"""
Author: Benjamin Arroquia Cuadros
12/06/2023

Module to process DOA index.

- ProvPolygons get convexhull over all geometries.
- Program just process DOA index in this bounding box.

"""

import numpy as np
from osgeo import gdal, osr, ogr
import os
import settings
import glob
from datetime import datetime
import pyproj
import math
import shutil
import re


# CLASS
class ProvPolygons:
    """
    Class to calculate data of provinces
    Create a dict with NAMEUNIT and NATCODE fields.
    Input layer must have this fields.
    Input: shapefile of provinces
    """
    def __init__(self, shape_path, name_layer):
        """
        Input: shapefile with provinces
        """
        self.lyr = None
        self.gdb = None
        self.read_gdb(shape_path, name_layer)
        extra_epsg = re.findall(r'\b\d+\b', self.lyr.GetSpatialRef()\
                                .ExportToWkt())[-1]
        self.epsg = int(extra_epsg)
        self.define_geotransform(self.epsg)
        self.ls_idfeat = list(range(self.lyr.GetFeatureCount()))
        self.dc_feat = dict(zip(self.ls_idfeat, [{} for i in self.ls_idfeat]))
        self.ly_x_min = None
        self.ly_x_max = None
        self.ly_y_min = None
        self.ly_y_max = None
        self.convexhull_from_layer()
        self.read_features()

    def read_gdb(self, gdb_path, name_layer):
        driver = ogr.GetDriverByName("GPKG")
        # opening the FileGDB
        try:
            self.gdb = driver.Open(gdb_path, 0)
            for l in self.gdb:
                if l.GetName() == name_layer:
                    self.lyr = l
        except Exception as e:
            raise Exception(e, "Can not open GPKG")

        if self.lyr == None:      
                raise Exception(f"Layer name {name_layer} not in {gdb_path}")
        
    def read_features(self):
        # Get metadata from provincias and bbox
        for FID, feat in enumerate(self.lyr):
            # feat = self.lyr.GetFeature(FID)
            self.dc_feat[FID]['entity'] = feat.GetField("NAMEUNIT")
            self.dc_feat[FID]['cod_entity'] = feat.GetField("NATCODE")[4:6]
            self.create_dict_features(feat, FID)
    
    def define_geotransform(self, from_epsg=4326):
        source = pyproj.CRS.from_epsg(from_epsg)
        ecmwf_lcc = pyproj.CRS.from_proj4(
            '+proj=lcc +lat_0=50 +lat_1=50 +lat_2=50 +lon_0=8 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs'
        )
        self.transform_to_ecmwf = pyproj.Transformer.from_crs(source, ecmwf_lcc, always_xy=True)
        
    def get_bbox_layer(self):
        dc_bbox = {
            'ly_x_min': self.ly_x_min,
            'ly_x_max': self.ly_x_max,
            'ly_y_min': self.ly_y_min,
            'ly_y_max': self.ly_y_max
        }
        return dc_bbox

    def create_dict_features(self, feat, fid):
        geom = feat.GetGeometryRef()
        # create list of point to get bbox
        pointsX = []; pointsY = []

        if (geom.GetGeometryName() == 'MULTIPOLYGON'):
            count = 0
            for polygon in geom:
                geomInner = geom.GetGeometryRef(count)
                ring = geomInner.GetGeometryRef(0)
                numpoints = ring.GetPointCount()
                for p in range(numpoints):
                        lon, lat, z = ring.GetPoint(p)
                        x, y = self.transform_to_ecmwf.transform(lon, lat)
                        pointsX.append(x)
                        pointsY.append(y)
                count += 1
        elif (geom.GetGeometryName() == 'POLYGON'):
            ring = geom.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                    lon, lat, z = ring.GetPoint(p)
                    x, y = self.transform_to_ecmwf.transform(lon, lat)
                    pointsX.append(x)
                    pointsY.append(y)
        self.dc_feat[fid]['xmin'] = min(pointsX)
        self.dc_feat[fid]['xmax'] = max(pointsX)
        self.dc_feat[fid]['ymin'] = min(pointsY)
        self.dc_feat[fid]['ymax'] = max(pointsY)
    
    def get_list_fields_name(self):
        schema = []
        ldefn = self.lyr.GetLayerDefn()
        for n in range(ldefn.GetFieldCount()):
            fdefn = ldefn.GetFieldDefn(n)
            schema.append(fdefn.name)
        print(schema)
        
    def convexhull_from_layer(self):
        """
        Calculate bbox to define mask area before calculate doa.
        The aim is reduce computacional cost of calculate doa.
        """
        # Collect all Geometry
        geomcol = ogr.Geometry(ogr.wkbGeometryCollection)
        for feature in self.lyr:
            geomcol.AddGeometry(feature.GetGeometryRef())
        pointsX = []; pointsY = []
        # Calculate convex hull
        convexhull = geomcol.ConvexHull()
        ring = convexhull.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                x, y = self.transform_to_ecmwf.transform(lon, lat)
                pointsX.append(x)
                pointsY.append(y)
        self.ly_x_min = min(pointsX)
        self.ly_x_max = max(pointsX)
        self.ly_y_min = min(pointsY)
        self.ly_y_max = max(pointsY)

        
    def select_by_atribute(self, value):
        # Value of nameunit or privince name
        self.lyr.SetAttributeFilter('"NAMEUNIT"=\'%s\'' % (value))
    
    def get_layer(self):
        return self.lyr
    
    def clear(self):
        self.shp = None

class DOA_change:
    """
    Class to create memfile of doa
    Create a new raster with multiple bands.
    Raster in same BBox defined x_lon_min, y_lat_max, xcount, ycount
    To get zonal stats iterate over bands
    """
    def __init__(self, grib_file):
        # Mem file
        self.raster = gdal.Open(grib_file)
        self.number_bands = self.raster.RasterCount
        self.change_metadata()

    def change_metadata(self):
        for band in range(1, self.number_bands+1):
            self.append_new_band(band)

    def append_new_band(self, band):
        band_sel = self.raster.GetRasterBand(band)
        meta_band = band_sel.GetMetadata()
        date = dict([i.split('=') for i in meta_band['GRIB_IDS'].split(' ')])['REF_TIME']
        meta_band['REF_TIME'] = date
        meta_band['GRIB_COMMENT'] = 'Density of Oxygen in Air (DOA)'
        meta_band['GRIB_UNIT'] = '[o²]'
        meta_band['GRIB_SHORT_NAME'] = 'DOA'
        meta_band['GRIB_ELEMENT'] = 'DOA'
        meta_band['GRIB_DISCIPLINE'] = 'Biometeorology'
        band_sel.SetMetadata(meta_band)


class DOA:
    """
    Class to create memfile of doa
    Create a new raster with multiple bands.
    Raster in same BBox defined x_lon_min, y_lat_max, xcount, ycount
    To get zonal stats iterate over bands
    """
    def __init__(self, pixelWidth, pixelHeight, x_lon_min, 
                 y_lat_max, xcount, ycount, raster_proj, 
                 metadata, bands):
        # Mem file
        self.target_doa = gdal.GetDriverByName('MEM').Create('', xcount, ycount, bands, gdal.GDT_Float32)
        self.target_doa.SetGeoTransform((
            x_lon_min, pixelWidth, 0,
            y_lat_max, 0, pixelHeight,
        ))
        self.band_number = 1
        self.target_doa.SetProjection(raster_proj)
        transform = self.target_doa.GetGeoTransform()
        self.number_bands = self.target_doa.RasterCount
        self.xOrigin = transform[0]
        self.yOrigin = transform[3]
        self.pixelWidth = transform[1]
        self.pixelHeight = transform[5]
        
    def append_new_band(self, temp, press, humi, metadata):
        matx_res = self.calculate_doa(temp, press, humi)
        band = self.target_doa.GetRasterBand(self.band_number)
        meta_band = metadata
        meta_band['GRIB_IDS'] = metadata['GRIB_IDS']
        meta_band['GRIB_REF_TIME'] = metadata['GRIB_REF_TIME']
        meta_band['GRIB_VALID_TIME'] = metadata['GRIB_VALID_TIME']
        date = dict([i.split('=') for i in metadata['GRIB_IDS'].split(' ')])['REF_TIME']
        meta_band['REF_TIME'] = date
        meta_band['GRIB_COMMENT'] = 'Density of Oxygen in Air (DOA)'
        meta_band['GRIB_UNIT'] = '[o²]'
        meta_band['GRIB_SHORT_NAME'] = 'DOA'
        meta_band['GRIB_ELEMENT'] = 'DOA'
        meta_band['GRIB_DISCIPLINE'] = 'Biometeorology'
        band.SetMetadata(meta_band)
        band.WriteArray(matx_res)
        self.band_number = self.band_number + 1
        return self.band_number -1

    def calculate_doa(self, temp, press, humi):
        """
        Calculate DOA index.
        Needed a n-matrix dimensioned and ordered: pressure, humidity, temperature
        :param matx: numpy matrix. 0-pressure, 1-humidity, 2-temperature. [pressure, humidity, temperature]
        :return: numpy matx_res
        """
        if (temp.shape == press.shape) and (humi.shape == press.shape):
            r, c = temp.shape
            ls_matrix = []
            for i in range(0, r):
                ls_row = []
                for j in range(0, c):
                    try:
                        p = press[i, j]
                        h = humi[i, j]+273.15
                        t = temp[i, j]
                        # relative humidity
                        hpa = p / 100.0
                        humidity_rel = (h) / 100.0
                        h6 = t + 35 * math.log(humidity_rel)
                        TVA = 6.112 * math.exp((17.7 * h6) / (h6 + 243.5))
                        doa_value = round((80.51 * hpa) / (t + 273) * (1 - TVA / hpa), 2)
                        ls_row.append(doa_value)
                    except Exception as e:
                        ls_row.append(0.0)
                        print("DOA: ", e, humidity_rel, h, p, t)
                ls_matrix.append(ls_row)
                matx_res = np.array(ls_matrix)
        else:
            matx_res = None
        return matx_res

    def get_doa_matrix(self, band):
        return self.target_doa.GetRasterBand(band).ReadAsArray().astype(float)
    
    def get_raster_params(self, band):
        str_format = "%Y-%m-%d %H:%M:%S"
        band_sel = self.target_doa.GetRasterBand(band)
        var_name = band_sel.GetMetadata()['GRIB_COMMENT']
        time_sec = band_sel.GetMetadata()['GRIB_REF_TIME']
        time_sec = datetime.fromtimestamp(int(time_sec.strip().split()[0])).strftime(str_format)
        return time_sec, var_name

    def print_geotransform(self):
        print('Doa geoT: ', self.target_doa.GetGeoTransform())
        
    def create_tiff():
        # save in file doa matrix
        pass
    
    def save_raster_to_check(self, file_name):
        out_raster = f"{settings.PROCESSED_FOLDERS}/{file_name}.grib"
        driver = gdal.GetDriverByName('GRIB')
        if os.path.exists(out_raster):
            os.remove(out_raster)
        dst = driver.CreateCopy(out_raster, self.target_doa, 0) 
        dst = None
        driver_ds = None
        
    def get_band_by_datetime(self, datetime):
        band_num = None
        for i in range(1, self.band+1):
            btime = self.target_doa.GetRasterBand(i).GetMetadata()['GRIB_REF_TIME']
            if btime == datetime:
                band_num = i
        return band_num
    
    def crop_raster_from_polygon(self, lyr, dc_entities, band_num):
        """
        Calculate new mem raster data using geometry
        Parameters:
            lyr, dc_entities, datetime, band_num
        Return:
            None
        """
        # Specify offset and rows and columns to read
        xoff = abs(int((self.xOrigin - dc_entities['xmin'])/self.pixelWidth))
        yoff = int((self.yOrigin - dc_entities['ymax'])/self.pixelWidth)
        xcount = int((dc_entities['xmax'] - dc_entities['xmin'])/self.pixelWidth)
        ycount = abs(int((dc_entities['ymax'] - dc_entities['ymin'])/self.pixelWidth))
        # Establece el origen correcto para el plot
        x_lon_min = xoff * self.pixelWidth + self.xOrigin
        y_lat_max = (yoff *self.pixelWidth - self.yOrigin) * -1

        # Create memory target raster
        self.target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Float32)
        self.target_ds.SetGeoTransform((
            x_lon_min, self.pixelWidth, 0,
            y_lat_max, 0, self.pixelHeight,
        ))
        self.target_ds.SetProjection(self.target_doa.GetProjection())
        # rasterize
        gdal.RasterizeLayer(self.target_ds, [1], lyr)
        # Read raster as arrays
        try:
            dataraster = self.target_doa.GetRasterBand(band_num).ReadAsArray(xoff, yoff, xcount, ycount).astype(float)
            bandmask = self.target_ds.GetRasterBand(1)
            datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(float)
            # Mask zone of raster
            zoneraster = np.ma.masked_array(dataraster,  np.logical_not(datamask))
            zoneraster = np.ma.filled(zoneraster, np.nan)
        except Exception as e:
            print("poligonize: ", e)
        matrix_poligonized = np.ma.filled(zoneraster, np.nan)
        return matrix_poligonized
    

class RasterExtraction:
    """
    Class to get stats from netcdf files
    """
    ls_vars_doa = ['Relative humidity [%]', 
                       'Temperature [C]', 'Pressure [Pa]']
    def __init__(self, grib_file):
        self.path = grib_file
        # raster data
        self.raster = gdal.Open(grib_file)
        # Get raster georeference info
        transform = self.raster.GetGeoTransform()
        self.number_bands = self.raster.RasterCount
        self.xOrigin = transform[0]
        self.yOrigin = transform[3]
        self.pixelWidth = transform[1]
        self.pixelHeight = transform[5]
        self.band = None
        self.bandraster = None
        self.ly_xoff = None
        self.ly_yoff = None
        self.ly_xcount = None
        self.ly_ycount = None
        self.doa_obj = None

    
    def get_raster_params(self, band):
        str_format = "%Y-%m-%d %H:%M:%S"
        band_sel = self.raster.GetRasterBand(band)
        var_name = band_sel.GetMetadata()['GRIB_COMMENT']
        time_sec = band_sel.GetMetadata()['GRIB_REF_TIME']
        time_sec = datetime.fromtimestamp(int(time_sec.strip().split()[0])).strftime(str_format)
        return time_sec, var_name

        
    def has_all_doa_variables(self, dc_vars):
        result = True
        for k, v in dc_vars.items():
            if k not in self.ls_vars_doa:
                result = False
        return result
    
    def process_bbox_to_crop(self, dc_bbox):
        # Define vars to crop from vectotrial data 
        # This should be abstract for any projection.
        # here only valid for spain.
        self.ly_xoff = abs(int((self.xOrigin - dc_bbox['ly_x_min'])/self.pixelWidth))
        self.ly_yoff = int((self.yOrigin - dc_bbox['ly_y_max'])/self.pixelWidth)
        self.ly_xcount = int((dc_bbox['ly_x_max'] - dc_bbox['ly_x_min'])/self.pixelWidth)
        self.ly_ycount = abs(int((dc_bbox['ly_y_max'] - dc_bbox['ly_y_min'])/self.pixelWidth))
        self.x_lon_min = self.ly_xoff * self.pixelWidth + self.xOrigin
        self.y_lat_max = (self.ly_yoff *self.pixelWidth - self.yOrigin) * -1
    
    def get_array_band_crop(self, band_number, crop=False):
        band = self.raster.GetRasterBand(band_number)
        # check if process doa with all raster or crop is fast
        if crop:
            if (self.ly_ycount == None) and (self.ly_xoff == None):
                self.process_bbox_to_crop(crop)
            array_band = band.ReadAsArray(self.ly_xoff, self.ly_yoff, 
                                    self.ly_xcount, self.ly_ycount).astype(float)
        else:
            array_band = band.ReadAsArray().astype(float)
        return array_band
    
    def get_list_with_variables(self, crop_raster=False):
        dc_time = {}
        ls_doas = []
        for i in range(1, self.number_bands+1):
            # construct a dict with all raster grouped by time
            time, var_name = self.get_raster_params(band=i)
            if time in dc_time:
                dc_time[time][var_name] = i
            else:
                dc_time[time] = {var_name: i}  
        for k, v in dc_time.items():
            # check if exists 3 variables to calculate doa
            if len(v) == 3 and self.has_all_doa_variables(dc_vars=v):
                humi = self.get_array_band_crop(v[self.ls_vars_doa[0]], crop_raster)
                temp = self.get_array_band_crop(v[self.ls_vars_doa[1]], crop_raster)
                press = self.get_array_band_crop(v[self.ls_vars_doa[2]], crop_raster)
                dc_vars = {
                    'time': k,
                    'humidity': humi,
                    'temperature': temp,
                    'pressure': press,
                    'band': v[self.ls_vars_doa[0]]
                }
                ls_doas.append(dc_vars)
        return ls_doas
    
    def print_stats(self, ar_no_nans):
        stats = ar_no_nans.shape[0], np.mean(ar_no_nans),\
        np.median(ar_no_nans), np.max(ar_no_nans),\
        np.min(ar_no_nans)
        print(stats)
    
    def create_doa_from_raster(self, layer):
        # Two way to create DOA and compare processing time
        # ls_doas = self.get_list_with_variables(crop_raster=False)
        ls_doas = self.get_list_with_variables(crop_raster=layer.get_bbox_layer())
        ls_time = []
        if self.doa_obj is None:
            metadata_r = self.raster.GetRasterBand(1).GetMetadata()
            self.doa_obj = DOA(pixelWidth=self.pixelWidth, 
                               pixelHeight=self.pixelHeight,
                            x_lon_min=self.x_lon_min, y_lat_max=self.y_lat_max, 
                          xcount=self.ly_xcount, ycount=self.ly_ycount, 
                          raster_proj=self.raster.GetProjection(),
                          metadata=metadata_r, bands=len(ls_doas))
        for i, vars_doa in enumerate(ls_doas):
            # create doas by time and append in object
            metadata_r = self.raster.GetRasterBand(vars_doa['band']).GetMetadata()
            b = self.doa_obj.append_new_band(temp=vars_doa['temperature'], 
                               press=vars_doa['pressure'], 
                               humi=vars_doa['humidity'],
                               metadata=metadata_r)
            ls_doas[i]['doa_band'] = b
            time, var = self.doa_obj.get_raster_params(b)
            ls_time.append(time)
        # uncomment this and create a new file tiff
        min_t, max_t = min(ls_time), max(ls_time)
        file_doa = f"DOA_{min_t}_{max_t}"
        self.doa_obj.save_raster_to_check(file_name=file_doa)
        return file_doa
        
        
    def crop_raster_from_polygon(self, lyr, dc_entities, band_num):
        """
        Calculate new mem raster data using geometry
        Parameters:
            lyr, dc_entities, datetime, band_num
        Return:
            None
        """
        # Specify offset and rows and columns to read
        xoff = abs(int((self.xOrigin - dc_entities['xmin'])/self.pixelWidth))
        yoff = int((self.yOrigin - dc_entities['ymax'])/self.pixelWidth)
        xcount = int((dc_entities['xmax'] - dc_entities['xmin'])/self.pixelWidth)
        ycount = abs(int((dc_entities['ymax'] - dc_entities['ymin'])/self.pixelWidth))
        # Establece el origen correcto para el plot
        x_lon_min = xoff * self.pixelWidth + self.xOrigin
        y_lat_max = (yoff *self.pixelWidth - self.yOrigin) * -1

        # Create memory target raster
        self.target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Float32)
        self.target_ds.SetGeoTransform((
            x_lon_min, self.pixelWidth, 0,
            y_lat_max, 0, self.pixelHeight,
        ))
        self.target_ds.SetProjection(self.raster.GetProjection())
        # rasterize
        gdal.RasterizeLayer(self.target_ds, [1], lyr)
        # Read raster as arrays
        try:
            dataraster = self.raster.GetRasterBand(band_num).ReadAsArray(xoff, yoff, xcount, ycount).astype(float)
            bandmask = self.target_ds.GetRasterBand(1)
            datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(float)
            # Mask zone of raster
            zoneraster = np.ma.masked_array(dataraster,  np.logical_not(datamask))
            if 'Relative humidity' in self.raster.GetRasterBand(band_num).GetMetadata()['GRIB_COMMENT']:
                # band of humidity is wrong
                zoneraster = np.ma.filled(zoneraster, np.nan) + 273.15
            else:
                zoneraster = np.ma.filled(zoneraster, np.nan)
            self.target_ds.GetRasterBand(1).WriteArray(np.ma.filled(zoneraster, np.nan))
            self.target_ds.GetRasterBand(1).SetNoDataValue(0.0)
        except Exception as e:
            print("poligonize: ", e)
        return np.ma.filled(zoneraster, np.nan)
        
    def set_band_raster(self, band):
        if band != self.band:
            self.band = band
            self.bandraster = self.raster.GetRasterBand(band)
        
    def get_doa(self):
        return self.doa_obj
            
    def clear(self):
        self.raster = None


# FUNCTIONS

def round_float(i):
    return round(i, 3)


def read_download_files(processed=True):
    """
    Get list of grib files to process.

    Parameters
    ----------
    processed : bool, optional
        only includes files contained in process folder.

    Return 
    ----------
    type:
        list of file names
    """
    txtfiles = []
    folder = os.path.join(settings.DOWNLOAD_FOLDERS, "*.grib")
    if processed:
        folder = os.path.join(settings.PROCESSED_FOLDERS, "*.grib")
    for file in glob.glob(folder):
        txtfiles.append(file)
    return txtfiles


def create_folders():
    ls_folders = [
        './data/download',
        './data/failures',
        './data/processed'
    ]
    for path in ls_folders:
        if not os.path.exists(path):
            os.mkdir(path)


def process_copernicus_data(processed=False):
    create_folders()
    # get path of grib files 
    files = read_download_files(processed=processed)
    # create bounding box with polygons
    layer_prov = ProvPolygons(shape_path=settings.POL_PROVINCES,
                                name_layer=settings.LAYER_PROVINCES)
    for file in files:
        if 'DOA' in file:
            continue
        # read first and execute 
        rpol = RasterExtraction(file)
        file_doa = rpol.create_doa_from_raster(layer=layer_prov)
        DOA_change(os.path.join(settings.PROCESSED_FOLDERS, file_doa+'.grib'))
        # Copy and remove files from download folder
        if not processed:
            shutil.copyfile(file, os.path.join(settings.PROCESSED_FOLDERS, 
                                            os.path.basename(file)))
            os.remove(file) 
        rpol.clear()
    layer_prov.clear()

if __name__ == "__main__":
    process_copernicus_data(processed=False)
    print("Finish")