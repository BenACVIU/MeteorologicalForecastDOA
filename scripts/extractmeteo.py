"""
Author: Benjamin Arroquia Cuadros
10/06/2026

Extract data from raster copernicus grib files to sqlite.
Module to clip and extract statistics from raster file.

Dependencies: osgeo, numpy, pandas

Install GDAL: https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html
Note, check ogrinfo: pip install GDAL==<GDAL VERSION FROM OGRINFO>

Workflow of data processing:
1. Read download folder
2. If exist '*.grib' file process
3. Get all metadata of raster: time and variables
4. Process data: DOA
    4.1 Get all bands and create dict with dates and 3 vars to calculate DOA
    4.2 Crop all raster to create DOA obj inside vectorial layer
    4.3 Create DOAs band
    4.4 Iterate over polygon entities and gather zonal statistics
    4.5 Insert in db
    4.6 Create gtiff of doas
5. Process data: copernicus variables
6. Copy file in processed folder

ProvPolygons transform coordinates into Lambert Conical.

Data raster from Copernicus has to be allocated in a folder './data/download'.
All data in dowload folder is processed, deleted and copied in processed.

DOA error in humidity variable from copernicus.
Correct using (h+273.15) in doa function.

DOA is calculated croping full raster using bbox from polygon layer.
This aims to reduce time processing and be efficient.
Then stats are calculated over single entities in polygon layer (provinces).

Use case: get pixel values stats about every polygon
        in a shapefile that intersects with a raster layer 
"""


import numpy as np
from osgeo import gdal, ogr
import os
import settings
import glob
from datetime import datetime
import sqlalchemy
from sqlalchemy import Column, Float, String, MetaData, DateTime
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base
import pyproj
import re

# MODELS
meta = MetaData()
Base = declarative_base()


class StatsLayerClusters(Base):
    __tablename__ = "stats_cluster"
    variable = Column('variable', String, primary_key=True)
    # algorithm used to extract zonal stats
    time = Column('time', DateTime(timezone=False), 
                     default=datetime.utcnow, primary_key=True)
    entity = Column('entity', String) 
    cod_entity = Column('cod_entity', String, primary_key=True) 
    method = Column('method', String, primary_key=True)
    shape = Column('shape', Float)
    mean = Column('mean', Float) 
    median = Column('median', Float) 
    stdd = Column('stdd', Float)
    variance = Column('variance', Float) 
    maxi = Column('maxi', Float) 
    mini = Column('mini', Float) 
    def __repr__(self):
        return f"StatsCop(variable={self.variable!r}, entity={self.entity!r}, mean={self.mean!r})"
    

class SqlDB:
    def __init__(self, path, meta, Base):
        self.path_db = path
        self.meta = meta
        self.Base = Base
        self.create_db()

    def create_db(self):
        self.engine = sqlalchemy.create_engine(self.path_db)
        self.Base.metadata.create_all(self.engine)
            
    def inser_many_to_table(self, table, ls_data):
        """
        :param table: class of table
        :param ls_data: list of lists of values
        """
        ls_rows = []
        cols = [c.key for c in table.__table__.columns]
        for data in ls_data:
            row = dict(zip(cols, data))
            ls_rows.append(table(**row))
        with Session(self.engine) as session:
            session.add_all(ls_rows)
            session.commit()
    
    def update_table_download(self, table, name, time, value):
        with Session(self.engine) as session:
            down = session.query(table).filter(table.name == name, 
                                               table.time == time).one()
            down.status = value
            session.commit()

# CLASS

class RasterExtraction:
    """
    Class to get stats from netcdf or grib files
    """
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
        self.metadata = self.raster.GetRasterBand(1).GetMetadata()

    @staticmethod
    def round_float(i):
        return round(i, 3)

    def get_raster_params(self, band):
        str_format = "%Y-%m-%d %H:%M:%S"
        band_sel = self.raster.GetRasterBand(band)
        var_name = band_sel.GetMetadata()['GRIB_COMMENT']
        time_sec = band_sel.GetMetadata()['GRIB_REF_TIME']
        time_sec = datetime.fromtimestamp(int(time_sec.strip().split()[0])).strftime(str_format)
        return time_sec, var_name

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
        xcount = int((dc_entities['xmax'] - dc_entities['xmin'])/self.pixelWidth) + 1
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
            zoneraster = np.ma.filled(zoneraster, np.nan)
            self.target_ds.GetRasterBand(1)\
                            .WriteArray(np.ma.filled(zoneraster, np.nan))
            self.target_ds.GetRasterBand(1).SetNoDataValue(0.0)
        except Exception as e:
            print("poligonize: ", e)
        return np.ma.filled(zoneraster, np.nan)
        
    def set_band_raster(self, band):
        if band != self.band:
            self.band = band
            self.bandraster = self.raster.GetRasterBand(band)
            
    def get_stats(self, lyr, dc_entities, band_number):
        """
        Calculate tuple of statistics creating a new raster layer.
        Parameters:
            dc_entities (dict): data about geometry related with the mask
            lyr (ogr): layer with geometry selected to create a mask
            band_number (int): number band
        Return:
            stats (tuple): shape, mean, median, std, var, max, min
        """
        npdata = self.crop_raster_from_polygon(lyr, dc_entities, band_number)
        ar_no_nans = npdata.flatten()[~np.isnan(npdata.flatten())]
        stats = ar_no_nans.shape[0], np.mean(ar_no_nans),\
                np.median(ar_no_nans), np.std(ar_no_nans),\
                np.var(ar_no_nans), np.max(ar_no_nans),\
                np.min(ar_no_nans)
        return list(map(self.round_float, map(float, stats)))
    

    def get_stats_provinces(self, layer_prov):
        """
        Loop over bands and create a lis of zonal statistics
        """
        ls_rows = []
        for i in range(1, self.number_bands+1):
            fecha = datetime.strptime(self.get_raster_params(i)[0], 
                                      '%Y-%m-%d %H:%M:%S')
            variable_atm = self.get_raster_params(i)[1]
            for k, feat in layer_prov.dc_feat.items():
                layer_prov.select_by_atribute(feat['cod_entity'], feat['method'])
                lsstats = self.get_stats(layer_prov.get_layer(), feat, i)
                # # formating insert in table
                row = [variable_atm, fecha, feat['entity'], 
                       feat['cod_entity'], feat['method']] + lsstats
                ls_rows.append(row)
        return ls_rows

    def clear(self):
        self.raster = None


class ClusterPolygons:
    """
    Class to load GPKG with layers
    Create a dict with NAMEUNIT and NATCODE fields.
    Input layer must have this fields.
    Input: shapefile of provinces
    """
    def __init__(self, gdb_path, name_layer):
        """
        Input: GPKG with clusters of provincies
        """
        # list to store layers'names
        self.dc_layers = {}
        self.gdb = None
        self.read_gdb(gdb_path, name_layer)
        extra_epsg = re.findall(r'\b\d+\b', self.lyr.GetSpatialRef()\
                                .ExportToWkt())[-1]
        self.epsg = int(extra_epsg)
        self.define_geotransform(self.epsg)
        self.ls_idfeat = list(range(self.lyr.GetFeatureCount()))
        self.dc_feat = dict(zip(self.ls_idfeat, 
                                [{} for i in self.ls_idfeat]))
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
            self.dc_feat[FID]['entity'] = feat.GetField("entity")
            self.dc_feat[FID]['cod_entity'] = feat.GetField("cod_entity")
            self.dc_feat[FID]['method'] = feat.GetField("k_method")
            # self.dc_feat[FID]['layer'] = feat.GetField("layer")
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

    def select_by_atribute(self, cod_entity, method):
        # Value of nameunit or privince name
        self.lyr.SetAttributeFilter('"cod_entity"=\'%s\' and "k_method"=\'%s\'' % (cod_entity, method))

    def get_layer(self):
        return self.lyr
    
    def clear(self):
        self.shp = None


# FUNCTIONS

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


def process_copernicus_data(processed=False):
    # get path of grib files 
    files = read_download_files(processed=processed)
    # db to register processed data
    db_geo = SqlDB(path=f"sqlite:///{settings.DB_PATH}",
                   meta=meta, Base=Base)
    polclus = ClusterPolygons(gdb_path=settings.POL_PROVINCES,
                                name_layer=settings.LAYER_CLUSTERS)
    for file in files:
        # read first and execute 
        coperaster = RasterExtraction(file)
        # Extract stats from raster and insert in db 
        ls_rows = coperaster.get_stats_provinces(polclus)
        db_geo.inser_many_to_table(table=StatsLayerClusters, 
                                     ls_data=ls_rows) 
        coperaster.clear()
    polclus.clear()
        

if __name__ == "__main__":
    print(settings.DB_PATH)
    process_copernicus_data(processed=True)
    print("Finish")