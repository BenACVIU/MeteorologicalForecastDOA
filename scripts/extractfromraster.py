"""
Author: B. Arroquia Cuadros
Processing meteorological data from Copernicus
Example of get zonal statistics of single polygon.
"""

from osgeo import gdal, osr, ogr
from datetime import datetime
import re
import pyproj
import numpy as np
import matplotlib.pyplot as plt
import os
import settings


def round_float(number):
    return round(number, 2)

# Input param
cod_prov = 46

# Files
path = os.getcwd()
grib_file = './data/processed/DOA_2016-08-01 02:00:00_2016-08-31 20:00:00.grib'
# grib_file = os.path.join(path, grib_file)
provinces_ly = './data/provinces_spain.gpkg'
# provinces_ly = os.path.join(path, provinces_ly)
print("Files:\n", grib_file, "\n", provinces_ly)

# Load data and get metadata

driver = ogr.GetDriverByName("GPKG")
gdb = driver.Open(provinces_ly, 0)
for l in gdb:
    if l.GetName() == settings.LAYER_CLUSTERS:
         lyr = l


# gdb = driver.Open(provinces_ly, 0)
# lyr = gdb.GetLayerByIndex(0)
extra_epsg = re.findall(r'\b\d+\b', lyr.GetSpatialRef().ExportToWkt())[-1]
epsg = int(extra_epsg)
print("CRS: ", epsg)

# define geo transform
source = pyproj.CRS.from_epsg(epsg)
ecmwf_lcc = pyproj.CRS.from_proj4(
    '+proj=lcc +lat_0=50 +lat_1=50 +lat_2=50 +lon_0=8 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs'
)
transform_to_ecmwf = pyproj.Transformer.from_crs(source, ecmwf_lcc, always_xy=True)

# Loop over provinces
ls_idfeat = list(range(lyr.GetFeatureCount()))
dc_entity = dict(zip(ls_idfeat, [{} for i in ls_idfeat]))
for fid, feat in enumerate(lyr):
    dc_entity[fid]["name"] = feat.GetField("entity")
    dc_entity[fid]["cod_entity"] = feat.GetField("cod_entity")
    dc_entity[fid]["method"] = feat.GetField("k_method")
    geom = feat.GetGeometryRef()
    # create list of point to get bbox
    pointsX = []; pointsY = []
    # Get coordinates from polygons
    if (geom.GetGeometryName() == 'MULTIPOLYGON'):
        count = 0
        for polygon in geom:
            geomInner = geom.GetGeometryRef(count)
            ring = geomInner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                    lon, lat, z = ring.GetPoint(p)
                    x, y = transform_to_ecmwf.transform(lon, lat)
                    pointsX.append(x)
                    pointsY.append(y)
            count += 1
    elif (geom.GetGeometryName() == 'POLYGON'):
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                x, y = transform_to_ecmwf.transform(lon, lat)
                pointsX.append(x)
                pointsY.append(y)
    # Extract bounding box
    dc_entity[fid]['xmin'] = min(pointsX)
    dc_entity[fid]['xmax'] = max(pointsX)
    dc_entity[fid]['ymin'] = min(pointsY)
    dc_entity[fid]['ymax'] = max(pointsY)

# Read raster
raster = gdal.Open(grib_file)
transform = raster.GetGeoTransform()
number_bands = raster.RasterCount
xOrigin = transform[0]
yOrigin = transform[3]
pixelWidth = transform[1]
pixelHeight = transform[5]
band = raster.GetRasterBand(1)
str_format = "%Y-%m-%d %H:%M:%S"
time_sec = band.GetMetadata()['GRIB_REF_TIME']
time_sec = datetime.fromtimestamp(int(time_sec.strip().split()[0])).strftime(str_format)
print("Date of band: ", time_sec)
ly_xoff = None
ly_yoff = None
ly_xcount = None
ly_ycount = None
doa_obj = None

dc_feat = [v for k, v in dc_entity.items() if (int(v['cod_entity']) == cod_prov)][0]
# Specify offset and rows and columns to read
xoff = abs(int((xOrigin - dc_feat['xmin'])/pixelWidth))
yoff = int((yOrigin - dc_feat['ymax'])/pixelWidth)
xcount = int((dc_feat['xmax'] - dc_feat['xmin']) / pixelWidth) + 1
ycount = abs(int((dc_feat['ymax'] - dc_feat['ymin']) / pixelWidth))
# Origin of new raster
x_lon_min = xoff * pixelWidth + xOrigin
y_lat_max = (yoff *pixelWidth - yOrigin) * -1
print(x_lon_min, y_lat_max)

# Create memory target raster
target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Float32)
target_ds.SetGeoTransform((
    x_lon_min, pixelWidth, 0,
    y_lat_max, 0, pixelHeight,
))
target_ds.SetProjection(raster.GetProjection())
print(dc_feat)
# rasterize by selected province in vector layer
lyr.SetAttributeFilter('"cod_entity"=\'%s\' and "k_method"=\'%s\'' % (str(cod_prov), dc_feat['method']))
# lyr.SetAttributeFilter('"k_method"="lisa"')
for l in lyr:
     print(l.GetField("cod_entity"), l.GetField("k_method"))
gdal.RasterizeLayer(target_ds, [1], lyr)
# Read raster as arrays
try:
    dataraster = band.ReadAsArray(xoff, yoff, xcount, ycount).astype(float)
    bandmask = target_ds.GetRasterBand(1)
    datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(float)
    # Mask zone of raster
    zoneraster = np.ma.masked_array(dataraster,  np.logical_not(datamask))
    print(band.GetMetadata()['GRIB_COMMENT'])
    zoneraster = np.ma.filled(zoneraster, np.nan)
    target_ds.GetRasterBand(1).WriteArray(np.ma.filled(zoneraster, np.nan))
    target_ds.GetRasterBand(1).SetNoDataValue(0.0)
except Exception as e:
    print("poligonize: ", e)

npdata = np.ma.filled(zoneraster, np.nan)
ar_no_nans = npdata.flatten()[~np.isnan(npdata.flatten())]
# Zonal statistics: shape, mean, median, std, var, max, min
stats = ar_no_nans.shape[0], np.mean(ar_no_nans),\
        np.median(ar_no_nans), np.std(ar_no_nans),\
        np.var(ar_no_nans), np.max(ar_no_nans),\
        np.min(ar_no_nans)

plt.imshow(zoneraster, interpolation='nearest')
plt.colorbar()
plt.show()

print(list(map(round_float, map(float, stats))))
