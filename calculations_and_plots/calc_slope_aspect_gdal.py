"""
after the example: https://www.leahwasser.com/dev-earthlab-site/tutorials/DEM-slope-aspect-python/
because xDEM needs a vCRS, and richDEM fails installation...
"""

from __future__ import division
from osgeo import gdal
from matplotlib.colors import ListedColormap
from matplotlib import colors
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)


#Open the front_range.dem file using gdal
filename = '../data/front_range_dem.tif'

# define helper functions
def getResolution(rasterfn):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    res = {"east-west": abs(geotransform[1]),
           "north-south": abs(geotransform[5])}
    return res

# next 2 functions: https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
def raster2array(rasterfn):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    return band.ReadAsArray()

def getNoDataValue(rasterfn):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    return band.GetNoDataValue()

# load raster as 2D numpy array
data_array = raster2array(filename)
nodataval = getNoDataValue(filename)
resolution = getResolution(filename)
print(resolution)
