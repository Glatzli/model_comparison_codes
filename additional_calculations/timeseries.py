import sys
sys.path.append("D:/MSc_Arbeit/model_comparison_codes")
import importlib
importlib.reload(read_icon_model_3D)
import read_icon_model_3D
import confg
import dask
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

icon15 = read_icon_model_3D.read_icon_fixed_point_multiple_hours(day = 15, hours = range(12, 24), lon = 11.4011756,
                                                 lat = 47.266076, variant = "ICON")
icon16 = read_icon_model_3D.read_icon_fixed_point_multiple_hours(day = 16, hours = range(00, 13), lon = 11.4011756,
                                                 lat = 47.266076, variant = "ICON")

variables = ["temperature", "pressure", "temp", "pres", "u", "v", "w"]
icon = xr.concat([icon15[variables], icon16[variables]], dim = "time")