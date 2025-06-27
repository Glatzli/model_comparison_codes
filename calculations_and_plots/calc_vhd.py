"""
In this script some functinons should be defined that calculate the valley heat deficit (VHD) from the potential temperature
for each model. VHD will be calc between lowest model level & Hafelekar at xxxx m.

At first calc VHD for grid point of Innsbruck.
"""

import sys

from read_icon_model_3D import read_icon_fixed_time

sys.path.append("D:/MSc_Arbeit/model_comparison_codes")
import importlib
import read_in_arome
import read_icon_model_3D
import read_ukmo
# importlib.reload(read_icon_model_3D)
import read_wrf_helen
# importlib.reload(read_in_arome)
import confg
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from colorspace import diverging_hcl


def calc_vhd_single_point(ds_point):
    ds_below_hafelekar = ds_point.where(ds_point.height <= hafelekar_height, drop=True)
    th_hafelekar = ds_below_hafelekar.isel(height=0).th
    vhd_point = c_p*((th_hafelekar - ds_below_hafelekar.th.isel(height=slice(1, 100))) * ds_below_hafelekar.rho).sum(dim="height")  # pot temp deficit

    # th_deficit[th_deficit == th_deficit.max()]  # max pot temp deficit at 01UTC in icon model


if __name__ == '__main__':
    matplotlib.use('Qt5Agg')
    lat_ibk = 47.259998
    lon_ibk = 11.384167
    hafelekar_height = 2279  # m, highest HOBO from https://zenodo.org/records/4672313 hobo dataset
    c_p = 1005  # J/(kg*K), specific heat capacity of air at constant pressure
    # icon_ibk_ngp = read_icon_model_3D.read_icon_fixed_point(lat=lat_ibk, lon=lon_ibk, variant="ICON")  # ngp = nearest grid point
    icon_ibk_ngp = xr.open_dataset(confg.icon_folder_3D + "/ICON_temp_p_rho_timeseries_ibk.nc")
    calc_vhd_single_point(icon_ibk_ngp)