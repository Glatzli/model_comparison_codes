"""
This file should include functions for calculating the PBL height using the methods described in
Wagner et al. 2015: The impact of valley geometry on daytime thermally driven flows and vertical transport processes

Idea: I would like to search for lat/lons
"""
import confg
import read_in_arome
import read_icon_model_3D
import read_ukmo
import read_wrf_helen

import xarray as xr
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import pandas as pd


def calc_pbl1(ds, model="AROME"):
    """
    calculates PBL height according to Wagner et al. (2015)
    is this calculation right? I think!
    :param ds:
    :param model:
    :return:
    """
    dth = ds.th.differentiate(coord="height")  # dθ/d(height index)
    if model in ["ICON", "WRF"]:  # take unstaggered height vals for ICON & WRF models
        # where pot temp gradient first exceeds threshold (w NaN below): take min along height to get first value from surface up
        pbl_height1 = ds.where(dth > -0.001, drop=True).z_unstag.min(dim="height")
        pbl_height2 = ds.where(dth > -0.001, drop=True).z_unstag.max(dim="height")  # take max to get first value from top
    else:
        pbl_height1 = ds.where(dth > -0.001, drop=True).z.min(dim="height")
        pbl_height2 = ds.where(dth > -0.001, drop=True).z.max(dim="height")

    return pbl_height1, pbl_height2



if __name__ == '__main__':


    lat_point, lon_point = 47.31, 11.6
    vars = ["p", "temp", "th", "z", "z_unstag"]
    arome = read_in_arome.read_in_arome_fixed_point(lat=lat_point, lon=lon_point, variables=vars[:-1],
                                                               height_as_z_coord=True)
    um = read_ukmo.read_ukmo_fixed_point(lat=lat_point, lon=lon_point, variables=vars[:-1], height_as_z_coord=True)

    # till here it works! continue...
    nwms = [arome, um]  # [icon, um, wrf]
    pbl_height1 = np.array([0])
    for nwm in nwms:
        np.append(calc_pbl1(ds=nwm, model="AROME"))

    icon = read_icon_model_3D.read_icon_fixed_point(lat=lat_point, lon=lon_point, variant="ICON",
                                                    variables=vars, height_as_z_coord=True)
    icon2te = read_icon_model_3D.read_icon_fixed_point(lat=lat_point, lon=lon_point, variant="ICON2TE",
                                                       variables=vars, height_as_z_coord=True)
    wrf = read_wrf_helen.read_wrf_fixed_point(lat=lat_point, lon=lon_point, variables=vars, height_as_z_coord=True)

    models = [icon, icon2te, wrf]
