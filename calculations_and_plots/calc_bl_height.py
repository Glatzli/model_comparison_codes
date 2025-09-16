"""
This file should include functions for calculating the PBL height using the methods described in
Wagner et al. 2015: The impact of valley geometry on daytime thermally driven flows and vertical transport processes
=> method for the CBL,

Idea: I would like to search for lat/lons
"""
import confg
import read_in_arome
import read_icon_model_3D
import read_ukmo
import read_wrf_helen

from calc_vhd import open_save_timeseries_main
import xarray as xr
import numpy as np
import matplotlib
import datetime
import matplotlib.pyplot as plt
from colorspace import terrain_hcl, qualitative_hcl, sequential_hcl
import pandas as pd


def calc_pbl_heights(ds, model="AROME"):
    """
    calculates PBL height according to Wagner et al. (2015): method to find the PBL height for a CBL and not for a SBL!
    when pot temp. differential is getting larger than the given threshold then the lower PBL height is reached (PBL1) etc
    :param ds:
    :param model:
    :return:
    a numpy array with the 3 PBL heights in it
    """
    dth = ds.th.differentiate(coord="height")  # dθ/d(height index)

    if model in ["ICON", "WRF"]:  # take unstaggered height vals for ICON & WRF models
        # where pot temp gradient first exceeds threshold (w NaN below): take min along height to get first value from surface up
        pbl_height1 = ds.where(dth > 0.001).z_unstag.min(dim="height")
        pbl_height2 = ds.where(dth < 0.001).z_unstag.max(dim="height")
        # take max to get first value from top where it gets below 0.001 K/m
    else:
        pbl_height1 = ds.where(dth > 0.001).z.min(dim="height")
        pbl_height2 = ds.where(dth < 0.001).z.max(dim="height")  # that is wrong! pbl2 should be higher than pbl1...

        # pbl_height3 = ds.where(ds.height > pbl_height2 and dth.max())  # for pbl 3 only take values above pbl2...

    return pbl_height1, pbl_height2


def plot_vert_profiles(timestamp_idx=34):
    # plot the pot. temp profiles with heights to enable a plausibility check:
    fig, axs = plt.subplots(figsize=(10, 8))
    arome.isel(time=timestamp_idx).th.plot(y="height", ax=axs, color=qualitative_colors[0], label="AROME")
    icon.isel(time=timestamp_idx).th.plot(y="height", ax=axs, color=qualitative_colors[2], label="ICON")
    icon2te.isel(time=timestamp_idx).th.plot(y="height", ax=axs, color=qualitative_colors[3], label="ICON2TE")
    um.isel(time=timestamp_idx).th.plot(y="height", ax=axs, color=qualitative_colors[4], label="UM")
    wrf.isel(time=timestamp_idx).th.plot(y="height", ax=axs, color=qualitative_colors[6], label="WRF")

    """
    plt.axhline(y=um_pbl_height1.isel(time=timestamp_idx).data, linestyle="--", color=qualitative_colors[4])
    axs.text(x=300, y=um_pbl_height1.isel(time=timestamp_idx).data, s="UM PBL height1")
    plt.axhline(y=um_pbl_height2.isel(time=timestamp_idx).data, linestyle="--", color=qualitative_colors[4])
    axs.text(x=350, y=um_pbl_height2.isel(time=timestamp_idx).data, s="UM PBL height2")

    plt.axhline(y=wrf_pbl_height1.isel(time=timestamp_idx).data, linestyle="--", linewidth=0.5, color=qualitative_colors[6])
    axs.text(x=300, y=wrf_pbl_height1.isel(time=timestamp_idx).data, s="WRF PBL height1")
    plt.axhline(y=wrf_pbl_height2.isel(time=timestamp_idx).data, linestyle="--", linewidth=0.5, color=qualitative_colors[6])
    axs.text(x=350, y=wrf_pbl_height2.isel(time=timestamp_idx).data, s="WRF PBL height2")
    """
    plt.legend()
    plt.grid()
    # plt.ylim([500, 5000])
    #plt.xlim([290, 330])
    plt.show()
    print(4)


if __name__ == '__main__':
    qualitative_colors = qualitative_hcl(palette="Dark 3").colors()
    # using PCGP-method:
    (arome, icon, icon2te, um, wrf,
     radio, hatpro) = open_save_timeseries_main(lat=confg.ibk_uni["lat"], lon=confg.ibk_uni["lon"],
                                                point_name=confg.ibk_uni["name"], variables=["th", "temp", "z", "z_unstag"],
                                                height_as_z_coord=True)
    plot_vert_profiles(timestamp_idx=34)

    """
    # variant for any point w/o using PCGP method:
    lat_point, lon_point = 47.31, 11.6
    vars = ["th", "z", "z_unstag"]
    # this would be the version for selecting a point and then calculating the PBL height
    # but as for VHD calculation: It would be faster to compute the height for the full domain first and then select the
    # point where the timeseries is wanted...

    arome = read_in_arome.read_in_arome_fixed_point(lat=lat_point, lon=lon_point, variables=vars[:-1],
                                                               height_as_z_coord=True)
    icon = read_icon_model_3D.read_icon_fixed_point(lat=lat_point, lon=lon_point, variant="ICON",
                                                    variables=vars, height_as_z_coord=True)
    icon2te = read_icon_model_3D.read_icon_fixed_point(lat=lat_point, lon=lon_point, variant="ICON2TE",
                                                       variables=vars, height_as_z_coord=True)
    um = read_ukmo.read_ukmo_fixed_point(lat=lat_point, lon=lon_point, variables=vars[:-1], height_as_z_coord=True)
    # don't read give the funct z_unstag as variable to read, cause only ICON & WRF are staggered!
    wrf = read_wrf_helen.read_wrf_fixed_point(lat=lat_point, lon=lon_point, variables=vars, height_as_z_coord=True)

    # calculate PBL heights

    arome_pbl_height1, arome_pbl_height2 = calc_pbl_heights(arome, model="AROME")
    icon_pbl_height1, icon_pbl_height2 = calc_pbl_heights(icon, model="ICON")
    icon2te_pbl_height1, icon2te_pbl_height2 = calc_pbl_heights(icon2te, model="ICON")

    um_pbl_height1, um_pbl_height2 = calc_pbl_heights(um, model="UM")
    wrf_pbl_height1, wrf_pbl_height2 = calc_pbl_heights(wrf, model="WRF")

    plot_vert_profiles(timestamp_idx=34)

    um_pbl_heights = xr.Dataset(data_vars={"pbl_height1": um_pbl_height1,
                                           "pbl_height2": um_pbl_height2},
                                attrs={"modelinfo": "UM boundary layer height computed as defined in DOI:10.1002/qj.2481"})
    um_pbl_heights.to_netcdf("UM_pbl_heights.nc")
    wrf
    """
