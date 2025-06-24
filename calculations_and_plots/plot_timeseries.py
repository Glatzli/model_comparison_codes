"""
This script is used to plot the time series of the vertical distribution of potential temperature for all models.
problem: vertical coordinate is not the same for all models => use pressure?
"""

import sys
sys.path.append("D:/MSc_Arbeit/model_comparison_codes")
import importlib
import read_in_arome
import read_icon_model_3D
import read_ukmo
# importlib.reload(read_icon_model_3D)
import read_wrf_helen
importlib.reload(read_in_arome)
import confg
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from colorspace import diverging_hcl



def plot_pot_temp_time_contours(pot_temp, model="AROME"):
    """
    plot pot temp time & height series for all models. HATPRO was interpolated to AROME levels & it's pressure is used
    to compute pot temp.
    thin 1 K pot temp contour lines, thick 5 K pot temp contour lines and red/blue shading for the 1/2 hrly
    warming/cooling in pot temp is plotted

    :param pot_temp:
    :param model:
    :return:
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    vmin, vmax = -2, 2  # uniform colorbar
    levels = np.arange(vmin, vmax + 0.5, 0.5)
    # limit the time range for the plot
    start_time = pd.to_datetime('2017-10-15 13:00:00', format='%Y-%m-%d %H:%M:%S')
    end_time = pd.to_datetime('2017-10-16 12:00:00', format='%Y-%m-%d %H:%M:%S')
    # pot_temp = pot_temp.sel(time=slice(start_time, end_time))

    # Plot the filled contours
    contourf = (pot_temp.diff("time", n=1) * 2).plot.contourf(ax=ax, x="time", y="height", levels=levels, cmap=pal1.cmap(),
                                                   add_colorbar=False, vmin=vmin, vmax=vmax)

    # Plot the contour lines
    contour1 = pot_temp.plot.contour(ax=ax, x="time", y="height",
                                     levels=np.arange(np.round(pot_temp.min()), np.round(pot_temp.max()), 1),
                                     colors='black', linewidths=0.5)   #.isel(time=slice(1, 100))
    contour5 = pot_temp.plot.contour(ax=ax, x="time", y="height", levels=np.arange(290, np.round(pot_temp.max()), 5),
                                     colors='black', linewidths=1.5)  #.isel(time=slice(1, 100))
    ax.clabel(contour5)

    ax.set_xlim(start_time, end_time)
    # Add a colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('K hr$^{-1}$')

    ax.set_title(model + " potential temp time series")
    if model ==  "HATPRO":
        ax.set_ylabel(f"height [m]")
    elif model == "ICON" or model == "ICON2TE":
        ax.set_ylabel(f"geometric height [m]")
    else:
        ax.set_ylabel(f"geopotential height [m]")
    ax.set_xlabel("")

    plt.savefig(confg.dir_PLOTS + model + f"_pot_temp_timeseries_{interface_height}_ibk.png", dpi=500)


def plot_temp_time_contours(temp, model="AROME"):
    """
    plot temp over time & height for all models incl HATPRO.
    :param temp:
    :param model:
    :return:
    """
    """
    match model:
        case "AROME":
            "time"="time"; "height" = "height"
        case ("ICON" | "ICON2TE"):  # same for ICON2TE
            "time"="time"; "height" = "height"
        case "UKMO":
            "time"="time"; "height" = "height"
        case "WRF":
            "time"="time"; "height" = "height"
        case "HATPRO":
            "time" = "time"; "height" = "height" """

    fig, ax = plt.subplots(figsize=(12, 6))

    vmin, vmax = -1.5, 1.5  # uniform colorbar
    levels = np.arange(vmin, vmax + 0.3, 0.3)

    start_time = pd.to_datetime('2017-10-15 14:00:00', format='%Y-%m-%d %H:%M:%S')
    end_time = pd.to_datetime('2017-10-16 12:00:00', format='%Y-%m-%d %H:%M:%S')
    # temp = temp.sel(time=slice(start_time, end_time))

    # Plot the filled contours
    contourf = temp.diff("time").plot.contourf(ax=ax, x="time", y="height", levels=levels, cmap=pal1.cmap(),
                                                   add_colorbar=False, vmin=-2, vmax=2)

    # Plot the contour lines
    contour1 = temp.plot.contour(ax=ax, x="time", y="height",
                                                                    levels=np.arange(np.round(temp.min()),
                                                                                     np.round(temp.max()), 1),
                                                                    colors='black', linewidths=0.5)   #.isel(time=slice(1, 100))
    contour5 = temp.plot.contour(ax=ax, x="time", y="height",
                                                                    levels=np.arange(-50, np.round(temp.max()), 5),
                                                                    colors='black', linewidths=1.5)  #.isel(time=slice(1, 100))
    ax.clabel(contour5)

    ax.set_xlim(start_time, end_time)
    # Add a colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('K hr$^{-1}$')
    ax.set_title(model + " temp time series")
    if model ==  "HATPRO":
        ax.set_ylabel(f"height [m]")
    else:
        ax.set_ylabel(f"geopotential height [m]")
    ax.set_xlabel("")
    plt.savefig(confg.dir_PLOTS + model + "_temp_timeseries_ibk.png", dpi=300)
    plt.show()


def plot_arome():
    # arome = read_in_arome.read_in_arome_fixed_point(lat=lat_ibk, lon=lon_ibk, )
    # arome = read_in_arome.read_3D_variables_AROME(variables=["p", "th", "z"], method="sel", lat=lat_ibk, lon=lon_ibk)
    # pot_temp = arome.th.isel(nz=np.arange(40, 90))

    arome = xr.open_dataset(confg.model_folder + "/AROME/" + "AROME_temp_timeseries_ibk.nc")
    pot_temp = arome.th.where(arome["height"] <= interface_height, drop=True)
    plot_pot_temp_time_contours(pot_temp, model="AROME")

    # temp = arome.temperature.where(arome.height <= 4000, drop=True)  # tried with normal temp, but you don't see much...
    # plot_temp_time_contours(temp, model="AROME")

def plot_icon():
    """icon15 = read_icon_model_3D.read_icon_fixed_point_multiple_hours(day=15, hours=np.arange(14, 23), lon=lon_ibk, lat=lat_ibk, variant="ICON")
    icon16 = read_icon_model_3D.read_icon_fixed_point_multiple_hours(day=16, hours=np.arange(0, 9), lon=lon_ibk, lat=lat_ibk, variant="ICON")
    variables = ["th", "temp", "z_ifc"]  # "temp", "pres", "u", "v", "w",
    icon = xr.concat([icon15[variables], icon16[variables]], dim="time")"""
    icon = xr.open_dataset(confg.icon_folder_3D + "/ICON_temp_timeseries_ibk.nc")
    pot_temp = icon.th.where(icon["height"] <= interface_height, drop=True)
    plot_pot_temp_time_contours(pot_temp, model="ICON")

def plot_icon2te():
    """icon15_2te = read_icon_model_3D.read_icon_fixed_point_multiple_hours(day=15, hours=range(12, 24), lon=lon_ibk,
                                                                         lat=lat_ibk, variant="ICON2TE")
    icon16_2te = read_icon_model_3D.read_icon_fixed_point_multiple_hours(day=16, hours=range(00, 13), lon=lon_ibk,
                                                                         lat=lat_ibk, variant="ICON2TE")
    variables = ["th", "temp"]  # ["temp", "pressure", "pres", "u", "v", "w"]
    icon2te = xr.concat([icon15_2te[variables], icon16_2te[variables]], dim="time")
    icon2te_pot_temp = icon2te.th.isel(height=np.arange(40, 90))"""

    icon_2te = xr.open_dataset(confg.icon2TE_folder_3D + "/ICON_2TE_temp_timeseries_ibk.nc")
    icon_2te_pot_temp = icon_2te.th.where(icon_2te["height"] <= interface_height, drop=True)
    plot_pot_temp_time_contours(pot_temp=icon_2te_pot_temp, model="ICON2TE")

def plot_ukmo():
    um = xr.open_dataset(confg.ukmo_folder + "/UKMO_temp_timeseries_ibk.nc")
    um_pot_temp = um.th.where(um["height"] <= interface_height, drop=True)

    plot_pot_temp_time_contours(pot_temp=um_pot_temp, model="UKMO")

def plot_wrf():
    # wrf = read_wrf_helen.read_wrf_fixed_point(lat=lat_ibk, lon=lon_ibk)
    # wrf_pot_temp = wrf.th.isel(height=slice(0, 50))

    wrf = xr.open_dataset(confg.wrf_folder + "/WRF_temp_timeseries_ibk.nc")
    wrf_pot_temp = wrf.th.where(wrf["height"] <= interface_height, drop=True)

    plot_pot_temp_time_contours(pot_temp=wrf_pot_temp, model="WRF")


def plot_hatpro():
    # hatpro = xr.open_dataset(f"{confg.hatpro_folder}/hatpro_merged.nc")
    # hatpro_temp = hatpro["temperature"].sel(height=slice(0, 4400))  # select up to 4.400 m
    # plot_temp_time_contours(temp=hatpro_temp, model="HATPRO")
    # hatpro

    # try with hatpro interpolated data
    hatpro = xr.open_dataset(f"{confg.hatpro_folder}/hatpro_interpolated_arome.nc")
    hatpro_pot_temp = hatpro["th"].where(hatpro["height"] <= interface_height, drop=True)  # hatpro["th"].sel(height=slice(0, 4400))  # select up to 4.400 m
    plot_pot_temp_time_contours(pot_temp=hatpro_pot_temp, model="HATPRO")

if __name__ == '__main__':
    lat_ibk = 47.259998
    lon_ibk = 11.384167
    interface_height = 4000  # what is max height that should be plotted?
    pal1 = diverging_hcl(palette="Blue-Red 2")

    matplotlib.use('Qt5Agg')  # Use the Qt5Agg backend for interactive plotting

    plot_arome()

    plot_icon()
    plot_icon2te()

    plot_ukmo()
    plot_wrf()
    plot_hatpro()
    plt.show()
