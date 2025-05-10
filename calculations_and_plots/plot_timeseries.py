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
    plot pot temp time & height series for all models. problem: I don't have pressure for HATPRO (plot temp from models?)!
    :param pot_temp:
    :param model:
    :return:
    """
    match model:
        case "AROME":
            variable_x="time"; variable_y = "nz"
        case ("ICON" | "ICON2TE"):  # same for ICON2TE
            variable_x="time"; variable_y = "height"
        case "UKMO":
            variable_x="time"; variable_y = "height"
        case "WRF":
            variable_x="time"; variable_y = "height"
        case "HATPRO":
            variable_x = "time"; variable_y = "height"

    fig, ax = plt.subplots(figsize=(12, 6))

    vmin, vmax = -2, 2  # uniform colorbar
    levels = np.arange(vmin, vmax + 0.5, 0.5)
    # limit the time range for the plot
    start_time = pd.to_datetime('2017-10-15 14:00:00', format='%Y-%m-%d %H:%M:%S')
    end_time = pd.to_datetime('2017-10-16 12:00:00', format='%Y-%m-%d %H:%M:%S')
    # pot_temp = pot_temp.sel(time=slice(start_time, end_time))

    # Plot the filled contours
    contourf = pot_temp.diff(variable_x).plot.contourf(ax=ax, x=variable_x, y=variable_y, levels=levels, cmap=pal1.cmap(),
                                                   add_colorbar=False, vmin=-2, vmax=2)

    # Plot the contour lines
    contour1 = pot_temp.plot.contour(ax=ax, x=variable_x, y=variable_y,
                                                                    levels=np.arange(np.round(pot_temp.min()),
                                                                                     np.round(pot_temp.max()), 1),
                                                                    colors='black', linewidths=0.5)   #.isel(time=slice(1, 100))
    contour5 = pot_temp.plot.contour(ax=ax, x=variable_x, y=variable_y,
                                                                    levels=np.arange(290, np.round(pot_temp.max()), 5),
                                                                    colors='black', linewidths=1.5)  #.isel(time=slice(1, 100))
    ax.clabel(contour5)
    if model in ["AROME", "ICON"]:
        ax.invert_yaxis()

    ax.set_xlim(start_time, end_time)
    # Add a colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('K hr$^{-1}$')
    ax.set_title(model + " potential temp time series")
    ax.set_ylabel(f"variable {variable_y}")
    ax.set_xlabel("")

    plt.savefig(confg.dir_PLOTS + model + "_pot_temp_timeseries_ibk.png", dpi=300)



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
    arome = read_in_arome.read_3D_variables_AROME(variables=["p", "th", "z"], method="sel", lat=lat_ibk, lon=lon_ibk)
    # pot_temp = arome.th.isel(nz=np.arange(40, 90))
    # plot_pot_temp_time_contours(pot_temp, model="AROME")

    temp = arome.temperature.where(arome.height <= 4000, drop=True)
    plot_temp_time_contours(temp, model="AROME")

def plot_icon():
    icon15 = read_icon_model_3D.read_icon_fixed_point_multiple_hours(day=15, hours=range(12, 14), lon=lon_ibk,
                                                                     lat=lat_ibk, variant="ICON")
    icon16 = read_icon_model_3D.read_icon_fixed_point_multiple_hours(day=16, hours=range(00, 13), lon=lon_ibk,
                                                                     lat=lat_ibk, variant="ICON")
    variables = ["th", "temp"]  # "temp", "pres", "u", "v", "w",
    icon = xr.concat([icon15[variables], icon16[variables]], dim="time")
    pot_temp = icon.th.isel(height=np.arange(40, 90))
    plot_pot_temp_time_contours(pot_temp, model="ICON")

def plot_icon2te():
    icon15_2te = read_icon_model_3D.read_icon_fixed_point_multiple_hours(day=15, hours=range(12, 24), lon=lon_ibk,
                                                                         lat=lat_ibk, variant="ICON2TE")
    icon16_2te = read_icon_model_3D.read_icon_fixed_point_multiple_hours(day=16, hours=range(00, 13), lon=lon_ibk,
                                                                         lat=lat_ibk, variant="ICON2TE")
    variables = ["th", "temp"]  # ["temp", "pressure", "pres", "u", "v", "w"]
    icon2te = xr.concat([icon15_2te[variables], icon16_2te[variables]], dim="time")
    icon2te_pot_temp = icon2te.th.isel(height=np.arange(40, 90))
    plot_pot_temp_time_contours(pot_temp=icon2te_pot_temp, model="ICON2TE")

def plot_ukmo():
    um = read_ukmo.read_ukmo_fixed_point(lat=lat_ibk, lon=lon_ibk)
    um_pot_temp = um.air_potential_temperature.isel(model_level_number=np.arange(0, 50))
    plot_pot_temp_time_contours(pot_temp=um_pot_temp, model="UKMO")

def plot_wrf():
    wrf = read_wrf_helen.read_wrf_fixed_point(lat=lat_ibk, lon=lon_ibk)
    wrf_pot_temp = wrf.th.isel(height=slice(0, 50))
    plot_pot_temp_time_contours(pot_temp=wrf_pot_temp, model="WRF")


def plot_hatpro():
    # hatpro = xr.open_dataset(f"{confg.hatpro_folder}/hatpro_merged.nc")
    # hatpro_temp = hatpro["temperature"].sel(height=slice(0, 4400))  # select up to 4.400 m
    # plot_temp_time_contours(temp=hatpro_temp, model="HATPRO")
    # hatpro

    # try with hatpro interpolated data
    hatpro = xr.open_dataset(f"{confg.hatpro_folder}/hatpro_interpolated_arome.nc")
    hatpro_pot_temp = hatpro["th"].where(hatpro["height"] <= 4000, drop=True)  # hatpro["th"].sel(height=slice(0, 4400))  # select up to 4.400 m
    plot_pot_temp_time_contours(pot_temp=hatpro_pot_temp, model="HATPRO")

if __name__ == '__main__':
    lat_ibk = 47.259998
    lon_ibk = 11.384167
    pal1 = diverging_hcl(palette="Blue-Red 2")

    matplotlib.use('Qt5Agg')  # Use the Qt5Agg backend for interactive plotting

    # plot_arome()

    #plot_icon()
    # plot_icon2te()

    # plot_ukmo()
    # plot_wrf()
    plot_hatpro()
    plt.show()
