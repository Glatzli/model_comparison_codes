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
# importlib.reload(read_wrf_helen)
import confg
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from colorspace import diverging_hcl



def plot_temp_time_contours(pot_temp, model="AROME"):
    match model:
        case "AROME":
            variable_x="time"; variable_y = "nz"
        case ("ICON" | "ICON2TE"):  # same for ICON2TE
            variable_x="time"; variable_y = "height"
        case "UKMO":
            variable_x="time"; variable_y = "height"
        case "WRF":
            variable_x="time"; variable_y = "height"

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the filled contours
    contourf = pot_temp.diff(variable_x).plot.contourf(ax=ax, x=variable_x, y=variable_y, levels=20, cmap=pal1.cmap(),
                                                   add_colorbar=False, vmin=-2, vmax=2)

    # Plot the contour lines
    contour1 = pot_temp.isel(time=slice(1, 100)).plot.contour(ax=ax, x=variable_x, y=variable_y,
                                                                    levels=np.arange(np.round(pot_temp.min()),
                                                                                     np.round(pot_temp.max()), 1),
                                                                    colors='black', linewidths=0.5)
    contour5 = pot_temp.isel(time=slice(1, 100)).plot.contour(ax=ax, x=variable_x, y=variable_y,
                                                                    levels=np.arange(290, np.round(pot_temp.max()), 5),
                                                                    colors='black', linewidths=1.5)
    ax.clabel(contour5)
    if model in ["AROME", "ICON"]:
        ax.invert_yaxis()
    # Add a colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('K hr$^{-1}$')
    ax.set_title(model + " pot temp time series")
    ax.set_ylabel(f"variable {variable_y}");
    ax.set_xlabel("")
    plt.savefig(confg.dir_PLOTS + model + " temp_timeseries_ibk.png", dpi=300)


def plot_arome():
    arome = read_in_arome.read_3D_variables_AROME(variables=["th", "z", "p"], method="sel", lat=lat_ibk, lon=lon_ibk)
    pot_temp = arome.th.isel(nz=np.arange(40, 90))
    plot_temp_time_contours(pot_temp, model="AROME")

def plot_icon():
    icon15 = read_icon_model_3D.read_icon_fixed_point_multiple_hours(day=15, hours=range(12, 14), lon=lon_ibk,
                                                                     lat=lat_ibk, variant="ICON")
    icon16 = read_icon_model_3D.read_icon_fixed_point_multiple_hours(day=16, hours=range(00, 13), lon=lon_ibk,
                                                                     lat=lat_ibk, variant="ICON")
    variables = ["th", "temp"]  # "temp", "pres", "u", "v", "w",
    icon = xr.concat([icon15[variables], icon16[variables]], dim="time")
    pot_temp = icon.th.isel(height=np.arange(40, 90))
    plot_temp_time_contours(pot_temp, model="ICON")

def plot_icon2te():
    icon15_2te = read_icon_model_3D.read_icon_fixed_point_multiple_hours(day=15, hours=range(12, 24), lon=lon_ibk,
                                                                         lat=lat_ibk, variant="ICON2TE")
    icon16_2te = read_icon_model_3D.read_icon_fixed_point_multiple_hours(day=16, hours=range(00, 13), lon=lon_ibk,
                                                                         lat=lat_ibk, variant="ICON2TE")
    variables = ["th", "temp"]  # ["temp", "pressure", "pres", "u", "v", "w"]
    icon2te = xr.concat([icon15_2te[variables], icon16_2te[variables]], dim="time")
    icon2te_pot_temp = icon2te.th.isel(height=np.arange(40, 90))
    plot_temp_time_contours(pot_temp=icon2te_pot_temp, model="ICON2TE")

def plot_ukmo():
    um = read_ukmo.read_ukmo_fixed_point(lat=lat_ibk, lon=lon_ibk)
    um_pot_temp = um.air_potential_temperature.isel(model_level_number=np.arange(0, 50))
    plot_temp_time_contours(pot_temp=um_pot_temp, model="UKMO")

def plot_wrf():
    wrf = read_wrf_helen.read_wrf_fixed_point(lat=lat_ibk, lon=lon_ibk)
    wrf_pot_temp = wrf.th.isel(height=slice(0, 50))
    plot_temp_time_contours(pot_temp=wrf_pot_temp, model="WRF")


if __name__ == '__main__':
    lat_ibk = 47.259998
    lon_ibk = 11.384167
    pal1 = diverging_hcl(palette="Blue-Red 2")

    matplotlib.use('Qt5Agg')  # Use the Qt5Agg backend for interactive plotting

    # plot_arome()

    plot_icon()
    # plot_icon2te()

    # plot_ukmo()
    # plot_wrf()
