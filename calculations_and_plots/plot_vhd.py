"""
This script should plot the timeseries of the VHD for all models and the change of spatial extent of a defined threshold
in time.

"""

import importlib
import read_in_arome
import read_icon_model_3D
import read_ukmo
from calc_vhd import calc_vhd_single_point
import read_wrf_helen
import confg
import xarray as xr

# importlib.reload(calc_vhd_single_point)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_vhds():
    fig, ax = plt.subplots(figsize=(10, 6))
    (vhd_arome / 10**6).plot(ax=ax, label="AROME")
    (vhd_icon / 10**6).plot(ax=ax, label="ICON")
    (vhd_icon2te / 10**6).plot(ax=ax, label="ICON2TE")

    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.ylabel(r"valley heat deficit $[\frac{MJ}{m^2}]$")  # units are still pfusch...
    plt.title("")
    plt.legend(loc='upper left')
    plt.savefig(confg.dir_PLOTS + "vhd_model_comp_ibk.svg")
    plt.show()


"""
def plot_vhds(vhd, labels=None, ylabel="Wert [$10^6$]", title="Zeitreihenvergleich"):

    plt.figure(figsize=(12, 6))
    for i, da in enumerate(dataarrays):
        label = labels[i] if labels is not None else f"Serie {i+1}"
        plt.plot(da.time.values, da.values / 1e6, marker="o", label=label)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xlabel("Zeit")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
"""


if __name__ == '__main__':
    matplotlib.use('Qt5Agg')

    # first open already saved point-timeseries for ibk point
    arome_timeseries = xr.open_dataset(confg.dir_AROME + f"/arome_lat47.25_lon11.39_timeseries.nc")
    icon_timeseries = xr.open_dataset \
        (confg.icon_folder_3D + f"/icon_lat47.26_lon11.38_timeseries.nc"  )# read saved AROME timeseries
    icon2te_timeseries = xr.open_dataset \
        (confg.icon2TE_folder_3D + f"/icon_2te_lat47.26_lon11.38_timeseries.nc")  # read saved ICON timeseries
    # calc VHD for model data for PCGP
    vhd_arome = calc_vhd_single_point(arome_timeseries, model="AROME")
    vhd_icon = calc_vhd_single_point(icon_timeseries, model="ICON")
    vhd_icon2te = calc_vhd_single_point(icon2te_timeseries, model="ICON")
    plot_vhds()