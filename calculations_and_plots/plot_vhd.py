"""
domain calc and point calculation both work now, but they are different! Why?

This script should plot the timeseries of the VHD for all models and the change of spatial extent of a defined threshold
in time.

"""
import math
import importlib
from pathlib import Path

from bokeh.util.logconfig import level
import read_in_arome
import read_icon_model_3D
import read_ukmo
from calc_vhd import calc_vhd_single_point, select_pcgp_vhd, calc_vhd_full_domain, calc_vhd_single_point_main
import read_wrf_helen
import confg
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

importlib.reload(confg)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from colorspace import terrain_hcl, qualitative_hcl, sequential_hcl


def plot_vhds_point(vhd_arome, vhd_icon, vhd_icon2te, point_name=confg.ibk_uni["name"], vhd_origin="point"):
    """
    plots the VHD for a single point, which is already calced
    :param vhd_arome:
    :param vhd_icon:
    :param vhd_icon2te:
    :param point_name:
    :param vhd_origin: should be either ["point", "domain"]; stands for how the VHD is calculated that is used for plotting.
        either from full domain calculation or from reading single point timeseries...
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    (vhd_arome.vhd / 10**6).plot(ax=ax, label="AROME")
    (vhd_icon.vhd / 10**6).plot(ax=ax, label="ICON")
    (vhd_icon2te.vhd / 10**6).plot(ax=ax, label="ICON2TE")

    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.ylim(0.08, 0.35)
    plt.ylabel(r"valley heat deficit $[\frac{MJ}{m^2}]$")  # units are still pfusch...
    plt.title(f"VHD timeline for {point_name} via {vhd_origin} calc")
    plt.legend(loc='upper left')
    plt.savefig(confg.dir_PLOTS + "vhd_plots/" + f"vhd_model_comp_{point_name}_{vhd_origin}.svg")



def read_vhd_full_domain_and_plot_vhds_point(lat= confg.ibk_uni["lat"], lon=confg.ibk_uni["lon"], point_name=confg.ibk_uni["name"]):
    """
    reads only functions for reading VHD full domain, calcing PCGP (fct in calc_vhd.py) and plotting the VHDs

    Problem: calculation of VHD for single point with extra reading data & calcing gives a different VHD than precomputed
    VHD for full domain and the selecting the PCGP?! -> why?

    :return:
    """
    vhd_arome, vhd_icon, vhd_icon2te = select_pcgp_vhd(lat=lat, lon=lon, point_name=point_name)
    plot_vhds_point(vhd_arome=vhd_arome, vhd_icon=vhd_icon, vhd_icon2te=vhd_icon2te, point_name=point_name,
                    vhd_origin="domain")


def plot_vhd_full_domain(ds_extent, time, model="ICON"):
    """
    not used right now
    :param ds_extent:
    :param time:
    :param model:
    :return:
    """
    fig, axis = plt.subplots(figsize=(8, 5), subplot_kw={'projection': ccrs.Mercator()})  # , subplot_kw={'projection': ccrs.PlateCarree()}
    model_vhd = (ds_extent / 10**6).sel(lat=slice(confg.lat_min, confg.lat_max),
                  lon=slice(confg.lon_min, confg.lon_max)).plot(ax=axis, cmap=darkblue_hcl_rev,
                                                                transform=ccrs.Mercator(),
                                                                add_colorbar=False)
    cbar = fig.colorbar(model_vhd, ax=axis, orientation='vertical', pad=0.02)
    cbar.set_label("valley heat deficit [$J/m^2$]", rotation=90, labelpad=15)
    axis.add_feature(cfeature.BORDERS, linestyle=':')
    axis.add_feature(cfeature.LAKES, alpha=0.5)
    axis.add_feature(cfeature.RIVERS)

    plt.title(f"{model} valley heat deficit at {time} UTC")
    plt.xlabel("longitude [°E]")
    plt.ylabel("latitude [°N]")
    plt.savefig(confg.dir_PLOTS + "vhd_plots/" + f"{model}_VHD_{time}_UTC.svg")
    plt.show()


def plot_vhd_small_multiples(ds_extent, times, model="ICON"):
    """
    written by ChatGPT, but modified
    plots VHD as hourly data with small multiples, need revision because I now calced it 1/2 hourly
    :param ds_extent:
    :param times:
    :param model:
    :return:
    """

    n = len(times)
    cols = 6
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6),
                             subplot_kw={'projection': ccrs.Mercator()})
    norm = mpl.colors.Normalize(vmin=0, vmax=0.5)  # normalize the colorbar
    axes = axes.flatten()
    for i, time in enumerate(times):
        ax = axes[i]
        vhd = (ds_extent.sel(time=time) / 1e6).sel(lat=slice(confg.lat_min, confg.lat_max),
                                                   lon=slice(confg.lon_min, confg.lon_max))
        im = vhd.vhd.plot(ax=ax, cmap=darkblue_hcl_rev, transform=ccrs.Mercator(), add_colorbar=False)
        ax.set_title(f"{time.dt.strftime('%H:%M').item()}", y = 1.0, pad = -25)  # title above plot

        #ax.add_feature(cfeature.BORDERS, linestyle=':')
        #ax.add_feature(cfeature.LAKES, alpha=0.5)
        #ax.add_feature(cfeature.RIVERS)
        ax.set_xlabel("")
        ax.set_ylabel("")
    # fig.set
    # plt.title("valley heat deficit spatial evolution " + model)
    # Farbleiste nur einmal für alle Plots
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=darkblue_hcl_rev), cax=cbar_ax,
                 label="valley heat deficit [$MJ/m^2$]", ticks=np.arange(0, 0.5, 0.05))

    #fig.colorbar(im, ax=axes, orientation='vertical', pad=0.02, fraction=0.02).set_label(
    #"valley heat deficit [$J/m^2$]", rotation=90, labelpad=15)
    # plt.tight_layout()
    plt.savefig(confg.dir_PLOTS + "vhd_plots/" + f"{model}_VHD_small_multiples.png", dpi=600)
    plt.show()


if __name__ == '__main__':
    mpl.use('Qt5Agg')
    darkblue_hcl = sequential_hcl(palette="Blues 3")  # colors for slope profiles
    darkblue_hcl_rev = mcolors.ListedColormap(darkblue_hcl.colors()[::-1])
    # darkred_hcl = sequential_hcl(palette="Reds 3").colors()[4]
    # black_hcl = sequential_hcl(palette="Grays").colors()[0]
    # pal = sequential_hcl("Terrain")
    # calc_and_plot_vhds_point(lat=confg.ibk_uni["lat"], lon=confg.ibk_uni["lon"], point_name=confg.ibk_uni["name"])  # old stuff, prob überflüssig

    point_name, point_lat, point_lon = confg.kiefersfelden["name"], confg.kiefersfelden["lat"], confg.kiefersfelden["lon"]

    # via single point VHD calculation
    vhd_arome_single, vhd_icon_single, vhd_icon2te_single = calc_vhd_single_point_main(lat=point_lat, lon=point_lon,
                                                                  point_name=point_name)  # call main fct which calls others
    plot_vhds_point(vhd_arome=vhd_arome_single, vhd_icon=vhd_icon_single, vhd_icon2te=vhd_icon2te_single, point_name=point_name,
                    vhd_origin="point")

    # via full domain VHD calculation
    vhd_arome_domain, vhd_icon_domain, vhd_icon2te_domain = select_pcgp_vhd(lat=point_lat, lon=point_lon, point_name=point_name)
    plot_vhds_point(vhd_arome=vhd_arome_domain, vhd_icon=vhd_icon_domain, vhd_icon2te=vhd_icon2te_domain, point_name=point_name,
                    vhd_origin="domain")
    plt.show()


    # arome = read_in_arome.read_in_arome_fixed_time(time="2017-10-15T12:00:00", variables=["z"])  # variables=["p", "temp", "th", "z", "rho"]
    """
    vhd_arome = xr.open_dataset(confg.dir_AROME + "/AROME_vhd_full_domain_full_time.nc")
    vhd_icon = xr.open_dataset(confg.icon_folder_3D + "/ICON_vhd_full_domain_full_time.nc")
    vhd_icon2te = xr.open_dataset(confg.icon2TE_folder_3D + "/ICON2TE_vhd_full_domain_full_time.nc")

    plot_vhd_small_multiples(vhd_arome, vhd_arome.time, model="AROME")
    plot_vhd_small_multiples(vhd_icon, vhd_icon.time, model="ICON")
    plot_vhd_small_multiples(vhd_icon2te, vhd_icon2te.time, model="ICON2TE")
    """