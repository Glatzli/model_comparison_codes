"""
This script should plot the timeseries of the VHD for all models and the change of spatial extent of a defined threshold
in time.
An hourly small multiple plot of the VHD over the full valley extent was done, with a contour line at 80% of the
maximum VHD, to show the extent of the maximum in each timestep.

(A lot could be programmed shorter with loops through all models etc, but the effort isn't worth it for how often I will use
this...)

"""
import datetime
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


def plot_vhds_point(vhd_arome, vhd_icon, vhd_icon2te, vhd_um, vhd_wrf, point_name=confg.ibk_uni["name"],
                    vhd_origin="point", vhd_hatpro=None, vhd_radio=None, *args, **kwargs):
    """
    plots the VHD for a single point, which is already calced
    :param vhd_arome:
    :param vhd_icon:
    :param vhd_icon2te:
    :param vhd_um:
    :param vhd_hatpro: optional HATPRO (only for Ibk points)
    :param point_name:
    :param vhd_origin: should be either ["point", "domain"]; stands for how the VHD is calculated that is used for plotting.
        either from full domain calculation or from reading single point timeseries...
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    if vhd_origin == "point":  # add used lat & lon for each plot type (single point calc & domain calc)
        (vhd_arome.vhd / 10**6).plot(ax=ax, label=f"AROME", color=qualitative_colors[0], linewidth=2)  # for debugging: lat {vhd_arome.lat.item():.3f}, lon {vhd_arome.lon.item():.3f}
        (vhd_icon.vhd / 10**6).plot(ax=ax, label=f"ICON", color=qualitative_colors[2], linewidth=2)  #  lat {vhd_icon.lat.item():.3f}, lon {vhd_icon.lon.item():.3f}
        (vhd_icon2te.vhd / 10**6).plot(ax=ax, label=f"ICON2TE", color=qualitative_colors[2], linewidth=2, linestyle="dashed")  #  lat {vhd_icon2te.lat.item():.3f}, lon {vhd_icon2te.lon.item():.3f}
        (vhd_um.vhd/ 10**6).plot(ax=ax, label=f"UM", color=qualitative_colors[4], linewidth=2)  #  lat {vhd_um.lat.item():.3f}, lon {vhd_um.lon.item():.3f}
        (vhd_wrf.vhd / 10 **6).plot(ax=ax, label=f"WRF", color=qualitative_colors[6], linewidth=2)  #  lat {vhd_wrf.lat.item():.3f}, lon {vhd_wrf.lon.item():.3f}
        if "ibk" in point_name:  # for points in ibk add HATPRO & radiosonde data
            (vhd_hatpro.vhd / 10 ** 6).plot(ax=ax, label=f"HATPRO (uni)", color=qualitative_colors[8], linewidth=2)
            # ax.scatter(datetime.datetime(2017, 10, 16, 4, 0, 0), (vhd_radio.vhd / 10 ** 6),
            #         label="Radiosonde (airport)", marker="*")
            # (vhd_radio.vhd / 10 ** 6).plot(ax=ax, label=f"Radiosonde (airport)")
    elif vhd_origin == "domain":
        (vhd_arome.vhd / 10**6).plot(ax=ax, label=f"AROME", color=qualitative_colors[0], linewidth=2)
        (vhd_icon.vhd / 10**6).plot(ax=ax, label=f"ICON", color=qualitative_colors[2], linewidth=2)
        (vhd_icon2te.vhd / 10**6).plot(ax=ax, label=f"ICON2TE", color=qualitative_colors[2], linestyle="dashed", linewidth=2)
        (vhd_um.vhd / 10**6).plot(ax=ax, label=f"UM", color=qualitative_colors[4], linewidth=2)
        (vhd_wrf.vhd / 10 ** 6).plot(ax=ax, label=f"WRF", color=qualitative_colors[6], linewidth=2)

    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.ylim(0.08, 0.6)
    plt.ylabel(r"valley heat deficit $[\frac{MJ}{m^2}]$")
    plt.grid()
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
    vhd_arome, vhd_icon, vhd_icon2te, vhd_um, vhd_wrf = select_pcgp_vhd(lat=lat, lon=lon, point_name=point_name)
    plot_vhds_point(vhd_arome=vhd_arome, vhd_icon=vhd_icon, vhd_icon2te=vhd_icon2te, vhd_um=vhd_um, vhd_wrf=vhd_wrf,
                    point_name=point_name, vhd_origin="domain")


def plot_vhd_full_domain(ds_extent, time, model="ICON"):
    """ deprecated
    not used right now
    :param ds_extent:
    :param time:
    :param model:
    :return:
    """
    fig, axis = plt.subplots(figsize=(8, 5), subplot_kw={'projection': ccrs.Mercator()})  # , subplot_kw={'projection': ccrs.PlateCarree()}
    model_vhd = (ds_extent / 10**6).sel(lat=slice(confg.lat_min_vhd, confg.lat_max_vhd),
                                        lon=slice(confg.lon_min_vhd, confg.lon_max_vhd)).plot(ax=axis, cmap=darkblue_hcl_rev,
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


def plot_vhd_small_multiples(ds_extent, model="ICON"):
    """
    written by ChatGPT, but modified
    plots VHD as hourly data with small multiples, need revision because I now calced it 1/2 hourly
    :param ds_extent:
    :param times:
    :param model:
    :return:
    """
    projection = ccrs.Mercator()
    ds_extent = ds_extent.isel(time=slice(4, 100, 4))
    nplots, ncols = len(ds_extent.time), 3
    nrows = int((nplots + ncols - 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6), layout="compressed", subplot_kw={'projection': projection})
    # norm = mpl.colors.Normalize(vmin=0, vmax=0.5)  # normalize the colorbar
    axes = axes.flatten()
    # levels = np.linspace(0.05, 0.36, 10)  # try without distinct levels
    for i, time in enumerate(ds_extent.time):
        ax = axes[i]
        ds_extent_sel = (ds_extent.sel(time=time) / 1e6).sel(lat=slice(confg.lat_min_vhd, confg.lat_max_vhd),
                                                   lon=slice(confg.lon_min_vhd, confg.lon_max_vhd))
        im = ds_extent_sel.vhd.plot(ax=ax, cmap=darkblue_hcl_cont_rev, transform=projection, vmin=0.05, vmax=0.36,
                                    add_colorbar=False)

        # shows extent of max: plot a contour line for 80% of the maximum of current VHD:
        contours = [ds_extent_sel.vhd.max().item() * 0.8]
        cs = ax.contour(ds_extent_sel.lon, ds_extent_sel.lat, ds_extent_sel.vhd.values, levels=contours,
                        colors="k", linewidths=0.5, transform=projection)

        # maybe add topography contours? would need height info in dataset...
        ax.text(0.1, 0.8, f"{time.dt.hour.item() :02d}h", transform=ax.transAxes,  # create hour text label w white box
                fontsize=10, fontweight="bold", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
    # plt.title("valley heat deficit spatial evolution " + model)
    # add 1 colorbar for all plots
    # cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=darkblue_hcl_rev), cax=cbar_ax,
    #             label=model + " valley heat deficit [$MJ/m^2$]", ticks=np.arange(0, 0.5, 0.05))
    cbar = plt.colorbar(im, ax = axes, label=model + " valley heat deficit [$MJ/m^2$]")  # , ticks=np.round(levels, 2)
    cbar.ax.tick_params(size=0)
    #fig.colorbar(im, ax=axes, orientation='vertical', pad=0.02, fraction=0.02).set_label(
    #"valley heat deficit [$J/m^2$]", rotation=90, labelpad=15)

    # plt.tight_layout()
    plt.savefig(confg.dir_PLOTS + "vhd_plots/" + f"{model}_VHD_small_multiples.png", dpi=600)


if __name__ == '__main__':
    mpl.use('Qt5Agg')
    darkblue_hcl = sequential_hcl(palette="Blues 3")  # colors for slope profiles
    darkblue_hcl_rev = mcolors.ListedColormap(darkblue_hcl.colors()[::-1])
    darkblue_hcl_cont_rev = darkblue_hcl.cmap().reversed()
    # darkred_hcl = sequential_hcl(palette="Reds 3").colors()[4]
    # black_hcl = sequential_hcl(palette="Grays").colors()[0]
    # pal = sequential_hcl("Terrain")
    qualitative_colors = qualitative_hcl(palette="Dark 3").colors()
    # calc_and_plot_vhds_point(lat=confg.ibk_uni["lat"], lon=confg.ibk_uni["lon"], point_name=confg.ibk_uni["name"])  # old stuff, prob überflüssig

    # define point for which VHD timeline should be calculated
    point = confg.telfs
    """
    # via single point VHD calculation
    (vhd_arome_single, vhd_icon_single, vhd_icon2te_single,
     vhd_um_single, vhd_wrf_single, vhd_hatpro, vhd_radio) = calc_vhd_single_point_main(lat=point["lat"], lon=point["lon"],
                                                                  point_name=point["name"])  # call main fct which calls others
    plot_vhds_point(vhd_arome=vhd_arome_single, vhd_icon=vhd_icon_single, vhd_icon2te=vhd_icon2te_single,
                    vhd_um=vhd_um_single, vhd_wrf=vhd_wrf_single, point_name=point["name"], vhd_origin="point",
                    vhd_hatpro=vhd_hatpro, vhd_radio=vhd_radio)

    # via full domain VHD calculation
    vhd_arome_domain, vhd_icon_domain, vhd_icon2te_domain, vhd_um_domain, vhd_wrf_domain = select_pcgp_vhd(lat=point["lat"], lon=point["lon"])
    plot_vhds_point(vhd_arome=vhd_arome_domain, vhd_icon=vhd_icon_domain, vhd_icon2te=vhd_icon2te_domain,
                    vhd_um=vhd_um_domain, vhd_wrf=vhd_wrf_domain, point_name=point["name"], vhd_origin="domain")

    """
    vhd_arome = xr.open_dataset(confg.dir_AROME + "/AROME_vhd_full_domain_full_time.nc")
    vhd_icon = xr.open_dataset(confg.icon_folder_3D + "/ICON_vhd_full_domain_full_time.nc")
    vhd_icon2te = xr.open_dataset(confg.icon2TE_folder_3D + "/ICON2TE_vhd_full_domain_full_time.nc")
    vhd_um = xr.open_dataset(confg.ukmo_folder + "/UM_vhd_full_domain_full_time.nc")
    vhd_wrf = xr.open_dataset(confg.wrf_folder + "/WRF_vhd_full_domain_full_time.nc")

    plot_vhd_small_multiples(vhd_arome, model="AROME")
    plot_vhd_small_multiples(vhd_icon, model="ICON")
    plot_vhd_small_multiples(vhd_icon2te, model="ICON2TE")
    plot_vhd_small_multiples(vhd_um, model="UM")
    plot_vhd_small_multiples(vhd_wrf, model="WRF")

    plt.show()

