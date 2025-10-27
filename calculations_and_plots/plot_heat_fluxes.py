"""
sensible heat flux spatial plots:
during night AROME is mostly negative, during
now added lowest level wind arrows. AROME looks strange? WRF much better up-valley flow?
espc. AROME has low values near the surface, further up they increase (see Hannes' plots...)

sunset at 16:25 UTC: temp falls already since ~15:30? => heat flux turns around at sunset
WRF hfs: UPWARD HEAT FLUX AT THE SURFACE
"""

import confg
import read_in_arome
import read_icon_model_3D
import read_ukmo
import read_wrf_helen

import math
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from colorspace import terrain_hcl, qualitative_hcl, sequential_hcl, diverging_hcl
import numpy as np
import matplotlib.pyplot as plt
from metpy.plots.declarative import BarbPlot
import metpy.units as units
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.dates as mdates
from datetime import datetime


def plot_heatflux(ds):
    """
    Plot sensible heat flux with topography contours on a map projection.

    :param ds: xarray Dataset with variables 'lon', 'lat', 'hfs', 'z_unstag', and 'time'
    """
    projection = ccrs.Mercator()
    # Set up figure with projection
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": projection})

    # Add map features
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # Plot heat flux (color mesh)
    pcm = ax.pcolormesh(ds.lon.values, ds.lat.values, ds.hfs.values, cmap=colormap,
                        shading="auto", transform=projection)  # input coords are lon/lat

    # Add contours of topography
    z = ds.z_unstag.isel(height=0)
    thin_levels = list(range(0, int(z.max()) + 100, 100))
    # thick_levels = list(range(500, int(z.max()) + 500, 500))

    ax.contour(ds.lon.values, ds.lat.values, z, ls=thin_levels, colors="k", linewidths=0.3, transform=projection)
    # ax.contour(ds.lon.values, ds.lat.values, z, levels=thick_levels, colors="k", linewidths=1.0, transform=projection)

    cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", shrink=0.7, vmin=-250, vmax=250, label="Sensible Heat Flux [W m$^{-2}$]")
    time_val = ds.time.values
    # Convert to datetime and format
    time_val = pd.to_datetime(time_val)
    time_str = time_val.strftime("%dth %H:%M")
    ax.set_title(f"Surface Heat Flux (hfs) at {time_str}", fontsize=14)
    plt.xlim([confg.lon_hf_min, confg.lon_hf_max])
    plt.ylim([confg.lat_hf_min, confg.lat_hf_max])

    plt.tight_layout()
    plt.show()

# assuming you have lon_hf_min/max, lat_hf_min/max defined

def make_times(start_day=15, start_hour=14, start_minute=0,
               end_day=16, end_hour=10, end_minute=0,
               freq="2h"):
    """
    a bit useless, just creates a pd daterange (altough functions for read in only take day, hour & min...)
    :param start_day:
    :param start_hour:
    :param start_minute:
    :return:
    """

    start_dt = datetime(2017, 10, start_day, start_hour, start_minute)
    end_dt = datetime(2017, 10, end_day, end_hour, end_minute)
    times = pd.date_range(start=start_dt, end=end_dt, freq=freq)
    return times

def read_wrf_for_times(times, variables):
    """
    reads for all given times the WRF data with the given variables.

    :param times: pandas.DatetimeIndex
    :param variables:
    """
    ds_list = []
    for t in times:
        ds = read_wrf_helen.read_wrf_fixed_time(day=t.day, hour=t.hour, min=t.minute, variables=variables)
        if "time" in getattr(ds, "dims", []):
            ds = ds.isel(time=0)
        ds_list.append(ds)
    wrf = xr.concat(ds_list, dim="time")
    return wrf


def plot_small_multiples(ds, model="WRF", **kwargs):
    """
    plots small multiples of sensible heat flux with topography contours and wind barbs
    difficulties: for AROME we have the heat flux in 2D variables, not in the 3D ones => needs extra handling & read

    """
    projection = ccrs.Mercator()  # use a mercator projection per default
    nplots, ncols = len(ds.time), 3
    nrows = int((nplots + ncols - 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), layout="compressed", subplot_kw={'projection': projection})
    # norm = mpl.colors.Normalize(vmin=0, vmax=0.5)  # normalize the colorbar
    axes = axes.flatten()
    for i, time in enumerate(times):
        ax = axes[i]
        ds_sel = ds.sel(time=time)  # .sel(lat=slice(confg.lat_min, confg.lat_max), lon=slice(confg.lon_min, confg.lon_max))
        im = ax.pcolormesh(ds_sel.lon.values, ds_sel.lat.values, ds_sel.hfs.values, cmap=colormap, vmin=-100, vmax=100,
                           transform=projection)  # HF field
        step = 2
        if model=="WRF":  # for WRF I need extra handling cause u/v are 3D ...
            z = ds_sel.z_unstag.isel(height=0)
            u, v = ds_sel.sel(height=1).u.values[::step, ::step], ds_sel.sel(height=1).v.values[::step, ::step]

        elif model=="AROME":
            z = ds_sel.hgt
            # lat, lon = ds_sel.lat.values[::step], ds_sel.lon.values[::step]
            u, v = ds_sel.u.values[::step, ::step], ds_sel.v.values[::step, ::step]

        lat, lon = ds_sel.lat.values[::step], ds_sel.lon.values[::step]
        # add wind quivers (arrows): what length is which speed?
        quiver = ax.quiver(x=lon, y=lat, u=u, v=v, scale=40, scale_units="inches", transform=projection)

        thin_levels = list(range(0, int(z.max()) + 100, 250))
        ax.contour(ds_sel.lon.values, ds_sel.lat.values, z.values, levels=thin_levels, colors="k", linewidths=0.3,
                   transform=projection)  # height contours

        ax.add_feature(cfeature.BORDERS, linewidth=0.5, transform=projection)
        ax.text(0.1, 0.8, f"{time.hour:02d}h", transform=ax.transAxes,  # create hour text label w white box
                fontsize=10, fontweight="bold", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        ax.set_xlabel(""), ax.set_ylabel("")
        ax.set_xlim([confg.lon_hf_min, confg.lon_hf_max]), ax.set_ylim([confg.lat_hf_min, confg.lat_hf_max])

    fig.subplots_adjust(bottom=0.15)  # Platz für die Legende schaffen
    qk = ax.quiverkey(quiver, X=1.4, Y=0.25, U=5, label='5 m/s', labelpos='E', coordinates='axes')
    cbar = plt.colorbar(im, ax = axes, label=model + "sensible heat flux at surface [$W/m^2$]")
    cbar.ax.tick_params(size=0)
    # plt.tight_layout()
    plt.savefig(confg.dir_PLOTS + "heat_flux/" + f"heat_flux_{model}_small_multiples.png", dpi=500)


def plot_heatflux_small_multiples(start_day=15, start_hour=14,
                                  end_day=16, end_hour=10,
                                  variables=["hfs", "z", "z_unstag"]):
    """
    deprecated
    Create small multiples of sensible heat flux from WRF output.
    """
    projection = ccrs.Mercator()
    # I wouldn't need a datetime for the input, but it's easiest programmed...
    start_time = datetime(2017, 10, start_day, start_hour, 0)  # dummy year/month
    end_time = datetime(2017, 10, end_day, end_hour, 0)
    times = pd.date_range(start=start_time, end=end_time, freq="2h")  #  30min

    nplots, ncols = len(times), 4
    nrows = int((nplots + ncols - 1) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 3*nrows),
                             subplot_kw={"projection": projection})
    axes = axes.flatten()
    for i, t in enumerate(times):
        # --- read dataset ---
        ds = read_wrf_helen.read_wrf_fixed_time(day=t.day, hour=t.hour, min=t.minute, variables=variables)
        ds = ds.isel(time=0)
        ax = axes[i]

        ax.add_feature(cfeature.BORDERS, linewidth=0.5, transform=projection)

        # Heat flux field
        pcm = ax.pcolormesh(ds.lon.values, ds.lat.values, ds.hfs.values,
                            cmap=colormap, shading="auto", transform=projection)

        # Topography contours (thin only)
        z = ds.z_unstag.isel(height=0)
        thin_levels = list(range(0, int(z.max()) + 100, 250))
        ax.contour(ds.lon.values, ds.lat.values, z, levels=thin_levels, colors="k", linewidths=0.3,
                   transform=projection)
        # Zoom into domain
        ax.set_xlim([confg.lon_hf_min, confg.lon_hf_max]), ax.set_ylim([confg.lat_hf_min, confg.lat_hf_max])

        # Remove ticks
        ax.set_xticks([]), ax.set_yticks([])

        # Annotate with hour only
        ax.text(0.05, 0.9, f"{t.hour:02d}h",
                transform=ax.transAxes,
                fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    # Remove unused axes (if any)
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    # Shared colorbar
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])  # [left, bottom, width, height]
    fig.colorbar(pcm, cax=cbar_ax, orientation="horizontal",
                 label="Sensible Heat Flux [W m$^{-2}$]")

    fig.suptitle("WRF heat flux", fontsize=14)
    plt.tight_layout()
    plt.savefig(confg.dir_PLOTS + "heat_flux/" + "heat_flux_wrf_small_multiples.png", dpi=600)
    plt.show()


if __name__ == '__main__':
    mpl.use('Qt5Agg')
    colormap = diverging_hcl(palette="Blue-Red 2").cmap()
    # wrf_extent = read_wrf_helen.read_wrf_fixed_time(day=15, hour=15, min=0, variables=["hfs", "z", "z_unstag"])  #  "p", "temp", "th",
    # plot_heatflux(ds=wrf_extent.isel(time=0))

    times = make_times(start_day=15, start_hour=14, start_minute=0, end_day=16, end_hour=12, end_minute=0, freq="2h")
    # i saved lowest level of u&v 3D var as "u_v_from_3d"
    arome2d = read_in_arome.read_2D_variables_AROME(variableList=["hfs", "hgt", "lfs", "u_v_from_3d"],  # reads all timestamps
                                                    lon=slice(confg.lon_min, confg.lon_max),
                                                    lat=slice(confg.lat_min, confg.lat_max), slice_lat_lon=True)
    arome2d = arome2d.sel(time=times)  # select only the times we want to plot
    plot_small_multiples(ds=arome2d, model="AROME")

    wrf_hf = read_wrf_for_times(times=times, variables=["hfs", "z", "z_unstag", "u", "v"])
    plot_small_multiples(ds=wrf_hf, model="WRF")
    plt.show()

    #plot_heatflux_small_multiples(start_day=15, start_hour=14, end_day=16, end_hour=10,
    #                              variables=["hfs", "z", "z_unstag"])


