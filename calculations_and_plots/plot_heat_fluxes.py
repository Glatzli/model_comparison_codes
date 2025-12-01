"""
sensible heat flux spatial plots:
during night AROME is mostly negative, during
now added lowest level wind arrows. AROME looks strange? WRF much better up-valley flow?
espc. AROME has low values near the surface, further up they increase (see Hannes' plots...)

sunset at 16:25 UTC: temp falls already since ~15:30? => heat flux turns around at sunset
WRF hfs: UPWARD HEAT FLUX AT THE SURFACE
"""

from datetime import datetime
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
from colorspace import diverging_hcl

import confg
import read_in_arome
import read_wrf_helen


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
    pcm = ax.pcolormesh(ds.lon.values, ds.lat.values, ds.hfs.values, cmap=colormap, shading="auto",
                        transform=projection)  # input coords are lon/lat

    # Add contours of topography
    z = ds.z_unstag.isel(height=0)
    thin_levels = list(range(0, int(z.max()) + 100, 100))
    # thick_levels = list(range(500, int(z.max()) + 500, 500))

    ax.contour(ds.lon.values, ds.lat.values, z, ls=thin_levels, colors="k", linewidths=0.3, transform=projection)
    # ax.contour(ds.lon.values, ds.lat.values, z, levels=thick_levels, colors="k", linewidths=1.0, transform=projection)

    cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", shrink=0.7, vmin=-250, vmax=250,
                        label="Sensible Heat Flux [W m$^{-2}$]")
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

def make_times(start_day=15, start_hour=14, start_minute=0, end_day=16, end_hour=10, end_minute=0, freq="2h"):
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


def plot_small_multiples(ds, model="WRF", variable="hfs", vmin=None, vmax=None, **kwargs):
    """
    plots small multiples of heat budget variables with topography contours and wind barbs

    :param ds: xarray Dataset with the heat budget variable, topography, and optionally wind
    :param model: Model name (WRF or AROME)
    :param variable: Variable name to plot (hfs, lfs, lwd, lwu, swd, swu)
    :param vmin: Minimum value for colorbar (if None, will be determined automatically)
    :param vmax: Maximum value for colorbar (if None, will be determined automatically)
    """
    # Variable metadata for labels and colormaps
    var_metadata = {
        "hfs": {"label": "Sensible Heat Flux", "cmap": "RdBu_r", "symmetric": True},
        "lfs": {"label": "Latent Heat Flux", "cmap": "RdBu_r", "symmetric": True},
        "lwd": {"label": "Downward Longwave Flux", "cmap": "YlOrRd", "symmetric": False},
        "lwu": {"label": "Upward Longwave Flux", "cmap": "YlOrRd", "symmetric": False},
        "swd": {"label": "Downward Shortwave Flux", "cmap": "YlOrRd", "symmetric": False},
        "swu": {"label": "Upward Shortwave Flux", "cmap": "YlOrRd", "symmetric": False},
    }

    metadata = var_metadata.get(variable, {"label": variable, "cmap": "viridis", "symmetric": False})

    # Use diverging colormap if specified, otherwise use the metadata colormap
    if metadata["symmetric"]:
        cmap = diverging_hcl(palette="Blue-Red 2").cmap()
    else:
        cmap = plt.colormaps[metadata["cmap"]]

    projection = ccrs.Mercator()  # use mercator projection per default
    nplots, ncols = len(ds.time), 3
    nrows = int((nplots + ncols - 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), subplot_kw={'projection': projection})
    axes = axes.flatten()

    for i, time in enumerate(ds.time.values):
        ax = axes[i]
        ds_sel = ds.sel(time=time)

        # Plot the selected variable
        var_data = ds_sel[variable].values
        im = ax.pcolormesh(ds_sel.lon.values, ds_sel.lat.values, var_data, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=projection)

        # --- select topography & subsetted wind data (use only every 2nd grid point for wind)
        # first select
        step = 2  # plot only every 2nd grid point wind arrow
        if model == "WRF":
            z = ds_sel.z_unstag.isel(height=0)
            # Only add wind arrows if u and v are available
            if "u" in ds_sel and "v" in ds_sel:
                u, v = ds_sel.sel(height=1).u.values[::step, ::step], ds_sel.sel(height=1).v.values[::step, ::step]
            else:
                u, v = None, None
        elif model == "AROME":
            z = ds_sel.hgt
            # Only add wind arrows if u and v are available
            if "u" in ds_sel and "v" in ds_sel:
                u, v = ds_sel.u.values[::step, ::step], ds_sel.v.values[::step, ::step]
            else:
                u, v = None, None

        # --- plot topo contours
        levels_thin = np.arange(0, 3500, 100)  # same as in plot_topo_comparison.py
        ax.contour(ds_sel.lon.values, ds_sel.lat.values, z.values, levels=levels_thin, colors="k", linewidths=0.3,
                   transform=projection)

        # Add wind quivers only if wind data is available
        lat, lon = ds_sel.lat.values[::step], ds_sel.lon.values[::step]
        if u is not None and v is not None:  # only added if there's meaningful wind data
            quiver = ax.quiver(x=lon, y=lat, u=u, v=v, scale=40, scale_units="inches", transform=projection)

        ax.add_feature(cfeature.BORDERS, linewidth=0.5, transform=projection)

        # Format timestamp
        time_pd = pd.to_datetime(time)
        ax.text(0.1, 0.8, f"{time_pd.hour:02d}h", transform=ax.transAxes,
                fontsize=10, fontweight="bold", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        ax.set_xlabel(""), ax.set_ylabel("")
        ax.set_xlim([confg.lon_hf_min, confg.lon_hf_max]), ax.set_ylim([confg.lat_hf_min, confg.lat_hf_max])

    # Remove unused axes
    # for j in range(i + 1, len(axes)):
    #   fig.delaxes(axes[j])

    # Add quiver key only if wind data was plotted
    if u is not None and v is not None:
        qk = ax.quiverkey(quiver, X=1.4, Y=0.25, U=5, label='5 m/s', labelpos='E', coordinates='axes')

    # Place colorbar on the right edge of the figure
    cbar = plt.colorbar(im, ax=axes[:i+1], label=f"{model} {metadata['label']} [$W/m^2$]", orientation='vertical')
                        # , fraction=0.046, pad=0.04)
    cbar.ax.tick_params(size=0)

    plt.tight_layout()
    plt.savefig(os.path.join(confg.dir_PLOTS, "heat_flux", f"{variable}_{model}_small_multiples.png"), dpi=500)
    # plt.close(fig)


def plot_all_heat_budget_variables(arome_ds, wrf_ds, times):
    """
    Plot all heat budget variables for both AROME and WRF models.

    :param arome_ds: AROME dataset with all heat budget variables
    :param wrf_ds: WRF dataset with all heat budget variables
    :param times: Time range to plot
    """
    # Define uniform colorbar ranges for each variable (same for both models)
    var_ranges = {
        "hfs": {"vmin": -100, "vmax": 100},    # Sensible heat flux (can be negative/positive)
        "lfs": {"vmin": -50, "vmax": 150},     # Latent heat flux (mostly positive)
        "lwd": {"vmin": 200, "vmax": 400},     # Downward longwave (always positive)
        "lwu": {"vmin": 50, "vmax": 450},     # Upward longwave (always positive)
        "swd": {"vmin": 0, "vmax": 800},       # Downward shortwave (0 at night)
        "swu": {"vmin": 0, "vmax": 200},       # Upward shortwave (reflected, 0 at night)
    }

    variables_to_plot = ["hfs", "lfs"]  #, "lwd", "lwu", "swd", "swu"]

    print(f"\n{'='*70}")
    print(f"Plotting heat budget variables for AROME and WRF")
    print(f"{'='*70}\n")

    for var in variables_to_plot:
        print(f"  Processing {var}...")

        # Check if variable exists in datasets
        if var in arome_ds:
            print(f"    Plotting AROME {var}...")
            plot_small_multiples(ds=arome_ds.sel(time=times), model="AROME", variable=var,
                               vmin=var_ranges[var]["vmin"], vmax=var_ranges[var]["vmax"])
        else:
            print(f"    Warning: {var} not found in AROME dataset")

        if var in wrf_ds:
            print(f"    Plotting WRF {var}...")
            plot_small_multiples(ds=wrf_ds.sel(time=times), model="WRF", variable=var,
                               vmin=var_ranges[var]["vmin"], vmax=var_ranges[var]["vmax"])
        else:
            print(f"    Warning: {var} not found in WRF dataset")

    print(f"\n{'='*70}")
    print(f"✓ All heat budget plots created successfully!")
    print(f"  Location: {confg.dir_PLOTS}heat_flux/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    mpl.use('Qt5Agg')
    colormap = diverging_hcl(palette="Blue-Red 2").cmap()

    times = make_times(start_day=15, start_hour=14, start_minute=0, end_day=16, end_hour=12, end_minute=0, freq="2h")

    print("Loading AROME data...")
    arome2d = read_in_arome.read_2D_variables_AROME(
        variableList=["hfs", "lfs", "lwd", "lwu", "swd", "swu", "hgt", "u_v_from_3d"],
        lon=slice(confg.lon_min, confg.lon_max), lat=slice(confg.lat_min, confg.lat_max), slice_lat_lon=True)


    print("Loading WRF data...")
    wrf_hf = read_wrf_for_times(times=times,
                                 variables=["hfs", "lfs", "lwd", "lwu", "swd", "swu", "z", "z_unstag", "u", "v"])

    # Plot all heat budget variables for both models
    plot_all_heat_budget_variables(arome_ds=arome2d, wrf_ds=wrf_hf, times=times)
    plt.show()
