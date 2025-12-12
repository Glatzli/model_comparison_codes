"""
sensible heat flux spatial plots:
during night AROME is mostly negative, during
now added lowest level wind arrows. AROME looks strange? WRF much better up-valley flow?
espc. AROME has low values near the surface, further up they increase (see Hannes' plots...)

sunset at 16:25 UTC: temp falls already since ~15:30? => heat flux turns around at sunset
WRF hfs: UPWARD HEAT FLUX AT THE SURFACE
"""
import fix_win_DLL_loading_issue

import os
from datetime import datetime

import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot to avoid Qt5 crashes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from colorspace import diverging_hcl

import sys
sys.path.append("C:/Users/eleme/Documents/1Uni_Laptop/model_comparison_codes/calculations_and_plots")
import confg
import read_in_arome
import read_wrf_helen
from plot_topo_comparison import calculate_lon_extent_for_km

variables_to_plot = ["hfs", "lfs"] # , "lwd", "lwu", "swd", "swu"]


def add_scalebar(ax, length_km=10, location='lower right'):
    """
    Add a scalebar to a cartopy axes.

    Args:
        ax: Cartopy axes object
        length_km: Length of scalebar in kilometers (default: 10)
        location: Location string for the scalebar (default: 'lower right')
    """
    # Get the current extent in data coordinates
    extent = ax.get_extent(crs=ccrs.PlateCarree())
    lon_min, lon_max, lat_min, lat_max = extent

    # Calculate center latitude for scalebar positioning
    center_lat = (lat_min + lat_max) / 2

    # Use the existing function to calculate longitude extent for km
    scalebar_lon_size = calculate_lon_extent_for_km(center_lat, length_km)

    # Position the scalebar
    if 'right' in location:
        scalebar_lon_start = lon_max - scalebar_lon_size - 0.01
    else:
        scalebar_lon_start = lon_min + 0.01

    if 'lower' in location:
        scalebar_lat = lat_min + 0.02
    else:
        scalebar_lat = lat_max - 0.02

    # Draw the scalebar as a thick black line
    ax.plot([scalebar_lon_start, scalebar_lon_start + scalebar_lon_size],
            [scalebar_lat, scalebar_lat],
            color='black', linewidth=3, transform=ccrs.PlateCarree())

    # Add text label
    scalebar_lon_center = scalebar_lon_start + scalebar_lon_size / 2
    ax.text(scalebar_lon_center, scalebar_lat + 0.01, f'{length_km} km',
            transform=ccrs.PlateCarree(), ha='center', va='bottom',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))


# ============================================================================
# CONFIGURATION AND HELPER FUNCTIONS
# ============================================================================

def get_variable_metadata():
    """
    Get metadata for heat budget variables (labels, colormaps, symmetry).

    Returns:
        dict: Dictionary with variable metadata
    """
    return {"hfs": {"label": "Sensible Heat Flux", "cmap": "RdBu_r", "symmetric": True},
        "lfs": {"label": "Latent Heat Flux", "cmap": "RdBu_r", "symmetric": True},
        "lwd": {"label": "Downward Longwave Flux", "cmap": "YlOrRd", "symmetric": False},
        "lwu": {"label": "Upward Longwave Flux", "cmap": "YlOrRd", "symmetric": False},
        "swd": {"label": "Downward Shortwave Flux", "cmap": "YlOrRd", "symmetric": True},
        "swu": {"label": "Upward Shortwave Flux", "cmap": "YlOrRd", "symmetric": True}, }


def get_variable_ranges():
    """
    Get default value ranges for heat budget variables.

    Returns:
        dict: Dictionary with vmin/vmax for each variable
    """
    return {"hfs": {"vmin": -100, "vmax": 100},  # Sensible heat flux (can be negative/positive)
        "lfs": {"vmin": -150, "vmax": 150},  # Latent heat flux (mostly positive)
        "lwd": {"vmin": 0, "vmax": 400},  # Downward longwave (always positive)
        "lwu": {"vmin": 0, "vmax": 450},  # Upward longwave (always positive)
        "swd": {"vmin": -800, "vmax": 800},  # Downward shortwave (0 at night)
        "swu": {"vmin": -800, "vmax": 800},  # Upward shortwave (reflected, 0 at night)
    }


def get_colormap_for_variable(variable):
    """
    Get the appropriate colormap for a given variable.

    Args:
        variable: Variable name (hfs, lfs, etc.)

    Returns:
        matplotlib colormap object
    """
    var_metadata = get_variable_metadata()
    metadata = var_metadata.get(variable, {"label": variable, "cmap": "viridis", "symmetric": False})

    if metadata["symmetric"]:
        return diverging_hcl(palette="Blue-Red 2").cmap()
    else:
        return plt.colormaps[metadata["cmap"]]


def extract_topography_and_wind(ds_sel, model, step=2):
    """
    Extract topography and wind data from a dataset slice.

    Args:
        ds_sel: Dataset slice for a specific time
        model: Model name ("WRF" or "AROME")
        step: Subsample step for wind vectors (default: 2)

    Returns:
        tuple: (z, u, v) - topography and wind components (u, v can be None)
    """
    if model == "WRF":
        z = ds_sel.z_unstag.isel(height=0)
        if "u" in ds_sel and "v" in ds_sel:
            u = ds_sel.sel(height=1).u.values[::step, ::step]
            v = ds_sel.sel(height=1).v.values[::step, ::step]
        else:
            u, v = None, None
    elif model == "AROME":
        z = ds_sel.hgt
        if "u" in ds_sel and "v" in ds_sel:
            u = ds_sel.u.values[::step, ::step]
            v = ds_sel.v.values[::step, ::step]
        else:
            u, v = None, None
    else:
        raise ValueError(f"Unknown model: {model}")

    return z, u, v


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

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
    # Get variable metadata and colormap
    var_metadata = get_variable_metadata()
    metadata = var_metadata.get(variable, {"label": variable, "cmap": "viridis", "symmetric": False})
    cmap = get_colormap_for_variable(variable)

    projection = ccrs.Mercator()  # use mercator projection per default
    nplots, ncols = len(ds.time), 3
    nrows = int((nplots + ncols - 1) / ncols)  # = 4 for 2hourly timesteps

    # Increase figure size and adjust spacing for larger subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), subplot_kw={'projection': projection},
                             gridspec_kw={'hspace': 0.15, 'wspace': 0.05})
    axes = axes.flatten() if nrows > 1 else [axes] if nplots == 1 else axes

    for i, time in enumerate(ds.time.values):
        ax = axes[i]
        ds_sel = ds.sel(time=time)

        # Plot the selected variable
        var_data = ds_sel[variable].values
        lat, lon = ds_sel.lat.values, ds_sel.lon.values
        im = ax.pcolormesh(lon, lat, var_data, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=projection)

        # Extract topography and wind data (by subsetting, want to plot only ever 2nd arrow)
        step = 2  # plot only every 2nd grid point wind arrow
        z, u, v = extract_topography_and_wind(ds_sel, model, step)

        # --- plot topo contours
        levels_thin = np.arange(0, 3500, 250)  # same as in plot_topo_comparison.py
        ax.contour(lon, lat, z.values, levels=levels_thin, colors="k", linewidths=0.3,
                   transform=projection)

        # Add wind quivers if wind data is available
        if u is not None and v is not None:  # only added if there's meaningful wind data
            quiver = ax.quiver(x=lon[::step], y=lat[::step], u=u, v=v, scale=40, scale_units="inches", transform=projection)


        ax.add_feature(cfeature.BORDERS, linewidth=0.5, transform=projection)

        # Format timestamp
        time_pd = pd.to_datetime(time)
        ax.text(0.1, 0.8, f"{time_pd.hour:02d}h", transform=ax.transAxes, fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        ax.set_xlabel(""), ax.set_ylabel("")
        ax.set_xlim([confg.lon_hf_min, confg.lon_hf_max]), ax.set_ylim(
            [confg.lat_hf_min, confg.lat_hf_max])  # ax.set_xlim([11.76, 11.95]), ax.set_ylim([47.15, 47.4])

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Use subplots_adjust first to create space for colorbar
    plt.subplots_adjust(left=0.05, right=0.80, top=0.95, bottom=0.05, hspace=0.15, wspace=0.05)

    if u is not None and v is not None:  # add wind quiver key only if wind data was plotted
        qk = ax.quiverkey(quiver, X=1.4, Y=-0.1, U=5, label='5 m/s', labelpos='E', coordinates='axes')
    # Place colorbar with precise positioning using fig.add_axes
    # [left, bottom, width, height] in figure coordinates
    # cax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(im, label=f"{model} {metadata['label']} [$W/m^2$]", orientation='vertical')  # cax=cax,
    # cbar.ax.tick_params(size=0)

    # Ensure the heat_flux directory exists
    heat_flux_dir = os.path.join(confg.dir_PLOTS, "heat_flux")
    os.makedirs(heat_flux_dir, exist_ok=True)

    plt.savefig(os.path.join(heat_flux_dir, f"{variable}_{model}_small_multiples.png"), dpi=300)


def plot_small_multiples_ziller_detail(ds, model="WRF", variable="hfs", vmin=None, vmax=None, lon_extent=(11.76, 11.95),
        lat_extent=(47.15, 47.4), **kwargs):
    """
    Plot detailed small multiples for Zillertal region with specific extent and times.
    Uses the same visualization as plot_small_multiples but for a zoomed-in region.

    :param ds: xarray Dataset with the heat budget variable, topography, and optionally wind
    :param model: Model name (WRF or AROME)
    :param variable: Variable name to plot (hfs, lfs, lwd, lwu, swd, swu)
    :param vmin: Minimum value for colorbar (if None, will be determined automatically)
    :param vmax: Maximum value for colorbar (if None, will be determined automatically)
    :param lon_extent: Tuple (lon_min, lon_max) for plot extent (only sets axis limits, not data subset)
    :param lat_extent: Tuple (lat_min, lat_max) for plot extent (only sets axis limits, not data subset)
    """
    # Get variable metadata and colormap
    var_metadata = get_variable_metadata()
    metadata = var_metadata.get(variable, {"label": variable, "cmap": "viridis", "symmetric": False})
    cmap = get_colormap_for_variable(variable)

    projection = ccrs.Mercator()  # use mercator projection per default
    nplots, ncols = len(ds.time), 2  # Use 2 columns for fewer timesteps
    nrows = int((nplots + ncols - 1) / ncols)

    # Increase figure size and adjust spacing for larger subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 6), subplot_kw={'projection': projection},
                             gridspec_kw={'hspace': 0.15, 'wspace': 0.05})
    axes = axes.flatten() if nrows > 1 else [axes] if nplots == 1 else axes

    for i, time in enumerate(ds.time.values):
        ax = axes[i]
        ds_sel = ds.sel(time=time)

        # Plot the selected variable
        im = ax.pcolormesh(ds_sel.lon.values, ds_sel.lat.values, ds_sel[variable].values, cmap=cmap, vmin=vmin,
                           vmax=vmax, transform=projection)

        # Extract topography and wind data
        step = 2  # plot only every 2nd grid point wind arrow
        z, u, v = extract_topography_and_wind(ds_sel, model, step)

        # --- plot topo contours
        levels_thin = np.arange(0, 3500, 250)
        ax.contour(ds_sel.lon.values, ds_sel.lat.values, z.values, levels=levels_thin, colors="k", linewidths=0.3,
                   transform=projection)

        # Add wind quivers only if wind data is available
        lat, lon = ds_sel.lat.values[::step], ds_sel.lon.values[::step]
        if u is not None and v is not None:
            # scale_units='width': scale relative to plot width (simpler than inches)
            # scale=200: 200 m/s wind would span the entire plot width
            quiver = ax.quiver(x=lon, y=lat, u=u, v=v, scale=100, scale_units="width", transform=projection)

        ax.add_feature(cfeature.BORDERS, linewidth=0.5, transform=projection)

        # Format timestamp
        time_pd = pd.to_datetime(time)
        ax.text(0.1, 0.8, f"{time_pd.hour:02d}h", transform=ax.transAxes, fontsize=10,
                fontweight="bold", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        ax.set_xlabel(""), ax.set_ylabel("")
        ax.set_xlim(lon_extent), ax.set_ylim(lat_extent)

    # Remove unused axes; necessary?!
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add quiver key only if wind data was plotted
    if u is not None and v is not None:
        qk = ax.quiverkey(quiver, X=1.2, Y=0, U=5, label='5 m/s', labelpos='E', coordinates='axes')

    # Use subplots_adjust first to create space for colorbar
    plt.subplots_adjust(left=0.05, right=0.80, top=0.95, bottom=0.05, hspace=0.15, wspace=0.05)

    # Place colorbar with precise positioning using fig.add_axes
    # [left, bottom, width, height] in figure coordinates
    cax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(im, cax=cax, label=f"{model} {metadata['label']} [$W/m^2$]", orientation='vertical')
    cbar.ax.tick_params(size=0)

    # Save with descriptive filename
    filename = f"{variable}_{model}_small_multiples_ziller_detail.png"
    plt.savefig(os.path.join(confg.dir_PLOTS, "heat_flux", filename), dpi=300)
    print(f"    ✓ Saved: {filename}")


def plot_all_heat_budget_variables(arome_ds, wrf_ds, times):
    """
    Plot all heat budget variables for both AROME and WRF models.

    :param arome_ds: AROME dataset with all heat budget variables
    :param wrf_ds: WRF dataset with all heat budget variables
    :param times: Time range to plot
    """
    # Get uniform colorbar ranges for each variable
    var_ranges = get_variable_ranges()

    print(f"\n{'=' * 70}")
    print(f"Plotting heat budget variables for AROME and WRF")
    print(f"{'=' * 70}\n")

    for var in variables_to_plot:
        print(f"  Processing {var}...")

        # Check if variable exists in datasets
        if var in arome_ds:
            print(f"    Plotting AROME {var}...")
            plot_small_multiples(ds=arome_ds.sel(time=times), model="AROME", variable=var, vmin=var_ranges[var]["vmin"],
                                 vmax=var_ranges[var]["vmax"])
        else:
            print(f"    Warning: {var} not found in AROME dataset")

        if var in wrf_ds:
            print(f"    Plotting WRF {var}...")
            plot_small_multiples(ds=wrf_ds.sel(time=times), model="WRF", variable=var, vmin=var_ranges[var]["vmin"],
                                 vmax=var_ranges[var]["vmax"])
        else:
            print(f"    Warning: {var} not found in WRF dataset")

    print(f"\n{'=' * 70}")
    print(f"✓ All heat budget plots created successfully!")
    print(f"  Location: {confg.dir_PLOTS}/heat_flux/")
    print(f"{'=' * 70}\n")


def plot_shortwave_comparison_arome(ds, time, lon_extent=(11.76, 11.95), lat_extent=(47.15, 47.4)):
    """
    Plot AROME swd (left) and swu (right) as a side-by-side comparison for a single time.
    Uses shared colorbar for both plots.

    :param ds: AROME xarray Dataset with swd, swu, hgt, and optionally wind (u, v)
    :param time: Single timestamp to plot (datetime-like)
    :param lon_extent: Tuple (lon_min, lon_max) for plot extent
    :param lat_extent: Tuple (lat_min, lat_max) for plot extent
    """
    var_metadata = get_variable_metadata()
    var_ranges = get_variable_ranges()
    cmap = get_colormap_for_variable("swd")  # Same colormap for both swd and swu

    # Use same vmin/vmax for both to have consistent colorbar
    vmin = 0
    vmax = max(var_ranges["swd"]["vmax"], var_ranges["swu"]["vmax"])

    projection = ccrs.Mercator()

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), subplot_kw={'projection': projection})
    #                              gridspec_kw={'wspace': 0.05}

    ds_sel = ds.sel(time=time, method="nearest")
    variables = ["swd", "swu"]
    # titles = ["Downward Shortwave Flux", "Upward Shortwave Flux"]

    for i, (ax, var) in enumerate(zip(axes, variables)):
        # Plot the variable
        im = ax.pcolormesh(ds_sel.lon.values, ds_sel.lat.values, ds_sel[var].values,
                           cmap=cmap, vmin=vmin, vmax=vmax, transform=projection)

        # Extract topography and wind data
        step = 2
        z, u, v = extract_topography_and_wind(ds_sel, model="AROME", step=step)

        # Plot topo contours
        levels_thin = np.arange(0, 3500, 100)
        ax.contour(ds_sel.lon.values, ds_sel.lat.values, z.values, levels=levels_thin,
                   colors="k", linewidths=0.3, transform=projection)

        # Add wind quivers only if wind data is available
        if u is not None and v is not None:
            lat_arr, lon_arr = ds_sel.lat.values[::step], ds_sel.lon.values[::step]
            quiver = ax.quiver(x=lon_arr, y=lat_arr, u=u, v=v, scale=100, scale_units="width",
                               transform=projection)

        ax.add_feature(cfeature.BORDERS, linewidth=0.5, transform=projection)
        ax.set_xlim(lon_extent), ax.set_ylim(lat_extent)

        # Add subplot labels a) and b) with white background
        label = "a)" if i == 0 else "b)"
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1))

        # scalebar is not shown?!
        # Add scalebar to both subplots
        add_scalebar(ax, length_km=10, location='lower right')

    # Add quiver key on the right side
    if u is not None and v is not None:
        qk = axes[1].quiverkey(quiver, X=1.1, Y=0, U=5, label='5 m/s', labelpos='E', coordinates='axes')

    # Add shared colorbar
    plt.subplots_adjust(left=0.05, right=0.85, top=0.92, bottom=0.08, wspace=0.05)
    cax = fig.add_axes([0.87, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(im, cax=cax, label="Shortwave Flux [$W/m^2$]", orientation='vertical')
    cbar.ax.tick_params(size=0)

    # Add time as suptitle
    time_pd = pd.to_datetime(time)
    # fig.suptitle(f"AROME Shortwave Radiation - {time_pd.strftime('%Y-%m-%d %H:%M')} UTC", fontsize=12, y=0.98)

    # Save figure
    filename = f"swd_swu_AROME_comparison_{time_pd.strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(os.path.join(confg.dir_PLOTS, "heat_flux", filename), dpi=300)
    print(f"    ✓ Saved: {filename}")


def plot_all_heat_budget_variables_ziller_detail(arome_ds, wrf_ds, times, lon_extent=(11.76, 11.95),
        lat_extent=(47.15, 47.4)):
    """
    Plot all heat budget variables for Zillertal detail region (zoomed-in view).

    :param arome_ds: AROME dataset with all heat budget variables
    :param wrf_ds: WRF dataset with all heat budget variables
    :param times: Time range to plot (specific timesteps)
    :param lon_extent: Tuple (lon_min, lon_max) for plot extent
    :param lat_extent: Tuple (lat_min, lat_max) for plot extent
    """
    # Get uniform colorbar ranges for each variable
    var_ranges = get_variable_ranges()

    print(f"\n{'=' * 70}")
    print(f"Plotting DETAILED heat budget variables for Zillertal region")
    print(f"  Extent: lon {lon_extent}, lat {lat_extent}")
    print(f"  Times: {len(times)} timesteps")
    print(f"{'=' * 70}\n")

    for var in variables_to_plot:
        print(f"  Processing {var}...")

        # Check if variable exists in datasets
        if var in arome_ds:
            print(f"    Plotting AROME {var} (Zillertal detail)...")
            plot_small_multiples_ziller_detail(ds=arome_ds.sel(time=times), model="AROME", variable=var,
                                               vmin=var_ranges[var]["vmin"], vmax=var_ranges[var]["vmax"],
                                               lon_extent=lon_extent, lat_extent=lat_extent)
        else:
            print(f"    Warning: {var} not found in AROME dataset")

        if var in wrf_ds:
            print(f"    Plotting WRF {var} (Zillertal detail)...")
            plot_small_multiples_ziller_detail(ds=wrf_ds.sel(time=times), model="WRF", variable=var,
                                               vmin=var_ranges[var]["vmin"], vmax=var_ranges[var]["vmax"],
                                               lon_extent=lon_extent, lat_extent=lat_extent)
        else:
            print(f"    Warning: {var} not found in WRF dataset")

    print(f"\n{'=' * 70}")
    print(f"✓ All detailed Zillertal heat budget plots created successfully!")
    print(f"  Location: {confg.dir_PLOTS}heat_flux/")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    colormap = diverging_hcl(palette="Blue-Red 2").cmap()

    # Choose which plots to create:
    # Option 1: Full extent, longer time period (2-hourly from 14:00 to 12:00 next day)
    create_full_extent_plots = True

    # Option 2: Detailed Zillertal region, specific morning hours (hourly from 10:00 to 11:30)
    create_ziller_detail_plots = False

    # Option 3: AROME swd/swu comparison plot at 10:00 UTC
    create_shortwave_comparison = False

    if create_full_extent_plots:
        print("\n" + "=" * 70)
        print("Creating FULL EXTENT plots")
        print("=" * 70)

        times = make_times(start_day=15, start_hour=14, start_minute=0, end_day=16, end_hour=12, end_minute=0,
                           freq="2h")

        print("Loading AROME data...")
        arome2d = read_in_arome.read_2D_variables_AROME(
            variableList=["hfs", "lfs", "lwd", "lwu", "swd", "swu", "hgt", "u_v_from_3d"],
            lon=slice(confg.lon_min, confg.lon_max), lat=slice(confg.lat_min, confg.lat_max), slice_lat_lon=True)

        print("Loading WRF data...")
        wrf_hf = read_wrf_for_times(times=times,
                                    variables=["hfs", "lfs", "lwd", "lwu", "swd", "swu", "z", "z_unstag", "u", "v"])

        # Plot all heat budget variables for both models
        plot_all_heat_budget_variables(arome_ds=arome2d, wrf_ds=wrf_hf, times=times)

    if create_ziller_detail_plots:
        print("\n" + "=" * 70)
        print("Creating ZILLERTAL DETAIL plots")
        print("=" * 70)

        # Specific times for detailed view: 10:00 to 11:30 on day 16, hourly
        times_detail = make_times(start_day=16, start_hour=10, start_minute=0, end_day=16, end_hour=11, end_minute=30,
                                  freq="1h")

        # Zillertal extent (will only be used for axis limits)
        lon_extent = (11.76, 11.95)
        lat_extent = (47.15, 47.4)

        print(f"Loading AROME data (full extent, Zillertal will be shown via axis limits)...")
        arome2d_detail = read_in_arome.read_2D_variables_AROME(
            variableList=["hfs", "lfs", "lwd", "lwu", "swd", "swu", "hgt", "u_v_from_3d"],
            lon=slice(confg.lon_min, confg.lon_max), lat=slice(confg.lat_min, confg.lat_max), slice_lat_lon=True)

        print("Loading WRF data (full extent)...")
        wrf_hf_detail = read_wrf_for_times(times=times_detail,
                                           variables=["hfs", "lfs", "lwd", "lwu", "swd", "swu", "z", "z_unstag", "u",
                                                      "v"])

        # Plot all heat budget variables for Zillertal detail (axis limits only)
        plot_all_heat_budget_variables_ziller_detail(arome_ds=arome2d_detail, wrf_ds=wrf_hf_detail, times=times_detail,
                                                     lon_extent=lon_extent, lat_extent=lat_extent)

    if create_shortwave_comparison:
        print("\n" + "=" * 70)
        print("Creating AROME SHORTWAVE COMPARISON plot (swd vs swu)")
        print("=" * 70)

        # Zillertal extent
        lon_extent = (11.76, 11.95)
        lat_extent = (47.15, 47.4)

        # Time for the plot: 10:00 UTC on day 16
        time_10h = datetime(2017, 10, 16, 10, 0)

        print(f"Loading AROME data...")
        arome2d_sw = read_in_arome.read_2D_variables_AROME(
            variableList=["swd", "swu", "hgt", "u_v_from_3d"],
            lon=slice(confg.lon_min, confg.lon_max), lat=slice(confg.lat_min, confg.lat_max), slice_lat_lon=True)

        print(f"Plotting AROME shortwave comparison at {time_10h}...")
        plot_shortwave_comparison_arome(ds=arome2d_sw, time=time_10h, lon_extent=lon_extent, lat_extent=lat_extent)

    plt.show()