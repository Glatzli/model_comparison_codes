"""
sensible heat flux spatial plots:
during night AROME is mostly negative, during
now added lowest level wind arrows. AROME looks strange? WRF much better up-valley flow?
espc. AROME has low values near the surface, further up they increase (see Hannes' plots...)

sunset at 16:25 UTC: temp falls already since ~15:30? => heat flux turns around at sunset
WRF hfs: UPWARD HEAT FLUX AT THE SURFACE

lat/lons are only limited, not subsetted. Therefore pretty large files.
"""
import sys

sys.path.append("C:/Users/eleme/Documents/1Uni_Laptop/model_comparison_codes")
import fix_win_DLL_loading_issue
fix_win_DLL_loading_issue

import os
from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from colorspace import diverging_hcl

import confg
import read_in_arome
import read_wrf_helen
from plot_topo_comparison import calculate_lon_extent_for_km

variables_to_plot = ["hfs", "lfs", "lwd", "lwu", "swd", "swu"]  # plot all heat budget variables


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
    ax.plot([scalebar_lon_start, scalebar_lon_start + scalebar_lon_size], [scalebar_lat, scalebar_lat], color='black',
            linewidth=3, transform=ccrs.PlateCarree())

    # Add text label
    scalebar_lon_center = scalebar_lon_start + scalebar_lon_size / 2
    ax.text(scalebar_lon_center, scalebar_lat + 0.01, f'{length_km} km', transform=ccrs.PlateCarree(), ha='center',
            va='bottom', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))


# ============================================================================
# CONFIGURATION AND HELPER FUNCTIONS
# ============================================================================

# Simple dictionaries for variable properties
VARIABLE_COLORMAPS = {"hfs": diverging_hcl(palette="Blue-Red 2").cmap(),  # Sensible Heat Flux
                      "lfs": diverging_hcl(palette="Blue-Red 2").cmap(),  # Latent Heat Flux
                      "lwd": plt.colormaps["YlOrRd"],  # Downward Longwave Flux
                      "lwu": plt.colormaps["YlOrRd"],  # Upward Longwave Flux
                      "swd": plt.colormaps["YlOrRd"],  # Downward Shortwave Flux
                      "swu": plt.colormaps["YlOrRd"],  # Upward Shortwave Flux
                      }

VARIABLE_LABELS = {"hfs": "Sensible Heat Flux", "lfs": "Latent Heat Flux", "lwd": "Downward Longwave Flux",
                   "lwu": "Upward Longwave Flux", "swd": "Downward Shortwave Flux", "swu": "Upward Shortwave Flux", }

VARIABLE_RANGES = {"hfs": {"vmin": -100, "vmax": 100}, "lfs": {"vmin": -150, "vmax": 150},
                   "lwd": {"vmin": 0, "vmax": 400}, "lwu": {"vmin": 0, "vmax": 450}, "swd": {"vmin": -800, "vmax": 800},
                   "swu": {"vmin": -800, "vmax": 800}, }


def extract_topography_and_wind(ds_sel, model, step=2):
    """
    Extract topography and wind data from a dataset slice.

    Args:
        ds_sel: Dataset slice for a specific time
        model: Model name ("WRF" or "AROME")
        step: Subsample step for wind barbs (default: 2). Controls distance between wind barbs.
              Lower values = more barbs (denser), higher values = fewer barbs (sparser).
              E.g., step=1 plots every grid point, step=3 plots every 3rd grid point.

    Returns:
        tuple: (z, u, v) - topography and wind components (u, v can be None)
    """
    if model == "AROME":
        z = ds_sel.hgt
        if "u" in ds_sel and "v" in ds_sel:
            u = ds_sel.u.values[::step, ::step]
            v = ds_sel.v.values[::step, ::step]
        else:
            u, v = None, None

    elif model in ["ICON", "ICON2TE"]:
        z = ds_sel.z
        if "u" in ds_sel and "v" in ds_sel:
            u = ds_sel.u.values[::step, ::step]  # .sel(height=1, method="nearest")
            v = ds_sel.v.values[::step, ::step]  # .sel(height=1, method="nearest").
    elif model == "UM":
        z = ds_sel.z
        if "u" in ds_sel and "v" in ds_sel:
            u = ds_sel.u.values[::step, ::step]  # (height=1, method="nearest").
            v = ds_sel.v.values[::step, ::step]  # .sel(height=1, method="nearest")
    elif model == "WRF":
        z = ds_sel.z  # for heat flux plot: _unstag.isel(height=0)
        if "u" in ds_sel and "v" in ds_sel:
            u = ds_sel.u.values[::step, ::step]  # hf-plot: .sel(height=1)
            v = ds_sel.v.values[::step, ::step]  # hf-plot: .sel(height=1)
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
    Deprecated?
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

    return wrf.sel(height=1, bottom_top_stag=1)  # return only the lowest model level


def plot_small_multiples(ds, model="WRF", variable="hfs", vmin=None, vmax=None, figsize=(12, 8), ncols=3,
        lon_extent=None, lat_extent=None, filename_suffix="", save_file=True, contour_line_dist=250, barb_length=3,
        step=2, plot_dir="heat_flux", custom_label=None, **kwargs):
    """
    Base function for plotting small multiples with common functionality.
    This eliminates code duplication between different plot types.

    :param ds: xarray Dataset with the heat budget variable, topography, and optionally wind
    :param model: Model name (WRF or AROME)
    :param variable: Variable name to plot (hfs, lfs, lwd, lwu, swd, swu)
    :param vmin: Minimum value for colorbar
    :param vmax: Maximum value for colorbar
    :param figsize: Figure size tuple
    :param ncols: Number of columns in subplot grid
    :param lon_extent: Tuple (lon_min, lon_max) for axis limits (None uses config defaults)
    :param lat_extent: Tuple (lat_min, lat_max) for axis limits (None uses config defaults)
    :param filename_suffix: Suffix to add to filename
    :param save_file: Whether to save the plot
    :param contour_line_dist: Distance between contour lines in meters (for full extent 250m, for detail plots use
    100m)
    :param barb_length: Length of wind barbs (default: 3). Controls visual size of wind barbs.
                        Larger values = longer barbs, smaller values = shorter barbs.
                        Note: This only affects visual appearance, not the meteorological interpretation.
    :param step: Subsample step for wind barbs (default: 2). Controls distance between wind barbs.
                 Lower values = more barbs (denser), higher values = fewer barbs (sparser).
    :param plot_dir: Directory name within confg.dir_PLOTS for saving plots (default: "heat_flux")
    :param custom_label: Optional custom label to override default (default: None)
    """
    # Get colormap and label for the variable
    cmap = confg.temperature_colormap
    label = custom_label or VARIABLE_LABELS.get(variable, variable)

    projection = ccrs.Mercator()
    nplots = len(ds.time)
    nrows = int((nplots + ncols - 1) / ncols)

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw={'projection': projection},
                             gridspec_kw={'hspace': 0.15, 'wspace': 0.05})
    axes = axes.flatten() if nrows > 1 else [axes] if nplots == 1 else axes

    # Initialize variables to avoid "might be referenced before assignment" errors
    im = None
    u, v = None, None
    ax = None
    barbs = None

    # Set default extents if not provided
    if lon_extent is None:
        lon_extent = (confg.lon_hf_min, confg.lon_hf_max)
    if lat_extent is None:
        lat_extent = (confg.lat_hf_min, confg.lat_hf_max)

    for i, time in enumerate(ds.time.values):
        ax = axes[i]
        ds_sel = ds.sel(time=time)

        # Plot the selected variable
        im = ax.pcolormesh(ds_sel.lon.values, ds_sel.lat.values, ds_sel[variable].values, cmap=cmap, vmin=vmin,
                           vmax=vmax, transform=projection)

        # Extract topography and wind data
        z, u, v = extract_topography_and_wind(ds_sel, model, step)

        # Plot topo contours
        levels_thin = np.arange(0, 3500, contour_line_dist)
        ax.contour(ds_sel.lon.values, ds_sel.lat.values, z.values, levels=levels_thin, colors="k", linewidths=0.15,
                   transform=projection)

        # Add wind barbs if wind data is available (meteorologically correct)
        if u is not None and v is not None:
            lat_subset, lon_subset = ds_sel.lat.values[::step], ds_sel.lon.values[::step]

            # Convert wind speeds from m/s to knots (multiply by 1.94384)
            u_knots = u * 1.94384
            v_knots = v * 1.94384

            # Wind barbs - meteorologically correct, no scaling needed!
            barbs = ax.barbs(x=lon_subset, y=lat_subset, u=u_knots, v=v_knots,
                           transform=projection, color='black', length=barb_length, linewidth=0.35)

        # Format timestamp
        time_pd = pd.to_datetime(time)
        ax.text(0.1, 0.8, f"{time_pd.hour:02d}h", transform=ax.transAxes, fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        ax.set_xlabel(""), ax.set_ylabel("")
        ax.set_xlim(lon_extent), ax.set_ylim(lat_extent)

    # Remove unused axes
    if 'i' in locals():
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

    # Use subplots_adjust to create space for colorbar
    plt.subplots_adjust(left=0.05, right=0.80, top=0.95, bottom=0.05, hspace=0.15, wspace=0.05)

    # Wind barbs are self-explanatory and don't need a legend
    # Barb convention: half barb ≈ 3 m/s, full barb = 5 m/s, two full barbs = 10 m/s, pennant = 50 m/s

    # Create colorbar
    if im is not None:
        cax = fig.add_axes([0.82, 0.1, 0.015, 0.7])
        # Choose units based on variable type
        if variable in ["temp"]:
            units = "[°C]"
        elif variable in ["th"]:
            units = "[K]"
        else:
            units = "[$W/m^2$]"
        cbar = fig.colorbar(im, cax=cax, label=f"{model} {label} {units}", orientation='vertical')
        cbar.ax.tick_params(size=0)

    # Save file if requested
    if save_file:
        filename = f"{variable}_{model}_small_multiples{filename_suffix}.png"  # Changed from .svg to .png

        # Create directory if it doesn't exist
        plots_dir = os.path.join(confg.dir_PLOTS, plot_dir)
        os.makedirs(plots_dir, exist_ok=True)

        filepath = os.path.join(plots_dir, filename)

        # Delete existing file if it exists to ensure clean overwrite
        if os.path.exists(filepath):
            os.remove(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')  # Added dpi=300
        print(f"    ✓ Saved: {filename}")
        plt.close()  # Close figure to free memory


def plot_detail_for_extent(arome_ds, wrf_ds, times, lon_extent, lat_extent, figsize, contour_line_dist,
        extent_name="detail", variables_to_plot=None, barb_length=None, step=2):
    """
    Plot detailed small multiples for both models for a given extent.

    Args:
        arome_ds: AROME dataset
        wrf_ds: WRF dataset
        times: Time selection for plotting
        lon_extent: Tuple (lon_min, lon_max) for plot extent
        lat_extent: Tuple (lat_min, lat_max) for plot extent
        extent_name: Name for the extent (used in print messages and filenames)
        variables_to_plot: List of variables to plot (if None, uses global variables_to_plot)
        barb_length: Length of wind barbs. Controls visual size of wind barbs.
                     Larger values = longer barbs, smaller values = shorter barbs.
        step: Subsample step for wind barbs (default: 2). Controls distance between wind barbs.
              Lower values = more barbs (denser), higher values = fewer barbs (sparser).

    Example usage:
        # Plot Zillertal detail
        times_detail = make_times(start_day=16, start_hour=10, start_minute=0,
                                 end_day=16, end_hour=11, end_minute=30, freq="1h")
        plot_detail_for_extent(arome2d, wrf_hf, times_detail,
                              lon_extent=(11.76, 11.95), lat_extent=(47.15, 47.4),
                              extent_name="Zillertal", step=3)
    """
    if variables_to_plot is None:
        variables_to_plot = globals()['variables_to_plot']

    print(f"\n" + "=" * 70)
    print(f"Creating {extent_name.upper()} DETAIL plots")
    print("=" * 70)

    # Subset datasets to the specified extent
    arome_detail = arome_ds.sel(lat=slice(lat_extent[0], lat_extent[1]), lon=slice(lon_extent[0], lon_extent[1]))
    wrf_detail = wrf_ds.sel(lat=slice(lat_extent[0], lat_extent[1]), lon=slice(lon_extent[0], lon_extent[1]))

    # Plot all heat budget variables for the detail extent
    for var in variables_to_plot:
        print(f"  Processing {var}...")

        # Check if variable exists in AROME dataset
        if var in arome_detail:
            print(f"    Plotting AROME {var} ({extent_name} detail)...")
            plot_small_multiples(ds=arome2d.sel(time=times), model="AROME", variable=var,
                                 vmin=VARIABLE_RANGES[var]["vmin"], vmax=VARIABLE_RANGES[var]["vmax"],
                                 lon_extent=lon_extent, lat_extent=lat_extent, figsize=figsize,
                                 filename_suffix=extent_name, contour_line_dist=contour_line_dist,
                                 barb_length=barb_length, step=step)

        else:
            print(f"    Warning: {var} not found in AROME dataset")

        # Check if variable exists in WRF dataset
        if var in wrf_detail:
            print(f"    Plotting WRF {var} ({extent_name} detail)...")
            # first must select WRF to 2D for plotting:

            plot_small_multiples(ds=wrf_detail.sel(time=times), model="WRF", variable=var,
                                 vmin=VARIABLE_RANGES[var]["vmin"], vmax=VARIABLE_RANGES[var]["vmax"],
                                 lon_extent=lon_extent, lat_extent=lat_extent, figsize=figsize,
                                 filename_suffix=extent_name, contour_line_dist=contour_line_dist,
                                 barb_length=barb_length, step=step)
        else:
            print(f"    Warning: {var} not found in WRF dataset")


def plot_shortwave_comparison_arome(ds, time, lon_extent=(11.76, 11.95), lat_extent=(47.15, 47.4),
                                   step=2, barb_length=4):
    """
    Plot AROME swd (left) and swu (right) as a side-by-side comparison for a single time.
    Uses shared colorbar for both plots.

    :param ds: AROME xarray Dataset with swd, swu, hgt, and optionally wind (u, v)
    :param time: Single timestamp to plot (datetime-like)
    :param lon_extent: Tuple (lon_min, lon_max) for plot extent
    :param lat_extent: Tuple (lat_min, lat_max) for plot extent
    :param step: Subsample step for wind barbs (default: 2). Controls distance between wind barbs.
                 Lower values = more barbs (denser), higher values = fewer barbs (sparser).
    :param barb_length: Length of wind barbs (default: 4). Controls visual size of wind barbs.
                        Larger values = longer barbs, smaller values = shorter barbs.
    """
    cmap = VARIABLE_COLORMAPS.get("swd", plt.colormaps["YlOrRd"])  # Same colormap for both swd and swu

    # Use same vmin/vmax for both to have consistent colorbar
    vmin = 0
    vmax = max(VARIABLE_RANGES["swd"]["vmax"], VARIABLE_RANGES["swu"]["vmax"])

    projection = ccrs.Mercator()

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), subplot_kw={'projection': projection})
    #                              gridspec_kw={'wspace': 0.05}

    ds_sel = ds.sel(time=time, method="nearest")
    variables = ["swd", "swu"]
    # titles = ["Downward Shortwave Flux", "Upward Shortwave Flux"]

    for i, (ax, var) in enumerate(zip(axes, variables)):
        # Plot the variable
        im = ax.pcolormesh(ds_sel.lon.values, ds_sel.lat.values, ds_sel[var].values, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=projection)

        # Extract topography and wind data
        z, u, v = extract_topography_and_wind(ds_sel, model="AROME", step=step)

        # Plot topo contours
        levels_thin = np.arange(0, 3500, 100)
        ax.contour(ds_sel.lon.values, ds_sel.lat.values, z.values, levels=levels_thin, colors="k", linewidths=0.3,
                   transform=projection)

        # Add wind barbs if wind data is available (meteorologically correct)
        if u is not None and v is not None:
            lat_arr, lon_arr = ds_sel.lat.values[::step], ds_sel.lon.values[::step]

            # Convert wind speeds from m/s to knots (multiply by 1.94384)
            u_knots = u * 1.94384
            v_knots = v * 1.94384

            # Wind barbs - meteorologically correct, no scaling needed!
            barbs = ax.barbs(x=lon_arr, y=lat_arr, u=u_knots, v=v_knots,
                           transform=projection, color='black', length=barb_length, linewidth=0.35)

        ax.add_feature(cfeature.BORDERS, linewidth=0.5, transform=projection)
        ax.set_xlim(lon_extent), ax.set_ylim(lat_extent)

        # Add subplot labels a) and b) with white background
        label = "a)" if i == 0 else "b)"
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=12, verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1))

        # scalebar is not shown?!
        # Add scalebar to both subplots
        add_scalebar(ax, length_km=10, location='lower right')

    # Wind barbs are self-explanatory and don't need a legend

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
    filepath = os.path.join(confg.dir_PLOTS, "heat_flux", filename)

    # Delete existing file if it exists to ensure clean overwrite
    if os.path.exists(filepath):
        os.remove(filepath)

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {filename}")
    plt.close()  # Close figure to free memory


if __name__ == '__main__':
    colormap = diverging_hcl(palette="Blue-Red 2").cmap()

    # Choose which plots to create:
    create_full_extent_plots = False  # Full extent plots for all times

    create_ziller_detail_plots = True  # Detailed Zillertal region

    create_wipp_detail_plots = True  # Detailed Wipp Valley region

    create_valley_exit_detail = False  # Specific detail plots for the valley exit region

    # Option 3: AROME swd/swu comparison plot at 10:00 UTC
    create_shortwave_comparison = False

    times = make_times(start_day=15, start_hour=14, start_minute=0, end_day=16, end_hour=12, end_minute=0, freq="2h")
    print("Loading AROME data...")
    arome2d = read_in_arome.read_2D_variables_AROME(
        variableList=["hfs", "lfs", "lwd", "lwu", "swd", "swu", "hgt", "u_v_from_3d"],
        lon=slice(confg.lon_min, confg.lon_max), lat=slice(confg.lat_min, confg.lat_max), slice_lat_lon=True)
    print("Loading WRF data...")
    wrf_hf = read_wrf_for_times(times=times,
                                variables=["hfs", "lfs", "lwd", "lwu", "swd", "swu", "z", "z_unstag", "u", "v"])

    if create_full_extent_plots:
        print("\n" + "=" * 70)
        print("Creating FULL EXTENT plots")
        print("=" * 70)

        lon_extent = (confg.lon_hf_min, confg.lon_hf_max)  # plot full extent
        lat_extent = (confg.lat_hf_min, confg.lat_hf_max)

        plot_detail_for_extent(arome_ds=arome2d, wrf_ds=wrf_hf, times=times, lon_extent=lon_extent,
                               lat_extent=lat_extent, figsize=(12, 8), contour_line_dist=250, extent_name="_full",
                               variables_to_plot=variables_to_plot, barb_length=2, step=2)

        # Close WRF dataset to free RAM  # wrf_hf.close()

    if create_wipp_detail_plots:
        plot_detail_for_extent(arome_ds=arome2d, wrf_ds=wrf_hf, times=times, lon_extent=confg.lon_wipp_extent,
                               lat_extent=confg.lat_wipp_extent, figsize=(8, 8), contour_line_dist=100,
                               extent_name="_wipp_valley", variables_to_plot=variables_to_plot, barb_length=3,
                               step=2)

    if create_valley_exit_detail:
        # from Achensee till Zell am See
        # and Jenbach till Rosenheim

        # Use the generic function for Zillertal plots
        plot_detail_for_extent(arome_ds=arome2d, wrf_ds=wrf_hf, times=times, lon_extent=confg.lon_inn_exit_extent,
                               lat_extent=confg.lat_inn_exit_extent, figsize=(11, 8), contour_line_dist=100,
                               extent_name="_valley_exit", variables_to_plot=variables_to_plot, barb_length=3.2,
                               step=2)

    if create_ziller_detail_plots:
        # Specific times for detailed view: 10:00 to 11:30 on day 16, hourly
        # times_detail = make_times(start_day=16, start_hour=10, start_minute=0, end_day=16, end_hour=11, end_minute=30,
        #                           freq="1h")
        # Zillertal extent

        # Use the generic function for Zillertal plots
        plot_detail_for_extent(arome_ds=arome2d, wrf_ds=wrf_hf, times=times, lon_extent=confg.lon_ziller_extent,
                               lat_extent=confg.lat_ziller_extent, figsize=(8, 8), contour_line_dist=100,
                               extent_name="_ziller_valley", variables_to_plot=variables_to_plot, barb_length=3.5,
                               step=2)



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
        arome2d_sw = read_in_arome.read_2D_variables_AROME(variableList=["swd", "swu", "hgt", "u_v_from_3d"],
                                                           lon=slice(confg.lon_min, confg.lon_max),
                                                           lat=slice(confg.lat_min, confg.lat_max), slice_lat_lon=True)

        print(f"Plotting AROME shortwave comparison at {time_10h}...")
        plot_shortwave_comparison_arome(ds=arome2d_sw, time=time_10h, lon_extent=lon_extent, lat_extent=lat_extent,
                                       step=2, barb_length=4)

        # Close shortwave dataset to free RAM
        arome2d_sw.close()

    plt.show()