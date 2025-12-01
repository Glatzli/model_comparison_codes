"""
Plot time series of vertical distribution of potential temperature using saved timeseries data.
This script uses the saved timeseries data for the ibk_uni gridpoint with "above_terrain" height coordinate.
Based on plot_timeseries_old_for_concept.py but using the new data management system.
"""

# Fix for OpenMP duplicate library error on Windows
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

sys.path.append("C:/Users/eleme/Documents/1Uni_Laptop/model_comparison_codes")

import confg
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from colorspace import diverging_hcl
from calculations_and_plots.manage_timeseries import load_or_read_timeseries, MODEL_ORDER


def plot_pot_temp_time_contours(pot_temp, wind_u=None, wind_v=None, model="AROME", interface_height=2500, point_name="ibk_uni"):
    """
    Plot potential temperature time & height series for a model with wind barbs.
    Thin 1 K pot temp contour lines, thick 5 K pot temp contour lines and red/blue shading for the 1/2 hrly
    warming/cooling in pot temp is plotted. Wind barbs show wind speed and direction.

    Args:
        pot_temp: xarray DataArray with potential temperature data (time, height)
        wind_u: xarray DataArray with u-component of wind (time, height) [m/s]
        wind_v: xarray DataArray with v-component of wind (time, height) [m/s]
        model: Name of the model
        interface_height: Maximum height for plotting [m]
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    vmin, vmax = -2, 2  # uniform colorbar
    levels = np.arange(vmin, vmax + 0.5, 0.5)

    # Limit the time range for the plot
    start_time = pd.to_datetime('2017-10-15 13:00:00', format='%Y-%m-%d %H:%M:%S')
    end_time = pd.to_datetime('2017-10-16 12:00:00', format='%Y-%m-%d %H:%M:%S')

    # Create diverging colormap
    pal1 = diverging_hcl(palette="Blue-Red 2")

    # Plot the filled contours (hourly warming/cooling rate)
    contourf = (pot_temp.diff("time", n=1) * 2).plot.contourf(ax=ax, x="time", y="height", levels=levels,
                                                              cmap=pal1.cmap(), add_colorbar=False, vmin=vmin,
                                                              vmax=vmax)

    # Plot the contour lines (1K intervals)
    contour1 = pot_temp.plot.contour(ax=ax, x="time", y="height",
                                     levels=np.arange(np.round(pot_temp.min()), np.round(pot_temp.max()), 1),
                                     colors='black', linewidths=0.5)

    # Plot the contour lines (5K intervals, labeled)
    contour5 = pot_temp.plot.contour(ax=ax, x="time", y="height", levels=np.arange(290, np.round(pot_temp.max()), 5),
                                     colors='black', linewidths=1.5)
    ax.clabel(contour5)

    # ax.set_xlim(start_time, end_time)
    plt.xlim(start_time, end_time)
    plt.ylim([0, interface_height])

    # Add wind barbs if wind data is provided
    if wind_u is not None and wind_v is not None:
        # Select every 4th height level to avoid cluttering
        if model == "HATPRO":
            height_skip = 8  # for lidar wind data we need to skip more levels due to higher vertical resolution
        else:
            height_skip = 4

        # Create hourly time array between start and end time
        hourly_times = pd.date_range(start=start_time, end=end_time, freq='h')
        # Get height coordinates
        heights = wind_u.height.values[::height_skip]

        # Create meshgrid for barb positions
        time_mesh, height_mesh = np.meshgrid(hourly_times, heights)

        # Plot wind barbs
        if model == "HATPRO":  # for LIDAR wind data has timestamps often 1 min before full hour -> use nearest method
            ax.barbs(time_mesh, height_mesh, wind_u.sel(time=hourly_times, height=heights, method="nearest").values,
                     wind_v.sel(time=hourly_times, height=heights, method="nearest").values, length=6, linewidth=0.5,
                     color='black')
        else:
            ax.barbs(time_mesh, height_mesh, wind_u.sel(time=hourly_times, height=heights).values,
                     wind_v.sel(time=hourly_times, height=heights).values, length=6, linewidth=0.5, color='black')

    # Add a colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('K hr$^{-1}$')
    if model == "HATPRO":
        ax.set_title(f"{model} potential temp time series with LIDAR wind data for {point_name}")
    else:
        ax.set_title(f"{model} potential temp time series for {point_name}")
    ax.set_ylabel(f"height above terrain [m]")
    ax.set_xlabel("")

    # Create subfolder for vertical timeseries plots if it doesn't exist
    vertical_timeseries_dir = os.path.join(confg.dir_PLOTS, "vertical_timeseries")
    os.makedirs(vertical_timeseries_dir, exist_ok=True)

    # Save the figure as SVG
    output_path = os.path.join(vertical_timeseries_dir, f"{model}_pot_temp_timeseries_{interface_height}m_{point_name}.svg")
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"  Saved: {output_path}")

    return fig, ax


def plot_model_timeseries(model, point, point_name, interface_height=2500):
    """
    Plot timeseries for a single model using saved timeseries data.
    Use different fct for models & HATPRO cause HATPRO is saved differently and models would be read if the timeseries
    wouldn't exist yet.

    Args:
        model: Model name (AROME, ICON, ICON2TE, UM, WRF)
        point: Dictionary with lat, lon, height keys
        point_name: Name of the point (e.g., 'ibk_uni')
        interface_height: Maximum height for plotting [m]
    """
    print(f"\nPlotting {model} timeseries for {point_name}...")

    # Load timeseries data (including wind components)
    ds = load_or_read_timeseries(model=model, point=point, point_name=point_name,
                                 variables_list=["th", "u", "v", "udir", "wspd"], height_as_z_coord="above_terrain")

    if ds is None:
        print(f"  Warning: Could not load data for {model}")
        return None

    # Check if 'th' variable exists
    if "th" not in ds:
        print(f"  Warning: 'th' variable not found in {model} dataset")
        ds.close()
        return None

    # Create the plot
    fig, ax = plot_pot_temp_time_contours(pot_temp=ds["th"], wind_u=ds["u"], wind_v=ds["v"], model=model,
                                          interface_height=interface_height, point_name=point_name)
    ds.close()
    return fig, ax


def load_lidar_wind_data():
    """
    Load lidar wind data from the merged SL88 file.

    Returns:
        wind_u, wind_v: xarray DataArrays with u and v wind components, or (None, None) if loading fails
    """
    try:
        if not os.path.exists(confg.lidar_sl88_merged_path):
            print(f"  Warning: Lidar file not found: {confg.lidar_sl88_merged_path}")
            return None, None

        # Load lidar data
        ds_lidar = xr.open_dataset(confg.lidar_sl88_merged_path)

        # Don't close the dataset here - return the data arrays
        return ds_lidar["ucomp_unfiltered"], ds_lidar["vcomp_unfiltered"]

    except Exception as e:
        print(f"  Error loading lidar wind data: {e}")
        return None, None


def plot_hatpro_timeseries(interface_height=2500, point_name="ibk_uni"):
    """
    Plot HATPRO timeseries using the calced vars file with potential temperature.
    For HATPRO, load and overlay lidar wind data from SL88 merged file.

    Args:
        interface_height: Maximum height for plotting [m]
        point_name: Name of the point (e.g., 'ibk_uni')
    """
    print(f"\nPlotting HATPRO timeseries...")

    # Check if HATPRO file exists
    if not os.path.exists(confg.hatpro_calced_vars):
        print(f"  Warning: HATPRO file not found: {confg.hatpro_calced_vars}")
        return None

    # Load HATPRO data
    ds_hatpro = xr.open_dataset(confg.hatpro_calced_vars)

    # Check if 'th' variable exists
    if "th" not in ds_hatpro:
        print(f"  Warning: 'th' variable not found in HATPRO dataset")
        ds_hatpro.close()
        return None

    # Get potential temperature data (no filtering - interface_height only sets ylim)
    pot_temp = ds_hatpro["th"]

    # Load lidar wind data for HATPRO plots
    wind_u, wind_v = load_lidar_wind_data()

    if wind_u is not None and wind_v is not None:
        print(f"  Using lidar wind data for HATPRO plot")
    else:
        print(f"  Note: HATPRO plot will be created without wind barbs")

    # Create the plot
    fig, ax = plot_pot_temp_time_contours(pot_temp, wind_u=wind_u, wind_v=wind_v, model="HATPRO",
                                          interface_height=interface_height, point_name=point_name)

    ds_hatpro.close()
    return fig, ax


def plot_all_models(point_name="ibk_uni", interface_height=2500, models_to_plot=None):
    """
    Plot timeseries for all models.

    Args:
        point_name: Name of the point (default: 'ibk_uni')
        interface_height: Maximum height for plotting [m] (default: 2500)
        models_to_plot: List of models to plot (default: all models from MODEL_ORDER)
    """
    print(f"\n{'=' * 70}")
    print(f"Plotting timeseries for {point_name} up to {interface_height}m")
    print(f"{'=' * 70}")

    # Get point coordinates
    if point_name not in confg.ALL_POINTS:
        print(f"Error: Point '{point_name}' not found in confg.ALL_POINTS")
        return

    point = confg.ALL_POINTS[point_name]

    # Use all models if not specified
    if models_to_plot is None:
        models_to_plot = MODEL_ORDER

    # Plot each model
    for model in models_to_plot:
        try:
            plot_model_timeseries(model, point, point_name, interface_height)
        except Exception as e:
            print(f"  Error plotting {model}: {e}")

    # Plot HATPRO (only for Innsbruck points)
    if point_name.startswith("ibk"):
        try:
            plot_hatpro_timeseries(interface_height, point_name)
        except Exception as e:
            print(f"  Error plotting HATPRO: {e}")

    print(f"\n{'=' * 70}")
    print("Done!")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    # Use Qt5Agg backend for interactive plotting (optional)
    matplotlib.use('Qt5Agg')

    # Configuration
    point_name = "ibk_uni"
    interface_height = 2500  # default y limit for plots

    # Plot all models
    plot_all_models(point_name=point_name, interface_height=interface_height)

    # Show all plots
    plt.show()