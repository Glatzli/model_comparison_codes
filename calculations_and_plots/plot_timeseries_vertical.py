"""
Plot time series of vertical distribution of potential temperature using saved timeseries data (like Figure 6 in
Lareau et al. 2013: The persistent cold air pool study).
This script uses the saved timeseries data for a given gridpoint with "above_terrain" height coordinate.
Based on plot_timeseries_old_for_concept.py but using the new data management system.

Due to past saving issues: if the plots are already saved, they are deleted and then the new ones are saved.

NOTE on wind barbs dimensions:
- xarray.sel() returns data in (time, height) order
- np.meshgrid(times, heights) returns arrays in (height, time) order
- matplotlib.barbs(X, Y, U, V) requires all arrays to have the same shape
- Therefore, wind data needs to be transposed (.T) for model data, but not for LIDAR data, the script automatically
detects if transposing is necessary to match meshgrid
"""
import os

import fix_win_DLL_loading_issue  # Must be first import on Windows to avoid DLL loading issues

fix_win_DLL_loading_issue
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from colorspace import diverging_hcl, sequential_hcl

import confg
from calculations_and_plots.manage_timeseries import load_or_read_timeseries, MODEL_ORDER


# import sys
# sys.path.append("C:/Users/eleme/Documents/1Uni_Laptop/model_comparison_codes")

def add_wind_barbs(ax, wind_u, wind_v, model, interface_height, debug=False):
    """
    Add wind barbs to the plot.

    :param ax: Matplotlib axis to add barbs to
    :param wind_u: u-component of wind (xarray DataArray)
    :param wind_v: v-component of wind (xarray DataArray)
    :param model: Model name string
    :param interface_height: Height limit for the plot
    :param debug: Print dimension information for debugging (default: False)
    :return: ax_wind (secondary axis)
    """
    # Create secondary y-axis for wind barbs (shares x-axis with main plot)
    ax_wind = ax.twinx()

    # Select height levels to avoid cluttering
    if model == "HATPRO":
        height_skip = 4  # for lidar wind data we need to skip more levels due to higher vertical resolution
    else:
        height_skip = 3

    # Get height coordinates - skip every nth level
    heights = wind_u.height.values[::height_skip]

    # Select every 2nd timestep to avoid cluttering
    times_selected = wind_u.time.values[1::2]

    # Create meshgrid for barb positions
    time_mesh, height_mesh = np.meshgrid(times_selected, heights)

    # Get wind data based on model type
    if "HATPRO" in model:  # for LIDAR wind data data has timestamps often 1 min before full hour -> use nearest method
        wind_u_data = wind_u.sel(time=times_selected, height=heights, method="nearest").values
        wind_v_data = wind_v.sel(time=times_selected, height=heights, method="nearest").values
    else:
        wind_u_data = wind_u.sel(time=times_selected, height=heights).values
        wind_v_data = wind_v.sel(time=times_selected, height=heights).values

    # Check and fix dimensions automatically:
    if debug:
        print(f"  Wind barbs for {model}:")
        print(f"    meshgrid shape: {time_mesh.shape} (height, time)")
        print(f"    wind data shape: {wind_u_data.shape}")

    # matplotlib.barbs() requires all arrays to have the same shape
    if wind_u_data.shape != time_mesh.shape:
        if debug:
            print(f"    → Transposing wind data from {wind_u_data.shape} to {wind_u_data.T.shape}")
        wind_u_data = wind_u_data.T  # (time, height) → (height, time)
        wind_v_data = wind_v_data.T

        # Verify the fix
        if debug:
            if wind_u_data.shape == time_mesh.shape:
                print(f"    ✓ Dimensions now match: {wind_u_data.shape}")
            else:
                print(f"    ✗ Error: Dimensions still don't match! wind_data: {wind_u_data.shape}, meshgrid: {time_mesh.shape}")
    else:
        if debug:
            print(f"    ✓ Dimensions already match: {wind_u_data.shape}")

    # Plot wind barbs
    ax_wind.barbs(time_mesh, height_mesh, wind_u_data, wind_v_data, length=6, linewidth=0.5, color='black')

    ax_wind.set_ylim([0, interface_height])  # Same limits as main axis
    # ax_wind.grid()
    ax_wind.set_ylabel('')  # No label needed for secondary axis
    ax_wind.set_yticks([])  # Hide tick labels on secondary axis
    return ax_wind


def plot_pot_temp_time_contours(pot_temp, wind_u=None, wind_v=None, model="AROME", interface_height=2500,
        point_name="ibk_uni"):
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

    # Limit the time range for the plot
    start_time = pd.to_datetime('2017-10-15 13:00:00', format='%Y-%m-%d %H:%M:%S')
    end_time = pd.to_datetime('2017-10-16 12:00:00', format='%Y-%m-%d %H:%M:%S')
    plt.xlim(start_time, end_time)
    plt.ylim([0, interface_height])

    if wind_u is not None and wind_v is not None:
        add_wind_barbs(ax, wind_u, wind_v, model, interface_height)

    # Add a colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('K hr$^{-1}$')
    # if model == "HATPRO":
    # ax.set_title(f"{model} potential temp time series with SL88 LIDAR wind data for {point_name}")
    # else:
    #     ax.set_title(f"{model} potential temp time series for {point_name}")
    ax.set_title("")
    ax.set_ylabel(f"height above terrain [m]")
    ax.set_xlabel("")

    # Create subfolder for vertical timeseries plots if it doesn't exist
    vertical_timeseries_dir = os.path.join(confg.dir_PLOTS, "vertical_timeseries")
    os.makedirs(vertical_timeseries_dir, exist_ok=True)

    # Save the figure as SVG with improved error handling
    output_path = os.path.join(vertical_timeseries_dir,
                               f"{point_name}_{model}_pot_temp_timeseries_{interface_height}m.svg")

    try:
        # Close existing file if it exists and might be locked
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except PermissionError:
                print(f"  Warning: Could not remove existing file {output_path}. File may be open in another program.")

        plt.savefig(output_path, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  Saved: {output_path}")
    except Exception as e:
        print(f"  Error saving plot to {output_path}: {e}")

    return fig, ax


def plot_dewpoint_depression_time_contours(td_dep, pot_temp, wind_u=None, wind_v=None, model="AROME",
        interface_height=2500, point_name="ibk_uni"):
    """
    Plot dewpoint depression time & height series for a model with wind barbs.
    Potential temperature contour lines (1K & 5K intervals) and green shading for dewpoint depression.
    Wind barbs show wind speed and direction.

    Args:
        td_dep: xarray DataArray with dewpoint depression data (time, height) [°C]
        pot_temp: xarray DataArray with potential temperature data (time, height) [K]
        wind_u: xarray DataArray with u-component of wind (time, height) [m/s]
        wind_v: xarray DataArray with v-component of wind (time, height) [m/s]
        model: Name of the model
        interface_height: Maximum height for plotting [m]
        point_name: Name of the point (e.g., 'ibk_uni')
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    vmin, vmax = 0, 15  # dewpoint depression range in °C
    levels = np.arange(vmin, vmax + 1, 1)

    # Create sequential green colormap (Greens 3)
    pal_green = sequential_hcl(palette="Greens 3")

    # Plot the filled contours (dewpoint depression)
    contourf = td_dep.plot.contourf(ax=ax, x="time", y="height", levels=levels, cmap=pal_green.cmap(),
                                    add_colorbar=False, vmin=vmin, vmax=vmax)

    # Plot the potential temperature contour lines (1K intervals)
    contour1 = pot_temp.plot.contour(ax=ax, x="time", y="height",
                                     levels=np.arange(np.round(pot_temp.min()), np.round(pot_temp.max()), 1),
                                     colors='black', linewidths=0.5)

    # Plot the potential temperature contour lines (5K intervals, labeled)
    contour5 = pot_temp.plot.contour(ax=ax, x="time", y="height", levels=np.arange(290, np.round(pot_temp.max()), 5),
                                     colors='black', linewidths=1.5)
    ax.clabel(contour5)

    # Limit the time range for the plot
    start_time = pd.to_datetime('2017-10-15 13:00:00', format='%Y-%m-%d %H:%M:%S')
    end_time = pd.to_datetime('2017-10-16 12:00:00', format='%Y-%m-%d %H:%M:%S')
    plt.xlim(start_time, end_time)
    plt.ylim([0, interface_height])

    if wind_u is not None and wind_v is not None:
        add_wind_barbs(ax, wind_u, wind_v, model, interface_height)

    # Add a colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Dewpoint Depression [°C]')

    # if model == "HATPRO":  # add title
    #     ax.set_title(f"{model} dewpoint depression time series with SL88 LIDAR wind data for {point_name}")
    # else:
    #     ax.set_title(f"{model} dewpoint depression time series for {point_name}")
    ax.set_title("")
    ax.set_ylabel(f"height above terrain [m]")
    ax.set_xlabel("")

    # Create subfolder for vertical timeseries plots if it doesn't exist
    vertical_timeseries_dir = os.path.join(confg.dir_PLOTS, "vertical_timeseries")
    os.makedirs(vertical_timeseries_dir, exist_ok=True)

    # Save the figure as SVG with improved error handling
    output_path = os.path.join(vertical_timeseries_dir,
                               f"{point_name}_{model}_dewpoint_depression_timeseries_{interface_height}m.svg")

    try:
        # Close existing file if it exists and might be locked
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except PermissionError:
                print(f"  Warning: Could not remove existing file {output_path}. File may be open in another program.")

        plt.savefig(output_path, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  Saved: {output_path}")
    except Exception as e:
        print(f"  Error saving plot to {output_path}: {e}")

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

    # Load timeseries data (including wind components and dewpoint depression)
    ds = load_or_read_timeseries(model=model, point=point, point_name=point_name,
                                 variables_list=["u", "v", "udir", "wspd", "q", "Td", "Td_dep", "p", "th", "temp",
                                                 "rho", "z", "z_unstag"], height_as_z_coord="above_terrain")

    if ds is None:
        print(f"  Warning: Could not load data for {model}")
        return None

    # Check if 'th' variable exists
    if "th" not in ds:
        print(f"  Warning: 'th' variable not found in {model} dataset")
        ds.close()
        return None

    # Create the potential temperature plot
    fig, ax = plot_pot_temp_time_contours(pot_temp=ds["th"], wind_u=ds["u"], wind_v=ds["v"], model=model,
                                          interface_height=interface_height, point_name=point_name)

    # Create the dewpoint depression plot if Td_dep is available
    if "Td_dep" in ds:
        fig_td, ax_td = plot_dewpoint_depression_time_contours(td_dep=ds["Td_dep"], pot_temp=ds["th"], wind_u=ds["u"],
                                                               wind_v=ds["v"], model=model,
                                                               interface_height=interface_height, point_name=point_name)
    else:
        print(f"  Warning: 'Td_dep' variable not found in {model} dataset, skipping dewpoint depression plot")

    ds.close()
    return fig, fig_td


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

    # Get potential temperature data
    pot_temp = ds_hatpro["th"]

    # Load lidar wind data for HATPRO plots
    sl88_data = xr.open_dataset(confg.lidar_sl88_merged_path)
    # slxr142_data = xr.open_dataset(confg.lidar_slxr142_merged_path)

    if sl88_data is not None:
        print(f"  Using lidar wind data for HATPRO plot")
    else:
        print(f"  Note: HATPRO plot will be created without wind barbs")

    # Create the pot temp plot
    fig, ax = plot_pot_temp_time_contours(pot_temp, wind_u=sl88_data.ucomp.compute(),
                                          wind_v=sl88_data.vcomp.compute(),
                                          model="HATPRO", interface_height=interface_height, point_name=point_name)
    # Close the figure to free memory
    # plt.close(fig)
    # Check if 'th' variable exists
    if "th" not in ds_hatpro:
        print(f"  Warning: 'th' variable not found in HATPRO dataset")
        ds_hatpro.close()
        return None
    # Create the dewpoint depression plot
    fig_td, ax_td = plot_dewpoint_depression_time_contours(td_dep=ds_hatpro["Td_dep"], pot_temp=pot_temp,
                                                           wind_u=sl88_data.ucomp.compute(),
                                                           wind_v=sl88_data.vcomp.compute(), model="HATPRO",
                                                           interface_height=interface_height, point_name=point_name)
    # Close the figure to free memory
    # plt.close(fig)

    ds_hatpro.close()
    return fig, fig_td


def plot_models_and_measurements(point_name="ibk_uni", interface_height=2500, models_to_plot=None):
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
            fig, fig_td = plot_model_timeseries(model, point, point_name, interface_height)
            plt.close(fig)
            plt.close(fig_td)
        except Exception as e:
            print(f"  Error plotting {model}: {e}")

    # Plot HATPRO (only for Innsbruck points)
    if point_name.startswith("ibk"):
        try:
            fig, fig_td = plot_hatpro_timeseries(interface_height, point_name)
            plt.close(fig)
            plt.close(fig_td)
        except Exception as e:
            print(f"  Error plotting HATPRO: {e}")

    print(f"\n{'=' * 70}")
    print("Done!")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    # points to plot:
    point_names = ["ibk_uni"] # "telfs", "ziller_ried", "jenbach",  # "woergl", "rosenheim"  , "ibk_airport",
    # "jenbach", "woergl", "hafelekar"
    interface_height = 1650  # default y limit for plots

    # Plot all models
    for point_name in point_names:
        plot_models_and_measurements(point_name=point_name, interface_height=interface_height)

    # Show all plots  # plt.show()