"""
Plot four vertical profiles in the same plot for comparison.

This module creates a static matplotlib plot showing temperature and wind profiles from 4 points
at a single timestamp using a single model (AROME, ICON, ICON2TE, UM, or WRF).

Features:
- Loads timeseries data for a single model at multiple points
- Compares 4 different points at a single timestamp
- Uses HCL uniform colors and different line styles for each point
- Layout: left subplot for temperature & dewpoint depression,
  right subplot for wind speed & direction
"""

import fix_win_DLL_loading_issue

fix_win_DLL_loading_issue

import os
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from colorspace import qualitative_hcl

import confg
from calculations_and_plots.manage_timeseries import load_or_read_timeseries

# Define line styles for the 4 points

LINE_WIDTHS = [2.0, 2.0, 2.0, 2.0]
MARKER_STYLES = ["o", "o", "o", "o"]  # circle, down triangle, square, up triangle

# Define line styles for the 4 points
POINT_LINE_STYLES = ["-", "--", "-.", "-."]


def _get_point_colors(n_points: int) -> List[str]:
    """Return HCL-uniform colors for each point."""
    return qualitative_hcl(palette="Dark 3").colors()[:n_points]


def plot_four_vertical_profiles(point_names: List[str], model: str = "AROME", timestamp: str = None,
        plot_max_height: float = 1650, temperature_var: str = "temp", variables: list = None,
        figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Create a matplotlib plot comparing vertical profiles from 4 different points at a single timestamp.

    Uses load_or_read_timeseries to load data from managed timeseries files.
    Shows all 4 points in the same plot with 2 subplots:
    - Left: Temperature & Dewpoint Depression for all points
    - Right: Wind Speed & Direction for all points

    Args:
        point_names: List of point location names to plot (one line per point in the same subplot)
        model: Which model to extract from the timeseries files (default: "AROME")
        timestamp: ISO format timestamp string. If None, statement is raised
        plot_max_height: Maximum height in meters to display on y-axis (default: 2000m)
        temperature_var: "temp" for temperature in °C or "th" for potential temperature in K
        variables: List of variables to load
        figsize: Figure size as (width, height) tuple

    Returns:
        Matplotlib figure object
    """
    if len(point_names) == 0:
        raise ValueError("At least 1 point name must be provided")

    print(f"Creating plot for model '{model}' at single timestamp")
    print(f"Points: {point_names}")
    print(f"Timestamp: {timestamp if timestamp else 'first available'}")

    # If timestamp not specified, get it from the first timeseries/model
    if timestamp is None:
        print(f"Insert a timestamp! Given now: {timestamp}")
        pass

    # Create figure with 2 subplots: [Temp/Humidity] | [Wind]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Create top x-axes once to avoid duplicate labels
    ax1_top = ax1.twiny()  # ax1 = left axis, left subplot
    ax2_top = ax2.twiny()  # ax2 = right axis, right subplot
    ax1_top.xaxis.set_ticks_position('top')
    ax1_top.xaxis.set_label_position('top')
    ax1_top.tick_params(axis='x', labeltop=True, labelbottom=False)
    ax2_top.xaxis.set_ticks_position('top')
    ax2_top.xaxis.set_label_position('top')
    ax2_top.tick_params(axis='x', labeltop=True, labelbottom=False)

    # HCL-uniform colors for each point
    point_colors = _get_point_colors(len(point_names))

    # Process each point
    for pt_idx, point_name in enumerate(point_names):
        print(f"\n  Processing point: {point_name}")

        point = confg.ALL_POINTS.get(point_name)
        if point is None:
            print(f"    Warning: Point {point_name} not found in confg")
            continue

        # Use different line style for each point
        point_line_style = POINT_LINE_STYLES[pt_idx] if pt_idx < len(POINT_LINE_STYLES) else "-"
        point_color = point_colors[pt_idx]

        try:
            # Load timeseries data directly
            print(f"    Loading data for {point_name}...")
            ds = load_or_read_timeseries(model=model, point=point, point_name=point_name, variables_list=variables,
                                         height_as_z_coord="direct")

            if ds is None:
                print(f"      Warning: Could not load timeseries for {model} at {point_name}")
                continue

            # Get height coordinate
            height = ds.coords["height"].values
            ts = np.datetime64(timestamp)

            # Filter to max height
            # valid_height = height <= plot_max_height
            # height_filtered = height[valid_height]
            ds_subset = ds.sel(time=ts)  # select only specific, wanted timestamp

            # ====== SUBPLOT 1 (LEFT): Temperature & Dewpoint Depression ======

            # Add temperature trace
            if temperature_var in ds:
                temp = ds_subset[temperature_var].values
                ax1.plot(temp, height, linestyle=point_line_style, linewidth=2.0, color=point_color,
                         label=f"{confg.ALL_POINTS[point_name]['name']} ({confg.ALL_POINTS[point_name]['height']} m)")

            # Add dewpoint depression on secondary x-axis
            if "Td_dep" in ds:
                Td_dep = ds_subset["Td_dep"].values
                ax1_top.plot(Td_dep, height, linestyle=point_line_style, linewidth=1.0, color=point_color)

            # ====== SUBPLOT 2 (RIGHT): Wind Speed & Direction ======

            # Add wind speed trace
            if "wspd" in ds:
                wspd = ds_subset["wspd"].values
                ax2.plot(wspd, height, linestyle=point_line_style, linewidth=2.0, color=point_color)

            # Add wind direction as scatter
            if "udir" in ds:
                wdir = ds_subset["udir"].values
                ax2_top.scatter(wdir, height, marker=MARKER_STYLES[pt_idx], s=15, color=point_color)

            ds.close()

        except Exception as e:
            print(f"      Error loading data for {model} at {point_name}: {e}")
            continue

    # Configure subplot 1 (Temperature & Humidity)
    ax1.set_xlabel("Temperature [°C]" if temperature_var == "temp" else "Potential Temperature [K]", fontsize=12)
    ax1.set_xlim([285, 310])
    ax1.set_ylabel("Height above terrain [m]", fontsize=12)
    ax1.set_ylim([600, plot_max_height])
    ax1.grid(True, alpha=0.3)

    ax1_top.set_xlabel("Dewpoint Depression [°C]", fontsize=12)
    ax1_top.set_xlim([0, 70])
    ax1.legend(loc='upper left', fontsize=12)

    # Configure subplot 2 (Wind Speed & Direction)
    ax2.set_xlim([0, 10])
    ax2.set_xlabel("Wind Speed [m/s]", fontsize=12)
    ax2.set_ylim([600, plot_max_height])
    ax2.grid(True, alpha=0.3)

    ax2_top.set_xlabel("Wind Direction", fontsize=12)
    ax2_top.set_xlim([0, 360])
    ax2_top.set_xticks([0, 90, 180, 270, 360])
    ax2_top.set_xticklabels(["N", "E", "S", "W", "N"])

    # Format timestamp for overall title
    try:
        formatted_ts = pd.to_datetime(timestamp).strftime('%dth %H:%M')
    except:
        formatted_ts = timestamp

    # Add overall title
    fig.suptitle(f"{model} at {formatted_ts} UTC", fontsize=12)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    """
    Example: Plot vertical profiles for multiple points at a single timestamp.

    Uses load_or_read_timeseries to load data from managed timeseries files.
    This function shows one model at multiple points for a single timestamp.
    """
    # Configuration
    # List of point names to plot (now 4 points)
    point_names = ["brenner_saddle", "wipp_schoenberg_matrei", "patsch_EC_south", "ibk_uni"]

    # Which models to extract and compare
    models = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]  # Can be: AROME, ICON, ICON2TE, UM, WRF

    # Select timestamp once (ISO format)
    timestamp = "2017-10-16T00:30:00"

    # Other parameters
    plot_max_height = 2500  # m
    temperature_var = "th"  # "temp" or "th"

    # Save plot directory
    output_dir = os.path.join(confg.dir_PLOTS, "vertical_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Create filename w only hh_mm; no year or month in filename
    ts_clean = timestamp[8:-3].replace(":", "")

    # Loop through all models and create plots
    for model in models:
        print(f"\n{'='*60}")
        print(f"Creating plot for model: {model}")
        print(f"{'='*60}")

        try:
            # Create plot for this model
            fig = plot_four_vertical_profiles(point_names=point_names, model=model, timestamp=timestamp,
                                              plot_max_height=plot_max_height, temperature_var=temperature_var)

            # Save plot
            output_file = os.path.join(output_dir, f"brenner_cross_section_{model}_{ts_clean}.pdf")
            fig.savefig(output_file, format="pdf", bbox_inches='tight')
            print(f"✓ Plot saved to: {output_file}")

            # Close figure to free memory
            plt.close(fig)

        except Exception as e:
            print(f"✗ Error creating plot for {model}: {e}")
