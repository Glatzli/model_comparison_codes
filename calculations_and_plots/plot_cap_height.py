"""
Compute and plot CAP height (inversion base) for all points as small multiples timeline.

This module imports timeseries loading and CAP computation functions from plot_vertical_profiles.py
to ensure consistent data handling across scripts.

Workflow:
- Load or read timeseries data for each model and point (using functions from plot_vertical_profiles)
- Compute CAP height from timeseries data at each point (using cap_height_profile function)
- Plot: small multiples timeline showing CAP height at all points over time
"""
import fix_win_DLL_loading_issue
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

import confg
from calculations_and_plots.calc_cap_height import cap_height_profile
# Import timeseries and CAP computation functions from manage_timeseries
from calculations_and_plots.manage_timeseries import (load_or_read_timeseries, MODEL_ORDER, variables)
from confg import model_colors_temp_wind, icon_2te_hatpro_linestyle


def compute_cap_for_point(model: str, point: dict, point_name: str, timestamps: List[str]) -> xr.DataArray:
    """
    Compute CAP height timeseries for a single model and point.
    
    Uses the same approach as plot_vertical_profiles: load timeseries from saved file
    or read fresh data, then compute CAP height from the timeseries using cap_height_profile.
    
    Args:
        model: Model name (AROME, ICON, ICON2TE, UM, WRF)
        point: Dictionary with 'lat' and 'lon' keys
        point_name: Name of the point location
        timestamps: List of timestamp strings
    
    Returns:
        DataArray with CAP heights (time dimension)
    """
    # Load timeseries data (from saved file or read fresh)
    ds = load_or_read_timeseries(model=model, point=point, point_name=point_name, variables_list=variables,
                                 height_as_z_coord="above_terrain")
    if ds is None:
        print(f"    Warning: Could not load {model} data for {point_name}")
        return None

    # Check for height coordinate
    if "height" not in ds.coords:
        print(f"    Warning: No height coordinate found for {model} at {point_name}")
        ds.close()
        return None

    # Convert timestamps to numpy datetime64
    ts_array = [np.datetime64(ts) for ts in timestamps]

    # Select only the requested timestamps
    ds_selected = ds.sel(time=ts_array, method="nearest")

    # Compute CAP height using the same function as plot_vertical_profiles
    try:
        ds_with_cap = cap_height_profile(ds_selected, consecutive=3, model=model)
        cap_height_da = ds_with_cap["cap_height"]
    except Exception as e:
        print(f"    Warning: Could not compute CAP height for {model} at {point_name}: {e}")
        ds.close()
        return None

    ds.close()
    return cap_height_da


def compute_cap_all_points_all_models(point_names: List[str], timestamps: List[str]) -> Dict[
    str, Dict[str, xr.DataArray]]:
    """
    Compute CAP heights for all models at all specified points.
    
    Args:
        point_names: List of point names from confg.py
        timestamps: List of ISO format timestamp strings
    
    Returns:
        Nested dict: {model: {point_name: cap_height_da}}
    """
    print(f"\n{'=' * 70}")
    print(f"Computing CAP heights for {len(MODEL_ORDER)} models at {len(point_names)} points")
    print(f"{'=' * 70}\n")

    # Structure: {model: {point_name: cap_height_da}}
    cap_data = {model: {} for model in MODEL_ORDER}

    # Add keys for observations
    cap_data["HATPRO"] = {}
    cap_data["radiosonde"] = {}

    for model in MODEL_ORDER:
        print(f"\n{'-' * 70}")
        print(f"Processing model: {model}")
        print(f"{'-' * 70}")

        for point_name in point_names:
            point = confg.ALL_POINTS[point_name]  # index point dict by it's name
            if point is None:
                print(f"  ⚠ Skipping {point_name} - not found in confg")
                continue

            print(f"  Computing CAP for {point['name']} ({point_name})...")

            cap_da = compute_cap_for_point(model, point, point_name, timestamps)
            if cap_da is not None:
                cap_data[model][point_name] = cap_da
                print(f"    ✓ Success")
            else:
                print(f"    ✗ Failed")

            # Load observation CAP heights for Innsbruck points
            if point_name.startswith("ibk"):
                print(f"  Loading observations for {point['name']} ({point_name})...")
                obs_cap = load_observation_cap_heights(point_name, timestamps)

                for obs_type, cap_da in obs_cap.items():
                    cap_data[obs_type][point_name] = cap_da

    print(f"\n{'=' * 70}")
    print(f"✓ CAP computation complete!")
    print(f"{'=' * 70}\n")

    return cap_data


def plot_cap_timeseries_small_multiples(cap_data: Dict[str, Dict[str, xr.DataArray]], point_names: List[str],
        ymin: int = 0, ymax: int = 800) -> go.Figure:
    """
    Create small multiples plot of CAP height timelines for multiple points.
    
    Args:
        cap_data: Nested dict {model: {point_name: cap_height_da}}
        point_names: List of point names to plot
    
    Returns:
        Plotly figure object with small multiples
    """
    # Calculate grid layout with 2 columns
    n_points = len(point_names)
    n_cols = 2
    n_rows = int(np.ceil(n_points / n_cols))

    # Create subplot titles
    subplot_titles = []
    for point_name in point_names:
        point = confg.ALL_POINTS.get(point_name)  # index point dict by its name
        if point:
            subplot_titles.append(point["name"])
        else:
            subplot_titles.append(point_name)

    # Create subplots (function from plotly)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, vertical_spacing=0.12,
                        horizontal_spacing=0.1)

    # Plot for each point
    for idx, point_name in enumerate(point_names):
        point = confg.ALL_POINTS[point_name]  # index point dict by its name
        if point is None:
            continue

        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Plot model data
        for model in MODEL_ORDER:
            # Check if we have CAP data for this model and point
            if model not in cap_data or point_name not in cap_data[model]:
                continue

            cap_da = cap_data[model][point_name]

            # Filter times from 14:00 onwards
            cap_filtered = cap_da.where(cap_da.time >= np.datetime64("2017-10-15T14:00"), drop=True)

            # Determine line style: dashed for ICON2TE, solid otherwise
            line_dash = icon_2te_hatpro_linestyle if model == "ICON2TE" else "solid"

            # Only show legend for first subplot
            show_legend = (idx == 0)

            fig.add_trace(go.Scatter(x=cap_filtered["time"].values, y=cap_filtered.values, mode='lines', name=model,
                                     line=dict(color=model_colors_temp_wind[model], dash=line_dash, width=1.5),
                                     legendgroup=model, showlegend=show_legend), row=row, col=col)

        # Plot observation data for Innsbruck points
        if point_name.startswith("ibk"):
            # HATPRO
            if "HATPRO" in cap_data and point_name in cap_data["HATPRO"]:
                cap_hatpro = cap_data["HATPRO"][point_name]
                cap_hatpro_filtered = cap_hatpro.where(cap_hatpro.time >= np.datetime64("2017-10-15T14:00"), drop=True)

                show_legend = (idx == 0)
                fig.add_trace(
                    go.Scatter(x=cap_hatpro_filtered["time"].values, y=cap_hatpro_filtered.values, mode='lines',
                        name='HATPRO',
                        line=dict(color=model_colors_temp_wind["HATPRO"], dash=icon_2te_hatpro_linestyle, width=1.5),
                        legendgroup="HATPRO", showlegend=show_legend), row=row, col=col)

            # Radiosonde
            if "radiosonde" in cap_data and point_name in cap_data["radiosonde"]:
                cap_radiosonde = cap_data["radiosonde"][point_name]
                cap_radiosonde_filtered = cap_radiosonde.where(cap_radiosonde.time >= np.datetime64("2017-10-15T14:00"),
                                                               drop=True)

                show_legend = (idx == 0)
                fig.add_trace(go.Scatter(x=cap_radiosonde_filtered["time"].values, y=cap_radiosonde_filtered.values,
                    mode='markers', name='Radiosonde',
                    marker=dict(symbol='star', size=12, color=model_colors_temp_wind["Radiosonde"]),
                    legendgroup="radiosonde", showlegend=show_legend), row=row, col=col)

    fig.update_layout(title_text="CAP Height Timelines at Multiple Points", height=350 * n_rows, hovermode='x unified',
                      template='plotly_white',
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5))

    # Update axes labels and limits
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            # Set uniform y-axis range
            fig.update_yaxes(range=[ymin, ymax], row=i, col=j)
            # Set uniform x-axis range: 14:00 on 15th to 10:00 on 16th
            fig.update_xaxes(range=[np.datetime64("2017-10-15T14:00"), np.datetime64("2017-10-16T10:00")], row=i, col=j)
            # Only add axis labels to the first (upper left) subplot
            if i == 1 and j == 1:
                fig.update_xaxes(title_text="Time", row=i, col=j)
                fig.update_yaxes(title_text="CAP height [m]", row=i, col=j)
    return fig


def compute_and_plot_cap_all_points(start_time: str = "2017-10-15T12:00:00", end_time: str = "2017-10-16T12:00:00",
        time_step_hours: float = 0.5, max_height: float = 5000, point_names: List[str] = confg.ALL_POINTS) -> None:
    """
    Main function: Compute CAP heights for all models and points, then create small multiples plot.
    
    Args:
        start_time: Start timestamp ISO format
        end_time: End timestamp ISO format
        time_step_hours: Time step in hours between timestamps
        max_height: Maximum height for CAP computation
        point_names: List of points to process (default: ALL_POINTS from confg)
    """
    import pandas as pd

    # Generate list of timestamps
    timestamps = pd.date_range(start=start_time, end=end_time, freq=f"{int(time_step_hours * 60)}min").strftime(
        "%Y-%m-%dT%H:%M:%S").tolist()

    print(f"\nTime range: {start_time} to {end_time}")
    print(f"Time step: {time_step_hours}h ({len(timestamps)} timesteps)")
    print(f"Points: {len(point_names)}")

    # Compute CAP heights for all models and points
    cap_data = compute_cap_all_points_all_models(point_names, timestamps)

    # Create small multiples plot
    print("\nCreating small multiples plot...")
    fig = plot_cap_timeseries_small_multiples(cap_data, point_names, ymin=0, ymax=1000)

    # Save plot
    html_dir = os.path.join(confg.dir_PLOTS, "cap_depth")
    os.makedirs(html_dir, exist_ok=True)
    html_path = os.path.join(html_dir, "cap_depth_all_points_small_multiples.html")
    fig.write_html(html_path)
    fig.show(renderer="browser")

    print(f"\n{'=' * 70}")
    print(f"✓ Plot saved to: {html_path}")
    print(f"{'=' * 70}\n")


def load_observation_cap_heights(point_name: str, timestamps: List[str]) -> Dict[str, xr.DataArray]:
    """
    Load CAP heights for observations (HATPRO and Radiosonde) for Innsbruck points.

    Args:
        point_name: Name of the point location (must contain "ibk" for Innsbruck)
        timestamps: List of ISO format timestamp strings

    Returns:
        Dict with keys "HATPRO" and/or "radiosonde" containing CAP height DataArrays
    """
    obs_cap_data = {}

    # Only load observations for Innsbruck points
    if "ibk" not in point_name.lower():
        return obs_cap_data

    ts_array = [np.datetime64(ts) for ts in timestamps]

    # Load HATPRO CAP height
    try:
        if os.path.exists(confg.hatpro_with_cap_height):
            print(f"    Loading HATPRO CAP height")
            ds_hatpro = xr.open_dataset(confg.hatpro_with_cap_height)

            # Select timestamps
            cap_hatpro = ds_hatpro["cap_height"].sel(time=ts_array, method="nearest")
            obs_cap_data["HATPRO"] = cap_hatpro

            ds_hatpro.close()
            print(f"    ✓ HATPRO CAP height loaded")
    except Exception as e:
        print(f"    Warning: Could not load HATPRO CAP height: {e}")

    # Load Radiosonde CAP height (single point at 02:15 UTC)
    try:
        # Radiosonde CAP height from confg (already terrain-corrected)
        radiosonde_time = np.datetime64("2017-10-16T02:15:00")

        # Create DataArray with single point at 02:15 UTC
        cap_da = xr.DataArray([confg.radiosonde_cap_height], coords={"time": [radiosonde_time]}, dims=["time"],
            name="cap_height")
        obs_cap_data["radiosonde"] = cap_da
        print(f"    ✓ Radiosonde CAP height: {confg.radiosonde_cap_height:.0f} m at 02:15 UTC (defined in confg)")
    except Exception as e:
        print(f"    Warning: Could not load Radiosonde CAP height: {e}")

    return obs_cap_data


if __name__ == "__main__":
    # Compute and plot CAP heights for all valley points
    compute_and_plot_cap_all_points(start_time="2017-10-15T14:00:00", end_time="2017-10-16T12:00:00",
                                    time_step_hours=0.5, max_height=5000, point_names=confg.get_valley_points_only())