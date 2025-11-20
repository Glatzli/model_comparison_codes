"""
Compute and plot CAP height (inversion base) for all points as small multiples timeline.

This module imports timeseries loading and CAP computation functions from plot_vertical_profiles.py
to ensure consistent data handling across scripts.

Workflow:
- Load or read timeseries data for each model and point (using functions from plot_vertical_profiles)
- Compute CAP height from timeseries data at each point (using cap_height_profile function)
- Plot: small multiples timeline showing CAP height at all points over time
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from colorspace import qualitative_hcl
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

import confg
from calculations_and_plots.calc_cap_height import cap_height_profile
# Import timeseries and CAP computation functions from manage_timeseries
from calculations_and_plots.manage_timeseries import (
    load_or_read_timeseries,
    MODEL_ORDER,
    ALL_POINTS,
    variables
)

# --- Colors ---
qualitative_colors = qualitative_hcl(palette="Dark 3").colors()
# Model color mapping - ICON and ICON2TE share the same color
MODEL_COLORS = {"AROME": qualitative_colors[0], "ICON": qualitative_colors[2], "ICON2TE": qualitative_colors[2],
                # Same color as ICON, differentiated by line style
                "UM": qualitative_colors[4], "WRF": qualitative_colors[6], "Radiosonde": "grey",
                # Black for observations
                "HATPRO": "grey"  # Dark grey for HATPRO
                }


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
    ds = load_or_read_timeseries(model, point, point_name, variables)
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
        ds_with_cap = cap_height_profile(ds_selected, consecutive=3, model=model, subtract=True)
        cap_height_da = ds_with_cap["cap_height"]
    except Exception as e:
        print(f"    Warning: Could not compute CAP height for {model} at {point_name}: {e}")
        ds.close()
        return None
    
    ds.close()
    return cap_height_da


def compute_cap_all_points_all_models(point_names: List[str], timestamps: List[str]) -> Dict[str, Dict[str, xr.DataArray]]:
    """
    Compute CAP heights for all models at all specified points.
    
    Args:
        point_names: List of point names from confg.py
        timestamps: List of ISO format timestamp strings
    
    Returns:
        Nested dict: {model: {point_name: cap_height_da}}
    """
    print(f"\n{'='*70}")
    print(f"Computing CAP heights for {len(MODEL_ORDER)} models at {len(point_names)} points")
    print(f"{'='*70}\n")
    
    # Structure: {model: {point_name: cap_height_da}}
    cap_data = {model: {} for model in MODEL_ORDER}
    
    for model in MODEL_ORDER:
        print(f"\n{'-'*70}")
        print(f"Processing model: {model}")
        print(f"{'-'*70}")
        
        for point_name in point_names:
            point = getattr(confg, point_name, None)
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
    
    print(f"\n{'='*70}")
    print(f"✓ CAP computation complete!")
    print(f"{'='*70}\n")
    
    return cap_data


def plot_cap_timeseries_small_multiples(cap_data: Dict[str, Dict[str, xr.DataArray]],
                                        point_names: List[str], ymin: int = 0, ymax: int=600) -> go.Figure:
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
        point = getattr(confg, point_name, None)
        if point:
            subplot_titles.append(point["name"])
        else:
            subplot_titles.append(point_name)
    
    # Create subplots
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, vertical_spacing=0.12,
        horizontal_spacing=0.1)
    
    # Plot for each point
    for idx, point_name in enumerate(point_names):
        point = getattr(confg, point_name, None)
        if point is None:
            continue
        
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        for model in MODEL_ORDER:
            # Check if we have CAP data for this model and point
            if model not in cap_data or point_name not in cap_data[model]:
                continue
            
            cap_da = cap_data[model][point_name]
            
            # Filter times from 14:00 onwards
            cap_filtered = cap_da.where(cap_da.time >= np.datetime64("2017-10-15T14:00"), drop=True)
            
            # Determine line style: dashed for ICON2TE, solid otherwise
            line_dash = "dash" if model == "ICON2TE" else "solid"
            
            # Only show legend for first subplot
            show_legend = (idx == 0)
            
            fig.add_trace(go.Scatter(x=cap_filtered["time"].values, y=cap_filtered.values, mode='lines', name=model,
                line=dict(color=MODEL_COLORS[model], dash=line_dash, width=1.5), legendgroup=model,
                showlegend=show_legend), row=row, col=col)
    
    fig.update_layout(title_text="CAP Height Timelines at Multiple Points", height=350 * n_rows, hovermode='x unified',
        template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5))
    
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
                                    time_step_hours: float = 0.5, max_height: float = 5000,
                                    point_names: List[str] = None) -> None:
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
    
    if point_names is None:
        point_names = ALL_POINTS
    
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
    fig = plot_cap_timeseries_small_multiples(cap_data, point_names, ymin=0, ymax=600)
    
    # Save plot
    html_dir = os.path.join(confg.dir_PLOTS, "cap_depth")
    os.makedirs(html_dir, exist_ok=True)
    html_path = os.path.join(html_dir, "cap_depth_all_points_small_multiples.html")
    fig.write_html(html_path)
    fig.show(renderer="browser")
    
    print(f"\n{'=' * 70}")
    print(f"✓ Plot saved to: {html_path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    # Compute and plot CAP heights for all points
    compute_and_plot_cap_all_points(start_time="2017-10-15T12:00:00", end_time="2017-10-16T12:00:00",
        time_step_hours=0.5, max_height=5000)

