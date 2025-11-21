"""
Plot vertical temperature profiles for all models at fixed points.

This module creates interactive plots showing temperature profiles with height for different
weather models (AROME, ICON, ICON2TE, UM, WRF) and observations (Radiosonde, HATPRO)
at various locations. The plots include CAP (Cold Air Pool) height markers when available.

Main functionality:
- Load timeseries data from saved files (or read fresh data if not available)
- Add radiosonde and HATPRO observations for Innsbruck points
- Create small multiples plots showing all points side-by-side
- Save interactive HTML plots
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr

from plotly.subplots import make_subplots

import confg
from confg import model_colors_temp_wind, model_colors_humidity, icon_2te_hatpro_linestyle
from read_in_hatpro_radiosonde import read_radiosonde_dataset
from calculations_and_plots.calc_cap_height import cap_height_profile
# Import timeseries management functions
from calculations_and_plots.manage_timeseries import (load_or_read_timeseries, MODEL_ORDER, ALL_POINTS)




# Observation data paths; is actually in confg!
# RADIOSONDE_PATH = r"D:\MSc_Arbeit\data\radiosonde_ibk_smoothed.nc"
# HATPRO_PATH = r"D:\MSc_Arbeit\data\Observations\HATPRO_obs\hatpro_interpolated_arome_height_as_z.nc"


def get_obs_cap_path(obs_type: str) -> str:
    """
    Get the path to the computed CAP height file for observations.
    
    Args:
        obs_type: Either "radiosonde" or "hatpro"
    
    Returns:
        Full path to the CAP height NetCDF file
    """
    import confg
    cap_dir = os.path.join(confg.data_folder, "calculated_cap_height")
    return os.path.join(cap_dir, f"{obs_type}_cap_height.nc")


def _get_cap_height_value(cap_height_da: xr.DataArray, point: dict, ts: np.datetime64) -> float:
    """
    Extract CAP height value from DataArray, handling different dimension structures.
    Returns NaN if extraction fails.
    """
    if not cap_height_da.dims:
        # Scalar value
        return cap_height_da.item()

    if "lat" in cap_height_da.dims and "lon" in cap_height_da.dims:
        # Spatial + time dimensions
        return cap_height_da.sel(lat=point["lat"], lon=point["lon"], time=ts).item()

    # Only time dimension
    return cap_height_da.sel(time=ts, method="nearest").item()


def _load_or_compute_cap_heights(model: str, ds_filtered: xr.Dataset, point: dict, timestamps: List[str],
                                 ts_array: List[np.datetime64], model_data: dict, max_height: float) -> dict:
    """
    Compute CAP heights from timeseries (point) data.
    Returns dict mapping timestamp strings to (temp_at_cap, cap_height) tuples (tuple is needed so that the
    cap-marker is at
    the right temperature in the plot afterwards).
    """
    cap_data = {}

    ds_with_cap = cap_height_profile(ds_filtered, consecutive=3, model=model)
    cap_height_da = ds_with_cap["cap_height"]

    # Extract CAP heights for each timestamp
    for ts_str, ts in zip(timestamps, ts_array):
        cap_height = _get_cap_height_value(cap_height_da, point, ts)

        if np.isnan(cap_height) or cap_height > max_height:
            continue

        if ts_str not in model_data[model]:
            continue

        temp_data, height_data = model_data[model][ts_str]
        if len(height_data) == 0:
            continue

        idx = np.argmin(np.abs(height_data - cap_height))
        temp_at_cap = temp_data[idx]
        cap_data[ts_str] = (temp_at_cap, cap_height)

    return cap_data


def plot_single_point_with_slider(point_name: str, timestamps: List[str], max_height: float = 5000,
                                  plot_max_height: float = 2000,
                                  variables: list = ["udir", "wspd", "q", "p", "th", "temp", "z",
                                                     "z_unstag"]) -> go.Figure:
    """
    Create an interactive plot with time slider for a single point location.
    
    Users can slide through different timesteps and see the temperature profiles update dynamically.
    The plot shows all models and observations at one location.
    
    Args:
        point_name: Point location name from confg.py (e.g. "ibk_villa")
        timestamps: List of ISO format timestamp strings
        max_height: Maximum height in meters to load data (default: 5000m)
        plot_max_height: Maximum height in meters to display on y-axis (default: 2000m)
    
    Returns:
        Plotly figure object with the interactive time slider
    """
    point = getattr(confg, point_name, None)
    if point is None:
        raise ValueError(f"Point {point_name} not found in confg")

    print(f"Creating slider plot for {point['name']} with {len(timestamps)} timesteps...")

    # Convert timestamp strings to numpy datetime64
    ts_array = [np.datetime64(ts) for ts in timestamps]

    # Pre-load all data for all timesteps
    print(f"  Loading data for all models and timesteps...")
    model_data = {}  # {model: {timestamp: (temp, height)}}
    model_humidity_data = {}  # {model: {timestamp: (q, height)}}
    model_wspd_data = {}  # {model: {timestamp: (wspd, height)}}
    model_udir_data = {}  # {model: {timestamp: (wdir, height)}}
    obs_data = {}  # {obs_type: data}
    cap_data = {}  # {model: {timestamp: (temp_at_cap, cap_height)}}

    # Load model data
    for model in MODEL_ORDER:
        model_data[model] = {}
        model_humidity_data[model] = {}
        model_wspd_data[model] = {}
        model_udir_data[model] = {}

        # Load timeseries dataset
        ds = load_or_read_timeseries(model=model, point=point, point_name=point_name, variables_list=variables,
                                     height_as_z_coord="above_terrain")
        if ds is None:
            print(f"    Warning: Could not load {model} data")
            continue

        # Get height variable
        if "height" not in ds.coords:
            print(f"    Warning: No height coordinate found for {model}")
            ds.close()
            continue

        height_var = ds.coords["height"]
        ds_filtered = ds.where(height_var <= max_height, drop=True)

        # Extract temperature and height values for each timestamp
        for ts_str, ts in zip(timestamps, ts_array):
            temp = ds_filtered["temp"].sel(time=ts).values
            height = ds_filtered.coords["height"].values

            # Filter NaNs for temperature
            valid = ~np.isnan(temp) & ~np.isnan(height)
            model_data[model][ts_str] = (temp[valid], height[valid])

            # Extract humidity data (q) if available
            if "q" in ds_filtered:
                q = ds_filtered["q"].sel(time=ts).values
                # Convert from kg/kg to g/kg
                q = q * 1000
                # Filter NaNs for humidity
                valid_q = ~np.isnan(q) & ~np.isnan(height)
                model_humidity_data[model][ts_str] = (q[valid_q], height[valid_q])

            # Extract wind speed data (wspd) if available
            if "wspd" in ds_filtered:
                wspd = ds_filtered["wspd"].sel(time=ts).values
                valid_wspd = ~np.isnan(wspd) & ~np.isnan(height)
                model_wspd_data[model][ts_str] = (wspd[valid_wspd], height[valid_wspd])

            # Extract wind direction data (udir) if available
            if "udir" in ds_filtered:
                udir = ds_filtered["udir"].sel(time=ts).values
                valid_udir = ~np.isnan(udir) & ~np.isnan(height)
                model_udir_data[model][ts_str] = (udir[valid_udir], height[valid_udir])

        # Load or compute CAP heights
        try:
            cap_data[model] = _load_or_compute_cap_heights(model, ds_filtered, point, timestamps, ts_array, model_data,
                                                           max_height)
        except Exception as e:
            print(f"    Warning: Could not load/compute CAP height for {model}: {e}")

        ds.close()

    # Load observation data (only for Innsbruck points)
    if point_name.startswith("ibk"):
        # Radiosonde (no time dimension)
        try:
            ds_radiosonde = read_radiosonde_dataset(height_as_z_coord="above_terrain")
            ds_filtered = ds_radiosonde.where(ds_radiosonde["height"] <= max_height, drop=True)
            temp = ds_filtered["temp"].values
            height = ds_filtered["height"].values
            valid = ~np.isnan(temp) & ~np.isnan(height)
            obs_data["radiosonde"] = (temp[valid], height[valid])

            # Extract humidity data (q) if available
            if "q" in ds_filtered:
                q = ds_filtered["q"].values  # in kg/kg
                # Convert from kg/kg to g/kg
                q = q * 1000
                valid_q = ~np.isnan(q) & ~np.isnan(height)  # filter NaNs ...
                obs_data["radiosonde_humidity"] = (q[valid_q], height[valid_q])

            # Extract wind speed data (wspd) if available
            if "wspd" in ds_filtered:
                wspd = ds_filtered["wspd"].values
                valid_wspd = ~np.isnan(wspd) & ~np.isnan(height)
                obs_data["radiosonde_wspd"] = (wspd[valid_wspd], height[valid_wspd])

            # Extract wind direction data (udir) if available
            if "udir" in ds_filtered:
                udir = ds_filtered["udir"].values
                valid_udir = ~np.isnan(udir) & ~np.isnan(height)
                obs_data["radiosonde_udir"] = (udir[valid_udir], height[valid_udir])

            # set radiosonde CAP height manually
            cap_height = 1537 - confg.ibk_airport[
                "height"]  # chosen manually from height_as_z_coord="direct" plot & subtract (real) terrain height

            if not np.isnan(cap_height) and cap_height <= max_height:
                idx = np.argmin(np.abs(height[valid] - cap_height))
                obs_data["radiosonde_cap"] = (temp[valid][idx], cap_height)

            ds_radiosonde.close()
        except Exception as e:
            print(f"    Warning: Error in loading Radiosonde: {e}")

        # HATPRO (time-dependent)
        try:
            # Check for HATPRO file with CAP height first
            if os.path.exists(confg.hatpro_with_cap_height):
                # Load HATPRO data with pre-computed CAP height
                print(f"    Loading HATPRO data with CAP height")
                ds_hatpro = xr.open_dataset(confg.hatpro_with_cap_height)
                hatpro_cap_da = ds_hatpro["cap_height"]

            elif os.path.exists(confg.hatpro_calced_vars):  # used interpolated heights to arome levels -> change to
                # smooth HATPRO!
                # Load original HATPRO data and compute CAP height
                print(f"    Loading HATPRO data")
                ds_hatpro = xr.open_dataset(confg.hatpro_calced_vars)

                print(f"    Computing CAP height for HATPRO...")
                # Filter to max_height before computing CAP
                ds_hatpro_filtered = ds_hatpro.where(ds_hatpro["height"] <= max_height, drop=True)
                ds_hatpro_with_cap = cap_height_profile(ds_hatpro_filtered, consecutive=3, model="HATPRO")

                # Add CAP height to original (non-filtered) dataset
                ds_hatpro["cap_height"] = ds_hatpro_with_cap["cap_height"]
                hatpro_cap_da = ds_hatpro["cap_height"]

                # Save complete dataset with CAP height
                print(f"    Saving HATPRO dataset with CAP height to {confg.hatpro_with_cap_height}")
                ds_hatpro.to_netcdf(confg.hatpro_with_cap_height)
                print(f"    ✓ Saved successfully")
            else:
                print(f"    Warning: HATPRO file not found")
                ds_hatpro = None
                hatpro_cap_da = None

            if ds_hatpro is not None:
                obs_data["hatpro"] = {}
                obs_data["hatpro_humidity"] = {}
                obs_data["hatpro_wspd"] = {}
                obs_data["hatpro_udir"] = {}

                for ts_str, ts in zip(timestamps, ts_array):
                    ds_ts = ds_hatpro.sel(time=ts)
                    ds_filtered = ds_ts.where(ds_ts["height"] <= max_height, drop=True)

                    temp = ds_filtered["temp"].values
                    height = ds_filtered["height"].values
                    valid = ~np.isnan(temp) & ~np.isnan(height)
                    obs_data["hatpro"][ts_str] = (temp[valid], height[valid])
                    # creates large dict w. radiosonde w. cap_height & hatpro data for that timestamp

                    # Extract humidity data (q) if available; wind data isn't available -> maybe include LIDAR?
                    if "q" in ds_filtered:
                        q = ds_filtered["q"].values  # in kg/kg
                        # Convert from kg/kg to g/kg
                        q = q * 1000
                        valid_q = ~np.isnan(q) & ~np.isnan(height)  # filter NaNs ...
                        obs_data["hatpro_humidity"][ts_str] = (q[valid_q], height[valid_q])

                    # CAP height
                    if hatpro_cap_da is not None:
                        cap_height = hatpro_cap_da.sel(time=ts, method="nearest").item()
                        if not np.isnan(cap_height) and cap_height <= max_height:
                            idx = np.argmin(np.abs(height[valid] - cap_height))
                            key = f"hatpro_cap_{ts_str}"
                            obs_data[key] = (temp[valid][idx], cap_height)

                ds_hatpro.close()
        except Exception as e:
            print(f"    Warning: Error in loading HATPRO: {e}")

    print(f"  Creating frames...")

    max_q = 30  # Maximum humidity for scale

    # Create figure with subplots: [Temp/Humidity] | [Wind Speed/Direction]
    fig = make_subplots(rows=1, cols=2, specs=[[{"secondary_y": False}, {"secondary_y": False}]],
                        horizontal_spacing=0.12, column_widths=[0.6, 0.4])

    # Create frames for slider - one frame per timestep
    frames = []
    for ts_str in timestamps:
        frame_traces = []

        # ====== SUBPLOT 1: Temperature & Humidity ======
        # Add model traces (temperature)
        for model in MODEL_ORDER:
            if ts_str in model_data.get(model, {}):
                temp, height = model_data[model][ts_str]
                line_dash = icon_2te_hatpro_linestyle if model == "ICON2TE" else "solid"
                frame_traces.append(go.Scatter(x=temp, y=height, mode='lines', name=model,
                                               line=dict(color=model_colors_temp_wind[model], dash=line_dash,
                                                         width=1.5), legendgroup=model,
                                               showlegend=True, xaxis='x1', yaxis='y1'))

                # Add CAP marker
                if ts_str in cap_data.get(model, {}):
                    temp_cap, height_cap = cap_data[model][ts_str]
                    frame_traces.append(go.Scatter(x=[temp_cap], y=[height_cap], mode='markers',
                                                   marker=dict(symbol='x', size=8, color=model_colors_temp_wind[model],
                                                               line=dict(width=0.5,
                                                                         color=model_colors_temp_wind[model])),
                                                   name=f"{model} CAP",
                                                   legendgroup=model, showlegend=False,
                                                   hovertemplate=f"{model} CAP: {height_cap:.0f}m<extra></extra>",
                                                   xaxis='x1', yaxis='y1'))

            # Add specific humidity traces (on secondary x-axis x3 = top of subplot 1)
            if ts_str in model_humidity_data.get(model, {}):
                q, height = model_humidity_data[model][ts_str]
                line_dash = icon_2te_hatpro_linestyle if model == "ICON2TE" else "solid"

                frame_traces.append(go.Scatter(x=q, y=height, mode='lines', name=f"{model} q",
                                               line=dict(color=model_colors_humidity[model], dash=line_dash, width=1.0),
                                               legendgroup=model,
                                               showlegend=False, xaxis='x3', yaxis='y1'))

        # Add observations (only for Innsbruck points; names in confg always start with "ibk") - subplot 1
        if point_name.startswith("ibk"):
            # Add Radiosonde: all variables but only 1 measurement at 02:15 UTC
            if "radiosonde" in obs_data:
                temp, height = obs_data["radiosonde"]
                frame_traces.append(go.Scatter(x=temp, y=height, mode='lines', name="Radiosonde (from 02:18 UTC)",
                                               line=dict(color=model_colors_temp_wind["Radiosonde"], width=1.5),
                                               legendgroup="Radiosonde",
                                               showlegend=True, xaxis='x1', yaxis='y1'))

                if "radiosonde_cap" in obs_data:
                    temp_cap, height_cap = obs_data["radiosonde_cap"]
                    frame_traces.append(go.Scatter(x=[temp_cap], y=[height_cap], mode='markers',
                                                   marker=dict(symbol='x', size=8,
                                                               color=model_colors_temp_wind["Radiosonde"],
                                                               line=dict(width=0.5,
                                                                         color=model_colors_temp_wind["Radiosonde"])),
                                                   name="Radiosonde CAP", legendgroup="Radiosonde", showlegend=False,
                                                   hovertemplate=f"Radiosonde CAP: {height_cap:.0f}m<extra></extra>",
                                                   xaxis='x1', yaxis='y1'))
            # Add Radiosonde humidity
            if "radiosonde_humidity" in obs_data:
                q, height = obs_data["radiosonde_humidity"]
                frame_traces.append(
                    go.Scatter(x=q, y=height, mode='lines', name="Radiosonde q",
                               line=dict(color=model_colors_humidity["Radiosonde"], width=1.0),
                               legendgroup="Radiosonde", showlegend=False, xaxis='x3', yaxis='y1'))

            # HATPRO (time-dependent)
            if "hatpro" in obs_data and ts_str in obs_data["hatpro"]:
                temp, height = obs_data["hatpro"][ts_str]
                frame_traces.append(go.Scatter(x=temp, y=height, mode='lines', name="HATPRO",
                                               line=dict(color=model_colors_temp_wind["HATPRO"], width=2.0, dash="dot"),
                                               legendgroup="HATPRO",
                                               showlegend=True, xaxis='x1', yaxis='y1'))

                cap_key = f"hatpro_cap_{ts_str}"
                if cap_key in obs_data:
                    temp_cap, height_cap = obs_data[cap_key]
                    frame_traces.append(go.Scatter(x=[temp_cap], y=[height_cap], mode='markers',
                                                   marker=dict(symbol='x', size=8,
                                                               color=model_colors_temp_wind["HATPRO"],
                                                               line=dict(width=0.8,
                                                                         color=model_colors_temp_wind["HATPRO"])),
                                                   name="HATPRO CAP",
                                                   legendgroup="HATPRO", showlegend=False,
                                                   hovertemplate=f"HATPRO CAP: {height_cap:.0f}m<extra></extra>",
                                                   xaxis='x1', yaxis='y1'))

            # HATPRO humidity (time-dependent)
            if "hatpro_humidity" in obs_data and ts_str in obs_data["hatpro_humidity"]:
                q, height = obs_data["hatpro_humidity"][ts_str]
                frame_traces.append(go.Scatter(x=q, y=height, mode='lines', name="HATPRO q",
                                               line=dict(color=model_colors_humidity["HATPRO"], width=1, dash="dot"),
                                               legendgroup="HATPRO",
                                               showlegend=False, xaxis='x3', yaxis='y1'))

        # ====== SUBPLOT 2: Wind Speed & Direction ======
        # Add wind speed traces (bottom x-axis of subplot 2) - row=1, col=2
        for model in MODEL_ORDER:
            if ts_str in model_wspd_data.get(model, {}):
                wspd, height = model_wspd_data[model][ts_str]
                line_dash = icon_2te_hatpro_linestyle if model == "ICON2TE" else "solid"

                frame_traces.append(go.Scatter(x=wspd, y=height, mode='lines', name=f"{model} wspd",
                                               line=dict(color=model_colors_temp_wind[model], dash=line_dash,
                                                         width=1.5), legendgroup=model,
                                               showlegend=False, xaxis='x2', yaxis='y2'))

            # Add wind direction traces (top x-axis x4 of subplot 2)
            if ts_str in model_udir_data.get(model, {}):
                wdir, height = model_udir_data[model][ts_str]

                # Use open circles for ICON2TE to match its dashed line style
                marker_symbol = 'circle-open' if model == "ICON2TE" else 'circle'

                frame_traces.append(go.Scatter(x=wdir, y=height, mode='markers', name=f"{model} wdir",
                                               marker=dict(color=model_colors_temp_wind[model], size=5,
                                                           symbol=marker_symbol), legendgroup=model,
                                               showlegend=False, xaxis='x4', yaxis='y2'))

        # Add observation wind data (only for Innsbruck points) - subplot 2
        if point_name.startswith("ibk"):
            # Radiosonde wind speed (constant)
            if "radiosonde_wspd" in obs_data:
                wspd, height = obs_data["radiosonde_wspd"]
                frame_traces.append(go.Scatter(x=wspd, y=height, mode='lines', name="Radiosonde wspd",
                                               line=dict(color=model_colors_temp_wind["Radiosonde"], width=1.5),
                                               legendgroup="Radiosonde", showlegend=False, xaxis='x2', yaxis='y2'))

            # Radiosonde wind direction (constant)
            if "radiosonde_udir" in obs_data:
                wdir, height = obs_data["radiosonde_udir"]
                frame_traces.append(go.Scatter(x=wdir, y=height, mode='markers', name="Radiosonde wdir",
                                               marker=dict(color=model_colors_temp_wind["Radiosonde"], size=5,
                                                           symbol='circle'),
                                               legendgroup="Radiosonde", showlegend=False, xaxis='x4', yaxis='y2'))

        # Format current timestamp for this frame
        formatted_ts = pd.to_datetime(ts_str).strftime('%dth %H:%M')
        frames.append(go.Frame(data=frame_traces, name=ts_str,
                               layout=go.Layout(
                                   title_text=f"Vertical profiles at {point['name']}, {point['height']} m - {formatted_ts} UTC")))

    # Add initial data (first timestep)
    for trace in frames[0].data:
        fig.add_trace(trace, row=1, col=1 if trace.xaxis in ['x1', 'x3'] else 2)

    # Assign frames
    fig.frames = frames

    # Create slider
    sliders = [dict(active=0, yanchor="top", y=-0.12, xanchor="left", x=0.1,
                    currentvalue=dict(prefix="Time: ", visible=True, xanchor="center", font=dict(size=14)),
                    pad=dict(b=10, t=50),
                    len=0.8, transition=dict(duration=0), steps=[dict(
            args=[[ts_str], dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))],
            label="",  # Empty label
            method="animate") for ts_str in timestamps])]

    # Update layout
    formatted_ts_initial = pd.to_datetime(timestamps[0]).strftime('%dth %H:%M')
    fig.update_layout(title_text=f"Vertical profiles at {point['name']}, {point['height']} m - {formatted_ts_initial}",
                      height=700, width=1400, hovermode='closest', template='plotly_white', sliders=sliders,
                      legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5,
                                  bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="lightgray", borderwidth=1),
                      updatemenus=[
                          dict(type="buttons", direction="left", x=0.0, y=-0.12, xanchor="left", yanchor="top",
                               pad=dict(t=10, b=10),
                               buttons=[dict(label="▶ Play", method="animate",
                                             args=[None, dict(frame=dict(duration=800, redraw=True),
                                                              fromcurrent=True, mode="immediate",
                                                              transition=dict(duration=0))]),
                                        dict(label="⏸ Pause", method="animate",
                                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                mode="immediate",
                                                                transition=dict(duration=0))])])])

    # Update x-axes and y-axes
    # Subplot 1 (Temperature & Humidity)
    fig.update_xaxes(title_text="Temperature [°C]", range=[8, 20], row=1, col=1)
    fig.update_yaxes(title_text="Height above terrain [m]", range=[0, plot_max_height], row=1, col=1)

    # Subplot 2 (Wind) - no y-axis labels (redundant with left plot), but synchronized zoom
    fig.update_xaxes(title_text="Wind Speed [m/s]", range=[0, 10], row=1, col=2)
    fig.update_yaxes(title_text="", range=[0, plot_max_height], showticklabels=False, matches='y', row=1, col=2)

    # Add secondary x-axis for humidity (top of subplot 1) - x3
    fig.update_layout(
        xaxis3=dict(title="Specific Humidity [g/kg]", overlaying='x', side='top', range=[0, max_q], anchor='y',
                    showgrid=False))

    # Add secondary x-axis for wind direction (top of subplot 2) - x4
    fig.update_layout(xaxis4=dict(title="Wind Direction [°]", overlaying='x2', side='top', range=[0, 360], anchor='y2',
                                  showgrid=False))
    return fig


def plot_save_all_points_with_slider(start_time: str = "2017-10-16T00:00:00", end_time: str = "2017-10-16T12:00:00",
                                     time_step_hours: float = 1.0, max_height: float = 5000,
                                     plot_max_height: float = 2000, point_names: List[str] = ALL_POINTS) -> None:
    """
    Create and save individual slider plots for each point location.
    
    Creates one HTML file per point, each with its own time slider.
    This is more reliable than trying to animate small multiples.
    
    Args:
        start_time: Start timestamp ISO format (e.g. "2017-10-16T00:00:00")
        end_time: End timestamp ISO format (e.g. "2017-10-16T12:00:00")
        time_step_hours: Time step in hours between frames (default: 1.0)
        max_height: Maximum height in meters to load data (default: 5000m)
        plot_max_height: Maximum height in meters to display initially (default: 2000m)
        point_names: List of points to plot (default: ALL_POINTS)
    """
    print(f"\n{'=' * 70}")
    print(f"Creating individual slider plots for {len(point_names)} points")
    print(f"Time range: {start_time} to {end_time}, step: {time_step_hours}h")
    print(f"{'=' * 70}\n")
    # Generate list of timestamps
    timestamps = pd.date_range(start=start_time, end=end_time, freq=f"{int(time_step_hours * 60)}min").strftime(
        "%Y-%m-%dT%H:%M:%S").tolist()

    print(f"Total timesteps: {len(timestamps)}\n")
    # Ensure output directory exists
    html_dir = os.path.join(confg.dir_PLOTS, "vertical_plots")
    os.makedirs(html_dir, exist_ok=True)

    # Create plot for each point
    for point_name in point_names:
        try:
            point = getattr(confg, point_name, None)
            if point is None:
                print(f"⚠ Skipping {point_name} - not found in confg")
                continue

            print(f"\n{'-' * 70}")
            print(f"Processing: {point['name']} ({point_name})")
            print(f"{'-' * 70}")

            # Create the plot with slider
            fig = plot_single_point_with_slider(point_name, timestamps=timestamps, max_height=max_height,
                                                plot_max_height=plot_max_height)
            # Save to HTML
            html_path = os.path.join(html_dir, f"vertical_profile_{point_name}_slider.html")
            fig.write_html(html_path)
            print(f"✓ Saved: {html_path}")

        except Exception as e:
            print(f"✗ Error processing {point_name}: {e}")
            continue

    print(f"\n{'=' * 70}")
    print(f"✓ All plots created successfully!")
    print(f"  Location: {html_dir}")
    print(f"  - Use the slider to move through timesteps")
    print(f"  - Click 'Play' to animate")
    print(f"  - Data loaded up to {max_height}m, displayed up to {plot_max_height}m")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    # Create vertical temperature profile plot for 04:00 UTC on October 16, 2017
    # Data is loaded up to 5000m but initially displayed up to 2000m
    # Users can zoom out in the interactive HTML to see higher altitudes
    # plot_save_vertical_profiles(timestamp="2017-10-16T04:00:00", max_height=5000, plot_max_height=2000)

    # Create interactive plot with time slider
    # Shows profiles from midnight to noon on October 16, 2017
    plot_save_all_points_with_slider(start_time="2017-10-15T14:00:00", end_time="2017-10-16T12:00:00",
                                     time_step_hours=0.5, max_height=3000, plot_max_height=800)
                                     # point_names=["ibk_uni", "telfs"])
