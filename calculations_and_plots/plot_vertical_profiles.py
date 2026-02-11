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

Would plotting VHD-area be useful?
"""
# from __future__ import annotations
import fix_win_DLL_loading_issue

fix_win_DLL_loading_issue

import os
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr

from plotly.subplots import make_subplots

import confg
from confg import model_colors_temp_wind, model_colors_humidity  # , icon_2te_hatpro_linestyle
from read_in_hatpro_radiosonde import read_radiosonde_dataset
from calculations_and_plots.calc_cap_height import cap_height_profile
# Import timeseries management functions
from calculations_and_plots.manage_timeseries import (load_or_read_timeseries, MODEL_ORDER)


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
        ts_array: List[np.datetime64], model_data: dict) -> dict:
    """
    Compute CAP heights from timeseries (point) data.
    Returns dict mapping timestamp strings to (temp_at_cap, cap_height) tuples (tuple is needed so that the
    cap-marker is at the right temperature in the plot afterwards).
    """
    cap_data = {}

    ds_with_cap = cap_height_profile(ds_filtered, consecutive=3, model=model)
    cap_height_da = ds_with_cap["cap_height"]

    # Extract CAP heights for each timestamp
    for ts_str, ts in zip(timestamps, ts_array):
        cap_height = _get_cap_height_value(cap_height_da, point, ts)

        if np.isnan(cap_height):
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


def _create_vhd_area_trace(th_values: np.ndarray, height_values: np.ndarray, th_hafelekar: float, point_height: float,
        color: str, name: str, legendgroup: str) -> go.Scatter:
    """
    Create a filled area trace for VHD (Vertical Heat Deficit) visualization.

    The VHD area is between:
    - A vertical (neutral, well mixed) line at the minimum temperature (from terrain to hafelekar height)
    - The potential temperature profile

    The area represents the heat deficit: the difference between a well-mixed neutral state
    and the actual stratified temperature profile.

    Args:
        th_values: Potential temperature values [K]
        height_values: Height values above terrain [m]
        th_hafelekar: Potential temperature at Hafelekar height [K] (used for height limit)
        point_height: Terrain height of the point location [m]
        color: Color for the fill
        name: Name for the trace
        legendgroup: Legend group for linking with main trace

    Returns:
        Plotly Scatter trace with fill
    """
    # Calculate the height relative to terrain for Hafelekar
    hafelekar_height_above_terrain = confg.hafelekar_height - point_height

    # Filter data to only include heights up to Hafelekar
    mask = height_values <= hafelekar_height_above_terrain
    th_filtered = th_values[mask]
    height_filtered = height_values[mask]

    if len(height_filtered) == 0:
        # Return empty trace if no data
        return go.Scatter(x=[], y=[], mode='none', showlegend=False)

    # Combine: vertical line up + profile points back down (reversed)
    if name in ["AROME", "ICON", "ICON2TE"]:
        th_filtered_model, height_filtered_model = th_filtered, height_filtered
    else:
        th_filtered_model, height_filtered_model = (th_filtered[::-1],  height_filtered[::-1])
        # need to turn around due to different indexing

    # more intuitive, direct approach: for x I need the pot. temperatures; and for y heights!
    # Rectangle bottom-right corner to top-right, then follow profile back down
    x_fill = np.concatenate([[th_filtered.min(), th_filtered.max(), th_filtered.max()], th_filtered_model])
    y_fill = np.concatenate([[height_filtered.min(), height_filtered.min(), height_filtered.max()], height_filtered_model])

    # Create the filled polygon:
    # 1. Start with the vertical line at(min) from (bottom to top)
    # 2. Go back down following the temperature profile
    # 3. Close at bottom
    # The area between the vertical line (well-mixed) and the profile (stratified) is the VHD
    # x_fill = np.concatenate([[np.min(th_filtered), np.min(th_filtered)], th_filtered_model])
    # y_fill = np.concatenate([[np.min(height_filtered), np.max(height_filtered)], height_filtered_model])


    # Convert color to rgba with transparency; is this working?
    if 'rgba' in color:
        # Already rgba format - replace the alpha value
        import re
        fillcolor = re.sub(r',\s*[\d.]+\s*\)', ', 0.025)', color)
    elif 'rgb' in color:
        # rgb format - convert to rgba
        fillcolor = color.replace('rgb', 'rgba').replace(')', ', 0.025)')
    else:
        # Hex or named color - use as is with opacity parameter
        fillcolor = color

    return go.Scatter(x=x_fill, y=y_fill, mode='none', fill='toself', fillcolor=fillcolor, opacity=0.2,
        # Make transparent (20% opacity)
        name=f"{name} VHD", legendgroup=legendgroup, showlegend=False, hoverinfo='skip', xaxis='x1', yaxis='y1')


def plot_single_point_with_slider(point_name: str, timestamps: List[str], plot_max_height: float = 2000,
        variables: list = ["udir", "wspd", "Td_dep", "p", "th", "temp", "z", "z_unstag"],
        temperature_var: str = "temp") -> go.Figure:
    """
    Create an interactive plot with time slider for a single point location.
    
    Users can slide through different timesteps and see the temperature profiles update dynamically.
    The plot shows all models and observations at one location.
    
    Args:
        point_name: Point location name from confg.py (e.g. "ibk_villa")
        timestamps: List of ISO format timestamp strings
        max_height: Maximum height in meters to load data (default: 5000m)
        plot_max_height: Maximum height in meters to display on y-axis (default: 2000m)
        temperature_var: Temperature variable to plot - "temp" for temperature in °C or "th" for potential temperature in K (default: "temp")

    Returns:
        Plotly figure object with the interactive time slider
    """
    point = confg.ALL_POINTS.get(point_name)
    if point is None:
        raise ValueError(f"Point {point_name} not found in confg")

    print(f"Creating slider plot for {point['name']} with {len(timestamps)} timesteps...")

    # Convert timestamp strings to numpy datetime64
    ts_array = [np.datetime64(ts) for ts in timestamps]

    # Pre-load all data for all timesteps
    print(f"  Loading data for all models and timesteps...")
    model_data = {}  # {model: {timestamp: (temp, height)}}
    model_humidity_data = {}  # {model: {timestamp: (q, height)}}
    model_dewpoint_dep_data = {}  # {model: {timestamp: (Td_dep, height)}}
    model_wspd_data = {}  # {model: {timestamp: (wspd, height)}}
    model_udir_data = {}  # {model: {timestamp: (wdir, height)}}
    obs_data = {}  # {obs_type: data}
    cap_data = {}  # {model: {timestamp: (temp_at_cap, cap_height)}}
    vhd_hafelekar_th = {}  # {timestamp: th_hafelekar} - potential temp at Hafelekar height

    # Load model data
    for model in MODEL_ORDER:
        model_data[model] = {}
        model_humidity_data[model] = {}
        model_dewpoint_dep_data[model] = {}
        model_wspd_data[model] = {}
        model_udir_data[model] = {}

        # Load timeseries dataset, for model data
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
        # ds = ds.where(height_var <= max_height, drop=True)  # I formerly subsetted the dataset

        # Extract temperature and height values for each timestamp
        for ts_str, ts in zip(timestamps, ts_array):
            temp = ds[temperature_var].sel(time=ts).values
            height = ds.coords["height"].values

            # Filter NaNs for temperature
            valid = ~np.isnan(temp) & ~np.isnan(height)
            model_data[model][ts_str] = (temp[valid], height[valid])

            # For VHD calculation: get potential temperature at Hafelekar height (only for th plots)
            if temperature_var == "th" and ts_str not in vhd_hafelekar_th:
                hafelekar_height_above_terrain = confg.hafelekar_height - point["height"]
                # Find th at Hafelekar height by interpolation
                if len(height[valid]) > 0 and hafelekar_height_above_terrain <= height[valid].max():
                    th_hafelekar = np.interp(hafelekar_height_above_terrain, height[valid], temp[valid])
                    vhd_hafelekar_th[ts_str] = th_hafelekar

            # deprecated: Extract humidity data (q) if available
            if "q" in ds:
                q = ds["q"].sel(time=ts).values
                # Convert from kg/kg to g/kg
                q = q * 1000
                # Filter NaNs for humidity
                valid_q = ~np.isnan(q) & ~np.isnan(height)
                model_humidity_data[model][ts_str] = (q[valid_q], height[valid_q])

            # Extract dewpoint depression (Td_dep) if available
            if "Td_dep" in ds:
                Td_dep = ds["Td_dep"].sel(time=ts).values
                # Filter NaNs for dewpoint depression
                valid_Td_dep = ~np.isnan(Td_dep) & ~np.isnan(height)
                model_dewpoint_dep_data[model][ts_str] = (Td_dep[valid_Td_dep], height[valid_Td_dep])

            # Extract wind speed data (wspd) if available
            if "wspd" in ds:
                wspd = ds["wspd"].sel(time=ts).values
                valid_wspd = ~np.isnan(wspd) & ~np.isnan(height)
                model_wspd_data[model][ts_str] = (wspd[valid_wspd], height[valid_wspd])

            # Extract wind direction data (udir) if available
            if "udir" in ds:
                udir = ds["udir"].sel(time=ts).values
                valid_udir = ~np.isnan(udir) & ~np.isnan(height)
                model_udir_data[model][ts_str] = (udir[valid_udir], height[valid_udir])

        # Load or compute CAP heights
        try:
            cap_data[model] = _load_or_compute_cap_heights(model=model, ds_filtered=ds, point=point,
                                                           timestamps=timestamps, ts_array=ts_array,
                                                           model_data=model_data)

        except Exception as e:
            print(f"    Warning: Could not load/compute CAP height for {model}: {e}")

        ds.close()

    # Load observation data (only for Innsbruck points)
    if point_name.startswith("ibk"):
        # Radiosonde (no time dimension)
        try:
            ds_radiosonde = read_radiosonde_dataset(height_as_z_coord="above_terrain")
            # ds = ds_radiosonde.where(ds_radiosonde["height"] <= max_height, drop=True)  # formerly subsetted max
            # height...
            temp = ds_radiosonde[temperature_var].values
            height = ds_radiosonde["height"].values
            valid = ~np.isnan(temp) & ~np.isnan(height)
            obs_data["radiosonde"] = (temp[valid], height[valid])

            # deprecated: Extract humidity data (q) if available
            if "q" in ds_radiosonde:
                q = ds_radiosonde["q"].values  # in kg/kg
                # Convert from kg/kg to g/kg
                q = q * 1000
                valid_q = ~np.isnan(q) & ~np.isnan(height)  # filter NaNs ...
                obs_data["radiosonde_humidity"] = (q[valid_q], height[valid_q])

            # Extract dewpoint depression (Td_dep) if available
            if "Td_dep" in ds_radiosonde:
                Td_dep = ds_radiosonde["Td_dep"].values  # in °C
                valid_Td_dep = ~np.isnan(Td_dep) & ~np.isnan(height)  # filter NaNs ...
                obs_data["radiosonde_Td_dep"] = (Td_dep[valid_Td_dep], height[valid_Td_dep])

            # Extract wind speed data (wspd) if available
            if "wspd" in ds_radiosonde:
                wspd = ds_radiosonde["wspd"].values
                valid_wspd = ~np.isnan(wspd) & ~np.isnan(height)
                obs_data["radiosonde_wspd"] = (wspd[valid_wspd], height[valid_wspd])

            # Extract wind direction data (udir) if available
            if "udir" in ds_radiosonde:
                udir = ds_radiosonde["udir"].values
                valid_udir = ~np.isnan(udir) & ~np.isnan(height)
                obs_data["radiosonde_udir"] = (udir[valid_udir], height[valid_udir])

            # Radiosonde CAP height was searched in plot and defined in confg
            if not np.isnan(confg.radiosonde_cap_height) and confg.radiosonde_cap_height:  # <= max_height
                idx = np.argmin(np.abs(height[valid] - confg.radiosonde_cap_height))
                obs_data["radiosonde_cap"] = (temp[valid][idx], confg.radiosonde_cap_height)

            ds_radiosonde.close()
        except Exception as e:
            print(f"    Warning: Error in loading Radiosonde: {e}")

        # SL88 LIDAR (time-dependent wind data)
        try:
            if os.path.exists(confg.lidar_sl88_merged_path):
                print(f"    Loading SL88 LIDAR data")
                ds_lidar = xr.open_dataset(confg.lidar_sl88_merged_path)

                obs_data["lidar_wspd"] = {}
                obs_data["lidar_udir"] = {}

                for ts_str, ts in zip(timestamps, ts_array):
                    # Select nearest time
                    ds_ts = ds_lidar.sel(time=ts, method='nearest', tolerance='1min')

                    # Get height values in meters (from height_m coordinate)
                    if 'height_m' in ds_ts.coords:
                        height = ds_ts['height_m'].values
                    else:
                        print(f"    Warning: No height_m coordinate in SL88 LIDAR data")
                        continue

                    # Filter to max_height
                    # height_mask = height <= max_height
                    # height = height[height_mask]

                    # Extract wind speed (ff variable)
                    if 'ff' in ds_ts:
                        wspd = ds_ts['ff'].values
                        wspd_filtered = wspd  # [height_mask]
                        # valid_wspd = ~np.isnan(wspd_filtered) & ~np.isnan(height)
                        ds_lidar.ucomp_unfiltered.dropna(dim="height")
                        obs_data["lidar_wspd"][ts_str] = (wspd, height)  # [valid_wspd]

                    # Extract wind direction (dd variable)
                    if 'dd' in ds_ts:
                        wdir = ds_ts['dd'].values
                        wdir_filtered = wdir  # [height_mask]
                        # valid_wdir = ~np.isnan(wdir_filtered) & ~np.isnan(height)
                        obs_data["lidar_udir"][ts_str] = (wdir_filtered, height)  # [valid_wdir]

                ds_lidar.close()
                print(f"    ✓ SL88 LIDAR data loaded successfully")
            else:
                print(f"    Warning: SL88 LIDAR merged file not found at {confg.lidar_sl88_merged_path}")
        except Exception as e:
            print(f"    Warning: Error in loading SL88 LIDAR: {e}")

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
                # ds_hatpro_filtered = ds_hatpro.where(ds_hatpro["height"] <= max_height, drop=True)
                ds_hatpro_with_cap = cap_height_profile(ds_hatpro, consecutive=3, model="HATPRO")

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
                obs_data["hatpro_humidity"] = {}  # deprecated
                obs_data["hatpro_Td_dep"] = {}
                obs_data["hatpro_wspd"] = {}
                obs_data["hatpro_udir"] = {}

                for ts_str, ts in zip(timestamps, ts_array):
                    ds_ts = ds_hatpro.sel(time=ts)
                    # ds = ds_ts.where(ds_ts["height"] <= max_height, drop=True)

                    temp = ds_ts[temperature_var].values
                    height = ds_ts["height"].values
                    valid = ~np.isnan(temp) & ~np.isnan(height)
                    obs_data["hatpro"][ts_str] = (temp[valid], height[valid])
                    # creates large dict w. radiosonde w. cap_height & hatpro data for that timestamp

                    # deprecated (not used): Extract humidity data (q) if available
                    if "q" in ds_ts:
                        q = ds_ts["q"].values  # in kg/kg
                        # Convert from kg/kg to g/kg
                        q = q * 1000
                        valid_q = ~np.isnan(q) & ~np.isnan(height)  # filter NaNs ...
                        obs_data["hatpro_humidity"][ts_str] = (q[valid_q], height[valid_q])

                    # Extract dewpoint depression (Td_dep) if available
                    if "Td_dep" in ds_ts:
                        Td_dep = ds_ts["Td_dep"].values  # in °C
                        valid_Td_dep = ~np.isnan(Td_dep) & ~np.isnan(height)  # filter NaNs ...
                        obs_data["hatpro_Td_dep"][ts_str] = (Td_dep[valid_Td_dep], height[valid_Td_dep])

                    # CAP height
                    if hatpro_cap_da is not None:
                        cap_height = hatpro_cap_da.sel(time=ts, method="nearest").item()
                        if not np.isnan(cap_height):  # and cap_height <= max_height
                            idx = np.argmin(np.abs(height[valid] - cap_height))
                            key = f"hatpro_cap_{ts_str}"
                            obs_data[key] = (temp[valid][idx], cap_height)

                ds_hatpro.close()
        except Exception as e:
            print(f"    Warning: Error in loading HATPRO: {e}")

    print(f"  Creating frames...")

    # max_q = 30  # deprecated: Maximum humidity for scale
    max_Td_dep = 15  # Maximum dewpoint depression for scale [°C]

    # Create figure with subplots: [Temp/Humidity] | [Wind Speed/Direction]
    fig = make_subplots(rows=1, cols=2, specs=[[{"secondary_y": False}, {"secondary_y": False}]],
                        horizontal_spacing=0.12, column_widths=[0.6, 0.4])

    # Create frames for slider - one frame per timestep
    frames = []
    for ts_str in timestamps:
        frame_traces = []

        # ====== SUBPLOT 1: Temperature & Humidity ======

        # Add VHD area traces first (so they appear behind the lines)
        if temperature_var == "th" and ts_str in vhd_hafelekar_th:
            th_hafelekar = vhd_hafelekar_th[ts_str]

            # Add VHD areas for models
            for model in MODEL_ORDER:
                if ts_str in model_data.get(model, {}):
                    temp, height = model_data[model][ts_str]
                    vhd_trace = _create_vhd_area_trace(th_values=temp, height_values=height, th_hafelekar=th_hafelekar,
                        point_height=point["height"], color=model_colors_temp_wind[model], name=model,
                        legendgroup=model)
                    frame_traces.append(vhd_trace)

        # Add model traces (temperature)
        for model in MODEL_ORDER:
            if ts_str in model_data.get(model, {}):
                temp, height = model_data[model][ts_str]
                # line_dash = icon_2te_hatpro_linestyle if model == "ICON2TE" else "solid"
                frame_traces.append(go.Scatter(x=temp, y=height, mode='lines', name=model,
                                               line=dict(color=model_colors_temp_wind[model],  # dash=line_dash,
                                                         width=1.5), legendgroup=model, showlegend=True, xaxis='x1',
                                               yaxis='y1'))

                # Add CAP marker
                if ts_str in cap_data.get(model, {}):
                    temp_cap, height_cap = cap_data[model][ts_str]
                    frame_traces.append(go.Scatter(x=[temp_cap], y=[height_cap], mode='markers',
                                                   marker=dict(symbol='x', size=8, color=model_colors_temp_wind[model],
                                                               line=dict(width=0.5,
                                                                         color=model_colors_temp_wind[model])),
                                                   name=f"{model} CAP", legendgroup=model, showlegend=False,
                                                   hovertemplate=f"{model} CAP: {height_cap:.0f}m<extra></extra>",
                                                   xaxis='x1', yaxis='y1'))

            # deprecated: Add specific humidity traces (on secondary x-axis x3 = top of subplot 1)
            # if ts_str in model_humidity_data.get(model, {}):
            #     q, height = model_humidity_data[model][ts_str]
            #     line_dash = icon_2te_hatpro_linestyle if model == "ICON2TE" else "solid"
            #
            #     frame_traces.append(go.Scatter(x=q, y=height, mode='lines', name=f"{model} q",
            #                                    line=dict(color=model_colors_humidity[model], dash=line_dash, width=1.0),
            #                                    legendgroup=model, showlegend=False, xaxis='x3', yaxis='y1'))

            # Add dewpoint depression traces (on secondary x-axis x3 = top of subplot 1)
            if ts_str in model_dewpoint_dep_data.get(model, {}):
                Td_dep, height = model_dewpoint_dep_data[model][ts_str]
                # line_dash = icon_2te_hatpro_linestyle if model == "ICON2TE" else "solid"

                frame_traces.append(
                    go.Scatter(x=Td_dep, y=height, mode='lines', name=f"{model} Td_dep",  # dash=line_dash,
                               line=dict(color=model_colors_humidity[model], width=1.0), legendgroup=model,
                               showlegend=False, xaxis='x3', yaxis='y1'))

        # Add observations (only for Innsbruck points; names in confg always start with "ibk") - subplot 1
        if point_name.startswith("ibk"):
            # Add VHD area for Radiosonde (only for potential temperature)
            if temperature_var == "th" and ts_str in vhd_hafelekar_th and "radiosonde" in obs_data:
                th_hafelekar = vhd_hafelekar_th[ts_str]
                temp, height = obs_data["radiosonde"]
                vhd_trace = _create_vhd_area_trace(th_values=temp, height_values=height, th_hafelekar=th_hafelekar,
                    point_height=point["height"], color=model_colors_temp_wind["Radiosonde"], name="Radiosonde",
                    legendgroup="Radiosonde")
                frame_traces.append(vhd_trace)

            # Add Radiosonde: all variables but only 1 measurement at 02:15 UTC
            if "radiosonde" in obs_data:
                temp, height = obs_data["radiosonde"]
                frame_traces.append(go.Scatter(x=temp, y=height, mode='lines', name="Radiosonde (from 02:18 UTC)",
                                               line=dict(color=model_colors_temp_wind["Radiosonde"], dash="dot",
                                                         width=1.5), legendgroup="Radiosonde", showlegend=True,
                                               xaxis='x1', yaxis='y1'))

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
            # deprecated: Add Radiosonde humidity
            # if "radiosonde_humidity" in obs_data:
            #     q, height = obs_data["radiosonde_humidity"]
            #     frame_traces.append(go.Scatter(x=q, y=height, mode='lines', name="Radiosonde q",
            #                                    line=dict(color=model_colors_humidity["Radiosonde"], width=1.0),
            #                                    legendgroup="Radiosonde", showlegend=False, xaxis='x3', yaxis='y1'))

            # Add Radiosonde dewpoint depression
            if "radiosonde_Td_dep" in obs_data:
                Td_dep, height = obs_data["radiosonde_Td_dep"]
                frame_traces.append(go.Scatter(x=Td_dep, y=height, mode='lines', name="Radiosonde Td_dep",
                                               line=dict(color=model_colors_humidity["Radiosonde"], dash="dot",
                                                         width=1.0), legendgroup="Radiosonde", showlegend=False,
                                               xaxis='x3', yaxis='y1'))

            # HATPRO (time-dependent)
            if "hatpro" in obs_data and ts_str in obs_data["hatpro"]:
                # Add VHD area for HATPRO (only for potential temperature)
                if temperature_var == "th" and ts_str in vhd_hafelekar_th:
                    th_hafelekar = vhd_hafelekar_th[ts_str]
                    temp, height = obs_data["hatpro"][ts_str]
                    vhd_trace = _create_vhd_area_trace(th_values=temp, height_values=height, th_hafelekar=th_hafelekar,
                        point_height=point["height"], color=model_colors_temp_wind["HATPRO"], name="HATPRO",
                        legendgroup="HATPRO")
                    frame_traces.append(vhd_trace)

                temp, height = obs_data["hatpro"][ts_str]
                frame_traces.append(go.Scatter(x=temp, y=height, mode='lines', name="HATPRO",
                                               line=dict(color=model_colors_temp_wind["HATPRO"], width=1.5, dash="dot"),
                                               legendgroup="HATPRO", showlegend=True, xaxis='x1', yaxis='y1'))

                cap_key = f"hatpro_cap_{ts_str}"
                if cap_key in obs_data:
                    temp_cap, height_cap = obs_data[cap_key]
                    frame_traces.append(go.Scatter(x=[temp_cap], y=[height_cap], mode='markers',
                                                   marker=dict(symbol='x', size=8,
                                                               color=model_colors_temp_wind["HATPRO"],
                                                               line=dict(width=0.8,
                                                                         color=model_colors_temp_wind["HATPRO"])),
                                                   name="HATPRO CAP", legendgroup="HATPRO", showlegend=False,
                                                   hovertemplate=f"HATPRO CAP: {height_cap:.0f}m<extra></extra>",
                                                   xaxis='x1', yaxis='y1'))

            # deprecated: HATPRO humidity (time-dependent)
            # if "hatpro_humidity" in obs_data and ts_str in obs_data["hatpro_humidity"]:
            #     q, height = obs_data["hatpro_humidity"][ts_str]
            #     frame_traces.append(go.Scatter(x=q, y=height, mode='lines', name="HATPRO q",
            #                                    line=dict(color=model_colors_humidity["HATPRO"], width=1, dash="dot"),
            #                                    legendgroup="HATPRO", showlegend=False, xaxis='x3', yaxis='y1'))

            # HATPRO dewpoint depression (time-dependent)
            if "hatpro_Td_dep" in obs_data and ts_str in obs_data["hatpro_Td_dep"]:
                Td_dep, height = obs_data["hatpro_Td_dep"][ts_str]
                frame_traces.append(go.Scatter(x=Td_dep, y=height, mode='lines', name="HATPRO Td_dep",
                                               line=dict(color=model_colors_humidity["HATPRO"], width=1, dash="dot"),
                                               legendgroup="HATPRO", showlegend=False, xaxis='x3', yaxis='y1'))

        # ====== SUBPLOT 2: Wind Speed & Direction ======
        # Add wind speed traces (bottom x-axis of subplot 2) - row=1, col=2
        for model in MODEL_ORDER:
            if ts_str in model_wspd_data.get(model, {}):
                wspd, height = model_wspd_data[model][ts_str]
                # line_dash = icon_2te_hatpro_linestyle if model == "ICON2TE" else "solid"

                frame_traces.append(go.Scatter(x=wspd, y=height, mode='lines', name=f"{model} wspd",
                                               line=dict(color=model_colors_temp_wind[model],  # dash=line_dash,
                                                         width=1.5), legendgroup=model, showlegend=False, xaxis='x2',
                                               yaxis='y2'))

            # Add wind direction traces (top x-axis x4 of subplot 2)
            if ts_str in model_udir_data.get(model, {}):
                wdir, height = model_udir_data[model][ts_str]

                # Use open circles for ICON2TE to match its dashed line style
                # marker_symbol = 'circle-open' if model == "ICON2TE" else 'circle'

                # take filled circle for models
                frame_traces.append(go.Scatter(x=wdir, y=height, mode='markers', name=f"{model} wdir",
                                               marker=dict(color=model_colors_temp_wind[model], size=5,
                                                           symbol="circle"), legendgroup=model, showlegend=False,
                                               xaxis='x4', yaxis='y2'))

        # Add observation wind data (only for Innsbruck points) - subplot 2
        if point_name.startswith("ibk"):
            # Radiosonde wind speed (constant)
            if "radiosonde_wspd" in obs_data:
                wspd, height = obs_data["radiosonde_wspd"]
                frame_traces.append(go.Scatter(x=wspd, y=height, mode='lines', name="Radiosonde wspd",
                                               line=dict(color=model_colors_temp_wind["Radiosonde"], width=1.5,
                                                         dash="dot"), legendgroup="Radiosonde", showlegend=False,
                                               xaxis='x2', yaxis='y2'))

            # Radiosonde wind direction (constant)
            if "radiosonde_udir" in obs_data:
                wdir, height = obs_data["radiosonde_udir"]
                frame_traces.append(go.Scatter(x=wdir, y=height, mode='markers', name="Radiosonde wdir",
                                               marker=dict(color=model_colors_temp_wind["Radiosonde"], size=5,
                                                           symbol='circle-open'), legendgroup="Radiosonde",
                                               showlegend=False, xaxis='x4', yaxis='y2'))

            # SL88 LIDAR wind speed (time-dependent)
            if "lidar_wspd" in obs_data and ts_str in obs_data["lidar_wspd"]:
                wspd, height = obs_data["lidar_wspd"][ts_str]
                frame_traces.append(go.Scatter(x=wspd, y=height, mode='lines', name="SL88 LIDAR",
                                               line=dict(color=model_colors_temp_wind["HATPRO"], width=1.5, dash="dot"),
                                               legendgroup="SL88_LIDAR", showlegend=True, xaxis='x2', yaxis='y2'))

            # SL88 LIDAR wind direction (time-dependent)
            if "lidar_udir" in obs_data and ts_str in obs_data["lidar_udir"]:
                wdir, height = obs_data["lidar_udir"][ts_str]
                frame_traces.append(go.Scatter(x=wdir, y=height, mode='markers', name="SL88 LIDAR wdir",
                                               marker=dict(color=model_colors_temp_wind["HATPRO"], size=5,
                                                           symbol='circle-open'), legendgroup="SL88_LIDAR",
                                               showlegend=False, xaxis='x4', yaxis='y2'))

        # Format current timestamp for this frame
        formatted_ts = pd.to_datetime(ts_str).strftime('%dth %H:%M')
        frames.append(go.Frame(data=frame_traces, name=ts_str, layout=go.Layout(
            title_text=f"Vertical profiles at {point['name']}, {point['height']} m - {formatted_ts} UTC")))

    # Add initial data (first timestep)
    for trace in frames[0].data:
        fig.add_trace(trace, row=1, col=1 if trace.xaxis in ['x1', 'x3'] else 2)

    # Assign frames
    fig.frames = frames

    # Create slider
    sliders = [dict(active=0, yanchor="top", y=-0.12, xanchor="left", x=0.1,
                    currentvalue=dict(prefix="Time: ", visible=True, xanchor="center", font=dict(size=14)),
                    pad=dict(b=10, t=50), len=0.8, transition=dict(duration=0), steps=[dict(
            args=[[ts_str], dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))],
            label="",  # Empty label
            method="animate") for ts_str in timestamps])]

    # Update layout
    formatted_ts_initial = pd.to_datetime(timestamps[0]).strftime('%dth %H:%M')
    fig.update_layout(title_text=f"Vertical profiles at {point['name']}, {point['height']} m - {formatted_ts_initial}",
                      height=700, width=1400, hovermode='closest', template='plotly_white', sliders=sliders,
                      legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5,
                                  bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="lightgray", borderwidth=1),
                      updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.12, xanchor="left", yanchor="top",
                                        pad=dict(t=10, b=10), buttons=[dict(label="▶ Play", method="animate",
                                                                            args=[None, dict(
                                                                                frame=dict(duration=800, redraw=True),
                                                                                fromcurrent=True, mode="immediate",
                                                                                transition=dict(duration=0))]),
                                                                       dict(label="⏸ Pause", method="animate",
                                                                            args=[[None], dict(
                                                                                frame=dict(duration=0, redraw=False),
                                                                                mode="immediate",
                                                                                transition=dict(duration=0))])])])

    # Update x-axes and y-axes
    # Subplot 1 (Temperature & Humidity)
    if temperature_var == "th":
        fig.update_xaxes(title_text="Potential Temperature [K]", range=[280, 307], row=1, col=1)
    else:
        fig.update_xaxes(title_text="Temperature [°C]", range=[8, 25], row=1, col=1)
    fig.update_yaxes(title_text="Height above terrain [m]", range=[0, plot_max_height], row=1, col=1)

    # Subplot 2 (Wind) - no y-axis labels (redundant with left plot), but synchronized zoom
    fig.update_xaxes(title_text="Wind Speed [m/s]", range=[0, 10], row=1, col=2)
    fig.update_yaxes(title_text="", range=[0, plot_max_height], showticklabels=False, showgrid=True, matches='y',
                     row=1, col=2)

    # deprecated: Add secondary x-axis for humidity (top of subplot 1) - x3
    # fig.update_layout(
    #     xaxis3=dict(title="Specific Humidity [g/kg]", overlaying='x', side='top', range=[0, max_q], anchor='y',
    #                 showgrid=False))

    # Add secondary x-axis for dewpoint depression (top of subplot 1) - x3
    fig.update_layout(
        xaxis3=dict(title="Dewpoint Depression [°C]", overlaying='x', side='top', range=[0, 50], anchor='y',
                    showgrid=False))

    # Add secondary x-axis for wind direction (top of subplot 2) - x4
    fig.update_layout(xaxis4=dict(title="Wind Direction", overlaying='x2', side='top', range=[0, 360], anchor='y2',
                                  showgrid=False,
                                  tickmode='array',
                                  tickvals=[0, 90, 180, 270, 360],
                                  ticktext=['N', 'E', 'S', 'W', 'N']))
    return fig


def plot_save_all_points_with_slider(start_time: str = "2017-10-12T00:00:00", end_time: str = "2017-10-16T12:00:00",
        time_step_hours: float = 1.0, plot_max_height: float = 2000, point_names: List[str] = confg.ALL_POINTS,
        temperature_var: str = "temp") -> None:
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
        point_names: List of points to plot (default: confg.ALL_POINTS)
        temperature_var: Temperature variable to plot - "temp" for temperature in °C or "th" for potential temperature in K (default: "temp")
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
            point = confg.ALL_POINTS.get(point_name)
            if point is None:
                print(f"⚠ Skipping {point_name} - not found in confg")
                continue

            print(f"\n{'-' * 70}")
            print(f"Processing: {point['name']} ({point_name})")
            print(f"{'-' * 70}")

            # Create the plot with slider
            fig = plot_single_point_with_slider(point_name, timestamps=timestamps, plot_max_height=plot_max_height,
                                                temperature_var=temperature_var)
            # Save to HTML
            temp_suffix = "_theta" if temperature_var == "th" else ""
            html_path = os.path.join(html_dir, f"vertical_profile_{point_name}{temp_suffix}_slider.html")
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
    print(f"  - Data displayed up to {plot_max_height}m")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    # Create vertical temperature profile plot for 04:00 UTC on October 16, 2017
    # Data is loaded up to 5000m but initially displayed up to 2000m
    # Users can zoom out in the interactive HTML to see higher altitudes
    # plot_save_vertical_profiles(timestamp="2017-10-16T04:00:00", max_height=5000, plot_max_height=2000)

    # Create interactive plot with time slider - Temperature in °C
    # Shows profiles from midnight to noon on October 16, 2017
    plot_save_all_points_with_slider(start_time="2017-10-15T12:00:00", end_time="2017-10-16T12:00:00",
                                     time_step_hours=0.5, plot_max_height=1650, point_names= confg.VALLEY_POINTS,
                                     # ["ibk_uni"], # ["patsch_EC_south"],
                                     temperature_var="temp")  #  ['ibk_airport', 'woergl', 'jenbach',
    # 'kufstein', 'kiefersfelden', 'telfs', 'wipp_valley', 'ziller_valley',  # 'ziller_ried'])