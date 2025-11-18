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
from colorspace import qualitative_hcl
from plotly.subplots import make_subplots

import confg
import read_icon_model_3D
import read_in_arome
import read_ukmo
import read_wrf_helen
from calculations_and_plots.calc_cap_height import cap_height_profile
# Import timeseries management functions
from calculations_and_plots.manage_timeseries import (
    get_timeseries_path,
    save_timeseries,
    load_or_read_timeseries,
    load_timeseries,
    MODEL_ORDER,
    ALL_POINTS,
    variables
)
from calculations_and_plots.calc_vhd import read_dems_calc_pcgp

# --- Color scheme for models (consistent with plot_cap_height) ---
qualitative_colors = qualitative_hcl(palette="Dark 3").colors()

# Model color mapping - ICON and ICON2TE share the same color
MODEL_COLORS = {"AROME": qualitative_colors[0], "ICON": qualitative_colors[2], "ICON2TE": qualitative_colors[2],
                # Same color as ICON, differentiated by line style
                "UM": qualitative_colors[4], "WRF": qualitative_colors[6], "Radiosonde": "grey",
                # Black for observations
                "HATPRO": "grey"  # Dark grey for HATPRO
                }

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


def _load_model_timeseries(model: str, point: dict, point_name: str, variables_list: list) -> xr.Dataset | None:
    """
    Helper function to load timeseries data for a model.
    
    Wrapper around load_or_read_timeseries from manage_timeseries module.
    
    Args:
        model: Name of the weather model
        point: Dictionary with 'lat' and 'lon' keys
        point_name: Name of the point location
        variables_list: List of variable names to read
    
    Returns:
        xarray Dataset with timeseries data, or None if loading fails
    """
    return load_or_read_timeseries(model, point, point_name, variables_list, height_as_z_coord=True)


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
    Returns dict mapping timestamp strings to (temp_at_cap, cap_height) tuples (tuple is needed so that the cap-marker is at
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


def plot_vertical_profiles_small_multiples(point_names: List[str], timestamp: str = "2017-10-16T04:00:00",
                                           max_height: float = 5000, plot_max_height: float = 2000) -> go.Figure:
    """
    deprecated?
    Create a small multiples plot of vertical temperature profiles for multiple points.
    
    Each subplot shows temperature vs height for all models at one location.
    The plot includes CAP height markers (X symbols) when available.
    Data is first loaded from saved timeseries files if available, otherwise read fresh.
    
    Args:
        point_names: List of point location names from confg.py (e.g. ["ibk_villa", "ibk_uni"])
        timestamp: ISO format timestamp string (e.g. "2017-10-16T04:00:00")
        max_height: Maximum height in meters to load data (default: 5000m)
        plot_max_height: Maximum height in meters to display on y-axis (default: 2000m)
    
    Returns:
        Plotly figure object with the small multiples plot

    # Convert timestamp string to numpy datetime64 for data selection
    ts = np.datetime64(timestamp)
    
    # Calculate grid layout: 2 columns, rows as needed
    n_points = len(point_names)
    n_cols = 2
    n_rows = int(np.ceil(n_points / n_cols))
    
    # Create subplot titles using the human-readable names from confg
    subplot_titles = []
    for point_name in point_names:
        point = getattr(confg, point_name, None)
        if point:
            # Add height information to the title
            height = point.get("height", "N/A")
            subplot_titles.append(f"{point['name']} ({height} m)")
        else:
            subplot_titles.append(point_name)
    
    # Create the subplot grid
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, vertical_spacing=0.06,
                        horizontal_spacing=0.08)
    
    # Loop through each point location
    for idx, point_name in enumerate(point_names):
        # Get point coordinates from confg
        point = getattr(confg, point_name, None)
        if point is None:
            continue
        
        # Calculate subplot position (1-indexed for plotly)
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        # Loop through each model
        for model in MODEL_ORDER:
            try:
                # Load timeseries data (from file if exists, otherwise read fresh and save)
                ds = load_or_read_timeseries(model, point, point_name, variables, height_as_z_coord=True)
                
                # --- Step 3: Filter data to maximum height (for data loading, not display) ---
                # Check if height is a coordinate (from saved files) or a variable (from fresh read)
                if "height" in ds.coords:
                    height_var = ds.coords["height"]
                elif "z_unstag" in ds and model in ["ICON", "WRF"]:
                    height_var = ds["z_unstag"]
                elif "z" in ds:
                    height_var = ds["z"]
                else:
                    print(f"Warning: No height coordinate found for {model} at {point_name}")
                    continue
                
                # Select only heights up to max_height (data loading limit)
                ds_filtered = ds.where(height_var <= max_height, drop=True)
                
                # --- Step 4: Extract temperature and height values ---
                if "temp" in ds_filtered:
                    # Get temperature at the requested timestamp
                    temp = ds_filtered["temp"].sel(time=ts, method="nearest").values
                    
                    # Get height values (handle both coordinate and variable cases)
                    if "height" in ds.coords:
                        # For timeseries files, height is already a coordinate
                        height = ds_filtered.coords["height"].values
                    else:
                        # For fresh data, extract from variable at the timestamp
                        height = height_var.sel(time=ts, method="nearest").values
                    
                    # Filter out NaN values
                    valid = ~np.isnan(temp) & ~np.isnan(height)
                    temp = temp[valid]
                    height = height[valid]
                    
                    # --- Step 5: Plot temperature profile ---
                    # Use dashed line for ICON2TE to differentiate from ICON (same color)
                    line_dash = "dash" if model == "ICON2TE" else "solid"
                    
                    # Only show legend for first subplot to avoid clutter
                    show_legend = (idx == 0)
                    
                    # Add temperature profile trace
                    fig.add_trace(go.Scatter(x=temp, y=height, mode='lines', name=model,
                                             line=dict(color=MODEL_COLORS[model], dash=line_dash, width=1.5),
                                             legendgroup=model, showlegend=show_legend), row=row, col=col)
                    
                    # --- Step 6: Add CAP height marker if available ---
                    try:
                        cap_path = default_cap_path(model)
                        if os.path.exists(cap_path):
                            # Load CAP height data
                            with xr.open_dataset(cap_path) as cap_ds:
                                cap_height_da = cap_ds["cap_height"].load()
                            
                            # Select CAP height for this point and time
                            cap_height = cap_height_da.sel(lat=point["lat"], lon=point["lon"], time=ts,
                                                           method="nearest").item()
                            
                            # Only add marker if CAP height is valid and within data range
                            if not np.isnan(cap_height) and cap_height <= max_height:
                                # Get temperature at CAP height
                                if "height" in ds_filtered.coords:
                                    temp_at_cap = ds_filtered["temp"].sel(time=ts, height=cap_height,
                                                                          method="nearest").item()
                                else:
                                    temp_at_cap = ds_filtered["temp"].sel(time=ts, method="nearest").sel(
                                        height=cap_height, method="nearest").item()
                                
                                # Add X marker at CAP height
                                fig.add_trace(go.Scatter(x=[temp_at_cap], y=[cap_height], mode='markers',
                                                         marker=dict(symbol='x', size=8, color=MODEL_COLORS[model],
                                                                     line=dict(width=0.5, color=MODEL_COLORS[model])),
                                                         name=f"{model} CAP", legendgroup=model, showlegend=False,
                                                         hovertemplate=f"{model} CAP: "
                                                                       f"{cap_height:.0f}m<extra></extra>"), row=row,
                                              col=col)
                    except Exception as e:
                        # Skip marker if CAP data not available (not an error)
                        pass
                
                # Close dataset if it was opened from file
                timeseries_path = get_timeseries_path(model, point_name)
                if ds is not None and timeseries_path and os.path.exists(timeseries_path):
                    ds.close()
            
            except Exception as e:
                print(f"Warning: Could not load {model} data for {point_name}: {e}")
                continue
        
        # --- Step 7: Add observation data for Innsbruck points ---
        if point_name.startswith("ibk"):  # Only for Innsbruck points
            # Only show legend for first Innsbruck point
            show_obs_legend = (idx == 0 or (idx == 1 and point_names[0] not in ["ibk_villa", "ibk_uni", "ibk_airport"]))
            
            try:
                # --- Radiosonde data ---
                if os.path.exists(confg.radiosonde_smoothed):
                    print(f"Loading Radiosonde data for {point_name}")
                    ds_radiosonde = xr.open_dataset(confg.radiosonde_smoothed)
                    
                    # Radiosonde has no time dimension, it's a single profile
                    # Filter to max_height (data loading limit, not display limit)
                    ds_radiosonde_filtered = ds_radiosonde.where(ds_radiosonde["height"] <= max_height, drop=True)
                    
                    # Extract temperature and height
                    temp_radiosonde = ds_radiosonde_filtered["temp"].values
                    height_radiosonde = ds_radiosonde_filtered["height"].values
                    
                    # Filter out NaN values
                    valid = ~np.isnan(temp_radiosonde) & ~np.isnan(height_radiosonde)
                    temp_radiosonde = temp_radiosonde[valid]
                    height_radiosonde = height_radiosonde[valid]
                    
                    # Add Radiosonde data trace
                    fig.add_trace(go.Scatter(x=temp_radiosonde, y=height_radiosonde, mode='lines',
                                             name="Radiosonde (from 02:15 UTC)",
                                             line=dict(color=MODEL_COLORS["Radiosonde"], width=2.5),
                                             legendgroup="Radiosonde", showlegend=show_obs_legend), row=row, col=col)
                    
                    # --- Add CAP height marker for Radiosonde ---
                    try:
                        radiosonde_cap_path = get_obs_cap_path("radiosonde")
                        if os.path.exists(radiosonde_cap_path):
                            with xr.open_dataset(radiosonde_cap_path) as cap_ds:
                                cap_height_radiosonde = cap_ds["cap_height"].item()
                            
                            if not np.isnan(cap_height_radiosonde) and cap_height_radiosonde <= max_height:
                                # Find temperature at CAP height
                                height_diff = np.abs(height_radiosonde - cap_height_radiosonde)
                                nearest_idx = np.argmin(height_diff)
                                temp_at_cap = temp_radiosonde[nearest_idx]
                                
                                # Add X marker at CAP height
                                fig.add_trace(go.Scatter(x=[temp_at_cap], y=[cap_height_radiosonde], mode='markers',
                                                         marker=dict(symbol='x', size=8,
                                                                     color=MODEL_COLORS["Radiosonde"],
                                                                     line=dict(width=0.5,
                                                                               color=MODEL_COLORS["Radiosonde"])),
                                                         name="Radiosonde CAP", legendgroup="Radiosonde",
                                                         showlegend=False, hovertemplate=f"Radiosonde CAP: "
                                                                                         f"{cap_height_radiosonde:.0f}m<extra></extra>"),
                                              row=row, col=col)
                    except Exception as e:
                        pass  # Skip marker if CAP not available
                    
                    ds_radiosonde.close()
            
            except Exception as e:
                print(f"Warning: Could not load Radiosonde data for {point_name}: {e}")
            
            try:
                # --- HATPRO data ---
                if os.path.exists(confg.hatpro_interp_arome_height_as_z):
                    print(f"Loading HATPRO data for {point_name}")
                    ds_hatpro = xr.open_dataset(confg.hatpro_interp_arome_height_as_z)
                    
                    # Select time slice for the given timestamp
                    ds_hatpro_ts = ds_hatpro.sel(time=ts)
                    
                    # Filter to max_height (data loading limit, not display limit)
                    ds_hatpro_filtered = ds_hatpro_ts.where(ds_hatpro_ts["height"] <= max_height, drop=True)
                    
                    # Extract temperature and height
                    temp_hatpro = ds_hatpro_filtered["temp"].values
                    height_hatpro = ds_hatpro_filtered["height"].values
                    
                    # Add HATPRO data trace
                    fig.add_trace(go.Scatter(x=temp_hatpro, y=height_hatpro, mode='lines', name="HATPRO",
                                             line=dict(color=MODEL_COLORS["HATPRO"], width=1.5, dash="dot"),
                                             legendgroup="HATPRO", showlegend=show_obs_legend), row=row, col=col)
                    
                    # --- Add CAP height marker for HATPRO ---
                    try:
                        hatpro_cap_path = get_obs_cap_path("hatpro")
                        if os.path.exists(hatpro_cap_path):
                            with xr.open_dataset(hatpro_cap_path) as cap_ds:
                                # Select CAP height for this timestamp
                                cap_height_hatpro = cap_ds["cap_height"].sel(time=ts, method="nearest").item()
                            
                            if not np.isnan(cap_height_hatpro) and cap_height_hatpro <= max_height:
                                # Find temperature at CAP height
                                height_diff = np.abs(height_hatpro - cap_height_hatpro)
                                nearest_idx = np.argmin(height_diff)
                                temp_at_cap = temp_hatpro[nearest_idx]
                                
                                # Add X marker at CAP height
                                fig.add_trace(go.Scatter(x=[temp_at_cap], y=[cap_height_hatpro], mode='markers',
                                                         marker=dict(symbol='x', size=8, color=MODEL_COLORS["HATPRO"],
                                                                     line=dict(width=0.5,
                                                                               color=MODEL_COLORS["HATPRO"])),
                                                         name="HATPRO CAP", legendgroup="HATPRO", showlegend=False, ),
                                              row=row, col=col)
                    except Exception as e:
                        pass  # Skip marker if CAP not available
                    
                    ds_hatpro.close()
            
            except Exception as e:
                print(f"Warning: Could not load HATPRO data for {point_name}: {e}")
    
    # --- Step 8: Configure plot layout ---
    fig.update_layout(title_text=f"Vertical Temperature Profiles at {timestamp}", height=350 * n_rows,
                      hovermode='closest', template='plotly_white',
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5))
    
    # --- Step 9: Configure axes for all subplots ---
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            # Set y-axis range to plot_max_height for initial display (but data goes up to max_height)
            fig.update_yaxes(range=[0, plot_max_height], row=i, col=j)
            
            # Set x-axis range to [8, 20]°C for all plots
            fig.update_xaxes(range=[8, 20], row=i, col=j)
            
            # Add axis labels and tick labels only for first column (left side)
            if j == 1:
                # First column: both x and y labels and tick annotations
                fig.update_xaxes(title_text="Temperature [°C]", row=i, col=j)
                fig.update_yaxes(title_text="Height [m]", row=i, col=j)
            else:
                # Second column: no labels and no tick annotations
                fig.update_xaxes(title_text="", showticklabels=False, row=i, col=j)
                fig.update_yaxes(title_text="", showticklabels=False, row=i, col=j)
    
    return fig
    """


def plot_single_point_with_slider(point_name: str, timestamps: List[str], max_height: float = 5000,
                                  plot_max_height: float = 2000, variables: list = ["udir", "wspd", "q", "p", "th",
                                                                                    "temp", "hgt", "z", "z_unstag"]) -> go.Figure:
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
    obs_data = {}  # {obs_type: data}
    cap_data = {}  # {model: {timestamp: (temp_at_cap, cap_height)}}
    
    # Load model data
    for model in MODEL_ORDER:
        model_data[model] = {}
        model_humidity_data[model] = {}
        
        # Load timeseries dataset
        ds = _load_model_timeseries(model, point, point_name, variables)
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
        
        # Extract temperature data for each timestamp
        for ts_str, ts in zip(timestamps, ts_array):
            temp = ds_filtered["temp"].sel(time=ts, method="nearest").values
            height = ds_filtered.coords["height"].values
            
            # Filter NaNs
            valid = ~np.isnan(temp) & ~np.isnan(height)
            model_data[model][ts_str] = (temp[valid], height[valid])
        
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
            if os.path.exists(confg.radiosonde_smoothed):
                print(f"    Loading Radiosonde data")
                ds_radiosonde = xr.open_dataset(confg.radiosonde_smoothed)
                ds_filtered = ds_radiosonde.where(ds_radiosonde["height"] <= max_height, drop=True)
                
                temp = ds_filtered["temp"].values
                height = ds_filtered["height"].values
                valid = ~np.isnan(temp) & ~np.isnan(height)
                obs_data["radiosonde"] = (temp[valid], height[valid])
                
                # CAP height
                radiosonde_cap_path = get_obs_cap_path("radiosonde")
                if os.path.exists(radiosonde_cap_path):
                    with xr.open_dataset(radiosonde_cap_path) as cap_ds:
                        cap_height = cap_ds["cap_height"].item()
                else:
                    # Compute CAP height from radiosonde data if file doesn't exist
                    print(f"    CAP height file not found for radiosonde, computing from data...")
                    ds_with_cap = cap_height_profile(ds_filtered, model="radiosonde")
                    cap_height = ds_with_cap["cap_height"].item()
                    
                    # Save the computed CAP height for future use
                    ds_with_cap["cap_height"].to_netcdf(radiosonde_cap_path)
                    print(f"    Computed and saved radiosonde CAP height to {radiosonde_cap_path}")
                
                if not np.isnan(cap_height) and cap_height <= max_height:
                    idx = np.argmin(np.abs(height[valid] - cap_height))
                    obs_data["radiosonde_cap"] = (temp[valid][idx], cap_height)
                
                ds_radiosonde.close()
        except Exception as e:
            print(f"    Warning: Could not load Radiosonde: {e}")
        
        # HATPRO (time-dependent)
        try:
            # Check for HATPRO file with CAP height first
            if os.path.exists(confg.hatpro_with_cap_height):
                # Load HATPRO data with pre-computed CAP height
                print(f"    Loading HATPRO data with CAP height")
                ds_hatpro = xr.open_dataset(confg.hatpro_with_cap_height)
                hatpro_cap_da = ds_hatpro["cap_height"]
                
            elif os.path.exists(confg.hatpro_interp_arome_height_as_z):  # used interpolated heights to arome levels -> change to smooth HATPRO!
                # Load original HATPRO data and compute CAP height
                print(f"    Loading HATPRO data")
                ds_hatpro = xr.open_dataset(confg.hatpro_interp_arome_height_as_z)
                
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
                
                for ts_str, ts in zip(timestamps, ts_array):
                    ds_ts = ds_hatpro.sel(time=ts)
                    ds_filtered = ds_ts.where(ds_ts["height"] <= max_height, drop=True)
                    
                    temp = ds_filtered["temp"].values
                    height = ds_filtered["height"].values
                    valid = ~np.isnan(temp) & ~np.isnan(height)
                    obs_data["hatpro"][ts_str] = (temp[valid], height[valid])  # creates large dict w. radiosonde w. cap_height &
                    # hatpro data for that timestamp
                    
                    # CAP height
                    if hatpro_cap_da is not None:
                        cap_height = hatpro_cap_da.sel(time=ts, method="nearest").item()
                        if not np.isnan(cap_height) and cap_height <= max_height:
                            idx = np.argmin(np.abs(height[valid] - cap_height))
                            key = f"hatpro_cap_{ts_str}"
                            obs_data[key] = (temp[valid][idx], cap_height)
                
                ds_hatpro.close()
        except Exception as e:
            print(f"    Warning: Could not load HATPRO: {e}")
    
    print(f"  Creating frames...")
    
    # Create figure with secondary x-axis for humidity
    fig = go.Figure()
    
    # Create frames for slider - one frame per timestep
    frames = []
    for ts_str in timestamps:
        frame_traces = []
        
        # Add model traces (temperature)
        for model in MODEL_ORDER:
            if ts_str in model_data.get(model, {}):
                temp, height = model_data[model][ts_str]
                line_dash = "dash" if model == "ICON2TE" else "solid"
                
                frame_traces.append(go.Scatter(x=temp, y=height, mode='lines', name=model,
                                               line=dict(color=MODEL_COLORS[model], dash=line_dash, width=1.5),
                                               legendgroup=model, xaxis='x1'))
                
                # Add CAP marker
                if ts_str in cap_data.get(model, {}):
                    temp_cap, height_cap = cap_data[model][ts_str]
                    frame_traces.append(go.Scatter(x=[temp_cap], y=[height_cap], mode='markers',
                                                   marker=dict(symbol='x', size=8, color=MODEL_COLORS[model],
                                                               line=dict(width=0.5, color=MODEL_COLORS[model])),
                                                   name=f"{model} CAP", legendgroup=model, showlegend=False,
                                                   hovertemplate=f"{model} CAP: {height_cap:.0f}m<extra></extra>",
                                                   xaxis='x1'))
        
        # Add humidity traces (on secondary x-axis)
        for model in MODEL_ORDER:
            if ts_str in model_humidity_data.get(model, {}):
                q, height = model_humidity_data[model][ts_str]
                line_dash = "dash" if model == "ICON2TE" else "solid"
                
                frame_traces.append(go.Scatter(x=q, y=height, mode='lines', name=f"{model} q",
                                               line=dict(color=MODEL_COLORS[model], dash=line_dash, width=1.0),
                                               legendgroup=f"{model}_humidity", xaxis='x2', visible='legendonly'
                                               # Hidden by default
                                               ))
        
        # Add observations (only for Innsbruck points)
        if point_name.startswith("ibk"):
            # Radiosonde (constant)
            if "radiosonde" in obs_data:
                temp, height = obs_data["radiosonde"]
                frame_traces.append(go.Scatter(x=temp, y=height, mode='lines', name="Radiosonde (from 02:18 UTC)",
                                               line=dict(color=MODEL_COLORS["Radiosonde"], width=2.5),
                                               legendgroup="Radiosonde", xaxis='x1'))
                
                if "radiosonde_cap" in obs_data:
                    temp_cap, height_cap = obs_data["radiosonde_cap"]
                    frame_traces.append(go.Scatter(x=[temp_cap], y=[height_cap], mode='markers',
                                                   marker=dict(symbol="x", size=8, color=MODEL_COLORS["Radiosonde"],
                                                               line=dict(width=0.5, color=MODEL_COLORS["Radiosonde"])),
                                                   name="Radiosonde CAP", legendgroup="Radiosonde", showlegend=False,
                                                   hovertemplate=f"Radiosonde CAP: {height_cap:.0f}m<extra></extra>",
                                                   xaxis='x1'))
            
            # HATPRO (time-dependent)
            if "hatpro" in obs_data and ts_str in obs_data["hatpro"]:
                temp, height = obs_data["hatpro"][ts_str]
                frame_traces.append(go.Scatter(x=temp, y=height, mode='lines', name="HATPRO",
                                               line=dict(color=MODEL_COLORS["HATPRO"], width=2.5, dash="dot"),
                                               legendgroup="HATPRO", xaxis='x1'))
                
                cap_key = f"hatpro_cap_{ts_str}"
                if cap_key in obs_data:
                    temp_cap, height_cap = obs_data[cap_key]
                    frame_traces.append(go.Scatter(x=[temp_cap], y=[height_cap], mode='markers',
                                                   marker=dict(symbol='x', size=8, color=MODEL_COLORS["HATPRO"],
                                                               line=dict(width=0.5, color=MODEL_COLORS["HATPRO"])),
                                                   name="HATPRO CAP", legendgroup="HATPRO", showlegend=False,
                                                   hovertemplate=f"HATPRO CAP: {height_cap:.0f}m<extra></extra>",
                                                   xaxis='x1'))
        
        frames.append(go.Frame(data=frame_traces, name=ts_str, layout=go.Layout(
            title_text=f"Vertical Temperature Profile at {point['name']} - {ts_str}")))
    
    # Add initial data (first timestep)
    for trace in frames[0].data:
        fig.add_trace(trace)
    
    # Assign frames
    fig.frames = frames
    
    # Create slider
    sliders = [dict(active=0, yanchor="top", y=-0.15, xanchor="left", x=0.1,
                    currentvalue=dict(prefix="Time: ", visible=True, xanchor="center", font=dict(size=16)),
                    pad=dict(b=10, t=50), len=0.8, transition=dict(duration=0), steps=[dict(
            args=[[ts_str], dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))],
            label="",  # Empty label to hide slider annotations
            method="animate") for ts_str in timestamps])]
    
    # Update layout with dual x-axes
    fig.update_layout(title_text=f"Vertical Temperature Profile at {point['name']} - {timestamps[0]}", height=700,
                      width=1200, hovermode='closest', template='plotly_white', sliders=sliders,
                      legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
                                  bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="lightgray", borderwidth=1),
                      xaxis=dict(title="Temperature [°C]", range=[8, 20], domain=[0, 0.88]
                                 # Leave space for second x-axis
                                 ),
                      xaxis2=dict(title="Specific Humidity [g/kg]", overlaying='x', side='top', range=[0, 4],
                                  # Adjust based on typical values
                                  domain=[0, 0.88]), yaxis=dict(title="Height [m]", range=[0, plot_max_height]),
                      updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.15, xanchor="left", yanchor="top",
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
    
    return fig


def plot_save_vertical_profiles(timestamp: str = "2017-10-16T04:00:00", max_height: float = 5000,
                                plot_max_height: float = 2000) -> None:
    """
    Create and save vertical temperature profile plots for all defined points.
    
    This is the main function that creates a small multiples plot showing temperature profiles
    for all locations and saves it as an interactive HTML file. It also displays the plot
    in the browser.
    
    Args:
        timestamp: ISO format timestamp string (e.g. "2017-10-16T04:00:00")
        max_height: Maximum height in meters to load data (default: 5000m)
        plot_max_height: Maximum height in meters to display initially (default: 2000m)
    """
    print(f"\nCreating vertical temperature profile plots for {timestamp}...")
    
    # Create the small multiples plot for all points
    # Data is loaded up to max_height, but displayed up to plot_max_height
    fig = plot_vertical_profiles_small_multiples(ALL_POINTS, timestamp=timestamp, max_height=max_height,
                                                 plot_max_height=plot_max_height)
    
    # Ensure output directory exists
    html_dir = os.path.join(confg.dir_PLOTS, "vertical_plots")
    os.makedirs(html_dir, exist_ok=True)
    
    # Create filename from timestamp (replace special characters)
    ts_str = timestamp.replace(":", "").replace("-", "").replace("T", "_")
    html_path = os.path.join(html_dir, f"vertical_profiles_{ts_str}.html")
    
    # Save as interactive HTML file
    fig.write_html(html_path)
    print(f"Saved vertical profile plot to: {html_path}")
    print(f"Data loaded up to {max_height}m, displayed up to {plot_max_height}m (you can zoom out in the HTML)")
    
    # Also display in browser
    fig.show()


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
    plot_save_all_points_with_slider(start_time="2017-10-15T15:00:00", end_time="2017-10-16T10:00:00",
                                     time_step_hours=1, max_height=4000,  # point_names=["ibk_uni"]
                                     plot_max_height=1500)
