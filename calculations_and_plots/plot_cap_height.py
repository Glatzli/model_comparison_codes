"""
calc_cap_height
Compute, save and plot CAP height (inversion base) over a full region, similar to plot_vhd.

Workflow:
- Build a list of timestamps
- For each model and timestamp: read a 3D slice (time, height, lat, lon)
- Compute dT/dz and CAP height per column (time, lat, lon)
- Concatenate over time and save to NetCDF per model
- Plot: (a) timeline at a given point; (b) small multiples maps
"""
from __future__ import annotations

import datetime
from typing import Iterable, List, Optional
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from colorspace import qualitative_hcl, sequential_hcl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import confg
import read_icon_model_3D
import read_in_arome
import read_ukmo
import read_wrf_helen
from calculations_and_plots.calc_cap_height import cap_height_region

# --- Colors (reuse style similar to plot_vhd) ---
qualitative_colors = qualitative_hcl(palette="Dark 3").colors()
blues_hcl = sequential_hcl(palette="Blues 3")
blues_hcl_cont = blues_hcl.cmap().reversed()  # reversed colormap


# --- Time helpers ---
def build_times(start_day: int, start_hour: int, end_day: int, end_hour: int, step_minutes: int = 30) -> List[
    datetime.datetime]:
    """Create half-hourly (or given step) timestamps within the experiment days.
    Example: build_times(15, 12, 16, 12) -> 2017-10-15 12:00 to 2017-10-16 12:00 (inclusive)
    """
    t0 = datetime.datetime(2017, 10, start_day, start_hour, 0)
    t1 = datetime.datetime(2017, 10, end_day, end_hour, 0)
    times: List[datetime.datetime] = []
    t = t0
    delta = datetime.timedelta(minutes=step_minutes)
    while t <= t1:
        times.append(t)
        t += delta
    return times


# --- Reading per model ---
AROME_VARS = ["p", "th", "temp", "z"]
ICON_VARS = ["p", "th", "temp", "z", "z_unstag"]
UM_VARS = ["p", "th", "temp", "z"]
WRF_VARS = ["p", "th", "temp", "z", "z_unstag"]


def read_model_fixed_time(model: str, t: datetime.datetime, height_as_z_coord: bool,
                          variant: Optional[str] = None) -> xr.Dataset:
    """Read a given model at a fixed time, returning at least (time, height, lat, lon) with temp available.
    model: one of {"AROME","ICON","ICON2TE","UM","WRF"}
    variant: for ICON family, either "ICON" or "ICON2TE"
    """
    if model == "AROME":
        return read_in_arome.read_in_arome_fixed_time(day=t.day, hour=t.hour, min=t.minute, variables=AROME_VARS)
    if model in ("ICON", "ICON2TE"):
        v = variant or model
        return read_icon_model_3D.read_icon_fixed_time(day=t.day, hour=t.hour, min=t.minute, variant=v,
                                                       variables=ICON_VARS)
    if model == "UM":
        return read_ukmo.read_ukmo_fixed_time(day=t.day, hour=t.hour, min=t.minute, variables=UM_VARS)
    if model == "WRF":
        return read_wrf_helen.read_wrf_fixed_time(day=t.day, hour=t.hour, min=t.minute, variables=WRF_VARS)
    raise ValueError(f"Unknown model '{model}'")


# --- CAP computation and accumulation over time ---
def compute_cap_timeseries(model: str, times: Iterable[datetime.datetime], height_as_z_coord: bool) -> xr.DataArray:
    """Compute CAP height for a list of times for a model. Returns DataArray (time, lat, lon).
    If saved file exists, load it; otherwise compute and return (without saving here)."""
    # If a saved file exists for this model, load and return it (skip expensive recomputation)
    out_path = default_cap_path(model)
    if os.path.exists(out_path):
        print(f"Loading existing CAP file for {model}: {out_path}")
        try:
            # Load dataset into memory and close file handle immediately to avoid permission errors
            with xr.open_dataset(out_path) as loaded:
                if "cap_height" in loaded:
                    cap = loaded["cap_height"].load()  # load into memory
                else:
                    # fallback: try as DataArray
                    cap = xr.open_dataarray(out_path).load()
            cap.attrs["loaded_from"] = out_path
            return cap
        except Exception as e:
            print(f"Warning: Could not load {out_path}: {e}. Recomputing...")
            # fall back to recompute if reading fails
            pass

    # Compute CAP for all times over the FULL domain
    print(f"Computing CAP timeseries for {model} over full domain...")  # single status message per model
    cap_list: List[xr.DataArray] = []
    for i, t in enumerate(times):
        # Read full domain (no spatial subsetting here)
        ds = read_model_fixed_time(model, t, height_as_z_coord=True,
                                   variant=model if model in ("ICON", "ICON2TE") else None)
        # compute cap height for this time slice over FULL (lat, lon)
        cap_ds = cap_height_region(ds, consecutive=3)
        # extract cap_height DataArray; ensure it has a time coordinate equal to current timestamp
        if "cap_height" in cap_ds:
            cap_t = cap_ds["cap_height"]
        else:
            raise RuntimeError("cap_height not produced by cap_height_region")

        # If cap_t contains a time dim of length 1, keep it; otherwise set/assign time
        if "time" in cap_t.dims and cap_t.sizes.get("time", 0) == 1:
            # ensure the timestamp matches requested time
            cap_t = cap_t.assign_coords(time=("time", [np.datetime64(t)]))
        else:
            # add time dim
            cap_t = cap_t.expand_dims(time=[np.datetime64(t)])

        cap_list.append(cap_t)

    if len(cap_list) == 0:
        raise RuntimeError("No times processed for compute_cap_timeseries")

    cap = xr.concat(cap_list, dim="time")
    cap.name = "cap_height"
    cap.attrs.update({"description": "First height (bottom-up) where dT_dz < 0 for 3 consecutive levels",
                      "units": ds["height"].attrs.get("units", "") if "height" in ds.coords else ""})
    return cap


# --- Saving helpers ---
def default_cap_path(model: str) -> str:
    """Build default output path per model for NetCDF save."""
    if model == "AROME":
        base = confg.dir_AROME
    elif model == "ICON":
        base = confg.icon_folder_3D
    elif model == "ICON2TE":
        base = confg.icon2TE_folder_3D
    elif model == "UM":
        base = confg.ukmo_folder
    elif model == "WRF":
        base = confg.wrf_folder
    else:
        base = "."
    return f"{base}/{model}_cap_height_full_domain_full_time.nc"


def save_cap(cap: xr.DataArray, path: Optional[str] = None) -> None:
    """Save CAP DataArray to NetCDF. Skip if file already exists."""
    out_path = path or default_cap_path(cap.attrs.get("model", "CAP"))
    
    # EDITED: Skip saving if file already exists (avoid permission errors on Windows)
    if os.path.exists(out_path):
        print(f"File already exists, skipping save: {out_path}")
        return
    
    # Ensure we save only the cap_height variable inside a Dataset
    if isinstance(cap, xr.DataArray):
        ds_out = cap.to_dataset(name=cap.name or "cap_height")
    elif isinstance(cap, xr.Dataset):
        # keep only cap_height variable if present
        if "cap_height" in cap:
            ds_out = cap[["cap_height"]]
        else:
            # fallback: convert first data var
            first = list(cap.data_vars)[0]
            ds_out = cap[[first]]
    else:
        raise TypeError("cap must be an xarray DataArray or Dataset")

    # Make sure directory exists (avoid write errors)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving CAP data to: {out_path}")
    ds_out.to_netcdf(out_path)


# --- Plotting ---
def plot_cap_timeseries_at_point(cap_dict: dict, lat: float, lon: float, point_name: str) -> None:
    """Plot CAP height timeline at a given point for multiple models (interactive plotly version).
    cap_dict: {model_name: cap_height DataArray (time, lat, lon)}
    ICON and ICON2TE use the same color; ICON2TE is dashed.
    """
    fig = go.Figure()
    
    # Model order and color mapping
    model_order = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]
    model_colors = {
        "AROME": qualitative_colors[0],
        "ICON": qualitative_colors[2],
        "ICON2TE": qualitative_colors[2],  # same color as ICON
        "UM": qualitative_colors[4],
        "WRF": qualitative_colors[6]
    }
    
    for model in model_order:
        if model not in cap_dict:
            continue
        cap = cap_dict[model]
        # Filter times from 14:00 onwards
        cap_filtered = cap.where(cap.time >= np.datetime64("2017-10-15T14:00"), drop=True)
        # Select point AFTER computing over full domain
        series = cap_filtered.sel(lat=lat, lon=lon, method="nearest")  # (time)
        
        # Determine line style: dashed for ICON2TE, solid otherwise
        line_dash = "dash" if model == "ICON2TE" else "solid"
        
        fig.add_trace(go.Scatter(
            x=series["time"].values,
            y=series.values,
            mode='lines',
            name=model,
            line=dict(color=model_colors[model], dash=line_dash, width=2)
        ))
    
    fig.update_layout(
        title=f"CAP height timeline at {point_name}",
        xaxis_title="Time",
        yaxis_title="CAP height [m]",
        hovermode='x unified',
        template='plotly_white',
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top')
    )
    
    fig.show()


def plot_cap_small_multiples(cap: xr.DataArray, model: str, vmin: Optional[float] = None,
                             vmax: Optional[float] = None) -> None:
    """Small multiples of CAP height maps (2-hourly from 14:00 to 12:00 next day), similar to plot_vhd_small_multiples."""
    projection = ccrs.Mercator()
    
    # Filter times: 14:00 (day 15) to 12:00 (day 16)
    cap_filtered = cap.where(
        (cap.time >= np.datetime64("2017-10-15T14:00")) &
        (cap.time <= np.datetime64("2017-10-16T12:00")),
        drop=True
    )
    
    # Subsample every 2nd time step (2-hourly if data is hourly, or every 4th if 30-min)
    cap_sub = cap_filtered.isel(time=slice(0, None, 4)) if cap_filtered.sizes.get("time", 0) > 1 else cap_filtered
    
    # EDITED: Restrict to CAP extent (not VHD extent) for plotting, using values from confg
    lat_min, lat_max = confg.lat_min_cap_height, confg.lat_max_cap_height
    lon_min, lon_max = confg.lon_min_cap_height, confg.lon_max_cap_height
    # EDITED: Use .values for coordinates to ensure we work with actual coordinate values
    cap_sub = cap_sub.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    nplots, ncols = cap_sub.sizes.get("time", 1), 3
    nrows = int((nplots + ncols - 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6), layout="compressed", subplot_kw={'projection': projection})
    axes = np.atleast_1d(axes).flatten()

    im = None
    for i, t in enumerate(cap_sub.time.values):
        ax = axes[i]
        scene = cap_sub.sel(time=t)
        # EDITED: Use .values for lat and lon to ensure proper indexing
        im = ax.pcolormesh(scene.lon.values, scene.lat.values, scene.values, cmap=blues_hcl_cont, shading="auto",
                           transform=projection, vmin=vmin, vmax=vmax)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.set_title(f"{np.datetime_as_string(t, unit='h')}")
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Colorbar
    if im is not None:
        cbar = plt.colorbar(im, ax=axes, label=f"{model} CAP height [m]")
        cbar.ax.tick_params(size=0)
    # Do NOT show here - let caller handle plt.show()


# --- End-to-end runner ---
MODELS = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]

# EDITED: Define all points from confg.py
ALL_POINTS = [
    "ibk_villa",
    "ibk_uni",
    "ibk_airport",
    "woergl",
    "kiefersfelden",
    "telfs",
    "wipp_valley",
    "ziller_valley",
    "ziller_ried"
]


def compute_save_plot_cap(start_day: int = 15, start_hour: int = 12, end_day: int = 16, end_hour: int = 12,
                          point: Optional[dict] = None, save_nc: bool = True, save_html: bool = True) -> None:
    """End-to-end function: compute CAP per model and time, save NetCDFs, and plot.
    point: dict with keys {'lat', 'lon', 'name'}; defaults to confg.ibk_villa.
    save_html: if True, save the timeline plot as HTML file instead of showing it.
    """
    if point is None:
        point = confg.ibk_villa

    times = build_times(start_day, start_hour, end_day, end_hour, step_minutes=30)

    cap_by_model = {}
    for model in MODELS:
        cap = compute_cap_timeseries(model, times, height_as_z_coord=True)
        cap.attrs["model"] = model
        cap_by_model[model] = cap
        if save_nc:
            save_cap(cap, path=default_cap_path(model))

    # Plot timeseries at the chosen point
    fig = plot_cap_timeseries_at_point_fig(cap_by_model, lat=point["lat"], lon=point["lon"], point_name=point["name"])
    
    # EDITED: Save as HTML or show
    if save_html:
        # Ensure plots/cap_depth directory exists
        html_dir = os.path.join(confg.dir_PLOTS, "cap_depth")
        os.makedirs(html_dir, exist_ok=True)
        html_path = os.path.join(html_dir, f"cap_depth_{point['name'].replace(' ', '_')}.html")
        fig.write_html(html_path)
        print(f"Saved CAP timeline to: {html_path}")
    else:
        fig.show()

    # Plot small multiples per model (all at once) with uniform colorbar
    for model, cap in cap_by_model.items():
        plot_cap_small_multiples(cap, model=model, vmin=0, vmax=1000)
        # Save figure immediately after creating
        plt.savefig(confg.dir_PLOTS + "cap_depth/" + f"{model}_CAP_small_multiples.png", dpi=300)
    
    # Show all figures at once (only if not saving HTML)
    if not save_html:
        plt.show()


# EDITED: New function that returns figure instead of showing it
def plot_cap_timeseries_at_point_fig(cap_dict: dict, lat: float, lon: float, point_name: str) -> go.Figure:
    """Plot CAP height timeline at a given point for multiple models (interactive plotly version).
    Returns the figure object for saving or display.
    cap_dict: {model_name: cap_height DataArray (time, lat, lon)}
    ICON and ICON2TE use the same color; ICON2TE is dashed.
    """
    fig = go.Figure()
    
    # Model order and color mapping
    model_order = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]
    model_colors = {
        "AROME": qualitative_colors[0],
        "ICON": qualitative_colors[2],
        "ICON2TE": qualitative_colors[2],  # same color as ICON
        "UM": qualitative_colors[4],
        "WRF": qualitative_colors[6]
    }
    
    for model in model_order:
        if model not in cap_dict:
            continue
        cap = cap_dict[model]
        # Filter times from 14:00 onwards
        cap_filtered = cap.where(cap.time >= np.datetime64("2017-10-15T14:00"), drop=True)
        # Select point AFTER computing over full domain
        series = cap_filtered.sel(lat=lat, lon=lon, method="nearest")  # (time)
        
        # Determine line style: dashed for ICON2TE, solid otherwise
        line_dash = "dash" if model == "ICON2TE" else "solid"
        
        fig.add_trace(go.Scatter(
            x=series["time"].values,
            y=series.values,
            mode='lines',
            name=model,
            line=dict(color=model_colors[model], dash=line_dash, width=2)
        ))
    
    fig.update_layout(
        title=f"CAP height timeline at {point_name}",
        xaxis_title="Time",
        yaxis_title="CAP height [m]",
        hovermode='x unified',
        template='plotly_white',
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top')
    )
    
    return fig


# EDITED: New function to create small multiples of CAP timeseries for multiple points
def plot_cap_timeseries_small_multiples(cap_dict: dict, point_names: List[str]) -> go.Figure:
    """Create small multiples plot of CAP height timelines for multiple points.
    cap_dict: {model_name: cap_height DataArray (time, lat, lon)}
    point_names: list of point names from confg.py (e.g. ["ibk_villa", "ibk_uni", ...])
    Returns the figure object.
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
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Model order and color mapping
    model_order = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]
    model_colors = {
        "AROME": qualitative_colors[0],
        "ICON": qualitative_colors[2],
        "ICON2TE": qualitative_colors[2],
        "UM": qualitative_colors[4],
        "WRF": qualitative_colors[6]
    }
    
    # Plot for each point
    for idx, point_name in enumerate(point_names):
        point = getattr(confg, point_name, None)
        if point is None:
            continue
        
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        for model in model_order:
            if model not in cap_dict:
                continue
            cap = cap_dict[model]
            # Filter times from 14:00 onwards
            cap_filtered = cap.where(cap.time >= np.datetime64("2017-10-15T14:00"), drop=True)
            # Select point
            series = cap_filtered.sel(lat=point["lat"], lon=point["lon"], method="nearest")
            
            # Determine line style: dashed for ICON2TE, solid otherwise
            line_dash = "dash" if model == "ICON2TE" else "solid"
            
            # Only show legend for first subplot
            show_legend = (idx == 0)
            
            fig.add_trace(
                go.Scatter(
                    x=series["time"].values,
                    y=series.values,
                    mode='lines',
                    name=model,
                    line=dict(color=model_colors[model], dash=line_dash, width=1.5),
                    legendgroup=model,
                    showlegend=show_legend
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title_text="CAP Height Timelines at Multiple Points",
        height=350 * n_rows,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update axes labels
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            fig.update_yaxes(title_text="CAP height [m]", row=i, col=j)
            if i == n_rows:
                fig.update_xaxes(title_text="Time", row=i, col=j)
    
    return fig


# EDITED: New function to process all points
def compute_save_plot_cap_all_points(start_day: int = 15, start_hour: int = 12, end_day: int = 16, end_hour: int = 12,
                                     save_nc: bool = True) -> None:
    """Compute CAP for all models once, then plot timelines for all points defined in confg.py.
    Saves ONLY the small multiples timeline plot (not individual point files).
    """
    times = build_times(start_day, start_hour, end_day, end_hour, step_minutes=30)

    # Compute CAP for all models once (expensive operation)
    cap_by_model = {}
    for model in MODELS:
        cap = compute_cap_timeseries(model, times, height_as_z_coord=True)
        cap.attrs["model"] = model
        cap_by_model[model] = cap
        if save_nc:
            save_cap(cap, path=default_cap_path(model))

    # Ensure plots/cap_depth directory exists
    html_dir = os.path.join(confg.dir_PLOTS, "cap_depth")
    os.makedirs(html_dir, exist_ok=True)

    # EDITED: Only create and save small multiples plot for all points (no individual files)
    print("\nCreating small multiples plot for all points...")
    fig_small_multiples = plot_cap_timeseries_small_multiples(cap_by_model, ALL_POINTS)
    sm_path = os.path.join(html_dir, "cap_depth_all_points_small_multiples.html")
    fig_small_multiples.write_html(sm_path)
    print(f"Saved small multiples plot to: {sm_path}")


# EDITED: New function to plot vertical temperature profiles as small multiples
def plot_vertical_profiles_small_multiples(point_names: List[str], timestamp: str = "2017-10-16T04:00:00",
                                           max_height: float = 3000):
    """Create small multiples plot of vertical temperature profiles for all points at a given timestamp.
    Each subplot shows all models for one point, with CAP height markers.
    point_names: list of point names from confg.py
    timestamp: ISO format timestamp string (e.g. "2017-10-16T04:00:00")
    max_height: maximum height in meters to plot (default 3000m)
    """
    # Convert timestamp string to numpy datetime64
    ts = np.datetime64(timestamp)
    
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
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.08
    )
    
    # Model order and color mapping
    model_order = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]
    model_colors = {
        "AROME": qualitative_colors[0],
        "ICON": qualitative_colors[2],
        "ICON2TE": qualitative_colors[2],
        "UM": qualitative_colors[4],
        "WRF": qualitative_colors[6]
    }
    
    # Variables to read for each model
    variables = ["p", "th", "temp", "z", "z_unstag"]
    
    # Helper function to get timeseries file path
    def get_timeseries_path(model: str, point_name: str) -> str:
        """Build path to timeseries file with _height_as_z.nc suffix."""
        if model == "AROME":
            base = confg.dir_AROME
        elif model == "ICON":
            base = confg.icon_folder_3D
        elif model == "ICON2TE":
            base = confg.icon2TE_folder_3D
        elif model == "UM":
            base = confg.ukmo_folder
        elif model == "WRF":
            base = confg.wrf_folder
        else:
            return ""
        
        # Construct path: MODEL_FOLDER/timeseries/modelname_pointname_timeseries_height_as_z.nc
        model_name_lower = model.lower()
        if model == "ICON2TE":
            model_name_lower = "icon2te"
        
        return os.path.join(base, "timeseries", f"{model_name_lower}_{point_name}_timeseries_height_as_z.nc")
    
    # Helper function to save timeseries to file
    def save_timeseries(ds: xr.Dataset, model: str, point_name: str) -> None:
        """Save timeseries dataset to file if it doesn't exist yet."""
        timeseries_path = get_timeseries_path(model, point_name)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(timeseries_path), exist_ok=True)
        
        # Only save if file doesn't exist yet
        if not os.path.exists(timeseries_path):
            print(f"Saving {model} timeseries for {point_name} to: {timeseries_path}")
            try:
                ds.to_netcdf(timeseries_path)
            except Exception as e:
                print(f"Warning: Could not save timeseries file {timeseries_path}: {e}")
        else:
            print(f"Timeseries file already exists: {timeseries_path}")
    
    # Read data and plot for each point
    for idx, point_name in enumerate(point_names):
        point = getattr(confg, point_name, None)
        if point is None:
            continue
        
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        for model in model_order:
            try:
                # EDITED: First try to load from saved timeseries file
                timeseries_path = get_timeseries_path(model, point_name)
                ds = None
                
                if os.path.exists(timeseries_path):
                    print(f"Loading {model} timeseries from: {timeseries_path}")
                    try:
                        ds = xr.open_dataset(timeseries_path)
                    except Exception as e:
                        print(f"Warning: Could not load timeseries file {timeseries_path}: {e}")
                        ds = None
                
                # If timeseries file doesn't exist or failed to load, read fresh data
                if ds is None:
                    print(f"Reading fresh {model} data for {point_name}...")
                    if model == "AROME":
                        ds = read_in_arome.read_in_arome_fixed_point(
                            lat=point["lat"], lon=point["lon"],
                            variables=variables, height_as_z_coord=True
                        )
                    elif model == "ICON":
                        ds = read_icon_model_3D.read_icon_fixed_point(
                            lat=point["lat"], lon=point["lon"],
                            variant="ICON", variables=variables, height_as_z_coord=True
                        )
                    elif model == "ICON2TE":
                        ds = read_icon_model_3D.read_icon_fixed_point(
                            lat=point["lat"], lon=point["lon"],
                            variant="ICON2TE", variables=variables, height_as_z_coord=True
                        )
                    elif model == "UM":
                        ds = read_ukmo.read_ukmo_fixed_point(
                            lat=point["lat"], lon=point["lon"],
                            variables=variables, height_as_z_coord=True
                        )
                    elif model == "WRF":
                        ds = read_wrf_helen.read_wrf_fixed_point(
                            lat=point["lat"], lon=point["lon"],
                            variables=variables, height_as_z_coord=True
                        )
                    else:
                        continue
                    
                    # Save the timeseries for future use
                    save_timeseries(ds, model, point_name)
                
                # Filter to max_height
                # Check if height is a coordinate (from timeseries files) or a variable
                if "height" in ds.coords:
                    height_var = ds.coords["height"]
                elif "z_unstag" in ds and model in ["ICON", "WRF"]:
                    height_var = ds["z_unstag"]
                elif "z" in ds:
                    height_var = ds["z"]
                else:
                    print(f"Warning: No height coordinate found for {model} at {point_name}")
                    continue
                
                # Select only heights up to max_height
                ds_filtered = ds.where(height_var <= max_height, drop=True)
                
                # Get temperature and height values
                if "temp" in ds_filtered:
                    temp = ds_filtered["temp"].sel(time=ts, method="nearest").values
                    
                    # Get height values (handle both coordinate and variable cases)
                    if "height" in ds.coords:
                        # For timeseries files, height is already a coordinate
                        height = ds_filtered.coords["height"].values
                    else:
                        height = height_var.sel(time=ts, method="nearest").values
                    
                    # Filter out NaNs
                    valid = ~np.isnan(temp) & ~np.isnan(height)
                    temp = temp[valid]
                    height = height[valid]
                    
                    # Determine line style
                    line_dash = "dash" if model == "ICON2TE" else "solid"
                    
                    # Only show legend for first subplot
                    show_legend = (idx == 0)
                    
                    # Add temperature profile trace
                    fig.add_trace(
                        go.Scatter(
                            x=temp,
                            y=height,
                            mode='lines',
                            name=model,
                            line=dict(color=model_colors[model], dash=line_dash, width=1.5),
                            legendgroup=model,
                            showlegend=show_legend
                        ),
                        row=row, col=col
                    )
                    
                    # Add CAP height marker if available
                    # Try to load cap_height from saved files
                    try:
                        cap_path = default_cap_path(model)
                        if os.path.exists(cap_path):
                            with xr.open_dataset(cap_path) as cap_ds:
                                cap_height_da = cap_ds["cap_height"].load()
                            
                            # Select point and time
                            cap_height = cap_height_da.sel(
                                lat=point["lat"], lon=point["lon"],
                                time=ts, method="nearest"
                            ).item()
                            
                            if not np.isnan(cap_height) and cap_height <= max_height:
                                # Get temperature at cap height
                                # For timeseries files with height as coord, use height directly
                                if "height" in ds_filtered.coords:
                                    temp_at_cap = ds_filtered["temp"].sel(
                                        time=ts, height=cap_height, method="nearest"
                                    ).item()
                                else:
                                    temp_at_cap = ds_filtered["temp"].sel(
                                        time=ts, method="nearest"
                                    ).sel(height=cap_height, method="nearest").item()
                                
                                # Add marker
                                fig.add_trace(
                                    go.Scatter(
                                        x=[temp_at_cap],
                                        y=[cap_height],
                                        mode='markers',
                                        marker=dict(symbol='x', size=10, color=model_colors[model],
                                                   line=dict(width=2, color=model_colors[model])),
                                        name=f"{model} CAP",
                                        legendgroup=model,
                                        showlegend=False,
                                        hovertemplate=f"{model} CAP: {cap_height:.0f}m<extra></extra>"
                                    ),
                                    row=row, col=col
                                )
                    except Exception as e:
                        # Skip marker if CAP data not available
                        pass
                
                # Close dataset if it was opened from file
                if ds is not None and timeseries_path and os.path.exists(timeseries_path):
                    ds.close()
                    
            except Exception as e:
                print(f"Warning: Could not load {model} data for {point_name}: {e}")
                continue
    
    # Update layout
    fig.update_layout(
        title_text=f"Vertical Temperature Profiles at {timestamp}",
        height=350 * n_rows,
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update axes
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            # Set y-axis range [0, max_height]
            fig.update_yaxes(range=[0, max_height], row=i, col=j)
            
            # Only show axis titles and ticks for top-left plot
            if i == 1 and j == 1:
                fig.update_xaxes(title_text="Temperature [°C]", row=i, col=j)
                fig.update_yaxes(title_text="Height [m]", row=i, col=j)
            else:
                fig.update_xaxes(title_text="", showticklabels=False, row=i, col=j)
                fig.update_yaxes(title_text="", showticklabels=False, row=i, col=j)
    
    return fig


# EDITED: New function to save vertical profile plots
def plot_save_vertical_profiles(timestamp: str = "2017-10-16T04:00:00", max_height: float = 3000) -> None:
    """Create and save vertical temperature profile small multiples plot for all points.
    timestamp: ISO format timestamp string (e.g. "2017-10-16T04:00:00")
    max_height: maximum height in meters to plot (default 3000m)
    """
    print(f"\nCreating vertical temperature profile plots for {timestamp}...")
    
    # Create small multiples plot
    fig = plot_vertical_profiles_small_multiples(ALL_POINTS, timestamp=timestamp, max_height=max_height)
    
    # Save as HTML
    html_dir = os.path.join(confg.dir_PLOTS, "cap_depth")
    os.makedirs(html_dir, exist_ok=True)
    
    # Create filename from timestamp
    ts_str = timestamp.replace(":", "").replace("-", "").replace("T", "_")
    html_path = os.path.join(html_dir, f"vertical_profiles_{ts_str}.html")
    
    fig.write_html(html_path)
    print(f"Saved vertical profile plot to: {html_path}")
    
    # Also show in browser
    fig.show()


if __name__ == "__main__":
    # EDITED: Run for all points and save as HTML files
    compute_save_plot_cap_all_points()
    
    # EDITED: Also create vertical profile plot for 04 UTC on Oct 16
    plot_save_vertical_profiles(timestamp="2017-10-16T04:00:00", max_height=2000)
