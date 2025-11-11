"""
calc_cap_height
compute, save and plot CAP height (inversion base) for all points and plot it as small multiples,
similar to plot_vhd.

Workflow:
- Build a list of timestamps
- For each model and timestamp: read a 3D slice (time, height, lat, lon)
- Compute dT and CAP height per column (time, lat, lon)
- Concatenate over time and save to NetCDF per model
- Plot: (a) timeline at a given point; (b) small multiples maps

Note: Vertical temperature profile plotting has been moved to plot_vertical_profiles.py
"""
from __future__ import annotations

import datetime
import os
from typing import Iterable, List, Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from colorspace import qualitative_hcl, sequential_hcl
from plotly.subplots import make_subplots

import confg
import read_icon_model_3D
import read_in_arome
import read_ukmo
import read_wrf_helen
from calculations_and_plots.calc_cap_height import cap_height_region, calc_dT

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


def read_model_timeseries(model: str, times: Iterable[datetime.datetime],
                          variant: Optional[str] = None) -> xr.Dataset:
    """Read a model for multiple timesteps and concatenate along time dimension.
    
    This is more efficient than looping because:
    1. The reading functions may already load multiple times internally
    2. We can process all times at once with vectorized operations
    
    Args:
        model: one of {"AROME","ICON","ICON2TE","UM","WRF"}
        times: Iterable of datetime objects
        variant: for ICON family, either "ICON" or "ICON2TE"
    
    Returns:
        Dataset with time dimension containing all requested timesteps
    """
    times_list = list(times)
    
    # Read all timesteps and concatenate
    datasets = []
    for t in times_list:
        ds = read_model_fixed_time(model, t, variant=variant)
        # Ensure time dimension exists
        if "time" not in ds.dims or ds.sizes.get("time", 0) != 1:
            ds = ds.expand_dims(time=[np.datetime64(t)])
        else:
            ds = ds.assign_coords(time=("time", [np.datetime64(t)]))
        datasets.append(ds)
    
    # Concatenate along time dimension
    ds_all = xr.concat(datasets, dim="time")
    return ds_all


def read_model_fixed_time(model: str, t: datetime.datetime,
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
def compute_cap_timeseries(model: str, times: Iterable[datetime.datetime]) -> xr.DataArray:
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
    
    # Compute CAP for all times over the FULL domain (vectorized approach)
    print(f"Computing CAP timeseries for {model} over full domain...")
    
    # Read all timesteps at once
    ds = read_model_timeseries(model, times, variant=model if model in ("ICON", "ICON2TE") else None)
    
    # Compute dT for all times at once (vectorized)
    ds = calc_dT(ds)
    
    # Compute cap height for all times at once (vectorized)
    cap_ds = cap_height_region(ds, consecutive=3)
    
    # Extract cap_height DataArray
    if "cap_height" not in cap_ds:
        raise RuntimeError("cap_height not produced by cap_height_region")
    
    cap = cap_ds["cap_height"]
    cap.name = "cap_height"
    cap.attrs.update({
        "description": "First height (bottom-up) where dT < 0 for 3 consecutive levels",
        "units": ds["height"].attrs.get("units", "") if "height" in ds.coords else ""
    })
    
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
    model_colors = {"AROME": qualitative_colors[0], "ICON": qualitative_colors[2], "ICON2TE": qualitative_colors[2],
        # same color as ICON
        "UM": qualitative_colors[4], "WRF": qualitative_colors[6]}
    
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
        
        fig.add_trace(go.Scatter(x=series["time"].values, y=series.values, mode='lines', name=model,
            line=dict(color=model_colors[model], dash=line_dash, width=2)))
    
    fig.update_layout(title=f"CAP height timeline at {point_name}", xaxis_title="Time", yaxis_title="CAP height [m]",
        hovermode='x unified', template='plotly_white', legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'))
    
    fig.show()


def plot_cap_small_multiples(cap: xr.DataArray, model: str, vmin: Optional[float] = None,
                             vmax: Optional[float] = None) -> None:
    """Small multiples of CAP height maps (2-hourly from 14:00 to 12:00 next day), similar to
    plot_vhd_small_multiples."""
    projection = ccrs.Mercator()
    
    # Filter times: 14:00 (day 15) to 12:00 (day 16)
    cap_filtered = cap.where(
        (cap.time >= np.datetime64("2017-10-15T14:00")) & (cap.time <= np.datetime64("2017-10-16T12:00")), drop=True)
    
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
        cbar.ax.tick_params(size=0)  # Do NOT show here - let caller handle plt.show()


# --- End-to-end runner ---
MODELS = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]

# EDITED: Define all points from confg.py
ALL_POINTS = ["ibk_villa", "ibk_uni", "ibk_airport", "woergl", "kiefersfelden", "telfs", "wipp_valley", "ziller_valley",
    "ziller_ried"]


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
    model_colors = {"AROME": qualitative_colors[0], "ICON": qualitative_colors[2], "ICON2TE": qualitative_colors[2],
        # same color as ICON
        "UM": qualitative_colors[4], "WRF": qualitative_colors[6]}
    
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
        
        fig.add_trace(go.Scatter(x=series["time"].values, y=series.values, mode='lines', name=model,
            line=dict(color=model_colors[model], dash=line_dash, width=2)))
    
    fig.update_layout(title=f"CAP height timeline at {point_name}", xaxis_title="Time", yaxis_title="CAP height [m]",
        hovermode='x unified', template='plotly_white', legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'))
    
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
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, vertical_spacing=0.12,
        horizontal_spacing=0.1)
    
    # Model order and color mapping
    model_order = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]
    model_colors = {"AROME": qualitative_colors[0], "ICON": qualitative_colors[2], "ICON2TE": qualitative_colors[2],
        "UM": qualitative_colors[4], "WRF": qualitative_colors[6]}
    
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
            
            fig.add_trace(go.Scatter(x=series["time"].values, y=series.values, mode='lines', name=model,
                line=dict(color=model_colors[model], dash=line_dash, width=1.5), legendgroup=model,
                showlegend=show_legend), row=row, col=col)
    
    fig.update_layout(title_text="CAP Height Timelines at Multiple Points", height=350 * n_rows, hovermode='x unified',
        template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5))
    
    # Update axes labels and limits
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            # Set uniform y-axis range
            fig.update_yaxes(range=[500, 1420], row=i, col=j)
            # Set uniform x-axis range: 14:00 on 15th to 10:00 on 16th
            fig.update_xaxes(
                range=[np.datetime64("2017-10-15T14:00"), np.datetime64("2017-10-16T10:00")],
                row=i, col=j
            )
            # Only add axis labels to the first (upper left) subplot
            if i == 1 and j == 1:
                fig.update_xaxes(title_text="Time", row=i, col=j)
                fig.update_yaxes(title_text="CAP height [m]", row=i, col=j)
    
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
        cap = compute_cap_timeseries(model, times)
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


if __name__ == "__main__":
    # EDITED: Run for all points and save as HTML files
    compute_save_plot_cap_all_points()
