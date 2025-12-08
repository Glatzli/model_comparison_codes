"""
Plot heat budget timeseries for specific points from AROME and WRF models.

This script creates interactive Plotly plots showing the time evolution of all heat budget
variables (hfs, lfs, lwd, lwu, swd, swu) for selected points. Each point gets its own plot
saved as an HTML file.

The script uses PCGP (Physically Consistent Grid Point) selection to ensure accurate
point representation across different model grids.
"""

import os
import sys
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
import xarray as xr

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import confg
import read_in_arome
import read_wrf_helen
from calculations_and_plots.calc_vhd import read_dems_calc_pcgp

# Heat budget variables to plot
HEAT_BUDGET_VARS = ["hfs", "lfs", "lwd", "lwu", "swd", "swu"]

# Variable metadata for labels with meaningful colors
# Sensible heat = red/orange (warm), Latent = blue (water),
# Shortwave = yellow/orange (sun), Longwave = brown/gray (earth/atmosphere)
VAR_METADATA = {"hfs": {"label": "Sensible Heat Flux", "color": "#E16A86"},  # Red-pink (warm)
                "lfs": {"label": "Latent Heat Flux", "color": "#50B2AD"},  # Blue-green (water/evaporation)
                "swd": {"label": "Downward Shortwave", "color": "#FFBF00"},  # Yellow-orange (sun)
                "swu": {"label": "Upward Shortwave", "color": "#FFA040"},  # Light orange (reflected sun)
                "lwd": {"label": "Downward Longwave", "color": "#A0A0A0"},  # Gray (atmospheric radiation)
                "lwu": {"label": "Upward Longwave", "color": "#8B7355"},  # Brown (earth radiation)
                }


def read_wrf_2d_timeseries(lat, lon, variables):
    """
    Read WRF 2D variable timeseries for a specific point.

    Parameters
    ----------
    lat : float
        Latitude of the point
    lon : float
        Longitude of the point
    variables : list
        List of 2D variables to read

    Returns
    -------
    xr.Dataset
        Dataset with 2D variables at the specified point
    """
    print(f"  Reading WRF 2D variables for lat={lat:.3f}, lon={lon:.3f}")

    # Create time range - only from 14:00 on 15th to 11:00 on 16th
    start_dt = datetime(2017, 10, 15, 14, 0)
    end_dt = datetime(2017, 10, 16, 11, 0)
    times = pd.date_range(start=start_dt, end=end_dt, freq="1h")  # Use hourly instead of 30min for speed

    ds_list = []
    for i, t in enumerate(times):
        try:
            # Read full domain and then select the point
            ds_time = read_wrf_helen.read_wrf_fixed_time(day=t.day, hour=t.hour, min=t.minute, variables=variables)

            # Select the point from the full domain
            ds_time = ds_time.sel(lat=lat, lon=lon, method="nearest")

            # Extract only surface variables (remove height dimension if present)
            for var in variables:
                if var in ds_time and "height" in ds_time[var].dims:
                    ds_time[var] = ds_time[var].isel(height=0)

            ds_list.append(ds_time)

            if (i + 1) % 5 == 0:
                print(f"    Progress: {i + 1}/{len(times)} timesteps loaded")

        except Exception as e:
            print(f"  ⚠ Warning: Could not read WRF data for {t}: {e}")
            continue

    if len(ds_list) == 0:
        print("  ✗ No WRF data could be read")
        return None

    try:
        ds = xr.concat(ds_list, dim="time")
        print(f"  ✓ Successfully loaded {len(ds_list)} WRF timesteps")
        return ds
    except Exception as e:
        print(f"  ✗ Error concatenating WRF data: {e}")
        return None


def plot_heat_budget_timeseries_for_point(point_name, point_info, save_dir):
    """
    Create interactive Plotly plot showing all heat budget variables for a specific point.

    Parameters
    ----------
    point_name : str
        Name of the point (from confg.ALL_POINTS)
    point_info : dict
        Dictionary with 'lat', 'lon', 'height' keys
    save_dir : str
        Directory to save the HTML plot
    """
    print(f"\n{'=' * 70}")
    print(f"Processing point: {point_name}")
    print(f"  Coordinates: lat={point_info['lat']:.4f}, lon={point_info['lon']:.4f}")
    print(f"  Height: {point_info['height']} m")
    print(f"{'=' * 70}")

    # Get PCGP coordinates for accurate point representation
    pcgp_arome, pcgp_icon, pcgp_um, pcgp_wrf = read_dems_calc_pcgp(lat=point_info["lat"], lon=point_info["lon"])

    # Read data from both models
    arome_ds = read_in_arome.read_2D_variables_AROME(variableList=HEAT_BUDGET_VARS, lon=pcgp_arome.x.values,
                                                     lat=pcgp_arome.y.values, slice_lat_lon=False)

    wrf_ds = read_wrf_2d_timeseries(lat=pcgp_wrf.y.values, lon=pcgp_wrf.x.values, variables=HEAT_BUDGET_VARS)

    # Create figure with subplots for each variable
    fig = go.Figure()

    # Track which variables were successfully plotted
    plotted_vars = []

    # Plot each heat budget variable
    for var in HEAT_BUDGET_VARS:
        var_meta = VAR_METADATA.get(var, {"label": var, "color": "#000000"})

        # Plot AROME data
        if arome_ds is not None and var in arome_ds:
            try:
                data = arome_ds[var]
                # Remove any extra dimensions
                if len(data.dims) > 1:
                    # Keep only time dimension
                    for dim in data.dims:
                        if dim != "time":
                            data = data.isel({dim: 0})

                fig.add_trace(
                    go.Scatter(x=data.time.values, y=data.values, mode='lines', name=f'AROME - {var_meta["label"]}',
                               line=dict(color=var_meta["color"], width=2.5, dash='solid'), legendgroup=var,
                               showlegend=True))
                plotted_vars.append(var)
            except Exception as e:
                print(f"  ⚠ Warning: Could not plot AROME {var}: {e}")

        # Plot WRF data
        if wrf_ds is not None and var in wrf_ds:
            try:
                data = wrf_ds[var]
                # Remove any extra dimensions
                if len(data.dims) > 1:
                    # Keep only time dimension
                    for dim in data.dims:
                        if dim != "time":
                            data = data.isel({dim: 0})

                fig.add_trace(
                    go.Scatter(x=data.time.values, y=data.values, mode='lines', name=f'WRF - {var_meta["label"]}',
                               line=dict(color=var_meta["color"], width=2.5, dash='dash'), legendgroup=var,
                               showlegend=True))
                if var not in plotted_vars:
                    plotted_vars.append(var)
            except Exception as e:
                print(f"  ⚠ Warning: Could not plot WRF {var}: {e}")

    if len(plotted_vars) == 0:
        print(f"  ✗ No data could be plotted for {point_name}")
        return

    # Update layout
    title_text = f'Heat Budget Timeseries - {point_info.get("name", point_name)} ({point_info["height"]} m)'

    fig.update_layout(title=dict(text=title_text, x=0.5, font=dict(size=18, family="Arial, sans-serif")),
                      xaxis=dict(title='Time', showgrid=True, gridcolor='lightgray', gridwidth=1,
                                 range=['2017-10-15 14:00:00', '2017-10-16 11:00:00']),
                      yaxis=dict(title='Heat Flux [W/m²]', showgrid=True, gridcolor='lightgray', gridwidth=1),
                      hovermode='x unified', template='plotly_white', width=1400, height=700,
                      margin=dict(l=80, r=50, t=100, b=80),
                      legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(size=10)))

    # Save the plot
    point_name_safe = point_name.replace(" ", "_")
    output_file = os.path.join(save_dir, f"heat_budget_timeseries_{point_name_safe}.html")

    pyo.plot(fig, filename=output_file, auto_open=False)
    print(f"  ✓ Plot saved to: {output_file}")

    # Also show the plot
    fig.show()


def plot_all_heat_budget_timeseries(point_names=None):
    """
    Create heat budget timeseries plots for all specified points.

    Parameters
    ----------
    point_names : list, optional
        List of point names to process. If None, uses all points from confg.ALL_POINTS
    """
    # Use all points if none specified
    if point_names is None:
        point_names = list(confg.ALL_POINTS.keys())

    # Create output directory
    save_dir = os.path.join(confg.dir_PLOTS, "heat_flux")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"# Creating heat budget timeseries plots")
    print(f"# Points to process: {len(point_names)}")
    print(f"# Output directory: {save_dir}")
    print(f"{'#' * 70}")

    # Process each point
    for point_name in point_names:
        if point_name not in confg.ALL_POINTS:
            print(f"\n⚠ Warning: Point '{point_name}' not found in confg.ALL_POINTS, skipping...")
            continue

        point_info = confg.ALL_POINTS[point_name]

        try:
            plot_heat_budget_timeseries_for_point(point_name=point_name, point_info=point_info, save_dir=save_dir)
        except Exception as e:
            print(f"\n✗ Error processing {point_name}: {e}")
            continue

    print(f"\n{'#' * 70}")
    print(f"# ✓ All heat budget timeseries plots completed!")
    print(f"# Output location: {save_dir}")
    print(f"{'#' * 70}\n")


if __name__ == '__main__':
    # Example: Plot for a subset of valley points
    # You can change this to confg.VALLEY_POINTS or specific point names
    points_to_plot = ["ibk_uni", "ibk_airport", "kufstein", "jenbach"]

    # Or use all points:
    # points_to_plot = None

    plot_all_heat_budget_timeseries(point_names=points_to_plot)