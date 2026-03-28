"""
Plot heat budget timeseries for specific points from AROME and WRF models. The data is freshly read each execution time,
no timeseries is saved for the heat budget variables (otherwise than for other model data)!

This script creates interactive Plotly plots showing the time evolution of all heat budget
variables (hfs, lfs, lwd, lwu, swd, swu) for selected points. Each point gets its own plot
saved as an HTML file.

The script uses PCGP (Physically Consistent Grid Point) selection to ensure accurate
point representation across different model grids.
"""
import fix_win_DLL_loading_issue

fix_win_DLL_loading_issue
import os
import sys
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import confg
import read_in_arome
import read_wrf_helen
from calculations_and_plots.calc_vhd import read_dems_calc_pcgp
from calculations_and_plots.energy_balance_EC_hannes import prepare_EC_datasets

# Heat budget variables to plot
HEAT_BUDGET_VARS = ["hfs", "lfs", "swd", "swu", "lwd", "lwu"]

# Mapping of point names to EC station keys
# key = 0: ibk_airport (approx.)
# key = 3: ibk_uni
EC_STATION_MAPPING = {"ibk_airport": 0, "ibk_uni": 3, "patsch_EC_south": 1}  #

# Variable metadata for labels with meaningful colors
# Sensible heat = red/orange (warm), Latent = blue (water),
# Shortwave = yellow/orange (sun), Longwave = brown/gray (earth/atmosphere)
VAR_METADATA = {"hfs": {"label": "Sensible heat flux", "color": "#E16A86"},  # Red-pink (warm)
                "lfs": {"label": "Latent heat flux", "color": "#50B2AD"},  # Blue-green (water/evaporation)
                "swd": {"label": "Downward shortwave", "color": "#FFBF00"},  # Yellow-orange (sun)
                "swu": {"label": "Upward shortwave", "color": "#FFA040"},  # Light orange (reflected sun)
                "lwd": {"label": "Downward longwave", "color": "#A0A0A0"},  # Gray (atmospheric radiation)
                "lwu": {"label": "Upward longwave", "color": "#8B7355"},  # Brown (earth radiation)
                }


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

    wrf_ds = read_wrf_helen.read_wrf_fixed_point(lat=pcgp_wrf.y.values, lon=pcgp_wrf.x.values,
                                                 variables=HEAT_BUDGET_VARS, height_as_z_coord=False)

    # Load observation data if available for this point (only for ibk_airport & ibk_uni)
    obs_ds = None
    if point_name in EC_STATION_MAPPING:
        try:
            station_key = EC_STATION_MAPPING[point_name]
            obs_ds = prepare_EC_datasets(station_key)
            print(f"  Loaded observation data for station key {station_key}")
        except Exception as e:
            print(f"  Warning: Could not load observation data: {e}")
            obs_ds = None

    # Create figure with subplots for each variable
    fig = go.Figure()

    # Track which variables were successfully plotted and which models are available
    plotted_vars = []
    has_arome = False
    has_wrf = False
    has_obs = obs_ds is not None

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
                    go.Scatter(x=data.time.values, y=data.values, mode='lines', name=f'AROME',  # - {var_meta["label"]}
                               line=dict(color=var_meta["color"], width=2, dash='dash'), legendgroup=var,
                               legendgrouptitle_text=var_meta["label"], showlegend=True))
                plotted_vars.append(var)
                has_arome = True
            except Exception as e:
                print(f"  Warning: Could not plot AROME {var}: {e}")

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
                    go.Scatter(x=data.time.values, y=data.values, mode='lines', name=f'WRF',  # - {var_meta["label"]}
                               line=dict(color=var_meta["color"], width=2, dash='dot'), legendgroup=var,
                               legendgrouptitle_text=var_meta["label"], showlegend=True))
                if var not in plotted_vars:
                    plotted_vars.append(var)
                has_wrf = True
            except Exception as e:
                print(f"  Warning: Could not plot WRF {var}: {e}")

        # Plot observation data if available
        if obs_ds is not None:
            try:
                if var in obs_ds:
                    data = obs_ds[var]

                    # For upward fluxes and turbulent fluxes, apply sign convention
                    # In observations: swout, lwout, h, le are stored as positive
                    # We don't need to negate them as they're already in the right convention

                    fig.add_trace(go.Scatter(x=data.time.values, y=data.values, mode='lines', name=f'OBS',
                                             # - {var_meta["label"]}
                                             line=dict(color=var_meta["color"], width=2, dash="solid"), legendgroup=var,
                                             legendgrouptitle_text=var_meta["label"], showlegend=True))
                    if var not in plotted_vars:
                        plotted_vars.append(var)
            except Exception as e:
                print(f"  Warning: Could not plot observation {var}: {e}")

    if len(plotted_vars) == 0:
        print(f" No data could be plotted for {point_name}")
        return

    # Update layout
    title_text = f'Heat Budget Timeseries - {point_info.get("name", point_name)} ({point_info["height"]} m)'

    fig.update_layout(title=dict(text=title_text, x=0.5, font=dict(size=18, family="Arial, sans-serif")),
                      xaxis=dict(title='Time', showgrid=True, gridcolor='lightgray', gridwidth=1,
                                 range=['2017-10-15 13:00:00', '2017-10-16 12:00:00']),
                      yaxis=dict(title='Heat flux [W/m²]', showgrid=True, gridcolor='lightgray', gridwidth=1),
                      hovermode='x unified', template='plotly_white', width=1400, height=700,
                      margin=dict(l=80, r=50, t=100, b=80),
                      legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(size=10)))

    # Save the plot
    point_name_safe = point_name.replace(" ", "_")
    output_file = os.path.join(save_dir, f"heat_budget_timeseries_{point_name_safe}.html")

    pyo.plot(fig, filename=output_file, auto_open=False)
    print(f"  Plot saved to: {output_file}")
    return fig


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
            print(f"\n Warning: Point '{point_name}' not found in confg.ALL_POINTS, skipping...")
            continue

        point_info = confg.ALL_POINTS[point_name]

        try:
            fig = plot_heat_budget_timeseries_for_point(point_name=point_name, point_info=point_info, save_dir=save_dir)

        except Exception as e:
            print(f"\n Error processing {point_name}: {e}")
            continue

    # show only the last created figure
    fig.show()
    print(f"\n{'#' * 70}")
    print(f"#  All heat budget timeseries plots completed!")
    print(f"# Output location: {save_dir}")
    print(f"{'#' * 70}\n")


def plot_static_heat_budget_timeseries(point_names=None, variables=None, ylimits=None, xlimits=None):
    """
    Create static matplotlib plots showing heat budget variables for specified points.

    Parameters
    ----------
    point_names : str or list, optional
        Name(s) of the point(s) (from confg.ALL_POINTS). Can be a single string or a list of strings.
        If None, uses all points from confg.ALL_POINTS
    variables : list, optional
        List of variables to plot. If None, uses HEAT_BUDGET_VARS (default: all 6 variables)
    ylimits : tuple, optional
        Y-axis limits as (ymin, ymax). If None, auto-scales
    xlimits : tuple, optional
        X-axis limits as (xmin, xmax) with datetime strings or datetime objects.
        If None, defaults to ['2017-10-15 13:00:00', '2017-10-16 12:00:00']

    Returns
    -------
    list : list of matplotlib.figure.Figure
        The created figures
    """
    # Handle point_names input
    if point_names is None:
        point_names = list(confg.ALL_POINTS.keys())
    elif isinstance(point_names, str):
        point_names = [point_names]

    # Use default variables if none specified
    if variables is None:
        variables = HEAT_BUDGET_VARS

    # Use default time limits if none specified
    if xlimits is None:
        xlimits = [datetime(2017, 10, 15, 13, 0, 0), datetime(2017, 10, 16, 12, 0, 0)]
    else:
        # Convert to datetime if strings
        xlimits = [datetime.fromisoformat(str(x)) if isinstance(x, str) else x for x in xlimits]

    # Create output directory
    save_dir = os.path.join(confg.dir_PLOTS, "heat_flux")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"# Creating static heat budget timeseries plots")
    print(f"# Points to process: {len(point_names)}")
    print(f"# Output directory: {save_dir}")
    print(f"{'#' * 70}")

    figures = []

    # Process each point
    for point_name in point_names:
        # Get point info
        if point_name not in confg.ALL_POINTS:
            print(f"\n Warning: Point '{point_name}' not found in confg.ALL_POINTS, skipping...")
            continue

        point_info = confg.ALL_POINTS[point_name]

        print(f"\n{'=' * 70}")
        print(f"Creating static plot for: {point_name}")
        print(f"  Coordinates: lat={point_info['lat']:.4f}, lon={point_info['lon']:.4f}")
        # print(f"  Height: {point_info['height']} m")
        print(f"  Variables: {variables}")
        print(f"{'=' * 70}")

        try:
            # Get PCGP coordinates for accurate point representation
            pcgp_arome, pcgp_icon, pcgp_um, pcgp_wrf = read_dems_calc_pcgp(lat=point_info["lat"], lon=point_info["lon"])

            # Read data from both models
            arome_ds = read_in_arome.read_2D_variables_AROME(variableList=variables, lon=pcgp_arome.x.values,
                                                             lat=pcgp_arome.y.values, slice_lat_lon=False)

            wrf_ds = read_wrf_helen.read_wrf_fixed_point(lat=pcgp_wrf.y.values, lon=pcgp_wrf.x.values,
                                                         variables=variables, height_as_z_coord=False)

            # Load observation data if available for this point
            obs_ds = None
            if point_name in EC_STATION_MAPPING:
                try:
                    station_key = EC_STATION_MAPPING[point_name]
                    obs_ds = prepare_EC_datasets(station_key)
                    print(f"  Loaded observation data for station key {station_key}")
                except Exception as e:
                    print(f"  Warning: Could not load observation data: {e}")
                    obs_ds = None

            # Create figure
            fig, ax = plt.subplots(figsize=(14, 7))

            # Track which variables and models were plotted
            plotted_vars = set()
            plotted_models = set()

            # Plot each heat budget variable
            for var in variables:
                if var not in VAR_METADATA:
                    print(f"  Warning: Variable '{var}' not in VAR_METADATA, skipping...")
                    continue

                var_meta = VAR_METADATA[var]

                # Plot AROME data
                if arome_ds is not None and var in arome_ds:
                    try:
                        data = arome_ds[var]
                        # Remove any extra dimensions
                        if len(data.dims) > 1:
                            for dim in data.dims:
                                if dim != "time":
                                    data = data.isel({dim: 0})

                        # Only add label if this variable hasn't been plotted yet
                        label = var_meta["label"] if var not in plotted_vars else None
                        ax.plot(data.time.values, data.values, color=var_meta["color"], linewidth=2, linestyle='--',
                                label=label)
                        plotted_vars.add(var)
                        plotted_models.add('AROME')
                    except Exception as e:
                        print(f"  Warning: Could not plot AROME {var}: {e}")

                # Plot WRF data
                if wrf_ds is not None and var in wrf_ds:
                    try:
                        data = wrf_ds[var]
                        # Remove any extra dimensions
                        if len(data.dims) > 1:
                            for dim in data.dims:
                                if dim != "time":
                                    data = data.isel({dim: 0})

                        # Only add label if this variable hasn't been plotted yet
                        label = var_meta["label"] if var not in plotted_vars else None
                        ax.plot(data.time.values, data.values, color=var_meta["color"], linewidth=2, linestyle=':',
                                label=label)
                        plotted_vars.add(var)
                        plotted_models.add('WRF')
                    except Exception as e:
                        print(f"  Warning: Could not plot WRF {var}: {e}")

                # Plot observation data if available
                if obs_ds is not None and var in obs_ds:
                    try:
                        data = obs_ds[var]
                        # Only add label if this variable hasn't been plotted yet
                        label = var_meta["label"] if var not in plotted_vars else None
                        ax.plot(data.time.values, data.values, color=var_meta["color"], linewidth=2, linestyle='-',
                                label=label)
                        plotted_vars.add(var)
                        plotted_models.add('OBS')
                    except Exception as e:
                        print(f"  Warning: Could not plot observation {var}: {e}")

            # Set labels
            # title_text = f'Heat Budget Timeseries - {point_info.get("name", point_name)} ({point_info["height"]} m)'
            # ax.set_title(title_text, fontsize=13)
            ax.set_ylabel('Heat flux [W m$^{-2}$]', fontsize=13)

            # Set axis limits
            if xlimits is not None:
                ax.set_xlim(xlimits)
            if ylimits is not None:
                ax.set_ylim(ylimits)

            # Format x-axis for dates - same style as plot_dwd_temperature.py
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))  # Major ticks every 4 hours
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%dth\n%H:%M'))  # Format: DD and HH:MM on separate lines
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # Minor ticks every hour

            # Grid - both major and minor
            ax.grid(True, alpha=0.3, which='both')
            ax.grid(True, alpha=0.1, which='minor')

            # Set tick label sizes
            plt.xticks(rotation=0, fontsize=13)
            plt.yticks(fontsize=13)

            # Create custom legend with variables (colors) and models (linestyles)
            from matplotlib.lines import Line2D

            # First, add variable legends (colored lines)
            variable_handles = []
            for var in plotted_vars:
                if var in VAR_METADATA:
                    var_meta = VAR_METADATA[var]
                    variable_handles.append(
                        Line2D([0], [0], color=var_meta["color"], linewidth=2, label=var_meta["label"]))

            # Then, add model legends (linestyles) - only once
            model_handles = []
            if 'AROME' in plotted_models:
                model_handles.append(Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='AROME'))
            if 'WRF' in plotted_models:
                model_handles.append(Line2D([0], [0], color='black', linewidth=2, linestyle=':', label='WRF'))
            if 'OBS' in plotted_models:
                model_handles.append(Line2D([0], [0], color='black', linewidth=2, linestyle='-', label='OBS'))

            # Combine both legends
            all_handles = variable_handles + model_handles
            ax.legend(handles=all_handles, fontsize=13, loc='best', ncol=2)

            plt.tight_layout()

            # Save the plot
            point_name_safe = point_name.replace(" ", "_")
            vars_str = "_".join(variables)
            output_file = os.path.join(save_dir, f"heat_budget_static_{point_name_safe}_{vars_str}.pdf")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Plot saved to: {output_file}")

            figures.append(fig)

        except Exception as e:
            print(f"\n Error processing {point_name}: {e}")
            continue

    print(f"\n{'#' * 70}")
    print(f"#  All static heat budget timeseries plots completed!")
    print(f"# Output location: {save_dir}")
    print(f"{'#' * 70}\n")

    return figures


if __name__ == '__main__':
    # Example: Plot for a subset of valley points
    # You can change this to confg.VALLEY_POINTS or specific point names
    points_to_plot = ["ibk_uni"]  # "patsch_EC_south", "ibk_uni", "ibk_airport", "kufstein", "jenbach", "hafelekar", "hohe_warte",

    # Or use all points:
    # points_to_plot = None

    # Create interactive Plotly plots for multiple points
    # plot_all_heat_budget_timeseries(point_names=points_to_plot)

    # Example 1: Create static matplotlib plots for multiple points with all variables (default)
    plot_static_heat_budget_timeseries(point_names=points_to_plot, variables=["hfs", "lfs"],
                                       ylimits=[-60, 60], xlimits=['2017-10-15 15:00:00', '2017-10-16 06:00:00'])
    # ["hfs", "lfs", "swd", "swu", "lwd", "lwu"]

    # Example 2: Create static plot for a single point (pass string instead of list)
    # plot_static_heat_budget_timeseries(point_names="ibk_uni", variables=None, ylimits=None, xlimits=None)

    # Example 3: Plot only specific variables with custom limits for multiple points
    # plot_static_heat_budget_timeseries(
    #     point_names=["ibk_uni", "ibk_airport"],
    #     variables=["hfs", "lfs"],  # Only sensible and latent heat
    #     ylimits=(-100, 400),
    #     xlimits=["2017-10-15 14:00:00", "2017-10-16 10:00:00"]
    # )

    plt.show()