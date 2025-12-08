"""
somehow I get the error if running the script completely (problem w.
backend...):
Process finished with exit code -1066598274 (0xC06D007E)
Avoid by setting a breakpoint before saving the plot, then the rest is working somehow...

Plot pressure and temperature along the Inn Valley from ZAMG/Geosphere station data.

This module contains plotting functions for visualizing station data along the Inn Valley,
including pressure reduced to a reference elevation and temperature.
"""

import os

import plotly.graph_objects as go
import plotly.offline as pyo

import confg
from calculations_and_plots.manage_timeseries import load_or_read_timeseries, MODEL_ORDER
from download_geosphere_data import (load_or_download_all_stations, reduce_pressure_to_reference_station,
                                     reduce_model_pressure_to_reference_station)

# Use consistent colors matching ZAMG station colors
# Map point names to the same color indices as ZAMG stations
station_color_map = {'Kufstein': confg.qualitative_colors_temp[0], 'Jenbach': confg.qualitative_colors_temp[2],
                     'Innsbruck Uni': confg.qualitative_colors_temp[4],
                     'Innsbruck Airport': confg.qualitative_colors_temp[6]}
model_color_map = {'kufstein': confg.qualitative_colors_temp[0],  # Same as Kufstein ZAMG
                   'jenbach': confg.qualitative_colors_temp[2],  # Same as Jenbach ZAMG
                   'ibk_uni': confg.qualitative_colors_temp[4],  # Same as Innsbruck Uni ZAMG
                   'ibk_airport': confg.qualitative_colors_temp[6]  # Same as Innsbruck Airport ZAMG
                   }


def plot_zamg_measurements(stations_data_reduced, stations_metadata, save_path=None):
    """
    Plot ZAMG station measurements in a separate figure.

    Parameters
    ----------
    stations_data_reduced : dict
        Dictionary with station name as key and DataFrame as value
    stations_metadata : dict
        Dictionary with station metadata including heights
    save_path : str, optional
        Path to save figure (as HTML)
    """
    if len(stations_data_reduced) == 0:
        print("No ZAMG data to plot!")
        return None

    # Create single plot for ZAMG measurements
    fig = go.Figure()

    # Plot ZAMG Station Pressure
    for station_name, api_df in stations_data_reduced.items():
        # Drop NaN values for cleaner plotting
        valid_data = api_df['p_reduced'].dropna()

        # Use mapped colors defined at beginning of file
        color = station_color_map[station_name]

        if station_name == "Innsbruck Uni":
            # height = confg.ALL_POINTS["ibk_uni"]["height"]  # set correct height for Innsbruck Uni
            height = 609.5  # m pressure height is at 609.5m (https://acinn-data.uibk.ac.at/pages/tawes-uibk.html)
        else:
            height = stations_metadata.get(station_name, {}).get('altitude', 'N/A')
        label = f"{station_name} ({height:.0f} m)"  # Create label with station height

        # Add station data trace
        fig.add_trace(go.Scatter(x=valid_data.index, y=valid_data.values, mode='lines', name=label,
                                 line=dict(color=color, width=3), opacity=0.9))

    # Update layout
    fig.update_layout(title=dict(text='ZAMG Station Measurements - Inn Valley Pressure', x=0.5,
                                 font=dict(size=18, family="Arial, sans-serif")), xaxis=dict(title='Time'),
                      yaxis=dict(title='Pressure reduced to Kufstein height [hPa]'), hovermode='x unified',
                      template='plotly_white', width=1200, height=600, margin=dict(l=80, r=50, t=100, b=80),
                      legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(size=11)))

    # Update axes
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=1,
                     range=['2017-10-15 14:00:00', '2017-10-16 11:00:00'])
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1)

    # Save and show
    if save_path:
        html_path = save_path.replace('.html', '_zamg_measurements.html')
        pyo.plot(fig, filename=html_path, auto_open=False)
        print(f"ZAMG measurements figure saved to: {html_path}")

    fig.show()
    return fig


def plot_model_data(model_data, model_name):
    """
    Plot data for a specific model in a separate figure.

    Parameters
    ----------
    model_data : dict
        Dictionary with model data {point_name: {model: xr.Dataset}}
    model_name : str
        Name of the model to plot
    save_path : str, optional
        Path to save figure (as HTML)
    """
    # Check if model has any data
    model_has_data = False
    for point_name, models in model_data.items():
        if model_name in models:
            model_has_data = True
            break

    if not model_has_data:
        print(f"No data available for model {model_name}")
        return None

    # Create single plot for this model
    fig = go.Figure()

    # Plot model data for all points
    for color_index, (point_name, models) in enumerate(model_data.items()):
        if model_name in models:
            ds = models[model_name]

            # Use p_reduced if available, otherwise skip
            if 'p_reduced' not in ds.variables:
                print(f"Warning: No p_reduced data for {model_name} at {point_name}")
                continue

            pressure_values = ds['p_reduced'].values

            # Use mapped colors defined at beginning of file
            color = model_color_map[point_name]  # .get(point_name, confg.qualitative_colors_temp[color_index * 2])

            # Add model data trace
            fig.add_trace(go.Scatter(x=ds.time.values, y=pressure_values, mode='lines', name=point_name,
                                     line=dict(color=color, width=2), opacity=0.8))

    # Update layout
    fig.update_layout(title=dict(text=f'{model_name} Model - Inn Valley Pressure', x=0.5,
                                 font=dict(size=18, family="Arial, sans-serif")), xaxis=dict(title='Time'),
                      yaxis=dict(title='Pressure reduced to Kufstein height [hPa]'), hovermode='x unified',
                      template='plotly_white', width=1200, height=600, margin=dict(l=80, r=50, t=100, b=80),
                      legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(size=11)))

    # Update axes
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=1,
                     range=['2017-10-15 14:00:00', '2017-10-16 11:00:00'])
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1)

    # Save and show
    html_path = os.path.join(confg.dir_PLOTS, "pressure_along_valley", f"pressure_comparison_{model_name.lower()}.html")
    # save_path.replace(
    # '.html',
    # f'_{model_name.lower()}_model.html')
    pyo.plot(fig, filename=html_path, auto_open=False)
    print(f"{model_name} model figure saved to: {html_path}")

    fig.show()
    return fig


def plot_all_separate(stations_data_reduced, stations_metadata, model_data=None, save_path=None):
    """
    Create separate plots for ZAMG measurements and each model.

    Parameters
    ----------
    stations_data_reduced : dict
        Dictionary with station name as key and DataFrame as value
    stations_metadata : dict
        Dictionary with station metadata including heights
    model_data : dict, optional
        Dictionary with model data {point_name: {model: xr.Dataset}}
    save_path : str, optional
        Base path to save figures (as HTML)
    """
    figures = {}

    # Plot ZAMG measurements
    print("Creating ZAMG measurements plot...")
    figures['zamg'] = plot_zamg_measurements(stations_data_reduced, stations_metadata, save_path)

    # Plot each model separately
    if model_data is not None:
        models_available = ['AROME', 'ICON', 'ICON2TE', 'UM']  # skip WRF

        for model in models_available:
            print(f"Creating {model} model plot...")
            figures[model.lower()] = plot_model_data(model_data, model)

    return figures


def plot_combined_subplots(stations_data_reduced, stations_metadata, model_data=None, save_path=None):
    """
    Create a single matplotlib plot with multiple subplots for ZAMG measurements and each model.

    Parameters
    ----------
    stations_data_reduced : dict
        Dictionary with station name as key and DataFrame as value
    stations_metadata : dict
        Dictionary with station metadata including heights
    model_data : dict, optional
        Dictionary with model data {point_name: {model: xr.Dataset}}
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd

    # Define models to plot
    models_available = ['AROME', 'ICON', 'ICON2TE', 'UM'] if model_data else []
    total_plots = 1 + len(models_available)  # 1 for ZAMG + models

    # Calculate rows and columns for subplots (3 rows, 2 columns for 5 plots)
    rows = 3
    cols = 2

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12), sharex=True)
    axes = axes.flatten()  # Make it easier to iterate

    # Define time range
    time_start = pd.to_datetime('2017-10-15 14:00:00')
    time_end = pd.to_datetime('2017-10-16 11:00:00')

    plot_idx = 0

    # Plot 1: ZAMG Station Measurements
    if len(stations_data_reduced) > 0:
        ax = axes[plot_idx]

        legend_handles = []
        legend_labels = []

        for station_name, api_df in stations_data_reduced.items():
            valid_data = api_df['p_reduced'].dropna()
            color = station_color_map.get(station_name, confg.qualitative_colors_temp[0])

            if station_name == "Innsbruck Uni":
                # height = confg.ALL_POINTS["ibk_uni"]["height"]
                height = 609.5  # m pressure height is at 609.5m (https://acinn-data.uibk.ac.at/pages/tawes-uibk.html)
            else:
                height = stations_metadata.get(station_name, {}).get('altitude', 'N/A')
            label = f"{station_name} ({height:.0f} m)"

            # Plot with consistent styling
            line = ax.plot(valid_data.index, valid_data.values,
                          color=color, linewidth=2.5, alpha=0.9, label=label)

            # Collect handles and labels for single legend
            legend_handles.append(line[0])
            legend_labels.append(label)

        ax.set_title('ZAMG Station Measurements', fontsize=14, fontweight='bold')
        ax.set_ylabel('Pressure reduced to\nKufstein height [hPa]', fontsize=11)
        ax.grid(True, alpha=0.3)
        # Remove individual legend - we'll create one global legend
        ax.set_xlim(time_start, time_end)
        ax.set_ylim(967, 974)  # Set uniform y-axis range
        plot_idx += 1

    # Plot 2-5: Model data
    if model_data is not None:
        for model_name in models_available:
            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]
            model_has_data = False

            for point_name, models in model_data.items():
                if model_name in models:
                    ds = models[model_name]

                    if 'p_reduced' not in ds.variables:
                        continue

                    model_has_data = True
                    pressure_values = ds['p_reduced'].values
                    color = model_color_map.get(point_name, confg.qualitative_colors_temp[0])

                    # Convert xarray time to pandas datetime for matplotlib
                    time_values = pd.to_datetime(ds.time.values)

                    # Plot with consistent styling
                    ax.plot(time_values, pressure_values,
                           color=color, linewidth=2, alpha=0.8, label=point_name)

            ax.set_title(f'{model_name} Model', fontsize=14, fontweight='bold')

            # Only add y-label for left plots (even indices: 0, 2, 4)
            if plot_idx % 2 == 1:  # Left plots (index 1, 3, 5 -> plot positions 0, 2, 4 in subplot grid)
                ax.set_ylabel('Pressure reduced to\nKufstein height [hPa]', fontsize=11)

            ax.grid(True, alpha=0.3)
            # Remove individual legend
            ax.set_xlim(time_start, time_end)
            ax.set_ylim(967, 974)  # Set uniform y-axis range

            if not model_has_data:
                print(f"No data available for model {model_name}")
                ax.text(0.5, 0.5, 'No data available',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, style='italic', alpha=0.7)

            plot_idx += 1

    # Hide unused subplot(s)
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    # Format x-axis for all visible plots
    for i in range(plot_idx):
        ax = axes[i]
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=2))

        # Remove x-axis labels (no "Time" label)
        plt.setp(ax.xaxis.get_majorticklabels())

    # Create single legend in lower right using ZAMG station data
    if len(stations_data_reduced) > 0:
        fig.legend(legend_handles, legend_labels, loc='lower right',
                  bbox_to_anchor=(0.8, 0.2), fontsize=11, frameon=True,
                  fancybox=True, shadow=True)

    # Overall title
    # fig.suptitle('Pressure Along Inn Valley - Combined View',
    #            fontsize=18, fontweight='bold', y=0.95)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.15)
    # plt.show()
    # Save and show
    # combined_path = save_path.replace('_combined_subplots.')
    plt.savefig(save_path, dpi=400)
    print(f"Combined subplot figure saved to: {save_path}")

    return fig


if __name__ == "__main__":
    """
    Main function to plot pressure along the valley.
    Loads data from CSV files, reduces pressure to reference station, and plots.
    """
    print("=" * 70)
    print("Plotting Pressure Along Inn Valley")
    print("=" * 70)

    # Load or download all station data
    stations_data, stations_metadata = load_or_download_all_stations()

    if not stations_data:
        print("\n✗ No data loaded or downloaded! Please check your configuration.")
        exit(1)

    # Reduce pressure to reference station (Innsbruck Uni)
    print(f"\n{'=' * 70}")
    print("Reducing pressure to Innsbruck Uni elevation...")
    print(f"{'=' * 70}")

    reference_station = 'Kufstein'
    stations_data_reduced = reduce_pressure_to_reference_station(stations_data, stations_metadata,
                                                                 reference_station=reference_station)

    # Load model data for the corresponding points
    print(f"\n{'=' * 70}")
    print("Loading model data for comparison points...")
    print(f"{'=' * 70}")

    # Map station names to confg.ALL_POINTS keys
    station_to_point_map = {"Kufstein": "kufstein", "Jenbach": "jenbach", "Innsbruck Uni": "ibk_uni",
                            "Innsbruck Airport": "ibk_airport"}

    model_data = {}  # {point_name: {model: xr.Dataset}}

    for station_name, point_key in station_to_point_map.items():
        if point_key is None:
            continue  # Skip if no corresponding point

        if point_key not in confg.ALL_POINTS:
            print(f"  ✗ Point {point_key} not found in confg.ALL_POINTS")
            continue

        point = confg.ALL_POINTS[point_key]
        model_data[point_key] = {}

        print(f"\n  Loading models for {point['name']}:")
        for model in MODEL_ORDER:
            try:
                ds = load_or_read_timeseries(model=model, point=point, point_name=point_key, height_as_z_coord="direct")
                if ds is not None:
                    model_data[point_key][model] = ds
                    print(f"    ✓ {model} loaded")
                else:
                    print(f"    ✗ {model} not available")
            except Exception as e:
                print(f"    ✗ {model} error: {e}")

    # Reduce model pressures to reference station elevation
    print(f"\n{'=' * 70}")
    print("Reducing model pressures to Kufstein elevation...")
    print(f"{'=' * 70}")

    model_data_reduced = reduce_model_pressure_to_reference_station(model_data, stations_metadata,
                                                                    reference_station=reference_station)

    # Create separate plots for thesis
    print(f"\n{'=' * 70}")
    print("Creating separate plots for thesis...")
    print(f"{'=' * 70}")

    # Create output directory if it doesn't exist
    output_dir = os.path.join(confg.dir_PLOTS, "pressure_along_valley")
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, "geosphere_api_downloaded_data.html")
    # figures = plot_all_separate(stations_data_reduced, stations_metadata, model_data=model_data_reduced,
    #                             save_path=save_path)
    # Create combined subplot view
    print(f"\n{'=' * 70}")
    print("Creating combined subplot view...")
    print(f"{'=' * 70}")

    combined_save_path = os.path.join(output_dir, "pressure_comparison_small_multiples.png")
    combined_figure = plot_combined_subplots(stations_data_reduced, stations_metadata, model_data=model_data_reduced,
                                             save_path=combined_save_path)

    print(f"\n{'=' * 70}")
    print("Plotting complete!")
    print(f"{'=' * 70}")