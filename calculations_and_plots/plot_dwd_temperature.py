"""
Simple script to plot temperature timeseries from DWD CSV file
Uses AROME color from confg.py

Author: GitHub Copilot
Date: 2026-02-11
"""
import fix_win_DLL_loading_issue

fix_win_DLL_loading_issue
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime
import os

import confg
from manage_timeseries import load_or_read_timeseries
# from hobo_utils import load_hobo_data, get_westernmost_hobo_station, add_hobo_timeseries_to_plot
from momaa_hobo_utils import load_momaa_data, get_momma_station_timeseries


def _plot_temperature_comparison(df, obs_label, output_filename, model_data=None, momma_station_id=None, ds_momma=None):
    """
    Shared plotting function for temperature comparisons (DRY principle)

    Parameters:
    df (pd.DataFrame): DataFrame with datetime index and 'temp' column
    obs_label (str): Label for the observation data
    output_filename (str): Filename for saving the plot
    model_data (dict): Dictionary with model names as keys and xarray datasets as values
    momma_station_id (str): Station ID for MOMMA station to plot as dotted line (e.g., "PM02")
    ds_momma (xr.Dataset): MOMMA dataset (optional, will be loaded if not provided)
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot temperature timeseries from observations
    ax.plot(df.index, df['temp'], color=confg.model_colors_temp_wind["HATPRO"], linewidth=2, label=obs_label,
            linestyle="--", zorder=10)

    # # Plot HOBO station timeseries if requested
    # if hobo_station_key is not None:
    #     if ds_hobo is None:
    #         ds_hobo = load_hobo_data()
    #     if ds_hobo is not None:
    #         add_hobo_timeseries_to_plot(
    #             ax=ax,
    #             ds_hobo=ds_hobo,
    #             station_key=hobo_station_key,
    #             color=confg.model_colors_temp_wind["HATPRO"],
    #             linewidth=1,
    #             linestyle=":",
    #             label=f'HOBO Station {hobo_station_key}'
    #         )

    # Plot MOMMA station timeseries if requested
    if momma_station_id is not None:
        if ds_momma is not None:
            times, temps = get_momma_station_timeseries(ds_momma, momma_station_id)
            if times is not None and len(times) > 0:
                ax.plot(times, temps, color=confg.model_colors_temp_wind["HATPRO"], linewidth=1, linestyle="--",
                        label=f'MOMMA {momma_station_id}')

    # Plot model temperatures
    if model_data is not None:
        for model_name, ds in model_data.items():
            # Select the lowest level (height=0) temperature using nearest neighbor selection
            if 'z' in ds.dims:
                temp_data = ds['temp'].sel(z=0, method="nearest")
            elif 'height' in ds.dims:
                temp_data = ds['temp'].sel(height=0, method="nearest")
            else:
                # If no vertical dimension, just use temp as is
                temp_data = ds['temp']

            # print(f"{model_name} Model")
            # print(f"Min: {temp_data['temp'].min():.1f}, Max: {df['temp'].max():.1f}")
            # print(f"Diff:  {(df['temp'].max() - df['temp'].min()):.1f}°C")

            # Plot the model temperature
            ax.plot(temp_data.time, temp_data, color=confg.model_colors_temp_wind[model_name], linewidth=2,
                    label=f'{model_name}', alpha=0.8)

    # Formatting with nice x-axis timestamps (like xarray)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))  # Major ticks every 4 hours
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%dth\n%H:%M'))  # Format: DD and HH:MM on separate lines
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # Minor ticks every hour

    # Set x-axis limits
    ax.set_xlim([datetime.datetime(2017, 10, 15, 12, 0, 0), datetime.datetime(2017, 10, 16, 12, 0, 0)])
    ax.set_ylabel('Temperature [°C]', fontsize=13)
    plt.ylim([4, 27])

    # Grid
    ax.grid(True, alpha=0.3, which='both')
    ax.grid(True, alpha=0.1, which='minor')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0, fontsize=13)
    plt.yticks(fontsize=13)
    ax.legend(fontsize=13)  # Add legend

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    output_file = os.path.join(confg.dir_PLOTS, "temperature_wind", output_filename)
    plt.savefig(output_file)
    print(f"Plot saved as: {output_file}")

    # Show plot  # plt.show()


def plot_dwd_temperature_timeseries(csv_file_path, model_data=None):
    """
    Plot temperature timeseries from DWD CSV file and model data

    Parameters:
    csv_file_path (str): Path to the CSV file with temperature data
    model_data (dict): Dictionary with model names as keys and xarray datasets as values

    Returns:
    pd.DataFrame: DataFrame with datetime index and temperature data
    """
    # Read the CSV data - only columns "Zeitstempel" and "Wert" needed
    print(f"Reading DWD data from: {csv_file_path}")
    df = pd.read_csv(csv_file_path, header=0, usecols=["SDO_ID", "Zeitstempel"])

    # Convert timestamp to datetime index and temperature to numeric
    df.index = pd.to_datetime(df['SDO_ID'])
    df = df.drop(["SDO_ID"], axis=1)
    df = df.rename(columns={"Zeitstempel": "temp"})

    # Use shared plotting function
    _plot_temperature_comparison(df, obs_label='TAWES T2m', output_filename="rosenheim_temperature_timeseries.pdf",
                                 model_data=model_data)

    return df


def plot_zamg_temperature_timeseries(csv_file_path, model_data=None,
        output_filename="innsbruck_uni_temperature_timeseries.pdf", add_momma="PM02"):
    """
    Plot temperature timeseries from ZAMG CSV file and model data

    Parameters:
    csv_file_path (str): Path to the CSV file with temperature data (contains 'tl' column)
    model_data (dict): Dictionary with model names as keys and xarray datasets as values
    output_filename (str): Filename for saving the plot
    add_momma_pm02 (bool): If True, adds MOMMA PM02 station as dotted line

    Returns:
    pd.DataFrame: DataFrame with datetime index and temperature data
    """
    # Read the CSV data - extract time and tl (temperature) column
    print(f"Reading ZAMG data from: {csv_file_path}")
    df = pd.read_csv(csv_file_path, header=0, usecols=["time", "tl"])

    # Convert timestamp to datetime index
    df.index = pd.to_datetime(df['time'])
    df = df.drop(["time"], axis=1)
    df = df.rename(columns={"tl": "temp"})

    # Remove any NaN values that might be present in the data
    df = df.dropna(subset=['temp'])

    # # Get westernmost HOBO station key and dataset if requested
    # hobo_station_key = None
    # ds_hobo = None
    # if add_hobo_westernmost:
    #     ds_hobo = load_hobo_data()
    #     if ds_hobo is not None:
    #         hobo_station_key = get_westernmost_hobo_station(ds_hobo)
    #         print(f"Adding westernmost HOBO station: {hobo_station_key}")

    # Get requested MOMMA station
    momma_station_id = None
    ds_momma = None
    if add_momma is not None:
        ds_momma = load_momaa_data()
        if ds_momma is not None:
            print(f"Adding MOMMA station: {add_momma}")

    # Use shared plotting function
    _plot_temperature_comparison(df, obs_label='T2m Observation', output_filename=output_filename,
                                 model_data=model_data, momma_station_id=momma_station_id, ds_momma=ds_momma)

    return df


def plot_all_momma_temperatures():
    """
    Plot temperature timeseries for all available MOMMA stations in one plot.

    Color scheme:
    - PM03, PM04, PM05: higher up/ in Wipp valley/foehn influence: Yellow/orange-ish colors (similar)
    - PM06, PM07: Innsbruck: Similar colors
    - PM08: east of Ibk, in drainage-area of Inn-valley: Dashed line
    - PM02, PM09, PM10: west of Ibk: in pooling area of Inn-valley, colder: Dotted lines
    """
    # Define specific colors and line styles for each station

    station_styles = {"PM02": {"color": confg.model_colors_temp_wind["HATPRO"], "linestyle": "-"},
        "PM03": {"color": confg.model_colors_temp_wind["AROME"], "linestyle": "-"},  #
        "PM04": {"color": confg.qualitative_colors_temp[1], "linestyle": "-"},
        "PM05": {"color": confg.model_colors_temp_wind["AROME"], "linestyle": "--"},
        "PM06": {"color": confg.model_colors_temp_wind["ICON"], "linestyle": "-"},
        "PM07": {"color": confg.model_colors_temp_wind["ICON2TE"], "linestyle": "-"},
        "PM08": {"color": confg.model_colors_temp_wind["Radiosonde"], "linestyle": "dotted"},
        "PM09": {"color": confg.model_colors_temp_wind["HATPRO"], "linestyle": "--"},
        "PM10": {"color": confg.model_colors_temp_wind["LIDAR142"], "linestyle": "-"}, }

    # Load MOMMA data
    ds_momma = load_momaa_data()
    if ds_momma is None:
        print("✗ Error: Could not load MOMMA data")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each MOMMA station
    station_ids = list(confg.MOMMA_stations_PM.keys())
    print(f"Plotting {len(station_ids)} MOMMA stations...")

    for station_id in station_ids:
        times, temps = get_momma_station_timeseries(ds_momma, station_id)

        if times is not None and len(times) > 0:
            station_name = confg.MOMMA_stations_PM[station_id]['name']
            station_height = confg.MOMMA_stations_PM[station_id]['height']

            # Get custom style for this station
            style = station_styles.get(station_id, {"color": "black", "linestyle": "-", "linewidth": 1.5})

            ax.plot(times, temps, color=style["color"], linewidth=2, linestyle=style["linestyle"],
                    label=f'{station_id}: {station_name} ({station_height}m)')

            print(f"  ✓ {station_id}: {station_name} - {len(times)} data points, "
                  f"Temp: {temps.min():.1f} to {temps.max():.1f}°C")
        else:
            print(f"  ✗ {station_id}: No data available")

    # Formatting
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m\n%H:%M'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

    # Set x-axis limits
    ax.set_xlim([datetime.datetime(2017, 10, 15, 12, 0, 0), datetime.datetime(2017, 10, 16, 12, 0, 0)])
    ax.set_ylabel('Temperature [°C]', fontsize=13)

    # Grid
    ax.grid(True, alpha=0.3, which='both')
    ax.grid(True, alpha=0.1, which='minor')

    # Tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=13)

    # Legend - place outside the plot area
    ax.legend(fontsize=13, loc='best')

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    output_file = os.path.join(confg.dir_PLOTS, "temperature_wind", "all_momma_temperatures.pdf")
    plt.savefig(output_file, bbox_inches='tight')
    print(f"\n✓ Plot saved as: {output_file}")

    # Show plot
    plt.show()


if __name__ == "__main__":
    """Main function"""
    point = confg.ALL_POINTS["inzing"]  # "rosenheim", "ibk_uni"  "ibk_airport"
    point_name = point["name"]
    models = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]

    # Load model timeseries
    model_data = {}
    for model in models:
        ds = load_or_read_timeseries(model=model, point=point, point_name=point_name, height_as_z_coord="above_terrain")
        model_data[model] = ds

    if point_name == "Rosenheim":
        try:
            df = plot_dwd_temperature_timeseries(csv_file_path=confg.rosenheim_data, model_data=model_data)
            print("Observation")
            print(f"Min: {df['temp'].min():.1f}, Max: {df['temp'].max():.1f}")
            print(f"Diff:  {(df['temp'].max() - df['temp'].min()):.1f}°C")

        except FileNotFoundError:
            print(f"Error: CSV file '{confg.rosenheim_data}' not found!")
            print("Please make sure the file exists or adjust the file path.")
        except Exception as e:
            print(f"Error: {e}")
    elif point_name == "ibk uni":
        df = plot_zamg_temperature_timeseries(csv_file_path=confg.innsbruck_uni_zamg_new, model_data=model_data,
                                              output_filename="ibk_uni_temperature_timeseries.pdf", add_momma="PM02")
        for model in model_data:
            print(f"{model} Model")
            print(f"Diff:  {(df['temp'].max() - df['temp'].min()):.1f}°C")

    elif point_name == "ibk airport":
        df = plot_zamg_temperature_timeseries(csv_file_path=confg.innsbruck_airport_zamg_new, model_data=model_data,
                                              output_filename="ibk_airport_temperature_timeseries.pdf")

    elif point_name == "inzing":
        df = plot_zamg_temperature_timeseries(csv_file_path=confg.innsbruck_airport_zamg_new, model_data=model_data,
                                              add_momma="PM10", output_filename="inzing_temperature_timeseries.pdf")

    # Plot all MOMMA stations in one plot
    print("\n" + "=" * 60)
    print("Plotting all MOMMA stations...")
    print("=" * 60)  # plot_all_momma_temperatures()