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


def _plot_temperature_comparison(df, obs_label, output_filename, model_data=None):
    """
    Shared plotting function for temperature comparisons (DRY principle)

    Parameters:
    df (pd.DataFrame): DataFrame with datetime index and 'temp' column
    obs_label (str): Label for the observation data
    output_filename (str): Filename for saving the plot
    model_data (dict): Dictionary with model names as keys and xarray datasets as values
    """
    print(f"Data loaded: {len(df)} records")
    print(f"Time period: {df.index.min()} to {df.index.max()}")
    print(f"Temperature range: {df['temp'].min():.1f}°C to {df['temp'].max():.1f}°C")

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot temperature timeseries from observations
    ax.plot(df.index, df['temp'], color=confg.model_colors_temp_wind["HATPRO"], linewidth=2, label=obs_label,
            linestyle="--", zorder=10)

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

            # Plot the model temperature
            ax.plot(temp_data.time, temp_data, color=confg.model_colors_temp_wind[model_name], linewidth=2,
                    label=f'{model_name}', alpha=0.8)

    # Formatting with nice x-axis timestamps (like xarray)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))  # Major ticks every 4 hours
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%dth\n%H:%M'))  # Format: DD and HH:MM on separate lines
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # Minor ticks every hour

    # Set x-axis limits
    ax.set_xlim([datetime.datetime(2017, 10, 15, 12, 0, 0), datetime.datetime(2017, 10, 16, 12, 0, 0)])
    ax.set_ylabel('Temperature [°C]', fontsize=12)
    plt.ylim([4, 27])

    # Grid
    ax.grid(True, alpha=0.3, which='both')
    ax.grid(True, alpha=0.1, which='minor')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)

    ax.legend(fontsize=10)  # Add legend

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    output_file = os.path.join(confg.dir_PLOTS, "temperature_wind", output_filename)
    plt.savefig(output_file)
    print(f"Plot saved as: {output_file}")

    # Show plot
    plt.show()


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
    _plot_temperature_comparison(df, obs_label='DWD Observation',
                                 output_filename="rosenheim_temperature_timeseries.pdf", model_data=model_data)

    return df


def plot_zamg_temperature_timeseries(csv_file_path, model_data=None,
        output_filename="innsbruck_uni_temperature_timeseries.pdf"):
    """
    Plot temperature timeseries from ZAMG CSV file and model data

    Parameters:
    csv_file_path (str): Path to the CSV file with temperature data (contains 'tl' column)
    model_data (dict): Dictionary with model names as keys and xarray datasets as values
    output_filename (str): Filename for saving the plot

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

    # Use shared plotting function
    _plot_temperature_comparison(df, obs_label='Geosphere Observation', output_filename=output_filename,
                                 model_data=model_data)

    return df


if __name__ == "__main__":
    """Main function"""
    point = confg.ALL_POINTS["rosenheim"]  # "rosenheim", "ibk_uni"  "ibk_airport"
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

            # Optional: print some statistics
            print(f"\nTemperature statistics:")
            print(f"Mean: {df['temp'].mean():.1f}°C")
            print(f"Min:  {df['temp'].min():.1f}°C")
            print(f"Max:  {df['temp'].max():.1f}°C")
            print(f"Std:  {df['temp'].std():.1f}°C")

        except FileNotFoundError:
            print(f"Error: CSV file '{confg.rosenheim_data}' not found!")
            print("Please make sure the file exists or adjust the file path.")
        except Exception as e:
            print(f"Error: {e}")
    elif point_name == "ibk uni":
        df = plot_zamg_temperature_timeseries(csv_file_path=confg.innsbruck_uni_zamg_new, model_data=model_data,
                                              output_filename="ibk_uni_temperature_timeseries.pdf")

    elif point_name == "ibk airport":
        df = plot_zamg_temperature_timeseries(csv_file_path=confg.innsbruck_airport_zamg_new, model_data=model_data,
                                              output_filename="ibk_airport_temperature_timeseries.pdf")