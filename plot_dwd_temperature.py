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


def plot_temperature_timeseries(csv_file_path):
    """
    Plot temperature timeseries from DWD CSV file

    Parameters:
    csv_file_path (str): Path to the CSV file with temperature data
    """

    # Read the CSV data - only columns "Zeitstempel" and "Wert" needed
    print(f"Reading data from: {csv_file_path}")
    df = pd.read_csv(csv_file_path, header=0, usecols=["SDO_ID", "Zeitstempel"])

    # Convert timestamp to datetime index and temperature to numeric
    df.index = pd.to_datetime(df['SDO_ID'])
    df = df.drop(["SDO_ID"], axis=1)
    df = df.rename(columns={"Zeitstempel": "temp"})

    print(f"Data loaded: {len(df)} records")
    print(f"Time period: {df.index.min()} to {df.index.max()}")
    print(f"Temperature range: {df['temp'].min():.1f}°C to {df['temp'].max():.1f}°C")

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot temperature timeseries
    ax.plot(df.index, df['temp'], color=confg.model_colors_temp_wind["AROME"], linewidth=1.5, label='DWD Rosenheim 2m Temperature')

    # Formatting with nice x-axis timestamps (like xarray)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Major ticks every 6 hours
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%dth\n%H:%M'))  # Format: DD and HH:MM on separate lines
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # Minor ticks every hour

    # Set x-axis limits
    ax.set_xlim([datetime.datetime(2017, 10, 15, 12, 0, 0), datetime.datetime(2017, 10, 16, 12, 0, 0)])
    ax.set_ylabel('Temperature [°C]', fontsize=12)
    # ax.set_title(f'2m-Temperature Rosenheim ({confg.ALL_POINTS["rosenheim"]["height"]})', fontsize=14,
    #              fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, which='both')
    ax.grid(True, alpha=0.1, which='minor')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)

    # Add legend
    ax.legend(fontsize=10)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    output_file = os.path.join(confg.dir_PLOTS, "temperature_wind", "rosenheim_temperature_timeseries.svg")
    plt.savefig(output_file)
    print(f"Plot saved as: {output_file}")

    # Show plot
    plt.show()

    return df


if __name__ == "__main__":
    """Main function"""

    try:
        df = plot_temperature_timeseries(confg.rosenheim_data)

        # Optional: print some statistics
        print(f"\nTemperature statistics:")
        print(f"Mean: {df['temperature'].mean():.1f}°C")
        print(f"Min:  {df['temperature'].min():.1f}°C")
        print(f"Max:  {df['temperature'].max():.1f}°C")
        print(f"Std:  {df['temperature'].std():.1f}°C")

    except FileNotFoundError:
        print(f"Error: CSV file '{confg.rosenheim_data}' not found!")
        print("Please make sure the file exists or adjust the file path.")
    except Exception as e:
        print(f"Error: {e}")