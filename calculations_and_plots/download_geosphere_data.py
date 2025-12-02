"""
somehow I get the error if running the script completely:
Process finished with exit code -1066598274 (0xC06D007E)
Avoid by setting a breakpoint before saving the plot, then the rest is working somehow...

Download and verify/plot the ZAMG/Geosphere station data via API.

This script downloads data from the Geosphere Austria API for comparison with existing CSV files.
API Documentation: https://dataset.api.hub.geosphere.at/v1/docs/

Stations:
- Kufstein (station 9016)
- Jenbach (station 11901)
- Innsbruck University (station 11803)
- Innsbruck Airport (station 11804)

Time period: 2017-10-15 12:00:00 to 2017-10-16 12:00:00
Dataset: 10-minute observations (klima-v2-10min)
"""

import os

import json
# Import required modules for datetime formatting
import matplotlib.dates as mdates
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

import confg

# Geosphere API base URL
API_BASE_URL = "https://dataset.api.hub.geosphere.at/v1"

# Station IDs and names from confg.station_files_zamg_new
# Mapping from confg station codes to Geosphere API station IDs
STATION_ID_MAPPING = {'KUF': '9016',  # Kufstein
                      'JEN': '11901',  # Jenbach
                      'IAO': '11803',  # Innsbruck University
                      'LOWI': '11804'  # Innsbruck Airport
                      }

# Time period
START_TIME = "2017-10-15T12:00:00"
END_TIME = "2017-10-16T12:00:00"

# Dataset resource ID for 10-minute climate data
# Based on API docs: klima-v2-10min is the 10-minute historical dataset
DATASET_RESOURCE = "klima-v2-10min"


def get_station_metadata(station_id):
    """
    Get metadata for a specific station.

    Parameters
    ----------
    station_id : str
        Station ID

    Returns
    -------
    dict
        Station metadata
    """
    url = f"{API_BASE_URL}/station/historical/klima-v2-10min/metadata"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()

        # The response has a 'stations' key containing the list
        stations_list = data.get('stations', [])

        # Find the specific station - the ID field is 'id', not 'station_id'
        for station in stations_list:
            if str(station.get('id')) == str(station_id):
                return station

        print(f"Warning: Station {station_id} not found in metadata")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching station metadata: {e}")
        return None


def download_station_data(station_id, start_time, end_time, parameters=None):
    """
    Download data for a specific station from Geosphere API.

    Parameters
    ----------
    station_id : str
        Station ID (e.g., '9016' for Kufstein)
    start_time : str
        Start time in ISO format (e.g., '2017-10-15T12:00:00')
    end_time : str
        End time in ISO format
    parameters : list, optional
        List of parameters to download (default: ['P', 'TL', 'RF'])

    Returns
    -------
    pd.DataFrame
        DataFrame with downloaded data
    """
    if parameters is None:
        # Default parameters: Pressure, Temperature, Relative Humidity, Wind
        # P = station pressure, PRED = reduced pressure to MSL, TL = air temperature
        # RF = relative humidity, FF = wind speed, DD = wind direction
        # Quality flags are automatically included with _FLAG suffix
        parameters = ['P', 'PRED', 'TL', 'RF', 'FF', 'DD', 'P_FLAG', 'PRED_FLAG', 'TL_FLAG', 'RF_FLAG', 'FF_FLAG',
                      'DD_FLAG']

    # Build API URL for station data
    # Correct Geosphere API structure: /station/historical/{resource_id}
    # Station ID is passed as query parameter, not in URL path
    url = f"{API_BASE_URL}/station/historical/klima-v2-10min"

    # Parameters for the request - try CSV format first as it's more reliable
    params = {'parameters': ','.join(parameters), 'station_ids': station_id,  # Station ID as query parameter
              'start': start_time, 'end': end_time, 'output_format': 'csv'  # Changed from geojson to csv
              }

    print(f"  Requesting URL: {url}")
    print(f"  With parameters: {params}")
    full_url = f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
    print(f"  Full URL: {full_url}")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        # Parse CSV response
        from io import StringIO

        # Read CSV data directly into pandas
        df = pd.read_csv(StringIO(response.text), parse_dates=['time'])

        if df.empty:
            print(f"  Warning: No data returned for station {station_id}")
            return None

        # Set time as index
        df.set_index('time', inplace=True)

        return df

    except requests.exceptions.RequestException as e:
        print(f"  Error downloading data: {e}")
        if hasattr(e.response, 'text'):
            print(f"  Response: {e.response.text[:500]}")
        return None
    except json.JSONDecodeError as e:
        print(f"  Error parsing JSON response: {e}")
        return None


def read_csv_data(filepath, start_time, end_time):
    """
    Read existing CSV data for comparison.

    Parameters
    ----------
    filepath : str
        Path to CSV file
    start_time : str
        Start time for filtering
    end_time : str
        End time for filtering

    Returns
    -------
    pd.DataFrame
        DataFrame with CSV data
    """
    try:
        df = pd.read_csv(filepath, sep=',', parse_dates=['time'], index_col='time')
        # Filter to time range
        df_filtered = df.loc[start_time:end_time]
        return df_filtered
    except Exception as e:
        print(f"  Error reading CSV: {e}")
        return None


def compare_data(api_df, csv_df, station_name):
    """
    Compare API data with CSV data.

    Parameters
    ----------
    api_df : pd.DataFrame
        Data from API
    csv_df : pd.DataFrame
        Data from CSV
    station_name : str
        Station name for labeling
    """
    print(f"\n  Comparison for {station_name}:")
    print(f"  API data points: {len(api_df)}")
    print(f"  CSV data points: {len(csv_df)}")

    if api_df is None or csv_df is None:
        print("  Cannot compare - missing data")
        return

    # Compare pressure values
    if 'P' in api_df.columns and 'P' in csv_df.columns:
        # Align timestamps
        common_times = api_df.index.intersection(csv_df.index)

        if len(common_times) == 0:
            print("  Warning: No common timestamps found")
            return

        api_pressure = api_df.loc[common_times, 'P']
        csv_pressure = csv_df.loc[common_times, 'P']

        # Calculate differences
        diff = api_pressure - csv_pressure

        print(f"  Common timestamps: {len(common_times)}")
        print(f"  Pressure difference (API - CSV):")
        print(f"    Mean: {diff.mean():.4f} hPa")
        print(f"    Std:  {diff.std():.4f} hPa")
        print(f"    Max:  {diff.max():.4f} hPa")
        print(f"    Min:  {diff.min():.4f} hPa")

        # Check if data matches
        if diff.abs().max() < 0.1:
            print("  ✓ Data matches well (max difference < 0.1 hPa)")
        elif diff.abs().max() < 1.0:
            print("  ⚠ Data has small differences (max difference < 1.0 hPa)")
        else:
            print("  ✗ Data has significant differences!")


def plot_downloaded_data(stations_data_reduced, stations_metadata, save_path=None, reference_station='Kufstein'):
    """
    Plot downloaded API data for all stations in one plot.

    Parameters
    ----------
    stations_data_reduced : dict
        Dictionary with station name as key and DataFrame as value
    stations_metadata : dict
        Dictionary with station metadata including heights
    save_path : str, optional
        Path to save figure
    """
    if len(stations_data_reduced) == 0:
        print("No data to plot!")
        return

    # Create figure with two subplots: one for P, one for TL
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    # Plot Station Pressure (P)
    for color_index, (station_name, api_df) in enumerate(stations_data_reduced.items()):
        # Drop NaN values for cleaner plotting
        valid_data = api_df['p_reduced'].dropna()
        # Use confg.qualitative_colors_temp with incrementing index
        color = confg.qualitative_colors_temp[color_index * 2]

        if station_name == "Innsbruck Uni":
            height = confg.ALL_POINTS["ibk_uni"]["height"]  # set correct height for Innsbruck Uni
        else:
            height = stations_metadata.get(station_name, {}).get('altitude', 'N/A')
        label = f"{station_name} ({height:.0f} m)"  # Create label with station height

        ax[0].plot(valid_data.index, valid_data.values, '-', label=label, color=color, linewidth=2.5, alpha=0.9)

    ax[0].set_ylabel('Pressure reduced to Innsbruck Uni height [hPa]')
    ax[0].legend(loc='best', fontsize=11, framealpha=0.9)
    ax[0].grid(True, linestyle='--')

    # Plot Temperature (TL)
    for color_index, (station_name, api_df) in enumerate(stations_data_reduced.items()):
        # Drop NaN values for cleaner plotting
        valid_data = api_df['tl'].dropna()
        if len(valid_data) > 0:
            # Use confg.qualitative_colors_temp with incrementing index
            color = confg.qualitative_colors_temp[color_index * 2]

            # Create label with station height
            height = stations_metadata.get(station_name, {}).get('altitude', 'N/A')
            label = f"{station_name} ({height:.0f} m)"

            ax[1].plot(valid_data.index, valid_data.values, '-', label=label, color=color, linewidth=2.5, alpha=0.9)

    ax[1].set_ylabel('2m air Temperature TL [°C]')
    ax[1].legend(loc='best', fontsize=11, framealpha=0.9)
    ax[1].grid(True, linestyle='--')

    # Format x-axis with proper datetime formatting
    # Show date and time on x-axis
    date_format = mdates.DateFormatter('%Y-%m-%d\n%H:%M')
    ax[1].xaxis.set_major_formatter(date_format)

    # Set major ticks every 3 hours
    ax[1].xaxis.set_major_locator(mdates.HourLocator(interval=4))
    # Set minor ticks every hour
    ax[1].xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    # Rotate and align the tick labels so they look better
    plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=0, ha='center')

    plt.suptitle('Geosphere API Downloaded Data - Inn Valley Stations', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path, format='svg')
    print(f"\nFigure saved to: {save_path}")


def reduce_pressure_to_reference_station(stations_data, stations_metadata, reference_station='Kufstein'):
    """
    Reduce pressure from all stations to the reference station elevation using barometric formula.

    The barometric formula accounts for temperature variation and is more accurate than
    simple exponential approximation.

    Parameters
    ----------
    stations_data : dict
        Dictionary with station names as keys and DataFrames as values
    stations_metadata : dict
        Dictionary with station names as keys and metadata dicts as values
    reference_station : str
        short Geosphere name of reference station (default: 'Kufstein')

    Returns
    -------
    dict
        Dictionary with station names as keys and DataFrames with reduced pressure as values
    """
    # Physical constants
    R = 287.05  # Specific gas constant for dry air [J/(kg·K)]
    g = 9.80665  # Standard gravity [m/s²]

    # Get reference station elevation
    if reference_station not in stations_metadata:
        print(f"Warning: Reference station '{reference_station}' not found in metadata!")
        return stations_data

    ref_elevation = stations_metadata[reference_station]['altitude']
    print(f"\nReducing all pressures to {reference_station} elevation: {ref_elevation} m")

    stations_data_reduced = {}

    for station_name, df in stations_data.items():
        if station_name not in stations_metadata:
            print(f"Warning: No metadata for station '{station_name}' - keeping original pressure")
            stations_data_reduced[station_name] = df.copy()
            continue

        # wrong height of pressure measurement for Ibk uni station in the Metadata!
        if station_name == "Innsbruck Uni":
            station_elevation = confg.ALL_POINTS["ibk_uni"]["height"]
        else:
            station_elevation = stations_metadata[station_name]['altitude']
        elevation_diff = station_elevation - ref_elevation  # Positive if station is higher

        print(f"  {station_name}: {station_elevation} m -> {ref_elevation} m (Δh = {elevation_diff:.1f} m)")

        # Create copy of DataFrame
        df_reduced = df.copy()

        if 'p' in df.columns and 'tl' in df.columns:
            # Barometric formula: P_reduced = P_station * exp((g * Δh) / (R * T))
            # If station is higher (elevation_diff > 0), pressure increases when reduced to lower level
            # If station is lower (elevation_diff < 0), pressure decreases when reduced to higher level

            df_reduced['p_reduced'] = df['p'] * np.exp((g * elevation_diff) / (R * (df['tl'] + 273.15)))

            # Show statistics
            original_mean = df['p'].mean()
            reduced_mean = df_reduced['p_reduced'].mean()
            print(f"    Original pressure: {original_mean:.2f} hPa")
            print(f"    Reduced pressure:  {reduced_mean:.2f} hPa")
            print(f"    Mean difference:   {reduced_mean - original_mean:.2f} hPa")

        elif 'p' in df.columns:
            # Fallback: Use standard atmosphere if no temperature data
            print(f"    Warning: No temperature data for {station_name}, using standard atmosphere (T=15°C)")
            T_standard = 288.15  # 15°C in Kelvin
            pressure_factor = np.exp((g * elevation_diff) / (R * T_standard))
            df_reduced['p_reduced'] = df['p'] * pressure_factor

        else:
            print(f"    Warning: No pressure data for {station_name}")

        stations_data_reduced[station_name] = df_reduced

    return stations_data_reduced


def plot_pressure_comparison(stations_data, stations_data_reduced, save_path=None):
    """
    Plot comparison of original vs reduced pressure for all stations.

    Parameters
    ----------
    stations_data : dict
        Original station data
    stations_data_reduced : dict
        Station data with reduced pressure
    save_path : str, optional
        Path to save figure
    """
    if len(stations_data) == 0:
        print("No data to plot!")
        return

    import matplotlib.dates as mdates

    # Create figure with three subplots: original P, reduced P, and temperature
    fig, ax = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Plot Original Pressure
    for color_index, (station_name, api_df) in enumerate(stations_data.items()):
        if 'p_reduced' in api_df.columns:
            valid_data = api_df['p'].dropna()
            color = confg.qualitative_colors_temp[color_index * 2]
            ax[0].plot(valid_data.index, valid_data.values, '-', label=station_name, color=color, linewidth=2.5,
                       alpha=0.9)

    ax[0].set_ylabel('Original Pressure [hPa]')
    ax[0].set_title('Original Station Pressure', fontweight='bold')
    ax[0].legend(loc='best', fontsize=10, framealpha=0.9)
    ax[0].grid(True, linestyle='--', alpha=0.7)

    # Plot Reduced Pressure
    for color_index, (station_name, api_df_red) in enumerate(stations_data_reduced.items()):
        if 'p_reduced' in api_df_red.columns:
            valid_data = api_df_red['p_reduced'].dropna()
            color = confg.qualitative_colors_temp[color_index * 2]
            ax[1].plot(valid_data.index, valid_data.values, '-', label=station_name, color=color, linewidth=2.5,
                       alpha=0.9)

    ax[1].set_ylabel('Reduced Pressure [hPa]')
    ax[1].set_title('Pressure Reduced to Innsbruck University Level (578 m)', fontweight='bold')
    ax[1].legend(loc='best', fontsize=10, framealpha=0.9)
    ax[1].grid(True, linestyle='--', alpha=0.7)

    # Plot Temperature (for reference)
    for color_index, (station_name, api_df) in enumerate(stations_data.items()):
        if 'tl' in api_df.columns:
            valid_data = api_df['tl'].dropna()
            color = confg.qualitative_colors_temp[color_index * 2]
            ax[2].plot(valid_data.index, valid_data.values, '-', label=station_name, color=color, linewidth=2.5,
                       alpha=0.9)

    ax[2].set_ylabel('Temperature [°C]')
    ax[2].set_title('Air Temperature (used for pressure reduction)', fontweight='bold')
    ax[2].legend(loc='best', fontsize=10, framealpha=0.9)
    ax[2].grid(True, linestyle='--', alpha=0.7)

    # Format x-axis
    date_format = mdates.DateFormatter('%Y-%m-%d\n%H:%M')
    ax[2].xaxis.set_major_formatter(date_format)
    ax[2].xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax[2].xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    plt.setp(ax[2].xaxis.get_majorticklabels(), rotation=0, ha='center')

    plt.suptitle('Pressure Reduction Analysis - Inn Valley Stations', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"\nComparison figure saved to: {save_path}")

    plt.show()
    return fig


def save_downloaded_data(stations_data):
    """Save downloaded station data to CSV files if they don't exist."""

    # Get file paths from confg.station_files_zamg_new
    for station_name, df in stations_data.items():
        file_path = None

        # Find the corresponding file path in confg.station_files_zamg_new
        for station_code, station_info in confg.station_files_zamg_new.items():
            if station_info['name'] == station_name:
                file_path = station_info.get('filepath')
                break

        # Check if directory exists
        if not os.path.exists(os.path.dirname(file_path)):
            print(f"Warning: Directory not found for {station_name}: {os.path.dirname(file_path)}")
            continue
        # Save data
        df_to_save = df.reset_index()
        df_to_save.to_csv(file_path, index=False)
        print(f"Saved {station_name} to {file_path}")


def save_station_metadata(stations_metadata, metadata_file_path=None):
    """Save station metadata to JSON file."""
    if metadata_file_path is None:
        metadata_file_path = os.path.join(confg.data_folder, "Observations", "ZAMG_Tawes", "station_metadata.json")

    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)

        with open(metadata_file_path, 'w') as f:
            json.dump(stations_metadata, f, indent=2)
        print(f"Saved metadata to {metadata_file_path}")
    except Exception as e:
        print(f"Error saving metadata: {e}")


def load_station_metadata(metadata_file_path=None):
    """Load station metadata from JSON file."""
    if metadata_file_path is None:
        metadata_file_path = os.path.join(confg.data_folder, "Observations", "ZAMG_Tawes", "station_metadata.json")

    if os.path.exists(metadata_file_path):
        try:
            with open(metadata_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")

    return {}


if __name__ == "__main__":
    """
    Main function to download Geosphere data via API.
    """
    print("=" * 70)
    print("Geosphere Austria API Data Download")
    print("=" * 70)

    stations_data = {}

    # Load existing metadata if available
    stations_metadata = load_station_metadata()

    # Loop through stations from confg.station_files_zamg_new
    for station_code, station_info in confg.station_files_zamg_new.items():
        # Skip stations not in our mapping
        if station_code not in STATION_ID_MAPPING:
            print(f"\nSkipping {station_code} - not in mapping")
            continue

        station_id = STATION_ID_MAPPING[station_code]
        station_name = station_info['name']

        print(f"\n{'=' * 70}")
        print(f"Processing: {station_name} (Code: {station_code}, ID: {station_id})")
        print(f"{'=' * 70}")

        # Check if CSV file already exists and load it
        file_path = station_info.get('filepath')
        if file_path and os.path.exists(file_path):
            print(f"\n1. Loading existing data from {file_path}")
            try:
                api_df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
                print(f"  ✓ Loaded {len(api_df)} data points from existing file")
                stations_data[station_name] = api_df

                # Load metadata from saved file if we have it
                if station_name not in stations_metadata:
                    print("  Getting metadata from API...")
                    metadata = get_station_metadata(station_id)
                    if metadata:
                        stations_metadata[station_name] = metadata
                continue

            except Exception as e:
                print(f"  Error loading existing file: {e}")
                print("  Will download fresh data instead...")

        # Get station metadata
        print("\n1. Fetching station metadata...")
        metadata = get_station_metadata(station_id)
        if metadata:
            print(f"  Station: {metadata.get('name')}")
            print(f"  Location: {metadata.get('lat')}, {metadata.get('lon')}")
            print(f"  Elevation: {metadata.get('altitude')} m")
            stations_metadata[station_name] = metadata

        # Download API data
        print("\n2. Downloading data from API...")
        api_df = download_station_data(station_id, START_TIME, END_TIME)

        if api_df is not None:
            print(f"  Successfully downloaded {len(api_df)} data points")
            print(f"  Time range: {api_df.index[0]} to {api_df.index[-1]}")

            # Show sample data
            if 'p' in api_df.columns:
                print(f"  Pressure (p) range: {api_df['p'].min():.2f} - {api_df['p'].max():.2f} hPa")
            if 'tl' in api_df.columns:
                print(f"  Temperature (tl) range: {api_df['tl'].min():.2f} - {api_df['tl'].max():.2f} °C")

            stations_data[station_name] = api_df
        else:
            print(f"  Failed to download data for {station_name}")

    # Save downloaded data and metadata
    if stations_data:
        save_downloaded_data(stations_data)
        save_station_metadata(stations_metadata)

    # Reduce pressure to reference station
    print(f"\n{'=' * 70}")
    print("Reducing pressure to reference station...")
    print(f"{'=' * 70}")

    reference_station = 'Kufstein'
    stations_data_reduced = reduce_pressure_to_reference_station(stations_data, stations_metadata,
                                                                 reference_station=reference_station)

    # Create plots
    print(f"\n{'=' * 70}")
    print("Creating plots...")
    print(f"{'=' * 70}")

    save_path = os.path.join(confg.dir_PLOTS, "geosphere_api_downloaded_data.svg")
    plot_downloaded_data(stations_data_reduced, stations_metadata, save_path=save_path)

    print(f"\n{'=' * 70}")
    print("Download and processing complete!")
    print(f"{'=' * 70}")
