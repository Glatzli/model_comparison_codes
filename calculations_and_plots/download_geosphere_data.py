"""


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
import fix_win_DLL_loading_issue
import json
import os

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

R = 287.05  # Specific gas constant for dry air [J/(kg·K)]
g = 9.80665  # Standard gravity [m/s²]


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
    params = {'parameters': ','.join(parameters), 'station_ids': station_id, # Station ID as query parameter
              'start': start_time, 'end': end_time, 'output_format': 'csv'# Changed from geojson to csv
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
            print("  [OK] Data matches well (max difference < 0.1 hPa)")
        elif diff.abs().max() < 1.0:
            print("  [WARNING] Data has small differences (max difference < 1.0 hPa)")
        else:
            print("  [ERROR] Data has significant differences!")


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
            # station_elevation = confg.ALL_POINTS["ibk_uni"]["height"]  # this is HATPRO height!
            station_elevation = 609.5  # m pressure height is at 609.5m (https://acinn-data.uibk.ac.at/pages/tawes-uibk.html)
        else:
            station_elevation = stations_metadata[station_name]['altitude']
        elevation_diff = station_elevation - ref_elevation  # Positive if station is higher

        print(f"  {station_name}: {station_elevation} m -> {ref_elevation} m (dh = {elevation_diff:.1f} m)")

        # Create copy of DataFrame
        df_reduced = df.copy()

        if 'p' in df.columns and 'tl' in df.columns:
            # Barometric formula: P_reduced = P_station * exp((g * dh) / (R * T))
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
            print(f"    Warning: No temperature data for {station_name}")  # , using standard atmosphere (T=15°C)
            continue
            T_standard = 288.15  # 15°C in Kelvin
            pressure_factor = np.exp((g * elevation_diff) / (R * T_standard))
            df_reduced['p_reduced'] = df['p'] * pressure_factor

        else:
            print(f"    Warning: No pressure data for {station_name}")

        stations_data_reduced[station_name] = df_reduced

    return stations_data_reduced


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


def load_or_download_all_stations(start_time=START_TIME, end_time=END_TIME):
    """
    Load all stations from CSV files, or download from API if files don't exist.

    This is the main function to use - it handles everything in one call (DRY principle).

    Parameters
    ----------
    start_time : str, optional
        Start time in ISO format (default: START_TIME)
    end_time : str, optional
        End time in ISO format (default: END_TIME)

    Returns
    -------
    tuple
        (stations_data, stations_metadata) where:
        - stations_data: dict with station names as keys and DataFrames as values
        - stations_metadata: dict with station metadata
    """
    stations_data = {}
    stations_metadata = load_station_metadata()

    for station_code, station_info in confg.station_files_zamg_new.items():
        station_name = station_info['name']
        file_path = station_info.get('filepath')

        # Try to load from CSV first
        if file_path and os.path.exists(file_path):
            print(f"\nLoading {station_name} from {file_path}")
            try:
                df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
                stations_data[station_name] = df
                print(f"  [OK] Loaded {len(df)} data points")

                # Load metadata if not already present
                if station_name not in stations_metadata:
                    if station_code in STATION_ID_MAPPING:
                        metadata = get_station_metadata(STATION_ID_MAPPING[station_code])
                        if metadata:
                            stations_metadata[station_name] = metadata
                continue
            except Exception as e:
                print(f"  [ERROR] Error loading data: {e}")
                print("  Will try downloading from API...")

        # File doesn't exist or failed to load - download from API
        print(f"\n[WARNING] File not found for {station_name}: {file_path}")
        print(f"  Downloading data from Geosphere API...")

        if station_code not in STATION_ID_MAPPING:
            print(f"  [ERROR] Station code {station_code} not in API mapping, skipping")
            continue

        station_id = STATION_ID_MAPPING[station_code]

        # Get metadata
        if station_name not in stations_metadata:
            metadata = get_station_metadata(station_id)
            if metadata:
                stations_metadata[station_name] = metadata
                print(f"  [OK] Retrieved metadata for {station_name}")

        # Download data
        try:
            df = download_station_data(station_id, start_time, end_time)
            if df is not None:
                stations_data[station_name] = df
                print(f"  [OK] Downloaded {len(df)} data points")

                # Save for future use
                print(f"  Saving to {file_path}...")
                save_downloaded_data({station_name: df})
                print(f"  [OK] Data saved")
            else:
                print(f"  [ERROR] Failed to download data for {station_name}")
        except Exception as e:
            print(f"  [ERROR] Error downloading data: {e}")

    # Save updated metadata
    if stations_metadata:
        save_station_metadata(stations_metadata)

    return stations_data, stations_metadata


def reduce_model_pressure_to_reference_station(model_data, stations_metadata, reference_station='Kufstein'):
    """
    Reduce pressure from all model data to the reference station elevation using barometric formula.

    Uses the lowest model level (height variable) as reference height for each model.

    Parameters
    ----------
    model_data : dict
        Dictionary with point names as keys and model dictionaries as values
        {point_name: {model: xr.Dataset}}
    stations_metadata : dict
        Dictionary with station metadata including heights
    reference_station : str
        Name of reference station (default: 'Kufstein')

    Returns
    -------
    dict
        Dictionary with reduced model pressure data
    """
    import numpy as np

    # Get reference station elevation
    if reference_station not in stations_metadata:
        print(f"Warning: Reference station '{reference_station}' not found in metadata!")
        return model_data

    ref_elevation = stations_metadata[reference_station]['altitude']
    print(f"\nReducing model pressures to {reference_station} elevation: {ref_elevation} m")

    model_data_reduced = {}

    for point_name, models in model_data.items():
        model_data_reduced[point_name] = {}

        for model_name, ds in models.items():
            if model_name == "WRF":
                # Skip WRF due to no time dimension for pressure
                print(f"  Skipping {model_name} - no time dimension for pressure")
                continue

            if 'p' not in ds.variables or 'height' not in ds.variables or 'temp' not in ds.variables:
                print(f"  Warning: Missing required variables for {model_name} at {point_name}")
                model_data_reduced[point_name][model_name] = ds
                continue

            # Get the lowest model level (surface level)
            model_height = float(ds['height'].sel(height=0, method="nearest").values)

            # Calculate elevation difference
            elevation_diff = model_height - ref_elevation  # Positive if model level is higher
            print(
                f"  {model_name} at {point_name}: {model_height:.1f} m -> {ref_elevation} m (dh = {elevation_diff:.1f} m)")

            # Create a copy of the dataset
            ds_reduced = ds.copy()
            # Get pressure and temperature at lowest level
            p_model_surface = ds['p'].sel(height=0, method="nearest") * 100  # Surface pressure in Pa
            temp_model_surface = ds['temp'].sel(height=0, method="nearest") + 273.15  # Surface temperature in K

            # Barometric formula: P_reduced = P_model * exp((g * dh) / (R * T))
            p_reduced_hpa = (p_model_surface * np.exp((g * elevation_diff) / (R * temp_model_surface))) / 100

            # Convert back to Pa and store in dataset
            ds_reduced['p_reduced'] = (['time'], p_reduced_hpa.values)
            ds_reduced['p_reduced'].attrs['units'] = 'Pa'
            ds_reduced['p_reduced'].attrs['long_name'] = f'Pressure reduced to {reference_station} elevation'

            # Show statistics
            original_mean = float(p_model_surface.mean() / 100)
            reduced_mean = float(p_reduced_hpa.mean())
            print(f"    mean Original pressure: {original_mean:.2f} hPa")
            print(f"    mean Reduced pressure:  {reduced_mean:.2f} hPa")
            print(f"    Mean difference:   {reduced_mean - original_mean:.2f} hPa")

            model_data_reduced[point_name][model_name] = ds_reduced

    return model_data_reduced


if __name__ == "__main__":
    """
    Main function to download Geosphere data via API.
    Simply calls load_or_download_all_stations() to do all the work.
    """
    print("=" * 70)
    print("Geosphere Austria API Data Download")
    print("=" * 70)

    # Use the unified function (DRY principle)
    stations_data, stations_metadata = load_or_download_all_stations()

    print(f"\n{'=' * 70}")
    print("Download and processing complete!")
    print(f"{'=' * 70}")
    print(f"Loaded/downloaded {len(stations_data)} stations")