"""
Utility functions for loading and plotting HOBO station observation data.

This module provides functions to:
- Load HOBO station data from netCDF files
- Extract temperature data for specific time points
- Add HOBO stations as scatter points to matplotlib plots

These functions are used by plot_heat_fluxes.py and plot_wind_temperature_horiz.py
to overlay observation data on model temperature plots.
"""
import sys

sys.path.append("C:/Users/eleme/Documents/1Uni_Laptop/model_comparison_codes")
import fix_win_DLL_loading_issue

fix_win_DLL_loading_issue

import numpy as np
import xarray as xr
import pandas as pd

import confg


def load_hobo_data():
    """
    Load HOBO station data from netCDF file.

    Returns:
        xarray.Dataset: HOBO dataset with station locations and temperature data
                       Returns None if loading fails
    """
    try:
        ds_hobo = xr.open_dataset(confg.hobos_file)
        print(f"✓ HOBO data loaded successfully")
        return ds_hobo
    except Exception as e:
        print(f"✗ Warning: Could not load HOBO data: {e}")
        return None


def load_zamg_data():
    """
    Load ZAMG station data from CSV files.

    Returns:
        dict: Dictionary with station data {station_id: {'name': ..., 'lat': ..., 'lon': ..., 'data': DataFrame}}
              Returns empty dict if loading fails
    """
    try:
        zamg_data = {}
        for station_id, station_info in confg.station_files_zamg_new.items():
            try:
                df = pd.read_csv(station_info['filepath'])
                # Parse time column
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')

                zamg_data[station_id] = {'name': station_info['name'], 'lat': station_info['lat'],
                                         'lon': station_info['lon'], 'height': station_info['hoehe'], 'data': df}
            except Exception as e:
                print(f"  Warning: Could not load ZAMG station {station_id}: {e}")
                continue

        if zamg_data:
            print(f"✓ ZAMG data loaded successfully ({len(zamg_data)} stations)")
        else:
            print(f"✗ Warning: No ZAMG data could be loaded")
        return zamg_data
    except Exception as e:
        print(f"✗ Warning: Could not load ZAMG data: {e}")
        return {}


def load_momaa_data():
    """
    Load MOMMA station data from netCDF file.

    Returns:
        xarray.Dataset: MOMMA dataset with station locations and meteorological data
                       Returns None if loading fails
    """
    try:
        ds_momaa = xr.open_dataset(confg.momma_our_period_file)
        print(f"✓ MOMMA data loaded successfully")
        return ds_momaa
    except Exception as e:
        print(f"✗ Warning: Could not load MOMMA data: {e}")
        return None


def get_hobo_temperatures_at_time(ds_hobo, time):
    """
    Extract HOBO station temperatures for a specific time. This function works now only for one HOBO station (
    Hafelekar-station is indexed already before, by a for loop through all stations it could easily extract all
    station data with the station key as dict key f.e.)

    Args:
        ds_hobo: xarray Dataset with HOBO data
        time: Datetime object for which to extract temperatures

    Returns:
        dict: Dictionary with station data including lat, lon, and temperature.
              Only includes stations with valid (non-NaN) temperature data.
              Format: {station_key: {'lat': float, 'lon': float, 'temp': float}}
    """
    if ds_hobo is None:
        return {}

    station_data = {}

    try:

        # Extract coordinates and temperature
        lat = float(ds_hobo['lat'].values)
        lon = float(ds_hobo['lon'].values)
        # Get temperature at the specified time (nearest match)
        temp = ds_hobo['ta'].sel(TIME=time, method="nearest", tolerance="5min").values.item()

        # create dict with lat, lon and temp for this station
        station_data = {'lat': lat, 'lon': lon, 'temp': temp}
    except Exception as e:
        print(f"  Warning: Error extracting HOBO data for time {time}: {e}")

    return station_data


def get_momma_temperatures_at_time(ds_momma, time):
    """
    Extract MOMMA station temperatures for a specific time.

    Args:
        ds_momma: xarray Dataset with MOMMA data
        time: Datetime object for which to extract temperatures

    Returns:
        dict: Dictionary with station data including lat, lon, and temperature.
              Only includes stations with valid (non-NaN) temperature data.
              Format: {station_key: {'lat': float, 'lon': float, 'temp': float}}
    """
    if ds_momma is None:
        return {}

    station_data = {}

    try:
        time_pd = pd.to_datetime(time)

        # Iterate through MOMMA stations
        for station_key, station_info in confg.MOMMA_stations_PM.items():
            try:
                # Select data for this station
                station = ds_momma.sel(STATION_KEY=station_info['key'])

                # Extract temperature at the specified time (nearest match)
                temp_data = station['ta'].sel(time=time_pd, method="nearest", tolerance="5min")
                temp = float(temp_data.values)

                # Only include stations with valid temperature data
                if not np.isnan(temp):
                    station_data[station_key] = {'lat': station_info['latitude'], 'lon': station_info['longitude'],
                                                 'temp': temp}
            except Exception as e:
                # Silently skip stations with no data at this time
                continue

    except Exception as e:
        print(f"  Warning: Error extracting MOMMA data for time {time}: {e}")

    return station_data


def add_station_data_to_plot(ax, ds_hobo=None, ds_momma=None, time=None, marker_size=80, edge_color='black',
        edge_width=1.5, vmin=5, vmax=25, cmap=None):
    """
    Add observation station temperature data as scatter points to an existing plot.

    Station types:
    - MOMMA: Circles (filled with temperature colors)

    Args:
        ax: Matplotlib axis object to add station data to
        ds_momma: xarray Dataset with MOMMA data
        time: Datetime object for which to plot temperatures
        marker_size: Size of scatter markers (default: 80)
        edge_color: Color of marker edge (default: 'black')
        edge_width: Width of marker edge (default: 1.5)
        vmin: Minimum temperature value for colormap (default: 5°C)
        vmax: Maximum temperature value for colormap (default: 25°C)
        cmap: Colormap to use (default: None, will use confg.temperature_colormap)

    Returns:
        dict: Dictionary with scatter objects {'hobo': scatter, 'zamg': scatter, 'momma': scatter}
              None values if no data available for that type
    """
    if cmap is None:
        cmap = confg.temperature_colormap

    scatter_objects = {}

    # Plot HOBO stations (circles)
    if ds_hobo is not None:
        hobo_data = get_hobo_temperatures_at_time(ds_hobo, time)

        # lats = [data['lat'] for data in hobo_data.values()]  # for plotting all HOBOs
        # lons = [data['lon'] for data in hobo_data.values()]
        # temps = [data['temp'] for data in hobo_data.values()]

        scatter_objects['hobo'] = ax.scatter(hobo_data['lon'], hobo_data['lat'], c=hobo_data['temp'], s=marker_size,
                                             cmap=cmap, vmin=vmin, vmax=vmax, alpha=1.0, edgecolors=edge_color,
                                             linewidth=edge_width, marker='s', zorder=10, label='HOBO')

    if ds_momma is not None:
        # Plot MOMMA stations (circles)
        momma_data = get_momma_temperatures_at_time(ds_momma, time)

        lats = [data['lat'] for data in momma_data.values()]
        lons = [data['lon'] for data in momma_data.values()]
        temps = [data['temp'] for data in momma_data.values()]

        scatter_objects['momma'] = ax.scatter(lons, lats, c=temps, s=marker_size, cmap=cmap, vmin=vmin, vmax=vmax,
                                              alpha=1.0, edgecolors=edge_color, linewidth=edge_width, marker='o',
                                              # 's' = square
                                              zorder=12, label='MOMMA')

    return scatter_objects if scatter_objects else None


def add_white_station_symbols_to_plot(ax, ds_hobo=None, ds_momma=None, marker_size=80, edge_color='black',
                                    edge_width=1.5):
    """
    Add white observation station symbols to plots (for non-temperature plots or special styling).

    Station types:
    - HOBO: White squares
    - MOMMA: White crosses

    Args:
        ax: Matplotlib axis object to add station data to
        ds_hobo: xarray Dataset with HOBO data (only Hafelekar H38)
        ds_momma: xarray Dataset with MOMMA data
        marker_size: Size of scatter markers (default: 80)
        edge_color: Color of marker edge (default: 'black')
        edge_width: Width of marker edge (default: 1.5)

    Returns:
        dict: Dictionary with scatter objects {'hobo': scatter, 'momma': scatter}
    """
    scatter_objects = {}

    # Plot HOBO station (white square) - only Hafelekar H38
    if ds_hobo is not None:
        try:
            # Get Hafelekar HOBO coordinates
            lat = float(ds_hobo.lat.values)
            lon = float(ds_hobo.lon.values)

            scatter_objects['hobo'] = ax.scatter(lon, lat, c='white', s=marker_size,
                                               alpha=1.0, edgecolors=edge_color, linewidth=edge_width,
                                               marker='s', zorder=15, label='HOBO H38')
        except Exception as e:
            print(f"Warning: Could not plot HOBO station: {e}")

    # Plot MOMMA stations (white crosses)
    if ds_momma is not None:
        try:
            # Get all MOMMA station coordinates
            lats = []
            lons = []

            for station_key in ds_momma['STATION_KEY'].values:
                station_data = ds_momma.sel(STATION_KEY=station_key)
                lat = float(station_data['lat'].values)
                lon = float(station_data['lon'].values)
                lats.append(lat)
                lons.append(lon)

            scatter_objects['momma'] = ax.scatter(lons, lats, c='white', s=marker_size,
                                                alpha=1.0, edgecolors=edge_color, linewidth=edge_width,
                                                marker='+', zorder=15, label='MOMMA')
        except Exception as e:
            print(f"Warning: Could not plot MOMMA stations: {e}")

    return scatter_objects if scatter_objects else None


def get_westernmost_hobo_station(ds_hobo):
    """
    Find the westernmost HOBO station (lowest longitude value).

    Args:
        ds_hobo: xarray Dataset with HOBO data

    Returns:
        int: Station key of the westernmost HOBO station
             Returns None if no data available
    """
    if ds_hobo is None:
        return None

    try:
        keys = list(ds_hobo['STATION_KEY'].values)
        lons = [float(ds_hobo.sel(STATION_KEY=key).lon.values) for key in keys]
        westernmost_idx = lons.index(min(lons))
        return keys[westernmost_idx]
    except Exception as e:
        print(f"✗ Warning: Could not find westernmost HOBO station: {e}")
        return None


def get_hobo_station_timeseries(ds_hobo, station_key):
    """
    Extract temperature timeseries for a specific HOBO station.

    Args:
        ds_hobo: xarray Dataset with HOBO data
        station_key: Station key identifier

    Returns:
        tuple: (times, temperatures) as pandas DatetimeIndex and numpy array
               Returns (None, None) if no data available
    """
    if ds_hobo is None:
        return None, None

    try:
        hobo_station = ds_hobo.sel(STATION_KEY=station_key)
        hobo_temps = hobo_station['ta'].values
        hobo_times = pd.to_datetime(hobo_station['TIME'].values)

        # Filter out NaN values
        valid_mask = ~np.isnan(hobo_temps)
        hobo_times = hobo_times[valid_mask]
        hobo_temps = hobo_temps[valid_mask]

        return hobo_times, hobo_temps
    except Exception as e:
        print(f"✗ Warning: Could not extract HOBO station {station_key} timeseries: {e}")
        return None, None


def add_hobo_timeseries_to_plot(ax, ds_hobo, station_key, color, linewidth=1, linestyle=':', label=None):
    """
    Add HOBO station temperature timeseries as a line to an existing plot.

    Args:
        ax: Matplotlib axis object to add HOBO timeseries to
        ds_hobo: xarray Dataset with HOBO data
        station_key: Station key identifier for the HOBO station
        color: Line color
        linewidth: Width of the line (default: 1)
        linestyle: Line style (default: ':' for dotted)
        label: Label for the line in legend (default: None, will use 'HOBO Station {key}')

    Returns:
        Line2D object or None if no data available
    """
    if ds_hobo is None:
        return None

    try:
        times, temps = get_hobo_station_timeseries(ds_hobo, station_key)

        if times is not None and len(times) > 0:
            if label is None:
                label = f'HOBO Station {station_key}'

            line = ax.plot(times, temps, color=color, linewidth=linewidth, linestyle=linestyle, label=label, zorder=9)
            return line[0]
        else:
            return None
    except Exception as e:
        print(f"✗ Warning: Could not plot HOBO station {station_key} timeseries: {e}")
        return None


def get_momma_station_timeseries(ds_momma, station_id):
    """
    Extract temperature timeseries for a specific MOMMA station.

    Args:
        ds_momma: xarray Dataset with MOMMA data
        station_id: Station identifier (e.g., "PM02")

    Returns:
        tuple: (times, temperatures) as pandas DatetimeIndex and numpy array
               Returns (None, None) if no data available
    """
    if ds_momma is None:
        return None, None

    try:
        # Get station info from confg
        if station_id not in confg.MOMMA_stations_PM:
            print(f"✗ Warning: MOMMA station {station_id} not found in confg.MOMMA_stations_PM")
            return None, None

        station_info = confg.MOMMA_stations_PM[station_id]
        station_key = station_info['key']

        # Select data for the station
        momma_station = ds_momma.sel(STATION_KEY=station_key)
        momma_temps = momma_station['ta'].values
        momma_times = pd.to_datetime(momma_station['time'].values)

        # Filter out NaN values
        valid_mask = ~np.isnan(momma_temps)
        momma_times = momma_times[valid_mask]
        momma_temps = momma_temps[valid_mask]

        return momma_times, momma_temps
    except Exception as e:
        print(f"✗ Warning: Could not extract MOMMA station {station_id} timeseries: {e}")
        return None, None


def add_momma_timeseries_to_plot(ax, ds_momma, station_id, color, linewidth=1, linestyle=':', label=None):
    """
    Add MOMMA station temperature timeseries as a line to an existing plot.

    Args:
        ax: Matplotlib axis object to add MOMMA timeseries to
        ds_momma: xarray Dataset with MOMMA data
        station_id: Station identifier (e.g., "PM02")
        color: Line color
        linewidth: Width of the line (default: 1)
        linestyle: Line style (default: ':' for dotted)
        label: Label for the line in legend (default: None, will use station name)

    Returns:
        Line2D object or None if no data available
    """
    if ds_momma is None:
        return None

    try:
        times, temps = get_momma_station_timeseries(ds_momma, station_id)

        if times is not None and len(times) > 0:
            if label is None:
                station_name = confg.MOMMA_stations_PM[station_id]['name']
                label = f'MOMMA {station_id} ({station_name})'

            line = ax.plot(times, temps, color=color, linewidth=linewidth, linestyle=linestyle, label=label, zorder=9)
            return line[0]
        else:
            return None
    except Exception as e:
        print(f"✗ Warning: Could not plot MOMMA station {station_id} timeseries: {e}")
        return None