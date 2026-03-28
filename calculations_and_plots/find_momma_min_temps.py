"""
Script to find the minimum temperature for each STATION_KEY in the MOMMA dataset.
"""
import fix_win_DLL_loading_issue
fix_win_DLL_loading_issue
import numpy as np
import xarray as xr
import sys
import os

# Add the project root to path so we can import confg
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import confg

# Load the MOMMA dataset
ds_momma = xr.open_dataset(confg.momma_our_period_file)

# Find minimum temperature for each station
for station_idx in range(ds_momma.dims['STATION_KEY']):
    station_data = ds_momma.isel(STATION_KEY=station_idx)
    temp_data = station_data['ta'].values
    valid_temps = temp_data[~np.isnan(temp_data)]

    if len(valid_temps) > 0:
        min_temp = np.min(valid_temps)

        # Find station name
        station_name = "Unknown"
        for pm_id, station_info in confg.MOMMA_stations_PM.items():
            if station_info['key'] == station_idx:
                station_name = f"{pm_id} ({station_info['name']})"
                break

        print(f"{station_name}: {min_temp:.2f}°C")

