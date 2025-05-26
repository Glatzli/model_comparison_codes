
import confg
import os
import pandas as pd
import xarray as xr
import glob
# import numpy as np
from metpy.units import units
import metpy



def read_timeSeries_AROME(location):
    """The Timeseries is a direct AROME model output which holds all variables (see Data_structure.md) for a specific
    location
    ::param location: is the selected location
    """
    pattern = f"AROME_Geosphere_20171015T1200Z_{location}_timeseries_40s_*.nc"
    final_path_pattern = os.path.join(confg.dir_timeseries_AROME, pattern)

    # Use glob to find files that match the pattern
    matching_files = glob.glob(final_path_pattern)

    # Assuming there's only one match per location, open the dataset
    if matching_files:
        return xr.open_dataset(matching_files[0])
    else:
        raise FileNotFoundError(f"No files found for location {location}")


def read_2D_variables_AROME(variableList, lon, lat, slice_lat_lon=False):
    """ WITH the sel Method
    Read all the 2D variables (single netcdf per variable) and merge them

    :param variableList: List of the selected variables
    :param lon: Longitude of the MOMAA station
    :param lat: Latitude of the MOMAA station
    :param slice_lat_lon: Method for selecting latitude and longitude ('nearest' for nearest neighbor, None for exact match)
    :return: Merged DataFrame with all the variables
    """
    datasets = []  # List to hold the interpolated datasets for each variable

    for variable in variableList:
        file_path = os.path.join(confg.dir_2D_AROME, f"AROME_Geosphere_20171015T1200Z_CAP02_2D_30min_1km_best_{variable}.nc")

        ds = xr.open_dataset(file_path)

        # Use no method if lat or lon are slice objects
        if slice_lat_lon:
            ds = ds.sel(longitude=lon, latitude=lat).isel(time=slice(4, None))
        else:
            ds = ds.sel(longitude=lon, latitude=lat, method="nearest").isel(time=slice(4, None))

        for var, units in confg.variables_units_2D_AROME.items():
            if var in ds:
                ds[var].attrs['units'] = units

        ds_quantified = ds.metpy.quantify()
        datasets.append(ds_quantified)

    return xr.merge(datasets, join="exact")


def read_3D_variables_AROME(variables, method, lon, lat, slice_lat_lon=False, level=None, time=None):
    """
    Merge datasets for a list of variables at a specific location and time.
    The (lat, lon, time) parameters can also be arrays, e.g., [10, 12, 13].

    :param variables: List of variable names to include in the final merged dataset.
    :param method: Selection method ('sel' or 'interp') for data points.
    :param level: optional nz coordinate for data selection.
    :param lon: Longitude coordinate for data selection.
    :param lat: Latitude coordinate for data selection.
    :param time: Optional time (is set from 4 to None) assuming it starts at 12:00
    :param slice_lat_lon: default False, says if it is a slice object or not
    :return: Merged xarray Dataset for the specified variables, location, and time.
    """
    datasets = []  # List to hold datasets for each variable

    for var in variables:
        # Construct the file path and open the dataset
        file_path = os.path.join(confg.dir_3D_AROME, f"AROME_Geosphere_20171015T1200Z_CAP02_3D_30min_1km_best_{var}.nc")
        ds = xr.open_dataset(file_path)

        if time is None:
            time_start = pd.to_datetime('2017-10-15 14:00:00',
                                        format='%Y-%m-%d %H:%M:%S')
            time_end = pd.to_datetime('2017-10-16 12:00:00',
                                      format='%Y-%m-%d %H:%M:%S')
            time = pd.date_range(start=time_start, end=time_end, freq='30min')

        # Select or interpolate the dataset based on the method
        ds_selected = ds.isel(x=398, y=215)

        # Update variable units
        for variable, units in confg.variables_units_3D_AROME.items():
            if variable in ds_selected:
                ds_selected[variable].attrs['units'] = units

        # Quantify the dataset and append to the list
        datasets.append(ds_selected.metpy.quantify())

    # Merge all datasets
    return xr.merge(datasets, join="exact")

def read_in_arome(time, method, lon, lat):
    """plot the MODEL output of AROME as it would be a Radiosonde"""
    my_variable_list = ["p", "q", "th", "u", "v", "z"]

    if (method == "sel") | (method == "interp"):
        print(f"Your selected method is {method}")
    else:
        raise AttributeError(
            "You have to define a method (sel or interp) how the point near the LOWI should be selected")

    df_final = read_3D_variables_AROME(variables=my_variable_list, method=method, lon=lon, lat=lat, time=time)

    print(df_final["p"].metpy.unit_array.magnitude)
    #Extract values
    df_final["temperature"] = metpy.calc.temperature_from_potential_temperature(df_final["p"], df_final["th"])
    p = df_final["p"].metpy.unit_array.to(units.hPa)  # Metadata is removed
    T = df_final["temperature"].metpy.unit_array.to(units.degC)

    ds = xr.Dataset()

    return df_final