"""
re-written by Daniel
"""


import confg
import importlib
importlib.reload(confg)
import os
import pandas as pd
import xarray as xr
import glob
# import numpy as np
from metpy.units import units
import metpy
import metpy.calc as mpcalc
import tarfile
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

from confg import variables_units_2D_AROME

matplotlib.use('Qt5Agg')



def convert_calc_variables(ds, vars_to_calc=["temp", "rh", "rho"]):
    """
    Converts and calculates meteorological variables for a xarray Dataset.
    by Daniel
    cal
    Idea: calculate variables which are wanted with the vars_to_calc list...

    :param ds: arome dataset
    :param vars_to_calc: list of variables to calculate, possible are: ["temp", "rh", "rho", "qv"], attention:
        there is no check if all needed vars are there for the calculation!
    return:
    :ds: xarray Dataset with the calculated variables: p, temp [°C], rH [%] ...
    """
    if "p" in ds:
        # Convert pressure from Pa to hPa
        ds['p'] = (ds['p'] / 100.0) * units.hPa
        if "temp" in vars_to_calc:
            # calc temp
            ds["temp"] = mpcalc.temperature_from_potential_temperature(ds["p"], ds["th"] * units("K"))

            if "rho" in vars_to_calc:  # using ideal gas law: rho = p / (R * T) with R_dryair = 287.05 J/kgK
                ds["rho"] = (ds["p"] * 100) / (287.05 * ds["temp"])
        if "rh" in vars_to_calc:
            # calculate relative humidity only if it's loaded in the dataset
            ds['rh'] = mpcalc.relative_humidity_from_specific_humidity(ds['p'], ds["temp"], ds['q']* units("kg/kg")) * 100  # for percent


    # calculate dewpoint
    #ds["Td"] = mpcalc.dewpoint_from_specific_humidity(pressure = ds['pressure'],
    #                                                  specific_humidity = ds['qv']) # , temperature = ds["temp"]
    ds = ds.metpy.dequantify()
    if "temp" in vars_to_calc:
        # convert temp to °C
        ds["temp"] = ds["temp"] - 273.15
    return ds


def create_ds_geopot_height_as_z_coordinate(ds):
    """
    create a new dataset with geopotential height as vertical coordinate for temperature for plotting
    :param ds:
    :return:
    :ds_new: new dataset with geopotential height as vertical coordinate
    """
    geopot_height = ds.z.isel(time=20).compute()

    ds_new = xr.Dataset(  # somehow lat & lon doesn't work => w/o those coords
        data_vars=dict(
            th=(["time", "height"], ds.th.values),
            p=(["time", "height"], ds.p.values),
        ),
        coords=dict(
            height=("height", geopot_height.values),
            time=("time", ds.time.values)
        ),
        attrs=dict(description="AROME data with geopotential height at mid of ds as vertical coordinate"))

    return ds_new



# those 3 functions are old one used by hannes, not used now!
def read_timeSeries_AROME(location):
    """The Timeseries is a direct AROME model output which holds all variables (see Data_structure.md) for a specific
    location -> interpolated to 2m(?), only lowest level!
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

def read_2D_variables_AROME(lon, lat, variableList=["hfs", "hgt", "lfs", "lwd"], slice_lat_lon=False):
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
    ancient from hannes
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

    for i, var in enumerate(variables):
        if var != "rho":
            # Construct the file path and open the dataset except for rho, which is calculated later
            file_path = os.path.join(confg.dir_3D_AROME, f"AROME_Geosphere_20171015T1200Z_CAP02_3D_30min_1km_best_{var}.nc")
            ds = xr.open_dataset(file_path)

        if time is None:  # if no time is given, read full timerange
            time_start = pd.to_datetime('2017-10-15 12:00:00',
                                        format='%Y-%m-%d %H:%M:%S')
            time_end = pd.to_datetime('2017-10-16 12:00:00',
                                      format='%Y-%m-%d %H:%M:%S')
            time = pd.date_range(start=time_start, end=time_end, freq='30min')

        # select point in space in domain either through interpolation or nearest point
        if method == "interp":
            ds_selected = ds.interp(longitude=lon, latitude=lat)
        elif method == "sel":
            ds_selected = ds.sel(longitude=lon, latitude=lat, method="nearest")
        # shorter, used by Daniel (Hannes had a lot of checks for lat lon time etc -> I will make seperate function for that)

        # Update variable units
        # for variable, units in confg.variables_units_3D_AROME.items():
        #    if variable in ds_selected:
        #        ds_selected[variable].attrs['units'] = units

        datasets.append(ds_selected)

    # Merge all datasets
    ds = xr.merge(datasets, join="exact")
    ds = ds.isel(nz=slice(None, None, -1))  # reverse nz axis to have uniform 0 at ground level!
    ds = create_ds_geopot_height_as_z_coordinate(ds)  # create new dataset with geopotential height as vertical coordinate
    ds = convert_calc_variables(ds)
    return ds


def read_in_arome(variables=["p", "th", "z"]):
    """
    reads in all arome data, all vars that are given that are not saved as files are calculated later

    (fast) by Daniel
    exception for "rho", which is calculated later
    :param variables: list of variables to read in, possible are: ["ciwc", "clwc", "p", "q", "th", "tke", "u", "v", "w", "z"]
        and ["rho", "temp"]
    :return: ds with all variables in the list
    """
    data_vars = ["ciwc", "clwc", "p", "q", "th", "tke", "u", "v", "w", "z"]  # saved file vars
    vars_to_calculate = set(variables) - set(data_vars)
    # vars_to_read = set(variables) & set(data_vars)
    """if {"temp", "q"} & vars_to_calculate and "th" not in vars_to_read:
        variables.append("th")
    if "p" in vars_to_read:
        
    if "rh" in vars_to_calculate and "q" not in vars_to_read:
        variables.append("q")"""

    arome_paths = [confg.dir_3D_AROME + f"/AROME_Geosphere_20171015T1200Z_CAP02_3D_30min_1km_best_{var}.nc" for var in
                   variables if var in data_vars]  # only read in variables that are saved as files, others need to be calc.
    ds = xr.open_mfdataset(arome_paths, combine="by_coords", data_vars='minimal',
                           coords='minimal', compat='override', decode_timedelta=True)
    return ds, vars_to_calculate

def rename_vars(data):
    """
    Rename the 'nz' coordinate to 'height' and reverse the height axis to have uniform 0 at ground level.
    :param ds: arome dataset
    :return: edited arome ds
    """
    data = data.rename({"nz": "height"})  # rename to uniform height coordinate
    data = data.assign_coords(height=data.height.values[::-1])
    return data


def read_in_arome_fixed_point(lat=47.259998, lon=11.384167, method="sel", variables=["p", "th", "z"]):  # , variable_list=
    """
    Read the AROME model output for a fixed point at a specific location with full time range.
    The method can be 'sel' or 'interp' for selecting the nearest point or interpolating to the point.

    :param lat: Latitude of the fixed point.
    :param lon: Longitude of the fixed point.
    :param method: Selection method of point ('sel' or 'interp').
    :param variables: List of variables to include in the dataset ["ciwc", "clwc", "p", "q", "th", "tke", "u", "v", "w", "z", "rho"]
        if "rho" is needed, it will be calculated using ideal gas law in "convert_calc_variables"
    :return: Merged xarray Dataset
    """
    ds, vars_to_calculate = read_in_arome(variables=variables)
    if method == "interp":  # interpolate to point, uses numpy/scipy interp routines...
        ds = ds.interp(latitude=lat, longitude=lon)
    elif method == "sel":   # selects nearest point
        ds = ds.sel(latitude=lat, longitude=lon, method="nearest")

    ds = rename_vars(data=ds)
    ds = convert_calc_variables(ds, vars_to_calc=vars_to_calculate)
    ds = ds.compute()
    return ds

def read_in_arome_fixed_time(time="2017-10-15T14:00:00", variables=["p", "th", "z"]):
    """
    read arome data for a fixed time
    :param time: time as string f.e. "2017-10-15T12:00:00", you can use pd.to_datetime() to convert a string to a timestamp
    :return:
    ds of arome data with only wanted timestamp (~2GB)
    """
    ds, vars_to_calculate = read_in_arome(variables=variables)
    ds = ds.sel(time=time)  # select just needed timestep

    ds = rename_vars(data=ds)
    ds = convert_calc_variables(ds, vars_to_calc=vars_to_calculate)
    return ds


def save_arome_topography(arome3d):
    """
    saves the geopotential height of the lowest level of AROME as a 2D .netcdf file for topography plotting and as .tif
    file for aspect calculation with xdem (for PCGP)...
    :return:
    """
    arome = arome3d.isel(height=0).compute()
    arome = arome.rename({"latitude": "lat", "longitude": "lon"})  # rename to uniform z coordinate
    arome.to_netcdf(confg.dir_AROME + "AROME_geopot_height_3dlowest_level.nc", mode="w", format="NETCDF4")

    # round coords due to strange error during slope calc..
    arome["lat"] = arome["lat"].round(4)
    arome["lon"] = arome["lon"].round(4)
    # rename coords for xdem/rasterio compatibility, for slope and aspect calculation
    arome.rename({"lat":"y", "lon":"x"}).rio.to_raster(confg.dir_AROME + "AROME_geopot_height_3dlowest_level.tif")


if __name__ == '__main__':
    lat_ibk = 47.259998
    lon_ibk = 11.384167
    # arome = read_timeSeries_AROME(location)

    # arome3d = read_3D_variables_AROME(lon= lon_ibk, lat=lat_ibk, variables=["p", "th", "z", "rho"], method="sel")

    arome = read_in_arome_fixed_point(lon= lon_ibk, lat= lat_ibk, variables=["p", "th", "temp", "rho"], method="sel")
    # right now I have for height coord. 1 at the bottom, and 90 at top, but also lowest temps, lowest p at 1...
    # arome = read_in_arome_fixed_time(time="2017-10-15T12:00:00", variables=["temp", "th", "p", "rho"])
    arome = arome.compute()
    arome

    # arome_path = Path(confg.data_folder + "AROME_temp_timeseries_ibk.nc")
    # arome_path = Path(confg.model_folder + "/AROME/" + "AROME_temp_timeseries_ibk.nc")

    # arome3d_new# .to_netcdf(confg.dir_3D_AROME + "/AROME_temp_timeseries_ibk.nc", mode="w", format="NETCDF4")
    # read_2D_variables_AROME(lon, lat, variableList=["hfs", "hgt", "lfs", "lwd"], slice_lat_lon=False)

