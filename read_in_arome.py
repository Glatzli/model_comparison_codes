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
import datetime
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
        ds['p'] = ds['p'].assign_attrs(units="hPa", description="pressure")
        if "temp" in vars_to_calc:
            # calc temp
            ds["temp"] = mpcalc.temperature_from_potential_temperature(ds["p"], ds["th"] * units("K"))

            if "rho" in vars_to_calc:  # using ideal gas law: rho [kg/m^3] = p [Pa] / (R * T [K]) with R_dryair = 287.05 J/kgK
                ds["rho"] = (ds["p"] * 100) / (287.05 * ds["temp"])
                ds["rho"] = ds['rho'].assign_attrs(units="kg/m^3", description="air density calced from p & temp (ideal gas law)")
        if "rh" in vars_to_calc:
            # not checked yet
            ds['rh'] = mpcalc.relative_humidity_from_specific_humidity(ds['p'], ds["temp"], ds['q']* units("kg/kg")) * 100  # for percent
            ds['rh'] = ds['rh'].assign_attrs(units="%", description="relative humidity calced from p, temp & q")

    # calculate dewpoint
    #ds["Td"] = mpcalc.dewpoint_from_specific_humidity(pressure = ds['pressure'],
    #                                                  specific_humidity = ds['qv']) # , temperature = ds["temp"]
    ds = ds.metpy.dequantify()
    if "temp" in vars_to_calc:
        # convert temp to °C
        ds["temp"] = ds["temp"] - 273.15
        ds["temp"] = ds['temp'].assign_attrs(units="degC", description="temperature calced from th & p")
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
    vars_to_calculate = set(variables) - set(data_vars)  # need to calculate the var's that are not in ds and are given

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
    data = data.rename({"nz": "height", "latitude": "lat", "longitude": "lon"})  # rename to uniform height coordinate
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


def read_in_arome_fixed_time(day, hour, min, variables=["p", "th", "z"], min_lat=46.5, max_lat=48.2,
                             min_lon=9.2, max_lon=13):
    """
    read arome data for a fixed time,
    by default indexes the data to the chosen box (icon)
    :param time: time as string f.e. "2017-10-15T12:00:00", you can use pd.to_datetime() to convert a string to a timestamp
    :return:
    ds of arome data with only wanted timestamp (~2GB)
    """
    ds, vars_to_calculate = read_in_arome(variables=variables)
    timestamp = datetime.datetime(2017, 10, day, hour, min, 00)
    ds = ds.sel(time=timestamp)  # select just needed timestep
    ds = ds.sel(latitude=slice(min_lat, max_lat + 0.01), longitude=slice(min_lon, max_lon + 0.01))  # include lon=13.0° & lat=48.2°

    ds = rename_vars(data=ds)
    ds = convert_calc_variables(ds, vars_to_calc=vars_to_calculate)
    ds = ds.compute()
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
    # arome = read_timeSeries_AROME(location)

    # arome3d = read_3D_variables_AROME(lon= lon_ibk, lat=lat_ibk, variables=["p", "th", "z", "rho"], method="sel")

    # arome = read_in_arome_fixed_point(lon= confg.lon_ibk, lat= confg.lat_ibk, variables=["p", "th", "temp", "rho"], method="sel")
    # right now I have for height coord. 1 at the bottom, and 90 at top, but also lowest temps, lowest p at 1...
    arome = read_in_arome_fixed_point(lat=confg.ibk_uni["lat"], lon=confg.ibk_uni["lon"], variables=["p", "temp", "th", "z", "rho"])
    # arome = read_in_arome_fixed_time(day=16, hour=12, min=0, variables=["p", "temp", "th", "z", "rho"])
    arome

    # arome_z_subset = xr.open_dataset(confg.dir_AROME + "AROME_subset_z.nc", mode="w", format="NETCDF4")
    # arome_z
    # arome_path = Path(confg.data_folder + "AROME_temp_timeseries_ibk.nc")
    # arome_path = Path(confg.model_folder + "/AROME/" + "AROME_temp_timeseries_ibk.nc")

    # arome3d_new.to_netcdf(confg.dir_3D_AROME + "/AROME_temp_timeseries_ibk.nc", mode="w", format="NETCDF4")


