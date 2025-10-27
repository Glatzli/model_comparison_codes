"""
re-written by Daniel
"""

import importlib

import confg

importlib.reload(confg)
import os
import pandas as pd
import xarray as xr
# import numpy as np
from metpy.units import units
import metpy.calc as mpcalc
import matplotlib
import datetime
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
    if ("wspd" or "udir") in vars_to_calc:
        # if ("u" or "v") not in ds not in ds:  # check if needed vars are in dataset: could be probably done more
        # beautiful, but it works...
        #    raise ValueError("u and v wind components not in dataset, which are required to calculate wind speed and
        #    direction.")
        ds["wspd"] = mpcalc.wind_speed(ds["u"] * units("m/s"), ds["v"] * units("m/s"))
        ds["wspd"] = ds['wspd'].assign_attrs(units="m/s", description="wind speed calced from u & v using MetPy")
        ds["udir"] = mpcalc.wind_direction(ds["u"].compute() * units("m/s"), ds["v"].compute() * units("m/s"))
        ds["udir"] = ds['udir'].assign_attrs(units="deg", description="wind direction calced from u & v using MetPy")
    if "p" in ds:
        # Convert pressure from Pa to hPa
        ds['p'] = (ds['p'] / 100.0) * units.hPa
        ds['p'] = ds['p'].assign_attrs(units="hPa", description="pressure")
    if "temp" in vars_to_calc:
        # if ("p" or "th") not in ds:  #
        #     raise ValueError("u and v wind components not in dataset, which are required to calculate wind speed
        #     and direction.")
        # calc temp
        ds["temp"] = mpcalc.temperature_from_potential_temperature(ds["p"], ds["th"] * units("K"))
    
    if "rho" in vars_to_calc:  # using ideal gas law: rho [kg/m^3] = p [Pa] / (R * T [K]) with R_dryair = 287.05 J/kgK
        # if ("p" or "temp") not in ds:
        #     raise ValueError("p and temp not in dataset, which are required to calculate air density rho.")
        ds["rho"] = (ds["p"] * 100) / (287.05 * ds["temp"])
        ds["rho"] = ds['rho'].assign_attrs(units="kg/m^3",
                                           description="air density calced from p & temp (ideal gas law)")
    if ("p" and "T" and "qv") in vars_to_calc:
        ds["Td"] = mpcalc.dewpoint_from_specific_humidity(ds["p"] * units.hPa, ds["T"] * units.degC, ds["qv"])
        ds["Td"] = ds['Td'].assign_attrs(units="degC",
                                         description="dewpoint Temp calculated from p, T & qv using MetPy")
    # if "rh" in vars_to_calc:
    #     if ("p" or "temp" or "q") not in ds:
    #         raise ValueError("p, temp and q not in dataset, which are required to calculate relative humidity rh.")
    # not checked yet
    #     ds['rh'] = mpcalc.relative_humidity_from_specific_humidity(ds['p'], ds["temp"], ds['q']* units("kg/kg")) *
    #     100  # for percent
    #    ds['rh'] = ds['rh'].assign_attrs(units="%", description="relative humidity calced from p, temp & q")
    
    ds = ds.metpy.dequantify()
    if "temp" in vars_to_calc:
        # convert temp to °C
        ds["temp"] = ds["temp"] - 273.15
        ds["temp"] = ds['temp'].assign_attrs(units="degC", description="temperature calced from th & p")
    return ds


def check_add_needed_variables(variables, vars_to_calculate):
    """
    checks if needed variables for calculation are in the variable list, if not adds them
    :param variables: list of variables to read in
    :param vars_to_calculate: list of variables to calculate
    :return: updated variable list
    """
    # check if needed variables are in dataset, if not add them:
    # f.e. wspd or wind dir should be calculated, I need u & v: add that...
    variables = list(set(variables) | {"u", "v"}) if (
                "udir" in vars_to_calculate or "wspd" in vars_to_calculate) else variables
    variables = list(set(variables) | {"p", "th"}) if ("temp" in vars_to_calculate) else variables
    variables = list(set(variables) | {"p", "temp"}) if ("rho" in vars_to_calculate) else variables
    variables = list(set(variables) | {"T", "p", "qv"}) if "Td" in vars_to_calculate else variables
    return variables


def read_in_arome(variables=["p", "th", "z"]):
    """
    reads in all arome data, all vars that are given that are not saved as files are calculated later

    (fast) by Daniel
    exception for "rho", which is calculated later
    :param variables: list of variables to read in, possible are: ["ciwc", "clwc", "p", "q", "th", "tke", "u", "v",
    "w", "z"]
        and ["rho", "temp"]
    :return: ds with all variables in the list
    """
    data_vars = ["ciwc", "clwc", "p", "q", "th", "tke", "u", "v", "w", "z"]  # saved file vars
    vars_to_calculate = set(variables) - set(data_vars)  # need to calculate the var's that are not in ds and are given
    
    variables = check_add_needed_variables(variables, vars_to_calculate)
    
    arome_paths = [confg.dir_3D_AROME + f"/AROME_Geosphere_20171015T1200Z_CAP02_3D_30min_1km_best_{var}.nc" for var in
                   variables if
                   var in data_vars]  # only read in variables that are saved as files, others need to be calc.
    ds = xr.open_mfdataset(arome_paths, combine="by_coords", data_vars='minimal', coords='minimal', compat='override',
                           decode_timedelta=True)
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


def read_in_arome_fixed_point(lat=47.259998, lon=11.384167, method="sel", variables=["p", "th", "z"],
                              height_as_z_coord=False):  # , variable_list=
    """
    Read the AROME model output for a fixed point at a specific location with full time range.
    The method can be 'sel' or 'interp' for selecting the nearest point or interpolating to the point.

    :param lat: Latitude of the fixed point.
    :param lon: Longitude of the fixed point.
    :param method: Selection method of point ('sel' or 'interp').
    :param variables: List of variables to include in the dataset ["ciwc", "clwc", "p", "q", "th", "tke", "u", "v",
    "w", "z", "rho"]
        if "rho" is needed, it will be calculated using ideal gas law in "convert_calc_variables"
    :param: height_as_z_coord: sets mean geopot. height as the coordinate in z (which simplifies some computations)
    :return: Merged xarray Dataset
    """
    ds, vars_to_calculate = read_in_arome(variables=variables)
    if method == "interp":  # interpolate to point, uses numpy/scipy interp routines...
        ds = ds.interp(latitude=lat, longitude=lon)
    elif method == "sel":  # selects nearest point
        ds = ds.sel(latitude=lat, longitude=lon, method="nearest")
    
    ds = rename_vars(data=ds)
    ds = convert_calc_variables(ds, vars_to_calc=vars_to_calculate)
    if height_as_z_coord:  # take mean over all geopot. height vars (skip first 2 hours due to possible model init.
        # issues)
        if not "z" in ds:
            raise ValueError("z (geopotential height) not in dataset, can't set height as z coordinate.")
        ds["height"] = ds.z.isel(time=slice(4, 100)).mean(dim="time").values
    ds = ds.compute()
    return ds


def read_in_arome_fixed_time(day, hour, min, variables=["p", "th", "z"], min_lat=46.5,
                             max_lat=48.2, min_lon=9.2, max_lon=13):
    """
    read arome data for a fixed time,
    by default indexes the data to the chosen box (icon)
    :param day:
    :param hour:
    :param min:
    :param variables: List of variables to include in the dataset ["p", "th"]
    :param min_lat, max_lat, min_lon, max_lon: per default select the defined subset (for which also ICON was
    regridded...)
    :return: ds with all variables in the list
    """
    ds, vars_to_calculate = read_in_arome(variables=variables)
    timestamp = datetime.datetime(2017, 10, day, hour, min, 00)
    ds = ds.sel(time=timestamp)  # select just needed timestep
    ds = ds.sel(latitude=slice(min_lat, max_lat + 0.01),
                longitude=slice(min_lon, max_lon + 0.01))  # include lon=13.0° & lat=48.2°
    
    ds = rename_vars(data=ds)
    ds = convert_calc_variables(ds, vars_to_calc=vars_to_calculate)
    
    ds = ds.compute()
    return ds


def read_2D_variables_AROME(variableList, lon, lat, slice_lat_lon=False):
    """ function from Hannes, with sel-method
    Read all the 2D variables (single netcdf per variable) and merge them
    hfs: sensible heat flux is vice versa than in WRF -> invert sign!?

    :param variableList: List of the selected variables
    :param lon: Longitude of the MOMAA station
    :param lat: Latitude of the MOMAA station
    :param slice_lat_lon: True: lon & lat are slices, if False not, Method for selecting latitude and longitude
    :return: Merged DataFrame with all the variables
    """
    datasets = []  # List to hold the interpolated datasets for each variable
    
    for variable in variableList:
        file_path = os.path.join(confg.dir_2D_AROME,
                                 f"AROME_Geosphere_20171015T1200Z_CAP02_2D_30min_1km_best_{variable}.nc")
        
        ds = xr.open_dataset(file_path)
        if not variable == "u_v_from_3d":  # create exception for afterwards saved u & v from 3D vars:
            ds = ds.rename({"latitude": "lat", "longitude": "lon"})  # they already have short lat & lon names!
        # Use no method if lat or lon are slice objects
        if slice_lat_lon:
            ds = ds.sel(lon=lon, lat=lat).isel(time=slice(4, None))  # , method="nearest"
        else:
            ds = ds.sel(lon=lon, lat=lat, method="nearest").isel(time=slice(4, None))
        
        for var, units in variables_units_2D_AROME.items():
            if var in ds:
                ds[var].attrs['units'] = units
        
        # ds_quantified = ds.metpy.quantify()
        datasets.append(ds)
    data = xr.merge(datasets, join="override")  # former: join="exact"
    # downgrade coords to float32, for uniformity with 3D data! anyways not needed that precise...
    data = data.assign_coords(lat=data.lat.astype("float32"), lon=data.lon.astype("float32"))
    if "hfs" in data:
        data["hfs"] = -data["hfs"]  # invert sign of sensible heat flux to be consistent with WRF
        data["hfs"].attrs['description'] = "sensible heat flux, positive: heat transport toward surface"
    
    return data


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
    arome.rename({"lat": "y", "lon": "x"}).rio.to_raster(confg.dir_AROME + "AROME_geopot_height_3dlowest_level.tif")


def extract_3d_variable_define_2D(variables=["u", "v"]):
    """
    extract 3D variables and save it like 2D variables (espc needed for heat flux plotting to avoid reading 3D
    files again and again...)
    :param arome3d:
    :return:
    """
    timerange = pd.date_range("2017-10-15 12:00:00", periods=49, freq="30min")
    timestamp_ds = []
    for timestamp in timerange:
        arome = read_in_arome_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
                                         variables=variables)  # subset lat/lon per default
        timestamp_ds.append(arome.sel(height=1))
    
    arome2d_new = xr.concat(timestamp_ds, dim="time")
    # arome2d_new = arome2d_new.assign_coords(lat=arome2d_new.lat.astype("float32"), lon=arome2d_new.lon.astype(
    # "float32"))
    # problem: 2D data have float 32 lat/lon coords, 3D have float 64
    # need to chang
    
    arome2d_new.to_netcdf(
        confg.dir_2D_AROME + f"/AROME_Geosphere_20171015T1200Z_CAP02_2D_30min_1km_best_{variables}_float32.nc")


if __name__ == '__main__':
    # arome = read_timeSeries_AROME(location)
    # arome3d = read_3D_variables_AROME(lon= lon_ibk, lat=lat_ibk, variables=["p", "th", "z", "rho"], method="sel")
    
    # right now I have for height coord. 1 at the bottom, and 90 at top, but also lowest temps, lowest p at 1...
    # arome = read_in_arome_fixed_point(lat=confg.ibk_uni["lat"], lon=confg.ibk_uni["lon"],
    #                                   variables=["u", "v", "udir", "wspd", "z"],
    #                                   height_as_z_coord=True)  # ["p", "temp", "th", "z", "udir", "wspd"]
    arome = read_in_arome_fixed_time(day=16, hour=12, min=0, variables=["p", "temp", "th", "z"], height_as_z_coord=True)
    arome
    
    # arome2d = read_2D_variables_AROME(variableList=["hfs", "hgt", "lfs", "lwnet", "lwu", "swd", "swnet"],
    #                                   lon=slice(confg.lon_hf_min, confg.lon_hf_max),
    #                                   lat=slice(confg.lat_hf_min, confg.lat_hf_max), slice_lat_lon=True)
    # arome2d
    
    # arome_z_subset = xr.open_dataset(confg.dir_AROME + "AROME_subset_z.nc", mode="w", format="NETCDF4")
    # arome_z
    # arome_path = Path(confg.data_folder + "AROME_temp_timeseries_ibk.nc")
    # arome_path = Path(confg.model_folder + "/AROME/" + "AROME_temp_timeseries_ibk.nc")
    
    # arome3d_new.to_netcdf(confg.dir_3D_AROME + "/AROME_temp_timeseries_ibk.nc", mode="w", format="NETCDF4")
    
    extract_3d_variable_define_2D(variables=["hfs"])  # used to save u & v like 2D variables
