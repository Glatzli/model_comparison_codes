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



def convert_calc_variables(ds):
    """
    Converts and calculates meteorological variables for a xarray Dataset.
    by Daniel
    Idea: calculate only variables which are possible with read ds

    Parameters:
    - df: A xarray Dataset containing the columns 'p' for pressure in Pa
          and 'th' for potential temperature in Kelvin.

    Returns:
    - A xarray Dataset with the original data and new columns:
      'pressure' in hPa and 'temperature' in degrees Celsius.
    """
    if "p" in ds:
        # Convert pressure from Pa to hPa
        ds['pressure'] = (ds['p'] / 100.0) * units.hPa
        if "th" in ds:
            # calc temp
            ds["temperature"] = mpcalc.temperature_from_potential_temperature(ds["pressure"], ds["th"] * units("K"))
            if "q" in ds:
                # calculate relative humidity only if it's loaded in the dataset
                ds['rh'] = mpcalc.relative_humidity_from_specific_humidity(ds['pressure'], ds["temperature"], ds['q']* units("kg/kg")) * 100  # for percent

    # calculate dewpoint
    #ds["Td"] = mpcalc.dewpoint_from_specific_humidity(pressure = ds['pressure'],
    #                                                  specific_humidity = ds['qv']) # , temperature = ds["temp"]
    ds = ds.metpy.dequantify()
    # convert temp to °C
    ds["temperature"] = ds["temperature"] - 273.15

    return ds.compute()


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
        # Construct the file path and open the dataset
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
    doesn't work due to dask issues when calculating the variables... use rather original read_3D variables func...
    reads in all arome data (fast) by Daniel
    :param variables: list of variables to read in, max possible: ["ciwc", "clwc", "p", "q", "th", "tke", "u", "v", "w", "z"]
    :return: ds with all raw arome data (~40GB!)
    """
    arome_paths = [confg.dir_3D_AROME + f"/AROME_Geosphere_20171015T1200Z_CAP02_3D_30min_1km_best_{var}.nc" for var in
                   variables]
    ds = xr.open_mfdataset(arome_paths, combine="by_coords", data_vars='minimal',
                           coords='minimal', compat='override', decode_timedelta=True)
    ds = ds.rename({"nz": "height"})  # rename to uniform height coordinate
    ds = ds.isel(height=slice(None, None, -1))  # reverse height axis to have uniform 0 at ground level!
    return ds

def read_in_arome_fixed_point(lat=47.259998, lon=11.384167, method="sel", variables=["p", "th", "z"]):  # , variable_list=
    """
    Read the AROME model output for a fixed point at a specific location with full time range.
    The method can be 'sel' or 'interp' for selecting the nearest point or interpolating to the point.

    :param lat: Latitude of the fixed point.
    :param lon: Longitude of the fixed point.
    :param method: Selection method of point ('sel' or 'interp').
    :param variables: List of variables to include in the dataset ["ciwc", "clwc", "p", "q", "th", "tke", "u", "v", "w", "z"]
    :return: Merged xarray Dataset
    """
    ds = read_in_arome(variables)
    if method == "interp":  # interpolate to point, uses numpy/scipy interp routines...
        ds = ds.interp(latitude=lat, longitude=lon)
    elif method == "sel":   # selects nearest point
        ds = ds.sel(latitude=lat, longitude=lon, method="nearest")

    ds = convert_calc_variables(ds)
    return ds

def read_in_arome_fixed_time(time="2017-10-15T14:00:00", variables=["p", "th", "z"]):
    """
    read arome data for a fixed time
    :param time: time as string f.e. "2017-10-15T12:00:00", you can use pd.to_datetime() to convert a string to a timestamp
    :return:
    ds of arome data with only wanted timestamp (~2GB)
    """
    ds = read_in_arome(variables)
    ds = ds.sel(time=time)  # select just needed timestep

    ds = convert_calc_variables(ds)
    return ds


def read_in_arome_radiosonde(time, method, lon, lat):
    """
    from hannes' plotting routines...
    read the MODEL output of AROME as it would be a Radiosonde with geopot height as vertical coordinate"""
    my_variable_list = ["p", "q", "th", "u", "v", "z"]

    if (method == "sel") | (method == "interp"):
        print(f"Your selected method is {method}")
    else:
        raise AttributeError(
            "You have to define a method (sel or interp) how the point near the LOWI should be selected")

    df_final = read_3D_variables_AROME(variables=my_variable_list, method=method, lon=lon, lat=lat, time=time)

    # print(df_final["p"].metpy.unit_array.magnitude) Extract values

    df_final["windspeed"] = metpy.calc.wind_speed(df_final["u"], df_final["v"])
    df_final["wind direction"] = metpy.calc.wind_direction(df_final["u"], df_final["v"], convention='from')
    df_final["temperature"] = metpy.calc.temperature_from_potential_temperature(df_final["p"], df_final["th"])
    df_final["dewpoint"] = metpy.calc.dewpoint_from_specific_humidity(pressure=df_final["p"],
                                                                      temperature=df_final["temperature"],
                                                                      specific_humidity=df_final["q"])

    p = df_final["p"].metpy.unit_array.to(units.hPa)  # Metadata is removed
    T = df_final["temperature"].metpy.unit_array.to(units.degC)

    Td = df_final["dewpoint"].metpy.unit_array
    wind_speed = df_final["windspeed"].metpy.unit_array.to(units.knots)
    wind_dir = df_final['wind direction']
    u, v = metpy.calc.wind_components(wind_speed, wind_dir)  # orig mpcalc

    ds = xr.Dataset()

    # Add variables to the dataset
    ds['u_wind'] = xr.DataArray(u.magnitude, dims=('height',),
                                coords={'height': df_final["z"].values},
                                attrs={'units': str(u.units)})
    ds['v_wind'] = xr.DataArray(v.magnitude, dims=('height',),
                                coords={'height': df_final["z"].values},
                                attrs={'units': str(v.units)})
    ds['pressure'] = xr.DataArray(p.magnitude, dims=('height',),
                                  coords={'height': df_final["z"].values},
                                  attrs={'units': str(p.units)})
    ds['temperature'] = xr.DataArray(T.magnitude, dims=('height',),
                                     coords={'height': df_final["z"].values},
                                     attrs={'units': str(T.units)})
    ds['dewpoint'] = xr.DataArray(Td.magnitude, dims=('height',),
                                  coords={'height': df_final["z"].values},
                                  attrs={'units': str(Td.units)})

    return ds.metpy.quantify()


if __name__ == '__main__':
    lat_ibk = 47.259998
    lon_ibk = 11.384167
    # arome = read_timeSeries_AROME(location)


    # arome3d = read_3D_variables_AROME(variables=["z"], method="sel")
    # arome = read_in_arome_fixed_point(variables=["p", "th", "z"])
    arome3d_new = read_in_arome_fixed_time(time="2017-10-15T12:00:00", variables=["z"])

    # arome_path = Path(confg.data_folder + "AROME_temp_timeseries_ibk.nc")
    # arome_path = Path(confg.model_folder + "/AROME/" + "AROME_temp_timeseries_ibk.nc")

    # arome = arome3d_new.isel(height=0).compute()
    # arome.to_netcdf(confg.dir_AROME + "AROME_geopot_height_3dlowest_level.nc", mode="w", format="NETCDF4")

    # arome3d_new# .to_netcdf(confg.dir_3D_AROME + "/AROME_temp_timeseries_ibk.nc", mode="w", format="NETCDF4")
    # read_2D_variables_AROME(lon, lat, variableList=["hfs", "hgt", "lfs", "lwd"], slice_lat_lon=False)

