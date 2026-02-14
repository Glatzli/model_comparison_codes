"""
re-written by Daniel
"""
import fix_win_DLL_loading_issue

fix_win_DLL_loading_issue
import importlib

import confg

importlib.reload(confg)
import os
import pandas as pd
import xarray as xr
# import numpy as np
from metpy.units import units
import metpy.calc as mpcalc
import datetime
from confg import variables_units_2D_AROME


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
    # Constants
    R_dryair = 287.05  # J/(kg*K), specific gas constant for dry air

    # Calculate wind speed and/or direction
    if ("wspd" in vars_to_calc) or ("udir" in vars_to_calc):
        try:
            u_wind = ds["u"].compute() * units("m/s")
            v_wind = ds["v"].compute() * units("m/s")
            ds["wspd"] = mpcalc.wind_speed(u_wind, v_wind)
            ds["wspd"] = ds['wspd'].assign_attrs(units="m/s", description="wind speed calced from u & v using MetPy")
            ds["udir"] = mpcalc.wind_direction(u_wind, v_wind)
            ds["udir"] = ds['udir'].assign_attrs(units="deg",
                                                 description="wind direction calced from u & v using MetPy")
        except Exception as e:
            print(f"  ✗ Error/or not needed to calculate wind speed/direction: {e}")

    # Convert pressure from Pa to hPa
    if "p" in ds:
        try:
            ds['p'] = (ds['p'] / 100.0) * units.hPa
            ds['p'] = ds['p'].assign_attrs(units="hPa", description="pressure")
        except Exception as e:
            print(f"  ✗ Error calculating pressure: {e}")

    # Calculate temperature from potential temperature
    if "temp" in vars_to_calc:
        try:
            ds["temp"] = mpcalc.temperature_from_potential_temperature(ds["p"], ds["th"] * units("K"))
        except Exception as e:
            print(f"  ✗ Error calculating temperature: {e}")

    # Calculate air density using ideal gas law
    if "rho" in vars_to_calc:
        try:
            # rho [kg/m^3] = p [Pa] / (R * T [K])
            ds["rho"] = (ds["p"] * 100) / (R_dryair * ds["temp"])
            ds["rho"] = ds['rho'].assign_attrs(units="kg/m^3",
                                               description=f"air density calced from p & temp using ideal gas law (R_dry = {R_dryair} J/(kg*K))")
        except Exception as e:
            print(f"  ✗ Error calculating density: {e}")

    # Calculate dewpoint temperature & dewpoint depression
    if "Td" in vars_to_calc:
        try:
            ds["Td"] = mpcalc.dewpoint_from_specific_humidity(pressure=ds["p"],
                                                              specific_humidity=ds["q"] * units("kg/kg"))
            ds["Td"] = ds['Td'].assign_attrs(units="degC",
                                             description="dewpoint Temp calculated from p and q using MetPy")
            ds["Td_dep"] = ds.temp - ds.Td
            ds["Td_dep"] = ds["Td_dep"].assign_attrs(units="degC",
                                                     description="Dewpoint temperature depression (temp - Td)")
        except Exception as e:
            print(f"  ✗ Error calculating dewpoint temperature: {e}")

    # Dequantify metpy units
    ds = ds.metpy.dequantify()

    # Convert temperature to Celsius
    if "temp" in vars_to_calc and "temp" in ds:
        try:
            ds["temp"] = ds["temp"] - 273.15
            ds["temp"] = ds['temp'].assign_attrs(units="degC", description="temperature calced from th & p")
        except Exception as e:
            print(f"  ✗ Error converting temperature to Celsius: {e}")

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
    variables = list(set(variables) | {"T", "p", "q"}) if "Td" in vars_to_calculate else variables
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

    # only read in variables that are saved as files, others need to be calculated
    arome_paths = [confg.dir_3D_AROME + f"/AROME_Geosphere_20171015T1200Z_CAP02_3D_30min_1km_best_{var}.nc" for var in
                   variables if var in data_vars]

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


def change_flux_signs(ds):
    """
    change signs of fluxes to have positive downward (atmosphere -> surface)
    :param ds:
    :return:
    """
    # Invert AROME turbulent fluxes to match convention: positive
    if "hfs" in ds:
        ds["hfs"] = -ds["hfs"]  # invert sign of sensible heat flux to be consistent with WRF
        ds["hfs"].attrs['description'] = "sensible heat flux (inverted), positive: upward (surface->atmosphere)"

    if "lfs" in ds:
        ds["lfs"] = -ds["lfs"]  # invert sign of latent heat flux to be consistent with WRF
        ds["lfs"].attrs['description'] = "latent heat flux (inverted), positive: upward (surface->atmosphere)"
    return ds


def read_in_arome_fixed_point(lat: float = confg.ALL_POINTS["ibk_uni"]["lat"],
        lon: float = confg.ALL_POINTS["ibk_uni"]["lon"], method: str = "sel", variables: list = ["p", "th", "z"],
        height_as_z_coord: str | bool = "direct"):
    """
    Read the AROME model output for a fixed point at a specific location with full time range.
    The method can be 'sel' or 'interp' for selecting the nearest point or interpolating to the point.

    :param lat: Latitude of the fixed point (default: Innsbruck University).
    :param lon: Longitude of the fixed point (default: Innsbruck University).
    :param method: Selection method ('sel' for nearest neighbor, 'interp' for linear interpolation).
    :param variables: List of variables to include in the dataset ["ciwc", "clwc", "p", "q", "th", "tke", "u", "v",
        "w", "z", "rho"]. If "rho" is needed, it will be calculated using ideal gas law in "convert_calc_variables".
        Default: ["p", "th", "z"]
    :param height_as_z_coord: How to set the vertical coordinate:
        - "direct": Use geopotential height and set it directly as vertical coord.
        - "above_terrain": Height above terrain at this point
        - False/None: Keep original model level indexing (1=lowest, 90=highest)
    :return: xarray.Dataset with selected variables at the specified point
    :raises ValueError: If method is invalid, or if 'z' is not in dataset when height_as_z_coord is set
    """

    valid_methods = ["sel", "interp"]
    if method not in valid_methods:  # Validate method parameter
        raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")

    # Read AROME data
    ds, vars_to_calculate = read_in_arome(variables=variables)

    # Select point using specified method
    if method == "interp":  # interpolate to point, uses numpy/scipy interp routines
        ds = ds.interp(latitude=lat, longitude=lon)
    elif method == "sel":  # selects nearest point
        ds = ds.sel(latitude=lat, longitude=lon, method="nearest")

    # Rename variables and calculate derived variables
    ds = rename_vars(data=ds)
    ds = convert_calc_variables(ds, vars_to_calc=vars_to_calculate)

    time_idx = 5  # skips first 2 hours of model initialization
    lowest_model_lvl_above_terrain = 5.1  # m, constant height of lowest model level above terrain
    if height_as_z_coord == "direct":
        # set geopotential height directly as vertical coordinate
        if "z" not in ds:
            raise ValueError("Variable 'z' (geopotential height) not in dataset. "
                             "Cannot set height as z coordinate. Add 'z' to variables list.")
        ds["height"] = ds.z.isel(time=time_idx)
        ds["height"] = ds["height"].assign_attrs(units="m", description="geopotential height amsl")

    elif height_as_z_coord == "above_terrain":
        # Calculate height above terrain at this point (when lowest level is subtracted, we would be on the terrain, therefore
        # add terrain height again...)
        if "z" not in ds:
            raise ValueError("Variable 'z' (geopotential height) not in dataset. "
                             "Cannot set height as z coordinate. Add 'z' to variables list.")
        z_lowest_model_lvl = ds.z.isel(time=time_idx).sel(height=1)  # geopot. height of lowest model level
        ds["height"] = ds.z.isel(time=time_idx) - z_lowest_model_lvl + lowest_model_lvl_above_terrain
        ds["height"] = ds["height"].assign_attrs(units="m", description="geopotential height above terrain")

    elif height_as_z_coord not in [False, None]:
        # Warn if invalid value provided, but continue with default behavior
        print(f"Warning: Invalid height_as_z_coord value '{height_as_z_coord}'. "
              f"Using original model level indexing. Valid options: 'direct', 'above_terrain', False, None")

    ds = ds.compute()
    return ds


def read_in_arome_fixed_time(day, hour, min, variables=["p", "th", "z"], min_lat=46.5, max_lat=48.2, min_lon=9.2,
        max_lon=13):
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

    if "hgt" in variables:  # geopotential height above terrain height (skip first 2 hours due to possible model init.
        # issues)
        # need to read 2d variable at that point to get terrain height
        # set geopot. height as vertical coordinate, subtract height of terrain at that point to compensate column depth
        ds["hgt"] = read_2D_variables_AROME(variableList=["hgt"], slice_lat_lon=True, lat=slice(min_lat, max_lat),
                                            lon=slice(min_lon, min_lat)).sel(time=timestamp).hgt

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
            ds = ds.sel(lon=lon, lat=lat)  # formerly indexed time: .isel(time=slice(4, None))
        else:
            ds = ds.sel(lon=lon, lat=lat, method="nearest")  # formerly indexed time: .isel(time=slice(4, None))

        for var, units in variables_units_2D_AROME.items():
            if var in ds:
                ds[var].attrs['units'] = units

        # ds_quantified = ds.metpy.quantify()
        datasets.append(ds)
    data = xr.merge(datasets, compat="no_conflicts", join="override")  # led to problems in past with join="outer" ...
    # to keep attr infos downgrade coords to float32, for uniformity with 3D data! anyways not needed that precise...
    data = data.assign_coords(lat=data.lat.astype("float32"), lon=data.lon.astype("float32"))

    data = change_flux_signs(ds=data)  # change flux signs to positive downward

    return data.compute()


def save_arome_topography(arome3d):
    """
    deprecated?
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

    # arome2d = read_2D_variables_AROME(variableList=["hfs", "lfs", "lwd", "lwu", "swd", "swu", "hgt", "u_v_from_3d"],
    #     lon=slice(confg.lon_hf_min, confg.lon_hf_max), lat=slice(confg.lat_hf_min, confg.lat_hf_max),
    #     slice_lat_lon=True)

    arome_point = read_in_arome_fixed_point(lat=confg.ALL_POINTS["telfs"]["lat"], lon=confg.ALL_POINTS["telfs"]["lon"],
                                            variables=["p", "q", "th", "z", "temp", "Td", "Td_dep"],
                                            height_as_z_coord="direct")  # ["p", "temp", "th", "z", "udir", "wspd"]

    # arome = read_in_arome_fixed_time(day=16, hour=12, min=0, variables=["z", "hgt"], min_lat=confg.lat_hf_min,
    #                                  max_lat=confg.lat_hf_max, min_lon=confg.lon_hf_min, max_lon=confg.lon_hf_max)
    arome_point  # arome2d

# arome_z_subset = xr.open_dataset(confg.dir_AROME + "AROME_subset_z.nc", mode="w", format="NETCDF4")  # arome_z  # arome_path = Path(confg.data_folder + "AROME_temp_timeseries_ibk.nc")  # arome_path = Path(confg.model_folder +  # "/AROME/" + "AROME_temp_timeseries_ibk.nc")

# arome3d_new.to_netcdf(confg.dir_3D_AROME + "/AROME_temp_timeseries_ibk.nc", mode="w", format="NETCDF4")

# extract_3d_variable_define_2D(variables=["hfs"])  # used to save u & v like 2D variables