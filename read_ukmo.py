"""Script to read in UKMO Model: at fixed time OR at fixed height (model level)

re-written by Daniel

"""
import cartopy.crs as ccrs
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import xarray as xr
from metpy.calc import dewpoint_from_specific_humidity, relative_humidity_from_dewpoint
from metpy.units import units
import pyproj
import matplotlib
import matplotlib.pyplot as plt
import datetime
from pyproj import Transformer, CRS
# from rasterio.env import local
from scipy.interpolate import interp1d
from pathlib import Path

import confg
from confg import station_files_zamg, stations_ibox, MOMMA_stations_PM, ukmo_folder


def get_coordinates_by_station_name(station_name):
    """
    deprecated?
    extract latitude and longitude by station_name"""
    # Iterate over all station entries in the dictionary
    if station_name in ["Innsbruck Uni", "Kufstein", "Jenbach", "Innsbruck Airport"]:
        for station_code, station_info in station_files_zamg.items():
            # Check if the current entry's name matches the provided station name
            if station_info['name'].lower() == station_name.lower():
                # If a match is found, return the latitude and longitude
                return station_info['lat'], station_info['lon']
        # If no match is found, return None to indicate that the station was not found
        return None, None
    elif station_name in station_files_zamg.keys():
        return station_files_zamg[station_name]["lat"], station_files_zamg[station_name]["lon"]
    elif station_name in stations_ibox.keys():
        return stations_ibox[station_name]["latitude"], stations_ibox[station_name]["longitude"]
    elif station_name in MOMMA_stations_PM.keys():
        return MOMMA_stations_PM[station_name]["latitude"], MOMMA_stations_PM[station_name]["longitude"]
    else:
        raise AssertionError("No station found with this name!")


def convert_calc_variables(ds, vars_to_calc=["temp", "rho"]):  # , multiple_levels=True
    """converts & calculates variables using metpy, ideal gas law etc
    UM is quite similar as AROME, but there will be some differences (wind level staggering etc), therefore leave extra
    function for every model

    interpolates wind on pressure levels only if dataset in multiple levels

    :param ds: xarray dataset
    :param vars_to_calc: list of variables to calculate
    """
    # Constants
    R_dryair = 287.05  # J/(kg*K), specific gas constant for dry air

    # Calculate wind speed and/or direction
    try:
        if "wspd" in vars_to_calc or "udir" in vars_to_calc:
            u_wind = ds["u"].compute() * units("m/s")  # somehow sometimes brought error if not computed before...
            v_wind = ds["v"].compute() * units("m/s")
            if "wspd" in vars_to_calc:
                ds["wspd"] = mpcalc.wind_speed(u_wind, v_wind)
                ds["wspd"] = ds['wspd'].assign_attrs(units="m/s", description="wind speed calced from u & v using MetPy")
            if "udir" in vars_to_calc:
                ds["udir"] = mpcalc.wind_direction(u_wind, v_wind)
                ds["udir"] = ds['udir'].assign_attrs(units="deg", description="wind direction calced from u & v using MetPy")
    except Exception as e:
        print(f"  ✗ Error calculating wind speed/direction: {e}")

    # Convert pressure from Pa to hPa
    try:
        if "p" in ds:
            ds["p"] = (ds["p"] / 100) * units("hPa")
            ds['p'] = ds['p'].assign_attrs(units="hPa", description="pressure")
    except Exception as e:
        print(f"  ✗ Error calculating pressure: {e}")

    # Calculate temperature from potential temperature
    try:
        if "temp" in vars_to_calc:
            ds["temp"] = mpcalc.temperature_from_potential_temperature(ds["p"], ds["th"] * units("K"))
    except Exception as e:
        print(f"  ✗ Error calculating temperature: {e}")

    # Calculate air density using ideal gas law
    try:
        if "rho" in vars_to_calc:
            # rho [kg/m^3] = p [Pa] / (R * T [K])
            ds["rho"] = (ds["p"] * 100) / (R_dryair * ds["temp"])
            ds["rho"] = ds['rho'].assign_attrs(
                units="kg/m^3",
                description=f"air density calced from p & temp using ideal gas law (R_dry = {R_dryair} J/(kg*K))")
    except Exception as e:
        print(f"  ✗ Error calculating density: {e}")

    # Calculate relative humidity; not used...
    try:
        if "rh" in vars_to_calc:
            ds['rh'] = mpcalc.relative_humidity_from_specific_humidity(ds['p'], ds["temp"], ds['q'] * units("kg/kg")
            ) * 100  # convert to percent
            ds['rh'] = ds['rh'].assign_attrs(units="%", description="relative humidity calced from p, temp & q")
    except Exception as e:
        print(f"  ✗ Error calculating relative humidity: {e}")

    # Dequantify metpy units
    ds = ds.metpy.dequantify()

    # Convert temperature to Celsius
    try:
        if "temp" in vars_to_calc and "temp" in ds:
            ds["temp"] = ds["temp"] - 273.15
            ds["temp"] = ds['temp'].assign_attrs(units="degC", description="temperature calced from th & p")
    except Exception as e:
        print(f"  ✗ Error converting temperature to Celsius: {e}")

    return ds


def create_ds_geopot_height_as_z_coordinate(ds):
    """
    create a new dataset with geopotential height as vertical coordinate for temperature for plotting
    :param ds:
    :return:
    :ds_new: new dataset with geopotential height as vertical coordinate
    """
    geopot_height = ds.z.isel(time=20).compute()

    ds_new = xr.Dataset(
        data_vars=dict(
            th=(["time", "height"], ds.th.values),
            p=(["time", "height"], ds.p.values),
        ),
        coords=dict(
            height=("height", geopot_height.values),
            time=("time", ds.time.values)
        ),
        attrs=dict(description="UKMO data with geopotential height at mid of ds as vertical coordinate"))

    return ds_new


def rename_variables(ds):
    """rename variables to have a consistent naming convention
    :param ds: xarray dataset
    :return: ds with renamed variables
    """
    ds = ds.rename({"surface_altitude" : "hgt", "model_level_number": "height",
                    "grid_latitude": "lat", "grid_longitude": "lon","air_potential_temperature": "th",
                    "transformed_x_wind": "u", "transformed_y_wind": "v", "upward_air_velocity": "w",
                    "air_pressure": "p", "specific_humidity": "q", "geopotential_height": "z"})
    ds['lat'] = ds['lat'].assign_attrs(description="regridded latitude", units="degrees_north")
    ds['lon'] = ds['lon'].assign_attrs(description="regridded longitude", units="degrees_east")
    ds["u"] = ds["u"].assign_attrs(description="regridded wind speed in x dir")
    ds["v"] = ds["v"].assign_attrs(description="regridded wind speed in y dir")
    ds["w"] = ds["w"].assign_attrs(description="regridded wind speed in w dir (small differences due to "
                                               "interpolation of the small values)")
    return ds


def read_ukmo(variables=["p", "temp", "th", "rho", "z"]):
    """
    reads all regridded UM files as xarray dataset using DASK ~3GB
    also renames the vars immediately for consistency and adds attrs
    :param variables: list of variables to be read/calculated in
    :return:
    """
    data_vars = ["hgt", "p", "q", "th", "u", "v", "w", "z"]  # saved file vars except "lct":land binary mask, made problems
    # whilst interpolating (and I'm not even sure what this var is...)
    vars_to_calculate = set(variables) - set(data_vars)  # need to calculate the var's that are not in ds and are given

    ukmo_files = [confg.ukmo_folder + (f"MetUM_MetOffice_20171015T1200Z_CAP02_{'2D' if var == 'hgt' else '3D'}"
                                       f"_30min_1km_optimal_{var}_regrid.nc") for var in data_vars]  # read in all saved vars
    # only the terrain height "hgt" is 2D, and therefore also has a different filename
    data = xr.open_mfdataset(ukmo_files, combine = "by_coords", data_vars = 'minimal',
                             coords = 'minimal', compat = 'override', decode_timedelta=True)  #concat_dim="time",
    data = rename_variables(data)
    return data, vars_to_calculate


def read_ukmo_fixed_point(lat=confg.ALL_POINTS["ibk_uni"]["lat"], lon=confg.ALL_POINTS["ibk_uni"]["lon"], variables=None, height_as_z_coord="direct"):
    """
    Read in UKMO Model at a fixed point and select the lowest level, either with city_name or with (lat, lon)
    now with xr mfdataset much faster!
    
    :param lat: Latitude of the fixed point (default: Innsbruck University).
    :param lon: Longitude of the fixed point (default: Innsbruck University).
    :param variables: List of variables to include in the dataset
    :param height_as_z_coord: How to set the vertical coordinate:
        - "direct": Use geopotential height and set it directly as vertical coord.
        - "above_terrain": Height above terrain at this point
        - False/None: Keep original model level indexing
    :return: xarray.Dataset with selected variables at the specified point
    :raises ValueError: If 'z' is not in dataset when height_as_z_coord is set
    """
    data, vars_to_calculate = read_ukmo(variables=variables)
    data = data.sel(lat=lat, lon=lon, method="nearest")  # selects lat, lon
    data = convert_calc_variables(data, vars_to_calc=vars_to_calculate)
    data = data[variables]  # subset to have only the variables wanted before computing
    
    time_idx = 5  # skips first 2 hours of model initialization
    lowest_model_lvl_above_terrain = 10.0  # m, constant height of lowest model level above terrain (for UM assume 10, as
    # this was the target value)
    
    if height_as_z_coord == "direct":
        # set geopotential height directly as vertical coordinate
        if "z" not in data:
            raise ValueError("Variable 'z' (geopotential height) not in dataset. ")
        data["height"] = data.z.isel(time=time_idx)
        data["height"] = data["height"].assign_attrs(units="m", description="geopotential height amsl")
    
    elif height_as_z_coord == "above_terrain":
        # Calculate height above terrain at this point
        if "z" not in data:
            raise ValueError("Variable 'z' (geopotential height) not in dataset. ")
        z_lowest_model_lvl = data.z.isel(time=time_idx).sel(height=1)  # geopot. height of lowest model level
        data["height"] = data.z.isel(time=time_idx) - z_lowest_model_lvl + lowest_model_lvl_above_terrain
        data["height"] = data["height"].assign_attrs(units="m", description="geopotential height above model terrain")
    
    elif height_as_z_coord not in [False, None]:
        # Warn if invalid value provided, but continue with default behavior
        print(f"Warning: Invalid height_as_z_coord value '{height_as_z_coord}'. "
              f"Using original model level indexing. Valid options: 'direct', 'above_terrain', False, None")

    return data.compute()


def read_ukmo_fixed_time(day=16, hour=12, min=0, variables=None):
    """
    read full domain of UKMO Model at a fixed time
    converts the lat/lon vars from rotated pole to regular lat/lon
    :param day: int [15, 16]
    :param variables: list of variables to read in
    """
    data, vars_to_calculate = read_ukmo(variables=variables)
    timestamp = datetime.datetime(2017, 10, day, hour, min, 00)
    data = data.sel(time=timestamp)  # selects time , bnds=1
    data = convert_calc_variables(data, vars_to_calc=vars_to_calculate)
    data = data[variables]  # subset to have only the variables wanted before computing

    return data.compute()

"""
def read_ukmo_fixed_point_and_time(city_name=None, time="2017-10-15T14:00:00", lat=None, lon=None):
    deprecated
    read in UKMO Model at fixed point w all levels, either with city_name or with lat/lon, get xarray ds!

    city_name: str
    time: str
    lat: float
    lon: float
    

    if city_name is not None:
        lat, lon = get_coordinates_by_station_name(city_name)
    # original: my_lat, my_lon = get_coordinates_by_station_name(city_name)
    xi, yi = get_rotated_index_of_lat_lon(latitude=lat, longitude=lon)

    datasets = []  # List to hold datasets for each variable
    for var in ["u", "v", "w", "z", "th", "q", "p"]:
        data = xr.open_dataset(f"{ukmo_folder}/MetUM_MetOffice_20171015T1200Z_CAP02_3D_30min_1km_optimal_{var}.nc", decode_timedelta = True)
        data = data.isel(grid_latitude=yi, grid_longitude=xi, bnds=1)
        datasets.append(data)

    dat = xr.merge(datasets, compat='override')
    dat = convert_calc_variables(dat)
    dat = dat.rename({"model_level_number": "height"})
    return dat


def read_full_ukmo(variables= ["u", "v", "w", "z", "th", "q", "p"]):
    deprecated?
    read all ukmo data (~40GB) as xarray dataset
    Problem: "regular_longitude" and "regular_latitude" are not coordinates, but data variables!
    (written by chatgpt, maybe coord. transform is totally useless)

    
    um_files = [ukmo_folder + "/MetUM_MetOffice_20171015T1200Z_CAP02_3D_30min_1km_optimal_" + var + ".nc"
                for var in variables]
    um = xr.open_mfdataset(um_files, combine='by_coords', compat='override', decode_timedelta=True)

    # transform coordinates by chatgpt
    proj_rot = ccrs.RotatedPole(pole_longitude= -168.6, pole_latitude= 42.7)
    proj_ll = ccrs.PlateCarree()
    lonr = um['grid_longitude'].values
    latr = um['grid_latitude'].values
    lonr2d, latr2d = np.meshgrid(lonr, latr)
    lonlat = proj_ll.transform_points(proj_rot, lonr2d, latr2d)
    regular_lon, regular_lat = lonlat[..., 0], lonlat[..., 1]

    um['regular_longitude'] = (('grid_latitude', 'grid_longitude'), regular_lon)
    um['regular_latitude'] = (('grid_latitude', 'grid_longitude'), regular_lat)

    return um
    """


def save_um_topography(ds):
    """
    save the terrain height as netcdf and .tif file (for PCGP calc)
    """

    # save um extent terrain height as xr dataset
    ds.to_netcdf(confg.ukmo_folder + "/UM_geometric_height_3dlowest_level.nc", mode="w", format="NETCDF4")
    ds.rio.write_crs("EPSG:4326", inplace=True)  # add some projection info ()
    # rename coords for xdem calc of slope => need .tif file
    ds.rename({"lat": "y", "lon": "x"}).hgt.rio.to_raster(confg.ukmo_folder + "/UM_geometric_height_3dlowest_level.tif")


if __name__ == '__main__':
    matplotlib.use('Qt5Agg')
    # get values on lowest level
    # get_coordinates_by_station_name("IAO")
    # um = read_ukmo_fixed_point_and_time("IAO", "2017-10-15T14:00:00")
    variables = ["udir", "wspd", "q", "p", "th", "temp", "z"]
    um = read_ukmo_fixed_point(lat=confg.ALL_POINTS["ibk_uni"]["lat"], lon=confg.ALL_POINTS["ibk_uni"]["lon"],
                               variables=variables, height_as_z_coord="above_terrain")  # , "hgt" , "rho"
    # um_extent = read_ukmo_fixed_time(day=16, hour=12, min=0, variables=["p", "temp", "th", "z"])
    um

    # save lowest level as nc file for topo plotting
    # save_um_topography(ds=um_extent)

    # um_geopot = create_ds_geopot_height_as_z_coordinate(um)

    # save um for plotting temp timeseries with geopot height as z coord
    # um_path = Path(confg.ukmo_folder + "/UKMO_temp_timeseries_ibk.nc")
    # um_geopot.to_netcdf(um_path, mode="w", format="NETCDF4")
    # um

 # are these the correct lats&lons of the UM model? just thought by myself, not sure if this is correct
    # lat = um.rotated_latitude_longitude.grid_north_pole_latitude + um.grid_latitude
    # lon = (um.grid_longitude - 360) + um.rotated_latitude_longitude.grid_north_pole_longitude
    # um["grid_latitude"] = lat
    # um["grid_longitude"] = lon
