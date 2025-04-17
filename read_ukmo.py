"""Script to read in UKMO Model: at fixed time OR at fixed height (model level)"""
import cartopy.crs as ccrs
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import xarray as xr
from metpy.calc import dewpoint_from_specific_humidity, relative_humidity_from_dewpoint
from metpy.units import units
# from rasterio.env import local
from scipy.interpolate import interp1d

import confg
from confg import station_files_zamg, stations_ibox, MOMMA_stations_PM, ukmo_folder


def get_coordinates_by_station_name(station_name):
    """extract latitude and longitude by station_name"""
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


def get_rotated_index_of_lat_lon(latitude, longitude):
    """Function to get the index of the selected latitude and longitude"""
    dat = xr.open_dataset(f"{ukmo_folder}/MetUM_MetOffice_20171015T1200Z_CAP02_3D_30min_1km_optimal_v.nc", decode_timedelta = True)

    # Define the rotated pole coordinates and the regular longitude-latitude projection
    lon0, lat0 = -168.6, 42.7
    proj_rot = ccrs.RotatedPole(pole_longitude=lon0, pole_latitude=lat0)
    proj_ll = ccrs.PlateCarree()

    # Extract rotated latitude and longitude values and create 2D grids
    lonr, latr = dat["grid_longitude"].values, dat["grid_latitude"].values
    lonr2d, latr2d = np.meshgrid(lonr, latr)
    lonlat = proj_ll.transform_points(proj_rot, lonr2d, latr2d)
    regular_lon, regular_lat = lonlat[..., 0], lonlat[..., 1]

    # Calculate distances and find the index of the nearest grid point
    distances = np.sqrt((regular_lon - longitude) ** 2 + (regular_lat - latitude) ** 2)
    yi, xi = np.unravel_index(np.argmin(distances),
                              distances.shape)  # kriege hier yi und xi her das ist der wichtige schritt, somit kann ich alle rausholen

    # Output the nearest x and y projection coordinates
    # print(f"Value at nearest xpoint: {dat['projection_x_coordinate'][yi, xi].values}")
    # print(f"Value at nearest ypoint: {dat['projection_y_coordinate'][yi, xi].values}")

    assert np.isclose(dat['projection_x_coordinate'][yi, xi].values, longitude, atol=0.3)  # renamed from projection_x_coordinate to lon_proj
    assert np.isclose(dat['projection_y_coordinate'][yi, xi].values, latitude, atol=0.3)

    return xi, yi


def convert_variables(ds, multiple_levels=True):
    """converts variables with metpy, interpolates wind on pressure levels only if dataset in multiple levels

    ds: ukmo-xarray dataset
    multiple_levels: bool
    """

    ds["pressure"] = (ds["air_pressure"] / 100) * units("hPa")
    # p = ds["pressure"] * units.hPa
    ds["temperature"] = mpcalc.temperature_from_potential_temperature(ds["pressure"], ds["air_potential_temperature"] *
                                                                      units("K"))  # .metpy.dequantify() - 273.15 # temp in Celsius w/o unit

    qv = ds["specific_humidity"] * units("kg/kg")  # from kg / kg in g/kg
    # u_icon = ds["transformed_x_wind"] * units("m/s")  # I don't need this now...
    # v_icon = ds["transformed_y_wind"] * units("m/s")

    # ds['wind_dir'] = mpcalc.wind_direction(u_icon, v_icon, convention='from')
    # ds["windspeed"] = mpcalc.wind_speed(u_icon, v_icon)

    ds["rh"] = mpcalc.relative_humidity_from_specific_humidity(ds["pressure"], ds["temperature"], qv)
    # o.k. if interpolation is not necessary

    #if multiple_levels:
    #    ds = ds.rename_vars(name_dict={"level_height": "level_height_u_v_wind"})
        # Wind is defined on another level than pressure, but I need the pressure at the wind level, so extrapolate it
        # Use this function to calculate pressures at wind levels,
        # ds = ds.interp(pressure=ds["level_height_u_v_wind"], method="linear", kwargs={"fill_value": "extrapolate"})
        # but they are saved in the dataset within the same height coord?!?

    #else: ds["relative_humidity"] = mpcalc.relative_humidity_from_specific_humidity(ds["pressure"], temp, qv).to("percent")
    ds = ds.metpy.dequantify()
    ds["temperature"] = ds["temperature"] - 273.15  # convert to Celsius
    ds["rh"] = ds["rh"] * 100  # convert to percent
    return ds

def read_ukmo_fixed_point_lowest_level(city_name=None, lat=None, lon=None):
    """read in UKMO Model at a fixed point and select the lowest level, either with city_name or with (lat, lon)"""
    if city_name is not None:
        lat, lon = get_coordinates_by_station_name(city_name)

    xi, yi = get_rotated_index_of_lat_lon(latitude=lat, longitude=lon)

    for i, var in enumerate(["u", "v", "w", "z", "th", "q", "p"]):
        data = xr.open_dataset(f"{ukmo_folder}/MetUM_MetOffice_20171015T1200Z_CAP02_3D_30min_1km_optimal_{var}.nc", decode_timedelta = True)
        data = data.sel(time=slice("2017-10-15T14:00:00", "2017-10-16T12:00:00.000000000"))  # delete first 2h
        data = data.isel(grid_latitude=yi, grid_longitude=xi, model_level_number=0, bnds=1)  # selects lowest level, and lat, lon

        if i == 0:  # this is not beautiful but o.k.
            dat = data
        else:
            dat = xr.merge([dat, data], compat='override')

        # print(data_final["level_height"].values) # u and v are on 2.5 m, all other variables at 5m
    # dat = convert_variables(dat, multiple_levels=False)
    return dat

def read_ukmo_fixed_point(city_name=None, lat=None, lon=None, variable_list=["u", "v", "w", "z", "th", "q", "p"]):
    """read in UKMO Model at a fixed point and select the lowest level, either with city_name or with (lat, lon)
    now with xr mfdataset much faster!
    """
    if city_name is not None:
        lat, lon = get_coordinates_by_station_name(city_name)

    xi, yi = get_rotated_index_of_lat_lon(latitude=lat, longitude=lon)

    ukmo_files = [confg.ukmo_folder + f"/MetUM_MetOffice_20171015T1200Z_CAP02_3D_30min_1km_optimal_{var}.nc" for var in variable_list]
    data = xr.open_mfdataset(ukmo_files, combine = "by_coords", data_vars = 'minimal',
                             coords = 'minimal', compat = 'override', decode_timedelta=True)  #concat_dim="time",

    data = data.isel(grid_latitude=yi, grid_longitude=xi, bnds=1)  # selects lat, lon

    data = data.rename({"model_level_number": "height"})
    data = convert_variables(data, multiple_levels=True)
    return data

def read_ukmo_fixed_time(time="2017-10-15T14:00:00", variable_list=["u", "v", "w", "z", "th", "q", "p"]):
    """read full domain of UKMO Model at a fixed time"""

    ukmo_files = [confg.ukmo_folder + f"/MetUM_MetOffice_20171015T1200Z_CAP02_3D_30min_1km_optimal_{var}.nc" for var in variable_list]
    data = xr.open_mfdataset(ukmo_files, combine = "by_coords", data_vars = 'minimal',
                             coords = 'minimal', compat = 'override', decode_timedelta=True)  #concat_dim="time",

    data = data.sel(time=time)  # selects time , bnds=1

    data = data.rename({"model_level_number": "height"})
    data = convert_variables(data, multiple_levels=True)
    return data


def read_ukmo_fixed_point_and_time(city_name=None, time="2017-10-15T14:00:00", lat=None, lon=None):
    """read in UKMO Model at fixed point w all levels, either with city_name or with lat/lon, get xarray ds!

    city_name: str
    time: str
    lat: float
    lon: float
    """

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
    dat = convert_variables(dat, multiple_levels=True)
    dat = dat.rename({"model_level_number": "height"})
    return dat


def read_full_ukmo(variables= ["u", "v", "w", "z", "th", "q", "p"]):
    """read all ukmo data (~40GB) as xarray dataset
    Problem: "regular_longitude" and "regular_latitude" are not coordinates, but data variables!
    (written by chatgpt, maybe coord. transform is totally useless)

    """
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


def get_ukmo_height_of_specific_lat_lon(lat, lon):
    """Get ukmo height for a specific lat lon"""
    # They have no time, hgt = Terrain height

    xi, yi = get_rotated_index_of_lat_lon(latitude=lat, longitude=lon)

    # ignore lct (land binary mask) we are only looking at land
    for var in ["hgt"]:
        dat = xr.open_dataset(f"{ukmo_folder}/MetUM_MetOffice_20171015T1200Z_CAP02_2D_30min_1km_optimal_{var}.nc")

        data_final = dat.isel(grid_latitude=yi, grid_longitude=xi)

        if var == "hgt":
            altitude = data_final["surface_altitude"].values

    return altitude


def get_ukmo_height_of_city_name(city_name):
    """get the altitude of a specific city"""
    lat, lon = get_coordinates_by_station_name(city_name)
    return get_ukmo_height_of_specific_lat_lon(lat=lat, lon=lon)


if __name__ == '__main__':
    # get values on lowest level
    # get_coordinates_by_station_name("IAO")
    # um = read_ukmo_fixed_point_and_time("IAO", "2017-10-15T14:00:00")

    #um = read_ukmo_fixed_point(lat=47.266076, lon=11.4011756)
    um = read_ukmo_fixed_time(time="2017-10-15T14:00:00")
    um
    # um = read_ukmo_fixed_point_and_time(lat=47.266076, lon=11.4011756, time="2017-10-15T14:00:00")
    # um

    # um = read_full_ukmo(["th", "p"])
    # um


    # xi, yi = get_rotated_index_of_lat_lon(latitude=47.266076, longitude=11.4011756)
    # df3d = get_ukmo_fixed_point_lowest_level("Kufstein")
