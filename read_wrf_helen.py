"""Script to read in WRF ACINN data
Note: it is a sort of WRF Data, but still different than the WRF_ETH files.
re-written by Daniel, regridded by Manuela

Hannes:
Use salem from Prof. Maussion (https://salem.readthedocs.io/en/stable/) to read it in, had to modify it sometimes

Most important functions:
- read_wrf_fixed_point_and_time() # at fixed point and time, only independent coordinate is the height
- read_wrf_fixed_time() # get a 2D slice if i select afterwards also a certain height
- read_wrf_fixed_point() # get a timeseries with variable height
"""

import metpy.calc as mpcalc
import netCDF4
import pandas as pd
import xarray as xr
from fsspec.compression import compr
from matplotlib import pyplot as plt
from salem import wrftools
import datetime
from scripts.regsetup import description

from shapely.geometry import Point, Polygon
from sklearn.covariance import ledoit_wolf
# from wrf import combine_dims
from xarray.backends.netCDF4_ import NetCDF4DataStore
import confg
from metpy.units import units
import cartopy.crs as ccrs
import pyproj
# from pyproj import Proj, Transformer
import numpy as np
#import matplotlib
#from pathlib import Path
#matplotlib.use('Qt5Agg')

def __open_wrf_dataset_my_version(file, **kwargs):
    """deprecated?
    Hannes used originally for loops through time to append each dataset to a list and then to combine them...

    Updated Function from salem, the problem is that our time has no units (so i had to remove the check in the original function)
    Internally used to unstagger, and read in WRF 3D File.

    Wrapper around xarray's open_dataset to make WRF files a bit better.

    This is needed because variables often have not enough georef attrs
    to be understood alone, and datasets tend to loose their attrs with
    operations...

    Parameters
    ----------
    file : str
        the path to the WRF file
    **kwargs : optional
        Additional arguments passed on to ``xarray.open_dataset``.

    Returns
    -------
    an xarray Dataset
    """

    nc = netCDF4.Dataset(file)
    nc.set_auto_mask(False)

    # Change staggered variables to unstaggered ones (did not work on salem)
    for vn, v in nc.variables.items():
        if wrftools.Unstaggerer.can_do(v):
            nc.variables[vn] = wrftools.Unstaggerer(v)

    # Check if we can add diagnostic variables to the pot
    for vn in wrftools.var_classes:
        cl = getattr(wrftools, vn)
        if vn not in nc.variables and cl.can_do(nc):
            nc.variables[vn] = cl(nc)

    # trick xarray with our custom netcdf
    ds = xr.open_dataset(NetCDF4DataStore(nc), **kwargs)

    # remove time dimension to lon lat
    for vn in ['XLONG', 'XLAT']:
        try:
            v = ds[vn].isel(Time=0)
            ds[vn] = xr.DataArray(v.values, dims=['south_north', 'west_east'])
        except (ValueError, KeyError):
            pass

    # add cartesian coords
    # ds['longitude'] = ds.salem.grid.x_coord  # adapted
    # ds['latitude'] = ds.salem.grid.y_coord

    # add pyproj string everywhere
    ds.attrs['pyproj_srs'] = ds.salem.grid.proj.srs
    for v in ds.data_vars:
        ds[v].attrs['pyproj_srs'] = ds.salem.grid.proj.srs

    return ds

def assign_rename_coords(ds):
    """
    assign and rename coordinates for WRF dataset: time and lat/lon

    """

    # WRF has originally "Time" as a dimension, but that are integer vals; it also has "time" as a data variable, which is
    # in minutes since 2017-10-15T12:00:00Z (so mins after simulation start)
    # => assign time to Time coordinate, delete "time" data var. rename "Time" to "time" for consistency and calculate
    # it as datetime vals...
    ds = ds.assign_coords(Time=("Time", ds.time.compute().data))
    ds = ds.drop_vars("time")
    new_time = [pd.Timestamp('2017-10-15T12:00:00') + pd.Timedelta(minutes=m) for m in ds.Time.values]
    ds = ds.assign_coords(Time=("Time", new_time))

    # define the lat&lon vals that are saved in data vars as coordinates
    ds = ds.assign_coords(west_east=("west_east", ds.isel(south_north=1, Time=0).lon.data))
    ds = ds.assign_coords(south_north=("south_north", ds.isel(west_east=1, Time=0).lat.data))
    ds = ds.drop_vars(["lon", "lat"])  # drop lon & lat data vars, because they are now coords
    ds = ds.rename({"Time": "time", "bottom_top": "height", "south_north": "lat", "west_east": "lon"})
    ds = ds.assign_coords(height=("height", ds.height.data + 1))  # height is just dim w/o coordinate,
    # also start by 1 till 80 not 0 to 79 as original
    ds = ds.assign_coords(bottom_top_stag=("bottom_top_stag", ds.bottom_top_stag.data + 1))  # same for bottom_top_stag
    return ds


def open_wrf_mfdataset(filepaths, variables):
    """changed function to work for multiple files with xarray's mfdataset
    by Daniel to work for regridded WRF data...

    :param filepaths: list of filepaths to regridded WRF-files
    :param variables: list of variables that need to be calculated (that are not in ds)
    """
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]  # Einzelnen Pfad in eine Liste umwandeln
    datasets = []  # i don't know exactly what these lines are doing...
    for filepath in filepaths:  # iterate through all paths, do commands for each file
        nc = netCDF4.Dataset(filepath)
        nc.set_auto_mask(False)

        # unstagger variables; from Hannes...
        for vn, v in nc.variables.items():
            if wrftools.Unstaggerer.can_do(v):
                nc.variables[vn] = wrftools.Unstaggerer(v)

        # Hinzufügen diagnostischer Variablen
        for vn in wrftools.var_classes:
            cl = getattr(wrftools, vn)
            if vn not in nc.variables and cl.can_do(nc):
                nc.variables[vn] = cl(nc)

        # Trick xarray mit unserem benutzerdefinierten NetCDF
        datasets.append(NetCDF4DataStore(nc))"""

    ds = xr.open_mfdataset(filepaths, chunks="auto", concat_dim="Time", combine="nested", data_vars='minimal',
                                coords='minimal', compat='override', decode_cf=False)
    # need to calculate the var's that are not in ds and are given (there are also lat&lon, time in this list, but those
    # are not calculated...)
    vars_to_calculate = set(variables) - set(list(ds.data_vars))
    ds = assign_rename_coords(ds)
    ds.attrs["history"] = "regridded to regular lat/lon grid by Manuela Lehner " + ds.attrs["history"]

    return ds, vars_to_calculate


def salem_example_plots(ds):
    """Make some example plots with salem plotting functions need to have some slice of lat lon
    Could be used in future to plot 2D maps with a certain variable and a certain extent
    """
    hmap = ds.salem.get_map(cmap='topo')

    hmap.set_data(ds['alb'])
    hmap.set_points(confg.station_files_zamg["LOWI"]["lon"], confg.station_files_zamg["LOWI"]["lat"])
    hmap.set_text(confg.station_files_zamg["LOWI"]["lon"], confg.station_files_zamg["LOWI"]["lat"], 'Innsbruck', fontsize=17)
    hmap.visualize()

    # psrs = 'epsg:4236'  # http://spatialreference.org/ref/epsg/wgs-84-utm-zone-30n/

    # ds.attrs['pyproj_srs'] = psrs

    print(ds["alb"])
    padding = 0.02
    min_lon = 11.3398110103 - padding
    max_lon = 11.4639758751 + padding
    min_lat = 47.2403724414 - padding
    max_lat = 47.321

    box_polygon2 = Polygon(
        [(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat), (min_lon, min_lat)])

    plt.figure()
    ds["alb"].salem.quick_map(interp='linear')


def convert_calc_variables(ds, vars_to_calc=["temp", "rho"]):
    """
    calculate only variables that are wanted, and change attributes to new units/varibles

    Parameters:
    - ds: A xarray Dataset containing the columns 'p' for pressure in Pa
          and 'th' for potential temperature in Kelvin.

    Returns:
    - A xarray Dataset with the original data and new columns:
      'pressure' in hPa and 'temperature' in degrees Celsius.
    """
    if "th" in ds:
        # ds["th"] = ds["th"] + 300 # th is original the perturbation potential temp,
        # WRF user manual says https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/output.html
        ds["th"] = ds['th'].assign_attrs(units= "K", description="potential temperature, calced from pert. pot. temp + 300K")

    if "p" in ds:
        # Convert pressure from Pa to hPa
        ds["p"] = (ds["p"] / 100) * units.hPa
        ds["p"] = ds["p"].assign_attrs(units="hPa", description="pressure")
        if "temp" in vars_to_calc:
            # calculate temp in K
            ds["temp"] = mpcalc.temperature_from_potential_temperature(ds['p'], ds["th"] * units("K"))
            if "rho" in vars_to_calc:
                ds["rho"] = (ds["p"] * 100) / (287.05 * ds["temp"])  # using ideal gas law: rho [kg/m^3] = p [Pa] / (R * T [K]) with R_dryair = 287.05 J/kgK
                ds["rho"] = ds['rho'].assign_attrs(units="kg/m^3", description="air density, calced w R_dryair = 287.05")

    # ds["rh"] = mpcalc.relative_humidity_from_mixing_ratio(ds["p"], ds["temperature"], ds["q_mixingratio"] * units("kg/kg")) * 100  # for percent
    # ds["Td"] = mpcalc.dewpoint_from_relative_humidity(ds["temp"], ds["rh"])  # I don't need it now, evtl. there is an error in calc of rh...

    ds = ds.metpy.dequantify()
    if "temp" in vars_to_calc:
        ds["temp"]  = ds["temp"] - 273.15  # convert temp to °C
        ds["temp"] = ds['temp'].assign_attrs(units="°C", description="temperature")

    return ds


def create_ds_geopot_height_as_z_coordinate(ds):
    """
    create a new dataset with geopotential height as vertical coordinate for temperature for plotting
    :param ds:
    :return:
    :ds_new: new dataset with geopotential height as vertical coordinate
    """
    geometric_height = ds.z.compute()

    ds_new = xr.Dataset(
        data_vars=dict(
            th=(["time", "height"], ds.th.values),
            # p=(["time", "height"], ds.p.values),
        ),
        coords=dict(
            height=("height", geometric_height.values),
            time=("time", ds.time.values)
        ),
        attrs=dict(description="WRF data with geopotential height at mid of ds as vertical coordinate"))

    return ds_new


def read_wrf_fixed_point_and_time(day: int, hour: int, latitude: float, longitude: float, minute: int):
    """deprecated
    Read in WRF ACINN at a fixed time (hour, day, min) and location (lat, lon)

    :param day: day can be 15 or 16 (October)
    :param hour: can be from 12 to 00 for 15.October to 00 to 12 for 16. October
    :param latitude:
    :param longitude:
    :param minute: can be 00 or 30 (default is 00)
    """

    if day not in [15, 16]:
        raise ValueError("Only October day 15 or 16 is available!")

    day = str(day)

    formatted_hour = f"{hour:02d}"
    if minute not in [0, 30]:
        raise ValueError("Only 0 or 30 min are possibile. Values available only for full hour or half hour")
    formatted_min = f"{minute:02d}"

    date = f"201710{day}"
    my_time = f"{date}T{formatted_hour}{formatted_min}Z"
    filepath2 = f"{confg.wrf_folder}/WRF_ACINN_{date}/WRF_ACINN_20171015T1200Z_CAP02_3D_30min_1km_HCW_{my_time}.nc"

    ds = __open_wrf_dataset_my_version(filepath2)
    # salem_example_plots(ds) # could plot some salem 2D plots
    """ print variables and units
    for var_name, data_array in ds.data_vars.items():
        long_name = data_array.attrs.get('description', 'No long name available')
        units = data_array.attrs.get('units', 'Units not specified')
        print(f"- {var_name}: {long_name}, Units: {units}")
    """

    df = ds.salem.subset(geometry=Point(longitude, latitude),
                         crs='epsg:4236')
    df = df.isel(Time=0, south_north=0, west_east=0)  # select all isel =0
    df = df.rename_dims({"Time": "time", "bottom_top": "height"})  # rename dimensions to uniform names
    df = convert_calc_variables(df)

    return df


def rename_drop_vars(ds):
    """ deprecated, cause ds is created new due to lat/lon dimensions
    rename and drop variables for consistent naming

    :param ds:
    :return:
    """
    ds["Time"] = ds.time  # add correct Time value to coord
    ds["south_north"] = ds.lat
    ds["west_east"] = ds.lon
    ds = ds.drop_vars(["time", "lat", "lon"])  # before also included: "latitude", "longitude"
    ds = ds.rename({"Time": "time", "bottom_top": "height", "south_north": "lat", "west_east": "lon"})
    # rename dimensions to uniform names
    return ds


def unstagger_z_point(ds):
    """
    for unstaggering the geopotential height var, i.e. calculating it on the same levels as p and other
    vars are available, only for the point-read in: take mean between every 2nd level to compute it
    :param ds:
    :return:
    """
    z_unstag = ds.z.rolling(bottom_top_stag=2, center=True).mean()[1:]  # geopot height is on staggered variable: take
    # mean to compute it on unstaggered grid (height variable), where temp etc are calculated
    ds = ds.assign(z_unstag=(("height"), z_unstag.values))  # assign unstaggered variables as new data var (only height coord)
    return ds


def unstagger_z_domain(ds):
    """
    same as unstagger_z_point but only with lat&lon coords for full domain read in
    :param ds:
    :return:
    """
    z_unstag = ds.z.rolling(bottom_top_stag=2, center=True).mean()[1:]
    ds = ds.assign(z_unstag=(("height", "lat", "lon"), z_unstag.values))  # also has lat&lon vals
    return ds


def create_new_dataset(ds):
    """needed due to regridding, create a new dataset with lat, lon, height and time as coordinates
    only used for read wrf fixed time over full domain, for a single point it's not needed!

    """
    # Problem: we have lat & lon not as coordinates => need to redefine the dataset
    # extract 1D- coordinates
    lat_1d = ds['lat'][0, :, 0].values
    lon_1d = ds['lon'][0, 0, :].values

    # Erzeuge ein neues Dataset mit 1D-Koordinaten
    wrf = xr.Dataset(
        data_vars=dict(
            alb=(["time", "lat", "lon"], ds.alb.values),
            hfs= (["time", "lat", "lon"], ds.hfs.values),
            lfs=(["time", "lat", "lon"], ds.lfs.values),
            lwd=(["time", "lat", "lon"], ds.lwd.values),
            lwu=(["lat", "lon"], ds.lwu.values),
            p=(["height", "lat", "lon"], ds.p.values),
            swd=(["time", "lat", "lon"], ds.swd.values),
            swu=(["lat", "lon"], ds.swu.values),
            th=(["time", "height", "lat", "lon"], ds.th.values),
            tke=(["time", "height", "lat", "lon"], ds.tke.values),
            u=(["time", "height", "lat", "lon"], ds.u.values),
            v=(["time", "height", "lat", "lon"], ds.v.values),
            w=(["time", "height", "lat", "lon"], ds.w.values),
            z=(["height", "lat", "lon"], ds.z.values),
        ),
        coords=dict(
            height=("height", ds.bottom_top.values),
            time=("time", ds.Time.values),
            lat=("lat", lat_1d),
            lon=("lon", lon_1d)
        ),
        attrs=dict(description="WRF data with regridded lat/lon as coordinates")
    )
    return wrf

def generate_filenames():
    """
    create list of all wrf file names:
    :return:
    """
    hours_15 = [f"{hour:02d}{minute:02d}" for hour in range(12, 24) for minute in [0, 30]]  # 1200, 1230, ..., 2330 (list of strings)
    hours_16 = [f"{hour:02d}{minute:02d}" for hour in range(0, 12) for minute in [0, 30]] + ["1200"]  # 0000, 0030, ..., 1200
    wrf_files_15 = [confg.wrf_folder + f"/WRF_ACINN_20171015/WRF_ACINN_20171015T1200Z_CAP02_3D_30min_1km_HCW_20171015T"
                 + hour + "Z_regrid.nc" for hour in hours_15]
    wrf_files_16 = [confg.wrf_folder + f"/WRF_ACINN_20171016/WRF_ACINN_20171015T1200Z_CAP02_3D_30min_1km_HCW_20171016T"
                    + hour + "Z_regrid.nc" for hour in hours_16]
    wrf_files = wrf_files_15 + wrf_files_16  # list of path to all wrf files, not beautiful but works
    return wrf_files


def read_wrf_fixed_point(lat=47.259998, lon=11.384167, variables=["p", "temp", "th", "rho", "z"], height_as_z_coord=False):
    """calls fct to read and merge WRF files across multiple days and times for a specified location. (used for lidar plots)
    It is also possible to define the lowest_level = True, selects only lowest level
    then adjust dimensions to have time & height as coordinates
    & calculate vars like rh etc with metpy in convert_calc_variables

    :param variables: a variable list to keep only certain variables, if empty default is used
    :param lat: Latitude of the location.
    :param lon: Longitude of the location.
    :param height_as_z_coord: set mean geopot. height as height variable
    """
    # combined_ds = generate_datasets(lat, lon, start_day, end_day, variable_list=["th", "p", "time"])  # deleted by daniel

    wrf_files = generate_filenames()
    combined_ds, vars_to_calculate = open_wrf_mfdataset(filepaths=wrf_files, variables=variables)

    ds = combined_ds.sel(lat=lat, lon=lon, method= "nearest")  # index given point
    if "z_unstag" in variables:
        ds = unstagger_z_point(ds)
    ds = convert_calc_variables(ds, vars_to_calc=vars_to_calculate)
    ds = ds[variables]
    if height_as_z_coord:  # set unstaggered geopot. height as height coord. values
        ds["height"] = ds.z_unstag.values

    return ds.compute()


def read_wrf_fixed_time(day=16, hour=12, min=0, variables=["p", "temp", "th", "rho", "z"]):  # min_lat=46.5, max_lat=48.2, min_lon=9.2, max_lon=13
    """ reads 1 single WRF file at a spec. time over the full domain

    :param my_time: selected time

    :param lowest_level: Default False, but if True then select only lowest level
    :param variables: a variable list to keep only certain variables; default are those needed for VHD calc

    """
    time = datetime.datetime(2017, 10, day, hour, min, 00)
    filepath = (confg.wrf_folder + f"/WRF_ACINN_201710{time.day:02d}/WRF_ACINN_20171015T1200Z_CAP02_3D_30min_1km_HCW_201710"
                                   f"{time.day:02d}T{time.hour:02d}{time.minute:02d}Z_regrid.nc")

    ds, vars_to_calc = open_wrf_mfdataset(filepaths=filepath, variables=variables)
    if "z_unstag" in variables:
        ds = unstagger_z_domain(ds)
    ds = convert_calc_variables(ds, vars_to_calc=vars_to_calc)  # calculate variables like temp, rho, etc.
    ds = ds[variables]

    return ds.compute()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Qt5Agg')

    #wrf_plotting = create_ds_geopot_height_as_z_coordinate(wrf)
    #wrf_path = Path(confg.wrf_folder + "/WRF_temp_timeseries_ibk.nc")
    #wrf_plotting.to_netcdf(wrf_path, mode="w", format="NETCDF4")

    # wrf = read_wrf_fixed_point(lat=confg.ibk_villa["lat"], lon=confg.ibk_villa["lon"],
    #                            variables=["p", "temp", "th", "rho", "z", "z_unstag"], height_as_z_coord=True)
    wrf_extent = read_wrf_fixed_time(day=16, hour=4, min=0, variables=["hfs", "p", "temp", "th", "z", "z_unstag"])
    wrf_extent

    # what would be better to take as var for model topography? terrain height hgt or geometric height z for consistency
    # with other models?! ~ 20m difference...
    # wrf_extent.hgt.to_netcdf(confg.wrf_folder + "/WRF_geometric_height_3dlowest_level.nc", mode="w", format="NETCDF4")
    # wrf_tif = wrf_extent.rename({"lat": "y", "lon": "x", "hgt": "band_data"})  # rename for tif export
    # wrf_tif.rio.write_crs("EPSG:4326", inplace=True)  # add WGS84-projection
    # wrf_tif.isel(time=0).band_data.rio.to_raster(confg.wrf_folder + "/WRF_geometric_height_3dlowest_level.tif")  # for xdem calc of slope I need .tif file
