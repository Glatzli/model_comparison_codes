"""Script to read in WRF ACINN data
Note: it is a sort of WRF Data, but still different than the WRF_ETH files.

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
from shapely.geometry import Point, Polygon
from wrf import combine_dims
from xarray.backends.netCDF4_ import NetCDF4DataStore
import warnings
import confg
from metpy.units import units
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import matplotlib
matplotlib.use('Qt5Agg')

def __open_wrf_dataset_my_version(file, **kwargs):
    """Updated Function from salem, the problem is that our time has no units (so i had to remove the check in the original function)
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
    ds['longitude'] = ds.salem.grid.x_coord  # adapted
    ds['latitude'] = ds.salem.grid.y_coord

    # add pyproj string everywhere
    ds.attrs['pyproj_srs'] = ds.salem.grid.proj.srs
    for v in ds.data_vars:
        ds[v].attrs['pyproj_srs'] = ds.salem.grid.proj.srs

    return ds

def __open_wrf_dataset_my_version_open_mfdataset(filepaths):
    """changed function to work for multiple files with xarray's mfdataset
    specifically for read_wrf_fixed_point

    Parameters
    ----------
    file : str
        the path to the WRF file
    **kwargs : optional
        Additional arguments passed on to ``xarray.open_mfdataset``.

    Returns
    -------
    an xarray Dataset
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]  # Einzelnen Pfad in eine Liste umwandeln

    datasets = []
    for filepath in filepaths:  # iterate through all paths, do commands for each file
        nc = netCDF4.Dataset(filepath)
        nc.set_auto_mask(False)

        # unstagger variables
        for vn, v in nc.variables.items():
            if wrftools.Unstaggerer.can_do(v):
                nc.variables[vn] = wrftools.Unstaggerer(v)

        # Hinzufügen diagnostischer Variablen
        for vn in wrftools.var_classes:
            cl = getattr(wrftools, vn)
            if vn not in nc.variables and cl.can_do(nc):
                nc.variables[vn] = cl(nc)

        # Trick xarray mit unserem benutzerdefinierten NetCDF
        datasets.append(NetCDF4DataStore(nc))

    # trick xarray with our custom netcdf
    ds = xr.open_mfdataset(datasets, concat_dim="Time", combine="nested", data_vars='minimal',
                                coords='minimal', compat='override', decode_cf=False)

    # somehow time-dimension didn't work correctly, so add it by hand...
    ds["Time"] = [pd.Timestamp('2017-10-15T12:00:00') + pd.Timedelta(minutes=m) for m in ds.time.values]

    # remove time dimension to lon lat
    for vn in ['XLONG', 'XLAT']:
        try:
            v = ds[vn].isel(Time=0)
            ds[vn] = xr.DataArray(v.values, dims=['south_north', 'west_east'])
        except (ValueError, KeyError):
            pass

    # add cartesian coords
    ds['longitude'] = ds.salem.grid.x_coord  # adapted
    ds['latitude'] = ds.salem.grid.y_coord

    # add pyproj string everywhere
    ds.attrs['pyproj_srs'] = ds.salem.grid.proj.srs
    for v in ds.data_vars:
        ds[v].attrs['pyproj_srs'] = ds.salem.grid.proj.srs

    return ds

def salem_example_plots(ds):
    """Make some example plots with salem plotting functions need to have some slice of lat lon
    Could be used in future to plot 2D maps with a certain variable and a certain extent
    """
    hmap = ds.salem.get_map(cmap='topo')

    hmap.set_data(ds['alb'])
    hmap.set_points(confg.station_files_zamg["LOWI"]["lon"], confg.station_files_zamg["LOWI"]["lat"])
    hmap.set_text(confg.station_files_zamg["LOWI"]["lon"], confg.station_files_zamg["LOWI"]["lat"], 'Innsbruck', fontsize=17)
    hmap.visualize()

    psrs = 'epsg:4236'  # http://spatialreference.org/ref/epsg/wgs-84-utm-zone-30n/

    ds.attrs['pyproj_srs'] = psrs

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

def convert_calc_variables(ds):
    """
    changed from ICON, original from Hannes was insanely slow
    cals temp (degC), rh (%), pressure (hpa) & evtl Td (degC)

    Parameters:
    - ds: A xarray Dataset containing the columns 'p' for pressure in Pa
          and 'th' for potential temperature in Kelvin.

    Returns:
    - A xarray Dataset with the original data and new columns:
      'pressure' in hPa and 'temperature' in degrees Celsius.
    """
    # Convert pressure from Pa to hPa
    ds['pressure'] = (ds['p'] / 100) * units.hPa

    # calculate temp in K
    ds["temp"] = mpcalc.temperature_from_potential_temperature(ds['pressure'], ds["th"] * units("K"))

    ds["rh"] = mpcalc.relative_humidity_from_mixing_ratio(ds["pressure"], ds["temp"], ds["q_mixingratio"] * units("kg/kg")) * 100  # for percent
    # ds["Td"] = mpcalc.dewpoint_from_relative_humidity(ds["temp"], ds["rh"])  # I don't need it now, evtl. there is an error in calc of rh...

    ds = ds.metpy.dequantify()
    ds["temp"]  = ds["temp"] - 273.15  # convert temp to °C

    return ds

def read_wrf_fixed_point_and_time(day: int, hour: int, latitude: float, longitude: float, minute: int):
    """Read in WRF ACINN at a fixed time (hour, day, min) and location (lat, lon)
    I probably won't need this function!

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



def generate_datasets(lat=47.259998, lon=11.384167, start_day=15, end_day=16,
                      variable_list=["u", "v", "z", "th", "time", "p", "q_mixingratio"], lowest_level=False):
    """read in wrf datasets, put them into list together & merge them w xarray"""
    datasets = []
    for day in range(start_day, end_day + 1):
        date = f"201710{day:02d}"
        day_folder = f"WRF_ACINN_{date}"
        for hour in range(14 if day == 15 else 0, 24 if day == 15 else 13):  # adapted logic
            for minute in [0, 30]:
                if (hour == 12) & (minute == 30):
                    continue  # skip 12:30 for day 16
                file_name = f"WRF_ACINN_201710{15:02d}T{12:02d}{0:02d}Z_CAP02_3D_30min_1km_HCW_{date}T{hour:02d}{minute:02d}Z.nc"
                filepath = f"{confg.wrf_folder}/{day_folder}/{file_name}"

                ds = __open_wrf_dataset_my_version(filepath)
                # Directly subset the dataset for the given point and time
                ds = ds.salem.subset(geometry=Point(lon, lat), crs='epsg:4236')
                if lowest_level:  # from hannes, I probably won't need only lowest level
                    ds = ds.isel(Time=0, south_north=0, west_east=0, bottom_top=0)
                else:
                    ds = ds.isel(Time=0, south_north=0, west_east=0)

                ds = ds[variable_list]  # define variables
                datasets.append(ds)

    return xr.concat(generate_datasets(), dim='Time')


def read_wrf_fixed_point(lat=47.259998, lon= 11.384167, variable_list=["u", "v", "z", "th", "time", "p", "q_mixingratio"],
                         lowest_level=False):
    """calls fct to read and merge WRF files across multiple days and times for a specified location. (used for lidar plots)
    It is also possible to define the lowest_level = True, selects only lowest level
    then adjust dimensions to have time & height as coordinates
    & calculate vars like rh etc with metpy in convert_calc_variables

    :param lowest_level: Default False, but if True then select only lowest level
    :param variable_list: a variable list to keep only certain variables, if empty default is used
    :param lat: Latitude of the location.
    :param lon: Longitude of the location.
    """


    # combined_ds = generate_datasets(lat, lon, start_day, end_day, variable_list=["th", "p", "time"])  # deleted by daniel
    # create list of all wrf file names:
    hours_15 = [f"{hour:02d}{minute:02d}" for hour in range(12, 24) for minute in [0, 30]]  # 1200, 1230, ..., 2330 (list of strings)
    hours_16 = [f"{hour:02d}{minute:02d}" for hour in range(0, 12) for minute in [0, 30]] + ["1200"]  # 0000, 0030, ..., 1200
    wrf_files_15 = [confg.wrf_folder + f"/WRF_ACINN_20171015/WRF_ACINN_20171015T1200Z_CAP02_3D_30min_1km_HCW_20171015T"
                 + hour + "Z.nc" for hour in hours_15]
    wrf_files_16 = [confg.wrf_folder + f"/WRF_ACINN_20171016/WRF_ACINN_20171015T1200Z_CAP02_3D_30min_1km_HCW_20171016T"
                    + hour + "Z.nc" for hour in hours_16]
    wrf_files = wrf_files_15 + wrf_files_16  # list of path to all wrf files, not beautiful but works

    combined_ds = __open_wrf_dataset_my_version_open_mfdataset(wrf_files)

    ds = combined_ds.salem.subset(geometry=Point(lon, lat), crs='epsg:4236')  #from Hannes' code, subset to point
    ds = ds.isel(south_north=0, west_east=0)  # select point, from here on only 2D-dataset: Time & bottom_top

    # assign data variable as coordinate
    ds = ds.drop_vars(["time"])
    ds = ds.rename({"Time": "time", "bottom_top": "height"})  # rename dimensions to uniform names
    # assign bottom top as coordinate, and give it the values of z (height) m
    #z_values_at_time0 = combined_ds.isel(Time=0)['z']

    ds = convert_calc_variables(ds)
    return ds


def read_wrf_fixed_time(my_time="2017-10-15T14:00:00", min_lon=11, max_lon=13, min_lat=47, max_lat=48, variable_list=["time","u", "v", "z", "th", "p", "alb", "q_mixingratio"]):  #,lowest_level=False
    """Read and merge WRF files across multiple days and times for a specified location. (used for lidar plots)
    It is also possible to define the lowest_level = True, selects only lowest level

    :param my_time: selected time
    :param min_lon, max_lon, min_lat, max_lat: minimum and maximum latitude and longitude of Box
    :param lowest_level: Default False, but if True then select only lowest level
    :param variable_list: a variable list to keep only certain variables, if it is NONE, then preselection is done

    """
    box_polygon = Polygon(
        [(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat), (min_lon, min_lat)])

    time = pd.to_datetime(my_time)

    filepath = (confg.wrf_folder + f"/WRF_ACINN_201710{time.day:02d}/WRF_ACINN_20171015T1200Z_CAP02_3D_30min_1km_HCW_201710"
                                   f"{time.day:02d}T{time.hour:02d}{time.minute:02d}Z.nc")

    ds = __open_wrf_dataset_my_version(filepath)
    ds["Time"] = ds.time  # add correct Time value to coord
    ds = ds.drop_vars(["time"])
    ds = ds.rename({"Time": "time", "bottom_top": "height"}) # rename dimensions to uniform names
    ds = ds.salem.subset(geometry=box_polygon, crs='epsg:4236')  # .isel(Time=0)

    ds = convert_calc_variables(ds)
    return ds

if __name__ == '__main__':
    #wrf = read_wrf_fixed_point(lat=47.259998, lon=11.384167)
    #wrf.rh.isel(time=0).plot(y="height")

    wrf = read_wrf_fixed_time(my_time="2017-10-15T14:00:00", min_lon=12, max_lon=13, min_lat=49, max_lat=50)

    # df = read_wrf_fixed_point_and_time(day=16, hour=3, latitude=confg.station_files_zamg["IAO"]["lat"],
    #                               longitude=confg.station_files_zamg["IAO"]["lon"], minute=0)



    # print(df)
    #print(read_wrf_fixed_point(longitude=11.3857,
    #                           latitude=47.2640, lowest_level=True))

    #salem_example_plots(ds)
    #plt.show()
    wrf
