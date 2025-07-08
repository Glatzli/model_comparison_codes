"""Read in the 3D ICON Model, re-written by Daniel

functions used from outside:
- read_icon_fixed_point() need dask to read it in, a lot of RAM used
- read_icon_fixed_point_and_time()
"""
import sys
from operator import concat

from xarray import decode_cf

sys.path.append("D:/MSc_Arbeit/model_comparison_codes")
import confg
import xarray as xr
import numpy as np
from functools import partial
import metpy.calc as mpcalc
from metpy.units import units
import datetime
from pathlib import Path



# 3 Mal ", dims=["height"] entfernt"
def convert_calc_variables(ds):
    """
    Converts and calculates meteorological variables for a xarray Dataset.

    Parameters:
    - df: A xarray Dataset containing the columns 'p' for pressure in Pa
          and 'th' for potential temperature in Kelvin.

    Returns:
    - A xarray Dataset with the original data and new columns:
      'pressure' in hPa and 'temperature' in degrees Celsius.
    """

    # Convert pressure from Pa to hPa
    ds['p'] = (ds['pres'] / 100.0) * units.hPa

    # calc pot temp
    ds["th"] = mpcalc.potential_temperature(ds['p'], ds["temp"] * units.kelvin)

    # convert temp to °C
    ds["temp"]  = (ds["temp"] - 273.15) * units.degC

    # ds['qv'] = ds["qv"] * units("kg/kg")  # originally has kg/kg

    # calculate relative humidity
    # ds['rh'] = mpcalc.relative_humidity_from_specific_humidity(ds['pressure'], ds["temp"], ds['qv']) * 100  # for percent

    # calculate dewpoint
    #ds["Td"] = mpcalc.dewpoint_from_specific_humidity(pressure = ds['pressure'],
    #                                                  specific_humidity = ds['qv']) # , temperature = ds["temp"]

    return ds.metpy.dequantify()  # remove units from the dataset


def create_ds_geopot_height_as_z_coordinate(ds):
    """
    create a new dataset with geopotential height as vertical coordinate for temperature for plotting, orig copied from
    AROME
    :param ds:
    :return:
    :ds_new: new dataset with geopotential height as vertical coordinate
    """
    geopot_height = ds.z
    # ds.z_ifc.isel(height_3=slice(1, 91))

    ds_new = xr.Dataset(  # somehow lat & lon doesn't work => w/o those coords
        data_vars=dict(
            th=(["time", "height"], ds.th.values),
            temp=(["time", "height"], ds.temp.values),
            p=(["time", "height"], ds.p.values),
            rho=(["time", "height"], ds.rho.values),
        ),
        coords=dict(
            height=("height", ds.z.isel(height_3=slice(1, 91)).values),
            # skip most upper level, different height coordinates => just trust in hannes' notes...
            time=("time", ds.time.values)
        ),
        attrs=dict(description="ICON data with z_ifc geometric height at half level center as vertical coordinate"))

    return ds_new


def find_min_index(ds_icon, lon, lat):
    """
    Distances are relatively short where the curvature of the Earth can be neglected (fast 0.04 seconds)
    deleted old function, still in 2TE version
    """
    # Convert degrees to radians for calculation
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    lon_diff_squared = (ds_icon.clon - lon_rad) ** 2
    lat_diff_squared = (ds_icon.clat - lat_rad) ** 2

    # Sum the squared differences to get squared Euclidean distances
    squared_distances = lon_diff_squared + lat_diff_squared

    # Find the index of the minimum squared distance
    min_idx = squared_distances.argmin()
    return min_idx.values

def read_full_icon(variant="ICON"):
    """Read the full ICON 3D model dataset for a given variant. ~8GB"""
    if variant == "ICON":
        ds_path = confg.icon_folder_3D + "/ICON_20171015_latlon.nc"
    elif variant == "ICON2TE":
        ds_path = confg.icon2TE_folder_3D + "/ICON2TE_20171015_latlon.nc"
    else:
        raise ValueError("wrong variant")
    icon_full = xr.open_dataset(ds_path, chunks="auto", engine="netcdf4")
    return icon_full

def rename_icon_variables(ds):

    ds = ds.rename({"z_ifc": "z"})
    return ds

def read_icon_fixed_point(lat, lon, variant="ICON"):
    """ Read ICON 3D model at a fixed point & compute variables"""
    icon_full = read_full_icon(variant=variant)

    icon_point = icon_full.sel(lat=lat, lon=lon, method="nearest")
    icon_point = convert_calc_variables(icon_point)
    icon = rename_icon_variables(ds=icon_point)  # rename z_ifc to z
    icon_point = icon_point.compute()
    return icon_point

def read_icon_fixed_time(day=16, hour=12, min=0, variant="ICON"):
    icon_full = read_full_icon(variant=variant)

    timestamp = datetime.datetime(2017, 10, day, hour, min, 00)
    icon = icon_full.sel(time=timestamp, height=90, height_3=91, method="nearest")
    icon = convert_calc_variables(icon)
    icon = rename_icon_variables(ds=icon)  # rename z_ifc to z
    icon = icon.compute()
    return icon

def read_icon_fixed_point_and_time_hexa(day, hour, lon, lat, variant="ICON"):
    """
    deprecated from hannes, use read_icon_fixed_point_multiple_hours instead
    Read Icon 3D model at a fixed point and a fixed time
    """

    if day not in [15, 16]:
        raise ValueError("Only October day 15 or 16 is available!")

    formatted_hour = f"{hour:02d}"

    if variant == "ICON":  # put together read icon & read icon2TE script
        icon_file = f'ICON_BLM-GUF_20171015T1200Z_CAP02_2D-3D_10min_1km_all_201710{day}T{formatted_hour}0000Z.nc'
        folder = confg.icon_folder_3D
    elif variant == "ICON2TE":
        icon_file = f'ICON_2TE_BLM-GUF_20171015T1200Z_CAP02_2D-3D_10min_1km_all_201710{day}T{formatted_hour}0000Z.nc'
        folder = confg.icon2TE_folder_3D
    else:
        print("invalid model variant, either ICON or ICON2TE")

    ds_icon = xr.open_dataset(f"{folder}/" + icon_file)

    min_idx = find_min_index(ds_icon, lon, lat)

    nearest_data = ds_icon.isel(ncells=min_idx).isel(time=0)

    return convert_calc_variables(nearest_data)  # calculate temp, pressure

def generate_icon_filenames_hexa(day, hours=[12], variant="ICON"):
    """
    creates a list of filenames for the half-hourly ICON data.

    Parameters:
    - day: Tag im Format '15' oder '16'.
    - hours: list of hours
    - variant: model variant, either "ICON" or "ICON2TE".

    Returns:
    - Eine Liste von Dateinamen als Strings.
    """
    if isinstance(hours, int):  # if a single hour is given, convert it to a list
        hours = [hours]

        filenames = [confg.icon_folder_3D +
            f"/{variant}_BLM-GUF_20171015T1200Z_CAP02_2D-3D_10min_1km_all_201710{day}T{hour:02d}{minute:02d}00Z.nc"
            for hour in hours for minute in [0, 30]
        ]
    elif variant == "ICON_2TE":
        filenames = [confg.icon2TE_folder_3D +
             f"/{variant}_BLM-GUF_20171015T1200Z_CAP02_2D-3D_10min_1km_all_201710{day}T{hour:02d}{minute:02d}00Z.nc"
             for hour in hours for minute in [0, 30]
         ]

    return filenames

def read_icon_fixed_point_multiple_hours_hexa(day=16, hours=[12], lon=11.4011756, lat=47.266076, variant="ICON"):  # , variables=["height", "time", "temp"]
    """ Read ICON 3D model at a fixed point with multiple hours """

    if day not in [15, 16]:
        raise ValueError("Only October day 15 or 16 is available!")

    icon_filepaths = generate_icon_filenames_hexa(day=day, hours=hours, variant=variant)
    if day == 16 and 12 in hours:
        icon_filepaths = icon_filepaths[:-1]  # remove last file, cause it's only available till 1200

    ds_icon = xr.open_mfdataset(icon_filepaths, combine = "by_coords", data_vars = 'minimal',
                             coords = 'minimal', compat = 'override', decode_timedelta=True)
                                # concat_dim="time", combine="nested", data_vars='minimal',
                                # coords='minimal', compat='override', decode_cf=False)
    # ds_icon = ds_icon[variables]
    min_idx = find_min_index(ds_icon, lon, lat)  # no clue what that's doing
    ds_icon = ds_icon.isel(ncells=min_idx)
    ds_icon = convert_calc_variables(ds_icon)

    return  ds_icon  # calculate temp, pressure

def read_icon_fixed_time_hexa(day=16, hour=[12], variant="ICON"):  # , variables=["height", "time", "temp"]
    """
    Read full ICON domain at fixed time
    probably interpolate it afterwards to regular lat/lon grid

    add variables subset to get smaller dataset...
    """
    if day not in [15, 16]:
        raise ValueError("Only October day 15 or 16 is available!")

    icon_filepaths = generate_icon_filenames_hexa(day=day, hours=hour, variant=variant)

    ds_icon = xr.open_mfdataset(icon_filepaths, combine = "by_coords", data_vars = 'minimal',
                             coords = 'minimal', compat = 'override', decode_timedelta=True)
                                # concat_dim="time", combine="nested", data_vars='minimal',
                                # coords='minimal', compat='override', decode_cf=False)
    # ds_icon = ds_icon[variables]
    ds_icon = convert_calc_variables(ds_icon)

    return  ds_icon  # calculate temp, pressure

def read_icon_fixed_point_hexa(nearest_grid_cell, day=16, variant="ICON"):
    """
    Reads ICON 3D datasets for a given day and a given grid cell
    NOTE: Since the files are large we need dask to not get a overflow in RAM used

    Parameters:
    - nearest_grid_cell: The index of the nearest cell

    Returns:
    - Combined xarray dataset along dimensions, with selected ICON variables.
    """
    if day not in [15, 16]:
        raise ValueError("Only October day 15 or 16 is available!")

    # Preprocess function to select data at a specific location
    def _preprocess(x):
        return x.isel(ncells=nearest_grid_cell)

    # Use open_mfdataset with the partial function as a preprocess argument
    partial_func = partial(_preprocess)

    if variant == "ICON":  # put together read icon & read icon2TE script
        file_pattern = f'ICON_BLM-GUF_20171015T1200Z_CAP02_2D-3D_10min_1km_all_201710{str(day)}T????00Z.nc'
        folder = confg.icon_folder_3D  # variable from confg-file
    elif variant == "ICON2TE":
        file_pattern = f'ICON_2TE_BLM-GUF_20171015T1200Z_CAP02_2D-3D_10min_1km_all_201710{str(day)}T????00Z.nc'
        folder = confg.icon2TE_folder_3D
    else:
        print("invalid model variant, either ICON or ICON2TE")

    # Load and concatenate datasets automatically by coordinates
    ds = xr.open_mfdataset(
        folder + f"/{file_pattern}",
        combine='by_coords',
        preprocess=partial_func
    )

    # Handling 'z_ifc' to include only in the first dataset
    if 'z_ifc' in ds.variables:
        z_ifc = ds['z_ifc'].isel(time=0).expand_dims('time')  # Get 'z_ifc' from the first time point
        ds = ds.drop_vars('z_ifc', errors='ignore')  # Drop 'z_ifc' from all datasets
        ds = ds.assign({'z_ifc': z_ifc})  # Reassign 'z_ifc' only for the first time point

    return ds


if __name__ == '__main__':
    lat_ibk = 47.259998
    lon_ibk = 11.384167

    model = "ICON"  # either "ICON" or "ICON2TE"
    # testing cdo generates nc files:
    # icon_latlon = xr.open_dataset(confg.icon_folder_3D + "/ICON_20171015_latlon.nc")

    icon_extent = read_icon_fixed_time(day=16, hour=12, min=0, variant="ICON")
    # save lowest level as nc file for topo plotting
    icon_extent.z.to_netcdf(confg.icon_folder_3D + "/ICON_geometric_height_3dlowest_level.nc", mode="w", format="NETCDF4")
    icon_extent.z.rio.to_raster(confg.icon_folder_3D + "/ICON_geometric_height_3dlowest_level.tif")  # for xdem calc of slope I need .tif file

    # icon_latlon = read_icon_fixed_point(lat=lat_ibk, lon=lon_ibk, variant=model)
    #icon_plotting = create_ds_geopot_height_as_z_coordinate(icon_latlon)
    #icon_path = Path(confg.model_folder + f"/{model}/" + f"{model}_temp_p_rho_timeseries_ibk.nc")
    #icon_plotting.to_netcdf(icon_path, mode="w", format="NETCDF4")


    # with this code the ICON model is read in, one specific latitude is extracted and saved as new dataset with geometric
    # height as vert coord
    """
    icon15 = read_icon_fixed_point_multiple_hours(day=15, hours=np.arange(12, 24), lon=lon_ibk, lat=lat_ibk, variant="ICON")
    icon16 = read_icon_fixed_point_multiple_hours(day=16, hours=np.arange(0, 13), lon=lon_ibk, lat=lat_ibk, variant="ICON")
    variables = ["th", "temp", "z_ifc"]  # "temp", "pres", "u", "v", "w",
    icon = xr.concat([icon15[variables], icon16[variables]], dim="time")
    """
    # create a new dataset with geometric height z_ifc as vertical coordinate
    """
    icon_plotting = create_ds_geopot_height_as_z_coordinate(icon)
    icon_path = Path(confg.model_folder + "/ICON/" + "ICON_temp_timeseries_ibk.nc")
    icon_plotting.to_netcdf(icon_path, mode="w", format="NETCDF4")
    """
    # icon 2te
    # icon_2te_path = Path(confg.model_folder + "/ICON2TE/" + "ICON_2TE_temp_timeseries_ibk.nc")
    # icon_plotting.to_netcdf(icon_2te_path, mode="w", format="NETCDF4")
    # icon_plotting