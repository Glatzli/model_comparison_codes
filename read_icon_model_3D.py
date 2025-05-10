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


def read_icon_fixed_point(nearest_grid_cell, day=16, variant="ICON"):
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
    ds['pressure'] = (ds['pres'] / 100.0) * units.hPa

    # calc pot temp
    ds["th"] = mpcalc.potential_temperature(ds['pressure'], ds["temp"] * units.kelvin)

    # convert temp to °C
    ds["temp"]  = (ds["temp"] - 273.15) * units.degC

    ds['qv'] = ds["qv"] * units("kg/kg")  # originally has kg/kg

    # calculate relative humidity
    ds['rh'] = mpcalc.relative_humidity_from_specific_humidity(ds['pressure'], ds["temp"], ds['qv']) * 100  # for percent

    # calculate dewpoint
    #ds["Td"] = mpcalc.dewpoint_from_specific_humidity(pressure = ds['pressure'],
    #                                                  specific_humidity = ds['qv']) # , temperature = ds["temp"]

    return ds.metpy.dequantify()  # remove units from the dataset

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


def read_icon_fixed_point_and_time(day, hour, lon, lat, variant="ICON"):
    """
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

def read_icon_fixed_point_multiple_hours(day=16, hours=[12], lon=11.4011756, lat=47.266076, variant="ICON"):  # , variables=["height", "time", "temp"]
    """ Read ICON 3D model at a fixed point with multiple hours """

    if day not in [15, 16]:
        raise ValueError("Only October day 15 or 16 is available!")

    formatted_hours = [f"{hour:02d}" for hour in hours]  # create list of strings w hours

    if variant == "ICON":
        icon_files = [confg.icon_folder_3D + (f'/ICON_BLM-GUF_20171015T1200Z_CAP02_2D-3D_10min_1km_all_201710'
                                              f'{day}T' + hour + '0000Z.nc') for hour in formatted_hours]
    elif variant == "ICON2TE":
        icon_files = [confg.icon2TE_folder_3D + (f'/ICON_2TE_BLM-GUF_20171015T1200Z_CAP02_2D-3D_10min_1km_all_201710'
                                                 f'{day}T' + hour + '0000Z.nc') for hour in formatted_hours]
    else:
        print("invalid model variant, either ICON or ICON2TE")

    ds_icon = xr.open_mfdataset(icon_files, concat_dim="time", combine="nested", data_vars='minimal',
                                coords='minimal', compat='override', decode_cf=False) # combine='by_coords')
    # ds_icon = ds_icon[variables]
    min_idx = find_min_index(ds_icon, lon, lat)  # no clue what that's doing
    ds_icon = ds_icon.isel(ncells=min_idx)

    return convert_calc_variables(ds_icon)  # calculate temp, pressure

if __name__ == '__main__':
    icon15 = read_icon_fixed_point_multiple_hours(day = 15, hours = range(12, 18), lon = 11.4011756, lat = 47.266076,
                                                  variant="ICON")
    icon15