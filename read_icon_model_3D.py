"""Read in the 3D ICON Model

functions used from outside:
- read_icon_fixed_point() need dask to read it in, a lot of RAM used
- read_icon_fixed_point_and_time()
"""
import sys
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
        folder = icon_folder_3D  # variable from confg-file
    elif variant == "ICON2TE":
        file_pattern = f'ICON_2TE_BLM-GUF_20171015T1200Z_CAP02_2D-3D_10min_1km_all_201710{str(day)}T????00Z.nc'
        folder = icon2TE_folder_3D
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
def convert_calc_variables(df):
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
    df['pressure'] = df['pres'] / 100.0
    p = df['pressure'].values * units.hPa

    # Calculate temperature from potential temperature
    temp_C = df["temp"] - 273.15
    df["temperature"] = temp_C

    # calculate specific + relativ humidity
    temp_values = temp_C.values * units.degC
    specific_humidity = df["qv"].values * 1000 * units("g/kg")
    df['specific_humidity'] = xr.DataArray(data=specific_humidity.magnitude)

    # the variables that go into mpcalc have to Arrays (Quantitys) without Dimension (important to take .values before)
    relative_humidity = mpcalc.relative_humidity_from_specific_humidity(p, temp_values, specific_humidity).to('percent')
    df['relative_humidity'] = xr.DataArray(data=relative_humidity.magnitude)

    # calculate dewpoint
    rh_values = df["relative_humidity"].values * units.percent
    dewpoint = mpcalc.dewpoint_from_relative_humidity(temp_values, rh_values).to("degC")
    df['dewpoint'] = xr.DataArray(data=dewpoint.magnitude)

    return df

def find_min_index(ds_icon, my_lon, my_lat):
    """
    Distances are relatively short where the curvature of the Earth can be neglected (fast 0.04 seconds)
    deleted old function, still in 2TE version
    """
    # Convert degrees to radians for calculation
    lon_rad = np.radians(my_lon)
    lat_rad = np.radians(my_lat)

    lon_diff_squared = (ds_icon.clon - lon_rad) ** 2
    lat_diff_squared = (ds_icon.clat - lat_rad) ** 2

    # Sum the squared differences to get squared Euclidean distances
    squared_distances = lon_diff_squared + lat_diff_squared

    # Find the index of the minimum squared distance
    min_idx = squared_distances.argmin()
    return min_idx.values

def read_icon_fixed_point_and_time(day, hour, my_lon, my_lat, variant="ICON"):
    """
    Read Icon 3D model at a fixed point and a fixed time

    """

    if day not in [15, 16]:
        raise ValueError("Only October day 15 or 16 is available!")

    formatted_hour = f"{hour:02d}"

    if variant == "ICON":  # put together read icon & read icon2TE script
        icon_file = f'ICON_BLM-GUF_20171015T1200Z_CAP02_2D-3D_10min_1km_all_201710{day}T{formatted_hour}0000Z.nc'
        folder = icon_folder_3D
    elif variant == "ICON2TE":
        icon_file = f'ICON_2TE_BLM-GUF_20171015T1200Z_CAP02_2D-3D_10min_1km_all_201710{day}T{formatted_hour}0000Z.nc'
        folder = icon2TE_folder_3D
    else:
        print("invalid model variant, either ICON or ICON2TE")

    ds_icon = xr.open_dataset(f"{folder}/" + icon_file)

    min_idx = find_min_index(ds_icon, my_lon, my_lat)

    nearest_data = ds_icon.isel(ncells=min_idx).isel(time=0)

    return convert_calc_variables(nearest_data)  # calculate temp, pressure