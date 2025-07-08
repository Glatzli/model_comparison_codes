"""
In this script some functinons should be defined that calculate the valley heat deficit (VHD) from the potential temperature
for each model. VHD will be calc between lowest model level & Hafelekar at xxxx m.

At first calc VHD for grid point of Innsbruck.
"""

import sys
from read_icon_model_3D import read_icon_fixed_time
sys.path.append("D:/MSc_Arbeit/model_comparison_codes")
import importlib
import read_in_arome
import read_icon_model_3D
import read_ukmo
# importlib.reload(read_icon_model_3D)
import read_wrf_helen
# importlib.reload(read_in_arome)
import confg
import xarray as xr
import numpy as np
import matplotlib
from plot_topography import calculate_km_for_lon_extent
import matplotlib.pyplot as plt
import pandas as pd
from colorspace import diverging_hcl



def read_dem_xarray(file_name=confg.TIROL_DEMFILE):
    """reads DEM=digital elevation model as an xarray dataset from a .tiff file
    enables to have measured heights as a dataset"""

    dem = xr.open_dataset(file_name, engine="rasterio")
    dem = dem.rename({"x": "lon", "y": "lat", "band_data": "height"})  # rename the coordinates to lon and lat
    return dem


def choose_gpe(ds, lat_ngp, lon_ngp):
    """choose grid point ensemble (GPE) around NGP (nearest grid point) from each dataset with indices of lat and lon
    written by ChatGPT"""
    lat_idx = ds.lat.to_index().get_indexer([lat_ngp], method="nearest")[0]
    lon_idx = ds.lon.to_index().get_indexer([lon_ngp], method="nearest")[0]
    # choose GPE by indices aroung NGP
    gpe = ds.isel(
        lat=slice(max(lat_idx-1, 0), lat_idx+2),
        lon=slice(max(lon_idx-1, 0), lon_idx+2)
    )

    return gpe


# def select_pcgp(lat, lon, ds_extent):

def calculate_slope(elevation_data, x_res):
    """
    Calculates slope angle (degrees) and aspect (degrees) from elevation data.

    Args:
        elevation_data (np.ndarray): 2D NumPy array of elevation values.
        x_res (float): Resolution of the DEM in the x direction (meters).

    Returns:
        - slope (np.ndarray): Slope angle in degrees.
    """
    # Calculate gradients using finite difference method
    px, py = np.gradient(elevation_data, x_res)
    slope = np.sqrt(px ** 2 + py ** 2)
    slope_deg = np.degrees(np.arctan(slope))
    """
    chat GPT try:
    slope_deg = np.degrees(np.arctan(slope))
    # Calculate slope in radians
    slope_radians = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    # Calculate aspect in radians
    aspect_radians = np.arctan2(-dz_dx, dz_dy)
    # Convert radians to degrees
    slope_degrees = np.degrees(slope_radians)
    aspect_degrees = np.degrees(aspect_radians)
    # Ensure aspect is within 0-360 degrees
    aspect_degrees = (aspect_degrees + 360) % 360
    """
    return slope * 100, slope_deg  # slope in percent

def read_topo_calc_slope(dem_array):
    """
    general function to read topography of DEM and models, calculate slope and save it (needed for PCGP selection)!
    :param ds:
    :return:
    """
    # dem_yres_m = (dem_array.lat[1].values - dem_array.lat[0].values) * 1.11 * 10**5  # 1 deg lat = 111 km approx.
    dem_xres_deg = dem_array.lon[1].values - dem_array.lon[0].values
    dem_xres_m = calculate_km_for_lon_extent(latitude=dem_array.lat[0].values,
                                             lon_extent_deg=dem_xres_deg) * 1000  # convert to meters

    slope, slope_deg = calculate_slope(elevation_data=dem_array, x_res=dem_xres_m)
    return slope



def calc_vhd_single_point(ds_point):
    ds_below_hafelekar = ds_point.where(ds_point.height <= hafelekar_height, drop=True)
    th_hafelekar = ds_below_hafelekar.isel(height=0).th
    vhd_point = c_p*((th_hafelekar - ds_below_hafelekar.th.isel(height=slice(1, 100))) * ds_below_hafelekar.rho).sum(dim="height")  # pot temp deficit

    # th_deficit[th_deficit == th_deficit.max()]  # max pot temp deficit at 01UTC in icon model


if __name__ == '__main__':
    matplotlib.use('Qt5Agg')
    lat_ibk = 47.259998
    lon_ibk = 11.384167
    hafelekar_height = 2279  # m, highest HOBO from https://zenodo.org/records/4672313 hobo dataset
    c_p = 1005  # J/(kg*K), specific heat capacity of air at constant pressure
    # icon_ibk_ngp = read_icon_model_3D.read_icon_fixed_point(lat=lat_ibk, lon=lon_ibk, variant="ICON")  # ngp = nearest grid point

    dem = read_dem_xarray()  # read DEM
    # dem_elevation = dem.isel(band=0).height.compute().values  # index to have only 2 coordinates: lat, lon

    model_topo = xr.open_dataset(confg.icon_folder_3D + "/ICON_geometric_height_3dlowest_level.nc")
    icon_topo_array = model_topo.z # icon topography as DataArray
    slope = read_topo_calc_slope(icon_topo_array)
    model_topo["slope"] = (("lat", "lon"), slope)


    gpe = choose_gpe(ds=model_topo, lat_ngp=lat_ibk, lon_ngp=lon_ibk)

    icon_ibk_ngp = xr.open_dataset(confg.icon_folder_3D + "/ICON_temp_p_rho_timeseries_ibk.nc")
    calc_vhd_single_point(icon_ibk_ngp)

