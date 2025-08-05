from pathlib import Path

import numpy as np
import xarray as xr

import sys
sys.path.append("/mnt/c/Users/eleme/Documents/1Uni_Laptop/model_comparison_codes")
# import importlib
# importlib.import_module(confg)

import confg
import read_wrf_helen
import read_ukmo
import cartopy.crs as ccrs


def reproject_um():
    um = read_ukmo.read_ukmo_fixed_time(time="2017-10-15T14:00:00", variable_list=["z", "th", "p", "q"])
    # are these the correct lats&lons of the UM model?
    lat = um.rotated_latitude_longitude.grid_north_pole_latitude + um.grid_latitude
    lon = (um.grid_longitude - 360) + um.rotated_latitude_longitude.grid_north_pole_longitude

    # define rotated pole crs using cartopy
    um_orig_crs = ccrs.RotatedPole(pole_longitude=um.rotated_latitude_longitude.grid_north_pole_longitude,
                                   pole_latitude=um.rotated_latitude_longitude.grid_north_pole_latitude)




def regrid_wrf_win():
    wrf = read_wrf_helen.read_wrf_fixed_time(my_time="2017-10-15T14:00:00", min_lon=min_lon_subset,  # _wsl
                                             max_lon=max_lon_subset,
                                             min_lat=min_lat_subset, max_lat=max_lat_subset)  #

    wrf_small = wrf[["hgt"]]
    # or use xESMF?
    # code written by ChatGPT: Zielgitter definieren, take same vals as for ICON regridding
    lon_new = np.arange(min_lon_subset, max_lon_subset, 0.01398)
    lat_new = np.arange(min_lat_subset, max_lat_subset, 0.00988)

    grid_out = xr.Dataset({'longitude': (['longitude'], list(lon_new)),
                           'latitude': (['latitude'], list(lat_new))})


def regrid_wrf():
    wrf = read_wrf_helen.read_wrf_fixed_time_wsl(my_time="2017-10-15T14:00:00", min_lon=min_lon_subset,  #
                                             max_lon=max_lon_subset,
                                             min_lat=min_lat_subset, max_lat=max_lat_subset)  #

    wrf_small = wrf[["hgt"]]
    # or use xESMF?
    # code written by ChatGPT: Zielgitter definieren, take same vals as for ICON regridding
    lon_new = np.arange(min_lon_subset, max_lon_subset, 0.01398)
    lat_new = np.arange(min_lat_subset, max_lat_subset, 0.00988)

    grid_out = xr.Dataset({'lon': (['lon'], list(lon_new), {"units": "degrees_east"}),
                           'lat': (['lat'], list(lat_new), {"units": "degrees_north"})})

    # from here not Win compatible code! need to run via WSL
    import xesmf as xe
    # Regridder erstellen (z.B. bilinear)
    regridder = xe.Regridder(wrf_small, grid_out, 'conservative')

    # Interpoliertes Dataset erzeugen
    ds_wgs84 = regridder(wrf)
    ds_wgs84.to_netcdf(Path("d/MSc_Arbeit/WRF_ACINN/WRF_ACINN_20171015" + "/WRF_2017-10-15T14:00:00_wgs84.nc"),
                       mode="w", format="NETCDF4")


if __name__ == '__main__':
    lat_ibk = 47.259998
    lon_ibk = 11.384167
    # wrf = read_wrf_fixed_point(lat=lat_ibk, lon=lon_ibk)
    # wrf_plotting = create_ds_geopot_height_as_z_coordinate(wrf)
    # wrf_path = Path(confg.wrf_folder + "/WRF_temp_timeseries_ibk.nc")
    # wrf_plotting.to_netcdf(wrf_path, mode="w", format="NETCDF4")
    min_lon_subset, max_lon_subset = 9.2, 13
    min_lat_subset, max_lat_subset = 46.5, 48.2

    reproject_um()

    # regrid_wrf_win()
    # regrid_wrf()


