from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe

import confg
import read_wrf_helen

if __name__ == '__main__':
    lat_ibk = 47.259998
    lon_ibk = 11.384167
    # wrf = read_wrf_fixed_point(lat=lat_ibk, lon=lon_ibk)
    # wrf_plotting = create_ds_geopot_height_as_z_coordinate(wrf)
    # wrf_path = Path(confg.wrf_folder + "/WRF_temp_timeseries_ibk.nc")
    # wrf_plotting.to_netcdf(wrf_path, mode="w", format="NETCDF4")
    min_lon_subset, max_lon_subset = 9.2, 13
    min_lat_subset, max_lat_subset = 46.5, 48.2

    wrf = read_wrf_helen.read_wrf_fixed_time(my_time="2017-10-15T14:00:00", min_lon=min_lon_subset,
                                             max_lon=max_lon_subset,
                                             min_lat=min_lat_subset, max_lat=max_lat_subset)  #
    # or use xESMF?
    # code written by ChatGPT: Zielgitter definieren, take same vals as for ICON regridding
    lon_new = np.arange(min_lon_subset, max_lon_subset, 0.01398)
    lat_new = np.arange(min_lat_subset, max_lat_subset, 0.00988)
    grid_out = xr.Dataset({'lon': (['lon'], lon_new),
                           'lat': (['lat'], lat_new)})

    # Regridder erstellen (z.B. bilinear)
    regridder = xe.Regridder(wrf, grid_out, 'conservative')

    # Interpoliertes Dataset erzeugen
    ds_wgs84 = regridder(wrf)
    ds_wgs84.to_netcdf(Path(confg.wrf_folder + "/WRF_2017-10-15T14:00:00_wgs84.nc"), mode="w", format="NETCDF4")
