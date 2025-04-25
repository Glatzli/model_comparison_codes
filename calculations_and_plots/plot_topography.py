import pathlib, netCDF4, cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import read_in_arome
import read_wrf_helen
import confg
import xarray as xr
import numpy as np
from colorspace import terrain_hcl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


def plot_topography(ds, model_name="AROME"):
    """plot model topography as contour plot in the innvalley"""
    pal = terrain_hcl()
    # box to plot:
    min_lat = 46.5
    max_lat = 48
    min_lon = 10.5
    max_lon = 12.5
    levels = np.arange(0, 3500, 500)
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    match model_name:  # need to distinguish cause slightly different dataset setup...
        case "AROME":
            lat_name = "latitude"
            lon_name = "longitude"
            height_variable = "z"
            cs = ax.contourf(ds[lon_name], ds[lat_name], ds[height_variable].isel(height=0), cmap=pal.cmap(), levels=levels,
                             transform=ccrs.PlateCarree())
        case "ICON":
            height_variable = "z_ifc"  # geometric height at half level center

        case "UKMO":
            height_variable = "z"  # geopotential height

        case "WRF":
            lat_name = "south_north"
            lon_name = "west_east"
            height_variable = "z"  # terrain height
            cs = ax.contourf(ds[lon_name], ds[lat_name], ds[height_variable].isel(height=0), cmap=pal.cmap(), levels=levels,
                             transform=ccrs.PlateCarree())  # plot ends up being smth i don't understand, maybe it's rotated?!
      # pal = terrain_hcl()
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1)  # add national borders
    ax.scatter(confg.station_files_zamg["IAO"]["lon"], confg.station_files_zamg["IAO"]["lat"],  # add innsbruck marker
               label=confg.station_files_zamg["IAO"]["name"][0:-4], marker=".", s=20, color="black")
    ax.text(confg.station_files_zamg["IAO"]["lon"], confg.station_files_zamg["IAO"]["lat"],
            confg.station_files_zamg["IAO"]["name"][0:-4])  # plot name
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xticks(np.arange(min_lon, max_lon, 0.5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(min_lat, max_lat, 0.5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    #plt.xlabel("longitude")
    #plt.ylabel("latitude")

    # Adding a colorbar to represent the mapping from your contour levels to the terrain colors
    # sm = plt.cm.ScalarMappable(cmap="terrain", norm=norm)
    # sm.set_array([])
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('Elevation in meter')
    cbar.set_ticks(levels)
    plt.savefig(confg.dir_PLOTS + f"{model_name}_topography.png", dpi=300)


if __name__ == '__main__':
    ds = read_in_arome.read_in_arome_fixed_time()
    # ds = read_wrf_helen.read_wrf_fixed_time(min_lon=7, max_lon=15, min_lat=46, max_lat=48)

    # Plot stations as points, underneath as contour the AROME Model height
    plot_topography(ds, model_name="AROME")  # mit dist_degree = 0.4
    plt.show()
    print("susi")
