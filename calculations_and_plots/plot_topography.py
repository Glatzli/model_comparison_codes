"""Script to compare and plot the heights of the AROME model and the real world (DEM)
Plot also the locations of the stations
"""

import json
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import geopandas
import geopandas as gpd
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
from matplotlib.legend import Legend
from matplotlib.pyplot import ylabel
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import pycrs
import rasterio
import xarray as xr
from fiona.crs import from_epsg

from rasterio.mask import mask
from rasterio.plot import show
from shapely.geometry import box
from colorspace import terrain_hcl, qualitative_hcl, sequential_hcl

import confg
from AROME.profile_radiosonde import station_files_zamg
from confg import JSON_TIROL, TIROL_DEMFILE, cities, stations_ibox, MOMMA_stations_PM, dir_PLOTS, \
    ec_station_names, filepath_arome_height, dem_file_hobos_extent, data_folder
import confg as cfg


def read_plot_clip_tirol():
    """Plot the digital elevation data and coordinates of boundaries of Tirol
    """
    # Using geopandas for that
    AUT = geopandas.read_file(JSON_TIROL)
    print(AUT.NAME_1)  # all Bundesländer of Austria
    TIROL = AUT.loc[(AUT.NAME_1 == "Tirol")]  # can change here the Bundesland for example "Vorarlberg"

    # DEM90 digital elevation model
    dem = rasterio.open(TIROL_DEMFILE)

    print(dem.bounds)
    # plot it
    fig, ax = plt.subplots()
    TIROL.boundary.plot(ax=ax, edgecolor='C3')
    image_hidden = ax.imshow(dem.read()[0], cmap="Greys_r")
    fig.colorbar(image_hidden, ax=ax)
    show(dem, cmap="Greys_r", ax=ax)

    return dem


def read_dem_xarray(file_name = TIROL_DEMFILE):

    dem = xr.open_dataset(file_name, engine="rasterio")
    dem = dem.rename({"x": "lon", "y": "lat", "band_data":"height"})  # rename the coordinates to lon and lat
    return dem


def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""

    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def create_clipped_dem(data):
    """clip raster file data to a defined extend for HOBOS"""
    # select some latitudes and longitudes around the center point(which in this case is the Innsbruck airport)

    padding = 0.02
    min_lon = 11.3398110103 - padding
    max_lon = 11.4639758751 + padding
    min_lat = 47.2403724414 - padding
    max_lat = 47.321

    bbox = box(min_lon, min_lat, max_lon, max_lat)

    # clip the DEM
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))
    geo = geo.to_crs(crs=data.crs.data)
    coords = getFeatures(geo)  # get the coordinates of the geometry in such a format that rasterio wants them

    out_img, out_transform = mask(dataset=data, shapes=coords, crop=True)
    out_meta = data.meta.copy()

    # epsg_code = int(data.crs.data['init'][5:])
    out_meta.update(
        {"driver": "GTiff", "height": out_img.shape[1], "width": out_img.shape[2], "transform": out_transform,
         "crs": pycrs.parse.from_epsg_code(4326).to_proj4()})  # hard-coded EPSG code for WGS84 cause data['init'] not available
    # original: "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()})
    with rasterio.open(dem_file_hobos_extent, "w", **out_meta) as dest:
        dest.write(out_img)

    fig, ax = plt.subplots()

    # Display the clipped image
    plt.imshow(out_img[0], extent=[min_lon, max_lon, min_lat, max_lat], cmap='viridis', origin="upper")
    plt.contour(out_img[0], origin="upper")
    plt.colorbar(label='Value')  # Add colorbar
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Clipped Raster Image')


def add_lat_lon_plot(ax, df, rotate=None):
    """add description of longitude and latutide to plot"""
    # Define the xticks for longitude
    ax.set_xticks(np.arange(np.round(df["lon"].min()), df["lon"].max(), 0.5), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)

    # Define the yticks for latitude
    ax.set_yticks(np.arange(np.round(df["lat"].min()), df["lat"].max(), 0.5), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)

    # Rotate x-axis labels if rotation angle is provided
    """
    if rotate is not None:
        for label in ax.get_xticklabels():
            label.set_rotation(rotate)"""


def plot_stations_and_AROME_height(df, model_name, ext_lat, ext_lon):
    """plot model topography as contour plot in the innvalley, and on top as scatter points all stations"""
    fig, ax = plt.subplots(figsize=(16, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.coastlines()

    basis = 10
    min_ds_900 = np.floor(df["z"].min().item() / basis) * basis
    max_ds_900 = np.floor(df["z"].max().item() / basis) * basis
    dist = 50

    my_range = np.arange(min_ds_900, max_ds_900 + 1,
                         dist)  # interval of 60m

    # Get the terrain colormap
    terrain_cmap = plt.get_cmap('terrain')

    # Normalize your data to get corresponding colormap indices
    norm = plt.Normalize(vmin=df["z"].min(), vmax=df["z"].max())

    # Create contour lines with colors mapped from the terrain colormap
    cs = ax.contour(df['lon'], df['lat'], df["z"], my_range,
                    transform=ccrs.PlateCarree(),
                    colors=[terrain_cmap(norm(level)) for level in my_range])

    # Adding a colorbar to represent the mapping from your contour levels to the terrain colors
    sm = plt.cm.ScalarMappable(cmap=terrain_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, ticks=my_range, boundaries=my_range)
    cbar.set_label('Elevation in meter')

    add_lat_lon_plot(df=df, ax=ax, rotate=90)

    # Attention if I adjust the extent I have to adjust the markersize
    station_counter = 0
    for city_name, city_info in cities.items():
        # check if station location point is inside the extent
        if (ext_lat.start <= city_info["lat"] <= ext_lat.stop) and (
                ext_lon.start <= city_info["lon"] <= ext_lon.stop):

            station_counter += 1
            if city_name == "Innsbruck Airport":
                ax.plot(city_info["lon"], city_info["lat"], marker='o', color='red', markersize=10, alpha=1,
                        transform=ccrs.PlateCarree(), label=f"{city_name} (Radiosonde)", linestyle='none')
            else:
                ax.plot(city_info["lon"], city_info["lat"], marker='o', color='red', markersize=10, alpha=1,
                        transform=ccrs.PlateCarree(), label=f"{city_name}", linestyle='none')
        else:
            print(f"{city_name} is not inside lat-lon extent")
            continue

    for ibox_name, ibox_info in stations_ibox.items():
        ax.plot(ibox_info["longitude"], ibox_info["latitude"], marker='*', color='gold', markersize=15, alpha=1,
                transform=ccrs.PlateCarree(), label=f"{ibox_info['name']}", linestyle='none')
    for momma_name, momma_info in MOMMA_stations_PM.items():
        ax.plot(momma_info["longitude"], momma_info["latitude"], marker='H', color='black', markersize=10, alpha=1,
                transform=ccrs.PlateCarree(), label=f"{momma_info['name']}", linestyle='none')

    for ec_name, ec_info in ec_station_names.items():
        ax.plot(ec_info["lon"], ec_info["lat"], marker='s', markersize=15, alpha=0.7, color="orange",
                transform=ccrs.PlateCarree(), label=f"{ec_info['name']}", linestyle='none')

    plt.title(f"{model_name} height contour around Innsbruck airport, distance {dist} m")
    handles, labels = ax.get_legend_handles_labels()
    zamg_legend = Legend(ax, handles[:station_counter], labels[:station_counter], loc="upper left",
                         title="TAWES/DWD stations")
    ibox_legend = Legend(ax, handles[station_counter:station_counter + len(stations_ibox)],
                         labels[station_counter:station_counter + len(stations_ibox)], loc="lower left",
                         title="Ibox stations")
    momma_legend = Legend(ax, handles[station_counter + len(stations_ibox):station_counter + len(stations_ibox) + len(
        MOMMA_stations_PM)], labels[station_counter + len(stations_ibox):station_counter + len(stations_ibox) + len(
        MOMMA_stations_PM)], loc="lower right", title="MOMMMA stations")

    ec_legend = Legend(ax, handles[station_counter + len(stations_ibox) + len(MOMMA_stations_PM):],
                       labels[station_counter + len(stations_ibox) + len(
                           MOMMA_stations_PM):], loc="upper right", title="EC flux stations")

    ax.add_artist(zamg_legend)
    ax.add_artist(ibox_legend)
    ax.add_artist(momma_legend)
    ax.add_artist(ec_legend)
    plt.savefig(f"{dir_PLOTS}/contour_stations_with_AROME_height/contour_{model_name}_zoom.png")

def plot_stations_and_AROME_height_filled(df, model_name, ext_lat, ext_lon, ext_lat_small, ext_lon_small):
    """plot model topography as filled imshow plot in the innvalley, and on top as scatter points all stations"""
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    cs = ax.contourf(df['lon'], df['lat'], df["z"], levels=levels_filled,
                    transform=ccrs.PlateCarree(), cmap = pal.cmap())
    ax.contour(df['lon'], df['lat'], df["z"], levels=levels_contour_lines,
                    transform=ccrs.PlateCarree(), color = "grey", linewidths=0.2)
    rect = Rectangle((ext_lon_small.start, ext_lat_small.start), ext_lon_small.stop - ext_lon_small.start, ext_lat_small.stop -ext_lat_small.start,
                     linewidth=1, edgecolor='black', facecolor='gray', alpha=0.3)
    ax.add_patch(rect)

    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1)  # add national borders

    # Adding a colorbar to represent the mapping from your contour levels to the terrain colors
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('Geopotential Height [m]')

    add_lat_lon_plot(df=df, ax=ax, rotate=90)  # add rotated lat & lon axis labels

    # Plot station points
    station_counter = 0
    # Attention if I adjust the extent I have to adjust the markersize
    station_counter = 0
    for city_name, city_info in cities.items():
        # check if station location point is inside the extent
        if (ext_lat.start <= city_info["lat"] <= ext_lat.stop) and (
                ext_lon.start <= city_info["lon"] <= ext_lon.stop):

            station_counter += 1
            if city_name == "Innsbruck Airport":
                radiosonde = ax.plot(city_info["lon"], city_info["lat"], marker='*', color='orange', markersize=10,
                        transform=ccrs.PlateCarree(), label="Radiosonde", linestyle='none')
            elif city_name == 'Innsbruck Uni':
                lidar = ax.plot(city_info["lon"], city_info["lat"], marker='D', color=darkred_hcl, markersize=8,
                                transform=ccrs.PlateCarree(), label="LIDAR", linestyle='none')
            else:
                aws = ax.plot(city_info["lon"], city_info["lat"], marker='o', color=black_hcl, markersize=7, label="AWS")
        else:
            print(f"{city_name} is not inside lat-lon extent")
            continue

    for momma_name, momma_info in MOMMA_stations_PM.items():
        #if (ext_lat.start <= momma_info["latitude"] <= ext_lat.stop) and (
        #        ext_lon.start <= momma_info["longitude"] <= ext_lon.stop):
        momma = ax.plot(momma_info["longitude"], momma_info["latitude"], marker='o', color=black_hcl, markersize=8,
                            transform=ccrs.PlateCarree(), label="MOMMA", linestyle='none')

    for ec_name, ec_info in ec_station_names.items():
        ec = ax.plot(ec_info["lon"], ec_info["lat"], marker='s', markersize=6, color=black_hcl,  # , alpha=0.7
                transform=ccrs.PlateCarree(), label="EC station", linestyle='none')


    # ax.legend(handles=[tawes[0], momma[0], ec[0]], labels=["AWS", "T/RH Logger", "EC station"], bbox_to_anchor=(-0.2, 0.5))  # for legend outside the plot
    # ax.legend(handles=[momma[0], ec[0], lidar[0], radiosonde[0]], bbox_to_anchor=(-0.2, 1))  # , labels=["AWS", "EC station", "LIDAR", "Radiosonde", "T/RH Logger"]
    length_scalebar = 10  # length of the scalebar in km
    lon_extent = calculate_lon_extent_for_km(lat_ibk, length_scalebar)
    scalebar = AnchoredSizeBar(ax.transData, size=lon_extent, label=f'{length_scalebar} km', loc='lower right',
                               frameon=False, fontproperties=fontprops)
    ax.add_artist(scalebar)

    plt.tight_layout()
    plt.savefig(f"{dir_PLOTS}topography_plots/contour_{model_name}_filled.svg")

def plot_stations_detail(df, model_name, ext_lat, ext_lon):
    """plot model topography as filled imshow plot in the innvalley, and on top as scatter points all stations"""
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    # plt.rc('text', usetex=True)

    cs = ax.contourf(df['lon'], df['lat'], df["z"], levels=levels_filled,
                     transform=ccrs.PlateCarree(), cmap=pal.cmap())
    ax.contour(df['lon'], df['lat'], df["z"], levels=levels_contour_lines,
               transform=ccrs.PlateCarree(), color=gray_hcl, linewidths=0.2)
    thick_contours = ax.contour(df['lon'], df['lat'], df["z"], levels=levels_thick_contour_lines,
               transform=ccrs.PlateCarree(), color=gray_hcl, linewidths=0.5)
    ax.clabel(thick_contours, inline=True, fontsize=10, fmt='%d')

    # ax.add_feature(cfeature.BORDERS, edgecolor='grey', linewidth=0.5, label)  # add national borders

    # remove colorbar for detail plot
    # cbar = plt.colorbar(cs, ax=ax)
    # cbar.set_label('Geopotential Height [m]')

    # add_lat_lon_plot(df=df, ax=ax, rotate=90)  # add rotated lat & lon axis labels

    # Plot station points
    sp_north = list(np.arange(14, 21)) + list(np.arange(38, 46))
    sp_south = list(np.arange(10, 13)) + list(np.arange(34, 38))
    hobo_sp_north = hobo.isel(STATION_KEY=sp_north)
    hobo_sp_south = hobo.isel(STATION_KEY=sp_south)
    # Attention if I adjust the extent I have to adjust the markersize
    station_counter = 0
    for city_name, city_info in cities.items():
        # check if station location point is inside the extent
        if (ext_lat.start <= city_info["lat"] <= ext_lat.stop) and (
                ext_lon.start <= city_info["lon"] <= ext_lon.stop):

            station_counter += 1
            if city_name == "Innsbruck Airport":
                radiosonde = ax.plot(city_info["lon"], city_info["lat"], marker='*', color='orange', markersize=10,
                                     transform=ccrs.PlateCarree(), label="Radiosonde",
                                     linestyle='none')
            elif city_name == 'Innsbruck Uni':
                lidar = ax.plot(city_info["lon"], city_info["lat"], marker='D', color=darkred_hcl, markersize=8,
                                transform=ccrs.PlateCarree(), label="LIDAR & HATPRO", linestyle='none')
            else:
                aws = ax.plot(city_info["lon"], city_info["lat"], marker='o', color=black_hcl, markersize=7)
        else:
            print(f"{city_name} is not inside lat-lon extent")
            continue

    for momma_name, momma_info in MOMMA_stations_PM.items():
        if (ext_lat.start <= momma_info["latitude"] <= ext_lat.stop) and (
                ext_lon.start <= momma_info["longitude"] <= ext_lon.stop):
            momma = ax.plot(momma_info["longitude"], momma_info["latitude"], marker='o', color=black_hcl, markersize=8,
                            transform=ccrs.PlateCarree(), label="AWS", linestyle='none')

    for i, hobo_name in enumerate(hobo.hobo_id.values):  # plot slope profile north
        if (ext_lat.start  < hobo.lat[i] < ext_lat.stop) and (  # check if this hobo station is inside the extent
                ext_lon.start  < hobo.lon[i] < ext_lon.stop):  # (otherwise the plot would get too big...)
            if hobo_name in hobo_sp_north.hobo_id.values:
                # plot the hobo station points
                sp_north = ax.plot(hobo.lon.values[i - 1], hobo.lat.values[i - 1], marker='P', color=darkred_hcl, markersize=8,
                                      transform=ccrs.PlateCarree(), label=r"SP${_N}$", linestyle='none')
            elif hobo_name in hobo_sp_south.hobo_id.values:
                sp_south = ax.plot(hobo.lon.values[i], hobo.lat.values[i], marker='P', color=darkblue_hcl, markersize=8,
                                      transform=ccrs.PlateCarree(), label=r"SP${_S}$", linestyle='none')
            else:
                hobo_handle = ax.plot(hobo.lon.values[i - 1], hobo.lat.values[i - 1], marker='P', color=gray_hcl,
                                   markersize=8, transform=ccrs.PlateCarree(), label=f"HOBO", linestyle='none')
    ax.plot(hobo.lon.values[36], hobo.lat.values[36], marker='P', color=darkblue_hcl, markersize=8,
                       transform=ccrs.PlateCarree(), label=f"SP_S", linestyle='none')


    for ec_name, ec_info in ec_station_names.items():
        ec = ax.plot(ec_info["lon"], ec_info["lat"], marker='s', markersize=6, color=black_hcl,  # , alpha=0.7
                     transform=ccrs.PlateCarree(), label="EC station", linestyle='none')

    # ax.legend(handles=[tawes[0], momma[0], ec[0]], labels=["AWS", "T/RH Logger", "EC station"], bbox_to_anchor=(-0.2, 0.5))  # for legend outside the plot

    plt.xlim(ext_lon.start, ext_lon.stop)
    plt.ylim(ext_lat.start, ext_lat.stop)

    length_scalebar = 2  # length of the scalebar in km
    lon_extent = calculate_lon_extent_for_km(lat_ibk, length_scalebar)
    scalebar = AnchoredSizeBar(ax.transData, size=lon_extent, label=f'{length_scalebar} km', loc='lower right',
                               frameon=False, fontproperties=fontprops)
    ax.add_artist(scalebar)

    ax.legend(handles=[momma[0], ec[0], lidar[0], radiosonde[0], hobo_handle[0], sp_north[0], sp_south[0]],
              bbox_to_anchor=(-0.1, 1))
    #  labels=["EC station", "LIDAR", "Radiosonde", "AWS", "T/RH Logger", ""]
    plt.tight_layout()
    plt.savefig(f"{dir_PLOTS}topography_plots/contour_{model_name}_detail.svg")

def plot_lidar_and_Modelgrid_points(dataset, model_name, ext_lat, ext_lon):
    """Function to plot lidar station points and Model grid points"""
    """plot model topography as contour plot in the innvalley"""
    fig, ax = plt.subplots(figsize=(16, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.coastlines()

    basis = 10
    min_ds_900 = np.floor(dataset["z"].min().item() / basis) * basis
    max_ds_900 = np.floor(dataset["z"].max().item() / basis) * basis
    dist = 50

    my_range = np.arange(min_ds_900, max_ds_900 + 1,
                         dist)  # interval of 60m

    # Get the terrain colormap
    terrain_cmap = plt.get_cmap('terrain')

    # Normalize your data to get corresponding colormap indices
    norm = plt.Normalize(vmin=dataset["z"].min(), vmax=dataset["z"].max())

    # Create contour lines with colors mapped from the terrain colormap
    cs = ax.contour(dataset['longitude'], dataset['latitude'], dataset["z"], my_range,
                    transform=ccrs.PlateCarree(),
                    colors=[terrain_cmap(norm(level)) for level in my_range])

    # Adding a colorbar to represent the mapping from your contour levels to the terrain colors
    sm = plt.cm.ScalarMappable(cmap=terrain_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, ticks=my_range, boundaries=my_range)
    cbar.set_label('Elevation in meter')

    add_lat_lon_plot(df=dataset, ax=ax, rotate=90)

    station_counter = 0
    for city_name, city_info in cities.items():
        # check if station location point is inside the extent
        if (ext_lat.start <= city_info["lat"] <= ext_lat.stop) and (
                ext_lon.start <= city_info["lon"] <= ext_lon.stop):

            station_counter += 1
            if city_name == "Innsbruck Airport":
                ax.plot(city_info["lon"], city_info["lat"], marker='o', color=darkred_hcl, markersize=10, alpha=1,
                        transform=ccrs.PlateCarree(), label=f"{city_name} (Radiosonde)", linestyle='none')
            else:
                ax.plot(city_info["lon"], city_info["lat"], marker='o', color=darkred_hcl, markersize=10, alpha=1,
                        transform=ccrs.PlateCarree(), label=f"{city_name}", linestyle='none')
        else:
            print(f"{city_name} is not inside lat-lon extent")
            continue

    for station_lidar in ["SL88", "SL75", "SL74", "SLXR142"]:
        lidar_station_df = xr.open_dataset(
            f"{data_folder}Observations/LIDAR/{station_lidar}_vad_l2/{station_lidar}_20171015_vad.nc")

        ax.plot(lidar_station_df.attrs['lon'], lidar_station_df.attrs['lat'], marker='*', markersize=15,
                alpha=1,
                transform=ccrs.PlateCarree(), label=f"LIDAR: {station_lidar}", linestyle='none')

    plt.title(f"{model_name} height contour around Innsbruck airport, distance {dist} m")

    # Extract latitude and longitude values from the dataset
    latitude = dataset.latitude.values
    longitude = dataset.longitude.values

    # Create a meshgrid of latitude and longitude
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)

    ax.scatter(lon_grid, lat_grid, label="Grid points of model", marker="+", alpha=1, color="k",
               transform=ccrs.PlateCarree())

    handles, labels = ax.get_legend_handles_labels()
    zamg_legend = Legend(ax, handles[:station_counter], labels[:station_counter], loc="upper left",
                         title="TAWES/DWD stations")

    lidar_legend = Legend(ax, handles[station_counter: station_counter + len(station_lidar)],
                          labels[station_counter: station_counter + len(station_lidar)], loc="upper right",
                          title="LIDAR station")

    grid_legend = Legend(ax, handles[station_counter + len(station_lidar):],
                         labels[station_counter + len(station_lidar):],
                         loc="lower right",
                         title="LAT-LON GRID AROME")

    ax.add_artist(zamg_legend)
    ax.add_artist(lidar_legend)
    ax.add_artist(grid_legend)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Lidar station in comparison with AROME gridpoints')
    plt.savefig(f"{dir_PLOTS}/contour_stations_with_AROME_height/lidar_stations.png")


def plot_height_diff_LOWI(ds, model_name):
    """Plot the real height from a DEM in the background, and the model height as text on the gridpoints"""
    # Assuming 'ds' is your xarray dataset and 'out_file' is the path to your DEM file
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_title(f'{model_name} Latitude-Longitude Grid with the airport in the center')

    margin_x = 0.03
    margin_y = 0.02
    extent = [station_files_zamg["LOWI"]["lon"] - margin_x, station_files_zamg["LOWI"]["lon"] + margin_x,
              station_files_zamg["LOWI"]["lat"] - margin_y,
              station_files_zamg["LOWI"]["lat"] + margin_y]

    bbox = box(extent[0], extent[2], extent[1], extent[3])
    geo = gpd.GeoDataFrame({'geometry': [bbox]}, crs="EPSG:4326")

    # Open the raster file and clip it to the extent
    with rasterio.open(dem_file_hobos_extent) as clipped:
        geo = geo.to_crs(crs=clipped.crs.data)  # Ensure the CRS matches the source
        out_image, out_transform = mask(clipped, shapes=geo.geometry, crop=True)
        data = out_image[0]  # Assuming single band data

        # Calculate vmin and vmax from the clipped data
        vmin = data.min()
        vmax = data.max()

        # Plot the DEM data with dynamic scaling
        image = ax.imshow(data, cmap="Greys", vmin=vmin, vmax=vmax,
                          extent=[extent[0], extent[1], extent[2], extent[3]],
                          transform=ccrs.PlateCarree())

        # Add a colorbar to the plot
        cbar = fig.colorbar(image, ax=ax, extend='both')
        cbar.set_label('Altitude (m)')

    # Set the extent of the plot to focus tightly around the area of interest
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Add map features for context
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # Plot the grid lines more efficiently
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--', linewidth=0.5, color='black', alpha=0.5)

    # Disable labels on the top and right edges
    gl.top_labels = False
    gl.right_labels = False

    # Set the x and y spacing for the gridlines
    gl.xlocator = mticker.FixedLocator(np.arange(11.335, 11.395, 0.01))
    gl.ylocator = mticker.FixedLocator(np.arange(47.245, 47.275, 0.01))

    # Mark the position of Innsbruck airport
    ax.plot(station_files_zamg["LOWI"]["lon"], station_files_zamg["LOWI"]["lat"], marker='x', color='red',
            markersize=10, alpha=0.7,
            transform=ccrs.PlateCarree(), label="Position of Innsbruck airport",
            linestyle='none')  # trick that legend draws actually no line

    # Filter the dataset for the extent and convert to dataframe
    df = ds.sel(latitude=slice(extent[2], extent[3]),
                longitude=slice(extent[0], extent[1])).to_dataframe().reset_index()

    # Plot scatter points with colormap based on height values
    sc = ax.scatter('longitude', 'latitude', c='z', s=10, cmap='jet', data=df,
                    norm=plt.Normalize(df['z'].min(), df['z'].max()), transform=ccrs.PlateCarree(),
                    label='AROME Model Height')

    # Add colorbar for the scatter plot
    plt.colorbar(sc, ax=ax, extend='both').set_label('AROME Model Height')

    # Annotate points with height values
    for _, row in df.iterrows():
        ax.text(row['longitude'], row['latitude'], f"{row['z']:.0f}", fontsize=6, transform=ccrs.PlateCarree())

    plt.legend()
    plt.savefig(f"{dir_PLOTS}/contour_stations_with_AROME_height/AROME_height_ibk_airport.png")


def select_extent_around_LOWI(ds, lat_degree, lon_degree):
    """function to select the extent that should be plotted around Innsbruck airport"""
    extent_lat = slice(station_files_zamg["IAO"]["lat"] - lat_degree, station_files_zamg["IAO"]["lat"] + lat_degree)
    extent_lon = slice(station_files_zamg["IAO"]["lon"] - lon_degree, station_files_zamg["IAO"]["lon"] + lon_degree)
    return ds.sel(lat=extent_lat, lon=extent_lon), extent_lat, extent_lon  # nz=90,  (for hannes' topo .nc file)



def calculate_lon_extent_for_km(latitude, km):
    """
    Berechnet die Ausdehnung in Grad Längengrad für eine gegebene Entfernung in Kilometern.
    by ChatGPT...

    Parameters:
    latitude (float): Die gegebene Breite in Grad.
    km (float): Die Entfernung in Kilometern.

    Returns:
    float: Die Ausdehnung in Grad Längengrad.
    """
    # Radius der Erde in Kilometern
    earth_radius = 6371

    # Erdumfang in Kilometern
    earth_circumference = 2 * math.pi * earth_radius

    # Länge eines Längengrads in Kilometern an der gegebenen Breite
    lon_km = math.cos(math.radians(latitude)) * earth_circumference / 360

    # Ausdehnung in Grad Längengrad für die gegebene Entfernung
    lon_extent = km / lon_km
    return lon_extent

def calculate_km_for_lon_extent(latitude, lon_extent_deg):
    """
    Berechnet die Entfernung in km für eine gegebene Längendifferenz (in Grad) an einer bestimmten Breite.
    """
    earth_radius = 6371  # km
    earth_circumference = 2 * math.pi * earth_radius
    lon_km = math.cos(math.radians(latitude)) * earth_circumference / 360
    return lon_extent_deg * lon_km



if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Qt5Agg')

    lat_ibk = 47.259998  # only for maßstab
    fontprops = fm.FontProperties(size=10)

    # lon = calculate_lon_extent_for_km(lat_ibk, 1)
    # lon2 = calculate_lon_extent_for_km(lat_ibk, 1.5)

    pal = sequential_hcl("Terrain")  # palettes for terrain topo
    pal2 = qualitative_hcl("Dark 3")
    levels_filled = np.arange(500, 3500, 250)
    levels_contour_lines = np.arange(500, 3600, 100)
    levels_thick_contour_lines = np.arange(500, 3500, 500)

    darkblue_hcl = sequential_hcl(palette="Blues 3").colors()[3]  # colors for slope profiles
    darkred_hcl = sequential_hcl(palette="Reds 3").colors()[4]
    black_hcl = sequential_hcl(palette="Grays").colors()[0]
    gray_hcl = sequential_hcl(palette="Grays").colors()[3]

    #data = read_plot_clip_tirol()  # show whole Tyrol map
    # create_clipped_dem(data)  # create DEM90 digital elevation model
    hobo = xr.open_dataset(confg.data_folder + "201707_hobo.nc")

    # read_plot_clip_tirol()  # read & plot DEM of tirol

    # height_lowest = xr.open_dataset(confg.dir_AROME + "AROME_geopot_height_3dlowest_level.nc")  # open AROME MODEL Height lowest level
    height_lowest = xr.open_dataset(confg.icon_folder_3D + "/ICON_geometric_height_3dlowest_level.nc") # open ICON MODEL Height lowest level

    ds_small_extent, extent_lat_small, extent_lon_small = select_extent_around_LOWI(height_lowest, lat_degree=0.08, lon_degree=0.08)
    ds_small_extent2, extent_lat_small2, extent_lon_small2 = select_extent_around_LOWI(height_lowest, lat_degree=0.1, lon_degree=0.1)
    ds_large_extent, extent_lat_large, extent_lon_large = select_extent_around_LOWI(height_lowest, lat_degree=0.4, lon_degree=0.4)

    # Plot stations as points, underneath as contour the AROME Model height
    plot_stations_and_AROME_height_filled(ds_large_extent, "ICON", ext_lat=extent_lat_large, ext_lon=extent_lon_large,
                                          ext_lat_small = extent_lat_small, ext_lon_small = extent_lon_small)

    plot_stations_detail(ds_small_extent2, "ICON", ext_lat=extent_lat_small, ext_lon=extent_lon_small)

    # Plot the LOWI station point in center, and grid heights around it with real heights of DEM in the background
    # plot_height_diff_LOWI(ds_large_extent, "AROME")


    # plot the locations of Lidar
    #plot_lidar_and_Modelgrid_points(ds_small_extent, "AROME", ext_lat=extent_lat_small,
    #                                ext_lon=extent_lon_small)  # mit dist_degree = 0.05
    plt.show()


