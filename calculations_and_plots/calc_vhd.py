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
import xdem
import matplotlib.pyplot as plt
import pandas as pd
from colorspace import terrain_hcl, qualitative_hcl, sequential_hcl


def choose_gpe(ds, lat_ngp, lon_ngp):
    """choose grid point ensemble (GPE) around NGP (nearest grid point) from each dataset with indices of lat and lon
    written by ChatGPT"""
    lat_idx = ds.y.to_index().get_indexer([lat_ngp], method="nearest")[0]
    lon_idx = ds.x.to_index().get_indexer([lon_ngp], method="nearest")[0]
    # choose GPE by indices aroung NGP
    gpe = ds.isel(
        y=slice(max(lat_idx-1, 0), lat_idx+2),
        x=slice(max(lon_idx-1, 0), lon_idx+2)
    )

    return gpe


def calculate_slope_numpy(elevation_data, x_res):
    """
    Calculates slope angle (degrees) from elevation data using finite difference method & NumPy.

    Args:
        elevation_data (np.ndarray): 2D NumPy array of elevation values.
        x_res (float): Resolution of the DEM in the x (lon) direction [meters].

    Returns:
        - slope (np.ndarray): Slope angle in percent and degrees.
    """
    px, py = np.gradient(elevation_data, x_res)
    slope = np.sqrt(px ** 2 + py ** 2)
    slope_deg = np.degrees(np.arctan(slope))

    return slope * 100, slope_deg  # slope in percent


def calculate_slope(filepath):
    """
    Goal: calculate slope and aspect from elevation data using xdem
    :param elevation_data:
    :return:
    """
    model = xr.open_dataset(filepath, engine="rasterio")
    model_xres_deg = model.x[1].values - model.x[0].values
    model_xres_m = calculate_km_for_lon_extent(latitude=model.y[int(len(model.y)/2)].values,  # take middle latitude of dataset
                                             lon_extent_deg=model_xres_deg) * 1000  # convert to meters

    slope, slope_deg = calculate_slope_numpy(elevation_data=model.isel(band=0).band_data.values, x_res=model_xres_m)
    dem = xdem.DEM(filepath)  # , transform=transform
    aspect = xdem.terrain.aspect(dem)
    model["slope"] = (("y", "x"), slope)  # add slope to dem dataset
    model["aspect"] = (("y", "x"), aspect.data.data)
    return slope, aspect, model


def plot_height_slope_aspect(height_da, slope, aspect_raster, modeltype, title_height="Height", title_slope="Slope", title_aspect="Aspect"):
    """
    Plottet Höhe (xarray.DataArray), Slope (np.ndarray) und Aspect (Raster-Objekt) nebeneinander.
    """
    # Aspect-Daten extrahieren und fehlende Werte maskieren
    aspect = np.array(aspect_raster.data, dtype=np.float32)
    aspect = np.ma.masked_where(np.isnan(aspect) | (aspect == getattr(aspect_raster, "nodata", -99999)), aspect)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    if modeltype == "dem":  # somehow DEM is flipped -> choose different origin for imshow than for models
        origin_pic = "upper"
    else:
        origin_pic = "lower"

    im0 = axes[0].imshow(height_da.values, origin=origin_pic, cmap=pal.cmap())
    axes[0].set_title(title_height)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Height [m]")

    im1 = axes[1].imshow(slope, origin=origin_pic, cmap=pal.cmap())
    axes[1].set_title(title_slope)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Slope [°]")

    im2 = axes[2].imshow(aspect, origin=origin_pic, cmap="twilight", vmin=0, vmax=360)
    axes[2].set_title(title_aspect)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Aspect [°]")

    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    plt.show()

def read_topo_calc_slope(dem_array):
    """
    überflüssig?!

    general function to read topography of DEM and models, calculate slope and save it (needed for PCGP selection)!
    :param ds:
    :return:
    """
    # dem_yres_m = (dem_array.lat[1].values - dem_array.lat[0].values) * 1.11 * 10**5  # 1 deg lat = 111 km approx.
    dem_xres_deg = dem_array.lon[1].values - dem_array.lon[0].values
    dem_xres_m = calculate_km_for_lon_extent(latitude=dem_array.lat[0].values,
                                             lon_extent_deg=dem_xres_deg) * 1000  # convert to meters

    slope, slope_deg = calculate_slope_numpy(elevation_data=dem_array, x_res=dem_xres_m)
    return slope


def read_dem_xarray(filename):
    """
    überflüssig?!

    reads DEM=digital elevation model as an xarray dataset from a .tiff file
    enables to have measured heights as a dataset"""

    dem = xr.open_dataset(filename, engine="rasterio")
    dem = dem.rename({"x": "lon", "y": "lat", "band_data": "height"})  # rename the coordinates to lon and lat
    return dem


def calc_vhd_single_point(ds_point):
    ds_below_hafelekar = ds_point.where(ds_point.height <= hafelekar_height, drop=True)
    th_hafelekar = ds_below_hafelekar.isel(height=0).th
    vhd_point = c_p*((th_hafelekar - ds_below_hafelekar.th.isel(height=slice(1, 100))) * ds_below_hafelekar.rho).sum(dim="height")  # pot temp deficit

    # th_deficit[th_deficit == th_deficit.max()]  # max pot temp deficit at 01UTC in icon model
    return vhd_point

def calculate_select_pcgp(gpe_model, gpe_dem):
    """
    computes and selects the physically consistent grid point (PCGP) from a model grid point ensemble (GPE) and DEM-ensemble
    An ensemble consists of 1 nearest grid point in the middle and it's 8 surrounding points
    more details can be found in Simonet et al. (2025)  https://doi.org/10.21203/rs.3.rs-6050730/v1
    here only implemented with slope angle & aspect ratio calculated from model/dem height,
    implementation of land use is not possible due to missing measurements at the sites & variables in my model runs
    all variables are named in small letters for easier writing...

    :param gpe_model: grid point ensemble from a model (arome, icon, ...)
    :param gpe_dem: grid point ensemble from DEM
    :return:
    """
    # calc AD slope:
    ad_slope = np.abs(gpe_model.slope.values - gpe_dem.slope.values)  # calculate AD_beta i.e. slope
    ad_slope_n = ad_slope / ad_slope.max()  # AD_beta,n
    # calc AD aspect, use circular deviation Simonet et al. (2025) on p.20 (Equation 7)
    # diff > 180 degrees:
    aspect_identifier_big = np.abs(gpe_model.aspect.values - gpe_dem.aspect.values) > 180  # identifies which formula to use
    model_aspect_calc_big = gpe_model.aspect.where(aspect_identifier_big)
    dem_aspect_calc_big = gpe_dem.aspect.where(aspect_identifier_big)
    ad_aspect_big = ((model_aspect_calc_big.values - dem_aspect_calc_big.values) *
                     np.abs(1 - 360 / (np.abs(model_aspect_calc_big.values - dem_aspect_calc_big.values))))
    ad_aspect_big[np.isnan(ad_aspect_big)] = 0  # replace NaN with 0

    # 2. version of aspect calculation, if diff <= 180 degrees
    aspect_identifier_small = np.invert(aspect_identifier_big)
    ad_aspect_small = np.abs(gpe_model.aspect.where(aspect_identifier_small).values -
                             gpe_dem.aspect.where(aspect_identifier_small).values)
    ad_aspect_small[np.isnan(ad_aspect_small)] = 0  # replace NaN with 0
    ad_aspect = ad_aspect_big + ad_aspect_small  # add both aspect differences together to have 1 array with AD_aspect values

    ad = (ad_slope + ad_aspect) / 2  # average of both AD values, AD_beta,gamma i.e. AD_slope,aspect
    min_idx = ad.argmin()  # get flattened index of min AD value (index of pcgp)
    gpe_model["ad"] = (("lat", "lon"), ad)  # add AD_beta,n to gpe_model dataset
    # Now finished AD calculation, but how can I select PCGP where AD is minimal to find lat/lon coordinates of pcgp?
    # tried with new dataset, but pfusch...

    pcgp = xr.Dataset(
        data_vars=dict(
            band_data=(("lat", "lon", "ad"), gpe_model.band_data.values),
        ),
        coords=dict(
        x = ("x", gpe_model.x.values),
        y = ("y", gpe_model.y.values),
        ad = ("ad", ad),
    ),
    attrs=dict(decription="GPE of model with ad as coordinate to enable indexing by ad"),
    )
    pcgp_model = gpe_model.where(gpe_model.ad == gpe_model.ad.min(),
                                 drop=True)  # select pcgp from model topography
    return pcgp_model


if __name__ == '__main__':
    matplotlib.use('Qt5Agg')
    pal = sequential_hcl("Terrain")
    lat_ibk = 47.259998
    lon_ibk = 11.384167
    lat_nordkette = 47.3
    hafelekar_height = 2279  # m, highest HOBO from https://zenodo.org/records/4672313 hobo dataset
    c_p = 1005  # J/(kg*K), specific heat capacity of air at constant pressure
    # icon_ibk_ngp = read_icon_model_3D.read_icon_fixed_point(lat=lat_ibk, lon=lon_ibk, variant="ICON")  # ngp = nearest grid point

    # read DEM and calc slope
    slope_dem, aspect_dem, dem = calculate_slope(confg.TIROL_DEMFILE)
    # plot_height_slope_aspect(height_da=dem.isel(band=0).band_data, slope=slope_dem, aspect_raster=aspect_dem,
    #                         modeltype="dem", title_height="Height", title_slope="Slope", title_aspect="Aspect")

    # read AROME and calc slope
    slope_arome, aspect_arome, arome = calculate_slope(confg.dir_AROME + "AROME_geopot_height_3dlowest_level_w_crs.tif")
    # plot_height_slope_aspect(height_da=arome.isel(band=0).band_data, slope=slope_arome, aspect_raster=aspect_arome,
    #                         modeltype="model", title_height="Height", title_slope="Slope", title_aspect="Aspect")

    # read ICON, calc & plot slope + aspect, somehow ICON extent looks much bigger as what I cutted...
    slope_icon, aspect_icon, icon = calculate_slope(confg.icon_folder_3D + "/ICON_geometric_height_3dlowest_level_w_crs.tif")
    # plot_height_slope_aspect(height_da=icon.isel(band=0).band_data, slope=slope_icon, aspect_raster=aspect_icon,
    #                         modeltype="model", title_height="Height", title_slope="Slope", title_aspect="Aspect")

    # choose physically consistent grid point (pcgp) from model topography according to
    #
    gpe_dem = choose_gpe(ds=dem, lat_ngp=lat_nordkette, lon_ngp=lon_ibk)  # choose GPE around NGP from DEM -> real values
    gpe_arome = choose_gpe(ds=arome, lat_ngp=lat_nordkette, lon_ngp=lon_ibk)
    gpe_icon = choose_gpe(ds=icon, lat_ngp=lat_nordkette, lon_ngp=lon_ibk)

    pcgp_arome = calculate_select_pcgp(gpe_model=gpe_arome, gpe_dem=gpe_icon)
    pcgp_icon = calculate_select_pcgp(gpe_model=gpe_icon, gpe_dem=gpe_icon)

    icon_ibk_ngp = xr.open_dataset(confg.icon_folder_3D + "/ICON_temp_p_rho_timeseries_ibk.nc")
    vhd_continue = calc_vhd_single_point(icon_ibk_ngp)
    icon_ibk_ngp


    # old stuff
    # dem_elevation = dem.isel(band=0).height.compute().values  # index to have only 2 coordinates: lat, lon
    # slope_dem = read_topo_calc_slope(dem_xr.sel(band=1).height)  # calculate slope from DEM
    # read ICON and calc slope
    # model_topo = xr.open_dataset(confg.icon_folder_3D + "/ICON_geometric_height_3dlowest_level.nc")
    # slope_model = read_topo_calc_slope(model_topo.z)  # calculate slope from model topography, needs DataArray as input
    # model_topo["slope"] = (("lat", "lon"), slope_model)  # add slope to model_topo dataset
