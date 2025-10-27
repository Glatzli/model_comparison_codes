"""
In this script some functions are defined that calculate the valley heat deficit (VHD) from the potential temperature
for each model. VHD will be calc between lowest model level & Hafelekar (HAF) at 2279 m, at the highest HOBO from
https://zenodo.org/records/4672313 hobo dataset
Problem here: as topography value i.e. for indexing until HAF height geopotential height is used, which changes spatially
and temporarily (a bit). Therefore I searched for the index of HAF for the IBK gridpoint and then used the hardcoded
index for calculating the VHD (subsetting the dataset in function define_ds_below_hafelekar).

for point values of the VHD PCGP selection is used to find the physically consistent grid point (PCGP) from a model grid point ensemble (GPE)
as described in Simonet et al. (2025)  https://doi.org/10.21203/rs.3.rs-6050730/v1

For the calculation of the PCGP first the slope and aspect of each model & the DEM is calculated, which is done in
read_topos_calc_slope_aspect() and the other few first functions defined in this script

Then the VHD should be calculated for the full domain.
Plotting the VHD (timeseries and spatial extent over time) is done in plot_vhd.py
"""

import sys
from pathlib import Path
sys.path.append("D:/MSc_Arbeit/model_comparison_codes")
import read_in_arome
import richdem as rd
import read_icon_model_3D
import read_ukmo
import read_wrf_helen
import importlib
importlib.reload(read_wrf_helen)
import confg
import xarray as xr
import numpy as np
import matplotlib
from plot_topography import calculate_km_for_lon_extent
import xdem
from pyproj import CRS
import matplotlib.pyplot as plt
import pandas as pd
from colorspace import terrain_hcl, qualitative_hcl, sequential_hcl


def read_topos_calc_slope_aspect_main():
    """
    DEM calculation is wrong....
    reads DEM, AROME and ICON topography datasets, calculates slope and aspect for each dataset; is only used once for
    calculating for the models & DEM... That's also why it's not shortened...
    :return:
    """
    # read DEM and calc slope, save them as netcdf files
    dem = calculate_slope_aspect_richdem(confg.dem_smoothed)

    plot_height_slope_aspect(ds=dem.isel(band=0), modeltype="DEM smoothed", title_height="Height",
                             title_slope="Slope", title_aspect="Aspect")
    dem.to_netcdf(f"{confg.data_folder}/Height/" + "dem_with_slope_aspect.nc")

    # read AROME topo tif file and calc slope
    arome = calculate_slope_aspect_richdem(confg.dir_AROME + "AROME_geopot_height_3dlowest_level_w_crs.tif")
    arome = arome.sel(x=slice(9.2, 13), y=slice(46.5, 48.2))

    plot_height_slope_aspect(ds=arome.isel(band=0), modeltype="AROME", title_height="Height",
                             title_slope="Slope", title_aspect="Aspect")
    arome.to_netcdf(confg.data_folder + "/Height/arome_topo_with_slope_aspect.nc")

    # read ICON, calc & plot slope + aspect
    icon = calculate_slope_aspect_richdem(confg.icon_folder_3D + "/ICON_geometric_height_3dlowest_level_w_crs.tif")
    plot_height_slope_aspect(ds=icon.isel(band=0), modeltype="ICON", title_height="Height",
                             title_slope="Slope", title_aspect="Aspect")
    icon.to_netcdf(confg.data_folder + "/Height/icon_topo_with_slope_aspect.nc")

    # read UKMO and calc slope
    um = calculate_slope_aspect_richdem(confg.ukmo_folder + "/UM_geometric_height_3dlowest_level.tif")
    plot_height_slope_aspect(ds=um.isel(band=0), modeltype="UM", title_height="Height",
                             title_slope="Slope", title_aspect="Aspect")
    um.to_netcdf(confg.data_folder + "/Height/um_topo_with_slope_aspect.nc")

    # read WRF and calc slope
    wrf = calculate_slope_aspect_richdem(confg.wrf_folder + "/WRF_geometric_height_3dlowest_level.tif")
    plot_height_slope_aspect(ds=wrf.isel(band=0), modeltype="WRF", title_height="Height",
                             title_slope="Slope", title_aspect="Aspect")
    wrf.to_netcdf(confg.data_folder + "/Height/wrf_topo_with_slope_aspect.nc")
    plt.show()


def choose_gpe(ds, lat_ngp, lon_ngp):
    """choose grid point ensemble (GPE) around NGP (nearest grid point) from each dataset with indices of lat and lon
    written by ChatGPT => I thought maybe is here the Problem of the different results of VHDs, but the GPEs
    seem to be good ?!"""
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
    open .tif - dataset (either beforehand selected & saved model geopotential height or DEM) calculate slope (call numpy function)
    and aspect (xDEM) from elevation data
    :param: filepath: path to the .tif file with the model or DEM data
    :return: slope (np.ndarray), aspect (xarray.DataArray), model (xarray.Dataset with slope and aspect added)
    """
    model = xr.open_dataset(filepath, engine="rasterio")
    if "dem" in filepath:
        model = model.isel(y=slice(None, None, -1))

    model_xres_deg = model.x[1].values - model.x[0].values
    model_xres_m = calculate_km_for_lon_extent(latitude=model.y[int(len(model.y)/2)].values,  # take middle latitude of dataset
                                             lon_extent_deg=model_xres_deg) * 1000  # convert to meters

    slope, slope_deg = calculate_slope_numpy(elevation_data=model.isel(band=0).band_data.values, x_res=model_xres_m)
    vertical_crs = CRS("EVRF2019")

    dem = xdem.DEM(filepath)  # , transform=transform
    aspect = xdem.terrain.aspect(dem)
    model["slope"] = (("y", "x"), slope)  # add slope to dem dataset
    model["aspect"] = (("y", "x"), aspect.data.data)
    return slope, aspect, model


def calculate_slope_aspect_richdem(filepath):
    """
    calculates slope and aspect ratio using richdem
    first read .tif-topo file, squeeze height data, calculate z_scale (right?) and then compute slope&aspect with richDEM

    :param ds: fixed_time dataset of a model
    :return:
    :ds: dataset with slope and aspect ratio
    """
    ds = xr.open_dataset(filepath, engine="rasterio")  # for xDEM calculation I had a .tif file of topo, continue with that
    if "dem" in filepath:  # missing geotransform, try to set it by hand
        # In case of north up images, the GT(2) and GT(4) coefficients are zero, and the GT(1) is pixel width,
        # and GT(5) is pixel height.
        # The (GT(0),GT(3)) position is the top left corner of the top left pixel of the raster. from GDAL docs
        ds["y"] = ds["y"][::-1]

    xgeo = np.array([ds.y.max().item(), len(ds.x.values), 0, ds.x.min().item(), 0, len(ds.y.values)])
    hgt = ds['band_data'].values.squeeze()

    # calculate slope angle
    hgtrd = rd.rdarray(hgt, geotransform=xgeo, no_data=-9999)  # there's no value with -9999, all data is used!
    # zscale (float) – How much to scale the z-axis by prior to calculation
    # i.e. how much smaller is the distance in z between the levels than in x or y?
    # original by Manuela: 1/(how much km is 1° * dist_x in deg)
    # => I calculate it with searching the distance between points in lon [deg] and converting those to [m] with the function
    # "calculate_km_for_lon_extent", divide that through 1 cause Manuela did that also -> hopefully right?
    dist_x = calculate_km_for_lon_extent(confg.ibk_uni["lat"],
                                         (ds.isel(y=1, x=2).x - ds.isel(y=1, x=1).x)) * 1000
    z_scale = 1 / dist_x

    slope = rd.TerrainAttribute(hgtrd, attrib='slope_degrees', zscale=z_scale)
    aspect = rd.TerrainAttribute(hgtrd, attrib='aspect', zscale=z_scale)
    aspect = np.mod(360 - aspect + 180, 360)
    ds["slope"] = (("y", "x"), slope)  # add slope to dem dataset
    ds["aspect"] = (("y", "x"), aspect)
    return ds


def plot_height_slope_aspect(ds, modeltype, title_height="Height", title_slope="Slope", title_aspect="Aspect"):
    """
    Plottet Höhe (xarray.DataArray), Slope (np.ndarray) und Aspect (Raster-Objekt) nebeneinander.
    """
    # Aspect-Daten extrahieren und fehlende Werte maskieren
    # aspect = np.array(aspect_raster.data, dtype=np.float32)
    # aspect = np.ma.masked_where(np.isnan(aspect) | (aspect == getattr(aspect_raster, "nodata", -99999)), aspect)
    # what's that?

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # if modeltype in ["DEM", "dem", "DEM smoothed"]:
    #     plot_origin = "upper"
    # else:
    plot_origin = "lower"

    im0 = axes[0].imshow(ds.band_data.values, origin=plot_origin, cmap=pal.cmap())
    axes[0].set_title(modeltype + " " + title_height)
    plt.colorbar(im0, ax=axes[0], fraction=0.025, pad=0.04, label="Height [m]")

    im1 = axes[1].imshow(ds.slope.values, origin=plot_origin, cmap=pal.cmap())
    axes[1].set_title(title_slope)
    plt.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.04, label="Slope [°]")

    im2 = axes[2].imshow(ds.aspect.values, origin=plot_origin, cmap="twilight", vmin=0, vmax=360)
    axes[2].set_title(title_aspect)
    plt.colorbar(im2, ax=axes[2], fraction=0.025, pad=0.04, label="Aspect [°]")

    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()


def define_ds_below_hafelekar(ds, model="AROME"):
    """
    indexes the dataset below Hafelekar height (2279 m) for the given model. Because the handling/height coords are
    slightly different for each model, each needs to be handeled differently...

    :param ds:
    :param model:
    :return:
    """
    # ds_below_hafelekar = ds.sel(height=slice(confg.hafelekar_height, 1))  # now I just use geopot height as z coordinate
    # and hardcoded indexes aren't needed anymore...
    """
    if model in ["AROME", "ICON", "ICON2TE"]:
        ds_below_hafelekar = ds.sel(height=slice(confg.hafelekar_height, 1))
    elif model in ["UM", "WRF", "HATPRO", "radio"]:
        ds_below_hafelekar = ds.sel(height=slice(1, confg.hafelekar_height))
    """
    if model == "AROME":  # back then I haven't had geopot height as z coord...
        # select full dataset below Hafelekar for AROME (and all else...)
        # ds_below_hafelekar = ds.where(ds.z <= confg.hafelekar_height, drop=True)  # for searching HAF height
        ds_below_hafelekar = ds.isel(height=slice(53, 100))
        # use uniformely level of HAF for Ibk gridpoint from bottom up till lvl 37, 90 (total vert. lvls) - 37 = 53...

    elif model in ["ICON", "ICON2TE"]:
        # for ICON we have different height coordinates (staggered & unstaggered), therefore I chose the height level
        # below Hafelekar with the height var of z (original z_ifc) and used index to be geopot height change-independent
        # (and with that uniformly in space and time)

        # (there is ~ 10% VHD error when taking an additional level for the computation for AROME, just took hafelekar
        # height +50m and looked at VHD...)

        ds_below_hafelekar = ds.isel(height=slice(59, 100))
    elif model == "UM":
        ds_below_hafelekar = ds.where(ds.z <= confg.hafelekar_height, drop=True)  # maybe the others are uselessly
        ds_below_hafelekar = ds.isel(height=slice(0, 21))
        # complicated? check again for errors...
    elif model == "WRF":
        ds_below_hafelekar = ds.isel(height=ds.z.where(ds.z <= confg.hafelekar_height, drop=True).bottom_top_stag.values)
        ds_below_hafelekar = ds.isel(height=slice(0, 30))
    elif model in ["HATPRO", "radio", "radiosonde"]:
        ds_below_hafelekar = ds.sel(height=slice(0, confg.hafelekar_height))   # for HATPRO

    return ds_below_hafelekar


def calc_vhd_single_point(ds_point, model="AROME"):
    """
    calculates the valley heat deficit (VHD) for a single point in a dataset, e.g. for Innsbruck.
    calc density from pressure and temperature using metpy/ ideal gas law: rho = p / (R * T) with R_dryair = 287.05

    param ds_point:
    :return: vhd_point: xarray.DataArray with valley heat deficit at the point with time as coord.
    """
    ds_below_hafelekar = define_ds_below_hafelekar(ds=ds_point, model=model)
    th_hafelekar = ds_below_hafelekar.sel(height=ds_below_hafelekar.height.max()).th
    vhd_point = confg.c_p*((th_hafelekar - ds_below_hafelekar.th) * ds_below_hafelekar.rho).sum(dim="height")

    return vhd_point.to_dataset(name="vhd")  # return as dataset with vhd as variable


def calc_vhd_full_domain(ds_extent, model="AROME"):
    """
    should calculate the VHD for the full domain which is in the dataset given. for indexing the full dataset below
    hafelekar, another function is used (due to different handling for all models...)

    large data, espc for AROME because I didn't subset it yet, CDO gives error... maybe subset in python first?
    :param ds_extent:
    :param model: model name, used to select the correct height coord for indexing
    :return:
    """
    if model == "AROME":  # searched for height-value of Hafelekar in the point-calculation for Ibk gridpoint and use
        # that height for the full domain (searching here with HAF-height doesn't work properly,
        # because some gridpoints are above HAF height...)
        ds_below_hafelekar = ds_extent.sel(height = slice(37, 1))
    elif model in ["ICON", "ICON2TE"]:
        ds_below_hafelekar = ds_extent.sel(height=slice(32, 1))
    elif model == "UM":  # which index?
        ds_below_hafelekar = ds_extent.sel(height=slice(1, 22))
    elif model == "WRF":  # I have no clue why I have to use a rising slice for WRF and a descending one for the other models
        ds_below_hafelekar = ds_extent.sel(height=slice(1, 31))
    else:
        print("search proper HAF height in indexes of ds before calculating!")

    th_hafelekar = ds_below_hafelekar.sel(height=ds_below_hafelekar.height.max()).th
    vhd_full_domain = confg.c_p*((th_hafelekar - ds_below_hafelekar.th) * ds_below_hafelekar.rho).sum(dim="height")
    # had before at ds_below_hafelekar .isel(height=slice(1, 100))

    return vhd_full_domain


def calculate_select_pcgp(gpe_model, gpe_dem):
    """
    computes and selects the physically consistent grid point (PCGP) from a model grid point ensemble (GPE) and
    DEM-ensemble and returns the PCGP as a xarray dataset with x as lon and y as lat coord.

    An ensemble consists of 1 nearest grid point in the middle and it's 8 surrounding points
    More details can be found in Simonet et al. (2025)  https://doi.org/10.21203/rs.3.rs-6050730/v1
    here only implemented with slope angle & aspect ratio calculated from the model/dem height,
    implementation of land use is not possible due to missing measurements at the sites & variables in my model runs
    all variables are named in small letters for easier writing...

    for testing it would be indeed better to write smaller functions...

    :param gpe_model: grid point ensemble from a model (arome, icon, ...)
    :param gpe_dem: grid point ensemble from DEM
    :return:
    pcgp: xarray dataset with the physically consistent grid point (PCGP) from the model x as lon and y as lat coord.
    """
    # calc AD slope:
    ad_slope = np.abs(gpe_model.slope.values - gpe_dem.slope.values)  # calculate AD_beta i.e. slope
    # ad_slope_n = ad_slope / ad_slope.max()  # AD_beta,n probably not needed!

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
    # min_idx = ad.argmin()  # get flattened index of min AD value (index of pcgp)
    gpe_model["ad"] = (("x", "y"), ad)  # add AD_beta,n to gpe_model dataset
    # Now finished AD calculation, but how can I select PCGP where AD is minimal to find lat/lon coordinates of pcgp?
    # tried with new dataset, but pfusch...

    # search for index where AD is minimal and index dataset to get dataset only for the PCGP
    min_idx_flat = gpe_model.ad.values.argmin()
    min_idx = np.unravel_index(min_idx_flat, gpe_model.ad.shape)
    pcgp = gpe_model.isel(band=0).sel(x=gpe_model.x.values[min_idx[1]], y=gpe_model.y.values[min_idx[0]]).compute()

    return pcgp


def read_dems_calc_pcgp(lat=None, lon=None):
    """
    reads model topos & DEM, calls choose gpe fct, calls calc & select pcgp fct and
    saves timeseries with all needed vars at the PCGP
    filepaths to which timeseries are saved are f.e. confg.dir_AROME + f"/arome_{point_name}_timeseries.nc" for AROME

    :param lat: latitude of wanted NGP
    :param lon: longitude of wanted NGP
    :param point_name: name of point, only for file-saving
    :return:
    pcgp_arome: xr ds with the physically consistent grid point (PCGP)
    """
    # read DEM, AROME and ICON topography ds with calc slope&aspect; those were originaly saved dir. from the read in files
    # for AROME the lowest lvl of gepot. height is used, for ICON the lowest level of geometric height,
    # for WRF terrain height var
    dem = xr.open_dataset(confg.data_folder + "/Height/dem_with_slope_aspect.nc")
    arome = xr.open_dataset(confg.data_folder + "/Height/arome_topo_with_slope_aspect.nc")
    icon = xr.open_dataset(confg.data_folder + "/Height/icon_topo_with_slope_aspect.nc")
    um = xr.open_dataset(confg.data_folder + "/Height/um_topo_with_slope_aspect.nc")
    wrf = xr.open_dataset(confg.data_folder + "/Height/wrf_topo_with_slope_aspect.nc")

    # choose physically consistent grid point (pcgp) from model topography according to Simonet et al.
    gpe_dem = choose_gpe(ds=dem, lat_ngp=lat, lon_ngp=lon)  # choose GPE around NGP from DEM -> real values
    gpe_arome = choose_gpe(ds=arome, lat_ngp=lat, lon_ngp=lon)
    gpe_icon = choose_gpe(ds=icon, lat_ngp=lat, lon_ngp=lon)
    gpe_um = choose_gpe(ds=um, lat_ngp=lat, lon_ngp=lon)
    gpe_wrf = choose_gpe(ds=wrf, lat_ngp=lat, lon_ngp=lon)

    pcgp_arome = calculate_select_pcgp(gpe_model=gpe_arome, gpe_dem=gpe_dem)
    pcgp_icon = calculate_select_pcgp(gpe_model=gpe_icon, gpe_dem=gpe_dem)
    pcgp_um = calculate_select_pcgp(gpe_model=gpe_um, gpe_dem=gpe_dem)
    pcgp_wrf = calculate_select_pcgp(gpe_model=gpe_wrf, gpe_dem=gpe_dem)

    return pcgp_arome, pcgp_icon, pcgp_um, pcgp_wrf


def save_timeseries(pcgp_arome, pcgp_icon, pcgp_um, pcgp_wrf, point_name=None, variables=None,
                    paths={"AROME": Path(confg.dir_AROME + "/timeseries/" +  "/arome_ibk_uni_timeseries.nc"),
                           "ICON": Path(confg.icon_folder_3D + "/timeseries/" +  "/icon_ibk_uni_timeseries.nc"),
                           "ICON2TE": Path(confg.icon2TE_folder_3D + "/timeseries/" +  "/icon_2te_ibk_uni_timeseries.nc"),
                           "UM": Path(confg.ukmo_folder + "/timeseries/" +  "/um_ibk_uni_timeseries.nc"),
                           "WRF": Path(confg.wrf_folder + "/timeseries/" +  "/wrf_ibk_uni_timeseries.nc")}, height_as_z_coord=False):
    """
    checks if timeseries already exists for the given point, if not it reads the timeseries at the given PCGP-point
    and saves the timeseries as .nc files.
    This function saves the timeseries of the models at the given PCGP-point and saves the timeseries as .nc files.

    """
    if not paths["AROME"].exists():
        print("AROME timeseries need to be read in first for that point, please wait...")
        if variables[-1] == "z_unstag":  # only ICON & WRF are staggered and need to be unstaggered...
            model_timeseries = read_in_arome.read_in_arome_fixed_point(lat=pcgp_arome.y.values, lon=pcgp_arome.x.values,
                                                                       variables=variables[:-1], height_as_z_coord=height_as_z_coord)
        else:
            model_timeseries = read_in_arome.read_in_arome_fixed_point(lat=pcgp_arome.y.values, lon=pcgp_arome.x.values,
                                                                       variables=variables, height_as_z_coord=height_as_z_coord)
        model_timeseries.to_netcdf(paths["AROME"])
    if not paths["ICON"].exists():
        print("ICON timeseries need to be read in first for that point, please wait...")
        model_timeseries = read_icon_model_3D.read_icon_fixed_point(lat=pcgp_icon.y.values, lon=pcgp_icon.x.values,
                                                                    variant="ICON", variables=variables,
                                                                    height_as_z_coord=height_as_z_coord)
        model_timeseries.to_netcdf(paths["ICON"])
    if not paths["ICON2TE"].exists():
        print("ICON2TE timeseries need to be read in first for that point, please wait...")
        model_timeseries = read_icon_model_3D.read_icon_fixed_point(lat=pcgp_icon.y.values, lon=pcgp_icon.x.values,
                                                                    variant="ICON2TE", variables=variables,
                                                                    height_as_z_coord=height_as_z_coord)
        model_timeseries.to_netcdf(paths["ICON2TE"])
    if not paths["UM"].exists():
        print("UM timeseries need to be read in first for that point, please wait...")
        if variables[-1] == "z_unstag":  #
            model_timeseries = read_ukmo.read_ukmo_fixed_point(lat=pcgp_um.y.values, lon=pcgp_um.x.values,
                                                               variables=variables[:-1], height_as_z_coord=height_as_z_coord)
        else:
            model_timeseries = read_ukmo.read_ukmo_fixed_point(lat=pcgp_um.y.values, lon=pcgp_um.x.values,
                                                               variables=variables, height_as_z_coord=height_as_z_coord)
        model_timeseries.to_netcdf(paths["UM"])
    if not paths["WRF"].exists():
        print("WRF timeseries need to be read in first for that point, please wait...")
        model_timeseries = read_wrf_helen.read_wrf_fixed_point(lat=pcgp_wrf.y.values, lon=pcgp_wrf.x.values,
                                                               variables=variables, height_as_z_coord=height_as_z_coord)
        model_timeseries.to_netcdf(paths["WRF"])


def open_save_timeseries_main(lat=None, lon=None, point_name=confg.ibk_uni["name"],
                              variables=["p", "th", "temp", "rho", "z", "z_unstag"], height_as_z_coord=False):
    """
    calculates PCGP aroung given NGP, calculates timeseries for every model at that point (only if there isn't
    already one saved for that point) and returns the timeseries
    :param lat:
    :param lon:
    :param point_name:
    :return:
    """
    pcgp_arome, pcgp_icon, pcgp_um, pcgp_wrf = read_dems_calc_pcgp(lat=lat, lon=lon)
    timeseries_paths = {"AROME": Path(confg.dir_AROME + "/timeseries/" + f"/arome_{point_name}_timeseries.nc"),
                        # define timeseries paths
                        "ICON": Path(confg.icon_folder_3D + "/timeseries/" +  f"/icon_{point_name}_timeseries.nc"),
                        "ICON2TE": Path(confg.icon2TE_folder_3D + "/timeseries/" +  f"/icon_2te_{point_name}_timeseries.nc"),
                        "UM": Path(confg.ukmo_folder + "/timeseries/" +  f"/um_{point_name}_timeseries.nc"),
                        "WRF": Path(confg.wrf_folder + "/timeseries/" +  f"/wrf_{point_name}_timeseries.nc"),
                        "HATPRO": Path(confg.hatpro_folder + f"/hatpro_interpolated_arome.nc")}
    if height_as_z_coord:  # if geopot. height as z coordinate needed: modify the dictionary accordingly
        timeseries_paths = {
            key: path.with_name(path.stem + "_height_as_z.nc") # if key != "HATPRO" else path  # for incl. hatpro?
            for key, path in timeseries_paths.items()
        }

    save_timeseries(pcgp_arome=pcgp_arome, pcgp_icon=pcgp_icon, pcgp_um=pcgp_um, pcgp_wrf=pcgp_wrf,
                    point_name=point_name, variables=variables, paths=timeseries_paths,
                    height_as_z_coord=height_as_z_coord)
    # if timeseries isn't already saved at that point, read it and save it
    # used geopot height for indexing, not "terrain height" vars!

    # read saved timeseries for ibk gridpoint defined above
    arome_timeseries = xr.open_dataset(timeseries_paths["AROME"])  # confg.dir_AROME + f"/arome_{point_name}_timeseries.nc")
    icon_timeseries = xr.open_dataset(timeseries_paths["ICON"])# read saved AROME timeseries
    icon2te_timeseries = xr.open_dataset(timeseries_paths["ICON2TE"])  # read saved ICON timeseries
    um_timeseries = xr.open_dataset(timeseries_paths["UM"])  # read saved UKMO timeseries
    wrf_timeseries = xr.open_dataset(timeseries_paths["WRF"])  # read saved WRF timeseries
    hatpro_timeseries = xr.open_dataset(timeseries_paths["HATPRO"])

    if height_as_z_coord:  # if geopot height wanted as z coordinate read the accordingly saved radiosonde data
        radio = xr.open_dataset(confg.radiosonde_smoothed)  # smoothed radiosonde data has also geopot height as z coord
    else:
        radio = xr.open_dataset(confg.radiosonde_dataset)

    return arome_timeseries, icon_timeseries, icon2te_timeseries, um_timeseries, wrf_timeseries, hatpro_timeseries, radio


def calc_vhd_single_point_main(lat=None, lon=None, point_name=None, height_as_z_coord=True):
    """
    does some things after each other (not a very testable function...)
    1. calls function that calcs PCGP aroung given NGP, calculates timeseries for every model at that point (only if
        there isn't already one saved for that point) and returns the timeseries
    2. calc the VHD from that timeseries and returns it

    HATPRO timeseries is only included for points with "ibk" in it's names

    :param lat: latitude of point
    :param lon: longitude of point
    :param point_name: name of point defined in confg.py
    :height_as_z_coord: if True, geopotential height is used as z coordinate instead of model height levels
    :return:
    vhd_arome:
    """
    arome_timeseries, icon_timeseries, icon2te_timeseries, um_timeseries, wrf_timeseries, hatpro_timeseries, radio \
        = open_save_timeseries_main(lat=lat, lon=lon, point_name=point_name,
                                    variables=["p", "th", "temp", "rho", "u", "v", "z", "z_unstag"],
                                    height_as_z_coord=height_as_z_coord)
    # calc VHD for model data for single PCGP
    vhd_arome = calc_vhd_single_point(arome_timeseries, model="AROME")
    vhd_icon = calc_vhd_single_point(icon_timeseries, model="ICON")
    vhd_icon2te = calc_vhd_single_point(icon2te_timeseries, model="ICON2TE")
    vhd_um = calc_vhd_single_point(um_timeseries, model="UM")
    vhd_wrf = calc_vhd_single_point(wrf_timeseries, model="WRF")
    vhd_hatpro = calc_vhd_single_point(hatpro_timeseries, model="HATPRO")
    vhd_radio = calc_vhd_single_point(radio, model="radio")
    return vhd_arome, vhd_icon, vhd_icon2te, vhd_um, vhd_wrf, vhd_hatpro, vhd_radio


def select_pcgp_vhd(lat=confg.ibk_uni["lat"], lon=confg.ibk_uni["lon"]):
    """
    opens already calculated VHD calculations of full domain and selects the PCGP for the given gridpoint from it
    :return:
    """
    # read calculated vhds to select PCGP for single points from that:
    vhd_arome = xr.open_dataset(confg.dir_AROME + "/AROME_vhd_full_domain_full_time.nc")
    vhd_icon = xr.open_dataset(confg.icon_folder_3D + "/ICON_vhd_full_domain_full_time.nc")
    vhd_icon2te = xr.open_dataset(confg.icon2TE_folder_3D + "/ICON2TE_vhd_full_domain_full_time.nc")
    vhd_um = xr.open_dataset(confg.ukmo_folder + "/UM_vhd_full_domain_full_time.nc")
    vhd_wrf = xr.open_dataset(confg.wrf_folder + "/WRF_vhd_full_domain_full_time.nc")

    pcgp_arome, pcgp_icon, pcgp_um, pcgp_wrf = read_dems_calc_pcgp(lat=lat, lon=lon)
    vhd_arome_pcgp = vhd_arome.sel(lat=pcgp_arome.y.item(), lon=pcgp_arome.x.item(), method="nearest")  # I thought method
    # "nearest" isn't needed, but somehow the exact lon of pcgp vhd is not exactly the same as lon of vhd_arome?!
    # difference is f.e. 12.064999 for vhd lon value and 12.065000 for pcgp lon value...
    vhd_icon_pcgp = vhd_icon.sel(lat=pcgp_icon.y.item(), lon=pcgp_icon.x.item(), method="nearest")
    vhd_icon2te_pcgp = vhd_icon2te.sel(lat=pcgp_icon.y.item(), lon=pcgp_icon.x.item(), method="nearest")
    vhd_um = vhd_um.sel(lat=pcgp_um.y.item(), lon=pcgp_um.x.item())  # , method="nearest" needed?
    vhd_wrf_pcgp = vhd_wrf.sel(lat=pcgp_wrf.y.item(), lon=pcgp_wrf.x.item(), method="nearest")  # maybe lies here the problem
    # with the unmatched results from domain & point VHD?!
    return vhd_arome_pcgp, vhd_icon_pcgp, vhd_icon2te_pcgp, vhd_um, vhd_wrf_pcgp


if __name__ == '__main__':
    matplotlib.use('Qt5Agg')
    pal = sequential_hcl("Terrain")

    # used only once to calc and plot slopes & aspect ratios
    # read_topos_calc_slope_aspect_main()

    # hatpro = xr.open_dataset(f"{confg.hatpro_folder}/hatpro_interpolated_arome.nc")
    # vhd_hatpro = calc_vhd_single_point(hatpro, model="HATPRO")

    point = confg.ibk_uni
    # calculate PCGP for the given point
    # pcgp_arome, pcgp_icon, pcgp_um, pcgp_wrf = read_dems_calc_pcgp(lat=point["lat"], lon=point["lon"])

    # calculate VHD for a single point (if timeseries isn't already saved, data will be read first):
    (vhd_arome, vhd_icon, vhd_icon2te,
    vhd_um, vhd_wrf, vhd_hatpro, vhd_radio) = calc_vhd_single_point_main(lat=point["lat"], lon=point["lon"],
                                                                         point_name=point["name"])  # call main fct which calls
    # vhd_arome  # all other fcts for calculating vhd for that point


    # radio = pd.read_csv(confg.radiosonde_edited)
    # calc_vhd_single_point(ds=radio, model="radio")

    # read full domain of model and calc VHD for every hour concatenate them into 1 dataset and write them to a .nc file
    timerange = pd.date_range("2017-10-15 12:00:00", periods=49, freq="30min")
    vhd_datasets, vhd_ds_arome, vhd_ds_icon, vhd_ds_icon2te, vhd_ds_um, vhd_ds_wrf = [], [], [], [], [], []

    
    for timestamp in timerange:
        """
        arome = read_in_arome.read_in_arome_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
                                                       variables=["p", "temp", "th", "z", "rho"])
        vhd_ds_arome.append(calc_vhd_full_domain(ds_extent=arome, model="AROME"))
        icon = read_icon_model_3D.read_icon_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
                                                    variant="ICON", variables=["p", "temp", "th", "z", "rho"])
        vhd_ds_icon.append(calc_vhd_full_domain(ds_extent=icon, model="ICON"))

        icon2te = read_icon_model_3D.read_icon_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
                                                    variant="ICON2TE", variables=["p", "temp", "th", "z", "rho"])
        vhd_ds_icon2te.append(calc_vhd_full_domain(ds_extent=icon2te, model="ICON2TE"))

        um = read_ukmo.read_ukmo_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
                                            variables=["p", "temp", "th", "z", "rho"])
        vhd_ds_um.append(calc_vhd_full_domain(ds_extent=um, model="UM"))
        wrf = read_wrf_helen.read_wrf_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
                                                      variables=["p", "temp", "th", "z", "rho"])
        vhd_ds_wrf.append(calc_vhd_full_domain(ds_extent=wrf, model="WRF"))
        """
    """
    vhd_arome_full = xr.concat(vhd_ds_arome, dim="time")
    vhd_arome_full.to_dataset(name="vhd").to_netcdf(confg.dir_AROME + "/AROME_vhd_full_domain_full_time.nc")
    vhd_icon_full = xr.concat(vhd_ds_icon, dim="time")
    vhd_icon_full.to_dataset(name="vhd").to_netcdf(confg.icon_folder_3D + "/ICON_vhd_full_domain_full_time.nc")

    vhd_icon2te_full = xr.concat(vhd_ds_icon2te, dim="time")
    vhd_icon2te_full.to_dataset(name="vhd").to_netcdf(confg.icon2TE_folder_3D + "/ICON2TE_vhd_full_domain_full_time.nc")
    vhd_um_full = xr.concat(vhd_ds_um, dim="time")
    vhd_um_full.to_dataset(name="vhd").to_netcdf(confg.ukmo_folder + "/UM_vhd_full_domain_full_time.nc")
    vhd_wrf_full = xr.concat(vhd_ds_wrf, dim="time")
    vhd_wrf_full.to_dataset(name="vhd").to_netcdf(confg.wrf_folder + "/WRF_vhd_full_domain_full_time.nc")
    """

    # vhd_arome_pcgp, vhd_icon_pcgp, vhd_icon2te_pcgp = select_pcgp_vhd(lat=confg.ibk_uni["lat"], lon=confg.ibk_uni["lon"],
    #                                                                   point_name=confg.ibk_uni["name"])
    #vhd_arome_pcgp
    # vhd_arome = calc_vhd_full_domain(ds_extent=arome, model="AROME")
    # vhd_icon = calc_vhd_full_domain(ds_extent=icon, model="ICON")
    # vhd_arome
    # vhd_icon

    # =====================
    # CAP height (domain): compute per time like VHD and save per model: Is this even a good idea?
    # =====================
    # Import calc_cap_height locally to avoid circular import
    from plot_vertical_calc_bl_height import calc_cap_height, calc_dT_dz
    

    
    cap_ds_arome, cap_ds_icon, cap_ds_icon2te, cap_ds_um, cap_ds_wrf = [], [], [], [], []

    for timestamp in timerange:
        # AROME
        arome = read_in_arome.read_in_arome_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
                                                       variables=["p", "temp", "th", "z"])  # rho not needed
        # calc dT/dz needed by calc_cap_height (simple differentiate along model levels)
        arome = calc_dT_dz(arome)
        res_arome = calc_cap_height(arome)
        cap_t = res_arome["cap_height"] if isinstance(res_arome, xr.Dataset) else res_arome
        cap_t = cap_t.assign_coords(time=("time", [np.datetime64(timestamp)]))
        cap_ds_arome.append(cap_t)

        # ICON
        icon = read_icon_model_3D.read_icon_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
                                                       variant="ICON", variables=["p", "temp", "th", "z"])
        # , icon, icon2te, um, wrf, radio, hatpro
        res_icon = calc_cap_height(icon)
        cap_t = res_icon["cap_height"] if isinstance(res_icon, xr.Dataset) else res_icon
        cap_t = cap_t.assign_coords(time=("time", [np.datetime64(timestamp)]))
        cap_ds_icon.append(cap_t)

        """
        # ICON2TE
        icon2te = read_icon_model_3D.read_icon_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
                                                           variant="ICON2TE", variables=["p", "temp", "th", "z"])
        icon2te = icon2te.assign(dT_dz=icon2te["temp"].differentiate(coord="height"))
        res_icon2te = calc_cap_height(icon2te)
        cap_t = res_icon2te["cap_height"] if isinstance(res_icon2te, xr.Dataset) else res_icon2te
        cap_t = cap_t.assign_coords(time=("time", [np.datetime64(timestamp)]))
        cap_ds_icon2te.append(cap_t)

        # UM
        um = read_ukmo.read_ukmo_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
                                             variables=["p", "temp", "th", "z"])
        um = um.assign(dT_dz=um["temp"].differentiate(coord="height"))
        res_um = calc_cap_height(um)
        cap_t = res_um["cap_height"] if isinstance(res_um, xr.Dataset) else res_um
        cap_t = cap_t.assign_coords(time=("time", [np.datetime64(timestamp)]))
        cap_ds_um.append(cap_t)

        # WRF
        wrf = read_wrf_helen.read_wrf_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
                                                  variables=["p", "temp", "th", "z"])
        wrf = wrf.assign(dT_dz=wrf["temp"].differentiate(coord="height"))
        res_wrf = calc_cap_height(wrf)
        cap_t = res_wrf["cap_height"] if isinstance(res_wrf, xr.Dataset) else res_wrf
        cap_t = cap_t.assign_coords(time=("time", [np.datetime64(timestamp)]))
        cap_ds_wrf.append(cap_t)
        """

    # Concat over time and save like VHD
    cap_arome_full = xr.concat(cap_ds_arome, dim="time")
    cap_arome_full.to_dataset(name="cap_height").to_netcdf(confg.dir_AROME + "/AROME_cap_height_full_domain_full_time.nc")

    cap_icon_full = xr.concat(cap_ds_icon, dim="time")
    cap_icon_full.to_dataset(name="cap_height").to_netcdf(confg.icon_folder_3D + "/ICON_cap_height_full_domain_full_time.nc")

    """
    cap_icon2te_full = xr.concat(cap_ds_icon2te, dim="time")
    cap_icon2te_full.to_dataset(name="cap_height").to_netcdf(confg.icon2TE_folder_3D + "/ICON2TE_cap_height_full_domain_full_time.nc")

    cap_um_full = xr.concat(cap_ds_um, dim="time")
    cap_um_full.to_dataset(name="cap_height").to_netcdf(confg.ukmo_folder + "/UM_cap_height_full_domain_full_time.nc")

    cap_wrf_full = xr.concat(cap_ds_wrf, dim="time")
    cap_wrf_full.to_dataset(name="cap_height").to_netcdf(confg.wrf_folder + "/WRF_cap_height_full_domain_full_time.nc")
    """

    # =====================
    # end CAP height block
    # =====================
