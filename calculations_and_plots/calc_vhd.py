"""
In this script some functions are defined that calculate the valley heat deficit (VHD) from the potential temperature
for each model. VHD will be calc between lowest model level & Hafelekar at 2279 m, at the highest HOBO from
https://zenodo.org/records/4672313 hobo dataset
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
import read_icon_model_3D
import read_ukmo
import read_wrf_helen
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
    slope_dem, aspect_dem, dem = calculate_slope(confg.dem_smoothed)
    plot_height_slope_aspect(height_da=dem.isel(band=0).band_data, slope=dem.isel(band=0).slope,
                             aspect_raster=dem.isel(band=0).aspect, modeltype="DEM smoothed", title_height="Height",
                             title_slope="Slope", title_aspect="Aspect")
    dem.to_netcdf(f"{confg.data_folder}/Height/" + "dem_with_slope_aspect.nc")

    # read AROME topo tif file and calc slope
    slope_arome, aspect_arome, arome = calculate_slope(confg.dir_AROME + "AROME_geopot_height_3dlowest_level_w_crs.tif")
    arome = arome.sel(x=slice(9.2, 13), y=slice(46.5, 48.2))

    plot_height_slope_aspect(height_da=arome.isel(band=0).band_data, slope=arome.isel(band=0).slope,
                             aspect_raster=arome.isel(band=0).aspect, modeltype="AROME", title_height="Height",
                             title_slope="Slope", title_aspect="Aspect")
    arome.to_netcdf(confg.data_folder + "/Height/arome_topo_with_slope_aspect.nc")

    # read ICON, calc & plot slope + aspect
    slope_icon, aspect_icon, icon = calculate_slope(confg.icon_folder_3D + "/ICON_geometric_height_3dlowest_level_w_crs.tif")
    plot_height_slope_aspect(height_da=icon.isel(band=0).band_data, slope=icon.isel(band=0).slope,
                             aspect_raster=icon.isel(band=0).aspect, modeltype="ICON", title_height="Height",
                             title_slope="Slope", title_aspect="Aspect")
    icon.to_netcdf(confg.data_folder + "/Height/icon_topo_with_slope_aspect.nc")

    # read UKMO and calc slope
    slope_um, aspect_um, um = calculate_slope(confg.ukmo_folder + "/UM_geometric_height_3dlowest_level.tif")
    plot_height_slope_aspect(height_da=um.isel(band=0).band_data, slope=um.isel(band=0).slope,
                             aspect_raster=um.isel(band=0).aspect, modeltype="UM", title_height="Height",
                             title_slope="Slope", title_aspect="Aspect")
    um.to_netcdf(confg.data_folder + "/Height/um_topo_with_slope_aspect.nc")

    # read WRF and calc slope
    slope_wrf, aspect_wrf, wrf = calculate_slope(confg.wrf_folder + "/WRF_geometric_height_3dlowest_level.tif")
    plot_height_slope_aspect(height_da=wrf.isel(band=0).band_data, slope=wrf.isel(band=0).slope,
                             aspect_raster=wrf.isel(band=0).aspect, modeltype="WRF", title_height="Height",
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


def plot_height_slope_aspect(height_da, slope, aspect_raster, modeltype, title_height="Height", title_slope="Slope", title_aspect="Aspect"):
    """
    Plottet Höhe (xarray.DataArray), Slope (np.ndarray) und Aspect (Raster-Objekt) nebeneinander.
    """
    # Aspect-Daten extrahieren und fehlende Werte maskieren
    aspect = np.array(aspect_raster.data, dtype=np.float32)
    # aspect = np.ma.masked_where(np.isnan(aspect) | (aspect == getattr(aspect_raster, "nodata", -99999)), aspect)
    # what's that?

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(height_da.values, origin="lower", cmap=pal.cmap())
    axes[0].set_title(modeltype + " " + title_height)
    plt.colorbar(im0, ax=axes[0], fraction=0.025, pad=0.04, label="Height [m]")

    im1 = axes[1].imshow(slope, origin="lower", cmap=pal.cmap())
    axes[1].set_title(title_slope)
    plt.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.04, label="Slope [°]")

    im2 = axes[2].imshow(aspect, origin="lower", cmap="twilight", vmin=0, vmax=360)
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
    if model in ["ICON", "ICON2TE"]:
        # for ICON we have different height coordinates (staggered & unstaggered), therefore I chose the height level
        # of Hafelekar with the height var of z (orig z_ifc, which is unstaggered or staggered?!?) and indexed with it
        # the "height" coord which is the coord for the other variables like th, rho, p etc
        # distance between vert. model levels is ~100m, so error is of max. 50m, which is acceptable
        # (which is ~ 10% error for AROME, just took hafelekar height +50m and looked at VHD...)
        ds_below_hafelekar = ds.sel(height=ds.z.where(ds.z <= confg.hafelekar_height, drop=True).height_3.values)
    elif model == "UM":
        ds_below_hafelekar = ds.where(ds.z <= confg.hafelekar_height, drop=True)  # maybe the others are uselessly
        # complicated? check again for errors...
    elif model == "WRF":
        ds_below_hafelekar = ds.sel(height=ds.z.where(ds.z <= confg.hafelekar_height, drop=True).bottom_top_stag.values)
    elif model == "HATPRO":
        ds_below_hafelekar = ds.sel(height = ds.height.where(ds.height <= confg.hafelekar_height, drop=True).height.values)
    else:
        # select full dataset below Hafelekar for AROME
        ds_below_hafelekar = ds.sel(height=ds.z.where(ds.z <= confg.hafelekar_height, drop=True).height.values)
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
        ds_below_hafelekar = ds_extent.sel(height=slice(33, 1))
    elif model == "UM":  # which index?
        ds_below_hafelekar = ds_extent.sel(height=slice(1, 21))
    elif model == "WRF":  # I have no clue why I have to use a rising slice for WRF and a descending one for the other models
        ds_below_hafelekar = ds_extent.sel(height=slice(1, 37))
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


def save_timeseries(pcgp_arome, pcgp_icon, pcgp_um, pcgp_wrf, point_name=None,
                    variables=None, paths={"AROME": Path(confg.dir_AROME + "/arome_ibk_uni_timeseries.nc"),
                                           "ICON": Path(confg.icon_folder_3D + "/icon_ibk_uni_timeseries.nc"),
                                           "ICON2TE": Path(confg.icon2TE_folder_3D + "/icon_2te_ibk_uni_timeseries.nc"),
                                           "UM": Path(confg.ukmo_folder + "/um_ibk_uni_timeseries.nc"),
                                           "WRF": Path(confg.wrf_folder + "/wrf_ibk_uni_timeseries.nc")}):
    """
    checks if timeseries already exists for the given point, if not it reads the timeseries at the given PCGP-point
    and saves the timeseries as .nc files.
    This function saves the timeseries of the models at the given PCGP-point and saves the timeseries as .nc files.

    """
    if not paths["AROME"].exists():
        print("vhd AROME need to be calculated first for that point, please wait...")
        model_timeseries = read_in_arome.read_in_arome_fixed_point(lat=pcgp_arome.y.values, lon=pcgp_arome.x.values,
                                                                   variables=variables)
        model_timeseries.to_netcdf(paths["AROME"])
    if not paths["ICON"].exists():
        print("vhd ICON need to be calculated first for that point, please wait...")
        model_timeseries = read_icon_model_3D.read_icon_fixed_point(lat=pcgp_icon.y.values, lon=pcgp_icon.x.values,
                                                                    variant="ICON", variables=variables)
        model_timeseries.to_netcdf(paths["ICON"])
    if not paths["ICON2TE"].exists():
        print("vhd ICON2TE need to be calculated first for that point, please wait...")
        model_timeseries = read_icon_model_3D.read_icon_fixed_point(lat=pcgp_icon.y.values, lon=pcgp_icon.x.values,
                                                                    variant="ICON2TE", variables=variables)
        model_timeseries.to_netcdf(paths["ICON2TE"])
    if not paths["UM"].exists():
        print("vhd UM need to be calculated first for that point, please wait...")
        model_timeseries = read_ukmo.read_ukmo_fixed_point(lat=pcgp_um.y.values, lon=pcgp_um.x.values,
                                                           variables=variables)
        model_timeseries.to_netcdf(paths["UM"])
    if not paths["WRF"].exists():
        print("vhd WRF need to be calculated first for that point, please wait...")
        model_timeseries = read_wrf_helen.read_wrf_fixed_point(lat=pcgp_wrf.y.values, lon=pcgp_wrf.x.values,
                                                               variables=variables)
        model_timeseries.to_netcdf(paths["WRF"])


def calc_vhd_single_point_main(lat=None, lon=None, point_name=confg.ibk_uni["name"]):
    """
    checks if timeseries is not already saved for the given point, if not it reads the DEMs, calculates the PCGP and
    then read & save the timeseries for the PCGP for the given point.

    :param lat: latitude of point
    :param lon: longitude of point
    :param point_name: name of point defined in confg.py
    :return:
    vhd_arome:
    """
    timeseries_paths = {"AROME": Path(confg.dir_AROME + f"/arome_{point_name}_timeseries.nc"),
                        "ICON": Path(confg.icon_folder_3D + f"/icon_{point_name}_timeseries.nc"),
                        "ICON2TE": Path(confg.icon2TE_folder_3D + f"/icon_2te_{point_name}_timeseries.nc"),
                        "UM": Path(confg.ukmo_folder + f"/um_{point_name}_timeseries.nc"),
                        "WRF": Path(confg.wrf_folder + f"/wrf_{point_name}_timeseries.nc")}
    # if [f.exists() for f in timeseries_paths].count(False) >= 1:  # check if timeseries were already saved for that point, if not read and save them!

    pcgp_arome, pcgp_icon, pcgp_um, pcgp_wrf = read_dems_calc_pcgp(lat=lat, lon=lon)
    save_timeseries(pcgp_arome=pcgp_arome, pcgp_icon=pcgp_icon, pcgp_um=pcgp_um, pcgp_wrf=pcgp_wrf,
                    point_name=point_name, variables=["p", "th", "temp", "rho", "z", "hgt"], paths=timeseries_paths)

    # read saved timeseries for ibk gridpoint defined above
    arome_timeseries = xr.open_dataset(confg.dir_AROME + f"/arome_{point_name}_timeseries.nc")
    icon_timeseries = xr.open_dataset(confg.icon_folder_3D + f"/icon_{point_name}_timeseries.nc")# read saved AROME timeseries
    icon2te_timeseries = xr.open_dataset(confg.icon2TE_folder_3D + f"/icon_2te_{point_name}_timeseries.nc")  # read saved ICON timeseries
    um_timeseries = xr.open_dataset(confg.ukmo_folder + f"/um_{point_name}_timeseries.nc")  # read saved UKMO timeseries
    wrf_timeseries = xr.open_dataset(confg.wrf_folder + f"/wrf_{point_name}_timeseries.nc")  # read saved WRF timeseries

    # calc VHD for model data for single PCGP
    vhd_arome = calc_vhd_single_point(arome_timeseries, model="AROME")
    vhd_icon = calc_vhd_single_point(icon_timeseries, model="ICON")
    vhd_icon2te = calc_vhd_single_point(icon2te_timeseries, model="ICON")
    vhd_um = calc_vhd_single_point(um_timeseries, model="UM")
    vhd_wrf = calc_vhd_single_point(wrf_timeseries, model="WRF")
    return vhd_arome, vhd_icon, vhd_icon2te, vhd_um, vhd_wrf


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
    read_topos_calc_slope_aspect_main()

    # hatpro = xr.open_dataset(f"{confg.hatpro_folder}/hatpro_interpolated_arome.nc")
    # vhd_hatpro = calc_vhd_single_point(hatpro, model="HATPRO")

    # to calculate the PCGP for this point and save timeseries of the models at the PCGP
    # read_dems_calc_pcgp(lat=confg.ibk_villa["lat"], lon=confg.ibk_villa["lon"])

    # if PCGP already saved as extra nc file:
    # vhd_arome, vhd_icon, vhd_icon2te, vhd_um, vhd_wrf = calc_vhd_single_point_main(lat=confg.ibk_villa["lat"], lon=confg.ibk_villa["lon"],
     #                                                             point_name=confg.ibk_villa["name"])  # call main fct which calls
    # vhd_arome  # all other fcts for calculating vhd for that point

    # read full domain of model and calc VHD for every hour concatenate them into 1 dataset and write them to a .nc file
    timerange = pd.date_range("2017-10-15 12:00:00", periods=49, freq="30min")
    vhd_datasets, vhd_ds_arome, vhd_ds_icon, vhd_ds_um, vhd_ds_wrf = [], [], [], [], []

    for timestamp in timerange:
        # arome = read_in_arome.read_in_arome_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
        #                                               variables=["p", "temp", "th", "z", "rho"])
        #vhd_ds_arome.append(calc_vhd_full_domain(ds_extent=arome, model="AROME"))
        # icon2te = read_icon_model_3D.read_icon_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
        #                                             variant="ICON2TE", variables=["p", "temp", "th", "z", "rho"])
        # vhd_ds_icon.append(calc_vhd_full_domain(ds_extent=icon2te, model="ICON2TE"))
        um = read_ukmo.read_ukmo_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
                                            variables=["p", "temp", "th", "z", "rho", "hgt"])
        vhd_ds_um.append(calc_vhd_full_domain(ds_extent=um, model="UM"))
        wrf = read_wrf_helen.read_wrf_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
                                                      variables=["p", "temp", "th", "z", "rho"])
        vhd_ds_wrf.append(calc_vhd_full_domain(ds_extent=wrf, model="WRF"))

    # vhd_arome_full = xr.concat(vhd_ds_arome, dim="time")
    # vhd_arome_full.to_dataset(name="vhd").to_netcdf(confg.dir_AROME + "/AROME_vhd_full_domain_full_time.nc")
    # vhd_icon_full = xr.concat(vhd_ds_icon, dim="time")
    # vhd_icon_full.to_dataset(name="vhd").to_netcdf(confg.icon_folder_3D + "/ICON_vhd_full_domain_full_time.nc")
    # vhd_icon_full.to_dataset(name="vhd").to_netcdf(confg.icon2TE_folder_3D + "/ICON2TE_vhd_full_domain_full_time.nc")
    vhd_um_full = xr.concat(vhd_ds_um, dim="time")
    vhd_um_full.to_dataset(name="vhd").to_netcdf(confg.ukmo_folder + "/UM_vhd_full_domain_full_time.nc")
    vhd_wrf_full = xr.concat(vhd_ds_wrf, dim="time")
    vhd_wrf_full.to_dataset(name="vhd").to_netcdf(confg.wrf_folder + "/WRF_vhd_full_domain_full_time.nc")

    # vhd_arome_pcgp, vhd_icon_pcgp, vhd_icon2te_pcgp = select_pcgp_vhd(lat=confg.ibk_uni["lat"], lon=confg.ibk_uni["lon"],
    #                                                                   point_name=confg.ibk_uni["name"])
    #vhd_arome_pcgp
    # vhd_arome = calc_vhd_full_domain(ds_extent=arome, model="AROME")
    # vhd_icon = calc_vhd_full_domain(ds_extent=icon, model="ICON")
    # vhd_arome
    # vhd_icon
