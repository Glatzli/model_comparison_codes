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
import fix_win_DLL_loading_issue

fix_win_DLL_loading_issue
import sys

sys.path.append("D:/MSc_Arbeit/model_comparison_codes")
import richdem as rd
import read_wrf_helen
import importlib
import pandas as pd

importlib.reload(read_wrf_helen)
import confg
import xarray as xr
import numpy as np
from calculations_and_plots.plot_topography import calculate_km_for_lon_extent

import matplotlib.pyplot as plt
from colorspace import sequential_hcl


def read_topos_calc_slope_aspect_main():
    """
    DEM calculation is wrong....
    reads DEM, AROME and ICON topography datasets, calculates slope and aspect for each dataset; is only used once for
    calculating for the models & DEM... That's also why it's not shortened...
    :return:
    """
    # read DEM and calc slope, save them as netcdf files
    dem = calculate_slope_aspect_richdem(confg.dem_smoothed)

    plot_height_slope_aspect(ds=dem.isel(band=0), modeltype="DEM smoothed", title_height="Height", title_slope="Slope",
                             title_aspect="Aspect")
    dem.to_netcdf(f"{confg.data_folder}/Height/" + "dem_with_slope_aspect.nc")

    # read AROME topo tif file and calc slope
    arome = calculate_slope_aspect_richdem(confg.dir_AROME + "AROME_geopot_height_3dlowest_level_w_crs.tif")
    arome = arome.sel(x=slice(9.2, 13), y=slice(46.5, 48.2))

    plot_height_slope_aspect(ds=arome.isel(band=0), modeltype="AROME", title_height="Height", title_slope="Slope",
                             title_aspect="Aspect")
    arome.to_netcdf(confg.data_folder + "/Height/arome_topo_with_slope_aspect.nc")

    # read ICON, calc & plot slope + aspect
    icon = calculate_slope_aspect_richdem(confg.icon_folder_3D + "/ICON_geometric_height_3dlowest_level_w_crs.tif")
    plot_height_slope_aspect(ds=icon.isel(band=0), modeltype="ICON", title_height="Height", title_slope="Slope",
                             title_aspect="Aspect")
    icon.to_netcdf(confg.data_folder + "/Height/icon_topo_with_slope_aspect.nc")

    # read UKMO and calc slope
    um = calculate_slope_aspect_richdem(confg.ukmo_folder + "/UM_geometric_height_3dlowest_level.tif")
    plot_height_slope_aspect(ds=um.isel(band=0), modeltype="UM", title_height="Height", title_slope="Slope",
                             title_aspect="Aspect")
    um.to_netcdf(confg.data_folder + "/Height/um_topo_with_slope_aspect.nc")

    # read WRF and calc slope
    wrf = calculate_slope_aspect_richdem(confg.wrf_folder + "/WRF_geometric_height_3dlowest_level.tif")
    plot_height_slope_aspect(ds=wrf.isel(band=0), modeltype="WRF", title_height="Height", title_slope="Slope",
                             title_aspect="Aspect")
    wrf.to_netcdf(confg.data_folder + "/Height/wrf_topo_with_slope_aspect.nc")
    plt.show()


def choose_gpe(ds, lat_ngp, lon_ngp):
    """choose grid point ensemble (GPE) around NGP (nearest grid point) from each dataset with indices of lat and lon
    written by ChatGPT => I thought maybe is here the Problem of the different results of VHDs, but the GPEs
    seem to be good ?!"""
    lat_idx = ds.y.to_index().get_indexer([lat_ngp], method="nearest")[0]
    lon_idx = ds.x.to_index().get_indexer([lon_ngp], method="nearest")[0]
    # choose GPE by indices aroung NGP
    gpe = ds.isel(y=slice(max(lat_idx - 1, 0), lat_idx + 2), x=slice(max(lon_idx - 1, 0), lon_idx + 2))

    return gpe


def calculate_slope_aspect_richdem(filepath):
    """
    calculates slope and aspect ratio using richdem
    first read .tif-topo file, squeeze height data, calculate z_scale (right?) and then compute slope&aspect with richDEM

    :param ds: fixed_time dataset of a model
    :return:
    :ds: dataset with slope and aspect ratio
    """
    ds = xr.open_dataset(filepath,
                         engine="rasterio")  # for xDEM calculation I had a .tif file of topo, continue with that
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
    dist_x = calculate_km_for_lon_extent(confg.ALL_POINTS["ibk_uni"]["lat"],
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

    ATTENTION: ONLY USE "direct"-timeseries: indexing is hardcoded cause geopot. height changes w. time and space
        -> "above_terrain"-timeseries is not implemented...

    :param ds:
    :param model:
    :return:
    """
    """
    if model in ["AROME", "ICON", "ICON2TE"]:
        ds_below_hafelekar = ds.sel(height=slice(confg.hafelekar_height, 1))
    elif model in ["UM", "WRF", "HATPRO", "radio"]:
        ds_below_hafelekar = ds.sel(height=slice(1, confg.hafelekar_height))
    """
    if model == "AROME":  # back then I haven't had geopot height as z coord...
        # select full dataset below Hafelekar for AROME (and all else...)
        ds_below_hafelekar = ds.sel(height=(
                    ds.height <= confg.hafelekar_height))  # use uniformely level of HAF for Ibk gridpoint from bottom up till lvl 37, 90 (total vert. lvls) - 37 = 53...

    elif model in ["ICON", "ICON2TE"]:
        # for ICON we have different height coordinates (staggered & unstaggered), therefore I chose the height level
        # below Hafelekar with the height var of z (original z_ifc) and used index to be geopot height change-independent
        # (and with that uniformly in space and time)

        # (there is ~ 10% VHD error when taking an additional level for the computation for AROME, just took hafelekar
        # height +50m and looked at VHD...)

        ds_below_hafelekar = ds.isel(height=slice(59, 100))
    elif model == "UM":
        ds_below_hafelekar = ds.where(ds.z <= confg.hafelekar_height, drop=True)  # maybe the others are uselessly
        ds_below_hafelekar = ds.isel(height=slice(0, 21))  # complicated? check again for errors...
    elif model == "WRF":
        ds_below_hafelekar = ds.isel(
            height=ds.z.where(ds.z <= confg.hafelekar_height, drop=True).bottom_top_stag.values)
        ds_below_hafelekar = ds.isel(height=slice(0, 30))
    elif model in ["HATPRO", "radio", "radiosonde"]:
        ds_below_hafelekar = ds.sel(height=slice(0, confg.hafelekar_height))  # for HATPRO

    return ds_below_hafelekar

def calculate_height_diff(ds, model):
    """
    calculates the height difference between model levels (dz) for the given model or Observation, which is needed for
    the VHD calculation (layer thickness to numerically solve the integral)
    :param ds:
    :param model:
    :return:
    """
    ds["dz"] = ds.height.diff("height")
    if model in ["AROME", "ICON", "ICON2TE"]:
        # indices are vice-versa for AROME and ICONs, therefore difference gets negative -> correct here!
        ds["dz"] = -ds["dz"]
    elif model in ["UM", "WRF", "HATPRO", "radio"]:  # AROME & ICON has NaN value for dz at highest lvl -> no problem
        ds["dz"] = ds.dz.shift(height=-1)
        # NaN value is at lowest level -> shift by 1 value to have NaN at highest lvl,  # which is not used in VHD calculation and therefore doesn't cause problems
    return ds


def calc_vhd_single_point(ds_point, model="AROME"):
    """
    calculates the valley heat deficit (VHD) for a single point in a dataset, e.g. for Innsbruck.
    calc density from pressure and temperature using metpy/ ideal gas law: rho = p / (R * T) with R_dryair = 287.05

    ATTENTION: ONLY USE "direct"-timeseries: indexing for HAF height is hardcoded in define_ds_below_hafelekar() cause
    geopot. height changes w. time and space -> "above_terrain"-timeseries is not implemented...

    param ds_point:
    :return: vhd_point: xarray.DataArray with valley heat deficit at the point with time as coord.
    """
    ds_point = calculate_height_diff(ds=ds_point, model=model)

    # ds_point["dz"] = ds_point["dz"]  # sometimes lowest lvl gives NaN, sometimes not?!
    ds_below_hafelekar = define_ds_below_hafelekar(ds=ds_point, model=model)

    th_hafelekar = ds_below_hafelekar.sel(height=ds_below_hafelekar.height.max()).th
    vhd_point = confg.c_p * (
            (th_hafelekar - ds_below_hafelekar.th) * ds_below_hafelekar.rho * ds_below_hafelekar.dz).sum(dim="height")
    # vhd_point = confg.c_p * ((th_hafelekar - th_layers) * rho_layers * dz_values).sum(dim="height")
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
    ds_extent = calculate_height_diff(ds=ds_extent, model=model)

    if model in ["ICON", "ICON2TE"]:  # need to use unstaggered height vars for ICON and invert sign
        ds_extent["dz"] = -ds_extent.z_unstag.diff(dim="height")
    elif model == "WRF":
        ds_extent["dz"] = ds_extent.z_unstag.diff(dim="height")  # only unstaggered for WRF
    else:
        ds_extent["dz"] = ds_extent.z.diff(dim="height")

    if model in ["AROME"]:
        # indices are vice-versa for AROME and ICONs, therefore difference gets negative -> correct here!
        ds_extent["dz"] = -ds_extent["dz"]

    if model == "AROME":  # searched for height-value of Hafelekar in the point-calculation for Ibk gridpoint and use
        # that height for the full domain (searching here with HAF-height doesn't work properly,
        # because some gridpoints are above HAF height...)
        ds_below_hafelekar = ds_extent.sel(height=slice(37, 1))
    elif model in ["ICON", "ICON2TE"]:
        ds_below_hafelekar = ds_extent.sel(height=slice(32, 1))
    elif model == "UM":  # for WRF and UM indices are vice-versa somehow => need to use different order of slicing
        ds_below_hafelekar = ds_extent.sel(height=slice(1, 22))
    elif model == "WRF":
        ds_below_hafelekar = ds_extent.sel(height=slice(1, 31))

    th_hafelekar = ds_below_hafelekar.sel(height=ds_below_hafelekar.height.max()).th
    vhd_full_domain = confg.c_p * (
            (th_hafelekar - ds_below_hafelekar.th) * ds_below_hafelekar.rho * ds_below_hafelekar.dz).sum(dim="height")

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
    aspect_identifier_big = np.abs(
        gpe_model.aspect.values - gpe_dem.aspect.values) > 180  # identifies which formula to use
    model_aspect_calc_big = gpe_model.aspect.where(aspect_identifier_big)
    dem_aspect_calc_big = gpe_dem.aspect.where(aspect_identifier_big)
    ad_aspect_big = ((model_aspect_calc_big.values - dem_aspect_calc_big.values) * np.abs(
        1 - 360 / (np.abs(model_aspect_calc_big.values - dem_aspect_calc_big.values))))
    ad_aspect_big[np.isnan(ad_aspect_big)] = 0  # replace NaN with 0

    # 2. version of aspect calculation, if diff <= 180 degrees
    aspect_identifier_small = np.invert(aspect_identifier_big)
    ad_aspect_small = np.abs(
        gpe_model.aspect.where(aspect_identifier_small).values - gpe_dem.aspect.where(aspect_identifier_small).values)
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


def select_pcgp_vhd(lat=confg.ALL_POINTS["ibk_uni"]["lat"], lon=confg.ALL_POINTS["ibk_uni"]["lon"]):
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
    vhd_arome_pcgp = vhd_arome.sel(lat=pcgp_arome.y.item(), lon=pcgp_arome.x.item(), method="nearest")  # I thought
    # "nearest" isn't needed, but somehow the exact lon of pcgp vhd is not exactly the same as lon of vhd_arome?!
    # difference is f.e. 12.064999 for vhd lon value and 12.065000 for pcgp lon value...
    vhd_icon_pcgp = vhd_icon.sel(lat=pcgp_icon.y.item(), lon=pcgp_icon.x.item(), method="nearest")
    vhd_icon2te_pcgp = vhd_icon2te.sel(lat=pcgp_icon.y.item(), lon=pcgp_icon.x.item(), method="nearest")
    vhd_um = vhd_um.sel(lat=pcgp_um.y.item(), lon=pcgp_um.x.item())  # , method="nearest" needed?
    vhd_wrf_pcgp = vhd_wrf.sel(lat=pcgp_wrf.y.item(), lon=pcgp_wrf.x.item(),
                               method="nearest")  # maybe lies here the problem
    # with the unmatched results from domain & point VHD?!
    return vhd_arome_pcgp, vhd_icon_pcgp, vhd_icon2te_pcgp, vhd_um, vhd_wrf_pcgp


if __name__ == '__main__':
    # matplotlib.use('Qt5Agg')
    pal = sequential_hcl("Terrain")

    # used only once to calc and plot slopes & aspect ratios
    # read_topos_calc_slope_aspect_main()

    # hatpro = xr.open_dataset(f"{confg.hatpro_folder}/hatpro_interpolated_arome.nc")
    # vhd_hatpro = calc_vhd_single_point(hatpro, model="HATPRO")

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
                                                       variant="ICON",
                                                       variables=["p", "temp", "th", "z", "z_unstag", "rho"])
        vhd_ds_icon.append(calc_vhd_full_domain(ds_extent=icon, model="ICON"))

        icon2te = read_icon_model_3D.read_icon_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
                                                          variant="ICON2TE",
                                                          variables=["p", "temp", "th", "z", "z_unstag", "rho"])
        vhd_ds_icon2te.append(calc_vhd_full_domain(ds_extent=icon2te, model="ICON2TE"))

        um = read_ukmo.read_ukmo_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
                                            variables=["p", "temp", "th", "z", "rho"])
        vhd_ds_um.append(calc_vhd_full_domain(ds_extent=um, model="UM"))
        """
        wrf = read_wrf_helen.read_wrf_fixed_time(day=timestamp.day, hour=timestamp.hour, min=timestamp.minute,
                                                 variables=["p", "temp", "th", "z", "z_unstag", "rho"])
        vhd_ds_wrf.append(calc_vhd_full_domain(ds_extent=wrf, model="WRF"))
        """
vhd_arome_full = xr.concat(vhd_ds_arome, dim="time").to_dataset(name="vhd")
vhd_arome_full = vhd_arome_full.assign_attrs(units="J/m^2", description="Valley heat deficit, calculated in calc_vhd.py")
vhd_arome_full.to_netcdf(confg.dir_AROME + "/AROME_vhd_full_domain_full_time.nc")

vhd_icon_full = xr.concat(vhd_ds_icon, dim="time").to_dataset(name="vhd")
vhd_icon_full = vhd_icon_full.assign_attrs(units="J/m^2", description="Valley heat deficit, calculated in calc_vhd.py")
vhd_icon_full.to_netcdf(confg.icon_folder_3D + "/ICON_vhd_full_domain_full_time.nc")

vhd_icon2te_full = xr.concat(vhd_ds_icon2te, dim="time").to_dataset(name="vhd")
vhd_icon2te_full = vhd_icon2te_full.assign_attrs(units="J/m^2", description="Valley heat deficit, calculated in "
                                                                            "calc_vhd.py")
vhd_icon2te_full.to_netcdf(confg.icon2TE_folder_3D + "/ICON2TE_vhd_full_domain_full_time.nc")

vhd_um_full = xr.concat(vhd_ds_um, dim="time").to_dataset(name="vhd")
vhd_um_full = vhd_um_full.vhd.assign_attrs(units="J/m^2", description="Valley heat deficit, calculated in "
                                                                  "calc_vhd.py")
vhd_um_full.to_netcdf(confg.ukmo_folder + "/UM_vhd_full_domain_full_time.nc")

vhd_wrf_full = xr.concat(vhd_ds_wrf, dim="time").to_dataset(name="vhd")
vhd_wrf_full = vhd_wrf_full.assign_attrs(units="J/m^2", description="Valley heat deficit, calculated in "
                                                                    "calc_vhd.py")
vhd_wrf_full.to_netcdf(confg.wrf_folder + "/WRF_vhd_full_domain_full_time.nc")
"""