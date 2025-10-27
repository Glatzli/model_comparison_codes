"""Read in the 3D ICON Model, re-written by Daniel
evtl add variables to calc as in AROME


"""
import sys

sys.path.append("D:/MSc_Arbeit/model_comparison_codes")
import confg
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import datetime


# 3 Mal ", dims=["height"] entfernt"
def convert_calc_variables(ds, variables):
    """
    Converts and calculates meteorological variables for a xarray Dataset.

    Parameters:
    - df: A xarray Dataset containing the columns 'p' for pressure in Pa
          and 'th' for potential temperature in Kelvin.

    Returns:
    - A xarray Dataset with the original data and new columns:
      'pressure' in hPa and 'temperature' in degrees Celsius.
    """
    
    if "p" in variables:
        # Convert pressure from Pa to hPa
        ds['p'] = (ds['p'] / 100.0) * units.hPa
        ds["p"] = ds["p"].assign_attrs(units="hPa", description="pressure")
    
    if "th" in variables:
        # calc pot temp
        ds["th"] = mpcalc.potential_temperature(ds['p'], ds["temp"] * units.kelvin)
        ds["th"] = ds['th'].assign_attrs(units="K", description="potential temperature calced from p and temp")
    
    if "temp" in variables:
        # convert temp to °C
        ds["temp"] = (ds["temp"] - 273.15) * units.degC
        ds["temp"] = ds['temp'].assign_attrs(units="degC", description="temperature")
    
    # ds['qv'] = ds["qv"] * units("kg/kg")  # originally has kg/kg
    
    # calculate relative humidity
    # ds['rh'] = mpcalc.relative_humidity_from_specific_humidity(ds['pressure'], ds["temp"], ds['qv']) * 100  # for
    # percent
    
    # calculate dewpoint
    # ds["Td"] = mpcalc.dewpoint_from_specific_humidity(pressure = ds['pressure'],
    #                                                  specific_humidity = ds['qv']) # , temperature = ds["temp"]
    
    return ds.metpy.dequantify()  # remove units from the dataset


def create_ds_geopot_height_as_z_coordinate(ds):
    """
    create a new dataset with geopotential height as vertical coordinate for temperature for plotting, orig copied from
    AROME
    :param ds:
    :return:
    :ds_new: new dataset with geopotential height as vertical coordinate
    """
    geopot_height = ds.z
    # ds.z_ifc.isel(height_3=slice(1, 91))
    
    ds_new = xr.Dataset(  # somehow lat & lon doesn't work => w/o those coords
        data_vars=dict(th=(["time", "height"], ds.th.values), temp=(["time", "height"], ds.temp.values),
            p=(["time", "height"], ds.p.values), rho=(["time", "height"], ds.rho.values), ),
        coords=dict(height=("height", ds.z.isel(height_3=slice(1, 91)).values),
            # skip most upper level, different height coordinates => just trust in hannes' notes...
            time=("time", ds.time.values)),
        attrs=dict(description="ICON data with z_ifc geometric height at half level center as vertical coordinate"))
    
    return ds_new


def read_full_icon(variant="ICON", variables=["p", "temp", "th", "rho", "z"]):
    """
    (lazy) Read the regridded, full ICON 3D model dataset for a given variant. ~8GB

    """
    data_vars = ["temp", "pres", "qv", "clc", "tke", "z_ifc", "rho", "theta_v", "u", "v", "w"]
    # list of available, original (regridded) ICON variables
    vars_to_calculate = set(variables) - set(data_vars)  # need to calculate the var's that are not in ds and are given
    
    if variant == "ICON":
        ds_path = confg.icon_folder_3D + "/ICON_latlon_subset_tirol.nc"
    elif variant == "ICON2TE":
        ds_path = confg.icon2TE_folder_3D + "/ICON2TE_latlon_subset_tirol.nc"
    else:
        raise ValueError("wrong variant")
    icon_full = xr.open_dataset(ds_path, chunks="auto", engine="netcdf4")
    return icon_full, vars_to_calculate


def rename_icon_variables(ds):
    """
    renames all variables to have a consistent names for all models
    :param ds: ds with original variable names f.e. z_ifc -> z, pres -> p...
    :return: renamed dataset with consistent variable names
    """
    ds = ds.rename({"z_ifc": "z", "pres": "p"})
    return ds


def unstagger_z_point(ds):
    """
    function for unstaggering the geometric height var (orig z_ifc, renamed to z)
    :param ds:
    :return:
    """
    z_unstag = ds.z.rolling(height_3=2, center=True).mean()[1:]
    ds = ds.assign(z_unstag=(("height"), z_unstag.values))
    return ds


def unstagger_z_domain(ds):
    """
    same as unstagger_z_point but only with lat&lon coords for full domain read in
    :param ds:
    :return:
    """
    z_unstag = ds.z.rolling(height_3=2, center=True).mean()[1:]
    ds = ds.assign(z_unstag=(("height", "lat", "lon"), z_unstag.values))
    return ds


def read_icon_fixed_point(lat, lon, variant="ICON", variables=["p", "temp", "th", "rho", "z"], height_as_z_coord=False):
    """
    Read ICON 3D model at a fixed point, edit for consistent names etc:
    1. read full icon ds, 2. select given point, 3. rename vars so that given var string are the consistent names with
    the other models, 4. compute the calc variables like temp, pressure, compute the ds, 5 subset the ds (select wanted
    vars), 6. reverse the height indices to have 1 at bottom (consistent with other models)

    :param lat: latitude of the point
    :param lon: longitude of the point
    :param variant: model variant, either "ICON" or "ICON2TE"
    :param variables: list of variables to select from the dataset with the consistent names-> document in github readme
    :param height_as_z_coord: set unstaggered geopot. height as values for the height coordinate
    """
    icon_full, vars_to_calculate = read_full_icon(variant=variant, variables=variables)
    icon_point = icon_full.sel(lat=lat, lon=lon, method="nearest")
    icon_point = rename_icon_variables(ds=icon_point)  # rename z_ifc to z
    if "z_unstag" in variables:  # if unstaggered geometric height is needed, calc it
        icon_point = unstagger_z_point(ds=icon_point)
    icon_point = convert_calc_variables(icon_point, variables=variables)
    icon_selected = icon_point[variables]  # select only the variables wanted
    if height_as_z_coord:  # set unstaggered geopot. height as height coord. values
        icon_selected["height"] = icon_selected.z_unstag.values[::-1]
    
    icon_selected = icon_selected.compute()
    icon_selected = reverse_height_indices(ds=icon_selected)
    return icon_selected


def read_icon_fixed_time(day=16, hour=12, min=0, variant="ICON", variables=["p", "temp", "th", "rho", "z"]):
    icon_full, vars_to_calculate = read_full_icon(variant=variant)
    
    timestamp = datetime.datetime(2017, 10, day, hour, min, 00)
    icon = icon_full.sel(time=timestamp, method="nearest")  # old, why? height=90, height_3=91,
    icon = rename_icon_variables(ds=icon)  # rename z_ifc to z
    if "z_unstag" in variables:
        icon = unstagger_z_domain(ds=icon)
    icon = convert_calc_variables(icon, variables=variables)
    icon_selected = icon[variables]  # select only the variables wanted
    
    icon_selected = icon_selected.compute()
    icon_selected = reverse_height_indices(ds=icon_selected)
    return icon_selected


def reverse_height_indices(ds):
    """
    turn height cordinate(s) upside down to have 1 at bottom (consistent with other models)
    :param ds: ds with orig height coords
    :return: ds with reversed height coordinates
    """
    if "height" in ds:
        ds = ds.assign_coords(height=ds.height[::-1])
    if "height_2" in ds:
        ds = ds.assign_coords(height_2=ds.height_2[::-1])
    if "height_3" in ds:
        ds = ds.assign_coords(height_3=ds.height_3[::-1])
    return ds


def save_icon_topo(icon_extent):
    """
    save height info as .tif file for pcgp computation i.e. calc of slope & aspect need crs info
    :param icon_extent:
    :return:
    """
    icon_tif = icon_extent.rename({"lat": "y", "lon": "x", "z": "band_data"})  # rename
    icon_tif.rio.write_crs("EPSG:4326", inplace=True)  # add WGS84-projection
    icon_tif.band_data.rio.to_raster(confg.icon_folder_3D + "/ICON_geometric_height_3dlowest_level_w_crs.tif")


if __name__ == '__main__':
    model = "ICON"  # either "ICON" or "ICON2TE"
    # testing cdo generates nc files:
    # icon_latlon = xr.open_dataset(confg.icon_folder_3D + "/ICON_20171015_latlon.nc")
    
    # save_icon_topo(icon_extent)
    
    # save lowest level as nc file for topo plotting
    # icon_extent.z.to_netcdf(confg.icon_folder_3D + "/ICON_geometric_height_3dlowest_level.nc", mode="w",
    # format="NETCDF4")
    # icon_extent.z.rio.to_raster(confg.icon_folder_3D + "/ICON_geometric_height_3dlowest_level.tif")  # for xdem
    # calc of slope I need .tif file
    
    # icon_point = read_icon_fixed_point(lat=confg.ibk_villa["lat"], lon=confg.ibk_villa["lon"], variant=model,
    #                                    variables=["p", "temp", "th", "z", "z_unstag"], height_as_z_coord=True)
    icon_extent = read_icon_fixed_time(day=16, hour=12, min=0, variant="ICON",
                                       variables=["p", "temp", "th", "rho", "z", "z_unstag"], height_as_z_coord=True)
    icon_extent
    
    # icon_plotting = create_ds_geopot_height_as_z_coordinate(icon_point)
    # icon_path = Path(confg.model_folder + f"/{model}/" + f"{model}_temp_p_rho_timeseries_ibk.nc")
    # icon_plotting.to_netcdf(icon_path, mode="w", format="NETCDF4")
    
    # create a new dataset with geometric height z_ifc as vertical coordinate
    """
    icon_plotting = create_ds_geopot_height_as_z_coordinate(icon)
    icon_path = Path(confg.model_folder + "/ICON/" + "ICON_temp_timeseries_ibk.nc")
    icon_plotting.to_netcdf(icon_path, mode="w", format="NETCDF4")
    """  # icon 2te  # icon_2te_path = Path(confg.model_folder + "/ICON2TE/" + "ICON_2TE_temp_timeseries_ibk.nc")  #
    # icon_plotting.to_netcdf(icon_2te_path, mode="w", format="NETCDF4")  # icon_plotting
