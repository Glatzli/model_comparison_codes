"""
Process and interpolate HATPRO and radiosonde data for model comparison.

The radiosonde data is calculated/transformed & saved in different formats (last few lines in main):
1. original, CSV-data: 2017101603_bufr309052.csv
2. deleted unused data, CSV: radiosonde_ibk_2017101603.csv
3. with calculated th & rho, transformed to dataset (with "height 0, 1, 2, ...") as z coord

Function read_radiosonde_dataset reads the radiosonde dataset with 3 different height options (as in the models, see fct
description):

The problem with this code is now that I used timerseries with the model levels as vertical coordinate => would need to
change the indexing for the code to work (to interpolate the radiosonde data f.e...)

(written by Daniel)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import pandas as pd
import xarray as xr
from metpy.units import units

import confg

mpl.use('Qt5Agg')


def calc_vars_hatpro_w_pressure(ds):
    """
    calculate pot temp & density of HATPRO data with added pressure
    formerly also used to compute it for Raso pressure and that of a model, but sens. analysis showed no difference...
    Also we have absolute humidity by default from HATPRO, not spec. humidity as for models => convert
    :param ds:
    :return:
    """
    ds["th"] = mpcalc.potential_temperature(ds["p"] * units.hPa, ds["temp"] * units("degC"))  # using model p
    ds["th"] = ds['th'].assign_attrs(units="K", description="pot. temp calced from HATPRO temp & AROME p with metpy")
    
    # ds["th_model"] = mpcalc.potential_temperature(ds["p_model"] * units.hPa,
    #                                               ds["temp"] * units("degC"))  # using model p
    # ds["th_raso"] = mpcalc.potential_temperature(ds["p_raso"] * units.hPa, ds["temp"] * units("degC"))  # using raso p
    
    # using ideal gas law: rho [kg/m^3] = p [Pa] / (R * T [K]) with R_dryair = 287.05 J/kgK
    ds["rho"] = (ds["p"] * 100) / (287.05 * (ds["temp"] + 273.15))
    # ds["rho_model"] = (ds["p_model"] * 100) / (287.05 * (ds["temp"] + 273.15))
    # ds["rho_raso"] = (ds["p_raso"] * 100) / (287.05 * (ds["temp"] + 273.15))
    ds["rho"] = ds['rho'].assign_attrs(units="kg/m^3",
                                       description="air density calced from HATPRO temp & AROME p with ideal gas law")
    # convert absolute humidity [g/m^3] to specific humidity [kg/kg]
    ds["q"] = ds["absolute_humidity"] / (ds["rho"] * 1000)
    ds["q"] = ds['q'].assign_attrs(units="kg/kg",
                                   description="spec. humidity calculated from absolute humidity (HATPRO) & density ("
                                               "interp. AROME)")
    ds = ds.metpy.dequantify()
    return ds


def calc_vars_radiosonde(df):
    """
    calculate potential temp & density of radiosonde data
    :param df:
    :return:
    """
    df["th"] = mpcalc.potential_temperature(pressure=(df['p'].values) * units.mbar,
                                            temperature=(df["temp"].values + 273.15) * units.kelvin)
    df["rho"] = (df["p"].values * 100) / (287.05 * (df["temp"].values + 273.15))
    df["q"] = mpcalc.specific_humidity_from_dewpoint(pressure=df["p"].values * units.hPa,
                                                     dewpoint=df["Td"].values * units.degC)
    return df


def calc_vars_hatpro_radio_wrf(radio_wrf_interp, hatpro_arome_interp, hatpro_icon_interp, hatpro_wrf_interp, arome,
                               wrf):
    """
    deprecated?
    adds pressure either from the radiosonde data or a model to the HATPRO data

    calculate pot temp for HATPRO using radiosonde and model pressures, compare them afterwards for a sensitivity
    analysis

    :param radio_wrf_interp: radiosonde data interpolated to WRF levels
    :param hatpro_arome_interp: hatpro data interpolated to AROME levels
    :return:
    """
    # hatpro = hatpro.isel(height=slice(0, 80))
    hatpro_radiop = hatpro_wrf_interp.assign(p=(("height"), radio_wrf_interp.p.values))  # assign
    hatpro_radiop = calc_vars_hatpro_w_pressure(ds=hatpro_radiop)
    
    hatrpo_aromep = hatpro_arome_interp.assign(p=(("height"), arome.p.values))
    hatpro_aromep = calc_vars_hatpro_w_pressure(ds=hatrpo_aromep)
    
    hatpro_iconp = hatpro_arome_interp.assign(p=(("height"), arome.p.values))  # some error, do that tomorrow...
    hatpro_aromep = calc_vars_hatpro_w_pressure(ds=hatrpo_aromep)
    
    hatpro_wrfp = hatpro_wrf_interp.assign(p=(("height"), wrf.p.values))
    hatpro_wrfp = calc_vars_hatpro_w_pressure(ds=hatpro_wrfp)
    
    plt.figure(figsize=(12, 8))
    hatpro_radiop.isel(time=12).th.plot(y="height", label="p from radiosonde")
    hatpro_aromep.isel(time=12).th.plot(y="height", label="p from AROME")
    hatpro_wrfp.isel(time=12).th.plot(y="height", label="p from WRF")
    plt.legend()
    plt.show()


def read_hatpro(filepath):
    """internally used function to read in hatpro Temperature or Humidity depending on the filepath (height in meter)
    originally __read_hatpro_intern(filepath) from hannes"""
    height_int = [int(height) for height in confg.hatpro_vertical_levels["height"]]
    
    # Read in the DataFrame from the CSV file
    df = pd.read_csv(filepath, sep=";")
    
    # Convert the 'rawdate' column to datetime if it's not already
    df['rawdate'] = pd.to_datetime(df['rawdate'])
    
    # Set the 'rawdate' column as the index
    df.set_index('rawdate', inplace=True)
    
    # Rename the columns to v01, v02, ..., v39
    df.columns = [f"v{i:02d}" for i in range(1, 40)]
    
    # Create a new index that includes 'rawdate' and 'v1' to 'v39'
    new_index = pd.MultiIndex.from_product([df.index, df.columns], names=['time', 'height_level'])
    
    # Create a new DataFrame with the new index
    if "temp" in filepath:
        df_new = pd.DataFrame(index=new_index, data=df.values.flatten(), columns=['th'])
    elif "humidity" in filepath:
        df_new = pd.DataFrame(index=new_index, data=df.values.flatten(), columns=['humidity'])
    
    # Convert the DataFrame to an xarray dataset
    ds = xr.Dataset.from_dataframe(df_new)
    
    # Assign the 'height_level' coordinate
    ds["height_level"] = height_int
    """
    if "T" in list(ds.keys()):
        # Set the units attribute for temperature variable 'T'
        # ds["T"].attrs['units'] = "K"

        # ds["T"].values = ds["T"].values * units.kelvin
        # ds["T"] = ds["T"].metpy.convert_units("degC")
    elif "humidity" in list(ds.keys()):
        # ds["humidity"].attrs['units'] = "g/m^3"  # absolute humidity
        # ds['humidity'] = ds['humidity'].metpy.convert_units("g/m^3")

        # print(ds["humidity"])
    """
    return ds


def merge_save_hatpro():
    # read hatpro data and save it as a merged .nc file
    filepath = f"{confg.hatpro_folder}/data_HATPRO_temp.csv"
    hatpro_temp = read_hatpro(filepath)
    filepath = f"{confg.hatpro_folder}/data_HATPRO_humidity.csv"
    hatpro_humidity = read_hatpro(filepath)
    # Merge the temp & humidity datasets
    hatpro = xr.merge([hatpro_temp, hatpro_humidity])
    hatpro = hatpro.rename({"height_level": "height"})  # rename coordinate name for uniform name
    hatpro["temp"] = hatpro["th"] - 273.15  # add temperature to have consistency with models
    hatpro = hatpro.drop_vars("th")  # drop "th" because that is pot temp in my naming!
    
    # Save the merged dataset to a NetCDF file
    hatpro.to_netcdf(f"{confg.hatpro_folder}/hatpro_merged.nc")


def interpolate_hatpro_arome(hatpro, arome):
    """
    interpolate the AROME data to the HATPRO model levels to add pressure levels and calculate the potential
    temperature & density with AROME pressure, save it as netcdf file
    :param hatpro: merged HATPRO dataset (temp & humidity, raw data w. 10 min timesteps)
    :param arome: AROME timeseries dataset at PCGP around HATPRO station (above_terrain height as z coord,
    lowest lvl is at 5.1 m...)
    :return:
    """
    hatpro_sel = hatpro.sel(time=slice('2017-10-15 12:00:00', '2017-10-16 12:00:00'))  # select modeled period
    hatpro_sel = hatpro_sel.resample(time="30min").mean()  # resample to 1/2 hourly timesteps(as in models)
    hatpro_sel = hatpro_sel.rename_vars({"humidity": "absolute_humidity"})  # rename for clarity: it's abs. humidity!
    
    # use pressure from AROME model to calc pot temp from hatpro data...
    # first bring HATPRO height lvl to height above m.s.l.:
    hatpro_sel["height"] = hatpro_sel.height + confg.ibk_uni["height"] - confg.ibk_villa[
        "height"]  # = 33m height difference between HATPRO station and normal ibk level
    
    # is it better to interpolate HATPRO to AROME or vice versa? AROME has lowest lvl at 5.1 m, which is far below
    # lowest HATPRO lvl at 33 m above topography...
    # => interpolate AROME values to HATPRO levels to get pressure...
    arome_interp = arome.interp(height=hatpro_sel.height)
    hatpro_sel["p"] = arome_interp.p
    
    # up to height = 75 valid values, above only NaNs, cause AROME data goes further up...
    hatpro_w_pressure = calc_vars_hatpro_w_pressure(ds=hatpro_sel)
    hatpro_w_pressure.to_netcdf(confg.hatpro_calced_vars)


def interpolate_hatpro_radiosonde(hatpro, radio, arome, icon, wrf):
    """
    interpolate hatpro & radiosonde data onto model-geopot (or geometric for ICON) height levels and add pressure
    values of the models & radiosonde directly to the dataset for comparison
    caution: HATPRPO's height difference between levels gets bigger with increasing
    height, for models that distance stays the same -> further up the interpolation
    error increases for the HATPRO-data

    :param hatpro:
    :param radio:
    :param wrf:
    :return:
    :param hatpro_arome_interp: hatpro data interpolated onto AROME levels
    :param radio_arome_interp: radiosonde data interpolated onto AROME levels
    """
    # interpolate radiosonde:
    radio["height"] = radio.z.values  # need geopot height as coordinate values for interpolation
    radio = radio.drop_vars("z")
    # Auch radio.height eindeutig machen (falls nötig)
    radio_mean = radio.groupby("height").mean(dim="height")  # radiosonde data has mutliple meas. on the same heights
    # => take mean over all values on the same height
    radio_arome_interp = radio_mean.interp(height=arome.isel(time=25).z.values[::-1], method="linear")
    radio_icon_interp = radio_mean.interp(height=icon.z_unstag.values[::-1], method="linear")
    radio_wrf_interp = radio_mean.interp(height=wrf.isel(time=0).z_unstag.values,
                                         method="linear")  # and then interpolate it onto WRF levels
    
    hatpro_arome_interp = hatpro.interp(height=arome.isel(time=25).z.values[::-1],
                                        method="linear")  # interp. HATPRO to AROME lvls
    hatpro_arome_interp = hatpro_arome_interp.assign(
        p_model=(("time", "height"), arome.isel(height=slice(None, None, -1)).p.values),  # add model & raso
        p_raso=(("height"), radio_arome_interp.p.values))  # pressure to dataset
    hatpro_icon_interp = hatpro.interp(height=icon.z_unstag.values[::-1], method="linear")  # interpolate HATPRO to
    # unstaggered icon geometric height vals
    hatpro_icon_interp = hatpro_icon_interp.assign(
        p_model=(("time", "height"), icon.isel(height=slice(None, None, -1)).p.values),
        p_raso=(("height"), radio_icon_interp.p.values))
    hatpro_wrf_interp = hatpro.interp(height=wrf.z_unstag.values, method="linear")
    hatpro_wrf_interp = hatpro_wrf_interp.assign(p_model=(("height"), wrf.p.values),
                                                 p_raso=(("height"), radio_wrf_interp.p.values))
    """
    plt.figure(figsize=(2, 5))  # check if unstaggering worked
    plt.plot(["z"] * len(wrf.z.values), wrf.z.values, linestyle="None", ms=5, marker="_")
    plt.plot(["z_stag"] * len(wrf.z_stag.values), wrf.z_stag.values, linestyle="None", ms=5, marker="_")
    plt.grid()
    plt.show()
    """
    return hatpro_arome_interp, hatpro_icon_interp, hatpro_wrf_interp


def smooth_radiosonde(radio):
    """
    deprecated -> chose cap height manually...
    
    Smoothes radiosonde data with rolling mean over specified window size.
    Needed for computing CAP height (which is determined by 3 levels of dT/dz < 0 K/m).
    With raw data CAP would be calculated too low (due to small scale "noise").
    Increased window size to better match vertical resolution of models (~30-100m) and HATPRO (~10-50m).
    :param radio: xarray Dataset with radiosonde data
    :return: smoothed radiosonde Dataset
    """
    # Entferne Duplikate durch Mittelwertbildung über gleiche Höhen
    radio_unique = radio.groupby("height").mean(dim="height")
    
    # Glättung über ein Fenster von 50 Leveln
    radio_smoothed = radio_unique.rolling(height=15, center=True, min_periods=12).mean().dropna(dim="height")
    
    # Füge den niedrigsten Punkt wieder hinzu
    radio_smoothed = xr.concat([radio_unique.sel(height=1, method="nearest"), radio_smoothed], dim="height")
    
    radio_smoothed.to_netcdf(confg.radiosonde_smoothed)


def edit_vars(df):
    """
    convert temp to degC, dewpoint to degC, pressure to hPa, rename variables and drop unused vars
    :param df:
    :return:
    """
    df["temp"] = df["temperature"] - 273.15
    df["Td"] = df["dewpoint"] - 273.15
    df["p"] = df["pressure"] / 100  # pressure in hPa
    df = df.rename(columns={"geopotential height": "z", "wind direction": "udir", "windspeed": "wspd"})
    df.drop(["time", "pressure", "latitude offset", "longitude offset", "temperature", "dewpoint"], axis=1,
            inplace=True)
    return df


def read_radiosonde_csv(filepath):
    """
    read in original radiosonde data
    and edit varibles to uniform units & names
    :param filepath:
    :return:
    """
    # Lese die Datei, überspringe die ersten 5 Kommentarzeilen
    df = pd.read_csv(filepath, comment='#')
    df = edit_vars(df)
    return df


def convert_to_dataset(radio):
    """
    converts edited radiosonde dataframe to dataset (only because handling is then the same as for model data...)
    throws away first datapoint, which is NaN data
    :param radio: finished edited radiosonde dataframe
    :return:
    :param ds: dataset of edited radiosonde data
    """
    # ToDo set coords time & others...
    #   * time     (time) datetime64[ns] 392B 2017-10-15T12:00:00 ... 2017-10-16T12...
    #     lat      float32 4B 47.28
    #     lon      float32 4B 11.4
    ds = xr.Dataset({col: ("height", radio[col].values[1::]) for col in radio.columns},
                    coords={"height": radio.index.values[1::]})
    ds["th"] = ds['th'].assign_attrs(units="K", description="pot. temp calced from p & temp with metpy")
    ds["q"] = ds['q'].assign_attrs(units="kg/kg", description="spec. hum. calced from p & Td with metpy")
    ds["rho"] = ds['rho'].assign_attrs(units="kg/m^3",
                                       description="air density calced from p & temp with ideal gas law")
    return ds


def plot_height_levels(arome_heights, icon_heights, um_heights, wrf_heights, radio_heights, hatpro_heights=None):
    """
    plots the different geopot. height levels for all models incl. radiosonde & hatpro data, for
    getting an overview which data should be interpolated onto which levels...
    :param arome_heights:
    :param icon_heights:
    :param um_heights:
    :param wrf_heights:
    :param radio_heights:
    :param hatpro_heights:
    :return:
    """
    model_names = ['AROME', 'ICON', 'UM', 'WRF', 'Radiosonde', 'HATPRO']
    height_arrays = [arome_heights, icon_heights, um_heights, wrf_heights, radio_heights, hatpro_heights]
    
    plt.figure(figsize=(5, 8))
    for i, heights in enumerate(height_arrays):
        plt.plot([model_names[i]] * len(heights), heights, linestyle="None", label=model_names[i], ms=5, marker="_")
    
    plt.ylabel('geopotential height [m]')
    plt.ylim([500, 3000])
    plt.title('vertical level comparison models & measurments')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(confg.dir_PLOTS + "model_infos_general/vertical_geopot_height_levels_comp.svg")
    plt.show()


def interpolate_hatpro(arome, icon, wrf):
    """
    only used once
    functionality used to interpolate HATPRO to all different model levels, tested also sensitivity of it. I.e.
    interpolated
    it and looked at pot. temp differences between interpolations:
        # the difference between using the pressure from the Radiosonde or the model pressures is approx 0.5K at 3000m
        # for the pot. temp calc => using any other pressure makes no difference! even for break-up-phase at 10:30 in
        lowest
        # lvls only 0.5 K difference...
    :return:
    """
    
    (hatpro_arome_interp, hatpro_icon_interp, hatpro_wrf_interp) = interpolate_hatpro_radiosonde(hatpro=hatpro_sel,
                                                                                                 radio=radio,
                                                                                                 arome=arome, icon=icon,
                                                                                                 wrf=wrf)  # for
    # interpolating HATPRO data to
    
    # sensitivity test of HATPRO pot. temp calculation with model p's and raso p:
    hatpro_arome_interp = calc_vars_hatpro_w_pressure(ds=hatpro_arome_interp)
    # hatpro_icon_interp = calc_vars_hatpro_w_pressure(ds=hatpro_icon_interp)
    # hatpro_wrf_interp = calc_vars_hatpro_w_pressure(ds=hatpro_wrf_interp)
    
    """
    plt.figure(figsize=(12, 8))
    hatpro_arome_interp.isel(time=45).th_model.plot(y="height", label="HATPRO interp AROME using AROME p")
    hatpro_arome_interp.isel(time=45).th_raso.plot(y="height", label="HATPRO interp AROME using Raso p")
    hatpro_icon_interp.isel(time=45).th_model.plot(y="height", label="HATPRO interp ICON using model p")
    hatpro_icon_interp.isel(time=45).th_raso.plot(y="height", label="HATPRO interp ICON using Raso p")
    hatpro_wrf_interp.isel(time=45).th_model.plot(y="height", label="HATPRO interp WRF using model p")
    hatpro_wrf_interp.isel(time=45).th_raso.plot(y="height", label="HATPRO interp WRF using Raso p")
    plt.legend()
    plt.show()"""
    # the difference between using the pressure from the Radiosonde or the model pressures is approx 0.5K at 3000m
    # for the pot. temp calc => using any other pressure makes no difference! even for break-up-phase at 10:30 in lowest
    # lvls only 0.5 K difference...
    
    # take the pressure of radiosonde for th and save dataset again (once with height index in z and once with geopot
    # height in z):
    hatpro_arome_interp["th"] = hatpro_arome_interp["th_raso"]
    
    (hatpro_arome_interp["th"], hatpro_arome_interp["p"], hatpro_arome_interp["rho"]) = (hatpro_arome_interp["th_raso"],
                                                                                         hatpro_arome_interp["p_raso"],
                                                                                         hatpro_arome_interp[
                                                                                             "rho_raso"])
    
    hatpro = hatpro_arome_interp.drop_vars(["p_model", "p_raso", "th_model", "th_raso", "rho_model", "rho_raso"])
    hatpro.to_netcdf(confg.hatpro_interp_arome_height_as_z)


def read_radiosonde_dataset(height_as_z_coord: str | bool = "direct"):
    """
    read in radiosonde dataset with geopot. height as z coordinate either direct or as height above terrain (lowest
    point
    1m above ground)
    :param height_as_z_coord: either "direct", "above_terrain" or False, None (then orig. indices are height var)
    :return:
        radiosonde dataset with geopot. height as z coordinate either amsl, above terrain or original height indices
    """
    radio = xr.open_dataset(confg.radiosonde_dataset)
    if height_as_z_coord == "direct":
        radio["height"] = radio["z"]
        radio["height"] = radio["height"].assign_attrs(units="m", description="geopotential height amsl")
    elif height_as_z_coord == "above_terrain":
        radio["height"] = radio.z - radio.isel(height=0).z + 1
        radio["height"] = radio["height"].assign_attrs(units="m", description="geopotential height above terrain")
    
    return radio.compute()


if __name__ == '__main__':
    """
    yes, there's some legacy code in here for transforming & saving hatpro & radiosonde data that I used only once,
    which
    is now not needed anymore. But I left it in here for documentation purposes."""
    
    # merge_save_hatpro()  # only used once to merge the T & rh files, saved again
    hatpro = xr.open_dataset(confg.hatpro_merged)
    hatpro_sel = hatpro.sel(time=slice('2017-10-15 12:00:00', '2017-10-16 12:00:00'))  # select modeled period
    # hatpro_sel = hatpro_sel.resample(time="30min").mean()  # resample to 1/2 hourly timesteps(as in models)
    # hatpro_sel["height"] = hatpro_sel.height + 612  # the HATPRO station is at 612 m a.s.l., models always have
    # height above m amsl.
    
    arome = xr.open_dataset(confg.dir_AROME + "/timeseries/" + "arome_ibk_uni_timeseries_height_as_z.nc")
    # read PCGP around HATPRO for comparing
    
    # icon = xr.open_dataset(confg.icon_folder_3D + "/timeseries/" + "/icon_ibk_uni_timeseries_height_as_z.nc")
    # um = xr.open_dataset(confg.ukmo_folder + "/timeseries/" + "um_ibk_uni_timeseries_height_as_z.nc")
    # wrf = xr.open_dataset(confg.wrf_folder + "/timeseries/" + "/wrf_ibk_uni_timeseries_height_as_z.nc")
    # interpolate_hatpro_arome(hatpro, arome)
    
    # plot_height_levels(arome_heights=arome.isel(time=0).z.values[::-1], icon_heights=icon.z.values[::-1],
    #                    um_heights=um.isel(time=0).z.values, wrf_heights=wrf.isel(time=0).z.values,  #  #  #  #  #
    #                    radio_heights=radio.z.values, hatpro_heights=hatpro_sel.height.values)
    
    # (not used anymore) smoothing radiosonde and save it: # radio_smoothed = smooth_radiosonde(radio)
    

    # lines used for reading orig. radiosonde data & manipulating it, calcing th & rho and saving it as a dataset
    radio_orig = read_radiosonde_csv(confg.radiosonde_csv)  # read in orignal radiosonde data and transform to
    # uniform units, drop not needed vars
    
    radio = calc_vars_radiosonde(df=radio_orig)  # calc pot temp, rho & q
    radio_ds = convert_to_dataset(radio)  # modify radiosonde data (calc th & rho) and save it as dataset
    radio_ds.to_netcdf(confg.radiosonde_dataset)  # save it as .nc file (dataset)

    
    radio = read_radiosonde_dataset(height_as_z_coord="above_terrain")
    radio
