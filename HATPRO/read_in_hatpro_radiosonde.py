"""
With this script I read in the HATPRO temp & humidity csv data (all except met-file) and merged it in "main" together,
and saved it as .nc file for a faster read in if it's needed for plotting f.e.
Also interpolated it on AROME levels (easiest for first analysis, maybe include standard atmo?), calc all th & rho for
VHD calc.

(written by Daniel)
"""

import pandas as pd
import xarray as xr
from metpy.units import units
import metpy
import metpy.calc as mpcalc
import confg


def calc_vars(ds):
    """
    calculate all needed vars like pot. temp & rho for VHD calculation
    :param ds:
    :return:
    """
    ds["th"] = mpcalc.potential_temperature(ds["p"] * units.hPa, ds["temp"] * units("degC"))  # calc pot temp
      # using ideal gas law: rho [kg/m^3] = p [Pa] / (R * T [K]) with R_dryair = 287.05 J/kgK
    ds["rho"] = (ds["p"] * 100) / (287.05 * (ds["temp"] + 273.15))
    ds["rho"] = ds['rho'].assign_attrs(units="kg/m^3", description="air density calced from HATPRO temp & AROME p with ideal gas law")
    ds = ds.metpy.dequantify()
    return ds


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
    interpolate the HATPRO data to the AROME model levels and calculate the potential temperature & density with
    AROME pressure and save it as a netcdf file
    :return:
    """
    hatpro_sel = hatpro.sel(time=slice('2017-10-15 12:00:00', '2017-10-16 12:00:00'))  # select modeled period
    hatpro_sel = hatpro_sel.resample(time="30min").mean()  # resample to 1/2 hourly timesteps(as in models)

    # try to use pressure from AROME model to calc pot temp from hatpro data...
    arome["height_above_hatpro"] = arome.z - 612  # the HATPRO station is at 612 m a.s.l., models always habve height abor m.s.l
    # lowest lvl of AROME is still 30m above HATPRO...
    hatpro_interp = hatpro_sel.interp(height=arome.height_above_hatpro)  # interpolate HATPRO to AROME lvls
    hatpro_interp = hatpro_interp.assign(z=(("time", "height"), arome.height_above_hatpro.values))  # add geopot height
    # as variable similar to models

    hatpro_interp["p"] = arome.p # use AROME pressure in HATPRO data, hatpro also get
    # up to height = 75 valid values, above only NaNs, cause AROME data goes farther up...
    hatpro = calc_vars(ds=hatpro_interp)
    hatpro.to_netcdf(f"{confg.hatpro_folder}/hatpro_interpolated_arome.nc")


def edit_vars(df):
    """

    :param df:
    :return:
    """
    df["temp"] = df["temperature"] - 273.15
    df["Td"] = df["dewpoint"] - 273.15
    df["p"] = df["pressure"]
    df = df.rename(columns={"geopotential height": "z", "wind direction": "wind_dir", "windspeed": "wspd"})
    df.drop(["time", "pressure", "latitude offset", "longitude offset", "temperature", "dewpoint"], axis=1, inplace=True)
    return df


def read_radiosonde_csv(filepath):
    # Lese die Datei, überspringe die ersten 5 Kommentarzeilen
    df = pd.read_csv(filepath, comment='#')
    df = edit_vars(df)
    return df


if __name__ == '__main__':
    # merge_save_hatpro()  # only used once to merge the T & rh files, saved again
    # hatpro = xr.open_dataset(f"{confg.hatpro_folder}/hatpro_merged.nc")
    # hatpro
    # arome = xr.open_dataset(confg.dir_AROME + "arome_ibk_uni_timeseries.nc")  # read PCGP around HATPRO for comparing

    # interpolate_hatpro_arome(hatpro, arome)
    radio = read_radiosonde_csv(confg.radiosonde_csv)
    radio.to_csv(confg.radiosonde_edited, index=False)

