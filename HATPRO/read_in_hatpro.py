"""
With this script I read in the HATPRO temp & humidity csv data (all except met-file) and merged it in "main" together,
 and saved it as .nc file for a faster read in if it's needed for plotting f.e. (written by Daniel)
"""

import pandas as pd
import xarray as xr
from metpy.units import units
import metpy
import metpy.calc as mpcalc
import confg


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


def interpolate_hatpro_arome():
    """
    interpolate the HATPRO data to the AROME model levels and calculate the potential temperature with AROME pressure,
    and save it as a netcdf file
    :return:
    """
    arome = xr.open_dataset(confg.model_folder + "/AROME/AROME_temp_timeseries_ibk.nc")
    hatpro = xr.open_dataset(f"{confg.hatpro_folder}/hatpro_merged.nc")
    # arome

    # try to use pressure from AROME model to calc pot temp from hatpro data...
    arome["height_above_ibk"] = arome.height - 612  # the HATPRO station is at 612 m a.s.l., lowest lvl of AROME is above...
    hatpro_interp = hatpro.interp(height=arome.height_above_ibk)

    # start_time = pd.to_datetime('2017-10-15 12:00:00', format='%Y-%m-%d %H:%M:%S')
    # end_time = pd.to_datetime('2017-10-16 12:00:00', format='%Y-%m-%d %H:%M:%S')
    hatpro_interp = hatpro_interp.sel(time=slice('2017-10-15 12:00:00', '2017-10-16 12:30:00'))
    hatpro_interp = hatpro_interp.resample(time="30min").mean()

    hatpro_interp["pressure"] = arome.pressure.compute()  # use AROME pressure in HATPRO data

    # hatpro_interp['pressure'] = (hatpro_interp['p'] / 100.0) * units.hPa
    # calc temp
    hatpro_interp["th"] = mpcalc.potential_temperature(hatpro_interp["pressure"] * units.hPa, hatpro_interp["temp"] * units("degC"))
    hatpro_interp = hatpro_interp.metpy.dequantify()
    hatpro_interp = hatpro_interp.rename({"pressure": "p"})
    hatpro_interp.to_netcdf(f"{confg.hatpro_folder}/hatpro_interpolated_arome.nc")


def interpolate_hatpro_arome_add_density():
    """
    interpolate the HATPRO data to the AROME model levels and calculate the potential temperature with AROME pressure and
    append calculated AROME density, espc for VHD calculation...
    :return:
    """
    arome = xr.open_dataset(confg.dir_AROME + "arome_ibk_uni_timeseries.nc")  # because that point is the PCGP around
    # the uni, it is not directly at the HATPRO but a bit easterly (although no point is exactly at the HATPRO station!)
    hatpro = xr.open_dataset(f"{confg.hatpro_folder}/hatpro_merged.nc")
    # hatpro
    hatpro["height"] = hatpro.height + 612  # Hatpro heights begin at 0 but HATPRO station is at 612 m a.s.l. => correct

    # hatpro.interp(height=arome.isel(time=1).z.compute().values)
    hatpro_interp = hatpro.interp(height=arome.isel(time=1).z.compute().values, method="linear")  # is that right?!


    hatpro_interp.isel(time=0).temp  # for checking...


if __name__ == '__main__':
    # merge_save_hatpro()
    #hatpro = xr.open_dataset(f"{confg.hatpro_folder}/hatpro_merged.nc")
    # hatpro
    # hatpro_interp = xr.open_dataset(f"{confg.hatpro_folder}/hatpro_interpolated_arome.nc")
    # interpolate_hatpro_arome()
    interpolate_hatpro_arome_add_density()
