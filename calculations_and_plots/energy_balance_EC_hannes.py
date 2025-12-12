"""Script for the `surface energy balance`:
- first you need to prepare_EC_files(), by read_EC_stations() in order to cut it to our time period
- plot EC stations and compare with AROME 2D data (ground 0m)

"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from matplotlib import pyplot as plt
from metpy.units import units

import confg
from confg import dir_EC_stations, EC_30min_final, EC_1min_final, ec_station_names
from read_in_arome import read_2D_variables_AROME


# First subset the large data to our time period (read_EC_stations() and prepare_EC_files())
def read_EC_stations(in_file, out_file):
    """Cut large EC File to keep only our period and set coordinates like in 2D (time, lat, lon) instead of TIME
    index"""
    file_path_ec = os.path.join(dir_EC_stations, in_file)
    # Chain operations to open, set coordinates, swap dimensions, and select time slice
    ds_EC = (xr.open_dataset(file_path_ec).set_coords(['time', 'lat', 'lon']).swap_dims({'TIME': 'time'}).sel(
        time=slice('2017-10-15T14:00:00', '2017-10-16T12:00:00')))

    # Save the Dataset as a NetCDF file
    ds_EC.to_netcdf(out_file)
    print(f"Dataset saved to {out_file}")


def prepare_EC_files():
    """Prepare the two EC files: cut it to our time period and save them as netcdf files"""
    read_EC_stations(in_file="PIANO_EC_FluxData_QC_30min_v1-00.nc", out_file=EC_30min_final)
    read_EC_stations(in_file="PIANO_EC_MetData_QC_1min_v1-00.nc", out_file=EC_1min_final)


def plot_energy_balance(df_obs, name):
    """plot the surface energy balance that was observed at the MOMMAA station and the 2D AROME MODELL"""

    df_obs["nr"] = df_obs["swnet"] + df_obs["lwnet"]
    surface_energy_balance_obs = -(
            df_obs["nr"] * units("W/m^2") - df_obs['h'] * units("W/m^2") - df_obs['le'] * units("W/m^2"))

    fig = go.Figure()

    # Plotting the observed components
    fig.add_trace(
        go.Scatter(x=df_obs['time'], y=df_obs['nr'], mode='lines', name='NET R Observation', marker_color='orange'))
    fig.add_trace(go.Scatter(x=df_obs['time'], y=df_obs['swin'], mode='lines', name='+SW (insolation) Observation',
                             marker_color='gold'))
    fig.add_trace(go.Scatter(x=df_obs['time'], y=-df_obs['swout'], mode='lines', name='-SW (reflection) Observation',
                             marker_color='red'))
    fig.add_trace(go.Scatter(x=df_obs['time'], y=df_obs['lwin'], mode='lines', name='+LW (infrared) Observation',
                             marker_color='cyan'))
    fig.add_trace(go.Scatter(x=df_obs['time'], y=-df_obs['lwout'], mode='lines', name='-LW (infrared) Observation',
                             marker_color='green'))
    fig.add_trace(
        go.Scatter(x=df_obs['time'], y=-df_obs['h'], mode='lines', name='HFS (sensible heat flux) Observation',
                   marker_color='saddlebrown'))

    fig.add_trace(go.Scatter(x=df_obs['time'], y=-df_obs['le'], mode='lines', name='LHF (latent heat flux) Observation',
                             marker_color='royalblue'))
    fig.add_trace(go.Scatter(x=df_obs['time'], y=-df_obs["shf1"], mode='lines', name='Ground flux Observation',
                             marker_color="black", line=dict(width=3)))

    fig.add_trace(go.Scatter(x=df_obs['time'], y=surface_energy_balance_obs, mode='lines', name='Residual Observation',
                             marker_color='purple', line=dict(width=3)))
    # Add AROME Modell

    variable_list = ["hfs", "lfs", "lwd", "lwnet", "lwu", "swd", "swnet", "swu"]
    print("Obs:", df_obs.lon.values, df_obs.lat.values)
    df_arome_2D = read_2D_variables_AROME(variableList=variable_list, lon=df_obs.lon, lat=df_obs.lat,
                                          slice_lat_lon=False)
    print("Model:", df_arome_2D.lon.values, df_arome_2D.lat.values)

    # Add traces for each component of radiation
    fig.add_trace(go.Scatter(x=df_arome_2D['time'], y=df_arome_2D['swd'], mode='lines', name='+SW (insolation) AROME',
                             marker_color='gold', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df_arome_2D['time'], y=df_arome_2D['swu'], mode='lines', name='-SW (reflection) AROME',
                             marker_color='red', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df_arome_2D['time'], y=df_arome_2D['lwd'], mode='lines', name='+LW (infrared) AROME',
                             marker_color='cyan', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df_arome_2D['time'], y=df_arome_2D['lwu'], mode='lines', name='-LW (infrared) AROME',
                             marker_color='green', line=dict(dash='dash')))

    fig.add_trace(go.Scatter(x=df_arome_2D['time'], y=df_arome_2D['hfs'], mode='lines+markers',
                             name='HFS (sensible heat flux) AROME', marker_color='saddlebrown', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df_arome_2D['time'], y=df_arome_2D['lfs'], mode='lines+markers',
                             name='LFS (latent heat flux) AROME', marker_color='royalblue', line=dict(dash='dash')))

    net_radiation = df_arome_2D['swd'] + df_arome_2D['lwd'] + df_arome_2D['swu'] + df_arome_2D['lwu']
    surface_energy_balance = -(net_radiation + df_arome_2D['hfs'] + df_arome_2D[
        'lfs'])  # define it also negative to show the equality for radiation (input)

    fig.add_trace(go.Scatter(x=df_arome_2D['time'], y=net_radiation, mode='lines', name='NET R AROME',
                             line=dict(color='orange', dash='dash')))
    fig.add_trace(go.Scatter(x=df_arome_2D['time'], y=surface_energy_balance, mode='lines', name='Residual AROME',
                             line=dict(color='purple', width=3, dash='dash')))

    # Update layout
    fig.update_layout(title=f'Simplified Surface Energy Balance for 2D (surface) AROME Modell and {name}',
                      xaxis_title='Time (UTC)', yaxis_title='Energy rate (W/m²) (positive = towards the surface)',
                      legend_title='Legend')

    # fig.write_html(f"{dir_PLOTS}/energy_balance/EC_stationsObs_vs_2D_AROME/sfc_energy_balance_{name}.html")

    fig.show()


def prepare_EC_datasets(station_key):
    """ from Hannes, edited by daniel
    read in newly created datasets and select the desired station_key
    """
    ds_30min = xr.open_dataset(EC_30min_final)  # open previously edited & cut dataset (not orig from zenodo)
    ds_1min = xr.open_dataset(EC_1min_final)

    df_obs_1min = ds_1min.sel(STATION_KEY=station_key)  # select the current, desired station via station key
    df_obs_30min = ds_30min.sel(STATION_KEY=station_key)

    # don't have swu & swd in 30 min data, therefore interpolate 1min data
    df_obs_30min_interp = df_obs_1min.interp(time=df_obs_30min.time.values, method="linear")
    # also set sign correctly to atm-earth: positive & earth-atm: negative, took only necessary vars
    df_obs_30min = df_obs_30min.assign(swd=df_obs_30min_interp["swin"], swu=-df_obs_30min_interp["swout"],
                                       lwd=df_obs_30min_interp["lwin"], lwu=-df_obs_30min_interp["lwout"],
                                       hfs=df_obs_30min["h"], lfs= df_obs_30min["le"])
    return df_obs_30min


def read_in_EC_datasets():

    ds_orig_30min = xr.open_dataset(confg.dir_EC_stations + "/PIANO_EC_FluxData_QC_30min_v1-00.nc")
    ds_orig_30min.set_coords(['time', 'lat', 'lon']).swap_dims({'TIME': 'time'}).sel(
        time=slice('2017-10-15T12:00:00', '2017-10-16T12:00:00'))


def detect_NAN_values():
    """detect NAN values for all 4 stations in EC observations"""
    dataset = pd.DataFrame()

    # Iterate over a range of station keys
    for key in np.arange(0, 4):
        df_obs_30min = prepare_EC_datasets(key)  # Call your function to prepare the dataset for the current station key

        # Convert the xarray Dataset to a pandas DataFrame and count NaN values
        nan_counts = df_obs_30min.to_dataframe().isnull().sum()
        dataset[f'Station_{key}'] = nan_counts

    # Transpose the DataFrame so that the variable names become the index
    dataset = dataset.T

    # If variable names are in the column index after transposition, set them as the DataFrame index
    if 'variable' in dataset.columns:
        dataset.set_index('variable', inplace=True)

    dataset.to_csv("Nan_values.csv")
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot a bar for each column in the dataset
    dataset[["ta", "humidity", "pa", "ts1", "wspeed", "swin", "swout", "lwin", "lwout", "shf1", "h", "le"]].plot(
        kind='bar', ax=ax, width=0.8)

    # Set plot title and labels
    ax.set_title('NaN Counts per Variable Across Station Keys', fontsize=16)
    ax.set_xlabel('Variables', fontsize=14)
    ax.set_ylabel('Count of NaN Values', fontsize=14)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Show legend
    ax.legend(title='Station Keys')

    # Show the plot
    plt.tight_layout()  # plt.savefig("../Plots/Arome_first_plots/NAN_Values_EC.png")


if __name__ == '__main__':
    """plot the surface energy balance of AROME"""
    # prepare_EC_files()  # to prepare "raw", original EC files from Zenodo
    # detect_NAN_values()  # filter again for NaNs
    # key = 0: ~ airport a bit south of it key = 1: EC_south (Patsch), key = 3: EC_C ...
    for key in np.arange(0, 4):
        df_obs_30min = prepare_EC_datasets(key)
        name = ec_station_names[key]["name"]

        plot_energy_balance(df_obs_30min,
                            name)