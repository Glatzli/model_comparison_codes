"""deprecated i.e. from Hannes, right?

Script to make some plots of the innsbruck radiosonde (observations and AROME)
Optional:
-   change time of the model 'time_for_model'
-   plot_radiosondes_of_adjacent_points() 6 nearest points
-   instead of using the nearest point (method=sel) use method=interp (interpolation between nearest points)"""
import math
import warnings

import matplotlib.pyplot as plt
import metpy
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
from matplotlib import gridspec
from metpy.plots import SkewT, Hodograph
from metpy.units import units

from read_in_arome import read_3D_variables_AROME
from confg import radiosonde_csv, station_files_zamg, dir_PLOTS

warnings.filterwarnings("ignore")

# Specify the launch date and time
launch_date = pd.to_datetime('2017-10-16 02:15:05',
                             format='%Y-%m-%d %H:%M:%S')

time_for_model = pd.to_datetime('2017-10-16 03:00:00',
                                format='%Y-%m-%d %H:%M:%S')


def plot_radiosonde_ibk_obs():
    """read csv and plot the radiosonde observation of the airport of ibk"""
    df = pd.read_csv(radiosonde_csv, comment="#")

    # Drop rows with NaN values in any column
    df_cleaned = df.dropna()

    # Convert 'time' from seconds since launch to actual datetime
    df_cleaned['time'] = pd.to_timedelta(df_cleaned['time'], unit='s') + launch_date

    # convert also the units of some other variables
    df_cleaned.loc[:, 'pressure'] = df.loc[df_cleaned.index, 'pressure'] / 100
    df_cleaned.loc[:, 'dewpoint'] = df.loc[df_cleaned.index, 'dewpoint'] - 273.15
    df_cleaned.loc[:, 'temperature'] = df.loc[df_cleaned.index, 'temperature'] - 273.15

    df_unique = df_cleaned.drop_duplicates(['time', 'geopotential height'])

    # Then set the index and convert to xarray
    df_unique.set_index(['time'], inplace=True)

    u = df_unique["windspeed"].values * units.meter / units.second

    p = df_unique['pressure'].values * units.hPa
    T = df_unique['temperature'].values * units.degC
    Td = df_unique['dewpoint'].values * units.degC
    wind_speed = u.to(units.knots)
    wind_dir = df_unique['wind direction'].values * units.degrees
    u, v = mpcalc.wind_components(wind_speed, wind_dir)
    z = df_unique["geopotential height"].values * units.meter

    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.55, 0.85))

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    skew.plot(p, T, 'r')
    skew.plot(p, Td, 'g')
    # Mask barbs to below 100 hPa only
    mask = p >= 100 * units.hPa

    # Plot wind barbs
    skew.plot_barbs(p[mask][::20], u[mask][::20], v[mask][::20], y_clip_radius=0.01)

    # Change to adjust data limits and give it a semblance of what we want
    skew.ax.set_adjustable('datalim')
    skew.ax.set_ylim(1000, 100)
    skew.ax.set_xlim(-20, 30)

    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    # Add some descriptive titles

    mask = z <= 10 * units.km

    # Custom colorscale for the wind profile
    intervals = np.array([0, 1, 3, 5, 10]) * units.km
    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:olive']
    # Convert the rounded values into strings with the desired format
    str_lat = f'{station_files_zamg["LOWI"]["lat"]:.3f}_N'
    str_lon = f'{station_files_zamg["LOWI"]["lon"]:.3f}_E'

    plt.title(f'{str_lat} {str_lon}', loc='left')
    plt.title(f'Launch Time: {pd.to_datetime(launch_date).strftime("%Y-%m-%d %H:%M")}',
              loc='right')
    skew.ax.set_xlabel('Temperature (°C)')
    skew.ax.set_ylabel('Pressure (hPa)')

    # Create a hodograph
    ax1 = plt.axes((0.7, 0.75, 0.2, 0.2))
    h = Hodograph(ax1, component_range=40.)
    h.add_grid(increment=5)
    u1 = u.to(units('m/s'))
    v1 = v.to(units('m/s'))
    h.plot_colormapped(u1[mask], v1[mask], z[mask], intervals=intervals, colors=colors, label="colors")
    legend_elements = [
        plt.Line2D([0], [0], color=color, lw=4, label=f'{intervals[i].magnitude} - {intervals[i + 1].magnitude} km')
        for i, color in enumerate(colors)]

    # Add the legend to the plot
    ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.6, 0.5), title='Altitude intervals')

    decimate = 50

    for p0, t0, h0 in zip(p[::decimate], T[::decimate],
                          z[::decimate]):
        if p0 >= 100 * units("hPa"):
            # skew.ax.text(1.08, p.magnitude, round(h0, 0), transform=skew.ax.get_yaxis_transform(which="tick2")) should work somehow to put the height text on the outerside
            skew.ax.text(t0, p0, round(h0, 0))
    # plt.show()
    plt.title(f"Observations")
    plt.savefig(f"{dir_PLOTS}/radiosonde_LOWI/radiosonde_ibk_observation.png")


def plot_radiosonde_ibk_model(time, method, lon, lat):
    """plot the MODEL output of AROME as it would be a Radiosonde"""
    my_variable_list = ["p", "q", "th", "u", "v", "z"]

    if (method == "sel") | (method == "interp"):
        print(f"Your selected method is {method}")
    else:
        raise AttributeError(
            "You have to define a method (sel or interp) how the point near the LOWI should be selected")

    df_final = read_3D_variables_AROME(variables=my_variable_list, method=method, lon=lon, lat=lat, time=time)

    # print(df_final["p"].metpy.unit_array.magnitude) Extract values

    df_final["windspeed"] = metpy.calc.wind_speed(df_final["u"], df_final["v"])
    df_final["wind direction"] = metpy.calc.wind_direction(df_final["u"], df_final["v"], convention='from')
    df_final["temperature"] = metpy.calc.temperature_from_potential_temperature(df_final["p"],
                                                                                df_final["th"])  # virtual temperature

    # df_final["temperature"] = calc_t_from_tv(tv=df_final["t"], q=df_final["q"])  # was added to calc real temp

    df_final["dewpoint"] = metpy.calc.dewpoint_from_specific_humidity(pressure=df_final["p"],
                                                                      temperature=df_final["temperature"],
                                                                      specific_humidity=df_final["q"])

    Td = df_final["dewpoint"].metpy.unit_array
    p = df_final["p"].metpy.unit_array.to(units.hPa)  # Metadata is removed
    T = df_final["temperature"].metpy.unit_array.to(units.degC)
    wind_speed = df_final["windspeed"].metpy.unit_array.to(units.knots)
    wind_dir = df_final['wind direction']
    u, v = mpcalc.wind_components(wind_speed, wind_dir)

    ax = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(3, 3)

    skew = SkewT(ax, rotation=45, rect=(0.1, 0.1, 0.55, 0.85))

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    skew.plot(p, T, 'r')
    skew.plot(p, Td, 'g')

    # Mask barbs to below 100 hPa only
    mask = p >= 100 * units.hPa

    # Plot wind barbs
    skew.plot_barbs(p[mask][::3], u[mask][::3], v[mask][::3], y_clip_radius=0.01)

    # Change to adjust data limits and give it a semblance of what we want
    skew.ax.set_adjustable('datalim')
    skew.ax.set_ylim(1000, 100)
    skew.ax.set_xlim(-20, 30)

    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()

    z = df_final['z'].metpy.unit_array.to(units.m).to(units.km)
    # Create a hodograph
    # Mask velocities to below 10 km only
    mask = z <= 10 * units.km

    # Custom colorscale for the wind profile
    intervals = np.array([0, 1, 3, 5, 10]) * units.km
    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:olive']
    # Convert the rounded values into strings with the desired format
    str_lat = f"{df_final['latitude'].values:.3f}_N"
    str_lon = f"{df_final['longitude'].values:.3f}_E"

    plt.title(f'{str_lat} {str_lon}', loc='left')
    plt.title(f'Valid Time: {pd.to_datetime(df_final.time.values).strftime("%Y-%m-%d %H:%M")}',
              loc='right')
    skew.ax.set_xlabel('Temperature (°C)')
    skew.ax.set_ylabel('Pressure (hPa)')

    # Create a hodograph
    ax1 = ax.add_subplot(gs[0, -1])
    h = Hodograph(ax1, component_range=40.)
    h.add_grid(increment=5)
    u1 = u.to(units('m/s'))
    v1 = v.to(units('m/s'))
    h.plot_colormapped(u1[mask], v1[mask], z[mask], intervals=intervals, colors=colors, label="colors")

    decimate = -10
    h = df_final['z'].metpy.unit_array

    for p0, t0, h0 in zip(p[::decimate], T[::decimate],
                          h[::decimate]):
        if p0 >= 100 * units("hPa"):
            height_without_units = h0.magnitude  # or h0.m for shorthand
            rounded_height = math.ceil(height_without_units)
            skew.ax.text(t0.magnitude - 0., p0, f"{rounded_height} m")

    plt.title(f"AROME Model")
    # Create custom legend entries
    legend_elements = [
        plt.Line2D([0], [0], color=color, lw=4, label=f'{intervals[i].magnitude} - {intervals[i + 1].magnitude} km')
        for i, color in enumerate(colors)]

    # Add the legend to the plot
    ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.6, 0.5), title='Altitude intervals')

    plt.savefig(f"{dir_PLOTS}radiosonde_LOWI/radio_AROME_{method}_{str_lat.replace('.', '_')}_"
                f"{str_lon.replace('.', '_')}_{str(time_for_model).replace(':', '_').replace(' ', '_')}.png")


def plot_radiosondes_of_adjacent_points():
    """Plot the 6 nearest points to exclude height error"""
    lons = np.arange(11.345, 11.366, 0.01)
    lats = [47.255, 47.265]
    for my_lon in lons:
        for my_lat in lats:
            plot_radiosonde_ibk_model(time=time_for_model, method="sel", lon=my_lon, lat=my_lat)


if __name__ == '__main__':
    plot_radiosonde_ibk_obs()
    # plot_radiosondes_of_adjacent_points() # remember to comment out plt.savefig!
    # plot_radiosonde_ibk_model(time=time_for_model, method="interp", lat=station_files_zamg["LOWI"]["lat"],
    #                          lon=station_files_zamg["LOWI"]["lon"])
    plot_radiosonde_ibk_model(time=time_for_model, method="sel", lat=station_files_zamg["LOWI"]["lat"],
                              lon=station_files_zamg["LOWI"]["lon"])

    plt.show()
