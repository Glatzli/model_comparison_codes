"""
This file should include functions for calculating the PBL height using the methods described in
Wagner et al. 2015: The impact of valley geometry on daytime thermally driven flows and vertical transport processes
=> method for the CBL, not for SBL!

eif run,

Idea: I would like to search for lat/lons
"""
import confg
import read_in_arome
import read_icon_model_3D
import read_ukmo
import read_wrf_helen

import os
from calc_vhd import open_save_timeseries_main
import xarray as xr
import numpy as np
import matplotlib
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from colorspace import terrain_hcl, qualitative_hcl, sequential_hcl
import pandas as pd


def calc_pbl_heights(ds, model="AROME"):
    """
    calculates PBL height according to Wagner et al. (2015): method to find the PBL height for a CBL and not for a SBL!
    when pot temp. differential is getting larger than the given threshold then the lower PBL height is reached (PBL1) etc
    :param ds:
    :param model:
    :return:
    a numpy array with the 3 PBL heights in it
    """
    dth = ds.th.differentiate(coord="height")  # dθ/d(height index)

    if model in ["ICON", "WRF"]:  # take unstaggered height vals for ICON & WRF models
        # where pot temp gradient first exceeds threshold (w NaN below): take min along height to get first value from surface up
        pbl_height1 = ds.where(dth > 0.001).z_unstag.min(dim="height")
        pbl_height2 = ds.where(dth < 0.001).z_unstag.max(dim="height")
        # take max to get first value from top where it gets below 0.001 K/m
    else:
        pbl_height1 = ds.where(dth > 0.001).z.min(dim="height")
        pbl_height2 = ds.where(dth < 0.001).z.max(dim="height")  # that is wrong! pbl2 should be higher than pbl1...

        # pbl_height3 = ds.where(ds.height > pbl_height2 and dth.max())  # for pbl 3 only take values above pbl2...

    return pbl_height1, pbl_height2


def plot_vert_profiles(timestamp, point=confg.ibk_uni, zoomed=True):
    """
    plot vertical profiles with matplotlib and save it as svg (old style)
    :param timestamp:
    :param point:
    :param zoomed:
    :return:
    """
    # plot the pot. temp profiles with heights to enable a plausibility check:
    fig, axs = plt.subplots(figsize=(10, 8))

    arome.sel(time=timestamp).th.plot(y="height", ax=axs, color=qualitative_colors[0], label="AROME")
    icon.sel(time=timestamp).th.plot(y="height", ax=axs, color=qualitative_colors[3], label="ICON")
    icon2te.sel(time=timestamp).th.plot(y="height", ax=axs, color=qualitative_colors[4], label="ICON2TE")
    um.sel(time=timestamp).th.plot(y="height", ax=axs, color=qualitative_colors[2], label="UM")
    wrf.sel(time=timestamp).th.plot(y="height", ax=axs, color=qualitative_colors[6], label="WRF")
    if "ibk" in point["name"]:
        # radiosonde has also time dimension due to conversion into dataset...
        radio.th.plot(y="height", ax=axs, color="grey", linestyle="--", label="Radiosonde at 02:15")
        hatpro.sel(time=timestamp).th.plot(y="height", ax=axs, color=qualitative_colors[8], label="HATPRO")

    """
    plt.axhline(y=um_pbl_height1.isel(time=timestamp_idx).data, linestyle="--", color=qualitative_colors[4])
    axs.text(x=300, y=um_pbl_height1.isel(time=timestamp_idx).data, s="UM PBL height1")
    plt.axhline(y=um_pbl_height2.isel(time=timestamp_idx).data, linestyle="--", color=qualitative_colors[4])
    axs.text(x=350, y=um_pbl_height2.isel(time=timestamp_idx).data, s="UM PBL height2")

    plt.axhline(y=wrf_pbl_height1.isel(time=timestamp_idx).data, linestyle="--", linewidth=0.5, color=qualitative_colors[6])
    axs.text(x=300, y=wrf_pbl_height1.isel(time=timestamp_idx).data, s="WRF PBL height1")
    plt.axhline(y=wrf_pbl_height2.isel(time=timestamp_idx).data, linestyle="--", linewidth=0.5, color=qualitative_colors[6])
    axs.text(x=350, y=wrf_pbl_height2.isel(time=timestamp_idx).data, s="WRF PBL height2")
    """
    plt.legend()
    plt.grid()
    plt.ylabel("geopotential height [m]")
    plt.xlabel("potential temperature [K]")
    plt.title(f"{point['name']} ({point['height']} m) at {timestamp.strftime('%d %b, %H:%M')}")
    if "ibk" in point["name"]:
        plt.ylim([500, 3500])
        plt.xlim([280, 320])
    else:
        plt.ylim([450, 3500])
        plt.xlim([288, 318])
    if zoomed:
        plt.ylim([450, 2500])
        plt.xlim([288, 305])
    plt.savefig(confg.dir_PLOTS + "/vertical_plots/" + f"pot_temp_vertical_{point['name']}_{timestamp.strftime('%d_%H%M')}" + "_zoomed.svg" if zoomed else ".svg")
    plt.show()
    print(4)


def plot_vert_profiles_plotly(timestamp, point=confg.ibk_uni):
    """
    plot vertical pot temp distribution as interactive plot and save it as html file
    :param timestamp:
    :param point:
    :return:
    """
    fig = go.Figure()

    # Add model lines
    fig.add_trace(go.Scatter(
        x=arome.sel(time=timestamp).th.values,
        y=arome.sel(time=timestamp).height.values,
        mode='lines',
        name='AROME',
        line=dict(color=qualitative_colors[0])
    ))

    fig.add_trace(go.Scatter(
        x=icon.sel(time=timestamp).th.values,
        y=icon.sel(time=timestamp).height.values,
        mode='lines',
        name='ICON',
        line=dict(color=qualitative_colors[3])
    ))

    fig.add_trace(go.Scatter(
        x=icon2te.sel(time=timestamp).th.values,
        y=icon2te.sel(time=timestamp).height.values,
        mode='lines',
        name='ICON2TE',
        line=dict(color=qualitative_colors[4])
    ))

    fig.add_trace(go.Scatter(
        x=um.sel(time=timestamp).th.values,
        y=um.sel(time=timestamp).height.values,
        mode='lines',
        name='UM',
        line=dict(color=qualitative_colors[2])
    ))

    fig.add_trace(go.Scatter(
        x=wrf.sel(time=timestamp).th.values,
        y=wrf.sel(time=timestamp).height.values,
        mode='lines',
        name='WRF',
        line=dict(color=qualitative_colors[6])
    ))

    # Add radiosonde and hatpro if applicable
    if "ibk" in point["name"]:
        fig.add_trace(go.Scatter(
            x=radio.th.values,
            y=radio.height.values,
            mode='lines',
            name='Radiosonde at 02:15',
            line=dict(color='grey', dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=hatpro.sel(time=timestamp).th.values,
            y=hatpro.sel(time=timestamp).height.values,
            mode='lines',
            name='HATPRO',
            line=dict(color=qualitative_colors[8])
        ))

    # Layout settings
    fig.update_layout(
        title=f"{point['name']} ({point['height']} m) at {timestamp.strftime('%d %b, %H:%M')}",
        xaxis_title='Potential Temperature [K]',
        yaxis_title='Geopotential Height [m]',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        height=700,
        width=900
    )
    fig.update_xaxes(range=[288, 318])  # add default limits
    fig.update_yaxes(range=[450, 5000])
    fig.write_html(confg.dir_PLOTS + "/vertical_plots/" + f"pot_temp_vertical_{point['name']}_{timestamp.strftime('%d_%H%M')}.html")

    # Reverse y-axis if desired (to show height increasing upwards)
    # fig.update_yaxes(autorange="reversed")

    fig.show()


def plot_vert_profiles_plotly_multi(point=confg.ibk_uni):
    """
    Plot vertical potential temperature profiles for multiple timesteps and models using Plotly.
    Each timestamp is grouped in the legend so all traces at that time can be toggled together.
    """
    import datetime
    import os
    import pandas as pd
    import plotly.graph_objects as go
    from colorspace import sequential_hcl  # Assuming you use this package for color generation

    # Generate 3-hourly timestamps from 19:00 (15 Oct) to 08:00 (16 Oct) 2017
    timestamps = pd.date_range(start=datetime.datetime(2017, 10, 15, 19),
                               end=datetime.datetime(2017, 10, 16, 8), freq='3h')

    fig = go.Figure()
    n = len(timestamps) + 4  # Extra colors to avoid white

    # Generate color palettes
    model_colors = {
        'AROME': sequential_hcl(palette="Reds 2").colors(n),
        'ICON': sequential_hcl(palette="Blues 2").colors(n),
        'ICON2TE': sequential_hcl(palette="Blues 3").colors(n),
        'UM': sequential_hcl(palette="Greens 2").colors(n),
        'WRF': sequential_hcl(palette="Purples 2").colors(n),
        'HATPRO': sequential_hcl(palette="Light Grays").colors(n)
    }

    for i, timestamp in enumerate(timestamps):
        timestamp_str = timestamp.strftime("%d %H:%M")
        legendgroup = timestamp_str
        show_time_legend = True  # only once per timestamp

        # Format legend name (e.g., "16th 04:00")
        day = timestamp.day
        suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        legend_name = f"{day}{suffix} {timestamp.strftime('%H:%M')}"

        def add_model_trace(data, model_name, showlegend=False):
            fig.add_trace(go.Scatter(
                x=data.sel(time=timestamp).th.values,
                y=data.sel(time=timestamp).height.values,
                mode='lines',
                name=legend_name if showlegend else f'{model_name}',
                legendgroup=legendgroup,
                showlegend=showlegend,
                line=dict(color=model_colors[model_name][i])
            ))

        # Add each model trace
        add_model_trace(arome, 'AROME', showlegend=True)  # Only this one shows the time label
        add_model_trace(icon, 'ICON')
        add_model_trace(icon2te, 'ICON2TE')
        add_model_trace(um, 'UM')
        add_model_trace(wrf, 'WRF')

        if "ibk" in point["name"]:
            add_model_trace(hatpro, 'HATPRO')

    # Add separate legend entries for models using dummy traces (color key)
    for model, color_list in model_colors.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            name=model,
            line=dict(color=color_list[0], width=4),
            legendgroup='models',
            showlegend=True
        ))

    # Layout
    fig.update_layout(
        title=f"{point['name']} ({point['height']} m) — Potential Temperature Profiles",
        xaxis_title='Potential Temperature [K]',
        yaxis_title='Geopotential Height [m]',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=1.0,
            groupclick="toggleitem",
            title_text="Click timestamp double <br>to show all models<br>at that time:<br>(model legend below)"
        ),
        template='plotly_white',
        height=900,
        width=1200
    )
    fig.update_xaxes(range=[288, 318])
    fig.update_yaxes(range=[450, 5000])

    # Save and show
    output_path = os.path.join(confg.dir_PLOTS, "vertical_plots", f"pot_temp_vertical_{point['name']}_multi.html")
    fig.write_html(output_path)
    fig.show()


if __name__ == '__main__':
    qualitative_colors = qualitative_hcl(palette="Dark 3").colors()
    timestamp = datetime.datetime(2017, 10, 15, 18, 0, 0)
    point = confg.kiefersfelden
    # using PCGP-method:
    (arome, icon, icon2te, um, wrf,
     hatpro, radio) = open_save_timeseries_main(lat=point["lat"], lon=point["lon"], point_name=point["name"],
                                                variables=["p", "th", "temp", "z", "z_unstag"], height_as_z_coord=True)
    # plot_vert_profiles(timestamp=timestamp, point=point, zoomed=True)
    plot_vert_profiles_plotly_multi(point=point)

    """
    # variant for any point w/o using PCGP method:
    lat_point, lon_point = 47.31, 11.6
    vars = ["th", "z", "z_unstag"]
    # this would be the version for selecting a point and then calculating the PBL height
    # but as for VHD calculation: It would be faster to compute the height for the full domain first and then select the
    # point where the timeseries is wanted...

    arome = read_in_arome.read_in_arome_fixed_point(lat=lat_point, lon=lon_point, variables=vars[:-1],
                                                               height_as_z_coord=True)
    icon = read_icon_model_3D.read_icon_fixed_point(lat=lat_point, lon=lon_point, variant="ICON",
                                                    variables=vars, height_as_z_coord=True)
    icon2te = read_icon_model_3D.read_icon_fixed_point(lat=lat_point, lon=lon_point, variant="ICON2TE",
                                                       variables=vars, height_as_z_coord=True)
    um = read_ukmo.read_ukmo_fixed_point(lat=lat_point, lon=lon_point, variables=vars[:-1], height_as_z_coord=True)
    # don't read give the funct z_unstag as variable to read, cause only ICON & WRF are staggered!
    wrf = read_wrf_helen.read_wrf_fixed_point(lat=lat_point, lon=lon_point, variables=vars, height_as_z_coord=True)

    # calculate PBL heights

    arome_pbl_height1, arome_pbl_height2 = calc_pbl_heights(arome, model="AROME")
    icon_pbl_height1, icon_pbl_height2 = calc_pbl_heights(icon, model="ICON")
    icon2te_pbl_height1, icon2te_pbl_height2 = calc_pbl_heights(icon2te, model="ICON")

    um_pbl_height1, um_pbl_height2 = calc_pbl_heights(um, model="UM")
    wrf_pbl_height1, wrf_pbl_height2 = calc_pbl_heights(wrf, model="WRF")

    plot_vert_profiles(timestamp_idx=34)

    um_pbl_heights = xr.Dataset(data_vars={"pbl_height1": um_pbl_height1,
                                           "pbl_height2": um_pbl_height2},
                                attrs={"modelinfo": "UM boundary layer height computed as defined in DOI:10.1002/qj.2481"})
    um_pbl_heights.to_netcdf("UM_pbl_heights.nc")
    wrf
    """
