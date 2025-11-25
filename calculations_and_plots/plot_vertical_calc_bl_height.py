"""
Deprecated! Not used anymore!

This file should include functions for calculating the PBL height using the methods described in
Wagner et al. 2015: The impact of valley geometry on daytime thermally driven flows and vertical transport processes
=> method for the CBL, not for SBL!

eif run,

Idea: I would like to search for lat/lons
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # neu: für 2‑stündige Zeit-Auswahl
import plotly.graph_objects as go
import xarray as xr

import confg


def calc_pbl_heights(ds, model="AROME"):
    """
    calculates PBL height according to Wagner et al. (2015): method to find the PBL height for a CBL and not for a SBL!
    when pot temp. differential is getting larger than the given threshold then the lower PBL height is reached (
    PBL1) etc
    :param ds:
    :param model:
    :return:
    a numpy array with the 3 PBL heights in it
    """
    dth = ds.th.differentiate(coord="height")  # dθ/d(height index)
    
    if model in ["ICON", "WRF"]:  # take unstaggered height vals for ICON & WRF models
        # where pot temp gradient first exceeds threshold (w NaN below): take min along height to get first value
        # from surface up
        pbl_height1 = ds.where(dth > 0.001).z_unstag.min(dim="height")
        pbl_height2 = ds.where(dth < 0.001).z_unstag.max(
            dim="height")  # take max to get first value from top where it gets below 0.001 K/m
    else:
        pbl_height1 = ds.where(dth > 0.001).z.min(dim="height")
        pbl_height2 = ds.where(dth < 0.001).z.max(dim="height")  # that is wrong! pbl2 should be higher than pbl1...
        
        # pbl_height3 = ds.where(ds.height > pbl_height2 and dth.max())  # for pbl 3 only take values above pbl2...
    
    return pbl_height1, pbl_height2


def _add_cap_marker(fig, ds, timestamp, label, color, x_var='temp'):
    """ by ChatGPT
    helper to add 'x' cap markers to profile plots

    Now supports selecting which variable to use for the x-coordinate (x_var),
    default is 'temp' for temperature plots. Use x_var='dT_dz' for dT/dz plots.
    """
    cap_name = 'cap_depth' if 'cap_depth' in ds else ('cap_height' if 'cap_height' in ds else None)
    if cap_name is None:
        return
    cap_da = ds[cap_name]
    # get cap height for this timestamp (or scalar if no time dimension)
    try:
        cap_y = cap_da.sel(time=timestamp).item() if 'time' in cap_da.dims else float(cap_da.values)
    except Exception:
        return
    if cap_y is None or np.isnan(cap_y):
        return
    
    # get x value (variable given by x_var) at nearest height to cap_y
    x_val = None
    try:
        if x_var in ds:
            x_val = ds.sel(time=timestamp)[x_var].sel(height=cap_y, method='nearest').item()
        else:
            # fallback: try common alternatives
            for alt in ('temp', 'dT_dz'):
                if alt in ds:
                    x_val = ds.sel(time=timestamp)[alt].sel(height=cap_y, method='nearest').item()
                    break
    except Exception:
        try:
            h = ds.sel(time=timestamp).height.values
            idx = int(np.nanargmin(np.abs(h - cap_y)))
            if x_var in ds:
                x_val = ds.sel(time=timestamp)[x_var].isel(height=idx).item()
            else:
                for alt in ('temp', 'dT_dz'):
                    if alt in ds:
                        x_val = ds.sel(time=timestamp)[alt].isel(height=idx).item()
                        break
            cap_y = float(h[idx])
        except Exception:
            return
    
    if x_val is None or (isinstance(x_val, float) and np.isnan(x_val)):
        return
    
    fig.add_trace(go.Scatter(x=[x_val], y=[cap_y], mode='markers',
                             marker=dict(symbol='x', size=10, color=color, line=dict(width=1.5, color=color)),
                             name=f"{label} cap", showlegend=False))


# neu: Helper, um nur volle 2‑Stunden‑Zeitpunkte (hh:00, hh%2==0) aus einem Dataset zu wählen
def _select_timestamps_every_2h(ds: xr.Dataset, start=None, end=None):
    times = pd.DatetimeIndex(ds.time.values)
    mask = (times.minute == 0) & (times.hour % 2 == 0)
    if start is not None:
        mask &= times >= pd.Timestamp(start)
    if end is not None:
        mask &= times <= pd.Timestamp(end)
    return times[mask]


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
    
    arome.sel(time=timestamp).temp.plot(y="height", ax=axs, color=qualitative_colors[0], label="AROME")
    icon.sel(time=timestamp).temp.plot(y="height", ax=axs, color=qualitative_colors[3], label="ICON")
    icon2te.sel(time=timestamp).temp.plot(y="height", ax=axs, color=qualitative_colors[4], label="ICON2TE")
    um.sel(time=timestamp).temp.plot(y="height", ax=axs, color=qualitative_colors[2], label="UM")
    wrf.sel(time=timestamp).temp.plot(y="height", ax=axs, color=qualitative_colors[6], label="WRF")
    if "ibk" in point["name"]:
        # radiosonde has also time dimension due to conversion into dataset...
        radio.temp.plot(y="height", ax=axs, color="grey", linestyle="--", label="Radiosonde at 02:15")
        hatpro.sel(time=timestamp).temp.plot(y="height", ax=axs, color=qualitative_colors[8], label="HATPRO")
    
    """
    plt.axhline(y=um_pbl_height1.isel(time=timestamp_idx).data, linestyle="--", color=qualitative_colors[4])
    axs.text(x=300, y=um_pbl_height1.isel(time=timestamp_idx).data, s="UM PBL height1")
    plt.axhline(y=um_pbl_height2.isel(time=timestamp_idx).data, linestyle="--", color=qualitative_colors[4])
    axs.text(x=350, y=um_pbl_height2.isel(time=timestamp_idx).data, s="UM PBL height2")

    plt.axhline(y=wrf_pbl_height1.isel(time=timestamp_idx).data, linestyle="--", linewidth=0.5,
    color=qualitative_colors[6])
    axs.text(x=300, y=wrf_pbl_height1.isel(time=timestamp_idx).data, s="WRF PBL height1")
    plt.axhline(y=wrf_pbl_height2.isel(time=timestamp_idx).data, linestyle="--", linewidth=0.5,
    color=qualitative_colors[6])
    axs.text(x=350, y=wrf_pbl_height2.isel(time=timestamp_idx).data, s="WRF PBL height2")
    """
    plt.legend()
    plt.grid()
    plt.ylabel("geopotential height [m]")
    plt.xlabel("temperature [°C]")
    plt.title(f"{point['name']} ({point['height']} m) at {timestamp.strftime('%d %b, %H:%M')}")
    # if "ibk" in point["name"]:
    #     plt.ylim([500, 3500])
    #     plt.xlim([280, 320])
    # else:
    #     plt.ylim([450, 3500])
    #     plt.xlim([288, 318])
    # if zoomed:
    #     plt.ylim([450, 2500])
    #     plt.xlim([288, 305])
    plt.savefig(
        confg.dir_PLOTS + "/vertical_plots/" + f"temp_vertical_{point['name']}_{timestamp.strftime('%d_%H%M')}" +
        "_zoomed.svg" if zoomed else ".svg")
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
    fig.add_trace(
        go.Scatter(x=arome.sel(time=timestamp).temp.values, y=arome.sel(time=timestamp).height.values, mode='lines',
                   name='AROME', line=dict(color=qualitative_colors[0])))
    
    fig.add_trace(
        go.Scatter(x=icon.sel(time=timestamp).temp.values, y=icon.sel(time=timestamp).height.values, mode='lines',
                   name='ICON', line=dict(color=qualitative_colors[3])))
    
    fig.add_trace(
        go.Scatter(x=icon2te.sel(time=timestamp).temp.values, y=icon2te.sel(time=timestamp).height.values, mode='lines',
                   name='ICON2TE', line=dict(color=qualitative_colors[4])))
    
    fig.add_trace(go.Scatter(x=um.sel(time=timestamp).temp.values, y=um.sel(time=timestamp).height.values, mode='lines',
                             name='UM', line=dict(color=qualitative_colors[2])))
    
    fig.add_trace(
        go.Scatter(x=wrf.sel(time=timestamp).temp.values, y=wrf.sel(time=timestamp).height.values, mode='lines',
                   name='WRF', line=dict(color=qualitative_colors[6])))
    
    # Add radiosonde and hatpro if applicable
    if "ibk" in point["name"]:
        fig.add_trace(go.Scatter(x=radio.temp.values, y=radio.height.values, mode='lines', name='Radiosonde at 02:15',
                                 line=dict(color='grey', dash='dash')))
        
        fig.add_trace(go.Scatter(x=hatpro.sel(time=timestamp).temp.values, y=hatpro.sel(time=timestamp).height.values,
                                 mode='lines', name='HATPRO', line=dict(color=qualitative_colors[8])))
        # add cap markers for observations
        _add_cap_marker(fig, radio, timestamp, 'Radiosonde', 'grey')
        _add_cap_marker(fig, hatpro, timestamp, 'HATPRO', qualitative_colors[8])
    
    # add 'x' cap markers for each model
    _add_cap_marker(fig, arome, timestamp, 'AROME', qualitative_colors[0])
    _add_cap_marker(fig, icon, timestamp, 'ICON', qualitative_colors[3])
    _add_cap_marker(fig, icon2te, timestamp, 'ICON2TE', qualitative_colors[4])
    _add_cap_marker(fig, um, timestamp, 'UM', qualitative_colors[2])
    _add_cap_marker(fig, wrf, timestamp, 'WRF', qualitative_colors[6])
    
    # Layout settings
    fig.update_layout(title=f"{point['name']} ({point['height']} m) at {timestamp.strftime('%d %b, %H:%M')}",
                      xaxis_title='temperature [°C]', yaxis_title='geopotential Height [m]',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      template='plotly_white', height=700, width=900)
    fig.update_xaxes(range=[-5, 20])  # add default limits
    fig.update_yaxes(range=[450, 3000])
    fig.write_html(
        confg.dir_PLOTS + "/vertical_plots/" + f"{point['name']}_{timestamp.strftime('%d_%H%M')}_temp_vertical.html")
    
    # Reverse y-axis if desired (to show height increasing upwards)
    # fig.update_yaxes(autorange="reversed")
    
    fig.show()


def assign_height_of_temp_max(ds):
    """
    Berechnet für jeden Zeitschritt die Höhe des Temperaturmaximums und fügt sie als Variable hinzu.
    :param ds: xarray.Dataset mit Dimensionen 'time' und 'height' sowie Variable 'temp'
    :return: xarray.Dataset mit neuer Variable 'height_temp_max' (nur von 'time' abhängig)
    """
    # Index des Maximums entlang der Höhe für jeden Zeitschritt
    idx_max = ds['temp'].argmax(dim='height')
    # Höhe an diesem Index extrahieren
    height_temp_max = ds['height'].isel(height=idx_max)
    # Als neue Variable hinzufügen (nur von 'time' abhängig)
    ds = ds.assign(height_temp_max=('time', height_temp_max.values))
    return ds


def plot_vert_profiles_plotly_multi(point=confg.ibk_uni):
    """
    Plot vertical potential temperature profiles for multiple timesteps (alle 2 Stunden) und Modelle mit Plotly.
    Jeder Zeitstempel ist in der Legende gruppiert, so dass alle Spuren zu diesem Zeitpunkt gemeinsam umschaltbar sind.
    """
    import os
    import plotly.graph_objects as go
    from colorspace import sequential_hcl  # Assuming you use this package for color generation
    
    # neu: 2‑stündige Zeitstempel aus den tatsächlich verfügbaren Zeiten von 'arome' wählen
    timestamps = _select_timestamps_every_2h(arome)
    
    fig = go.Figure()
    n = len(timestamps) + 4  # Extra Farben, um Weiß zu vermeiden
    
    # Generate color palettes
    model_colors = {'AROME': sequential_hcl(palette="Reds 2").colors(n),
                    'ICON': sequential_hcl(palette="Blues 2").colors(n),
                    'ICON2TE': sequential_hcl(palette="Blues 3").colors(n),
                    'UM': sequential_hcl(palette="Greens 2").colors(n),
                    'WRF': sequential_hcl(palette="Purples 2").colors(n),
                    'HATPRO': sequential_hcl(palette="Light Grays").colors(n)}
    
    for i, timestamp in enumerate(timestamps):
        timestamp_str = timestamp.strftime("%d %H:%M")
        legendgroup = timestamp_str
        show_time_legend = True
        
        # Format legend name (e.g., "16th 04:00")
        day = timestamp.day
        suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        legend_name = f"{day}{suffix} {timestamp.strftime('%H:%M')}"
        
        def add_model_trace(data, model_name, showlegend=False):
            fig.add_trace(
                go.Scatter(x=data.sel(time=timestamp).th.values, y=data.sel(time=timestamp).height.values, mode='lines',
                           name=legend_name if showlegend else f'{model_name}', legendgroup=legendgroup,
                           showlegend=showlegend, line=dict(color=model_colors[model_name][i])))
        
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
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name=model, line=dict(color=color_list[0], width=4),
                                 legendgroup='models', showlegend=True))
    
    # Layout
    fig.update_layout(title=f"{point['name']} ({point['height']} m) — Potential Temperature Profiles",
                      xaxis_title='Potential Temperature [K]', yaxis_title='Geopotential Height [m]',
                      legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="right", x=1.0,
                                  groupclick="toggleitem",
                                  title_text="Click timestamp double <br>to show all models<br>at that time:<br>("
                                             "model legend below)"), template='plotly_white', height=900, width=1200)
    fig.update_xaxes(range=[288, 318])
    fig.update_yaxes(range=[450, 5000])
    
    # Save and show
    output_path = os.path.join(confg.dir_PLOTS, "vertical_plots", f"pot_temp_vertical_{point['name']}_multi.html")
    fig.write_html(output_path)
    fig.show()


def calc_dT_dz(arome, icon, icon2te, um, wrf, radio, hatpro):
    """
    Calculate vertical temperature gradient (dT/dz) for each model and add as a new variable
    """
    arome = arome.assign(dT_dz=arome['temp'].differentiate(coord='height'))
    icon = icon.assign(dT_dz=icon['temp'].differentiate(coord='height'))
    icon2te = icon2te.assign(dT_dz=icon2te['temp'].differentiate(coord='height'))
    um = um.assign(dT_dz=um['temp'].differentiate(coord='height'))
    wrf = wrf.assign(dT_dz=wrf['temp'].differentiate(coord='height'))
    radio = radio.assign(dT_dz=radio['temp'].differentiate(coord='height'))
    hatpro = hatpro.assign(dT_dz=hatpro['temp'].differentiate(coord='height'))
    # Return all updated datasets
    return arome, icon, icon2te, um, wrf, radio, hatpro


def calc_cap_height(ds: xr.Dataset) -> xr.DataArray:
    """
    Find, for each time, the first (bottom-up) height where dT_dz is negative
    for 3 consecutive levels. Returns a 1D DataArray over 'time' with the height.
    If no such triplet exists for a time, returns NaN for that time.
    """
    # Ensure height is ascending so "first" means lowest (bottom-up)
    ds_sorted = ds.sortby("height", ascending=True)
    
    # Boolean mask where dT_dz is negative; treat NaNs as False so they break sequences
    neg = (ds_sorted["dT_dz"] < 0).fillna(False)
    
    # Rolling 3-level window: True where all 3 consecutive levels are negative
    # min_periods=3 ensures only full windows count
    neg3 = neg.rolling(height=3, min_periods=3).sum() == 3
    
    # Broadcast height over time and mask where condition holds
    heights_masked = ds_sorted["height"].where(neg3)
    
    # For each time: find the minimum (first from bottom) height that satisfies condition
    cap_height = heights_masked.min(dim="height")
    return cap_height


# EDITED: New function for small multiples vertical temperature profiles
def plot_vert_profiles_small_multiples(point_names: list, timestamp: str = "2017-10-16T04:00:00",
                                       max_height: float = 3000) -> go.Figure:
    """
    Create small multiples plot of vertical temperature profiles for all points at a given timestamp.
    Each subplot shows all models for one point, with CAP height markers.
    point_names: list of point names from confg.py
    timestamp: ISO format timestamp string (e.g. "2017-10-16T04:00:00")
    max_height: maximum height in meters to plot (default 3000m)
    """
    from plotly.subplots import make_subplots
    import read_in_arome
    import read_icon_model_3D
    import read_ukmo
    import read_wrf_helen
    import os
    
    # Convert timestamp string to numpy datetime64
    ts = np.datetime64(timestamp)
    
    # Calculate grid layout with 2 columns
    n_points = len(point_names)
    n_cols = 2
    n_rows = int(np.ceil(n_points / n_cols))
    
    # Create subplot titles
    subplot_titles = []
    for point_name in point_names:
        point = getattr(confg, point_name, None)
        if point:
            subplot_titles.append(point["name"])
        else:
            subplot_titles.append(point_name)
    
    # Create subplots
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, vertical_spacing=0.06,
        horizontal_spacing=0.08)
    
    # Model order and color mapping (same as plot_vert_profiles_plotly)
    model_order = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]
    model_colors = {"AROME": qualitative_colors[0], "ICON": qualitative_colors[3], "ICON2TE": qualitative_colors[4],
        "UM": qualitative_colors[2], "WRF": qualitative_colors[6]}
    
    # Variables to read for each model
    variables = ["p", "th", "temp", "z", "z_unstag"]
    
    # Read data and plot for each point
    for idx, point_name in enumerate(point_names):
        point = getattr(confg, point_name, None)
        if point is None:
            continue
        
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        for model in model_order:
            try:
                # Try to load saved timeseries first, otherwise read fresh data
                ds = None
                
                if model == "AROME":
                    # Check for saved timeseries file
                    saved_path = os.path.join(confg.dir_AROME, "timeseries",
                                              f"arome_{point_name}_timeseries_height_as_z.nc")
                    if os.path.exists(saved_path):
                        ds = xr.open_dataset(saved_path)
                    else:
                        ds = read_in_arome.read_in_arome_fixed_point(lat=point["lat"], lon=point["lon"],
                            variables=variables, height_as_z_coord=True)
                
                elif model == "ICON":
                    saved_path = os.path.join(confg.icon_folder_3D, "timeseries",
                                              f"icon_{point_name}_timeseries_height_as_z.nc")
                    if os.path.exists(saved_path):
                        ds = xr.open_dataset(saved_path)
                    else:
                        ds = read_icon_model_3D.read_icon_fixed_point(lat=point["lat"], lon=point["lon"],
                            variant="ICON", variables=variables, height_as_z_coord=True)
                
                elif model == "ICON2TE":
                    saved_path = os.path.join(confg.icon2TE_folder_3D, "timeseries",
                                              f"icon_2te_{point_name}_timeseries_height_as_z.nc")
                    if os.path.exists(saved_path):
                        ds = xr.open_dataset(saved_path)
                    else:
                        ds = read_icon_model_3D.read_icon_fixed_point(lat=point["lat"], lon=point["lon"],
                            variant="ICON2TE", variables=variables, height_as_z_coord=True)
                
                elif model == "UM":
                    saved_path = os.path.join(confg.ukmo_folder, "timeseries",
                                              f"ukmo_{point_name}_timeseries_height_as_z.nc")
                    if os.path.exists(saved_path):
                        ds = xr.open_dataset(saved_path)
                    else:
                        ds = read_ukmo.read_ukmo_fixed_point(lat=point["lat"], lon=point["lon"], variables=variables,
                            height_as_z_coord=True)
                
                elif model == "WRF":
                    saved_path = os.path.join(confg.wrf_folder, "timeseries",
                                              f"wrf_{point_name}_timeseries_height_as_z.nc")
                    if os.path.exists(saved_path):
                        ds = xr.open_dataset(saved_path)
                    else:
                        ds = read_wrf_helen.read_wrf_fixed_point(lat=point["lat"], lon=point["lon"],
                            variables=variables, height_as_z_coord=True)
                
                if ds is None:
                    continue
                
                # Determine height variable
                if "z_unstag" in ds and model in ["ICON", "WRF"]:
                    height_var = ds["z_unstag"]
                else:
                    height_var = ds["z"]
                
                # Get temperature and height values at the requested timestamp
                if "temp" in ds:
                    temp = ds["temp"].sel(time=ts, method="nearest").values
                    height = height_var.sel(time=ts, method="nearest").values
                    
                    # Filter to max_height and remove NaNs
                    valid = ~np.isnan(temp) & ~np.isnan(height) & (height <= max_height)
                    temp = temp[valid]
                    height = height[valid]
                    
                    # Determine line style
                    line_dash = "dash" if model == "ICON2TE" else "solid"
                    
                    # Only show legend for first subplot
                    show_legend = (idx == 0)
                    
                    # Add temperature profile trace
                    fig.add_trace(go.Scatter(x=temp, y=height, mode='lines', name=model,
                        line=dict(color=model_colors[model], dash=line_dash, width=1.5), legendgroup=model,
                        showlegend=show_legend), row=row, col=col)
                    
                    # Add CAP height marker if available
                    # Calculate dT/dz and cap_height
                    try:
                        ds_with_gradient = ds.assign(dT_dz=ds['temp'].differentiate(coord='height'))
                        cap_height_val = calc_cap_height(ds_with_gradient).sel(time=ts, method="nearest").item()
                        
                        if not np.isnan(cap_height_val) and cap_height_val <= max_height:
                            # Get temperature at cap height
                            temp_at_cap = ds["temp"].sel(time=ts, method="nearest").sel(height=cap_height_val,
                                                                                        method="nearest").item()
                            
                            # Add marker
                            fig.add_trace(go.Scatter(x=[temp_at_cap], y=[cap_height_val], mode='markers',
                                marker=dict(symbol='x', size=10, color=model_colors[model],
                                            line=dict(width=2, color=model_colors[model])), name=f"{model} CAP",
                                legendgroup=model, showlegend=False,
                                hovertemplate=f"{model} CAP: {cap_height_val:.0f}m<extra></extra>"), row=row, col=col)
                    except Exception:
                        # Skip marker if CAP calculation fails
                        pass
                
                # Close dataset if it was opened from file
                if saved_path and os.path.exists(saved_path):
                    ds.close()
            
            except Exception as e:
                print(f"Warning: Could not load {model} data for {point_name}: {e}")
                continue
    
    # Update layout
    fig.update_layout(title_text=f"Vertical Temperature Profiles at {timestamp}", height=350 * n_rows,
        hovermode='closest', template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5))
    
    # Update axes
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            # Set y-axis range [0, max_height]
            fig.update_yaxes(range=[0, max_height], row=i, col=j)
            
            # Only show axis titles and ticks for top-left plot
            if i == 1 and j == 1:
                fig.update_xaxes(title_text="Temperature [°C]", row=i, col=j)
                fig.update_yaxes(title_text="Geopotential Height [m]", row=i, col=j)
            else:
                fig.update_xaxes(title_text="", showticklabels=False, row=i, col=j)
                fig.update_yaxes(title_text="", showticklabels=False, row=i, col=j)
    
    return fig


# EDITED: Wrapper function to save vertical profile small multiples
def plot_save_vert_profiles_small_multiples(timestamp: str = "2017-10-16T04:00:00", max_height: float = 3000,
                                            point_names: list = None) -> None:
    """
    Create and save vertical temperature profile small multiples plot for all points.
    timestamp: ISO format timestamp string (e.g. "2017-10-16T04:00:00")
    max_height: maximum height in meters to plot (default 3000m)
    point_names: list of point names from confg.py (default: ALL_POINTS from confg)
    """
    import os
    
    print(f"\nCreating vertical temperature profile small multiples for {timestamp}...")
    
    # Create small multiples plot
    fig = plot_vert_profiles_small_multiples(point_names, timestamp=timestamp, max_height=max_height)
    
    # Save as HTML in vertical_plots directory
    html_dir = os.path.join(confg.dir_PLOTS, "vertical_plots")
    os.makedirs(html_dir, exist_ok=True)
    
    # Create filename from timestamp
    ts_str = timestamp.replace(":", "").replace("-", "").replace("T", "_")
    html_path = os.path.join(html_dir, f"vertical_profiles_small_multiples_{ts_str}.html")
    
    fig.write_html(html_path)
    print(f"Saved vertical profile small multiples to: {html_path}")
    
    # Also show in browser
    fig.show()


if __name__ == "__main__":
    # Example: Create small multiples plot for 04 UTC on Oct 16
    plot_save_vert_profiles_small_multiples(timestamp="2017-10-16T04:00:00", max_height=3000)
