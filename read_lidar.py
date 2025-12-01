import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

import confg


def read_edit_original_lidar_data(data_path, instrument_name):
    """
    Reads LIDAR data from specified path, merges both files, converts time to datetime coordinates,
    filters to desired time window and 30-min intervals, then saves as merged file

    Parameters:
    -----------
    data_path : str
        Path to LIDAR data folder
    instrument_name : str
        Instrument name (e.g. 'SL88' or 'SLXR142')

    Returns:
    --------
    xarray.Dataset
        Processed and saved dataset with datetime coordinates, filtered to time window and 30-min intervals
    """

    # Find all NetCDF files in folder (exclude merged files)
    nc_files = [f for f in glob.glob(os.path.join(data_path, "*.nc"))
                if 'merged' not in os.path.basename(f)]

    if not nc_files:
        print(f"No NetCDF files found in {data_path}")
        return None

    print(f"Found {len(nc_files)} files for {instrument_name}")

    try:
        # Load all files and merge them
        # Use decode_times=False to avoid cftime issues
        datasets = []
        height_array = None  # Store height array separately

        for file in sorted(nc_files):
            ds = xr.open_dataset(file, decode_times=False)

            # Store height from first file
            if height_array is None and 'height' in ds.variables:
                height_array = ds['height'].values

            # Add instrument info
            ds.attrs['instrument'] = instrument_name
            datasets.append(ds)

        # Combine along time index dimension
        combined_ds = xr.concat(datasets, dim='NUMBER_OF_SCANS')

        if 'time' in combined_ds.variables:
            # Convert time values to datetime
            datetime_values = pd.to_datetime(combined_ds.time, unit='s')

            # Remove old time variable
            combined_ds = combined_ds.drop_vars("time")

            # Rename dimension and set datetime as coordinate
            combined_ds = combined_ds.rename({'NUMBER_OF_SCANS': 'time'})
            combined_ds = combined_ds.assign_coords({'time': datetime_values})

        # Add height as 1D coordinate (not as concatenated variable)
        if height_array is not None:
            combined_ds = combined_ds.assign_coords({'height_m': ('NUMBER_OF_GATES', height_array)})

        # Fix height coordinate: use first timestamp's height values as constant coordinate
        if 'height' in combined_ds.variables and len(combined_ds['height'].dims) == 2:
            print(f"  Converting 2D height coordinate to 1D using first timestamp...")
            # Get height values from first timestamp
            first_timestamp_heights = combined_ds['height'].isel(time=0).values

            # Remove the old 2D height variable
            combined_ds = combined_ds.drop_vars('height')

            # Add as 1D coordinate
            combined_ds = combined_ds.assign_coords({'height': ('NUMBER_OF_GATES', first_timestamp_heights)})
            print(f"  Height coordinate converted from 2D to 1D with {len(first_timestamp_heights)} levels")

        print(f"  Full time range: {combined_ds.time.values[0]} to {combined_ds.time.values[-1]}")

        # Define time window: 2017-10-15 12:00 to 2017-10-16 12:00
        start_time = '2017-10-15 12:00:00'
        end_time = '2017-10-16 12:00:00'

        # Time window filtering first
        filtered_ds = combined_ds.sel(time=slice(start_time, end_time))

        # Create 30-min interval timestamps (aligned to :00 and :30)
        target_times = pd.date_range(start=start_time, end=end_time, freq='30min')

        print(f"  Selecting nearest timesteps to 30-min intervals...")

        # Select nearest timesteps to target times
        try:
            subset_ds = filtered_ds.sel(time=target_times, method='nearest', tolerance='10min')
        except ValueError:
            print()

        print(f"  After filtering: {len(subset_ds.time)} timesteps (30-min intervals)")
        print(f"  Final time range: {subset_ds.time.values[0]} to {subset_ds.time.values[-1]}")

        # Rename NUMBER_OF_GATES dimension to height for clarity
        if 'NUMBER_OF_GATES' in subset_ds.dims:
            subset_ds = subset_ds.rename({'NUMBER_OF_GATES': 'height'})
            print(f"  Renamed dimension NUMBER_OF_GATES to height")

        # Ensure height coordinate is properly set as 1D after filtering
        if 'height' in subset_ds.coords and len(subset_ds.coords['height'].dims) == 1:
            print(f"  Height coordinate is now 1D with {len(subset_ds.coords['height'])} levels")
            print(f"  Height range: {subset_ds.coords['height'].min().values:.2f} - {subset_ds.coords['height'].max().values:.2f} m")

        # Save path - use appropriate filename based on instrument
        if instrument_name == 'SL88':
            output_path = os.path.join(data_path, 'sl88_merged.nc')
        elif instrument_name == 'SLXR142':
            output_path = os.path.join(data_path, 'slxr142_merged.nc')
        else:
            output_path = os.path.join(data_path, f'{instrument_name.lower()}_merged.nc')

        # Save as NetCDF
        subset_ds.to_netcdf(output_path)
        print(f"  Saved to: {output_path}")

        return subset_ds

    except Exception as e:
        print(f"Error loading {instrument_name} data: {e}")
        import traceback
        traceback.print_exc()
        return None


def read_merged_lidar_data():
    """
    Reads both pre-processed merged LIDAR files if they exist

    Returns:
    --------
    tuple
        (sl88_data, slxr142_data) - Two xarray Datasets or None if not available
    """
    sl88_data, slxr142_data = None, None

    # Check and load SL88 merged file
    if os.path.exists(confg.lidar_sl88_merged_path):
        print(f"Loading existing SL88 merged file: {confg.lidar_sl88_merged_path}")
        try:
            sl88_data = xr.open_dataset(confg.lidar_sl88_merged_path)
            print(
                f"  SL88: {len(sl88_data.time)} timesteps from {sl88_data.time.values[0]} to {sl88_data.time.values[-1]}")
        except Exception as e:
            print(f"Error loading SL88 merged file: {e}")
            sl88_data = None
    else:
        print(f"SL88 merged file not found: {confg.lidar_sl88_merged_path}")

    # Check and load SLXR142 merged file
    if os.path.exists(confg.lidar_slxr142_merged_path):
        print(f"Loading existing SLXR142 merged file: {confg.lidar_slxr142_merged_path}")
        try:
            slxr142_data = xr.open_dataset(confg.lidar_slxr142_merged_path)
            print(
                f"  SLXR142: {len(slxr142_data.time)} timesteps from {slxr142_data.time.values[0]} to {slxr142_data.time.values[-1]}")
        except Exception as e:
            print(f"Error loading SLXR142 merged file: {e}")
            slxr142_data = None
    else:
        print(f"SLXR142 merged file not found: {confg.lidar_slxr142_merged_path}")

    return sl88_data, slxr142_data


def plot_lidar_comparison(sl88_data, slxr142_data, save_plot=True):
    """
    Creates interactive Plotly comparison plots with time slider for wind speed and direction of both LIDAR systems
    Data is expected to be already filtered to desired time range and 30-min intervals

    Parameters:
    -----------
    sl88_data : xarray.Dataset
        SL88 LIDAR data (with datetime coordinates, already filtered)
    slxr142_data : xarray.Dataset
        SLXR142 LIDAR data (with datetime coordinates, already filtered)
    save_plot : bool
        Whether to save the plot
    """

    if sl88_data is None or slxr142_data is None:
        print("One or both datasets are empty!")
        return

    # Colors for both systems
    colors = {'SL88': '#1f77b4', 'SLXR142': '#ff7f0e'}
    # extract heights
    heights_sl88 = sl88_data['height_m'].values
    heights_slxr142 = slxr142_data['height_m'].values

    print(f"SL88: {len(sl88_data.time)} timesteps from {sl88_data.time.values[0]} to {sl88_data.time.values[-1]}")
    print(f"SLXR142: {len(slxr142_data.time)} timesteps from {slxr142_data.time.values[0]} to {slxr142_data.time.values[-1]}")

    # Use full time series for slider
    max_frames = max(len(sl88_data.time), len(slxr142_data.time))

    print(f"Creating interactive plot with {max_frames} timesteps...")

    # Create subplots: [Wind Speed | Wind Direction]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Wind Speed [m/s]', 'Wind Direction [°]'),
        horizontal_spacing=0.15
    )

    # Create frames for slider
    frames = []

    for frame_idx in range(max_frames):
        frame_traces = []

        # ====== SUBPLOT 1: Wind Speed ======
        if 'ff' in sl88_data.variables and frame_idx < len(sl88_data.time):
            ff_sl88 = sl88_data['ff'].isel(time=frame_idx).values
            frame_traces.append(
                go.Scatter(
                    x=ff_sl88,
                    y=heights_sl88,
                    mode='lines+markers',
                    name='SL88',
                    line=dict(color=colors['SL88'], width=2),
                    marker=dict(size=4, symbol='circle'),
                    legendgroup='SL88',
                    showlegend=True,
                    xaxis='x1',
                    yaxis='y1'
                )
            )

        if 'ff' in slxr142_data.variables and frame_idx < len(slxr142_data.time):
            ff_slxr142 = slxr142_data['ff'].isel(time=frame_idx).values
            frame_traces.append(
                go.Scatter(
                    x=ff_slxr142,
                    y=heights_slxr142,
                    mode='lines+markers',
                    name='SLXR142',
                    line=dict(color=colors['SLXR142'], width=2),
                    marker=dict(size=4, symbol='square'),
                    legendgroup='SLXR142',
                    showlegend=True,
                    xaxis='x1',
                    yaxis='y1'
                )
            )

        # ====== SUBPLOT 2: Wind Direction ======
        if 'dd' in sl88_data.variables and frame_idx < len(sl88_data.time):
            dd_sl88 = sl88_data['dd'].isel(time=frame_idx).values
            frame_traces.append(
                go.Scatter(
                    x=dd_sl88,
                    y=heights_sl88,
                    mode='markers',
                    name='SL88',
                    marker=dict(size=8, symbol='circle', color=colors['SL88']),
                    legendgroup='SL88',
                    showlegend=False,
                    xaxis='x2',
                    yaxis='y2'
                )
            )

        if 'dd' in slxr142_data.variables and frame_idx < len(slxr142_data.time):
            dd_slxr142 = slxr142_data['dd'].isel(time=frame_idx).values
            frame_traces.append(
                go.Scatter(
                    x=dd_slxr142,
                    y=heights_slxr142,
                    mode='markers',
                    name='SLXR142',
                    marker=dict(size=8, symbol='square', color=colors['SLXR142']),
                    legendgroup='SLXR142',
                    showlegend=False,
                    xaxis='x2',
                    yaxis='y2'
                )
            )

        # Frame names from datetime coordinate - use SL88 time if available, else SLXR142
        if frame_idx < len(sl88_data.time):
            timestamp_str = pd.to_datetime(sl88_data.time.values[frame_idx]).strftime('%Y-%m-%d %H:%M')
        else:
            timestamp_str = pd.to_datetime(slxr142_data.time.values[frame_idx]).strftime('%Y-%m-%d %H:%M')

        frames.append(go.Frame(data=frame_traces, name=timestamp_str))

    # Add initial traces (first frame)
    if frames:
        for i, trace in enumerate(frames[0].data):
            col = 1 if i < 2 else 2
            fig.add_trace(trace, row=1, col=col)

    # Update layout with synchronized y-axes
    fig.update_xaxes(title_text="Wind Speed [m/s]", range=[0, 10], row=1, col=1)
    fig.update_xaxes(title_text="Wind Direction [°]", range=[0, 360], row=1, col=2)
    fig.update_yaxes(title_text="Height [m]", range=[0, 1000], row=1, col=1)
    fig.update_yaxes(title_text="Height [m]", range=[0, 1000], row=1, col=2, matches='y')

    # Create slider
    sliders = [dict(
        active=0,
        yanchor="top",
        y=-0.2,
        xanchor="left",
        x=0.0,
        currentvalue=dict(prefix="Time: ", visible=True, xanchor="left"),
        pad=dict(b=10, t=50),
        len=0.9,
        steps=[
            dict(
                args=[[frame.name],
                      dict(frame=dict(duration=300, redraw=True), mode="immediate", transition=dict(duration=300))],
                label=frame.name,
                method="animate"
            )
            for frame in frames
        ]
    )]

    # Play/Pause Buttons
    updatemenus = [dict(
        type="buttons",
        direction="left",
        x=0.0,
        y=-0.15,
        xanchor="left",
        yanchor="top",
        pad=dict(r=10, t=70),
        showactive=False,
        buttons=[
            dict(label="▶ Play", method="animate", args=[None,
                                                         dict(frame=dict(duration=500, redraw=True), fromcurrent=True,
                                                              mode="immediate", transition=dict(duration=300))]),
            dict(label="⏸ Pause", method="animate", args=[[None],
                                                          dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                                               transition=dict(duration=0))])
        ]
    )]

    fig.update_layout(
        title=dict(
            text='LIDAR Wind Measurements Comparison: SL88 vs SLXR142<br><sub>October 15, 2017 12:00 - October 16, 2017 12:00</sub>',
            x=0.5,
            xanchor='center'
        ),
        height=650,
        width=1400,
        hovermode='closest',
        updatemenus=updatemenus,
        sliders=sliders,
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top')
    )

    fig.frames = frames

    if save_plot:
        plot_path = os.path.join(confg.dir_PLOTS, 'lidar_comparison_interactive.html')
        os.makedirs(confg.dir_PLOTS, exist_ok=True)
        fig.write_html(plot_path)
        print(f"Interactive plot saved to: {plot_path}")

    print("Interactive plot created!")
    return fig


if __name__ == "__main__":
    """
    Main function to run LIDAR data analysis
    """
    print("Starting LIDAR data analysis...")

    # First check if processed merged files already exist
    sl88_data, slxr142_data = read_merged_lidar_data()

    # If merged files don't exist or are empty, process original data
    if sl88_data is None or slxr142_data is None:
        print("Merged files not found or faulty - processing original data...")
        sl88_data = read_edit_original_lidar_data(confg.lidar_sl88, "SL88")
        slxr142_data = read_edit_original_lidar_data(confg.lidar_slxr142, "SLXR142")

    else:
       print("Merged files already exist - skipping processing!")

    # Create comparison plot
    if sl88_data is not None or slxr142_data is not None:
        plot_lidar_comparison(sl88_data, slxr142_data)
    else:
        print("No data available for plot!")

    print("LIDAR analysis completed!")
