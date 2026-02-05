"""
Temperature and wind spatial plots:
Similar to plot_heat_fluxes but for temperature variables instead of heat flux variables.
Plots temperature with lowest level wind barbs overlaid.

The script reads data first, saves it to model timeseries folders (dir_timeseries_AROME),
and then creates plots for different regions (full extent, Zillertal, Wipp valley, valley exit).

Features:
- Wind barbs (u, v) from lowest model level
- Temperature (temp) as color background
- Terrain contours overlay
- Small multiples format across time
- Detail plots for specific regions
- Uses "alternative" approach with direct height indexing (method="nearest")
- Temperature range globally set to [5, 25]°C
- Data saved to timeseries folders before plotting
- All plots shown together at the end

Variables used: ["th", "temp", "u", "v"]
"""
import fix_win_DLL_loading_issue

fix_win_DLL_loading_issue
# import sys
# sys.path.append("C:/Users/eleme/Documents/1Uni_Laptop/model_comparison_codes")

import os

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import confg
import read_in_arome
import read_wrf_helen
import read_icon_model_3D
import read_ukmo
from plot_heat_fluxes import (plot_small_multiples, make_times)

# Simple dictionaries for variable properties

TEMPERATURE_LABELS = {"temp": "Temperature", "th": "Potential Temperature", }

# Temperature ranges - global definition for [5, 25] degrees Celsius
TEMPERATURE_RANGES = {"temp": {"vmin": 5, "vmax": 25},  # [5, 25]°C
                      "th": {"vmin": 278, "vmax": 298},  # [5, 25]°C in Kelvin (potential temp)
                      }


def read_and_save_arome_data(times, variables):
    """
    Read AROME data for given times and variables, optionally save to timeseries folder.
    Check if data already exists before reading from scratch.

    Args:
        times: List of datetime objects
        variables: List of variables to read
        save_data: Whether to save data to timeseries folder

    Returns:
        xarray.Dataset: Combined AROME dataset
    """
    print("Processing AROME data...")

    # Convert times list to pandas DatetimeIndex if needed
    if not isinstance(times, pd.DatetimeIndex):
        times = pd.DatetimeIndex(times)

    # Define save path
    save_dir = confg.dir_AROME_timeseries
    filename = f"arome_temperature_wind_{times[0].strftime('%Y%m%d_%H%M')}_{times[-1].strftime('%Y%m%d_%H%M')}.nc"
    filepath = os.path.join(save_dir, filename)

    # Check if data already exists
    if os.path.exists(filepath):
        print(f"  ✓ Loading existing AROME data from {filepath}")
        return xr.open_dataset(filepath)

    print("  Reading AROME data from scratch...")
    ds_list = []
    for time in times:
        ds = read_in_arome.read_in_arome_fixed_time(day=time.day, hour=time.hour, min=time.minute,
                                                    variables=variables + ["hgt"], min_lat=confg.lat_min,
                                                    max_lat=confg.lat_max, min_lon=confg.lon_min, max_lon=confg.lon_max)
        if "time" in getattr(ds, "dims", []):
            ds = ds.isel(time=0)

        # Select only lowest level for 3D variables to reduce file size
        data = {}
        for var in variables:
            if var in ds:
                if "height" in ds[var].dims:
                    data[var] = ds[var].sel(height=1, method="nearest")
                else:
                    data[var] = ds[var]

        # Add surface height and coordinates (no height dimension)
        data["hgt"] = ds["hgt"]
        data["lat"] = ds["lat"]
        data["lon"] = ds["lon"]
        data["time"] = ds["time"]

        ds_level1 = xr.Dataset(data)
        ds_list.append(ds_level1)

    arome_ds = xr.concat(ds_list, dim="time")

    os.makedirs(save_dir, exist_ok=True)
    print(f"  Saving AROME data (lowest level only) to {filepath}")
    arome_ds.to_netcdf(filepath)
    print(f"  ✓ Saved AROME data")

    return arome_ds


def read_and_save_icon_data(times, variables):
    """
    Read ICON data for given times and variables, optionally save to timeseries folder.
    Check if data already exists before reading from scratch.

    Args:
        times: List of datetime objects
        variables: List of variables to read
        save_data: Whether to save data to timeseries folder

    Returns:
        xarray.Dataset: Combined ICON dataset
    """
    print("Processing ICON data...")

    # Convert times list to pandas DatetimeIndex if needed
    if not isinstance(times, pd.DatetimeIndex):
        times = pd.DatetimeIndex(times)

    # Define save path
    save_dir = confg.dir_timeseries_ICON if hasattr(confg,
                                                    'dir_timeseries_ICON') else confg.dir_AROME_timeseries.replace(
        'AROME', 'ICON')
    filename = f"icon_temperature_wind_{times[0].strftime('%Y%m%d_%H%M')}_{times[-1].strftime('%Y%m%d_%H%M')}.nc"
    filepath = os.path.join(save_dir, filename)

    # Check if data already exists
    if os.path.exists(filepath):
        print(f"  ✓ Loading existing ICON data from {filepath}")
        return xr.open_dataset(filepath)

    print("  Reading ICON data from scratch...")
    ds_list = []
    for time in times:
        try:
            ds = read_icon_model_3D.read_icon_fixed_time(day=time.day, hour=time.hour, min=time.minute, variant="ICON",
                                                         variables=variables + ["z_unstag"])
            if "time" in getattr(ds, "dims", []):
                ds = ds.isel(time=0)

            # Select only lowest level for 3D variables to reduce file size
            data = {}
            for var in variables:
                if var in ds:
                    if "height" in ds[var].dims:
                        data[var] = ds[var].sel(height=1, method="nearest")
                    else:
                        data[var] = ds[var]

            # Add terrain height and coordinates
            if "z_unstag" in ds:
                if "height" in ds["z_unstag"].dims:
                    data["z"] = ds["z_unstag"].sel(height=1, method="nearest")
                else:
                    data["z"] = ds["z_unstag"]
            data["lat"] = ds["lat"]
            data["lon"] = ds["lon"]
            data["time"] = ds["time"]

            ds_level1 = xr.Dataset(data)
            ds_list.append(ds_level1)
        except Exception as e:
            print(f"    Warning: Could not read ICON data for {time}: {e}")
            continue

    if not ds_list:
        print("  ✗ No ICON data could be loaded")
        return None

    icon_ds = xr.concat(ds_list, dim="time")

    os.makedirs(save_dir, exist_ok=True)
    print(f"  Saving ICON data (lowest level only) to {filepath}")
    icon_ds.to_netcdf(filepath)
    print(f"  ✓ Saved ICON data")

    return icon_ds


def read_and_save_icon2te_data(times, variables):
    """
    Read ICON2TE data for given times and variables, optionally save to timeseries folder.
    Check if data already exists before reading from scratch.

    Args:
        times: List of datetime objects
        variables: List of variables to read
        save_data: Whether to save data to timeseries folder

    Returns:
        xarray.Dataset: Combined ICON2TE dataset
    """
    print("Processing ICON2TE data...")

    # Convert times list to pandas DatetimeIndex if needed
    if not isinstance(times, pd.DatetimeIndex):
        times = pd.DatetimeIndex(times)

    # Define save path
    save_dir = confg.icon2TE_folder_3D + "/timeseries"
    filename = f"icon2te_temperature_wind_{times[0].strftime('%Y%m%d_%H%M')}_{times[-1].strftime('%Y%m%d_%H%M')}.nc"
    filepath = os.path.join(save_dir, filename)

    # Check if data already exists
    if os.path.exists(filepath):
        print(f"  ✓ Loading existing ICON2TE data from {filepath}")
        return xr.open_dataset(filepath)

    print("  Reading ICON2TE data from scratch...")
    ds_list = []
    for time in times:
        try:
            ds = read_icon_model_3D.read_icon_fixed_time(day=time.day, hour=time.hour, min=time.minute,
                                                         variant="ICON2TE", variables=variables + ["z_unstag"])
            if "time" in getattr(ds, "dims", []):
                ds = ds.isel(time=0)

            # Select only lowest level for 3D variables to reduce file size
            data = {}
            for var in variables:
                if var in ds:
                    if "height" in ds[var].dims:
                        data[var] = ds[var].sel(height=1, method="nearest")
                    else:
                        data[var] = ds[var]

            # Add terrain height and coordinates
            if "z_unstag" in ds:
                if "height" in ds["z_unstag"].dims:
                    data["z"] = ds["z_unstag"].sel(height=1, method="nearest")
                else:
                    data["z"] = ds["z_unstag"]
            data["lat"] = ds["lat"]
            data["lon"] = ds["lon"]
            data["time"] = ds["time"]

            ds_level1 = xr.Dataset(data)
            ds_list.append(ds_level1)
        except Exception as e:
            print(f"    Warning: Could not read ICON2TE data for {time}: {e}")
            continue

    if not ds_list:
        print("  ✗ No ICON2TE data could be loaded")
        return None

    icon2te_ds = xr.concat(ds_list, dim="time")

    os.makedirs(save_dir, exist_ok=True)
    print(f"  Saving ICON2TE data (lowest level only) to {filepath}")
    icon2te_ds.to_netcdf(filepath)
    print(f"  ✓ Saved ICON2TE data")

    return icon2te_ds


def read_and_save_um_data(times, variables):
    """
    Read UM (UK Met Office) data for given times and variables, optionally save to timeseries folder.
    Check if data already exists before reading from scratch.

    Args:
        times: List of datetime objects
        variables: List of variables to read
        save_data: Whether to save data to timeseries folder

    Returns:
        xarray.Dataset: Combined UM dataset
    """
    print("Processing UM data...")

    # Convert times list to pandas DatetimeIndex if needed
    if not isinstance(times, pd.DatetimeIndex):
        times = pd.DatetimeIndex(times)

    # Define save path
    save_dir = confg.ukmo_folder + "/timeseries"
    filename = f"um_temperature_wind_{times[0].strftime('%Y%m%d_%H%M')}_{times[-1].strftime('%Y%m%d_%H%M')}.nc"
    filepath = os.path.join(save_dir, filename)

    # Check if data already exists
    if os.path.exists(filepath):
        print(f"  ✓ Loading existing UM data from {filepath}")
        return xr.open_dataset(filepath)

    print("  Reading UM data from scratch...")
    ds_list = []
    for time in times:
        try:
            ds = read_ukmo.read_ukmo_fixed_time(day=time.day, hour=time.hour, min=time.minute, variables=variables)
            if "time" in getattr(ds, "dims", []):
                ds = ds.isel(time=0)

            # Select only lowest level for 3D variables to reduce file size
            data = {}
            for var in variables:
                if var in ds:
                    if "height" in ds[var].dims:
                        data[var] = ds[var].sel(height=1, method="nearest")
                    else:
                        data[var] = ds[var]

            # Add coordinates
            data["lat"] = ds["lat"]
            data["lon"] = ds["lon"]
            data["time"] = ds["time"]

            ds_level1 = xr.Dataset(data)
            ds_list.append(ds_level1)
        except Exception as e:
            print(f"    Warning: Could not read UM data for {time}: {e}")
            continue

    if not ds_list:
        print("  ✗ No UM data could be loaded")
        return None

    um_ds = xr.concat(ds_list, dim="time")

    os.makedirs(save_dir, exist_ok=True)
    print(f"  Saving UM data (lowest level only) to {filepath}")
    um_ds.to_netcdf(filepath)
    print(f"  ✓ Saved UM data")

    return um_ds


def read_and_save_wrf_data(times, variables):
    """
    Read WRF data for given times and variables, optionally save to timeseries folder.
    Check if data already exists before reading from scratch.

    Args:
        times: List of datetime objects
        variables: List of variables to read
        save_data: Whether to save data to timeseries folder

    Returns:
        xarray.Dataset: Combined WRF dataset
    """
    print("Processing WRF data...")

    # Convert times list to pandas DatetimeIndex if needed
    if not isinstance(times, pd.DatetimeIndex):
        times = pd.DatetimeIndex(times)

    # Define save path
    save_dir = confg.wrf_folder + "/timeseries"
    filename = f"wrf_temperature_wind_{times[0].strftime('%Y%m%d_%H%M')}_{times[-1].strftime('%Y%m%d_%H%M')}.nc"
    filepath = os.path.join(save_dir, filename)

    # Check if data already exists
    if os.path.exists(filepath):
        print(f"  ✓ Loading existing WRF data from {filepath}")
        return xr.open_dataset(filepath)

    print("  Reading WRF data from scratch...")
    ds_list = []
    for time in times:
        ds = read_wrf_helen.read_wrf_fixed_time(day=time.day, hour=time.hour, min=time.minute,
                                                variables=variables + ["z_unstag", "hgt"])
        if "time" in getattr(ds, "dims", []):
            ds = ds.isel(time=0)

        # Select only lowest level for 3D variables to reduce file size
        data = {}
        for var in variables:
            if var in ds:
                if "height" in ds[var].dims:
                    data[var] = ds[var].sel(height=1, method="nearest")
                else:
                    data[var] = ds[var]

        # Add terrain height and surface height
        if "z_unstag" in ds:
            if "height" in ds["z_unstag"].dims:
                data["z_unstag"] = ds["z_unstag"].sel(height=1, method="nearest")
            else:
                data["z_unstag"] = ds["z_unstag"]
        if "hgt" in ds:
            data["hgt"] = ds["hgt"]
        data["lat"] = ds["lat"]
        data["lon"] = ds["lon"]
        data["time"] = ds["time"]

        ds_level1 = xr.Dataset(data)
        ds_list.append(ds_level1)

    wrf_ds = xr.concat(ds_list, dim="time")
    wrf_ds = wrf_ds.sel(bottom_top_stag=1, method="nearest")  # select & save lowest level only

    os.makedirs(save_dir, exist_ok=True)
    print(f"  Saving WRF data (lowest level only) to {filepath}")
    wrf_ds.to_netcdf(filepath)
    print(f"  ✓ Saved WRF data")

    return wrf_ds


def plot_temperature_detail_for_extent(model_datasets, times, lon_extent, lat_extent, figsize, contour_line_dist,
        extent_name="detail", variables_to_plot=None, barb_length=None, step=2):
    """
    Plot detailed small multiples for temperature variables for all available models for a given extent.
    Uses the imported plot_small_multiples function from plot_heat_fluxes.py but with temperature-specific settings.

    Args:
        model_datasets: Dictionary with model names as keys and datasets as values
        times: Time selection for plotting
        lon_extent: Tuple (lon_min, lon_max) for plot extent
        lat_extent: Tuple (lat_min, lat_max) for plot extent
        extent_name: Name for the extent (used in print messages and filenames)
        variables_to_plot: List of variables to plot (if None, uses ["temp"])
        barb_length: Length of wind barbs. Controls visual size of wind barbs.
        step: Subsample step for wind barbs (default: 2). Controls distance between wind barbs.
    """
    if variables_to_plot is None:
        variables_to_plot = ["temp"]

    print(f"\n" + "=" * 70)
    print(f"Creating {extent_name.upper()} DETAIL plots")
    print("=" * 70)

    # Process each available model
    for model_name, model_ds in model_datasets.items():
        # if model_name == "WRF":  # Skip WRF model, correct heat flux plot later...
        #     continue
        if model_ds is None:
            print(f"  Warning: {model_name} dataset not available, skipping...")
            continue

        print(f"  Processing {model_name}...")

        # Subset dataset to the specified extent
        try:
            model_detail = model_ds.sel(lat=slice(lat_extent[0], lat_extent[1]),
                                        lon=slice(lon_extent[0], lon_extent[1]))
        except Exception as e:
            print(f"    Warning: Could not subset {model_name} data to extent: {e}")
            continue

        # Plot all temperature variables for the detail extent
        for var in variables_to_plot:
            print(f"    Processing {model_name} {var}...")

            # Check if variable exists in dataset
            if var in model_detail:
                print(f"      Plotting {model_name} {var} ({extent_name} detail)...")

                # Use temperature-specific colormap and label directly
                plot_small_multiples(ds=model_detail.sel(time=times), model=model_name, variable=var,
                                     vmin=TEMPERATURE_RANGES[var]["vmin"], vmax=TEMPERATURE_RANGES[var]["vmax"],
                                     lon_extent=lon_extent, lat_extent=lat_extent, figsize=figsize,
                                     filename_suffix=f"{extent_name}_{model_name}",
                                     contour_line_dist=contour_line_dist, barb_length=barb_length, step=step,
                                     plot_dir="temperature_wind",
                                     custom_label=TEMPERATURE_LABELS[var])

            else:
                print(f"      Warning: {var} not found in {model_name} dataset")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


if __name__ == "__main__":
    # Choose which plots to create (similar to plot_heat_fluxes.py):
    create_orig_hf_plots = True  # extent of original heat flux plots (around Ibk)
    create_wipp_detail_plots = False  # Detailed Wipp Valley region
    create_valley_exit_detail = False  # Specific detail plots for the valley exit region
    create_ziller_detail_plots = True  # Detailed Zillertal region

    # Data saving options
    save_data_to_timeseries = True  # Save data to timeseries folders

    # Variables to process and plot (as requested: ["th", "temp", "u", "v"])
    variables_to_process = ["th", "temp", "u", "v", "z"]  # z needed for terrain contours
    variables_to_plot = ["temp"]  # Plot temperature with wind barbs

    # Create time range
    times = make_times(start_day=15, start_hour=14, start_minute=0, end_day=16, end_hour=12, end_minute=0, freq="2h")

    # Read and save data for all 5 models
    print("=" * 70)
    print("READING AND SAVING DATA FOR ALL 5 MODELS")
    print("=" * 70)

    # Dictionary to store all model datasets
    model_datasets = {}

    # Read all models with error handling
    model_datasets["AROME"] = read_and_save_arome_data(times, variables_to_process)
    model_datasets["ICON"] = read_and_save_icon_data(times, variables_to_process)
    model_datasets["ICON2TE"] = read_and_save_icon2te_data(times, variables_to_process)
    model_datasets["UM"] = read_and_save_um_data(times, variables_to_process)
    model_datasets["WRF"] = read_and_save_wrf_data(times, variables_to_process)

    # Print summary of successfully loaded models
    available_models = [name for name, ds in model_datasets.items() if ds is not None]
    unavailable_models = [name for name, ds in model_datasets.items() if ds is None]

    print(f"\n✓ Successfully loaded models: {', '.join(available_models)}")
    if unavailable_models:
        print(f"✗ Failed to load models: {', '.join(unavailable_models)}")

    # Create plots for different extents
    if create_orig_hf_plots:
        print("\n" + "=" * 70)
        print("Creating FULL EXTENT plots")
        print("=" * 70)

        # lon_extent = (confg.lon_min, confg.lon_max)  # plot full extent
        # lat_extent = (confg.lat_min, confg.lat_max)


        plot_temperature_detail_for_extent(model_datasets=model_datasets, times=times, lon_extent=confg.lon_hf_extent,
                                           lat_extent=confg.lat_hf_extent, figsize=(12, 8), contour_line_dist=250,
                                           extent_name="_zentral_inn", variables_to_plot=variables_to_plot,
                                           barb_length=3,
                                           step=2)
    if create_wipp_detail_plots:
        plot_temperature_detail_for_extent(model_datasets=model_datasets, times=times, lon_extent=confg.lon_wipp_extent,
                                           lat_extent=confg.lat_wipp_extent, figsize=(8, 8), contour_line_dist=100,
                                           extent_name="_wipp_valley", variables_to_plot=variables_to_plot,
                                           barb_length=3, step=2)

    if create_valley_exit_detail:
        # from Achensee till Zell am See and Jenbach till Rosenheim
        plot_temperature_detail_for_extent(model_datasets=model_datasets, times=times,
                                           lon_extent=confg.lon_inn_exit_extent, lat_extent=confg.lat_inn_exit_extent,
                                           figsize=(11, 8), contour_line_dist=100, extent_name="_valley_exit",
                                           variables_to_plot=variables_to_plot, barb_length=3, step=2)

    if create_ziller_detail_plots:
        # Use the generic function for Zillertal plots
        plot_temperature_detail_for_extent(model_datasets=model_datasets, times=times,
                                           lon_extent=confg.lon_ziller_extent, lat_extent=confg.lat_ziller_extent,
                                           figsize=(8, 8), contour_line_dist=100, extent_name="_ziller_valley",
                                           variables_to_plot=variables_to_plot, barb_length=3, step=1)


    print("\n" + "=" * 70)
    print("Temperature and wind plotting completed!")
    print(f"Processed models: {', '.join(available_models)}")
    print("=" * 70)

    # Show all plots at once
    plt.show()