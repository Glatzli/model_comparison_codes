"""
Manage timeseries data for all models at all defined points.

This module handles:
- Loading timeseries from saved NetCDF files
- Reading fresh data if files don't exist
- Saving timeseries for future reuse
- Computing and saving timeseries for all points and models

Functions are imported and used by plot_vertical_profiles.py and plot_cap_height.py
to ensure consistent data handling.
"""
from __future__ import annotations

import os
from typing import List

import xarray as xr

import confg
import read_icon_model_3D
import read_in_arome
import read_ukmo
import read_wrf_helen
from calculations_and_plots.calc_vhd import read_dems_calc_pcgp

# Variables needed for all models
variables = ["udir", "wspd", "q", "p", "th", "temp", "z", "z_unstag"]

# All point locations defined in confg.py
ALL_POINTS = ["ibk_villa", "ibk_uni", "ibk_airport", "woergl", "kiefersfelden", "telfs", "wipp_valley", "ziller_valley",
              "ziller_ried"]

# Model processing order
MODEL_ORDER = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]


def get_timeseries_path(model: str, point_name: str) -> str:
    """
    Build the file path to a saved timeseries NetCDF file for a specific model and point.
    
    The timeseries files are stored with the naming convention:
    MODEL_FOLDER/timeseries/modelname_pointname_timeseries_height_as_z.nc
    
    Args:
        model: Name of the weather model (AROME, ICON, ICON2TE, UM, WRF)
        point_name: Name of the point location (from confg.py)
    
    Returns:
        Full path to the timeseries file
    """
    # Get the base directory for each model
    if model == "AROME":
        base = confg.dir_AROME
    elif model == "ICON":
        base = confg.icon_folder_3D
    elif model == "ICON2TE":
        base = confg.icon2TE_folder_3D
    elif model == "UM":
        base = confg.ukmo_folder
    elif model == "WRF":
        base = confg.wrf_folder
    else:
        return ""
    
    # Construct the filename with lowercase model name
    model_name_lower = model.lower()
    if model == "ICON2TE":
        model_name_lower = "icon2te"
    
    return os.path.join(base, "timeseries", f"{model_name_lower}_{point_name}_timeseries_height_as_z.nc")


def save_timeseries(ds: xr.Dataset, model: str, point_name: str) -> None:
    """
    Save a timeseries dataset to a NetCDF file for future reuse.
    
    Only saves if the file doesn't already exist to avoid overwriting and permission issues.
    Creates the necessary directory structure if it doesn't exist.
    
    Args:
        ds: xarray Dataset containing the timeseries data
        model: Name of the weather model
        point_name: Name of the point location
    """
    timeseries_path = get_timeseries_path(model, point_name)
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(timeseries_path), exist_ok=True)
    
    # Only save if the file doesn't already exist
    if not os.path.exists(timeseries_path):
        print(f"  Saving {model} timeseries for {point_name} to: {timeseries_path}")
        try:
            ds.to_netcdf(timeseries_path)
            print(f"  ✓ Successfully saved {model} timeseries")
        except Exception as e:
            print(f"  ✗ Warning: Could not save timeseries file {timeseries_path}: {e}")
    else:
        print(f"  ℹ Timeseries file already exists, skipping save: {os.path.basename(timeseries_path)}")


def load_timeseries(model: str, point_name: str) -> xr.Dataset | None:
    """
    Load timeseries from saved NetCDF file.
    
    Args:
        model: Name of the weather model
        point_name: Name of the point location
    
    Returns:
        xarray Dataset with timeseries data, or None if file doesn't exist
    """
    timeseries_path = get_timeseries_path(model, point_name)
    
    if os.path.exists(timeseries_path):
        print(f"  Loading {model} from saved timeseries: {os.path.basename(timeseries_path)}")
        try:
            ds = xr.open_dataset(timeseries_path)
            return ds
        except Exception as e:
            print(f"  ✗ Warning: Could not load saved file. Error: {e}")
            return None
    
    return None


def read_fresh_timeseries(model: str, point: dict, point_name: str, variables_list: list,
                          height_as_z_coord: str = "above_terrain") -> xr.Dataset | None:
    """
    Read fresh timeseries data from model output files.
    
    Uses PCGP (Physically Consistent Grid Point) selection for accurate point representation.
    
    Args:
        model: Name of the weather model
        point: Dictionary with 'lat' and 'lon' keys
        point_name: Name of the point location
        variables_list: List of variable names to read
        height_as_z_coord: Whether to use height as z-coordinate
    
    Returns:
        xarray Dataset with fresh timeseries data, or None if reading fails
    """
    print(f"  Reading fresh {model} data for {point_name}...")
    
    try:
        # Get PCGP for accurate point representation
        pcgp_arome, pcgp_icon, pcgp_um, pcgp_wrf = read_dems_calc_pcgp(lat=point["lat"], lon=point["lon"])
        
        # AROME & UM aren't staggered -> remove z_unstag if present
        variables_for_reading = [v for v in variables_list if not (model in ["AROME", "UM"] and v == "z_unstag")]
        
        # Map model to appropriate read function and PCGP
        read_functions = {"AROME": (read_in_arome.read_in_arome_fixed_point, pcgp_arome, {}),
            "ICON": (read_icon_model_3D.read_icon_fixed_point, pcgp_icon, {"variant": "ICON"}),
            "ICON2TE": (read_icon_model_3D.read_icon_fixed_point, pcgp_icon, {"variant": "ICON2TE"}),
            "UM": (read_ukmo.read_ukmo_fixed_point, pcgp_um, {}),
            "WRF": (read_wrf_helen.read_wrf_fixed_point, pcgp_wrf, {})}
        
        if model not in read_functions:
            print(f"  ✗ Unknown model: {model}")
            return None
        
        read_func, pcgp, extra_kwargs = read_functions[model]
        ds = read_func(lat=pcgp.y.values, lon=pcgp.x.values, variables=variables_for_reading,
            height_as_z_coord=height_as_z_coord, **extra_kwargs)
        
        return ds
    
    except Exception as e:
        print(f"  ✗ Error reading fresh data for {model} at {point_name}: {e}")
        return None


def load_or_read_timeseries(model: str, point: dict, point_name: str, variables_list: list = variables,
                            height_as_z_coord: str = "above_terrain") -> xr.Dataset | None:
    """
    Load timeseries from saved file if it exists, otherwise read fresh data and save it.
    
    This function implements the core logic for efficient data loading:
    1. Check if saved timeseries file exists
    2. If yes: load from file
    3. If no: read fresh data from model output and save for future use
    
    Args:
        model: Name of the weather model (AROME, ICON, ICON2TE, UM, WRF)
        point: Dictionary with 'lat' and 'lon' keys
        point_name: Name of the point location (from confg.py)
        variables_list: List of variable names to read (default: uses global variables list)
        height_as_z_coord: same as in read in functions...
    
    Returns:
        xarray Dataset with timeseries data, or None if loading/reading fails
    """
    
    # Try to load from saved file first, if not returns None...
    ds = load_timeseries(model, point_name)
    if ds is not None:
        return ds
    
    # If no saved file exists, read fresh data
    ds = read_fresh_timeseries(model, point, point_name, variables_list, height_as_z_coord)
    if ds is not None:
        # Save the freshly read data for future use
        save_timeseries(ds, model, point_name)
    
    return ds


def compute_and_save_all_timeseries(point_names: List[str] = None, variables_list: list = None,
                                    height_as_z_coord: bool = True) -> None:
    """
    Compute and save timeseries for all models at all specified points.
    
    This is the main function to pre-compute and save timeseries data for later use
    by plotting and analysis scripts.
    
    Args:
        point_names: List of point names from confg.py (default: ALL_POINTS)
        variables_list: List of variable names to read (default: uses global variables list)
        height_as_z_coord: Whether to use height as z-coordinate (default: True)
    """
    if point_names is None:
        point_names = ALL_POINTS
    
    if variables_list is None:
        variables_list = variables
    
    print(f"\n{'=' * 70}")
    print(f"Computing and saving timeseries for {len(MODEL_ORDER)} models at {len(point_names)} points")
    print(f"{'=' * 70}\n")
    
    total_computed = 0
    total_skipped = 0
    
    for point_name in point_names:
        point = getattr(confg, point_name, None)
        if point is None:
            print(f"⚠ Skipping {point_name} - not found in confg")
            continue
        
        print(f"\n{'-' * 70}")
        print(f"Processing: {point['name']} ({point_name})")
        print(f"{'-' * 70}")
        
        for model in MODEL_ORDER:
            # Check if file already exists
            timeseries_path = get_timeseries_path(model, point_name)
            if os.path.exists(timeseries_path):
                print(f"  {model}: Already exists, skipping")
                total_skipped += 1
                continue
            
            # Read and save timeseries
            print(f"  {model}: Computing timeseries...")
            ds = read_fresh_timeseries(model, point, point_name, variables_list, height_as_z_coord)
            
            if ds is not None:
                save_timeseries(ds, model, point_name)
                ds.close()
                print(f"  {model}: ✓ Success")
                total_computed += 1
            else:
                print(f"  {model}: ✗ Failed")
    
    print(f"\n{'=' * 70}")
    print(f"✓ Timeseries computation complete!")
    print(f"  Computed: {total_computed}")
    print(f"  Skipped (already exist): {total_skipped}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    # Compute and save timeseries for all points and all models
    compute_and_save_all_timeseries()
