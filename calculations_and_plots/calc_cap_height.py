"""
calc_cap_height
CAP height calculation over a full lat/lon region and for 1D vertical profiles.

This mirrors the point-based logic used elsewhere: for each time and grid cell,
find the first (bottom-up) height where dT is negative for 3 consecutive levels.
If dT is not present, it will be computed from temp via diff over 'height'.

Output: adds a DataArray 'cap_height' with dims (time, lat, lon) to the dataset.
For 1D profiles (observations), computes cap_height for single vertical profiles.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import confg


def calc_dT(ds: xr.Dataset) -> xr.Dataset:
    """Ensure the dataset has dT; compute from temp if missing.
    Requires a 'height' dimension and 'temp' variable for fallback.
    
    Computes the temperature difference T[i+1] - T[i] between consecutive height levels.
    This is used to find where temperature decreases with height for consecutive levels.
    """
    if "dT" in ds:
        return ds
    if "temp" not in ds:
        raise ValueError("Dataset must contain 'dT' or 'temp' to compute dT.")
    
    # Compute temperature difference between consecutive height levels
    # dT = T[i+1] - T[i]
    # We want to find where dT < 0 (temperature decreases with height)
    dT = ds["temp"].diff(dim="height", n=1, label="lower")
    
    # Set attributes
    dT.attrs["long_name"] = "Temperature difference between consecutive height levels"
    dT.attrs["units"] = ds['temp'].attrs.get('units', 'K')
    
    return ds.assign(dT=dT)


def find_consecutive_negative_mask(dT: xr.DataArray, consecutive: int = 3) -> xr.DataArray:
    """
    PROBLEM: doesn't work! looked at 12 UTC till now, and probably on most upper lvls not at gorud level...
    => debug again!
    
    Find locations where dT is negative for consecutive levels.
    
    This is the core logic used by all cap_height functions to identify
    end of inversion layers (where temperature decreases with height).
    
    Args:
        dT: DataArray with temperature differences, must have 'height' dimension
        consecutive: Number of consecutive negative values required (default: 3)
    
    Returns:
        Boolean DataArray with same shape as dT, True where the condition holds
    """
    # Boolean mask where dT is negative; NaNs become False
    neg = (dT > 0) & dT.notnull()
    
    # Use rolling window along 'height' dimension to find consecutive negatives
    # rolling(height=consecutive).sum() counts how many negatives in each window
    # We want windows where ALL consecutive values are negative
    neg_count = neg.rolling(height=consecutive, min_periods=consecutive).sum()
    
    # Mask where we have 'consecutive' negative values in a row
    neg_consecutive = (neg_count == consecutive)

    # Shift the True values down by 'consecutive' positions to mark the FIRST (lowest) level
    # of each sequence where the condition holds
    neg_consecutive_shifted = neg_consecutive.shift(height=-consecutive, fill_value=False)
    
    return neg_consecutive_shifted


def cap_height_region(ds: xr.Dataset, consecutive: int = 3) -> xr.Dataset:
    """Compute CAP height per (time, lat, lon) grid cell.

    Contract:
    - Input: xr.Dataset with dims including 'time', 'height' and spatial dims 'lat', 'lon'.
      Must contain 'dT' (time,height,lat,lon) or 'temp' to derive it.
      Must contain 'z' (AROME, UM) or 'z_unstag' (ICON, WRF) for geopotential height.
    - Logic: for each time and grid cell, find the first (lowest) geopotential height where dT < 0
      holds for `consecutive` consecutive levels (default 3).
    - Output: returns a new dataset with an added variable 'cap_height' of shape (time, lat, lon)
      containing the geopotential height in meters.
    """
    # Detect which geopotential height variable is present
    if "z_unstag" in ds:
        z_var_name = "z_unstag"
    elif "z" in ds:
        z_var_name = "z"
    else:
        raise ValueError("Dataset must contain 'z' or 'z_unstag' for geopotential height")
    
    z_geopot = ds[z_var_name]  # This is the actual geopotential height we want to return
    
    # Find where dT is negative for consecutive levels (common logic)
    neg_consecutive = find_consecutive_negative_mask(ds["dT"], consecutive=consecutive)
    
    # For each column, find the LOWEST geopotential height where condition holds
    # Mask z_geopot where the condition is true
    z_masked = z_geopot.where(neg_consecutive)
    
    # The CAP height is the minimum geopotential height along the height dimension
    # (minimum because we want the lowest altitude where inversion ends)
    cap_height = z_masked.min(dim="height", skipna=True)
    cap_height.name = "cap_height"
    cap_height.attrs["description"] = (
        f"lowest geopotential height where dT < 0 for {consecutive} consecutive levels")
    cap_height.attrs["units"] = z_geopot.attrs.get("units", "m")
    
    # Attach to dataset and return
    return ds.assign(cap_height=cap_height)


def cap_height_region_da(ds: xr.Dataset, consecutive: int = 3) -> xr.DataArray:
    """Like cap_height_region, but return only the cap_height DataArray (time, lat, lon)."""
    out = cap_height_region(ds, consecutive=consecutive)
    return out["cap_height"]


def cap_height_profile(ds: xr.Dataset, consecutive: int = 3) -> xr.Dataset:
    """
    Compute CAP height for 1D vertical profiles (e.g., radiosonde, HATPRO observations).
    
    This function is designed for observational data where we have a single vertical profile
    or a time series of profiles without spatial dimensions (lat, lon).
    
    Args:
        ds: xr.Dataset with 'height' dimension and 'temp' variable
            May have 'time' dimension for time series of profiles
            Must contain either 'dT' or 'temp' (will compute dT from temp)
            Height coordinate should be the actual geopotential height in meters
        consecutive: number of consecutive levels with dT < 0 required (default: 3)
    
    Returns:
        Dataset with added 'cap_height' variable
        - If input has no time dim: cap_height is a scalar
        - If input has time dim: cap_height has shape (time,)
    """
    # Make sure we can compute/use dT
    ds = calc_dT(ds)
    
    # For observations, height coordinate IS the geopotential height
    # (not a separate variable like z or z_unstag)
    if "height" not in ds.coords:
        raise ValueError("Dataset must have 'height' as a coordinate")
    
    height_coord = ds.coords["height"]
    
    # Find where dT is negative for consecutive levels (common logic)
    neg_consecutive = find_consecutive_negative_mask(ds["dT"], consecutive=consecutive)
    
    # For each profile (or timestep), find the LOWEST height where condition holds
    # Mask height where the condition is true
    height_masked = height_coord.where(neg_consecutive)
    
    # The CAP height is the minimum height along the height dimension
    cap_height = height_masked.min(dim="height", skipna=True)
    cap_height.name = "cap_height"
    cap_height.attrs["description"] = (
        f"lowest height where dT < 0 for {consecutive} consecutive levels"
    )
    cap_height.attrs["units"] = height_coord.attrs.get("units", "m")
    
    # Attach to dataset and return
    return ds.assign(cap_height=cap_height)


def cap_height_profile_da(ds: xr.Dataset, consecutive: int = 3) -> xr.DataArray:
    """
    Like cap_height_profile, but return only the cap_height DataArray.
    
    Returns:
        - Scalar if input has no time dimension
        - DataArray with shape (time,) if input has time dimension
    """
    out = cap_height_profile(ds, consecutive=consecutive)
    return out["cap_height"]
