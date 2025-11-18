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

import plotly.graph_objects as go
import xarray as xr

# Model-specific shift because indexed height of CAP is a bit different (and height coords are differently ordered
# f.e.ICON & WRF: ICON needs shift by 1 up, WRF by 3 down, etc.
shift_map = {'AROME': 0, 'ICON': 1, 'ICON2TE': 1, 'WRF': -3, 'UM': -3, 'HATPRO': -3, "radiosonde": 3}


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
    dT = ds["temp"].differentiate(coord="height")  # former: .diff(dim="height", n=3, label="lower")
    
    # Set attributes
    dT.attrs["long_name"] = "Temperature difference between consecutive height levels"
    dT.attrs["units"] = ds['temp'].attrs.get('units', 'K')
    
    return ds.assign(dT=dT)


def find_consecutive_negative_mask(dT: xr.DataArray, consecutive: int = 3, model: str = None) -> xr.DataArray:
    """
    Find locations where dT is negative for consecutive levels.
    
    This is the core logic used by all cap_height functions to identify
    end of inversion layers (where temperature decreases with height).
    
    Args:
        dT: DataArray with temperature differences, must have 'height' dimension
        consecutive: Number of consecutive negative values required (default: 3)
        model: Model name ('ICON', 'ICON2TE', 'WRF', 'AROME', 'UM') for model-specific shift
    
    Returns:
        Boolean DataArray with same shape as dT, True where the condition holds
    """
    # Boolean mask where dT is negative; NaNs become False
    neg = (dT < 0) & dT.notnull()
    
    # Use rolling window along 'height' dimension to find consecutive negatives
    neg_count = neg.rolling(height=consecutive, min_periods=consecutive).sum()
    
    # Mask where we have 'consecutive' negative values in a row
    neg_consecutive = (neg_count == consecutive)
    
    neg_consecutive_shifted = neg_consecutive.shift(height=shift_map[model], fill_value=False)
    # because height coords are not completely uniform, we need to shift ICON by 1 up, WRF by 3 down, ...
    # former: neg_consecutive_shifted = neg_consecutive.shift(height=1, fill_value=False)
    # shift_amount = shift_map.get(model, 1) if model else 1
    
    return neg_consecutive_shifted


def cap_height_region(ds: xr.Dataset, consecutive: int = 3, model: str = None) -> xr.Dataset:
    """Compute CAP height per (time, lat, lon) grid cell.

    Args:
        ds: Input dataset with 'dT' or 'temp', and 'z' or 'z_unstag'
        consecutive: Number of consecutive negative dT levels required (default: 3)
        model: Model name ('ICON', 'ICON2TE', 'WRF', 'AROME', 'UM') for model-specific shift
    
    Returns:
        Dataset with added 'cap_height' variable (time, lat, lon)
    """
    # Detect which geopotential height variable is present
    if "z_unstag" in ds:
        z_var_name = "z_unstag"
    elif "z" in ds:
        z_var_name = "z"
    else:
        raise ValueError("Dataset must contain 'z' or 'z_unstag' for geopotential height")
    
    z_geopot = ds[z_var_name]
    
    # Find where dT is negative for consecutive levels with model-specific shift
    neg_consecutive = find_consecutive_negative_mask(ds["dT"], consecutive=consecutive, model=model)
    
    # For each column, find the LOWEST geopotential height where condition holds
    z_masked = z_geopot.where(neg_consecutive)
    cap_height = z_masked.min(dim="height", skipna=True)
    cap_height.name = "cap_height"
    cap_height.attrs["description"] = f"lowest geopotential height where dT < 0 for {consecutive} consecutive levels"
    cap_height.attrs["units"] = z_geopot.attrs.get("units", "m")
    
    return ds.assign(cap_height=cap_height)


def cap_height_region_da(ds: xr.Dataset, consecutive: int = 3, model: str = None) -> xr.DataArray:
    """Like cap_height_region, but return only the cap_height DataArray (time, lat, lon)."""
    out = cap_height_region(ds, consecutive=consecutive, model=model)
    return out["cap_height"]


def cap_height_profile(ds: xr.Dataset, consecutive: int = 3, model: str = None) -> xr.Dataset:
    """
    Compute CAP height for 1D vertical profiles (e.g., radiosonde, HATPRO observations).
    
    Args:
        ds: xr.Dataset with 'height' dimension and 'temp' variable
        consecutive: number of consecutive levels with dT < 0 required (default: 3)
        model: Model name ('ICON', 'ICON2TE', 'WRF', 'AROME', 'UM') for model-specific shift
    
    Returns:
        Dataset with added 'cap_height' variable
    """
    ds = calc_dT(ds)
    
    if "height" not in ds.coords:
        raise ValueError("Dataset must have 'height' as a coordinate")
    
    height_coord = ds.coords["height"]
    
    # Find where dT is negative for consecutive levels with model-specific shift
    neg_consecutive = find_consecutive_negative_mask(ds["dT"], consecutive=consecutive, model=model)
    
    # For each profile (or timestep), find the LOWEST height where condition holds
    height_masked = height_coord.where(neg_consecutive)
    cap_height = height_masked.min(dim="height", skipna=True)
    cap_height.name = "cap_height"
    cap_height.attrs["description"] = f"lowest height where dT/dz < 0 for {consecutive} consecutive levels"
    cap_height.attrs["units"] = height_coord.attrs.get("units", "m")
    
    return ds.assign(cap_height=cap_height)


def test_plot(ds, cap_height, timeidx: int = 24):
    """
    creates a test-plot of the temp profile with dT and cap_height in it as interactive plotly figure.
    :param ds:
    :param cap_height:
    :param timeidx:
    :return:
    """
    import numpy as np
    
    fig = go.Figure()
    
    # Get data for the selected time
    temp_profile = ds.isel(time=timeidx).temp.values
    dt_profile = ds.isel(time=timeidx).dT.values
    height_profile = ds.isel(time=timeidx).height.values
    cap_h = float(cap_height.isel(time=timeidx).values)
    
    # Temperature profile
    fig.add_trace(go.Scatter(x=temp_profile, y=height_profile, mode="lines", name="Temperature"))
    
    # dT profile (shifted by +18)
    fig.add_trace(go.Scatter(x=dt_profile + 18, y=height_profile, mode="lines", name="dT + 18"))
    
    # CAP height marker
    nearest_idx = np.argmin(np.abs(height_profile - cap_h))
    temp_at_cap = temp_profile[nearest_idx]
    fig.add_trace(go.Scatter(x=[temp_at_cap], y=[cap_h], mode="markers", name="CAP Height"))
    
    fig.update_layout(xaxis_title="Temperature (°C)", yaxis_title="Height (m)", xaxis_range=[15, 20],
                      yaxis_range=[500, 1500])
    fig.show(renderer='browser')


def cap_height_profile_da(ds: xr.Dataset, consecutive: int = 3, model: str = None) -> xr.DataArray:
    """
    Like cap_height_profile, but return only the cap_height DataArray.
    
    Returns:
        - Scalar if input has no time dimension
        - DataArray with shape (time,) if input has time dimension
    """
    out = cap_height_profile(ds, consecutive=consecutive, model=model)
    return out["cap_height"]
