"""
calc_cap_height
CAP height calculation over a full lat/lon region.

This mirrors the point-based logic used elsewhere: for each time and grid cell,
find the first (bottom-up) height where dT_dz is negative for 3 consecutive levels.
If dT_dz is not present, it will be computed from temp via differentiate over 'height'.

Output: adds a DataArray 'cap_height' with dims (time, lat, lon) to the dataset.
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def _ensure_dT_dz(ds: xr.Dataset) -> xr.Dataset:
    """Ensure the dataset has dT_dz; compute from temp if missing.
    Requires a 'height' coordinate and 'temp' variable for fallback.
    """
    if "dT_dz" in ds:
        return ds
    if "temp" not in ds:
        raise ValueError("Dataset must contain 'dT_dz' or 'temp' to compute dT_dz.")

    # If 'height' is a 1D coordinate (only depends on 'height'), we can use xarray.differentiate
    height = ds["height"]
    if set(height.dims) == {"height"}:
        return ds.assign(dT_dz=ds["temp"].differentiate(coord="height"))

    # Otherwise height varies per column (e.g. dims (height, lat, lon)).
    # We compute dT/dz column-wise using np.gradient via xr.apply_ufunc.

    def _grad_1d(t: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute dT/dz for 1D arrays t and z.
        - returns array of same length as input, with NaN where insufficient data.
        - handles NaNs in t/z and non-monotonic (descending) z by flipping.
        """
        # ensure float and 1d
        t = np.asarray(t, dtype=float)
        z = np.asarray(z, dtype=float)
        out = np.full_like(t, np.nan, dtype=float)

        valid = ~np.isnan(t) & ~np.isnan(z)
        if valid.sum() < 2:
            return out

        t_v = t[valid]
        z_v = z[valid]

        # If z is descending, flip before gradient
        flipped = False
        if z_v[0] > z_v[-1]:
            z_v = z_v[::-1]
            t_v = t_v[::-1]
            flipped = True

        # If any duplicate z lead to zero spacing, gradient will be inf/nan; guard it
        dz = np.diff(z_v)
        if np.allclose(dz, 0):
            return out

        g = np.gradient(t_v, z_v)
        if flipped:
            g = g[::-1]

        out[valid] = g
        return out

    # apply_ufunc: operate along 'height' core dimension, vectorized over lat/lon and time
    dT_dz = xr.apply_ufunc(
        _grad_1d,
        ds["temp"],
        ds["height"],
        input_core_dims=[["height"], ["height"]],
        output_core_dims=[["height"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[ds["temp"].dtype],
    )

    return ds.assign(dT_dz=dT_dz)


def cap_height_region(ds: xr.Dataset, consecutive: int = 3) -> xr.Dataset:
    """Compute CAP height per (time, lat, lon) grid cell.

    Contract:
    - Input: xr.Dataset with dims including 'time', 'height' and spatial dims 'lat', 'lon'.
      Must contain 'dT_dz' (time,height,lat,lon) or 'temp' to derive it.
      Must contain 'z' (AROME, UM) or 'z_unstag' (ICON, WRF) for geopotential height.
    - Logic: for each time and grid cell, find the first (lowest) geopotential height where dT_dz < 0
      holds for `consecutive` consecutive levels (default 3).
    - Output: returns a new dataset with an added variable 'cap_height' of shape (time, lat, lon)
      containing the geopotential height in meters.
    """
    # Make sure we can compute/use dT_dz
    ds = _ensure_dT_dz(ds)

    # EDITED: Detect which geopotential height variable is present
    if "z_unstag" in ds:
        z_var_name = "z_unstag"
    elif "z" in ds:
        z_var_name = "z"
    else:
        raise ValueError("Dataset must contain 'z' or 'z_unstag' for geopotential height")

    z_geopot = ds[z_var_name]  # This is the actual geopotential height we want to return

    # EDITED: No sorting - work with the data as-is (model levels may be top-down or bottom-up)
    # We'll use the geopotential height z to determine "first/lowest"

    # Boolean mask where dT_dz is negative; NaNs become False
    neg = (ds["dT_dz"] < 0) & ds["dT_dz"].notnull()

    # EDITED: Use rolling window along 'height' dimension to find consecutive negatives
    # rolling(height=consecutive).sum() counts how many negatives in each window
    # We want windows where ALL consecutive values are negative
    neg_count = neg.rolling(height=consecutive, min_periods=consecutive).sum()
    # mask where we have 'consecutive' negative values in a row
    neg_consecutive = (neg_count == consecutive)

    # EDITED: For each column, find the LOWEST geopotential height where condition holds
    # Mask z_geopot where the condition is true
    z_masked = z_geopot.where(neg_consecutive)

    # The CAP height is the minimum geopotential height along the height dimension
    # (minimum because we want the lowest altitude where inversion starts)
    cap_height = z_masked.min(dim="height", skipna=True)
    cap_height.name = "cap_height"
    cap_height.attrs["description"] = (
        f"lowest geopotential height where dT_dz < 0 for {consecutive} consecutive levels"
    )
    cap_height.attrs["units"] = z_geopot.attrs.get("units", "m")

    # Attach to dataset and return
    return ds.assign(cap_height=cap_height)


def cap_height_region_da(ds: xr.Dataset, consecutive: int = 3) -> xr.DataArray:
    """Like cap_height_region, but return only the cap_height DataArray (time, lat, lon)."""
    out = cap_height_region(ds, consecutive=consecutive)
    return out["cap_height"]
