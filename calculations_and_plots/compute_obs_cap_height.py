"""""
compute_obs_cap_height.py

Compute and save CAP height for observational data (Radiosonde and HATPRO).
This script calculates the CAP height using the same algorithm as for model data:
finding the lowest height where dT/dz < 0 for 3 consecutive levels.

Output files are saved to: {data_folder}/calculated_cap_height/
"""

from __future__ import annotations

import os
import sys
import xarray as xr

# Add parent directory to path to import confg
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import confg
from calculations_and_plots.calc_cap_height import cap_height_profile


def compute_radiosonde_cap_height() -> None:
    """
    Compute CAP height for radiosonde profile and save to NetCDF.
    
    The radiosonde data is a single vertical profile (no time dimension).
    Output: radiosonde_cap_height.nc with a single scalar cap_height value.
    """
    print("\n" + "="*70)
    print("Computing CAP height for Radiosonde...")
    print("="*70)
    
    # Load radiosonde data
    radiosonde_path = confg.radiosonde_smoothed
    print(f"Loading radiosonde data from: {radiosonde_path}")
    
    if not os.path.exists(radiosonde_path):
        print(f"Warning: Radiosonde file not found: {radiosonde_path}")
        return
    
    ds_radiosonde = xr.open_dataset(radiosonde_path)
    print(f"Radiosonde data loaded. Dimensions: {dict(ds_radiosonde.dims)}")
    
    # Compute CAP height
    print("Computing CAP height (3 consecutive levels with dT/dz < 0)...")
    ds_with_cap = cap_height_profile(ds_radiosonde, consecutive=3)
    
    cap_height_value = ds_with_cap["cap_height"].item()
    print(f"✓ Radiosonde CAP height: {cap_height_value:.1f} m")
    
    # Save to NetCDF
    output_dir = os.path.join(confg.data_folder, "calculated_cap_height")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "radiosonde_cap_height.nc")
    
    # Save only the cap_height as a dataset
    cap_ds = ds_with_cap[["cap_height"]].copy()
    cap_ds.attrs["description"] = "CAP height computed from radiosonde profile"
    cap_ds.attrs["source_file"] = radiosonde_path
    cap_ds.attrs["computation"] = "Lowest height where dT/dz < 0 for 3 consecutive levels"
    
    cap_ds.to_netcdf(output_path)
    print(f"✓ Saved to: {output_path}")
    
    ds_radiosonde.close()


def compute_hatpro_cap_height() -> None:
    """
    Compute CAP height for HATPRO profiles and save to NetCDF.
    
    HATPRO data has a time dimension, so we compute CAP height for each timestep.
    Output: hatpro_cap_height.nc with cap_height(time).
    """
    print("\n" + "="*70)
    print("Computing CAP height for HATPRO...")
    print("="*70)
    
    # Load HATPRO data
    hatpro_path = r"D:\MSc_Arbeit\data\Observations\HATPRO_obs\hatpro_interpolated_arome_height_as_z.nc"
    print(f"Loading HATPRO data from: {hatpro_path}")
    
    if not os.path.exists(hatpro_path):
        print(f"Warning: HATPRO file not found: {hatpro_path}")
        return
    
    ds_hatpro = xr.open_dataset(hatpro_path)
    print(f"HATPRO data loaded. Dimensions: {dict(ds_hatpro.dims)}")
    
    # Compute CAP height for all timesteps
    print("Computing CAP height for all timesteps (3 consecutive levels with dT/dz < 0)...")
    ds_with_cap = cap_height_profile(ds_hatpro, consecutive=3)
    
    # Print statistics
    cap_height_da = ds_with_cap["cap_height"]
    print(f"✓ HATPRO CAP height computed for {len(cap_height_da)} timesteps")
    print(f"  Mean CAP height: {float(cap_height_da.mean()):.1f} m")
    print(f"  Min CAP height:  {float(cap_height_da.min()):.1f} m")
    print(f"  Max CAP height:  {float(cap_height_da.max()):.1f} m")
    print(f"  Valid values:    {int((~cap_height_da.isnull()).sum())} / {len(cap_height_da)}")
    
    # Save to NetCDF
    output_dir = os.path.join(confg.data_folder, "calculated_cap_height")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "hatpro_cap_height.nc")
    
    # Save only the cap_height as a dataset
    cap_ds = ds_with_cap[["cap_height"]].copy()
    cap_ds.attrs["description"] = "CAP height computed from HATPRO temperature profiles"
    cap_ds.attrs["source_file"] = hatpro_path
    cap_ds.attrs["computation"] = "Lowest height where dT/dz < 0 for 3 consecutive levels"
    
    cap_ds.to_netcdf(output_path)
    print(f"✓ Saved to: {output_path}")
    
    ds_hatpro.close()


def main() -> None:
    """
    Main function to compute CAP heights for all observational datasets.
    """
    print("\n" + "="*70)
    print("COMPUTING CAP HEIGHT FOR OBSERVATIONAL DATA")
    print("="*70)
    
    # Compute for radiosonde
    compute_radiosonde_cap_height()
    
    # Compute for HATPRO
    compute_hatpro_cap_height()
    
    print("\n" + "="*70)
    print("✓ All observational CAP heights computed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
