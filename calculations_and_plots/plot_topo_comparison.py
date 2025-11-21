"""
Compare topography values from all models at a fixed time.

This script reads topography data from all models (AROME, ICON, ICON2TE, UM, WRF)
at the 4th timestamp (14:00 on the first day) and plots them for comparison.

For each model, the lowest model level height is extracted:
- AROME: z (lowest level) and hgt variable
- ICON: z_unstag (lowest level)
- ICON2TE: z_unstag (lowest level)
- UM: z (lowest level)
- WRF: z_unstag (lowest level) and hgt variable

All plots are shown over the domain defined in confg (Hafelekar extent):
lat: 47.0 - 47.6
lon: 11.1 - 12.1
"""
from __future__ import annotations

import os
import pickle

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from colorspace import sequential_hcl, diverging_hcl

import confg
import read_icon_model_3D
import read_in_arome
import read_ukmo
import read_wrf_helen


def check_read_topographies(day, hour, minute):
    """
    Check if topography data is already saved, if not read from models and save it.
    :param day:
    :param hour:
    :param minute:
    :return:
    """
    # Check if topography data is already saved
    if os.path.exists(confg.all_model_topographies):
        print(f"\n✓ Topography data already exists at: {confg.all_model_topographies}")
        print("Loading saved data...")
        with open(confg.all_model_topographies, 'rb') as f:  # read dict with datasets in it
            topo_data = pickle.load(f)
        return topo_data
    
    else:
        print("extract topography data from models first...")
        # Read all model topographies
        topo_data = read_all_model_topographies(day=day, hour=hour, minute=minute)
        
        with open(confg.all_model_topographies, 'wb') as f:
            pickle.dump(topo_data, f)
        return topo_data


def read_all_model_topographies(day: int = 15, hour: int = 14, minute: int = 0):
    """
    Read topography data from all models at a fixed time.
    
    Args:
        day: Day of month (default: 15 for 2017-10-15)
        hour: Hour of day (default: 14 for 14:00)
        minute: Minute of hour (default: 0)
    
    Returns:
        Dictionary containing topography data for each model
    """
    topo_data = {}
    
    print(f"\nReading topography data at 2017-10-{day:02d} {hour:02d}:{minute:02d}...")
    
    # AROME: z and hgt on lowest level
    try:
        print("  Reading AROME...")
        ds_arome = read_in_arome.read_in_arome_fixed_time(day=day, hour=hour, min=minute, variables=["z", "hgt"],
                                                          min_lat=confg.lat_min, max_lat=confg.lat_max,
                                                          min_lon=confg.lon_min, max_lon=confg.lon_max)
        
        # Get z on lowest level (height1 is always lowest, but not necessary index 1...)
        topo_data["AROME_z"] = ds_arome["z"].sel(height=1)  # lowest level
        
        topo_data["AROME_hgt"] = ds_arome["hgt"]
        print("    ✓ AROME loaded")
    except Exception as e:
        print(f"    ✗ Error loading AROME: {e}")
    
    # ICON: z_unstag on lowest level
    try:
        print("  Reading ICON...")
        ds_icon = read_icon_model_3D.read_icon_fixed_time(day=day, hour=hour, min=minute, variant="ICON",
                                                          variables=["z", "z_unstag"])
        # Get lowest level of z_unstag
        topo_data["ICON"] = ds_icon["z_unstag"].sel(height=1)
        
        print("    ✓ ICON loaded")
    except Exception as e:
        print(f"    ✗ Error loading ICON: {e}")
    
    # ICON2TE: z_unstag on lowest level
    try:
        print("  Reading ICON2TE...")
        ds_icon2te = read_icon_model_3D.read_icon_fixed_time(day=day, hour=hour, min=minute, variant="ICON2TE",
                                                             variables=["z", "z_unstag"])
        # Get lowest level of z_unstag
        topo_data["ICON2TE"] = ds_icon2te["z_unstag"].sel(height=1)
        
        print("    ✓ ICON2TE loaded")
    except Exception as e:
        print(f"    ✗ Error loading ICON2TE: {e}")
    
    # UM: z on lowest level
    try:
        print("  Reading UM...")
        ds_um = read_ukmo.read_ukmo_fixed_time(day=day, hour=hour, min=minute, variables=["z"])
        # Get lowest level of z
        topo_data["UM"] = ds_um["z"].sel(height=1)
        
        print("    ✓ UM loaded")
    except Exception as e:
        print(f"    ✗ Error loading UM: {e}")
    
    # WRF: z_unstag and hgt on lowest level
    try:
        print("  Reading WRF...")
        ds_wrf = read_wrf_helen.read_wrf_fixed_time(day=day, hour=hour, min=minute, variables=["z", "z_unstag", "hgt"])
        
        # Get lowest level of z_unstag
        topo_data["WRF_z_unstag"] = ds_wrf["z_unstag"].sel(height=1)
        topo_data["WRF_hgt"] = ds_wrf["hgt"].isel(time=0)
        print("    ✓ WRF loaded")
    except Exception as e:
        print(f"    ✗ Error loading WRF: {e}")
    
    return topo_data


def plot_topography_comparison(topo_data: dict, save_path: str = None):
    """
    Create a comparison plot of topography from all models.
    
    Args:
        topo_data: Dictionary with topography data for each model
        save_path: Path to save the figure
    """
    # Filter out AROME_hgt and WRF_hgt - only show _z values (hgt & z vals are compared in extra plot...)
    filtered_topo_data = {k: v for k, v in topo_data.items()
                          if k not in ["AROME_hgt", "WRF_hgt"]}
    
    # Define plot extent from confg (Hafelekar extent)
    
    # Count number of plots
    n_plots = len(filtered_topo_data)
    
    # Create figure with subplots (3 columns, 2 rows)
    n_cols = 3
    n_rows = 2
    
    # Use terrain colormap
    cmap = sequential_hcl("Terrain").cmap()
    
    # Create figure with space for colorbar at bottom (wider figure)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Flatten axes array for easier iteration
    axes = np.atleast_1d(axes).flatten()
    
    # Set fixed colorbar limits
    vmin = 400
    vmax = 3000
    
    print(f"\nTopography range: {vmin:.1f} m - {vmax:.1f} m")
    
    # Plot each model
    for idx, (model_name, data) in enumerate(filtered_topo_data.items()):
        ax = axes[idx]
        
        # Select data within extent
        # data_subset = data.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        
        # Plot
        im = ax.pcolormesh(data.lon, data.lat, data.values, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading='auto')
        
        # Add contour lines
        # Thin contours every 100m
        contours_thin = ax.contour(data.lon, data.lat, data.values, levels=np.arange(0, 3500, 100), colors='black',
                                   linewidths=0.5, alpha=0.3, transform=ccrs.PlateCarree())
        
        # Thick contours every 500m with labels
        contours_thick = ax.contour(data.lon, data.lat, data.values, levels=np.arange(0, 3500, 500), colors='black',
                                    linewidths=1, alpha=0.5, transform=ccrs.PlateCarree())
        ax.clabel(contours_thick, inline=True, fontsize=8, fmt='%1.0f')
        
        # Add features
        ax.coastlines(resolution='10m', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        
        # Set extent
        ax.set_extent([confg.lon_hf_min, confg.lon_hf_max, confg.lat_hf_min, confg.lat_hf_max], crs=ccrs.PlateCarree())
        
        # Add gridlines without labels
        gl = ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')
        
        # Add title inside the plot at the top
        ax.text(0.5, 0.98, f"{model_name}", transform=ax.transAxes, fontsize=12, fontweight='bold', ha='center',
                va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    # Add single colorbar at the bottom center
    fig.subplots_adjust(bottom=0.12)
    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.015])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Height [m]', fontsize=12)
    
    # Overall title
    fig.suptitle('Topography Comparison: All Models at 2017-10-15 14:00 UTC', fontsize=14, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {save_path}")
    
    return fig, axes


def plot_topography_comparison_main(day: int = 15, hour: int = 14, minute: int = 0):
    """
    Main function to read data and create topography comparison plot.
    
    Args:
        day: Day of month (default: 15)
        hour: Hour of day (default: 14 for 14:00)
        minute: Minute of hour (default: 0)
    """
    
    topo_data = check_read_topographies(day=day, hour=hour, minute=minute)
    
    # Create save path
    save_path = os.path.join(confg.dir_topo_plots, "topo_comparison.png")
    
    # Create plot
    fig, axes = plot_topography_comparison(topo_data, save_path=save_path)
    
    # Show plot
    plt.tight_layout()



def calculate_topography_differences(topo_data: dict):
    """
    Calculate differences between model topographies.
    
    Computes the following differences:
    - ICON - UM
    - ICON - WRF_z_unstag
    - WRF_z_unstag - WRF_hgt
    - AROME_z - AROME_hgt
    - ICON - AROME_z
    - ICON - ICON2TE
    
    Args:
        topo_data: Dictionary with topography data for each model
    
    Returns:
        Dictionary containing difference fields
    """
    diff_data = {}
    
    print("\nCalculating topography differences...")
    
    # AROME_z - AROME_hgt
    if "AROME_z" in topo_data and "AROME_hgt" in topo_data:
        diff_data["AROME_z - AROME_hgt"] = topo_data["AROME_z"] - topo_data["AROME_hgt"]
        print("  ✓ AROME_z - AROME_hgt")
    
    # ICON - ICON2TE
    if "ICON" in topo_data and "ICON2TE" in topo_data:
        # Interpolate ICON2TE to ICON grid
        diff_data["ICON - ICON2TE"] = topo_data["ICON"] - topo_data["ICON2TE"]
        print("  ✓ ICON - ICON2TE")
    
    # ICON - AROME_z
    if "ICON" in topo_data and "AROME_z" in topo_data:
        # Interpolate AROME to ICON grid linearly
        arome_interp = topo_data["AROME_z"].interp(lat=topo_data["ICON"].lat, lon=topo_data["ICON"].lon)
        diff_data["ICON - AROME (interp.)"] = topo_data["ICON"] - arome_interp
        print("  ✓ ICON - AROME")
    
    # ICON - UM
    if "ICON" in topo_data and "UM" in topo_data:
        # Interpolate UM to ICON grid
        um_interp = topo_data["UM"].interp(lat=topo_data["ICON"].lat, lon=topo_data["ICON"].lon)
        diff_data["ICON - UM (interp.)"] = topo_data["ICON"] - um_interp
        print("  ✓ ICON - UM")
    
    # ICON - WRF_z_unstag
    if "ICON" in topo_data and "WRF_z_unstag" in topo_data:
        # Interpolate WRF to ICON grid
        wrf_interp = topo_data["WRF_z_unstag"].interp(lat=topo_data["ICON"].lat, lon=topo_data["ICON"].lon)
        diff_data["ICON - WRF (interp.)"] = topo_data["ICON"] - wrf_interp
        print("  ✓ ICON - WRF")
    
    # WRF_z_unstag - WRF_hgt
    if "WRF_z_unstag" in topo_data and "WRF_hgt" in topo_data:
        diff_data["WRF_z - WRF_hgt"] = topo_data["WRF_z_unstag"] - topo_data["WRF_hgt"]
        print("  ✓ WRF_z - WRF_hgt")

    return diff_data


def plot_topography_differences(diff_data: dict, topo_data: dict, save_path: str = None):
    """
    Create a comparison plot of topography differences.
    
    Args:
        diff_data: Dictionary with difference fields
        topo_data: Dictionary with original topography data (for contour lines)
        save_path: Path to save the figure (optional)
    """
    # Count number of plots
    n_plots = len(diff_data)
    
    # Create figure with subplots (3 columns, 2 rows)
    n_cols = 3
    n_rows = 2
    
    # Use diverging colormap centered at 0 (blue for positive, red for negative)
    # Using Blue-Red 2 for a lighter palette
    from colorspace import diverging_hcl
    cmap = diverging_hcl("Blue-Red 2", l=[30, 90], c=80).cmap()
    
    # Create figure with space for colorbar at bottom
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Flatten axes array for easier iteration
    axes = np.atleast_1d(axes).flatten()
    
    # Find max absolute difference for symmetric colorbar
    max_abs_diff = 0
    for data in diff_data.values():
        max_abs_diff = max(max_abs_diff, np.abs(data.values).max())
    
    # Round to nearest 50 for nice numbers
    vmax = 12 # np.ceil(max_abs_diff / 50) * 50
    vmin = -vmax
    
    print(f"\nDifference range: {vmin:.1f} m to {vmax:.1f} m")
    
    # Mapping for which topography to use for contours
    contour_topo_map = {
        "AROME_z - AROME_hgt": "AROME_z",
        "ICON - AROME (interp.)": "ICON",
        "ICON - UM (interp.)": "ICON",
        "ICON - WRF (interp.)": "ICON",
        "WRF_z - WRF_hgt": "WRF_z_unstag"
    }
    
    # Plot each difference
    for idx, (diff_name, data) in enumerate(diff_data.items()):
        ax = axes[idx]
        
        # Plot difference
        im = ax.pcolormesh(data.lon, data.lat, data.values, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading='auto')
        
        # Get the corresponding topography for contour lines
        topo_key = contour_topo_map.get(diff_name)
        if topo_key and topo_key in topo_data:
            topo_for_contours = topo_data[topo_key]
            
            # shouldn't be needed?
            # Interpolate topography to difference data grid if needed
            # if not (topo_for_contours.lat.shape == data.lat.shape and
            #         topo_for_contours.lon.shape == data.lon.shape):
            #     topo_for_contours = topo_for_contours.interp(lat=data.lat, lon=data.lon)
            
            # Add contour lines: thin contours every 100m
            contours_thin = ax.contour(topo_for_contours.lon, topo_for_contours.lat,
                                      topo_for_contours.values,
                                      levels=np.arange(0, 3500, 100),
                                      colors='black', linewidths=0.5, alpha=0.5,
                                      transform=ccrs.PlateCarree())
            
            # Thick contours every 500m with labels
            contours_thick = ax.contour(topo_for_contours.lon, topo_for_contours.lat,
                                       topo_for_contours.values,
                                       levels=np.arange(0, 3500, 500),
                                       colors='black', linewidths=1,
                                       transform=ccrs.PlateCarree())
            ax.clabel(contours_thick, inline=True, fontsize=8, fmt='%1.0f')
        
        # Add features
        ax.coastlines(resolution='10m', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        
        # Set extent
        ax.set_extent([confg.lon_hf_min, confg.lon_hf_max, confg.lat_hf_min, confg.lat_hf_max], crs=ccrs.PlateCarree())
        
        # Add gridlines without labels
        gl = ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')
        
        # Calculate statistics, only from not NaN vals
        mean_diff = float(data.values[~np.isnan(data.values)].mean())
        std_diff = float(data.values[~np.isnan(data.values)].std())
        
        # Add title inside the plot at the top with statistics
        title_text = f"{diff_name}\nMean: {mean_diff:.1f} m, Std: {std_diff:.1f} m"
        ax.text(0.5, 0.98, title_text, transform=ax.transAxes, fontsize=10, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    # Add single colorbar at the bottom center
    fig.subplots_adjust(bottom=0.12)
    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.015])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Height Difference [m]', fontsize=12)
    
    # Overall title
    fig.suptitle('Topography Differences Between Models at 2017-10-15 14:00 UTC', fontsize=14, fontweight='bold',
                 y=0.98)
    
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {save_path}")
    
    return fig, axes


def plot_internal_model_differences(diff_data: dict, topo_data: dict, save_path: str = None):
    """
    Create a comparison plot for internal model topography differences (z vs hgt) and ICON vs ICON2TE.
    Plots AROME_z - AROME_hgt, WRF_z - WRF_hgt side by side.
    
    Args:
        diff_data: Dictionary with difference fields
        topo_data: Dictionary with original topography data (for contour lines)
        save_path: Path to save the figure
    """
    # dict for internal model differences (hgt) and ICON-ICON2TE (again key and then data...)
    internal_diffs = {k: v for k, v in diff_data.items() if k in ['AROME_z - AROME_hgt', 'WRF_z - WRF_hgt']}

    
    if not internal_diffs:
        print("\n✗ No internal model differences found.")
        return None, None
    
    n_plots = len(internal_diffs)
    
    # Use diverging colormap centered at 0
    cmap = diverging_hcl("Blue-Red 2", l=[30, 90], c=80).cmap()
    
    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 2, figsize=(18, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Flatten axes array
    axes = np.atleast_1d(axes).flatten()
    
    # Fixed colorbar limits for internal differences
    vmin = -12
    vmax = 12
    
    print(f"\nInternal model difference range: {vmin:.1f} m to {vmax:.1f} m")
    
    # Mapping for which topography to use for contours
    contour_topo_map = {
        "AROME_z - AROME_hgt": "AROME_z",
        "WRF_z - WRF_hgt": "WRF_z_unstag",
        "ICON - ICON2TE": "ICON"
    }
    
    # Plot each difference
    for idx, (diff_name, data) in enumerate(internal_diffs.items()):
        ax = axes[idx]
        
        # Plot difference
        im = ax.pcolormesh(data.lon, data.lat, data.values, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading='auto')
        
        # Get the corresponding topography for contour lines
        topo_key = contour_topo_map.get(diff_name)
        if topo_key and topo_key in topo_data:
            topo_for_contours = topo_data[topo_key]
            
            # Add contour lines: thin contours every 100m
            contours_thin = ax.contour(topo_for_contours.lon, topo_for_contours.lat,
                                      topo_for_contours.values,
                                      levels=np.arange(0, 3500, 100),
                                      colors='black', linewidths=0.5, alpha=0.5,
                                      transform=ccrs.PlateCarree())
            
            # Thick contours every 500m with labels
            contours_thick = ax.contour(topo_for_contours.lon, topo_for_contours.lat,
                                       topo_for_contours.values,
                                       levels=np.arange(0, 3500, 500),
                                       colors='black', linewidths=1,
                                       transform=ccrs.PlateCarree())
            ax.clabel(contours_thick, inline=True, fontsize=8, fmt='%1.0f')
        
        # Add features
        ax.coastlines(resolution='10m', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        
        # Set extent
        ax.set_extent([confg.lon_hf_min, confg.lon_hf_max, confg.lat_hf_min, confg.lat_hf_max],
                     crs=ccrs.PlateCarree())
        
        # Add gridlines without labels
        gl = ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')
        
        # Calculate statistics
        mean_diff = float(data.values[~np.isnan(data.values)].mean())
        std_diff = float(data.values[~np.isnan(data.values)].std())
        
        # Add title inside the plot at the top with statistics
        title_text = f"{diff_name}\nMean: {mean_diff:.1f} m, Std: {std_diff:.1f} m"
        ax.text(0.5, 0.98, title_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
               ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Add single colorbar at the bottom center
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Height Difference [m]', fontsize=12)
    
    # Overall title
    # fig.suptitle('Internal Model Topography Differences at 2017-10-15 14:00 UTC',
    #             fontsize=13, fontweight='bold', y=0.96)
    
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {save_path}")
    
    return fig, axes


def plot_model_to_model_differences(diff_data: dict, topo_data: dict, save_path: str = None):
    """
    Create a comparison plot for model-to-model topography differences.
    Plots ICON vs other models (AROME, UM, WRF).
    
    Args:
        diff_data: Dictionary with difference fields
        topo_data: Dictionary with original topography data (for contour lines)
        save_path: Path to save the figure (optional)
    """
    from colorspace import diverging_hcl
    
    # Filter for model-to-model differences (exclude internal hgt differences)
    model_diffs = {k: v for k, v in diff_data.items() if "hgt" not in k}
    
    if not model_diffs:
        print("\n✗ No model-to-model differences found.")
        return None, None
    
    n_plots = len(model_diffs)
    
    # Use diverging colormap centered at 0
    cmap = diverging_hcl("Blue-Red 2", l=[30, 90], c=80).cmap()
    
    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Flatten axes array
    axes = np.atleast_1d(axes).flatten()
    
    # Fixed colorbar limits for model-to-model differences
    vmin = -50
    vmax = 50
    
    print(f"\nModel-to-model difference range: {vmin:.1f} m to {vmax:.1f} m")
    
    # Mapping for which topography to use for contours
    contour_topo_map = {
        "ICON - AROME (interp.)": "ICON",
        "ICON - UM (interp.)": "ICON",
        "ICON - WRF (interp.)": "ICON"
    }
    
    # Plot each difference
    for idx, (diff_name, data) in enumerate(model_diffs.items()):
        ax = axes[idx - 1]
        
        # Plot difference
        im = ax.pcolormesh(data.lon, data.lat, data.values, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading='auto')
        
        # Get the corresponding topography for contour lines
        topo_key = contour_topo_map.get(diff_name)
        if topo_key and topo_key in topo_data:
            topo_for_contours = topo_data[topo_key]
            
            # Add contour lines: thin contours every 100m
            contours_thin = ax.contour(topo_for_contours.lon, topo_for_contours.lat,
                                      topo_for_contours.values,
                                      levels=np.arange(0, 3500, 100),
                                      colors='black', linewidths=0.5, alpha=0.5,
                                      transform=ccrs.PlateCarree())
            
            # Thick contours every 500m with labels
            contours_thick = ax.contour(topo_for_contours.lon, topo_for_contours.lat,
                                       topo_for_contours.values,
                                       levels=np.arange(0, 3500, 500),
                                       colors='black', linewidths=1,
                                       transform=ccrs.PlateCarree())
            ax.clabel(contours_thick, inline=True, fontsize=8, fmt='%1.0f')
        
        # Add features
        ax.coastlines(resolution='10m', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        
        # Set extent
        ax.set_extent([confg.lon_hf_min, confg.lon_hf_max, confg.lat_hf_min, confg.lat_hf_max],
                     crs=ccrs.PlateCarree())
        
        # Add gridlines without labels
        gl = ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')
        
        # Calculate statistics
        mean_diff = float(data.values[~np.isnan(data.values)].mean())
        std_diff = float(data.values[~np.isnan(data.values)].std())
        
        # Add title inside the plot at the top with statistics
        title_text = f"{diff_name}\nMean: {mean_diff:.1f} m, Std: {std_diff:.1f} m"
        ax.text(0.5, 0.98, title_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
               ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    # Add single colorbar at the bottom center
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Height Difference [m]', fontsize=12)
    
    # Overall title
    fig.suptitle('Model-to-Model Topography Differences at 2017-10-15 14:00 UTC',
                fontsize=13, fontweight='bold', y=0.96)
    
    # Save figure
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {save_path}")
    
    return fig, axes


def plot_topography_differences_main(day: int = 15, hour: int = 14, minute: int = 0):
    """
    Main function to read data and create topography difference plots.
    Creates two separate plots:
    1. Internal model differences (z - hgt) with colorbar -12 to 12
    2. Model-to-model differences with colorbar -50 to 50
    
    Args:
        day: Day of month (default: 15)
        hour: Hour of day (default: 14 for 14:00)
        minute: Minute of hour (default: 0)
    """
    topo_data = check_read_topographies(day=day, hour=hour, minute=minute)
    
    # Calculate differences
    diff_data = calculate_topography_differences(topo_data)
    
    if not diff_data:
        print("\n✗ No differences calculated. Cannot create plot.")
        return
    
    # Create save paths
    plot_dir = os.path.join(confg.dir_PLOTS, "topography_comparison")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot 1: Internal model differences (z - hgt)
    save_path_internal = os.path.join(plot_dir,
                                     f"topo_differences_internal_2017-10-{day:02d}_{hour:02d}{minute:02d}.png")
    fig1, axes1 = plot_internal_model_differences(diff_data, topo_data, save_path=save_path_internal)
    
    # Plot 2: Model-to-model differences
    save_path_models = os.path.join(plot_dir,
                                   f"topo_differences_models_2017-10-{day:02d}_{hour:02d}{minute:02d}.png")
    
    fig2, axes2 = plot_model_to_model_differences(diff_data, topo_data, save_path=save_path_models)
    
    # Show plots
    plt.tight_layout()



if __name__ == "__main__":
    # Create topography comparison plot for 14:00 on 2017-10-15 (4th timestamp)
    plot_topography_comparison_main(day=15, hour=14, minute=0)
    
    # Create topography difference plots for 14:00 on 2017-10-15
    plot_topography_differences_main(day=15, hour=14, minute=0)
    plt.show()