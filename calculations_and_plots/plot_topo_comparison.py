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

# Fix for OpenMP duplicate library error on Windows
import os
import sys

sys.path.append("D:/MSc_Arbeit/model_comparison_codes/calculations_and_plots")

import pickle
import math

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

# define fine contour levels
levels_thin = np.arange(0, 3500, 100)
levels_thick = np.arange(0, 3500, 500)


def calculate_lon_extent_for_km(latitude, km):
    """
    Berechnet die Ausdehnung in Grad Längengrad für eine gegebene Entfernung in Kilometern.
    by ChatGPT...

    Parameters:
    latitude (float): Die gegebene Breite in Grad.
    km (float): Die Entfernung in Kilometern.

    Returns:
    float: Die Ausdehnung in Grad Längengrad.
    """
    # Radius der Erde in Kilometern
    earth_radius = 6371

    # Erdumfang in Kilometern
    earth_circumference = 2 * math.pi * earth_radius

    # Länge eines Längengrads in Kilometern an der gegebenen Breite
    lon_km = math.cos(math.radians(latitude)) * earth_circumference / 360

    # Ausdehnung in Grad Längengrad für die gegebene Entfernung
    lon_extent = km / lon_km
    return lon_extent


def calculate_km_for_lon_extent(latitude, lon_extent_deg):
    """
    Berechnet die Entfernung in km für eine gegebene Längendifferenz (in Grad) an einer bestimmten Breite.
    """
    earth_radius = 6371  # km
    earth_circumference = 2 * math.pi * earth_radius
    lon_km = math.cos(math.radians(latitude)) * earth_circumference / 360
    return lon_extent_deg * lon_km

def add_contour_lines(ax, topo_data, levels_thin=levels_thin, levels_thick=levels_thick, add_labels=True):
    """
    Add topography contour lines to a plot.

    Args:
        ax: Matplotlib axis object to add contours to
        topo_data: Topography data with .lon, .lat, and .values attributes
        levels_thin: Array of levels for thin contour lines (default: every 100m)
        levels_thick: Array of levels for thick contour lines (default: every 500m)
        add_labels: Whether to add labels to thick contours (default: True)

    Returns:
        Tuple of (contours_thin, contours_thick)
    """
    # Add thin contour lines every 100m
    contours_thin = ax.contour(topo_data.lon, topo_data.lat, topo_data.values, levels=levels_thin, colors='black',
                               linewidths=0.5, alpha=0.5, transform=ccrs.PlateCarree())

    # Thick contours every 500m with optional labels
    contours_thick = ax.contour(topo_data.lon, topo_data.lat, topo_data.values, levels=levels_thick, colors='black',
                                linewidths=1, transform=ccrs.PlateCarree())

    if add_labels:
        ax.clabel(contours_thick, inline=True, fontsize=8, fmt='%1.0f')

    return contours_thin, contours_thick


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
        print("  Reading AROME...")  # AROME has still large extent, therefore limit lat/lon to values of other models
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


def plot_topography_comparison(topo_data: dict, save_path: str = None, add_points_confg: bool = True,
        extent: tuple = None):
    """
    Create a comparison plot of topography from all models.
    
    Args:
        topo_data: Dictionary with topography data for each model
        save_path: Path to save the figure
        add_points_confg: Whether to add location markers from confg.ALL_POINTS (default: True)
        extent: Tuple (lon_min, lon_max, lat_min, lat_max) for plot extent
    """
    # Filter out AROME_hgt and WRF_hgt - only show _z values (hgt & z vals are compared in extra plot...)
    filtered_topo_data = {k: v for k, v in topo_data.items() if k not in ["AROME_hgt", "ICON2TE", "WRF_hgt"]}

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
    # if n_rows * n_cols == 1:
    #     axes = [axes]  # Make single axis into a list
    # else:
    axes = axes.flatten()

    # Set fixed colorbar limits
    vmin = 400
    vmax = 3000

    print(f"\nTopography range: {vmin:.1f} m - {vmax:.1f} m")

    # Plot each model
    im = None  # Initialize im variable for colorbar
    for idx, (model_name, data) in enumerate(filtered_topo_data.items()):
        ax = axes[idx]

        # Plot
        im = ax.pcolormesh(data.lon, data.lat, data.values, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading='auto')
        # Add contour lines
        add_contour_lines(ax, data)

        # Add features
        ax.coastlines(resolution='10m', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=1)

        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Add gridlines without labels
        gl = ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')

        # Add title inside the plot at the top
        ax.text(0.5, 0.98, f"{model_name}", transform=ax.transAxes, fontsize=12, fontweight='bold', ha='center',
                va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

        if add_points_confg:
            # Add points from confg.ALL_POINTS with extent checking
            add_points_to_axes(ax, extent=extent)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    # Add single colorbar at the bottom center (OUTSIDE the loop)
    if im is not None:
        fig.subplots_adjust(bottom=0.12)
        cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.015])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Height [m]', fontsize=12)

    # Overall title
    fig.suptitle('Topography Comparison: All Models at 2017-10-15 14:00 UTC', fontsize=14, fontweight='bold', y=0.98)

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {save_path}")

    return fig, axes


def plot_topography_comparison_main(day: int = 15, hour: int = 14, minute: int = 0, add_points_confg: bool = True):
    """
    Main function to read data and create topography comparison plot.

    Args:
        day: Day of month (default: 15)
        hour: Hour of day (default: 14 for 14:00)
        minute: Minute of hour (default: 0)
        add_points_confg: Whether to add location markers from confg.ALL_POINTS (default: True)
    """
    # Define plot extent (lon_min, lon_max, lat_min, lat_max) - Hafelekar extent
    plot_extent = (confg.lon_hf_min, confg.lon_hf_max, confg.lat_hf_min, confg.lat_hf_max)

    print(
        f"\nPlot extent: lon [{plot_extent[0]:.2f}, {plot_extent[1]:.2f}], lat [{plot_extent[2]:.2f}, {plot_extent[3]:.2f}]")
    # read or load topo data newly, is saved in AROME-folder
    topo_data = check_read_topographies(day=day, hour=hour, minute=minute)

    # Create save path
    save_path = os.path.join(confg.dir_topo_plots, "topo_comparison.png")

    # Create plot with points and extent
    fig, axes = plot_topography_comparison(topo_data, save_path=save_path, add_points_confg=add_points_confg,
                                           extent=plot_extent)  # plt.tight_layout()  #


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
    if n_rows * n_cols == 1:
        axes = [axes]  # Make single axis into a list
    else:
        axes = axes.flatten()

    # Find max absolute difference for symmetric colorbar
    max_abs_diff = 0
    for data in diff_data.values():
        max_abs_diff = max(max_abs_diff, np.abs(data.values).max())

    # Round to nearest 50 for nice numbers
    vmax = 12  # np.ceil(max_abs_diff / 50) * 50
    vmin = -vmax

    print(f"\nDifference range: {vmin:.1f} m to {vmax:.1f} m")

    # Mapping for which topography to use for contours
    contour_topo_map = {"AROME_z - AROME_hgt": "AROME_z", "ICON - AROME (interp.)": "ICON",
        "ICON - UM (interp.)": "ICON", "ICON - WRF (interp.)": "ICON", "WRF_z - WRF_hgt": "WRF_z_unstag"}

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
            # Add contour lines
            add_contour_lines(ax, topo_for_contours)

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
    contour_topo_map = {"AROME_z - AROME_hgt": "AROME_z", "WRF_z - WRF_hgt": "WRF_z_unstag", "ICON - ICON2TE": "ICON"}

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
            # Add contour lines
            add_contour_lines(ax, topo_for_contours)

        # Add features
        ax.coastlines(resolution='10m', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=1)

        # Set extent
        ax.set_extent([confg.lon_hf_min, confg.lon_hf_max, confg.lat_hf_min, confg.lat_hf_max], crs=ccrs.PlateCarree())

        # Add gridlines without labels
        gl = ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')

        # Calculate statistics
        mean_diff = float(data.values[~np.isnan(data.values)].mean())
        std_diff = float(data.values[~np.isnan(data.values)].std())

        # Add title inside the plot at the top with statistics
        title_text = f"{diff_name}\nMean: {mean_diff:.1f} m, Std: {std_diff:.1f} m"
        ax.text(0.5, 0.98, title_text, transform=ax.transAxes, fontsize=11, fontweight='bold', ha='center', va='top',
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
    contour_topo_map = {"ICON - AROME (interp.)": "ICON", "ICON - UM (interp.)": "ICON", "ICON - WRF (interp.)": "ICON"}

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
            # Add contour lines
            add_contour_lines(ax, topo_for_contours)

        # Add features
        ax.coastlines(resolution='10m', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=1)

        # Set extent
        ax.set_extent([confg.lon_hf_min, confg.lon_hf_max, confg.lat_hf_min, confg.lat_hf_max], crs=ccrs.PlateCarree())

        # Add gridlines without labels
        gl = ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')

        # Calculate statistics
        mean_diff = float(data.values[~np.isnan(data.values)].mean())
        std_diff = float(data.values[~np.isnan(data.values)].std())

        # Add title inside the plot at the top with statistics
        title_text = f"{diff_name}\nMean: {mean_diff:.1f} m, Std: {std_diff:.1f} m"
        ax.text(0.5, 0.98, title_text, transform=ax.transAxes, fontsize=11, fontweight='bold', ha='center', va='top',
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
    fig.suptitle('Model-to-Model Topography Differences at 2017-10-15 14:00 UTC', fontsize=13, fontweight='bold',
                 y=0.96)

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
    save_path_models = os.path.join(plot_dir, f"topo_differences_models_2017-10-{day:02d}_{hour:02d}{minute:02d}.png")

    fig2, axes2 = plot_model_to_model_differences(diff_data, topo_data, save_path=save_path_models)

    # Show plots
    plt.tight_layout()


def plot_arome_wrf_topography_only(topo_data: dict, save_path: str = None, add_points_confg: bool = True,
        extent: tuple = None):
    """
    Create a side-by-side plot of AROME and WRF topography only.

    Args:
        topo_data: Dictionary with topography data for each model
        save_path: Path to save the figure
        add_points_confg: Whether to add location markers from confg.ALL_POINTS (default: True)
        extent: Tuple (lon_min, lon_max, lat_min, lat_max) for plot extent
    """
    # Use terrain colormap
    cmap = sequential_hcl("Terrain").cmap()

    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set fixed colorbar limits
    vmin = 400
    vmax = 3000

    print(f"\nTopography range: {vmin:.1f} m - {vmax:.1f} m")

    # Plot AROME
    if "AROME_z" in topo_data:
        ax = axes[0]
        data = topo_data["AROME_z"]

        im = ax.pcolormesh(data.lon, data.lat, data.values, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading='auto')

        # Add contour lines
        add_contour_lines(ax, data)

        # Add features
        ax.coastlines(resolution='10m', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=1)

        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        # Add title
        ax.text(0.5, 0.98, "AROME", transform=ax.transAxes, fontsize=14, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

        if add_points_confg:
            add_points_to_axes(ax, extent=extent)

    # Plot WRF
    if "WRF_z_unstag" in topo_data:
        ax = axes[1]
        data = topo_data["WRF_z_unstag"]

        im = ax.pcolormesh(data.lon, data.lat, data.values, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(), shading='auto')

        # Add contour lines
        add_contour_lines(ax, data)

        # Add features
        ax.coastlines(resolution='10m', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=1)

        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        # Add title
        ax.text(0.5, 0.98, "WRF", transform=ax.transAxes, fontsize=14, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

        if add_points_confg:
            add_points_to_axes(ax, extent=extent)

    # Add single colorbar at the bottom center
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Height [m]', fontsize=12)

    # Overall title
    fig.suptitle('AROME vs WRF Topography at 2017-10-15 14:00 UTC', fontsize=16, fontweight='bold', y=0.96)

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {save_path}")

    return fig, axes


def plot_arome_wrf_topography_main(day: int = 15, hour: int = 14, minute: int = 0, add_points_confg: bool = True):
    """
    Main function to read data and create AROME vs WRF topography plot.

    Args:
        day: Day of month (default: 15)
        hour: Hour of day (default: 14 for 14:00)
        minute: Minute of hour (default: 0)
        add_points_confg: Whether to add location markers from confg.ALL_POINTS (default: True)
    """
    # Define plot extent (lon_min, lon_max, lat_min, lat_max) - Hafelekar extent
    plot_extent = (confg.lon_hf_min, confg.lon_hf_max, confg.lat_hf_min, confg.lat_hf_max)

    print(f"\n{'=' * 70}")
    print(f"Creating AROME vs WRF Topography Plot")
    print(f"{'=' * 70}")
    print(
        f"Plot extent: lon [{plot_extent[0]:.2f}, {plot_extent[1]:.2f}], lat [{plot_extent[2]:.2f}, {plot_extent[3]:.2f}]")

    # read or load topo data
    topo_data = check_read_topographies(day=day, hour=hour, minute=minute)

    # Create save path
    save_path = os.path.join(confg.dir_topo_plots, "topo_arome_wrf_comparison.png")

    # Create plot
    fig, axes = plot_arome_wrf_topography_only(topo_data, save_path=save_path, add_points_confg=add_points_confg,
                                               extent=plot_extent)

    print(f"{'=' * 70}\n")
    return fig, axes


def add_points_to_axes(ax, extent=None):
    """
    Add location markers from confg.ALL_POINTS to a map axes.

    Args:
        ax: Matplotlib axes with cartopy projection
        extent: Tuple (lon_min, lon_max, lat_min, lat_max) to check if points are within extent
    """
    if extent is None:  # define default here so that it's not mutable
        extent = [confg.lon_hf_min, confg.lon_hf_max, confg.lat_hf_min, confg.lat_hf_max]

    # Unpack extent if provided
    if extent:
        lon_min, lon_max, lat_min, lat_max = extent

    # Plot each point from ALL_POINTS dictionary
    for point_name, point_data in confg.ALL_POINTS.items():
        # Check if point is within extent
        if extent:
            if not ((lon_min <= point_data["lon"] <= lon_max) and (lat_min <= point_data["lat"] <= lat_max)):
                continue  # Skip points outside extent

        # Plot marker
        ax.plot(point_data["lon"], point_data["lat"], marker='o', markersize=6, markerfacecolor='red',
                markeredgecolor='white', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=10)

        # Add label below the point_data with automatic adjustment
        ax.annotate(point_data['name'], xy=(point_data["lon"], point_data["lat"]), xytext=(0, -10),
                    # Offset: 10 points below
                    textcoords='offset points', transform=ccrs.PlateCarree(), fontsize=8, fontweight='bold',
                    ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5),
                    zorder=11)


if __name__ == "__main__":
    # Choose which plot(s) to create:

    # 1. Create topography comparison plot for ALL models (5 models in grid)
    # plot_topography_comparison_main(day=15, hour=14, minute=0, add_points_confg=True)

    # 2. Create topography difference plots
    # plot_topography_differences_main(day=15, hour=14, minute=0)

    # 3. Create AROME vs WRF topography comparison only (big side-by-side plot)
    plot_arome_wrf_topography_main(day=15, hour=14, minute=0, add_points_confg=True)

    plt.show()