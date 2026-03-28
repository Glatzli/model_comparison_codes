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

import fix_win_DLL_loading_issue

fix_win_DLL_loading_issue
import os
import sys

sys.path.append("D:/MSc_Arbeit/model_comparison_codes/calculations_and_plots")

import pickle
import math

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import xarray as xr
from colorspace import sequential_hcl, diverging_hcl

import confg
import read_icon_model_3D
import read_in_arome
import read_ukmo
import read_wrf_helen
from momaa_hobo_utils import load_hobo_data, load_momaa_data

# define fine contour levels
levels_thin = np.arange(0, 3500, 100)
levels_thick = np.arange(0, 3500, 500)

lon_ibk_surr_extent = (11.31, 11.46)  # coordinates for extra detailed ibk surroundings-plot
lat_ibk_surr_extent = (47.18, 47.337)


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
    # Add thin contour lines every 100m (in background)
    contours_thin = ax.contour(topo_data.lon, topo_data.lat, topo_data.values, levels=levels_thin, colors='black',
                               linewidths=0.2, transform=ccrs.PlateCarree())  # , zorder=1

    # Thick contours every 500m with optional labels (in background but above thin lines)
    contours_thick = ax.contour(topo_data.lon, topo_data.lat, topo_data.values, levels=levels_thick, colors='black',
                                linewidths=0.5, transform=ccrs.PlateCarree())  # , zorder=2

    if add_labels:
        # Contour labels also in background
        labels = ax.clabel(contours_thick, inline=True, fontsize=8,
                           fmt='%1.0f')  # Set zorder for all label texts  # for label in labels:  #     label.set_zorder(3)

    return contours_thin, contours_thick


def add_scalebar(ax, length_km=1, location='lower right'):
    """
    Add a simple black scalebar with text label to a cartopy axes.

    Args:
        ax: Cartopy axes object
        length_km: Length of scalebar in kilometers (default: 1)
        location: Location string for the scalebar (default: 'lower right')
    """
    # Get the current extent in data coordinates
    extent = ax.get_extent(crs=ccrs.PlateCarree())
    lon_min, lon_max, lat_min, lat_max = extent

    # Calculate center latitude for scalebar positioning
    center_lat = (lat_min + lat_max) / 2

    # Use the existing function to calculate longitude extent for km
    scalebar_lon_size = calculate_lon_extent_for_km(center_lat, length_km)

    # Position the scalebar
    if 'right' in location:
        scalebar_lon_start = lon_max - scalebar_lon_size - 0.01
    else:
        scalebar_lon_start = lon_min + 0.01

    if 'lower' in location:
        scalebar_lat = lat_min + 0.01
    else:
        scalebar_lat = lat_max - 0.01

    # Draw the scalebar as a white rectangle with black borderline
    from matplotlib.patches import Rectangle
    bar_height = 0.005  # height of scalebar in degrees
    rect = Rectangle((scalebar_lon_start, scalebar_lat), scalebar_lon_size, bar_height, fill=True, facecolor='white',
                     edgecolor='black', linewidth=0.8, transform=ccrs.PlateCarree(), zorder=12)
    ax.add_patch(rect)

    # Add text label
    scalebar_lon_center = scalebar_lon_start + scalebar_lon_size / 2
    ax.text(scalebar_lon_center, scalebar_lat + 0.005, f'{length_km} km', transform=ccrs.PlateCarree(), ha='center',
            va='bottom', fontsize=11)


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
    cmap = sequential_hcl("Terrain 2").cmap()

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
        ax.add_feature(cfeature.BORDERS, linewidth=1.5)

        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Add gridlines without labels
        gl = ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')

        # Add title inside the plot at the top
        ax.text(0.5, 0.98, f"{model_name}", transform=ax.transAxes, fontsize=13, fontweight='bold', ha='center',
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
        cbar.set_label('Height [m]', fontsize=13)
        cbar.ax.tick_params(labelsize=13)

    # Overall title
    fig.suptitle('Topography Comparison: All Models at 2017-10-15 14:00 UTC', fontsize=13, fontweight='bold', y=0.98)

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
                        "ICON - UM (interp.)": "ICON", "ICON - WRF (interp.)": "ICON",
                        "WRF_z - WRF_hgt": "WRF_z_unstag"}

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
        ax.text(0.5, 0.98, title_text, transform=ax.transAxes, fontsize=13, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    # Add single colorbar at the bottom center
    fig.subplots_adjust(bottom=0.12)
    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.015])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Height Difference [m]', fontsize=13)
    cbar.ax.tick_params(labelsize=13)

    # Overall title
    fig.suptitle('Topography Differences Between Models at 2017-10-15 14:00 UTC', fontsize=13, fontweight='bold',
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
        ax.text(0.5, 0.98, title_text, transform=ax.transAxes, fontsize=13, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    # Add single colorbar at the bottom center
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Height Difference [m]', fontsize=13)
    cbar.ax.tick_params(labelsize=13)

    # Overall title
    fig.suptitle('Internal Model Topography Differences at 2017-10-15 14:00 UTC', fontsize=13, fontweight='bold',
                 y=0.96)

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
        ax.text(0.5, 0.98, title_text, transform=ax.transAxes, fontsize=13, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    # Add single colorbar at the bottom center
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Height Difference [m]', fontsize=13)
    cbar.ax.tick_params(labelsize=13)

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
    cmap = sequential_hcl("Terrain 2").cmap()

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
        ax.text(0.5, 0.98, "AROME", transform=ax.transAxes, fontsize=13, fontweight='bold', ha='center', va='top',
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
        ax.text(0.5, 0.98, "WRF", transform=ax.transAxes, fontsize=13, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

        if add_points_confg:
            add_points_to_axes(ax, extent=extent)

    # Add single colorbar at the bottom center
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Height [m]', fontsize=13)
    cbar.ax.tick_params(labelsize=13)

    # Overall title
    fig.suptitle('AROME vs WRF Topography at 2017-10-15 14:00 UTC', fontsize=13, fontweight='bold', y=0.96)

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


def add_points_to_axes(ax, lon_extent=confg.lon_central_inn_extent, lat_extent=confg.lat_central_inn_extent,
        save_path="_ibk_surroundings"):
    """
    Add location markers from confg.ALL_POINTS to a map axes.
    Also adds MOMAA stations (white crosses) and Hafelekar HOBO (white square).

    Args:
        ax: Matplotlib axes with cartopy projection
        lon_extent: Tuple (lon_min, lon_max)
        lat_extent: Tuple (lat_min, lat_max) to check if points are within extent
        save_path: Path to save the figure (for checking which domain for legend)
    """
    # Unpack extent if provided
    lon_min, lon_max = lon_extent[0], lon_extent[1]
    lat_min, lat_max = lat_extent[0], lat_extent[1]

    # Initialize legend handle variables
    points = None
    momaa = None
    hobo = None
    ec_stations = None

    # plot only given points, others are not mentioned in thesis, map gets otherwise too crowded!
    selected_points = {point: confg.ALL_POINTS[point] for point in
                       ["telfs", "inzing", "ibk_uni", "ibk_airport", "hafelekar", "brenner_saddle",
                        "wipp_schoenberg_matrei", "patsch_EC_south", "volders", "jenbach", "woergl", "kufstein",
                        "rosenheim"]}
    point_marker_size = 80
    momaa_marker_size = 60
    hobo_marker_size = 100
    ec_marker_size = 120
    lidar_marker_size = 120
    for point_name, point_data in selected_points.items():  # add selected points, where timeseries are computed &
        # which are discussed in the thesis

        # Check if point is within extent
        if not ((lon_min < point_data["lon"] < lon_max) and (lat_min < point_data["lat"] < lat_max)):
            continue  # Skip points outside extent

        # Plot marker (above contours) - only set handle once for legend
        if points is None:
            points = ax.scatter(point_data["lon"], point_data["lat"], marker='x', s=point_marker_size, c="white",
                                # confg.model_colors_temp_wind['HATPRO'],
                                transform=ccrs.PlateCarree(), zorder=11, label="Points computed")
        else:
            ax.scatter(point_data["lon"], point_data["lat"], marker='x', s=point_marker_size, c="white",
                       # confg.model_colors_temp_wind['HATPRO'],
                       transform=ccrs.PlateCarree(), zorder=11)

        # Add label below the point_data with automatic adjustment (above contours)
        ax.annotate(point_data['name'], xy=(point_data["lon"], point_data["lat"]), xytext=(0, -8),
                    # Offset: 13 points below
                    textcoords='offset points', transform=ccrs.PlateCarree(), fontsize=11, ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='black', linewidth=0.5,
                              alpha=0.8), zorder=12)

    # Add MOMAA stations (white crosses)
    try:
        ds_momaa = load_momaa_data()

        for station_key in ds_momaa['STATION_KEY'].values:
            # if "central_inn" in save_path:  # skip MOMAAs for central Inn extent
            #     continue
            station_data = ds_momaa.sel(STATION_KEY=station_key)
            lat = float(station_data['lat'].values)
            lon = float(station_data['lon'].values)

            # Check if station is within extent
            if (lon_min < lon < lon_max) and (lat_min < lat < lat_max):
                # Plot white cross (above contours) - only set handle once for legend
                if momaa is None:
                    momaa = ax.scatter(lon, lat, c=confg.model_colors_temp_wind['AROME'], edgecolors='black',
                                       s=momaa_marker_size, linewidth=0.5, marker='o', zorder=10,
                                       transform=ccrs.PlateCarree(), label="MOMAAs")
                else:
                    ax.scatter(lon, lat, c=confg.model_colors_temp_wind['AROME'], edgecolors='black',
                               s=momaa_marker_size, linewidth=0.5, marker='o', zorder=10, transform=ccrs.PlateCarree())
            else:
                print("MOMAA outside of extent, not plotted")
                continue

    except Exception as e:
        print(f"Warning: Could not add MOMMA stations: {e}")

    # Add Hafelekar HOBO station (white square)
    try:
        ds_hobo = load_hobo_data()
        ds_hobo_h38 = ds_hobo.where(ds_hobo['hobo_id'] == 'H38', drop=True).squeeze()

        lat = float(ds_hobo_h38.lat.values)
        lon = float(ds_hobo_h38.lon.values)

        # Check if station is within extent
        if (lon_min < lon < lon_max) and (lat_min < lat < lat_max):
            # Plot white square (above contours: 15 for MOMAA and HOBO, even above the points from confg
            hobo = ax.scatter(lon, lat, c=confg.model_colors_temp_wind['AROME'], s=hobo_marker_size, edgecolors='black',
                              linewidth=0.5, marker='s', zorder=10, transform=ccrs.PlateCarree(), label="HOBO")

    except Exception as e:
        print(f"Warning: Could not add HOBO H38 station: {e}")

    # Add EC stations (tri-left marker with AROME color and black edgecolor)
    try:
        ec_dict = confg.ec_station_names.items()
        for ec_station in ec_dict:
            if "central_inn" in save_path:  # skip ECs for central Inn extent
                continue
            lat = ec_station[1]["lat"]
            lon = ec_station[1]["lon"]

            # Check if station is within extent
            if (lon_min < lon < lon_max) and (lat_min < lat < lat_max):
                # Plot EC station marker (tri-left, "3") - only set handle once for legend
                ec_stations = ax.scatter(lon, lat, c=confg.model_colors_temp_wind['AROME'], s=ec_marker_size,
                                         edgecolors='black', linewidth=0.5, marker='>', zorder=10,
                                         transform=ccrs.PlateCarree(),
                                         label="ECs")  # else:  #     ax.scatter(lon, lat,  # c=confg.model_colors_temp_wind['AROME'], s=ec_marker_size, linewidth=0.5,  #                marker='3', zorder=10, transform=ccrs.PlateCarree())

    except Exception as e:
        print(f"Warning: Could not add EC stations: {e}")

    if "ibk_surroundings" in save_path:
        lidar_data = xr.open_dataset(confg.lidar_sl88_merged_path)
        lidar = ax.scatter(lidar_data.lon, lidar_data.lat, c=confg.model_colors_temp_wind["AROME"], edgecolors='black',
                           s=lidar_marker_size, marker='^', linewidth=0.5, zorder=10, transform=ccrs.PlateCarree(),
                           label="Lidar")
    else:
        lidar = None

    if "wipp" in save_path:  # only for the
        legend_location = "lower left"
    else:
        legend_location = "upper right"

    # Collect only existing handles for the legend
    legend_handles = []
    # Check if each handle exists and is not None
    if points is not None:
        legend_handles.append(points)
    if momaa is not None:
        legend_handles.append(momaa)
    if hobo is not None:
        legend_handles.append(hobo)
    if ec_stations is not None:
        legend_handles.append(ec_stations)
    if lidar is not None:
        legend_handles.append(lidar)
    # Add legend for all extents- except the valley_exit-region
    if "valley_exit" not in save_path:
        ax.legend(handles=legend_handles, loc=legend_location, framealpha=0.9, facecolor="lightgray")


def plot_single_model_topography(topo_data: dict, model_key: str, save_path: str = None, add_points_confg: bool = True,
        lat_extent: tuple = None, lon_extent: tuple = None, add_ibk_surroundings_rectangle: bool = False):
    """
    Create a plot of topography from a single model.

    Args:
        topo_data: Dictionary with topography data for each model
        model_key: Key for the model to plot (e.g., 'AROME_z', 'WRF_z_unstag', 'ICON', 'UM', etc.)
        save_path: Path to save the figure
        add_points_confg: Whether to add location markers from confg.ALL_POINTS (default: True)
        lat_extent: Tuple (lat_min, lat_max) for plot extent
        lon_extent: Tuple (lon_min, lon_max) for plot extent
    """
    # Check if model exists in data
    if model_key not in topo_data:
        raise ValueError(f"Model '{model_key}' not found in topo_data. Available models: {list(topo_data.keys())}")

    # Use terrain colormap
    cmap = sequential_hcl("Terrain 2").cmap()

    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set fixed colorbar limits
    vmin = 400
    vmax = 3000

    print(f"\nTopography range: {vmin:.1f} m - {vmax:.1f} m")
    print(f"Plotting model: {model_key}")

    # Get data
    data = topo_data[model_key]
    # if "ibk_surroundings" in save_path:
    #     lon_extent = (lon_extent[0] - 0.1, lon_extent[1] + 0.1)

    data = data.sel(lat=slice(lat_extent[0] - 0.01, lat_extent[1] + 0.01),  # subset data, to have smaller .pdf-files...
                    lon=slice(lon_extent[0] - 0.01, lon_extent[1] + 0.01))

    # Create plot
    im = ax.pcolormesh(data.lon, data.lat, data.values, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(),
                       shading='auto')

    # Add contour lines
    add_contour_lines(ax, data)

    # Add features
    # ax.coastlines(resolution='10m', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=1.2)
    ax.set_extent((lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]), crs=ccrs.PlateCarree())

    # Clean up model name for title
    # model_name = model_key.replace('_z', '').replace('_unstag', '').replace('_hgt', '')
    # Add title
    # ax.text(0.5, 0.98, model_name, transform=ax.transAxes, fontsize=13, fontweight='bold', ha='center', va='top',
    #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    if add_points_confg:
        add_points_to_axes(ax, lon_extent=lon_extent, lat_extent=lat_extent, save_path=save_path)

    if add_ibk_surroundings_rectangle:
        rect = Rectangle((lon_ibk_surr_extent[0], lat_ibk_surr_extent[0]),
                         lon_ibk_surr_extent[1] - lon_ibk_surr_extent[0],
                         lat_ibk_surr_extent[1] - lat_ibk_surr_extent[0], fill=False, edgecolor="black", linestyle="-",
                         linewidth=1, transform=ccrs.PlateCarree(), zorder=12)

    # Add scalebar (1 km length)
    if ("central_inn" in save_path) or ("inn_exit" in save_path):
        add_scalebar(ax, length_km=10, location='lower right')
    else:
        add_scalebar(ax, length_km=1, location='lower right')

    # Add colorbar automatically below the plot
    # cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.08, aspect=30)
    # cbar.set_label('Height [m]', fontsize=13)
    # cbar.ax.tick_params(labelsize=13)
    # plt.grid(False)  # deactivate grid
    plt.xlabel("")
    plt.ylabel("")
    # plt.tight_layout()

    # Save figure
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {save_path}")

    return fig, ax


def plot_single_model_topography_main(model_key: str, day: int = 15, hour: int = 14, minute: int = 0,
        add_points_confg: bool = True, lat_extent: tuple = confg.lat_central_inn_extent,
        lon_extent: tuple = confg.lon_central_inn_extent, extent_name: str = "_central_inn",
        add_ibk_surroundings_rectangle: bool = False):
    """
    Main function to read data and create a single model topography plot.

    Args:
        model_key: Key for the model to plot (e.g., 'AROME_z', 'WRF_z_unstag', 'ICON', 'UM', etc.)
        day: Day of month (default: 15)
        hour: Hour of day (default: 14 for 14:00)
        minute: Minute of hour (default: 0)
        add_points_confg: Whether to add location markers from confg.ALL_POINTS (default: True)
        lat_extent: Tuple (lat_min, lat_max) for plot extent
        lon_extent: Tuple (lon_min, lon_max) for plot extent
    """
    if lat_extent is None or lon_extent is None:  # if no values are passed for lat/lon default is set:
        lat_extent = confg.lat_central_inn_extent
        lon_extent = confg.lon_central_inn_extent

    print(f"\n{'=' * 70}")
    print(f"Creating {model_key} Topography Plot")
    print(f"{'=' * 70}")
    print(f"Plot extent: lon [{lon_extent[0]:.2f}, {lon_extent[1]:.2f}], lat [{lat_extent[0]:.2f},"
          f" {lat_extent[1]:.2f}]")

    # Read or load topo data
    topo_data = check_read_topographies(day=day, hour=hour, minute=minute)

    # Create save path
    model_name = model_key.replace('_z', '').replace('_unstag', '').replace('_hgt', '').lower()
    save_path = os.path.join(confg.dir_topo_plots, f"topo_{model_name}" + extent_name + ".svg")

    # Create plot
    fig, ax = plot_single_model_topography(topo_data, model_key=model_key, save_path=save_path,
                                           add_points_confg=add_points_confg, lon_extent=lon_extent,
                                           lat_extent=lat_extent,
                                           add_ibk_surroundings_rectangle=add_ibk_surroundings_rectangle)

    print(f"{'=' * 70}\n")
    return fig, ax


def add_extent_rectangles_to_plot(ax, extent_dict: dict, zorder=8):
    """
    Add rectangles showing the outlines of different plotting extents to a topography plot.
    Rectangles have uniform style (solid black lines) with text labels for each extent.

    Args:
        ax: Matplotlib axis object to add rectangles to
        extent_dict: Dictionary with extent names and their (lon_min, lon_max, lat_min, lat_max) tuples
        zorder: Z-order for the rectangles (default: 8, above contours but below points)
    """

    # Define display names for extents
    extent_labels = {"ibk_surroundings": "Ibk surroundings", "central_inn": "central Inn", "wipp": "Wipp valley",
                     "inn_exit": "Inn exit"}

    for extent_key, (lon_min, lon_max, lat_min, lat_max) in extent_dict.items():
        if extent_key in extent_labels:
            # Create rectangle with uniform style
            rect = Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min, fill=False,
                             edgecolor=confg.model_colors_temp_wind['AROME'], linestyle="-", linewidth=2.5,
                             transform=ccrs.PlateCarree(), zorder=zorder)
            ax.add_patch(rect)

            # Calculate center of rectangle for text placement
            center_lon = (lon_min + lon_max) / 2
            if extent_key == "Ibk surroundings":
                lat_text = lat_min + 0.04
            else:
                lat_text = lat_max - 0.04

            # Add text label at center of extent
            ax.text(center_lon, lat_text, extent_labels[extent_key], fontsize=13, ha='center', va='center',
                    transform=ccrs.PlateCarree(), zorder=zorder + 1,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='black', linewidth=0.5))


def plot_full_extent_topography(model_key: str = 'AROME_z', day: int = 15, hour: int = 14, minute: int = 0,
        add_points_confg: bool = True, add_extent_rectangles: bool = True):
    """
    Plot topography for the full extent with outlines of all subplot extents.

    This function plots the full domain defined by confg.lat_full_extent and confg.lon_full_extent,
    with thick contour lines every 500m and thin lines every 200m. It optionally adds rectangles
    showing the outlines of ibk_surroundings, central_inn, wipp, and inn_exit regions.

    Args:
        model_key: Key for the model to plot (default: 'AROME_z')
        day: Day of month (default: 15)
        hour: Hour of day (default: 14 for 14:00)
        minute: Minute of hour (default: 0)
        add_points_confg: Whether to add location markers from confg.ALL_POINTS (default: True)
        add_extent_rectangles: Whether to add rectangles for subplot extents (default: True)
    """
    print(f"\n{'=' * 70}")
    print(f"Creating Full Extent Topography Plot ({model_key})")
    print(f"{'=' * 70}")

    # Define plot extent (limited to specific region)
    plot_lon_extent = (10.9, 12.7)
    plot_lat_extent = (46.9, 48.05)
    print(f"Plot extent: lon [{plot_lon_extent[0]:.2f}, {plot_lon_extent[1]:.2f}], "
          f"lat [{plot_lat_extent[0]:.2f}, {plot_lat_extent[1]:.2f}]")

    # Use terrain colormap
    cmap = sequential_hcl("Terrain 2").cmap()

    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set fixed colorbar limits
    vmin = 400
    vmax = 3000

    print(f"Topography range: {vmin:.1f} m - {vmax:.1f} m")

    # Read or load topo data
    topo_data = check_read_topographies(day=day, hour=hour, minute=minute)

    # Check if model exists in data
    if model_key not in topo_data:
        raise ValueError(f"Model '{model_key}' not found in topo_data. Available models: {list(topo_data.keys())}")

    # Get data
    data = topo_data[model_key]

    # Subset data to plot extent with small buffer
    data = data.sel(lat=slice(plot_lat_extent[0] - 0.01, plot_lat_extent[1] + 0.01),
                    lon=slice(plot_lon_extent[0] - 0.01, plot_lon_extent[1] + 0.01))

    # Create plot
    im = ax.pcolormesh(data.lon, data.lat, data.values, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(),
                       shading='auto', zorder=1)

    # Add contour lines (thin every 200m, thick every 500m as requested)
    levels_thin_full = np.arange(0, 3500, 200)
    levels_thick_full = np.arange(0, 3500, 500)
    add_contour_lines(ax, data, levels_thin=levels_thin_full, levels_thick=levels_thick_full, add_labels=True)

    # Add features
    ax.add_feature(cfeature.BORDERS, linewidth=1.2)
    ax.set_extent((plot_lon_extent[0], plot_lon_extent[1], plot_lat_extent[0], plot_lat_extent[1]),
                  crs=ccrs.PlateCarree())
    add_scalebar(ax, length_km=10, location='lower right')

    # Add gridlines
    # gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3, linestyle='--')
    # gl.top_labels = False
    # gl.right_labels = False

    # Add location markers if requested
    # if add_points_confg:
    #    add_points_to_axes(ax, lon_extent=plot_lon_extent, lat_extent=plot_lat_extent, save_path="full_extent")

    # Add rectangles for subplot extents if requested
    if add_extent_rectangles:
        extent_dict = {"ibk_surroundings": (lon_ibk_surr_extent[0], lon_ibk_surr_extent[1], lat_ibk_surr_extent[0],
                                            lat_ibk_surr_extent[1]),
                       # (confg.lon_ibk_surr_extent[0], confg.lon_ibk_surr_extent[1],
                       # confg.lat_ibk_surr_extent[0], confg.lat_ibk_surr_extent[1]),
                       "central_inn": (confg.lon_central_inn_extent[0], confg.lon_central_inn_extent[1],
                                       confg.lat_central_inn_extent[0], confg.lat_central_inn_extent[1]),
                       "wipp": (confg.lon_wipp_extent[0], confg.lon_wipp_extent[1], confg.lat_wipp_extent[0],
                                confg.lat_wipp_extent[1]),
                       "inn_exit": (confg.lon_inn_exit_extent[0], confg.lon_inn_exit_extent[1],
                                    confg.lat_inn_exit_extent[0], confg.lat_inn_exit_extent[1])}
        add_extent_rectangles_to_plot(ax, extent_dict, zorder=8)

    # Add colorbar automatically below the plot
    # cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.1, aspect=40)
    # cbar.set_label('Height [m]', fontsize=13)
    # cbar.ax.tick_params(labelsize=13)

    # Remove axis labels and ticks
    plt.xticks([])
    plt.yticks([])
    ax.set_xlabel("")
    ax.set_xticklabels([])
    ax.set_ylabel("")
    ax.set_yticklabels([])

    # Create save path
    model_name = model_key.replace('_z', '').replace('_unstag', '').replace('_hgt', '').lower()
    save_path = os.path.join(confg.dir_topo_plots, f"topo_{model_name}_full_extent.svg")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\n✓ Figure saved to: {save_path}")

    print(f"{'=' * 70}\n")

    return fig, ax


if __name__ == "__main__":
    # Choose which plot(s) to create:

    # 1. Create topography comparison plot for ALL models (5 models in grid)
    # plot_topography_comparison_main(day=15, hour=14, minute=0, add_points_confg=True)

    # 2. Create topography difference plots
    # plot_topography_differences_main(day=15, hour=14, minute=0)

    # 3. Create AROME vs WRF topography comparison only (big side-by-side plot)
    # plot_arome_wrf_topography_main(day=15, hour=14, minute=0, add_points_confg=True)

    # 4. Plot single model topography using main function
    # plot_single_model_topography_main(model_key='AROME_z', day=15, hour=14, minute=0, add_points_confg=True)
    # plot_single_model_topography_main(model_key='ICON', day=15, hour=14, minute=0, add_points_confg=True)
    # plot_single_model_topography_main(model_key='UM', day=15, hour=14, minute=0, add_points_confg=True)
    # plot_single_model_topography_main(model_key='WRF_z_unstag', day=15, hour=14, minute=0, add_points_confg=True)

    # plot areas of interest Topography plots:
    # _central_inn _ibk_surroundings _wipp_valley _valley_exit _ziller_valley
    plot_single_model_topography_main(model_key='AROME_hgt', add_points_confg=True, lon_extent=lon_ibk_surr_extent,
                                      lat_extent=lat_ibk_surr_extent,  # confg.lon_ibk_surr_extent
                                      # confg.lat_ibk_surr_extent
                                      extent_name="_ibk_surroundings")

    plot_single_model_topography_main(model_key='AROME_hgt', add_points_confg=True,
                                      lon_extent=confg.lon_central_inn_extent, lat_extent=confg.lat_central_inn_extent,
                                      extent_name="_central_inn", add_ibk_surroundings_rectangle=True)

    plot_single_model_topography_main(model_key='AROME_hgt', add_points_confg=True, lon_extent=confg.lon_wipp_extent,
                                      lat_extent=confg.lat_wipp_extent, extent_name="_wipp")
    plot_single_model_topography_main(model_key='AROME_hgt', add_points_confg=True,
                                      lon_extent=confg.lon_inn_exit_extent, lat_extent=confg.lat_inn_exit_extent,
                                      extent_name="_inn_exit")

    # Plot full extent with extent rectangles
    plot_full_extent_topography(model_key='AROME_z', day=15, hour=14, minute=0, add_points_confg=False,
                                add_extent_rectangles=True)
    plt.show()