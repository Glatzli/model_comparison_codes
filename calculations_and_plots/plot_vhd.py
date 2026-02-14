"""
This script should plot the timeseries of the VHD for all models and the change of spatial extent of a defined threshold
in time.
An hourly small multiple plot of the VHD over the full valley extent was done, with a contour line at 80% of the
maximum VHD, to show the extent of the maximum in each timestep.

For the VHD-point calculation, the "direct" height coordinate is used, which sets the geopot. height directly as height coord.
This has reasons in the past: If I would have changed it to use "above_terrain", I would have needed to rewrite
the calc_vhd_single_point(ds_point, model="AROME")-function in calc_vhd.py, espc. the indices...
=> therefore just saved the timeseries twice, once with "direct" for the VHD calc and once with the "above_terrain"
height coord.

"""
import datetime

import fix_win_DLL_loading_issue

fix_win_DLL_loading_issue
import importlib
import os

import cartopy.crs as ccrs
import xarray as xr
import pandas as pd

import confg
from calculations_and_plots.calc_vhd import calc_vhd_single_point
from calculations_and_plots.manage_timeseries import load_or_read_timeseries
from read_in_hatpro_radiosonde import read_radiosonde_dataset, read_hatpro_dataset

importlib.reload(confg)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from colorspace import sequential_hcl


def calc_vhd_using_timeseries(point_name: str):
    """
    Calculate VHD for all models using the new manage_timeseries system.
    Uses the "above_terrain" feature for consistent height coordinates.

    :param point_name: Name of the point (key in confg.get_valley_points_only())
    :return: Dictionary with VHD datasets for each model
    """
    point = confg.get_valley_points_only()[point_name]
    models = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]
    vhd_results = {}

    for model in models:
        print(f"Loading/calculating VHD for {model} at {point['name']}...")

        # Load timeseries using the new system
        ds = load_or_read_timeseries(model=model, point=point, point_name=point_name, height_as_z_coord="direct")
        # use direct geopot height as z coord for VHD calculation

        if ds is not None:
            # Calculate VHD using the existing function
            vhd = calc_vhd_single_point(ds, model=model)
            vhd_results[model] = vhd
            ds.close()
        else:
            print(f"  Warning: Could not load timeseries for {model}")
            vhd_results[model] = None

    # Add observational data for Innsbruck points
    if point_name.startswith("ibk"):
        print("Adding observational data (HATPRO)...")
        hatpro = read_hatpro_dataset(height_as_z_coord="direct")
        vhd_results["HATPRO"] = calc_vhd_single_point(hatpro, model="HATPRO")  # check again

        radio = read_radiosonde_dataset(height_as_z_coord="direct")
        vhd_results["radio"] = calc_vhd_single_point(radio, model="radio")

    return vhd_results


def plot_vhds_point(vhd_results: dict, point_name: str, vhd_origin: str = "new_timeseries"):
    """
    Plot VHD timeseries for all models using the saved timeseries data with the direct height coordinate

    :param vhd_results: Dictionary with VHD datasets for each model
    :param point_name: Name of the point location
    :param vhd_origin: Description of calculation method
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot model data
    models_to_plot = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]
    for model in models_to_plot:
        if model in vhd_results and vhd_results[model] is not None:
            vhd = vhd_results[model]
            (vhd.vhd / 10 ** 6).plot(ax=ax, label=model, color=confg.model_colors_temp_wind[model], linewidth=2)

    # Plot observational data if available
    if "HATPRO" in vhd_results and vhd_results["HATPRO"] is not None:
        (vhd_results["HATPRO"].vhd / 10 ** 6).plot(ax=ax, label="HATPRO (uni)",
                                                   color=confg.model_colors_temp_wind["HATPRO"], linewidth=2,
                                                   linestyle="dotted")
    # Plot observational data if available
    if "radio" in vhd_results and vhd_results["radio"] is not None:
        ax.scatter(datetime.datetime(2017, 10, 16, 2, 15, 0), vhd_results["radio"].vhd / 10 ** 6,
                   label="Radiosonde (Airport)", color=confg.model_colors_temp_wind["Radiosonde"], s=100, marker="*",
                   zorder=10)
    # plt.ylim(2, 21)
    plt.ylabel(r"valley heat deficit $[\frac{MJ}{m^2}]$")
    plt.grid()
    plt.title("")
    # plt.title(f"VHD timeline for {point_name} via {vhd_origin}")
    plt.legend(loc='upper left')

    # Create output directory if it doesn't exist
    output_dir = os.path.join(confg.dir_PLOTS, "vhd_plots")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"vhd_model_comp_{point_name.replace(" ", "_")}_{vhd_origin}.pdf"))


def plot_vhd_small_multiples(ds_extent, model="ICON", lat_extent: tuple = confg.lat_cap_height_extent,
        lon_extent: tuple = confg.lon_cap_height_extent, extent_name: str = "cap_height"):
    """
    plots VHD as hourly data with small multiples, need revision because I now calced it 1/2 hourly
    :param ds_extent: VHD dataset with time, lat, lon coordinates
    :param model: Model name (AROME, ICON, ICON2TE, UM, WRF)
    :param lat_extent: Tuple (lat_min, lat_max) for plot extent
    :param lon_extent: Tuple (lon_min, lon_max) for plot extent
    :param extent_name: Name of extent for saving purposes (e.g., _"cap_height")
    :return:
    """
    projection = ccrs.Mercator()
    ds_extent_subset = ds_extent.sel(time=pd.date_range("2017-10-15 14:00:00", periods=12, freq="2h"))
    nplots, ncols = len(ds_extent_subset.time), 3
    nrows = int((nplots + ncols - 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6), layout="compressed", subplot_kw={'projection': projection})
    # norm = mpl.colors.Normalize(vmin=0, vmax=0.5)  # normalize the colorbar
    axes = axes.flatten()
    # levels = np.linspace(0.05, 0.36, 10)  # try without distinct levels
    for i, time in enumerate(ds_extent_subset.time):
        ax = axes[i]
        ds_extent_sel = (ds_extent_subset.sel(time=time) / 1e6).sel(
            lat=slice(lat_extent[0] - 0.01, lat_extent[1] + 0.01),
            lon=slice(lon_extent[0] - 0.01, lon_extent[1] + 0.01))
        im = ds_extent_sel.vhd.plot(ax=ax, cmap=darkblue_hcl_cont_rev, transform=projection, vmin=0, vmax=10,
                                    add_colorbar=False)

        # shows extent of max: plot a contour line for 80% of the maximum of current VHD:
        contours = [ds_extent_sel.vhd.max().item() * 0.8]
        cs = ax.contour(ds_extent_sel.lon, ds_extent_sel.lat, ds_extent_sel.vhd.values, levels=contours, colors="k",
                        linewidths=0.5, transform=projection)

        # Set extent for the plot
        # ax.set_extent((lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]), crs=ccrs.PlateCarree())

        # maybe add topography contours? would need height info in dataset...
        ax.text(0.1, 0.8, f"{time.dt.hour.item() :02d}h", transform=ax.transAxes,  # create hour text label w white box
                fontsize=12, fontweight="bold", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
    cbar = plt.colorbar(im, ax=axes, label=model + r" valley heat deficit [MJ/m^2]")
    cbar.ax.tick_params(size=0, labelsize=12)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(confg.dir_PLOTS, "vhd_plots")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{model}_VHD_small_multiples_{extent_name}.png"), dpi=600)
    fig.show()


def plot_vhd_single_valley_point(point_name: str):
    """
    Create VHD plots for a single valley point with specified height coordinate systems.

    :param point_name: Name of the valley point (key in confg.get_valley_points_only())
    """
    valley_points = confg.get_valley_points_only()

    if point_name not in valley_points:
        print(f"Error: '{point_name}' not found in valley points.")
        print("Available valley points:")
        for name in valley_points.keys():
            print(f"  - {name}")
        return

    point_info = valley_points[point_name]
    print(f"Creating VHD plots for: {point_info['name']} ({point_name})")
    print(f"Location: {point_info['lat']:.3f}°N, {point_info['lon']:.3f}°E")
    print("-" * 50)

    try:
        print(f"  Calculating VHD...")
        vhd_results = calc_vhd_using_timeseries(point_name=point_name)

        plot_vhds_point(vhd_results=vhd_results, point_name=point_info['name'])

        print(f"  ✓ Plot saved: vhd_model_comp_{point_info['name']}.pdf")

    except Exception as e:
        print(f"  ✗ Error: {e}")


if __name__ == '__main__':
    darkblue_hcl = sequential_hcl(palette="Blues 3")  # colors for full domain, small multiple plots
    darkblue_hcl_rev = mcolors.ListedColormap(darkblue_hcl.colors()[::-1])
    darkblue_hcl_cont_rev = darkblue_hcl.cmap().reversed()

    valley_points = confg.get_valley_points_only()  # Get all valley points

    print(f"Creating VHD plots for {len(valley_points)} valley points")
    print("=" * 70)

    for i, (point_name, point_info) in enumerate(valley_points.items(), 1):
        print(f"\n[{i}/{len(valley_points)}] Processing: {point_info['name']} ({point_name})")
        print(f"Location: {point_info['lat']:.3f}°N, {point_info['lon']:.3f}°E")
        print("-" * 50)
        if point_name != "ibk_uni":
            print("  Skipping non-Innsbruck point for now (only ibk_uni has observational data).")
            continue

        try:
            # calculate VHD using "direct" height-coordinate (directly geopot. height as height coord)
            vhd_results = calc_vhd_using_timeseries(point_name=point_name)
            plot_vhds_point(vhd_results=vhd_results, point_name=point_info['name'])

            print(f"  ✓ Comparison plot saved: vhd_model_comp_{point_info['name']}_above_terrain.pdf")

        except Exception as e:
            print(f"  ✗ Error processing {point_name}: {e}")
            continue

    print("\n" + "=" * 70)
    print("✓ VHD plot generation completed for all valley points!")
    print(f"Plots saved to: {os.path.join(confg.dir_PLOTS, 'vhd_plots')}")
    plt.show()

    # Optional: Show plots at the end (comment out if running in batch mode)
    # plt.show()

    """
    # Uncomment these lines if you want to generate spatial VHD plots as well:
    print("\nGenerating spatial VHD small multiples plots...")
    vhd_arome = xr.open_dataset(os.path.join(confg.dir_AROME, "AROME_vhd_full_domain_full_time.nc"))
    vhd_icon = xr.open_dataset(os.path.join(confg.icon_folder_3D, "ICON_vhd_full_domain_full_time.nc"))
    vhd_icon2te = xr.open_dataset(os.path.join(confg.icon2TE_folder_3D, "ICON2TE_vhd_full_domain_full_time.nc"))
    vhd_um = xr.open_dataset(os.path.join(confg.ukmo_folder, "UM_vhd_full_domain_full_time.nc"))
    vhd_wrf = xr.open_dataset(os.path.join(confg.wrf_folder, "WRF_vhd_full_domain_full_time.nc"))

    lat_extent = confg.lat_hf_extent
    lon_extent = confg.lon_hf_extent
    extent_name = "heat_flux"  #  inn_exit heat_flux, cap_height, full_domain, wipp, ziller
    # just for saving of different extents, a fancier script was programmed in
    # plot_topo_comparison, but it is only plotted once...

    plot_vhd_small_multiples(vhd_arome, model="AROME", lat_extent=lat_extent, lon_extent=lon_extent, extent_name=extent_name)
    plot_vhd_small_multiples(vhd_icon, model="ICON", lat_extent=lat_extent, lon_extent=lon_extent, extent_name=extent_name)
    plot_vhd_small_multiples(vhd_icon2te, model="ICON2TE", lat_extent=lat_extent, lon_extent=lon_extent, extent_name=extent_name)
    plot_vhd_small_multiples(vhd_um, model="UM", lat_extent=lat_extent, lon_extent=lon_extent, extent_name=extent_name)
    plot_vhd_small_multiples(vhd_wrf, model="WRF", lat_extent=lat_extent, lon_extent=lon_extent, extent_name=extent_name)
    """
    plt.show()
    print("")