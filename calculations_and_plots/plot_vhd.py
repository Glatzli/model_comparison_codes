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
import pandas as pd
import xarray as xr

import confg
from calculations_and_plots.calc_vhd import calc_vhd_single_point, select_pcgp_vhd
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


def plot_vhds_point(vhd_results: dict, point_name: str, origin: str = "direct"):
    """
    Plot VHD timeseries for all models using the saved timeseries data with the direct height coordinate

    :param vhd_results: Dictionary with VHD datasets for each model
    :param point_name: Name of the point location
    """
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot model data
    models_to_plot = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]
    for model in models_to_plot:
        if model in vhd_results and vhd_results[model] is not None:
            vhd = vhd_results[model]
            (vhd.vhd / 10 ** 6).plot(ax=ax, label=model, color=confg.model_colors_temp_wind[model], linewidth=2)

    # Plot observational data if available
    if "HATPRO" in vhd_results and vhd_results["HATPRO"] is not None:
        (vhd_results["HATPRO"].vhd / 10 ** 6).plot(ax=ax, label="HATPRO",
                                                   color=confg.model_colors_temp_wind["HATPRO"], linewidth=2,
                                                   linestyle="dotted")
    # Plot observational data if available
    if "radio" in vhd_results and vhd_results["radio"] is not None:
        ax.scatter(datetime.datetime(2017, 10, 16, 2, 15, 0), vhd_results["radio"].vhd / 10 ** 6,
                   label="Radiosonde", color=confg.model_colors_temp_wind["Radiosonde"], s=100, marker="*",
                   zorder=10)
    plt.ylim(0, 15)
    plt.ylabel(r"VHD [MJ m$^{-2}$]", fontsize=13)
    plt.grid()
    plt.title("")
    # plt.title(f"VHD timeline for {point_name} via {vhd_origin}")    plt.legend(loc='upper left', fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    plt.legend(loc="upper left", fontsize=13)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(confg.dir_PLOTS, "vhd_plots")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"vhd_model_comp_{point_name.replace(" ", "_")}_{origin}.pdf"))


def plot_vhd_two_models_comparison(ds1, ds2, timestamp, model1="AROME", model2="WRF",
        lat_extent: tuple = confg.lat_cap_height_extent, lon_extent: tuple = confg.lon_cap_height_extent,
        extent_name: str = "cap_height"):
    """
    Plot VHD for two models side by side for a single timestamp with shared colorbar and extent.

    :param ds1: VHD dataset for first model with time, lat, lon coordinates
    :param ds2: VHD dataset for second model with time, lat, lon coordinates
    :param timestamp: Single timestamp to plot (string or datetime-like object)
    :param model1: First model name (AROME, ICON, ICON2TE, UM, WRF)
    :param model2: Second model name (AROME, ICON, ICON2TE, UM, WRF)
    :param lat_extent: Tuple (lat_min, lat_max) for plot extent (same for both models)
    :param lon_extent: Tuple (lon_min, lon_max) for plot extent (same for both models)
    :param extent_name: Name of extent for saving purposes (e.g., "cap_height")
    :return:
    """
    plt.rcParams.update({'font.size': 13})
    projection = ccrs.Mercator()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': projection})

    # Select the specific timestamp and spatial subset for both models (same extent)
    ds1_sel = (ds1.sel(time=timestamp) / 1e6).sel(lat=slice(lat_extent[0] - 0.01, lat_extent[1] + 0.01),
                                                  lon=slice(lon_extent[0] - 0.01, lon_extent[1] + 0.01))
    ds2_sel = (ds2.sel(time=timestamp) / 1e6).sel(lat=slice(lat_extent[0] - 0.01, lat_extent[1] + 0.01),
                                                  lon=slice(lon_extent[0] - 0.01, lon_extent[1] + 0.01))

    # Plot first model (left)
    im1 = ds1_sel.vhd.plot(ax=axes[0], cmap=darkblue_hcl_cont_rev, transform=projection, vmin=0, vmax=10,
                           add_colorbar=False)
    contours1 = [ds1_sel.vhd.max().item() * 0.8]
    axes[0].contour(ds1_sel.lon, ds1_sel.lat, ds1_sel.vhd.values, levels=contours1, colors="k", linewidths=1.0,
                    transform=projection)
    axes[0].set_title(model1, fontsize=13)

    # Plot second model (right)
    im2 = ds2_sel.vhd.plot(ax=axes[1], cmap=darkblue_hcl_cont_rev, transform=projection, vmin=0, vmax=10,
                           add_colorbar=False)
    contours2 = [ds2_sel.vhd.max().item() * 0.8]
    axes[1].contour(ds2_sel.lon, ds2_sel.lat, ds2_sel.vhd.values, levels=contours2, colors="k", linewidths=1.0,
                    transform=projection)
    axes[1].set_title(model2, fontsize=13)

    # Add shared colorbar to the right of both plots with equal sizing
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax, label=r"VHD [MJ m$^{-2}$]")
    cbar.ax.tick_params(labelsize=13)

    # Format time string
    time_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M')
    # fig.suptitle(time_str, fontsize=14, fontweight='bold', y=0.98)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(confg.dir_PLOTS, "vhd_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with timestamp
    time_file_str = pd.to_datetime(timestamp).strftime('%d_%H%M')
    fig.savefig(os.path.join(output_dir, f"{model1}_vs_{model2}_VHD_{extent_name}_{time_file_str}.pdf"))
    plt.close(fig)
    print(f"  ✓ Saved: {model1}_vs_{model2}_VHD_{extent_name}_{time_file_str}.pdf")


def plot_vhd_single_timestamp(ds_extent, timestamp, model="ICON", lat_extent: tuple = confg.lat_cap_height_extent,
        lon_extent: tuple = confg.lon_cap_height_extent, extent_name: str = "cap_height"):
    """
    Plot VHD for a single timestamp with contour line at 80% of maximum.

    :param ds_extent: VHD dataset with time, lat, lon coordinates
    :param timestamp: Single timestamp to plot (string or datetime-like object)
    :param model: Model name (AROME, ICON, ICON2TE, UM, WRF)
    :param lat_extent: Tuple (lat_min, lat_max) for plot extent
    :param lon_extent: Tuple (lon_min, lon_max) for plot extent
    :param extent_name: Name of extent for saving purposes (e.g., "cap_height")
    :return:
    """
    plt.rcParams.update({'font.size': 13})
    projection = ccrs.Mercator()
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': projection})

    # Select the specific timestamp and spatial subset
    ds_extent_sel = (ds_extent.sel(time=timestamp) / 1e6).sel(lat=slice(lat_extent[0] - 0.01, lat_extent[1] + 0.01),
                                                              lon=slice(lon_extent[0] - 0.01, lon_extent[1] + 0.01))

    # Plot VHD
    im = ds_extent_sel.vhd.plot(ax=ax, cmap=darkblue_hcl_cont_rev, transform=projection, vmin=0, vmax=10,
                                cbar_kwargs={'label': r" VHD [MJ m$^{-2}$]"})

    # Plot contour line at 80% of maximum VHD
    contours = [ds_extent_sel.vhd.max().item() * 0.8]
    cs = ax.contour(ds_extent_sel.lon, ds_extent_sel.lat, ds_extent_sel.vhd.values, levels=contours, colors="k",
                    linewidths=1.0, transform=projection)

    # Add timestamp as text label
    time_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M')
    # ax.text(0.05, 0.95, time_str, transform=ax.transAxes, fontsize=13, fontweight="bold",
    #         bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"), va='top')

    ax.set_title("")
    ax.tick_params(axis='both', labelsize=13)
    plt.tight_layout()

    # Create output directory if it doesn't exist
    output_dir = os.path.join(confg.dir_PLOTS, "vhd_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with timestamp
    time_file_str = pd.to_datetime(timestamp).strftime('%d_%H%M')
    fig.savefig(os.path.join(output_dir, f"{model}_VHD_single_{extent_name}_{time_file_str}.pdf"))
    print(f"  ✓ Saved: {model}_VHD_single_{extent_name}_{time_file_str}.pdf")


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
    plt.rcParams.update({'font.size': 13})
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
                fontsize=13, fontweight="bold", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
    cbar = plt.colorbar(im, ax=axes, label=r" VHD [MJ m$^{-2}$]")
    cbar.ax.tick_params(size=0, labelsize=13)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(confg.dir_PLOTS, "vhd_plots")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{model}_VHD_small_multiples_{extent_name}.png"), dpi=600)  # fig.show()


def plot_vhds_point_from_full_extent(vhd_datasets: dict, point_name: str, point_info: dict):
    """
    Plot VHD timeseries for all models extracted from full domain datasets at a specific point.

    :param vhd_datasets: Dictionary with full domain VHD datasets for each model
    :param point_name: Name of the point location
    :param point_info: Dictionary with 'name', 'lat', 'lon' keys
    """
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(figsize=(10, 6))

    lat = point_info['lat']
    lon = point_info['lon']

    # Plot model data
    models_to_plot = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]
    for model in models_to_plot:
        if model in vhd_datasets and vhd_datasets[model] is not None:
            vhd_full = vhd_datasets[model]
            # Extract VHD at the specific point using nearest neighbor selection
            vhd_point = vhd_full.sel(lat=lat, lon=lon, method="nearest")
            (vhd_point.vhd / 10 ** 6).plot(ax=ax, label=model, color=confg.model_colors_temp_wind[model], linewidth=2)

    plt.ylim(0, 15)
    plt.ylabel(r"VHD [MJ m$^{-2}$]", fontsize=13)
    plt.grid()
    plt.title("")
    plt.legend(loc='upper left', fontsize=13)
    ax.tick_params(axis='both', labelsize=13)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(confg.dir_PLOTS, "vhd_plots")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"vhd_model_comp_{point_name.replace(' ', '_')}_full_extent.pdf"))
    plt.close(fig)


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

    """
    print(f"Creating VHD plots for {len(valley_points)} valley points")
    print("=" * 70)
    for i, (point_name, point_info) in enumerate(valley_points.items(), 1):
        print(f"\n[{i}/{len(valley_points)}] Processing: {point_info['name']} ({point_name})")
        print(f"Location: {point_info['lat']:.3f}°N, {point_info['lon']:.3f}°E")
        print("-" * 50)
        # if point_name not in ["brenner_saddle", "wipp_stafflach_steinach", "wipp_schoenberg_matrei",
        # "wipp_schoenberg", "patsch_EC_south"]:
        # print("  Skipping non-Wippvalley point for now.")
        # continue

        try:
            # calculate VHD using "direct" height-coordinate (directly geopot. height as height coord)
            vhd_results = calc_vhd_using_timeseries(point_name=point_name)
            plot_vhds_point(vhd_results=vhd_results, point_name=point_info['name'], origin="direct")

            print(f"  ✓ Comparison plot saved: vhd_model_comp_{point_info['name']}_above_terrain.pdf")

        except Exception as e:
            print(f"  ✗ Error processing {point_name}: {e}")
            continue

    print("\n" + "=" * 70)
    print("✓ VHD plot generation completed for all valley points!")
    print(f"Plots saved to: {os.path.join(confg.dir_PLOTS, 'vhd_plots')}")
    # plt.show()
    """
    # Optional: Show plots at the end (comment out if running in batch mode)
    """
    # Uncomment these lines if you want to generate spatial VHD plots as well:
    print("\nGenerating spatial VHD small multiples plots...")

    print(f"\nCreating VHD point plots from full extent VHD calculation")
    for (point_name, point_info) in valley_points.items():
        print(f" Processing: {point_info['name']} ({point_name})")
        vhd_arome_pcgp, vhd_icon_pcgp, vhd_icon2te_pcgp, vhd_um_pcgp, vhd_wrf_pcgp = select_pcgp_vhd(
            lat=point_info['lat'], lon=point_info['lon'])
        vhd_full_datasets = {"AROME": vhd_arome_pcgp, "ICON": vhd_icon_pcgp, "ICON2TE": vhd_icon2te_pcgp,
                             "UM": vhd_um_pcgp, "WRF": vhd_wrf_pcgp}
        plot_vhds_point(vhd_results=vhd_full_datasets, point_name=point_name, origin="full_extent")
    """
    lat_extent = confg.lat_central_inn_extent
    lon_extent = confg.lon_central_inn_extent
    extent_name = "central_inn"  # inn_exit, ibk_surr, heat_flux (central_inn), full_inn, cap_height, full_domain,
    # wipp, ziller
    # just for saving of different extents, a fancier script was programmed in
    # plot_topo_comparison, but it is only plotted once...

    vhd_arome = xr.open_dataset(confg.dir_AROME + "/AROME_vhd_full_domain_full_time.nc")
    vhd_icon = xr.open_dataset(confg.icon_folder_3D + "/ICON_vhd_full_domain_full_time.nc")
    vhd_icon2te = xr.open_dataset(confg.icon2TE_folder_3D + "/ICON2TE_vhd_full_domain_full_time.nc")
    vhd_um = xr.open_dataset(confg.ukmo_folder + "/UM_vhd_full_domain_full_time.nc")
    vhd_wrf = xr.open_dataset(confg.wrf_folder + "/WRF_vhd_full_domain_full_time.nc")

    vhd_dict = {"AROME": vhd_arome, "ICON": vhd_icon, "ICON2TE": vhd_icon2te, "UM": vhd_um, "WRF": vhd_wrf}
    # for model in vhd_dict:
    #     plot_vhd_small_multiples(vhd_dict[model], model=model, lat_extent=lat_extent, lon_extent=lon_extent,
    #                              extent_name=extent_name)

    # plot a single timestamp and model with:
    for model in vhd_dict:
        plot_vhd_single_timestamp(vhd_dict[model], timestamp="2017-10-16 08:00:00", model=model, lat_extent=lat_extent,
                                  lon_extent=lon_extent, extent_name=extent_name)

    # plot_vhd_two_models_comparison(vhd_dict["AROME"], vhd_dict["WRF"], timestamp="2017-10-16 06:00:00",
    #                                model1="AROME", model2="WRF", lat_extent=lat_extent,
    #                                lon_extent=lon_extent, extent_name=extent_name)

    # plt.show()
    print("")