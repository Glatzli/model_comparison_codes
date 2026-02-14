"""
Here a comparison plot of the different height levels of the models and Observations is made
"""
import fix_win_DLL_loading_issue

fix_win_DLL_loading_issue

import matplotlib.pyplot as plt
import xarray as xr
import os

import confg
from manage_timeseries import load_or_read_timeseries
from read_in_hatpro_radiosonde import read_radiosonde_dataset


def plot_height_levels(arome_heights, icon_heights, um_heights, wrf_heights, radio_heights, hatpro_heights=None):
    """
    plots the different geopot. height levels for all models incl. radiosonde & hatpro data, for
    getting an overview which data should be interpolated onto which levels...
    :param arome_heights:
    :param icon_heights:
    :param um_heights:
    :param wrf_heights:
    :param radio_heights:
    :param hatpro_heights:
    :return:
    """
    model_names = ['AROME', 'ICON', 'UM', 'WRF', 'Radiosonde', 'HATPRO']
    height_arrays = [arome_heights, icon_heights, um_heights, wrf_heights, radio_heights, hatpro_heights]

    plt.figure(figsize=(5, 6))
    for i, heights in enumerate(height_arrays):
        plt.plot([model_names[i]] * len(heights), heights, color=confg.model_colors_temp_wind[model_names[i]],
                 linestyle="None", label=model_names[i], ms=12, marker="_")

    plt.ylabel('Height above terrain [m]')
    plt.ylim([0, 200])
    # plt.title('vertical level comparison models & measurments')
    plt.grid(True, alpha=0.4, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(confg.dir_PLOTS, "model_infos_general", "vertical_geopot_height_levels_comp.pdf"))
    plt.show()


if __name__ == "__main__":
    """
    Main function to load timeseries for Innsbruck Uni gridpoint and plot vertical height levels
    """
    print("Loading timeseries for Innsbruck Uni gridpoint...\n")

    # Define point
    point = confg.ALL_POINTS["ibk_uni"]
    point_name = "ibk uni"

    # ========== Load model heights ==========
    models = ["AROME", "ICON", "UM", "WRF"]
    model_heights = {}

    for model in models:
        print(f"Loading {model}...")
        ds = load_or_read_timeseries(model=model, point=point, point_name=point_name, height_as_z_coord="above_terrain")
        ds_sel = ds.isel(time=0)  # select only first time step for height levels

        model_heights[model] = ds_sel['height'].values
        print(f"  ✓ {len(model_heights[model])} height levels")

    # Unpack model heights for easier access
    arome_heights = model_heights["AROME"]
    icon_heights = model_heights["ICON"]
    um_heights = model_heights["UM"]
    wrf_heights = model_heights["WRF"]

    # ========== Load Radiosonde heights ==========
    print("Loading Radiosonde...")
    try:
        ds_radio = read_radiosonde_dataset(height_as_z_coord="above_terrain")
        radio_heights = ds_radio['height'].values
        print(f"  ✓ {len(radio_heights)} height levels")

    except Exception as e:
        print(f"  ✗ Error loading Radiosonde data: {e}")
        radio_heights = []

    # ========== Load HATPRO heights ==========
    print("Loading HATPRO...")
    try:
        ds_hatpro = xr.open_dataset(confg.hatpro_calced_vars)
        if 'height' in ds_hatpro.coords:
            hatpro_heights = ds_hatpro['height'].values
            print(f"  ✓ {len(hatpro_heights)} height levels")
        else:
            print(f"  ✗ Warning: No 'height' coordinate found in HATPRO dataset")
            hatpro_heights = None
    except Exception as e:
        print(f"  ✗ Error loading HATPRO data: {e}")
        hatpro_heights = None

    # ========== Create plot ==========
    print("\n" + "=" * 50)
    print("Creating vertical level comparison plot...")
    print("=" * 50)

    plot_height_levels(arome_heights=arome_heights, icon_heights=icon_heights, um_heights=um_heights,
        wrf_heights=wrf_heights, radio_heights=radio_heights, hatpro_heights=hatpro_heights)
    plt.show()
    print("\n✓ Done! Plot saved successfully.")