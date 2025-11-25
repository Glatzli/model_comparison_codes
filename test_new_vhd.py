"""
Test script for the new VHD calculation using manage_timeseries system.
This script tests the above_domain feature for VHD calculations.
"""
import os
import confg
import matplotlib.pyplot as plt
from calculations_and_plots.plot_vhd import calc_vhd_using_new_timeseries, calc_vhd_single_point, load_or_read_timeseries
import calculations_and_plots.calc_vhd as calc_vhd
from colorspace import qualitative_hcl


def calc_vhd_using_new_timeseries(point_name: str, height_as_z_coord: str = "above_domain"):
    """
    Calculate VHD for all models using the new manage_timeseries system.
    Uses the "above_domain" feature for consistent height coordinates.

    :param point_name: Name of the point (key in confg.ALL_POINTS)
    :param height_as_z_coord: Height coordinate system to use ("above_domain" recommended)
    :return: Dictionary with VHD datasets for each model
    """
    point = confg.ALL_POINTS[point_name]
    models = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]
    vhd_results = {}

    for model in models:
        print(f"Loading/calculating VHD for {model} at {point['name']}...")

        # Load timeseries using the new system
        ds = load_or_read_timeseries(
            model=model,
            point=point,
            point_name=point_name,
            height_as_z_coord=height_as_z_coord
        )

        if ds is not None:
            # Calculate VHD using the existing function
            vhd = calc_vhd_single_point(ds, model=model)
            vhd_results[model] = vhd
            ds.close()
        else:
            print(f"  Warning: Could not load timeseries for {model}")
            vhd_results[model] = None

    return vhd_results


def plot_vhds_point_new(vhd_results: dict, point_name: str, vhd_origin: str = "new_timeseries"):
    """
    Plot VHD timeseries for all models using the new data structure.

    :param vhd_results: Dictionary with VHD datasets for each model
    :param point_name: Name of the point location
    :param vhd_origin: Description of calculation method
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Model colors (consistent with existing code)
    qualitative_colors = qualitative_hcl(palette="Dark 3").colors()
    model_colors = {
        "AROME": qualitative_colors[0],
        "ICON": qualitative_colors[2],
        "ICON2TE": qualitative_colors[2],
        "UM": qualitative_colors[4],
        "WRF": qualitative_colors[6],
        "HATPRO": qualitative_colors[8]
    }

    # Plot model data
    models_to_plot = ["AROME", "ICON", "ICON2TE", "UM", "WRF"]
    for model in models_to_plot:
        if model in vhd_results and vhd_results[model] is not None:
            vhd = vhd_results[model]
            linestyle = "--" if model == "ICON2TE" else "-"
            (vhd.vhd / 10**6).plot(
                ax=ax,
                label=model,
                color=model_colors[model],
                linewidth=2,
                linestyle=linestyle
            )

    plt.ylim(0.08, 0.6)
    plt.ylabel(r"valley heat deficit $[\frac{MJ}{m^2}]$")
    plt.grid()
    plt.title(f"VHD timeline for {point_name} via {vhd_origin}")
    plt.legend(loc='upper left')

    # Create output directory if it doesn't exist
    output_dir = os.path.join(confg.dir_PLOTS, "vhd_plots")
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(os.path.join(output_dir, f"vhd_model_comp_{point_name}_{vhd_origin}.svg"))


if __name__ == "__main__":
    # Test point
    point_name = "ibk_villa"
    point = confg.ALL_POINTS[point_name]

    print(f"Testing new VHD calculation system for {point['name']}")
    print(f"Location: {point['lat']:.3f}°N, {point['lon']:.3f}°E")
    print("-" * 50)

    # Calculate VHD using new system with above_domain feature
    vhd_results = calc_vhd_using_new_timeseries(
        point_name=point_name,
        height_as_z_coord="above_domain"
    )

    print("\nVHD calculation results:")
    for model, vhd in vhd_results.items():
        if vhd is not None:
            max_vhd = vhd.vhd.max().values / 1e6
            min_vhd = vhd.vhd.min().values / 1e6
            print(f"  {model}: VHD range {min_vhd:.3f} - {max_vhd:.3f} MJ/m²")
        else:
            print(f"  {model}: Failed to load/calculate")

    # Create plot
    plt.style.use('default')

    plot_vhds_point_new(
        vhd_results=vhd_results,
        point_name=point['name'],
        vhd_origin="above_domain_test"
    )

    print(f"\nPlot saved to: {confg.dir_PLOTS}vhd_plots/vhd_model_comp_{point['name']}_above_domain_test.svg")
    plt.show()
