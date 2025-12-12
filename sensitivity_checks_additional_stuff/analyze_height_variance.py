"""
Analyze variance and standard deviation of height coordinate changes over time.
This script examines how the lidar height coordinates vary over time.
"""

import sys
sys.path.append("C:/Users/eleme/Documents/1Uni_Laptop/model_comparison_codes")

import confg
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def analyze_height_variance():
    """
    Analyze the variance and standard deviation of height coordinate changes over time.
    """
    print("Loading lidar data to analyze height variance...")

    # Load lidar data
    if not confg.lidar_sl88_merged_path or not confg.lidar_sl88_merged_path.endswith('.nc'):
        print("Error: Invalid lidar file path")
        return

    try:
        ds_lidar = xr.open_dataset(confg.lidar_sl88_merged_path)

        # Get the height coordinate (2D array: time, height)
        height_coord = ds_lidar['height']

        print(f"Height coordinate shape: {height_coord.shape}")
        print(f"Height coordinate dimensions: {height_coord.dims}")
        print(f"Time range: {height_coord.time.values[0]} to {height_coord.time.values[-1]}")
        print(f"Number of height levels: {height_coord.shape[1]}")
        print(f"Number of time steps: {height_coord.shape[0]}")

        # Calculate the change in height over time (diff along time dimension)
        # This gives us how much each height level changes from one time step to the next
        height_diff = height_coord.diff('time')

        print(f"\nHeight difference shape: {height_diff.shape}")
        print(f"Height difference dimensions: {height_diff.dims}")

        # Calculate variance and standard deviation along time dimension
        # This shows how much each height level varies over time
        height_var_time = height_coord.var('time')
        height_std_time = height_coord.std('time')

        # Calculate variance and standard deviation of the differences
        height_diff_var_time = height_diff.var('time')
        height_diff_std_time = height_diff.std('time')

        # Calculate variance and standard deviation along height dimension
        # This shows how much the height profile varies at each time step
        height_var_height = height_coord.var('height')
        height_std_height = height_coord.std('height')

        # Statistics summary
        print("\n" + "="*60)
        print("HEIGHT COORDINATE ANALYSIS")
        print("="*60)

        print(f"\n1. Overall statistics:")
        print(f"   Min height: {height_coord.min().values:.2f} m")
        print(f"   Max height: {height_coord.max().values:.2f} m")
        print(f"   Mean height: {height_coord.mean().values:.2f} m")
        print(f"   Global std: {height_coord.std().values:.2f} m")
        print(f"   Global var: {height_coord.var().values:.2f} m²")

        print(f"\n2. Variance over TIME (for each height level):")
        print(f"   Min variance: {height_var_time.min().values:.6f} m²")
        print(f"   Max variance: {height_var_time.max().values:.6f} m²")
        print(f"   Mean variance: {height_var_time.mean().values:.6f} m²")
        print(f"   Std of variance: {height_var_time.std().values:.6f} m²")

        print(f"\n3. Standard deviation over TIME (for each height level):")
        print(f"   Min std: {height_std_time.min().values:.6f} m")
        print(f"   Max std: {height_std_time.max().values:.6f} m")
        print(f"   Mean std: {height_std_time.mean().values:.6f} m")
        print(f"   Std of std: {height_std_time.std().values:.6f} m")

        print(f"\n4. TIME DIFFERENCES analysis:")
        print(f"   Min height diff: {height_diff.min().values:.6f} m")
        print(f"   Max height diff: {height_diff.max().values:.6f} m")
        print(f"   Mean height diff: {height_diff.mean().values:.6f} m")
        print(f"   Global std of diffs: {height_diff.std().values:.6f} m")

        print(f"\n5. Variance of TIME DIFFERENCES (for each height level):")
        print(f"   Min diff variance: {height_diff_var_time.min().values:.6f} m²")
        print(f"   Max diff variance: {height_diff_var_time.max().values:.6f} m²")
        print(f"   Mean diff variance: {height_diff_var_time.mean().values:.6f} m²")

        print(f"\n6. Standard deviation of TIME DIFFERENCES (for each height level):")
        print(f"   Min diff std: {height_diff_std_time.min().values:.6f} m")
        print(f"   Max diff std: {height_diff_std_time.max().values:.6f} m")
        print(f"   Mean diff std: {height_diff_std_time.mean().values:.6f} m")

        print(f"\n7. Variance over HEIGHT (for each time step):")
        print(f"   Min variance: {height_var_height.min().values:.2f} m²")
        print(f"   Max variance: {height_var_height.max().values:.2f} m²")
        print(f"   Mean variance: {height_var_height.mean().values:.2f} m²")

        # Check if heights are constant over time
        if height_std_time.max().values < 1e-6:
            print(f"\n>> Heights appear to be CONSTANT over time (max std = {height_std_time.max().values:.10f} m)")
            print("   This explains why you can't index by height - the height coordinate doesn't change!")
            print("   You should use the height_m coordinate instead, which is 1D.")
        else:
            print(f"\n>> Heights DO vary over time (max std = {height_std_time.max().values:.6f} m)")

        # Create some plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Height variance over height levels
        axes[0,0].plot(height_coord.height_m.values, height_var_time.values)
        axes[0,0].set_xlabel('Height (m)')
        axes[0,0].set_ylabel('Variance over time (m²)')
        axes[0,0].set_title('Height Variance vs Height Level')
        axes[0,0].grid(True)

        # Plot 2: Height std over height levels
        axes[0,1].plot(height_coord.height_m.values, height_std_time.values)
        axes[0,1].set_xlabel('Height (m)')
        axes[0,1].set_ylabel('Std dev over time (m)')
        axes[0,1].set_title('Height Standard Deviation vs Height Level')
        axes[0,1].grid(True)

        # Plot 3: Height variance over time
        axes[1,0].plot(height_coord.time.values, height_var_height.values)
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Variance over height (m²)')
        axes[1,0].set_title('Height Profile Variance vs Time')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True)

        # Plot 4: Sample height profiles at different times
        sample_times = [0, len(height_coord.time)//4, len(height_coord.time)//2, -1]
        for i, t_idx in enumerate(sample_times):
            axes[1,1].plot(height_coord.isel(time=t_idx).values, height_coord.height_m.values,
                          label=f'Time {t_idx}: {str(height_coord.time.values[t_idx])[:19]}',
                          alpha=0.7)
        axes[1,1].set_xlabel('Height coordinate value (m)')
        axes[1,1].set_ylabel('Height level (m)')
        axes[1,1].set_title('Height Profiles at Different Times')
        axes[1,1].legend(fontsize=8)
        axes[1,1].grid(True)

        plt.tight_layout()

        # Save the plot
        plot_dir = confg.dir_PLOTS
        output_path = f"{plot_dir}/height_variance_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved analysis plot to: {output_path}")

        plt.show()

        # Return useful values
        return {
            'height_var_time': height_var_time,
            'height_std_time': height_std_time,
            'height_diff_var': height_diff_var_time,
            'height_diff_std': height_diff_std_time,
            'height_coord': height_coord,
            'height_diff': height_diff
        }

    except Exception as e:
        print(f"Error analyzing height variance: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    results = analyze_height_variance()
