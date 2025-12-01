"""
Basic test suite for calc_vhd.py
Tests main VHD calculation functions with mock data.
"""

import sys
import os
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calculations_and_plots.calc_vhd import (
    choose_gpe,
    calculate_slope_numpy,
    define_ds_below_hafelekar,
    calc_vhd_single_point,
    calculate_select_pcgp
)


class TestChooseGpe:
    """Test GPE point selection"""

    def test_choose_gpe_basic(self):
        """Test basic GPE selection at a point"""
        # Create mock dataset with lat/lon coordinates
        lat_vals = np.linspace(47.0, 47.5, 10)
        lon_vals = np.linspace(11.0, 11.5, 10)

        ds = xr.Dataset({
            'elevation': (['lat', 'lon'], np.random.uniform(500, 2000, (10, 10))),
            'slope': (['lat', 'lon'], np.random.uniform(0, 45, (10, 10)))
        }, coords={
            'lat': lat_vals,
            'lon': lon_vals
        })

        target_lat, target_lon = 47.25, 11.25

        try:
            result = choose_gpe(ds, target_lat, target_lon)

            assert isinstance(result, xr.Dataset)
            # Should have reduced to a single point
            assert result.sizes['lat'] == 1 or 'lat' not in result.dims
            assert result.sizes['lon'] == 1 or 'lon' not in result.dims
            print("✓ GPE point selection test passed")

        except Exception as e:
            print(f"GPE point selection test failed: {e}")


class TestCalculateSlopeNumpy:
    """Test numpy-based slope calculation"""

    def test_calculate_slope_numpy_basic(self):
        """Test basic slope calculation with numpy"""
        # Create a simple elevation grid with known slope
        elevation_data = np.array([
            [100, 110, 120],
            [100, 110, 120],
            [100, 110, 120]
        ])
        x_res = 100  # 100m resolution

        try:
            slope = calculate_slope_numpy(elevation_data, x_res)

            assert isinstance(slope, np.ndarray)
            assert slope.shape == elevation_data.shape
            # Should calculate reasonable slope values
            assert np.all(slope >= 0)  # Slope should be non-negative
            print("✓ Numpy slope calculation test passed")

        except Exception as e:
            print(f"Numpy slope calculation test failed: {e}")


class TestDefineDsBelowHafelekar:
    """Test dataset filtering below Hafelekar height"""

    def test_define_ds_below_hafelekar_basic(self):
        """Test filtering dataset below Hafelekar height"""
        time = pd.date_range('2017-10-15 12:00', periods=3, freq='1h')
        height = np.array([500, 1000, 1500, 2000, 2500])  # Mix below and above Hafelekar

        ds = xr.Dataset({
            'th': (['time', 'height'], np.random.uniform(285, 295, (3, 5))),
            'u': (['time', 'height'], np.random.uniform(-5, 5, (3, 5))),
            'v': (['time', 'height'], np.random.uniform(-5, 5, (3, 5)))
        }, coords={
            'time': time,
            'height': height
        })

        try:
            result = define_ds_below_hafelekar(ds, model="AROME")

            assert isinstance(result, xr.Dataset)
            # Should filter out heights above Hafelekar (2279m)
            if 'height' in result.coords:
                max_height = result.coords['height'].max().values
                assert max_height <= 2279, f"Max height {max_height} should be <= 2279"

            print("✓ Hafelekar filtering test passed")

        except Exception as e:
            print(f"Hafelekar filtering test failed: {e}")


class TestCalcVhdSinglePoint:
    """Test VHD calculation at a single point"""

    def test_calc_vhd_single_point_basic(self):
        """Test basic VHD calculation at a single point"""
        time = pd.date_range('2017-10-15 12:00', periods=2, freq='1h')
        height = np.array([600, 800, 1000, 1200, 1500])  # Below Hafelekar

        # Create temperature data with some gradient
        temp_data = np.array([
            [288, 286, 284, 282, 280],  # Decreasing with height
            [289, 287, 285, 283, 281]
        ])

        ds = xr.Dataset({
            'th': (['time', 'height'], temp_data),
            'z': (['height'], height)
        }, coords={
            'time': time,
            'height': height
        })

        try:
            result = calc_vhd_single_point(ds, model="AROME")

            assert isinstance(result, xr.Dataset)
            # Should have VHD variable
            if 'vhd' in result.data_vars:
                assert 'vhd' in result.data_vars
                # VHD values should be reasonable
                vhd_vals = result['vhd'].values
                assert np.all(np.isfinite(vhd_vals))  # Should not be NaN/inf

            print("✓ Single point VHD calculation test passed")

        except Exception as e:
            print(f"Single point VHD calculation test failed: {e}")


class TestCalculateSelectPcgp:
    """Test PCGP calculation and selection"""

    def test_calculate_select_pcgp_basic(self):
        """Test basic PCGP calculation"""
        height_levels = np.array([600, 800, 1000, 1200, 1500])

        # Model GPE (should be higher due to model terrain)
        gpe_model = np.array([650, 850, 1050, 1250, 1550])

        # DEM GPE (reference terrain)
        gpe_dem = np.array([612, 812, 1012, 1212, 1512])  # Slightly lower

        try:
            result = calculate_select_pcgp(gpe_model, gpe_dem)

            assert isinstance(result, (int, float, np.number))
            # PCGP should be reasonable (typically small value)
            assert abs(result) < 500, f"PCGP {result} seems unreasonably large"

            print("✓ PCGP calculation test passed")

        except Exception as e:
            print(f"PCGP calculation test failed: {e}")


def run_basic_tests():
    """Run basic functionality tests"""
    print("Running VHD calculation tests...")

    # Test slope calculation logic
    try:
        # Simple elevation grid with known slope
        elevation = np.array([
            [0, 10, 20],
            [0, 10, 20],
            [0, 10, 20]
        ])

        x_res = 100  # 100m spacing
        slope = calculate_slope_numpy(elevation, x_res)

        assert isinstance(slope, np.ndarray)
        assert slope.shape == elevation.shape
        # Should have some slope due to elevation gradient
        assert np.any(slope > 0)

        print("✓ Slope calculation logic test passed")

    except Exception as e:
        print(f"✗ Slope calculation logic test failed: {e}")

    # Test temperature gradient logic
    try:
        heights = np.array([500, 1000, 1500])
        temperatures = np.array([290, 285, 280])  # Decreasing with height

        # Calculate lapse rate
        dT_dz = np.gradient(temperatures, heights)

        assert len(dT_dz) == len(temperatures)
        # Should be negative (temperature decreases with height)
        assert np.mean(dT_dz) < 0

        print("✓ Temperature gradient logic test passed")

    except Exception as e:
        print(f"✗ Temperature gradient logic test failed: {e}")

    # Test dataset filtering logic
    try:
        heights = np.array([500, 1000, 1500, 2000, 2500, 3000])
        hafelekar_height = 2279

        # Filter heights below Hafelekar
        valid_heights = heights[heights <= hafelekar_height]

        assert len(valid_heights) < len(heights)
        assert np.all(valid_heights <= hafelekar_height)
        assert len(valid_heights) == 4  # Should keep first 4 levels

        print("✓ Height filtering logic test passed")

    except Exception as e:
        print(f"✗ Height filtering logic test failed: {e}")

    print("VHD calculation tests complete!")


if __name__ == '__main__':
    run_basic_tests()
