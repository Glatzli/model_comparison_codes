"""
Basic testing for plot_timeseries_saved_data.py script.
Simple tests for main functionality.
"""

import fix_win_DLL_loading_issue
import sys
import os
sys.path.append("C:/Users/eleme/Documents/1Uni_Laptop/model_comparison_codes")

import unittest
from unittest.mock import patch
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from calculations_and_plots.plot_timeseries_saved_data import (
    plot_pot_temp_time_contours,
    load_lidar_wind_data
)


def create_simple_dataset():
    """Create simple test dataset."""
    time = pd.date_range('2017-10-15 12:00:00', '2017-10-16 12:00:00', freq='1h')
    height = np.linspace(0, 3000, 50)

    # Random temperature data
    temp_data = np.random.normal(300, 10, (len(time), len(height)))
    wind_u = np.random.normal(5, 2, (len(time), len(height)))
    wind_v = np.random.normal(3, 2, (len(time), len(height)))

    return xr.Dataset({
        'th': (['time', 'height'], temp_data),
        'u': (['time', 'height'], wind_u),
        'v': (['time', 'height'], wind_v),
    }, coords={'time': time, 'height': height})


class BasicTests(unittest.TestCase):
    """Basic test cases."""

    def setUp(self):
        self.test_data = create_simple_dataset()

    def tearDown(self):
        plt.close('all')

    def test_basic_plotting(self):
        """Test basic plotting works."""
        pot_temp = self.test_data['th']

        fig, ax = plot_pot_temp_time_contours(
            pot_temp=pot_temp,
            model="TEST",
            interface_height=2000,
            point_name="test_point"
        )

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        title = ax.get_title()
        self.assertIn("test_point", title)

    def test_plotting_with_wind(self):
        """Test plotting with wind data."""
        pot_temp = self.test_data['th']
        wind_u = self.test_data['u']
        wind_v = self.test_data['v']

        fig, ax = plot_pot_temp_time_contours(
            pot_temp=pot_temp,
            wind_u=wind_u,
            wind_v=wind_v,
            model="TEST_WIND",
            interface_height=2000,
            point_name="test_point"
        )

        self.assertIsNotNone(fig)
        self.assertIn("TEST_WIND", ax.get_title())

    @patch('os.path.exists')
    def test_load_lidar_no_file(self, mock_exists):
        """Test lidar loading when file missing."""
        mock_exists.return_value = False

        wind_u, wind_v = load_lidar_wind_data()

        self.assertIsNone(wind_u)
        self.assertIsNone(wind_v)


def run_basic_tests():
    """Run basic functionality test."""
    print("Running basic tests...")

    # Test dataset creation
    test_data = create_simple_dataset()
    print(f"✓ Test dataset created: {test_data.th.shape}")

    # Test basic plotting
    try:
        fig, ax = plot_pot_temp_time_contours(
            pot_temp=test_data['th'],
            model="TEST",
            interface_height=2000,
            point_name="test_location"
        )
        plt.close(fig)
        print("✓ Basic plotting works")
    except Exception as e:
        print(f"✗ Basic plotting failed: {e}")

    # Run unit tests
    suite = unittest.TestLoader().loadTestsFromTestCase(BasicTests)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("✓ All tests passed")
    else:
        print(f"✗ {len(result.failures + result.errors)} tests failed")

    print("Testing complete!")


if __name__ == '__main__':
    run_basic_tests()
