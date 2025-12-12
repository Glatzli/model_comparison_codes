"""
Test suite for plot_topography.py
Tests topography plotting and comparison functionality.
"""

import fix_win_DLL_loading_issue
import os
import sys
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after setting path
from calculations_and_plots import plot_topography


class TestTopographyModule:
    """Test topography module structure and imports"""

    def test_essential_imports(self):
        """Test that essential modules are imported"""
        # Check cartopy imports
        assert hasattr(plot_topography, 'ccrs')
        assert hasattr(plot_topography, 'cfeature')

        # Check matplotlib imports
        assert hasattr(plot_topography, 'plt')
        assert hasattr(plot_topography, 'np')

        # Check data handling imports
        assert hasattr(plot_topography, 'xr')
        assert hasattr(plot_topography, 'rasterio')

    def test_config_imports(self):
        """Test that configuration variables are imported"""
        # Check that confg module elements are imported
        assert hasattr(plot_topography, 'confg')

        # Check specific configuration items
        config_items = ['JSON_TIROL', 'TIROL_DEMFILE', 'cities', 'stations_ibox',
                       'dir_PLOTS', 'station_files_zamg']
        for item in config_items:
            assert hasattr(plot_topography, item)


class TestCoordinateUtilities:
    """Test coordinate utility functions"""

    def test_calculate_km_for_lon_extent(self):
        """Test longitude to km conversion function if it exists"""
        if hasattr(plot_topography, 'calculate_km_for_lon_extent'):
            # Test with typical Tyrol coordinates
            lat = 47.3  # Innsbruck latitude
            lon_extent = 1.0  # 1 degree longitude

            km_extent = plot_topography.calculate_km_for_lon_extent(lon_extent, lat)

            # Should return a reasonable distance in km
            assert isinstance(km_extent, (int, float))
            assert 50 <= km_extent <= 100  # 1 degree at this latitude ≈ 70-80 km


if __name__ == '__main__':
    pytest.main([__file__])
