"""
Test suite for download_geosphere_data.py
Tests data download functionality from Geosphere Austria API.
"""

import fix_win_DLL_loading_issue
import os
import sys
from unittest.mock import patch, MagicMock
import pytest
import requests
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after setting path
from calculations_and_plots import download_geosphere_data


class TestGeosphereAPI:
    """Test Geosphere API functionality"""

    def test_station_id_mapping(self):
        """Test that station ID mapping is correctly defined"""
        assert hasattr(download_geosphere_data, 'STATION_ID_MAPPING')
        assert isinstance(download_geosphere_data.STATION_ID_MAPPING, dict)

        # Check expected stations
        expected_stations = ['KUF', 'JEN', 'IAO', 'LOWI']
        for station in expected_stations:
            assert station in download_geosphere_data.STATION_ID_MAPPING
            assert isinstance(download_geosphere_data.STATION_ID_MAPPING[station], str)

    def test_constants_defined(self):
        """Test that physical constants are properly defined"""
        assert hasattr(download_geosphere_data, 'R')
        assert hasattr(download_geosphere_data, 'g')
        assert hasattr(download_geosphere_data, 'API_BASE_URL')
        assert hasattr(download_geosphere_data, 'DATASET_RESOURCE')
        assert hasattr(download_geosphere_data, 'START_TIME')
        assert hasattr(download_geosphere_data, 'END_TIME')

        # Check reasonable values
        assert 280 <= download_geosphere_data.R <= 290  # Specific gas constant for dry air
        assert 9.8 <= download_geosphere_data.g <= 9.81  # Standard gravity


if __name__ == '__main__':
    pytest.main([__file__])
