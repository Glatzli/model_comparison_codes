"""
Test suite for AROME/profile_radiosonde.py
Tests radiosonde data processing and plotting functionality.
"""

import fix_win_DLL_loading_issue
import os
import sys
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after setting path
from AROME import profile_radiosonde


class TestRadiosondeConstants:
    """Test constants and configuration"""

    def test_launch_date_defined(self):
        """Test that launch date is properly defined"""
        assert hasattr(profile_radiosonde, 'launch_date')
        launch_date = profile_radiosonde.launch_date

        assert isinstance(launch_date, pd.Timestamp)
        assert launch_date.year == 2017
        assert launch_date.month == 10
        assert launch_date.day == 16

    def test_model_time_defined(self):
        """Test that model time is properly defined"""
        assert hasattr(profile_radiosonde, 'time_for_model')
        model_time = profile_radiosonde.time_for_model

        assert isinstance(model_time, pd.Timestamp)
        assert model_time.year == 2017
        assert model_time.month == 10
        assert model_time.day == 16

    def test_time_consistency(self):
        """Test that launch and model times are consistent"""
        launch = profile_radiosonde.launch_date
        model = profile_radiosonde.time_for_model

        # Model time should be after launch time
        assert model >= launch

        # Should be within reasonable time difference (less than 12 hours)
        time_diff = model - launch
        assert time_diff.total_seconds() <= 12 * 3600


class TestPlottingIntegration:
    """Test plotting functionality and dependencies"""

    def test_plotting_imports(self):
        """Test that plotting libraries are imported"""
        # Check essential plotting imports
        assert hasattr(profile_radiosonde, 'plt')
        assert hasattr(profile_radiosonde, 'SkewT')
        assert hasattr(profile_radiosonde, 'Hodograph')

    def test_metpy_imports(self):
        """Test that MetPy libraries are imported"""
        assert hasattr(profile_radiosonde, 'metpy')
        assert hasattr(profile_radiosonde, 'mpcalc')
        assert hasattr(profile_radiosonde, 'units')


if __name__ == '__main__':
    pytest.main([__file__])
