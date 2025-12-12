"""
Test suite for manage_timeseries.py
Tests timeseries data management functionality.
"""

import fix_win_DLL_loading_issue
import os
import sys
from unittest.mock import patch, MagicMock
import pytest
import xarray as xr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after setting path
from calculations_and_plots import manage_timeseries


class TestTimeseriesManagement:
    """Test timeseries management functionality"""

    def test_variables_list_defined(self):
        """Test that required variables list is properly defined"""
        assert hasattr(manage_timeseries, 'variables')
        variables = manage_timeseries.variables

        assert isinstance(variables, list)
        assert len(variables) > 0

        # Check for essential meteorological variables
        expected_vars = ["u", "v", "p", "th", "temp", "z"]
        for var in expected_vars:
            assert var in variables

    def test_model_order_defined(self):
        """Test that model processing order is defined"""
        assert hasattr(manage_timeseries, 'MODEL_ORDER')
        model_order = manage_timeseries.MODEL_ORDER

        assert isinstance(model_order, list)
        assert len(model_order) > 0

        # Check for expected models
        expected_models = ["AROME", "ICON", "UM", "WRF"]
        for model in expected_models:
            assert model in model_order


if __name__ == '__main__':
    pytest.main([__file__])
