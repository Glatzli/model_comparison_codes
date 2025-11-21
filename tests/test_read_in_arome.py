"""
Test suite for read_in_arome.py, written by Claude 4.5

Tests core functions with mock data to ensure basic functionality.
Simplified version focusing on essential features.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import xarray as xr
import numpy as np
import pandas as pd

from read_in_arome import (
    convert_calc_variables,
    check_add_needed_variables,
    read_in_arome,
    rename_vars,
    read_2D_variables_AROME,
)


# Fixtures
@pytest.fixture
def dummy_ds():
    """Create a fake xarray dataset for testing with realistic values"""
    time = pd.date_range('2017-10-15 12:00', periods=5, freq='30min')
    height = np.arange(1, 11)
    lat = np.linspace(47.0, 48.0, 10)
    lon = np.linspace(11.0, 12.0, 10)

    ds = xr.Dataset(
        {
            "u": (["time", "height", "lat", "lon"], np.random.uniform(-5, 5, (5, 10, 10, 10))),
            "v": (["time", "height", "lat", "lon"], np.random.uniform(-5, 5, (5, 10, 10, 10))),
            "p": (["time", "height", "lat", "lon"], np.random.uniform(90000, 100000, (5, 10, 10, 10))),
            "th": (["time", "height", "lat", "lon"], np.random.uniform(290, 310, (5, 10, 10, 10))),
            "z": (["time", "height", "lat", "lon"], np.random.uniform(500, 3000, (5, 10, 10, 10))),
        },
        coords={
            "time": time,
            "height": height,
            "lat": lat,
            "lon": lon
        }
    )
    return ds


@pytest.fixture
def minimal_ds():
    """Minimal dataset for testing error handling"""
    return xr.Dataset(
        {
            "p": (["time", "height"], np.ones((2, 3)) * 100000),
            "th": (["time", "height"], np.ones((2, 3)) * 300),
        },
        coords={
            "time": [0, 1],
            "height": [1, 2, 3]
        }
    )


# Tests for convert_calc_variables
class TestConvertCalcVariables:
    """Test the convert_calc_variables function - basic tests only"""

    def test_wind_both(self, dummy_ds):
        """Test calculating both wind speed and direction"""
        ds = convert_calc_variables(dummy_ds, vars_to_calc=["wspd", "udir"])
        assert "wspd" in ds
        assert "udir" in ds

    def test_wind_missing_u_v(self, minimal_ds, capsys):
        """Test error handling when u and v are missing"""
        ds = convert_calc_variables(minimal_ds, vars_to_calc=["wspd"])
        assert "wspd" not in ds
        # Capture the error message that was printed
        captured = capsys.readouterr()
        assert "Error calculating wind speed/direction" in captured.out

    def test_temperature_calculation(self, dummy_ds):
        """Test temperature calculation from potential temperature"""
        ds = convert_calc_variables(dummy_ds, vars_to_calc=["temp"])
        assert "temp" in ds
        assert ds["temp"].attrs["units"] == "degC"
        # Temperature should be reasonable
        assert np.all(ds["temp"].values > -50)
        assert np.all(ds["temp"].values < 50)

    def test_density_calculation(self, dummy_ds):
        """Test air density calculation using ideal gas law"""
        ds = convert_calc_variables(dummy_ds, vars_to_calc=["temp", "rho"])
        assert "rho" in ds
        assert "units" in ds["rho"].attrs
        assert "287.05" in ds["rho"].attrs["description"]
        # Density should be reasonable
        assert np.all(ds["rho"].values > 0.5)
        assert np.all(ds["rho"].values < 1.5)

    def test_pressure_conversion(self, dummy_ds):
        """Test pressure conversion from Pa to hPa"""
        original_p = dummy_ds["p"].values.copy()
        ds = convert_calc_variables(dummy_ds, vars_to_calc=[])
        assert "p" in ds
        assert "units" in ds["p"].attrs
        # Check conversion (Pa to hPa is division by 100)
        assert np.allclose(ds["p"].values, original_p / 100, rtol=0.01)


# Tests for check_add_needed_variables
class TestCheckAddNeededVariables:
    """Test the check_add_needed_variables function"""

    def test_add_u_v_for_wspd(self):
        """Test adding u and v for wind speed calculation"""
        variables = ["p", "th"]
        result = check_add_needed_variables(variables, ["wspd"])
        assert "u" in result
        assert "v" in result

    def test_add_p_th_for_temp(self):
        """Test adding p and th for temperature calculation"""
        variables = ["z"]
        result = check_add_needed_variables(variables, ["temp"])
        assert "p" in result
        assert "th" in result

    def test_no_duplication(self):
        """Test that variables are not duplicated"""
        variables = ["u", "v", "p"]
        result = check_add_needed_variables(variables, ["wspd"])
        assert result.count("u") == 1
        assert result.count("v") == 1
        assert result.count("p") == 1


# Tests for rename_vars
class TestRenameVars:
    """Test the rename_vars function - basic test only"""

    def test_rename_coordinates(self, dummy_ds):
        """Test renaming coordinates from AROME names to standard names"""
        ds = dummy_ds.rename({"height": "nz", "lat": "latitude", "lon": "longitude"})
        ds2 = rename_vars(ds)
        assert "height" in ds2.coords
        assert "lat" in ds2.coords
        assert "lon" in ds2.coords


# Tests for read_in_arome
class TestReadInArome:
    """Test the read_in_arome function"""

    def test_read_basic_variables(self, monkeypatch):
        """Test reading basic variables"""

        def mock_open_mfdataset(*args, **kwargs):
            return xr.Dataset(
                {
                    "p": (["time", "height", "latitude", "longitude"], np.ones((2, 3, 2, 2)) * 100000),
                    "th": (["time", "height", "latitude", "longitude"], np.ones((2, 3, 2, 2)) * 300)
                },
                coords={
                    "time": pd.date_range('2017-10-15', periods=2, freq='30min'),
                    "height": [1, 2, 3],
                    "latitude": [47.0, 48.0],
                    "longitude": [11.0, 12.0]
                }
            )

        monkeypatch.setattr(xr, "open_mfdataset", mock_open_mfdataset)
        ds, vars_to_calc = read_in_arome(["p", "th"])

        assert "p" in ds
        assert "th" in ds
        assert isinstance(vars_to_calc, set)

    def test_read_with_calculated_vars(self, monkeypatch):
        """Test reading with variables that need to be calculated"""

        def mock_open_mfdataset(*args, **kwargs):
            return xr.Dataset(
                {
                    "p": (["time", "height"], np.ones((2, 3)) * 100000),
                    "th": (["time", "height"], np.ones((2, 3)) * 300)
                },
                coords={"time": [0, 1], "height": [1, 2, 3]}
            )

        monkeypatch.setattr(xr, "open_mfdataset", mock_open_mfdataset)
        ds, vars_to_calc = read_in_arome(["p", "th", "temp", "rho"])

        assert "p" in ds
        assert "th" in ds
        assert "temp" in vars_to_calc
        assert "rho" in vars_to_calc


# Tests for read_2D_variables_AROME
class TestRead2DVariablesArome:
    """Test the read_2D_variables_AROME function"""

    def test_basic_read(self, monkeypatch):
        """Test basic 2D variable reading"""

        def mock_open_dataset(path):
            return xr.Dataset(
                {
                    "hfs": (["time", "latitude", "longitude"], np.ones((10, 5, 5)) * 100)
                },
                coords={
                    "time": pd.date_range('2017-10-15', periods=10, freq='30min'),
                    "latitude": np.linspace(47, 48, 5),
                    "longitude": np.linspace(11, 12, 5)
                }
            )

        monkeypatch.setattr(xr, "open_dataset", mock_open_dataset)

        data = read_2D_variables_AROME(["hfs"], lon=11.5, lat=47.5)
        assert "hfs" in data

    def test_hfs_sign_inversion(self, monkeypatch):
        """Test that sensible heat flux sign is inverted"""

        def mock_open_dataset(path):
            return xr.Dataset(
                {
                    "hfs": (["time", "latitude", "longitude"], np.ones((10, 5, 5)) * 50)
                },
                coords={
                    "time": pd.date_range('2017-10-15', periods=10, freq='30min'),
                    "latitude": np.linspace(47, 48, 5),
                    "longitude": np.linspace(11, 12, 5)
                }
            )

        monkeypatch.setattr(xr, "open_dataset", mock_open_dataset)

        data = read_2D_variables_AROME(["hfs"], lon=11.5, lat=47.5)
        # Sign should be inverted
        assert np.all(data["hfs"].values < 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
