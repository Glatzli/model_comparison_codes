"""
written entirely by Claude AI
Test suite for read_in_hatpro_radiosonde.py

Tests all active functions with mock data to ensure proper functionality.
"""

import fix_win_DLL_loading_issue
import os
# Import the functions to test
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from read_in_hatpro_radiosonde import (calc_vars_hatpro_w_pressure, calc_vars_radiosonde, edit_vars,
                                       read_radiosonde_csv, convert_to_dataset, plot_height_levels,
                                       read_radiosonde_dataset, read_hatpro, interpolate_hatpro_arome)


class TestCalcVarsHatproWPressure:
    """Test the calc_vars_hatpro_w_pressure function"""

    def test_calc_vars_basic(self):
        """Test basic calculation of potential temperature and density"""
        # Create mock dataset
        time = pd.date_range('2017-10-15 12:00', periods=5, freq='30min')
        height = np.array([100, 200, 300, 400, 500])

        ds = xr.Dataset({'temp': (['time', 'height'], np.random.uniform(10, 20, (5, 5))),
            'p': (['time', 'height'], np.random.uniform(900, 1000, (5, 5))),
            'absolute_humidity': (['time', 'height'], np.random.uniform(5, 15, (5, 5)))},
            coords={'time': time, 'height': height})

        result = calc_vars_hatpro_w_pressure(ds)

        # Check that new variables were created
        assert 'th' in result.variables
        assert 'rho' in result.variables
        assert 'q' in result.variables

        # Check that values are reasonable
        assert np.all(result['th'].values > 200)  # Potential temp should be > 280K
        assert np.all(result['rho'].values > 0.8)  # Density should be positive and reasonable
        assert np.all(result['rho'].values < 1.5)
        assert np.all(result['q'].values > 0)  # Specific humidity should be positive
        assert np.all(result['q'].values < 0.03)  # and less than 3%


class TestCalcVarsRadiosonde:
    """Test the calc_vars_radiosonde function"""

    def test_calc_vars_radiosonde_basic(self):
        """Test radiosonde variable calculations"""
        # Create mock dataframe
        df = pd.DataFrame({'temp': np.array([15, 10, 5, 0, -5]), 'p': np.array([1000, 900, 800, 700, 600]),
            'Td': np.array([10, 5, 0, -5, -10]), 'z': np.array([100, 500, 1000, 1500, 2000])})

        result = calc_vars_radiosonde(df)

        # Check that new variables were created
        assert 'th' in result.columns
        assert 'rho' in result.columns
        assert 'q' in result.columns

        # Check that values are reasonable
        assert np.all(result['th'] > 150)  # Potential temp
        assert np.all(result['rho'] > 0.3)  # Density
        assert np.all(result['rho'] < 1.5)
        assert np.all(result['q'] > 0)  # Specific humidity


class TestEditVars:
    """Test the edit_vars function"""

    def test_edit_vars_conversion(self):
        """Test variable editing and unit conversions"""
        df = pd.DataFrame({'time': [0, 1, 2], 'temperature': [288.15, 283.15, 278.15],  # Kelvin
            'dewpoint': [285.15, 280.15, 275.15],  # Kelvin
            'pressure': [100000, 90000, 80000],  # Pa
            'geopotential height': [100, 500, 1000],  # m
            'wind direction': [180, 200, 220], 'windspeed': [5, 10, 15], 'latitude offset': [0, 0, 0],
            'longitude offset': [0, 0, 0]})

        result = edit_vars(df)

        # Check conversions
        assert 'temp' in result.columns
        assert 'Td' in result.columns
        assert 'p' in result.columns
        assert 'z' in result.columns

        # Check values
        assert np.allclose(result['temp'].values, [15, 10, 5])  # Celsius
        assert np.allclose(result['Td'].values, [12, 7, 2])  # Celsius
        assert np.allclose(result['p'].values, [1000, 900, 800])  # hPa

        # Check that original columns were dropped
        assert 'temperature' not in result.columns
        assert 'dewpoint' not in result.columns
        assert 'pressure' not in result.columns
        assert 'time' not in result.columns


class TestReadRadiosondeCsv:
    """Test the read_radiosonde_csv function"""

    def test_read_radiosonde_csv(self):
        """Test reading radiosonde CSV file"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("# Comment line 1\n")
            f.write("# Comment line 2\n")
            f.write("# Comment line 3\n")
            f.write("# Comment line 4\n")
            f.write("# Comment line 5\n")
            f.write(
                "time,temperature,dewpoint,pressure,geopotential height,wind direction,windspeed,latitude offset,longitude offset\n")
            f.write("0,288.15,285.15,100000,100,180,5,0,0\n")
            f.write("1,283.15,280.15,90000,500,200,10,0,0\n")
            f.write("2,278.15,275.15,80000,1000,220,15,0,0\n")
            temp_file = f.name

        try:
            result = read_radiosonde_csv(temp_file)

            # Check that data was read correctly
            assert len(result) == 3
            assert 'temp' in result.columns
            assert 'p' in result.columns
            assert 'z' in result.columns

            # Check conversions
            assert np.allclose(result['temp'].values, [15, 10, 5])
            assert np.allclose(result['p'].values, [1000, 900, 800])
        finally:
            os.unlink(temp_file)


class TestConvertToDataset:
    """Test the convert_to_dataset function"""

    def test_convert_to_dataset_basic(self):
        """Test conversion from DataFrame to xarray Dataset"""
        # Create mock dataframe with index
        df = pd.DataFrame({'temp': [np.nan, 15, 10, 5],  # First value is NaN (will be dropped)
            'p': [np.nan, 1000, 900, 800], 'th': [np.nan, 290, 292, 294], 'q': [np.nan, 0.01, 0.008, 0.006],
            'rho': [np.nan, 1.2, 1.1, 1.0]})
        df.index = [0, 100, 500, 1000]  # Height index

        result = convert_to_dataset(df)

        # Check that it's a dataset
        assert isinstance(result, xr.Dataset)

        # Check that first NaN value was dropped
        assert len(result['height']) == 3

        # Check that all variables are present
        assert 'temp' in result.variables
        assert 'p' in result.variables
        assert 'th' in result.variables

        # Check attributes
        assert 'units' in result['th'].attrs
        assert 'units' in result['q'].attrs
        assert 'units' in result['rho'].attrs


class TestPlotHeightLevels:
    """Test the plot_height_levels function"""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_height_levels(self, mock_savefig, mock_show):
        """Test plotting of height levels"""
        # Create mock height arrays
        arome_heights = np.linspace(600, 3000, 50)
        icon_heights = np.linspace(600, 3000, 45)
        um_heights = np.linspace(600, 3000, 40)
        wrf_heights = np.linspace(600, 3000, 48)
        radio_heights = np.array([600, 800, 1000, 1500, 2000, 2500, 3000])
        hatpro_heights = np.array([600, 700, 900, 1200, 1600, 2100, 2700])

        # Should not raise any errors
        plot_height_levels(arome_heights, icon_heights, um_heights, wrf_heights, radio_heights, hatpro_heights)

        # Check that plot functions were called
        mock_show.assert_called_once()


class TestReadRadiosondeDataset:
    """Test the read_radiosonde_dataset function"""

    def test_read_radiosonde_dataset_direct(self):
        """Test reading radiosonde dataset with direct height"""
        # Create temporary dataset
        height_idx = np.arange(10)
        z_values = np.array([600, 700, 850, 1000, 1200, 1500, 1800, 2100, 2500, 3000])

        ds = xr.Dataset(
            {'temp': (['height'], np.random.uniform(5, 15, 10)), 'p': (['height'], np.linspace(1000, 700, 10)),
                'z': (['height'], z_values), 'th': (['height'], np.random.uniform(285, 295, 10)),
                'q': (['height'], np.random.uniform(0.005, 0.015, 10)), 'rho': (['height'], np.linspace(1.2, 0.9, 10))},
            coords={'height': height_idx})

        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as f:
            temp_file = f.name

        try:
            ds.to_netcdf(temp_file)

            with patch('confg.radiosonde_dataset', temp_file):
                result = read_radiosonde_dataset(height_as_z_coord="direct")

                # Check that height was set to z values
                assert np.allclose(result['height'].values, z_values)
                assert result['height'].attrs['units'] == 'm'
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_read_radiosonde_dataset_above_terrain(self):
        """Test reading radiosonde dataset with height above terrain"""
        # Create temporary dataset
        height_idx = np.arange(10)
        z_values = np.array([600, 700, 850, 1000, 1200, 1500, 1800, 2100, 2500, 3000])

        ds = xr.Dataset(
            {'temp': (['height'], np.random.uniform(5, 15, 10)), 'p': (['height'], np.linspace(1000, 700, 10)),
                'z': (['height'], z_values), 'th': (['height'], np.random.uniform(285, 295, 10)),
                'q': (['height'], np.random.uniform(0.005, 0.015, 10)), 'rho': (['height'], np.linspace(1.2, 0.9, 10))},
            coords={'height': height_idx})

        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as f:
            temp_file = f.name

        try:
            ds.to_netcdf(temp_file)

            with patch('confg.radiosonde_dataset', temp_file):
                result = read_radiosonde_dataset(height_as_z_coord="above_terrain")

                # Check that height starts at 1m
                assert result['height'].values[0] == 1
                # Check that height differences are preserved
                expected_heights = z_values - z_values[0] + 1
                assert np.allclose(result['height'].values, expected_heights)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestReadHatpro:
    """Test the read_hatpro function"""

    def test_read_hatpro_temp(self):
        """Test reading HATPRO temperature data"""
        # Create temporary CSV file with HATPRO format
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_temp.csv') as f:
            f.write("rawdate;" + ";".join([f"h{i:02d}" for i in range(1, 40)]) + "\n")
            # Add some data rows
            for hour in range(12, 15):
                time_str = f"2017-10-15 {hour:02d}:00:00"
                values = ";".join([str(280 + i * 0.5 + np.random.random()) for i in range(39)])
                f.write(f"{time_str};{values}\n")
            temp_file = f.name

        try:
            # Mock the hatpro_vertical_levels config
            with patch('confg.hatpro_vertical_levels', {"height": [str(i * 50) for i in range(1, 40)]}):
                result = read_hatpro(temp_file)

                # Check structure
                assert isinstance(result, xr.Dataset)
                assert 'th' in result.variables
                assert 'time' in result.coords
                assert 'height_level' in result.coords

                # Check dimensions
                assert len(result['height_level']) == 39
        finally:
            os.unlink(temp_file)

    def test_read_hatpro_humidity(self):
        """Test reading HATPRO humidity data"""
        # Create temporary CSV file with HATPRO format
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_humidity.csv') as f:
            f.write("rawdate;" + ";".join([f"h{i:02d}" for i in range(1, 40)]) + "\n")
            # Add some data rows
            for hour in range(12, 15):
                time_str = f"2017-10-15 {hour:02d}:00:00"
                values = ";".join([str(8 - i * 0.1 + np.random.random()) for i in range(39)])
                f.write(f"{time_str};{values}\n")
            temp_file = f.name

        try:
            # Mock the hatpro_vertical_levels config
            with patch('confg.hatpro_vertical_levels', {"height": [str(i * 50) for i in range(1, 40)]}):
                result = read_hatpro(temp_file)

                # Check structure
                assert isinstance(result, xr.Dataset)
                assert 'humidity' in result.variables
        finally:
            os.unlink(temp_file)


class TestInterpolateHatproArome:
    """Test the interpolate_hatpro_arome function"""

    def test_interpolate_hatpro_arome_basic(self):
        """Test interpolation of HATPRO to AROME levels"""
        # Create mock HATPRO dataset
        time = pd.date_range('2017-10-15 12:00', '2017-10-16 12:00', freq='10min')
        height_hatpro = np.array([50, 100, 200, 350, 550, 800, 1100, 1450, 1850])

        hatpro = xr.Dataset({'temp': (['time', 'height'], np.random.uniform(10, 20, (len(time), len(height_hatpro)))),
            'humidity': (['time', 'height'], np.random.uniform(5, 15, (len(time), len(height_hatpro))))},
            coords={'time': time, 'height': height_hatpro})

        # Create mock AROME dataset
        time_arome = pd.date_range('2017-10-15 12:00', '2017-10-16 12:00', freq='30min')
        height_arome = np.array([5.1, 30, 60, 100, 150, 220, 310, 420, 560, 730, 930, 1160, 1430])

        arome = xr.Dataset(
            {'p': (['time', 'height'], np.random.uniform(900, 1000, (len(time_arome), len(height_arome))))},
            coords={'time': time_arome, 'height': height_arome})

        # Mock config values
        mock_all_points = {'ibk_uni': {'height': 612}, 'ibk_villa': {'height': 579}}
        with patch('confg.ALL_POINTS', mock_all_points), patch('confg.hatpro_calced_vars',
                                                               tempfile.mktemp(suffix='.nc')):
            # Should not raise errors
            interpolate_hatpro_arome(hatpro, arome)


class TestIntegration:
    """Integration tests for the full workflow"""

    def test_full_radiosonde_workflow(self):
        """Test complete radiosonde processing workflow"""
        # Create mock CSV data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("# Comment\n" * 5)
            f.write(
                "time,temperature,dewpoint,pressure,geopotential height,wind direction,windspeed,latitude offset,longitude offset\n")
            for i in range(10):
                temp_k = 288.15 - i * 2
                td_k = temp_k - 5
                p_pa = 100000 - i * 5000
                z = 100 + i * 200
                f.write(f"{i},{temp_k},{td_k},{p_pa},{z},180,5,0,0\n")
            temp_file = f.name

        try:
            # Read CSV
            df = read_radiosonde_csv(temp_file)
            assert len(df) == 10

            # Calculate variables
            df_calc = calc_vars_radiosonde(df)
            assert 'th' in df_calc.columns
            assert 'rho' in df_calc.columns
            assert 'q' in df_calc.columns

            # Convert to dataset
            df_calc.index = df_calc['z'].values
            ds = convert_to_dataset(df_calc)
            assert isinstance(ds, xr.Dataset)

        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])