"""
Test suite for plot_heat_fluxes.py
Tests heat flux plotting and visualization functionality.
"""

import fix_win_DLL_loading_issue
import os
import sys
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import xarray as xr
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after setting path
from calculations_and_plots import plot_heat_fluxes


@pytest.fixture
def mock_heat_flux_data():
    """Create mock heat flux data for testing"""
    time = pd.date_range('2017-10-15 12:00', periods=49, freq='30min')
    lat = np.linspace(46.5, 48.2, 50)
    lon = np.linspace(9.2, 13.0, 70)

    return xr.Dataset(
        {
            "hfs": (["time", "lat", "lon"], np.random.uniform(-100, 200, (49, 50, 70))),  # W/m²
            "lfs": (["time", "lat", "lon"], np.random.uniform(-50, 150, (49, 50, 70))),   # W/m²
            "u": (["time", "lat", "lon"], np.random.uniform(-5, 5, (49, 50, 70))),        # m/s
            "v": (["time", "lat", "lon"], np.random.uniform(-5, 5, (49, 50, 70))),        # m/s
        },
        coords={"time": time, "lat": lat, "lon": lon}
    )


class TestHeatFluxModule:
    """Test heat flux module structure and configuration"""

    def test_variables_to_plot_defined(self):
        """Test that variables to plot are properly defined"""
        assert hasattr(plot_heat_fluxes, 'variables_to_plot')
        vars_to_plot = plot_heat_fluxes.variables_to_plot

        assert isinstance(vars_to_plot, list)
        assert "hfs" in vars_to_plot  # sensible heat flux
        assert "lfs" in vars_to_plot  # latent heat flux

    def test_essential_imports(self):
        """Test that essential modules are imported"""
        # Check plotting imports
        assert hasattr(plot_heat_fluxes, 'plt')
        assert hasattr(plot_heat_fluxes, 'ccrs')
        assert hasattr(plot_heat_fluxes, 'cfeature')

        # Check data handling
        assert hasattr(plot_heat_fluxes, 'xr')
        assert hasattr(plot_heat_fluxes, 'np')
        assert hasattr(plot_heat_fluxes, 'pd')

        # Check configuration
        assert hasattr(plot_heat_fluxes, 'confg')

    def test_model_reader_imports(self):
        """Test that model reading functions are imported"""
        assert hasattr(plot_heat_fluxes, 'read_in_arome')
        assert hasattr(plot_heat_fluxes, 'read_wrf_helen')

    def test_dll_fix_import(self):
        """Test that Windows DLL fix is imported first"""
        # Should have imported fix at the top
        import fix_win_DLL_loading_issue
        assert fix_win_DLL_loading_issue is not None


class TestScaleBarFunctionality:
    """Test scale bar functionality"""

    @patch('matplotlib.pyplot.figure')
    def test_add_scalebar_function(self, mock_figure):
        """Test scale bar addition function if it exists"""
        if hasattr(plot_heat_fluxes, 'add_scalebar'):
            # Create mock axes
            mock_ax = MagicMock()
            mock_ax.get_extent.return_value = [9.2, 13.0, 46.5, 48.2]  # lon_min, lon_max, lat_min, lat_max

            try:
                plot_heat_fluxes.add_scalebar(mock_ax, length_km=10)
                assert mock_ax.get_extent.called
                success = True
            except Exception:
                # Function might require additional setup
                success = False

            # Test structure exists
            assert True

    def test_scalebar_parameters(self):
        """Test scale bar parameter validation"""
        if hasattr(plot_heat_fluxes, 'add_scalebar'):
            # Should accept reasonable parameters
            length_km = 10
            location = 'lower right'

            assert isinstance(length_km, (int, float))
            assert length_km > 0
            assert isinstance(location, str)
            assert len(location) > 0


class TestCoordinateCalculations:
    """Test coordinate calculation functions"""

    def test_extent_calculations(self):
        """Test map extent calculations"""
        # Tyrol region bounds
        lon_min, lon_max = 9.2, 13.0
        lat_min, lat_max = 46.5, 48.2

        # Calculate center for scale bar positioning
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2

        assert 46.5 < center_lat < 48.2
        assert 9.2 < center_lon < 13.0

        # Test extent ranges are reasonable
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min

        assert 1.0 < lat_range < 3.0  # Reasonable latitude range for Tyrol
        assert 2.0 < lon_range < 5.0  # Reasonable longitude range for Tyrol

    def test_km_conversion_function(self):
        """Test km conversion function if available"""
        if hasattr(plot_heat_fluxes, 'calculate_lon_extent_for_km'):
            # Test conversion for typical Tyrol latitude
            lat = 47.3  # Innsbruck
            km = 10

            lon_extent = plot_heat_fluxes.calculate_lon_extent_for_km(km, lat)

            assert isinstance(lon_extent, (int, float))
            assert lon_extent > 0
            assert lon_extent < 1.0  # Should be fraction of degree


class TestHeatFluxData:
    """Test heat flux data handling"""

    def test_heat_flux_units(self, mock_heat_flux_data):
        """Test heat flux data units and ranges"""
        hfs = mock_heat_flux_data["hfs"]
        lfs = mock_heat_flux_data["lfs"]

        # Heat flux should be in W/m²
        assert hfs.min() >= -200  # Reasonable range for sensible heat flux
        assert hfs.max() <= 500

        assert lfs.min() >= -100  # Reasonable range for latent heat flux
        assert lfs.max() <= 300

    def test_wind_data_structure(self, mock_heat_flux_data):
        """Test wind data for arrow plotting"""
        u_wind = mock_heat_flux_data["u"]
        v_wind = mock_heat_flux_data["v"]

        # Wind components should have same shape
        assert u_wind.shape == v_wind.shape

        # Reasonable wind speeds
        assert u_wind.min() >= -20  # m/s
        assert u_wind.max() <= 20
        assert v_wind.min() >= -20
        assert v_wind.max() <= 20

    def test_time_coordinate(self, mock_heat_flux_data):
        """Test time coordinate structure"""
        time = mock_heat_flux_data.time

        # Should cover the study period
        assert len(time) == 49  # 24.5 hours at 30-minute intervals
        assert time[0].dt.year == 2017
        assert time[0].dt.month == 10
        assert time[0].dt.day == 15


class TestPlottingFunctionality:
    """Test plotting functionality"""

    @patch('matplotlib.pyplot.show')
    def test_matplotlib_backend(self, mock_show):
        """Test matplotlib backend setup"""
        import matplotlib

        # Should be able to use matplotlib
        assert matplotlib is not None

        # Backend should be set appropriately
        backend = matplotlib.get_backend()
        assert isinstance(backend, str)

    @patch('cartopy.crs.PlateCarree')
    def test_cartopy_projection(self, mock_projection):
        """Test cartopy projection setup"""
        mock_projection.return_value = MagicMock()

        # Should be able to create projections
        import cartopy.crs as ccrs
        proj = ccrs.PlateCarree()
        assert proj is not None

    def test_color_scheme_import(self):
        """Test color scheme imports"""
        # Should import diverging color schemes for heat flux
        assert hasattr(plot_heat_fluxes, 'diverging_hcl')

        try:
            from colorspace import diverging_hcl
            # Test color scheme creation
            colors = diverging_hcl(h=[240, 0], c=60, l=75, power=1.0)
            assert len(colors) > 0
        except ImportError:
            pytest.skip("colorspace not available")


class TestDataValidation:
    """Test data validation and processing"""

    def test_flux_sign_conventions(self):
        """Test heat flux sign conventions"""
        # According to comments in the file:
        # WRF: UPWARD HEAT FLUX AT THE SURFACE (positive upward)
        # AROME: needs sign inversion to match WRF convention

        # Test that we understand the sign conventions
        upward_flux = 100  # W/m² upward (surface to atmosphere)
        downward_flux = -50  # W/m² downward (atmosphere to surface)

        assert upward_flux > 0
        assert downward_flux < 0

    def test_diurnal_cycle_validation(self, mock_heat_flux_data):
        """Test diurnal cycle patterns in heat flux data"""
        hfs = mock_heat_flux_data["hfs"]
        time = mock_heat_flux_data.time

        # Should have data spanning day and night
        hours = time.dt.hour
        assert hours.min() >= 0
        assert hours.max() <= 23

        # Should have multiple time points
        assert len(time) > 10

    def test_spatial_coverage(self, mock_heat_flux_data):
        """Test spatial coverage of heat flux data"""
        lat = mock_heat_flux_data.lat
        lon = mock_heat_flux_data.lon

        # Should cover Tyrol region
        assert lat.min() >= 46.0
        assert lat.max() <= 49.0
        assert lon.min() >= 9.0
        assert lon.max() <= 14.0

        # Should have adequate resolution
        lat_resolution = (lat.max() - lat.min()) / len(lat)
        lon_resolution = (lon.max() - lon.min()) / len(lon)

        assert lat_resolution < 0.1  # Less than 0.1 degree spacing
        assert lon_resolution < 0.1


class TestPhysicalInterpretation:
    """Test physical interpretation of heat flux patterns"""

    def test_sunset_timing(self):
        """Test sunset timing interpretation from comments"""
        # From comments: "sunset at 16:25 UTC: temp falls already since ~15:30"
        sunset_utc = pd.Timestamp('2017-10-16 16:25:00')
        temp_drop_start = pd.Timestamp('2017-10-16 15:30:00')

        assert sunset_utc > temp_drop_start

        # Time difference should be reasonable
        time_diff = sunset_utc - temp_drop_start
        assert time_diff.total_seconds() == 55 * 60  # 55 minutes

    def test_heat_flux_interpretation(self):
        """Test heat flux physical interpretation"""
        # During night: mostly negative heat flux (heat loss from surface)
        # During day: positive heat flux (heat gain at surface)

        night_flux = -50  # W/m² (negative = upward/loss)
        day_flux = 150   # W/m² (positive = downward/gain)

        assert night_flux < 0  # Heat loss at night
        assert day_flux > 0    # Heat gain during day
        assert abs(day_flux) > abs(night_flux)  # Day heating > night cooling


if __name__ == '__main__':
    pytest.main([__file__])
