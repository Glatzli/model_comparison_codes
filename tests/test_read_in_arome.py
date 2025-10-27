import sys
sys.path.append("D:/MSc_Arbeit/model_comparison_codes")
import confg
import pytest
import xarray as xr
import numpy as np

from read_in_arome import (
    convert_calc_variables,
    check_add_needed_variables,
    read_in_arome,
    rename_vars,
    read_in_arome_fixed_point,
    read_in_arome_fixed_time,
    read_2D_variables_AROME,
    save_arome_topography,
    extract_3d_variable_define_2D
)

# This fixture creates a fake xarray dataset for testing
@pytest.fixture
def dummy_ds():
    time = np.arange(2)
    height = np.arange(3)
    lat = np.linspace(47, 48, 2)
    lon = np.linspace(11, 12, 2)
    ds = xr.Dataset(
        {
            "u": (["time", "height", "lat", "lon"], np.ones((2, 3, 2, 2))),
            "v": (["time", "height", "lat", "lon"], np.ones((2, 3, 2, 2))),
            "p": (["time", "height", "lat", "lon"], np.ones((2, 3, 2, 2)) * 100000),
            "th": (["time", "height", "lat", "lon"], np.ones((2, 3, 2, 2)) * 300),
            "z": (["time", "height", "lat", "lon"], np.ones((2, 3, 2, 2)) * 1000),
        },
        coords={
            "time": time,
            "height": height,
            "lat": lat,
            "lon": lon
        }
    )
    return ds

# Test if convert_calc_variables adds "temp" and "rho" to the dataset
def test_convert_calc_variables(dummy_ds):
    ds = convert_calc_variables(dummy_ds, vars_to_calc=["temp", "rho"])
    assert "temp" in ds
    assert "rho" in ds

# Test if check_add_needed_variables adds "u" and "v" when "udir" is needed
def test_check_add_needed_variables():
    variables = ["p", "th"]
    result = check_add_needed_variables(variables, ["udir"])
    assert "u" in result and "v" in result

# Test if rename_vars renames coordinates back to "height", "lat", "lon"
def test_rename_vars(dummy_ds):
    ds = dummy_ds.rename({"height": "nz", "lat": "latitude", "lon": "longitude"})
    ds2 = rename_vars(ds)
    assert "height" in ds2.coords
    assert "lat" in ds2.coords
    assert "lon" in ds2.coords

# Test read_in_arome with a mock for open_mfdataset
def test_read_in_arome(monkeypatch):
    def mock_open_mfdataset(*args, **kwargs):
        return xr.Dataset({"p": (["time", "height", "latitude", "longitude"], np.ones((2, 3))),
                           "th": (["time", "height", "latitude", "longitude"], np.ones((2, 3)))},
                          coords={"time": [0, 1], "height": [0, 1, 2], "latitude": [47.0, 47.01, 47.02],
                                  "longitude": [13.0, 13.01, 13.02]})
    #def mock_open_mfdataset(*args, **kwargs):
    #    # This simulates loading two NetCDF files
    #    arome_paths = [confg.dir_3D_AROME + f"/AROME_Geosphere_20171015T1200Z_CAP02_3D_30min_1km_best_p.nc",
    #                  confg.dir_3D_AROME + f"/AROME_Geosphere_20171015T1200Z_CAP02_3D_30min_1km_best_th.nc"]
    #   return xr.open_mfdataset(arome_paths, combine="by_coords", data_vars='minimal',
    #                           coords='minimal', compat='override', decode_timedelta=True)
    monkeypatch.setattr(xr, "open_mfdataset", mock_open_mfdataset)
    ds, vars_to_calc = read_in_arome(["p", "th"])
    assert "p" in ds
    assert "th" in ds

# Test read_in_arome_fixed_point with a mock for read_in_arome
def test_read_in_arome_fixed_point(monkeypatch):
    def mock_read_in_arome(*args, **kwargs):
        ds = xr.Dataset({"p": (["time", "height"], np.ones((2, 3))),
                         "th": (["time", "height"], np.ones((2, 3))),
                         "u": (["time", "height"], np.ones((2, 3))),
                         "v": (["time", "height"], np.ones((2, 3))),
                         "z": (["time", "height"], np.ones((2, 3)))},
                        coords={"time": [0, 1], "height": [0, 1, 2]})
        return ds, ["udir", "wspd"]
    monkeypatch.setattr("read_in_arome.read_in_arome", mock_read_in_arome)
    ds = read_in_arome_fixed_point(lat=47.2, lon=11.3, variables=["z", "p", "th", "udir", "wspd"])
    assert "udir" in ds
    assert "wspd" in ds

# Test read_in_arome_fixed_time with a mock for read_in_arome
def test_read_in_arome_fixed_time(monkeypatch):
    def mock_read_in_arome(*args, **kwargs):
        ds = xr.Dataset({"p": (["time", "height"], np.ones((2, 3))),
                         "th": (["time", "height"], np.ones((2, 3))),
                         "z": (["time", "height"], np.ones((2, 3)))},
                        coords={"time": [0, 1], "height": [0, 1, 2], "latitude": [47, 48], "longitude": [11, 12]})
        return ds, []
    monkeypatch.setattr("read_in_arome.read_in_arome", mock_read_in_arome)
    ds = read_in_arome_fixed_time(day=15, hour=12, min=0)
    assert "p" in ds
    assert "th" in ds

# Test read_2D_variables_AROME with a mock for open_dataset
def test_read_2D_variables_AROME(monkeypatch):
    def mock_open_dataset(*args, **kwargs):
        return xr.Dataset({"hfs": (["time", "lat", "lon"], np.ones((2, 2, 2)))},
                         coords={"time": [0, 1], "lat": [47, 48], "lon": [11, 12]})
    monkeypatch.setattr(xr, "open_dataset", mock_open_dataset)
    data = read_2D_variables_AROME(["hfs"], lon=11, lat=47)
    assert "hfs" in data

# Test save_arome_topography with a dummy class
def test_save_arome_topography(monkeypatch, tmp_path):
    class DummyArome(xr.Dataset):
        def compute(self):
            return self
        def rename(self, mapping):
            return self
        def to_netcdf(self, path, mode, format):
            pass
        @property
        def lat(self):
            return xr.DataArray([47, 48])
        @property
        def lon(self):
            return xr.DataArray([11, 12])
        def round(self, n):
            return self
        def rio(self):
            class Rio:
                def to_raster(self, path):
                    pass
            return Rio()
    arome3d = DummyArome()
    save_arome_topography(arome3d)

# Test extract_3d_variable_define_2D with a mock for read_in_arome_fixed_time
def test_extract_3d_variable_define_2D(monkeypatch):
    def mock_read_in_arome_fixed_time(*args, **kwargs):
        ds = xr.Dataset({"u": (["time", "lat", "lon"], np.ones((1, 2, 2))),
                         "v": (["time", "lat", "lon"], np.ones((1, 2, 2))),
                         "height": (["height"], [1])},
                        coords={"time": [0], "lat": [47, 48], "lon": [11, 12], "height": [1]})
        return ds
    monkeypatch.setattr("read_in_arome.read_in_arome_fixed_time", mock_read_in_arome_fixed_time)
    extract_3d_variable_define_2D(variables=["u", "v"])
