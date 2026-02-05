import pytest
import numpy as np
from calculations_and_plots.plot_vertical_profiles import _create_vhd_area_trace

class TestVHDTrace:

    @pytest.fixture
    def mock_data(self):
        return {
            "th_values": np.array([280, 285, 290, 295]),
            "height_values": np.array([0, 1000, 2000, 3000]),
            "th_hafelekar": 290,
            "point_height": 600,
            "color": "rgb(255, 0, 0)",
            "name": "TestModel",
            "legendgroup": "TestGroup"
        }

    def test_vhd_area_correctness(self, mock_data):
        "Ensure VHD area is calculated correctly for valid data"
        trace = _create_vhd_area_trace(**mock_data)
        assert trace.x[0] == mock_data["th_values"].min()
        assert trace.y[0] == 0
        assert trace.y[1] == mock_data["height_values"].max()

    def test_empty_data(self):
        "Ensure empty data returns an empty trace"
        trace = _create_vhd_area_trace(
            th_values=np.array([]),
            height_values=np.array([]),
            th_hafelekar=290,
            point_height=600,
            color="rgb(255, 0, 0)",
            name="TestModel",
            legendgroup="TestGroup"
        )
        assert len(trace.x) == 0
        assert len(trace.y) == 0

    def test_partial_data(self):
        "Ensure partial data (e.g., no heights above Hafelekar) returns empty trace"
        trace = _create_vhd_area_trace(
            th_values=np.array([280, 285, 290]),
            height_values=np.array([0, 500, 1000]),
            th_hafelekar=290,
            point_height=2000,
            color="rgb(255, 0, 0)",
            name="TestModel",
            legendgroup="TestGroup"
        )
        assert len(trace.x) == 0
        assert len(trace.y) == 0

    def test_color_transparency(self, mock_data):
        "Ensure color transparency is applied correctly"
        trace = _create_vhd_area_trace(**mock_data)
        assert "rgba" in trace.fillcolor
        assert trace.fillcolor.endswith("0.1)")
