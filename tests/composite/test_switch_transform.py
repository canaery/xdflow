"""
Tests for SwitchTransform class.
"""

import numpy as np
import pytest
import xarray as xr

from xdflow.composite import SwitchTransform
from xdflow.core.data_container import DataContainer, TransformError


class MockTransform:
    """Simple mock transform for testing."""

    is_stateful = False
    input_dims = ("trial", "channel", "time")
    output_dims = ("trial", "features")

    def __init__(self, name):
        self.name = name
        self.fitted = False

    def fit(self, container, **kwargs):
        self.fitted = True
        return self

    def transform(self, container, **kwargs):
        result = container.copy()
        result.data.attrs.setdefault("transform_history", []).append(self.name)
        return result

    def get_expected_output_dims(self, input_dims):
        return self.output_dims


class DimTransform:
    """Transform that emits a specific feature dim for testing renaming."""

    is_stateful = False
    input_dims = ("trial", "channel")
    output_dims = ("trial", "features")

    def __init__(self, name):
        self.name = name

    def fit(self, container, **kwargs):
        return self

    def transform(self, container, **kwargs):
        data = container.data.sum(dim="channel")
        data = data.expand_dims({"features": np.arange(1)}, axis=-1)
        return DataContainer(data)

    def get_expected_output_dims(self, input_dims):
        return self.output_dims


@pytest.fixture
def sample_container():
    data = xr.DataArray(
        np.random.rand(10, 5, 100),
        dims=["trial", "channel", "time"],
        coords={
            "trial": np.arange(10),
            "channel": np.arange(5),
            "time": np.arange(100),
            "group": ("trial", ["group_a"] * 5 + ["group_b"] * 5),
        },
        attrs={"data_history": []},
    )
    return DataContainer(data)


def test_switch_transform_initialization():
    choices = [("transform_a", MockTransform("a")), ("transform_b", MockTransform("b"))]

    switch = SwitchTransform(choices=choices)

    assert set(switch.transform_from_name.keys()) == {"transform_a", "transform_b"}
    assert not switch.is_stateful


def test_switch_transform_accepts_dict_choices(sample_container):
    choices = {
        "transform_a": MockTransform("a"),
        "transform_b": MockTransform("b"),
    }

    switch = SwitchTransform(choices=choices, choose="transform_b")

    assert set(switch.transform_from_name.keys()) == {"transform_a", "transform_b"}
    result = switch.fit_transform(sample_container)
    assert "b" in result.data.attrs["transform_history"]


def test_switch_transform_empty_choices():
    with pytest.raises(ValueError, match="At least one choice must be provided"):
        SwitchTransform(choices=[])


def test_switch_transform_missing_choose_kwarg(sample_container):
    choices = [("transform_a", MockTransform("a")), ("transform_b", MockTransform("b"))]
    switch = SwitchTransform(choices=choices)

    with pytest.raises(TransformError, match="'choose' keyword argument must be provided"):
        switch.fit_transform(sample_container)


def test_switch_transform_runtime_selection(sample_container):
    choices = [("transform_a", MockTransform("a")), ("transform_b", MockTransform("b"))]
    switch = SwitchTransform(choices=choices)

    result_a = switch.fit_transform(sample_container, choose="transform_a")
    assert "a" in result_a.data.attrs["transform_history"]
    assert "b" not in result_a.data.attrs["transform_history"]

    result_b = switch.fit_transform(sample_container, choose="transform_b")
    assert "b" in result_b.data.attrs["transform_history"]
    assert "a" not in result_b.data.attrs["transform_history"]


def test_switch_transform_invalid_choice(sample_container):
    choices = [("transform_a", MockTransform("a")), ("transform_b", MockTransform("b"))]
    switch = SwitchTransform(choices=choices)

    with pytest.raises(TransformError, match="Selected choice 'invalid' is not a valid option"):
        switch.fit_transform(sample_container, choose="invalid")


def test_switch_transform_statefulness():
    class StatefulMock(MockTransform):
        is_stateful = True

    choices = [("stateless", MockTransform("stateless")), ("stateful", StatefulMock("stateful"))]
    switch = SwitchTransform(choices=choices)

    assert switch.is_stateful


def test_switch_transform_children_property():
    choices = [
        ("transform_a", MockTransform("a")),
        ("transform_b", MockTransform("b")),
        ("transform_c", MockTransform("c")),
    ]

    switch = SwitchTransform(choices=choices)

    children = list(switch.children)
    assert len(children) == 3
    assert all(choice in children for _, choice in choices)


def test_switch_transform_repr():
    choices = [("transform_a", MockTransform("a")), ("transform_b", MockTransform("b"))]
    switch = SwitchTransform(choices=choices)

    repr_str = repr(switch)
    assert "SwitchTransform" in repr_str
    assert "transform_a" in repr_str
    assert "transform_b" in repr_str


def test_switch_transform_renames_output_dim():
    feature_container = DataContainer(
        xr.DataArray(
            np.random.rand(4, 3),
            dims=("trial", "channel"),
            coords={"trial": np.arange(4), "channel": np.arange(3)},
        )
    )
    choices = [("dim_transform", DimTransform("d"))]
    switch = SwitchTransform(choices=choices, from_dim="features", to_dim="renamed_feature")

    result = switch.fit_transform(feature_container, choose="dim_transform")
    assert result.data.dims == ("trial", "renamed_feature")
