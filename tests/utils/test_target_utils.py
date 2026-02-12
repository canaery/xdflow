"""Tests for target utils."""

import numpy as np
import xarray as xr

from xdflow.utils.target_utils import extract_target_array, resolve_target_coords


def _make_data():
    data = xr.DataArray(
        np.random.randn(4, 2),
        dims=("trial", "feature"),
        coords={
            "trial": np.arange(4),
            "feature": np.arange(2),
            "target_a": ("trial", np.arange(4)),
            "target_b": ("trial", np.arange(10, 14)),
        },
    )
    return data


def test_resolve_target_coords_list():
    data = _make_data()
    targets = resolve_target_coords(["target_a", "target_b"], data)
    assert targets == ["target_a", "target_b"]


def test_resolve_target_coords_pattern():
    data = _make_data()
    targets = resolve_target_coords("target_*", data)
    assert set(targets) == {"target_a", "target_b"}


def test_extract_target_array_stack():
    data = _make_data()
    arr = extract_target_array(["target_a", "target_b"], data, validate=False)
    assert arr.shape == (4, 2)
    assert np.all(arr[:, 0] == np.arange(4))
