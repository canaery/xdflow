"""Tests for sklearn CV adapter."""

import numpy as np
import xarray as xr

from xdflow.core.data_container import DataContainer
from xdflow.cv.sklearn_adapter import SklearnCVAdapter, set_cv_container


class FakeCV:
    """Minimal CV implementation for adapter testing."""

    def _split_holdout(self, container: DataContainer):
        trials = container.data.trial.values
        return trials, []

    def _get_splits(self, container: DataContainer, train_val_indices):
        trials = list(train_val_indices)
        mid = len(trials) // 2
        yield trials[:mid], trials[mid:]


def test_sklearn_adapter_split_indices():
    """Adapter should map trial labels to 0-based indices."""
    data = xr.DataArray(
        np.random.randn(4, 2),
        dims=("trial", "feature"),
        coords={"trial": [10, 11, 12, 13], "feature": [0, 1]},
    )
    container = DataContainer(data)

    adapter = SklearnCVAdapter(FakeCV())

    with set_cv_container(container):
        splits = list(adapter.split(np.zeros((4, 2))))

    assert len(splits) == 1
    train_idx, val_idx = splits[0]
    assert np.array_equal(train_idx, np.array([0, 1]))
    assert np.array_equal(val_idx, np.array([2, 3]))
