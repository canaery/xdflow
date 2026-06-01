import numpy as np
import pytest
import xarray as xr

from xdflow.core import DataContainer
from xdflow.transforms.domain_adaptation import CoralAligner, ProcrustesAligner, SAAligner


def _domain_container() -> DataContainer:
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape(4, 3),
        dims=("trial", "feature"),
        coords={
            "trial": np.arange(4),
            "feature": ["f0", "f1", "f2"],
            "domain": ("trial", ["source", "source", "target", "target"]),
        },
        attrs={
            "data_history": [{"class": "PreviousTransform", "params": {}}],
            "experiment": "attrs-are-provenance",
        },
    )
    return DataContainer(data)


@pytest.mark.parametrize(
    ("aligner", "adapted_params", "target_params"),
    [
        (
            ProcrustesAligner(target_coord="domain", sample_dim="trial", group_coord="domain", target_group="target"),
            {"centroid": np.zeros(3), "rotation_matrix": np.eye(3), "scale": 1.0},
            {"centroid": np.zeros(3)},
        ),
        (
            CoralAligner(sample_dim="trial", group_coord="domain", target_group="target"),
            {"centroid": np.zeros(3), "transform_matrix": np.eye(3)},
            {"centroid": np.zeros(3)},
        ),
        (
            SAAligner(sample_dim="trial", group_coord="domain", target_group="target"),
            {"centroid": np.zeros(3), "transform_matrix": np.eye(3)},
            {"centroid": np.zeros(3)},
        ),
    ],
)
def test_single_target_aligners_preserve_attrs(aligner, adapted_params, target_params):
    container = _domain_container()

    result = aligner._adapted_transform(container, adapted_params, target_params)

    assert result.data.attrs["experiment"] == "attrs-are-provenance"
    assert [entry["class"] for entry in result.data.attrs["data_history"]] == ["PreviousTransform"]
    assert result.data.attrs["data_history"] is not container.data.attrs["data_history"]
