"""
Tests for GroupApplyTransform class.
"""

import numpy as np
import pytest
import xarray as xr
from sklearn.preprocessing import LabelEncoder

from xdflow.composite import GroupApplyTransform
from xdflow.core.base import Predictor, Transform
from xdflow.core.data_container import DataContainer, TransformError


class MockTransform(Transform):
    """Simple mock transform for testing."""

    is_stateful = False
    input_dims = ("trial", "channel", "time")
    output_dims = ("trial", "features")

    def __init__(self, name, multiplier=1.0):
        super().__init__()
        self.name = name
        self.multiplier = multiplier
        self.fitted = False
        self.fit_data_shape = None

    def _fit(self, container, **kwargs):
        self.fitted = True
        self.fit_data_shape = container.data.shape
        return self

    def _transform(self, container, **kwargs):
        data = container.data * self.multiplier
        flattened = data.stack(features=("channel", "time"))
        result_data = flattened.transpose("trial", "features")
        return DataContainer(result_data)

    def get_expected_output_dims(self, input_dims):
        return self.output_dims

    def get_params(self, deep=False):
        return {"name": self.name, "multiplier": self.multiplier}


class MockStatefulTransform(MockTransform):
    """Mock stateful transform for testing."""

    is_stateful = True

    def __init__(self, name, multiplier=1.0):
        super().__init__(name, multiplier)
        self.mean_value = None

    def _fit(self, container, **kwargs):
        self.fitted = True
        self.fit_data_shape = container.data.shape
        self.mean_value = float(container.data.mean())
        return self

    def _transform(self, container, **kwargs):
        if self.mean_value is None:
            raise ValueError("Transform must be fitted before transform")
        centered_data = container.data - self.mean_value
        flattened = (centered_data * self.multiplier).stack(features=("channel", "time")).transpose("trial", "features")
        return DataContainer(flattened)


class MockPredictor(Predictor):
    """Mock predictor for testing."""

    is_stateful = True
    input_dims = ("trial", "channel", "time")
    output_dims = ("trial",)

    def __init__(self, name="mock_predictor"):
        self.name = name
        self.fitted = False
        self.feature_mean = None
        super().__init__(
            sample_dim="trial",
            target_coord="label",
            is_classifier=True,
            encoder=LabelEncoder(),
        )

    def _fit(self, container, **kwargs):
        self.fitted = True
        self.feature_mean = float(container.data.mean())
        return self

    def _predict(self, data, **kwargs):
        if self.feature_mean is None:
            raise ValueError("Predictor must be fitted before predict")
        predictions = (data.mean(dim=["channel", "time"]) > self.feature_mean).astype(int)
        if len(self.encoder.classes_) == 1:
            predictions = predictions * 0
        else:
            predictions = predictions % len(self.encoder.classes_)
        return predictions.values

    def get_expected_output_dims(self, input_dims):
        return self.output_dims

    def get_params(self, deep=False):
        return {"name": self.name}


class MockUnequalDimTransform(Transform):
    """Transform that produces output with features dim size equal to trial dim size."""

    is_stateful = False
    input_dims = ("trial", "channel", "time")

    def __init__(self, name="unequal"):
        super().__init__()
        self.name = name

    def _fit(self, container, **kwargs):
        return self

    def _transform(self, container, **kwargs):
        num_trials = container.data.sizes["trial"]
        new_array = xr.DataArray(
            np.random.rand(num_trials, num_trials),
            dims=["trial", "features"],
            coords={"trial": container.data.trial, "features": np.arange(num_trials)},
        )
        for c in container.data.coords:
            if "trial" in container.data.coords[c].dims and c != "trial":
                new_array.coords[c] = container.data.coords[c]
        return DataContainer(new_array)

    def get_expected_output_dims(self, input_dims):
        return "trial", "features"

    def get_params(self, deep=False):
        return {"name": self.name}


@pytest.fixture
def sample_container():
    np.random.seed(42)

    trial_indices = np.arange(20)
    animal_values = ["animal_A"] * 8 + ["animal_B"] * 7 + ["animal_C"] * 5
    session_values = ["session_1"] * 10 + ["session_2"] * 10
    labels = ["class_0"] * 10 + ["class_1"] * 10

    data = xr.DataArray(
        np.random.rand(20, 5, 10),
        dims=["trial", "channel", "time"],
        coords={
            "trial": trial_indices,
            "channel": np.arange(5),
            "time": np.arange(10),
            "animal": ("trial", animal_values),
            "session": ("trial", session_values),
            "label": ("trial", labels),
        },
        attrs={"data_history": []},
    )

    return DataContainer(data)


def test_group_apply_initialization():
    transform = MockTransform("base_transform")

    group_apply = GroupApplyTransform(
        group_coord="animal", transform_template=transform, unseen_policy="error", n_jobs=1
    )

    assert group_apply.group_coord == ["animal"]
    assert group_apply.transform_template == transform
    assert group_apply.unseen_policy == "error"
    assert group_apply.n_jobs == 1
    assert not group_apply.is_stateful


def test_group_apply_initialization_with_list():
    transform = MockTransform("base_transform")
    group_coords = ["animal", "session"]
    group_apply = GroupApplyTransform(group_coord=group_coords, transform_template=transform)
    assert group_apply.group_coord == group_coords


def test_group_apply_initialization_errors():
    with pytest.raises(TypeError):
        GroupApplyTransform(group_coord="animal")


def test_group_apply_stateful_detection():
    stateful_transform = MockStatefulTransform("stateful")
    group_apply = GroupApplyTransform(group_coord="animal", transform_template=stateful_transform)
    assert group_apply.is_stateful


def test_group_apply_fit_transform_template(sample_container):
    transform = MockStatefulTransform("base", multiplier=2.0)
    group_apply = GroupApplyTransform(group_coord="animal", transform_template=transform)

    result = group_apply.fit_transform(sample_container)

    assert len(group_apply.seen_groups) == 3
    assert set(group_apply.seen_groups) == {"animal_A", "animal_B", "animal_C"}
    assert len(group_apply.per_group_fitted) == 3

    for fitted_transform in group_apply.per_group_fitted.values():
        assert fitted_transform.fitted
        assert fitted_transform.mean_value is not None

    assert result.data.dims == ("trial", "features")
    assert result.data.shape[0] == 20
    assert "animal" in result.data.coords
    assert set(result.data.coords["animal"].values) == {"animal_A", "animal_B", "animal_C"}


def test_group_apply_fit_transform_multi_coord(sample_container):
    transform = MockStatefulTransform("base", multiplier=2.0)
    group_apply = GroupApplyTransform(group_coord=["animal", "session"], transform_template=transform)

    result = group_apply.fit_transform(sample_container)

    expected_groups = {
        "animal_A_session_1",
        "animal_B_session_1",
        "animal_B_session_2",
        "animal_C_session_2",
    }
    assert len(group_apply.seen_groups) == 4
    assert set(group_apply.seen_groups) == expected_groups
    assert len(group_apply.per_group_fitted) == 4

    for group, fitted_transform in group_apply.per_group_fitted.items():
        assert group in expected_groups
        assert fitted_transform.fitted
        assert fitted_transform.mean_value is not None

    assert result.data.dims == ("trial", "features")
    assert result.data.shape[0] == 20
    assert "animal" in result.data.coords
    assert "session" in result.data.coords


def test_group_apply_transform_unseen_error(sample_container):
    transform = MockTransform("base")
    group_apply = GroupApplyTransform(group_coord="animal", transform_template=transform, unseen_policy="error")

    animal_mask = sample_container.data.coords["animal"].isin(["animal_A", "animal_B"])
    fit_data = sample_container.data.where(animal_mask, drop=True)
    fit_container = DataContainer(fit_data)
    group_apply.fit_transform(fit_container)

    with pytest.raises(TransformError, match="Group 'animal_C' was not seen during fit"):
        group_apply.transform(sample_container)


def test_group_apply_transform_unseen_average(sample_container):
    transform = MockStatefulTransform("base", multiplier=1.0)
    group_apply = GroupApplyTransform(group_coord="animal", transform_template=transform, unseen_policy="average")

    animal_mask = sample_container.data.coords["animal"].isin(["animal_A", "animal_B"])
    fit_data = sample_container.data.where(animal_mask, drop=True)
    fit_container = DataContainer(fit_data)
    group_apply.fit_transform(fit_container)

    result = group_apply.transform(sample_container)

    assert result.data.shape[0] == 20
    assert "animal_C" in result.data.coords["animal"].values


def test_group_apply_transform_unseen_weighted_average(sample_container):
    transform = MockStatefulTransform("base", multiplier=1.0)
    group_apply = GroupApplyTransform(
        group_coord="animal", transform_template=transform, unseen_policy="weighted_average"
    )

    animal_mask = sample_container.data.coords["animal"].isin(["animal_A", "animal_B"])
    fit_data = sample_container.data.where(animal_mask, drop=True)
    fit_container = DataContainer(fit_data)
    group_apply.fit_transform(fit_container)

    assert "animal_A" in group_apply.train_counts
    assert "animal_B" in group_apply.train_counts
    assert group_apply.train_counts["animal_A"] == 8
    assert group_apply.train_counts["animal_B"] == 7

    result = group_apply.transform(sample_container)

    assert result.data.shape[0] == 20
    assert "animal_C" in result.data.coords["animal"].values


def test_group_apply_parallel_processing(sample_container):
    transform = MockStatefulTransform("base", multiplier=2.0)
    group_apply = GroupApplyTransform(group_coord="animal", transform_template=transform, n_jobs=2)

    result = group_apply.fit_transform(sample_container)

    assert len(group_apply.seen_groups) == 3
    assert result.data.dims == ("trial", "features")
    assert result.data.shape[0] == 20


def test_group_apply_predictor_support(sample_container):
    predictor = MockPredictor("mock_pred")
    group_apply = GroupApplyTransform(group_coord="animal", transform_template=predictor)

    group_apply.fit_transform(sample_container)
    predictions = group_apply.predict(sample_container)

    assert predictions.data.dims == ("trial",)
    assert predictions.data.shape[0] == 20

    for fitted_transform in group_apply.per_group_fitted.values():
        assert isinstance(fitted_transform, MockPredictor)
        assert fitted_transform.fitted


def test_group_apply_axis_preservation_validation(sample_container):
    class BadTransform(MockTransform):
        def _transform(self, container, **kwargs):
            data = container.data.mean(dim="trial")
            return DataContainer(data)

    bad_transform = BadTransform("bad")
    group_apply = GroupApplyTransform(group_coord="animal", transform_template=bad_transform)

    with pytest.raises(TransformError, match="removed the grouped dimension"):
        group_apply.fit_transform(sample_container)


def test_group_apply_get_expected_output_dims():
    transform = MockTransform("base")
    group_apply = GroupApplyTransform(group_coord="animal", transform_template=transform)

    input_dims = ("trial", "channel", "time")
    output_dims = group_apply.get_expected_output_dims(input_dims)

    assert output_dims == ("trial", "features")


def test_group_apply_children_property(sample_container):
    transform = MockTransform("base")
    group_apply = GroupApplyTransform(group_coord="animal", transform_template=transform)

    assert len(list(group_apply.children)) == 0

    group_apply.fit_transform(sample_container)
    children = list(group_apply.children)
    assert len(children) == 3
    assert all(isinstance(child, MockTransform) for child in children)


def test_group_apply_repr():
    transform = MockTransform("base")
    group_apply = GroupApplyTransform(
        group_coord="animal", transform_template=transform, unseen_policy="weighted_average", n_jobs=2
    )

    repr_str = repr(group_apply)
    assert "GroupApplyTransform" in repr_str
    assert "animal" in repr_str
    assert "weighted_average" in repr_str
    assert "n_jobs=2" in repr_str


def test_group_apply_missing_group_coord():
    data = xr.DataArray(
        np.random.rand(10, 5, 10),
        dims=["trial", "channel", "time"],
        coords={
            "trial": np.arange(10),
            "channel": np.arange(5),
            "time": np.arange(10),
            "session": ("trial", ["session_1"] * 10),
            "label": ("trial", ["class_0"] * 10),
        },
    )
    container = DataContainer(data)

    transform = MockTransform("base")
    group_apply = GroupApplyTransform(group_coord="animal", transform_template=transform)

    with pytest.raises(ValueError, match="Group coordinate 'animal' not found"):
        group_apply.fit_transform(container)


def test_group_apply_multi_dim_group_coord():
    data = xr.DataArray(
        np.random.rand(10, 5, 10),
        dims=["trial", "channel", "time"],
        coords={
            "trial": np.arange(10),
            "channel": np.arange(5),
            "time": np.arange(10),
            "animal": (["trial", "channel"], np.random.choice(["A", "B"], (10, 5))),
            "session": ("trial", ["session_1"] * 10),
            "label": ("trial", ["class_0"] * 10),
        },
    )
    container = DataContainer(data)

    transform = MockTransform("base")
    group_apply = GroupApplyTransform(group_coord="animal", transform_template=transform)

    with pytest.raises(ValueError, match="must index exactly one dimension"):
        group_apply.fit_transform(container)


def test_group_apply_unequal_output_dims_error(sample_container):
    transform = MockUnequalDimTransform()
    group_apply = GroupApplyTransform(
        group_coord="animal", transform_template=transform, unequal_output_dims_strategy="error"
    )

    with pytest.raises(TransformError, match="Sizes for dimension features are not the same"):
        group_apply.fit_transform(sample_container)


def test_group_apply_unequal_output_dims_cut_to_min(sample_container):
    transform = MockUnequalDimTransform()
    group_apply = GroupApplyTransform(
        group_coord="animal", transform_template=transform, unequal_output_dims_strategy="cut_to_min"
    )

    with pytest.warns(UserWarning, match="Sizes for dimension features are not the same"):
        result = group_apply.fit_transform(sample_container)

    expected_min_features = 5
    assert result.data.sizes["features"] == expected_min_features

    assert result.data.sizes["trial"] == 20
    assert "animal" in result.data.coords
    assert set(result.data.coords["animal"].values) == {"animal_A", "animal_B", "animal_C"}

    assert "features" in group_apply.max_size_per_dim
    assert group_apply.max_size_per_dim["features"] == expected_min_features

    result = group_apply.transform(sample_container)
    assert result.data.sizes["features"] == expected_min_features


def test_group_apply_clone_vs_deepcopy_independence(sample_container):
    class TrackingTransform(MockStatefulTransform):
        def __init__(self, name, multiplier=1.0):
            super().__init__(name, multiplier)
            self.fit_calls = 0
            self.transform_calls = 0

        def _fit(self, container, **kwargs):
            self.fit_calls += 1
            return super()._fit(container, **kwargs)

        def _transform(self, container, **kwargs):
            self.transform_calls += 1
            return super()._transform(container, **kwargs)

    template = TrackingTransform("template")
    group_apply = GroupApplyTransform(group_coord="animal", transform_template=template)

    group_apply.fit_transform(sample_container)

    assert template.fit_calls == 0
    assert template.transform_calls == 0

    for fitted_transform in group_apply.per_group_fitted.values():
        assert fitted_transform.fit_calls == 1
        assert fitted_transform.transform_calls == 1
