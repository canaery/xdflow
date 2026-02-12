"""Tests for intertrial filtering in cross-validation."""

import numpy as np
import xarray as xr
from sklearn.preprocessing import LabelEncoder

from xdflow.composite import Pipeline
from xdflow.core.base import Predictor
from xdflow.core.data_container import DataContainer
from xdflow.cv import KFoldValidator


class IntertrialAwarePredictor(Predictor):
    """Predictor that deliberately mislabels intertrial trials."""

    input_dims = ("trial", "feature")
    output_dims = ("trial",)

    def __init__(self, wrong_label: str = "classA"):
        super().__init__(
            sample_dim="trial",
            target_coord="stimulus",
            is_classifier=True,
            encoder=LabelEncoder(),
        )
        self.wrong_label = wrong_label

    def _label_values(self, data_array):
        coord_name = "stimulus_orig" if "stimulus_orig" in data_array.coords else "stimulus"
        return data_array.coords[coord_name].values.astype(str)

    def _fit(self, container, **kwargs):
        self.encoder.fit(self._label_values(container.data))
        return self

    def _predict(self, data, **kwargs):
        labels = self._label_values(data)
        event_types = data.coords["event_type"].values.astype(str)
        predictions = labels.copy()
        intertrial_mask = event_types == "intertrial"
        if intertrial_mask.any():
            predictions[intertrial_mask] = self.wrong_label
        return self.encoder.transform(predictions)

    def get_expected_output_dims(self, input_dims):
        return self.output_dims


def _create_intertrial_eval_data():
    """Synthetic container with explicit intertrial trials for scoring tests."""
    n_trials = 6
    n_features = 4
    rng = np.random.default_rng(123)
    data = rng.normal(size=(n_trials, n_features))
    stimuli = np.array(["classA", "blank", "classB", "blank", "classA", "classB"], dtype=object)
    event_types = np.array(["poke", "intertrial", "poke", "intertrial", "poke", "poke"], dtype=object)

    da = xr.DataArray(
        data,
        dims=["trial", "feature"],
        coords={
            "trial": np.arange(n_trials),
            "feature": np.arange(n_features),
            "session": ("trial", ["s1"] * n_trials),
            "stimulus": ("trial", stimuli),
            "event_type": ("trial", event_types),
        },
    )
    return DataContainer(da)


def test_cross_validator_can_exclude_intertrial_from_scoring():
    """Ensure the CV toggle filters intertrial trials when computing metrics."""
    data = _create_intertrial_eval_data()

    pipeline_with_intertrial = Pipeline(
        "intertrial_pipeline",
        [("predictor", IntertrialAwarePredictor(wrong_label="classA"))],
    )
    cv_with_intertrial = KFoldValidator(n_splits=2, shuffle=True, random_state=0)
    cv_with_intertrial.pipeline = pipeline_with_intertrial
    score_with_intertrial = cv_with_intertrial.cross_validate(data, verbose=False)

    pipeline_filtered = Pipeline(
        "intertrial_pipeline_filtered",
        [("predictor", IntertrialAwarePredictor(wrong_label="classA"))],
    )
    cv_filtered = KFoldValidator(
        n_splits=2,
        shuffle=True,
        random_state=0,
        exclude_intertrial_from_scoring=True,
    )
    cv_filtered.pipeline = pipeline_filtered
    score_without_intertrial = cv_filtered.cross_validate(data, verbose=False)

    assert score_without_intertrial > score_with_intertrial
    assert np.isclose(score_without_intertrial, 1.0)
