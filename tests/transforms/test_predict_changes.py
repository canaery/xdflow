"""
Tests for prediction output behavior in SKLearnPredictor.
"""

import numpy as np
import pytest
import xarray as xr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder

from xdflow.core.data_container import DataContainer
from xdflow.transforms.sklearn_transform import SKLearnPredictor


def test_predict_outputs_primary_data_and_preserves_coords():
    n_trials = 100
    n_features = 50

    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_trials, n_features))
    sessions = rng.choice(["session_1", "session_2", "session_3"], n_trials)
    labels = rng.choice(["A", "B", "C"], n_trials)

    data_array = xr.DataArray(
        data,
        dims=["trial", "feature"],
        coords={
            "trial": np.arange(n_trials),
            "feature": np.arange(n_features),
            "session": ("trial", sessions),
            "label": ("trial", labels),
        },
    )
    container = DataContainer(data_array)

    predictor = SKLearnPredictor(
        estimator_cls=LogisticRegression,
        sample_dim="trial",
        target_coord="label",
        max_iter=200,
    )

    predictor.fit(container)
    prediction_result = predictor.predict(container)

    assert prediction_result.data.dims == ("trial",)
    assert prediction_result.data.shape == (n_trials,)

    # Coordinates preserved for trial-aligned metadata
    assert "trial" in prediction_result.data.coords
    assert "session" in prediction_result.data.coords
    assert "label" in prediction_result.data.coords

    # Predictions should be data, not a coordinate
    with pytest.raises(KeyError):
        _ = prediction_result.data.coords["prediction"]


def test_encoder_and_proba_outputs():
    n_trials = 80
    n_features = 20

    rng = np.random.default_rng(123)
    data = rng.standard_normal((n_trials, n_features))
    labels = rng.choice(["A", "B"], n_trials)

    data_array = xr.DataArray(
        data,
        dims=["trial", "feature"],
        coords={
            "trial": np.arange(n_trials),
            "feature": np.arange(n_features),
            "label": ("trial", labels),
        },
    )
    container = DataContainer(data_array)

    encoder = LabelEncoder()
    encoder.fit(labels)

    predictor = SKLearnPredictor(
        estimator_cls=LogisticRegression,
        sample_dim="trial",
        target_coord="label",
        encoder=encoder,
        max_iter=200,
    )

    predictor.fit(container)

    proba_result = predictor.predict_proba(container)
    assert proba_result.data.shape == (n_trials, len(encoder.classes_))
    assert list(proba_result.data.dims) == ["trial", "class"]
    np.testing.assert_array_equal(proba_result.data.coords["class"].values, encoder.classes_)

    predict_result_encoded = predictor.predict(container)
    assert predict_result_encoded.data.values[0] in encoder.classes_

    transform_result = predictor.transform(container)
    assert transform_result.data.shape == (n_trials, 1)
    assert list(transform_result.data.dims) == ["trial", "prediction"]
    assert transform_result.data.values[0][0] in encoder.classes_


def test_sklearn_regressor_behavior():
    n_trials = 100
    n_features = 10
    rng = np.random.default_rng(7)

    X = rng.standard_normal((n_trials, n_features))
    true_weights = rng.standard_normal(n_features)
    y_continuous = X @ true_weights + rng.standard_normal(n_trials) * 0.1

    data_array = xr.DataArray(
        X,
        dims=["trial", "feature"],
        coords={
            "trial": np.arange(n_trials),
            "feature": np.arange(n_features),
            "target": ("trial", y_continuous),
        },
    )
    container = DataContainer(data_array)

    regressor = SKLearnPredictor(
        estimator_cls=Ridge,
        sample_dim="trial",
        target_coord="target",
        alpha=1.0,
    )

    assert not regressor.is_classifier
    assert regressor.encoder is None

    regressor.fit(container)
    prediction_result = regressor.predict(container)

    assert prediction_result.data.shape == (n_trials,)
    assert "float" in str(prediction_result.data.dtype)

    with pytest.raises(AttributeError, match="not been instantiated as a classifier"):
        regressor.predict_proba(container)
