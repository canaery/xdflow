"""
Tests for multi-target regression support and pattern matching.
"""

import numpy as np
import pytest
import xarray as xr
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from xdflow.composite import Pipeline
from xdflow.core.data_container import DataContainer
from xdflow.cv import KFoldValidator
from xdflow.transforms.sklearn_transform import SKLearnPredictor


def _create_multi_target_data(n_trials=120, n_features=10, n_targets=3) -> DataContainer:
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_trials, n_features))

    targets = {}
    for i in range(n_targets):
        targets[f"t{i}_target"] = rng.standard_normal(n_trials) * (i + 1)

    coords = {
        "trial": np.arange(n_trials),
        "feature": np.arange(n_features),
    }
    for name, values in targets.items():
        coords[name] = ("trial", values)

    da = xr.DataArray(data, dims=["trial", "feature"], coords=coords)
    return DataContainer(da)


def test_single_target_backward_compatibility():
    data = _create_multi_target_data(n_trials=80)

    regressor = SKLearnPredictor(
        Ridge,
        sample_dim="trial",
        target_coord="t0_target",
        alpha=1.0,
    )

    assert regressor.is_regressor is True
    assert regressor.is_classifier is False
    assert regressor.is_multi_target is False

    regressor.fit(data)
    predictions = regressor.predict(data)
    assert predictions.data.ndim == 1
    assert predictions.data.dims == ("trial",)


def test_multi_target_predictor():
    data = _create_multi_target_data(n_trials=80)
    target_coords = ["t0_target", "t1_target", "t2_target"]

    regressor = SKLearnPredictor(
        Ridge,
        sample_dim="trial",
        target_coord=target_coords,
        alpha=1.0,
    )

    assert regressor.is_regressor is True
    assert regressor.is_classifier is False
    assert regressor.is_multi_target is True
    assert regressor.target_coord_list == target_coords

    regressor.fit(data)
    predictions = regressor.predict(data)

    assert predictions.data.ndim == 2
    assert predictions.data.dims == ("trial", "target")
    assert predictions.data.sizes["target"] == 3
    assert list(predictions.data.coords["target"].values) == target_coords


def test_multi_target_cv():
    data = _create_multi_target_data(n_trials=100)
    target_coords = ["t0_target", "t1_target", "t2_target"]

    pipeline = Pipeline(
        "test_multi_target",
        [
            (
                "regressor",
                SKLearnPredictor(
                    Ridge,
                    sample_dim="trial",
                    target_coord=target_coords,
                    alpha=1.0,
                ),
            )
        ],
    )

    cv = KFoldValidator(n_splits=3, scoring="r2")
    cv.set_pipeline(pipeline)
    cv.cross_validate(data, verbose=False)

    assert cv.metric_name_ == "r2"
    assert len(cv.cv_scores_) == 3
    assert isinstance(cv.oof_score_, float)

    for pred in cv.oof_predictions_:
        assert pred.ndim == 2
        assert pred.shape[1] == 3

    for true in cv.true_labels_:
        assert true.ndim == 2
        assert true.shape[1] == 3


def test_multi_target_different_metrics():
    data = _create_multi_target_data(n_trials=100)
    target_coords = ["t0_target", "t1_target", "t2_target"]

    pipeline = Pipeline(
        "test_metrics",
        [
            (
                "regressor",
                SKLearnPredictor(
                    Ridge,
                    sample_dim="trial",
                    target_coord=target_coords,
                    alpha=1.0,
                ),
            )
        ],
    )

    for metric in ["r2", "mse", "mae", "rmse"]:
        cv = KFoldValidator(n_splits=3, scoring=metric)
        cv.set_pipeline(pipeline)
        score = cv.cross_validate(data, verbose=False)
        assert cv.metric_name_ == metric
        assert isinstance(score, float)


def test_multi_target_with_lgbm():
    lightgbm = pytest.importorskip("lightgbm")
    data = _create_multi_target_data(n_trials=100)
    target_coords = ["t0_target", "t1_target", "t2_target"]

    regressor = SKLearnPredictor(
        lambda **kwargs: MultiOutputRegressor(lightgbm.LGBMRegressor(**kwargs)),
        sample_dim="trial",
        target_coord=target_coords,
        n_estimators=50,
        verbose=-1,
        is_classifier=False,
    )

    assert regressor.is_regressor is True
    assert regressor.is_multi_target is True

    regressor.fit(data)
    predictions = regressor.predict(data)

    assert predictions.data.ndim == 2
    assert predictions.data.sizes["target"] == 3


def test_classifier_rejects_multi_target():
    from sklearn.linear_model import LogisticRegression

    target_coords = ["t0_target", "t1_target"]

    with pytest.raises(ValueError, match="Multi-target prediction is only supported for regressors"):
        SKLearnPredictor(
            LogisticRegression,
            sample_dim="trial",
            target_coord=target_coords,
            is_classifier=True,
        )


def test_holdout_scoring_multi_target():
    data = _create_multi_target_data(n_trials=100)
    target_coords = ["t0_target", "t1_target", "t2_target"]

    pipeline = Pipeline(
        "test_holdout",
        [
            (
                "regressor",
                SKLearnPredictor(
                    Ridge,
                    sample_dim="trial",
                    target_coord=target_coords,
                    alpha=1.0,
                ),
            )
        ],
    )

    cv = KFoldValidator(n_splits=3, test_size=0.2, scoring="r2")
    cv.set_pipeline(pipeline)
    cv.cross_validate(data, verbose=False)
    holdout_score = cv.score_on_holdout(data, verbose=False)

    assert cv.holdout_pred_labels_.ndim == 2
    assert cv.holdout_true_labels_.ndim == 2
    assert cv.holdout_pred_labels_.shape[1] == 3
    assert isinstance(holdout_score, float)


def test_pattern_matching_auto_discovery():
    data = _create_multi_target_data(n_trials=100)

    regressor = SKLearnPredictor(
        Ridge,
        sample_dim="trial",
        target_coord="*_target",
        alpha=1.0,
    )

    regressor.fit(data)

    assert regressor.is_multi_target is True
    assert len(regressor.target_coord_list) == 3
    assert set(regressor.target_coord_list) == {"t0_target", "t1_target", "t2_target"}

    predictions = regressor.predict(data)
    assert predictions.data.ndim == 2
    assert predictions.data.sizes["target"] == 3


def test_pattern_matching_with_cv():
    data = _create_multi_target_data(n_trials=100)

    pipeline = Pipeline(
        "test_pattern_cv",
        [
            (
                "regressor",
                SKLearnPredictor(
                    Ridge,
                    sample_dim="trial",
                    target_coord="*_target",
                    alpha=1.0,
                ),
            )
        ],
    )

    cv = KFoldValidator(n_splits=3, test_size=0.2, scoring="r2")
    cv.set_pipeline(pipeline)
    cv.cross_validate(data, verbose=False)

    assert cv.metric_name_ == "r2"
    assert len(cv.cv_scores_) == 3
    assert isinstance(cv.oof_score_, float)

    for pred in cv.oof_predictions_:
        assert pred.ndim == 2
        assert pred.shape[1] == 3

    for true in cv.true_labels_:
        assert true.ndim == 2
        assert true.shape[1] == 3

    holdout_score = cv.score_on_holdout(data, verbose=False)
    assert isinstance(holdout_score, float)
    assert cv.holdout_pred_labels_.ndim == 2
    assert cv.holdout_true_labels_.ndim == 2
