"""Tests for cross-validator scoring behavior."""

import numpy as np
import xarray as xr
from sklearn.linear_model import LogisticRegression, Ridge

from xdflow.composite import Pipeline
from xdflow.core.data_container import DataContainer
from xdflow.cv import KFoldValidator
from xdflow.transforms.sklearn_transform import SKLearnPredictor


def _create_eval_data(task: str = "classification") -> DataContainer:
    n_trials = 90
    n_features = 6
    rng = np.random.default_rng(42)
    data = rng.normal(size=(n_trials, n_features))

    if task == "classification":
        target = rng.choice(["A", "B", "C"], size=n_trials)
    else:
        target = rng.normal(loc=0.0, scale=1.0, size=n_trials)

    da = xr.DataArray(
        data,
        dims=["trial", "feature"],
        coords={
            "trial": np.arange(n_trials),
            "feature": np.arange(n_features),
            "stimulus": ("trial", target),
            "session": ("trial", rng.choice(["s1", "s2", "s3"], size=n_trials)),
            "event_type": ("trial", ["stim"] * n_trials),
        },
    )
    return DataContainer(da)


def test_auto_metric_selection_classification():
    data = _create_eval_data(task="classification")
    clf = SKLearnPredictor(
        LogisticRegression,
        sample_dim="trial",
        target_coord="stimulus",
        max_iter=200,
    )
    pipeline = Pipeline("clf_pipeline", [("clf", clf)])
    cv = KFoldValidator(n_splits=3)
    cv.pipeline = pipeline

    cv.cross_validate(data, verbose=False)

    assert cv.metric_name_ == "f1_weighted"
    assert cv.oof_score_ == cv.oof_f1_score_


def test_auto_metric_selection_regression():
    data = _create_eval_data(task="regression")
    reg = SKLearnPredictor(
        Ridge,
        sample_dim="trial",
        target_coord="stimulus",
        is_classifier=False,
        alpha=1.0,
    )
    pipeline = Pipeline("reg_pipeline", [("reg", reg)])
    cv = KFoldValidator(n_splits=3)
    cv.pipeline = pipeline

    cv.cross_validate(data, verbose=False)

    assert cv.metric_name_ == "r2"
    assert np.isfinite(cv.oof_score_)


def test_custom_string_metrics():
    data = _create_eval_data(task="regression")
    reg = SKLearnPredictor(Ridge, sample_dim="trial", target_coord="stimulus", is_classifier=False)
    pipeline = Pipeline("metrics_pipeline", [("reg", reg)])

    cv_mse = KFoldValidator(n_splits=3, scoring="mse")
    cv_mse.pipeline = pipeline
    cv_mse.cross_validate(data, verbose=False)
    assert cv_mse.metric_name_ == "mse"

    cv_rmse = KFoldValidator(n_splits=3, scoring="rmse")
    cv_rmse.pipeline = pipeline
    cv_rmse.cross_validate(data, verbose=False)
    assert cv_rmse.metric_name_ == "rmse"
    assert abs((abs(cv_rmse.oof_score_) ** 2) - abs(cv_mse.oof_score_)) < 0.05

    cv_mae = KFoldValidator(n_splits=3, scoring="mae")
    cv_mae.pipeline = pipeline
    cv_mae.cross_validate(data, verbose=False)
    assert cv_mae.metric_name_ == "mae"


def test_container_aware_scorer():
    data = _create_eval_data(task="regression")
    reg = SKLearnPredictor(Ridge, sample_dim="trial", target_coord="stimulus", is_classifier=False)
    pipeline = Pipeline("custom_pipeline", [("reg", reg)])

    called = {"count": 0}

    def container_metric(y_true, y_pred, container):
        assert container is not None
        assert "trial" in container.data.dims
        called["count"] += 1
        return -np.mean(np.abs(y_true - y_pred))

    cv = KFoldValidator(n_splits=3, scoring=container_metric)
    cv.pipeline = pipeline
    cv.cross_validate(data, verbose=False)

    assert cv.metric_name_ == "custom"
    assert called["count"] > 0


def test_stratify_coord_runs():
    data = _create_eval_data(task="classification")
    clf = SKLearnPredictor(
        LogisticRegression,
        sample_dim="trial",
        target_coord="stimulus",
        max_iter=200,
    )
    pipeline = Pipeline("stratify_pipeline", [("clf", clf)])
    cv = KFoldValidator(n_splits=3, stratify_coord="stimulus")
    cv.pipeline = pipeline

    score = cv.cross_validate(data, verbose=False)

    assert np.isfinite(score)
    assert cv.stratify_coord == "stimulus"
