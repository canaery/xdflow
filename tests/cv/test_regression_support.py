"""
Regression and classification support tests for CrossValidator + SKLearnPredictor.
"""

import numpy as np
import pytest
import xarray as xr
from sklearn.linear_model import LogisticRegression, Ridge

from xdflow.composite import Pipeline
from xdflow.core.data_container import DataContainer
from xdflow.cv import KFoldValidator
from xdflow.transforms.sklearn_transform import SKLearnPredictor


def _create_test_data(n_trials: int = 120, n_features: int = 12, task: str = "classification") -> DataContainer:
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_trials, n_features))

    if task == "classification":
        target = rng.choice(["A", "B", "C"], size=n_trials)
    else:
        target = rng.standard_normal(n_trials) * 10 + 50

    da = xr.DataArray(
        data,
        dims=["trial", "feature"],
        coords={
            "trial": np.arange(n_trials),
            "feature": np.arange(n_features),
            "label": ("trial", target),
            "session": ("trial", rng.choice(["s1", "s2", "s3"], size=n_trials)),
        },
    )
    return DataContainer(da)


def test_classification_pipeline():
    data = _create_test_data(task="classification")

    clf = SKLearnPredictor(
        LogisticRegression,
        sample_dim="trial",
        target_coord="label",
        max_iter=200,
    )

    assert clf.is_classifier is True
    assert clf.is_regressor is False

    pipeline = Pipeline("test_clf_pipeline", [("classifier", clf)])
    cv = KFoldValidator(n_splits=3)
    cv.set_pipeline(pipeline)

    cv.cross_validate(data, verbose=False)

    assert cv.metric_name_ == "f1_weighted"
    assert cv.oof_score_ == cv.oof_f1_score_
    conf_matrix = cv.oof_confusion_matrix_
    assert conf_matrix.shape[0] == conf_matrix.shape[1]


def test_regression_pipeline():
    data = _create_test_data(task="regression")

    reg = SKLearnPredictor(
        Ridge,
        sample_dim="trial",
        target_coord="label",
        alpha=1.0,
    )

    assert reg.is_classifier is False
    assert reg.is_regressor is True

    pipeline = Pipeline("test_reg_pipeline", [("regressor", reg)])
    cv = KFoldValidator(n_splits=3)
    cv.set_pipeline(pipeline)

    cv.cross_validate(data, verbose=False)

    assert cv.metric_name_ == "r2"
    with pytest.raises(ValueError, match="classification tasks"):
        _ = cv.oof_confusion_matrix_


def test_custom_metrics():
    data = _create_test_data(task="regression")

    reg = SKLearnPredictor(Ridge, sample_dim="trial", target_coord="label")
    pipeline = Pipeline("test_metrics_pipeline", [("regressor", reg)])

    cv_mse = KFoldValidator(n_splits=3, scoring="mse")
    cv_mse.set_pipeline(pipeline)
    cv_mse.cross_validate(data, verbose=False)
    assert cv_mse.metric_name_ == "mse"

    cv_rmse = KFoldValidator(n_splits=3, scoring="rmse")
    cv_rmse.set_pipeline(pipeline)
    cv_rmse.cross_validate(data, verbose=False)
    assert cv_rmse.metric_name_ == "rmse"
    assert abs((abs(cv_rmse.oof_score_) ** 2) - abs(cv_mse.oof_score_)) < 0.01

    cv_mae = KFoldValidator(n_splits=3, scoring="mae")
    cv_mae.set_pipeline(pipeline)
    cv_mae.cross_validate(data, verbose=False)
    assert cv_mae.metric_name_ == "mae"

    def custom_metric(y_true, y_pred):
        return -np.mean(np.abs(y_true - y_pred))

    cv_custom = KFoldValidator(n_splits=3, scoring=custom_metric)
    cv_custom.set_pipeline(pipeline)
    cv_custom.cross_validate(data, verbose=False)
    assert cv_custom.metric_name_ == "custom"
    assert abs(cv_custom.oof_score_ - cv_mae.oof_score_) < 0.01


def test_manual_override():
    clf = SKLearnPredictor(
        Ridge,
        sample_dim="trial",
        target_coord="label",
        is_classifier=True,
    )

    assert clf.is_classifier is True
    assert clf.is_regressor is False
