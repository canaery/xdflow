"""Tests for cross-validator scoring behavior."""

import numpy as np
import xarray as xr
from sklearn.linear_model import LogisticRegression, Ridge

from xdflow.composite import Pipeline
from xdflow.core.data_container import DataContainer
from xdflow.cv import KFoldValidator, SampledDomainKFoldValidator
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


def _create_multilabel_data() -> DataContainer:
    n_trials = 90
    rng = np.random.default_rng(123)
    target_a = np.tile([0, 1], n_trials // 2)
    target_b = np.tile([0, 0, 1], n_trials // 3)
    data = np.column_stack(
        [
            target_a + rng.normal(scale=0.15, size=n_trials),
            target_b + rng.normal(scale=0.15, size=n_trials),
            rng.normal(size=n_trials),
        ]
    )
    da = xr.DataArray(
        data,
        dims=["trial", "feature"],
        coords={
            "trial": np.arange(n_trials),
            "feature": np.arange(data.shape[1]),
            "target__a": ("trial", target_a),
            "target__b": ("trial", target_b),
            "event_type": ("trial", ["stim"] * n_trials),
        },
    )
    return DataContainer(da)


def test_multilabel_probability_scoring():
    data = _create_multilabel_data()
    clf = SKLearnPredictor(
        LogisticRegression,
        sample_dim="trial",
        target_coord=["target__a", "target__b"],
        is_classifier=True,
        is_multilabel=True,
        max_iter=200,
    )
    pipeline = Pipeline("multilabel_pipeline", [("clf", clf)])
    cv = KFoldValidator(n_splits=3, scoring="ap_macro", verbose=False)
    cv.pipeline = pipeline

    score = cv.cross_validate(data, verbose=False)

    assert cv.metric_name_ == "ap_macro"
    assert np.isfinite(score)
    assert len(cv.oof_probabilities_) == 3
    assert np.concatenate(cv.oof_probabilities_).shape == (90, 2)


def test_sampled_domain_kfold_splits_target_validation_and_sampled_target_training():
    n_source = 20
    n_target = 30
    n_trials = n_source + n_target
    rng = np.random.default_rng(123)
    labels = np.tile(["a", "b"], n_trials // 2)
    da = xr.DataArray(
        rng.normal(size=(n_trials, 4)),
        dims=["trial", "feature"],
        coords={
            "trial": np.arange(n_trials),
            "feature": np.arange(4),
            "stimulus": ("trial", labels),
            "domain": ("trial", np.array(["source"] * n_source + ["target"] * n_target)),
        },
    )
    data = DataContainer(da)
    clf = SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="stimulus", max_iter=200)
    pipeline = Pipeline("domain_pipeline", [("clf", clf)])
    cv = SampledDomainKFoldValidator(
        domain_coord="domain",
        target_domains="target",
        label_coord="stimulus",
        default_samples_per_label=1,
        n_splits=3,
        verbose=False,
    )
    cv.pipeline = pipeline

    splits = list(cv._get_splits(data, data.data.trial.values))

    assert len(splits) == 3
    for train_indices, validation_indices in splits:
        train_domains = data.data.sel(trial=train_indices).coords["domain"].values
        validation_domains = data.data.sel(trial=validation_indices).coords["domain"].values
        assert set(validation_domains) == {"target"}
        assert np.sum(train_domains == "source") == n_source
        assert np.sum(train_domains == "target") == 2
