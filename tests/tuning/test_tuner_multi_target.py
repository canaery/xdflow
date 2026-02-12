"""
Test tuner integration with pattern-based multi-target regression.
"""

import numpy as np
import xarray as xr
from sklearn.linear_model import Ridge

from xdflow.composite import Pipeline
from xdflow.core.data_container import DataContainer
from xdflow.cv import KFoldValidator
from xdflow.transforms.sklearn_transform import SKLearnPredictor
from xdflow.tuning.tuner_utils import run_tuning_pipeline


def _create_test_data(n_trials=100, n_features=10, n_targets=3):
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_trials, n_features))

    coords = {
        "trial": np.arange(n_trials),
        "feature": np.arange(n_features),
    }
    for i in range(n_targets):
        coords[f"t{i}_target"] = ("trial", rng.standard_normal(n_trials) * (i + 1))

    da = xr.DataArray(data, dims=["trial", "feature"], coords=coords)
    return DataContainer(da)


def test_tuner_with_pattern_matching():
    data = _create_test_data(n_trials=80)

    pipeline = Pipeline(
        "test_pattern_tune",
        [
            (
                "regressor",
                SKLearnPredictor(
                    Ridge,
                    sample_dim="trial",
                    target_coord="*_target",
                    alpha=1.0,
                    is_classifier=False,
                ),
            )
        ],
    )

    param_grid = {
        "test_pattern_tune": {
            "regressor": {
                "alpha": (0.1, 10.0),
            }
        }
    }

    cv = KFoldValidator(n_splits=3, test_size=0.2, scoring="r2")

    finalized_pipelines = run_tuning_pipeline(
        pipelines_to_tune=pipeline,
        cv_strategy=cv,
        param_grid=param_grid,
        initial_data_container=data,
        experiment_name="test_pattern_tuning",
        n_seeds=2,
        n_trials=3,
        plot_importances=False,
        plot_combined_conf_matrix=False,
        plot_each_seed_conf_matrix=False,
        use_cache=False,
        use_mlflow=False,
        verbose=False,
        score_on_holdout=True,
    )

    assert len(finalized_pipelines) == 2

    for pipeline in finalized_pipelines:
        predictor = pipeline.predictive_transform
        assert predictor.target_coord == "*_target"
        assert len(predictor.target_coord_list) == 3
        assert predictor.is_multi_target is True

    cloned = finalized_pipelines[0].clone()
    cloned_predictor = cloned.predictive_transform

    # Re-fit should resolve pattern again
    cloned.fit(data)
    assert len(cloned_predictor.target_coord_list) == 3
    assert cloned_predictor.is_multi_target is True
