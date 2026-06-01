"""Core-only XDFlow quickstart example.

Run from the repository root with:

    python examples/quickstart.py

This example uses only the base xdflow dependencies.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from sklearn.linear_model import LogisticRegression

from xdflow.composite import Pipeline
from xdflow.core import DataContainer
from xdflow.cv import KFoldValidator
from xdflow.transforms.basic_transforms import FlattenTransform
from xdflow.transforms.normalization import ZScoreTransform
from xdflow.transforms.sklearn_transform import SKLearnPredictor


def make_toy_trials(
    n_trials_per_class: int = 60,
    n_channels: int = 4,
    n_time: int = 25,
    seed: int = 0,
) -> DataContainer:
    """Create a small labeled trial x channel x time dataset."""
    rng = np.random.default_rng(seed)

    stimuli = np.repeat(["rest", "tone", "odor"], n_trials_per_class)
    rng.shuffle(stimuli)

    values = rng.normal(0.0, 0.8, size=(stimuli.size, n_channels, n_time))
    values[stimuli == "tone", 1, 8:15] += 2.0
    values[stimuli == "odor", 2, 14:22] += 2.0

    trial_ids = np.arange(stimuli.size)
    sessions = np.where(trial_ids < stimuli.size // 2, "session_a", "session_b")

    data = xr.DataArray(
        values,
        dims=("trial", "channel", "time"),
        coords={
            "trial": trial_ids,
            "channel": [f"ch{i}" for i in range(n_channels)],
            "time": np.linspace(-0.2, 0.8, n_time),
            "stimulus": ("trial", stimuli),
            "session": ("trial", sessions),
        },
        attrs={"sampling_frequency_hz": 25.0},
    )
    return DataContainer(data)


def build_pipeline() -> Pipeline:
    """Build a minimal dimension-aware classifier pipeline."""
    return Pipeline(
        name="core_quickstart",
        steps=[
            ("zscore", ZScoreTransform(per_dim="trial")),
            ("flatten", FlattenTransform(dims=("channel", "time"))),
            (
                "classifier",
                SKLearnPredictor(
                    LogisticRegression,
                    sample_dim="trial",
                    target_coord="stimulus",
                    max_iter=500,
                ),
            ),
        ],
    )


def run_quickstart(
    n_trials_per_class: int = 60,
    n_channels: int = 4,
    n_time: int = 25,
    seed: int = 0,
) -> tuple[float, DataContainer]:
    """Run cross-validation, fit the final pipeline, and return predictions."""
    container = make_toy_trials(
        n_trials_per_class=n_trials_per_class,
        n_channels=n_channels,
        n_time=n_time,
        seed=seed,
    )
    pipeline = build_pipeline()

    cv = KFoldValidator(
        n_splits=5,
        shuffle=True,
        random_state=seed,
        stratify_coord="stimulus",
        scoring="f1_weighted",
        verbose=False,
    )
    cv.set_pipeline(pipeline)
    score = cv.cross_validate(container, verbose=False)

    pipeline.fit(container)
    predictions = pipeline.predict(container)
    return score, predictions


if __name__ == "__main__":
    quickstart_score, quickstart_predictions = run_quickstart()
    print(f"Weighted F1: {quickstart_score:.3f}")
    print(f"Prediction dims: {quickstart_predictions.data.dims}")
