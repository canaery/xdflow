# Basic Pipeline Walkthrough

This walkthrough mirrors the intent of the notebook in `docs/tutorials/basic_tutorial.ipynb`, but keeps the published documentation lightweight and RTD-friendly.

It uses `MultiTaperTransform`, so install `xdflow[spectral]` or `xdflow[all]` before running the example.

## 1. Create labeled input data

`xdflow` expects an `xarray.DataArray` with named dimensions and coordinates. The repo includes a synthetic generator for a trial-aligned LFP-like dataset:

```python
from docs.tutorials.synthetic_lfp import make_synthetic_lfp
from xdflow.core import DataContainer

da = make_synthetic_lfp(n_trials=600, n_channels=32, seed=42)
container = DataContainer(da)
```

That array has:

- dimensions: `trial`, `channel`, `time`
- trial coordinates: `stimulus`, `session`, `animal`
- attrs such as `sampling_frequency_hz`

## 2. Build a pipeline

The example below performs common-average re-referencing, z-scoring, spectral feature extraction, flattening, PCA, and logistic regression:

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from xdflow.composite.pipeline import Pipeline
from xdflow.transforms.basic_transforms import FlattenTransform
from xdflow.transforms.cleaning import CARTransform
from xdflow.transforms.normalization import ZScoreTransform
from xdflow.transforms.sklearn_transform import SKLearnPredictor, SKLearnTransformer
from xdflow.transforms.spectral import MultiTaperTransform

fs = container.attrs["sampling_frequency_hz"]
freq_ranges = {
    "delta": [2, 4],
    "theta": [4, 8],
    "alpha": [8, 12],
    "beta": [12, 30],
    "gamma": [30, 58],
}

pipeline = Pipeline(
    name="decode_stimulus",
    steps=[
        ("car", CARTransform(car_method="all")),
        ("zscore", ZScoreTransform(by_dim=["trial"])),
        (
            "multitaper",
            MultiTaperTransform(
                fs=fs,
                num_time_windows=4,
                time_halfbandwidth_product=2,
                avg_over_time_windows=True,
                avg_over_freq_bands=True,
                freq_ranges=freq_ranges,
                n_jobs=-1,
            ),
        ),
        ("flatten", FlattenTransform(dims=("channel", "freq_band"))),
        (
            "pca",
            SKLearnTransformer(
                estimator_cls=PCA,
                sample_dim="trial",
                output_dim_name="feature",
                n_components=30,
            ),
        ),
        (
            "logreg",
            SKLearnPredictor(
                estimator_cls=LogisticRegression,
                sample_dim="trial",
                target_coord="stimulus",
                max_iter=500,
            ),
        ),
    ],
)
```

You can inspect the expected shape evolution ahead of time:

```python
pipeline.get_expected_output_dims(("trial", "channel", "time"), print_steps=True)
```

## 3. Evaluate with cross-validation

```python
from xdflow.cv.kfold import KFoldValidator

cv = KFoldValidator(
    n_splits=5,
    shuffle=True,
    random_state=0,
    test_size=0.2,
    scoring="f1_weighted",
)
cv.set_pipeline(pipeline)

score = cv.cross_validate(container, verbose=False)
print(score)
```

What `xdflow` handles for you here:

- trial labels stay attached through every step
- stateless preprocessing can be reused across folds
- stateful steps such as PCA and logistic regression are refit per fold
- predictors read targets from coordinates instead of a separate `y`

## 4. Predict on new data

Once fit, the same pipeline can be used directly:

```python
pipeline.fit(container)
predictions = pipeline.predict(container)
print(predictions.data.dims)
```

For classification pipelines, the final dimension is usually `prediction`.

## 5. Next steps

- Use [Data Contract](../concepts/data_contract.md) when adapting your own data.
- Explore grouped workflows with `GroupApplyTransform` if models or preprocessors should be fit per subject or session.
- Use the [API Reference](../api/index.md) to inspect transform signatures and available CV strategies.
