# Using XDFlow With LLMs

XDFlow includes a repository-level agent instruction file:

```text
AGENTS.md
```

For Codex-style coding agents, this file is read automatically when the agent works in the repository. For other LLM tools, attach `AGENTS.md` or point the tool at it before asking for code changes.

That file is the canonical instruction source for XDFlow-specific implementation rules: how to write transforms, how to honor the xarray data contract, how cloning works, when to use sklearn wrappers, and when to use XDFlow validators instead of manual split loops. For human-readable extension guides, see [Writing Custom Transforms](writing-transforms.md) and [Writing Custom Cross-Validators](writing-cross-validators.md).

## Why This Helps

XDFlow works well with LLM coding tools because there are a few concrete interfaces to target:

- data is an `xarray.DataArray` wrapped in `DataContainer`
- transforms implement `fit`, `transform`, and `fit_transform` through the `Transform` base class
- predictors implement `predict` and optionally `predict_proba`
- pipelines and validators own orchestration, fold-invariant caching, splitting, refitting, and scoring

The practical goal is to have the LLM write one small XDFlow piece at a time instead of inventing a full analysis script with manual side arrays, split loops, and cache logic.

## Recommended Workflow

1. Let the agent read `AGENTS.md`.
2. Give the scientific task and the data contract: dimensions, coordinates, sample dimension, target coordinates, grouping coordinates, and whether new transforms are fold-invariant or stateful.
3. Ask for one bounded implementation: a transform, a pipeline, a validator configuration, or tests.
4. Have the agent run `uv run ruff check xdflow tests` and `uv run pytest`.

This keeps the long-lived rules in version control and leaves each request focused on the actual scientific operation.

## Good Agent Tasks

- write a new `Transform` subclass for one preprocessing operation
- wrap a scikit-learn estimator with `SKLearnTransformer` or `SKLearnPredictor`
- compose existing transforms into a `Pipeline`
- add a `CrossValidator` configuration for a specific experimental design
- add focused tests for dimensions, coordinates, and immutability

Avoid asking the LLM to manually manage train/test splits, reshape arrays across many functions, or pass raw numpy arrays through a long script. Those are exactly the responsibilities XDFlow should centralize.

## Transform Skeleton For Agents

When a custom preprocessing operation is needed, the agent should target this shape. The full authoring guide, including dimension validation and tests, is [Writing Custom Transforms](writing-transforms.md).

```python
from typing import Any

import xarray as xr

from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer


class MyFeatureTransform(Transform):
    is_stateful = False
    input_dims = ()
    output_dims = ()

    def __init__(self, feature_dim: str = "feature", sel: dict[str, Any] | None = None):
        super().__init__(sel=sel)
        self.feature_dim = feature_dim

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        data = container.data
        transformed: xr.DataArray = ...
        return DataContainer(transformed)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        return input_dims
```

For stateful transforms, set `is_stateful = True`, implement `_fit`, and store learned values in private attributes such as `self._mean_` or `self._projection_`.

## Prefer Existing Building Blocks

Before asking for a new class, ask the LLM to use existing XDFlow components:

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from xdflow.composite import Pipeline
from xdflow.cv import KFoldValidator
from xdflow.transforms.basic_transforms import FlattenTransform
from xdflow.transforms.normalization import ZScoreTransform
from xdflow.transforms.sklearn_transform import SKLearnPredictor, SKLearnTransformer


pipeline = Pipeline(
    name="classifier",
    steps=[
        ("zscore", ZScoreTransform(by_dim=["trial"])),
        ("flatten", FlattenTransform(dims=("channel", "time"))),
        ("pca", SKLearnTransformer(PCA, sample_dim="trial", n_components=20)),
        ("clf", SKLearnPredictor(LogisticRegression, sample_dim="trial", target_coord="stimulus")),
    ],
)

cv = KFoldValidator(n_splits=5, test_size=0.2, stratify_coord="stimulus", verbose=False)
cv.set_pipeline(pipeline)
```

This keeps expensive preprocessing, fitting, prediction, scoring, and cache reuse inside XDFlow instead of spreading orchestration across generated helper functions.

## Review Checklist

When reviewing agent-generated XDFlow code, check:

- all constructor hyperparameters are public attributes with the same names
- fitted state is private and not included in `__init__`
- transforms return a new `DataContainer`
- xarray dimensions and coordinates are preserved or intentionally changed
- stateless transforms are genuinely fold-invariant if they will run before CV split boundaries
- sample-level coordinates remain aligned to the sample dimension
- sklearn models are wrapped with `SKLearnTransformer` or `SKLearnPredictor`
- cross-validation uses `xdflow.cv` validators rather than manual split loops
- tests cover dimensions, coordinates, immutability, and clone behavior

If the generated code violates these, update `AGENTS.md` if the rule is general, then ask the agent for a targeted fix.
