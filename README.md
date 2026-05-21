# XDFlow

**Dimension-aware, metadata-driven ML pipelines for structured data**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

`xdflow` is a machine learning framework for scientific data stored as labeled [xarray](https://xarray.dev/) objects. Transforms, predictors, validators, and tuners work with dimensions and coordinates such as `trial`, `channel`, `time`, `stimulus`, `session`, and `subject`, instead of losing that information when data is reshaped to anonymous `(samples, features)` arrays.

In a NumPy/sklearn workflow, that structure usually becomes side data:

```text
X.shape          # (180, 100)        # was (180, 4, 25)
y                # was a coordinate
sessions         # tracked separately
channel_names    # tracked separately
```

Every split, transform, prediction, and score then depends on keeping those arrays in sync. A bad reindex or reshape can still produce valid-looking arrays and scores.

`xdflow` keeps dimensions, coordinates, targets, groups, split settings, and transform state in the pipeline contract. Validators can split by named coordinates, stateful steps refit inside each fold, and fold-invariant work can be reused across CV and tuning.

This cuts down custom split/cache code and catches many alignment and leakage errors earlier.

---

## Installation

```bash
# Core framework (minimal dependencies)
pip install xdflow

# With hyperparameter tuning
pip install xdflow[tuning]

# With all extras (LightGBM, visualization, MLflow, spectral analysis)
pip install xdflow[all]
```

**Requirements**: Python 3.11+

---

## Quick Example

This example works with the core install. A full runnable version lives at `examples/quickstart.py`.

```python
import numpy as np
import xarray as xr
from sklearn.linear_model import LogisticRegression

from xdflow.composite import Pipeline
from xdflow.core import DataContainer
from xdflow.cv import KFoldValidator
from xdflow.transforms.basic_transforms import FlattenTransform
from xdflow.transforms.normalization import ZScoreTransform
from xdflow.transforms.sklearn_transform import SKLearnPredictor

rng = np.random.default_rng(0)
stimuli = np.repeat(["rest", "tone", "odor"], 60)
rng.shuffle(stimuli)

values = rng.normal(0.0, 1.2, size=(stimuli.size, 4, 25))
values[stimuli == "tone", 1, 8:15] += 1.5
values[stimuli == "odor", 2, 14:22] += 1.5

data = xr.DataArray(
    values,
    dims=("trial", "channel", "time"),
    coords={
        "trial": np.arange(stimuli.size),
        "channel": [f"ch{i}" for i in range(4)],
        "time": np.linspace(-0.2, 0.8, 25),
        "stimulus": ("trial", stimuli),
    },
)
container = DataContainer(data)

pipeline = Pipeline(
    name="core_quickstart",
    steps=[
        ("zscore", ZScoreTransform(by_dim=["trial"])),
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

cv = KFoldValidator(
    n_splits=5,
    shuffle=True,
    random_state=0,
    stratify_coord="stimulus",
    scoring="f1_weighted",
    verbose=False,
)
cv.set_pipeline(pipeline)

score = cv.cross_validate(container, verbose=False)
print(f"Weighted F1: {score:.3f}")

# Coordinates stay attached, and stateful steps are refit per fold.
```

---

## Key Features

- **Dimension contracts**: transforms declare the dimensions they consume and produce, so invalid reshapes and step handoffs fail earlier.
- **Composition primitives**: sequential `Pipeline`, parallel `TransformUnion`, conditional `SwitchTransform`, per-group `GroupApplyTransform`, and toggleable `OptionalTransform`.
- **Leakage-aware cross-validation**: validators split on named trial, target, and group coordinates; stateful steps refit only on training folds; out-of-fold predictions are emitted for stacking and ensembles.
- **Fold-invariant caching**: expensive stateless work such as spectrograms or filtering can run once and be reused across folds and tuning trials.
- **Hyperparameter tuning**: Optuna-based search across pipeline architectures, reusable static-prefix caching across tuning trials, optional MLflow logging, and explicit seed management.
- **Pluggable estimators**: any class implementing the sklearn API — sklearn itself, LightGBM, XGBoost, or your own — drops into the pipeline via `SKLearnTransformer` and `SKLearnPredictor`.
- **Multi-output support**: a unified interface for multi-target regression, multilabel classification, and sample-weighted training with sklearn-compatible estimators.
- **Explicit API contracts**: focused transform and pipeline interfaces keep custom code small enough to write, test, review, and generate with LLM coding tools.

---

## Relation to sklearn and pipeline frameworks

`xdflow` does not replace sklearn. It wraps sklearn estimators when they fit, and keeps named dimensions and coordinates intact around them. It also does not replace pipeline orchestrators like Kedro or ZenML; an `xdflow` pipeline can live inside one of their nodes. The layer it fills is between: opinionated transform and CV semantics for data whose axes mean something.

---

## Documentation

**[Full Documentation](https://xdflow.readthedocs.io/en/latest/)**

**Quick Links**:

- [Data Contract](docs/concepts/data_contract.md)
- [Installation & Setup](docs/installation.md)
- [5-Minute Core Quickstart](docs/tutorials/quickstart.md)
- [Core Concepts](docs/concepts/index.md)
- [Tutorials](docs/tutorials/index.md)
- [Using XDFlow With LLMs](docs/guides/llm.md)
- [API Reference](docs/api/index.md)

---

## Contributing

We welcome contributions! Whether you're:

- Adding a new transform
- Improving documentation
- Reporting bugs
- Requesting features
- Sharing use cases

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Early adopters especially welcome** - the API is still stabilizing, and your feedback shapes the future of this project.

### Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/canaery/xdflow.git
cd xdflow
uv sync --all-extras    # creates .venv and installs everything

# Run commands via uv
uv run pytest                            # run tests
uv run ruff check xdflow tests examples  # lint

# Or activate the venv directly
source .venv/bin/activate
pytest
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [xarray](https://xarray.dev/) - Labeled multidimensional arrays
- [scikit-learn](https://scikit-learn.org/) - ML fundamentals
- [Optuna](https://optuna.org/) - Hyperparameter optimization
- [MLflow](https://mlflow.org/) - Experiment tracking

---

## Contact

- **Lead maintainer**: [Julien Bloch](https://github.com/julien-bloch)
- **GitHub Issues**: [Report bugs or request features](https://github.com/canaery/xdflow/issues)
- **Discussions**: [Ask questions, share ideas](https://github.com/canaery/xdflow/discussions)
