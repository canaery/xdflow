# xdflow

**Dimension-aware ML pipelines for scientific data**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

`xdflow` is a machine learning framework designed for structured, multidimensional scientific data. Built on [xarray](https://xarray.dev/), it brings reproducible, metadata-aware pipelines to domains where sklearn falls short: neuroscience, sensor arrays, time series, medical imaging, and any field working with labeled, high-dimensional data.

---

## The Problem

If you work with scientific data, you've probably hit these walls:

**sklearn pipelines break on structured data**
```python
# Your data has dimensions: (trials × channels × time × frequency)
# sklearn expects: (samples × features)
# You spend hours reshaping, lose metadata, break reproducibility
```

**No standard way to handle trial structure, sessions, or groups**
```python
# You need: "fit PCA per subject, then pool for classifier"
# sklearn offers: global fit() or manual loops
```

**Cross-validation doesn't respect your data's structure**
```python
# You need: "leave-one-session-out, stratify by condition"
# sklearn offers: basic K-fold, group CV with no stratification
```

**Transforms don't preserve metadata**
```python
# After 5 pipeline steps, you've lost track of which channel is which
# Debugging is impossible, reproducibility is a prayer
```

---

## The Solution

`xdflow` provides:

- ✅ **Dimension-aware transforms** that preserve labeled axes
- ✅ **Reproducible pipelines** with deterministic state tracking
- ✅ **Sophisticated cross-validation** that respects trial/session/subject structure
- ✅ **First-class metadata** propagation through every step
- ✅ **Flexible composition** patterns (sequential, parallel, conditional, per-group)
- ✅ **Native xarray integration** with seamless sklearn interop
- ✅ **Multi-output and few-shot transfer workflows** for scientific ML
- ✅ **Experiment tracking** with MLflow out of the box

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

values = rng.normal(0.0, 0.8, size=(stimuli.size, 4, 25))
values[stimuli == "tone", 1, 8:15] += 2.0
values[stimuli == "odor", 2, 14:22] += 2.0

container = DataContainer(
    xr.DataArray(
        values,
        dims=("trial", "channel", "time"),
        coords={
            "trial": np.arange(stimuli.size),
            "channel": [f"ch{i}" for i in range(4)],
            "time": np.linspace(-0.2, 0.8, 25),
            "stimulus": ("trial", stimuli),
        },
    )
)

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

## Installation

```bash
# Core framework (minimal dependencies)
pip install xdflow

# With hyperparameter tuning
pip install xdflow[tuning]

# With all extras (LightGBM, visualization, MLflow, spectral analysis)
pip install xdflow[all]
```

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
uv run pytest                         # run tests
uv run ruff check xdflow tests examples  # lint

# Or activate the venv directly
source .venv/bin/activate
pytest
```

**Requirements**: Python 3.11+

---

## Key Features

### 1. **Transform System**
All transforms follow a `fit()` / `transform()` / `fit_transform()` contract with:
- Automatic input/output dimension validation
- Deterministic state serialization (every transform is exactly reproducible)
- Metadata preservation (channel names, coordinates, etc. flow through)
- Immutability (safe for parallel execution, nested CV)

### 2. **Composite Transforms**
Build complex pipelines with:
- **Pipeline**: Sequential composition (`A → B → C`)
- **TransformUnion**: Parallel feature extraction (`[A, B, C] → concatenate`)
- **SwitchTransform**: Conditional selection (`if condition: A else: B`)
- **GroupApplyTransform**: Per-group fitting (`fit PCA separately per subject`)
- **OptionalTransform**: Toggle transforms on/off for ablation studies

### 3. **Intelligent Cross-Validation**
- Automatically separates **stateless** preprocessing (computed once) from **stateful** models (refitted per fold)
- Orders-of-magnitude speedup on expensive transforms (spectrograms, wavelets)
- Supports grouping, stratification, custom CV strategies
- Out-of-fold predictions for stacking/ensembles

### 4. **Hyperparameter Tuning**
- Optuna integration with Bayesian optimization
- Multi-pipeline comparison (compare architectures, not just hyperparams)
- Automatic MLflow logging
- Seed management for reproducibility

### 5. **Multi-Output Support**
- Native multi-target regression (predict multiple outputs simultaneously)
- Multilabel classification with sklearn-compatible estimators
- Proper handling of sample weights
- Classification and regression in unified interface

### 6. **LLM-Friendly API Contracts**
- Small transform and pipeline contracts that are easy to implement and review
- Clear separation between scientific computation, orchestration, and evaluation
- A scaffold for having LLMs write focused transforms instead of fragile one-off scripts

---

## Who Should Use This?

`xdflow` is designed for researchers and engineers working with:

**Domains**:
- Neuroscience (EEG, ECoG, MEG, calcium imaging, spike trains)
- Biosignals (ECG, EMG, respiration)
- Sensor arrays (industrial IoT, environmental monitoring)
- Medical time series (sleep studies, patient monitoring)
- Geophysical signals (seismology, climate data)
- Any labeled, multidimensional scientific data

**Use Cases**:
- You have metadata that must flow through your pipeline
- You need cross-validation that respects experiment structure
- You want reproducible experiments without custom infrastructure
- You're tired of reshaping data to fit sklearn's assumptions
- You need to compare dozens of pipeline architectures systematically

---

## Comparison to Other Tools

| Feature | xdflow | sklearn | Kedro/ZenML |
|---------|---------|---------|-------------|
| Dimension-aware transforms | ✅ | ❌ | ❌ |
| Metadata preservation | ✅ | ❌ | ⚠️ (manual) |
| Structured CV semantics | ✅ | ⚠️ (basic) | ❌ |
| xarray-native | ✅ | ❌ | ❌ |
| Stateful/stateless optimization | ✅ | ❌ | ❌ |
| Reproducible by default | ✅ | ⚠️ (manual) | ✅ |
| Scientific data focus | ✅ | ❌ | ❌ |
| Learning curve | Medium | Low | High |

**When to use sklearn**: Tabular data, classic ML problems, well-established workflows
**When to use Kedro/ZenML**: Large-scale MLOps, multi-team production deployments
**When to use xdflow**: Structured scientific data, experiment reproducibility, metadata-aware pipelines

---

## Documentation

**[Full Documentation](https://xdflow.readthedocs.io)**

**Quick Links**:
- [Data Contract](docs/concepts/data_contract.md) — understand the container + transform rules
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

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built on the shoulders of giants:
- [xarray](https://xarray.dev/) - Labeled multidimensional arrays
- [scikit-learn](https://scikit-learn.org/) - ML fundamentals
- [Optuna](https://optuna.org/) - Hyperparameter optimization
- [MLflow](https://mlflow.org/) - Experiment tracking

Inspired by the needs of the scientific computing community and years of building neural decoding pipelines.

---

## Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/canaery/xdflow/issues)
- **Discussions**: [Ask questions, share ideas](https://github.com/canaery/xdflow/discussions)

---

**Built by scientists, for scientists. Let's make reproducible ML the default.**
