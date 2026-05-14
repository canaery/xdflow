# xdflow

**Dimension-aware ML pipelines for scientific data**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

`xdflow` is a machine learning framework designed for structured, multidimensional scientific data. Built on [xarray](https://xarray.dev/), it brings reproducible, metadata-aware pipelines to domains where sklearn falls short: neuroscience, sensor arrays, time series, medical imaging, and any field working with labeled, high-dimensional data.

---

## The Problem

sklearn assumes `(samples, features)`. Scientific data isn't shaped like that — and the workarounds leak, lose metadata, or both.

Real samples carry structure: trials, channels, time bins, frequency bands, sessions, subjects, conditions, devices. The standard Python ML stack forces that into a 2D array plus side tables, and four failure modes follow.

**Reshaping erases the contract.** Flatten `trial × channel × time` into `sample × feature` and you lose the names that explain what each axis means. Is feature 143 channel `C3` at 220 ms, beta power in a window, or the output of a previous projection? Shape mistakes still produce valid arrays and plausible metrics, so the bugs survive past the point they should have been caught.

**Metadata becomes side-channel bookkeeping.** `stimulus`, `session`, `subject`, `time_offset_ms` live in separate arrays from the data they describe:

```text
# Before flattening — coordinates are attached to the data
<xarray.DataArray (trial: 180, channel: 4, time: 25)>
Coordinates:
  * channel    (channel) <U3 'ch0' 'ch1' 'ch2' 'ch3'
  * time       (time) float64 -0.2 -0.158 ... 0.758 0.8
    stimulus   (trial) <U4 'odor' 'rest' 'tone' ...

# After sklearn reshape — coordinates are gone, you maintain them by hand
X.shape           # (180, 100)
y = stimuli       # separate array, aligned by convention
channels = ...    # another separate array
times = ...       # another separate array
```

Every filter, split, resample, prediction, and score then has to keep those side arrays aligned by hand. Intermediate outputs become opaque because the data no longer says what each sample or feature represents.

**Cross-validation forces a leakage/speed tradeoff.** Some transforms are expensive but stateless (fixed filtering, feature extraction). Others must be learned inside each training fold (PCA, normalization statistics, classifiers, domain adapters). Without an explicit pipeline model, the choice is: recompute everything per fold (safe but slow), or preprocess once (fast but leaks).

**Structured evaluation doesn't fit basic split loops.** Real experiments need leave-one-session-out, leave-one-subject-out, stratification by condition, trial grouping, few-shot target-domain sampling, or scoring rules that exclude specific event types or offsets. In plain sklearn, each of these becomes a custom split loop with manually synchronized metadata.

---

## The Solution

`xdflow` provides:

- ✅ **Dimension-aware transforms** with explicit shape contracts and labeled axes
- ✅ **First-class metadata** that propagates through every step
- ✅ **Leakage-aware cross-validation** that separates stateless preprocessing from per-fold fitted steps, with structured splits across trial/session/subject/domain
- ✅ **Reproducible pipelines** with deterministic state tracking
- ✅ **Flexible composition** patterns (sequential, parallel, conditional, per-group)
- ✅ **Native xarray integration** with seamless sklearn interop
- ✅ **Multi-output and few-shot transfer workflows** for scientific ML
- ✅ **Optional experiment tracking** with MLflow

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
A `fit()` / `transform()` / `fit_transform()` contract with input/output dimension validation, deterministic state serialization, and immutability — safe for parallel execution and nested CV.

### 2. **Composite Transforms**
Concrete building blocks for pipelines that branch, switch, or specialize per group:
- **Pipeline**: Sequential composition (`A → B → C`)
- **TransformUnion**: Parallel feature extraction (`[A, B, C] → concatenate`)
- **SwitchTransform**: Conditional selection (`if condition: A else: B`)
- **GroupApplyTransform**: Per-group fitting (`fit PCA separately per subject`)
- **OptionalTransform**: Toggle transforms on/off for ablation studies

### 3. **Cross-Validation**
Stateless preprocessing is computed once; stateful steps are refit per fold — large speedups on expensive transforms like spectrograms and wavelets. Out-of-fold predictions are emitted for stacking and ensembles.

### 4. **Hyperparameter Tuning**
Optuna-based Bayesian optimization with multi-pipeline comparison (compare architectures, not just hyperparams), optional MLflow logging, and explicit seed management.

### 5. **Multi-Output Support**
A unified interface for multi-target regression, multilabel classification, and sample-weighted training with sklearn-compatible estimators.

### 6. **LLM-Friendly API Contracts**
Transform and pipeline contracts are small enough that an LLM can write a focused, reviewable transform instead of a fragile one-off script.

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

XDFlow grew out of research and engineering work by Julien Bloch and Joanna Chang, who worked closely together on the reusable scientific ML infrastructure behind the project. Julien shaped much of the core architecture, Joanna contributed substantially across development and research use cases, and Yash helped with the final push to prepare the project for open source release.

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
