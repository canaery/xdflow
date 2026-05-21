# XDFlow

`xdflow` is a framework for building machine learning pipelines on labeled scientific data. It keeps `xarray` dimensions and coordinates with the data as it moves through transforms, predictors, validators, and tuners.

In NumPy/sklearn-style code, targets, sessions, subjects, channels, and timestamps often become side arrays. `xdflow` keeps those labels in the `DataContainer` and pipeline contract, so validators and tuners can split, refit, cache, score, and align predictions using named coordinates instead of positional conventions.

## Why use it

Use `xdflow` when you do not want to rewrite split, cache, refit, and metadata-alignment logic for every experiment:

- **Faster CV and tuning**: fold-invariant stateless transforms can run once and be reused across folds, tuning trials, and pipeline comparisons.
- **Better leakage safety**: stateful or split-dependent steps are cloned and refit inside each training fold.
- **Metadata tracking**: targets, groups, sessions, subjects, channels, timestamps, and predictions stay attached to the data instead of drifting into side arrays.
- **Named-coordinate validation**: validators can split, group, or stratify by coordinates such as `stimulus`, `session`, `subject`, or `animal`.
- **Modular pipelines**: transforms, predictors, unions, switches, and per-group steps compose without handwritten split loops.
- **Automatic split, cache, and refit planning**: validators and tuners use dimension contracts, coordinates, split settings, and transform state to decide what can be reused and what must be refit.

These patterns show up in neural recordings, biosignals, sensor arrays, medical time series, geophysical data, and other labeled datasets where the metadata is part of the experiment.

The library is built around a small set of abstractions:

- `DataContainer` wraps an `xarray.DataArray` and keeps dimensions, coordinates, attrs, and data history together
- `Transform` defines immutable preprocessing steps with explicit dimension and state contracts
- `Pipeline` composes transforms and predictors into reusable workflows
- `CrossValidator` runs structure-aware evaluation while separating fold-invariant stateless work from stateful or split-dependent steps

## Package layout

- `xdflow.core`: base container and transform interfaces
- `xdflow.composite`: sequential, grouped, conditional, and ensemble composition
- `xdflow.cv`: cross-validation orchestration and splitting strategies
- `xdflow.transforms`: preprocessing, sklearn adapters, spectral transforms, and predictors
- `xdflow.utils`: sampling, caching, target resolution, and plotting helpers
- `examples/`: runnable scripts for public workflows

## Start here

1. Read [Installation](installation.md) to set up the package and optional extras.
2. Read [Concepts](concepts/index.md) for the runtime model and transform rules.
3. Run the [5-Minute Core Quickstart](tutorials/quickstart.md) for a base-install example.
4. Follow the [Spectral Pipeline Walkthrough](tutorials/basic-pipeline.md) for an end-to-end signal-processing example.
5. Read [Reusable ML Patterns](tutorials/reusable-ml-patterns.md) for multilabel, weighting, and domain-transfer examples.
6. Use [XDFlow With LLMs](guides/llm.md) when asking an LLM to implement against the framework.
7. Use the [API Reference](api/index.md) for class and function details.
