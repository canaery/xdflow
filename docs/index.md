# xdflow

`xdflow` is a framework for building machine learning pipelines on labeled scientific data. It keeps `xarray` dimensions and coordinates intact while still exposing familiar `fit`, `transform`, `predict`, and cross-validation workflows.

## What it is for

Use `xdflow` when your data is more structured than a plain 2D feature matrix:

- neural recordings with `trial`, `channel`, and `time` dimensions
- sensor arrays with per-sample metadata such as subject, session, or condition
- multidimensional features where dimension names must survive preprocessing

The library is built around a small set of abstractions:

- `DataContainer` wraps an `xarray.DataArray` and preserves data history
- `Transform` defines immutable, dimension-aware preprocessing steps
- `Pipeline` composes transforms and predictors into reusable workflows
- `CrossValidator` runs structure-aware evaluation while separating stateless and stateful steps

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
