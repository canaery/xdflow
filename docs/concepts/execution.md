# Execution Model

XDFlow pipelines are ordinary transform graphs, but validators and tuners inspect those graphs before running them. The execution plan is based on three pieces of information:

- dimension and coordinate contracts on the data
- transform statefulness
- split and tuning settings

This is what lets XDFlow keep expensive preprocessing out of repeated loops while still refitting learned steps only where they are allowed to see the data.

## Cross-validation boundary

For a sequential `Pipeline`, cross-validation splits the pipeline at the first stateful step:

- the stateless prefix is fit/transformed once on the full input container
- the stateful suffix is cloned and fit separately inside each training fold
- validation data is passed through the fitted fold-specific suffix for prediction and scoring

For example:

```text
CARTransform -> MultiTaperTransform -> ZScoreTransform -> PCA -> LogisticRegression
stateless       stateless              stateless         stateful stateful
```

The common-average reference, spectral feature extraction, and per-trial z-scoring can run once before fold splitting if they are truly fold-invariant. PCA and the classifier must be fit only on each fold's training samples, so they live after the boundary.

This rule is simple, but it matters. It avoids recomputing deterministic preprocessing for every fold, and it prevents learned steps from seeing held-out samples.

## What counts as stateless

A transform should be stateless only when its output for one sample does not depend on held-out samples or on the split boundary.

Good stateless examples:

- reshaping or renaming dimensions
- flattening labeled feature dimensions
- deterministic per-sample transforms
- feature extraction that does not estimate statistics across trials

Stateful or split-dependent examples:

- PCA, whitening, or learned projections
- normalization fitted from training samples
- classifiers and regressors
- any transform that estimates cross-trial statistics

Do not mark a transform stateless just because it has no external estimator object. If it computes statistics across the sample dimension, make it stateful or place it after the split boundary.

## Tuning cache boundary

Tuning adds a second reuse opportunity. Before each trial, the tuner can split a pipeline into:

- a static prefix with no tunable or stateful steps
- a dynamic suffix containing tunable choices, stateful transforms, or predictors

The static prefix can be cached and reused as the input to multiple trials. The dynamic suffix is cloned, configured with the trial's parameters, and evaluated through the cross-validator.

This means a deterministic feature step before the first tunable choice can be computed once, while model families, optional steps, and hyperparameters remain part of the search.

## Limits and responsibilities

XDFlow can only plan correctly when transforms declare their behavior correctly:

- `is_stateful` must be true for learned or split-dependent operations
- custom transforms should preserve coordinates unless they intentionally change the data contract
- predictors should read targets from coordinates instead of side arrays
- validators should be used for split logic instead of external train/test loops

The framework catches many shape, alignment, and leakage mistakes early, but it is not a substitute for domain-specific validation. Transform authors still need to validate required coordinates, units, sampling assumptions, and scientific constraints inside their transforms.

## Current data scope

XDFlow currently operates on in-memory `xarray.DataArray` objects wrapped by `DataContainer`. Pipelines can run inside larger workflow systems, but XDFlow itself is not an Airflow, Dagster, Kedro, or ZenML replacement. Its responsibility is the scientific ML pipeline layer: labeled transforms, validation, tuning, caching, scoring, and aligned outputs.

Out-of-core and Dask-backed execution are future-facing areas rather than part of the current core runtime.
