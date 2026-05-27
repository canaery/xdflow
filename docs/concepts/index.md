# Core Concepts

`xdflow` is intentionally close to scikit-learn in workflow, but it changes the unit of computation from an unlabeled `ndarray` to a labeled `xarray.DataArray` wrapped by `DataContainer`.

Those labels are used by the runtime. Dimensions, coordinates, targets, groups, split settings, and transform state tell validators and tuners how to split, refit, cache, score, and keep predictions aligned.

## Data model

- The canonical input is `xarray.DataArray`.
- Dimensions are named and preserved through the pipeline.
- Coordinates carry metadata such as class labels, sessions, animals, or timestamps.
- `DataContainer` wraps the array and keeps operations returning containers instead of bare arrays.

The runtime expects at least a `trial` dimension for most supervised workflows. Additional dimensions such as `channel`, `time`, `frequency`, or `feature` are transform-specific.

## Transform model

Every processing step inherits from `Transform`.

- `fit` learns state when needed.
- `transform` returns a new `DataContainer`.
- `fit_transform` combines both operations.
- `predict` / `predict_proba` are added by `Predictor` subclasses.

Important conventions enforced throughout the library:

- transforms should operate on named dimensions, not positional axes
- incoming data should be treated as immutable
- learned state belongs on fitted attributes, not in constructor arguments
- new dimensions should receive descriptive names and coordinates where possible

See [Writing Custom Transforms](../guides/writing-transforms.md) for concrete authoring patterns, dimension checks, and test examples.

## Fold-invariant vs stateful steps

Cross-validation uses the `is_stateful` flag to decide where the fold boundary belongs:

- fold-invariant stateless transforms can run once before fold splitting
- stateful or split-dependent transforms are re-fit inside each fold

This matters for expensive preprocessing such as spectral feature extraction, where fold-invariant work should not be recomputed for every fold. A transform should only be marked stateless for CV reuse when it does not learn from held-out samples or depend on the split boundary.

## Composition patterns

The main composition tools are:

- `Pipeline`: sequential processing
- `TransformUnion`: parallel branches that are concatenated
- `GroupApplyTransform`: independent fitting per metadata-defined group
- `SwitchTransform` and `OptionalTransform`: conditional or toggleable branches
- `EnsemblePredictor`: weighted combination of multiple predictors

See [Composing Pipelines](composition.md) for examples and guidance on when to use each composite.

## Tuning

Tuning is a first-class XDFlow workflow. A tuner searches over pipeline parameters and architecture choices while the validator still owns splitting, scoring, refitting, and leakage boundaries.

The tuning install extra only controls dependency weight by installing Optuna. It does not make tuning a secondary part of the framework. See [Tuning](tuning.md) for the workflow model and [Hyperparameter Tuning](../tutorials/tuning.md) for a runnable pattern.

## Target handling

Predictors usually read labels from coordinates rather than from a separate `y` array. This keeps targets aligned with the same labeled trial axis used by the rest of the pipeline.

For details on required dimensions, coordinates, selection semantics, and immutability guarantees, see [Data Contract](data_contract.md).
