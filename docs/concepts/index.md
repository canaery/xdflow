# Core Concepts

`xdflow` is intentionally close to scikit-learn in workflow, but it changes the unit of computation from an unlabeled `ndarray` to a labeled `xarray.DataArray` wrapped by `DataContainer`.

## Data model

- The canonical input is `xarray.DataArray`.
- Dimensions are named and preserved through the pipeline.
- Coordinates carry metadata such as class labels, sessions, animals, or timestamps.
- `DataContainer` wraps the array and ensures operations keep returning containers instead of bare arrays.

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

## Stateless vs stateful steps

Cross-validation is optimized around the `is_stateful` flag:

- stateless transforms can run once before fold splitting
- stateful transforms are re-fit inside each fold

This matters for expensive preprocessing such as spectral feature extraction, where reusing stateless work can remove a large amount of repeated computation.

## Composition patterns

The main composition tools are:

- `Pipeline`: sequential processing
- `TransformUnion`: parallel branches that are concatenated
- `GroupApplyTransform`: independent fitting per metadata-defined group
- `SwitchTransform` and `OptionalTransform`: conditional or toggleable branches
- `EnsemblePredictor`: weighted combination of multiple predictors

## Target handling

Predictors usually read labels from coordinates rather than from a separate `y` array. This keeps targets aligned with the same labeled trial axis used by the rest of the pipeline.

For details on required dimensions, coordinates, selection semantics, and immutability guarantees, see [Data Contract](data_contract.md).
