# Data Contract

`xdflow` works on `xarray.DataArray` objects. `DataContainer` is a thin framework wrapper around a `DataArray`, not a separate data model. The wrapped xarray object remains the source of truth for values, dimensions, coordinates, and attrs.

The wrapper gives transforms, predictors, validators, and tuners a consistent object to pass around. It initializes `data.attrs["data_history"]`, rewraps xarray methods such as `sel` and `mean` when they return a new `DataArray`, and exposes the underlying array through `.data`.

The library does not require a fixed schema beyond a few conventions, but it assumes your data is labeled consistently enough for transforms to use dimensions by name.

The data contract is also what validators and tuners use at runtime. Dimensions and coordinates tell the framework what can be selected, split, transformed, scored, cached, and aligned without moving metadata into separate side arrays.

## Required structure

- supervised workflows are organized around a `trial` dimension
- target labels typically live in coordinates attached to `trial`
- transforms may require additional dimensions such as `channel`, `time`, `feature`, or `freq_band`

Each transform advertises dimension expectations through `input_dims` and `output_dims`, or computes them dynamically through `get_expected_output_dims`.

## Dimensions

- dimension names are part of the contract, not incidental metadata
- transforms should use dimension names such as `data.mean(dim="time")`
- positional-axis logic should be avoided unless a transform is explicitly reshaping into a new labeled representation

Common dimension names in this repo include:

- `trial`
- `channel`
- `time`
- `feature`
- `prediction`
- `freq_band`

Domain-specific labels are fine as long as they remain internally consistent.

## Runtime use

`xdflow` uses the data contract together with transform state to run pipelines:

- `sample_dim` identifies the independent sample axis for predictors
- target coordinates such as `stimulus` stay attached to that sample axis
- group coordinates such as `session`, `subject`, or `animal` can define split boundaries
- transform `input_dims` and `output_dims` catch invalid handoffs between steps
- `is_stateful` tells validators which steps must be cloned and refit per fold
- fold-invariant stateless steps can be reused across folds or tuning trials

Do not mark a transform stateless just because it has no fitted Python object. If it computes cross-sample statistics or otherwise depends on the validation split, model it as stateful or keep it after the split boundary.

## Transform history

`DataContainer` initializes a `data_history` list in the wrapped array attrs. The base `Transform` appends each completed transform with its class name and public parameters, so transformed outputs carry a lightweight provenance trail.

This history is useful for inspection and debugging. It is not a replacement for experiment tracking, model cards, or persisted pipeline configuration.

## Automatic validation

XDFlow uses transform dimension declarations as lightweight contract checks:

- adjacent pipeline steps are checked at construction time when one step declares concrete `output_dims` and the next declares concrete `input_dims`
- transforms with dynamic output shapes expose `get_expected_output_dims(input_dims)`
- `Pipeline(expected_input_dims=...)` validates the expected input dims for every step and checks handoffs through `get_expected_output_dims`
- during `fit_transform` and `transform`, `expected_input_dims` checks the actual dims before each step runs
- `pipeline.get_expected_output_dims(input_dims, print_steps=True)` shows expected dimension flow without running data
- `transform_sel` and `transform_drop_sel` write-back checks dims, sizes, and dimension coordinates before replacing a selected subset

These checks catch many invalid reshapes and step handoffs early, but they are not a full semantic schema validator. Transform authors should still validate required coordinates, attrs, units, size constraints, and domain-specific assumptions inside the transform.

See [Writing Custom Transforms](../guides/writing-transforms.md) for the authoring workflow and examples.

## Coordinates and attrs

Coordinates are the main place for labels and grouping metadata:

- class labels, sessions, animals, subjects, or conditions should be stored as coordinates
- timestamps and channel labels should remain attached to the relevant dimension
- additional metadata can live in `attrs`

Predictors and splitters often depend on coordinates such as `stimulus`, `session`, or `animal`, so those names need to exist on the data used by the relevant workflow.

## Immutability

Transforms are expected to behave functionally:

- `transform()` should return a new `DataContainer`
- the incoming container should not be mutated in place
- selective transforms that write results back into a larger array must do so on a copied container

This contract is exercised by the test suite, especially the transform immutability tests.

## Selection semantics

All transforms support optional selection arguments:

- `sel`: apply an `xarray.sel(...)` selection before the transform
- `drop_sel`: drop labels before the transform
- `transform_sel`: transform only a selected subset, then write it back into the original structure
- `transform_drop_sel`: inverse form of `transform_sel`

Selective in-place replacement is only valid for transforms that declare support for it via `_supports_transform_sel`.

## Fitted state

Stateful transforms must keep learned parameters out of the constructor so cloning stays safe during cross-validation.

Good pattern:

- constructor arguments are pure hyperparameters
- fitted artifacts are stored on private attributes such as `_estimator`, `_encoder`, or `_stats`

Bad pattern:

- populating constructor-declared attributes with learned values during `fit`

## Guidance for authoring new transforms

1. Define explicit constructor parameters and assign them to matching public attributes.
2. Validate dimensions and coordinate assumptions early.
3. Preserve or intentionally recompute coordinates when reshaping data.
4. Keep transform logic label-aware.
5. Add focused unit tests, including immutability coverage where applicable.

For examples and test patterns, see [Writing Custom Transforms](../guides/writing-transforms.md).
