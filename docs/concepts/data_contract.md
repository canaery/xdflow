# Data & Dimension Contract

`xdflow` pipelines operate on `xarray.DataArray` objects wrapped by `DataContainer`. This section defines the minimum contract expected by the runtime so users can adapt their own scientific data with confidence.

## Dimensions

- Pipelines are dimension-aware: every transform declares `input_dims` / `output_dims`.
- The only dimension **required** by the base container is `trial`. Everything else is optional but, when present, must be named consistently.
- Common dims: `trial`, `channel`, `time`, `frequency`, `feature`. You can add domain-specific dims (`subject`, `roi`, `phase`) without touching the runtime as long as you keep them labeled.
- Transforms must refer to dims by name (`data.mean(dim="time")`) and never rely on positional axes.

## Coordinates & metadata

- Coordinates store metadata such as channel names, timestamps, subject IDs, etc.
- `DataContainer` keeps coordinates immutable; transforms should modify `.data` but avoid mutating `.coords` unless that is the goal of the transform (e.g., renaming dims).
- When creating new dims, generate descriptive coordinate labels automatically whenever possible.
- Attach experiment metadata via coordinates/attrs rather than relying on ad-hoc dictionaries.

## Selection semantics

- All transforms accept optional `sel`, `drop_sel`, `transform_sel`, and `transform_drop_sel` arguments.
- `sel` / `drop_sel` run before the transform executes and effectively narrow or drop data along labeled dimensions.
- `transform_sel` / `transform_drop_sel` operate **inside** the transform by copying the container, applying the transform on a sub-selection, and writing back into the full array. Transforms must declare `_supports_transform_sel = True` to enable this advanced mode.
- Any transform exposing these selectors must preserve immutability: never mutate the incoming container or its `.data`.

## Immutability & history

- All `transform()` implementations must return a **new** `DataContainer`.
- Tests in `tests/test_transform_immutability.py` enforce this contract: they fail if a transform mutates the original container or returns the same object.
- `DataContainer` automatically maintains `data_history` in `xarray` attrs; transform authors should append summaries (e.g., transform name + params) so downstream debugging is trivial.

## Fit/transform separation

- Stateless transforms (`is_stateful = False`) may run once and be reused across cross-validation folds.
- Stateful transforms must keep learned parameters out of `__init__` so cloning/cloning-based CV works.
- Any learned state should be stored on private attributes (e.g., `_encoder`), and those attributes should be reset when `fit()` is called.

## Best practices for new transforms

1. **Explicit constructor signature** – All hyperparameters must be explicit keyword arguments recorded on `self`.
2. **Type hints** – Use Python typing (≥3.9) for clarity.
3. **Dimension checks** – Validate `container.data.dims` early; raise `ValueError` when expectations are not met.
4. **Metadata propagation** – When operations reduce dims, propagate or recompute coords/attrs so users keep interpretability.
5. **Testing** – Add cases to `tests/test_transform_immutability.py` and supply focused unit tests under `tests/transforms/` describing the new behavior.

Following this contract keeps every pipeline reproducible, composable, and predictable—regardless of the scientific domain layered on top.
