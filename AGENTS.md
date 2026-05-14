# Repository Guidelines

## Project Structure & Module Organization
- `xdflow/` holds the core library modules (e.g., `core/`, `transforms/`, `cv/`, `tracking/`).
- `tests/` mirrors library packages (`tests/transforms`, `tests/cv`, etc.) and contains `conftest.py`.
- `docs/` contains conceptual and development notes; `examples/` has runnable usage samples.
- `scripts/` is reserved for helper utilities and project automation.

## Build, Test, and Development Commands
- `python -m pip install -e ".[dev]"` installs the package in editable mode with lint/test tools.
- `pytest` runs the full test suite (use `-m "not slow"` to skip slow tests).
- `ruff check .` runs lint checks; `ruff format .` auto-formats the codebase.
- `uvx ty check xdflow` performs static type checks on library code.

## Coding Style & Naming Conventions
- Python 3.11+ with 4-space indentation.
- Formatting and linting are enforced via Ruff (line length 120, double quotes).
- Use descriptive snake_case for functions and modules; class names use CapWords.
- Keep public APIs in `xdflow/__init__.py` minimal and explicit.

## XDFlow Agent Contract
- Treat `xdflow` as a framework, not a script collection. Prefer adding small `Transform`, `Predictor`, `Pipeline`, or `CrossValidator` pieces over writing manual analysis loops.
- Data should flow through `DataContainer`, which wraps an `xarray.DataArray`. Preserve dimension names and coordinates unless a transform intentionally changes them.
- Use dimension names (`trial`, `channel`, `time`, `feature`, etc.) instead of positional axes whenever possible.
- Transforms must not mutate their input container. Return a new `DataContainer` or an immutable view consistent with existing transform behavior.
- All transform hyperparameters should be explicit `__init__` arguments and public attributes with matching names. Store learned/fitted state in private attributes outside constructor parameters.
- Do not override `clone()` for new transforms unless there is a strong reason. Prefer `_get_clone_kwargs()` when constructor kwargs need special handling.
- For sklearn-compatible models, prefer `SKLearnTransformer` and `SKLearnPredictor` over raw estimator calls. Use `sample_dim="trial"` for trial-level prediction.
- Use `is_multilabel=True` only for classifiers with multiple binary target coordinates. Use multi-target regression for continuous multi-output targets.
- Use `xdflow.cv` validators for train/validation/holdout logic. Do not write ad hoc split loops unless implementing a new validator.
- Keep expensive stateless computations inside stateless transforms so `CrossValidator` can compute them once and refit only stateful steps per fold.
- Add focused tests for output dims, coordinate preservation, immutability, clone behavior, and CV scoring/splitting when behavior changes.

## Testing Guidelines
- Tests are written with `pytest` and live under `tests/`.
- Test files follow `test_*.py`, functions `test_*`, and classes `Test*`.
- Use markers from `pyproject.toml` (e.g., `@pytest.mark.slow`).

## Commit & Pull Request Guidelines
- Follow a concise, conventional style: `feat: add group cv strategy`, `fix: handle empty coords`.
- Keep commits focused; avoid mixing refactors with behavior changes.
- PRs should include a clear summary, testing notes, and links to related issues or docs.

## Configuration & Dependencies
- Optional extras are defined in `pyproject.toml` (e.g., `.[torch]`, `.[tuning]`, `.[all]`).
- Prefer reproducible examples that use `xarray.DataArray` inputs and preserve metadata.
