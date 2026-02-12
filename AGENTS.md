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
- `mypy xdflow` performs static type checks on library code.

## Coding Style & Naming Conventions
- Python 3.9+ with 4-space indentation.
- Formatting and linting are enforced via Ruff (line length 120, double quotes).
- Use descriptive snake_case for functions and modules; class names use CapWords.
- Keep public APIs in `xdflow/__init__.py` minimal and explicit.

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
