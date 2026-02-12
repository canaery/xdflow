# Contributing to xdflow

Thanks for your interest in contributing! We welcome bug reports, feature requests, docs improvements, and new transforms.

## Development setup

```bash
python -m pip install -e ".[dev]"
```

## Tests and checks

```bash
pytest
ruff check .
ruff format .
mypy xdflow
```

## Guidelines

- Keep changes focused and easy to review.
- Prefer reproducible examples that use `xarray.DataArray`.
- Add or update tests for behavior changes.
- Follow the style config in `pyproject.toml` (Ruff, line length 120).

## Submitting a pull request

1. Create a topic branch.
2. Ensure tests and lint pass.
3. Open a PR with a clear summary and testing notes.
