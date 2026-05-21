# Installation

## Core install

```bash
pip install xdflow
```

This installs the core runtime for metadata-driven pipelines built on `xarray`, `numpy`, `pandas`, `scikit-learn`, `scipy`, and `joblib`.

## Optional extras

Install only the dependency sets you need:

```bash
pip install "xdflow[tuning]"
pip install "xdflow[lightgbm]"
pip install "xdflow[mlflow]"
pip install "xdflow[viz]"
pip install "xdflow[spectral]"
pip install "xdflow[adaptation]"
pip install "xdflow[all]"
```

Current extras in `pyproject.toml`:

- `tuning`: Optuna-backed tuning support. Tuning is a core XDFlow workflow; the extra only keeps optimizer dependencies out of the minimal install.
- `lightgbm`: LightGBM predictor wrapper
- `mlflow`: experiment tracking integration
- `viz`: plotting helpers
- `spectral`: spectral-analysis dependencies
- `adaptation`: domain-adaptation dependencies
- `all`: all optional runtime extras

## Development install

For editable development with linting and test tools:

```bash
python -m pip install -e ".[dev]"
```

This repository also includes a `uv.lock`, so a reproducible local setup can be created with:

```bash
uv sync --extra dev --extra docs
```

For tuning development or docs examples:

```bash
uv sync --extra dev --extra docs --extra tuning
```

Run the core quickstart example from the repository root:

```bash
uv run python examples/quickstart.py
```

To run the spectral tutorial examples locally, include the spectral extra:

```bash
uv sync --extra dev --extra docs --extra spectral
```

## Documentation build

Local docs build:

```bash
uv run mkdocs build --strict
```

The published docs are configured for Read the Docs through the top-level `.readthedocs.yaml` and `mkdocs.yml`.
