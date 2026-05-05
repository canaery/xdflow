# Development

## Local workflow

Common commands for working on the repo:

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format .
uvx ty check xdflow
```

If you use the checked-in lockfile:

```bash
uv sync --extra dev --extra docs
uv run pytest
uv run mkdocs serve
```

## Documentation maintenance

The docs site is built with MkDocs Material and published via Read the Docs.

Important files:

- `mkdocs.yml`: site structure, theme, plugins, and nav
- `.readthedocs.yaml`: RTD build configuration
- `docs/`: Markdown content and static docs assets

Before submitting doc changes, run:

```bash
uv run mkdocs build --strict
```

## Writing docs for new features

When adding a new transform or workflow:

1. Update the relevant concept or tutorial page if the user-facing behavior changes.
2. Add or expand API documentation if the module belongs to the core dependency set.
3. Keep examples based on `xarray.DataArray` and preserve dimension/coordinate semantics.
4. Add tests alongside the feature when behavior or contracts change.
