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
uv run ruff check xdflow tests examples
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

## Read the Docs publishing

Read the Docs uses `.readthedocs.yaml`, installs the `docs` extra from `pyproject.toml`, and builds the site with `mkdocs.yml`.

To publish documentation updates:

1. Add or edit pages under `docs/`.
2. Add new public pages to the `nav` section in `mkdocs.yml`.
3. Run `uv run mkdocs build --strict` locally.
4. Push or merge to the branch configured as the Read the Docs default branch.

The GitHub Actions docs job also runs `mkdocs build --strict`, so docs failures should be visible before tagging or publishing a release.

## Writing docs for new features

When adding a new transform or workflow:

1. Update the relevant concept or tutorial page if the user-facing behavior changes.
2. Link to or update [Writing Custom Transforms](guides/writing-transforms.md) or [Writing Custom Cross-Validators](guides/writing-cross-validators.md) when extension behavior changes.
3. Add or expand API documentation if the module belongs to the core dependency set.
4. Keep examples based on `xarray.DataArray` and make dimensions, coordinates, targets, groups, and split policies explicit.
5. Add tests alongside the feature when behavior or contracts change.
