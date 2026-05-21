# API Reference

The API pages document the stable, core parts of the library that build without optional runtime extras: labeled containers, dimension-aware transforms, composable pipelines, validators, and utilities for target and metadata handling.

Included here:

- core container and transform interfaces
- composition primitives
- cross-validation orchestrators
- common transforms used by the current tutorial flow
- utility helpers that are part of the public package surface

Some advanced modules in `xdflow.transforms` depend on optional extras such as LightGBM, Optuna integrations, or domain-adaptation packages. Those modules are intentionally not imported into the published API site unless their dependency set is part of the docs build.
