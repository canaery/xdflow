# Tuning API

Tuning APIs connect pipelines, validators, search spaces, and optional experiment tracking.

`Tuner` is the main programmatic interface. It evaluates candidate pipeline configurations through a `CrossValidator`, using Optuna for parameter suggestions and XDFlow's pipeline contracts for cloning, caching, and nested parameter setting.

::: xdflow.tuning.base.Tuner

## Helper Utilities

`run_tuning_pipeline` is a higher-level helper for running tuning over one or more prebuilt pipelines and returning finalized pipelines.

::: xdflow.tuning.tuner_utils.run_tuning_pipeline
