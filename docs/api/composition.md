# Composition API

## `xdflow.composite.pipeline`

Primary class:

- `Pipeline`: sequential composition of named steps

Use it when transforms should run in order and each step consumes the previous step's output.

```python
from xdflow.composite.pipeline import Pipeline
```

## `xdflow.composite.group_apply`

Primary class:

- `GroupApplyTransform`: clones and applies a transform independently per group coordinate

Typical use cases:

- fit preprocessing per subject
- fit a model per session
- aggregate unseen groups with configured fallback behavior

## `xdflow.composite.transform_union`

Primary classes:

- `TransformUnion`: run parallel branches and concatenate outputs
- `UnionWithInput`: same as `TransformUnion` but also keeps the original input branch

## `xdflow.composite.switch_transform`

Primary classes:

- `SwitchTransform`: choose between transform branches
- `OptionalTransform`: enable or disable a transform through configuration

## `xdflow.composite.ensemble`

Primary classes:

- `EnsemblePredictor`
- `EnsembleMember`

This module provides weighted combination of multiple predictors, including optional score-based weighting.
