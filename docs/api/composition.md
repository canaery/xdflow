# Composition API

Composition APIs define how transforms are combined while keeping the pipeline visible to validators and tuners. Use them for branching, per-group fitting, optional steps, and ensembles when those choices should participate in split, refit, and cache planning.

## `xdflow.composite.pipeline`

Primary class:

- `Pipeline`: sequential composition of named steps

Use it when transforms should run in order and each step consumes the previous step's output. Named steps give validators and tuners a graph they can clone, split, cache, and refit.

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
- avoid hand-written loops over metadata groups

## `xdflow.composite.transform_union`

Primary classes:

- `TransformUnion`: run parallel branches and concatenate outputs
- `UnionWithInput`: same as `TransformUnion` but also keeps the original input branch

Use these when several feature extractors should share the same fold and caching rules.

## `xdflow.composite.switch_transform`

Primary classes:

- `SwitchTransform`: choose between transform branches
- `OptionalTransform`: enable or disable a transform through configuration

Use these for ablations and architecture comparisons without moving conditional logic outside the pipeline.

## `xdflow.composite.ensemble`

Primary classes:

- `EnsemblePredictor`
- `EnsembleMember`

This module provides weighted combination of multiple predictors, including optional score-based weighting.
