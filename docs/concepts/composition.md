# Composing Pipelines

XDFlow is built around one compositional rule: every processing unit is a `Transform`.

A transform takes a `DataContainer`, does one labeled operation, and returns a `DataContainer`. A `Pipeline` is also a transform. Composite transforms are transforms too; they just contain other transforms and define how those children are executed.

This is the composite design pattern in practice: simple transforms and composed transforms expose the same interface. A `Pipeline`, `TransformUnion`, `SwitchTransform`, or `GroupApplyTransform` can be passed anywhere a single `Transform` is expected because all of them support the same core methods.

This means you can build small reusable pieces, combine them into pipelines, put pipelines inside larger pipelines, and still use the same `fit`, `transform`, `fit_transform`, `predict`, and cross-validation APIs.

## Sequential composition

`Pipeline` is the standard linear composition tool. It chains named steps and sends the output of each step into the next one.

```python
from xdflow.composite import Pipeline
from xdflow.transforms.basic_transforms import AverageTransform, FlattenTransform

feature_pipeline = Pipeline(
    name="trial_features",
    steps=[
        ("average_time", AverageTransform(dims="time")),
        ("flatten", FlattenTransform(dims=("channel", "freq_band"))),
    ],
)
```

Because `Pipeline` is a transform, it can be used anywhere a single transform can be used. A feature pipeline can become one branch of a union, one choice in a switch, or the inner transform applied per group.

## Choosing between alternatives

`SwitchTransform` represents one step with multiple possible implementations. It is useful when you want to choose between preprocessing strategies, feature extractors, or predictors without rewriting the surrounding pipeline.

```python
from xdflow.composite import SwitchTransform
from xdflow.transforms.basic_transforms import AverageTransform, IdentityTransform

feature_choice = SwitchTransform(
    choices=[
        ("raw", IdentityTransform()),
        ("time_average", AverageTransform(dims="time")),
    ],
    choose="time_average",
)
```

The selected branch can also be supplied later through `choose`, which makes `SwitchTransform` useful for tuning workflows. All choices should produce compatible output dimensions for the same input.

## Optional steps

`OptionalTransform` is a convenience wrapper for a common switch pattern: use a transform or skip it.

```python
from xdflow.composite import OptionalTransform
from xdflow.transforms.normalization import ZScoreTransform

maybe_zscore = OptionalTransform(
    ZScoreTransform(by_dim="channel", use_fit=True),
    use=True,
)
```

This is the clearest way to express a toggleable step in a pipeline or tuner search space. The skip branch is an identity transform, so the wrapped transform should preserve dimension names if both branches need to fit in the same validated pipeline position.

## Parallel feature branches

`TransformUnion` runs multiple transforms on the same input and concatenates their outputs. It is for feature pipelines that should be computed independently and then joined into one representation.

```python
from xdflow.composite import Pipeline, TransformUnion
from xdflow.transforms.basic_transforms import AverageTransform, FlattenTransform

time_branch = Pipeline(
    name="time_branch",
    steps=[
        ("average_time", AverageTransform(dims="time")),
    ],
)

channel_branch = Pipeline(
    name="channel_branch",
    steps=[
        ("average_channel", AverageTransform(dims="channel")),
    ],
)

features = TransformUnion(
    transforms_list=[time_branch, channel_branch],
    from_dims=["channel", "time"],
    to_dim="feature",
)
```

Each branch sees the same input container. The branch outputs must agree on all dimensions except the declared branch-specific join dimension in `from_dims`. The union renames or concatenates those branch dimensions into `to_dim`.

`UnionWithInput` is a shortcut for the case where one branch should be the original input and the other branch should be a transformed version of that input.

```python
from xdflow.composite import UnionWithInput

augmented = UnionWithInput(
    transform_template=time_branch,
    join_dim="channel",
)
```

## Per-group composition

`GroupApplyTransform` clones a template transform once per group, fits each clone on that group's data, applies each clone to that same group, and reassembles the result.

```python
from xdflow.composite import GroupApplyTransform
from xdflow.transforms.normalization import ZScoreTransform

subject_norm = GroupApplyTransform(
    group_coord="subject",
    transform_template=ZScoreTransform(by_dim="channel", use_fit=True),
)
```

Use this when the correct fitted state is group-specific: per-subject normalization, per-session preprocessing, separate device calibration, or separate models per condition. The grouping variable stays in the data as a coordinate, so the pipeline does not need a side vector passed through split and transformation code.

`GroupApplyTransform` is also useful with stateless templates when the operation should be constrained to one group at a time. If the template is stateful, each group gets its own fitted clone; if the template is stateless, each group is transformed independently and then reassembled.

The group coordinate must index exactly one dimension, and the inner transform must preserve the grouped axis so outputs can be reassembled.

For multiple coordinates, pass a list. XDFlow combines those coordinate values into a group key:

```python
per_animal_session = GroupApplyTransform(
    group_coord=["animal", "session"],
    transform_template=ZScoreTransform(by_dim="channel", use_fit=True),
)
```

## Nested parameters

Composite transforms support nested parameter setting with `__`, similar to scikit-learn.

```python
pipeline.set_params(
    feature_choice__choose="raw",
    classifier__estimator__C=0.5,
)
```

The first segment names the child step; the remaining segments are passed to that child. This is what lets tuners search over choices and hyperparameters without breaking the pipeline into manual loops.

## Choosing the right composite

Use `Pipeline` when steps are sequential.

Use `SwitchTransform` when one pipeline position has several possible implementations.

Use `OptionalTransform` when the choice is simply use or skip.

Use `TransformUnion` when independent feature branches should be concatenated.

Use `UnionWithInput` when a transformed branch should be concatenated with the original input.

Use `GroupApplyTransform` when fitting or transforming must happen independently per metadata-defined group.

These tools stay modular because they all preserve the same transform contract: `DataContainer` in, `DataContainer` out.

For the custom transform side of that contract, see [Writing Custom Transforms](../guides/writing-transforms.md).
