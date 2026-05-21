# Core API

## Top-level exports

The root package currently exposes the main workflow primitives:

```python
from xdflow import (
    CrossValidator,
    DataContainer,
    GroupApplyTransform,
    OptionalTransform,
    Pipeline,
    Predictor,
    SwitchTransform,
    Transform,
    TransformUnion,
)
```

`Tuner` is conditionally exported when the tuning extra is installed.

## `xdflow.core.data_container`

Key objects:

- `DataContainer`: immutable wrapper around `xarray.DataArray` that keeps data, dimensions, coordinates, attrs, and history together
- `TransformError`: shared exception used across pipeline execution

Common usage:

```python
from xdflow.core.data_container import DataContainer
```

Important behaviors:

- `DataContainer.data` returns the wrapped `DataArray`
- many `xarray.DataArray` methods are proxied and rewrapped into a new `DataContainer`
- `data_history` is stored in `attrs`
- coordinates remain available for predictors, validators, scoring, and downstream alignment

## `xdflow.core.base`

Key base classes:

- `Transform`: base class for all preprocessing and reshaping steps
- `Predictor`: mixin-style transform subclass for predictive endpoints
- `SampleWeightMixin`: helper for transforms that propagate sample weights

Common authoring expectations for subclasses:

- declare hyperparameters explicitly in `__init__`
- declare consumed and produced dimensions where they are known
- implement `_transform` and, for stateful steps, `_fit`
- keep learned state off constructor parameters so cloning remains safe
- mark a transform stateless only when it is safe to reuse before CV split boundaries
