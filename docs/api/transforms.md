# Transforms API

These modules cover the transforms used by the current maintained docs and the core dependency set.

## `xdflow.transforms.basic_transforms`

Common classes:

- `IdentityTransform`
- `AverageTransform`
- `FlattenTransform`
- `UnflattenTransform`
- `FunctionTransform`
- `CropTimeTransform`
- `SampleWeightTransform`

## `xdflow.transforms.cleaning`

Common classes:

- `CARTransform`
- `RegressOutReferenceTransform`
- `RemoveMissingBanksTransform`
- `RemoveOutliersTransform`

## `xdflow.transforms.normalization`

Common classes:

- `DemeanTransform`
- `ZScoreTransform`

## `xdflow.transforms.spectral`

Common classes:

- `MultiTaperTransform`
- `BandpassFilterTransform`

These operate on time-series style arrays and typically expect `trial`, `channel`, and `time` dimensions.

## `xdflow.transforms.sklearn_transform`

Common classes:

- `SKLearnTransform`
- `SKLearnTransformer`
- `SKLearnPredictor`

These adapters let you wrap scikit-learn estimators while preserving `xdflow`'s labeled data model.

## Advanced optional modules

The transform package also contains additional modules that may require optional extras or more specialized dependencies, including:

- `xdflow.transforms.lgbm_predictor`
- `xdflow.transforms.domain_adaptation`
- `xdflow.transforms.adapt_wrapper`
- `xdflow.transforms.zca`
- `xdflow.transforms.spatial`
- `xdflow.transforms.lda`
- `xdflow.transforms.pca`
