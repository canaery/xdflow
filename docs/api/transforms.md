# Transforms API

These modules cover the transforms used by the current maintained docs. Most are available with the core dependency set; modules noted as optional require their matching extra.

Validators and tuners use transform metadata during execution. Dimension declarations define valid handoffs between steps, and `is_stateful` controls which steps are refit inside each fold versus reused as fold-invariant work.

## `xdflow.transforms.basic_transforms`

Common classes:

- `IdentityTransform`
- `AverageTransform`
- `FlattenTransform`
- `UnflattenTransform`
- `FunctionTransform`
- `CropTimeTransform`
- `SampleWeightTransform`
- `BalanceClassWeightTransform`

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

Install `xdflow[spectral]` or `xdflow[all]` before importing this module.

## `xdflow.transforms.sklearn_transform`

Common classes:

- `SKLearnTransform`
- `SKLearnTransformer`
- `SKLearnPredictor`

These adapters let you wrap scikit-learn estimators while preserving `xdflow`'s labeled data contract: sample dimensions, target coordinates, sample weights, and prediction metadata stay attached at the container boundary.

`SKLearnPredictor` supports:

- single-label classification with a `LabelEncoder`
- multi-target regression
- multilabel classification with `is_multilabel=True`
- optional `sample_weight` coordinates

## `xdflow.transforms.multi_output_wrapper`

Common classes:

- `MultiOutputRegressorFactory`
- `MultiOutputClassifierFactory`
- `MultiOutputEstimatorFactory` (backward-compatible alias for `MultiOutputRegressorFactory`)

## Advanced optional modules

The transform package also contains additional modules that may require optional extras or more specialized dependencies, including:

- `xdflow.transforms.lgbm_predictor`
- `xdflow.transforms.domain_adaptation`
- `xdflow.transforms.adapt_wrapper`
- `xdflow.transforms.zca`
- `xdflow.transforms.spatial`
- `xdflow.transforms.lda`
- `xdflow.transforms.pca`
