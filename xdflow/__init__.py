"""
xdflow: Dimension-aware ML pipelines for scientific data

xdflow is a machine learning framework designed for structured, multidimensional
scientific data. Built on xarray, it brings reproducible, metadata-aware pipelines
to domains where sklearn falls short.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xdflow")
except PackageNotFoundError:  # pragma: no cover - direct source-tree import fallback
    __version__ = "0+unknown"

from xdflow.composite import (
    EnsemblePredictor,
    GroupApplyTransform,
    OptionalTransform,
    Pipeline,
    SwitchTransform,
    TransformUnion,
    UnionWithInput,
)
from xdflow.core import DataContainer, Predictor, Transform, TransformError
from xdflow.cv import (
    CrossValidator,
    GroupedKFoldValidator,
    KFoldValidator,
    LeaveAnimalOutValidator,
    LeaveGroupOutValidator,
    LeaveSessionOutValidator,
    SampledDomainKFoldValidator,
)

# Hyperparameter tuning (optional dependency: optuna)
try:
    from xdflow.tuning import Tuner, run_tuning_pipeline
except ModuleNotFoundError as exc:
    if exc.name != "optuna":
        raise
    Tuner = None
    run_tuning_pipeline = None

__all__ = [
    "__version__",
    # Core
    "Transform",
    "Predictor",
    "DataContainer",
    "TransformError",
    # Composite
    "Pipeline",
    "TransformUnion",
    "UnionWithInput",
    "GroupApplyTransform",
    "SwitchTransform",
    "OptionalTransform",
    "EnsemblePredictor",
    # Cross-validation
    "CrossValidator",
    "KFoldValidator",
    "GroupedKFoldValidator",
    "SampledDomainKFoldValidator",
    "LeaveGroupOutValidator",
    "LeaveSessionOutValidator",
    "LeaveAnimalOutValidator",
    # Tuning
    "Tuner",
    "run_tuning_pipeline",
]
