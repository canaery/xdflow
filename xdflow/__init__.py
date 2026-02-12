"""
xdflow: Dimension-aware ML pipelines for scientific data

xdflow is a machine learning framework designed for structured, multidimensional
scientific data. Built on xarray, it brings reproducible, metadata-aware pipelines
to domains where sklearn falls short.
"""

__version__ = "0.1.0"

# Core abstractions
# Composite patterns
from xdflow.composite import (
    GroupApplyTransform,
    OptionalTransform,
    Pipeline,
    SwitchTransform,
    TransformUnion,
)
from xdflow.core.base import Predictor, Transform
from xdflow.core.data_container import DataContainer

# Cross-validation
from xdflow.cv.base import CrossValidator

# Hyperparameter tuning (optional dependency: optuna)
try:
    from xdflow.tuning.base import Tuner
except ImportError:
    Tuner = None

__all__ = [
    "__version__",
    # Core
    "Transform",
    "Predictor",
    "DataContainer",
    # Composite
    "Pipeline",
    "TransformUnion",
    "GroupApplyTransform",
    "SwitchTransform",
    "OptionalTransform",
    # Cross-validation
    "CrossValidator",
    # Tuning
    "Tuner",
]
