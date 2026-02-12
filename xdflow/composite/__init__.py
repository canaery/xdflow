"""Composite transforms for building complex pipelines."""

from xdflow.composite.base import CompositeTransform
from xdflow.composite.ensemble import EnsemblePredictor
from xdflow.composite.group_apply import GroupApplyTransform
from xdflow.composite.pipeline import Pipeline
from xdflow.composite.switch_transform import OptionalTransform, SwitchTransform
from xdflow.composite.transform_union import TransformUnion, UnionWithInput

__all__ = [
    "GroupApplyTransform",
    "Pipeline",
    "SwitchTransform",
    "TransformUnion",
    "UnionWithInput",
    "EnsemblePredictor",
    "OptionalTransform",
    "CompositeTransform",
]
