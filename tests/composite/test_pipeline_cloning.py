"""Test that GroupApplyTransform can clone Pipeline templates."""

from xdflow.composite import GroupApplyTransform, Pipeline
from xdflow.core.base import Transform


class DummyTransform(Transform):
    def __init__(self, multiplier=1.0):
        super().__init__()
        self.multiplier = multiplier

    def _transform(self, container, **kwargs):
        return container

    def get_expected_output_dims(self, input_dims):
        return input_dims


def test_pipeline_cloning():
    pipeline = Pipeline(name="test_pipeline", steps=[("step1", DummyTransform(multiplier=2.0))])

    group_apply = GroupApplyTransform(group_coord="group", transform_template=pipeline)

    cloned = group_apply.transform_template.clone()
    assert cloned is not pipeline
    assert cloned.name == pipeline.name
