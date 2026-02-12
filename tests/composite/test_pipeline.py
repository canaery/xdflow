"""Tests for Pipeline class."""

import numpy as np
import xarray as xr

from xdflow.composite import Pipeline, SwitchTransform
from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer
from xdflow.transforms.basic_transforms import IdentityTransform
from xdflow.transforms.pca import GlobalFeaturePCA


class DummyClassTransform(Transform):
    """Simple transform that emits a (trial, class) output."""

    is_stateful: bool = False
    input_dims = ("trial", "feature")
    output_dims = ("trial", "class")

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        n_trials = container.data.sizes["trial"]
        data = np.zeros((n_trials, 2))
        output = xr.DataArray(
            data,
            dims=("trial", "class"),
            coords={"trial": container.data.coords["trial"], "class": [0, 1]},
        )
        return DataContainer(output)


class TestPipelineCreation:
    """Test Pipeline creation."""

    def test_empty_pipeline(self):
        """Test creating an empty pipeline."""
        # Empty pipelines aren't supported - need at least one step
        # Test with a minimal identity transform instead
        pipeline = Pipeline(name="empty", steps=[("identity", IdentityTransform())])
        assert len(pipeline.steps) == 1

    def test_pipeline_with_transforms(self):
        """Test creating pipeline with transforms."""
        pipeline = Pipeline(
            name="test_pipeline", steps=[("pca", GlobalFeaturePCA(n_components=5)), ("identity", IdentityTransform())]
        )

        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].name == "pca"
        assert pipeline.steps[1].name == "identity"

    def test_pipeline_from_list_of_transforms(self):
        """Test creating pipeline from list of TransformSteps."""
        from xdflow.composite.base import TransformStep

        steps = [TransformStep("pca", GlobalFeaturePCA(n_components=5)), TransformStep("identity", IdentityTransform())]
        pipeline = Pipeline(name="test_pipeline", steps=steps)

        assert len(pipeline.steps) == 2


class TestPipelineExecution:
    """Test Pipeline execution."""

    def test_pipeline_fit_transform(self, timeseries_container):
        """Test pipeline fit_transform."""
        pipeline = Pipeline(name="pca_pipeline", steps=[("pca", GlobalFeaturePCA(n_components=5))])

        result = pipeline.fit_transform(timeseries_container)
        assert result.shape[1] == 5

    def test_pipeline_multi_step(self, timeseries_container):
        """Test multi-step pipeline with two transforms."""
        # Use identity transform followed by PCA to test multi-step execution
        # timeseries_container has 8 channels, so PCA can produce max 8 components
        pipeline = Pipeline(
            name="identity_pca", steps=[("identity", IdentityTransform()), ("pca", GlobalFeaturePCA(n_components=5))]
        )

        result = pipeline.fit_transform(timeseries_container)
        assert result.shape[1] == 5


class TestPipelineCloning:
    """Test Pipeline cloning."""

    def test_clone_pipeline(self):
        """Test cloning a pipeline."""
        pipeline = Pipeline(name="pca_pipeline", steps=[("pca", GlobalFeaturePCA(n_components=5))])

        cloned = pipeline.clone()
        assert cloned is not pipeline
        assert len(cloned.steps) == len(pipeline.steps)


class TestSwitchTransform:
    """Test SwitchTransform behavior."""

    def test_switch_transform_renames_output_dim(self, simple_container):
        """SwitchTransform should rename output dims when configured."""
        switch = SwitchTransform(
            choices=[("dummy", DummyClassTransform())],
            choose="dummy",
            from_dim="class",
            to_dim="label",
        )

        result = switch.transform(simple_container)

        assert "label" in result.data.dims
        assert "class" not in result.data.dims
        assert switch.get_expected_output_dims(("trial", "feature")) == ("trial", "label")
