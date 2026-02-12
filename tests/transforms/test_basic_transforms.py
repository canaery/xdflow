"""Tests for basic transforms."""

import numpy as np

from xdflow.transforms.basic_transforms import (
    AverageTransform,
    CropTimeTransform,
    FlattenTransform,
    FunctionTransform,
    IdentityTransform,
    SampleWeightTransform,
    TransposeDimsTransform,
    TrialSampler,
    UnflattenTransform,
)


class TestAverageTransform:
    """Test AverageTransform functionality."""

    def test_average_time(self, data_container_factory):
        """Test averaging across time dimension."""
        container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
        transform = AverageTransform(dims="time")

        result = transform.transform(container)

        # Should have (trial, channel) dims
        assert "time" not in result.data.dims
        assert result.data.shape == (10, 8)

    def test_average_multiple_dims(self, data_container_factory):
        """Test averaging across multiple dimensions."""
        container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
        transform = AverageTransform(dims=["channel", "time"])

        result = transform.transform(container)

        # Should only have trial dim
        assert result.data.dims == ("trial",)
        assert result.data.shape == (10,)


class TestFlattenUnflattenTransform:
    """Test FlattenTransform and UnflattenTransform."""

    def test_flatten_two_dims(self, data_container_factory):
        """Test flattening two dimensions."""
        container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
        transform = FlattenTransform(dims=("channel", "time"))

        result = transform.transform(container)

        # Should have (trial, flat_channel__time)
        assert "channel" not in result.data.dims
        assert "time" not in result.data.dims
        assert "flat_channel__time" in result.data.dims
        assert result.data.shape == (10, 8 * 100)

    def test_flatten_unflatten_roundtrip(self, data_container_factory):
        """Test that flatten then unflatten recovers original structure."""
        container = data_container_factory(n_trials=10, n_channels=8, n_time=100)

        flatten = FlattenTransform(dims=("channel", "time"))
        flattened = flatten.transform(container)

        unflatten = UnflattenTransform(dim="flat_channel__time")
        unflattened = unflatten.transform(flattened)

        # Should recover original dimensions
        assert set(unflattened.data.dims) == set(container.data.dims)
        # Values should be preserved (might be reordered)
        np.testing.assert_array_almost_equal(
            np.sort(unflattened.data.values.flatten()), np.sort(container.data.values.flatten())
        )


class TestCropTimeTransform:
    """Test CropTimeTransform functionality."""

    def test_crop_time_window(self, data_container_factory):
        """Test cropping to a time window."""
        container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
        # Time coordinates are 0 to 1, so crop to middle half
        transform = CropTimeTransform(time_window_start_ms=250, time_window_end_ms=750)

        result = transform.transform(container)

        # Should have fewer time points
        assert result.data.sizes["time"] < container.data.sizes["time"]
        assert result.data.sizes["trial"] == 10
        assert result.data.sizes["channel"] == 8


class TestFunctionTransform:
    """Test FunctionTransform functionality."""

    def test_function_abs(self, data_container_factory):
        """Test applying absolute value function."""
        container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
        transform = FunctionTransform(func=np.abs, expected_output_dims=("trial", "channel", "time"))

        result = transform.transform(container)

        # All values should be non-negative
        assert np.all(result.data.values >= 0)
        assert result.data.shape == container.data.shape

    def test_function_square(self, data_container_factory):
        """Test applying square function."""
        container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
        transform = FunctionTransform(func=np.square, expected_output_dims=("trial", "channel", "time"))

        result = transform.transform(container)

        # All values should be non-negative
        assert np.all(result.data.values >= 0)
        # Squared values should be larger (in absolute terms) for values > 1
        assert result.data.shape == container.data.shape


class TestIdentityTransform:
    """Test IdentityTransform."""

    def test_identity_returns_same(self, data_container_factory):
        """Test that identity returns the same container."""
        container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
        transform = IdentityTransform()

        result = transform.transform(container)

        # Should return the exact same instance
        assert result is container


class TestTransposeDimsTransform:
    """Test TransposeDimsTransform."""

    def test_transpose_dims(self, data_container_factory):
        """Test transposing dimensions."""
        container = data_container_factory(n_trials=10, n_channels=8, n_time=100)
        # Original order: (trial, channel, time)
        transform = TransposeDimsTransform(dims=("time", "channel", "trial"))

        result = transform.transform(container)

        # Should have new dimension order
        assert result.data.dims == ("time", "channel", "trial")
        assert result.data.shape == (100, 8, 10)


class TestTrialSampler:
    """Test TrialSampler functionality."""

    def test_sample_trials(self, data_container_factory):
        """Test sampling a subset of trials."""
        container = data_container_factory(n_trials=100, n_channels=8, n_time=50)
        transform = TrialSampler(n_trials=10, shuffle=False)

        result = transform.transform(container)

        # Should have exactly 10 trials
        assert result.data.sizes["trial"] == 10
        assert result.data.sizes["channel"] == 8
        assert result.data.sizes["time"] == 50

    def test_sample_with_shuffle(self, data_container_factory):
        """Test sampling with shuffling."""
        container = data_container_factory(n_trials=100, n_channels=8, n_time=50)
        transform = TrialSampler(n_trials=20, shuffle=True)

        result1 = transform.transform(container)
        result2 = transform.transform(container)

        # Both should have 20 trials
        assert result1.data.sizes["trial"] == 20
        assert result2.data.sizes["trial"] == 20

        # With shuffling, the trial order might differ (though could be same by chance)
        # Just verify shapes are correct
        assert result1.data.shape == result2.data.shape

    def test_sample_more_than_available(self, data_container_factory):
        """Test that sampling more trials than available is handled."""
        container = data_container_factory(n_trials=10, n_channels=8, n_time=50)
        transform = TrialSampler(n_trials=20, shuffle=False)

        # Should handle gracefully (likely return all trials or raise error)
        # Check what the actual behavior is
        try:
            result = transform.transform(container)
            # If it succeeds, should have at most 10 trials
            assert result.data.sizes["trial"] <= 10
        except (ValueError, IndexError):
            # Or it might raise an error, which is also acceptable
            pass


class TestSampleWeightTransform:
    """Test SampleWeightTransform behavior."""

    def test_weight_map_numpy_scalars(self, data_container_factory):
        """Test weight lookup when coord values are numpy scalars."""
        container = data_container_factory(n_trials=5, n_channels=2, n_time=1)
        session_values = np.array([0, 1, 1, 2, 2], dtype=np.int64)
        container.data.coords["session"] = ("trial", session_values)

        transform = SampleWeightTransform(
            coord_name="session",
            weight_map={0: 1.0, 1: 2.0},
            default_weight=3.0,
        )

        result = transform.transform(container)
        weights = result.data.coords["sample_weight"].values

        assert np.allclose(weights, [1.0, 2.0, 2.0, 3.0, 3.0])

    def test_weight_map_set_params(self, data_container_factory):
        """Test nested weight_map updates via set_params."""
        container = data_container_factory(n_trials=3, n_channels=2, n_time=1)
        container.data.coords["session"] = ("trial", [0, 1, 2])

        transform = SampleWeightTransform(coord_name="session", weight_map={0: 1.0, 1: 2.0}, default_weight=0.5)
        transform.set_params(weight_map__2=4.0)

        result = transform.transform(container)
        weights = result.data.coords["sample_weight"].values

        assert np.allclose(weights, [1.0, 2.0, 4.0])
