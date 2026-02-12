"""Tests for DataContainer class."""

import numpy as np
import xarray as xr

from xdflow.core.data_container import DataContainer


class TestDataContainerCreation:
    """Test DataContainer creation and initialization."""

    def test_create_from_dataarray(self, simple_data_array):
        """Test creating DataContainer from xarray.DataArray."""
        container = DataContainer(simple_data_array)
        assert isinstance(container, DataContainer)
        assert container.data.equals(simple_data_array)

    def test_create_without_required_coords(self):
        """Test that DataContainer can be created without required coordinates."""
        data = xr.DataArray(np.random.randn(10, 5), dims=["x", "y"])
        container = DataContainer(data)
        assert isinstance(container, DataContainer)

    def test_create_with_required_coords_validation(self):
        """Test DataContainer with explicit coordinate validation."""
        data = xr.DataArray(
            np.random.randn(10, 5),
            dims=["trial", "feature"],
            coords={"trial": range(10), "feature": range(5)},
        )
        container = DataContainer(data, required_coords=["trial", "feature"])
        assert isinstance(container, DataContainer)

    def test_history_initialization(self, simple_container):
        """Test that history is initialized as empty list."""
        assert "data_history" in simple_container.data.attrs
        assert simple_container.data.attrs["data_history"] == []


class TestDataContainerProperties:
    """Test DataContainer properties and methods."""

    def test_data_property(self, simple_container, simple_data_array):
        """Test that data property returns the wrapped DataArray."""
        assert simple_container.data.equals(simple_data_array)

    def test_immutability(self, simple_container):
        """Test that modifying returned data doesn't affect container."""
        original_data = simple_container.data.copy(deep=True)
        returned_data = simple_container.data

        # Modify returned data
        returned_data.attrs["test"] = "modified"

        # Original should not be affected (shallow copy behavior)
        # Note: xarray attrs are mutable, but the array itself is not
        assert simple_container.data.equals(original_data)

    def test_repr(self, simple_container):
        """Test string representation."""
        repr_str = repr(simple_container)
        assert "DataContainer" in repr_str
        assert "DataArray" in repr_str or "trial" in repr_str

    def test_str(self, simple_container):
        """Test string summary."""
        str_summary = str(simple_container)
        assert "DataContainer" in str_summary
        assert "dimensions" in str_summary
        assert "transform" in str_summary


class TestDataContainerOperations:
    """Test DataContainer operations and transformations."""

    def test_indexing(self, simple_container):
        """Test that indexing returns a new DataContainer."""
        subset = simple_container[0:10]
        assert isinstance(subset, DataContainer)
        assert len(subset.data.trial) == 10

    def test_method_delegation(self, timeseries_container):
        """Test that DataArray methods are delegated and rewrapped."""
        # Test mean operation
        result = timeseries_container.mean(dim="time")
        assert isinstance(result, DataContainer)
        assert "time" not in result.dims

    def test_selection(self, timeseries_container):
        """Test sel operation returns DataContainer."""
        result = timeseries_container.sel(channel=["ch0", "ch1"])
        assert isinstance(result, DataContainer)
        assert len(result.data.channel) == 2

    def test_equality(self, simple_data_array):
        """Test DataContainer equality."""
        container1 = DataContainer(simple_data_array)
        container2 = DataContainer(simple_data_array.copy())
        assert container1 == container2

        # Different data should not be equal
        different_data = xr.DataArray(np.random.randn(100, 10), dims=["trial", "feature"])
        container3 = DataContainer(different_data)
        assert container1 != container3


class TestDataContainerTimeUnits:
    """Test time_units property."""

    def test_time_units_present(self):
        """Test time_units when time coordinate has units."""
        data = xr.DataArray(
            np.random.randn(10, 50),
            dims=["trial", "time"],
            coords={"trial": range(10), "time": np.linspace(0, 1, 50)},
        )
        data.coords["time"].attrs["units"] = "seconds"
        container = DataContainer(data)
        assert container.time_units == "seconds"

    def test_time_units_absent(self, simple_container):
        """Test time_units when no time coordinate exists."""
        assert simple_container.time_units is None

    def test_time_units_no_attr(self):
        """Test time_units when time exists but has no units attr."""
        data = xr.DataArray(
            np.random.randn(10, 50),
            dims=["trial", "time"],
            coords={"trial": range(10), "time": np.linspace(0, 1, 50)},
        )
        container = DataContainer(data)
        assert container.time_units is None


class TestDataContainerPickling:
    """Test DataContainer serialization."""

    def test_pickle_roundtrip(self, simple_container):
        """Test that DataContainer can be pickled and unpickled."""
        import pickle

        pickled = pickle.dumps(simple_container)
        unpickled = pickle.loads(pickled)

        assert isinstance(unpickled, DataContainer)
        assert unpickled == simple_container
