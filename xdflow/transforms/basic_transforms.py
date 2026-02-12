from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from xdflow.core.base import Transform, _infer_dict_key_type

# Assuming the base class and DataContainer are in these locations
from xdflow.core.data_container import DataContainer


class TransposeDimsTransform(Transform):
    """Transpose dimensions of the data container."""

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(
        self, dims: tuple[str, ...], sel: dict[str, Any] | None = None, drop_sel: dict[str, Any] | None = None
    ):
        """
        Initializes the TransposeDimsTransform.

        Args:
            dims: The dimensions to transpose to.
            sel: Optional selection to apply before transforming.
            drop_sel: Optional drop selection to apply before transforming.
        """
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.dims = dims

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        return container.transpose(*self.dims)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(self.dims)


class RenameDimsTransform(Transform):
    """Rename xarray dimension names in the data container.

    This transform applies `xarray.DataArray.rename` to change dimension names
    (and matching coordinate names) according to a provided mapping. It preserves
    data and coordinate values.

    Example:
        Rename `feature` to `channel` after a union or feature step:
        transform = RenameDimsTransform(rename_map={"feature": "channel"})

    Args:
        rename_map: Mapping from old dimension names to new names.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(
        self, rename_map: dict[str, str], sel: dict[str, Any] | None = None, drop_sel: dict[str, Any] | None = None
    ):
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.rename_map = rename_map

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """Apply dimension renaming.

        Args:
            container: Input data container.

        Returns:
            A new `DataContainer` with renamed dimensions.
        """
        data = container.data
        renamed = data.rename(self.rename_map)
        return DataContainer(renamed)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """Infer output dims by applying the mapping to the input dims.

        Args:
            input_dims: Tuple of input dim names.

        Returns:
            Tuple of output dim names with mapping applied.
        """
        return tuple(self.rename_map.get(d, d) for d in input_dims)


class IdentityTransform(Transform):
    """A no-op transform that returns the input unchanged.

    This is useful as a selectable option in a `SwitchTransform` when you want
    the step to optionally do nothing while preserving dimension contracts.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(self, sel: dict[str, Any] | None = None, drop_sel: dict[str, Any] | None = None):
        """Initialize an identity transform.

        Args:
            sel: Optional selection to apply before transforming.
            drop_sel: Optional drop selection to apply before transforming.
        """
        super().__init__(sel=sel, drop_sel=drop_sel)

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """Return the input container unchanged.

        Args:
            container: Input container.

        Returns:
            The same `DataContainer` instance.
        """
        return container

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """Return the same dims as input.

        Args:
            input_dims: Input dimension names.

        Returns:
            The same dimension names.
        """
        return input_dims


class SampleWeightTransform(Transform):
    """Attach a `sample_weight` coordinate derived from an existing coordinate.

    The transform maps values from a source coordinate (e.g. ``session``) to
    scalar weights using either a ``weight_map`` or a callable ``weight_func``.
    The resulting weights are written as a new coordinate (default
    ``sample_weight``) on the same dimension(s) as the source, so downstream
    predictors can consume them without changing the data layout.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()
    _supports_transform_sel: bool = True

    def __init__(
        self,
        coord_name: str,
        weight_map: Mapping[Any, float] | None = None,
        weight_func: Callable[[Any], float] | None = None,
        default_weight: float = 1.0,
        target_coord: str = "sample_weight",
        dtype: np.dtype | type = np.float64,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
    ):
        """
        Args:
            coord_name: Name of the coordinate to read values from.
            weight_map: Mapping of coordinate values to weights.
            weight_func: Callable taking a coordinate value and returning a weight.
            default_weight: Weight to use when neither map nor func provide a value.
            target_coord: Name of the coordinate that will store computed weights.
            dtype: Numeric dtype used for the weight array.
        """
        if weight_map is not None and weight_func is not None:
            raise ValueError("Specify either 'weight_map' or 'weight_func', not both.")
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.coord_name = coord_name
        self.weight_map = {key: float(value) for key, value in weight_map.items()} if weight_map is not None else None
        self.weight_func = weight_func
        self.default_weight = float(default_weight)
        self.target_coord = target_coord
        self.dtype = dtype

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep=deep)
        if self.weight_map is not None:
            for key, value in self.weight_map.items():
                params[f"weight_map__{key}"] = value
        return params

    def set_params(self, **params: Any) -> SampleWeightTransform:
        params = dict(params)
        map_updates: dict[Any, float] = {}
        for key in list(params.keys()):
            if key.startswith("weight_map__"):
                if self.weight_func is not None:
                    raise ValueError("Cannot tune 'weight_map__*' parameters when 'weight_func' is set.")
                _, map_key = key.split("__", 1)
                map_updates[map_key] = params.pop(key)

        if "weight_map" in params:
            new_map = params.pop("weight_map")
            if new_map is None:
                self.weight_map = None
            else:
                self.weight_map = {key: float(value) for key, value in dict(new_map).items()}

        if map_updates:
            if self.weight_map is None:
                self.weight_map = {}
            else:
                self.weight_map = dict(self.weight_map)
            for map_key, map_value in map_updates.items():
                typed_key = _infer_dict_key_type(self.weight_map, map_key)
                self.weight_map[typed_key] = float(map_value)

        super().set_params(**params)
        return self

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        data = container.data
        if self.coord_name not in data.coords:
            raise ValueError(f"Coordinate '{self.coord_name}' not found in DataArray.")

        source_coord = data.coords[self.coord_name]
        coord_values = source_coord.values

        if self.weight_map is not None:
            # Convert numpy scalar types to Python native types for dict lookup
            # (numpy.int64 != int in dict.get(), causing all lookups to fail)
            def safe_lookup(value):
                # Convert numpy scalars to Python native types
                if hasattr(value, "item"):
                    value = value.item()
                return self.weight_map.get(value, self.default_weight)

            vectorized_lookup = np.vectorize(safe_lookup)
            weights = vectorized_lookup(coord_values)
        elif self.weight_func is not None:
            vectorized_func = np.vectorize(self.weight_func)
            weights = vectorized_func(coord_values)
        else:
            weights = np.full_like(coord_values, fill_value=self.default_weight, dtype=np.float64)

        try:
            weights_array = np.asarray(weights, dtype=self.dtype)
        except TypeError as exc:
            raise TypeError(f"Could not convert computed weights to dtype '{self.dtype}'.") from exc

        if weights_array.shape != source_coord.shape:
            raise ValueError(
                "Computed weights must match the shape of the source coordinate "
                f"(expected {source_coord.shape}, got {weights_array.shape})."
            )

        weight_coord = source_coord.copy(data=weights_array)
        weight_coord.name = self.target_coord
        updated = data.assign_coords({self.target_coord: weight_coord})

        return DataContainer(updated)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        return input_dims


class AverageTransform(Transform):
    """
    Averages data along one or more specified dimensions.

    This transform computes the mean of the data array along the given dimension(s),
    effectively removing them from the data's shape.
    """

    is_stateful: bool = False
    # No constraints on input or output dimensions, as they are dynamic.
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(self, dims: str | tuple[str, ...], sel: dict[str, Any] = None, drop_sel: dict[str, Any] = None):
        """
        Initializes the AverageTransform.

        Args:
            dims: The dimension or tuple of dimensions to average over.
        """
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.dims = (dims,) if isinstance(dims, str) else dims

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Applies the averaging transformation.

        Args:
            container: The DataContainer to transform.
            **kwargs: Additional context/parameters (ignored by this transform)

        Returns:
            A new DataContainer with data averaged over the specified dimensions.
        """
        data = container.data

        # Perform the averaging using xarray's mean function
        averaged_data = data.mean(dim=self.dims, keep_attrs=True)

        # Create, log history, and return the new container
        new_container = DataContainer(averaged_data)
        return new_container

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """
        Determines the expected output dimensions by removing the averaged dimensions.

        Args:
            input_dims: A tuple of dimension names of the input data.

        Returns:
            A tuple of dimension names for the output data.
        """
        return tuple(dim for dim in input_dims if dim not in self.dims)


class FlattenTransform(Transform):
    """
    Flattens (stacks) multiple dimensions into a single new dimension. 🥞

    The new dimension is named automatically based on the dimensions being
    flattened (e.g., 'flat_dim1__dim2').
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(self, dims: tuple[str, ...], sel: dict[str, Any] = None, drop_sel: dict[str, Any] = None):
        """
        Initializes the FlattenTransform.

        Args:
            dims: A tuple of dimension names to flatten into one.
        """
        super().__init__(sel=sel, drop_sel=drop_sel)
        if not isinstance(dims, tuple) or len(dims) < 2:
            raise ValueError("`dims` must be a tuple of at least two strings.")
        self.dims = dims
        self.new_dim_name = f"flat_{'__'.join(self.dims)}"

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Applies the flattening transformation using xarray's stack method.

        Args:
            container: The DataContainer to transform.
            **kwargs: Additional context/parameters (ignored by this transform)

        Returns:
            A new DataContainer with specified dimensions stacked into a single new one.
        """
        data = container.data

        # Validate that all requested dims exist in the data
        missing_dims = [dim for dim in self.dims if dim not in data.dims]
        if missing_dims:
            raise ValueError(
                f"FlattenTransform: requested dims {missing_dims} not found in data dims {list(data.dims)}"
            )

        # Stack the specified dimensions into a new dimension
        flattened_data = data.stack({self.new_dim_name: self.dims})

        # Create, log history, and return the new container
        new_container = DataContainer(flattened_data)
        return new_container

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """
        Determines the expected output dimensions after flattening. The new
        flattened dimension is appended at the end.

        Args:
            input_dims: A tuple of dimension names of the input data.

        Returns:
            A tuple of dimension names for the output data.
        """
        # Remove the original dimensions that are being flattened
        remaining_dims = [dim for dim in input_dims if dim not in self.dims]
        # Add the new flattened dimension to the end
        return tuple(remaining_dims) + (self.new_dim_name,)


class FunctionTransform(Transform):
    """
    Applies a function to the whole xarray.DataArray. The function must work on xarray.DataArray/numpy.
    Useful for applying simple mathemtical functions like np.abs, np.log, etc.
    For xarray functions with additional arguments, use partial functions, e.g.

        from functools import partial
        FunctionTransform(func=partial(xr.DataArray.max, dim="time"), expected_output_dims=("trial", "channel", "freq_band"))
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(
        self,
        func: Callable,
        expected_output_dims: tuple[str, ...] = None,  # needed for functions that change dimensionality (e.g. np.mean)
        sel: dict[str, Any] = None,
        drop_sel: dict[str, Any] = None,
    ):
        """
        Initializes the FunctionTransform.

        Args:
            func: The function to apply to the data. The function must work on xarray.DataArray/numpy.
            sel: A dictionary to select a subset of data for transformation.
            drop_sel: A dictionary to drop a subset of data for transformation.

            It is highly recommended to use a vectorized function from NumPy (e.g., `np.abs`) or from xarray
        """
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.func = func

        # Perform a check to ensure the function is compatible with xarray DataArrays at initialization.
        try:
            # Use a very small, simple DataArray for the check.
            test_data = xr.DataArray(np.array([1.0, 2.0]))
            self.func(test_data)
        except Exception as e:
            raise ValueError(
                f"The provided function '{getattr(self.func, '__name__', 'unknown')}' is not compatible with xarray.DataArray. "
                "Please provide a vectorized function (like those from NumPy). "
            ) from e

        self.expected_output_dims = expected_output_dims

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        data = container.data
        transformed_data = self.func(data)
        return DataContainer(transformed_data)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        if self.expected_output_dims is not None:
            warnings.warn(
                f"Expected output dimensions are manually specified for {self.func.__name__}. Might be inaccurate since not computed."
            )
            return self.expected_output_dims

        return input_dims


class UnflattenTransform(Transform):
    """
    Unflattens (unstacks) a dimension into multiple dimensions.
    The dimension must have been flattened before and must follow the naming output
    of FlattenTransform (e.g. 'flat_dim1__dim2').
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(self, dim: str, sel: dict[str, Any] = None, drop_sel: dict[str, Any] = None):
        """
        Initializes the UnflattenTransform.

        Args:
            dim: dimension to unflatten, must follow naming output of FlattenTransform (e.g. 'flat_dim1__dim2').
        """
        super().__init__(sel=sel, drop_sel=drop_sel)
        if not dim.startswith("flat_"):
            raise ValueError("`dim` must have been flattened before, and must start with 'flat_'.")
        self.dim = dim
        self.new_dim_names = dim.replace("flat_", "").split("__")

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Applies the unflattening transformation using xarray's unstack method.

        Args:
            container: The DataContainer to transform.
            **kwargs: Additional context/parameters (ignored by this transform)

        Returns:
            A new DataContainer with specified dimensions stacked into a single new one.
        """
        data = container.data

        # Validate that requested dim exists in the data
        if self.dim not in data.dims:
            raise ValueError(f"UnflattenTransform: requested dim {self.dim} not found in data dims {list(data.dims)}")

        # Check if the coordinate for the dimension is a MultiIndex
        if not isinstance(data.coords[self.dim].to_index(), pd.MultiIndex):
            # If not, try to create one assuming the coordinate values are tuples
            try:
                multi_index = pd.MultiIndex.from_tuples(data.coords[self.dim].values, names=self.new_dim_names)
                data = data.assign_coords({self.dim: multi_index})
            except Exception as e:
                # If that fails, re-raise with a more informative error
                raise ValueError(
                    f"Failed to unstack '{self.dim}'. It is not a MultiIndex, "
                    f"and could not be converted from coordinate values. Original error: {e}"
                ) from e

        # Unstack the specified dimension into multiple dimensions
        unflattened_data = data.unstack(self.dim)

        # When unstacking, coordinates that should only depend on 'trial' can get
        # broadcast across the newly created dimensions. This block reverts them
        # so they only depend on 'trial'.
        if "trial" in self.new_dim_names:
            # Find the other dimensions that came from unstacking
            other_dims = [dim for dim in self.new_dim_names if dim != "trial"]

            coords_to_update = {}
            for coord_name, coord_data in unflattened_data.coords.items():
                # We are looking for coordinates that are not the dimensions themselves
                if coord_name in self.new_dim_names:
                    continue

                # Check if the coordinate depends on 'trial' and at least one of the other new dims
                if "trial" in coord_data.dims and any(d in coord_data.dims for d in other_dims):
                    # To drop the other dims, we select the first element along them.
                    # Since the values are broadcast, any index is fine.
                    selector = {dim: 0 for dim in other_dims if dim in coord_data.dims}
                    if selector:
                        # Drop the other dimensions
                        new_coord_data = coord_data.isel(**selector, drop=True)
                        coords_to_update[coord_name] = new_coord_data

            if coords_to_update:
                unflattened_data = unflattened_data.assign_coords(coords_to_update)

        # Create, log history, and return the new container
        new_container = DataContainer(unflattened_data)
        return new_container

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """
        Determines the expected output dimensions after unflattening. The new
        unflattened dimensions are appended at the end.

        Args:
            input_dims: A tuple of dimension names of the input data.

        Returns:
            A tuple of dimension names for the output data.
        """
        # Remove the original dimensions that are being unflattened
        remaining_dims = [dim for dim in input_dims if dim not in self.dim]
        # Add the new unflattened dimensions to the end
        return tuple(remaining_dims) + tuple(self.new_dim_names)


class TrialSampler(Transform):
    """
    Samples trials from the data.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(
        self, n_trials: int, shuffle: bool = True, sel: dict[str, Any] = None, drop_sel: dict[str, Any] = None
    ):
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.n_trials = n_trials
        self.shuffle = shuffle

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Samples trials from the data.

        Args:
            container: The DataContainer to transform.
            **kwargs: Additional context/parameters (ignored by this transform)
        """
        data = container.data

        num_trials = data.sizes.get("trial", 0)
        if self.n_trials > num_trials:
            raise ValueError(f"Cannot sample {self.n_trials} trials when only {num_trials} are available.")

        trial_indices = np.arange(num_trials)
        if self.shuffle:
            np.random.shuffle(trial_indices)

        sampled_indices = trial_indices[: self.n_trials]
        sampled_data = data.isel(trial=sampled_indices)

        new_container = DataContainer(sampled_data)
        return new_container

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """
        Determines the expected output dimensions after sampling.
        """
        return input_dims


class CropTimeTransform(Transform):
    """
    Crop data to a time window using the time coordinate, inclusive.

    This transform selects a subset of the data along a time-like coordinate using
    label-based slicing, preserving all other dimensions and coordinates.

    The start and end are inclusive and interpreted in the same units as the
    DataArray's time coordinate values.

    Args:
        time_window_start_ms: Start of the time window (inclusive).
        time_window_end_ms: End of the time window (inclusive).
        time_coord: Name of the time coordinate/dimension to slice over. Defaults to "time".
        sel: Optional selection to apply before transforming.
        drop_sel: Optional drop selection to apply before transforming.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(
        self,
        time_window_start_ms: float,
        time_window_end_ms: float,
        time_coord: str = "time",
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
    ):
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.time_window_start_ms = time_window_start_ms
        self.time_window_end_ms = time_window_end_ms
        self.time_coord = time_coord

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """Apply inclusive time-window cropping using label-based selection.

        Args:
            container: Input data container.

        Returns:
            A new `DataContainer` cropped to the specified time window.
        """
        data = container.data

        if self.time_coord not in data.coords:
            raise ValueError(f"Time coordinate '{self.time_coord}' not found in DataArray coords: {list(data.coords)}")

        # Inclusive label-based slice on the time coordinate
        sliced = container.sel(**{self.time_coord: slice(self.time_window_start_ms, self.time_window_end_ms)})
        return sliced

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """Cropping preserves dimension names; sizes may shrink.

        Args:
            input_dims: Input dimension names.

        Returns:
            The same dimension names.
        """
        return input_dims
