"""
Core data structure for the pipeline framework.
"""

import functools
import warnings

import xarray as xr


class TransformError(Exception):
    """Error raised when a transform or pipeline step fails."""

    pass


class DataContainer:
    """Thin framework wrapper around an `xarray.DataArray`.

    XDFlow's data model is xarray. `DataContainer` is not a parallel array
    abstraction; it is the object passed between transforms, predictors, and
    cross-validation utilities so the framework has a consistent boundary. The
    wrapped `xarray.DataArray` remains the source of truth for values,
    dimensions, coordinates, and attrs.

    The wrapper initializes the `data_history` attribute used to track pipeline
    operations and rewraps common xarray operations so chained calls stay inside
    the XDFlow transform contract.

    Most xarray methods can be called directly on the container. Methods that
    return a new `xarray.DataArray` are rewrapped as a new `DataContainer`, so
    calls such as `container.sel(...)` or `container.mean(...)` remain inside
    the XDFlow container contract.

    The wrapped array is shallow-copied on construction. Transforms should still
    treat containers as immutable and return new containers instead of mutating
    their inputs.
    """

    def __init__(self, data: xr.DataArray, required_coords: list[str] | None = None):
        """Initialize a container from an xarray data array.

        Args:
            data: Array with labeled dimensions and coordinates.
            required_coords: Optional coordinate names to check for. Missing
                coordinates emit warnings rather than raising, which lets callers
                decide how strict to be for a given pipeline.

        Notes:
            The constructor ensures `data.attrs["data_history"]` exists. It does
            not validate dimension names or coordinate schemas beyond
            `required_coords`.
        """
        if required_coords is not None:
            for coord in required_coords:
                if coord not in data.coords:
                    warnings.warn(f"Missing required coordinate: '{coord}'")

        # Create a shallow copy to ensure immutability (deep copy not needed for xarray immutability)
        self._data = data.copy(deep=False)

        # Initialize History
        if "data_history" not in self._data.attrs:
            self._data.attrs["data_history"] = []

    def __getstate__(self):
        """Return the state to be pickled."""
        return self.__dict__

    def __setstate__(self, state):
        """Restore the state from the unpickled state."""
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return f"DataContainer(data=\n{self._data}\n)"

    def __str__(self) -> str:
        history_len = len(self._data.attrs.get("data_history", []))
        return (
            f"DataContainer with {len(self._data.dims)} dimensions, "
            f"shape {dict(zip(self._data.dims, self._data.shape, strict=True))}, "
            f"and {history_len} transform(s) in history"
        )

    def __eq__(self, other):
        if not isinstance(other, DataContainer):
            return NotImplemented
        return self._data.equals(other._data)

    def __getitem__(self, key):
        """
        Enable slice indexing on DataContainer.

        Args:
            key: Index, slice, or tuple of indices/slices

        Returns:
            DataContainer: New DataContainer with indexed data
        """
        result = self._data[key]
        if isinstance(result, xr.DataArray):
            return type(self)(result)
        return result

    @property
    def data(self) -> xr.DataArray:
        """Public accessor for the wrapped DataArray. Used in order to ensure immutability."""
        return self._data

    @property
    def time_units(self) -> str | None:
        """Return declared time units for the `time` coordinate if present.

        Returns:
            The value of `data.coords['time'].attrs['units']` if available, otherwise None.
        """
        try:
            return self._data.coords["time"].attrs.get("units")
        except Exception:
            return None

    def __getattr__(self, name: str):
        """
        Delegate attribute access to the underlying xarray.DataArray.

        If the attribute is a method that returns a new DataArray, it is
        wrapped to return a new DataContainer instance. This preserves the
        wrapper's validation and immutability for chained operations.
        """
        # Prevent recursion during pickle deserialization by checking if _data exists
        if not hasattr(self, "_data"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Retrieve the attribute from the wrapped _data object
        attr = getattr(self._data, name)

        # Check if the retrieved attribute is a callable method (e.g., .sel, .mean)
        if callable(attr):
            # Create a wrapper to intercept the method call
            @functools.wraps(attr)
            def wrapper(*args, **kwargs):
                # Execute the original xarray method
                result = attr(*args, **kwargs)

                # If the result is a new DataArray, re-wrap it in a new DataContainer
                if isinstance(result, xr.DataArray):
                    return type(self)(result)

                # Otherwise, return the result as-is (e.g., a NumPy scalar, a number)
                return result

            return wrapper
        else:
            # If the attribute is a property (e.g., .coords, .dims), return it directly
            return attr
