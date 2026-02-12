import time
import warnings
from typing import Any

import numpy as np
import xarray as xr
from scipy.ndimage import convolve, gaussian_filter

from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer

# TODO: ICA


def _create_laplacian_kernel(radius: float, weighted: bool = False, scaling: float = 1.0) -> np.ndarray:
    """
    Creates a square kernel for a Laplacian filter of a given radius.

    Args:
        radius: The radius of the kernel in grid units. Can be a float.
                The kernel size is determined by int(radius).
        weighted: If True, weights are calculated based on the inverse Euclidean
                  distance from the center. If False, all neighbors within the
                  radius are given equal weight.
        scaling: The scaling factor applied to neighbor weights in the Laplacian
                calculation.

    Returns:
        A 2D NumPy array representing the Laplacian kernel.
    """
    if not isinstance(radius, (int, float)) or radius < 1:
        raise ValueError("Radius must be a number greater than or equal to 1.")

    kernel_int_radius = int(radius)
    size = 2 * kernel_int_radius + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center_idx = kernel_int_radius

    if weighted:
        total_weight = 0
        for r in range(size):
            for c in range(size):
                if r == center_idx and c == center_idx:
                    continue
                dist = np.sqrt((r - center_idx) ** 2 + (c - center_idx) ** 2)
                if dist <= radius:
                    weight = 1.0 / dist
                    kernel[r, c] = weight
                    total_weight += weight
        if total_weight > 0:
            kernel /= total_weight
            kernel[kernel != 0] *= scaling
    else:  # Unweighted
        count = 0
        for r in range(size):
            for c in range(size):
                if r == center_idx and c == center_idx:
                    continue
                dist = np.sqrt((r - center_idx) ** 2 + (c - center_idx) ** 2)
                if dist <= radius:
                    kernel[r, c] = 1
                    count += 1
        if count > 0:
            kernel /= count
            kernel[kernel != 0] *= scaling

    kernel[center_idx, center_idx] = -1

    return kernel


def _get_grid_mapping(channel_ids: np.ndarray, electrode_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper to get channel-to-grid mappings.

    Args:
        channel_ids: A 1D array of channel identifiers.
        electrode_grid: The 2D array representing the electrode layout, where values correspond to channel IDs.

    Returns:
        A tuple of (target_rows, target_cols), where each is a 1D array
        containing the grid coordinates for each channel ID.
    """
    n_channels = len(channel_ids)
    target_rows = np.full(n_channels, -1, dtype=int)
    target_cols = np.full(n_channels, -1, dtype=int)

    for i, channel_id in enumerate(channel_ids):
        channel_positions = np.where(electrode_grid == channel_id)
        if channel_positions[0].size == 0:
            raise ValueError(f"Channel ID {channel_id} not found in the electrode grid.")
        target_rows[i] = channel_positions[0][0]
        target_cols[i] = channel_positions[1][0]

    return target_rows, target_cols


def _map_to_grid(data: xr.DataArray, electrode_grid: np.ndarray) -> xr.DataArray:
    """
    Maps LFP data from a channel list to a spatial grid.

    This function takes a DataArray with a 'channel' dimension and maps it to
    a 2D spatial grid, preserving all other dimensions. Only numeric channel
    names that can be mapped to the electrode grid are included; non-numeric
    channels (e.g., auxiliary reference sensors) are automatically excluded.

    Args:
        data: Input DataArray with a 'channel' dimension.
        electrode_grid: 2D numpy array defining the electrode layout.

    Returns:
        The gridded DataArray (with 'row' and 'col' dimensions). Non-numeric
        channels are excluded and must be handled by the caller if needed.
    """
    if "channel" not in data.dims:
        raise ValueError("Input DataArray must have a 'channel' dimension.")

    # Only keep numeric channels that can be mapped to the electrode grid
    channel_values = data.coords["channel"].values
    is_numeric = np.array([str(c).isdigit() for c in channel_values])
    numeric_channels = channel_values[is_numeric]

    if len(numeric_channels) == 0:
        raise ValueError("No numeric channels found for spatial grid mapping.")

    data = data.sel(channel=numeric_channels)

    other_dims = [d for d in data.dims if d != "channel"]
    grid_shape = electrode_grid.shape
    output_dims = (*other_dims, "row", "col")
    output_coords = {d: data.coords[d] for d in data.coords.keys() if d != "channel"}
    output_shape = (*[data.sizes[d] for d in other_dims], *grid_shape)

    gridded_data = xr.DataArray(np.full(output_shape, np.nan), dims=output_dims, coords=output_coords)

    target_rows, target_cols = _get_grid_mapping(data.coords["channel"].values.astype(int), electrode_grid)

    data_permuted = data.transpose(..., "channel")
    gridded_data.values[..., target_rows, target_cols] = data_permuted.values

    return gridded_data


def _fill_nan_mean_neighbors(data: xr.DataArray) -> xr.DataArray:
    """
    Fills NaNs with the mean of spatial 3x3 neighbors (ignoring NaNs).
    This version uses two compiled convolutions (SciPy) for performance.

    Args:
        data: xr.DataArray with spatial dimensions 'row' and 'col'.

    Returns:
        xr.DataArray: Copy of data with NaNs replaced by the mean of their
        spatial neighbours.
    """
    if "row" not in data.dims or "col" not in data.dims:
        raise ValueError("Input must have 'row' and 'col' dimensions.")

    nan_mask = np.isnan(data.values)
    if not np.any(nan_mask):
        return data.copy(deep=True)

    # --- 1. Create the convolution kernel ---
    # The kernel is designed to sum values in a 3x3 spatial window, excluding the center.
    # It is n-dimensional, with size 1 for all non-spatial dimensions.
    spatial_dims = ["row", "col"]
    kernel = np.ones([3 if dim in spatial_dims else 1 for dim in data.dims], dtype=np.float32)
    # Set the center of the kernel to 0 to exclude the cell itself from the sum.
    center_indices = tuple(1 if dim in spatial_dims else 0 for dim in data.dims)
    kernel[center_indices] = 0

    # --- 2. Calculate the sum of neighbors for each cell ---
    # Replace NaNs with 0 to ensure they don't poison the sum during convolution.
    work_zero = np.nan_to_num(data.values, nan=0.0)
    sum_nb = convolve(work_zero, kernel, mode="constant", cval=0.0)

    # --- 3. Count the number of valid (non-NaN) neighbors for each cell ---
    # Create a mask where valid numbers are 1 and NaNs are 0.
    valid_mask = (~nan_mask).astype(np.float32)
    # Convolving this mask with the same kernel counts the number of valid neighbors.
    cnt_nb = convolve(valid_mask, kernel, mode="constant", cval=0.0)

    # --- 4. Calculate the mean and fill the original NaNs ---
    # Suppress "invalid value" warnings for division by zero (where cnt_nb is 0).
    # The result of 0/0 is NaN, which is the desired outcome for cells with no valid neighbors.
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_nb = sum_nb / cnt_nb

    # Create a copy of the original data and fill only the NaN values.
    filled_values = data.values.copy()
    filled_values[nan_mask] = mean_nb[nan_mask]

    # Warn if any NaNs could not be filled (e.g., islands of NaNs).
    if np.any(np.isnan(filled_values)):
        warnings.warn("NaNs remain after fill attempt. Laplacian will contain NaNs.", RuntimeWarning, stacklevel=2)

    return xr.DataArray(filled_values, dims=data.dims, coords=data.coords, attrs=data.attrs)


def _map_from_grid(
    gridded_data: xr.DataArray,
    electrode_grid: np.ndarray,
    original_dims: tuple[str, ...],
    original_coords: dict[str, Any],
    original_sizes: dict[str, int],
    original_attrs: dict[str, Any],
) -> xr.DataArray:
    """
    Maps spatially transformed data back from a grid to a channel list.

    Reconstructs the original channel structure, filling in transformed data for
    numeric channels that were processed on the spatial grid. Non-numeric channels
    (e.g., reference sensors) are left as NaN in the output array and should be
    handled separately by the caller if needed.

    Args:
        gridded_data: DataArray in grid format (dims must include 'row' and 'col').
        electrode_grid: 2D numpy array defining the electrode layout.
        original_dims: The dimensions of the original DataArray.
        original_coords: The coordinates of the original DataArray.
        original_sizes: The sizes of the dimensions of the original DataArray.
        original_attrs: The attributes of the original DataArray.

    Returns:
        A DataArray with the transformed data in the original channel format.
        Numeric channels contain the spatially transformed data; non-numeric
        channels are filled with NaN.
    """
    output_shape = tuple(original_sizes[d] for d in original_dims)
    output_data = xr.DataArray(
        np.full(output_shape, np.nan, dtype=gridded_data.dtype),
        dims=original_dims,
        coords=original_coords,
        attrs=original_attrs,
    )

    # Only map LFP channels (exclude non-numeric channels)
    channel_values = original_coords["channel"].values
    is_numeric = np.array([str(c).isdigit() for c in channel_values])
    channel_ids = channel_values[is_numeric].astype(int)
    target_rows, target_cols = _get_grid_mapping(channel_ids, electrode_grid)

    gridded_values_for_channels = gridded_data.values[..., target_rows, target_cols]

    channel_axis_index = original_dims.index("channel")
    permuted_gridded_values = np.moveaxis(gridded_values_for_channels, -1, channel_axis_index)

    # Write back only numeric channels; leave others untouched
    # Build an indexer that selects numeric channels along the channel axis
    idx = [slice(None)] * output_data.values.ndim
    idx[channel_axis_index] = np.where(is_numeric)[0]
    output_data.values[tuple(idx)] = permuted_gridded_values

    return output_data


def _pyramid_to_dataarray(
    pyramid: list[np.ndarray],
    keep_only_top_pyramid: bool = True,
    original_dims: tuple[str, ...] = None,
    original_coords: dict[str, Any] = None,
    original_sizes: dict[str, int] = None,
    original_attrs: dict[str, Any] = None,
) -> xr.DataArray:
    """
    Converts a pyramid of spatial features into a single feature array.
    """
    levels_to_keep = [pyramid[-1]] if keep_only_top_pyramid else pyramid[1:]

    other_dims = [d for d in original_dims if d != "channel"]
    coords = {d: original_coords[d] for d in original_coords.keys() if d != "channel"}

    # Flatten spatial dimensions for each level and collect feature parts
    feature_parts = []
    for level in levels_to_keep:
        # Collapse leading non-spatial dimensions into one
        num_samples = np.prod(level.shape[:-2])
        feature_parts.append(level.reshape(num_samples, -1))

    # Stack features from all levels horizontally
    all_features_flat = np.hstack(feature_parts)

    # Reshape back to include the original non-spatial dimensions
    other_dims_sizes = [original_sizes[d] for d in other_dims]
    final_shape = (*other_dims_sizes, -1)  # -1 infers the number of features

    all_features = all_features_flat.reshape(final_shape)

    output_da = xr.DataArray(
        all_features,
        dims=(*other_dims, "feature"),
        coords=coords,
        attrs=original_attrs,
    )

    return output_da


# TODO: Implement this?
# class MapToGrid(Transform):
#     """
#     Maps LFP data from a channel list to a spatial grid.
#     """

#     is_stateful: bool = False

#     def __init__(self, sel: dict = None, drop_sel: dict = None):
#         super().__init__(sel=sel, drop_sel=drop_sel)


class LaplacianCSDTransform(Transform):
    """
    Computes the Current Source Density (CSD) using a Laplacian filter.

    This transform applies spatial filtering to neural electrode data arranged on a 2D grid.
    The CSD is computed as the negative spatial Laplacian, which estimates local current
    sources and sinks in neural tissue.

    **Channel Handling:**
    - Only processes numeric channel names (e.g., '0', '1', '127') that correspond to
      electrode positions in the spatial grid
    - Non-numeric channels (e.g., reference or auxiliary sensors) are automatically preserved
      unchanged in the output
    - Channel order in the output matches the input exactly

    **Spatial Processing:**
    - Maps numeric channels to a 2D electrode grid using predefined coordinates
    - Applies Laplacian kernel convolution for spatial filtering
    - Maps results back to the original channel structure

    Args:
        grid_layout: 2D numpy array defining the spatial electrode layout, where values
            correspond to channel IDs. Required for spatial mapping.
        radius: Spatial radius for the Laplacian kernel (in grid units).
        weighted: If True, weights kernel by inverse distance from center.
        scaling: Scaling factor applied to neighbor weights.
        handle_nans: If True, fills NaN values using spatial neighbors before processing.
        verbosity: Verbosity level for timing output (0=silent, 1=basic, 2=detailed).

    Example:
        >>> # Data with both LFP channels ('0'-'127') and a reference channel
        >>> data.coords['channel'] = ['0', '1', '2', ..., '127', 'ref']
        >>> grid = np.array([[0, 1, 2], [3, 4, 5]])  # example grid layout
        >>> transform = LaplacianCSDTransform(grid_layout=grid, radius=1.5)
        >>> result = transform.transform(data)
        >>> # result: LFP channels contain CSD values, reference channel unchanged
    """

    is_stateful: bool = False
    _supports_transform_sel: bool = True

    def __init__(
        self,
        grid_layout: np.ndarray,
        radius: float = 1.0,
        weighted: bool = False,
        scaling: float = 1.0,
        handle_nans: bool = True,
        verbosity: int = 0,
        sel: dict = None,
        drop_sel: dict = None,
        transform_sel: dict = None,
        transform_drop_sel: dict = None,
    ):
        super().__init__(sel=sel, drop_sel=drop_sel, transform_sel=transform_sel, transform_drop_sel=transform_drop_sel)
        self.grid_layout = grid_layout
        self.radius = radius
        self.weighted = weighted
        self.scaling = scaling
        self.handle_nans = handle_nans
        self.verbosity = verbosity

    def _transform(self, data_container: DataContainer, **kwargs) -> DataContainer:
        data = data_container.data

        start_total = time.perf_counter()
        if self.verbosity:
            print("[LaplacianCSD] Start transform")

        # Save original data info for later
        original_dims = data.dims
        original_coords = data.coords
        original_sizes = data.sizes
        original_attrs = data.attrs

        # Map data to electrode grid (non-numeric channels automatically excluded)
        t0 = time.perf_counter()
        gridded_data = _map_to_grid(data, self.grid_layout)
        if self.verbosity:
            dt = time.perf_counter() - t0
            msg = f"map_to_grid done in {dt * 1000:.1f} ms"
            if self.verbosity >= 2:
                msg += f" | grid shape: {tuple(gridded_data.sizes[d] for d in gridded_data.dims)}"
            print(f"[LaplacianCSD] {msg}")

        # Fill NaNs with the mean of spatial 3x3 neighbors
        if self.handle_nans and np.any(np.isnan(gridded_data.values)):
            t0 = time.perf_counter()
            gridded_data = _fill_nan_mean_neighbors(gridded_data)  # takes a while
            if self.verbosity:
                dt = time.perf_counter() - t0
                print(f"[LaplacianCSD] fill_nans done in {dt:.2f} s")

        # Create the spatial kernel
        t0 = time.perf_counter()
        spatial_kernel = _create_laplacian_kernel(radius=self.radius, weighted=self.weighted, scaling=self.scaling)
        if self.verbosity:
            dt = time.perf_counter() - t0
            if self.verbosity >= 2:
                print(f"[LaplacianCSD] create_kernel done in {dt * 1000:.1f} ms | kernel shape: {spatial_kernel.shape}")
            else:
                print(f"[LaplacianCSD] create_kernel done in {dt * 1000:.1f} ms")

        # Keep track of dimensions to convolve over
        input_dims = gridded_data.dims
        convolve_dims = ("row", "col")
        other_dims = [d for d in input_dims if d not in convolve_dims]
        t0 = time.perf_counter()
        data_to_convolve = gridded_data.transpose(*other_dims, *convolve_dims).values
        original_shape = data_to_convolve.shape
        reshaped_shape = (-1, *original_shape[-2:])
        data_reshaped = data_to_convolve.reshape(reshaped_shape)
        if self.verbosity:
            dt = time.perf_counter() - t0
            msg = f"prepare_convolution done in {dt * 1000:.1f} ms"
            if self.verbosity >= 2:
                msg += f" | original shape: {original_shape} -> reshaped: {reshaped_shape}"
            print(f"[LaplacianCSD] {msg}")

        # Perform spatial convolution with efficient reshaping
        t0 = time.perf_counter()
        # Expand 2D kernel to ND (singleton along non-spatial dims) and convolve in a single call
        kernel_nd_shape = (1,) * (data_reshaped.ndim - 2) + spatial_kernel.shape
        kernel_nd = spatial_kernel.reshape(kernel_nd_shape)
        laplacian_reshaped = convolve(data_reshaped, kernel_nd, mode="reflect")
        laplacian_grid_values = laplacian_reshaped.reshape((*original_shape[:-2], *original_shape[-2:]))
        if self.verbosity:
            dt = time.perf_counter() - t0
            print(f"[LaplacianCSD] convolution done in {dt:.2f} s")

        # CSD is proportional to the negative Laplacian
        t0 = time.perf_counter()
        laplacian_csd_grid = xr.DataArray(-laplacian_grid_values, dims=gridded_data.dims, coords=gridded_data.coords)
        if self.verbosity:
            dt = time.perf_counter() - t0
            print(f"[LaplacianCSD] assemble_csd done in {dt * 1000:.1f} ms")

        # Map back to original channel list
        t0 = time.perf_counter()
        output_data = _map_from_grid(
            laplacian_csd_grid,
            self.grid_layout,
            original_dims,
            original_coords,
            original_sizes,
            original_attrs,
        )

        # Preserve non-numeric channels unchanged from original data
        channel_values = data.coords["channel"].values
        is_numeric = np.array([str(c).isdigit() for c in channel_values])
        non_numeric_channels = channel_values[~is_numeric]

        if len(non_numeric_channels) > 0:
            for channel in non_numeric_channels:
                output_data.loc[{"channel": channel}] = data.sel(channel=channel)
        if self.verbosity:
            dt = time.perf_counter() - t0
            msg = f"map_from_grid done in {dt * 1000:.1f} ms"
            if self.verbosity >= 2:
                msg += f" | output shape: {tuple(output_data.sizes[d] for d in output_data.dims)}"
            print(f"[LaplacianCSD] {msg}")

        if self.verbosity:
            total_dt = time.perf_counter() - start_total
            print(f"[LaplacianCSD] total done in {total_dt:.2f} s")
        return DataContainer(output_data)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        if "channel" not in input_dims:
            raise ValueError("Input must have a 'channel' dimension.")

        return input_dims


class WindowMeanPyramidTransform(Transform):
    """
    Creates multiple levels of spatial averaging using a 2x2 sliding window.
    Requires a "channel" dimension, but otherwise dimension-agnostic.
    #TODO: do we want to support transform_sel even with different dimensions? so we can keep reference channels?

    Args:
        grid_layout: 2D numpy array defining the spatial electrode layout, where values
            correspond to channel IDs. Required for spatial mapping.
        levels: Number of pyramid levels to create.
        keep_only_top_pyramid: If True, only keeps the top level of the pyramid.
        handle_nans: If True, fills NaN values using spatial neighbors before processing.
    """

    is_stateful: bool = False

    def __init__(
        self,
        grid_layout: np.ndarray,
        levels: int = 1,
        keep_only_top_pyramid: bool = True,
        handle_nans: bool = True,
        sel: dict = None,
        drop_sel: dict = None,
    ):
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.grid_layout = grid_layout
        self.levels = levels
        self.keep_only_top_pyramid = keep_only_top_pyramid
        self.handle_nans = handle_nans

    def _transform(self, data_container: DataContainer, **kwargs) -> DataContainer:
        data = data_container.data

        # Save original data info for later
        original_dims = data.dims
        original_coords = data.coords
        original_sizes = data.sizes
        original_attrs = data.attrs

        gridded_data = _map_to_grid(data, self.grid_layout)

        # make sure last two dimensions are spatial
        assert gridded_data.dims[-2] == "row" and gridded_data.dims[-1] == "col", (
            f"Last two dimensions must be row and col, got dims {gridded_data.dims[-2:]} (gridded_data.dims: {gridded_data.dims})"
        )

        if self.handle_nans and np.any(np.isnan(gridded_data.values)):
            gridded_data = _fill_nan_mean_neighbors(gridded_data)

        pyramid = [gridded_data.values]
        current_stack = gridded_data.values

        for i in range(self.levels):
            # Check we have minimum 2x2 grid for the windowing operation
            if current_stack.shape[-2] < 2 or current_stack.shape[-1] < 2:
                if i == 0:
                    raise ValueError("Grid too small for window mean at this level, stopping pyramid.")
                warnings.warn(
                    f"Grid too small for window mean at this level, stopping pyramid at level {i}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            # Create the four shifted versions using ellipsis to handle N-dimensions
            original = current_stack[..., :-1, :-1]
            right = current_stack[..., :-1, 1:]
            down = current_stack[..., 1:, :-1]
            diagonal = current_stack[..., 1:, 1:]
            stacked = np.stack([original, right, down, diagonal], axis=0)

            # Suppress warnings for slices that are all NaN
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"Mean of empty slice")
                next_stack = np.nanmean(stacked, axis=0)
            pyramid.append(next_stack)
            current_stack = next_stack

        output_da = _pyramid_to_dataarray(
            pyramid,
            self.keep_only_top_pyramid,
            original_dims=original_dims,
            original_coords=original_coords,
            original_sizes=original_sizes,
            original_attrs=original_attrs,
        )
        return DataContainer(output_da)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        if "channel" not in input_dims:
            raise ValueError("Input must have a 'channel' dimension.")

        other_dims = [d for d in input_dims if d != "channel"]
        return (*other_dims, "feature")


class GaussianPyramidTransform(Transform):
    """
    Creates multiple levels of a Gaussian pyramid for spatial feature extraction.
    Requires a "channel" dimension, but otherwise dimension-agnostic.
    Levels = 1 means 1 level of gaussian filtering (unlike old implementation).
    #TODO: do we want to support transform_sel even with different dimensions? so we can keep reference channels?

    Args:
        grid_layout: 2D numpy array defining the spatial electrode layout, where values
            correspond to channel IDs. Required for spatial mapping.
        levels: Number of pyramid levels to create.
        sigma: Standard deviation for Gaussian kernel.
        keep_only_top_pyramid: If True, only keeps the top level of the pyramid.
        handle_nans: If True, fills NaN values using spatial neighbors before processing.
        gaussian_kwargs: Additional keyword arguments for gaussian_filter.
    """

    is_stateful: bool = False

    def __init__(
        self,
        grid_layout: np.ndarray,
        levels: int = 1,
        sigma: float = 1.0,
        keep_only_top_pyramid: bool = True,
        handle_nans: bool = True,
        gaussian_kwargs: dict | None = None,
        sel: dict = None,
        drop_sel: dict = None,
    ):
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.grid_layout = grid_layout
        self.levels = levels
        self.sigma = sigma
        self.keep_only_top_pyramid = keep_only_top_pyramid
        self.handle_nans = handle_nans
        self.gaussian_kwargs = gaussian_kwargs if gaussian_kwargs is not None else {}

    def _transform(self, data_container: DataContainer, **kwargs) -> DataContainer:
        data = data_container.data

        # Save original data info for later
        original_dims = data.dims
        original_coords = data.coords
        original_sizes = data.sizes
        original_attrs = data.attrs

        gridded_data = _map_to_grid(data, self.grid_layout)

        if self.handle_nans and np.any(np.isnan(gridded_data.values)):
            gridded_data = _fill_nan_mean_neighbors(gridded_data)

        pyramid = [gridded_data.values]
        current_stack = gridded_data.values

        for i in range(self.levels):
            # Check we have enough data
            if current_stack.shape[-2] < 1 or current_stack.shape[-1] < 1:
                if i == 0:
                    raise ValueError(f"Grid too small {current_stack.shape}, stopping pyramid.")
                warnings.warn(
                    f"Grid too small {current_stack.shape}, stopping pyramid at level {i}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            if self.handle_nans and np.any(np.isnan(current_stack)):
                current_stack = _fill_nan_mean_neighbors(current_stack)

            # TODO: skip outer edges for first level?

            # The gaussian_filter can take a sequence for sigma for each dimension.
            input_dims = gridded_data.dims
            spatial_dims_indices = [input_dims.index("row"), input_dims.index("col")]

            # create sigma for all dimensions, applying it only to spatial dims
            sigma_per_dim = [0.0] * len(input_dims)
            sigma_per_dim[spatial_dims_indices[0]] = self.sigma
            sigma_per_dim[spatial_dims_indices[1]] = self.sigma

            filtered_stack = gaussian_filter(current_stack, sigma=sigma_per_dim, **self.gaussian_kwargs)

            # downsample spatially by a factor of 2
            slicer = [slice(None)] * len(input_dims)
            slicer[spatial_dims_indices[0]] = slice(None, None, 2)
            slicer[spatial_dims_indices[1]] = slice(None, None, 2)

            next_stack = filtered_stack[tuple(slicer)]

            pyramid.append(next_stack)
            current_stack = next_stack

        output_da = _pyramid_to_dataarray(
            pyramid,
            self.keep_only_top_pyramid,
            original_dims=original_dims,
            original_coords=original_coords,
            original_sizes=original_sizes,
            original_attrs=original_attrs,
        )

        return DataContainer(output_da)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        if "channel" not in input_dims:
            raise ValueError("Input must have a 'channel' dimension.")

        other_dims = [d for d in input_dims if d != "channel"]
        return (*other_dims, "feature")
