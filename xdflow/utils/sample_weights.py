from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def extract_sample_weights(
    data: xr.DataArray,
    sample_dim: str,
    coord_name: str | None,
    sample_index: pd.Index | xr.DataArray | None = None,
) -> np.ndarray | None:
    """Extract 1D sample weights aligned to a sample dimension."""
    if not coord_name:
        return None
    if coord_name not in data.coords:
        return None

    if sample_dim not in data.dims:
        raise ValueError(f"Sample dimension '{sample_dim}' not found in data dims {data.dims}.")

    weights = data.coords[coord_name]
    if sample_dim not in weights.dims:
        raise ValueError(f"Sample weight coordinate '{coord_name}' must include the sample dimension '{sample_dim}'.")

    if sample_index is None:
        sample_index = data.coords[sample_dim]

    try:
        aligned = weights.reindex({sample_dim: sample_index})
    except ValueError:
        aligned = weights.sel({sample_dim: sample_index})

    aligned = aligned.transpose(sample_dim, ...)
    if aligned.ndim != 1:
        raise ValueError(
            f"Sample weight coordinate '{coord_name}' must be 1D over '{sample_dim}', "
            f"but has dimensions {aligned.dims}."
        )

    weight_values = aligned.astype(float).values
    if np.isnan(weight_values).any():
        raise ValueError(f"Sample weight coordinate '{coord_name}' contains NaN values after alignment.")

    expected_len = len(sample_index)
    if weight_values.shape[0] != expected_len:
        raise ValueError(
            f"Sample weight coordinate '{coord_name}' length ({weight_values.shape[0]}) "
            f"does not match the number of samples ({expected_len})."
        )

    return weight_values
