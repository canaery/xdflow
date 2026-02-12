"""Resolve target coordinate specs (str/list/pattern) and stack target arrays."""

import fnmatch
from collections.abc import Sequence

import numpy as np
import xarray as xr


def resolve_target_coords(target_coord: str | Sequence[str], data: xr.DataArray) -> list[str]:
    """
    Accept a single coord name, an explicit list/tuple of coord names, or a glob pattern (e.g., ``"*_target"``).
    Returns a validated list of coord names present in ``data``.

    Pattern matching is only activated when the string contains a ``*`` wildcard character.
    Explicit coordinate names are matched exactly.
    """
    if isinstance(target_coord, (list, tuple)):
        missing = [coord for coord in target_coord if coord not in data.coords]
        if missing:
            raise ValueError(
                f"Target coordinates not found in DataArray: {missing}. Available: {list(data.coords.keys())}"
            )
        return list(target_coord)

    if isinstance(target_coord, str):
        # Only treat as pattern if it contains a wildcard
        if "*" in target_coord:
            matches = [coord for coord in data.coords.keys() if fnmatch.fnmatch(coord, target_coord)]
            if not matches:
                raise ValueError(
                    f"No coordinates found matching pattern '{target_coord}'. Available: {list(data.coords.keys())}"
                )
            return sorted(matches)
        # Exact string match required (no implicit pattern expansion)
        if target_coord not in data.coords:
            raise ValueError(f"Required target coordinate '{target_coord}' not found in DataArray.")
        return [target_coord]

    raise ValueError(f"target_coord must be string or list, got {type(target_coord)}")


def extract_target_array(target_coord: str | Sequence[str], data: xr.DataArray, validate: bool = True) -> np.ndarray:
    """
    Resolve target coordinates and return a stacked numpy array.

    Args:
        target_coord: String pattern/name or iterable of coord names. If ``validate`` is False and
            ``target_coord`` is a list/tuple, it is assumed to be pre-resolved.
        data: DataArray containing target coords.
        validate: Whether to validate/resolve the target coordinates. Defaults to True.
    """
    if not validate and isinstance(target_coord, (list, tuple)):
        targets = list(target_coord)
    else:
        targets = resolve_target_coords(target_coord, data)

    return (
        data.coords[targets[0]].values
        if len(targets) == 1
        else np.column_stack([data.coords[t].values for t in targets])
    )
