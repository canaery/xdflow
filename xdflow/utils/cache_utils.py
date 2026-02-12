"""
Cache utilities for preprocessed and featurized data.

This module provides functions for caching and retrieving preprocessed and featurized data
based on input parameters. It uses a hash of the input parameters to create unique cache keys.
"""

import hashlib
import inspect
import json
import os
import pickle
import shutil
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xdflow.core.data_container import DataContainer

DEFAULT_CACHE_DIR = Path(os.environ.get("XDFLOW_CACHE_DIR", Path.home() / ".cache" / "xdflow"))

# Default maximum cache size in GB
DEFAULT_MAX_CACHE_SIZE_GB = 50

# Default maximum cache size in bytes
DEFAULT_MAX_CACHE_SIZE = DEFAULT_MAX_CACHE_SIZE_GB * 1024 * 1024 * 1024

# Default maximum age of cache files in days
DEFAULT_MAX_CACHE_AGE = 7


def _get_module_hash_from_obj(obj: Any) -> str:
    """Get hash of the module containing the object."""
    if inspect.isclass(obj):
        module = inspect.getmodule(obj)
    else:
        module = inspect.getmodule(obj.__class__)

    if module is None:
        return None

    source_file = inspect.getsourcefile(module)
    if source_file is None:
        return None

    with open(source_file, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _get_all_children(obj: Any) -> list[Any]:
    """Recursively get all children of a composite object."""
    children = []
    if hasattr(obj, "children") and obj.children:
        for child in obj.children:
            children.append(child)
            children.extend(_get_all_children(child))
    return children


def get_module_hashes_for_object(instance: Any) -> dict[str, str]:
    """Return module path -> hash for an object and all its children.

    Args:
        instance: Composite object (e.g., Pipeline) whose module hashes to collect

    Returns:
        Dict mapping module file paths to md5 hashes of their contents
    """
    all_objects = [instance] + _get_all_children(instance)
    module_hashes: dict[str, str] = {}
    for obj in all_objects:
        try:
            module_path = inspect.getsourcefile(inspect.getmodule(obj.__class__))
            if module_path and module_path not in module_hashes:
                module_hashes[module_path] = _get_module_hash_from_obj(obj)
        except TypeError:
            # Happens for some built-in types
            continue
    return module_hashes


def _get_cache_path_for_key(prefix: str, key_dict: dict[str, Any]) -> Path:
    """Compute cache file path for a given key dict."""
    cache_key = hash_dict(key_dict)
    return _get_cache_dir(prefix) / f"{cache_key}.pkl"


def try_load_cached_object(prefix: str, key_dict: dict[str, Any]) -> Any | None:
    """Load a cached Python object by key if present, else return None.

    This uses the same directory structure and hashing as the function result cache.
    """
    # Clean and enforce limits before reading
    _cleanup_old_cache_files(DEFAULT_MAX_CACHE_AGE, prefix)
    _enforce_cache_size_limit(DEFAULT_MAX_CACHE_SIZE, prefix)

    cache_path = _get_cache_path_for_key(prefix, key_dict)
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, FileNotFoundError):
            return None
    return None


def save_cached_object(prefix: str, key_dict: dict[str, Any], obj: Any) -> None:
    """Persist a Python object in the cache under the key derived from key_dict."""
    cache_path = _get_cache_path_for_key(prefix, key_dict)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(obj, f)
    except PermissionError:
        # Skip caching if filesystem forbids writing (e.g., immutable files/attrs)
        return


def _safe_scalar(value: Any) -> Any:
    """Convert a value to a JSON-serializable, deterministic scalar.

    - Converts numpy scalars to Python scalars
    - Replaces NaN/Inf with None
    """
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
    return value


def get_pipeline_cache_key_dict(func: Callable, instance: Any, *args, **kwargs) -> dict[str, Any]:
    """
    Generate a cache key dictionary for a pipeline.

    This function creates a detailed dictionary that includes:
    - The function's arguments.
    - The configuration of the pipeline instance and all its nested transforms.
    - The code hashes of the modules of the pipeline and all its transforms.

    Args:
        func: The function being called (e.g., fit_transform).
        instance: The pipeline instance.
        *args: Positional arguments to the function.
        **kwargs: Keyword arguments to the function.

    Returns:
        A dictionary to be hashed for the cache key.
    """
    # 1. Get function arguments
    bound_args = inspect.signature(func).bind(instance, *args, **kwargs)
    bound_args.apply_defaults()
    key_dict = {k: _get_object_metadata(v) for k, v in bound_args.arguments.items() if k != "self"}

    # 2. Get instance configuration (recursively)
    key_dict["config"] = _get_object_params(instance)

    # 3. Get module hashes for the instance and all its children
    all_objects = [instance] + _get_all_children(instance)
    module_hashes = {}
    for obj in all_objects:
        try:
            module_path = inspect.getsourcefile(inspect.getmodule(obj.__class__))
            if module_path and module_path not in module_hashes:
                module_hashes[module_path] = _get_module_hash_from_obj(obj)
        except TypeError:
            pass  # Happens for some built-in types

    key_dict["module_hashes"] = module_hashes

    return key_dict


def cache_result(
    prefix: str,
    max_size: int = DEFAULT_MAX_CACHE_SIZE,
    max_age_days: float = DEFAULT_MAX_CACHE_AGE,
    key_gen_func: Callable | None = None,
) -> Callable:
    """
    Decorator that caches function results based on all function and class instance parameters.

    Args:
        prefix: Prefix for the cache directory (e.g., 'preprocess', 'featurize')
        max_size: Maximum cache size in bytes for this prefix
        max_age_days: Maximum age of cache files in days
        key_gen_func: Optional function to generate the cache key dictionary.
                      If None, a default key generation logic is used.

    Returns:
        Callable: Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the instance if this is an instance method
            instance = args[0] if args and hasattr(args[0], "__dict__") else None

            if instance and not getattr(instance, "use_cache", False):
                return func(*args, **kwargs)

            # Clean up cache files
            _cleanup_old_cache_files(max_age_days, prefix)
            _enforce_cache_size_limit(max_size, prefix)

            print("Checking cache")

            if key_gen_func and instance:
                key_dict = key_gen_func(func, *args, **kwargs)
            else:
                # Get arguments of function
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                bound_args.apply_defaults()
                key_dict = {
                    k: _get_object_metadata(v) for k, v in bound_args.arguments.items() if k != "self" or not instance
                }
                if instance:
                    # Make sure the class instance is the same
                    key_dict["config"] = _get_object_params(instance)
                    key_dict["module_hash"] = _get_module_hash_from_obj(instance)

            cache_key = hash_dict(key_dict)
            cache_path = _get_cache_dir(prefix) / f"{cache_key}.pkl"

            print(f"Cache path: {cache_path}", "Cache exists:", cache_path.exists())

            # Return cached result if it exists
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        # print(f"Loading cached result from {cache_path}")
                        return pickle.load(f)
                except (pickle.UnpicklingError, EOFError, FileNotFoundError):
                    # If cache is corrupted or gets deleted mid-read, recompute
                    pass

            # Compute result and cache it
            result = func(*args, **kwargs)
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
            except PermissionError:
                # Skip caching if filesystem forbids writing (e.g., immutable files/attrs)
                pass

            return result

        return wrapper

    return decorator


def _get_ndarray_metadata(arr: np.ndarray) -> dict[str, Any]:
    """Build a lightweight, deterministic signature for a numpy array."""
    meta: dict[str, Any] = {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
    }
    flat = arr.ravel()
    size = int(flat.size)
    sample_n = min(32, size)
    if sample_n > 0:
        idx = np.unique(np.linspace(0, size - 1, num=sample_n, dtype=int))
        sample_vals = flat[idx]
        meta["sample"] = tuple(_safe_scalar(v) for v in sample_vals)
    return meta


def _get_datacontainer_metadata(dc: DataContainer) -> dict[str, Any]:
    """Build a deterministic signature for a DataContainer, including trial signature and small data sample."""
    meta: dict[str, Any] = {
        "shape": tuple(dc.shape),
        "coords": tuple(dc.coords.keys()),
        "dims": tuple(dc.dims),
    }

    # Include coordinate value fingerprints to detect label changes
    coord_metadata: dict[str, Any] = {}
    for coord_name in dc.coords.keys():
        coord_values = np.asarray(dc.data.coords[coord_name].values)
        coord_info: dict[str, Any] = {
            "shape": tuple(coord_values.shape),
            "dtype": str(getattr(coord_values, "dtype", type(coord_values).__name__)),
        }
        flat = coord_values.ravel()
        size = int(flat.size)
        sample_n = min(32, size)
        if sample_n > 0:
            idx = np.unique(np.linspace(0, size - 1, num=sample_n, dtype=int))
            sample_vals = flat[idx]
            coord_info["sample"] = tuple(_safe_scalar(v) for v in sample_vals)
        coord_metadata[coord_name] = coord_info
    meta["coord_metadata"] = coord_metadata

    # Trial signature to uniquely identify splits
    if "trial" in dc.data.coords:
        trial_values = np.asarray(dc.data.trial.values)
        num_trials = int(trial_values.shape[0])

        head = trial_values[:8].tolist() if num_trials > 0 else []
        mid_start = max(0, num_trials // 2)
        mid = trial_values[mid_start : mid_start + 8].tolist() if num_trials > 0 else []
        tail = trial_values[-8:].tolist() if num_trials > 0 else []

        if num_trials > 0:
            sample_idx = np.unique(np.linspace(0, num_trials - 1, num=min(32, num_trials), dtype=int))
            sample = trial_values[sample_idx].tolist()
        else:
            sample = []

        def _to_native_list(values):
            return [v.item() if hasattr(v, "item") else v for v in values]

        meta["trial_signature"] = {
            "len": num_trials,
            "head": tuple(_to_native_list(head)),
            "mid": tuple(_to_native_list(mid)),
            "tail": tuple(_to_native_list(tail)),
            "sample": tuple(_to_native_list(sample)),
        }

    # Small content fingerprint of underlying data values
    data_vals = dc.data.values
    meta["data_dtype"] = str(getattr(data_vals, "dtype", "unknown"))
    flat = np.asarray(data_vals).ravel()
    size = int(flat.size)
    sample_n = min(32, size)
    if sample_n > 0:
        idx = np.unique(np.linspace(0, size - 1, num=sample_n, dtype=int))
        sample_vals = flat[idx]
        meta["data_sample"] = tuple(_safe_scalar(v) for v in sample_vals)

    return meta


def _get_dataframe_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Build a deterministic signature for a pandas DataFrame."""
    meta_df: dict[str, Any] = {
        "shape": tuple(df.shape),
        "columns": tuple(str(c) for c in df.columns),
        "dtypes": tuple(str(t) for t in df.dtypes.values),
    }
    meta_df["null_counts"] = tuple(int(n) for n in df.isna().sum().values)
    n_rows = int(df.shape[0])
    if n_rows > 0:
        row_idx = np.unique(np.linspace(0, n_rows - 1, num=min(16, n_rows), dtype=int))
        sampled = df.iloc[row_idx]
        sample_repr = []
        for _, row in sampled.iterrows():
            sample_repr.append(tuple(_safe_scalar(v) for v in row.values))
        meta_df["row_sample"] = tuple(sample_repr)
    return meta_df


def _get_series_metadata(s: pd.Series) -> dict[str, Any]:
    """Build a deterministic signature for a pandas Series."""
    meta_s: dict[str, Any] = {
        "length": int(s.shape[0]),
        "name": str(s.name),
        "dtype": str(s.dtype),
    }
    meta_s["null_count"] = int(s.isna().sum())
    n = int(s.shape[0])
    if n > 0:
        idx = np.unique(np.linspace(0, n - 1, num=min(32, n), dtype=int))
        meta_s["sample"] = tuple(_safe_scalar(v) for v in s.iloc[idx].values)
    return meta_s


def _get_dict_metadata(d: dict[Any, Any]) -> dict[str, Any]:
    """Build a deterministic signature for a dictionary."""
    keys_sorted = sorted(d.keys(), key=lambda x: str(x))
    meta_dict: dict[str, Any] = {
        "length": len(d),
        "keys": tuple(str(k) for k in keys_sorted),
    }
    n = len(keys_sorted)
    if n > 0:
        idx = np.unique(np.linspace(0, n - 1, num=min(32, n), dtype=int))
        sample_items = [(keys_sorted[i], d[keys_sorted[i]]) for i in idx]
        meta_dict["sample_items"] = tuple((str(k), _get_object_metadata(v)) for k, v in sample_items)
    return meta_dict


def _get_sequence_metadata(seq: Any) -> dict[str, Any]:
    """Build a deterministic signature for a list/tuple."""
    n = len(seq)
    meta_seq: dict[str, Any] = {
        "type": type(seq).__name__,
        "length": n,
    }
    if n > 0:
        idx = np.unique(np.linspace(0, n - 1, num=min(32, n), dtype=int))
        meta_seq["sample"] = tuple(_get_object_metadata(seq[i]) for i in idx)
    return meta_seq


def _get_set_metadata(s: set) -> dict[str, Any]:
    """Build a deterministic signature for a set."""
    sorted_vals = sorted(s, key=lambda x: str(x))
    n = len(sorted_vals)
    meta_set: dict[str, Any] = {
        "type": "set",
        "length": n,
    }
    if n > 0:
        idx = np.unique(np.linspace(0, n - 1, num=min(32, n), dtype=int))
        meta_set["sample"] = tuple(_get_object_metadata(sorted_vals[i]) for i in idx)
    return meta_set


def _get_callable_metadata(fn: Callable) -> dict[str, Any]:
    """Build a deterministic signature for a callable."""
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", str(fn)))
    return {"callable": str(qualname), "module_hash": _get_module_hash_from_obj(fn)}


def _get_object_with_dict_metadata(obj: Any) -> dict[str, Any]:
    """Build a deterministic signature for a generic object with __dict__."""
    return {
        "__class__": obj.__class__.__name__,
        "config": _get_object_params(obj),
        "module_hash": _get_module_hash_from_obj(obj),
    }


def _get_object_metadata(obj: Any) -> Any:
    """
    Get metadata for an object to use in cache key creation.
    Handles various data types including numpy arrays, pandas DataFrames, and nested structures.

    Args:
        obj: Object to get metadata from

    Returns:
        Metadata representation of the object suitable for hashing
    """
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, np.ndarray):
        return _get_ndarray_metadata(obj)
    if isinstance(obj, DataContainer):
        return _get_datacontainer_metadata(obj)
    if isinstance(obj, pd.DataFrame):
        return _get_dataframe_metadata(obj)
    if isinstance(obj, pd.Series):
        return _get_series_metadata(obj)
    if isinstance(obj, dict):
        return _get_dict_metadata(obj)
    if isinstance(obj, (list, tuple)):
        return _get_sequence_metadata(obj)
    if isinstance(obj, set):
        return _get_set_metadata(obj)
    if callable(obj):
        return _get_callable_metadata(obj)
    if hasattr(obj, "__dict__"):
        return _get_object_with_dict_metadata(obj)
    return {"__class__": obj.__class__.__name__, "repr": str(obj)[:128]}


def _get_object_params(obj: Any) -> Any:
    """
    Get a dictionary of an object's parameters, handling nested objects.

    Args:
        obj: Object to get parameters from

    Returns:
        dict: Dictionary of parameter names and values
    """

    params_to_exclude = ["scent_encoder", "transform_from_name", "fitted_"]  # TODO: make this more robust/flexible?

    if not hasattr(obj, "__dict__"):
        if isinstance(obj, (list, tuple)):
            return [_get_object_params(item) for item in obj]
        return str(obj)

    params = {}
    # Get class name for type identification
    params["__class__"] = obj.__class__.__name__

    # Get all attributes that don't start with '_'
    for key, value in obj.__dict__.items():
        if key in params_to_exclude:
            continue
        if not key.startswith("_"):  # Skip private attributes
            if hasattr(value, "__dict__"):
                # Recursively get parameters of nested objects
                params[key] = _get_object_params(value)
            elif isinstance(value, (list, tuple)):
                params[key] = [_get_object_params(item) for item in value]
            else:
                params[key] = str(value)

    return params


def hash_dict(d: dict[str, Any]) -> str:
    """
    Create a deterministic hash of a dictionary using MD5.
    Uses MD5 like tuner_utils for consistency, with robust handling of pandas/numpy objects.

    Args:
        d: Dictionary to hash

    Returns:
        str: MD5 hash of the dictionary
    """
    serialized = json.dumps(_get_object_metadata(d), sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def _get_cache_root() -> Path:
    """Get the root directory for caches, allowing an environment override."""
    override = os.environ.get("XDFLOW_CACHE_DIR")
    return Path(override) if override else DEFAULT_CACHE_DIR


def _get_cache_dir(prefix: str) -> Path:
    """Get the cache directory for a given prefix."""
    cache_dir = _get_cache_root() / prefix
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cache_size(prefix: str | None = None) -> int:
    """
    Get the total size of the cache in bytes.

    Args:
        prefix: Optional prefix to check specific cache directory

    Returns:
        int: Total size in bytes
    """
    cache_root = _get_cache_root()
    if not cache_root.exists():
        return 0

    total_size = 0
    if prefix:
        cache_dir = cache_root / prefix
        if cache_dir.exists():
            total_size = sum(f.stat().st_size for f in cache_dir.glob("*.pkl"))
    else:
        for d in cache_root.iterdir():
            if d.is_dir():
                total_size += sum(f.stat().st_size for f in d.glob("*.pkl"))

    return total_size


def _cleanup_old_cache_files(max_age_days: float = DEFAULT_MAX_CACHE_AGE, prefix: str | None = None) -> None:
    """
    Remove cache files older than max_age_days.

    Args:
        max_age_days: Maximum age of cache files in days
        prefix: Optional prefix to clean specific cache directory
    """
    cache_root = _get_cache_root()
    if not cache_root.exists():
        return

    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60

    if prefix:
        cache_dirs = [cache_root / prefix] if (cache_root / prefix).exists() else []
    else:
        cache_dirs = [d for d in cache_root.iterdir() if d.is_dir()]

    for cache_dir in cache_dirs:
        for f in cache_dir.glob("*.pkl"):
            if (current_time - f.stat().st_mtime) > max_age_seconds:
                try:
                    f.unlink()
                except PermissionError:
                    # Ignore files we cannot remove due to filesystem restrictions
                    continue


def _enforce_cache_size_limit(
    max_size: int = DEFAULT_MAX_CACHE_SIZE, prefix: str | None = None, verbose: bool = False
) -> None:
    """
    Remove oldest cache files until total size is under max_size if there is more than 1 file.

    Args:
        max_size: Maximum cache size in bytes
        prefix: Optional prefix to enforce limit on specific cache directory
    """
    cache_root = _get_cache_root()
    if not cache_root.exists():
        return

    if prefix:
        cache_dirs = [cache_root / prefix] if (cache_root / prefix).exists() else []
    else:
        cache_dirs = [d for d in cache_root.iterdir() if d.is_dir()]

    for cache_dir in cache_dirs:
        # Get list of files with their modification times
        files = [(f, f.stat().st_mtime) for f in cache_dir.glob("*.pkl")]
        if not files or len(files) <= 1:
            continue

        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x[1])

        # Remove oldest files until under size limit
        current_size = _get_cache_size(prefix)
        if verbose:
            print(f"Current cache size: {current_size / 1024 / 1024 / 1024:.2f} GB")
        for f, _ in files:
            if current_size <= max_size:
                break
            size = f.stat().st_size
            try:
                f.unlink()
                current_size -= size
            except PermissionError:
                # Ignore files we cannot remove; continue trimming others
                continue


def clear_cache(prefix: str | None = None) -> None:
    """
    Clear the cache for a given prefix or all caches if no prefix is specified.

    Args:
        prefix: Optional prefix to clear specific cache directory
    """
    cache_root = _get_cache_root()
    if prefix:
        cache_dir = cache_root / prefix
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
    else:
        if cache_root.exists():
            shutil.rmtree(cache_root)
