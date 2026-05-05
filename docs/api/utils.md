# Utilities API

## `xdflow.utils`

Top-level public re-exports include:

- `cache_result`
- `get_pipeline_cache_key_dict`
- `get_container_by_conditions`
- `stratified_sample`
- `bandpass_filter`
- `get_freq_band_indices`
- `get_remove_freq_ranges`
- `extract_target_array`
- `resolve_target_coords`

## `xdflow.utils.sampling`

Useful helpers for:

- filtering data by coordinate conditions
- train/test splits on `DataContainer`
- stratified sampling
- group discovery and group selection

## `xdflow.utils.target_utils`

Useful helpers for:

- resolving one or more target coordinates
- stacking multi-target coordinate arrays for predictors and scorers

## `xdflow.utils.visualizations`

Useful helpers for:

- plotting single-fold confusion matrices
- aggregating confusion matrices across folds
- plotting Optuna parameter importances when tuning dependencies are installed
