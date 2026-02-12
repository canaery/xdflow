"""
Transform classes for the pipeline framework.
"""

import time
import warnings
from abc import ABC, abstractmethod
from inspect import Parameter, signature
from typing import Any, Self

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import LabelEncoder

from xdflow.core.data_container import DataContainer


def _infer_dict_key_type(dict_obj: dict[Any, Any], raw_key: str) -> Any:
    """Infer the appropriate key type for nested dict parameter updates."""
    if dict_obj:
        exemplar = next(iter(dict_obj.keys()))
        target_type = type(exemplar)
        if target_type is bool:
            return raw_key == "True"
        if target_type is int:
            return int(raw_key)
        if target_type is float:
            return float(raw_key)
        if target_type is str and exemplar in {"True", "False"}:
            all_bool_like = all(isinstance(key, str) and key in {"True", "False"} for key in dict_obj.keys())
            if all_bool_like:
                return raw_key == "True"
    if raw_key in ("True", "False"):
        return raw_key == "True"
    return raw_key


def _set_nested_dict_value(dict_obj: dict[Any, Any], key_path: list[str], value: Any) -> dict[Any, Any]:
    """Return a copy of dict_obj with nested key path set to value."""
    if not key_path:
        return dict_obj

    head, *tail = key_path
    typed_head = _infer_dict_key_type(dict_obj, head)
    new_dict = dict_obj.copy()

    if not tail:
        new_dict[typed_head] = value
    else:
        current = new_dict.get(typed_head, {})
        if current and not isinstance(current, dict):
            raise ValueError(f"Cannot set nested dict key '{'__'.join(key_path)}' on non-dict value.")
        new_dict[typed_head] = _set_nested_dict_value(current if isinstance(current, dict) else {}, tail, value)

    return new_dict


class Transform(ABC):
    """
    Abstract base class for all pipeline transformations.

    This is the fundamental building block for all data processing steps.
    Hyperparameters are set when the transform object is instantiated.
    Core principle: Takes a DataContainer, performs an operation, and returns
    a new, modified DataContainer.

    Hard rules:
    - Label-based Operations: When feasible, implementations should operate on dimension names
      (e.g., data.mean(dim='time')) rather than positional axes
    - Data is immutable: Transforms should not modify the original data, but rather create a new DataContainer with the transformed data.
      New transforms should test for this in test_transform_immutability.py

    Soft rules:
    - Auto-generated Labels: When creating new dimensions, generate descriptive,
      automatic names for them when possible
    - Modify data not coords: In general, Transform should modify the data attribute of the DataContainer, not the coords.
      Transforms may reorganize the data which coincides with coords.

    Required Class Attributes (must be defined by each concrete subclass):
        is_stateful: bool - Whether this transform requires fitting to data. Especially relevant for CV
        input_dims: tuple[str, ...] - Expected input dimension names (empty tuple means no constraint)
        output_dims: tuple[str, ...] - Expected output dimension names (empty tuple means dynamic)

    Cloning and authoring policy
    ----------------------------
    - Initialization contract: define all hyperparameters as explicit __init__
      keyword arguments and store them as public attributes with matching names
      (e.g., self.foo for __init__(..., foo=...)). Do not mutate hyperparameters
      after initialization.
    - Fitted state isolation: store any learned/transient state outside the
      constructor in private/non-constructor attributes (e.g., self._stats). These
      must not appear in the __init__ signature so that cloning produces unfitted
      instances.
    - get_params/clone: the default get_params returns public attributes; the
      default clone constructs a new instance using only parameters present in the
      __init__ signature. If you override get_params, return only hyperparameters,
      not fitted state. For composites, use CompositeTransform which performs
      recursive cloning of child transforms via their clone() implementations.
    - No side-channel config: avoid deriving constructor hyperparameters from
      external mutable state at fit time; clone should reproduce initialization
      deterministically.

    kwargs and cooperative multiple inheritance
    ------------------------------------------
    - The base Transform accepts **kwargs solely to support cooperative
      multiple inheritance (e.g., SKLearnPredictor(SKLearnTransform, Predictor)).
    - Rules to avoid breaking clone():
      1) Do not silently consume novel parameters. Every real hyperparameter
         must be an explicit __init__ argument and stored on self.
      2) If a subclass forwards **kwargs to super().__init__, it must also
         separate and store its own hyperparameters explicitly first.
      3) get_params() should only surface public hyperparameters, not fitted
         state; clone() filters to the subclass __init__ signature to ensure
         only constructor-declared params are reconstructed.
      4) If you wrap external estimators (e.g., sklearn), keep their
         constructor kwargs in a dedicated attribute (e.g., self._estimator_kwargs)
         and include them in your clone() override so they are preserved.
    """

    # These should be overridden by each subclass
    is_stateful: bool = False
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()
    _supports_transform_sel: bool = False  # does not change dimensions or coords

    def __init__(
        self,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        transform_sel: dict | None = None,
        transform_drop_sel: dict | None = None,
        **kwargs,
    ):
        """
        Initializes the Transform.

        The `sel` and `drop_sel` arguments are used to subset the data.
        The output will only include the subsetted data.

        The `transform_sel` and `transform_drop_sel`  are used to subset the data specifically for fitting/transforming.
        Fitting/transforming will only be applied to the subsetted data, leaving the remaining data intact.
        The output will include the transformed subsetted data and the untransformed remaining data.

        Args:
            sel: A dictionary to select a subset of data.
            drop_sel: A dictionary to drop a subset of data.
            transform_sel: A dictionary to select a subset of data for transformation.
            transform_drop_sel: A dictionary to drop a subset of data for transformation.
        """
        # kwargs are accepted for cooperative inheritance but not used by Transform itself
        self.sel = sel
        self.drop_sel = drop_sel
        self.transform_sel = transform_sel
        self.transform_drop_sel = transform_drop_sel
        if self.transform_sel and self.transform_drop_sel:
            raise ValueError("Cannot specify both 'transform_sel' and 'transform_drop_sel'.")
        if (self.transform_sel or self.transform_drop_sel) and not self.supports_transform_sel:
            raise TypeError(f"{self.__class__.__name__} does not support selective transformation.")

    @property
    def supports_transform_sel(self) -> bool:
        """Whether this transform supports `transform_sel` semantics.

        Defaults to the class attribute `_supports_transform_sel` but allows subclasses
        to compute support dynamically via an override.
        """
        return self._supports_transform_sel

    def _get_effective_transform_sel(self, container: DataContainer) -> dict | None:
        """
        Resolves transform_sel and transform_drop_sel into a single selection dictionary.

        Args:
            container: The DataContainer to get coordinates from.

        Returns:
            A dictionary suitable for use with .sel() or None.
        """
        if self.transform_sel:
            return self.transform_sel

        if self.transform_drop_sel:
            selection_to_apply = {}
            for dim, labels_to_drop in self.transform_drop_sel.items():
                if dim not in container.data.dims:
                    raise ValueError(f"Dimension '{dim}' in transform_drop_sel not found in DataArray.")

                if isinstance(labels_to_drop, str) or not hasattr(labels_to_drop, "__iter__"):
                    labels_to_drop = [labels_to_drop]

                all_labels = container.data.coords[dim].values
                labels_to_drop_set = set(labels_to_drop)
                labels_to_keep = [label for label in all_labels if label not in labels_to_drop_set]
                selection_to_apply[dim] = labels_to_keep
            return selection_to_apply

        return None

    def _apply_selection(self, container: DataContainer) -> DataContainer:
        """
        Applies the initial `sel` and `drop_sel` transformations.

        Args:
            container: DataContainer to apply selections to

        Returns:
            DataContainer with selections applied
        """
        if self.sel:
            container = container.sel(**self.sel)
        if self.drop_sel:
            container = container.drop_sel(**self.drop_sel)
        return container

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """Determines expected output dims based on manually inputed input_dims"""
        if self.output_dims:
            return self.output_dims
        # e.g. for average, output_dims = tuple([dim for dim in input_dims if dim != self.dim_to_average])
        raise NotImplementedError("Subclasses must either specify output_dims or implement get_expected_output_dims.")

    @abstractmethod
    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Applies the transformation. Must be implemented by all subclasses.
        """
        raise NotImplementedError

    def transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Applies the transformation.

        Args:
            container: DataContainer to transform
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            New DataContainer with transformation applied
        """
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Starting {self.__class__.__name__}.transform")
            start_time = time.time()

        container = self._apply_selection(container)

        effective_transform_sel = self._get_effective_transform_sel(container)

        if effective_transform_sel and self.supports_transform_sel:
            # Create a deep copy to ensure immutability when writing back selected data
            new_container = container.copy(deep=True)  # NOTE: This needs to be deep as sel works on slices.
            # 1. Select the part of the data to be transformed from the original container state
            data_to_transform = container.sel(**effective_transform_sel)  # TODO Takes too long

            # 2. Transform the selected data
            transformed_part = self._transform(data_to_transform, **kwargs)

            # 3. check that the transform_sel output matches the input structure
            self._check_transform_sel_output(data_to_transform, transformed_part)

            # Update the new container with the transformed part
            new_container.data.loc[effective_transform_sel] = transformed_part.data  # TODO takes a bit long too
            transformed_container = new_container
        else:
            transformed_container = self._transform(container, **kwargs)

        if verbose:
            end_time = time.time()
            duration = end_time - start_time
            print(f"Ending {self.__class__.__name__}.transform - took {duration:.3f}s")

        return self._log_history(transformed_container)

    def _check_transform_sel_output(self, data_to_transform: DataContainer, transformed_part: DataContainer) -> bool:
        """
        If the transform changes dims/sizes/coords, raise an error.
        """
        dims_match = tuple(transformed_part.data.dims) == tuple(data_to_transform.data.dims)
        sizes_match = dims_match and all(
            transformed_part.data.sizes[d] == data_to_transform.data.sizes[d] for d in transformed_part.data.dims
        )
        coords_match = sizes_match and all(
            (
                d in transformed_part.data.coords
                and d in data_to_transform.data.coords
                and np.array_equal(transformed_part.data.coords[d].values, data_to_transform.data.coords[d].values)
            )
            for d in transformed_part.data.dims
            if d in data_to_transform.data.dims
        )

        if not (dims_match and sizes_match and coords_match):
            raise ValueError(
                f"Selective transformation via transform_sel is not supported for {self.__class__.__name__} because the transform changes dims/sizes/coords."
            )

    def fit(self, container: DataContainer, **kwargs) -> "Transform":
        """
        Fits the transform to the data.

        Args:
            container: DataContainer to fit on
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            Self (fitted transform)
        """
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Starting {self.__class__.__name__}.fit")
            start_time = time.time()

        if self.is_stateful:  # Stateful transforms require fitting to data.
            container = self._apply_selection(container)

            effective_transform_sel = self._get_effective_transform_sel(
                container
            )  # Returns None if no transform_sel or transform_drop_sel is set

            if effective_transform_sel and self.supports_transform_sel:
                container = container.sel(**effective_transform_sel)

            result = self._fit(container, **kwargs)
        else:
            result = self

        if verbose:
            end_time = time.time()
            duration = end_time - start_time
            print(f"Ending {self.__class__.__name__}.fit - took {duration:.3f}s")

        return result

    def _fit(self, container: DataContainer, **kwargs) -> "Transform":
        """
        Fits the transform to the data.

        Stateful subclasses MUST override this method.
        Stateless transforms SHOULD NOT override it.
        """
        if self.is_stateful:
            raise NotImplementedError("Stateful transforms must implement the 'fit' method.")
        # For stateless transforms, fit does nothing.
        return self

    def fit_transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Fit then transform in a single pass. Note that predictors have their own fit_transform.

        Args:
            container: DataContainer to fit and transform
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            DataContainer with the fit and transform applied
        """
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Starting {self.__class__.__name__}.fit_transform")
            start_time = time.time()

        if self.is_stateful:
            container = self._apply_selection(container)

            effective_transform_sel = self._get_effective_transform_sel(container)

            if effective_transform_sel and self.supports_transform_sel:
                container_to_fit = container.sel(**effective_transform_sel)
                self._fit(container_to_fit, **kwargs)
                data_to_transform = container.sel(**effective_transform_sel)
                transformed_part = self._transform(data_to_transform, **kwargs)

                # check that the transform_sel output matches the input structure
                self._check_transform_sel_output(data_to_transform, transformed_part)

                new_container = container.copy(deep=True)
                new_container.data.loc[effective_transform_sel] = transformed_part.data
                transformed_container = new_container
            else:
                self._fit(container, **kwargs)
                transformed_container = self._transform(container, **kwargs)
            result = self._log_history(transformed_container)
        else:
            # For stateless transforms, just transform
            result = self.transform(container, **kwargs)

        if verbose:
            end_time = time.time()
            duration = end_time - start_time
            print(f"Ending {self.__class__.__name__}.fit_transform - took {duration:.3f}s")

        return result

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get parameters for this transform.

        Args:
            deep (bool): If True, will return the parameters for this transform and
                         contained sub-objects that are themselves transforms.

        Returns:
            dict[str, Any]: Parameter names mapped to their values.
        """
        params = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return params

    def clone(self) -> Self:
        """Return a fresh instance with the same constructor parameters.

        Default implementation constructs a new instance from the filtered
        constructor signature using public parameters. Composite transforms
        should override this to clone children appropriately.
        """
        ctor = signature(type(self).__init__)
        ctor_param_names = {
            name
            for name, p in ctor.parameters.items()
            if name != "self" and p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        }
        raw_params = self.get_params(deep=False) or {}

        filtered_params = {}
        for ctor_param_name in ctor_param_names:
            # all constructor parameters must be present in the raw params
            assert ctor_param_name in raw_params, (
                f"Constructor parameter {ctor_param_name} not found as parameter in {self.__class__.__name__}."
            )
            filtered_params[ctor_param_name] = raw_params[ctor_param_name]

        return type(self)(**filtered_params)

    def set_params(self, **params: Any) -> "Transform":
        """
        Set the parameters of this transform.

        Supports nested parameter setting for dict/object attributes using '__' delimiter.
        For example, 'weight_map__stim_A' will set the "stim_A" key in the weight_map dict.
        Keys are type-inferred from existing dict keys when possible (e.g., "False" -> False).

        Returns:
            self: The transform instance.
        """
        for key, value in params.items():
            if "__" not in key:
                setattr(self, key, value)
                continue

            attr_name, nested_keys_str = key.split("__", 1)
            if not hasattr(self, attr_name):
                raise ValueError(f"'{type(self).__name__}' object has no attribute '{attr_name}'")

            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, dict):
                nested_keys = nested_keys_str.split("__")
                setattr(self, attr_name, _set_nested_dict_value(attr_value, nested_keys, value))
            elif hasattr(attr_value, "set_params"):
                attr_value.set_params(**{nested_keys_str: value})
            else:
                raise ValueError(
                    f"Cannot set nested parameter '{key}': "
                    f"attribute '{attr_name}' is not a dict or doesn't have set_params method"
                )
        return self

    def _log_history(self, container: DataContainer) -> DataContainer:
        """Helper to log the transform to the DataArray's attributes."""
        log_entry = {"class": self.__class__.__name__, "params": self.get_params(deep=False)}
        container.data.attrs["data_history"].append(log_entry)
        return container


class Predictor(Transform, ABC):
    """
    Abstract base class for all estimators that make predictions from transformed data.

    This class uses the template method pattern. It provides the final `predict`
    and `predict_proba` methods, which handle the structuring of the output
    DataContainer. Subclasses must implement the `_predict` and `_predict_proba`
    methods to perform the core prediction logic.
    """

    is_stateful: bool = True  # All predictors require training
    allow_unknown_targets: bool = True  # Whether to allow unknown target values that were not seen during fitting.
    unknown_target_encoding: int = -1
    _cooperative_init_kwarg_names = {"allow_unknown_targets", "unknown_target_encoding"}

    def __init__(
        self,
        sample_dim: str,
        target_coord: str | list[str],
        is_classifier: bool,
        encoder: LabelEncoder | None = None,
        proba: bool = False,
        sel: dict | None = None,
        drop_sel: dict | None = None,
        transform_sel: dict | None = None,
        transform_drop_sel: dict | None = None,
        **kwargs,
    ):
        """
        Args:
            sample_dim: Name of the sample dimension.
            target_coord: Name of the target coordinate (string) or list of target coordinate names (for multi-output).
            is_classifier: Whether the estimator is a classifier (True) or regressor (False).
            encoder: The encoder to use for the predictor. Should be set before predict or transform.
            proba: Whether to return the prediction as probabilities (trials, stimuli) or as a single encoding (trials)
            sel: A dictionary to select a subset of data.
            drop_sel: A dictionary to drop a subset of data.
            transform_sel: A dictionary to select a subset of data for transformation.
            transform_drop_sel: A dictionary to drop a subset of data for transformation.
        """
        allow_unknown_targets = kwargs.pop("allow_unknown_targets", self.allow_unknown_targets)
        unknown_target_encoding = kwargs.pop("unknown_target_encoding", self.unknown_target_encoding)

        # Check if target_coord is explicitly a list (not a pattern string)
        is_multi_target = isinstance(target_coord, list)

        # For now, create a provisional target_coord_list
        # This will be resolved at fit time if it's a pattern
        if is_multi_target:
            target_coord_list = target_coord
        else:
            # Could be single coord or pattern - will be resolved at fit time
            target_coord_list = [target_coord] if isinstance(target_coord, str) else []

        if not is_classifier:
            if proba:
                raise ValueError(
                    f"{self.__class__.__name__} has been initialized with is_classifier=False and proba=True. "
                    "Probabilities should not be returned for continuous target coordinates."
                )
            if encoder is not None:
                raise ValueError(
                    f"{self.__class__.__name__} initialized with is_classifier=False but an encoder was provided. "
                    "Encoders are only valid for classifiers."
                )
        else:
            # For classifiers, multi-target is not supported
            if is_multi_target:
                raise ValueError(
                    f"{self.__class__.__name__} initialized with is_classifier=True and multiple target_coord. "
                    "Multi-target prediction is only supported for regressors."
                )
            # For classifiers, always ensure an encoder exists to standardize label handling.
            if encoder is None:
                encoder = LabelEncoder()

        # With the above defaulting, encoder is guaranteed for classifiers.

        # Initialize the parent Transform (cooperative inheritance)
        super().__init__(
            sel=sel, drop_sel=drop_sel, transform_sel=transform_sel, transform_drop_sel=transform_drop_sel, **kwargs
        )

        self.allow_unknown_targets = allow_unknown_targets
        self.unknown_target_encoding = unknown_target_encoding
        self.sample_dim = sample_dim
        self.target_coord = target_coord  # Keep original (string or list)
        self.target_coord_list = target_coord_list  # Normalized to list
        self.is_multi_target = is_multi_target
        self.is_classifier = is_classifier
        self.encoder = encoder
        self.proba = proba
        self._is_fitted = False

    # get_expected_output_dims is still abstract and must be implemented by subclasses

    @property
    def is_regressor(self) -> bool:
        """Whether this is a regression task (inverse of is_classifier)."""
        return not self.is_classifier

    def get_labels(self) -> list[Any]:
        """
        Return the learned label ordering for classifiers.

        Requires the predictor to be configured as a classifier with a fitted encoder.
        """
        if not self.is_classifier:
            raise TypeError(f"{self.__class__.__name__} is configured as a regressor; labels are undefined.")

        if self.encoder is None:
            raise RuntimeError(
                f"{self.__class__.__name__} does not have an encoder. Classifiers require an encoder to expose labels."
            )

        if not hasattr(self.encoder, "classes_"):
            raise ValueError(
                f"Encoder for {self.__class__.__name__} has not been fitted yet, so classes_ is unavailable."
            )

        return list(self.encoder.classes_)

    def set_encoder(self, encoder: LabelEncoder):
        """
        Sets the encoder for the predictor.

        Args:
            encoder: The encoder to set
        """

        if not self.is_classifier:
            raise ValueError(
                f"{self.__class__.__name__} has been initialized with is_classifier=False. Continuous target coordinates for regression should not be encoded."
            )

        self.encoder = encoder

    def fit_and_set_encoder(self, data: xr.DataArray) -> None:
        """
        Fits the encoder and sets it for the predictor.

        Args:
            data: The data to fit the encoder on
        """
        self.encoder.fit(data.coords[self.target_coord].values)
        self.set_encoder(self.encoder)

    def _encode_target_coord(self, data: xr.DataArray) -> xr.DataArray:
        """
        Encodes the target coordinate using the encoder.

        Args:
            data: The data to encode the target coordinate on

        Returns:
            xr.DataArray: The data with the target coordinate encoded. The original values for the target coordinate are kept in a new coordinate.
        """
        if self.encoder is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a fitted 'encoder' for encoding the target coordinate."
            )

        if self.sample_dim not in data.coords[self.target_coord].dims:
            raise ValueError(
                f"Target coordinate {self.target_coord}'s dimensions ({data.coords[self.target_coord].dims}) must include the sample dimension {self.sample_dim}."
            )

        coord_name_orig = f"{self.target_coord}_orig"

        # Save the original target coordinate
        orig_target_coord_data = data.coords[self.target_coord]
        data = data.drop_vars(self.target_coord)

        # Encode the target coordinate
        try:
            encoded_target_coord = self.encoder.transform(orig_target_coord_data.values)
        except ValueError:
            if not getattr(self, "allow_unknown_targets", False):
                raise

            # Handle unknown target values by encoding them as the unknown target encoding
            values = orig_target_coord_data.values
            flat_values = values.reshape(-1)
            mapping = {label: idx for idx, label in enumerate(self.encoder.classes_)}
            encoded_flat = np.full(flat_values.shape, self.unknown_target_encoding, dtype=np.int64)
            for label, idx in mapping.items():
                encoded_flat[flat_values == label] = idx
            unknown_mask = encoded_flat == self.unknown_target_encoding
            if unknown_mask.any():
                # leave unknown labels encoded as placeholder value
                pass
            encoded_target_coord = encoded_flat.reshape(values.shape)

        # Replace the original target coordinate with the encoded one
        data = data.assign_coords(
            {
                self.target_coord: (orig_target_coord_data.dims, encoded_target_coord),
                coord_name_orig: orig_target_coord_data,  # Keep the original target coordinate as a new coordinate
            }
        )

        return data

    def _reset_target_coord(self, data: xr.DataArray) -> xr.DataArray:
        """
        Resets the target coordinate back to the original (not encoded) values. Undoes _encode_target_coord.

        Args:
            data: The data to reset the target coordinate on

        Returns:
            xr.DataArray: The data with the target coordinate reset to the original values.
        """
        if self.encoder is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a fitted 'encoder' for decoding the target coordinate."
            )

        coord_name_orig = f"{self.target_coord}_orig"

        # Remove the encoded target coordinate
        data = data.drop_vars(self.target_coord)

        # Reassign the original target coordinate
        orig_target_coord_data = data.coords[coord_name_orig]
        data = data.assign_coords({self.target_coord: orig_target_coord_data})
        data = data.drop_vars(coord_name_orig)

        return data

    def _get_output_coords(self, data: xr.DataArray) -> dict[str, Any]:
        """
        Gets the output coordinates for the predictor.

        Args:
            data: The data to get the output coordinates from

        Returns:
            A dictionary of the output coordinates
        """
        output_coords = {self.sample_dim: data.coords[self.sample_dim].values}

        for coord_name, coord_data in data.coords.items():
            if (coord_name != self.sample_dim) and (self.sample_dim in coord_data.dims):
                output_coords[coord_name] = coord_data

        return output_coords

    def fit(self, container: DataContainer, **kwargs) -> "Transform":
        """
        Fits the transform to the data.
        Handles encoding of the target coordinate.

        Args:
            container: DataContainer to fit on
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            Self (fitted transform)
        """
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Starting {self.__class__.__name__}.fit")
            start_time = time.time()

        container = self._apply_selection(container)

        effective_transform_sel = self._get_effective_transform_sel(container)

        if effective_transform_sel and self.supports_transform_sel:
            container = container.sel(**effective_transform_sel)

        # Encode the target coordinate if a classifier
        if self.is_classifier:
            # Fit encoder if not yet fitted
            if not hasattr(self.encoder, "classes_"):
                self.fit_and_set_encoder(container.data)
            container = DataContainer(self._encode_target_coord(container.data))

        # Fit the transform
        fitted = self._fit(container, **kwargs)

        # After fitting, check if target coordinates were resolved (for pattern matching)
        if hasattr(fitted, "_resolved_target_coords"):
            resolved = fitted._resolved_target_coords
            fitted.target_coord_list = resolved
            fitted.is_multi_target = len(resolved) > 1

        if verbose:
            end_time = time.time()
            duration = end_time - start_time
            print(f"Ending {self.__class__.__name__}.fit - took {duration:.3f}s")

        self._is_fitted = True

        return fitted

    def fit_transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """Predictor-specific fit/transform to avoid double selection and ensure encoded y during fit.

        Applies selection once, fits on an encoded view (for classifiers), then transforms
        the unencoded selected view directly via the protected `_transform` path.

        Args:
            container: DataContainer to fit and transform
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            DataContainer with the fit and transform applied
        """
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Starting {self.__class__.__name__}.fit_transform")
            start_time = time.time()

        # Apply selection once
        container = self._apply_selection(container)
        effective_transform_sel = self._get_effective_transform_sel(container)
        perform_transform_sel = effective_transform_sel and self._supports_transform_sel

        # Get effective container
        selected_container = container.sel(**effective_transform_sel) if perform_transform_sel else container

        # Default fit container is the selected view; classifiers override with encoded targets
        container_for_fit = selected_container

        # Prepare data for fit: encode labels for supervised classifierss
        if self.is_classifier:
            # Fit encoder if not fitted
            if not hasattr(self.encoder, "classes_"):
                self.fit_and_set_encoder(selected_container.data)
            encoded_da = self._encode_target_coord(selected_container.data)
            container_for_fit = DataContainer(encoded_da)

        # Fit using encoded labels
        self._fit(container_for_fit, **kwargs)

        # After fitting, check if target coordinates were resolved (for pattern matching)
        if hasattr(self, "_resolved_target_coords"):
            resolved = self._resolved_target_coords
            self.target_coord_list = resolved
            self.is_multi_target = len(resolved) > 1

        self._is_fitted = True

        # Transform directly on original-selected data to avoid double encoding/selection
        transformed_container = self._transform(selected_container, **kwargs)  # handles encoding/decoding

        if perform_transform_sel:
            self._check_transform_sel_output(selected_container, transformed_container)
            # Update the full container with the transformed part
            new_container = container.copy(deep=False)  # Shallow copy sufficient
            new_container.data.loc[effective_transform_sel] = transformed_container.data
            transformed_container = new_container

        result = self._log_history(transformed_container)

        if verbose:
            end_time = time.time()
            duration = end_time - start_time
            print(f"Ending {self.__class__.__name__}.fit_transform - took {duration:.3f}s")

        return result

    @abstractmethod
    def _predict(self, data: xr.DataArray, **kwargs) -> np.ndarray:
        """
        Performs the core prediction logic, returning an array of encoded predictions.
        Output shape should be (n_samples)

        Args:
            data: The data to predict on
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            An array of predictions (n_samples)
        """
        raise NotImplementedError

    def _predict_proba(self, data: xr.DataArray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs the core probability prediction, returning probabilities and class encodings.
        Output shape for probabilties should be (n_samples, n_classes), where n_classes are in the order of class encodings.

        Classes are encoded because the target coordinate is encoded before fit/predict for classifiers.
        Class encodings should be the the length of self.encoder.classes_ unless there are missing classes, but they should still be in the same order.
        Missing classes are handled in _align_proba_to_global.

        Args:
            data: The data to predict probabilities on
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            A tuple of (probabilities with shape (n_samples, n_classes), encoded_classes)
        """
        if not self.is_classifier:
            raise AttributeError(
                f"'{self.__class__.__name__}' has not been instantiated as a classifier (is_classifier=False) so should not call the 'predict_proba' method."
            )
        raise NotImplementedError

    def _align_proba_to_global(self, probs: np.ndarray, encoded_classes: np.ndarray) -> np.ndarray:
        """
        Align per-fold probabilities to the global encoder class order, zero-filling missing classes. The

        Args:
            probs: Array of shape (n_samples, encoded_classes)
            encoded_classes: Array of encoded class labels used by the predictor.

        Returns:
            Array of shape (n_samples, n_global_classes) aligned to `self.encoder.classes_`.
        """

        # Normalize shapes
        encoded_classes = np.asarray(encoded_classes).ravel()

        # Strictly require 2D probabilities
        if probs.ndim != 2:
            raise ValueError(
                f"Probabilities must be 2D with shape (n_samples, n_local_classes). Got shape {probs.shape}."
            )
        probs_2d = probs

        # Columns must correspond one-to-one to provided encoded_classes
        if probs_2d.shape[1] != len(encoded_classes):
            raise ValueError(
                f"probs has {probs_2d.shape[1]} columns but encoded_classes has length {len(encoded_classes)}."
            )

        class_labels = self.encoder.inverse_transform(encoded_classes)  # decode the class encodings to labels

        if len(class_labels) != len(self.encoder.classes_):
            warnings.warn(
                f"The number of class labels ({len(class_labels)}) does not match the number of encoder classes ({len(self.encoder.classes_)}). Missing classes will be zero-filled.",
                stacklevel=2,
            )

        class_to_col: dict[Any, int] = {cls: j for j, cls in enumerate(class_labels)}
        aligned = np.zeros((probs_2d.shape[0], len(self.encoder.classes_)), dtype=probs_2d.dtype)
        for j, cls in enumerate(self.encoder.classes_):
            if cls in class_to_col:
                aligned[:, j] = probs_2d[:, class_to_col[cls]]
        return aligned

    # ---- Internal encoded-entrypoints for composites ----
    def _fit_from_encoded(self, container: DataContainer, **kwargs) -> "Transform":
        """
        Fit using a container whose target coordinate is already encoded.

        This bypasses the outer encode step and is intended for composites that have
        already performed encoding at a higher level (e.g., EnsemblePredictor).

        Selection and transform_sel are still applied locally.
        """
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Starting {self.__class__.__name__}._fit_from_encoded")

        if not self.is_classifier:
            # For regressors, same as normal fit without special handling
            return self.fit(container, **kwargs)

        # Ensure encoder is available; when called from a composite, the encoder
        # should already be fitted and aligned.
        if self.encoder is None:
            raise ValueError(f"{self.__class__.__name__}._fit_from_encoded requires an encoder for classifiers.")

        # Apply local selections while preserving encoded view
        local = self._apply_selection(container)
        eff_sel = self._get_effective_transform_sel(local)
        if eff_sel and self.supports_transform_sel:
            local = local.sel(**eff_sel)

        # Delegate to core fit on the already-encoded container
        return self._fit(local, **kwargs)

    def _predict_from_encoded(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Predict from an already-encoded container, returning a DataContainer result.

        Intended for internal use by composites to avoid double-encoding.
        """
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Starting {self.__class__.__name__}._predict_from_encoded")

        data = container.data

        # Call core predictor on encoded view
        predictions = self._predict(data, **kwargs)

        # Validate prediction shape: can be 1D (single target) or 2D (multi-target)
        if predictions.ndim == 1:
            # Single target - keep as 1D
            output_dims = [self.sample_dim]
        elif predictions.ndim == 2:
            # Multi-target - predictions shape is (n_samples, n_targets)
            if not self.is_multi_target:
                raise ValueError(
                    f"Predictor returned 2D predictions but was initialized with single target_coord. "
                    f"Got prediction shape {predictions.shape}"
                )
            output_dims = [self.sample_dim, "target"]
        else:
            raise ValueError(
                f"Predictions must be 1D (single target) or 2D (multi-target), but got {predictions.ndim}D"
            )

        # Decode predictions and restore original target coord for output coords
        if self.is_classifier:
            if self.encoder is None:
                raise ValueError(
                    f"{self.__class__.__name__}._predict_from_encoded requires an encoder for classifiers."
                )
            predictions = self.encoder.inverse_transform(predictions)
            data = self._reset_target_coord(data)

        output_coords = self._get_output_coords(data)

        # Add target coordinate for multi-target predictions
        if predictions.ndim == 2:
            output_coords["target"] = self.target_coord_list

        output_da = xr.DataArray(
            predictions,
            dims=output_dims,
            coords=output_coords,
            attrs=data.attrs,
            name="prediction",
        )
        return DataContainer(output_da)

    def _predict_proba_from_encoded(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Predict probabilities from an already-encoded container.

        Aligns class axis to the global encoder classes and restores original coords.
        """
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Starting {self.__class__.__name__}._predict_proba_from_encoded")

        if not self.is_classifier:
            raise AttributeError(
                f"'{self.__class__.__name__}' has not been instantiated as a classifier (is_classifier=False) so should not call the 'predict_proba' method."
            )
        if self.encoder is None:
            raise ValueError(
                f"{self.__class__.__name__}._predict_proba_from_encoded requires an encoder for classifiers."
            )

        data = container.data
        probs, class_labels = self._predict_proba(data, **kwargs)
        probs = self._align_proba_to_global(probs, class_labels)

        # Restore original target coord for output coords
        data = self._reset_target_coord(data)
        output_coords = self._get_output_coords(data)
        output_coords["class"] = self.encoder.classes_

        output_da = xr.DataArray(probs, dims=(self.sample_dim, "class"), coords=output_coords, attrs=data.attrs)
        return DataContainer(output_da)

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Transforms data based on the `proba` setting.
        This allows a Predictor to be used as a featurizer in a pipeline.

        Handles encoding and decoding of the target coordinate.

        Args:
            container: DataContainer to transform
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            DataContainer with the transform applied.
                If proba is True, the output will have shape (n_trials, n_classes).
                If proba is False, the output will have shape (n_trials, n_targets).
        """
        data = container.data
        if self.is_classifier:
            data = self._encode_target_coord(data)  # already handles ValueError if encoder is not fitted

        if self.proba:  # only for classifiers, checked in __init__
            # Use the core _predict_proba logic to get raw probabilities
            probabilities, class_labels = self._predict_proba(data, **kwargs)

            # Align columns to global classes
            probabilities = self._align_proba_to_global(probabilities, class_labels)
            transformed_data = probabilities
            output_dim_name = "class"
        else:
            # Use the core _predict logic to get predictions
            predictions = self._predict(data, **kwargs)
            if self.is_classifier:
                predictions = self.encoder.inverse_transform(predictions)

            # Handle both 1D (single target) and 2D (multi-target) predictions
            if predictions.ndim == 1:
                transformed_data = predictions[:, np.newaxis]  # Make 2D (n_samples, 1)
                output_dim_name = "prediction"
            elif predictions.ndim == 2:
                transformed_data = predictions  # Already 2D (n_samples, n_targets)
                output_dim_name = "target"
            else:
                raise ValueError(f"Predictions must be 1D or 2D, got {predictions.ndim}D")

        # Set output coordinates
        if self.is_classifier:
            data = self._reset_target_coord(data)

        output_coords = self._get_output_coords(data)

        if self.proba:
            output_coords[output_dim_name] = self.encoder.classes_
        elif predictions.ndim == 2 and self.is_multi_target:
            # Multi-target: use target coordinate names
            output_coords[output_dim_name] = self.target_coord_list
        else:
            # Single target or classifier
            output_coords[output_dim_name] = np.arange(transformed_data.shape[1])

        output_dims = [self.sample_dim, output_dim_name]
        output_da = xr.DataArray(transformed_data, dims=output_dims, coords=output_coords, attrs=data.attrs)

        return DataContainer(output_da)

    def predict(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Predicts labels, handling data selection and output structuring.

        Args:
            container: DataContainer to predict on
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            DataContainer with predictions, data is 1D with shape (n_trials,)
        """
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Starting {self.__class__.__name__}.predict")
            start_time = time.time()

        container = self._apply_selection(container)
        data = container.data

        # Encode target_coord if classifier
        if self.is_classifier:
            data = self._encode_target_coord(data)

        # Make the prediction
        predictions = self._predict(data, **kwargs)

        # Validate prediction shape: can be 1D (single target) or 2D (multi-target)
        if predictions.ndim == 1:
            output_dims = [self.sample_dim]
        elif predictions.ndim == 2:
            if not self.is_multi_target:
                raise ValueError(
                    f"Predictor returned 2D predictions but was initialized with single target_coord. "
                    f"Got prediction shape {predictions.shape}"
                )
            output_dims = [self.sample_dim, "target"]
        else:
            raise ValueError(
                f"Predictions must be 1D (single target) or 2D (multi-target), but got {predictions.ndim}D"
            )

        # Inverse transform the prediction if encoded
        if self.is_classifier:
            predictions = self.encoder.inverse_transform(predictions)
            data = self._reset_target_coord(data)

        # Set the coordinates
        output_coords = self._get_output_coords(data)

        # Add target coordinate for multi-target predictions
        if predictions.ndim == 2:
            output_coords["target"] = self.target_coord_list

        # Create the output DataArray
        output_da = xr.DataArray(
            predictions, dims=output_dims, coords=output_coords, attrs=data.attrs, name="prediction"
        )
        result = DataContainer(output_da)

        if verbose:
            end_time = time.time()
            duration = end_time - start_time
            print(f"Ending {self.__class__.__name__}.predict - took {duration:.3f}s")

        return result

    def predict_proba(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Predicts probabilities, handling data selection and output structuring.
        DataContainer has data with shape (sample_dim, class) (e.g. n_trials, n_stimuli)
        """
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(f"Starting {self.__class__.__name__}.predict_proba")
            start_time = time.time()

        if not self.is_classifier:
            raise AttributeError(
                f"'{self.__class__.__name__}' has not been instantiated as a classifier (is_classifier=False) so should not call the 'predict_proba' method."
            )

        container = self._apply_selection(container)
        data = container.data

        # Encode target_coord
        data = self._encode_target_coord(data)

        # Make the prediction
        probabilities, class_labels = self._predict_proba(data, **kwargs)

        # Align to global classes
        probabilities = self._align_proba_to_global(probabilities, class_labels)

        assert probabilities.ndim == 2, f"Probabilities should have 2 dimensions, but got {probabilities.ndim}"

        # Set the coordinates
        data = self._reset_target_coord(data)
        output_coords = self._get_output_coords(data)
        output_coords["class"] = self.encoder.classes_

        output_da = xr.DataArray(probabilities, dims=(self.sample_dim, "class"), coords=output_coords, attrs=data.attrs)
        result = DataContainer(output_da)

        if verbose:
            end_time = time.time()
            duration = end_time - start_time
            print(f"Ending {self.__class__.__name__}.predict_proba - took {duration:.3f}s")

        return result


class SampleWeightMixin:
    """
    Mixin providing generic coordinate-to-array extraction for sample weights.

    This mixin decouples weight reading/alignment (generic across frameworks) from
    signature inspection and kwargs building (framework-specific).

    Any transform that wants to support sample weights can inherit this mixin to gain:
    - `sample_weight_coord` attribute for specifying the weight coordinate name
    - `_extract_sample_weight()` method for reading and aligning weights from a DataArray

    The transform is then responsible for:
    - Checking if its underlying estimator/model supports sample weights
    - Passing the weights to the appropriate fit/train method

    Example:
        class MyPredictor(Transform, SampleWeightMixin):
            def __init__(self, sample_weight_coord=None, **kwargs):
                super().__init__(**kwargs)
                self.sample_weight_coord = sample_weight_coord

            def _fit(self, container: DataContainer, **kwargs):
                X, sample_index = self._prepare_data(container.data)
                weights = self._extract_sample_weight(container.data, sample_index)
                if weights is not None:
                    self.model.fit(X, sample_weight=weights)
                else:
                    self.model.fit(X)
    """

    sample_weight_coord: str | None = None

    def _extract_sample_weight(self, data: xr.DataArray, sample_index: pd.Index) -> np.ndarray | None:
        """
        Read and align sample weights from the configured coordinate, if present.

        Args:
            data: xarray DataArray containing the sample weight coordinate
            sample_index: pandas Index of samples to align weights to

        Returns:
            1D numpy array of weights aligned to sample_index, or None if no weights configured

        Raises:
            ValueError: If weight coordinate is misconfigured or contains invalid data
        """
        coord_name = getattr(self, "sample_weight_coord", None)
        if not coord_name:
            return None
        if coord_name not in data.coords:
            return None

        # Get sample_dim from the transform (required for all transforms that use this mixin)
        sample_dim = getattr(self, "sample_dim", "trial")

        weights = data.coords[coord_name]
        if sample_dim not in weights.dims:
            raise ValueError(
                f"Sample weight coordinate '{coord_name}' must include the sample dimension '{sample_dim}'."
            )

        # Align weights to sample_index
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

        if weight_values.shape[0] != sample_index.shape[0]:
            raise ValueError(
                f"Sample weight coordinate '{coord_name}' length ({weight_values.shape[0]}) "
                f"does not match the number of samples ({sample_index.shape[0]})."
            )

        return weight_values
