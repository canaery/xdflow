import inspect
import warnings
from inspect import Parameter, signature
from typing import Any, Self

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.base import BaseEstimator, is_regressor
from sklearn.preprocessing import LabelEncoder

from xdflow.core.base import Predictor, SampleWeightMixin, Transform
from xdflow.core.data_container import DataContainer
from xdflow.utils.inspection import collect_super_init_param_names
from xdflow.utils.target_utils import extract_target_array, resolve_target_coords


class SKLearnTransform(Transform, SampleWeightMixin):
    """
    A wrapper that converts a scikit-learn compatible estimator (supervised or
    unsupervised) into a Transform, assuming the input data is 2D.

    Its constructor takes an uninitialized estimator class and its parameters
    directly. Any keyword arguments not used by this wrapper are passed to the
    estimator's constructor.

    Notes on cooperative multiple inheritance and kwargs
    ----------------------------------------------------
    - This class participates in cooperative multiple inheritance with
      Predictor via SKLearnPredictor(SKLearnTransform, Predictor).
    - It forwards **kwargs up the MRO to allow Predictor to receive its
      parameters (e.g., sample_dim, target_coord) while passing estimator
      kwargs to the wrapped estimator. Do not consume arbitrary kwargs here;
      store wrapper hyperparameters explicitly and keep estimator kwargs in
      self._estimator_kwargs so clone() can reconstruct the estimator.
    """

    is_stateful: bool = True
    _supports_transform_sel: bool = False  # should not support transform_sel because dims are changed

    def __init__(
        self,
        estimator_cls: type[BaseEstimator],
        sample_dim: str,
        target_coord: str = None,
        _estimator_instance: BaseEstimator = None,  # SKLearnPredictor needs to instantiate the estimator; only used internally, not by user
        sample_weight_coord: str | None = "sample_weight",
        sel: dict = None,
        drop_sel: dict = None,
        **kwargs: Any,
    ):
        """
        Initializes the wrapper and the wrapped scikit-learn estimator.

        Args:
            estimator_cls (Type[BaseEstimator]): The uninitialized scikit-learn
                estimator class (e.g., PCA, LogisticRegression).
            sample_dim (str): The name of the dimension that corresponds to samples.
            target_coord (str, optional): The name of the coordinate containing the
                target variable `y` for supervised fitting. If None, the estimator
                is treated as unsupervised. Defaults to None.
            _estimator_instance (BaseEstimator, optional): A pre-initialized estimator.
                If provided, `estimator_cls` and its kwargs are ignored. Defaults to None.
            sample_weight_coord (str | None, optional): Name of the coordinate containing
                optional sample weights to forward to ``fit``. Set to None to disable.
            **kwargs (Any): All other keyword arguments are passed
                to the parent constructor and the constructor of `estimator_cls`.
                'sel' and 'drop_sel' are passed to the parent constructor, the rest are passed to the estimator_cls.
        """
        # For cooperative inheritance, we pass all kwargs up the chain.
        # for SKLearnPredictor, SKLearnTransform is initialized first, then Predictor.
        super().__init__(sample_dim=sample_dim, target_coord=target_coord, sel=sel, drop_sel=drop_sel, **kwargs)

        parent_param_names = collect_super_init_param_names(type(self), SKLearnTransform)
        self.estimator_cls = estimator_cls  # needed for clone
        self.sample_weight_coord = sample_weight_coord
        self._fit_param_support_cache: dict[str, bool] = {}

        # Extract estimator-specific parameters (everything not used by Transform or Predictor)
        if _estimator_instance is not None:
            self.estimator = _estimator_instance
            self._estimator_kwargs = getattr(self, "_estimator_kwargs", None)
            if self._estimator_kwargs is None:
                self._estimator_kwargs = {k: v for k, v in kwargs.items() if k not in parent_param_names}
        else:
            self._estimator_kwargs = {k: v for k, v in kwargs.items() if k not in parent_param_names}
            self.estimator = estimator_cls(**self._estimator_kwargs)

        if not hasattr(self.estimator, "fit"):
            raise TypeError(
                f"The provided estimator class '{estimator_cls.__name__}' must produce an object with a 'fit' method."
            )

        self.sample_dim = sample_dim
        self.target_coord = target_coord

    def _prepare_data(self, data: xr.DataArray) -> tuple[np.ndarray, pd.Index]:
        """
        Validates that data is 2D and returns it in the expected format.
        """
        if data.ndim != 2:
            raise ValueError(
                f"Input data must be 2D (samples, features). Received data with {data.ndim} dimensions: {data.dims}"
            )

        if data.dims[0] != self.sample_dim:
            data = data.transpose(self.sample_dim, ...)

        return data.values, data.coords[self.sample_dim]

    def _resolve_target_coords(self, data: xr.DataArray) -> list[str]:
        """
        Resolve target coordinates from string, list, or pattern.

        Returns:
            List of resolved target coordinate names
        """
        return resolve_target_coords(self.target_coord, data)

    def _fit(self, container: DataContainer, **kwargs) -> "SKLearnTransform":
        """Fits the scikit-learn estimator (supervised or unsupervised)."""
        data = container.data
        X, sample_index = self._prepare_data(data)

        y = None
        if self.target_coord:
            # Resolve target coordinates (handles patterns, lists, and single coords)
            resolved_targets = self._resolve_target_coords(data)
            self._resolved_target_coords = resolved_targets
            y = extract_target_array(resolved_targets, data, validate=False)

            # Check if we need to wrap in MultiOutputRegressor for patterns that resolved to multiple targets
            if (
                hasattr(self, "multi_output")
                and not self.multi_output  # Not already wrapped (only for Predictors)
                and hasattr(self, "is_classifier")
                and not self.is_classifier  # Only for regressors
                and len(resolved_targets) > 1  # Pattern resolved to multiple targets
                and isinstance(self.target_coord, str)
                and "*" in self.target_coord  # Was a pattern
                and hasattr(self, "_base_estimator_cls")
                and inspect.isclass(self._base_estimator_cls)  # Skip lambdas/callables
            ):
                # Need to wrap the estimator in MultiOutputRegressor
                from xdflow.transforms.multi_output_wrapper import MultiOutputEstimatorFactory

                # Re-create estimator with wrapping
                estimator_cls = MultiOutputEstimatorFactory(self._base_estimator_cls)
                self.estimator = estimator_cls(**self._estimator_kwargs)
                self.multi_output = True

        # If supervised, delegate any encoding decisions to Predictor upstream
        fit_kwargs = self._build_fit_kwargs(data, sample_index)

        # Check if estimator uses SklearnCVAdapter and automatically set container context
        from xdflow.cv.sklearn_adapter import set_cv_container

        cv_param = getattr(self.estimator, "cv", None)
        uses_cv_adapter = cv_param is not None and hasattr(cv_param, "cross_validator")

        if uses_cv_adapter:
            # Automatically set container context for SklearnCVAdapter
            with set_cv_container(container):
                if y is not None:
                    self.estimator.fit(X, y, **fit_kwargs)
                else:
                    self.estimator.fit(X, **fit_kwargs)
        else:
            # Normal fit without context
            if y is not None:
                self.estimator.fit(X, y, **fit_kwargs)
            else:
                self.estimator.fit(X, **fit_kwargs)
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get parameters for this transform, including the wrapped estimator.
        This is part of the scikit-learn estimator API.
        """
        # Get parameters from the wrapper itself
        params = super().get_params(deep=deep)
        if hasattr(self, "estimator"):
            # Get parameters from the wrapped estimator
            estimator_params = self.estimator.get_params(deep=deep)
            params.update(estimator_params)
        return params

    def set_params(self, **params: Any) -> "SKLearnTransform":
        """
        Set the parameters of this transform and its wrapped estimator.
        This is part of the scikit-learn estimator API.
        """
        estimator_params = {}
        wrapper_params = {}

        # Separate parameters for the estimator and the wrapper
        for key, value in params.items():
            if hasattr(self, "estimator") and key in self.estimator.get_params(deep=True):
                estimator_params[key] = value
            else:
                wrapper_params[key] = value

        # Set parameters on the estimator
        if hasattr(self, "estimator") and estimator_params:
            self.estimator.set_params(**estimator_params)

        # Set parameters on the wrapper
        super().set_params(**wrapper_params)

        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Override setattr to act as a proxy.

        If the attribute is a parameter of the underlying estimator, set it there.
        Otherwise, set it on the wrapper.
        """
        if hasattr(self, "estimator") and name in self.estimator.get_params():
            setattr(self.estimator, name, value)
        else:
            super().__setattr__(name, value)

    def __hasattr__(self, name: str) -> bool:
        """
        Override hasattr to check if the parameter exists either on the wrapper or the estimator.
        """
        if name in ["estimator", "sample_dim", "target_coord"]:
            return super().__hasattr__(name)
        return hasattr(self.estimator, name) or super().__hasattr__(name)

    def clone(self) -> Self:
        """Return a fresh instance with the same constructor parameters.

        Same as base class, but also includes parameters from the estimator.
        """
        ctor = signature(type(self).__init__)
        ctor_param_names = {
            name
            for name, p in ctor.parameters.items()
            if name != "self" and p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        }
        raw_params = self.get_params(deep=False) or {}
        filtered_params = {k: v for k, v in raw_params.items() if k in ctor_param_names}

        # For SKLearnPredictor with multi_output, use the base estimator class
        if hasattr(self, "_base_estimator_cls") and hasattr(self, "multi_output"):
            filtered_params["estimator_cls"] = self._base_estimator_cls

        # combine filtered_params and estimator_params
        estimator_params = self._estimator_kwargs
        combined_params = {**filtered_params, **estimator_params}

        return type(self)(**combined_params)

    def _build_fit_kwargs(self, data: xr.DataArray, sample_index: pd.Index) -> dict[str, Any]:
        """Collect optional keyword arguments to forward to estimator.fit."""
        fit_kwargs: dict[str, Any] = {}

        sample_weight = self._extract_sample_weight(data, sample_index)
        if sample_weight is not None:
            if self._supports_fit_param("sample_weight"):
                fit_kwargs["sample_weight"] = sample_weight
            else:
                warnings.warn(
                    f"Ignoring coordinate '{self.sample_weight_coord}' because "
                    f"{self.estimator.__class__.__name__}.fit does not accept 'sample_weight'.",
                    RuntimeWarning,
                )
        return fit_kwargs

    def _supports_fit_param(self, param_name: str) -> bool:
        """Check whether the wrapped estimator accepts a specific fit parameter."""
        if param_name not in self._fit_param_support_cache:
            try:
                sig = signature(self.estimator.fit)
            except (TypeError, ValueError) as exc:
                warnings.warn(
                    f"Could not inspect signature for {self.estimator.__class__.__name__}.fit "
                    f"(error: {exc}); assuming it does not accept '{param_name}'.",
                    RuntimeWarning,
                )
                self._fit_param_support_cache[param_name] = False
            else:
                params = sig.parameters
                accepts = param_name in params or any(p.kind is Parameter.VAR_KEYWORD for p in params.values())
                self._fit_param_support_cache[param_name] = accepts
        return self._fit_param_support_cache[param_name]

    # _extract_sample_weight is now inherited from SampleWeightMixin


class SKLearnTransformer(SKLearnTransform):
    """
    A wrapper that converts a scikit-learn compatible transformer into a Transform.
    It assumes the input data is 2D and the estimator has a 'transform' method.
    """

    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()

    def __init__(
        self,
        estimator_cls: type[BaseEstimator],
        sample_dim: str,
        target_coord: str = None,
        output_dim_name: str = "component",
        sel: dict = None,
        drop_sel: dict = None,
        **kwargs: Any,
    ):
        """
        Initializes the wrapper.

        Args:
            estimator_cls (Type[BaseEstimator]): The uninitialized scikit-learn
                estimator class (e.g., PCA).
            sample_dim (str): The name of the dimension that corresponds to samples.
            target_coord (str, optional): The name of the coordinate containing the
                target variable `y` for supervised fitting. If None, the estimator
                is treated as unsupervised. Defaults to None.
            output_dim_name (str): The name for the new dimension created by the transformer.
            **kwargs (Any): All other keyword arguments are passed
                to the parent constructor and the constructor of `estimator_cls`.
                sel and drop_sel are passed to the parent constructor, the rest are passed to the estimator_cls.
        """
        super().__init__(
            estimator_cls=estimator_cls,
            sample_dim=sample_dim,
            target_coord=target_coord,
            sel=sel,
            drop_sel=drop_sel,
            **kwargs,
        )
        self.output_dim_name = output_dim_name
        if not hasattr(self.estimator, "transform"):
            raise TypeError(
                f"The provided estimator class '{self.estimator.__class__.__name__}' must have a 'transform' method."
            )

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """Applies the estimator's transform method to the data."""
        data = container.data
        X, sample_index = self._prepare_data(data)

        transformed_data = self.estimator.transform(X)
        assert transformed_data.ndim == 2, (
            f"Sklearn estimators should output 2D data, but got {transformed_data.ndim} dimensions."
        )

        output_coords = {self.sample_dim: sample_index.values}
        output_dims = [self.sample_dim, self.output_dim_name]

        # Preserve all coordinates that are associated with the sample dimension
        for coord_name, coord_data in data.coords.items():
            if coord_name != self.sample_dim and self.sample_dim in coord_data.dims:
                output_coords[coord_name] = coord_data
        output_coords[self.output_dim_name] = np.arange(transformed_data.shape[1])
        output_da = xr.DataArray(transformed_data, dims=output_dims, coords=output_coords, attrs=data.attrs)

        return DataContainer(output_da)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """Determines the expected output dimensions."""
        if len(input_dims) != 2:
            raise ValueError(f"Expected 2 input dimensions, but got {len(input_dims)}")
        return (self.sample_dim, self.output_dim_name)


class SKLearnPredictor(SKLearnTransform, Predictor):
    """
    A wrapper that converts a scikit-learn compatible classifier into a Predictor.
    It implements the core prediction logic by calling the estimator's
    `predict` and `predict_proba` methods.

    It can also be used as a Transform within a pipeline, where its
    `.transform()` behavior is configurable via the `proba` argument.
    """

    def __init__(
        self,
        estimator_cls: type[BaseEstimator],
        sample_dim: str,
        target_coord: str | list[str],
        encoder: LabelEncoder = None,
        proba: bool = False,
        is_classifier: bool | None = None,
        multi_output: bool = False,
        sel: dict = None,
        drop_sel: dict = None,
        sample_weight_coord: str | None = "sample_weight",
        **kwargs: Any,
    ):
        """
        Initializes the wrapper.

        Args:
            estimator_cls (Type[BaseEstimator]): The uninitialized scikit-learn
                estimator class (e.g., LGBMClassifier, Ridge, LGBMRegressor).
            sample_dim (str): The name of the dimension that corresponds to samples.
            target_coord (str or list of str): The name of the coordinate containing the
                target variable `y` for supervised fitting. Can be:
                - A single coordinate name (e.g., 'oil_dilution')
                - A list of coordinate names (e.g., ['stim_0_concentration', 'stim_1_concentration'])
                - A pattern with wildcards (e.g., '*_concentration' to auto-discover all concentration targets)
                Multi-target regression is only supported for regressors.
            encoder (LabelEncoder): The encoder to use for the predictor. Should be set before predict or transform.
                Only valid for classifiers.
            proba (bool): Whether to call the estimator's `.predict_proba()` method for transform.
                If False, calls the estimator's `.predict()` method.
            is_classifier (bool, optional): Manually specify if this is a classifier (True) or regressor (False).
                If None, auto-detects based on estimator's _estimator_type attribute.
            multi_output (bool): If True, automatically wraps the estimator in MultiOutputRegressor
                for multi-target regression. Only valid for regressors. Default: False.

                When multi_output=True, access the underlying estimators via:
                - predictor.estimator.estimator: The base estimator (e.g., LGBMRegressor)
                - predictor.estimator.estimators_: List of fitted estimators (one per target)
            sample_weight_coord (str, optional): The name of the coordinate containing the sample weights.
                Default: "sample_weight".
            **kwargs (Any): All other keyword arguments are passed to the estimator constructor.
        """
        parent_param_names = collect_super_init_param_names(type(self), SKLearnTransform)
        # Separate kwargs for the estimator from the rest
        self._estimator_kwargs = {k: v for k, v in kwargs.items() if k not in parent_param_names}

        # Store the original estimator class before potential wrapping
        self._base_estimator_cls = estimator_cls

        # Auto-enable multi_output for regressors when multiple targets are requested
        estimator_type_hint = getattr(estimator_cls, "_estimator_type", None)

        # Auto-enable MultiOutputRegressor when the target spec clearly implies multiple targets.
        # Guardrails:
        #   - Respect explicit multi_output flag (do nothing if user already set it).
        #   - Never auto-enable for classifiers (explicit or inferred).
        #   - Only act on estimator classes; lambdas/callables may already wrap MultiOutputRegressor.
        #   - Only trigger when the target spec is a list/tuple with >1 items.
        #   - For patterns (with *), defer the decision until fit() when we can resolve the pattern.
        auto_enable_multi_output = (
            not multi_output
            and is_classifier is not True
            and estimator_type_hint != "classifier"
            and inspect.isclass(estimator_cls)
            and isinstance(target_coord, (list, tuple))
            and len(target_coord) > 1
        )
        if auto_enable_multi_output:
            multi_output = True

        self.multi_output = multi_output

        # Wrap in MultiOutputRegressor if requested
        if multi_output:
            from sklearn.multioutput import MultiOutputRegressor

            from xdflow.transforms.multi_output_wrapper import MultiOutputEstimatorFactory

            _already_multioutput = estimator_cls is MultiOutputRegressor or isinstance(
                estimator_cls, MultiOutputEstimatorFactory
            )
            if not _already_multioutput:
                # Create a wrapper that preserves the original class for inspection
                _original_cls = estimator_cls
                estimator_cls = MultiOutputEstimatorFactory(_original_cls)
                wrapped_name = _original_cls.__name__
            else:
                wrapped_name = getattr(estimator_cls, "__name__", str(estimator_cls))

            # Log the wrapping for debugging
            if kwargs.get("verbose", False):
                print(
                    f"[SKLearnPredictor] Wrapping {wrapped_name} with MultiOutputRegressor for multi-target regression"
                )

        # Initialize the estimator to inspect its type
        if multi_output:
            # For multi-output, create a base instance to inspect type
            _estimator_instance = estimator_cls(**self._estimator_kwargs)
            # Store the actual base estimator type info
            _base_instance = self._base_estimator_cls(**self._estimator_kwargs)
            _estimator_type_source = _base_instance
        else:
            _estimator_instance = estimator_cls(**self._estimator_kwargs)
            _estimator_type_source = _estimator_instance

        # Auto-detect or use manual override
        if is_classifier is None:
            try:
                is_classifier = not is_regressor(_estimator_type_source)
            except Exception as exc:
                raise ValueError(
                    f"Could not auto-detect task type for {self._base_estimator_cls.__name__}. "
                    f"Type cannot be determined with sklearn.base.is_regressor."
                    f"Please explicitly specify is_classifier=True or is_classifier=False."
                ) from exc

        if proba and not hasattr(_estimator_instance, "predict_proba"):
            raise AttributeError(
                f"Estimator '{_estimator_instance.__class__.__name__}' has no method 'predict_proba' but 'proba' was set to True."
            )

        # Correctly initialize both parent classes for cooperative multiple inheritance
        super().__init__(
            estimator_cls=estimator_cls,
            _estimator_instance=_estimator_instance,
            sample_dim=sample_dim,
            target_coord=target_coord,
            encoder=encoder,
            is_classifier=is_classifier,
            proba=proba,
            sel=sel,
            drop_sel=drop_sel,
            sample_weight_coord=sample_weight_coord,
            **kwargs,
        )

    # _transform implemented by Predictor

    def _predict(self, data: xr.DataArray, **kwargs) -> np.ndarray:
        """
        Implements the core prediction logic, returning the predictions (based on target_coord values)
        """
        if not hasattr(self.estimator, "predict"):
            raise AttributeError(f"Estimator '{self.estimator.__class__.__name__}' has no method 'predict'.")
        X, _ = self._prepare_data(data)
        return self.estimator.predict(X)

    def _predict_proba(self, data: xr.DataArray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Implements the core probability prediction logic. Returns raw probabilities.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - probabilities: Array of shape (n_samples, n_classes)
                - class_labels: Array of shape (n_classes,)
        """
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"Estimator '{self.estimator.__class__.__name__}' has no method 'predict_proba'.")
        X, _ = self._prepare_data(data)
        probabilities = self.estimator.predict_proba(X)
        return probabilities, self.estimator.classes_

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """
        Determines the expected output dimensions for the `.transform()` method.
        The public `.predict()` method will always produce a 1D output.
        """
        if len(input_dims) != 2:
            raise ValueError(f"Expected 2 input dimensions, but got {len(input_dims)}")

        if self.proba:
            return (self.sample_dim, "class")
        else:
            return (self.sample_dim, "prediction")
