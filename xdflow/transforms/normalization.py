import warnings
from typing import Any

from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer
from xdflow.utils.sampling import get_container_by_conditions


def _normalize_per_dim(per_dim: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return (per_dim,) if isinstance(per_dim, str) else tuple(per_dim)


class DemeanTransform(Transform):
    """Subtract a mean independently per selected dimension label.

    `per_dim` names the dimensions whose labels remain distinct while the mean
    is computed over all other dimensions. For data with dimensions `("trial",
    "channel", "time")`, `per_dim="channel"` subtracts a separate channel mean
    computed across trials and time. Unlike xarray's `dim=`, `per_dim` does
    not name dimensions to reduce. When multiple dimensions are provided,
    statistics are computed separately for each coordinate tuple across those dimensions.

    By default the mean is computed from the data being transformed. With
    `use_fit=True`, the mean is learned during `fit` and reused during
    `transform`, which is the usual choice inside cross-validation.
    """

    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()
    _supports_transform_sel: bool = True

    def __init__(
        self,
        per_dim: str | list[str] | tuple[str, ...],
        use_fit: bool = False,
        fit_sel: dict[str, Any] | None = None,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        transform_sel: dict[str, Any] | None = None,
        transform_drop_sel: dict[str, Any] | None = None,
    ):
        """
        Initializes the DemeanTransform.

        Args:
            per_dim: Dimension or dimensions to keep distinct while computing the mean.
            use_fit: Whether to use the fit data to compute the mean.
            fit_sel: A dictionary to select a subset of data for fitting.
                Useful if you want to demean in reference to a subset of the data.
                If specified, use_fit will be set to True.
            sel: A dictionary to select a subset of data for transformation.
            drop_sel: A dictionary to drop a subset of data for transformation.
            transform_sel: A dictionary to select a subset of data for transformation.
            transform_drop_sel: A dictionary to drop a subset of data for transformation.
        """

        super().__init__(sel=sel, drop_sel=drop_sel, transform_sel=transform_sel, transform_drop_sel=transform_drop_sel)
        self.per_dim = per_dim
        self.use_fit = use_fit
        self.is_stateful = self.use_fit
        self.fit_sel = fit_sel

        if self.fit_sel is not None and not self.use_fit:
            warnings.warn("fit_sel is specified but use_fit is False. use_fit will be set to True.")
            self.use_fit = True

    def _fit(self, data_container: DataContainer, **kwargs) -> "DemeanTransform":
        """
        Fit the transform to the data.
        """
        if self.use_fit:
            if self.fit_sel is not None:
                data_container = get_container_by_conditions(data_container, self.fit_sel)

            per_dims = _normalize_per_dim(self.per_dim)
            mean_dims = [d for d in data_container.data.dims if d not in per_dims]
            self.mean = data_container.data.mean(dim=mean_dims)
        return self

    def _transform(self, data_container: DataContainer, **kwargs) -> DataContainer:
        """
        Apply the demean transformation.

        Args:
            data_container: The DataContainer holding the data.

        Returns:
            A new DataContainer with the transformed data.
        """
        data = data_container.data
        if self.use_fit:
            transformed_data = data - self.mean
        else:
            per_dims = _normalize_per_dim(self.per_dim)
            mean_dims = [d for d in data.dims if d not in per_dims]
            transformed_data = data - data.mean(dim=mean_dims)
        return DataContainer(transformed_data)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        return input_dims


class ZScoreTransform(Transform):
    """Apply z-score normalization independently per selected dimension label.

    `per_dim` names the dimensions whose labels remain distinct while the mean
    and standard deviation are computed over all other dimensions. For data with
    dimensions `("trial", "channel", "time")`, `per_dim="channel"`
    normalizes each channel using statistics computed across trials and time.
    Unlike xarray's `dim=`, `per_dim` does not name dimensions to reduce. When
    multiple dimensions are provided, statistics are computed separately for
    each coordinate tuple across those dimensions.

    By default statistics are computed from the data being transformed. With
    `use_fit=True`, statistics are learned during `fit` and reused during
    `transform`, which avoids validation leakage inside cross-validation.
    """

    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()
    _supports_transform_sel: bool = True

    def __init__(
        self,
        per_dim: str | list[str] | tuple[str, ...],
        use_fit: bool = False,
        fit_sel: dict[str, Any] | None = None,
        sel: dict[str, Any] | None = None,
        drop_sel: dict[str, Any] | None = None,
        transform_sel: dict[str, Any] | None = None,
        transform_drop_sel: dict[str, Any] | None = None,
    ):
        """Initialize a z-score transform.

        Args:
            per_dim: Dimension or dimensions to keep distinct while computing the mean and std.
            use_fit: Whether to use the fit data to compute the mean and std.
            fit_sel: A dictionary to select a subset of data for fitting.
                Useful if you want to zscore in reference to a subset of the data.
                If specified, use_fit will be set to True.
            sel: A dictionary to select a subset of data for transformation.
            drop_sel: A dictionary to drop a subset of data for transformation.
            transform_sel: A dictionary to select a subset of data for transformation.
            transform_drop_sel: A dictionary to drop a subset of data for transformation.
        """

        super().__init__(sel=sel, drop_sel=drop_sel, transform_sel=transform_sel, transform_drop_sel=transform_drop_sel)
        self.per_dim = per_dim
        self.use_fit = use_fit
        self.is_stateful = self.use_fit
        self.fit_sel = fit_sel

        if self.fit_sel is not None and not self.use_fit:
            warnings.warn("fit_sel is specified but use_fit is False. use_fit will be set to True.")
            self.use_fit = True

    def _fit(self, data_container: DataContainer, **kwargs) -> "ZScoreTransform":
        """
        Fit the transform to the data.
        """
        if self.use_fit:
            if self.fit_sel is not None:
                data_container = get_container_by_conditions(data_container, self.fit_sel)

            per_dims = _normalize_per_dim(self.per_dim)
            mean_dims = [d for d in data_container.data.dims if d not in per_dims]
            self.mean = data_container.data.mean(dim=mean_dims)
            self.std = data_container.data.std(dim=mean_dims)
        return self

    def _transform(self, data_container: DataContainer, **kwargs) -> DataContainer:
        """
        Apply the Z-score transformation.

        Args:
            data_container: The DataContainer holding the data.

        Returns:
            A new DataContainer with the transformed data.
        """
        data = data_container.data
        if self.use_fit:
            mean = self.mean
            std = self.std
        else:
            per_dims = _normalize_per_dim(self.per_dim)
            mean_dims = [d for d in data.dims if d not in per_dims]
            mean = data.mean(dim=mean_dims)
            std = data.std(dim=mean_dims)
        transformed_data = (data - mean) / std
        return DataContainer(transformed_data)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        return input_dims
