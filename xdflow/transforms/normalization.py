import warnings

from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer
from xdflow.utils.sampling import get_container_by_conditions


class DemeanTransform(Transform):
    """
    Subtract the mean per specified dimension.

    E.g. if your input dims are ("trial", "channel", "time"), and you set by_dim to "channel",
    then the data will be demeaned per channel by subtracting the mean of trial and time per channel.
    """

    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()
    _supports_transform_sel: bool = True

    def __init__(
        self,
        by_dim: str | list[str] | tuple[str, ...],
        use_fit: bool = False,
        fit_sel: dict = None,
        sel: dict = None,
        drop_sel: dict = None,
        transform_sel: dict = None,
        transform_drop_sel: dict = None,
    ):
        """
        Initializes the DemeanTransform.

        Args:
            by_dim: The dimension or dimensions along which to compute the mean.
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
        self.by_dim = by_dim if isinstance(by_dim, (list, tuple)) else (by_dim,)
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

            mean_dims = [d for d in data_container.data.dims if d not in self.by_dim]
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
            mean_dims = [d for d in data.dims if d not in self.by_dim]
            transformed_data = data - data.mean(dim=mean_dims)
        return DataContainer(transformed_data)

    def get_expected_output_dims(self, input_dims: tuple[str]) -> tuple[str]:
        return input_dims


class ZScoreTransform(Transform):
    """
    Apply Z-score normalization per specified dimension.

    This transform standardizes the data to have a mean of 0 and a standard
    deviation of 1 per specified dimension.

    """

    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()
    _supports_transform_sel: bool = True

    def __init__(
        self,
        by_dim: str | list[str] | tuple[str, ...],
        use_fit: bool = False,
        fit_sel: dict = None,
        sel: dict = None,
        drop_sel: dict = None,
        transform_sel: dict = None,
        transform_drop_sel: dict = None,
    ):
        """

        Args:
            by_dim: The dimension or dimensions along which to compute the mean and std.
            use_fit: Whether to use the fit data to compute the mean and std.
            fit_sel: A dictionary to select a subset of data for fitting.
                Useful if you want to zscore in reference to a subset of the data.
                If specified, use_fit will be set to True.
            sel: A dictionary to select a subset of data for transformation.
            drop_sel: A dictionary to drop a subset of data for transformation.
            transform_sel: A dictionary to select a subset of data for transformation.
            transform_drop_sel: A dictionary to drop a subset of data for transformation.

        E.g. if your input dims are ("trial", "channel", "time"), and you set by_dim to "channel",
        then the data will be z-scored per channel by subtracting the mean of trial and time per channel
        and dividing by the std of trial and time per channel.
        """

        super().__init__(sel=sel, drop_sel=drop_sel, transform_sel=transform_sel, transform_drop_sel=transform_drop_sel)
        self.by_dim = by_dim if isinstance(by_dim, (list, tuple)) else (by_dim,)
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

            mean_dims = [d for d in data_container.data.dims if d not in self.by_dim]
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
            mean_dims = [d for d in data.dims if d not in self.by_dim]
            mean = data.mean(dim=mean_dims)
            std = data.std(dim=mean_dims)
        transformed_data = (data - mean) / std
        return DataContainer(transformed_data)

    def get_expected_output_dims(self, input_dims: tuple[str]) -> tuple[str]:
        return input_dims
