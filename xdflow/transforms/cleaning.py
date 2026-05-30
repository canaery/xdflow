from collections.abc import Sequence

import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression

from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer


def _signal_channel_mask(channel_values: np.ndarray, excluded_channels: Sequence[str] | None) -> np.ndarray:
    """Return a boolean mask selecting signal channels (i.e., not excluded)."""
    if not excluded_channels:
        return np.ones(len(channel_values), dtype=bool)
    return ~np.isin(channel_values, np.asarray(excluded_channels, dtype=object))


class CARTransform(Transform):
    """
    Apply Common Average Referencing (CAR) to the data.

    CAR can be applied to all signal channels or disabled.

    Args:
        car_method: 'all' or 'none'.
        excluded_channels: Channels to leave untouched (e.g., reference sensors).
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ("trial", "channel", "time")
    output_dims: tuple[str, ...] = ("trial", "channel", "time")

    def __init__(
        self,
        car_method: str = "all",
        excluded_channels: Sequence[str] | None = None,
        sel: dict[str, object] | None = None,
        drop_sel: dict[str, object] | None = None,
    ):
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.car_method = car_method
        self.excluded_channels = excluded_channels  # for clone
        self._excluded_channels = tuple(excluded_channels or ())

    def _transform(self, data_container: DataContainer, **kwargs) -> DataContainer:
        """
        Apply the CAR transformation.
        """
        # This transform modifies the data in-place, so we need a deep copy.
        data = data_container.data.copy(deep=True)

        channel_values = np.asarray(data.coords["channel"].values, dtype=object)
        lfp_mask = _signal_channel_mask(channel_values, self._excluded_channels)
        lfp_channels = channel_values[lfp_mask]

        if self.car_method is None or self.car_method == "none":
            return DataContainer(data)

        elif self.car_method == "all":
            mean_all = data.sel(channel=lfp_channels).mean(dim="channel")
            data.loc[{"channel": lfp_channels}] -= mean_all
        else:
            raise ValueError(
                f"CAR method '{self.car_method}' not recognized. Must be one of 'all' or 'none'."
            )

        return DataContainer(data)


class RegressOutReferenceTransform(Transform):
    """
    Regress out a reference channel from all other channels.

    Uses linear regression to model the relationship between the reference signal
    and each target channel, then subtracts the predicted component.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ("trial", "channel", "time")
    output_dims: tuple[str, ...] = ("trial", "channel", "time")

    def __init__(
        self,
        reference_channel: str,
        excluded_channels: Sequence[str] | None = None,
        sel: dict[str, object] | None = None,
        drop_sel: dict[str, object] | None = None,
    ):
        if not reference_channel:
            raise ValueError("reference_channel must be provided.")
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.reference_channel = reference_channel
        self.excluded_channels = excluded_channels  # for clone
        extras = tuple(excluded_channels or ())
        self._excluded_channels = tuple(set(extras) | {reference_channel})

    def _transform(self, data_container: DataContainer, **kwargs) -> DataContainer:
        """Apply the reference regression."""
        data = data_container.data.copy(deep=True)

        if self.reference_channel not in data.coords["channel"]:
            raise ValueError(f"Reference channel '{self.reference_channel}' not present in data.")

        ref_signal_da = data.sel(channel=self.reference_channel)
        ref_signal_flat = ref_signal_da.values.reshape(-1, 1)

        channel_values = np.asarray(data.coords["channel"].values)
        target_mask = _signal_channel_mask(channel_values, self._excluded_channels)
        target_channels = channel_values[target_mask]
        for chan in target_channels:
            lfp_signal_da = data.sel(channel=chan)
            lfp_signal_flat = lfp_signal_da.values.flatten()

            model = LinearRegression()
            model.fit(ref_signal_flat, lfp_signal_flat)
            predicted_flat = model.predict(ref_signal_flat)

            predicted_reshaped = predicted_flat.reshape(lfp_signal_da.shape)
            predicted_da = xr.DataArray(predicted_reshaped, dims=lfp_signal_da.dims, coords=lfp_signal_da.coords)
            data.loc[{"channel": chan}] -= predicted_da
        return DataContainer(data)


class RemoveOutliersTransform(Transform):
    """
    Remove outliers from the data by clipping.

    Outliers are identified as values exceeding a specified number of standard
    deviations from the mean. They are replaced by the boundary value.

    Args:
        by_dim (str): The dimension along which to compute stats.
        std_threshold (float): The number of standard deviations to use as the
            threshold. Defaults to 5.0.
        use_fit (bool): Whether to use the fit data to compute the mean and std.
            Defaults to False.
        transform_sel (dict, optional): A dictionary to select a subset of data
            for transformation, leaving the rest untouched.
        transform_drop_sel (dict, optional): A dictionary to select a subset of
            data for transformation by excluding labels, leaving the rest untouched.

    E.g. if your input dims are ("trial", "channel", "time"), and you set by_dim to "channel",
    then the data will be clipped per channel by clipping the data to the std_threshold from the mean of trial and time per channel.
    """

    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()
    _supports_transform_sel: bool = True

    def __init__(
        self,
        by_dim: str,
        std_threshold: float = 5.0,
        use_fit: bool = False,
        sel: dict[str, object] | None = None,
        drop_sel: dict[str, object] | None = None,
        transform_sel: dict[str, object] | None = None,
        transform_drop_sel: dict[str, object] | None = None,
    ):
        super().__init__(sel=sel, drop_sel=drop_sel, transform_sel=transform_sel, transform_drop_sel=transform_drop_sel)
        self.std_threshold = std_threshold
        self.by_dim = by_dim
        self.use_fit = use_fit
        self.is_stateful = self.use_fit

    def _fit(self, data_container: DataContainer, **kwargs) -> "RemoveOutliersTransform":
        """
        Fit the transform to the data.

        Note: This method receives a pre-sliced DataContainer if transform_sel
        or transform_drop_sel was used, so it only needs to compute stats on
        the data it is given.
        """
        if self.use_fit:
            mean_dims = [d for d in data_container.data.dims if d != self.by_dim]
            self.mean = data_container.data.mean(dim=mean_dims)
            self.std = data_container.data.std(dim=mean_dims)
        return self

    def _transform(self, data_container: DataContainer, **kwargs) -> DataContainer:
        """
        Apply the outlier removal transformation.

        Args:
            data_container: The DataContainer holding the data.

        Returns:
            A new DataContainer with outliers clipped.
        """
        data = data_container.data
        std_threshold = self.std_threshold

        if self.use_fit:
            mean = self.mean
            std = self.std
        else:
            mean_dims = [d for d in data.dims if d != self.by_dim]
            mean = data.mean(dim=mean_dims)
            std = data.std(dim=mean_dims)
        lower_bound = mean - std_threshold * std
        upper_bound = mean + std_threshold * std
        transformed_data = data.clip(lower_bound, upper_bound)
        return DataContainer(transformed_data)

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        return input_dims
