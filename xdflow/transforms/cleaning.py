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


# # clean events dict -- in extractor?
# # remove rapid repeats -- not a problem anymore
# # parse events get indices -- in extractor
class RemoveMissingBanksTransform(Transform):
    """
    Detects and removes sessions with missing banks in LFP data.

    A channel block is considered missing if its statistical properties meet
    certain criteria. Missing blocks are replaced with NaNs. This transform
    operates per session.

    Args:
        std_of_means_max (float): Threshold for the max std of channel means.
        mean_of_stds_min (float): Threshold for the min mean of channel stds.
        mean_of_means_min (float): Threshold for the min mean of channel means.
        blocks (dict): A dictionary defining the channel slices for each bank.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ("trial", "channel", "time")
    output_dims: tuple[str, ...] = ("trial", "channel", "time")
    _supports_transform_sel: bool = True

    def __init__(
        self,
        std_of_means_max: float = 3000.0,
        mean_of_stds_min: float = 100.0,
        mean_of_means_min: float = 10000.0,
        blocks: dict = None,
        excluded_channels: Sequence[str] | None = None,
        expected_signal_channels: int | None = 127,
        sel: dict = None,
        drop_sel: dict = None,
        transform_sel: dict = None,
        transform_drop_sel: dict = None,
    ):
        super().__init__(sel=sel, drop_sel=drop_sel, transform_sel=transform_sel, transform_drop_sel=transform_drop_sel)
        if blocks is None:
            blocks = {"first_block": slice(0, 64), "second_block": slice(64, 127)}  # based on idx, not channel ID
        self.std_of_means_max = std_of_means_max
        self.mean_of_stds_min = mean_of_stds_min
        self.mean_of_means_min = mean_of_means_min
        self.blocks = blocks
        self.excluded_channels = excluded_channels  # for clone
        self._excluded_channels = tuple(excluded_channels or ())
        self.expected_signal_channels = expected_signal_channels

    def _transform(self, data_container: DataContainer, **kwargs) -> DataContainer:
        """
        Apply the missing bank removal transformation.
        """
        data = data_container.data

        channel_values = np.asarray(data.coords["channel"].values)
        lfp_mask = _signal_channel_mask(channel_values, self._excluded_channels)
        lfp_channels = channel_values[lfp_mask]

        if self.expected_signal_channels is not None and len(lfp_channels) != self.expected_signal_channels:
            raise ValueError(
                "RemoveMissingBanksTransform expected "
                f"{self.expected_signal_channels} signal channels but found {len(lfp_channels)}. "
                "Set expected_signal_channels=None to disable this validation."
            )

        # Create a 'block' coordinate mapping each channel to its block name
        block_labels = np.full(data.sizes["channel"], "", dtype=object)
        for block_name, block_slice in self.blocks.items():
            # This assumes that the channels in data are the LFP channels and are ordered
            block_labels[block_slice] = block_name
        data = data.assign_coords(block=("channel", block_labels))

        # 2. Group by the new 'block' coordinate first
        grouped_by_block = data.sel(channel=lfp_channels).groupby("block")

        # 3. Define a function to compute all necessary stats per session for a given block
        def compute_session_stats_for_block(block_data):
            # Group by session to compute stats for each session within this block
            session_grouped = block_data.groupby("session")

            # Calculate mean and std per channel for each session
            per_channel_means = session_grouped.mean(dim=("trial", "time"))
            per_channel_stds = session_grouped.std(dim=("trial", "time"))

            # Calculate the final block-level stats for each session
            std_of_means = per_channel_means.std(dim="channel")
            mean_of_stds = per_channel_stds.mean(dim="channel")
            mean_of_means = per_channel_means.mean(dim="channel")

            return xr.Dataset(
                {"std_of_means": std_of_means, "mean_of_stds": mean_of_stds, "mean_of_means": mean_of_means}
            )

        # 4. Apply the function to each block. xarray combines the results.
        all_block_stats = grouped_by_block.apply(compute_session_stats_for_block)

        # 5. Identify all bad banks across all sessions at once from the stats dataset
        is_bad_bank = (
            (all_block_stats["std_of_means"] > self.std_of_means_max)
            | (all_block_stats["mean_of_stds"] < self.mean_of_stds_min)
            | (all_block_stats["mean_of_means"] < self.mean_of_means_min)
        )

        # 6. Build a mask for only signal channels to avoid selecting an unknown 'block' for excluded channels
        lfp_block_labels = data.coords["block"].sel(channel=lfp_channels)
        mask_lfp = is_bad_bank.sel(block=lfp_block_labels, session=data.coords["session"])  # (trial, channel_lfp)

        # print how many bad banks are detected
        n_bad_banks = is_bad_bank.values.flatten().sum()
        if n_bad_banks > 0:
            print(f"Number of bad banks detected: {n_bad_banks}")

        # 7. Apply the mask to LFP data and reconstruct the full dataset
        lfp_data = data.sel(channel=lfp_channels)
        lfp_data_masked = lfp_data.where(~mask_lfp)
        if "block" in lfp_data_masked.coords:
            lfp_data_masked = lfp_data_masked.reset_coords("block", drop=True)

        # Reconstruct data by concatenating masked LFP with unchanged non-LFP channels
        non_lfp_channels = channel_values[~lfp_mask]
        if non_lfp_channels.size > 0:
            non_lfp_data = data.sel(channel=non_lfp_channels)
            if "block" in non_lfp_data.coords:
                non_lfp_data = non_lfp_data.reset_coords("block", drop=True)
            data = xr.concat([lfp_data_masked, non_lfp_data], dim="channel")
            # Restore original channel order
            data = data.sel(channel=data_container.data.coords["channel"])
        else:
            data = lfp_data_masked

        if "block" in data.coords:
            data = data.reset_coords("block", drop=True)

        return DataContainer(data)


class CARTransform(Transform):
    """
    Apply Common Average Referencing (CAR) to the data.

    It can be applied to the entire set of channels ('all') or
    in predefined banks ('by_32').

    Args:
        car_method: 'by_32', 'all', or 'none'.
        excluded_channels: Channels to leave untouched (e.g., reference sensors).
        expected_signal_channels: Validate the number of channels subject to CAR.
    """

    is_stateful: bool = False
    input_dims: tuple[str, ...] = ("trial", "channel", "time")
    output_dims: tuple[str, ...] = ("trial", "channel", "time")

    def __init__(
        self,
        car_method: str = "by_32",
        excluded_channels: Sequence[str] | None = None,
        expected_signal_channels: int | None = 127,
        sel: dict = None,
        drop_sel: dict = None,
    ):
        super().__init__(sel=sel, drop_sel=drop_sel)
        self.car_method = car_method
        self.excluded_channels = excluded_channels  # for clone
        self._excluded_channels = tuple(excluded_channels or ())
        self.expected_signal_channels = expected_signal_channels

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

        elif self.car_method == "by_32":
            if self.expected_signal_channels is not None and len(lfp_channels) != self.expected_signal_channels:
                raise ValueError(
                    f"CAR method 'by_32' expected {self.expected_signal_channels} channels "
                    f"but found {len(lfp_channels)}. Set expected_signal_channels=None to disable this check."
                )

            # Define bank groupings and create a new coordinate
            bank_labels = np.full(data.sizes["channel"], "other", dtype=object)
            bank_map = {
                "bank0": slice(0, 32),
                "bank1": slice(32, 64),
                "bank2": slice(64, 95),
                "bank3": slice(95, 127),
            }
            for bank_name, ch_slice in bank_map.items():
                bank_labels[np.isin(channel_values, lfp_channels[ch_slice])] = bank_name

            data = data.assign_coords(car_bank=("channel", bank_labels))

            # Group by the new coordinate, calculate mean, and subtract
            lfp_data = data.sel(channel=lfp_channels)
            grouped = lfp_data.groupby("car_bank")
            mean_per_bank = grouped.mean(dim="channel")
            transformed_lfp = grouped - mean_per_bank

            # Update the original data array
            data.loc[{"channel": lfp_channels}] = transformed_lfp
            data = data.drop_vars("car_bank")

        elif self.car_method == "all":
            mean_all = data.sel(channel=lfp_channels).mean(dim="channel")
            data.loc[{"channel": lfp_channels}] -= mean_all
        else:
            raise ValueError(
                f"CAR method '{self.car_method}' not recognized. Must be one of 'by_32', 'all', or 'none'."
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
        sel: dict = None,
        drop_sel: dict = None,
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
        sel: dict = None,
        drop_sel: dict = None,
        transform_sel: dict = None,
        transform_drop_sel: dict = None,
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

    def get_expected_output_dims(self, input_dims: tuple[str]) -> tuple[str]:
        return input_dims
