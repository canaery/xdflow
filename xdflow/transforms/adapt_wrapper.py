import inspect
from inspect import Parameter, signature
from typing import Any

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder

from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer, TransformError
from xdflow.transforms.domain_adaptation import AdaptiveStrategy, AdaptiveTransform

NON_ADAPT_ESTIMATOR_KWARGS = [
    "strategy_name",
    "sample_dim",
    "group_coord",
    "source_group",
    "adapt_sel",
    "sel",
    "drop_sel",
]


class AdaptWrapperStrategy(AdaptiveStrategy):
    """
    An adaptation strategy for adapt classes from the adapt package. This is used with the AdaptWrapperTransform.
    During adaptation, it fits the one source domain and one target domain.
    During transformation, it transforms the source and target domains according to the adapt class.
    Domains are determined by the group_coord.
    """

    def __init__(self, group_coord: str, target_group: str | int | float, n_jobs: int = 1, adapt_sel: dict = None):
        """
        Initialize the AdaptWrapperStrategy.

        Args:
            group_coord: The coordinate to group by. Determines different domains.
            target_group: The target group to adapt to.
            n_jobs: The number of jobs to use for parallel processing.
            adapt_sel: The selection criteria for data used for adaptation calculations. None means all data is used.
        """
        super().__init__(group_coord=group_coord, n_jobs=n_jobs, adapt_sel=adapt_sel)
        self.target_group = target_group
        self.target_params = {}
        self.adapted_params = {}
        self.group_dim = None
        self.seen_target_groups_ = []
        self.seen_groups_ = []

    def _adapt(self, aligner: "AdaptWrapperTransform", container: DataContainer, **kwargs) -> None:
        """
        Adapt the data using a single-source strategy.
        Fit and adapt the target and source domains according to the adapt class.
        Domains are determined by the group_coord.

        Args:
            aligner: The AdaptWrapperAligner instance using this strategy.
            container: The DataContainer with all data for adaptation.
            **kwargs: Additional arguments.
        """

        # Discover grouping structure
        self.group_dim = self._get_group_dim(container)
        all_groups = self._discover_groups(container)
        self.seen_source_groups_ = [g for g in all_groups if g != self.target_group]
        self.seen_groups_ = all_groups

        # TODO: handle more domains for certain classes
        if len(all_groups) != 2:
            raise ValueError("AdaptWrapperStrategy requires exactly two groups for adaptation.")

        if self.target_group not in all_groups:
            raise ValueError(f"Target group '{self.target_group}' not found in data")

        if not self.seen_source_groups_:
            raise ValueError("No source group found in the provided container for adaptation.")

        self.source_group = self.seen_source_groups_[0]  # should only be one source group

        # Convert data to right format
        source_container = self._select_group(container, self.source_group)
        target_container = self._select_group(container, self.target_group)

        aligner._fit_adapt(source_container, target_container, **kwargs)

    def transform(self, aligner: "AdaptWrapperTransform", container: DataContainer, **kwargs) -> DataContainer:
        """
        Transforms data by applying the appropriate source or adapted model to each group.

        Args:
            aligner: The AdaptWrapperAligner instance using this strategy.
            container: The DataContainer to be transformed.
            **kwargs: Additional arguments.

        Returns:
            The transformed DataContainer.
        """
        current_groups = self._discover_groups(container)

        def transform_group(group_val):
            group_container = self._select_group(container, group_val)
            if group_val == self.source_group:
                return aligner._adapted_transform(group_container, domain="source", **kwargs)
            elif group_val == self.target_group:
                return aligner._adapted_transform(group_container, domain="target", **kwargs)
            else:  # Unseen target group
                raise TransformError(
                    f"Group '{group_val}' was not seen during 'adapt'. Seen groups: {self.seen_groups_}"
                )

        group_outputs = []
        if self.n_jobs != 1:
            transformed_containers = Parallel(n_jobs=self.n_jobs)(
                delayed(transform_group)(group_val) for group_val in current_groups
            )
            group_outputs.extend([output.data for output in transformed_containers])
        else:
            for group_val in current_groups:
                transformed_container = transform_group(group_val)
                group_outputs.append(transformed_container.data)

        # Reassemble outputs #TODO: do we need to reassemble in the same order?
        reassembled = xr.concat(group_outputs, dim=self.group_dim)

        # Note: Reordering to match original is complex and may not be necessary.
        # If order is critical, it should be handled carefully.

        return DataContainer(reassembled)


class AdaptWrapperTransform(AdaptiveTransform):
    is_stateful: bool = True
    _supports_transform_sel: bool = False

    _STRATEGY_MAP = {"adapt_wrapper": AdaptWrapperStrategy}

    def __init__(
        self,
        adapt_estimator_cls,
        sample_dim: str,
        target_coord: str,
        group_coord: str,
        target_group: str | int | float,
        random_state: int = 0,
        adapt_sel: dict = None,
        sel: dict = None,
        drop_sel: dict = None,
        **kwargs,
    ):
        strategy_name = "adapt_wrapper"
        super().__init__(
            # base adaptive transform kwargs
            strategy_name=strategy_name,
            sample_dim=sample_dim,
            # strategy kwargs
            group_coord=group_coord,
            target_group=target_group,
            adapt_sel=adapt_sel,
            # transform kwargs
            sel=sel,
            drop_sel=drop_sel,
            **kwargs,
        )

        self.target_coord = target_coord
        self.adapt_estimator_cls = adapt_estimator_cls  # needed for clone
        self.random_state = random_state

        # Extract estimator-specific parameters (everything not used by Transform or Strategy)
        self._adapt_estimator_kwargs = {k: v for k, v in kwargs.items() if k not in NON_ADAPT_ESTIMATOR_KWARGS}
        self.adapt_estimator = adapt_estimator_cls(**self._adapt_estimator_kwargs)

        # check if adapt estimator has "domain" parameter in fit method
        if "domain" in inspect.signature(self.adapt_estimator.fit).parameters:
            self._specifies_transform_domain = True
        else:
            self._specifies_transform_domain = False

    def _fit_adapt(
        self, source_container: DataContainer, target_container: DataContainer, **kwargs
    ) -> "AdaptiveTransform":
        # get source data
        source_data = self._prepare_data(source_container.data)
        source_input = source_data.values
        source_output = source_data.coords[self.target_coord].values

        assert source_data.ndim == 2, "AdaptWrapperTransform requires 2D data"

        # get target data
        target_data = self._prepare_data(target_container.data)
        target_input = target_data.values
        target_output = target_data.coords[self.target_coord].values

        # encoder for outputs, only needed for adapt estimators' fit method, does not affect transform
        encoder = LabelEncoder()
        all_outputs = np.concatenate([source_output, target_output])
        encoder.fit(all_outputs)
        source_output = encoder.transform(source_output)
        target_output = encoder.transform(target_output)

        self.adapt_estimator.fit(X=source_input, y=source_output, Xt=target_input, yt=target_output)

    def _adapted_transform(self, data_container: DataContainer, domain: str, **kwargs) -> DataContainer:
        original_dims = data_container.data.dims
        data = self._prepare_data(data_container.data)

        data_input = data.values

        if self._specifies_transform_domain:
            assert domain in ["source", "target"], "Domain must be either 'source' or 'target'"
            transformed_values = self.adapt_estimator.transform(X=data_input, domain=domain, **kwargs)
        else:
            transformed_values = self.adapt_estimator.transform(X=data_input, **kwargs)

        output_coords = self._get_output_coords(data)
        output_dim_name = [dim for dim in data.dims if dim != self.sample_dim][0]  # should only be one
        output_coords[output_dim_name] = np.arange(transformed_values.shape[1])

        transformed_da = xr.DataArray(transformed_values, dims=data.dims, coords=output_coords, attrs=data.attrs)
        transformed_da = transformed_da.transpose(*original_dims)

        return DataContainer(transformed_da)

    def clone(self) -> "Transform":
        """Return a fresh instance with the same constructor parameters.

        Same as base class, but also includes parameters for the strategy.
        """
        ctor = signature(type(self).__init__)
        ctor_param_names = {
            name
            for name, p in ctor.parameters.items()
            if name != "self" and p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        }

        raw_params = self.get_params(deep=False) or {}
        filtered_params = {k: v for k, v in raw_params.items() if k in ctor_param_names}

        # combine filtered_params, strategy_params and adapt_estimator_params
        strategy_params = self._strategy_kwargs
        adapt_estimator_params = self._adapt_estimator_kwargs
        combined_params = {**filtered_params, **strategy_params, **adapt_estimator_params}

        return type(self)(**combined_params)

    def _get_output_coords(self, data: xr.DataArray) -> dict[str, Any]:
        """
        Gets the output coordinates for the predictor.
        """
        output_coords = {self.sample_dim: data.coords[self.sample_dim].values}

        for coord_name, coord_data in data.coords.items():
            if (coord_name != self.sample_dim) and (self.sample_dim in coord_data.dims):
                output_coords[coord_name] = coord_data

        return output_coords
