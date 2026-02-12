import warnings
from collections.abc import Callable, Hashable, Iterator

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from xdflow.core.data_container import DataContainer
from xdflow.cv.base import CrossValidator


class KFoldValidator(CrossValidator):
    """
    Implements cross-validation using a stratified K-Fold strategy with optional holdout set.

    This provides a concrete implementation of CrossValidator using
    scikit-learn's StratifiedKFold for balanced splits across classes.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 0,
        test_size: float = None,
        pooling_score_weight: float = 0.0,
        scoring: str | Callable | None = None,
        stratify_coord: str | None = None,
        exclude_intertrial_from_scoring: bool = False,
        use_stateful_fit_cache: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize KFold cross-validator.

        Args:
            n_splits: Number of folds for cross-validation
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
            test_size: Proportion of data to use as holdout test set (0.0-1.0).
                      If None or 0, no holdout set is created.
            pooling_score_weight: Interpolation factor between the average fold
                score (0.0) and the pooled OOF score (1.0). Defaults to 0.0.
            scoring: Scoring metric to use. If None, auto-selects based on task type.
            stratify_coord: Optional coordinate name to use for stratified splits (train/val/holdout).
            exclude_intertrial_from_scoring: Whether to drop intertrial segments when evaluating folds/holdout.
            use_stateful_fit_cache: Whether to cache stateful transforms during CV.
            verbose: Whether to print verbose output specific to cross-validation.
        """
        super().__init__(
            pooling_score_weight=pooling_score_weight,
            scoring=scoring,
            stratify_coord=stratify_coord,
            verbose=verbose,
            use_stateful_fit_cache=use_stateful_fit_cache,
            exclude_intertrial_from_scoring=exclude_intertrial_from_scoring,
        )
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.test_size = test_size

    def _split_holdout(self, container: DataContainer) -> tuple[np.ndarray, np.ndarray]:
        """
        Splits data into train/validation and holdout sets using stratified sampling.

        Args:
            container: DataContainer to split

        Returns:
            Tuple of (train_val_indices, holdout_indices)
        """
        all_trials = container.data.trial.values

        if self.test_size is None or self.test_size == 0:
            # No holdout set - use all data for cross-validation
            return all_trials, np.array([])

        # Determine if we can do stratified splitting
        final_predictor = self._get_final_predictor()
        stratify_coord = self.stratify_coord

        if stratify_coord is None:
            # Default: single-target classification stratifies on final target coord
            if final_predictor and final_predictor.is_classifier and not isinstance(self.final_target_coord_, list):
                stratify_coord = self.final_target_coord_

        if stratify_coord:
            if stratify_coord not in container.data.coords:
                raise ValueError(
                    f"Stratification coordinate '{stratify_coord}' not found in container coords. "
                    f"Available: {list(container.data.coords.keys())}"
                )
            labels = container.data.coords[stratify_coord].values
            self._validate_stratify_labels(labels, n_splits=None, test_size=self.test_size, context="holdout split")

            # Perform stratified split
            train_val_indices, holdout_indices = train_test_split(
                all_trials, test_size=self.test_size, stratify=labels, random_state=self.random_state
            )
        else:
            # Regression or multi-target: random split without stratification
            train_val_indices, holdout_indices = train_test_split(
                all_trials, test_size=self.test_size, random_state=self.random_state
            )

        return train_val_indices, holdout_indices

    def _get_splits(
        self, container: DataContainer, indices_to_split: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generates splits using scikit-learn's StratifiedKFold on the specified indices.

        Args:
            container: DataContainer to split
            indices_to_split: Trial indices to use for splitting

        Returns:
            Iterator yielding (train_indices, validation_indices) for each fold
        """
        if len(indices_to_split) == 0:
            return iter([])

        # Select the relevant part of the container first
        cv_container = container.data.sel(trial=indices_to_split)

        # Detect task type and use appropriate splitter
        final_predictor = self._get_final_predictor()
        stratify_coord = self.stratify_coord

        if (
            stratify_coord is None
            and final_predictor
            and final_predictor.is_classifier
            and not isinstance(self.final_target_coord_, list)
        ):
            stratify_coord = self.final_target_coord_

        if stratify_coord:
            from sklearn.model_selection import StratifiedKFold

            if stratify_coord not in cv_container.coords:
                raise ValueError(f"Stratification coordinate '{stratify_coord}' not found in container coords.")
            labels_array = cv_container.coords[stratify_coord].values
            self._validate_stratify_labels(labels_array, n_splits=self.n_splits, test_size=None, context="CV split")

            splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            splits = splitter.split(np.arange(len(labels_array)), labels_array)
        else:
            from sklearn.model_selection import KFold

            splitter = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            splits = splitter.split(np.arange(len(indices_to_split)))

        # Yield train/val indices (map positions back to trial IDs)
        for train_pos, val_pos in splits:
            yield cv_container.trial.values[train_pos], cv_container.trial.values[val_pos]


class GroupedKFoldValidator(CrossValidator):
    """
    Implements cross-validation using a stratified K-Fold strategy.
    Groups are specified by the group_coord parameter. K-folds are stratified by both the group and target coordinates.
    Specific groups can be specified for training, validation, and testing using the values of the group_coord coordinate.
    If no groups are specified, all groups are used for training/validation/testing.

    E.g. if group_coord = 'animal', train_groups = None, val_groups = [35], and test_groups = [35],
    all data will be used for training, but only animal 35 will be used for validation and testing.

    Useful for testing the performance of a model across different groups, especially for domain adaptation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 0,
        test_size: float = None,
        pooling_score_weight: float = 0.0,
        group_coord: str = None,
        train_groups: list[Hashable] | Hashable = None,
        val_groups: list[Hashable] | Hashable = None,
        test_groups: list[Hashable] | Hashable = None,
        scoring: str | Callable | None = None,
        stratify_coord: str | None = None,
        stratify_by_group: bool = True,
        exclude_intertrial_from_scoring: bool = False,
        use_stateful_fit_cache: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize GroupedKFoldValidator.

        Args:
            n_splits: Number of folds for cross-validation
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
            test_size: Proportion of data to use as holdout test set (0.0-1.0).
                      If None or 0, no holdout set is created.
            pooling_score_weight: Interpolation factor between the average fold
                score (0.0) and the pooled OOF score (1.0). Defaults to 0.0.
            group_coord: Coordinate to group by.
            train_groups: Groups to use for training. If None, all groups are used.
            val_groups: Groups to use for validation. If None, all groups are used.
            test_groups: Groups to use for testing. If None, all groups are used.
            scoring: Scoring metric to use. If None, auto-selects based on task type.
            stratify_coord: Optional coordinate name to use for stratified splits.
            stratify_by_group: Whether to stratify splits by group coordinate in addition to target.
                              If True (default), stratifies by group+target combination.
                              If False, only stratifies by target (or stratify_coord if set).
            exclude_intertrial_from_scoring: Whether to drop intertrial segments during evaluation.
            use_stateful_fit_cache: Whether to cache stateful transforms during CV.
            verbose: Whether to print verbose output specific to cross-validation.
        """
        super().__init__(
            pooling_score_weight=pooling_score_weight,
            scoring=scoring,
            use_stateful_fit_cache=use_stateful_fit_cache,
            exclude_intertrial_from_scoring=exclude_intertrial_from_scoring,
            stratify_coord=stratify_coord,
            verbose=verbose,
        )
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.test_size = test_size
        self.group_coord = group_coord
        self.stratify_by_group = stratify_by_group

        if not isinstance(train_groups, list) and train_groups is not None:
            train_groups = [train_groups]
        if not isinstance(val_groups, list) and val_groups is not None:
            val_groups = [val_groups]
        if not isinstance(test_groups, list) and test_groups is not None:
            test_groups = [test_groups]

        self.train_groups = train_groups
        self.val_groups = val_groups
        self.test_groups = test_groups

    def _check_groups(self, container: DataContainer):
        """Validates the dimension that the group_coord indexes. Checks that the specified groups are available in the data."""

        def _check_avail_groups(group_type: str, groups: list[Hashable], avail_groups: list[Hashable]):
            n_avail_groups = sum(group in avail_groups for group in groups)
            if n_avail_groups == 0:
                raise ValueError(f"No {group_type} groups {groups} found in data. Available groups: {avail_groups}")
            if n_avail_groups != len(groups):
                warnings.warn(
                    f"Not all {group_type} groups {groups} found in data. Available groups: {avail_groups}",
                    stacklevel=2,
                )

        if self.group_coord not in container.data.coords:
            raise ValueError(f"Group coordinate '{self.group_coord}' not found in data coordinates")

        coord_dims = container.data.coords[self.group_coord].dims
        if len(coord_dims) != 1:
            raise ValueError(
                f"Group coordinate '{self.group_coord}' must index exactly one dimension, "
                f"but it indexes {len(coord_dims)}: {coord_dims}"
            )

        unique_groups = self._discover_groups(container)
        if self.train_groups is not None:
            _check_avail_groups("train", self.train_groups, unique_groups)
        if self.val_groups is not None:
            _check_avail_groups("val", self.val_groups, unique_groups)
        if self.test_groups is not None:
            _check_avail_groups("test", self.test_groups, unique_groups)

        return coord_dims[0]

    def _discover_groups(self, container: DataContainer) -> list[Hashable]:
        """Discovers unique group values from the data."""
        group_values = container.data.coords[self.group_coord].values
        return sorted(np.unique(group_values).tolist())

    def _split_holdout(self, container: DataContainer) -> tuple[np.ndarray, np.ndarray]:
        """
        Splits data into train/validation and holdout sets using stratified sampling.
        Stratification is done by target coordinate (and optionally group coordinate if stratify_by_group=True).
        Keep only indices that are relevant to the specified groups for train/val/test.

        Args:
            container: DataContainer to split

        Returns:
            Tuple of (train_val_indices, holdout_indices)
        """
        self._check_groups(container)

        all_trials = container.data.trial.values
        all_groups = container.data.coords[self.group_coord].values

        if self.test_size is None or self.test_size == 0:
            # No holdout set - use all data for cross-validation
            return all_trials, np.array([])

        # Get labels for stratification
        # Determine stratification coordinate (use stratify_coord if set, otherwise use final_target_coord_)
        stratify_coord = self.stratify_coord if self.stratify_coord is not None else self.final_target_coord_
        if stratify_coord not in container.data.coords:
            raise ValueError(f"Stratification coordinate '{stratify_coord}' not found in container coords.")
        labels = container.data.coords[stratify_coord].values

        # Optionally combine group and label for stratification
        if self.stratify_by_group:
            try:
                # astype(str) is important if labels or groups are not strings
                stratify_labels = np.core.defchararray.add(all_groups.astype(str), labels.astype(str))
            except (TypeError, AttributeError):  # For object arrays
                stratify_labels = [str(grp) + "_" + str(lbl) for grp, lbl in zip(all_groups, labels)]
        else:
            stratify_labels = labels

        # Perform stratified split
        train_val_indices, holdout_indices = train_test_split(
            all_trials, test_size=self.test_size, stratify=stratify_labels, random_state=self.random_state
        )

        if (
            self.train_groups is not None and self.val_groups is not None
        ):  # if both are specified, otherwise keep all and split later
            # keep only indices that are in either train or val groups
            train_val_groups = set(self.train_groups + self.val_groups)
            train_val_indices = train_val_indices[np.isin(all_groups[train_val_indices], train_val_groups)]

        if self.test_groups is not None:
            # keep only indices that are in test_groups
            holdout_indices = holdout_indices[np.isin(all_groups[holdout_indices], self.test_groups)]

        return train_val_indices, holdout_indices

    def _get_splits(
        self, container: DataContainer, indices_to_split: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generates train/validation splits using scikit-learn's StratifiedKFold on the specified indices.
        Stratification is done by target coordinate (and optionally group coordinate if stratify_by_group=True).
        Keep only indices that are relevant to the specified groups for train/val.

        Args:
            container: DataContainer to split
            indices_to_split: Trial indices to use for splitting

        Returns:
            Iterator yielding (train_indices, validation_indices) for each fold
        """
        if len(indices_to_split) == 0:
            return iter([])

        # Initialize stratified k-fold splitter
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

        # Select the relevant part of the container first
        cv_container = container.data.sel(trial=indices_to_split)
        cv_groups = cv_container.coords[self.group_coord].values

        # Now get labels directly from this smaller container
        # Determine stratification coordinate (use stratify_coord if set, otherwise use final_target_coord_)
        stratify_coord = self.stratify_coord if self.stratify_coord is not None else self.final_target_coord_
        if stratify_coord not in cv_container.coords:
            raise ValueError(f"Stratification coordinate '{stratify_coord}' not found in container coords.")
        labels_array = cv_container.coords[stratify_coord].values

        # Optionally combine group and label for stratification
        if self.stratify_by_group:
            try:
                # astype(str) is important if labels or groups are not strings
                stratify_labels = np.core.defchararray.add(cv_groups.astype(str), labels_array.astype(str))
            except (TypeError, AttributeError):  # For object arrays
                stratify_labels = [str(grp) + "_" + str(lbl) for grp, lbl in zip(cv_groups, labels_array)]
        else:
            stratify_labels = labels_array

        # skf.split works on integer positions, so we map back to trial IDs
        for train_pos, val_pos in skf.split(np.arange(len(labels_array)), stratify_labels):
            # keep only indices that are relevant to the specified groups
            if self.train_groups is not None:
                train_pos = train_pos[np.isin(cv_groups[train_pos], self.train_groups)]
            if self.val_groups is not None:
                val_pos = val_pos[np.isin(cv_groups[val_pos], self.val_groups)]

            yield cv_container.trial.values[train_pos], cv_container.trial.values[val_pos]
