import xarray as xr
from joblib import Parallel, delayed

from xdflow.composite.base import CompositeTransform, Transform, TransformStep
from xdflow.composite.pipeline import Pipeline
from xdflow.core.data_container import DataContainer
from xdflow.transforms.basic_transforms import IdentityTransform


def _fit_transform_one(transform, container, **kwargs):
    """
    Helper function for parallel execution of fit_transform.

    Fits and transforms a single transform, returning both the fitted object and the result.
    This function is defined at module level so it can be pickled by joblib.

    Args:
        transform: The transform to fit and transform
        container: DataContainer to process
        **kwargs: Additional arguments

    Returns:
        Tuple of (fitted_transform, transformed_container)
    """
    transformed_container = transform.fit_transform(container, **kwargs)
    return transform, transformed_container


class TransformUnion(CompositeTransform):
    """
    Applies a set of transforms in parallel and concatenates their outputs.

    This is a special Transform that applies multiple transforms in parallel
    to the same input data and concatenates their results. This is useful for
    combining different types of features (e.g., spectral and temporal) into a
    single feature set.

    Note: This class computes is_stateful dynamically based on constituent transforms,
    so it overrides the class attribute with an instance attribute.

    Uses:
    TransformUnion(transforms_list=[
        Pipeline(name="time_average", steps=[("average_time", AverageTransform(dims="time"))]),
        ("average_channel", AverageTransform(dims="channel")),
    ])
    """

    # No class attributes defined here - will be computed and set as instance attributes

    def __init__(
        self,
        transforms_list: list[tuple[str, Transform] | Pipeline | TransformStep],
        from_dims: list[str] | None = None,
        to_dim: str | None = "feature",
        n_jobs: int = 1,
    ):
        """
        Initialize TransformUnion with multiple transforms.

        Args:
            transforms_list: List of (step_name, transform) tuples, Pipeline objects, or TransformStep objects
            n_jobs: Number of parallel jobs to run.
                   - n_jobs=1 (default): Sequential execution, maintains current behavior
                   - n_jobs=-1: Use all available CPU cores
                   - n_jobs>1: Use the specified number of worker processes
        """

        # store transforms as TransformSteps for easier handling
        self.transforms_list = []
        for transform in transforms_list:
            if isinstance(transform, Pipeline):
                self.transforms_list.append(TransformStep(transform.name, transform))
            elif isinstance(transform, TransformStep):
                self.transforms_list.append(transform)
            elif isinstance(transform, tuple):
                if isinstance(transform[1], Pipeline) and (transform[0] != transform[1].name):
                    raise ValueError(f"Pipeline name ({transform[1].name}) must match the step name ({transform[0]})")
                self.transforms_list.append(TransformStep(transform[0], transform[1]))
            else:
                raise ValueError(f"Invalid transform type: {type(transform)}")
        self.transform_from_name = {step.name: step.transform for step in self.transforms_list}

        # store dims to join
        if from_dims is None:
            raise ValueError("from_dims must be provided")
        self.from_dims = from_dims

        # store parallel processing parameter
        self.n_jobs = n_jobs
        # Name of the final joined dimension
        self.to_dim = to_dim

        # The super().__init__() call will now handle is_stateful
        super().__init__(sel=None, drop_sel=None)  # sel/drop_sel should be given to each transform

        # Compute input/output dimensions
        self.input_dims = ()  # Will be computed in validation if needed
        self.output_dims = ()  # Dynamic based on constituent transforms

        self._validate_composition()

    @property
    def children(self) -> list[Transform]:
        """Returns the pipeline objects from the dictionary."""
        return [step.transform for step in self.transforms_list]

    @property
    def is_predictor(self) -> bool:
        """Returns False for TransformUnion because it concatenates outputs and does not perform prediction."""
        return False

    def _validate_composition(self):
        """Validates that all transforms have the same inputs."""

        # check that from_dims are valid
        if len(self.from_dims) != len(self.transforms_list):
            raise ValueError(
                f"from_dims (len = {len(self.from_dims)}) must be the same length as transforms_list (len = {len(self.transforms_list)})"
            )

        # check that all pipelines have the same input dimensions, if automatically named
        input_dims = ()
        for step in self.transforms_list:
            if step.transform.input_dims:
                if not input_dims:
                    input_dims = step.transform.input_dims
                elif input_dims != step.transform.input_dims:
                    raise ValueError(
                        f"Transforms have different input dimensions: {input_dims} != {step.transform.input_dims}"
                    )
        self.input_dims = input_dims  # set input_dims as instance attribute

        if self.input_dims:  # if input_dims are automatically named, use them to get expected output dims
            expected_output_dims = []
            for step in self.transforms_list:
                step_output_dims = step.transform.get_expected_output_dims(self.input_dims)
                expected_output_dims.append(step_output_dims)

            self._validate_and_get_shared_output_dims(expected_output_dims)

    def fit_transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Fits and transforms the data in a single, efficient parallel pass.

        This method avoids double computation by running `fit_transform` on each
        child transform and collecting both the fitted transformer and the
        transformed data from each worker process.

        Args:
            container: DataContainer to fit and transform
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            DataContainer with concatenated results along 'feature' dimension
        """
        if self.is_stateful and self.n_jobs != 1:
            # Parallel execution
            # The helper returns a list of (fitted_transform, transformed_container) tuples
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_transform_one)(step.transform, container.copy(deep=False), **kwargs)
                for step in self.transforms_list
            )

            # Unzip the results
            fitted_transforms, output_containers = zip(*results)

            # Update the original transform objects with the fitted versions from the workers
            for i, fitted_transform in enumerate(fitted_transforms):
                self.transforms_list[i].transform = fitted_transform

            # Update the name-to-transform mapping as well
            self.transform_from_name = {step.name: step.transform for step in self.transforms_list}

            outputs = [c.data for c in output_containers]

        elif self.is_stateful:
            # Sequential execution for n_jobs=1
            outputs = []
            for step in self.transforms_list:
                # In the sequential case, the transform is modified in-place, and we just collect the data
                transformed_container = step.transform.fit_transform(container.copy(deep=False), **kwargs)
                outputs.append(transformed_container.data)
        else:
            # If not stateful, just transform (no fitting needed)
            outputs = [
                step.transform.transform(container.copy(deep=False), **kwargs).data for step in self.transforms_list
            ]

        # Concatenate the results along a new 'feature' dimension
        concat_outputs = self._concatenate_outputs(outputs)
        return DataContainer(concat_outputs)

    def fit(self, container: DataContainer, **kwargs) -> "TransformUnion":
        """
        Fits all stateful steps in parallel or sequentially.

        For parallel execution, fitted transforms are returned from worker processes
        and used to update the original transform objects.

        Args:
            container: DataContainer to fit on
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            Self (fitted TransformUnion)
        """
        if self.is_stateful and self.n_jobs != 1:
            # Parallel execution for stateful transforms
            fitted_transforms = Parallel(n_jobs=self.n_jobs)(
                delayed(step.transform.fit)(container.copy(deep=False), **kwargs) for step in self.transforms_list
            )
            # Update the original transform objects with fitted versions
            for i, fitted_transform in enumerate(fitted_transforms):
                self.transforms_list[i].transform = fitted_transform
            # Update the name-to-transform mapping
            self.transform_from_name = {step.name: step.transform for step in self.transforms_list}
        elif self.is_stateful:
            # Sequential execution (original behavior)
            for step in self.transforms_list:
                step.transform.fit(container.copy(deep=False), **kwargs)
        # If not stateful, no fitting is needed
        return self

    def _transform(self, container: DataContainer, **kwargs) -> DataContainer:
        """
        Transforms data with each transform in parallel or sequentially and concatenates the results.
        Expects transform outputs to have the same number of dimensions and exactly one unique dimension per output.

        Args:
            container: Input DataContainer
            **kwargs: Additional context/parameters passed through the pipeline

        Returns:
            DataContainer with concatenated results along 'feature' dimension
        """
        if self.n_jobs != 1:
            # Parallel execution
            output_containers = Parallel(n_jobs=self.n_jobs)(
                delayed(step.transform.transform)(container.copy(deep=False), **kwargs) for step in self.transforms_list
            )
            outputs = [c.data for c in output_containers]
        else:
            # Sequential execution (original behavior)
            outputs = [
                step.transform.transform(container.copy(deep=False), **kwargs).data for step in self.transforms_list
            ]

        # Concatenate the results along a new 'feature' dimension
        concat_outputs = self._concatenate_outputs(outputs)

        # Return the new container
        return DataContainer(concat_outputs)

    def _validate_and_get_shared_output_dims(self, output_dims: list[tuple[str, ...]]) -> tuple[str, ...]:
        """Validate outputs are compatible and return shared dims in a stable order.

        All outputs must have the same number of dimensions and they must only
        differ by exactly one dimension (the one listed in `from_dims` for
        that output). The returned tuple preserves the dimension order from the
        first output (excluding its join dim), and each subsequent output must
        match this ordered set for its non-joined dimensions.
        """

        # Validate dimension counts
        ndim = len(output_dims[0])
        if not all(len(output_dim) == ndim for output_dim in output_dims):
            raise ValueError("All outputs must have the same number of dimensions.")

        # Validate presence of join dims and collect ordered non-joined dims from first output
        first_output_dims = output_dims[0]
        first_join_dim = self.from_dims[0]
        if first_join_dim not in first_output_dims:
            raise ValueError(f"from_dims[0] = {first_join_dim} not found in output_dim[0] = {first_output_dims}")
        shared_dims_ordered: tuple[str, ...] = tuple(dim for dim in first_output_dims if dim != first_join_dim)

        # Check that all other outputs share the same ordered non-joined dims
        for i, (dims_i, join_dim_i) in enumerate(zip(output_dims[1:], self.from_dims[1:]), start=1):
            if join_dim_i not in dims_i:
                raise ValueError(f"from_dims[{i}] = {join_dim_i} not found in output_dim[{i}] = {dims_i}")
            non_joined_i = tuple(dim for dim in dims_i if dim != join_dim_i)
            if non_joined_i != shared_dims_ordered:
                raise ValueError(
                    "Output dims are not compatible. Dims apart from from_dims must be the same and in the same order"
                )

        return shared_dims_ordered

    def _concatenate_outputs(self, outputs: list[xr.DataArray]) -> DataContainer:
        """Concatenates the outputs along the appropriate join dimension using xarray's native functionality."""

        if not outputs:
            raise ValueError("No arrays provided.")

        # Validate dimension compatibility first
        output_dims = [output.dims for output in outputs]
        self._validate_and_get_shared_output_dims(output_dims)

        # Determine final join dimension name
        def _compute_join_dim_name(from_dims: list[str]) -> str:
            unique: list[str] = []
            for d in from_dims:
                if d not in unique:
                    unique.append(d)
            return unique[0] if len(unique) == 1 else "_".join(unique)

        computed_join_dim_name = _compute_join_dim_name(self.from_dims)
        final_join_dim_name = self.to_dim if self.to_dim is not None else computed_join_dim_name

        # Prepare outputs: rename join dims and prefix labels for uniqueness
        aligned_outputs: list[xr.DataArray] = []
        for step, output, dim_to_join in zip(self.transforms_list, outputs, self.from_dims):
            output_aligned = output

            # Rename the join dimension to a common name
            if dim_to_join != computed_join_dim_name:
                output_aligned = output_aligned.rename({dim_to_join: computed_join_dim_name})

            # Prefix labels along the join dimension to ensure uniqueness across branches
            if computed_join_dim_name in output_aligned.coords:
                orig_labels = output_aligned.coords[computed_join_dim_name].values
                prefixed_labels = [f"{step.name}_{str(lbl)}" for lbl in orig_labels]
                output_aligned = output_aligned.assign_coords({computed_join_dim_name: prefixed_labels})

            aligned_outputs.append(output_aligned)

        # Use xarray's native concat with exact coordinate matching for non-join dims
        concatenated = xr.concat(aligned_outputs, dim=computed_join_dim_name, join="exact", coords="minimal")

        # Rename the final join dimension if needed
        if computed_join_dim_name != final_join_dim_name:
            concatenated = concatenated.rename({computed_join_dim_name: final_join_dim_name})

        return concatenated

    def get_expected_output_dims(self, input_dims: tuple[str, ...]) -> tuple[str, ...]:
        """
        Returns the expected output dimensions for the PipelineUnion.
        """

        expected_output_dims = []
        for step in self.transforms_list:
            step_output_dims = step.transform.get_expected_output_dims(input_dims)
            expected_output_dims.append(step_output_dims)

        # Validate and recover the ordered shared dims (value unused; validates only)
        self._validate_and_get_shared_output_dims(expected_output_dims)

        # Determine final join dimension expected name
        join_dim_name = self.to_dim if self.to_dim is not None else self.from_dims[0]

        # Preserve the full order of the first child's output dims, but map its join dim
        # to the resolved name (the first join dim).
        first_output = expected_output_dims[0]
        first_join_dim = self.from_dims[0]
        ref_order = tuple(join_dim_name if d == first_join_dim else d for d in first_output)
        return ref_order


class UnionWithInput(TransformUnion):
    """
    Concatenates a transform's output with the original input along a join dimension.

    This is a convenience wrapper around `TransformUnion` that forms a two-branch
    union consisting of the provided `transform` and an identity branch. It is
    equivalent to:

        TransformUnion(
            transforms_list=[("transform", transform), ("identity", IdentityTransform())],
            from_dims=[join_dim, join_dim],
            to_dim=to_dim or join_dim,
            n_jobs=n_jobs,
        )

    Typical usage is to augment feature channels by concatenating the transform's
    output with the original input along `channel`.

    Args:
        transform: The transform or pipeline to apply in the non-identity branch.
        join_dim: The dimension name along which to concatenate both branches.
        to_dim: Optional name for the resulting join dimension. Defaults to `join_dim`.
        n_jobs: Parallelism parameter passed through to `TransformUnion`.
        name: Optional explicit name to assign to the transform branch.
    """

    def __init__(
        self,
        transform_template: Transform,
        join_dim: str,
        to_dim: str | None = None,
        n_jobs: int = 1,
        name: str | None = None,
    ):
        # Store constructor args for get_params/clone
        self.transform_template = transform_template
        self.join_dim = join_dim
        self.name = name

        step_name = name or transform_template.__class__.__name__.lower()
        transforms_list = [
            (step_name, transform_template),
            ("identity", IdentityTransform()),
        ]
        super().__init__(
            transforms_list=transforms_list,
            from_dims=[join_dim, join_dim],
            to_dim=(to_dim if to_dim is not None else join_dim),
            n_jobs=n_jobs,
        )
