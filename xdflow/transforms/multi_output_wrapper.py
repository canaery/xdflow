"""
Factory for creating multi-output versions of sklearn-compatible estimators.

This module provides a picklable way to wrap any sklearn estimator in MultiOutputRegressor.

**IMPORTANT**: This module is a required dependency for SKLearnPredictor's multi_output parameter.
Do not remove this module even if you only use the multi_output flag, as SKLearnPredictor
imports MultiOutputEstimatorFactory internally when multi_output=True.

The factory pattern solves the pickle/caching problem that arises when trying to use
lambda functions or locally-defined classes with MultiOutputRegressor.
"""

from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor


class MultiOutputEstimatorFactory:
    """
    Picklable factory that wraps any sklearn-compatible estimator in MultiOutputRegressor.

    This is needed because lambda functions cannot be pickled, which breaks caching.
    Using this class allows sklearn estimators to work with multi-target regression
    while maintaining compatibility with pickle/caching.

    Parameters
    ----------
    base_estimator_cls : Type[BaseEstimator]
        The sklearn-compatible estimator class to wrap (e.g., LGBMRegressor, Ridge, etc.)
    """

    def __init__(self, base_estimator_cls: type[BaseEstimator]):
        """
        Initialize the factory with a base estimator class.

        Parameters
        ----------
        base_estimator_cls : Type[BaseEstimator]
            The estimator class to wrap (not an instance)
        """
        self.base_estimator_cls = base_estimator_cls

    def __call__(self, **kwargs) -> MultiOutputRegressor:
        """
        Create a MultiOutputRegressor wrapping the base estimator.

        Parameters
        ----------
        **kwargs
            All keyword arguments are passed to the base estimator constructor

        Returns
        -------
        MultiOutputRegressor
            The wrapped estimator ready for multi-target regression
        """
        base_estimator = self.base_estimator_cls(**kwargs)
        return MultiOutputRegressor(base_estimator)

    def __repr__(self) -> str:
        """Readable representation for debugging."""
        return f"MultiOutputEstimatorFactory({self.base_estimator_cls.__name__})"

    def __reduce__(self):
        """Support for pickling."""
        return (self.__class__, (self.base_estimator_cls,))


def make_multi_output(estimator_cls: type[BaseEstimator]) -> MultiOutputEstimatorFactory:
    """
    Convenience function to create a multi-output factory.

    Parameters
    ----------
    estimator_cls : Type[BaseEstimator]
        The sklearn-compatible estimator class to wrap
    """
    return MultiOutputEstimatorFactory(estimator_cls)
