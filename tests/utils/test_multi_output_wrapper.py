"""Tests for multi-output wrapper."""

import pickle

from sklearn.linear_model import Ridge

from xdflow.transforms.multi_output_wrapper import MultiOutputEstimatorFactory


def test_multi_output_factory_pickles():
    factory = MultiOutputEstimatorFactory(Ridge)
    blob = pickle.dumps(factory)
    restored = pickle.loads(blob)
    assert callable(restored)
    assert restored.base_estimator_cls is Ridge
