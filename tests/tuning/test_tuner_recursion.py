from unittest.mock import MagicMock

import optuna
import pytest

from xdflow.composite.base import CompositeTransform
from xdflow.core.base import Transform
from xdflow.core.data_container import DataContainer
from xdflow.cv.base import CrossValidator
from xdflow.tuning.base import Tuner


class MockSimpleTransform(Transform):
    is_stateful = False

    def __init__(self, param_a=1, param_b="x", **kwargs):
        super().__init__(**kwargs)
        self.param_a = param_a
        self.param_b = param_b

    def _transform(self, data, **kwargs):
        return data  # pragma: no cover

    def get_expected_output_dims(self, input_dims):
        return input_dims


class MockTransformWithState(Transform):
    is_stateful = True

    def __init__(self, random_state=None, **kwargs):
        super().__init__(**kwargs)
        self.random_state = random_state

    def _transform(self, data, **kwargs):
        return data  # pragma: no cover

    def _fit(self, data, **kwargs):
        return self  # pragma: no cover

    def get_expected_output_dims(self, input_dims):
        return input_dims


class MockComposite(CompositeTransform):
    def __init__(self, name: str, steps: list, **kwargs):
        self.name = name
        self.steps = steps
        self.transform_from_name = {step[0]: step[1] for step in steps}
        super().__init__(**kwargs)

    @property
    def children(self):
        return [step[1] for step in self.steps]

    @property
    def is_predictor(self) -> bool:
        return False

    def _validate_composition(self):
        pass

    def _transform(self, data, **kwargs):
        for _, transform in self.steps:  # pragma: no cover
            data = transform.transform(data, **kwargs)  # pragma: no cover
        return data  # pragma: no cover

    def clone(self):
        cloned_steps = [(name, transform.clone()) for name, transform in self.steps]
        return MockComposite(name=self.name, steps=cloned_steps)

    def get_expected_output_dims(self, input_dims):
        return input_dims


MockPipeline = MockComposite
MockSwitchTransform = MockComposite


@pytest.fixture
def nested_pipeline():
    inner_pipeline = MockPipeline(
        name="inner_pipe",
        steps=[
            ("inner_step1", MockSimpleTransform(param_a=10)),
            ("inner_step2", MockTransformWithState(random_state=1)),
        ],
    )

    switch = MockSwitchTransform(
        name="switch",
        steps=[
            ("inner_pipe", inner_pipeline),
            ("switch_option2", MockSimpleTransform()),
        ],
    )

    main_pipeline = MockPipeline(
        name="main_pipe",
        steps=[
            ("main_step1", MockSimpleTransform(param_a=100)),
            ("main_step2", switch),
            ("main_step3", MockTransformWithState(random_state=42)),
        ],
    )
    return main_pipeline


@pytest.fixture
def param_grid(nested_pipeline):
    return {
        nested_pipeline.name: {
            "main_step1": {"param_a": [2]},
            "main_step2": {
                "inner_pipe": {
                    "inner_step1": {"param_a": (10, 20)},
                    "inner_step2": {"random_state": [1, 2, 123]},
                },
                "switch_option2": {"param_b": ["a"]},
            },
            "main_step3": {"random_state": (100, 200.0)},
        }
    }


@pytest.fixture
def mock_cv():
    cv = MagicMock(spec=CrossValidator)
    cv.cross_validate.return_value = 0.9
    return cv


@pytest.fixture
def mock_data_container():
    return MagicMock(spec=DataContainer)


@pytest.fixture
def tuner_instance(nested_pipeline, mock_cv, param_grid, mock_data_container):
    return Tuner(
        pipelines_to_tune=[nested_pipeline],
        cv_strategy=mock_cv,
        param_grid=param_grid,
        initial_data_container=mock_data_container,
        random_seed=123,
    )


def test_inject_random_seeds_into_pipelines(tuner_instance):
    main_pipeline = tuner_instance.pipelines_to_tune["main_pipe"]
    switch = main_pipeline.get_transform_from_name("main_step2")
    inner_pipe = switch.get_transform_from_name("inner_pipe")
    inner_step2 = inner_pipe.get_transform_from_name("inner_step2")
    main_step3 = main_pipeline.get_transform_from_name("main_step3")

    assert inner_step2.random_state == 123
    assert main_step3.random_state == 123


def test_enqueue_initial_trials_invalid_init_raises(tuner_instance, nested_pipeline):
    nested_pipeline.get_transform_from_name("main_step1").param_a = 999

    tuner_instance.study = optuna.create_study()
    with pytest.raises(ValueError, match="has initialized value"):
        tuner_instance._enqueue_initial_trials()


def test_suggest_and_set_params(tuner_instance, nested_pipeline, param_grid):
    study = optuna.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))

    trial.suggest_int = MagicMock(return_value=15)
    trial.suggest_categorical = MagicMock(return_value=2)
    trial.suggest_float = MagicMock(return_value=150.5)

    pipeline_param_grid = param_grid[nested_pipeline.name]

    tuner_instance._suggest_and_set_params(nested_pipeline, pipeline_param_grid, nested_pipeline.name, trial)

    main_step1 = nested_pipeline.get_transform_from_name("main_step1")
    switch = nested_pipeline.get_transform_from_name("main_step2")
    inner_pipe = switch.get_transform_from_name("inner_pipe")
    inner_step1 = inner_pipe.get_transform_from_name("inner_step1")
    inner_step2 = inner_pipe.get_transform_from_name("inner_step2")
    main_step3 = nested_pipeline.get_transform_from_name("main_step3")

    assert main_step1.param_a == 2
    assert inner_step1.param_a == 15
    assert inner_step2.random_state == 2
    assert main_step3.random_state == 150.5
