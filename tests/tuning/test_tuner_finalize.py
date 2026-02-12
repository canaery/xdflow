from types import SimpleNamespace

import numpy as np

from xdflow.tuning import tuner_utils
from xdflow.tuning.base import Tuner


class DummyPipeline:
    def __init__(self, name: str):
        self.name = name
        self.params = {}
        self.steps = []

    def clone(self):
        return DummyPipeline(self.name)

    def set_params(self, **params):
        self.params.update(params)
        return self


class DummyValidator:
    last_set_pipeline = None
    last_finalize_args = None

    def __init__(self):
        self.random_state = None
        self._pipeline = None

    def __deepcopy__(self, memo):
        new = type(self)()
        memo[id(self)] = new
        return new

    def set_pipeline(self, pipeline):
        self._pipeline = pipeline
        type(self).last_set_pipeline = pipeline

    def finalize_pipeline(self, container, verbose=False):
        type(self).last_finalize_args = {"container": container, "verbose": verbose, "pipeline": self._pipeline}
        return "finalized-model"


def test_finalize_best_pipeline_uses_validator_clone():
    DummyValidator.last_set_pipeline = None
    DummyValidator.last_finalize_args = None

    tuner = Tuner.__new__(Tuner)
    tuner.study = SimpleNamespace(best_trial=SimpleNamespace(params={"pipeline": "pipe"}))
    base_pipeline = DummyPipeline("pipe")
    tuner.pipelines_to_tune = {"pipe": base_pipeline}
    tuner.initial_data = object()
    tuner.cv_strategy = DummyValidator()
    tuner.verbose_transforms = False
    tuner.use_cache = False
    tuner.use_mlflow = False

    result = tuner.finalize_best_pipeline()

    assert result == "finalized-model"
    assert DummyValidator.last_finalize_args["container"] is tuner.initial_data
    assert DummyValidator.last_finalize_args["verbose"] is False
    assert DummyValidator.last_finalize_args["pipeline"] is not base_pipeline
    assert DummyValidator.last_finalize_args["pipeline"].name == "pipe"


class _StubValidator:
    def __init__(self):
        predictor = SimpleNamespace(encoder=SimpleNamespace(classes_=np.array(["A", "B"])), is_classifier=True)
        step = SimpleNamespace(transform=predictor)
        self.pipeline = SimpleNamespace(steps=[step], predictive_transform=predictor)
        self.holdout_confusion_matrix_normalized_ = np.eye(2)
        self.holdout_true_labels_ = np.array(["A", "B"])


class _BaseStubTuner:
    def __init__(self, *_, random_seed=None, **kwargs):
        self.random_seed = random_seed
        self.verbose_transforms = kwargs.get("verbose_transforms", False)
        self.study = SimpleNamespace()

    def tune(self, n_trials=None, show_progress_bar=None):
        self.tune_args = (n_trials, show_progress_bar)

    def score_best_pipeline_on_holdout(self, return_validator=False):
        return 0.9, _StubValidator()

    def finalize_best_pipeline(self, verbose=False):
        raise NotImplementedError


def test_run_tuning_pipeline_returns_finalized_models(monkeypatch):
    calls = {"finalize": []}

    class StubTuner(_BaseStubTuner):
        def finalize_best_pipeline(self, verbose=False):
            calls["finalize"].append((self.random_seed, verbose))
            return f"final-model-{self.random_seed}"

    monkeypatch.setattr(tuner_utils, "Tuner", StubTuner)

    finalized = tuner_utils.run_tuning_pipeline(
        pipelines_to_tune=[],
        cv_strategy=None,
        param_grid={},
        initial_data_container=None,
        n_seeds=2,
        n_trials=5,
        plot_importances=False,
        plot_combined_conf_matrix=False,
        plot_each_seed_conf_matrix=False,
    )

    assert finalized == ["final-model-0", "final-model-1"]
    assert calls["finalize"] == [(0, False), (1, False)]


def test_run_tuning_pipeline_sets_exclude_flag(monkeypatch):
    captured = []

    class DummyCV:
        def __init__(self):
            self.exclude_intertrial_from_scoring = False
            self.random_state = None

        def __deepcopy__(self, memo):
            new = type(self)()
            new.exclude_intertrial_from_scoring = self.exclude_intertrial_from_scoring
            memo[id(self)] = new
            return new

    class RecordingTuner(_BaseStubTuner):
        def __init__(self, *_, cv_strategy=None, exclude_intertrial_from_scoring=None, **kwargs):
            captured.append((cv_strategy.exclude_intertrial_from_scoring, exclude_intertrial_from_scoring))
            super().__init__(*_, **kwargs)
            self.cv_strategy = cv_strategy

        def finalize_best_pipeline(self, verbose=False):
            return "finalized"

    monkeypatch.setattr(tuner_utils, "Tuner", RecordingTuner)

    tuner_utils.run_tuning_pipeline(
        pipelines_to_tune=[],
        cv_strategy=DummyCV(),
        param_grid={},
        initial_data_container=None,
        n_seeds=1,
        n_trials=1,
        plot_importances=False,
        plot_combined_conf_matrix=False,
        plot_each_seed_conf_matrix=False,
        exclude_intertrial_from_scoring=True,
    )

    assert captured == [(False, True)]


def test_run_tuning_pipeline_cv_only_no_confusion_matrix(monkeypatch):
    class DummyCV:
        def __init__(self):
            self.random_state = None

        def __deepcopy__(self, memo):
            new = type(self)()
            memo[id(self)] = new
            return new

        @property
        def metric_name_(self):
            return "r2"

    class StubPipeline:
        def __init__(self):
            self.predictive_transform = SimpleNamespace(is_classifier=False)

    class StubTuner(_BaseStubTuner):
        def __init__(self, *_, **kwargs):
            super().__init__(*_, **kwargs)
            self.cv_strategy = DummyCV()
            self.pipeline = StubPipeline()
            self.study = SimpleNamespace(best_value=0.5)

        def get_best_pipeline(self):
            return self.pipeline

        def finalize_best_pipeline(self, verbose=False):
            return "finalized"

    monkeypatch.setattr(tuner_utils, "Tuner", StubTuner)

    finalized = tuner_utils.run_tuning_pipeline(
        pipelines_to_tune=[],
        cv_strategy=DummyCV(),
        param_grid={},
        initial_data_container=None,
        n_seeds=1,
        n_trials=1,
        plot_importances=False,
        plot_combined_conf_matrix=False,
        plot_each_seed_conf_matrix=False,
        score_on_holdout=False,
    )

    assert finalized == ["finalized"]
