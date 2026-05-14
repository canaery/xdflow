from __future__ import annotations

import importlib.util
from pathlib import Path


def test_core_quickstart_example_runs() -> None:
    example_path = Path(__file__).parents[2] / "examples" / "quickstart.py"
    spec = importlib.util.spec_from_file_location("xdflow_quickstart_example", example_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    score, predictions = module.run_quickstart(n_trials_per_class=12, n_channels=4, n_time=20, seed=1)

    assert 0.0 <= score <= 1.0
    assert predictions.data.dims == ("trial",)
    assert predictions.data.sizes["trial"] == 36
