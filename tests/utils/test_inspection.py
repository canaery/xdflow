"""Tests for inspection utilities."""

from xdflow.utils.inspection import collect_super_init_param_names


class BaseA:
    def __init__(self, alpha: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha


class BaseB:
    _cooperative_init_kwarg_names = {"beta"}

    def __init__(self, beta: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta


class Child(BaseA, BaseB):
    def __init__(self, gamma: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma


def test_collect_super_init_param_names():
    params = collect_super_init_param_names(Child, stop_class=BaseA)
    assert "beta" in params
    assert "alpha" not in params
