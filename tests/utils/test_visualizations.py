"""Tests for visualization utils."""

import pytest

from xdflow.utils import visualizations


def test_plot_confusion_matrix_requires_matplotlib(monkeypatch):
    """plot_confusion_matrix should error when matplotlib is unavailable."""
    monkeypatch.setattr(visualizations, "_require_matplotlib", lambda: (_ for _ in ()).throw(ImportError("missing")))

    with pytest.raises(ImportError):
        visualizations.plot_confusion_matrix([[1.0]], labels=["a"])
