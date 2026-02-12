"""Visualization utilities for xdflow."""

from collections.abc import Iterable
from typing import Any

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("plotting utilities require the 'viz' extra. Install with: pip install xdflow[viz]") from exc
    return plt


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    labels: Iterable[Any],
    want_plot: bool = False,
    want_confus: bool = False,
    save_as: str | None = None,
    title: str = "Confusion Matrix",
    test_trues: Iterable[Any] | None = None,
    ylabels: Iterable[Any] | None = None,
    xlabels: Iterable[Any] | None = None,
    ax=None,
    show_plot: bool = True,
    show_annotations: bool = True,
    cmap: str = "Blues",
):
    """
    Plot a confusion matrix heatmap with optional annotations.

    Returns:
        The matplotlib module if want_plot is True.
    """
    plt = _require_matplotlib()
    cm = np.array(confusion_matrix)
    labels = list(labels)

    if ylabels is None:
        ylabels = labels
    if xlabels is None:
        xlabels = labels

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
    else:
        fig = ax.figure

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)

    if show_annotations and cm.size:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j] * 100:.2f}%", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save_as:
        fig.savefig(save_as, bbox_inches="tight")

    if not show_plot:
        plt.close(fig)

    if want_confus:
        return cm
    if want_plot:
        return plt
    return None


def plot_combined_confusion_matrix(
    confusion_matrices: Iterable[np.ndarray],
    labels: Iterable[Any],
    f1_scores: Iterable[float] | None = None,
    sample_sizes: np.ndarray | None = None,
    test_trues: Iterable[Iterable[Any]] | None = None,
    want_plot: bool = False,
    want_confus: bool = False,
    title: str | None = None,
    save_as: str | None = None,
    xlabels: Iterable[Any] | None = None,
    ylabels: Iterable[Any] | None = None,
    cmap: str = "Blues",
):
    """
    Plot mean confusion matrix with standard error annotations across folds.
    """
    plt = _require_matplotlib()
    matrices = [np.array(cm) for cm in confusion_matrices]
    stacked = np.stack(matrices, axis=2)
    mean_matrix = np.mean(stacked, axis=2)
    std_matrix = np.std(stacked, axis=2)
    sem_matrix = std_matrix / np.sqrt(stacked.shape[2])

    if title is None:
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mean_matrix, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    labels = list(labels)
    if ylabels is None:
        ylabels = labels
    if xlabels is None:
        xlabels = labels
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)

    for i in range(mean_matrix.shape[0]):
        for j in range(mean_matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{mean_matrix[i, j] * 100:.2f}%\n±{sem_matrix[i, j] * 100:.2f}%",
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save_as:
        fig.savefig(save_as, bbox_inches="tight")

    if not want_plot:
        plt.close(fig)

    if want_confus:
        return mean_matrix, sem_matrix
    if want_plot:
        return plt
    return None


def plot_tune_importances(study, *, want_plot: bool = True):
    """
    Plot Optuna parameter importances for a study.

    Args:
        study: Optuna study object.
        want_plot: Whether to return the matplotlib module.
    """
    plt = _require_matplotlib()
    try:
        from optuna.importance import get_param_importances
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("plot_tune_importances requires Optuna. Install with: pip install xdflow[tuning]") from exc

    importances = get_param_importances(study)
    if not importances:
        raise ValueError("No parameter importances found for the provided study.")

    labels = list(importances.keys())
    values = list(importances.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values)
    ax.set_xlabel("Importance")
    ax.set_title("Optuna Parameter Importances")
    fig.tight_layout()

    if want_plot:
        return plt
    plt.close(fig)
    return None
