"""Confusion matrix and CV comparison visualization.

Supports arbitrary NxN confusion matrices (binary or multiclass).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import (
    CLASS_COLORS,
    CLASS_NAMES,
    FIGSIZE_SINGLE,
    save_figure,
    setup_style,
    get_class_names_dict,
)

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    output_path: Path | str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a heatmap confusion matrix with counts and percentages."""
    setup_style()
    import seaborn as sns

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    else:
        fig = ax.get_figure()

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float), row_sums,
        out=np.zeros_like(cm, dtype=float), where=row_sums != 0,
    )

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    n_classes = cm.shape[0]
    if class_names is None:
        names_dict = get_class_names_dict(n_classes)
        class_names = [names_dict.get(i, str(i)) for i in range(n_classes)]

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    if output_path is not None:
        save_figure(fig, output_path)

    return fig, ax


def plot_cv_comparison_bar(
    cv_results: dict[str, dict[str, float]],
    metric_name: str = "Accuracy",
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Compare CV methods side by side as a bar chart."""
    setup_style()

    cv_methods = list(cv_results.keys())
    clf_names = list(next(iter(cv_results.values())).keys()) if cv_results else []
    n_methods = len(cv_methods)
    n_clfs = len(clf_names)

    if n_methods == 0:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(max(6, 2 * n_methods), 4))

    x = np.arange(n_methods)
    bar_width = 0.8 / max(n_clfs, 1)
    colors = ["#0173B2", "#DE8F05", "#029E73", "#CC3311", "#56B4E9", "#E69F00"]

    for i, clf_name in enumerate(clf_names):
        values = [cv_results[m].get(clf_name, 0) for m in cv_methods]
        offset = (i - (n_clfs - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, bar_width * 0.9, label=clf_name, color=colors[i % len(colors)])
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("CV Method")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} by CV Method and Classifier")
    ax.set_xticks(x)
    ax.set_xticklabels(cv_methods, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    if output_path is not None:
        save_figure(fig, output_path)

    return fig
