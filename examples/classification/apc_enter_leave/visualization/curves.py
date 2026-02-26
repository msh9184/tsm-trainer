"""ROC curve, DET curve, confusion matrix, and CV comparison visualization.

All plots follow the publication-quality style from ``style.py`` and
support optional save-to-disk in multiple formats.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve as sklearn_roc_curve

from .style import (
    ACCENT_COLOR,
    CLASS_COLORS,
    CLASS_NAMES,
    FIGSIZE_SINGLE,
    FIGSIZE_WIDE,
    FONT_SIZE_ANNOTATION,
    save_figure,
    setup_style,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    output_path: Path | str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, float]:
    """Plot ROC curve with AUC annotation and diagonal reference."""
    setup_style()
    from sklearn.metrics import auc

    fpr, tpr, _ = sklearn_roc_curve(y_true, y_prob)
    auc_score = float(auc(fpr, tpr))

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    else:
        fig = ax.get_figure()

    ax.fill_between(fpr, tpr, alpha=0.15, color=CLASS_COLORS[0])
    ax.plot(
        fpr, tpr,
        color=CLASS_COLORS[0], lw=2,
        label=f"{model_name} (AUC = {auc_score:.3f})",
    )
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    if output_path is not None:
        save_figure(fig, output_path)

    return fig, ax, auc_score


# ---------------------------------------------------------------------------
# DET Curve
# ---------------------------------------------------------------------------

def _normal_deviate(p: np.ndarray) -> np.ndarray:
    """Map probabilities to normal-deviate (probit) scale."""
    from scipy.stats import norm

    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return norm.ppf(p)


def plot_det_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eer: float | None = None,
    eer_threshold: float | None = None,
    model_name: str = "Model",
    output_path: Path | str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot DET curve (FNR vs FPR) on normal-deviate scale with EER point."""
    setup_style()

    fpr, tpr, _ = sklearn_roc_curve(y_true, y_prob)
    fnr = 1.0 - tpr

    mask = (fpr > 0) & (fnr > 0)
    fpr_m, fnr_m = fpr[mask], fnr[mask]

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    else:
        fig = ax.get_figure()

    ax.plot(
        _normal_deviate(fpr_m), _normal_deviate(fnr_m),
        color=CLASS_COLORS[0], lw=2, label=model_name,
    )

    if eer is not None and not np.isnan(eer):
        eer_nd = _normal_deviate(np.array([eer]))[0]
        ax.plot(eer_nd, eer_nd, marker="*", markersize=12, color=ACCENT_COLOR, zorder=5)
        eer_label = f"EER = {eer:.3f}"
        if eer_threshold is not None:
            eer_label += f" (thr={eer_threshold:.3f})"
        ax.annotate(
            eer_label,
            xy=(eer_nd, eer_nd),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=FONT_SIZE_ANNOTATION,
            color=ACCENT_COLOR,
            arrowprops=dict(arrowstyle="->", color=ACCENT_COLOR, lw=1),
        )

    diag_vals = _normal_deviate(np.array([0.001, 0.5, 0.9]))
    ax.plot(diag_vals, diag_vals, ls="--", color="grey", lw=1)

    tick_vals = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    tick_labels = [f"{v:.0%}" for v in tick_vals]
    tick_nd = _normal_deviate(tick_vals)
    ax.set_xticks(tick_nd)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_nd)
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("False Negative Rate")
    ax.set_title("DET Curve")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")

    if output_path is not None:
        save_figure(fig, output_path)

    return fig, ax


# ---------------------------------------------------------------------------
# Confusion Matrix — supports 2x2 and NxN
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    output_path: Path | str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a heatmap confusion matrix with counts and percentages.

    Supports arbitrary NxN confusion matrices (binary or multiclass).
    """
    setup_style()
    import seaborn as sns

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    else:
        fig = ax.get_figure()

    # Normalized for color intensity
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float), row_sums,
        out=np.zeros_like(cm, dtype=float), where=row_sums != 0,
    )

    # Annotation: count + percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [CLASS_NAMES.get(i, str(i)) for i in range(n_classes)]

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


# ---------------------------------------------------------------------------
# CV comparison bar chart
# ---------------------------------------------------------------------------

def plot_cv_comparison_bar(
    cv_results: dict[str, dict[str, float]],
    metric_name: str = "AUC",
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Compare CV methods side by side as a bar chart.

    Parameters
    ----------
    cv_results : dict
        {cv_method_name: {classifier_name: metric_value}}.
    metric_name : str
        Name of the metric being plotted (for axis label).
    output_path : Path or str, optional
        Save path.
    """
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
    colors = ["#0173B2", "#DE8F05", "#029E73", "#CC3311"]

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


# ---------------------------------------------------------------------------
# All curves convenience
# ---------------------------------------------------------------------------

def plot_all_curves(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    metrics,
    output_dir: Path | str,
    model_name: str = "Model",
    class_names: list[str] | None = None,
) -> None:
    """Generate ROC, DET, and confusion matrix plots and save to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.lower().replace(" ", "_")

    # Confusion matrix
    try:
        fig, _ = plot_confusion_matrix(
            metrics.confusion_matrix,
            class_names=class_names,
            output_path=output_dir / f"confusion_matrix_{safe_name}",
        )
        plt.close(fig)
    except Exception:
        logger.warning("Failed to plot confusion matrix for %s", model_name, exc_info=True)

    if y_prob is None:
        logger.info("Skipping ROC/DET curves for %s (no probability output)", model_name)
        return

    # ROC curve (binary only)
    n_classes = len(np.unique(y_true))
    if n_classes <= 2:
        try:
            fig, _, _ = plot_roc_curve(
                y_true, y_prob,
                model_name=model_name,
                output_path=output_dir / f"roc_curve_{safe_name}",
            )
            plt.close(fig)
        except Exception:
            logger.warning("Failed to plot ROC curve for %s", model_name, exc_info=True)

        try:
            eer = getattr(metrics, "eer", None)
            eer_thr = getattr(metrics, "eer_threshold", None)
            fig, _ = plot_det_curve(
                y_true, y_prob,
                eer=eer, eer_threshold=eer_thr,
                model_name=model_name,
                output_path=output_dir / f"det_curve_{safe_name}",
            )
            plt.close(fig)
        except Exception:
            logger.warning("Failed to plot DET curve for %s", model_name, exc_info=True)
