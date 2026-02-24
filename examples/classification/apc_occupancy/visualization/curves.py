"""ROC curve, DET curve, and confusion matrix visualization.

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
    """Plot ROC curve with AUC annotation and diagonal reference.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    model_name : str
        Label for the legend entry.
    output_path : Path or str, optional
        If provided, save the figure to this path.
    ax : plt.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axes
    auc_score : float
    """
    setup_style()
    from sklearn.metrics import auc

    fpr, tpr, _ = sklearn_roc_curve(y_true, y_prob)
    auc_score = float(auc(fpr, tpr))

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    else:
        fig = ax.get_figure()

    # Shaded area under the curve
    ax.fill_between(fpr, tpr, alpha=0.15, color=CLASS_COLORS[1])
    ax.plot(
        fpr, tpr,
        color=CLASS_COLORS[1], lw=2,
        label=f"{model_name} (AUC = {auc_score:.3f})",
    )

    # Diagonal reference
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
    """Map probabilities to normal-deviate (probit) scale.

    Clips to (1e-4, 1-1e-4) to avoid infinities.
    """
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
    """Plot DET curve (FNR vs FPR) on normal-deviate scale with EER point.

    Parameters
    ----------
    y_true, y_prob : np.ndarray
        Ground truth and predicted probabilities.
    eer : float, optional
        Pre-computed Equal Error Rate.
    eer_threshold : float, optional
        Threshold at EER.
    model_name : str
        Legend label.
    output_path, ax : optional
        Save path and/or axes to draw on.
    """
    setup_style()

    fpr, tpr, _ = sklearn_roc_curve(y_true, y_prob)
    fnr = 1.0 - tpr

    # Filter out zero values (log/probit scale)
    mask = (fpr > 0) & (fnr > 0)
    fpr_m, fnr_m = fpr[mask], fnr[mask]

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    else:
        fig = ax.get_figure()

    # DET curve on normal-deviate scale
    ax.plot(
        _normal_deviate(fpr_m), _normal_deviate(fnr_m),
        color=CLASS_COLORS[1], lw=2, label=model_name,
    )

    # EER point
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

    # Diagonal (FPR == FNR line)
    diag_vals = _normal_deviate(np.array([0.001, 0.5, 0.9]))
    ax.plot(diag_vals, diag_vals, ls="--", color="grey", lw=1)

    # Tick labels in probability space
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
# Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path | str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a heatmap confusion matrix with counts and percentages.

    Parameters
    ----------
    cm : np.ndarray
        Shape (2, 2) confusion matrix: [[TN, FP], [FN, TP]].
    output_path, ax : optional
        Save path and/or axes to draw on.
    """
    setup_style()
    import seaborn as sns

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    else:
        fig = ax.get_figure()

    # Normalized version for color intensity (safe division for empty rows)
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

    labels = [CLASS_NAMES.get(i, str(i)) for i in range(cm.shape[0])]

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
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
# Convenience: generate all evaluation plots
# ---------------------------------------------------------------------------

def plot_all_curves(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    metrics,
    output_dir: Path | str,
    model_name: str = "Model",
) -> None:
    """Generate ROC, DET, and confusion matrix plots and save to *output_dir*.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    y_prob : np.ndarray or None
        Predicted probabilities. ROC/DET are skipped when None.
    metrics : ClassificationMetrics
        Pre-computed metrics (used for EER, confusion matrix).
    output_dir : Path or str
        Directory to save plots.
    model_name : str
        Model name used in titles and filenames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.lower().replace(" ", "_")

    # Confusion matrix (always available)
    try:
        fig, _ = plot_confusion_matrix(
            metrics.confusion_matrix,
            output_path=output_dir / f"confusion_matrix_{safe_name}",
        )
        plt.close(fig)
    except Exception:
        logger.warning("Failed to plot confusion matrix for %s", model_name, exc_info=True)

    if y_prob is None:
        logger.info("Skipping ROC/DET curves for %s (no probability output)", model_name)
        return

    # ROC curve
    try:
        fig, _, _ = plot_roc_curve(
            y_true, y_prob,
            model_name=model_name,
            output_path=output_dir / f"roc_curve_{safe_name}",
        )
        plt.close(fig)
    except Exception:
        logger.warning("Failed to plot ROC curve for %s", model_name, exc_info=True)

    # DET curve
    try:
        eer = getattr(metrics, "eer", None)
        eer_thr = getattr(metrics, "eer_threshold", None)
        fig, _ = plot_det_curve(
            y_true, y_prob,
            eer=eer,
            eer_threshold=eer_thr,
            model_name=model_name,
            output_path=output_dir / f"det_curve_{safe_name}",
        )
        plt.close(fig)
    except Exception:
        logger.warning("Failed to plot DET curve for %s", model_name, exc_info=True)
