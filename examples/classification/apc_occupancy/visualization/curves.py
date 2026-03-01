"""ROC curve, DET curve, confusion matrix, sweep analysis visualization.

All plots follow the publication-quality style from ``style.py`` and
support optional save-to-disk in multiple formats.

Includes sweep-specific plots for Phase 1/1.5 analysis:
  - Score distribution (class-separated probability histograms)
  - Threshold sensitivity curves (metrics vs decision threshold)
  - Context performance curves (AUC/EER vs context window size)
  - Sweep bar charts (grouped bar for multi-factor comparisons)
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
    eer: float | None = None,
    eer_threshold: float | None = None,
    output_path: Path | str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, float]:
    """Plot ROC curve with AUC annotation, EER marker, and diagonal reference."""
    setup_style()
    from sklearn.metrics import auc

    fpr, tpr, thresholds = sklearn_roc_curve(y_true, y_prob)
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

    # Mark EER point on ROC curve
    if eer is not None and not np.isnan(eer):
        fnr = 1.0 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        ax.plot(
            fpr[eer_idx], tpr[eer_idx],
            marker="*", markersize=14, color=ACCENT_COLOR, zorder=5,
        )
        eer_label = f"EER = {eer:.3f}"
        if eer_threshold is not None and not np.isnan(eer_threshold):
            eer_label += f" (thr={eer_threshold:.3f})"
        ax.annotate(
            eer_label,
            xy=(fpr[eer_idx], tpr[eer_idx]),
            xytext=(20, -20),
            textcoords="offset points",
            fontsize=FONT_SIZE_ANNOTATION,
            color=ACCENT_COLOR,
            arrowprops=dict(arrowstyle="->", color=ACCENT_COLOR, lw=1),
        )

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

    # ROC / DET / Score distribution / Threshold sensitivity (binary only)
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    eer = getattr(metrics, "eer", None)
    eer_thr = getattr(metrics, "eer_threshold", None)

    if n_classes <= 2:
        try:
            fig, _, _ = plot_roc_curve(
                y_true, y_prob,
                model_name=model_name,
                eer=eer, eer_threshold=eer_thr,
                output_path=output_dir / f"roc_curve_{safe_name}",
            )
            plt.close(fig)
        except Exception:
            logger.warning("Failed to plot ROC curve for %s", model_name, exc_info=True)

        try:
            fig, _ = plot_det_curve(
                y_true, y_prob,
                eer=eer, eer_threshold=eer_thr,
                model_name=model_name,
                output_path=output_dir / f"det_curve_{safe_name}",
            )
            plt.close(fig)
        except Exception:
            logger.warning("Failed to plot DET curve for %s", model_name, exc_info=True)

        try:
            fig, _ = plot_score_distribution(
                y_true, y_prob,
                eer_threshold=eer_thr,
                class_names=class_names,
                model_name=model_name,
                output_path=output_dir / f"score_dist_{safe_name}",
            )
            plt.close(fig)
        except Exception:
            logger.warning("Failed to plot score distribution for %s", model_name, exc_info=True)

        try:
            fig, _ = plot_threshold_sensitivity(
                y_true, y_prob,
                eer_threshold=eer_thr,
                output_path=output_dir / f"threshold_sensitivity_{safe_name}",
            )
            plt.close(fig)
        except Exception:
            logger.warning("Failed to plot threshold sensitivity for %s", model_name, exc_info=True)


# ---------------------------------------------------------------------------
# Score Distribution (class-separated probability histogram)
# ---------------------------------------------------------------------------

def plot_score_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eer_threshold: float | None = None,
    class_names: list[str] | None = None,
    model_name: str = "Model",
    output_path: Path | str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot predicted probability distributions separated by class.

    Shows overlaid histograms with KDE for class 0 vs class 1,
    optionally marking the EER threshold as a vertical line.
    """
    setup_style()

    if class_names is None:
        class_names = [CLASS_NAMES.get(0, "Class 0"), CLASS_NAMES.get(1, "Class 1")]

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    else:
        fig = ax.get_figure()

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    for cls_idx in [0, 1]:
        mask = y_true == cls_idx
        if mask.sum() == 0:
            continue
        scores = y_prob[mask]
        color = CLASS_COLORS.get(cls_idx, f"C{cls_idx}")
        ax.hist(
            scores, bins=30, alpha=0.45, color=color,
            label=f"{class_names[cls_idx]} (n={mask.sum()})",
            density=True, edgecolor="white", linewidth=0.5,
        )
        # KDE overlay
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(scores, bw_method=0.2)
            x_grid = np.linspace(0, 1, 200)
            ax.plot(x_grid, kde(x_grid), color=color, lw=2)
        except (ImportError, np.linalg.LinAlgError):
            pass

    if eer_threshold is not None and not np.isnan(eer_threshold):
        ax.axvline(
            eer_threshold, color=ACCENT_COLOR, ls="--", lw=2,
            label=f"EER threshold = {eer_threshold:.3f}",
        )

    ax.set_xlabel("Predicted P(Occupied)")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution — {model_name}")
    ax.legend(loc="upper center", fontsize=8)
    ax.set_xlim(-0.05, 1.05)

    if output_path is not None:
        save_figure(fig, output_path)

    return fig, ax


# ---------------------------------------------------------------------------
# Threshold Sensitivity Curve
# ---------------------------------------------------------------------------

def plot_threshold_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eer_threshold: float | None = None,
    output_path: Path | str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot Accuracy/Precision/Recall/F1 as functions of decision threshold.

    Demonstrates why AUC/EER are preferred over fixed-threshold metrics
    for binary classification.
    """
    setup_style()
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.linspace(0.01, 0.99, 99)

    accs, precs, recs, f1s = [], [], [], []
    for thr in thresholds:
        y_pred_t = (y_prob >= thr).astype(int)
        accs.append(accuracy_score(y_true, y_pred_t))
        precs.append(precision_score(y_true, y_pred_t, zero_division=0))
        recs.append(recall_score(y_true, y_pred_t, zero_division=0))
        f1s.append(f1_score(y_true, y_pred_t, zero_division=0))

    ax.plot(thresholds, accs, color="#0173B2", lw=2, label="Accuracy")
    ax.plot(thresholds, precs, color="#009E73", lw=2, label="Precision")
    ax.plot(thresholds, recs, color="#CC3311", lw=2, label="Recall")
    ax.plot(thresholds, f1s, color="#DE8F05", lw=2, label="F1")

    if eer_threshold is not None and not np.isnan(eer_threshold):
        ax.axvline(
            eer_threshold, color="grey", ls="--", lw=1.5,
            label=f"EER thr = {eer_threshold:.3f}",
        )
    # Default 0.5 threshold
    ax.axvline(0.5, color="grey", ls=":", lw=1, alpha=0.5, label="Default 0.5")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Threshold Sensitivity")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    if output_path is not None:
        save_figure(fig, output_path)

    return fig, ax


# ---------------------------------------------------------------------------
# Context Performance Curve (AUC/EER vs context window size)
# ---------------------------------------------------------------------------

def plot_context_performance(
    context_mins: list[int],
    auc_values: list[float | None],
    eer_values: list[float | None] = None,
    classifier_groups: dict[str, tuple[list[int], list[float | None]]] | None = None,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Plot AUC (and optionally EER) as a function of context window size.

    Parameters
    ----------
    context_mins : list[int]
        Total context in minutes.
    auc_values : list[float | None]
        AUC values corresponding to each context size.
    eer_values : list[float | None], optional
        EER values (plotted on secondary y-axis).
    classifier_groups : dict, optional
        {classifier_name: (context_mins_list, auc_list)} for multi-classifier comparison.
    output_path : Path, optional
        Save path.
    """
    setup_style()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    colors = ["#0173B2", "#DE8F05", "#029E73", "#CC3311", "#9467BD", "#8C564B"]

    if classifier_groups is not None:
        for i, (clf_name, (ctx_list, auc_list)) in enumerate(classifier_groups.items()):
            ctx_arr = np.array(ctx_list)
            auc_arr = np.array([v if v is not None else np.nan for v in auc_list])
            valid = ~np.isnan(auc_arr)
            ax1.plot(
                ctx_arr[valid], auc_arr[valid],
                marker="o", markersize=4, lw=2,
                color=colors[i % len(colors)], label=clf_name,
            )
    else:
        ctx_arr = np.array(context_mins)
        auc_arr = np.array([v if v is not None else np.nan for v in auc_values])
        valid = ~np.isnan(auc_arr)
        ax1.plot(
            ctx_arr[valid], auc_arr[valid],
            marker="o", markersize=5, lw=2, color=colors[0], label="AUC",
        )

    ax1.set_xlabel("Context Window (minutes)")
    ax1.set_ylabel("AUC")
    ax1.set_title("Context Window vs Classification Performance")

    # EER on secondary y-axis
    if eer_values is not None:
        ax2 = ax1.twinx()
        eer_arr = np.array([v if v is not None else np.nan for v in eer_values])
        valid_e = ~np.isnan(eer_arr)
        if valid_e.any():
            ax2.plot(
                np.array(context_mins)[valid_e], eer_arr[valid_e],
                marker="s", markersize=4, lw=2, ls="--",
                color="#CC3311", alpha=0.7, label="EER",
            )
            ax2.set_ylabel("EER (lower=better)", color="#CC3311")
            ax2.tick_params(axis="y", labelcolor="#CC3311")
            ax2.legend(loc="upper right", fontsize=8)

    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    if output_path is not None:
        save_figure(fig, output_path)

    return fig


# ---------------------------------------------------------------------------
# Sweep Bar Chart (grouped bar for multi-factor comparisons)
# ---------------------------------------------------------------------------

def plot_sweep_bar(
    results_df,
    group_col: str,
    metric_col: str = "auc",
    hue_col: str | None = None,
    title: str = "Sweep Results",
    output_path: Path | str | None = None,
    top_k: int = 20,
) -> plt.Figure:
    """Plot grouped horizontal bar chart from sweep results DataFrame.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have `group_col` and `metric_col` columns.
    group_col : str
        Column to group by (x-axis categories).
    metric_col : str
        Metric to display (bar height).
    hue_col : str, optional
        Column to color-code bars (e.g. classifier).
    title : str
        Plot title.
    output_path : Path, optional
        Save path.
    top_k : int
        Maximum number of bars to show.
    """
    setup_style()
    import pandas as pd

    df = results_df.copy()
    if metric_col not in df.columns:
        logger.warning("Column '%s' not found in results", metric_col)
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    df = df.dropna(subset=[metric_col])
    df = df.sort_values(metric_col, ascending=False).head(top_k)

    n_bars = len(df)
    fig_height = max(4, 0.4 * n_bars)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    colors_list = ["#0173B2", "#DE8F05", "#029E73", "#CC3311", "#9467BD",
                   "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]

    if hue_col is not None and hue_col in df.columns:
        hue_vals = df[hue_col].unique()
        color_map = {v: colors_list[i % len(colors_list)] for i, v in enumerate(hue_vals)}
        bar_colors = [color_map[v] for v in df[hue_col]]
    else:
        bar_colors = colors_list[0]

    y_pos = np.arange(n_bars)
    labels = df[group_col].astype(str).tolist()
    values = df[metric_col].tolist()

    ax.barh(y_pos, values, color=bar_colors, height=0.7, edgecolor="white", linewidth=0.5)

    for i, (val, label) in enumerate(zip(values, labels)):
        ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(metric_col.upper())
    ax.set_title(title)

    if hue_col is not None and hue_col in df.columns:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map[v], label=v) for v in hue_vals
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    if output_path is not None:
        save_figure(fig, output_path)

    return fig
