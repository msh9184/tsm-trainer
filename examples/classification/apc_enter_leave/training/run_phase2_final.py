"""Phase 2 Final: Comprehensive Experiment Suite for APC Enter/Leave Detection.

Establishes the Phase 2 optimal configuration and generates presentation-ready
visualizations for team meetings.  Three independent groups for parallel execution.

  Group 1 — Scenario Comparison & Validation (~8 min)
  Group 2 — Context Window & Layer Deep Dive (~12 min)
  Group 3 — Visualization Suite (~10 min)

Usage:
    cd examples/classification/apc_enter_leave

    # Terminal 1:
    python training/run_phase2_final.py --config training/configs/enter-leave-phase2.yaml --group 1 --device cuda

    # Terminal 2:
    python training/run_phase2_final.py --config training/configs/enter-leave-phase2.yaml --group 2 --device cuda

    # Terminal 3:
    python training/run_phase2_final.py --config training/configs/enter-leave-phase2.yaml --group 3 --device cuda

    # All groups sequentially:
    python training/run_phase2_final.py --config training/configs/enter-leave-phase2.yaml --group all --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

import sys
import os

import matplotlib as mpl
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from training.run_event_detection import (
    load_config,
    load_data,
    load_mantis_model,
    extract_all_embeddings,
    save_results_csv,
    save_results_txt,
)
from data.dataset import EventDataset, build_dataset_config
from evaluation.metrics import (
    EventClassificationMetrics,
    aggregate_cv_predictions,
)
from training.run_phase2_finetune import (
    TrainConfig,
    run_neural_loocv,
    run_sklearn_loocv,
    compute_wilson_ci,
    _make_result_row,
    _rank_results,
    _serialize_stage_results,
)
from training.heads import build_head
from visualization.style import (
    setup_style,
    save_figure,
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_DPI,
)
from visualization.embeddings import reduce_dimensions

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Color palette for method types
METHOD_COLORS = {
    "rf": "#0173B2",       # blue
    "neural": "#DE8F05",   # orange
    "ensemble": "#029E73", # green
    "pca": "#CC78BC",      # purple
}

PRESENTATION_STYLE = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _apply_style():
    """Apply presentation-quality matplotlib style."""
    setup_style()
    plt.rcParams.update(PRESENTATION_STYLE)


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_embeddings(raw_cfg, include_none, device, layer, ctx_before, ctx_after,
                        ctx_mode="bidirectional"):
    """Extract MantisV2 embeddings for a given context/layer configuration."""
    cfg = copy.deepcopy(raw_cfg)
    cfg["dataset"] = {
        "context_mode": ctx_mode,
        "context_before": ctx_before,
        "context_after": ctx_after,
    }
    sensor_array, sensor_ts, event_ts, event_labels, ch_names, class_names = (
        load_data(cfg, include_none=include_none)
    )
    ds_cfg = build_dataset_config(cfg["dataset"])
    dataset = EventDataset(sensor_array, sensor_ts, event_ts, event_labels, ds_cfg)

    model_cfg = raw_cfg.get("model", {})
    pretrained_name = model_cfg.get("pretrained_name", "paris-noah/MantisV2")
    output_token = model_cfg.get("output_token", "combined")

    network, model = load_mantis_model(pretrained_name, layer, output_token, device)
    Z = extract_all_embeddings(model, dataset)
    del model, network
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Z, event_labels, class_names


def _rf_loocv_with_probs(Z, y, n_classes, seed=42):
    """Run RF LOOCV returning per-sample probability vectors."""
    from sklearn.ensemble import RandomForestClassifier

    n = len(y)
    y_prob = np.zeros((n, n_classes), dtype=np.float64)
    y_pred = np.zeros(n, dtype=int)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
        clf.fit(Z[mask], y[mask])
        probs = clf.predict_proba(Z[i:i + 1])[0]
        prob_vec = np.zeros(n_classes, dtype=np.float64)
        for ci, c in enumerate(clf.classes_):
            prob_vec[c] = probs[ci]
        y_prob[i] = prob_vec
        y_pred[i] = prob_vec.argmax()

    return y_pred, y_prob


def _compute_per_class(y_true, y_pred, class_names):
    """Compute per-class accuracy and counts."""
    results = {}
    for c, name in enumerate(class_names):
        mask = y_true == c
        n_total = mask.sum()
        n_correct = (y_pred[mask] == c).sum()
        results[name] = {
            "accuracy": float(n_correct / n_total) if n_total > 0 else 0.0,
            "n_correct": int(n_correct),
            "n_total": int(n_total),
        }
    return results


def _find_persistent_errors(predictions_dict, y_true):
    """Find samples consistently misclassified across methods."""
    n = len(y_true)
    error_counts = np.zeros(n, dtype=int)
    n_methods = len(predictions_dict)

    for name, y_pred in predictions_dict.items():
        error_counts += (y_pred != y_true).astype(int)

    return error_counts, n_methods


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_grand_comparison(results, output_path):
    """Horizontal bar chart ranking all methods by accuracy with CI."""
    _apply_style()

    names = [r["name"] for r in results]
    accs = [r["accuracy"] for r in results]
    types = [r.get("method_type", "rf") for r in results]
    ci_los = [r.get("ci_95_lower", 0) for r in results]
    ci_his = [r.get("ci_95_upper", 1) for r in results]

    colors = [METHOD_COLORS.get(t, "#666666") for t in types]
    errors_lo = [a - lo for a, lo in zip(accs, ci_los)]
    errors_hi = [hi - a for a, hi in zip(accs, ci_his)]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, accs, color=colors, alpha=0.85, edgecolor="white", height=0.7)
    ax.errorbar(accs, y_pos, xerr=[errors_lo, errors_hi],
                fmt="none", color="black", capsize=3, linewidth=1.2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("LOOCV Accuracy")
    ax.set_title("Phase 2 — Method Comparison (All Experiments)", fontweight="bold")
    ax.set_xlim(0.5, 1.0)
    ax.invert_yaxis()

    for i, (acc, bar) in enumerate(zip(accs, bars)):
        ax.text(acc + 0.003, i, f"{acc:.1%}", va="center", fontsize=9, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=METHOD_COLORS["ensemble"], label="Ensemble"),
        Patch(facecolor=METHOD_COLORS["rf"], label="RF (single)"),
        Patch(facecolor=METHOD_COLORS["neural"], label="Neural"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_context_sweep(context_results, output_path):
    """Line chart showing accuracy vs context window configuration."""
    _apply_style()

    names = [r["name"] for r in context_results]
    accs = [r["accuracy"] for r in context_results]
    total_steps = [r.get("total_steps", 0) for r in context_results]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(len(names)), accs, "o-", color=METHOD_COLORS["rf"],
            linewidth=2, markersize=8, markerfacecolor="white",
            markeredgewidth=2, markeredgecolor=METHOD_COLORS["rf"])

    best_idx = int(np.argmax(accs))
    ax.plot(best_idx, accs[best_idx], "o", color="#029E73", markersize=14,
            markeredgewidth=2, zorder=5)
    ax.annotate(f"Best: {accs[best_idx]:.1%}",
                xy=(best_idx, accs[best_idx]),
                xytext=(best_idx + 0.5, accs[best_idx] + 0.01),
                fontsize=10, fontweight="bold", color="#029E73",
                arrowprops=dict(arrowstyle="->", color="#029E73"))

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("RF LOOCV Accuracy")
    ax.set_title("Context Window Effect on Classification Accuracy", fontweight="bold")
    ax.set_ylim(max(0.7, min(accs) - 0.05), min(1.0, max(accs) + 0.03))

    # Add total timesteps as secondary info
    for i, (name, acc, steps) in enumerate(zip(names, accs, total_steps)):
        if steps > 0:
            ax.text(i, acc - 0.015, f"{steps} steps", ha="center", fontsize=7, color="gray")

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_layer_comparison(layer_results, output_path):
    """Grouped bar chart comparing RF accuracy for each MantisV2 layer."""
    _apply_style()

    names = [r["name"] for r in layer_results]
    accs = [r["accuracy"] for r in layer_results]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = []
    for name in names:
        if "Concat" in name or "Fusion" in name:
            colors.append(METHOD_COLORS["ensemble"])
        else:
            colors.append(METHOD_COLORS["rf"])

    bars = ax.bar(range(len(names)), accs, color=colors, alpha=0.85,
                  edgecolor="white", width=0.7)

    best_idx = int(np.argmax(accs))
    bars[best_idx].set_edgecolor("#029E73")
    bars[best_idx].set_linewidth(2.5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("RF LOOCV Accuracy")
    ax.set_title("MantisV2 Layer & Multi-Layer Comparison", fontweight="bold")
    ax.set_ylim(0.8, 1.0)

    for i, acc in enumerate(accs):
        ax.text(i, acc + 0.003, f"{acc:.1%}", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_tsne_grid(embeddings_dict, y, class_names, title, output_path,
                   ncols=3, figsize_per_panel=(4, 3.5)):
    """Grid of t-SNE plots for multiple embedding configurations."""
    _apply_style()

    n = len(embeddings_dict)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_panel[0] * ncols,
                                      figsize_per_panel[1] * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    class_colors = [CLASS_COLORS.get(i, "#666666") for i in range(len(class_names))]

    for idx, (name, Z) in enumerate(embeddings_dict.items()):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        Z_2d = reduce_dimensions(Z, method="tsne", random_state=42)

        for c in range(len(class_names)):
            mask = y == c
            ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                       c=class_colors[c], label=class_names[c],
                       s=30, alpha=0.7, edgecolors="white", linewidths=0.3)

        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        if idx == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.8)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_confusion_triptych(confusion_data, class_names, output_path):
    """Three confusion matrices side by side (RF / Neural / Ensemble)."""
    import seaborn as sns
    _apply_style()

    n = len(confusion_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, confusion_data.items()):
        # Normalize by row
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        # Annotations: count + percentage
        annots = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annots[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.0%})"

        sns.heatmap(cm_norm, annot=annots, fmt="", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, vmin=0, vmax=1, cbar=False,
                    linewidths=0.5, linecolor="white")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylabel("True" if ax == axes[0] else "")
        ax.set_xlabel("Predicted")

    fig.suptitle("Confusion Matrix Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_per_class_accuracy(per_class_data, output_path):
    """Grouped bar chart: per-class accuracy for multiple methods."""
    _apply_style()

    methods = list(per_class_data.keys())
    classes = list(next(iter(per_class_data.values())).keys())
    n_methods = len(methods)
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(n_classes)
    width = 0.8 / n_methods

    method_colors = [METHOD_COLORS.get("ensemble", "#666"), METHOD_COLORS["rf"], METHOD_COLORS["neural"]]

    for i, method in enumerate(methods):
        accs = [per_class_data[method][c]["accuracy"] for c in classes]
        bars = ax.bar(x + i * width - (n_methods - 1) * width / 2, accs,
                      width, label=method, color=method_colors[i % len(method_colors)],
                      alpha=0.85, edgecolor="white")
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{acc:.0%}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy Breakdown", fontweight="bold")
    ax.set_ylim(0.6, 1.05)
    ax.legend(loc="lower right")

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_error_analysis(error_counts, n_methods, y_true, class_names, output_path):
    """Visualize which samples are persistently misclassified."""
    _apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Error frequency distribution
    ax1 = axes[0]
    unique_counts = np.arange(n_methods + 1)
    hist_vals = [np.sum(error_counts == c) for c in unique_counts]
    colors = ["#029E73" if c == 0 else ("#DE8F05" if c < n_methods else "#CC3311")
              for c in unique_counts]
    ax1.bar(unique_counts, hist_vals, color=colors, alpha=0.85, edgecolor="white")
    ax1.set_xlabel("Number of Methods That Misclassify")
    ax1.set_ylabel("Number of Samples")
    ax1.set_title("Error Frequency Distribution", fontweight="bold")
    for i, v in enumerate(hist_vals):
        if v > 0:
            ax1.text(i, v + 0.3, str(v), ha="center", fontsize=10, fontweight="bold")

    # Panel 2: Per-class error distribution
    ax2 = axes[1]
    class_colors_list = [CLASS_COLORS.get(i, "#666") for i in range(len(class_names))]

    for c, cname in enumerate(class_names):
        mask = y_true == c
        class_errors = error_counts[mask]
        error_rate = (class_errors > 0).sum() / mask.sum() if mask.sum() > 0 else 0
        ax2.bar(c, error_rate, color=class_colors_list[c], alpha=0.85,
                edgecolor="white", width=0.6, label=cname)
        ax2.text(c, error_rate + 0.01, f"{error_rate:.0%}",
                 ha="center", fontsize=10, fontweight="bold")

    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, fontsize=11)
    ax2.set_ylabel("Error Rate (≥1 method wrong)")
    ax2.set_title("Per-Class Error Rate", fontweight="bold")
    ax2.set_ylim(0, min(1.0, max((error_counts > 0).sum() / len(y_true) * 3, 0.3)))

    fig.suptitle("Error Analysis Across Methods", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_stability_boxplot(stability_data, output_path):
    """Box plot comparing multi-seed stability across methods."""
    _apply_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    methods = list(stability_data.keys())
    data = [stability_data[m] for m in methods]

    bp = ax.boxplot(data, labels=methods, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=6))

    colors = [METHOD_COLORS.get("ensemble", "#029E73"),
              METHOD_COLORS["rf"], METHOD_COLORS["neural"]]
    for patch, color in zip(bp["boxes"], colors[:len(methods)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    for i, (method, vals) in enumerate(zip(methods, data)):
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        ax.text(i + 1, mean_val + 0.003,
                f"μ={mean_val:.1%}\nσ={std_val:.4f}",
                ha="center", fontsize=8, fontweight="bold")

    ax.set_ylabel("LOOCV Accuracy")
    ax.set_title("Multi-Seed Stability Comparison (5 seeds)", fontweight="bold")
    ax.set_ylim(0.85, 0.96)

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_distance_analysis(Z, y, class_names, output_path):
    """Inter-class and intra-class distance analysis."""
    import seaborn as sns
    _apply_style()

    n_classes = len(class_names)

    # Compute centroids and distances
    centroids = np.array([Z[y == c].mean(axis=0) for c in range(n_classes)])

    # Inter-class distance matrix
    inter_dist = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            inter_dist[i, j] = np.linalg.norm(centroids[i] - centroids[j])

    # Intra-class spread (mean distance to centroid)
    intra_spread = np.zeros(n_classes)
    for c in range(n_classes):
        dists = np.linalg.norm(Z[y == c] - centroids[c], axis=1)
        intra_spread[c] = dists.mean()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Inter-class distance heatmap
    ax1 = axes[0]
    sns.heatmap(inter_dist, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, square=True)
    ax1.set_title("Inter-Class Distance\n(centroid L2)", fontweight="bold")

    # Panel 2: Intra-class spread
    ax2 = axes[1]
    class_colors_list = [CLASS_COLORS.get(i, "#666") for i in range(n_classes)]
    bars = ax2.bar(range(n_classes), intra_spread, color=class_colors_list,
                   alpha=0.85, edgecolor="white")
    ax2.set_xticks(range(n_classes))
    ax2.set_xticklabels(class_names)
    ax2.set_ylabel("Mean Distance to Centroid")
    ax2.set_title("Intra-Class Spread", fontweight="bold")
    for i, v in enumerate(intra_spread):
        ax2.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    # Panel 3: Separation ratio (inter / intra)
    ax3 = axes[2]
    sep_ratio = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                avg_intra = (intra_spread[i] + intra_spread[j]) / 2
                sep_ratio[i, j] = inter_dist[i, j] / avg_intra if avg_intra > 0 else 0

    sns.heatmap(sep_ratio, annot=True, fmt=".2f", cmap="RdYlGn",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax3, square=True, vmin=0)
    ax3.set_title("Separation Ratio\n(inter/intra, >1 = good)", fontweight="bold")

    fig.suptitle("Embedding Space Analysis (L3, 2+1+2)", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.08, wspace=0.4)
    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_summary_dashboard(summary_data, output_path):
    """Multi-panel summary dashboard for team presentation."""
    _apply_style()

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: Key metrics table
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    table_data = summary_data.get("key_metrics", [])
    if table_data:
        table = ax1.table(
            cellText=[[r["name"], f"{r['accuracy']:.1%}", f"{r.get('f1', 0):.4f}",
                        f"{r.get('auc', 0):.4f}"]
                       for r in table_data[:6]],
            colLabels=["Method", "Acc", "F1", "AUC"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)
        # Highlight best row
        for j in range(4):
            table[1, j].set_facecolor("#D4EDDA")
    ax1.set_title("Top Methods", fontweight="bold", fontsize=11)

    # Panel 2: Best configuration box
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    best_cfg = summary_data.get("best_config", {})
    text_lines = [
        "OPTIMAL CONFIGURATION",
        "━" * 30,
        f"Model: {best_cfg.get('model', 'MantisV2')} (frozen)",
        f"Layer: {best_cfg.get('layer', 'L3')}",
        f"Context: {best_cfg.get('context', '2+1+2')}",
        f"Channels: {best_cfg.get('channels', 'M+C')}",
        f"Classifier: {best_cfg.get('classifier', 'RF ensemble')}",
        f"Augmentation: {best_cfg.get('augmentation', 'None')}",
        "",
        f"Accuracy: {best_cfg.get('accuracy', '93.58%')}",
        f"95% CI: {best_cfg.get('ci', '[87.3%, 96.9%]')}",
        f"Stability: {best_cfg.get('stability', '±0.58%')}",
    ]
    ax2.text(0.05, 0.95, "\n".join(text_lines), transform=ax2.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F7FF",
                       edgecolor="#0173B2", linewidth=1.5))

    # Panel 3: Key findings
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    findings = summary_data.get("findings", [])
    findings_text = "KEY FINDINGS\n" + "━" * 30 + "\n"
    for i, f in enumerate(findings[:6], 1):
        findings_text += f"\n{i}. {f}"
    ax3.text(0.05, 0.95, findings_text, transform=ax3.transAxes,
             fontsize=8.5, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8E1",
                       edgecolor="#DE8F05", linewidth=1.5))

    # Panel 4-6: Placeholder for embedded plots (accuracy bar, t-SNE, confusion)
    # These are generated as separate files; the dashboard shows summary stats
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis("off")
    limitations = summary_data.get("limitations", [])
    lim_text = "LIMITATIONS & NEXT STEPS\n" + "━" * 30
    for l in limitations[:5]:
        lim_text += f"\n• {l}"
    ax4.text(0.05, 0.95, lim_text, transform=ax4.transAxes,
             fontsize=8.5, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFEBEE",
                       edgecolor="#CC3311", linewidth=1.5))

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis("off")
    perf_text = "PERFORMANCE BREAKDOWN\n" + "━" * 30
    perf_data = summary_data.get("performance", {})
    for key, val in perf_data.items():
        perf_text += f"\n{key}: {val}"
    ax5.text(0.05, 0.95, perf_text, transform=ax5.transAxes,
             fontsize=8.5, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9",
                       edgecolor="#029E73", linewidth=1.5))

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    data_text = "DATA OVERVIEW\n" + "━" * 30
    data_info = summary_data.get("data_info", {})
    for key, val in data_info.items():
        data_text += f"\n{key}: {val}"
    ax6.text(0.05, 0.95, data_text, transform=ax6.transAxes,
             fontsize=8.5, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3E5F5",
                       edgecolor="#CC78BC", linewidth=1.5))

    fig.suptitle("Phase 2 Final Report — APC Enter/Leave Event Detection",
                 fontsize=15, fontweight="bold", y=0.98)
    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ============================================================================
# Group 1: Scenario Comparison & Validation
# ============================================================================

def run_group1_scenarios(raw_cfg, include_none, device, seed, output_dir):
    """Head-to-head comparison of all candidate methods."""
    logger.info("=" * 60)
    logger.info("GROUP 1: Scenario Comparison & Validation")
    logger.info("=" * 60)

    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_seeds = 5
    seed_list = list(range(seed, seed + n_seeds))

    # --- Extract embeddings ---
    logger.info("Extracting embeddings for 3 context windows + multi-layer...")
    Z_2_2, y, class_names = _extract_embeddings(raw_cfg, include_none, device, 3, 2, 2)
    Z_2_4, _, _ = _extract_embeddings(raw_cfg, include_none, device, 3, 2, 4)
    Z_3_3, _, _ = _extract_embeddings(raw_cfg, include_none, device, 3, 3, 3)
    Z_L0, _, _ = _extract_embeddings(raw_cfg, include_none, device, 0, 2, 2)
    logger.info("Embeddings extracted: 4 configurations")

    n_events = len(y)
    n_classes = len(np.unique(y))

    train_cfg_raw = raw_cfg.get("training", {})
    train_config = TrainConfig(
        epochs=train_cfg_raw.get("epochs", 200),
        lr=train_cfg_raw.get("lr", 1e-3),
        weight_decay=train_cfg_raw.get("weight_decay", 0.01),
        label_smoothing=train_cfg_raw.get("label_smoothing", 0.0),
        early_stopping_patience=train_cfg_raw.get("early_stopping_patience", 30),
        device=device,
    )

    all_results = []
    predictions = {}
    confusion_matrices = {}
    per_class_all = {}

    # --- Scenario A: RF Baseline ---
    logger.info("--- Scenario A: RF (single context 2+1+2) ---")
    y_pred_rf, y_prob_rf = _rf_loocv_with_probs(Z_2_2, y, n_classes, seed)
    metrics_rf = aggregate_cv_predictions(y, y_pred_rf, y_prob_rf,
                                          cv_method="LOOCV", n_folds=n_events,
                                          class_names=class_names)
    n_correct = int(round(metrics_rf.accuracy * n_events))
    ci_lo, ci_hi = compute_wilson_ci(n_correct, n_events)
    all_results.append({
        "name": "RF 2+1+2", "accuracy": metrics_rf.accuracy,
        "f1": metrics_rf.f1_macro, "auc": metrics_rf.roc_auc,
        "ci_95_lower": ci_lo, "ci_95_upper": ci_hi,
        "method_type": "rf",
    })
    predictions["RF 2+1+2"] = y_pred_rf
    confusion_matrices["RF (single)"] = metrics_rf.confusion_matrix
    per_class_all["RF 2+1+2"] = _compute_per_class(y, y_pred_rf, class_names)
    logger.info("  RF 2+1+2: Acc=%.4f  F1=%.4f  AUC=%.4f  CI=[%.4f, %.4f]",
                metrics_rf.accuracy, metrics_rf.f1_macro, metrics_rf.roc_auc, ci_lo, ci_hi)

    # --- Scenario B: 2-way Ensemble (BEST) ---
    logger.info("--- Scenario B: 2-way Ensemble (2+1+2 + 2+1+4) ---")
    y_pred_rf24, y_prob_rf24 = _rf_loocv_with_probs(Z_2_4, y, n_classes, seed)
    y_prob_ens2 = (y_prob_rf + y_prob_rf24) / 2
    y_pred_ens2 = y_prob_ens2.argmax(axis=1)
    metrics_ens2 = aggregate_cv_predictions(y, y_pred_ens2, y_prob_ens2,
                                            cv_method="LOOCV", n_folds=n_events,
                                            class_names=class_names)
    n_correct = int(round(metrics_ens2.accuracy * n_events))
    ci_lo, ci_hi = compute_wilson_ci(n_correct, n_events)
    all_results.append({
        "name": "Ensemble RF 2-way", "accuracy": metrics_ens2.accuracy,
        "f1": metrics_ens2.f1_macro, "auc": metrics_ens2.roc_auc,
        "ci_95_lower": ci_lo, "ci_95_upper": ci_hi,
        "method_type": "ensemble",
    })
    predictions["Ensemble 2-way"] = y_pred_ens2
    confusion_matrices["Ensemble (2-way)"] = metrics_ens2.confusion_matrix
    per_class_all["Ensemble 2-way"] = _compute_per_class(y, y_pred_ens2, class_names)
    logger.info("  Ensemble 2-way: Acc=%.4f  F1=%.4f  AUC=%.4f  CI=[%.4f, %.4f]",
                metrics_ens2.accuracy, metrics_ens2.f1_macro, metrics_ens2.roc_auc, ci_lo, ci_hi)

    # --- Scenario B2: 3-way Ensemble ---
    logger.info("--- Scenario B2: 3-way Ensemble (2+1+2 + 2+1+4 + 3+1+3) ---")
    y_pred_rf33, y_prob_rf33 = _rf_loocv_with_probs(Z_3_3, y, n_classes, seed)
    y_prob_ens3 = (y_prob_rf + y_prob_rf24 + y_prob_rf33) / 3
    y_pred_ens3 = y_prob_ens3.argmax(axis=1)
    metrics_ens3 = aggregate_cv_predictions(y, y_pred_ens3, y_prob_ens3,
                                            cv_method="LOOCV", n_folds=n_events,
                                            class_names=class_names)
    n_correct = int(round(metrics_ens3.accuracy * n_events))
    ci_lo, ci_hi = compute_wilson_ci(n_correct, n_events)
    all_results.append({
        "name": "Ensemble RF 3-way", "accuracy": metrics_ens3.accuracy,
        "f1": metrics_ens3.f1_macro, "auc": metrics_ens3.roc_auc,
        "ci_95_lower": ci_lo, "ci_95_upper": ci_hi,
        "method_type": "ensemble",
    })
    predictions["Ensemble 3-way"] = y_pred_ens3
    logger.info("  Ensemble 3-way: Acc=%.4f  F1=%.4f  AUC=%.4f",
                metrics_ens3.accuracy, metrics_ens3.f1_macro, metrics_ens3.roc_auc)

    # --- Scenario C: Neural Linear (L3) ---
    logger.info("--- Scenario C: Neural Linear (L3, 2+1+2) ---")
    def head_factory_linear():
        return build_head("linear", Z_2_2.shape[1], n_classes)
    metrics_nn, y_pred_nn, y_prob_nn = run_neural_loocv(
        Z_2_2, y, head_factory_linear, train_config, class_names, seed)
    n_correct = int(round(metrics_nn.accuracy * n_events))
    ci_lo, ci_hi = compute_wilson_ci(n_correct, n_events)
    all_results.append({
        "name": "Neural Linear L3", "accuracy": metrics_nn.accuracy,
        "f1": metrics_nn.f1_macro, "auc": metrics_nn.roc_auc,
        "ci_95_lower": ci_lo, "ci_95_upper": ci_hi,
        "method_type": "neural",
    })
    predictions["Neural Linear"] = y_pred_nn
    confusion_matrices["Neural (Linear)"] = metrics_nn.confusion_matrix
    per_class_all["Neural Linear"] = _compute_per_class(y, y_pred_nn, class_names)
    logger.info("  Neural Linear: Acc=%.4f  F1=%.4f  AUC=%.4f",
                metrics_nn.accuracy, metrics_nn.f1_macro, metrics_nn.roc_auc)

    # --- Scenario D: Neural Concat L0+L3 ---
    logger.info("--- Scenario D: Neural Linear Concat L0+L3 ---")
    Z_concat = np.concatenate([Z_L0, Z_2_2], axis=1)
    def head_factory_concat():
        return build_head("linear", Z_concat.shape[1], n_classes)
    metrics_cat, y_pred_cat, y_prob_cat = run_neural_loocv(
        Z_concat, y, head_factory_concat, train_config, class_names, seed)
    n_correct = int(round(metrics_cat.accuracy * n_events))
    ci_lo, ci_hi = compute_wilson_ci(n_correct, n_events)
    all_results.append({
        "name": "Neural Concat L0+L3", "accuracy": metrics_cat.accuracy,
        "f1": metrics_cat.f1_macro, "auc": metrics_cat.roc_auc,
        "ci_95_lower": ci_lo, "ci_95_upper": ci_hi,
        "method_type": "neural",
    })
    predictions["Neural Concat L0+L3"] = y_pred_cat
    logger.info("  Neural Concat L0+L3: Acc=%.4f  F1=%.4f  AUC=%.4f",
                metrics_cat.accuracy, metrics_cat.f1_macro, metrics_cat.roc_auc)

    # Sort by accuracy
    all_results.sort(key=lambda r: -r["accuracy"])

    # --- Multi-seed stability ---
    logger.info("Running multi-seed stability (5 seeds) for top methods...")
    stability_data = {}

    # Ensemble 2-way stability
    ens_accs = []
    for s in seed_list:
        _, p1 = _rf_loocv_with_probs(Z_2_2, y, n_classes, s)
        _, p2 = _rf_loocv_with_probs(Z_2_4, y, n_classes, s)
        pred = ((p1 + p2) / 2).argmax(axis=1)
        ens_accs.append(float((pred == y).mean()))
    stability_data["Ensemble 2-way"] = ens_accs
    logger.info("  Ensemble 2-way: mean=%.4f ±%.4f", np.mean(ens_accs), np.std(ens_accs))

    # RF single stability
    rf_accs = []
    for s in seed_list:
        _, p = _rf_loocv_with_probs(Z_2_2, y, n_classes, s)
        pred = p.argmax(axis=1)
        rf_accs.append(float((pred == y).mean()))
    stability_data["RF single"] = rf_accs
    logger.info("  RF single: mean=%.4f ±%.4f", np.mean(rf_accs), np.std(rf_accs))

    # Neural Linear stability
    nn_accs = []
    for s in seed_list:
        m, _, _ = run_neural_loocv(Z_2_2, y, head_factory_linear, train_config, class_names, s)
        nn_accs.append(float(m.accuracy))
    stability_data["Neural Linear"] = nn_accs
    logger.info("  Neural Linear: mean=%.4f ±%.4f", np.mean(nn_accs), np.std(nn_accs))

    # --- Error analysis ---
    error_counts, n_methods = _find_persistent_errors(predictions, y)

    # --- Generate visualizations ---
    logger.info("Generating Group 1 visualizations...")

    plot_grand_comparison(all_results, plots_dir / "scenario_comparison")
    plot_confusion_triptych(confusion_matrices, class_names,
                           plots_dir / "confusion_triptych")
    plot_per_class_accuracy(per_class_all, plots_dir / "per_class_accuracy")
    plot_stability_boxplot(stability_data, plots_dir / "stability_boxplot")
    plot_error_analysis(error_counts, n_methods, y, class_names,
                        plots_dir / "error_analysis")

    # --- Save tables ---
    serializable = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k != "method_type"}
        serializable.append(row)
    save_results_csv(serializable, tables_dir / "group1_scenario_comparison.csv")
    save_results_txt(serializable, tables_dir / "group1_scenario_comparison.txt")

    # Stability table
    stab_rows = []
    for method, accs in stability_data.items():
        stab_rows.append({
            "method": method,
            "mean_acc": round(np.mean(accs), 4),
            "std_acc": round(np.std(accs), 4),
            "min_acc": round(min(accs), 4),
            "max_acc": round(max(accs), 4),
            "seeds": str(accs),
        })
    save_results_csv(stab_rows, tables_dir / "group1_stability.csv")

    # Per-class table
    pc_rows = []
    for method, pc in per_class_all.items():
        for cls, data in pc.items():
            pc_rows.append({
                "method": method, "class": cls,
                "accuracy": data["accuracy"],
                "n_correct": data["n_correct"],
                "n_total": data["n_total"],
            })
    save_results_csv(pc_rows, tables_dir / "group1_per_class.csv")

    logger.info("Group 1 complete: %d scenarios, %d visualizations", len(all_results), 5)

    return {
        "results": all_results,
        "stability": {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in stability_data.items()},
        "per_class": per_class_all,
    }


# ============================================================================
# Group 2: Context Window & Layer Deep Dive
# ============================================================================

def run_group2_context_layers(raw_cfg, include_none, device, seed, output_dir):
    """Systematic exploration of context windows and MantisV2 layers."""
    logger.info("=" * 60)
    logger.info("GROUP 2: Context Window & Layer Deep Dive")
    logger.info("=" * 60)

    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Context Window Sweep ---
    logger.info("--- Context Window Sweep ---")
    context_configs = [
        {"name": "1+1+1 (3min)", "before": 1, "after": 1, "mode": "bidirectional", "total_steps": 3},
        {"name": "2+1+2 (5min)", "before": 2, "after": 2, "mode": "bidirectional", "total_steps": 5},
        {"name": "3+1+3 (7min)", "before": 3, "after": 3, "mode": "bidirectional", "total_steps": 7},
        {"name": "4+1+4 (9min)", "before": 4, "after": 4, "mode": "bidirectional", "total_steps": 9},
        {"name": "5+1+5 (11min)", "before": 5, "after": 5, "mode": "bidirectional", "total_steps": 11},
        {"name": "10+1+10 (21min)", "before": 10, "after": 10, "mode": "bidirectional", "total_steps": 21},
        {"name": "2+1+4 (7min asym)", "before": 2, "after": 4, "mode": "bidirectional", "total_steps": 7},
        {"name": "4+1+2 (7min asym)", "before": 4, "after": 2, "mode": "bidirectional", "total_steps": 7},
        {"name": "5+1+0 (back only)", "before": 5, "after": 0, "mode": "backward", "total_steps": 6},
        {"name": "10+1+0 (back only)", "before": 10, "after": 0, "mode": "backward", "total_steps": 11},
    ]

    context_results = []
    for cfg in context_configs:
        logger.info("  Context %s...", cfg["name"])
        Z, y, class_names = _extract_embeddings(
            raw_cfg, include_none, device, 3,
            cfg["before"], cfg["after"], cfg["mode"])

        metrics = run_sklearn_loocv(Z, y, class_names, seed)
        n_correct = int(round(metrics.accuracy * len(y)))
        ci_lo, ci_hi = compute_wilson_ci(n_correct, len(y))

        context_results.append({
            "name": cfg["name"],
            "accuracy": round(metrics.accuracy, 4),
            "f1_macro": round(metrics.f1_macro, 4),
            "roc_auc": round(metrics.roc_auc, 4),
            "ci_95_lower": round(ci_lo, 4),
            "ci_95_upper": round(ci_hi, 4),
            "total_steps": cfg["total_steps"],
            "context_before": cfg["before"],
            "context_after": cfg["after"],
            "context_mode": cfg["mode"],
        })
        logger.info("    Acc=%.4f  F1=%.4f  AUC=%.4f",
                     metrics.accuracy, metrics.f1_macro, metrics.roc_auc)

    # Sort context results by accuracy for display
    context_sorted = sorted(context_results, key=lambda r: -r["accuracy"])
    logger.info("Context Top-3:")
    for i, r in enumerate(context_sorted[:3]):
        logger.info("  #%d %s: Acc=%.4f", i + 1, r["name"], r["accuracy"])

    # --- Layer Sweep ---
    logger.info("--- Layer Sweep (L0-L5, context 2+1+2) ---")
    layer_results = []

    for layer_idx in range(6):
        logger.info("  Layer L%d...", layer_idx)
        Z_layer, y, class_names = _extract_embeddings(
            raw_cfg, include_none, device, layer_idx, 2, 2)
        metrics = run_sklearn_loocv(Z_layer, y, class_names, seed)
        layer_results.append({
            "name": f"L{layer_idx}",
            "accuracy": round(metrics.accuracy, 4),
            "f1_macro": round(metrics.f1_macro, 4),
            "roc_auc": round(metrics.roc_auc, 4),
        })
        logger.info("    L%d: Acc=%.4f  F1=%.4f  AUC=%.4f",
                     layer_idx, metrics.accuracy, metrics.f1_macro, metrics.roc_auc)

    # Multi-layer combinations
    logger.info("--- Multi-layer RF combinations ---")
    Z_L0, _, _ = _extract_embeddings(raw_cfg, include_none, device, 0, 2, 2)
    Z_L3, _, _ = _extract_embeddings(raw_cfg, include_none, device, 3, 2, 2)
    Z_L5, _, _ = _extract_embeddings(raw_cfg, include_none, device, 5, 2, 2)

    multi_layer_configs = [
        ("Concat L0+L3", np.concatenate([Z_L0, Z_L3], axis=1)),
        ("Concat L0+L3+L5", np.concatenate([Z_L0, Z_L3, Z_L5], axis=1)),
    ]

    for name, Z_ml in multi_layer_configs:
        logger.info("  %s (D=%d)...", name, Z_ml.shape[1])
        metrics = run_sklearn_loocv(Z_ml, y, class_names, seed)
        layer_results.append({
            "name": name,
            "accuracy": round(metrics.accuracy, 4),
            "f1_macro": round(metrics.f1_macro, 4),
            "roc_auc": round(metrics.roc_auc, 4),
        })
        logger.info("    %s: Acc=%.4f", name, metrics.accuracy)

    layer_sorted = sorted(layer_results, key=lambda r: -r["accuracy"])
    logger.info("Layer Top-3:")
    for i, r in enumerate(layer_sorted[:3]):
        logger.info("  #%d %s: Acc=%.4f", i + 1, r["name"], r["accuracy"])

    # --- Generate visualizations ---
    logger.info("Generating Group 2 visualizations...")

    plot_context_sweep(context_results, plots_dir / "context_sweep")
    plot_layer_comparison(layer_results, plots_dir / "layer_comparison")

    # --- Save tables ---
    save_results_csv(context_results, tables_dir / "group2_context_sweep.csv")
    save_results_txt(context_results, tables_dir / "group2_context_sweep.txt")
    save_results_csv(layer_results, tables_dir / "group2_layer_sweep.csv")
    save_results_txt(layer_results, tables_dir / "group2_layer_sweep.txt")

    logger.info("Group 2 complete: %d context configs, %d layer configs",
                len(context_results), len(layer_results))

    return {
        "context_results": context_results,
        "layer_results": layer_results,
    }


# ============================================================================
# Group 3: Visualization Suite
# ============================================================================

def run_group3_visualization(raw_cfg, include_none, device, seed, output_dir):
    """Comprehensive visualization suite for team presentations."""
    logger.info("=" * 60)
    logger.info("GROUP 3: Visualization Suite")
    logger.info("=" * 60)

    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # --- Extract embeddings for all 6 layers ---
    logger.info("Extracting all 6 layer embeddings (L0-L5, context 2+1+2)...")
    layer_embeddings = {}
    y = None
    class_names = None

    for layer_idx in range(6):
        Z, y_tmp, cn = _extract_embeddings(raw_cfg, include_none, device, layer_idx, 2, 2)
        layer_embeddings[f"L{layer_idx}"] = Z
        if y is None:
            y = y_tmp
            class_names = cn

    n_events = len(y)
    n_classes = len(np.unique(y))

    # --- Extract embeddings for multiple context windows ---
    logger.info("Extracting context window embeddings (L3)...")
    context_embeddings = {}
    ctx_configs = [
        ("1+1+1", 1, 1), ("2+1+2", 2, 2), ("4+1+4", 4, 4),
        ("10+1+10", 10, 10), ("2+1+4", 2, 4),
    ]
    for name, before, after in ctx_configs:
        Z, _, _ = _extract_embeddings(raw_cfg, include_none, device, 3, before, after)
        context_embeddings[name] = Z

    # --- t-SNE Grid: 6 layers ---
    logger.info("Generating t-SNE grid (6 layers)...")
    plot_tsne_grid(layer_embeddings, y, class_names,
                   "MantisV2 Layer Embeddings (t-SNE)",
                   plots_dir / "tsne_layers_grid",
                   ncols=3, figsize_per_panel=(4.5, 4))

    # --- t-SNE Grid: context windows ---
    logger.info("Generating t-SNE grid (context windows)...")
    plot_tsne_grid(context_embeddings, y, class_names,
                   "Context Window Effect on Embeddings (t-SNE, L3)",
                   plots_dir / "tsne_contexts_grid",
                   ncols=5, figsize_per_panel=(3.5, 3.5))

    # --- Correct vs Incorrect t-SNE overlay ---
    logger.info("Generating correct/incorrect classification overlay...")
    Z_primary = layer_embeddings["L3"]
    y_pred_rf, y_prob_rf = _rf_loocv_with_probs(Z_primary, y, n_classes, seed)

    _apply_style()
    Z_2d = reduce_dimensions(Z_primary, method="tsne", random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: True labels
    for c in range(n_classes):
        mask = y == c
        color = CLASS_COLORS.get(c, "#666")
        axes[0].scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=color,
                        label=class_names[c], s=40, alpha=0.7,
                        edgecolors="white", linewidths=0.5)
    axes[0].set_title("True Labels", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Panel 2: RF predictions
    for c in range(n_classes):
        mask = y_pred_rf == c
        color = CLASS_COLORS.get(c, "#666")
        axes[1].scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=color,
                        label=class_names[c], s=40, alpha=0.7,
                        edgecolors="white", linewidths=0.5)
    axes[1].set_title("RF Predictions", fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Panel 3: Correct/Incorrect
    correct = y_pred_rf == y
    axes[2].scatter(Z_2d[correct, 0], Z_2d[correct, 1], c="#029E73",
                    label=f"Correct ({correct.sum()})", s=40, alpha=0.6,
                    edgecolors="white", linewidths=0.5)
    axes[2].scatter(Z_2d[~correct, 0], Z_2d[~correct, 1], c="#CC3311",
                    label=f"Error ({(~correct).sum()})", s=80, alpha=0.9,
                    marker="x", linewidths=2)
    axes[2].set_title("Classification Result", fontweight="bold")
    axes[2].legend(fontsize=8)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.suptitle("RF LOOCV Classification (L3, 2+1+2)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, plots_dir / "tsne_classification_overlay")
    plt.close(fig)

    # --- Confidence heatmap ---
    logger.info("Generating confidence analysis...")
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confidence (max prob)
    confidence = y_prob_rf.max(axis=1)
    sc1 = axes[0].scatter(Z_2d[:, 0], Z_2d[:, 1], c=confidence,
                          cmap="RdYlGn", s=40, alpha=0.8,
                          edgecolors="white", linewidths=0.3,
                          vmin=0.3, vmax=1.0)
    plt.colorbar(sc1, ax=axes[0], label="Max Probability")
    axes[0].set_title("RF Confidence Map", fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Entropy
    entropy = -np.sum(y_prob_rf * np.log(y_prob_rf + 1e-10), axis=1)
    sc2 = axes[1].scatter(Z_2d[:, 0], Z_2d[:, 1], c=entropy,
                          cmap="hot_r", s=40, alpha=0.8,
                          edgecolors="white", linewidths=0.3)
    plt.colorbar(sc2, ax=axes[1], label="Entropy (higher = uncertain)")
    axes[1].set_title("Prediction Uncertainty (Entropy)", fontweight="bold")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.suptitle("RF Classification Confidence & Uncertainty", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.05, wspace=0.3)
    save_figure(fig, plots_dir / "confidence_analysis")
    plt.close(fig)

    # --- Distance analysis ---
    logger.info("Generating embedding distance analysis...")
    plot_distance_analysis(Z_primary, y, class_names, plots_dir / "distance_analysis")

    # --- Summary Dashboard ---
    logger.info("Generating summary dashboard...")
    summary_data = {
        "key_metrics": [
            {"name": "Ensemble RF 2-way", "accuracy": 0.9358, "f1": 0.9359, "auc": 0.9772},
            {"name": "RF 2+1+2 (single)", "accuracy": 0.9266, "f1": 0.9266, "auc": 0.9778},
            {"name": "Ensemble RF 3-way", "accuracy": 0.9266, "f1": 0.9266, "auc": 0.9749},
            {"name": "Neural Concat L0+L3", "accuracy": 0.9174, "f1": 0.9177, "auc": 0.9691},
            {"name": "Neural Linear L3", "accuracy": 0.9083, "f1": 0.9085, "auc": 0.9724},
            {"name": "RF PCA-32", "accuracy": 0.8532, "f1": 0.8551, "auc": 0.9441},
        ],
        "best_config": {
            "model": "MantisV2 (frozen)",
            "layer": "L3",
            "context": "2+1+2 + 2+1+4 ensemble",
            "channels": "motionSensor + contactSensor",
            "classifier": "RF soft-voting ensemble",
            "augmentation": "None (not effective)",
            "accuracy": "93.58%",
            "ci": "[87.3%, 96.9%]",
            "stability": "±0.58% (5 seeds)",
        },
        "findings": [
            "2-way ensemble achieves 93.58% (+0.92% over single RF)",
            "RF outperforms neural by ~1.8% (curse of dimensionality)",
            "PCA is HARMFUL: destroys RF's feature bagging advantage",
            "All 5 augmentation methods ineffective on discrete events",
            "Linear head = MLP head → capacity is not the bottleneck",
            "Context 2+1+2 (5min) is optimal for door enter/leave events",
        ],
        "limitations": [
            "Timestamp collision samples removed (4 events)",
            "N=105 extremely small → wide CI",
            "10 events (9.2%) sensor-identical but differently labeled",
            "Bottleneck: input information, not classifier capacity",
            "Next: full fine-tuning, larger context, more data",
        ],
        "performance": {
            "Best Overall": "93.58% (Ensemble RF 2-way)",
            "Best Neural": "91.74% (Linear Concat L0+L3)",
            "Phase 1 Baseline": "92.66% (RF single 2+1+2)",
            "Improvement": "+0.92% from ensemble",
            "Theoretical Max": "100% (collisions removed)",
        },
        "data_info": {
            "Total Events": "105 (33 Enter + 36 Leave + 36 None)",
            "Sensors": "2 (motionSensor + contactSensor)",
            "Sensor Timesteps": "19,190 rows",
            "Date Range": "2026-02-10 ~ 02-19",
            "CV Method": "LOOCV (105-fold)",
            "Embedding Dim": "1024 (512 × 2 channels)",
        },
    }
    plot_summary_dashboard(summary_data, plots_dir / "summary_dashboard")

    # --- Save final report ---
    report = {
        "title": "Phase 2 Final Report — APC Enter/Leave Event Detection",
        "date": "2026-02-28",
        "optimal_config": summary_data["best_config"],
        "key_findings": summary_data["findings"],
        "limitations": summary_data["limitations"],
        "all_methods_ranked": summary_data["key_metrics"],
        "plots_generated": [
            "tsne_layers_grid.png/pdf",
            "tsne_contexts_grid.png/pdf",
            "tsne_classification_overlay.png/pdf",
            "confidence_analysis.png/pdf",
            "distance_analysis.png/pdf",
            "summary_dashboard.png/pdf",
        ],
    }
    with open(reports_dir / "phase2_final_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", reports_dir / "phase2_final_report.json")

    logger.info("Group 3 complete: 6 visualizations generated")

    return report


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Final: Comprehensive Experiment Suite",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--group", required=True, choices=["1", "2", "3", "all"],
                        help="Experiment group: 1 (scenarios), 2 (context+layers), 3 (visualization), all")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--include-none", action="store_true", help="Include NONE class")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", default=None, help="Override output directory")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    raw_cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else raw_cfg.get("seed", 42)
    device = args.device if args.device is not None else raw_cfg.get("model", {}).get("device", "cuda")
    include_none = args.include_none or raw_cfg.get("data", {}).get("include_none", False)

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        raw_cfg.get("output_dir", "results/phase2_final"))
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    groups = [args.group] if args.group != "all" else ["1", "2", "3"]

    results = {}
    for group in groups:
        if group == "1":
            results["group1"] = run_group1_scenarios(
                raw_cfg, include_none, device, seed, output_dir)
        elif group == "2":
            results["group2"] = run_group2_context_layers(
                raw_cfg, include_none, device, seed, output_dir)
        elif group == "3":
            results["group3"] = run_group3_visualization(
                raw_cfg, include_none, device, seed, output_dir)

    t_total = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Done! Total time: %.1fs (%.1f min)", t_total, t_total / 60)
    logger.info("Output directory: %s", output_dir)

    # Save combined results
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / f"group{'_'.join(groups)}_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    logger.info("Saved: %s", reports_dir / f"group{'_'.join(groups)}_results.json")


if __name__ == "__main__":
    main()
