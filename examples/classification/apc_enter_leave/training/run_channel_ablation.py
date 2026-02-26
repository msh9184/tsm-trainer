"""Channel ablation for enter/leave event detection.

Systematic sweep across channel combinations and transformer layers
to identify optimal sensor configurations for MantisV2 zero-shot
event classification.

Dimensions:
  - 23 channel combinations (6 uni + 9 pair + 4 triple + 2 quad + 1 five + 1 full)
  - 6 transformer layers (L0~L5)
  - Optional: multiple seq_len values
  - Primary: RF classifier + LOOCV (fast sweep)
  - Top-10: all 3 classifiers × all 3 CV methods (detailed eval)

Ablation design rationale:
  - Contact sensor (C) is unique to enter/leave (not in occupancy experiments)
  - Previous occupancy optimal: T1 alone + L0 (AUC=0.8937)
  - Hypothesis: C should be highly discriminative for enter vs leave
  - All 15 possible pairs tested for completeness (C(6,2)=15)
  - Triples focus on physically meaningful combinations

Output structure (results/channel_ablation/):
  grid_results.csv              138+ rows: channel_combo × layer
  top10_full_eval.csv           Top-10 × 3 classifiers × 3 CV = ~90 rows
  config.json                   Experiment configuration
  summary_report.txt            Text analysis report
  plots/
    heatmap_combo_layer.png     23 combos × 6 layers AUC heatmap
    univariate_ranking.png      6 sensors ranked by best AUC
    combo_ranking_top15.png     Top-15 combos bar chart
    channel_count_trend.png     AUC vs number of channels
    contact_effect.png          With vs without Contact sensor comparison
    top10_bar.png               Overall top-10 configs
    embeddings_best.png         t-SNE/PCA of best config
  tables/
    grid_results.csv            Full grid
    grid_results.txt            Text table
    univariate_ranking.csv      Univariate sensor ranking
    best_per_size.csv           Best combo per n_channels
    top10_full_eval.csv         Top-10 full eval

Usage:
    cd examples/classification/apc_enter_leave

    # Full channel × layer sweep (138 experiments, ~30 min on A100)
    python training/run_channel_ablation.py \\
        --config training/configs/enter-leave-zeroshot.yaml

    # Quick test: univariate only, L0+L2 only
    python training/run_channel_ablation.py \\
        --config ... --quick

    # Select specific layers
    python training/run_channel_ablation.py \\
        --config ... --layers 0 2 5

    # Add seq_len sweep (top-5 combos × multiple seq_lens)
    python training/run_channel_ablation.py \\
        --config ... --seq-lens 128 256 512

    # No plots (faster, for GPU batch)
    python training/run_channel_ablation.py \\
        --config ... --no-plots

    # CPU mode
    python training/run_channel_ablation.py \\
        --config ... --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

import sys
import os

# Headless matplotlib
import matplotlib as mpl
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    mpl.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_PROJECT_DIR))

# Reuse core components from run_event_detection
from run_event_detection import (
    load_mantis_model,
    extract_all_embeddings,
    build_classifier,
    run_loocv,
    run_stratified_kfold,
    run_lodo,
)
from data.preprocess import EventPreprocessConfig, load_sensor_and_events
from data.dataset import EventDatasetConfig, EventDataset
from visualization.style import setup_style, save_figure, configure_output

logger = logging.getLogger(__name__)


# ============================================================================
# Channel definitions
# ============================================================================

# Full names → short aliases
CHANNEL_ALIASES: dict[str, str] = {
    "408981c2_contactSensor": "C",
    "d620900d_motionSensor": "M",
    "f2e891c6_powerMeter": "P",
    "d620900d_temperatureMeasurement": "T1",
    "ccea734e_temperatureMeasurement": "T2",
    "f2e891c6_energyMeter": "E",
}

# 23 channel combinations: 6 uni + 9 pair + 4 triple + 2 quad + 1 five + 1 full
CHANNEL_COMBOS: dict[str, list[str]] = {
    # === Univariate (6) ===
    "C": ["408981c2_contactSensor"],
    "M": ["d620900d_motionSensor"],
    "P": ["f2e891c6_powerMeter"],
    "T1": ["d620900d_temperatureMeasurement"],
    "T2": ["ccea734e_temperatureMeasurement"],
    "E": ["f2e891c6_energyMeter"],

    # === Pairs (9) — all meaningful combinations ===
    "C+M": ["408981c2_contactSensor", "d620900d_motionSensor"],
    "C+P": ["408981c2_contactSensor", "f2e891c6_powerMeter"],
    "C+T1": ["408981c2_contactSensor", "d620900d_temperatureMeasurement"],
    "C+E": ["408981c2_contactSensor", "f2e891c6_energyMeter"],
    "M+P": ["d620900d_motionSensor", "f2e891c6_powerMeter"],
    "M+T1": ["d620900d_motionSensor", "d620900d_temperatureMeasurement"],
    "M+E": ["d620900d_motionSensor", "f2e891c6_energyMeter"],
    "T1+T2": ["d620900d_temperatureMeasurement", "ccea734e_temperatureMeasurement"],
    "P+T1": ["f2e891c6_powerMeter", "d620900d_temperatureMeasurement"],

    # === Triples (4) ===
    "C+M+P": [
        "408981c2_contactSensor", "d620900d_motionSensor", "f2e891c6_powerMeter",
    ],
    "C+M+T1": [
        "408981c2_contactSensor", "d620900d_motionSensor",
        "d620900d_temperatureMeasurement",
    ],
    "C+P+T1": [
        "408981c2_contactSensor", "f2e891c6_powerMeter",
        "d620900d_temperatureMeasurement",
    ],
    "M+P+T1": [
        "d620900d_motionSensor", "f2e891c6_powerMeter",
        "d620900d_temperatureMeasurement",
    ],

    # === Quads (2) ===
    "C+M+P+T1": [
        "408981c2_contactSensor", "d620900d_motionSensor",
        "f2e891c6_powerMeter", "d620900d_temperatureMeasurement",
    ],
    "M+P+T1+T2": [
        "d620900d_motionSensor", "f2e891c6_powerMeter",
        "d620900d_temperatureMeasurement", "ccea734e_temperatureMeasurement",
    ],

    # === 5-channel (1) ===
    "C+M+P+T1+T2": [
        "408981c2_contactSensor", "d620900d_motionSensor",
        "f2e891c6_powerMeter", "d620900d_temperatureMeasurement",
        "ccea734e_temperatureMeasurement",
    ],

    # === Full 6-channel (1) ===
    "ALL6": [
        "408981c2_contactSensor", "d620900d_motionSensor",
        "f2e891c6_powerMeter", "d620900d_temperatureMeasurement",
        "ccea734e_temperatureMeasurement", "f2e891c6_energyMeter",
    ],
}

UNIVARIATE_COMBOS = ["C", "M", "P", "T1", "T2", "E"]

LAYERS = [0, 1, 2, 3, 4, 5]

# seq_len descriptions (1-min resolution)
SEQ_LEN_DESC: dict[int, str] = {
    32: "32m",
    64: "1h4m",
    128: "2h8m",
    256: "4h16m",
    384: "6h24m",
    512: "8h32m",
}

# Grid result CSV fields
GRID_FIELDS = [
    "combo", "n_channels", "layer", "seq_len", "emb_dim",
    "accuracy", "f1_macro", "roc_auc", "eer", "n_samples", "embed_time_s",
]

# Top-10 full eval CSV fields
TOP10_FIELDS = [
    "rank", "combo", "n_channels", "layer", "emb_dim",
    "classifier", "cv_method", "accuracy", "f1_macro", "roc_auc", "eer",
]


# ============================================================================
# Config and data loading
# ============================================================================

def load_config(config_path: str | Path) -> dict:
    """Load raw YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_all_channel_data(
    raw_cfg: dict,
    include_none: bool = False,
) -> tuple[np.ndarray, "pd.DatetimeIndex", np.ndarray, np.ndarray, list[str], list[str]]:
    """Load sensor data with ALL channels (no channel filter)."""
    import pandas as pd

    data_cfg = raw_cfg.get("data", {})
    _fields = set(EventPreprocessConfig.__dataclass_fields__)
    base_cfg = {k: v for k, v in data_cfg.items() if k in _fields}

    # Remove channel filter to load ALL numeric channels
    base_cfg.pop("channels", None)
    base_cfg["nan_threshold"] = data_cfg.get("nan_threshold", 0.3)
    base_cfg["include_none"] = include_none

    config = EventPreprocessConfig(**base_cfg)
    return load_sensor_and_events(config)


def build_combo_dataset(
    full_sensor_array: np.ndarray,
    all_channel_names: list[str],
    target_channels: list[str],
    sensor_timestamps,
    event_timestamps: np.ndarray,
    event_labels: np.ndarray,
    seq_len: int = 512,
    target_seq_len: int | None = 512,
) -> EventDataset:
    """Build EventDataset for a specific channel combination."""
    ch_indices = [all_channel_names.index(ch) for ch in target_channels]
    combo_array = full_sensor_array[:, ch_indices]

    # No interpolation at native MantisV2 length (512)
    tgt = target_seq_len if seq_len != 512 else None

    ds_config = EventDatasetConfig(seq_len=seq_len, target_seq_len=tgt)
    return EventDataset(
        combo_array, sensor_timestamps, event_timestamps, event_labels, ds_config,
    )


# ============================================================================
# Result I/O helpers
# ============================================================================

def _nan_safe(v):
    """Convert NaN/Inf to None for JSON/CSV serialisation."""
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.floating,)):
        v = float(v)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def save_csv(results: list[dict], path: Path, fieldnames: list[str] | None = None) -> None:
    """Save results as CSV."""
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: _nan_safe(r.get(k, "")) for k in fieldnames})
    logger.info("Saved: %s (%d rows)", path, len(results))


def save_txt(results: list[dict], path: Path, fieldnames: list[str] | None = None) -> None:
    """Save results as aligned text table."""
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(results[0].keys())

    widths = {}
    for f in fieldnames:
        vals = [str(r.get(f, "")) for r in results]
        widths[f] = max(len(f), max((len(v) for v in vals), default=0))

    lines = []
    header = "  ".join(f"{f:<{widths[f]}}" for f in fieldnames)
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        line = "  ".join(f"{str(r.get(f, '')):<{widths[f]}}" for f in fieldnames)
        lines.append(line)

    path.write_text("\n".join(lines) + "\n")
    logger.info("Saved: %s", path)


# ============================================================================
# Visualization
# ============================================================================

def plot_heatmap_combo_layer(
    results: list[dict],
    output_path: Path,
    seq_len: int | None = None,
) -> None:
    """Heatmap: channel_combo (rows) × layer (cols), colored by AUC."""
    setup_style()

    # Filter by seq_len if specified
    data = [r for r in results if seq_len is None or r["seq_len"] == seq_len]
    if not data:
        return

    combos_present = list(dict.fromkeys(r["combo"] for r in data))
    layers_present = sorted(set(r["layer"] for r in data))

    matrix = np.full((len(combos_present), len(layers_present)), np.nan)
    for r in data:
        i = combos_present.index(r["combo"])
        j = layers_present.index(r["layer"])
        matrix[i, j] = r.get("roc_auc") or np.nan

    h = max(6, len(combos_present) * 0.35 + 1)
    w = max(5, len(layers_present) * 0.9 + 3)
    fig, ax = plt.subplots(figsize=(w, h))

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.4, vmax=1.0)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                txt, color = "-", "grey"
            else:
                txt = f"{val:.3f}"
                color = "white" if val < 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(range(len(layers_present)))
    ax.set_xticklabels([f"L{l}" for l in layers_present], fontsize=9)
    ax.set_yticks(range(len(combos_present)))
    ax.set_yticklabels(combos_present, fontsize=8)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Channel Combination")
    sl_desc = f" (seq_len={seq_len})" if seq_len else ""
    ax.set_title(f"AUC Heatmap — Channel × Layer{sl_desc}", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.6, label="AUC")

    save_figure(fig, output_path)
    plt.close(fig)


def plot_univariate_ranking(results: list[dict], output_path: Path) -> None:
    """Bar chart ranking 6 individual sensors by best AUC."""
    setup_style()

    uni_results = {}
    for r in results:
        if r["combo"] in UNIVARIATE_COMBOS:
            auc = r.get("roc_auc") or 0
            if r["combo"] not in uni_results or auc > uni_results[r["combo"]]["auc"]:
                uni_results[r["combo"]] = {
                    "auc": auc,
                    "layer": r["layer"],
                    "f1": r.get("f1_macro", 0),
                }

    if not uni_results:
        return

    sorted_sensors = sorted(uni_results.items(), key=lambda x: x[1]["auc"], reverse=True)
    names = [s[0] for s in sorted_sensors]
    aucs = [s[1]["auc"] for s in sorted_sensors]
    layers = [s[1]["layer"] for s in sorted_sensors]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#DE8F05"] + ["#0173B2"] * (len(names) - 1)
    bars = ax.bar(range(len(names)), aucs, color=colors, alpha=0.85, edgecolor="white")

    for i, (bar, auc, layer) in enumerate(zip(bars, aucs, layers)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{auc:.3f}\nL{layer}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_xlabel("Sensor (univariate)")
    ax.set_ylabel("AUC (best layer)")
    ax.set_title("Univariate Sensor Ranking — Enter/Leave Detection", fontsize=11)
    ax.set_ylim(0.3, 1.05)

    save_figure(fig, output_path)
    plt.close(fig)


def plot_combo_ranking(results: list[dict], output_path: Path, top_k: int = 15) -> None:
    """Horizontal bar chart of top-K channel combos by best AUC."""
    setup_style()

    best_per_combo = {}
    for r in results:
        c = r["combo"]
        auc = r.get("roc_auc") or 0
        if c not in best_per_combo or auc > best_per_combo[c]["roc_auc"]:
            best_per_combo[c] = r

    sorted_combos = sorted(best_per_combo.items(),
                           key=lambda x: x[1].get("roc_auc", 0) or 0, reverse=True)
    top = sorted_combos[:top_k]

    if not top:
        return

    labels = [f"{c} ({r['n_channels']}ch, L{r['layer']})" for c, r in top]
    aucs = [r.get("roc_auc") or 0 for _, r in top]

    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.4)))
    colors = ["#DE8F05"] + ["#0173B2"] * (len(top) - 1)
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, aucs, color=colors, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, aucs):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", ha="left", va="center", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUC")
    ax.set_title(f"Top {len(top)} Channel Combinations by AUC", fontsize=11)
    min_auc = min(aucs) if aucs else 0.4
    ax.set_xlim(max(0.3, min_auc - 0.05), 1.0)
    ax.invert_yaxis()

    save_figure(fig, output_path)
    plt.close(fig)


def plot_channel_count_trend(results: list[dict], output_path: Path) -> None:
    """Box plot: AUC distribution vs number of channels."""
    setup_style()

    by_nch = defaultdict(list)
    for r in results:
        auc = r.get("roc_auc")
        if auc is not None and not np.isnan(auc):
            by_nch[r["n_channels"]].append(auc)

    if not by_nch:
        return

    sizes = sorted(by_nch.keys())
    data = [by_nch[s] for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=[str(s) for s in sizes],
                    patch_artist=True, widths=0.5)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sizes)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, (s, aucs) in enumerate(zip(sizes, data)):
        x = np.random.normal(i + 1, 0.04, size=len(aucs))
        ax.scatter(x, aucs, alpha=0.4, s=15, color="black", zorder=3)

    # Best per size
    best_per_size = {s: max(aucs) for s, aucs in by_nch.items()}
    ax.plot([sizes.index(s) + 1 for s in sorted(best_per_size)],
            [best_per_size[s] for s in sorted(best_per_size)],
            "r--o", lw=1.5, markersize=6, label="Best AUC", zorder=4)

    ax.set_xlabel("Number of Channels")
    ax.set_ylabel("AUC (LOOCV, RF)")
    ax.set_title("AUC vs Channel Count — Univariate to Multivariate", fontsize=11)
    ax.legend(loc="lower right")
    ax.set_ylim(0.3, 1.05)

    save_figure(fig, output_path)
    plt.close(fig)


def plot_contact_effect(results: list[dict], output_path: Path) -> None:
    """Paired comparison: AUC with vs without Contact sensor (C).

    For each combo that includes C, find the matching combo without C
    and plot the AUC difference.
    """
    setup_style()

    best_per_combo = {}
    for r in results:
        c = r["combo"]
        auc = r.get("roc_auc") or 0
        if c not in best_per_combo or auc > best_per_combo[c]:
            best_per_combo[c] = auc

    # Find pairs: combo_with_C vs combo_without_C
    pairs = []
    for combo_name, channels in CHANNEL_COMBOS.items():
        if "408981c2_contactSensor" not in channels:
            continue
        if combo_name == "C":
            continue  # Skip univariate C itself

        # Find the combo without C
        channels_no_c = [ch for ch in channels if ch != "408981c2_contactSensor"]
        # Find matching combo name
        for other_name, other_channels in CHANNEL_COMBOS.items():
            if sorted(other_channels) == sorted(channels_no_c):
                if combo_name in best_per_combo and other_name in best_per_combo:
                    pairs.append({
                        "with_c": combo_name,
                        "without_c": other_name,
                        "auc_with": best_per_combo[combo_name],
                        "auc_without": best_per_combo[other_name],
                        "delta": best_per_combo[combo_name] - best_per_combo[other_name],
                    })
                break

    if not pairs:
        return

    pairs.sort(key=lambda p: p["delta"], reverse=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(pairs) * 0.6)))

    labels = [f"{p['with_c']} vs {p['without_c']}" for p in pairs]
    deltas = [p["delta"] for p in pairs]
    colors = ["#009E73" if d >= 0 else "#CC3311" for d in deltas]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, deltas, color=colors, alpha=0.85, edgecolor="white")

    for bar, p in zip(bars, pairs):
        x = bar.get_width()
        sign = "+" if x >= 0 else ""
        ax.text(x + 0.003 * (1 if x >= 0 else -1),
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{x:.3f} ({p['auc_with']:.3f} vs {p['auc_without']:.3f})",
                ha="left" if x >= 0 else "right", va="center", fontsize=7)

    ax.axvline(x=0, color="grey", ls="--", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUC Delta (with C - without C)")
    ax.set_title("Contact Sensor Effect — AUC Improvement", fontsize=11)
    ax.invert_yaxis()

    save_figure(fig, output_path)
    plt.close(fig)


# ============================================================================
# Report generation
# ============================================================================

def generate_report(
    grid_results: list[dict],
    top10_results: list[dict],
    total_time: float,
    combos_used: dict[str, list[str]],
    layers_used: list[int],
    seq_lens_used: list[int],
    available_channels: list[str],
) -> str:
    """Generate comprehensive text analysis report."""
    lines = []
    lines.append("=" * 72)
    lines.append("CHANNEL ABLATION REPORT — Enter/Leave Event Detection")
    lines.append("MantisV2 Zero-Shot × LOOCV × RandomForest")
    lines.append("=" * 72)
    lines.append("")

    # 1. Configuration
    lines.append("1. EXPERIMENT CONFIGURATION")
    lines.append("-" * 50)
    lines.append(f"  Channel combos: {len(combos_used)} ({len([c for c in combos_used if len(combos_used[c]) == 1])} uni, "
                 f"{len([c for c in combos_used if len(combos_used[c]) == 2])} pair, "
                 f"{len([c for c in combos_used if len(combos_used[c]) >= 3])} multi)")
    lines.append(f"  Layers: L{min(layers_used)}~L{max(layers_used)} ({len(layers_used)} layers)")
    lines.append(f"  seq_len: {seq_lens_used}")
    lines.append(f"  Grid size: {len(combos_used)} × {len(layers_used)} × {len(seq_lens_used)} "
                 f"= {len(combos_used) * len(layers_used) * len(seq_lens_used)}")
    lines.append(f"  Completed: {len(grid_results)} experiments in {total_time:.0f}s")
    lines.append(f"  Available channels: {available_channels}")
    lines.append(f"  Primary: RF (200 trees) + LOOCV (73 folds)")
    lines.append("")

    if not grid_results:
        lines.append("  [No results to report]")
        return "\n".join(lines)

    # 2. Overall best
    best = max(grid_results, key=lambda r: r.get("roc_auc") or 0)
    lines.append("2. OVERALL BEST CONFIGURATION")
    lines.append("-" * 50)
    lines.append(f"  Channels: {best['combo']} ({best['n_channels']} ch)")
    lines.append(f"  Layer: L{best['layer']}")
    lines.append(f"  seq_len: {best['seq_len']} ({SEQ_LEN_DESC.get(best['seq_len'], '')})")
    lines.append(f"  AUC: {best.get('roc_auc', 'N/A')}")
    lines.append(f"  EER: {best.get('eer', 'N/A')}")
    lines.append(f"  F1 macro: {best.get('f1_macro', 'N/A')}")
    lines.append(f"  Accuracy: {best.get('accuracy', 'N/A')}")
    lines.append(f"  Embedding dim: {best.get('emb_dim', 'N/A')}")
    lines.append("")

    # 3. Univariate ranking
    lines.append("3. UNIVARIATE SENSOR RANKING")
    lines.append("-" * 50)
    lines.append(f"  {'Sensor':<8s} {'Best AUC':>9s} {'Layer':>6s} {'F1':>7s} {'Acc':>7s}")
    lines.append("  " + "-" * 42)

    uni_best = {}
    for r in grid_results:
        if r["combo"] in UNIVARIATE_COMBOS:
            auc = r.get("roc_auc") or 0
            if r["combo"] not in uni_best or auc > (uni_best[r["combo"]].get("roc_auc") or 0):
                uni_best[r["combo"]] = r

    for combo, r in sorted(uni_best.items(),
                           key=lambda x: x[1].get("roc_auc") or 0, reverse=True):
        lines.append(
            f"  {combo:<8s} {r.get('roc_auc', 'N/A'):>9} L{r['layer']:>4d} "
            f"{r.get('f1_macro', 'N/A'):>7} {r.get('accuracy', 'N/A'):>7}"
        )
    lines.append("")

    # 4. Best per channel count
    lines.append("4. BEST CONFIG PER CHANNEL COUNT")
    lines.append("-" * 50)
    lines.append(f"  {'n_ch':<5s} {'Combo':<16s} {'Layer':>6s} {'AUC':>7s} {'F1':>7s}")
    lines.append("  " + "-" * 45)

    best_per_size = {}
    for r in grid_results:
        n = r["n_channels"]
        auc = r.get("roc_auc") or 0
        if n not in best_per_size or auc > (best_per_size[n].get("roc_auc") or 0):
            best_per_size[n] = r

    for n in sorted(best_per_size):
        r = best_per_size[n]
        lines.append(
            f"  {n:<5d} {r['combo']:<16s} L{r['layer']:>4d} "
            f"{r.get('roc_auc', 'N/A'):>7} {r.get('f1_macro', 'N/A'):>7}"
        )
    lines.append("")

    # 5. Best per combo (sorted by AUC)
    lines.append("5. ALL COMBOS RANKED BY BEST AUC")
    lines.append("-" * 50)
    lines.append(f"  {'Rank':<5s} {'Combo':<16s} {'n_ch':<5s} {'Layer':>6s} {'AUC':>7s} {'EER':>7s} {'F1':>7s}")
    lines.append("  " + "-" * 56)

    best_per_combo = {}
    for r in grid_results:
        c = r["combo"]
        auc = r.get("roc_auc") or 0
        if c not in best_per_combo or auc > (best_per_combo[c].get("roc_auc") or 0):
            best_per_combo[c] = r

    for rank, (c, r) in enumerate(
        sorted(best_per_combo.items(), key=lambda x: x[1].get("roc_auc") or 0, reverse=True),
        1,
    ):
        lines.append(
            f"  {rank:<5d} {c:<16s} {r['n_channels']:<5d} L{r['layer']:>4d} "
            f"{r.get('roc_auc', 'N/A'):>7} {r.get('eer', 'N/A'):>7} {r.get('f1_macro', 'N/A'):>7}"
        )
    lines.append("")

    # 6. Contact sensor effect
    lines.append("6. CONTACT SENSOR EFFECT (with C vs without C)")
    lines.append("-" * 50)

    best_auc = {c: r.get("roc_auc") or 0 for c, r in best_per_combo.items()}
    c_effects = []
    for combo_name, channels in combos_used.items():
        if "408981c2_contactSensor" not in channels or combo_name == "C":
            continue
        channels_no_c = [ch for ch in channels if ch != "408981c2_contactSensor"]
        for other_name, other_channels in combos_used.items():
            if sorted(other_channels) == sorted(channels_no_c):
                if combo_name in best_auc and other_name in best_auc:
                    delta = best_auc[combo_name] - best_auc[other_name]
                    c_effects.append((combo_name, other_name, delta,
                                     best_auc[combo_name], best_auc[other_name]))
                break

    if c_effects:
        c_effects.sort(key=lambda x: x[2], reverse=True)
        for with_c, without_c, delta, auc_w, auc_wo in c_effects:
            sign = "+" if delta >= 0 else ""
            lines.append(f"  {with_c:<12s} vs {without_c:<12s}: {sign}{delta:.4f} "
                         f"({auc_w:.4f} vs {auc_wo:.4f})")
        avg_delta = np.mean([x[2] for x in c_effects])
        lines.append(f"  Average C effect: {'+' if avg_delta >= 0 else ''}{avg_delta:.4f}")
    else:
        lines.append("  [No matching pairs found]")
    lines.append("")

    # 7. Layer statistics
    lines.append("7. LAYER STATISTICS")
    lines.append("-" * 50)
    layer_stats = defaultdict(list)
    for r in grid_results:
        auc = r.get("roc_auc")
        if auc is not None and not np.isnan(auc):
            layer_stats[r["layer"]].append(auc)

    lines.append(f"  {'Layer':<8s} {'Mean AUC':>10s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'n':>4s}")
    lines.append("  " + "-" * 44)
    for layer in sorted(layer_stats):
        aucs = layer_stats[layer]
        lines.append(
            f"  L{layer:<6d} {np.nanmean(aucs):>10.4f} {np.nanstd(aucs):>8.4f} "
            f"{np.nanmin(aucs):>8.4f} {np.nanmax(aucs):>8.4f} {len(aucs):>4d}"
        )
    lines.append("")

    # 8. Top-10 full eval
    if top10_results:
        lines.append("8. TOP-10 CONFIGS — ALL CLASSIFIERS × CV METHODS")
        lines.append("-" * 50)

        current_rank = None
        for r in top10_results:
            if r["rank"] != current_rank:
                current_rank = r["rank"]
                lines.append(f"\n  #{r['rank']} {r['combo']} ({r['n_channels']}ch) L{r['layer']}")
            lines.append(
                f"    {r['classifier']:<18s} {r['cv_method']:<20s} "
                f"AUC={r.get('roc_auc', 'N/A')}  F1={r.get('f1_macro', 'N/A')}"
            )
        lines.append("")

    # 9. Key findings
    lines.append("9. KEY FINDINGS")
    lines.append("-" * 50)

    if uni_best:
        best_uni = max(uni_best.items(), key=lambda x: x[1].get("roc_auc") or 0)
        worst_uni = min(uni_best.items(), key=lambda x: x[1].get("roc_auc") or 0)
        lines.append(f"  (a) Best univariate: {best_uni[0]} AUC={best_uni[1].get('roc_auc', 'N/A')}")
        lines.append(f"      Worst univariate: {worst_uni[0]} AUC={worst_uni[1].get('roc_auc', 'N/A')}")

    if best_per_size:
        lines.append(f"  (b) Multivariate effect:")
        for n in sorted(best_per_size):
            r = best_per_size[n]
            lines.append(f"      {n}-ch best: {r['combo']} AUC={r.get('roc_auc', 'N/A')}")

    if layer_stats:
        best_layer = max(layer_stats, key=lambda l: np.nanmean(layer_stats[l]))
        worst_layer = min(layer_stats, key=lambda l: np.nanmean(layer_stats[l]))
        lines.append(f"  (c) Best mean layer: L{best_layer} "
                     f"(avg={np.nanmean(layer_stats[best_layer]):.4f})")
        lines.append(f"      Worst mean layer: L{worst_layer} "
                     f"(avg={np.nanmean(layer_stats[worst_layer]):.4f})")

    if c_effects:
        positive_c = sum(1 for x in c_effects if x[2] > 0)
        lines.append(f"  (d) Contact sensor: positive effect in {positive_c}/{len(c_effects)} comparisons")

    lines.append("")
    lines.append(f"Total runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
    lines.append("=" * 72)

    return "\n".join(lines)


# ============================================================================
# Main experiment runner
# ============================================================================

def run_ablation(
    raw_cfg: dict,
    combos: dict[str, list[str]],
    layers: list[int],
    seq_lens: list[int],
    include_none: bool = False,
    device: str = "cuda",
    seed: int = 42,
    no_plots: bool = False,
    top_k: int = 10,
) -> tuple[list[dict], list[dict]]:
    """Run channel ablation sweep.

    Returns (grid_results, top10_results).
    """
    output_dir = Path(raw_cfg.get("output_dir", "results/channel_ablation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    configure_output(formats=["png"], dpi=200)

    # ----- Load ALL channel data -----
    logger.info("=" * 60)
    logger.info("Loading data (ALL channels, no filter)...")
    logger.info("=" * 60)

    (sensor_array, sensor_timestamps, event_timestamps,
     event_labels, all_channel_names, class_names) = load_all_channel_data(
        raw_cfg, include_none=include_none,
    )

    logger.info("Available channels (%d): %s", len(all_channel_names), all_channel_names)

    # Verify requested combos have available channels
    valid_combos = {}
    for combo_name, channels in combos.items():
        missing = [ch for ch in channels if ch not in all_channel_names]
        if missing:
            logger.warning("Skipping combo %s: missing channels %s", combo_name, missing)
            continue
        valid_combos[combo_name] = channels

    logger.info("Valid combos: %d / %d", len(valid_combos), len(combos))

    # Model config
    model_cfg = raw_cfg.get("model", {})
    pretrained_name = model_cfg.get("pretrained_name", "paris-noah/MantisV2")
    output_token = model_cfg.get("output_token", "combined")

    # CV config (for top-K full eval)
    cv_cfg = raw_cfg.get("cv", {})
    strat_k = cv_cfg.get("stratified_k", 5)
    strat_repeats = cv_cfg.get("stratified_repeats", 10)

    # ----- Main grid sweep: combo × layer × seq_len -----
    grid_results = []
    embedding_cache = {}  # (layer, combo, seq_len) → Z
    total_experiments = len(valid_combos) * len(layers) * len(seq_lens)
    experiment_idx = 0

    t_total_start = time.time()

    for layer in layers:
        logger.info("=" * 60)
        logger.info("Layer %d: Loading MantisV2...", layer)
        logger.info("=" * 60)

        network, model = load_mantis_model(pretrained_name, layer, output_token, device)

        for seq_len in seq_lens:
            for combo_name, channels in valid_combos.items():
                experiment_idx += 1
                logger.info(
                    "[%d/%d] L%d / %s (%d ch) / seq=%d",
                    experiment_idx, total_experiments,
                    layer, combo_name, len(channels), seq_len,
                )

                # Build dataset for this combo
                dataset = build_combo_dataset(
                    sensor_array, all_channel_names, channels,
                    sensor_timestamps, event_timestamps, event_labels,
                    seq_len=seq_len, target_seq_len=512,
                )

                # Extract embeddings
                t0 = time.time()
                Z = extract_all_embeddings(model, dataset)
                embed_time = time.time() - t0

                # Cache embeddings
                cache_key = (layer, combo_name, seq_len)
                embedding_cache[cache_key] = Z

                # Run LOOCV with RF
                clf_factory = lambda s=seed: build_classifier("random_forest", s)
                metrics = run_loocv(Z, event_labels, clf_factory, class_names)

                auc_val = metrics.roc_auc if not np.isnan(metrics.roc_auc) else None
                eer_val = metrics.eer if not np.isnan(metrics.eer) else None

                result = {
                    "combo": combo_name,
                    "n_channels": len(channels),
                    "layer": layer,
                    "seq_len": seq_len,
                    "emb_dim": Z.shape[1],
                    "accuracy": round(metrics.accuracy, 4),
                    "f1_macro": round(metrics.f1_macro, 4),
                    "roc_auc": round(auc_val, 4) if auc_val is not None else None,
                    "eer": round(eer_val, 4) if eer_val is not None else None,
                    "n_samples": metrics.n_samples,
                    "embed_time_s": round(embed_time, 1),
                }
                grid_results.append(result)

                logger.info(
                    "  AUC=%.4f  F1=%.4f  Acc=%.4f  (embed %.1fs)",
                    auc_val or 0, metrics.f1_macro, metrics.accuracy, embed_time,
                )

        # Free GPU memory after processing all combos for this layer
        del model, network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Layer %d complete, GPU memory freed", layer)

    grid_time = time.time() - t_total_start
    logger.info("Grid sweep complete: %d experiments in %.0fs", len(grid_results), grid_time)

    # ----- Save grid results -----
    save_csv(grid_results, tables_dir / "grid_results.csv", GRID_FIELDS)
    save_txt(grid_results, tables_dir / "grid_results.txt", GRID_FIELDS)

    # Univariate ranking
    uni_rows = [r for r in grid_results if r["combo"] in UNIVARIATE_COMBOS]
    if uni_rows:
        uni_best = {}
        for r in uni_rows:
            c = r["combo"]
            if c not in uni_best or (r.get("roc_auc") or 0) > (uni_best[c].get("roc_auc") or 0):
                uni_best[c] = r
        uni_ranking = sorted(uni_best.values(), key=lambda r: r.get("roc_auc") or 0, reverse=True)
        save_csv(uni_ranking, tables_dir / "univariate_ranking.csv", GRID_FIELDS)

    # Best per channel count
    best_per_size = {}
    for r in grid_results:
        n = r["n_channels"]
        if n not in best_per_size or (r.get("roc_auc") or 0) > (best_per_size[n].get("roc_auc") or 0):
            best_per_size[n] = r
    size_rows = [best_per_size[n] for n in sorted(best_per_size)]
    save_csv(size_rows, tables_dir / "best_per_size.csv", GRID_FIELDS)

    # ----- Top-K full evaluation -----
    top10_results = []
    if top_k > 0:
        logger.info("=" * 60)
        logger.info("Top-%d full evaluation (all classifiers × CV methods)...", top_k)
        logger.info("=" * 60)

        sorted_grid = sorted(grid_results, key=lambda r: r.get("roc_auc") or 0, reverse=True)

        # Deduplicate by combo (keep best layer/seq_len per combo)
        seen_combos = set()
        top_configs = []
        for r in sorted_grid:
            if r["combo"] not in seen_combos:
                seen_combos.add(r["combo"])
                top_configs.append(r)
            if len(top_configs) >= top_k:
                break

        classifier_names = raw_cfg.get("classifiers", ["random_forest", "svm", "nearest_centroid"])
        cv_methods = cv_cfg.get("methods", ["loocv", "stratified_kfold", "lodo"])

        for rank, config in enumerate(top_configs, 1):
            cache_key = (config["layer"], config["combo"], config["seq_len"])
            Z = embedding_cache.get(cache_key)
            if Z is None:
                logger.warning("Cache miss for %s, skipping", cache_key)
                continue

            # Build day groups for LODO
            combo_channels = valid_combos[config["combo"]]
            dataset = build_combo_dataset(
                sensor_array, all_channel_names, combo_channels,
                sensor_timestamps, event_timestamps, event_labels,
                seq_len=config["seq_len"], target_seq_len=512,
            )
            day_groups = dataset.get_day_groups()

            for clf_name in classifier_names:
                clf_factory = lambda name=clf_name, s=seed: build_classifier(name, s)

                for cv_method in cv_methods:
                    if cv_method == "loocv":
                        metrics = run_loocv(Z, event_labels, clf_factory, class_names)
                    elif cv_method == "stratified_kfold":
                        metrics = run_stratified_kfold(
                            Z, event_labels, clf_factory,
                            k=strat_k, n_repeats=strat_repeats, seed=seed,
                            class_names=class_names,
                        )
                    elif cv_method == "lodo":
                        metrics = run_lodo(
                            Z, event_labels, day_groups, clf_factory, class_names,
                        )
                    else:
                        continue

                    auc_val = metrics.roc_auc if not np.isnan(metrics.roc_auc) else None
                    eer_val = metrics.eer if not np.isnan(metrics.eer) else None

                    top10_results.append({
                        "rank": rank,
                        "combo": config["combo"],
                        "n_channels": config["n_channels"],
                        "layer": config["layer"],
                        "emb_dim": config["emb_dim"],
                        "classifier": clf_name,
                        "cv_method": cv_method,
                        "accuracy": round(metrics.accuracy, 4),
                        "f1_macro": round(metrics.f1_macro, 4),
                        "roc_auc": round(auc_val, 4) if auc_val is not None else None,
                        "eer": round(eer_val, 4) if eer_val is not None else None,
                    })

            logger.info(
                "  #%d %s L%d: completed %d × %d evaluations",
                rank, config["combo"], config["layer"],
                len(classifier_names), len(cv_methods),
            )

        save_csv(top10_results, tables_dir / "top10_full_eval.csv", TOP10_FIELDS)
        save_txt(top10_results, tables_dir / "top10_full_eval.txt", TOP10_FIELDS)

    total_time = time.time() - t_total_start

    # ----- Visualization -----
    if not no_plots and grid_results:
        logger.info("Generating plots...")

        try:
            plot_heatmap_combo_layer(grid_results, plots_dir / "heatmap_combo_layer")
        except Exception:
            logger.warning("Failed: heatmap_combo_layer", exc_info=True)

        try:
            plot_univariate_ranking(grid_results, plots_dir / "univariate_ranking")
        except Exception:
            logger.warning("Failed: univariate_ranking", exc_info=True)

        try:
            plot_combo_ranking(grid_results, plots_dir / "combo_ranking_top15")
        except Exception:
            logger.warning("Failed: combo_ranking_top15", exc_info=True)

        try:
            plot_channel_count_trend(grid_results, plots_dir / "channel_count_trend")
        except Exception:
            logger.warning("Failed: channel_count_trend", exc_info=True)

        try:
            plot_contact_effect(grid_results, plots_dir / "contact_effect")
        except Exception:
            logger.warning("Failed: contact_effect", exc_info=True)

        # Embedding visualization for best config
        if grid_results:
            best = max(grid_results, key=lambda r: r.get("roc_auc") or 0)
            best_key = (best["layer"], best["combo"], best["seq_len"])
            Z_best = embedding_cache.get(best_key)
            if Z_best is not None:
                try:
                    from visualization.embeddings import plot_embeddings_multi_method
                    fig = plot_embeddings_multi_method(
                        Z_best, event_labels,
                        title=f"Best: {best['combo']} L{best['layer']}",
                        output_path=plots_dir / "embeddings_best",
                    )
                    plt.close(fig)
                except Exception:
                    logger.warning("Failed: embeddings_best", exc_info=True)

    # ----- Report -----
    report_text = generate_report(
        grid_results, top10_results, total_time,
        valid_combos, layers, seq_lens, all_channel_names,
    )

    report_path = output_dir / "summary_report.txt"
    report_path.write_text(report_text)
    logger.info("Report saved: %s", report_path)
    print("\n" + report_text)

    # Config JSON
    config_json = {
        "channel_combos": {k: v for k, v in valid_combos.items()},
        "layers": layers,
        "seq_lens": seq_lens,
        "n_experiments": len(grid_results),
        "top_k": top_k,
        "seed": seed,
        "device": device,
        "total_time_s": round(total_time, 1),
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_json, f, indent=2)

    return grid_results, top10_results


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Channel ablation for enter/leave event detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Transformer layers to evaluate (default: 0 1 2 3 4 5)",
    )
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=None,
        help="Sequence lengths to evaluate (default: 512). "
             "Must be multiples of 32.",
    )
    parser.add_argument(
        "--combos", type=str, nargs="+", default=None,
        help="Channel combo names to evaluate (default: all 23). "
             "Use 'univariate' for all 6 single-channel combos.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: univariate only, L0+L2 only, no top-K eval",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of top configs for full evaluation (default: 10)",
    )
    parser.add_argument(
        "--include-none", action="store_true",
        help="Include NONE events as 3rd class",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda or cpu (overrides config)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation (faster)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    raw_cfg = load_config(args.config)

    # Override config with CLI args
    if args.output_dir:
        raw_cfg["output_dir"] = args.output_dir
    elif "output_dir" not in raw_cfg:
        raw_cfg["output_dir"] = "results/channel_ablation"
    else:
        # Default to channel_ablation subdirectory
        base_dir = Path(raw_cfg["output_dir"]).parent
        raw_cfg["output_dir"] = str(base_dir / "channel_ablation")

    seed = args.seed or raw_cfg.get("seed", 42)

    device = args.device or raw_cfg.get("model", {}).get("device", "cuda")

    # Determine layers
    if args.quick:
        layers = [0, 2]
    elif args.layers:
        layers = args.layers
    else:
        model_layers = raw_cfg.get("model", {}).get("layers", LAYERS)
        layers = model_layers if model_layers else LAYERS

    # Determine seq_lens
    if args.seq_lens:
        seq_lens = args.seq_lens
    else:
        seq_lens = [raw_cfg.get("dataset", {}).get("seq_len", 512)]

    # Determine combos
    if args.quick:
        combos = {k: v for k, v in CHANNEL_COMBOS.items() if k in UNIVARIATE_COMBOS}
        top_k = 0
    elif args.combos:
        if "univariate" in args.combos:
            combo_names = UNIVARIATE_COMBOS + [c for c in args.combos if c != "univariate"]
        else:
            combo_names = args.combos
        combos = {k: CHANNEL_COMBOS[k] for k in combo_names if k in CHANNEL_COMBOS}
        unknown = [k for k in combo_names if k not in CHANNEL_COMBOS and k != "univariate"]
        if unknown:
            logger.warning("Unknown combos: %s. Available: %s", unknown, list(CHANNEL_COMBOS))
        top_k = args.top_k
    else:
        combos = CHANNEL_COMBOS
        top_k = args.top_k

    # Log experiment plan
    total = len(combos) * len(layers) * len(seq_lens)
    logger.info("=" * 60)
    logger.info("Channel Ablation — Enter/Leave Event Detection")
    logger.info("=" * 60)
    logger.info("  Combos: %d (%s)", len(combos), list(combos.keys()))
    logger.info("  Layers: %s", layers)
    logger.info("  seq_lens: %s", seq_lens)
    logger.info("  Grid: %d × %d × %d = %d experiments", len(combos), len(layers), len(seq_lens), total)
    logger.info("  Top-K: %d (full eval with 3 classifiers × 3 CV)", top_k)
    logger.info("  Device: %s, Seed: %d", device, seed)
    logger.info("=" * 60)

    grid_results, top10_results = run_ablation(
        raw_cfg,
        combos=combos,
        layers=layers,
        seq_lens=seq_lens,
        include_none=args.include_none,
        device=device,
        seed=seed,
        no_plots=args.no_plots,
        top_k=top_k,
    )

    if grid_results:
        best = max(grid_results, key=lambda r: r.get("roc_auc") or 0)
        print(f"\nBest: {best['combo']} L{best['layer']} seq={best['seq_len']} "
              f"AUC={best.get('roc_auc')} F1={best.get('f1_macro')}")


if __name__ == "__main__":
    main()
