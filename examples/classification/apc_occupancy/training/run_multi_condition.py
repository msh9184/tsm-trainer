"""Multi-condition experiment: seq_len × channel_combo × layer grid sweep.

Systematic grid search across 3 dimensions to investigate how context
window length affects MantisV2 zero-shot classification across different
sensor combinations and transformer layers.

Dimensions:
  - 6 seq_len values: 32, 64, 128, 288, 384, 512
  - 6 channel combinations: T1, M, T1+T2, M+P, T1+M+P, M+P+T1+T2
  - 6 layers: L0~L5
  - Total: 216 experiments (RF classifier primary)

The top-10 configs by AUC are then evaluated with all 3 classifiers
(NearestCentroid, RandomForest, SVM) for a comprehensive comparison.

MantisV2 inference is the bottleneck, so embeddings are cached by
(layer, channel_combo, seq_len) key — each unique combo is computed
only once, and RF fit/predict runs in <1s per experiment.

Output structure (results/multi_condition/):
  grid_results.csv              216 rows: full grid sweep
  top10_full_eval.csv           Top-10 × 3 classifiers = 30 rows
  config.json                   Experiment configuration
  summary_report.txt            Text analysis report
  plots/
    heatmap_L{0..5}.png         seq_len × channel AUC heatmap per layer
    trend_seqlen.png            AUC vs seq_len trend lines (best layer)
    trend_seqlen_by_layer.png   AUC vs seq_len grouped by layer
    layer_sensitivity.png       AUC vs layer for each condition
    best_configs.png            Overall top-10 bar chart
  tables/
    grid_results.csv            Full grid (duplicate for convenience)
    best_per_seqlen.csv         Best config for each seq_len
    best_per_combo.csv          Best config for each channel combo
    seq_len_statistics.csv      Mean/std AUC per seq_len

Usage:
    cd examples/classification/apc_occupancy

    # Full grid sweep (216 experiments)
    python training/run_multi_condition.py \\
        --config training/configs/p4-zeroshot.yaml

    # Quick test (subset)
    python training/run_multi_condition.py \\
        --config training/configs/p4-zeroshot.yaml \\
        --seq-lens 128 288 512 \\
        --layers 0 2 5

    # No plots (faster, for GPU batch)
    python training/run_multi_condition.py \\
        --config training/configs/p4-zeroshot.yaml --no-plots

    # Custom output directory
    python training/run_multi_condition.py \\
        --config training/configs/p4-zeroshot.yaml \\
        --output-dir results/multi_condition_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
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
sys.path.insert(0, str(_PROJECT_DIR))

from data.preprocess import PreprocessConfig, load_sensor_and_labels
from data.dataset import DatasetConfig, OccupancyDataset
from evaluation.metrics import compute_metrics
from visualization.style import setup_style, save_figure, configure_output

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# 6 key sensor channels with short names
KEY_CHANNELS: dict[str, str] = {
    "d620900d_motionSensor": "M",
    "f2e891c6_powerMeter": "P",
    "d620900d_temperatureMeasurement": "T1",
    "ccea734e_temperatureMeasurement": "T2",
    "408981c2_contactSensor": "C",
    "f2e891c6_energyMeter": "E",
}

# 6 channel combinations to evaluate
CHANNEL_COMBOS: dict[str, list[str]] = {
    "T1": ["d620900d_temperatureMeasurement"],
    "M": ["d620900d_motionSensor"],
    "T1+T2": [
        "d620900d_temperatureMeasurement",
        "ccea734e_temperatureMeasurement",
    ],
    "M+P": [
        "d620900d_motionSensor",
        "f2e891c6_powerMeter",
    ],
    "T1+M+P": [
        "d620900d_temperatureMeasurement",
        "d620900d_motionSensor",
        "f2e891c6_powerMeter",
    ],
    "M+P+T1+T2": [
        "d620900d_motionSensor",
        "f2e891c6_powerMeter",
        "d620900d_temperatureMeasurement",
        "ccea734e_temperatureMeasurement",
    ],
}

# seq_len values (all multiples of 32, 5-min interval data)
# 32=2h40m, 64=5h20m, 128=10h40m, 288=24h, 384=32h, 512=42h40m
SEQ_LENS = [32, 64, 128, 288, 384, 512]

LAYERS = [0, 1, 2, 3, 4, 5]

# Grid result CSV field names
GRID_FIELDS = [
    "seq_len", "combo", "n_channels", "layer", "emb_dim",
    "auc", "eer", "eer_threshold", "f1", "accuracy",
    "precision", "recall", "n_train", "n_test", "embed_time",
]

# Top-10 full eval CSV field names
TOP10_FIELDS = [
    "rank", "seq_len", "combo", "n_channels", "layer", "emb_dim",
    "classifier", "auc", "eer", "eer_threshold", "f1", "accuracy",
    "precision", "recall",
]

# Readable descriptions for seq_len values
SEQ_LEN_DESC: dict[int, str] = {
    32: "2h40m",
    64: "5h20m",
    128: "10h40m",
    288: "24h",
    384: "32h",
    512: "42h40m",
}


# ============================================================================
# Data loading (reused from run_phase1_final.py)
# ============================================================================

def load_config(config_path: str | Path) -> dict:
    """Load raw YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(raw_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load P4 sensor data and labels (all channels, nan_threshold=0.3)."""
    data_cfg = raw_cfg.get("data", {})
    _preprocess_fields = set(PreprocessConfig.__dataclass_fields__)
    base_cfg = {k: v for k, v in data_cfg.items() if k in _preprocess_fields}

    # Load ALL numeric channels (remove explicit channel filter from YAML)
    base_cfg.pop("channels", None)
    base_cfg["nan_threshold"] = 0.3

    # Train labels
    train_cfg = PreprocessConfig(**base_cfg)
    sensor_array, train_labels, channel_names, sensor_ts, _ = load_sensor_and_labels(train_cfg)

    # Test labels
    test_cfg_dict = dict(base_cfg)
    test_cfg_dict["label_csv"] = data_cfg.get("test_label_csv", "")
    test_cfg = PreprocessConfig(**test_cfg_dict)
    _, test_labels, _, _, _ = load_sensor_and_labels(test_cfg)

    # Resolve overlap
    train_mask = train_labels >= 0
    test_mask = test_labels >= 0
    overlap = (train_mask & test_mask).sum()
    if overlap > 0:
        test_labels = test_labels.copy()
        test_labels[train_mask] = -1
        logger.info("Removed %d overlapping timesteps from test set", overlap)

    return sensor_array, train_labels, test_labels, channel_names


# ============================================================================
# Dataset construction
# ============================================================================

def build_multi_channel_dataset(
    sensor_array: np.ndarray,
    labels: np.ndarray,
    channel_names: list[str],
    target_channels: list[str],
    seq_len: int = 288,
    stride: int = 1,
    target_seq_len: int | None = 512,
) -> OccupancyDataset:
    """Build dataset for one or more channels.

    Parameters
    ----------
    sensor_array : np.ndarray
        Shape (T, C_all) — full sensor array with all channels.
    labels : np.ndarray
        Shape (T,) — train or test labels (-1 = unlabeled).
    channel_names : list[str]
        All channel names from sensor_array.
    target_channels : list[str]
        Subset of channels to use.
    seq_len : int
        Window length in time steps.
    stride : int
        Stride between windows.
    target_seq_len : int or None
        If set, interpolate to this length. None = no interpolation.
    """
    ch_indices = [channel_names.index(ch) for ch in target_channels]
    multi_ch_array = sensor_array[:, ch_indices]  # (T, len(target_channels))

    # No interpolation at native MantisV2 length (512)
    tgt = target_seq_len if seq_len != 512 else None

    ds_cfg = DatasetConfig(seq_len=seq_len, stride=stride, target_seq_len=tgt)
    return OccupancyDataset(
        sensor_array=multi_ch_array,
        label_array=labels,
        config=ds_cfg,
    )


# ============================================================================
# Model loading
# ============================================================================

def load_mantis_model(layer: int, device: str = "cuda"):
    """Load MantisV2 at a specific transformer layer.

    Returns (network, trainer) where trainer.transform() extracts embeddings.
    """
    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    network = MantisV2(
        device=device,
        return_transf_layer=layer,
        output_token="combined",
    )
    network = network.from_pretrained("paris-noah/MantisV2")
    model = MantisTrainer(device=device, network=network)
    return network, model


# ============================================================================
# Embedding extraction with caching
# ============================================================================

def extract_embeddings(
    model,
    train_ds: OccupancyDataset,
    test_ds: OccupancyDataset,
    cache: dict,
    cache_key: tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract embeddings with cache lookup.

    Parameters
    ----------
    model : MantisTrainer
        Model to use for transform().
    train_ds, test_ds : OccupancyDataset
        Datasets to extract embeddings from.
    cache : dict
        Embedding cache: key → (Z_train, y_train, Z_test, y_test).
    cache_key : tuple
        Cache key (layer, combo_name, seq_len).

    Returns
    -------
    Z_train, y_train, Z_test, y_test : np.ndarray
    """
    if cache_key in cache:
        return cache[cache_key]

    X_train, y_train = train_ds.get_numpy_arrays()
    X_test, y_test = test_ds.get_numpy_arrays()

    t0 = time.time()
    Z_train = model.transform(X_train)
    Z_test = model.transform(X_test)
    elapsed = time.time() - t0

    logger.info(
        "  Embeddings [%s]: train=%s test=%s (%.1fs)",
        cache_key, Z_train.shape, Z_test.shape, elapsed,
    )

    result = (Z_train, y_train, Z_test, y_test, elapsed)
    cache[cache_key] = result
    return result


# ============================================================================
# Classifier utilities
# ============================================================================

def build_rf_classifier(seed: int = 42):
    """Build RandomForest classifier (primary for grid sweep)."""
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)


def build_all_classifiers(seed: int = 42) -> dict:
    """Build all 3 classifiers for full evaluation."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import NearestCentroid
    from sklearn.svm import SVC

    return {
        "NearestCentroid": NearestCentroid(),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed,
        ),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=seed),
    }


def run_rf_experiment(
    Z_train: np.ndarray, y_train: np.ndarray,
    Z_test: np.ndarray, y_test: np.ndarray,
    seed: int = 42,
) -> dict:
    """Run RF classifier and compute all metrics. Returns dict of results."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_test_s = scaler.transform(Z_test)

    rf = build_rf_classifier(seed)
    rf.fit(Z_train_s, y_train)
    y_pred = rf.predict(Z_test_s)
    y_prob = rf.predict_proba(Z_test_s)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    return {
        "auc": metrics.roc_auc,
        "eer": metrics.eer,
        "eer_threshold": metrics.eer_threshold,
        "f1": metrics.f1,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
    }


def run_all_classifiers(
    Z_train: np.ndarray, y_train: np.ndarray,
    Z_test: np.ndarray, y_test: np.ndarray,
    seed: int = 42,
) -> dict[str, dict]:
    """Run all 3 classifiers. Returns {clf_name: metrics_dict}."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_test_s = scaler.transform(Z_test)

    classifiers = build_all_classifiers(seed)
    results = {}

    for clf_name, clf in classifiers.items():
        clf.fit(Z_train_s, y_train)
        y_pred = clf.predict(Z_test_s)
        y_prob = None
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(Z_test_s)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        results[clf_name] = {
            "auc": metrics.roc_auc,
            "eer": metrics.eer,
            "eer_threshold": metrics.eer_threshold,
            "f1": metrics.f1,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
        }

    return results


# ============================================================================
# Top config evaluation
# ============================================================================

def run_top_configs_full_eval(
    results: list[dict],
    embedding_cache: dict,
    top_k: int = 10,
    seed: int = 42,
) -> list[dict]:
    """Run all 3 classifiers on top-K configs by AUC.

    Reuses cached embeddings from the grid sweep.
    """
    # Sort by AUC descending, take top-K
    sorted_results = sorted(
        results, key=lambda r: r.get("auc", 0), reverse=True,
    )
    top_configs = sorted_results[:top_k]

    full_eval_results = []

    for rank, config in enumerate(top_configs, 1):
        cache_key = (config["layer"], config["combo"], config["seq_len"])

        if cache_key not in embedding_cache:
            logger.warning("Cache miss for %s, skipping full eval", cache_key)
            continue

        cached = embedding_cache[cache_key]
        Z_train, y_train, Z_test, y_test = cached[0], cached[1], cached[2], cached[3]

        clf_results = run_all_classifiers(Z_train, y_train, Z_test, y_test, seed)

        for clf_name, metrics in clf_results.items():
            full_eval_results.append({
                "rank": rank,
                "seq_len": config["seq_len"],
                "combo": config["combo"],
                "n_channels": config["n_channels"],
                "layer": config["layer"],
                "emb_dim": config["emb_dim"],
                "classifier": clf_name,
                **metrics,
            })

        best_clf = max(clf_results.items(), key=lambda x: x[1].get("auc", 0))
        print(
            f"  Top-{rank}: seq={config['seq_len']} {config['combo']} L{config['layer']} "
            f"best={best_clf[0]} AUC={best_clf[1]['auc']:.4f}"
        )

    return full_eval_results


# ============================================================================
# Visualization
# ============================================================================

def plot_heatmap_seqlen_channel(
    results: list[dict],
    layer: int,
    output_path: Path,
) -> None:
    """Heatmap: seq_len (rows) × channel_combo (cols), colored by AUC.

    One plot per layer.
    """
    setup_style()

    # Collect unique sorted values
    seq_lens_present = sorted(set(r["seq_len"] for r in results))
    combos_present = list(dict.fromkeys(r["combo"] for r in results))

    # Build matrix
    matrix = np.full((len(seq_lens_present), len(combos_present)), np.nan)
    for r in results:
        if r["layer"] != layer:
            continue
        i = seq_lens_present.index(r["seq_len"])
        j = combos_present.index(r["combo"])
        matrix[i, j] = r["auc"]

    h = max(3.5, len(seq_lens_present) * 0.65)
    w = max(6, len(combos_present) * 1.0 + 2)
    fig, ax = plt.subplots(figsize=(w, h))

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                txt = "-"
                color = "grey"
            else:
                txt = f"{val:.3f}"
                color = "white" if val < 0.65 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    row_labels = [f"{s} ({SEQ_LEN_DESC.get(s, '')})" for s in seq_lens_present]
    ax.set_xticks(range(len(combos_present)))
    ax.set_xticklabels(combos_present, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(seq_lens_present)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_ylabel("seq_len (context window)")
    ax.set_xlabel("Channel Combination")
    ax.set_title(f"AUC Heatmap — Layer {layer} (RF Classifier)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="AUC")

    save_figure(fig, output_path)
    plt.close(fig)


def plot_trend_seqlen(
    results: list[dict],
    output_path: Path,
    fixed_layer: int | None = None,
) -> None:
    """Line plot: AUC vs seq_len, one line per channel combo.

    If fixed_layer is None, uses the best layer for each (combo, seq_len).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    combos_present = list(dict.fromkeys(r["combo"] for r in results))
    colors = plt.cm.Set2(np.linspace(0, 1, len(combos_present)))

    for idx, combo in enumerate(combos_present):
        combo_results = [r for r in results if r["combo"] == combo]
        if fixed_layer is not None:
            combo_results = [r for r in combo_results if r["layer"] == fixed_layer]

        # Group by seq_len, take best AUC per seq_len
        seq_auc = {}
        for r in combo_results:
            s = r["seq_len"]
            if s not in seq_auc or r["auc"] > seq_auc[s]:
                seq_auc[s] = r["auc"]

        if not seq_auc:
            continue

        xs = sorted(seq_auc.keys())
        ys = [seq_auc[s] for s in xs]

        ax.plot(xs, ys, marker="o", color=colors[idx], lw=2, markersize=6,
                label=combo)

    ax.set_xlabel("seq_len (time steps)")
    ax.set_ylabel("AUC (best across layers)" if fixed_layer is None
                   else f"AUC (Layer {fixed_layer})")
    title = "AUC vs Context Length — Best Layer per Config"
    if fixed_layer is not None:
        title = f"AUC vs Context Length — Layer {fixed_layer}"
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(SEQ_LENS)
    ax.set_xticklabels([f"{s}\n({SEQ_LEN_DESC.get(s, '')})" for s in SEQ_LENS],
                        fontsize=7)

    save_figure(fig, output_path)
    plt.close(fig)


def plot_trend_seqlen_by_layer(
    results: list[dict],
    output_path: Path,
) -> None:
    """Multi-panel: AUC vs seq_len, grouped by layer (2×3 subplots)."""
    setup_style()
    layers_present = sorted(set(r["layer"] for r in results))
    n_layers = len(layers_present)

    nrows = (n_layers + 2) // 3
    ncols = min(3, n_layers)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)

    combos_present = list(dict.fromkeys(r["combo"] for r in results))
    colors = plt.cm.Set2(np.linspace(0, 1, len(combos_present)))

    for ax_idx, layer in enumerate(layers_present):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        layer_results = [r for r in results if r["layer"] == layer]

        for cidx, combo in enumerate(combos_present):
            combo_results = [r for r in layer_results if r["combo"] == combo]
            if not combo_results:
                continue

            xs = sorted(set(r["seq_len"] for r in combo_results))
            ys = [next(r["auc"] for r in combo_results if r["seq_len"] == s)
                  for s in xs]

            ax.plot(xs, ys, marker="o", color=colors[cidx], lw=1.5,
                    markersize=4, label=combo)

        ax.set_title(f"Layer {layer}", fontsize=10)
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("seq_len")
        ax.set_ylabel("AUC")
        if ax_idx == 0:
            ax.legend(fontsize=6, loc="lower right")

    # Hide empty subplots
    for ax_idx in range(n_layers, nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    fig.suptitle("AUC vs Context Length by Layer", fontsize=12, y=1.02)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_layer_sensitivity(
    results: list[dict],
    output_path: Path,
) -> None:
    """Line plot: AUC vs layer for each (seq_len, combo) condition.

    Uses only the best seq_len per combo for clarity.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    combos_present = list(dict.fromkeys(r["combo"] for r in results))
    colors = plt.cm.Set2(np.linspace(0, 1, len(combos_present)))

    for cidx, combo in enumerate(combos_present):
        combo_results = [r for r in results if r["combo"] == combo]

        # Find the best seq_len for this combo (highest mean AUC across layers)
        seq_lens_avail = sorted(set(r["seq_len"] for r in combo_results))
        best_seq_len = max(
            seq_lens_avail,
            key=lambda s: np.mean([
                r["auc"] for r in combo_results if r["seq_len"] == s
            ]),
        )

        # Plot AUC vs layer for best seq_len
        layer_results = [r for r in combo_results if r["seq_len"] == best_seq_len]
        layer_results.sort(key=lambda r: r["layer"])
        xs = [r["layer"] for r in layer_results]
        ys = [r["auc"] for r in layer_results]

        ax.plot(xs, ys, marker="o", color=colors[cidx], lw=2, markersize=6,
                label=f"{combo} (seq={best_seq_len})")

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("AUC")
    ax.set_title("Layer Sensitivity — AUC vs Layer (best seq_len per combo)", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{l}" for l in LAYERS])
    ax.set_ylim(0.4, 1.0)

    save_figure(fig, output_path)
    plt.close(fig)


def plot_best_configs_bar(
    results: list[dict],
    output_path: Path,
    top_k: int = 10,
) -> None:
    """Horizontal bar chart of top-K configs by AUC."""
    setup_style()
    sorted_results = sorted(results, key=lambda r: r.get("auc", 0), reverse=True)
    top = sorted_results[:top_k]

    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.5)))

    labels = [
        f"seq={r['seq_len']} {r['combo']} L{r['layer']}"
        for r in top
    ]
    aucs = [r["auc"] for r in top]

    # Highlight the best one
    colors = ["#DE8F05"] + ["#0173B2"] * (len(top) - 1)

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, aucs, color=colors, alpha=0.85, edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUC")
    ax.set_title(f"Top {len(top)} Configurations by AUC", fontsize=11)

    # Adaptive x-axis range
    min_auc = min(aucs) if aucs else 0.5
    ax.set_xlim(max(0.5, min_auc - 0.05), 1.0)
    ax.invert_yaxis()

    for bar, val in zip(bars, aucs):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", ha="left", va="center", fontsize=8)

    save_figure(fig, output_path)
    plt.close(fig)


# ============================================================================
# Result I/O
# ============================================================================

def _nan_safe(v):
    """Convert NaN/Inf to None for JSON serialisation."""
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.floating,)):
        v = float(v)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def save_grid_csv(results: list[dict], path: Path) -> None:
    """Save grid results as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GRID_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: _nan_safe(r.get(k, "")) for k in GRID_FIELDS})
    logger.info("Saved grid CSV: %s (%d rows)", path, len(results))


def save_top10_csv(results: list[dict], path: Path) -> None:
    """Save top-10 full eval results as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TOP10_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: _nan_safe(r.get(k, "")) for k in TOP10_FIELDS})
    logger.info("Saved top-10 CSV: %s (%d rows)", path, len(results))


def save_config_json(config: dict, path: Path) -> None:
    """Save experiment configuration as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = {k: _nan_safe(v) if isinstance(v, float) else v
                for k, v in config.items()}
    with open(path, "w") as f:
        json.dump(cleaned, f, indent=2)


def save_analysis_tables(results: list[dict], tables_dir: Path) -> None:
    """Save derived analysis tables from grid results."""
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full grid (copy for convenience)
    save_grid_csv(results, tables_dir / "grid_results.csv")

    # 2. Best config per seq_len
    best_per_seqlen = {}
    for r in results:
        s = r["seq_len"]
        if s not in best_per_seqlen or r["auc"] > best_per_seqlen[s]["auc"]:
            best_per_seqlen[s] = r

    rows = [
        {"seq_len": s, "seq_len_desc": SEQ_LEN_DESC.get(s, ""),
         "combo": v["combo"], "layer": v["layer"],
         "auc": round(v["auc"], 4), "eer": round(v["eer"], 4),
         "f1": round(v["f1"], 4)}
        for s, v in sorted(best_per_seqlen.items())
    ]
    path = tables_dir / "best_per_seqlen.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # 3. Best config per combo
    best_per_combo = {}
    for r in results:
        c = r["combo"]
        if c not in best_per_combo or r["auc"] > best_per_combo[c]["auc"]:
            best_per_combo[c] = r

    rows = [
        {"combo": c, "seq_len": v["seq_len"], "layer": v["layer"],
         "auc": round(v["auc"], 4), "eer": round(v["eer"], 4),
         "f1": round(v["f1"], 4)}
        for c, v in sorted(best_per_combo.items(),
                           key=lambda x: x[1]["auc"], reverse=True)
    ]
    path = tables_dir / "best_per_combo.csv"
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # 4. seq_len statistics (mean/std AUC per seq_len)
    seq_stats = {}
    for r in results:
        s = r["seq_len"]
        seq_stats.setdefault(s, []).append(r["auc"])

    rows = [
        {"seq_len": s, "seq_len_desc": SEQ_LEN_DESC.get(s, ""),
         "mean_auc": round(float(np.nanmean(aucs)), 4),
         "std_auc": round(float(np.nanstd(aucs)), 4),
         "min_auc": round(float(np.nanmin(aucs)), 4),
         "max_auc": round(float(np.nanmax(aucs)), 4),
         "n_experiments": len(aucs)}
        for s, aucs in sorted(seq_stats.items())
    ]
    path = tables_dir / "seq_len_statistics.csv"
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# ============================================================================
# Summary report
# ============================================================================

def generate_summary_report(
    results: list[dict],
    top10_results: list[dict],
    total_time: float,
    seq_lens: list[int],
    combos: dict[str, list[str]],
    layers: list[int],
) -> str:
    """Generate text summary report of multi-condition experiment."""
    lines = []
    lines.append("=" * 72)
    lines.append("MULTI-CONDITION EXPERIMENT REPORT")
    lines.append("seq_len × channel_combo × layer Grid Sweep")
    lines.append("=" * 72)
    lines.append("")

    # Section 1: Configuration
    lines.append("1. EXPERIMENT CONFIGURATION")
    lines.append("-" * 40)
    lines.append(f"  seq_len values: {seq_lens}")
    desc_str = ", ".join(f"{s}={SEQ_LEN_DESC.get(s, '?')}" for s in seq_lens)
    lines.append(f"    Time coverage: {desc_str}")
    lines.append(f"  Channel combos: {list(combos.keys())}")
    lines.append(f"  Layers: L{min(layers)}~L{max(layers)} ({len(layers)} layers)")
    lines.append(f"  Total grid size: {len(seq_lens)} × {len(combos)} × {len(layers)} "
                 f"= {len(seq_lens) * len(combos) * len(layers)}")
    lines.append(f"  Completed: {len(results)} experiments")
    lines.append(f"  Model: MantisV2 (4.2M params, pretrained)")
    lines.append(f"  Primary classifier: RandomForest (200 trees)")
    lines.append("")

    # Section 2: Overall best
    if results:
        best = max(results, key=lambda r: r.get("auc", 0))
        lines.append("2. OVERALL BEST CONFIGURATION")
        lines.append("-" * 40)
        lines.append(f"  seq_len: {best['seq_len']} ({SEQ_LEN_DESC.get(best['seq_len'], '')})")
        lines.append(f"  Channels: {best['combo']} ({best['n_channels']} ch)")
        lines.append(f"  Layer: L{best['layer']}")
        lines.append(f"  AUC: {best['auc']:.4f}")
        lines.append(f"  EER: {best['eer']:.4f}")
        lines.append(f"  F1: {best['f1']:.4f}")
        lines.append(f"  Embedding dim: {best['emb_dim']}")
        lines.append("")

    # Section 3: Best per seq_len
    lines.append("3. BEST CONFIG PER SEQ_LEN")
    lines.append("-" * 40)
    lines.append(f"  {'seq_len':<10s} {'time':<8s} {'combo':<12s} {'layer':<6s} "
                 f"{'AUC':>7s} {'EER':>7s} {'F1':>7s}")
    lines.append("  " + "-" * 58)

    best_per_sl = {}
    for r in results:
        s = r["seq_len"]
        if s not in best_per_sl or r["auc"] > best_per_sl[s]["auc"]:
            best_per_sl[s] = r

    for s in sorted(best_per_sl.keys()):
        r = best_per_sl[s]
        lines.append(
            f"  {s:<10d} {SEQ_LEN_DESC.get(s, ''):<8s} {r['combo']:<12s} L{r['layer']:<5d} "
            f"{r['auc']:>7.4f} {r['eer']:>7.4f} {r['f1']:>7.4f}"
        )
    lines.append("")

    # Section 4: Best per combo
    lines.append("4. BEST CONFIG PER CHANNEL COMBO")
    lines.append("-" * 40)
    lines.append(f"  {'combo':<12s} {'n_ch':<5s} {'seq_len':<8s} {'layer':<6s} "
                 f"{'AUC':>7s} {'EER':>7s} {'F1':>7s}")
    lines.append("  " + "-" * 55)

    best_per_combo = {}
    for r in results:
        c = r["combo"]
        if c not in best_per_combo or r["auc"] > best_per_combo[c]["auc"]:
            best_per_combo[c] = r

    for c, r in sorted(best_per_combo.items(),
                       key=lambda x: x[1]["auc"], reverse=True):
        lines.append(
            f"  {c:<12s} {r['n_channels']:<5d} {r['seq_len']:<8d} L{r['layer']:<5d} "
            f"{r['auc']:>7.4f} {r['eer']:>7.4f} {r['f1']:>7.4f}"
        )
    lines.append("")

    # Section 5: seq_len statistics
    lines.append("5. SEQ_LEN STATISTICS (mean ± std AUC)")
    lines.append("-" * 40)

    seq_stats = {}
    for r in results:
        seq_stats.setdefault(r["seq_len"], []).append(r["auc"])

    for s in sorted(seq_stats.keys()):
        aucs = seq_stats[s]
        lines.append(
            f"  seq_len={s:3d} ({SEQ_LEN_DESC.get(s, ''):>7s}): "
            f"mean={np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f}  "
            f"[{np.nanmin(aucs):.4f}, {np.nanmax(aucs):.4f}]  "
            f"(n={len(aucs)})"
        )
    lines.append("")

    # Section 6: Top-10 with all classifiers
    if top10_results:
        lines.append("6. TOP-10 CONFIGS — ALL CLASSIFIERS")
        lines.append("-" * 40)

        current_rank = None
        for r in top10_results:
            if r["rank"] != current_rank:
                current_rank = r["rank"]
                lines.append(
                    f"\n  #{r['rank']} seq={r['seq_len']} {r['combo']} L{r['layer']} "
                    f"({r['n_channels']}ch, {r['emb_dim']}d)"
                )
            lines.append(
                f"    {r['classifier']:<18s} AUC={r['auc']:.4f}  "
                f"EER={r['eer']:.4f}  F1={r['f1']:.4f}"
            )
        lines.append("")

    # Section 7: Key findings
    lines.append("7. KEY FINDINGS")
    lines.append("-" * 40)

    if results:
        # Analyze seq_len effect
        mean_by_sl = {
            s: np.nanmean(aucs) for s, aucs in seq_stats.items()
        }
        best_sl = max(mean_by_sl, key=mean_by_sl.get)
        worst_sl = min(mean_by_sl, key=mean_by_sl.get)
        lines.append(
            f"  (a) Best mean seq_len: {best_sl} ({SEQ_LEN_DESC.get(best_sl, '')}) "
            f"avg AUC={mean_by_sl[best_sl]:.4f}"
        )
        lines.append(
            f"      Worst mean seq_len: {worst_sl} ({SEQ_LEN_DESC.get(worst_sl, '')}) "
            f"avg AUC={mean_by_sl[worst_sl]:.4f}"
        )

        # Analyze combo effect
        combo_stats = {}
        for r in results:
            combo_stats.setdefault(r["combo"], []).append(r["auc"])
        mean_by_combo = {c: np.nanmean(a) for c, a in combo_stats.items()}
        best_combo = max(mean_by_combo, key=mean_by_combo.get)
        lines.append(
            f"  (b) Best mean combo: {best_combo} avg AUC={mean_by_combo[best_combo]:.4f}"
        )

        # Layer preference
        layer_stats = {}
        for r in results:
            layer_stats.setdefault(r["layer"], []).append(r["auc"])
        mean_by_layer = {l: np.nanmean(a) for l, a in layer_stats.items()}
        best_layer = max(mean_by_layer, key=mean_by_layer.get)
        lines.append(
            f"  (c) Best mean layer: L{best_layer} avg AUC={mean_by_layer[best_layer]:.4f}"
        )

    lines.append("")
    lines.append(f"Total experiment time: {total_time:.1f}s ({total_time/60:.1f} min)")
    lines.append("=" * 72)

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-condition experiment: seq_len × channel × layer grid sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="YAML config path (e.g., training/configs/p4-zeroshot.yaml)",
    )
    parser.add_argument(
        "--output-dir", default="results/multi_condition",
        help="Output directory (default: results/multi_condition)",
    )
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=None,
        help=f"Subset of seq_len values (default: {SEQ_LENS})",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help=f"Subset of layers (default: {LAYERS})",
    )
    parser.add_argument(
        "--combos", type=str, nargs="+", default=None,
        help=f"Subset of channel combos (default: {list(CHANNEL_COMBOS.keys())})",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation (faster, for batch execution)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of top configs for full classifier eval (default: 10)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Override device (default: from config or cuda)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    t_start = time.time()

    # ----------------------------------------------------------------
    # Parse config
    # ----------------------------------------------------------------
    raw_cfg = load_config(args.config)
    device = args.device or raw_cfg.get("model", {}).get("device", "cuda")
    seed = raw_cfg.get("seed", 42)

    seq_lens = args.seq_lens or SEQ_LENS
    layers = args.layers or LAYERS
    combos = CHANNEL_COMBOS
    if args.combos:
        combos = {k: v for k, v in CHANNEL_COMBOS.items() if k in args.combos}
        if not combos:
            parser.error(
                f"No valid combos. Choose from: {list(CHANNEL_COMBOS.keys())}"
            )

    # Validate seq_lens
    for s in seq_lens:
        if s % 32 != 0:
            parser.error(f"seq_len must be a multiple of 32, got {s}")

    # Output dirs
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    for d in [output_dir, plots_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    configure_output(formats=["png"], dpi=300)

    total_grid = len(seq_lens) * len(combos) * len(layers)

    print("=" * 72)
    print("MULTI-CONDITION EXPERIMENT")
    print(f"Grid: {len(seq_lens)} seq_lens × {len(combos)} combos × {len(layers)} layers "
          f"= {total_grid} experiments")
    print("=" * 72)

    # ----------------------------------------------------------------
    # Step 1: Load data
    # ----------------------------------------------------------------
    print("\n[STEP 1] Loading data (all channels)...")
    sensor_array, train_labels, test_labels, channel_names = load_data(raw_cfg)
    print(f"  Sensor shape: {sensor_array.shape}")
    print(f"  Channels ({len(channel_names)}): {channel_names}")
    print(f"  Train labels: {(train_labels >= 0).sum()}, "
          f"Test labels: {(test_labels >= 0).sum()}")

    # ----------------------------------------------------------------
    # Step 2: Grid sweep
    # ----------------------------------------------------------------
    print(f"\n[STEP 2] Running grid sweep ({total_grid} experiments)...")

    # We need to access the embedding cache after sweep for top-K eval
    embedding_cache: dict[tuple, tuple] = {}

    # Monkey-patch to capture cache from inside run_grid_sweep
    # Instead, we refactor to pass cache in and out
    results = _run_grid_sweep_with_cache(
        sensor_array, train_labels, test_labels, channel_names,
        seq_lens, combos, layers, embedding_cache,
        device=device, seed=seed,
    )

    # Save grid results
    save_grid_csv(results, output_dir / "grid_results.csv")

    # ----------------------------------------------------------------
    # Step 3: Top-K full evaluation
    # ----------------------------------------------------------------
    print(f"\n[STEP 3] Running top-{args.top_k} full classifier evaluation...")

    top10_results = run_top_configs_full_eval(
        results, embedding_cache, top_k=args.top_k, seed=seed,
    )
    if top10_results:
        save_top10_csv(top10_results, output_dir / "top10_full_eval.csv")

    # ----------------------------------------------------------------
    # Step 4: Visualizations
    # ----------------------------------------------------------------
    if not args.no_plots and results:
        print("\n[STEP 4] Generating visualizations...")

        # 4a: Heatmaps per layer
        for layer in layers:
            plot_heatmap_seqlen_channel(
                results, layer,
                plots_dir / f"heatmap_L{layer}",
            )
        print(f"  Heatmaps: {len(layers)} layer plots saved")

        # 4b: Trend lines (best layer)
        plot_trend_seqlen(results, plots_dir / "trend_seqlen")
        print("  Trend: AUC vs seq_len saved")

        # 4c: Trend lines by layer (2×3 panel)
        plot_trend_seqlen_by_layer(results, plots_dir / "trend_seqlen_by_layer")
        print("  Trend by layer: multi-panel saved")

        # 4d: Layer sensitivity
        plot_layer_sensitivity(results, plots_dir / "layer_sensitivity")
        print("  Layer sensitivity saved")

        # 4e: Top configs bar chart
        plot_best_configs_bar(results, plots_dir / "best_configs", top_k=args.top_k)
        print("  Best configs bar chart saved")
    elif args.no_plots:
        print("\n[STEP 4] Skipping visualizations (--no-plots)")

    # ----------------------------------------------------------------
    # Step 5: Save tables and reports
    # ----------------------------------------------------------------
    print("\n[STEP 5] Saving analysis tables...")
    save_analysis_tables(results, tables_dir)

    # Save experiment config
    save_config_json({
        "seq_lens": seq_lens,
        "channel_combos": {k: v for k, v in combos.items()},
        "layers": layers,
        "device": device,
        "seed": seed,
        "total_experiments": total_grid,
        "completed_experiments": len(results),
        "top_k": args.top_k,
        "config_path": str(args.config),
    }, output_dir / "config.json")

    # Generate summary report
    total_time = time.time() - t_start
    report_text = generate_summary_report(
        results, top10_results, total_time, seq_lens, combos, layers,
    )
    report_path = output_dir / "summary_report.txt"
    report_path.write_text(report_text)

    print(f"\n{report_text}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("MULTI-CONDITION EXPERIMENT COMPLETE")
    print("=" * 72)
    print(f"  Output: {output_dir}/")
    print(f"  Grid:   {output_dir}/grid_results.csv ({len(results)} rows)")
    if top10_results:
        print(f"  Top-10: {output_dir}/top10_full_eval.csv ({len(top10_results)} rows)")
    if not args.no_plots:
        n_plots = len(list(plots_dir.glob("*.png")))
        print(f"  Plots:  {plots_dir}/ ({n_plots} PNG files)")
    print(f"  Tables: {tables_dir}/ ({len(list(tables_dir.glob('*.csv')))} CSV files)")
    print(f"  Report: {report_path}")
    print(f"  Time:   {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 72)


def _run_grid_sweep_with_cache(
    sensor_array: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    channel_names: list[str],
    seq_lens: list[int],
    combos: dict[str, list[str]],
    layers: list[int],
    embedding_cache: dict,
    device: str = "cuda",
    seed: int = 42,
) -> list[dict]:
    """Grid sweep that populates the shared embedding_cache.

    Same as run_grid_sweep but writes to the provided cache dict so
    the caller can reuse embeddings for top-K full evaluation.
    """
    results = []
    total_experiments = len(seq_lens) * len(combos) * len(layers)
    experiment_idx = 0

    # Validate channels
    available_combos = {}
    for combo_name, combo_channels in combos.items():
        missing = [ch for ch in combo_channels if ch not in channel_names]
        if missing:
            logger.warning(
                "Skipping combo %s: missing channels %s", combo_name, missing,
            )
            continue
        available_combos[combo_name] = combo_channels

    if not available_combos:
        raise ValueError("No valid channel combinations found in data")

    # Cache models per layer
    models: dict[int, object] = {}

    for seq_len in seq_lens:
        for combo_name, combo_channels in available_combos.items():
            # Build datasets once per (seq_len, combo)
            try:
                train_ds = build_multi_channel_dataset(
                    sensor_array, train_labels, channel_names,
                    combo_channels, seq_len=seq_len, stride=1,
                    target_seq_len=512,
                )
                test_ds = build_multi_channel_dataset(
                    sensor_array, test_labels, channel_names,
                    combo_channels, seq_len=seq_len, stride=1,
                    target_seq_len=512,
                )
            except ValueError as e:
                logger.warning(
                    "Skip seq_len=%d combo=%s: %s", seq_len, combo_name, e,
                )
                for _ in layers:
                    experiment_idx += 1
                continue

            for layer in layers:
                experiment_idx += 1

                if layer not in models:
                    print(f"  Loading MantisV2 layer {layer}...")
                    _, models[layer] = load_mantis_model(layer, device)

                cache_key = (layer, combo_name, seq_len)

                try:
                    cached = extract_embeddings(
                        models[layer], train_ds, test_ds,
                        embedding_cache, cache_key,
                    )
                    Z_train, y_train, Z_test, y_test, embed_time = cached

                    metrics = run_rf_experiment(
                        Z_train, y_train, Z_test, y_test, seed,
                    )

                    row = {
                        "seq_len": seq_len,
                        "combo": combo_name,
                        "n_channels": len(combo_channels),
                        "layer": layer,
                        "emb_dim": Z_train.shape[1],
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                        "embed_time": round(embed_time, 2),
                        **metrics,
                    }
                    results.append(row)

                    print(
                        f"  [{experiment_idx:3d}/{total_experiments}] "
                        f"seq={seq_len:3d} {combo_name:<10s} L{layer} "
                        f"AUC={metrics['auc']:.4f} EER={metrics['eer']:.4f} "
                        f"F1={metrics['f1']:.4f}"
                    )

                except Exception as e:
                    logger.warning(
                        "Failed: seq_len=%d combo=%s layer=%d: %s",
                        seq_len, combo_name, layer, e,
                    )

    print(f"\nGrid sweep complete: {len(results)} / {total_experiments} experiments")
    return results


if __name__ == "__main__":
    main()
