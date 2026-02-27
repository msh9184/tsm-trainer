"""Phase 1 Zero-Shot Sweep: Layer x Sensor Combination x Classifier.

Comprehensive evaluation of MantisV2 zero-shot embeddings for 3-class
enter/leave/none event detection with LOOCV. Produces output matching
the Chronos-2 baseline experiment format (classification_report + confusion
matrix) for direct comparison.

Sweep dimensions:
  - 6 transformer layers (L0-L5)
  - 15 sensor combinations (7 baseline-equivalent + 8 extended)
  - 3 classifiers (RandomForest, SVM, NearestCentroid)
  - LOOCV evaluation (109 events: 35 Enter + 38 Leave + 36 None)

Output:
  - Baseline-format classification reports (per combo x layer)
  - O/X summary table matching Chronos-2 baseline format
  - t-SNE visualization for top-K configs with 3-class coloring
  - CSV/JSON exports of all results

Usage:
    cd examples/classification/apc_enter_leave

    # Full sweep (6 layers x 15 combos x 3 classifiers = 270 experiments)
    python training/run_phase1_sweep.py \\
        --config training/configs/enter-leave-phase1.yaml

    # Quick test: L0 only, RF only, baseline combos only
    python training/run_phase1_sweep.py --config ... --quick

    # Specific layers
    python training/run_phase1_sweep.py --config ... --layers 0 2 5

    # Baseline combos only (7 combos matching Chronos-2 experiment)
    python training/run_phase1_sweep.py --config ... --baseline-only

    # Binary mode (exclude NONE, 73 events)
    python training/run_phase1_sweep.py --config ... --binary

    # CPU mode
    python training/run_phase1_sweep.py --config ... --device cpu
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
sys.path.insert(0, str(_PROJECT_DIR))

from data.preprocess import EventPreprocessConfig, load_sensor_and_events
from data.dataset import EventDatasetConfig, EventDataset
from visualization.style import setup_style, save_figure, configure_output

logger = logging.getLogger(__name__)


# ============================================================================
# Sensor group and combination definitions
# ============================================================================

# Each sensor group maps to one or more physical channel names in the CSV
SENSOR_GROUPS: dict[str, list[str]] = {
    "M":  ["d620900d_motionSensor"],
    "P":  ["f2e891c6_powerMeter"],
    "T1": ["d620900d_temperatureMeasurement"],
    "T2": ["ccea734e_temperatureMeasurement"],
    "C":  ["408981c2_contactSensor"],
    "E":  ["f2e891c6_energyMeter"],
}

# Channel combinations for the sweep.
# Format: (short_key, display_name, group_keys, is_baseline)
# The 7 baseline combos match the Chronos-2 experiment sensor configurations.
CHANNEL_COMBOS: list[tuple[str, str, list[str], bool]] = [
    # --- Baseline-equivalent (7 combos, same as Chronos-2 experiment) ---
    ("M_P_T", "motionSensor, powerSensor, temperatureMeasurement(2)",
     ["M", "P", "T1", "T2"], True),
    ("M_P", "motionSensor, powerSensor",
     ["M", "P"], True),
    ("M_T", "motionSensor, temperatureMeasurement(2)",
     ["M", "T1", "T2"], True),
    ("P_T", "powerSensor, temperatureMeasurement(2)",
     ["P", "T1", "T2"], True),
    ("M", "motionSensor",
     ["M"], True),
    ("P", "powerSensor",
     ["P"], True),
    ("T", "temperatureMeasurement(2)",
     ["T1", "T2"], True),
    # --- Extended: individual channels not in baseline ---
    ("C", "contactSensor",
     ["C"], False),
    ("E", "energyMeter",
     ["E"], False),
    ("T1", "temperatureMeasurement_d620",
     ["T1"], False),
    ("T2", "temperatureMeasurement_ccea",
     ["T2"], False),
    # --- Extended: key combinations with C, E ---
    ("M_C", "motionSensor, contactSensor",
     ["M", "C"], False),
    ("M_T_C", "motionSensor, temperatureMeasurement(2), contactSensor",
     ["M", "T1", "T2", "C"], False),
    ("M_P_T_C", "motionSensor, powerSensor, temperatureMeasurement(2), contactSensor",
     ["M", "P", "T1", "T2", "C"], False),
    ("ALL", "All 6 channels (M+P+T1+T2+C+E)",
     ["M", "P", "T1", "T2", "C", "E"], False),
]


def resolve_combo_channels(group_keys: list[str]) -> list[str]:
    """Resolve sensor group keys to a flat list of channel names."""
    channels = []
    for key in group_keys:
        channels.extend(SENSOR_GROUPS[key])
    return channels


# ============================================================================
# Config and data loading
# ============================================================================

def load_config(config_path: str | Path) -> dict:
    """Load raw YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_all_data(raw_cfg: dict, include_none: bool = True):
    """Load sensor data (all channels) and events.

    Returns (sensor_array, sensor_timestamps, event_timestamps,
             event_labels, channel_names, class_names).
    """
    data_cfg = raw_cfg.get("data", {})
    config = EventPreprocessConfig(
        sensor_csv=data_cfg.get("sensor_csv", ""),
        events_csv=data_cfg.get("events_csv", ""),
        column_names_csv=data_cfg.get("column_names_csv"),
        channels=data_cfg.get("channels"),
        nan_threshold=data_cfg.get("nan_threshold", 0.3),
        include_none=include_none,
    )
    return load_sensor_and_events(config)


# ============================================================================
# Model loading and embedding extraction
# ============================================================================

def load_mantis_model(pretrained_name: str, layer: int, output_token: str, device: str):
    """Load MantisV2 at a specific transformer layer.

    Returns (network, trainer) tuple.
    """
    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU")

    network = MantisV2(
        device=device,
        return_transf_layer=layer,
        output_token=output_token,
    )
    network = network.from_pretrained(pretrained_name)
    model = MantisTrainer(device=device, network=network)
    return network, model


def extract_embeddings(model, dataset: EventDataset) -> np.ndarray:
    """Extract embeddings for all events. Returns (n_events, embed_dim)."""
    X, _ = dataset.get_numpy_arrays()
    t0 = time.time()
    Z = model.transform(X)
    elapsed = time.time() - t0
    logger.info("  Embeddings: %s -> %s (%.1fs)", X.shape, Z.shape, elapsed)
    return Z


# ============================================================================
# Classifier factory
# ============================================================================

def build_classifier(name: str, seed: int = 42):
    """Build a classifier by name."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import NearestCentroid
    from sklearn.svm import SVC

    factories = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed),
        "svm": lambda: SVC(
            kernel="rbf", C=1.0, probability=True, random_state=seed),
        "nearest_centroid": lambda: NearestCentroid(),
    }
    factory = factories.get(name)
    if factory is None:
        raise ValueError(f"Unknown classifier: {name!r}. Available: {list(factories)}")
    return factory()


# ============================================================================
# LOOCV — returns raw predictions for flexible output formatting
# ============================================================================

def run_loocv_raw(
    Z: np.ndarray,
    y: np.ndarray,
    clf_name: str,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Leave-One-Out CV returning per-sample predictions.

    Parameters
    ----------
    Z : np.ndarray, shape (n, embed_dim)
    y : np.ndarray, shape (n,)
    clf_name : str
    seed : int

    Returns
    -------
    y_pred : np.ndarray, shape (n,), predicted class labels
    y_prob : np.ndarray or None, shape (n, n_classes), predicted probabilities.
        None if classifier has no predict_proba.
    """
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    n_classes = len(np.unique(y))
    y_pred = np.zeros(n, dtype=np.int64)

    # Check if classifier supports predict_proba
    test_clf = build_classifier(clf_name, seed)
    has_prob = hasattr(test_clf, "predict_proba")
    del test_clf

    if has_prob:
        y_prob = np.zeros((n, n_classes), dtype=np.float64)
    else:
        y_prob = None

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        scaler = StandardScaler()
        Z_tr = scaler.fit_transform(Z[mask])
        Z_te = scaler.transform(Z[i:i + 1])

        clf = build_classifier(clf_name, seed)
        clf.fit(Z_tr, y[mask])
        y_pred[i] = clf.predict(Z_te)[0]

        if has_prob:
            proba = clf.predict_proba(Z_te)[0]  # (n_seen_classes,)
            # Map to full n_classes vector using clf.classes_
            if hasattr(clf, "classes_") and len(clf.classes_) < n_classes:
                for ci, cls_label in enumerate(clf.classes_):
                    y_prob[i, cls_label] = proba[ci]
            else:
                y_prob[i] = proba

    return y_pred, y_prob


# ============================================================================
# Baseline-format output
# ============================================================================

def format_classification_report(
    combo_display_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    layer: int | None = None,
    clf_name: str | None = None,
) -> str:
    """Format results matching the Chronos-2 baseline output.

    Produces sklearn classification_report + confusion matrix,
    identical to the format in experiment_settings_260227.txt.
    """
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    n = len(y_true)
    correct = int((y_true == y_pred).sum())
    acc = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=2,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    header = combo_display_name
    if layer is not None or clf_name is not None:
        parts = []
        if layer is not None:
            parts.append(f"L{layer}")
        if clf_name is not None:
            parts.append(clf_name)
        header += f"  [{', '.join(parts)}]"

    lines = [
        header,
        f"   Leave-One-Out Accuracy: {acc:.4f} ({correct}/{n})",
        "",
        "   Leave-One-Out Classification Report:",
        report,
        "   Confusion Matrix:",
        str(cm),
    ]
    return "\n".join(lines)


def format_summary_table_baseline(
    results: list[dict],
) -> str:
    """Create O/X summary table matching baseline format.

    Shows baseline-equivalent combos with M, P, T columns + Accuracy.
    Each row is the best accuracy across all classifiers for that combo.
    """
    # Group by combo_key, take best accuracy
    best_per_combo: dict[str, dict] = {}
    for r in results:
        if not r.get("is_baseline"):
            continue
        key = r["combo_key"]
        if key not in best_per_combo or r["accuracy"] > best_per_combo[key]["accuracy"]:
            best_per_combo[key] = r

    if not best_per_combo:
        return "(No baseline results found)"

    # Define display order (matching baseline document)
    baseline_order = ["M_P_T", "M_P", "M_T", "P_T", "M", "P", "T"]

    lines = []
    lines.append(f"{'motionSensor':>14}\t{'powerSensor':>14}\t{'temperature(2)':>16}\t{'Accuracy(%)':>12}")

    for combo_key in baseline_order:
        r = best_per_combo.get(combo_key)
        if r is None:
            continue
        groups = r["groups"]
        m_flag = "O" if "M" in groups else "X"
        p_flag = "O" if "P" in groups else "X"
        t_flag = "O" if ("T1" in groups or "T2" in groups) else "X"
        acc_pct = f"{r['accuracy'] * 100:.2f}"
        lines.append(f"{m_flag:>14}\t{p_flag:>14}\t{t_flag:>16}\t{acc_pct:>12}")

    return "\n".join(lines)


def format_extended_table(results: list[dict]) -> str:
    """Create extended results table including C and E columns."""
    best_per_combo: dict[str, dict] = {}
    for r in results:
        key = r["combo_key"]
        if key not in best_per_combo or r["accuracy"] > best_per_combo[key]["accuracy"]:
            best_per_combo[key] = r

    if not best_per_combo:
        return "(No results found)"

    # All combos in order
    combo_order = [c[0] for c in CHANNEL_COMBOS]

    lines = []
    header = (f"{'Combo':<12} {'M':>3} {'P':>3} {'T1':>3} {'T2':>3} "
              f"{'C':>3} {'E':>3}  {'Acc(%)':>8} {'F1(m)':>7} {'Clf':>18}")
    lines.append(header)
    lines.append("-" * len(header))

    for combo_key in combo_order:
        r = best_per_combo.get(combo_key)
        if r is None:
            continue
        groups = r["groups"]
        flags = ""
        for g in ["M", "P", "T1", "T2", "C", "E"]:
            flags += f" {'O':>3}" if g in groups else f" {'X':>3}"

        acc_pct = f"{r['accuracy'] * 100:.2f}"
        f1_m = f"{r['f1_macro']:.4f}"
        clf = r["classifier"]
        lines.append(f"{combo_key:<12}{flags}  {acc_pct:>8} {f1_m:>7} {clf:>18}")

    return "\n".join(lines)


# ============================================================================
# t-SNE Visualization
# ============================================================================

# Vivid, colorblind-friendly colors for 3-class t-SNE scatter plots
TSNE_COLORS = {
    0: "#2078B4",  # Rich blue — Enter
    1: "#FF6F00",  # Orange — Leave
    2: "#4CAF50",  # Green — None
}


def plot_tsne_3class(
    Z: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    subtitle: str,
    output_path: Path,
    class_names: list[str] | None = None,
    seed: int = 42,
) -> None:
    """Publication-quality t-SNE scatter for 3-class events.

    Correctly classified points are filled circles; misclassified points
    are marked with X and a black edge for visual emphasis.
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    setup_style()

    n = len(Z)
    n_classes = len(np.unique(y_true))

    # StandardScaler -> PCA pre-reduction -> t-SNE
    X_scaled = StandardScaler().fit_transform(Z)
    n_pca = min(50, X_scaled.shape[1], n)
    if X_scaled.shape[1] > n_pca:
        X_scaled = PCA(n_components=n_pca, random_state=seed).fit_transform(X_scaled)

    perp = min(30, max(5, n - 1))
    tsne = TSNE(n_components=2, perplexity=perp, max_iter=1000, random_state=seed)
    emb_2d = tsne.fit_transform(X_scaled)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    correct_mask = y_true == y_pred

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot correct predictions (filled circles)
    for cls in sorted(set(y_true)):
        mask = (y_true == cls) & correct_mask
        if not mask.any():
            continue
        color = TSNE_COLORS.get(cls, "#888888")
        name = class_names[cls] if cls < len(class_names) else str(cls)
        count = int((y_true == cls).sum())
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=color, s=60, alpha=0.8,
            edgecolors="white", linewidths=0.5,
            label=f"{name} (n={count})",
            zorder=3,
        )

    # Plot misclassified points (X markers with black edge)
    n_wrong = int((~correct_mask).sum())
    if n_wrong > 0:
        for cls in sorted(set(y_true)):
            mask = (y_true == cls) & (~correct_mask)
            if not mask.any():
                continue
            color = TSNE_COLORS.get(cls, "#888888")
            ax.scatter(
                emb_2d[mask, 0], emb_2d[mask, 1],
                c=color, s=100, alpha=0.9,
                marker="X",
                edgecolors="black", linewidths=1.0,
                zorder=4,
            )
        # Legend entry for misclassified
        ax.scatter([], [], marker="X", c="gray", edgecolors="black",
                   linewidths=1.0, s=80, label=f"Misclassified (n={n_wrong})")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
            fontsize=10, ha="center", va="bottom", color="#555555")
    ax.set_xlabel("t-SNE Component 1", fontsize=11)
    ax.set_ylabel("t-SNE Component 2", fontsize=11)
    ax.legend(loc="best", fontsize=9, framealpha=0.9, edgecolor="#cccccc")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, linestyle="--")

    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved t-SNE: %s", output_path)


def plot_tsne_layer_grid(
    embeddings_by_layer: dict[int, np.ndarray],
    y_true: np.ndarray,
    predictions_by_layer: dict[int, np.ndarray],
    combo_name: str,
    clf_name: str,
    output_path: Path,
    class_names: list[str] | None = None,
    seed: int = 42,
) -> None:
    """Grid of t-SNE plots across layers for a single combo/classifier."""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    setup_style()

    layers = sorted(embeddings_by_layer.keys())
    n_layers = len(layers)
    n_classes = len(np.unique(y_true))

    ncols = min(3, n_layers)
    nrows = (n_layers + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    for idx, layer in enumerate(layers):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        Z = embeddings_by_layer[layer]
        y_pred = predictions_by_layer.get(layer)

        # Reduce dimensions
        X_scaled = StandardScaler().fit_transform(Z)
        n_pca = min(50, X_scaled.shape[1], len(Z))
        if X_scaled.shape[1] > n_pca:
            X_scaled = PCA(n_components=n_pca, random_state=seed).fit_transform(X_scaled)

        perp = min(30, max(5, len(Z) - 1))
        tsne = TSNE(n_components=2, perplexity=perp, max_iter=1000, random_state=seed)
        emb_2d = tsne.fit_transform(X_scaled)

        if y_pred is not None:
            correct_mask = y_pred == y_true
        else:
            correct_mask = np.ones(len(y_true), dtype=bool)

        for cls in sorted(set(y_true)):
            color = TSNE_COLORS.get(cls, "#888888")
            # Correct
            mask = (y_true == cls) & correct_mask
            if mask.any():
                ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                           c=color, s=40, alpha=0.7, edgecolors="white",
                           linewidths=0.3, zorder=3)
            # Misclassified
            mask_wrong = (y_true == cls) & ~correct_mask
            if mask_wrong.any():
                ax.scatter(emb_2d[mask_wrong, 0], emb_2d[mask_wrong, 1],
                           c=color, s=60, alpha=0.8, marker="X",
                           edgecolors="black", linewidths=0.8, zorder=4)

        acc = float((y_pred == y_true).sum() / len(y_true)) if y_pred is not None else 0
        ax.set_title(f"L{layer} -- Acc={acc:.1%}", fontsize=11)
        ax.set_xlabel("t-SNE 1", fontsize=9)
        ax.set_ylabel("t-SNE 2", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, linestyle="--")

    # Hide unused subplots
    for idx in range(n_layers, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Legend on first subplot
    ax0 = axes[0, 0]
    for cls in sorted(set(y_true)):
        name = class_names[cls] if cls < len(class_names) else str(cls)
        color = TSNE_COLORS.get(cls, "#888888")
        ax0.scatter([], [], c=color, s=40, label=name)
    ax0.scatter([], [], marker="X", c="gray", edgecolors="black",
                linewidths=0.8, s=60, label="Misclassified")
    ax0.legend(loc="best", fontsize=8, framealpha=0.8)

    fig.suptitle(f"t-SNE across Layers -- {combo_name} / {clf_name}",
                 fontsize=14, fontweight="bold", y=1.02)

    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved layer grid t-SNE: %s", output_path)


# ============================================================================
# Results saving
# ============================================================================

def save_csv(rows: list[dict], path: Path, fieldnames: list[str] | None = None):
    """Save results list to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved: %s (%d rows)", path, len(rows))


# ============================================================================
# Main sweep
# ============================================================================

def run_phase1_sweep(
    raw_cfg: dict,
    layers: list[int],
    classifier_names: list[str],
    combos: list[tuple[str, str, list[str], bool]],
    include_none: bool = True,
    device: str = "cuda",
    seed: int = 42,
    top_k_tsne: int = 5,
) -> dict:
    """Run the full Phase 1 zero-shot sweep.

    Returns a summary dictionary with all results.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix as sk_confusion_matrix, roc_auc_score,
    )

    output_dir = Path(raw_cfg.get("output_dir", "results/phase1_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tsne_dir = plots_dir / "tsne"
    reports_dir = output_dir / "reports"
    for d in [tables_dir, plots_dir, tsne_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    configure_output(formats=["png"], dpi=200)

    # ---- Load all data ----
    logger.info("=" * 70)
    logger.info("PHASE 1 ZERO-SHOT SWEEP")
    logger.info("=" * 70)
    logger.info("Loading data (all channels, include_none=%s)...", include_none)

    sensor_array, sensor_ts, event_ts, event_labels, channel_names, class_names = (
        load_all_data(raw_cfg, include_none=include_none)
    )

    # Build channel name -> column index mapping
    ch_idx = {name: i for i, name in enumerate(channel_names)}
    logger.info("Available channels (%d): %s", len(channel_names), channel_names)

    # Dataset config
    ds_cfg_raw = raw_cfg.get("dataset", {})
    ds_config = EventDatasetConfig(
        seq_len=ds_cfg_raw.get("seq_len", 512),
        target_seq_len=ds_cfg_raw.get("target_seq_len"),
    )

    # Model config
    model_cfg = raw_cfg.get("model", {})
    pretrained_name = model_cfg.get("pretrained_name", "paris-noah/MantisV2")
    output_token = model_cfg.get("output_token", "combined")

    n_events = len(event_labels)
    if include_none:
        target_names = ["ENTER_HOME", "LEAVE_HOME", "NONE"]
    else:
        target_names = ["ENTER_HOME", "LEAVE_HOME"]

    logger.info("Events: %d (%s)", n_events,
                ", ".join(f"{name}={int((event_labels == i).sum())}"
                          for i, name in enumerate(class_names)))
    logger.info("Layers: %s", layers)
    logger.info("Combos: %d", len(combos))
    logger.info("Classifiers: %s", classifier_names)
    logger.info("Seq_len: %d (%.1fh)", ds_config.seq_len, ds_config.seq_len / 60)

    # ---- Main sweep ----
    all_results = []
    all_embeddings = {}  # {(layer, combo_key): Z_array}
    all_predictions = {}  # {(layer, combo_key, clf_name): y_pred}
    all_reports = []  # text classification reports

    t_sweep_start = time.time()

    for layer in layers:
        logger.info("=" * 70)
        logger.info("Layer %d: Loading MantisV2...", layer)
        logger.info("=" * 70)

        network, model = load_mantis_model(pretrained_name, layer, output_token, device)

        for combo_key, display_name, group_keys, is_baseline in combos:
            # Resolve channels for this combo
            combo_channels = resolve_combo_channels(group_keys)

            # Check all channels are available
            missing = [c for c in combo_channels if c not in ch_idx]
            if missing:
                logger.warning("Combo %s: missing channels %s, skipping", combo_key, missing)
                continue

            # Subset sensor array to this combo's channels
            indices = [ch_idx[c] for c in combo_channels]
            sensor_subset = sensor_array[:, indices]

            # Build dataset with subset
            dataset = EventDataset(
                sensor_subset, sensor_ts, event_ts, event_labels, ds_config,
            )

            # Extract embeddings
            logger.info("  L%d / %s (%d ch) ...", layer, combo_key, len(combo_channels))
            Z = extract_embeddings(model, dataset)
            all_embeddings[(layer, combo_key)] = Z

            # Run LOOCV for each classifier
            for clf_name in classifier_names:
                t0 = time.time()
                y_pred, y_prob = run_loocv_raw(Z, event_labels, clf_name, seed)
                elapsed = time.time() - t0

                all_predictions[(layer, combo_key, clf_name)] = y_pred

                # Compute metrics
                acc = accuracy_score(event_labels, y_pred)
                f1_mac = f1_score(event_labels, y_pred, average="macro", zero_division=0)
                f1_wt = f1_score(event_labels, y_pred, average="weighted", zero_division=0)
                prec_mac = precision_score(event_labels, y_pred, average="macro", zero_division=0)
                rec_mac = recall_score(event_labels, y_pred, average="macro", zero_division=0)

                auc_val = float("nan")
                if y_prob is not None:
                    try:
                        auc_val = roc_auc_score(
                            event_labels, y_prob, multi_class="ovr", average="macro")
                    except Exception:
                        pass

                correct = int((event_labels == y_pred).sum())

                logger.info(
                    "    %s: Acc=%.4f (%d/%d) F1m=%.4f AUC=%.4f (%.1fs)",
                    clf_name, acc, correct, n_events, f1_mac, auc_val, elapsed,
                )

                result = {
                    "layer": layer,
                    "combo_key": combo_key,
                    "combo_display": display_name,
                    "groups": group_keys,
                    "is_baseline": is_baseline,
                    "classifier": clf_name,
                    "accuracy": acc,
                    "correct": correct,
                    "total": n_events,
                    "f1_macro": f1_mac,
                    "f1_weighted": f1_wt,
                    "precision_macro": prec_mac,
                    "recall_macro": rec_mac,
                    "auc_macro": auc_val if not np.isnan(auc_val) else None,
                    "time_sec": round(elapsed, 1),
                }
                all_results.append(result)

                # Generate baseline-format classification report
                report_text = format_classification_report(
                    display_name, event_labels, y_pred, target_names,
                    layer=layer, clf_name=clf_name,
                )
                all_reports.append({
                    "layer": layer,
                    "combo_key": combo_key,
                    "classifier": clf_name,
                    "accuracy": acc,
                    "report": report_text,
                })

        # Free GPU memory after processing all combos for this layer
        del model, network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sweep_time = time.time() - t_sweep_start
    logger.info("Sweep completed in %.1fs (%.1f min)", sweep_time, sweep_time / 60)

    # ---- Save results ----
    logger.info("=" * 70)
    logger.info("Saving results...")
    logger.info("=" * 70)

    # 1. Full results CSV
    csv_fields = [
        "layer", "combo_key", "classifier", "accuracy", "correct", "total",
        "f1_macro", "f1_weighted", "precision_macro", "recall_macro",
        "auc_macro", "time_sec",
    ]
    csv_rows = []
    for r in all_results:
        row = {}
        for k in csv_fields:
            v = r.get(k)
            if isinstance(v, float):
                row[k] = round(v, 4)
            else:
                row[k] = v
        csv_rows.append(row)
    save_csv(csv_rows, tables_dir / "all_results.csv", csv_fields)

    # 2. Best per combo (across layers and classifiers)
    best_per_combo: dict[str, dict] = {}
    for r in all_results:
        key = r["combo_key"]
        if key not in best_per_combo or r["accuracy"] > best_per_combo[key]["accuracy"]:
            best_per_combo[key] = r

    best_combo_rows = []
    for c_key, c_display, c_groups, c_baseline in combos:
        r = best_per_combo.get(c_key)
        if r:
            best_combo_rows.append({
                "combo_key": c_key,
                "best_layer": r["layer"],
                "best_classifier": r["classifier"],
                "accuracy": round(r["accuracy"], 4),
                "f1_macro": round(r["f1_macro"], 4),
                "auc_macro": round(r["auc_macro"], 4) if r["auc_macro"] else None,
            })
    save_csv(best_combo_rows, tables_dir / "best_per_combo.csv")

    # 3. Per-layer summary
    layer_summary = []
    for layer in layers:
        layer_results = [r for r in all_results if r["layer"] == layer]
        if layer_results:
            best = max(layer_results, key=lambda r: r["accuracy"])
            layer_summary.append({
                "layer": layer,
                "best_combo": best["combo_key"],
                "best_classifier": best["classifier"],
                "accuracy": round(best["accuracy"], 4),
                "f1_macro": round(best["f1_macro"], 4),
            })
    save_csv(layer_summary, tables_dir / "layer_summary.csv")

    # 4. Baseline O/X summary tables (per layer)
    for layer in layers:
        layer_results = [r for r in all_results if r["layer"] == layer]
        table = format_summary_table_baseline(layer_results)
        table_path = tables_dir / f"baseline_table_L{layer}.txt"
        table_path.write_text(
            f"=== Baseline Comparison Table (Layer {layer}) ===\n"
            f"(Best accuracy across classifiers per combo)\n\n{table}\n"
        )

    # Extended table (all combos, per layer)
    for layer in layers:
        layer_results = [r for r in all_results if r["layer"] == layer]
        ext_table = format_extended_table(layer_results)
        ext_path = tables_dir / f"extended_table_L{layer}.txt"
        ext_path.write_text(
            f"=== Extended Results (Layer {layer}) ===\n"
            f"(Best accuracy across classifiers per combo)\n\n{ext_table}\n"
        )

    # 5. Save all classification reports
    report_path = reports_dir / "all_classification_reports.txt"
    sorted_reports = sorted(all_reports, key=lambda r: r["accuracy"], reverse=True)
    with open(report_path, "w") as f:
        f.write("Phase 1 Zero-Shot Sweep -- Classification Reports\n")
        f.write("=" * 70 + "\n")
        f.write(f"Events: {n_events} ({', '.join(f'{n}={int((event_labels == i).sum())}' for i, n in enumerate(class_names))})\n")
        f.write(f"Seq_len: {ds_config.seq_len} min ({ds_config.seq_len / 60:.1f}h)\n")
        f.write(f"Sorted by accuracy (descending)\n")
        f.write("=" * 70 + "\n\n")
        for rpt in sorted_reports:
            f.write(rpt["report"])
            f.write("\n" + "-" * 50 + "\n\n")
    logger.info("Saved %d classification reports: %s", len(all_reports), report_path)

    # 6. Best results report (one per combo, matching baseline format)
    best_report_path = reports_dir / "best_results_report.txt"
    with open(best_report_path, "w") as f:
        f.write("Phase 1 Zero-Shot -- Best Results per Sensor Combination\n")
        f.write("=" * 70 + "\n\n")

        for combo_key, display_name, group_keys, is_baseline in combos:
            combo_results = [r for r in all_results if r["combo_key"] == combo_key]
            if not combo_results:
                continue
            best = max(combo_results, key=lambda r: r["accuracy"])

            # Find matching report
            report_match = [
                rpt for rpt in all_reports
                if rpt["combo_key"] == combo_key
                and rpt["layer"] == best["layer"]
                and rpt["classifier"] == best["classifier"]
            ]
            if report_match:
                f.write(report_match[0]["report"])
            else:
                f.write(f"{display_name}: Acc={best['accuracy']:.4f}\n")
            f.write("\n" + "-" * 50 + "\n\n")
    logger.info("Saved best results: %s", best_report_path)

    # ---- t-SNE Visualization ----
    logger.info("=" * 70)
    logger.info("Generating t-SNE visualizations (top %d configs)...", top_k_tsne)
    logger.info("=" * 70)

    # Sort results by accuracy, pick top K unique (layer, combo) pairs
    sorted_results = sorted(all_results, key=lambda r: r["accuracy"], reverse=True)
    seen_configs = set()
    top_configs = []
    for r in sorted_results:
        config_key = (r["layer"], r["combo_key"])
        if config_key not in seen_configs:
            seen_configs.add(config_key)
            top_configs.append(r)
        if len(top_configs) >= top_k_tsne:
            break

    # Individual t-SNE plots for top configs
    for rank, r in enumerate(top_configs, 1):
        layer = r["layer"]
        combo_key = r["combo_key"]
        clf_name = r["classifier"]

        Z = all_embeddings.get((layer, combo_key))
        y_pred = all_predictions.get((layer, combo_key, clf_name))

        if Z is None or y_pred is None:
            continue

        acc_pct = r["accuracy"] * 100
        title = f"L{layer} / {r['combo_display']}"
        subtitle = (f"Accuracy: {acc_pct:.1f}% ({r['correct']}/{r['total']}) "
                    f"| {clf_name} | Rank #{rank}")

        try:
            plot_tsne_3class(
                Z, event_labels, y_pred,
                title=title,
                subtitle=subtitle,
                output_path=tsne_dir / f"tsne_top{rank}_L{layer}_{combo_key}_{clf_name}",
                class_names=target_names,
                seed=seed,
            )
        except Exception:
            logger.warning("Failed t-SNE for rank %d", rank, exc_info=True)

    # Layer grid t-SNE for the overall best combo
    if top_configs:
        best_combo_key = top_configs[0]["combo_key"]
        best_clf_name = top_configs[0]["classifier"]
        best_combo_display = top_configs[0]["combo_display"]

        layer_Z = {}
        layer_pred = {}
        for layer in layers:
            Z = all_embeddings.get((layer, best_combo_key))
            y_pred = all_predictions.get((layer, best_combo_key, best_clf_name))
            if Z is not None:
                layer_Z[layer] = Z
            if y_pred is not None:
                layer_pred[layer] = y_pred

        if layer_Z:
            try:
                plot_tsne_layer_grid(
                    layer_Z, event_labels, layer_pred,
                    combo_name=best_combo_display,
                    clf_name=best_clf_name,
                    output_path=tsne_dir / f"tsne_layer_grid_{best_combo_key}",
                    class_names=target_names,
                    seed=seed,
                )
            except Exception:
                logger.warning("Failed layer grid t-SNE", exc_info=True)

    # ---- JSON summary ----
    summary = {
        "experiment": "Phase 1 Zero-Shot Sweep (MantisV2)",
        "n_events": n_events,
        "class_distribution": {
            name: int((event_labels == i).sum())
            for i, name in enumerate(class_names)
        },
        "channels_available": channel_names,
        "seq_len": ds_config.seq_len,
        "layers": layers,
        "combos_tested": len(combos),
        "classifiers": classifier_names,
        "include_none": include_none,
        "seed": seed,
        "sweep_time_sec": round(sweep_time, 1),
        "total_experiments": len(all_results),
        "best_overall": {
            "layer": sorted_results[0]["layer"],
            "combo": sorted_results[0]["combo_key"],
            "classifier": sorted_results[0]["classifier"],
            "accuracy": round(sorted_results[0]["accuracy"], 4),
            "f1_macro": round(sorted_results[0]["f1_macro"], 4),
            "auc_macro": (round(sorted_results[0]["auc_macro"], 4)
                          if sorted_results[0]["auc_macro"] else None),
        } if sorted_results else {},
        "all_results": [
            {k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in r.items()
             if k in csv_fields}
            for r in all_results
        ],
    }

    json_path = reports_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved: %s", json_path)

    # ---- Console summary ----
    print("\n" + "=" * 70)
    print("PHASE 1 ZERO-SHOT SWEEP -- RESULTS SUMMARY")
    print("=" * 70)
    class_dist = ", ".join(
        f"{name}={int((event_labels == i).sum())}"
        for i, name in enumerate(class_names)
    )
    print(f"\nEvents: {n_events} ({class_dist})")
    print(f"Seq_len: {ds_config.seq_len} min ({ds_config.seq_len / 60:.1f}h)")
    print(f"Layers: {layers}")
    print(f"Combos: {len(combos)} | Classifiers: {classifier_names}")
    print(f"Total experiments: {len(all_results)} | Sweep time: {sweep_time:.1f}s")

    # Top 10 results
    print(f"\n{'Rank':<5} {'Layer':<6} {'Combo':<12} {'Classifier':<18} "
          f"{'Acc(%)':>8} {'F1(m)':>7} {'AUC':>7}")
    print("-" * 70)
    for rank, r in enumerate(sorted_results[:10], 1):
        auc_str = f"{r['auc_macro']:.4f}" if r['auc_macro'] is not None else "  N/A"
        print(f"{rank:<5} L{r['layer']:<5} {r['combo_key']:<12} {r['classifier']:<18} "
              f"{r['accuracy'] * 100:>7.2f} {r['f1_macro']:>7.4f} {auc_str}")

    # Baseline comparison table (best layer)
    if sorted_results:
        best_layer = sorted_results[0]["layer"]
        print(f"\n--- Baseline Comparison (L{best_layer}, best clf per combo) ---")
        print(format_summary_table_baseline(
            [r for r in all_results if r["layer"] == best_layer]))

    # Extended table (best layer)
    if sorted_results:
        best_layer = sorted_results[0]["layer"]
        print(f"\n--- Extended Results (L{best_layer}, best clf per combo) ---")
        print(format_extended_table(
            [r for r in all_results if r["layer"] == best_layer]))

    print(f"\nOutput: {output_dir}")
    print("=" * 70)

    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Zero-Shot Sweep: Layer x Sensor x Classifier",
    )
    parser.add_argument(
        "--config", required=True, help="YAML config file",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: L0 only, RF only, baseline combos only",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Transformer layers (default: from config or 0-5)",
    )
    parser.add_argument(
        "--classifiers", nargs="+", default=None,
        help="Classifiers: random_forest, svm, nearest_centroid",
    )
    parser.add_argument(
        "--combos", nargs="+", default=None,
        help="Combo keys to test (e.g., M M_P M_T ALL)",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Only test 7 baseline-equivalent combos",
    )
    parser.add_argument(
        "--binary", action="store_true",
        help="Binary mode: exclude NONE (73 events instead of 109)",
    )
    parser.add_argument(
        "--device", default=None, help="Device: cuda or cpu",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
    )
    parser.add_argument(
        "--top-k-tsne", type=int, default=5,
        help="Number of top configs for t-SNE (default: 5)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    raw_cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else raw_cfg.get("seed", 42)
    device = args.device or raw_cfg.get("model", {}).get("device", "cuda")
    include_none = not args.binary

    model_cfg = raw_cfg.get("model", {})

    if args.quick:
        layers = [0]
        classifier_names = ["random_forest"]
        combos = [c for c in CHANNEL_COMBOS if c[3]]  # baseline only
    else:
        layers = args.layers or model_cfg.get("layers", [0, 1, 2, 3, 4, 5])
        classifier_names = args.classifiers or raw_cfg.get(
            "classifiers", ["random_forest", "svm", "nearest_centroid"])

        if args.baseline_only:
            combos = [c for c in CHANNEL_COMBOS if c[3]]
        elif args.combos:
            combo_set = set(args.combos)
            combos = [c for c in CHANNEL_COMBOS if c[0] in combo_set]
            if not combos:
                logger.error("No valid combos found from: %s", args.combos)
                logger.info("Available: %s", [c[0] for c in CHANNEL_COMBOS])
                return
        else:
            combos = CHANNEL_COMBOS

    logger.info("=" * 70)
    logger.info("Phase 1 Zero-Shot Sweep")
    logger.info("=" * 70)
    logger.info("  Layers: %s", layers)
    logger.info("  Classifiers: %s", classifier_names)
    logger.info("  Combos: %d%s", len(combos),
                " (baseline only)" if args.baseline_only else "")
    logger.info("  Mode: %s", "binary (2-class)" if args.binary else "3-class (with NONE)")
    logger.info("  Device: %s", device)
    logger.info("  Seed: %d", seed)
    logger.info("  t-SNE top-K: %d", args.top_k_tsne)

    t_start = time.time()

    summary = run_phase1_sweep(
        raw_cfg,
        layers=layers,
        classifier_names=classifier_names,
        combos=combos,
        include_none=include_none,
        device=device,
        seed=seed,
        top_k_tsne=args.top_k_tsne,
    )

    t_total = time.time() - t_start
    logger.info("Total time: %.1fs (%.1f min)", t_total, t_total / 60)


if __name__ == "__main__":
    main()
