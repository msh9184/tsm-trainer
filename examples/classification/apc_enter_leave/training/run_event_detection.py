"""Enter/Leave Event Detection with MantisV2 Zero-Shot Embeddings.

Classifies ENTER_HOME vs LEAVE_HOME events using context windows of
SmartThings sensor data passed through frozen MantisV2 embeddings.

Key differences from occupancy classification:
  - One context window per EVENT (not per-timestep sliding windows)
  - Only 73 events (excl. NONE) -> cross-validation mandatory
  - MantisV2 is frozen: embeddings computed ONCE per layer, reused across CV folds
  - LOOCV with 73 folds on precomputed embeddings takes ~2 seconds

Execution flow:
  1. Load sensor data (1-min CSV) + events (excl. NONE)
  2. Build EventDataset: one context window per event
  3. For each layer: extract ALL embeddings (frozen model, one pass)
  4. For each layer x classifier x CV method: run cross-validation
  5. Generate tables, plots, and JSON report

Usage:
    cd examples/classification/apc_enter_leave
    python training/run_event_detection.py --config training/configs/enter-leave-zeroshot.yaml

    # Quick mode: L0 only, LOOCV only, RF only
    python training/run_event_detection.py --config ... --quick

    # Select specific layers and CV methods
    python training/run_event_detection.py --config ... --layers 0 2 --cv-methods loocv

    # Include NONE as 3rd class
    python training/run_event_detection.py --config ... --include-none

    # CPU-only
    python training/run_event_detection.py --config ... --device cpu

    # Backward-only context (8.5h, no future)
    python training/run_event_detection.py --config ... --backward-only

    # Custom bidirectional context (e.g., past 8min + future 8min)
    python training/run_event_detection.py --config ... --context-before 8 --context-after 8
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
from data.dataset import EventDatasetConfig, EventDataset, build_dataset_config
from evaluation.metrics import (
    EventClassificationMetrics,
    aggregate_cv_predictions,
    aggregate_repeated_cv_predictions,
)
from visualization.style import (
    FIGSIZE_SINGLE,
    setup_style, save_figure, configure_output,
)
from visualization.curves import (
    plot_confusion_matrix,
    plot_cv_comparison_bar,
)
from visualization.embeddings import (
    reduce_dimensions, plot_embeddings, plot_embeddings_multi_method,
)

logger = logging.getLogger(__name__)

ALL_LAYERS = [0, 1, 2, 3, 4, 5]


# ============================================================================
# Config loading
# ============================================================================

def load_config(config_path: str | Path) -> dict:
    """Load raw YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================================
# Data loading
# ============================================================================

def load_data(raw_cfg: dict, include_none: bool = False):
    """Load sensor data and events from config.

    Returns (sensor_array, sensor_timestamps, event_timestamps,
             event_labels, channel_names, class_names).
    """
    data_cfg = raw_cfg.get("data", {})

    config = EventPreprocessConfig(
        sensor_csv=data_cfg.get("sensor_csv", ""),
        events_csv=data_cfg.get("events_csv", ""),
        column_names_csv=data_cfg.get("column_names_csv"),
        column_names=data_cfg.get("column_names"),
        channels=data_cfg.get("channels"),
        exclude_channels=data_cfg.get("exclude_channels", []),
        nan_threshold=data_cfg.get("nan_threshold", 0.3),
        include_none=include_none,
        add_time_features=data_cfg.get("add_time_features", False),
    )

    return load_sensor_and_events(config)


# ============================================================================
# Model loading
# ============================================================================

def load_mantis_model(
    pretrained_name: str,
    layer: int,
    output_token: str,
    device: str,
):
    """Load MantisV2 at a specific transformer layer.

    Returns (network, trainer) tuple. The trainer's .transform() extracts
    embeddings from raw (n_channels, seq_len) windows.
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


def extract_all_embeddings(model, dataset: EventDataset) -> np.ndarray:
    """Extract embeddings for ALL events at once (frozen model = deterministic).

    Returns shape (n_events, embedding_dim).
    """
    X, _ = dataset.get_numpy_arrays()
    t0 = time.time()
    Z = model.transform(X)
    elapsed = time.time() - t0

    # Defensive NaN check (MantisV2 single-patch normalization edge case)
    nan_count = int(np.isnan(Z).sum())
    if nan_count > 0:
        logger.warning(
            "%d NaN values in embeddings (%.1f%% of %d), replacing with 0",
            nan_count, 100.0 * nan_count / Z.size, Z.size,
        )
        Z = np.nan_to_num(Z, nan=0.0)

    logger.info("Extracted embeddings: %s -> %s (%.1fs)", X.shape, Z.shape, elapsed)
    return Z


# ============================================================================
# Classifier factory
# ============================================================================

def build_classifier(name: str, seed: int = 42):
    """Build a single classifier by name."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import NearestCentroid
    from sklearn.svm import SVC

    classifiers = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed,
        ),
        "svm": lambda: SVC(kernel="rbf", C=1.0, probability=True, random_state=seed),
        "nearest_centroid": lambda: NearestCentroid(),
    }

    factory = classifiers.get(name)
    if factory is None:
        raise ValueError(f"Unknown classifier: {name!r}. Available: {list(classifiers)}")
    return factory()


def build_classifiers(names: list[str], seed: int = 42) -> dict:
    """Build multiple classifiers by name."""
    return {name: build_classifier(name, seed) for name in names}


# ============================================================================
# Probability extraction helper
# ============================================================================

def _extract_class1_prob(clf, proba: np.ndarray) -> np.ndarray:
    """Extract probability of class 1 from predict_proba output.

    Uses ``clf.classes_`` to correctly map columns to class labels,
    handling edge cases where the training fold may have seen only one class
    (in which case predict_proba returns fewer columns than expected).

    Parameters
    ----------
    clf : fitted classifier with predict_proba
    proba : np.ndarray, shape (n_samples, n_seen_classes)

    Returns
    -------
    np.ndarray, shape (n_samples,) — probability of class 1.
    """
    if hasattr(clf, "classes_"):
        classes = list(clf.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        # Class 1 not seen during training — probability is 0
        return np.zeros(proba.shape[0], dtype=np.float64)

    # Fallback: assume standard column ordering [class_0, class_1, ...]
    if proba.shape[1] >= 2:
        return proba[:, 1]
    return proba[:, 0]


# ============================================================================
# Cross-validation runners
# ============================================================================

def run_loocv(
    Z: np.ndarray,
    y: np.ndarray,
    clf_factory,
    class_names: list[str] | None = None,
) -> EventClassificationMetrics:
    """Leave-One-Out Cross-Validation.

    N folds (one per sample). Fit scaler+classifier on N-1, predict 1.
    Aggregate all N predictions for a single global evaluation.
    """
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    n_classes = len(np.unique(y))
    is_binary = n_classes <= 2

    # Check predict_proba availability once (same classifier type every fold)
    test_clf = clf_factory()
    has_prob = hasattr(test_clf, "predict_proba")
    del test_clf

    y_pred_all = np.zeros(n, dtype=np.int64)
    if has_prob:
        if is_binary:
            y_prob_all = np.zeros(n, dtype=np.float64)
        else:
            y_prob_all = np.zeros((n, n_classes), dtype=np.float64)
    else:
        y_prob_all = None

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        Z_train, y_train = Z[train_mask], y[train_mask]
        Z_test = Z[i : i + 1]

        scaler = StandardScaler()
        Z_train_s = scaler.fit_transform(Z_train)
        Z_test_s = scaler.transform(Z_test)

        clf = clf_factory()
        clf.fit(Z_train_s, y_train)

        y_pred_all[i] = clf.predict(Z_test_s)[0]

        if has_prob:
            proba = clf.predict_proba(Z_test_s)  # (1, n_seen_classes)
            if is_binary:
                y_prob_all[i] = _extract_class1_prob(clf, proba)[0]
            else:
                # Map to full n_classes vector using clf.classes_
                if hasattr(clf, "classes_") and len(clf.classes_) < n_classes:
                    full_prob = np.zeros(n_classes, dtype=np.float64)
                    for ci, cls in enumerate(clf.classes_):
                        full_prob[cls] = proba[0, ci]
                    y_prob_all[i] = full_prob
                else:
                    y_prob_all[i] = proba[0]

    return aggregate_cv_predictions(
        y, y_pred_all, y_prob_all,
        cv_method="LOOCV", n_folds=n, class_names=class_names,
    )


def run_stratified_kfold(
    Z: np.ndarray,
    y: np.ndarray,
    clf_factory,
    k: int = 5,
    n_repeats: int = 10,
    seed: int = 42,
    class_names: list[str] | None = None,
) -> EventClassificationMetrics:
    """Stratified K-Fold with repeats.

    Per-sample probabilities are averaged across all appearances in test folds,
    then a single global evaluation is computed.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.preprocessing import StandardScaler

    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats, random_state=seed)
    n = len(y)
    n_classes = len(np.unique(y))
    is_binary = n_classes <= 2

    # Check predict_proba availability once
    test_clf = clf_factory()
    has_prob = hasattr(test_clf, "predict_proba")
    del test_clf

    sample_indices_all = []
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    n_folds = 0

    for train_idx, test_idx in rskf.split(Z, y):
        n_folds += 1
        Z_train, y_train = Z[train_idx], y[train_idx]
        Z_test, y_test = Z[test_idx], y[test_idx]

        scaler = StandardScaler()
        Z_train_s = scaler.fit_transform(Z_train)
        Z_test_s = scaler.transform(Z_test)

        clf = clf_factory()
        clf.fit(Z_train_s, y_train)

        preds = clf.predict(Z_test_s)

        probs_class1 = None
        probs_full = None
        if has_prob:
            proba = clf.predict_proba(Z_test_s)
            if is_binary:
                probs_class1 = _extract_class1_prob(clf, proba)
            else:
                # Map to full n_classes columns if fewer seen
                if hasattr(clf, "classes_") and len(clf.classes_) < n_classes:
                    full = np.zeros((len(Z_test_s), n_classes), dtype=np.float64)
                    for ci, cls in enumerate(clf.classes_):
                        full[:, cls] = proba[:, ci]
                    probs_full = full
                else:
                    probs_full = proba

        for j, idx in enumerate(test_idx):
            sample_indices_all.append(idx)
            y_true_all.append(y_test[j])
            y_pred_all.append(preds[j])
            if is_binary and probs_class1 is not None:
                y_prob_all.append(probs_class1[j])
            elif not is_binary and probs_full is not None:
                y_prob_all.append(probs_full[j])

    sample_indices_all = np.array(sample_indices_all)
    y_true_all = np.array(y_true_all, dtype=np.int64)
    y_pred_all = np.array(y_pred_all, dtype=np.int64)

    if has_prob and y_prob_all:
        if is_binary:
            prob_arr = np.array(y_prob_all, dtype=np.float64)
        else:
            prob_arr = np.vstack(y_prob_all)  # (n_predictions, n_classes)
    else:
        prob_arr = None

    cv_name = f"StratifiedKFold{k}x{n_repeats}"
    return aggregate_repeated_cv_predictions(
        sample_indices_all, y_true_all, y_pred_all, prob_arr,
        n_samples=n, cv_method=cv_name, n_folds=n_folds, class_names=class_names,
    )


def run_lodo(
    Z: np.ndarray,
    y: np.ndarray,
    day_groups: np.ndarray,
    clf_factory,
    class_names: list[str] | None = None,
) -> EventClassificationMetrics:
    """Leave-One-Day-Out Cross-Validation.

    Each fold leaves out all events from one calendar day.
    Folds where the training set has only one class are skipped with a warning.
    """
    from sklearn.preprocessing import StandardScaler

    unique_days = sorted(np.unique(day_groups).tolist())
    n_folds = len(unique_days)
    n_classes = len(np.unique(y))
    is_binary = n_classes <= 2

    # Check predict_proba availability once
    test_clf = clf_factory()
    has_prob = hasattr(test_clf, "predict_proba")
    del test_clf

    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    skipped = 0

    for day in unique_days:
        test_mask = day_groups == day
        train_mask = ~test_mask

        y_train = y[train_mask]
        y_test = y[test_mask]

        # Skip if training set has only one class (classifier can't learn)
        if len(np.unique(y_train)) < 2:
            logger.warning(
                "LODO day %d: training set has only one class, skipping %d test events",
                day, test_mask.sum(),
            )
            skipped += 1
            continue

        Z_train = Z[train_mask]
        Z_test = Z[test_mask]

        scaler = StandardScaler()
        Z_train_s = scaler.fit_transform(Z_train)
        Z_test_s = scaler.transform(Z_test)

        clf = clf_factory()
        clf.fit(Z_train_s, y_train)

        preds = clf.predict(Z_test_s)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(preds.tolist())

        if has_prob:
            proba = clf.predict_proba(Z_test_s)
            if is_binary:
                class1_probs = _extract_class1_prob(clf, proba)
                y_prob_all.extend(class1_probs.tolist())
            else:
                # Map to full n_classes columns if fewer seen
                if hasattr(clf, "classes_") and len(clf.classes_) < n_classes:
                    full = np.zeros((len(Z_test_s), n_classes), dtype=np.float64)
                    for ci, cls in enumerate(clf.classes_):
                        full[:, cls] = proba[:, ci]
                    for row in full:
                        y_prob_all.append(row)
                else:
                    for row in proba:
                        y_prob_all.append(row)

    if skipped > 0:
        logger.warning("LODO: skipped %d/%d day folds due to single-class training sets", skipped, n_folds)

    if len(y_true_all) == 0:
        logger.error("LODO: all folds skipped — no predictions to aggregate")
        from evaluation.metrics import compute_event_metrics
        return compute_event_metrics(
            np.array([0, 1]), np.array([0, 0]),
            class_names=class_names,
        )

    y_true_all = np.array(y_true_all, dtype=np.int64)
    y_pred_all = np.array(y_pred_all, dtype=np.int64)

    if has_prob and y_prob_all:
        if is_binary:
            prob_arr = np.array(y_prob_all, dtype=np.float64)
        else:
            prob_arr = np.vstack(y_prob_all)
    else:
        prob_arr = None

    return aggregate_cv_predictions(
        y_true_all, y_pred_all, prob_arr,
        cv_method="LODO", n_folds=n_folds - skipped, class_names=class_names,
    )


# ============================================================================
# Visualization helpers
# ============================================================================

def plot_roc_overlay(results: dict, output_path: Path) -> None:
    """Plot ROC curves for all classifiers on one figure."""
    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    colors = ["#009E73", "#CC3311", "#0173B2", "#DE8F05"]
    for i, (clf_name, data) in enumerate(results.items()):
        metrics = data["metrics"]
        if metrics.roc_fpr is None or metrics.roc_tpr is None:
            continue
        ax.plot(
            metrics.roc_fpr, metrics.roc_tpr,
            color=colors[i % len(colors)], lw=2,
            label=f"{clf_name} (AUC={metrics.roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Classifiers")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    save_figure(fig, output_path)
    plt.close(fig)


def plot_layer_ablation_bar(
    layer_results: dict[int, dict[str, float]],
    metric_name: str,
    output_path: Path,
) -> None:
    """Bar chart of a metric across layers, best highlighted."""
    setup_style()
    layers = sorted(layer_results.keys())
    values = [layer_results[l].get("best", 0) for l in layers]

    best_idx = int(np.argmax(values))

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bar_colors = ["#0173B2" if i != best_idx else "#DE8F05" for i in range(len(layers))]
    bars = ax.bar([f"L{l}" for l in layers], values, color=bar_colors)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9,
        )

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} by Layer (best classifier)")
    ax.set_ylim(0, 1.05)

    save_figure(fig, output_path)
    plt.close(fig)


# ============================================================================
# Result saving helpers
# ============================================================================

def save_results_csv(
    results: list[dict],
    output_path: Path,
    fieldnames: list[str] | None = None,
) -> None:
    """Save results list to CSV."""
    if not results:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Saved: %s (%d rows)", output_path, len(results))


def save_results_txt(
    results: list[dict],
    output_path: Path,
    fieldnames: list[str] | None = None,
) -> None:
    """Save results as text table."""
    if not results:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(results[0].keys())

    # Calculate column widths
    widths = {f: max(len(f), max(len(str(r.get(f, ""))) for r in results)) for f in fieldnames}

    lines = []
    header = "  ".join(f"{f:<{widths[f]}}" for f in fieldnames)
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        line = "  ".join(f"{str(r.get(f, '')):<{widths[f]}}" for f in fieldnames)
        lines.append(line)

    output_path.write_text("\n".join(lines) + "\n")
    logger.info("Saved: %s", output_path)


# ============================================================================
# Main experiment runner
# ============================================================================

def run_experiment(
    raw_cfg: dict,
    layers: list[int],
    cv_methods: list[str],
    classifier_names: list[str],
    include_none: bool = False,
    device: str = "cuda",
    seed: int = 42,
    quick: bool = False,
) -> dict:
    """Run the full enter/leave detection experiment.

    Returns a comprehensive results dictionary.
    """
    output_dir = Path(raw_cfg.get("output_dir", "results/enter_leave_zeroshot"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "report").mkdir(exist_ok=True)

    configure_output(formats=["png"], dpi=200)

    # ----- Load data -----
    logger.info("=" * 60)
    logger.info("Loading data...")
    logger.info("=" * 60)

    sensor_array, sensor_timestamps, event_timestamps, event_labels, channel_names, class_names = (
        load_data(raw_cfg, include_none=include_none)
    )

    # ----- Build dataset (smart mode detection: bidirectional vs backward) -----
    ds_cfg_raw = raw_cfg.get("dataset", {})
    ds_config = build_dataset_config(ds_cfg_raw)
    dataset = EventDataset(
        sensor_array, sensor_timestamps, event_timestamps, event_labels, ds_config,
    )

    day_groups = dataset.get_day_groups()
    unique_days = sorted(np.unique(day_groups).tolist())
    logger.info("Events span %d calendar days", len(unique_days))

    # ----- Model config -----
    model_cfg = raw_cfg.get("model", {})
    pretrained_name = model_cfg.get("pretrained_name", "paris-noah/MantisV2")
    output_token = model_cfg.get("output_token", "combined")

    # ----- CV config -----
    cv_cfg = raw_cfg.get("cv", {})
    strat_k = cv_cfg.get("stratified_k", 5)
    strat_repeats = cv_cfg.get("stratified_repeats", 10)

    # ----- Extract embeddings for each layer -----
    all_embeddings = {}
    for layer in layers:
        logger.info("=" * 60)
        logger.info("Layer %d: Loading MantisV2 and extracting embeddings", layer)
        logger.info("=" * 60)

        network, model = load_mantis_model(pretrained_name, layer, output_token, device)
        Z = extract_all_embeddings(model, dataset)
        all_embeddings[layer] = Z

        # Free GPU memory (both trainer and network hold GPU tensors)
        del model, network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- Run cross-validation for each layer x classifier x CV method -----
    all_results = []  # list of dicts for CSV
    best_result = {"auc": -1}
    cv_method_results = defaultdict(lambda: defaultdict(dict))  # {cv_method: {clf: metric}}
    layer_auc = {}  # {layer: {best: best_auc}}
    stored_metrics = {}  # {(layer, clf_name, cv_method): metrics} for reuse

    n_events = len(event_labels)

    for layer in layers:
        Z = all_embeddings[layer]

        layer_best_auc = 0
        for clf_name in classifier_names:
            clf_factory = lambda name=clf_name, s=seed: build_classifier(name, s)

            for cv_method in cv_methods:
                logger.info("-" * 40)
                logger.info("L%d / %s / %s", layer, clf_name, cv_method)
                logger.info("-" * 40)

                t0 = time.time()

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
                    logger.warning("Unknown CV method: %s, skipping", cv_method)
                    continue

                elapsed = time.time() - t0
                stored_metrics[(layer, clf_name, cv_method)] = metrics

                logger.info(
                    "  Acc=%.4f  F1_macro=%.4f  AUC=%.4f  EER=%.4f  (%.1fs)",
                    metrics.accuracy, metrics.f1_macro, metrics.roc_auc, metrics.eer, elapsed,
                )

                result_row = {
                    "layer": layer,
                    "classifier": clf_name,
                    "cv_method": cv_method,
                    "n_folds": metrics.n_folds,
                    "accuracy": round(metrics.accuracy, 4),
                    "f1_macro": round(metrics.f1_macro, 4),
                    "f1_weighted": round(metrics.f1_weighted, 4),
                    "precision_macro": round(metrics.precision_macro, 4),
                    "recall_macro": round(metrics.recall_macro, 4),
                    "roc_auc": round(metrics.roc_auc, 4) if not np.isnan(metrics.roc_auc) else None,
                    "eer": round(metrics.eer, 4) if not np.isnan(metrics.eer) else None,
                    "n_samples": metrics.n_samples,
                    "time_sec": round(elapsed, 1),
                }
                all_results.append(result_row)

                # Track best
                auc_val = metrics.roc_auc if not np.isnan(metrics.roc_auc) else 0
                if auc_val > best_result["auc"]:
                    best_result = {
                        "auc": auc_val,
                        "layer": layer,
                        "classifier": clf_name,
                        "cv_method": cv_method,
                        "metrics": metrics,
                    }

                layer_best_auc = max(layer_best_auc, auc_val)

                # Track for CV comparison chart
                cv_method_results[cv_method][clf_name] = max(
                    cv_method_results[cv_method].get(clf_name, 0), auc_val,
                )

        layer_auc[layer] = {"best": layer_best_auc}

    # ----- Save CSV tables -----
    logger.info("=" * 60)
    logger.info("Saving results...")
    logger.info("=" * 60)

    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"

    # All results
    fieldnames = [
        "layer", "classifier", "cv_method", "n_folds",
        "accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro",
        "roc_auc", "eer", "n_samples", "time_sec",
    ]
    save_results_csv(all_results, tables_dir / "all_results.csv", fieldnames)
    save_results_txt(all_results, tables_dir / "all_results.txt", fieldnames)

    # Per-CV-method tables
    for cv_method in cv_methods:
        rows = [r for r in all_results if r["cv_method"] == cv_method]
        if rows:
            save_results_csv(rows, tables_dir / f"{cv_method}_results.csv", fieldnames)
            save_results_txt(rows, tables_dir / f"{cv_method}_results.txt", fieldnames)

    # Layer ablation table
    layer_rows = [{"layer": l, "best_auc": v["best"]} for l, v in sorted(layer_auc.items())]
    save_results_csv(layer_rows, tables_dir / "layer_ablation.csv")
    save_results_txt(layer_rows, tables_dir / "layer_ablation.txt")

    # CV comparison table
    cv_comp_rows = []
    for cv_method, clf_dict in cv_method_results.items():
        for clf_name, auc_val in clf_dict.items():
            cv_comp_rows.append({"cv_method": cv_method, "classifier": clf_name, "auc": round(auc_val, 4)})
    save_results_csv(cv_comp_rows, tables_dir / "cv_comparison.csv")
    save_results_txt(cv_comp_rows, tables_dir / "cv_comparison.txt")

    # ----- Generate plots -----
    logger.info("Generating plots...")

    # Layer ablation bar chart
    try:
        plot_layer_ablation_bar(layer_auc, "AUC", plots_dir / "layer_ablation_auc")
    except Exception:
        logger.warning("Failed to plot layer ablation", exc_info=True)

    # CV comparison bar chart
    try:
        cv_auc_data = {cv: {clf: round(v, 3) for clf, v in clfs.items()} for cv, clfs in cv_method_results.items()}
        if cv_auc_data:
            fig = plot_cv_comparison_bar(cv_auc_data, "AUC", plots_dir / "cv_comparison_bar")
            plt.close(fig)
    except Exception:
        logger.warning("Failed to plot CV comparison", exc_info=True)

    # Best config plots (ROC, confusion matrix, embeddings)
    best_metrics = best_result.get("metrics")
    if best_metrics is not None:
        best_layer = best_result["layer"]
        best_clf = best_result["classifier"]
        Z_best = all_embeddings[best_layer]

        # Confusion matrix from best
        try:
            fig, _ = plot_confusion_matrix(
                best_metrics.confusion_matrix,
                class_names=class_names,
                output_path=plots_dir / "confusion_matrix_best",
            )
            plt.close(fig)
        except Exception:
            logger.warning("Failed to plot confusion matrix", exc_info=True)

        # Embedding plots
        if not quick:
            try:
                fig = plot_embeddings_multi_method(
                    Z_best, event_labels,
                    title=f"Best: L{best_layer} / {best_clf}",
                    output_path=plots_dir / "embeddings_multi",
                )
                plt.close(fig)
            except Exception:
                logger.warning("Failed to plot embeddings", exc_info=True)
        else:
            try:
                emb_2d = reduce_dimensions(Z_best, method="pca")
                fig, _ = plot_embeddings(
                    emb_2d, event_labels,
                    title=f"PCA — L{best_layer}",
                    method="pca",
                    output_path=plots_dir / "embeddings_pca",
                )
                plt.close(fig)
            except Exception:
                logger.warning("Failed to plot PCA embeddings", exc_info=True)

    # ----- ROC overlay for best layer across classifiers -----
    if best_result.get("layer") is not None:
        best_layer = best_result["layer"]
        primary_cv = cv_methods[0] if cv_methods else "loocv"

        # Look up stored metrics instead of re-running CV
        overlay_results = {}
        for clf_name in classifier_names:
            key = (best_layer, clf_name, primary_cv)
            m = stored_metrics.get(key)
            if m is not None:
                overlay_results[clf_name] = {"metrics": m}

        try:
            if overlay_results:
                plot_roc_overlay(overlay_results, plots_dir / "roc_overlay")
        except Exception:
            logger.warning("Failed to plot ROC overlay", exc_info=True)

    # ----- Save JSON summary -----
    summary = {
        "best_config": {
            "layer": best_result.get("layer"),
            "classifier": best_result.get("classifier"),
            "cv_method": best_result.get("cv_method"),
            "auc": best_result.get("auc"),
        },
        "experiment_config": {
            "n_events": n_events,
            "n_channels": len(channel_names),
            "channel_names": channel_names,
            "class_names": class_names,
            "context_mode": ds_config.context_mode,
            "context_window": ds_config.describe(),
            "effective_seq_len": ds_config.effective_seq_len,
            "target_seq_len": ds_config.target_seq_len,
            "layers_tested": layers,
            "cv_methods": cv_methods,
            "classifiers": classifier_names,
            "include_none": include_none,
            "seed": seed,
        },
        "all_results": all_results,
    }

    if best_metrics is not None:
        summary["best_metrics"] = best_metrics.to_dict()

    report_path = output_dir / "report" / "summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved summary: %s", report_path)

    # Text analysis report
    report_lines = [
        "Enter/Leave Event Detection — Analysis Report",
        "=" * 60,
        "",
        f"Events: {n_events} ({', '.join(f'{n}={int((event_labels == i).sum())}' for i, n in enumerate(class_names))})",
        f"Channels: {len(channel_names)} ({', '.join(channel_names)})",
        f"Context window: {ds_config.describe()}",
        f"Calendar days: {len(unique_days)}",
        "",
        "Best Configuration:",
        f"  Layer: L{best_result.get('layer')}",
        f"  Classifier: {best_result.get('classifier')}",
        f"  CV Method: {best_result.get('cv_method')}",
        f"  AUC: {best_result.get('auc', 0):.4f}",
    ]

    if best_metrics is not None:
        report_lines.extend([
            f"  Accuracy: {best_metrics.accuracy:.4f}",
            f"  F1 (macro): {best_metrics.f1_macro:.4f}",
            f"  EER: {best_metrics.eer:.4f}",
            "",
            best_metrics.summary(class_names),
        ])

    report_lines.extend([
        "",
        "Layer Ablation (best AUC per layer):",
    ])
    for layer in sorted(layer_auc):
        report_lines.append(f"  L{layer}: {layer_auc[layer]['best']:.4f}")

    report_txt_path = output_dir / "report" / "analysis_report.txt"
    report_txt_path.write_text("\n".join(report_lines) + "\n")
    logger.info("Saved report: %s", report_txt_path)

    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enter/Leave Event Detection with MantisV2 Zero-Shot Embeddings",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: L0 only, LOOCV only, RF only",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Transformer layers to test (default: from config)",
    )
    parser.add_argument(
        "--cv-methods", nargs="+", default=None,
        help="CV methods: loocv, stratified_kfold, lodo (default: from config)",
    )
    parser.add_argument(
        "--classifiers", nargs="+", default=None,
        help="Classifiers: random_forest, svm, nearest_centroid (default: from config)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: cuda or cpu (default: from config)",
    )
    parser.add_argument(
        "--include-none", action="store_true",
        help="Include NONE as 3rd class",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: from config)",
    )
    parser.add_argument(
        "--context-mode", choices=["bidirectional", "backward"], default=None,
        help="Context window mode (default: from config or bidirectional)",
    )
    parser.add_argument(
        "--context-before", type=int, default=None,
        help="Minutes before event (bidirectional mode, default: 4)",
    )
    parser.add_argument(
        "--context-after", type=int, default=None,
        help="Minutes after event (bidirectional mode, default: 4)",
    )
    parser.add_argument(
        "--backward-only", action="store_true",
        help="Use backward-only context (seq_len from config, default 512)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    raw_cfg = load_config(args.config)

    # Apply CLI overrides to dataset config
    ds_overrides = {}
    if args.backward_only:
        ds_overrides["context_mode"] = "backward"
    elif args.context_mode:
        ds_overrides["context_mode"] = args.context_mode
    if args.context_before is not None:
        ds_overrides["context_before"] = args.context_before
    if args.context_after is not None:
        ds_overrides["context_after"] = args.context_after
    if ds_overrides:
        raw_cfg.setdefault("dataset", {}).update(ds_overrides)

    # Resolve parameters (CLI overrides config)
    seed = args.seed if args.seed is not None else raw_cfg.get("seed", 42)
    device = args.device if args.device is not None else raw_cfg.get("model", {}).get("device", "cuda")

    model_cfg = raw_cfg.get("model", {})
    cv_cfg = raw_cfg.get("cv", {})

    if args.quick:
        layers = [0]
        cv_methods = ["loocv"]
        classifier_names = ["random_forest"]
    else:
        layers = args.layers if args.layers is not None else model_cfg.get("layers", ALL_LAYERS)
        cv_methods = args.cv_methods if args.cv_methods is not None else cv_cfg.get("methods", ["loocv"])
        classifier_names = args.classifiers if args.classifiers is not None else raw_cfg.get(
            "classifiers", ["random_forest", "svm", "nearest_centroid"],
        )

    include_none = args.include_none or raw_cfg.get("data", {}).get("include_none", False)

    logger.info("=" * 60)
    logger.info("Enter/Leave Event Detection")
    logger.info("=" * 60)
    logger.info("  Layers: %s", layers)
    logger.info("  CV methods: %s", cv_methods)
    logger.info("  Classifiers: %s", classifier_names)
    logger.info("  Device: %s", device)
    logger.info("  Seed: %d", seed)
    logger.info("  Include NONE: %s", include_none)
    logger.info("  Quick mode: %s", args.quick)

    t_start = time.time()

    summary = run_experiment(
        raw_cfg,
        layers=layers,
        cv_methods=cv_methods,
        classifier_names=classifier_names,
        include_none=include_none,
        device=device,
        seed=seed,
        quick=args.quick,
    )

    t_total = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Done! Total time: %.1fs (%.1f min)", t_total, t_total / 60)

    best = summary.get("best_config", {})
    logger.info(
        "Best: L%s / %s / %s — AUC=%.4f",
        best.get("layer"), best.get("classifier"),
        best.get("cv_method"), best.get("auc", 0),
    )


if __name__ == "__main__":
    main()
