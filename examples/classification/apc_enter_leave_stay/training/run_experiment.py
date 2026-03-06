"""Enter/Leave/Stay Event Classification with MantisV2 Embeddings.

Dual-track experiment runner:
  Track 1 (sklearn): Frozen MantisV2 embeddings -> sklearn classifier -> LOOCV
  Track 2 (neural):  Frozen MantisV2 embeddings -> trainable head -> LOOCV

Supports 3 label settings (5-class, 3-class event, 3-class occupancy).
Default: Setting 3 (3-class occupancy) with LOOCV.

Usage:
    cd examples/classification/apc_enter_leave_stay
    python training/run_experiment.py --config training/configs/setting3-loocv.yaml

    # Quick mode: L0 only, RF only, no neural
    python training/run_experiment.py --config ... --quick

    # Setting 1 (5-class)
    python training/run_experiment.py --config ... --label-setting 1

    # Neural track only
    python training/run_experiment.py --config ... --neural-only

    # Sklearn track only (skip neural)
    python training/run_experiment.py --config ... --no-neural
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

import matplotlib as mpl
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    mpl.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from data.preprocess import EventPreprocessConfig, load_sensor_and_events, CLASS_NAMES_BY_SETTING
from data.dataset import EventDatasetConfig, EventDataset, build_dataset_config
from evaluation.metrics import (
    EventClassificationMetrics,
    aggregate_cv_predictions,
)
from visualization.style import setup_style, save_figure, configure_output, FIGSIZE_SINGLE
from visualization.curves import plot_confusion_matrix, plot_cv_comparison_bar
from visualization.embeddings import (
    reduce_dimensions, plot_embeddings, plot_embeddings_multi_method,
)

logger = logging.getLogger(__name__)

ALL_LAYERS = [0, 1, 2, 3, 4, 5]


# ============================================================================
# Config loading
# ============================================================================

def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================================
# Data loading
# ============================================================================

def load_data(raw_cfg: dict, label_setting: int = 3):
    """Load sensor data and events with label setting applied."""
    data_cfg = raw_cfg.get("data", {})

    config = EventPreprocessConfig(
        sensor_csv=data_cfg.get("sensor_csv", ""),
        events_csv=data_cfg.get("events_csv", ""),
        column_names_csv=data_cfg.get("column_names_csv"),
        column_names=data_cfg.get("column_names"),
        channels=data_cfg.get("channels"),
        exclude_channels=data_cfg.get("exclude_channels", []),
        nan_threshold=data_cfg.get("nan_threshold", 0.3),
        label_setting=label_setting,
        add_time_features=data_cfg.get("add_time_features", False),
    )

    return load_sensor_and_events(config)


# ============================================================================
# Model loading
# ============================================================================

def load_mantis_model(pretrained_name: str, layer: int, output_token: str, device: str):
    """Load MantisV2 at a specific transformer layer."""
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
    """Extract embeddings for ALL events at once (frozen model = deterministic)."""
    X, _ = dataset.get_numpy_arrays()
    t0 = time.time()
    Z = model.transform(X)
    elapsed = time.time() - t0

    nan_count = int(np.isnan(Z).sum())
    if nan_count > 0:
        logger.warning(
            "%d NaN values in embeddings (%.1f%%), replacing with 0",
            nan_count, 100.0 * nan_count / Z.size,
        )
        Z = np.nan_to_num(Z, nan=0.0)

    logger.info("Extracted embeddings: %s -> %s (%.1fs)", X.shape, Z.shape, elapsed)
    return Z


# ============================================================================
# Sklearn classifier factory
# ============================================================================

def build_sklearn_classifier(name: str, seed: int = 42):
    """Build a single sklearn classifier by name. Uses class_weight='balanced' where available."""
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestCentroid
    from sklearn.svm import SVC

    classifiers = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed,
            class_weight="balanced",
        ),
        "svm": lambda: SVC(
            kernel="rbf", C=1.0, probability=True, random_state=seed,
            class_weight="balanced",
        ),
        "logistic_regression": lambda: LogisticRegression(
            max_iter=1000, random_state=seed,
            class_weight="balanced",
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed,
            class_weight="balanced",
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=100, random_state=seed,
        ),
        "nearest_centroid": lambda: NearestCentroid(),
    }

    factory = classifiers.get(name)
    if factory is None:
        raise ValueError(f"Unknown classifier: {name!r}. Available: {list(classifiers)}")
    return factory()


# ============================================================================
# LOOCV — sklearn track
# ============================================================================

def run_loocv_sklearn(
    Z: np.ndarray,
    y: np.ndarray,
    clf_factory,
    class_names: list[str] | None = None,
) -> EventClassificationMetrics:
    """Leave-One-Out Cross-Validation with sklearn classifier."""
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    n_classes = len(np.unique(y))

    test_clf = clf_factory()
    has_prob = hasattr(test_clf, "predict_proba")
    del test_clf

    y_pred_all = np.zeros(n, dtype=np.int64)
    y_prob_all = np.zeros((n, n_classes), dtype=np.float64) if has_prob else None

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
            proba = clf.predict_proba(Z_test_s)
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


# ============================================================================
# LOOCV — neural track
# ============================================================================

def run_loocv_neural(
    Z: np.ndarray,
    y: np.ndarray,
    head_type: str = "mlp",
    n_classes: int = 3,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    dropout: float = 0.5,
    hidden_dims: list[int] | None = None,
    device: str = "cpu",
    class_names: list[str] | None = None,
) -> EventClassificationMetrics:
    """Leave-One-Out CV with neural classification head.

    Each fold trains a fresh head on N-1 embeddings, predicts 1.
    """
    from training.heads import build_head

    n = len(y)
    embed_dim = Z.shape[1]

    # Compute class weights for loss
    unique, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    weights = total / (len(unique) * counts)
    class_weights = torch.zeros(n_classes, dtype=torch.float32)
    for cls, w in zip(unique, weights):
        class_weights[cls] = w
    class_weights = class_weights.to(device)

    y_pred_all = np.zeros(n, dtype=np.int64)
    y_prob_all = np.zeros((n, n_classes), dtype=np.float64)

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        Z_train = torch.from_numpy(Z[train_mask]).float().to(device)
        y_train = torch.from_numpy(y[train_mask]).long().to(device)
        Z_test = torch.from_numpy(Z[i : i + 1]).float().to(device)

        # Normalize (per-fold)
        mean = Z_train.mean(dim=0, keepdim=True)
        std = Z_train.std(dim=0, keepdim=True) + 1e-8
        Z_train = (Z_train - mean) / std
        Z_test = (Z_test - mean) / std

        # Build head
        head_kwargs = {"dropout": dropout}
        if hidden_dims is not None and head_type == "mlp":
            head_kwargs["hidden_dims"] = hidden_dims
        head = build_head(head_type, embed_dim, n_classes, **head_kwargs).to(device)

        # Train
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        head.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = head(Z_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        # Predict
        head.eval()
        with torch.no_grad():
            logits = head(Z_test)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = int(logits.argmax(dim=-1).cpu().item())

        y_pred_all[i] = pred
        y_prob_all[i] = probs[0]

    return aggregate_cv_predictions(
        y, y_pred_all, y_prob_all,
        cv_method="LOOCV", n_folds=n, class_names=class_names,
    )


# ============================================================================
# Visualization helpers
# ============================================================================

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

def save_results_csv(results: list[dict], output_path: Path, fieldnames: list[str] | None = None) -> None:
    if not results:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen = set()
        fieldnames = []
        for r in results:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info("Saved: %s (%d rows)", output_path, len(results))


def save_results_txt(results: list[dict], output_path: Path, fieldnames: list[str] | None = None) -> None:
    if not results:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen = set()
        fieldnames = []
        for r in results:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)

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
    label_setting: int = 3,
    sklearn_classifiers: list[str] | None = None,
    run_neural: bool = True,
    neural_heads: list[str] | None = None,
    device: str = "cuda",
    seed: int = 42,
    quick: bool = False,
) -> dict:
    """Run the full enter/leave/stay classification experiment."""
    output_dir = Path(raw_cfg.get("output_dir", "results/enter_leave_stay"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "report").mkdir(exist_ok=True)

    configure_output(formats=["png"], dpi=200)

    class_names = CLASS_NAMES_BY_SETTING[label_setting]
    n_classes = len(class_names)

    # ----- Load data -----
    logger.info("=" * 60)
    logger.info("Loading data (label_setting=%d, %d classes)", label_setting, n_classes)
    logger.info("=" * 60)

    sensor_array, sensor_timestamps, event_timestamps, event_labels, channel_names, class_names = (
        load_data(raw_cfg, label_setting=label_setting)
    )

    # ----- Build dataset -----
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

    # ----- Neural config -----
    neural_cfg = raw_cfg.get("neural", {})
    neural_epochs = neural_cfg.get("epochs", 50)
    neural_lr = neural_cfg.get("lr", 0.001)
    neural_wd = neural_cfg.get("weight_decay", 0.01)
    neural_dropout = neural_cfg.get("dropout", 0.5)
    neural_hidden = neural_cfg.get("hidden_dims", [128, 64])

    if sklearn_classifiers is None:
        sklearn_classifiers = raw_cfg.get("sklearn_classifiers", ["random_forest", "svm", "nearest_centroid"])
    if neural_heads is None:
        neural_heads = neural_cfg.get("heads", ["linear", "mlp"])

    # ----- Extract embeddings for each layer -----
    all_embeddings = {}
    for layer in layers:
        logger.info("=" * 60)
        logger.info("Layer %d: Loading MantisV2 and extracting embeddings", layer)
        logger.info("=" * 60)

        network, model = load_mantis_model(pretrained_name, layer, output_token, device)
        Z = extract_all_embeddings(model, dataset)
        all_embeddings[layer] = Z

        del model, network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- Run experiments -----
    all_results = []
    best_result = {"f1_macro": -1}
    layer_best = {}
    n_events = len(event_labels)

    for layer in layers:
        Z = all_embeddings[layer]

        layer_best_f1 = 0

        # Track 1: sklearn classifiers
        for clf_name in sklearn_classifiers:
            clf_factory = lambda name=clf_name, s=seed: build_sklearn_classifier(name, s)

            logger.info("-" * 40)
            logger.info("[sklearn] L%d / %s / LOOCV", layer, clf_name)
            logger.info("-" * 40)

            t0 = time.time()
            metrics = run_loocv_sklearn(Z, event_labels, clf_factory, class_names)
            elapsed = time.time() - t0

            logger.info(
                "  Acc=%.4f  F1_macro=%.4f  AUC=%.4f  (%.1fs)",
                metrics.accuracy, metrics.f1_macro, metrics.roc_auc, elapsed,
            )

            result_row = _build_result_row(
                layer, f"sklearn/{clf_name}", "LOOCV", metrics, elapsed,
            )
            all_results.append(result_row)

            if metrics.f1_macro > best_result["f1_macro"]:
                best_result = {
                    "f1_macro": metrics.f1_macro,
                    "layer": layer,
                    "classifier": f"sklearn/{clf_name}",
                    "metrics": metrics,
                }

            layer_best_f1 = max(layer_best_f1, metrics.f1_macro)

        # Track 2: neural heads
        if run_neural:
            for head_type in neural_heads:
                logger.info("-" * 40)
                logger.info("[neural] L%d / %s / LOOCV", layer, head_type)
                logger.info("-" * 40)

                t0 = time.time()
                metrics = run_loocv_neural(
                    Z, event_labels,
                    head_type=head_type,
                    n_classes=n_classes,
                    epochs=neural_epochs,
                    lr=neural_lr,
                    weight_decay=neural_wd,
                    dropout=neural_dropout,
                    hidden_dims=neural_hidden,
                    device=device,
                    class_names=class_names,
                )
                elapsed = time.time() - t0

                logger.info(
                    "  Acc=%.4f  F1_macro=%.4f  AUC=%.4f  (%.1fs)",
                    metrics.accuracy, metrics.f1_macro, metrics.roc_auc, elapsed,
                )

                result_row = _build_result_row(
                    layer, f"neural/{head_type}", "LOOCV", metrics, elapsed,
                )
                all_results.append(result_row)

                if metrics.f1_macro > best_result["f1_macro"]:
                    best_result = {
                        "f1_macro": metrics.f1_macro,
                        "layer": layer,
                        "classifier": f"neural/{head_type}",
                        "metrics": metrics,
                    }

                layer_best_f1 = max(layer_best_f1, metrics.f1_macro)

        layer_best[layer] = {"best": layer_best_f1}

    # ----- Save results -----
    logger.info("=" * 60)
    logger.info("Saving results...")
    logger.info("=" * 60)

    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"

    fieldnames = [
        "layer", "classifier", "cv_method", "n_folds",
        "accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro",
        "roc_auc", "n_samples", "time_sec",
    ]
    save_results_csv(all_results, tables_dir / "all_results.csv", fieldnames)
    save_results_txt(all_results, tables_dir / "all_results.txt", fieldnames)

    # Layer ablation
    layer_rows = [{"layer": l, "best_f1_macro": round(v["best"], 4)} for l, v in sorted(layer_best.items())]
    save_results_csv(layer_rows, tables_dir / "layer_ablation.csv")

    # ----- Plots -----
    logger.info("Generating plots...")

    # Layer ablation bar chart
    try:
        plot_layer_ablation_bar(layer_best, "F1 (macro)", plots_dir / "layer_ablation_f1")
    except Exception:
        logger.warning("Failed to plot layer ablation", exc_info=True)

    # Best confusion matrix
    best_metrics = best_result.get("metrics")
    if best_metrics is not None:
        best_layer = best_result["layer"]
        Z_best = all_embeddings[best_layer]

        try:
            fig, _ = plot_confusion_matrix(
                best_metrics.confusion_matrix,
                class_names=class_names,
                output_path=plots_dir / "confusion_matrix_best",
            )
            plt.close(fig)
        except Exception:
            logger.warning("Failed to plot confusion matrix", exc_info=True)

        # Embedding visualizations
        if not quick:
            try:
                fig = plot_embeddings_multi_method(
                    Z_best, event_labels,
                    class_names=class_names,
                    title=f"Best: L{best_layer} / {best_result['classifier']}",
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
                    title=f"PCA - L{best_layer}",
                    method="pca",
                    class_names=class_names,
                    output_path=plots_dir / "embeddings_pca",
                )
                plt.close(fig)
            except Exception:
                logger.warning("Failed to plot PCA", exc_info=True)

    # ----- JSON summary -----
    summary = {
        "label_setting": label_setting,
        "n_classes": n_classes,
        "class_names": class_names,
        "best_config": {
            "layer": best_result.get("layer"),
            "classifier": best_result.get("classifier"),
            "f1_macro": best_result.get("f1_macro"),
        },
        "experiment_config": {
            "n_events": n_events,
            "n_channels": len(channel_names),
            "channel_names": channel_names,
            "context_window": ds_config.describe(),
            "layers_tested": layers,
            "sklearn_classifiers": sklearn_classifiers,
            "neural_heads": neural_heads if run_neural else [],
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

    # Text report
    report_lines = [
        f"Enter/Leave/Stay Classification - Setting {label_setting}",
        "=" * 60,
        "",
        f"Events: {n_events} ({', '.join(f'{n}={int((event_labels == i).sum())}' for i, n in enumerate(class_names))})",
        f"Channels: {len(channel_names)} ({', '.join(channel_names)})",
        f"Context: {ds_config.describe()}",
        f"Calendar days: {len(unique_days)}",
        "",
        "Best Configuration:",
        f"  Layer: L{best_result.get('layer')}",
        f"  Classifier: {best_result.get('classifier')}",
        f"  F1 (macro): {best_result.get('f1_macro', 0):.4f}",
    ]

    if best_metrics is not None:
        report_lines.extend([
            f"  Accuracy: {best_metrics.accuracy:.4f}",
            f"  AUC: {best_metrics.roc_auc:.4f}",
            "",
            best_metrics.summary(class_names),
        ])

    report_lines.extend([
        "",
        "Layer Ablation (best F1 macro per layer):",
    ])
    for layer in sorted(layer_best):
        report_lines.append(f"  L{layer}: {layer_best[layer]['best']:.4f}")

    report_txt_path = output_dir / "report" / "analysis_report.txt"
    report_txt_path.write_text("\n".join(report_lines) + "\n")
    logger.info("Saved report: %s", report_txt_path)

    return summary


def _build_result_row(
    layer: int, classifier: str, cv_method: str,
    metrics: EventClassificationMetrics, elapsed: float,
) -> dict:
    return {
        "layer": layer,
        "classifier": classifier,
        "cv_method": cv_method,
        "n_folds": metrics.n_folds,
        "accuracy": round(metrics.accuracy, 4),
        "f1_macro": round(metrics.f1_macro, 4),
        "f1_weighted": round(metrics.f1_weighted, 4),
        "precision_macro": round(metrics.precision_macro, 4),
        "recall_macro": round(metrics.recall_macro, 4),
        "roc_auc": round(metrics.roc_auc, 4) if not np.isnan(metrics.roc_auc) else None,
        "n_samples": metrics.n_samples,
        "time_sec": round(elapsed, 1),
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enter/Leave/Stay Classification with MantisV2 Embeddings",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--quick", action="store_true", help="Quick mode: L0, RF only, no neural")
    parser.add_argument("--label-setting", type=int, default=None, help="Label setting: 1, 2, or 3 (default: from config)")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Transformer layers to test")
    parser.add_argument("--classifiers", nargs="+", default=None, help="Sklearn classifiers to use")
    parser.add_argument("--neural-only", action="store_true", help="Run neural track only")
    parser.add_argument("--no-neural", action="store_true", help="Skip neural track")
    parser.add_argument("--neural-heads", nargs="+", default=None, help="Neural head types: linear, mlp")
    parser.add_argument("--device", default=None, help="Device: cuda or cpu")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--context-before", type=int, default=None, help="Minutes before event")
    parser.add_argument("--context-after", type=int, default=None, help="Minutes after event")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    raw_cfg = load_config(args.config)

    # CLI overrides
    ds_overrides = {}
    if args.context_before is not None:
        ds_overrides["context_before"] = args.context_before
    if args.context_after is not None:
        ds_overrides["context_after"] = args.context_after
    if ds_overrides:
        raw_cfg.setdefault("dataset", {}).update(ds_overrides)

    label_setting = args.label_setting if args.label_setting is not None else raw_cfg.get("label_setting", 3)
    seed = args.seed if args.seed is not None else raw_cfg.get("seed", 42)
    device = args.device if args.device is not None else raw_cfg.get("model", {}).get("device", "cuda")

    model_cfg = raw_cfg.get("model", {})

    if args.quick:
        layers = [0]
        sklearn_classifiers = ["random_forest"]
        run_neural = False
        neural_heads = []
    else:
        layers = args.layers if args.layers is not None else model_cfg.get("layers", ALL_LAYERS)
        sklearn_classifiers = args.classifiers if args.classifiers is not None else None
        neural_heads = args.neural_heads

        if args.neural_only:
            sklearn_classifiers = []
            run_neural = True
        elif args.no_neural:
            run_neural = False
        else:
            run_neural = raw_cfg.get("neural", {}).get("enabled", True)

    logger.info("=" * 60)
    logger.info("Enter/Leave/Stay Classification")
    logger.info("=" * 60)
    logger.info("  Label setting: %d (%s)", label_setting, CLASS_NAMES_BY_SETTING[label_setting])
    logger.info("  Layers: %s", layers)
    logger.info("  Sklearn classifiers: %s", sklearn_classifiers or "(from config)")
    logger.info("  Neural: %s (heads=%s)", run_neural, neural_heads or "(from config)")
    logger.info("  Device: %s", device)
    logger.info("  Seed: %d", seed)
    logger.info("  Quick mode: %s", args.quick)

    t_start = time.time()

    summary = run_experiment(
        raw_cfg,
        layers=layers,
        label_setting=label_setting,
        sklearn_classifiers=sklearn_classifiers,
        run_neural=run_neural,
        neural_heads=neural_heads,
        device=device,
        seed=seed,
        quick=args.quick,
    )

    t_total = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Done! Total time: %.1fs (%.1f min)", t_total, t_total / 60)

    best = summary.get("best_config", {})
    logger.info(
        "Best: L%s / %s — F1_macro=%.4f",
        best.get("layer"), best.get("classifier"), best.get("f1_macro", 0),
    )


if __name__ == "__main__":
    main()
