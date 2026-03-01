#!/usr/bin/env python3
"""Phase 1 Comprehensive Sweep: MantisV2 Frozen Embeddings + sklearn Classifiers.

Designed for 3 parallel GPU servers to find optimal configuration.

Three sweep groups (run independently on separate GPUs):

  Group A — Context Window × Layer (84 experiments)
    - 14 context settings × 6 layers
    - Fixed: RF, M+C channels, combined token
    - Goal: Find optimal temporal context and MantisV2 layer

  Group B — Channel Ablation (126 experiments)
    - 21 channel combos × 6 layers
    - Fixed: RF, 9min context (4+1+4), combined token
    - Goal: Find optimal sensor channel combination

  Group C — Classifier × Output Token × Interp Length (54 experiments)
    - 3 classifiers × 3 tokens × 6 target_seq_len
    - Fixed: best layer (default L3), M+C, 9min context
    - Goal: Fine-tune representation and classifier choice

Usage:
  cd examples/classification/apc_occupancy

  # Run all groups sequentially (~8-12h total on A100)
  python training/run_phase1_sweep.py \\
      --config training/configs/occupancy-phase1.yaml --device cuda

  # Run single group (for parallel GPU servers, ~3-4h each)
  python training/run_phase1_sweep.py \\
      --config training/configs/occupancy-phase1.yaml --group A --device cuda
  python training/run_phase1_sweep.py \\
      --config training/configs/occupancy-phase1.yaml --group B --device cuda
  python training/run_phase1_sweep.py \\
      --config training/configs/occupancy-phase1.yaml --group C --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Headless matplotlib
import matplotlib as mpl
if not os.environ.get("DISPLAY"):
    mpl.use("Agg")
import matplotlib.pyplot as plt

# Local imports — run from apc_occupancy/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import PreprocessConfig, load_occupancy_data
from data.dataset import DatasetConfig, OccupancyDataset, build_dataset_config
from evaluation.metrics import compute_metrics, compute_wilson_ci, ClassificationMetrics

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

ALL_LAYERS = [0, 1, 2, 3, 4, 5]

# 14 context window configurations (bidirectional + 1 backward comparison)
CONTEXT_CONFIGS = [
    {"name": "1min (0+1+0)", "context_before": 0, "context_after": 0, "context_mode": "bidirectional"},
    {"name": "3min (1+1+1)", "context_before": 1, "context_after": 1, "context_mode": "bidirectional"},
    {"name": "5min (2+1+2)", "context_before": 2, "context_after": 2, "context_mode": "bidirectional"},
    {"name": "7min (3+1+3)", "context_before": 3, "context_after": 3, "context_mode": "bidirectional"},
    {"name": "9min (4+1+4)", "context_before": 4, "context_after": 4, "context_mode": "bidirectional"},
    {"name": "11min (5+1+5)", "context_before": 5, "context_after": 5, "context_mode": "bidirectional"},
    {"name": "15min (7+1+7)", "context_before": 7, "context_after": 7, "context_mode": "bidirectional"},
    {"name": "21min (10+1+10)", "context_before": 10, "context_after": 10, "context_mode": "bidirectional"},
    {"name": "31min (15+1+15)", "context_before": 15, "context_after": 15, "context_mode": "bidirectional"},
    {"name": "41min (20+1+20)", "context_before": 20, "context_after": 20, "context_mode": "bidirectional"},
    {"name": "61min (30+1+30)", "context_before": 30, "context_after": 30, "context_mode": "bidirectional"},
    {"name": "121min (60+1+60)", "context_before": 60, "context_after": 60, "context_mode": "bidirectional"},
    {"name": "241min (120+1+120)", "context_before": 120, "context_after": 120, "context_mode": "bidirectional"},
    {"name": "31min backward (30+1+0)", "context_before": 30, "context_after": 0, "context_mode": "bidirectional"},
]

# 21 channel combinations
CHANNEL_CONFIGS = [
    # Single channels
    {"name": "M only", "channels": ["d620900d_motionSensor"]},
    {"name": "P only", "channels": ["f2e891c6_powerMeter"]},
    {"name": "T1 only", "channels": ["d620900d_temperatureMeasurement"]},
    {"name": "T2 only", "channels": ["ccea734e_temperatureMeasurement"]},
    {"name": "C only", "channels": ["408981c2_contactSensor"]},
    {"name": "E only", "channels": ["408981c2_temperatureMeasurement"]},
    # Pairs
    {"name": "M+C", "channels": ["d620900d_motionSensor", "408981c2_contactSensor"]},
    {"name": "M+T1", "channels": ["d620900d_motionSensor", "d620900d_temperatureMeasurement"]},
    {"name": "M+P", "channels": ["d620900d_motionSensor", "f2e891c6_powerMeter"]},
    {"name": "M+T2", "channels": ["d620900d_motionSensor", "ccea734e_temperatureMeasurement"]},
    {"name": "T1+C", "channels": ["d620900d_temperatureMeasurement", "408981c2_contactSensor"]},
    {"name": "P+C", "channels": ["f2e891c6_powerMeter", "408981c2_contactSensor"]},
    {"name": "M+E", "channels": ["d620900d_motionSensor", "408981c2_temperatureMeasurement"]},
    # Triples
    {"name": "M+C+T1", "channels": ["d620900d_motionSensor", "408981c2_contactSensor", "d620900d_temperatureMeasurement"]},
    {"name": "M+C+P", "channels": ["d620900d_motionSensor", "408981c2_contactSensor", "f2e891c6_powerMeter"]},
    {"name": "M+P+T1", "channels": ["d620900d_motionSensor", "f2e891c6_powerMeter", "d620900d_temperatureMeasurement"]},
    {"name": "M+C+E", "channels": ["d620900d_motionSensor", "408981c2_contactSensor", "408981c2_temperatureMeasurement"]},
    # Quads and more
    {"name": "M+P+T1+T2", "channels": ["d620900d_motionSensor", "f2e891c6_powerMeter", "d620900d_temperatureMeasurement", "ccea734e_temperatureMeasurement"]},
    {"name": "M+C+P+T1", "channels": ["d620900d_motionSensor", "408981c2_contactSensor", "f2e891c6_powerMeter", "d620900d_temperatureMeasurement"]},
    {"name": "M+C+P+T1+T2", "channels": ["d620900d_motionSensor", "408981c2_contactSensor", "f2e891c6_powerMeter", "d620900d_temperatureMeasurement", "ccea734e_temperatureMeasurement"]},
    {"name": "All 6", "channels": ["d620900d_motionSensor", "408981c2_contactSensor", "f2e891c6_powerMeter", "d620900d_temperatureMeasurement", "ccea734e_temperatureMeasurement", "408981c2_temperatureMeasurement"]},
]

CLASSIFIER_NAMES = ["random_forest", "svm", "nearest_centroid"]
OUTPUT_TOKENS = ["combined", "cls", "mean"]
TARGET_SEQ_LENS = [64, 96, 128, 192, 256, 512]


# ============================================================================
# Model + Embedding utilities
# ============================================================================

def load_mantis_model(pretrained_name: str, layer: int, output_token: str, device: str):
    """Load MantisV2 model + trainer wrapper."""
    from mantis import MantisV2, MantisTrainer

    model = MantisV2.from_pretrained(pretrained_name)
    trainer = MantisTrainer(model, return_transf_layer=layer, output_token=output_token)
    trainer.to(device)
    return trainer


def extract_embeddings(trainer, dataset: OccupancyDataset, device: str) -> np.ndarray:
    """Extract frozen embeddings for all windows in the dataset.

    Returns shape (n_windows, embed_dim).
    """
    import torch

    X, _ = dataset.get_numpy_arrays()  # (N, C, L)
    n_samples, n_channels, seq_len = X.shape

    all_embeddings = []
    trainer.model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            sample = torch.from_numpy(X[i]).to(device)  # (C, L)
            # MantisV2 expects (batch, 1, L) per channel
            channel_embeds = []
            for ch in range(n_channels):
                inp = sample[ch].unsqueeze(0).unsqueeze(0)  # (1, 1, L)
                emb = trainer.transform(inp)  # (1, D)
                channel_embeds.append(emb.cpu().numpy())
            # Concatenate channel embeddings
            combined = np.concatenate(channel_embeds, axis=-1)  # (1, C*D)
            all_embeddings.append(combined)

    Z = np.concatenate(all_embeddings, axis=0)  # (N, C*D)

    # NaN safety
    if np.isnan(Z).any():
        n_nan = np.isnan(Z).sum()
        logger.warning("Found %d NaN values in embeddings, replacing with 0", n_nan)
        Z = np.nan_to_num(Z, nan=0.0)

    return Z


def build_classifier(name: str, seed: int = 42):
    """Build sklearn classifier by name."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import NearestCentroid
    from sklearn.svm import SVC

    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    elif name == "svm":
        return SVC(kernel="rbf", C=1.0, probability=True, random_state=seed)
    elif name == "nearest_centroid":
        return NearestCentroid()
    else:
        raise ValueError(f"Unknown classifier: {name}")


def _extract_prob_positive(clf, X_test) -> np.ndarray | None:
    """Extract P(class=1) from classifier, handling edge cases."""
    if not hasattr(clf, "predict_proba"):
        return None
    try:
        proba = clf.predict_proba(X_test)
    except Exception:
        return None

    if hasattr(clf, "classes_"):
        classes = list(clf.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        else:
            return np.zeros(len(X_test))
    return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]


# ============================================================================
# Evaluation runner
# ============================================================================

def run_train_test_eval(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    clf_name: str,
    seed: int = 42,
) -> ClassificationMetrics:
    """Train classifier and evaluate on test set."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_test_s = scaler.transform(Z_test)

    clf = build_classifier(clf_name, seed)
    clf.fit(Z_train_s, y_train)

    y_pred = clf.predict(Z_test_s)
    y_prob = _extract_prob_positive(clf, Z_test_s)

    metrics = compute_metrics(y_test, y_pred, y_prob, class_names=["Empty", "Occupied"])
    return metrics


# ============================================================================
# Group A: Context Window × Layer
# ============================================================================

def run_group_a(cfg: dict, device: str, output_dir: Path) -> list[dict]:
    """Sweep context windows × layers with RF classifier.

    14 contexts × 6 layers = 84 experiments.
    """
    logger.info("=" * 70)
    logger.info("GROUP A: Context Window × Layer Sweep (84 experiments)")
    logger.info("=" * 70)

    results = []
    default_channels = cfg.get("default_channels", ["d620900d_motionSensor", "408981c2_contactSensor"])
    split_date = cfg.get("split_date", "2026-02-15")
    pretrained = cfg["model"]["pretrained_name"]
    seed = cfg.get("seed", 42)

    for ctx_cfg in CONTEXT_CONFIGS:
        ctx_name = ctx_cfg["name"]
        logger.info("--- Context: %s ---", ctx_name)

        # Build dataset with this context configuration
        ds_config = DatasetConfig(
            context_mode=ctx_cfg["context_mode"],
            context_before=ctx_cfg["context_before"],
            context_after=ctx_cfg["context_after"],
            stride=cfg.get("stride", 1),
        )

        prep_cfg = PreprocessConfig(
            sensor_csv=cfg["data"]["sensor_csv"],
            label_csv=cfg["data"]["label_csv"],
            label_format="events",
            initial_occupancy=cfg["data"].get("initial_occupancy", 0),
            binarize=True,
            channels=default_channels,
        )

        sensor_arr, train_labels, test_labels, ch_names, timestamps = (
            load_occupancy_data(prep_cfg, split_date=split_date)
        )

        # Build unified dataset then split
        all_labels = np.where(train_labels >= 0, train_labels, test_labels)
        dataset = OccupancyDataset(sensor_arr, all_labels, timestamps, ds_config)
        train_mask, test_mask = dataset.get_train_test_split(split_date)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            logger.warning("Skipping %s: empty train or test split", ctx_name)
            continue

        for layer in ALL_LAYERS:
            exp_name = f"{ctx_name} | L{layer}"
            logger.info("  %s", exp_name)
            t0 = time.time()

            try:
                trainer = load_mantis_model(pretrained, layer, "combined", device)
                Z = extract_embeddings(trainer, dataset, device)
                del trainer

                X_all, y_all = Z, dataset.labels
                Z_train = X_all[train_mask]
                y_train = y_all[train_mask]
                Z_test = X_all[test_mask]
                y_test = y_all[test_mask]

                metrics = run_train_test_eval(Z_train, y_train, Z_test, y_test, "random_forest", seed)
                elapsed = time.time() - t0

                row = {
                    "group": "A",
                    "context": ctx_name,
                    "context_before": ctx_cfg["context_before"],
                    "context_after": ctx_cfg["context_after"],
                    "layer": layer,
                    "channels": "+".join([c.split("_")[0][0].upper() for c in default_channels]),
                    "classifier": "RF",
                    "output_token": "combined",
                    "accuracy": metrics.accuracy,
                    "f1": metrics.f1,
                    "auc": metrics.roc_auc if not np.isnan(metrics.roc_auc) else None,
                    "eer": metrics.eer if not np.isnan(metrics.eer) else None,
                    "ci_lower": metrics.ci_lower,
                    "ci_upper": metrics.ci_upper,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "time_s": round(elapsed, 1),
                }
                results.append(row)
                logger.info(
                    "    Acc=%.4f  F1=%.4f  AUC=%.4f  (%.1fs)",
                    metrics.accuracy, metrics.f1,
                    metrics.roc_auc if not np.isnan(metrics.roc_auc) else 0.0,
                    elapsed,
                )
            except Exception as e:
                logger.error("    FAILED: %s", e)
                results.append({
                    "group": "A", "context": ctx_name, "layer": layer,
                    "error": str(e),
                })

    # Save results
    df = pd.DataFrame(results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "group_a_context_layer.csv", index=False)
    logger.info("Saved: %s (%d rows)", tables_dir / "group_a_context_layer.csv", len(df))

    return results


# ============================================================================
# Group B: Channel Ablation × Layer
# ============================================================================

def run_group_b(cfg: dict, device: str, output_dir: Path) -> list[dict]:
    """Sweep channel combinations × layers with RF classifier.

    21 channels × 6 layers = 126 experiments.
    """
    logger.info("=" * 70)
    logger.info("GROUP B: Channel Ablation × Layer Sweep (126 experiments)")
    logger.info("=" * 70)

    results = []
    split_date = cfg.get("split_date", "2026-02-15")
    pretrained = cfg["model"]["pretrained_name"]
    seed = cfg.get("seed", 42)

    # Use default 9min bidirectional context
    ctx_before = cfg.get("default_context_before", 4)
    ctx_after = cfg.get("default_context_after", 4)

    for ch_cfg in CHANNEL_CONFIGS:
        ch_name = ch_cfg["name"]
        channels = ch_cfg["channels"]
        logger.info("--- Channels: %s ---", ch_name)

        ds_config = DatasetConfig(
            context_mode="bidirectional",
            context_before=ctx_before,
            context_after=ctx_after,
            stride=cfg.get("stride", 1),
        )

        prep_cfg = PreprocessConfig(
            sensor_csv=cfg["data"]["sensor_csv"],
            label_csv=cfg["data"]["label_csv"],
            label_format="events",
            initial_occupancy=cfg["data"].get("initial_occupancy", 0),
            binarize=True,
            channels=channels,
        )

        try:
            sensor_arr, train_labels, test_labels, ch_names, timestamps = (
                load_occupancy_data(prep_cfg, split_date=split_date)
            )
        except Exception as e:
            logger.warning("Skipping %s: %s", ch_name, e)
            continue

        all_labels = np.where(train_labels >= 0, train_labels, test_labels)
        dataset = OccupancyDataset(sensor_arr, all_labels, timestamps, ds_config)
        train_mask, test_mask = dataset.get_train_test_split(split_date)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            logger.warning("Skipping %s: empty train or test split", ch_name)
            continue

        for layer in ALL_LAYERS:
            exp_name = f"{ch_name} | L{layer}"
            logger.info("  %s", exp_name)
            t0 = time.time()

            try:
                trainer = load_mantis_model(pretrained, layer, "combined", device)
                Z = extract_embeddings(trainer, dataset, device)
                del trainer

                Z_train = Z[train_mask]
                y_train = dataset.labels[train_mask]
                Z_test = Z[test_mask]
                y_test = dataset.labels[test_mask]

                metrics = run_train_test_eval(Z_train, y_train, Z_test, y_test, "random_forest", seed)
                elapsed = time.time() - t0

                row = {
                    "group": "B",
                    "channels": ch_name,
                    "n_channels": len(channels),
                    "layer": layer,
                    "context": f"{ctx_before}+1+{ctx_after}",
                    "classifier": "RF",
                    "accuracy": metrics.accuracy,
                    "f1": metrics.f1,
                    "auc": metrics.roc_auc if not np.isnan(metrics.roc_auc) else None,
                    "eer": metrics.eer if not np.isnan(metrics.eer) else None,
                    "ci_lower": metrics.ci_lower,
                    "ci_upper": metrics.ci_upper,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "time_s": round(elapsed, 1),
                }
                results.append(row)
                logger.info(
                    "    Acc=%.4f  F1=%.4f  AUC=%.4f  (%.1fs)",
                    metrics.accuracy, metrics.f1,
                    metrics.roc_auc if not np.isnan(metrics.roc_auc) else 0.0,
                    elapsed,
                )
            except Exception as e:
                logger.error("    FAILED: %s", e)
                results.append({
                    "group": "B", "channels": ch_name, "layer": layer,
                    "error": str(e),
                })

    df = pd.DataFrame(results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "group_b_channel_ablation.csv", index=False)
    logger.info("Saved: %s (%d rows)", tables_dir / "group_b_channel_ablation.csv", len(df))

    return results


# ============================================================================
# Group C: Classifier × Output Token × Target Seq Len
# ============================================================================

def run_group_c(cfg: dict, device: str, output_dir: Path) -> list[dict]:
    """Sweep classifiers × output tokens × interpolation lengths.

    3 classifiers × 3 tokens × 6 target_seq_len = 54 experiments.
    """
    logger.info("=" * 70)
    logger.info("GROUP C: Classifier × Token × SeqLen Sweep (54 experiments)")
    logger.info("=" * 70)

    results = []
    default_channels = cfg.get("default_channels", ["d620900d_motionSensor", "408981c2_contactSensor"])
    split_date = cfg.get("split_date", "2026-02-15")
    pretrained = cfg["model"]["pretrained_name"]
    seed = cfg.get("seed", 42)
    default_layer = cfg.get("default_layer", 3)
    ctx_before = cfg.get("default_context_before", 4)
    ctx_after = cfg.get("default_context_after", 4)

    prep_cfg = PreprocessConfig(
        sensor_csv=cfg["data"]["sensor_csv"],
        label_csv=cfg["data"]["label_csv"],
        label_format="events",
        initial_occupancy=cfg["data"].get("initial_occupancy", 0),
        binarize=True,
        channels=default_channels,
    )

    for output_token in OUTPUT_TOKENS:
        for tgt_len in TARGET_SEQ_LENS:
            logger.info("--- Token=%s, target_seq_len=%d ---", output_token, tgt_len)

            ds_config = DatasetConfig(
                context_mode="bidirectional",
                context_before=ctx_before,
                context_after=ctx_after,
                stride=cfg.get("stride", 1),
                target_seq_len=tgt_len,
            )

            try:
                sensor_arr, train_labels, test_labels, ch_names, timestamps = (
                    load_occupancy_data(prep_cfg, split_date=split_date)
                )
            except Exception as e:
                logger.warning("Skipping token=%s tgt=%d: %s", output_token, tgt_len, e)
                continue

            all_labels = np.where(train_labels >= 0, train_labels, test_labels)
            dataset = OccupancyDataset(sensor_arr, all_labels, timestamps, ds_config)
            train_mask, test_mask = dataset.get_train_test_split(split_date)

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            # Extract embeddings once for this token + layer
            t0_emb = time.time()
            try:
                trainer = load_mantis_model(pretrained, default_layer, output_token, device)
                Z = extract_embeddings(trainer, dataset, device)
                del trainer
            except Exception as e:
                logger.error("Embedding extraction failed: %s", e)
                continue
            emb_time = time.time() - t0_emb

            Z_train = Z[train_mask]
            y_train = dataset.labels[train_mask]
            Z_test = Z[test_mask]
            y_test = dataset.labels[test_mask]

            for clf_name in CLASSIFIER_NAMES:
                exp_name = f"Token={output_token} | tgt={tgt_len} | {clf_name}"
                t0 = time.time()

                try:
                    metrics = run_train_test_eval(Z_train, y_train, Z_test, y_test, clf_name, seed)
                    elapsed = time.time() - t0

                    row = {
                        "group": "C",
                        "classifier": clf_name,
                        "output_token": output_token,
                        "target_seq_len": tgt_len,
                        "layer": default_layer,
                        "channels": "+".join([c.split("_")[0][0].upper() for c in default_channels]),
                        "context": f"{ctx_before}+1+{ctx_after}",
                        "accuracy": metrics.accuracy,
                        "f1": metrics.f1,
                        "auc": metrics.roc_auc if not np.isnan(metrics.roc_auc) else None,
                        "eer": metrics.eer if not np.isnan(metrics.eer) else None,
                        "ci_lower": metrics.ci_lower,
                        "ci_upper": metrics.ci_upper,
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        "embed_dim": Z.shape[1],
                        "time_s": round(elapsed, 1),
                        "embed_time_s": round(emb_time, 1),
                    }
                    results.append(row)
                    logger.info(
                        "  %s: Acc=%.4f  F1=%.4f  AUC=%.4f  (%.1fs)",
                        exp_name, metrics.accuracy, metrics.f1,
                        metrics.roc_auc if not np.isnan(metrics.roc_auc) else 0.0,
                        elapsed,
                    )
                except Exception as e:
                    logger.error("  %s FAILED: %s", exp_name, e)
                    results.append({
                        "group": "C", "classifier": clf_name,
                        "output_token": output_token, "target_seq_len": tgt_len,
                        "error": str(e),
                    })

    df = pd.DataFrame(results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "group_c_classifier_token.csv", index=False)
    logger.info("Saved: %s (%d rows)", tables_dir / "group_c_classifier_token.csv", len(df))

    return results


# ============================================================================
# Summary + Ranking
# ============================================================================

def generate_summary(all_results: list[dict], output_dir: Path):
    """Generate combined ranking and summary report."""
    df = pd.DataFrame(all_results)
    valid = df[~df.get("error", pd.Series(dtype=str)).notna() | df.get("error", pd.Series(dtype=str)).isna()]
    if "accuracy" not in valid.columns or len(valid) == 0:
        logger.warning("No valid results to summarize")
        return

    valid = valid.dropna(subset=["accuracy"])
    ranked = valid.sort_values("accuracy", ascending=False).head(20)

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(tables_dir / "top20_ranking.csv", index=False)

    logger.info("")
    logger.info("=" * 70)
    logger.info("TOP 20 CONFIGURATIONS:")
    logger.info("=" * 70)
    for i, (_, row) in enumerate(ranked.iterrows()):
        group = row.get("group", "?")
        ctx = row.get("context", "?")
        layer = row.get("layer", "?")
        ch = row.get("channels", "?")
        clf = row.get("classifier", "?")
        token = row.get("output_token", "?")
        acc = row.get("accuracy", 0)
        f1_val = row.get("f1", 0)
        auc_val = row.get("auc", 0) or 0
        logger.info(
            "  #%2d [%s] %s | L%s | %s | %s | %s: Acc=%.4f F1=%.4f AUC=%.4f",
            i + 1, group, ctx, layer, ch, clf, token, acc, f1_val, auc_val,
        )

    # Save summary JSON
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_experiments": len(all_results),
        "successful": len(valid),
        "failed": len(all_results) - len(valid),
        "top5": ranked.head(5).to_dict("records"),
    }
    with open(reports_dir / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved: %s", reports_dir / "phase1_summary.json")


# ============================================================================
# Main
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Comprehensive Sweep")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--group", choices=["A", "B", "C"], default=None,
                        help="Run specific group only (A, B, or C)")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Override stride (default from config)")
    parser.add_argument("--split-date", default=None,
                        help="Override train/test split date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    if args.stride is not None:
        cfg["stride"] = args.stride
    if args.split_date is not None:
        cfg["split_date"] = args.split_date

    output_dir = Path(cfg.get("output_dir", "results/phase1_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    logger.info("Device: %s", device)
    logger.info("Config: %s", args.config)
    logger.info("Output: %s", output_dir)

    t_start = time.time()
    all_results = []

    groups = [args.group] if args.group else ["A", "B", "C"]

    for group in groups:
        if group == "A":
            results = run_group_a(cfg, device, output_dir)
        elif group == "B":
            results = run_group_b(cfg, device, output_dir)
        elif group == "C":
            results = run_group_c(cfg, device, output_dir)
        else:
            continue
        all_results.extend(results)

    generate_summary(all_results, output_dir)

    total_time = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("Done! Total time: %.1fs (%.1f min)", total_time, total_time / 60)
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
