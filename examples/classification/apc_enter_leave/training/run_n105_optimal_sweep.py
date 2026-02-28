"""N=105 Optimal Configuration Sweep for APC Enter/Leave Detection.

After removing 4 timestamp collision samples (N=109→105, theoretical ceiling
96.33%→100%), this script systematically re-searches the optimal configuration
across all axes.  Three independent groups for parallel GPU execution.

  Group A — Context Window × Layer Grid (~15 min)
      10 context configs × 6 layers = 60 RF LOOCV experiments
  Group B — Multi-Layer Concat + Ensemble Exploration (~15 min)
      15 multi-layer combos + 15 ensemble combos = 30 experiments
  Group C — Neural Heads + Final Ranking (~10 min)
      6 neural configs + top-10 Wilson CI + summary report

Usage:
    cd examples/classification/apc_enter_leave

    # Terminal 1:
    python training/run_n105_optimal_sweep.py --config training/configs/enter-leave-phase2.yaml --group A --device cuda

    # Terminal 2:
    python training/run_n105_optimal_sweep.py --config training/configs/enter-leave-phase2.yaml --group B --device cuda

    # Terminal 3:
    python training/run_n105_optimal_sweep.py --config training/configs/enter-leave-phase2.yaml --group C --device cuda

    # All groups sequentially:
    python training/run_n105_optimal_sweep.py --config training/configs/enter-leave-phase2.yaml --group all --device cuda
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

import sys
import os

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from training.run_event_detection import (
    load_config,
    load_data,
    load_mantis_model,
    extract_all_embeddings,
)
from data.dataset import EventDataset, build_dataset_config
from evaluation.metrics import (
    EventClassificationMetrics,
    aggregate_cv_predictions,
)
from training.run_phase2_finetune import (
    TrainConfig,
    run_neural_loocv,
    compute_wilson_ci,
)
from training.heads import build_head

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

CONTEXT_CONFIGS = [
    {"name": "1+1+1", "before": 1, "after": 1, "mode": "bidirectional"},
    {"name": "2+1+2", "before": 2, "after": 2, "mode": "bidirectional"},
    {"name": "2+1+4", "before": 2, "after": 4, "mode": "bidirectional"},
    {"name": "3+1+3", "before": 3, "after": 3, "mode": "bidirectional"},
    {"name": "4+1+4", "before": 4, "after": 4, "mode": "bidirectional"},
    {"name": "5+1+5", "before": 5, "after": 5, "mode": "bidirectional"},
    {"name": "6+1+6", "before": 6, "after": 6, "mode": "bidirectional"},
    {"name": "8+1+8", "before": 8, "after": 8, "mode": "bidirectional"},
    {"name": "10+1+10", "before": 10, "after": 10, "mode": "bidirectional"},
    {"name": "15+1+0 (bw)", "before": 15, "after": 0, "mode": "bidirectional"},
]

LAYERS = [0, 1, 2, 3, 4, 5]

MULTI_LAYER_COMBOS = [
    {"name": "L0+L3", "layers": [0, 3]},
    {"name": "L0+L5", "layers": [0, 5]},
    {"name": "L2+L3", "layers": [2, 3]},
    {"name": "L3+L5", "layers": [3, 5]},
    {"name": "L0+L2+L3", "layers": [0, 2, 3]},
    {"name": "L0+L3+L5", "layers": [0, 3, 5]},
    {"name": "L2+L3+L4", "layers": [2, 3, 4]},
    {"name": "L1+L3+L5", "layers": [1, 3, 5]},
    {"name": "L0+L1+L3+L5", "layers": [0, 1, 3, 5]},
    {"name": "L0+L2+L3+L5", "layers": [0, 2, 3, 5]},
]

# Top ensemble context pairs to explore (2-way and 3-way)
ENSEMBLE_2WAY = [
    ("2+1+2", "2+1+4"),
    ("2+1+2", "3+1+3"),
    ("2+1+2", "4+1+4"),
    ("2+1+2", "5+1+5"),
    ("2+1+4", "3+1+3"),
    ("2+1+4", "4+1+4"),
    ("3+1+3", "4+1+4"),
    ("3+1+3", "5+1+5"),
    ("4+1+4", "5+1+5"),
    ("2+1+2", "6+1+6"),
]

ENSEMBLE_3WAY = [
    ("2+1+2", "2+1+4", "3+1+3"),
    ("2+1+2", "2+1+4", "4+1+4"),
    ("2+1+2", "3+1+3", "4+1+4"),
    ("2+1+2", "3+1+3", "5+1+5"),
    ("2+1+4", "3+1+3", "4+1+4"),
]


# ============================================================================
# Embedding Extraction (shared logic)
# ============================================================================

def _extract_embeddings(raw_cfg, include_none, device, layer, ctx_before,
                        ctx_after, ctx_mode="bidirectional"):
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


def _rf_loocv(Z, y, n_classes, seed=42):
    """Run RF LOOCV returning predictions, probabilities, and accuracy."""
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

    acc = float((y_pred == y).mean())
    return y_pred, y_prob, acc


def _save_csv(results, output_path):
    """Save results list as CSV."""
    if not results:
        return
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    logger.info("Saved: %s (%d rows)", output_path, len(results))


# ============================================================================
# Group A: Context Window × Layer Grid
# ============================================================================

def run_group_a(raw_cfg, include_none, device, seed, output_dir):
    """Systematic grid: 10 context windows × 6 layers with RF LOOCV."""
    logger.info("=" * 70)
    logger.info("GROUP A: Context Window × Layer Grid (N=105)")
    logger.info("=" * 70)

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    results = []
    n_total = len(CONTEXT_CONFIGS) * len(LAYERS)
    count = 0

    # Cache: (ctx_name, layer) → (Z, y, class_names)
    embedding_cache = {}

    for ctx_cfg in CONTEXT_CONFIGS:
        for layer in LAYERS:
            count += 1
            ctx_name = ctx_cfg["name"]
            key = (ctx_name, layer)

            logger.info("[%d/%d] Context=%s, Layer=L%d",
                        count, n_total, ctx_name, layer)

            t0 = time.time()
            Z, y, class_names = _extract_embeddings(
                raw_cfg, include_none, device, layer,
                ctx_cfg["before"], ctx_cfg["after"], ctx_cfg["mode"])
            embedding_cache[key] = (Z, y, class_names)

            n_classes = len(np.unique(y))
            y_pred, y_prob, acc = _rf_loocv(Z, y, n_classes, seed)

            n_correct = int(round(acc * len(y)))
            ci_lo, ci_hi = compute_wilson_ci(n_correct, len(y))

            # Per-class accuracy
            per_class = {}
            for c, name in enumerate(class_names):
                mask = y == c
                n_c = mask.sum()
                if n_c > 0:
                    per_class[name] = float((y_pred[mask] == c).sum() / n_c)

            elapsed = time.time() - t0
            row = {
                "context": ctx_name,
                "layer": f"L{layer}",
                "accuracy": round(acc, 4),
                "ci_95_lower": round(ci_lo, 4),
                "ci_95_upper": round(ci_hi, 4),
                "n_correct": n_correct,
                "n_total": len(y),
                "n_errors": len(y) - n_correct,
                "time_sec": round(elapsed, 1),
            }
            for cname, cacc in per_class.items():
                row[f"acc_{cname}"] = round(cacc, 4)

            results.append(row)
            logger.info("  Acc=%.4f  CI=[%.4f, %.4f]  (%d errors, %.1fs)",
                        acc, ci_lo, ci_hi, len(y) - n_correct, elapsed)

    # Sort by accuracy descending
    results.sort(key=lambda r: -r["accuracy"])

    # Save results
    _save_csv(results, tables_dir / "group_a_context_layer_grid.csv")

    # Save JSON with metadata
    report = {
        "group": "A",
        "description": "Context Window × Layer Grid (RF LOOCV, N=105)",
        "n_experiments": len(results),
        "top_10": results[:10],
        "best": results[0] if results else None,
    }
    with open(tables_dir / "group_a_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("GROUP A SUMMARY — Top 10 Configurations:")
    logger.info("=" * 70)
    for i, r in enumerate(results[:10]):
        logger.info("  #%d: %s + %s = %.2f%% (CI [%.2f%%, %.2f%%], %d errors)",
                    i + 1, r["context"], r["layer"],
                    r["accuracy"] * 100, r["ci_95_lower"] * 100,
                    r["ci_95_upper"] * 100, r["n_errors"])

    # Save embedding cache info for Group B
    cache_path = output_dir / ".embedding_cache_keys.json"
    with open(cache_path, "w") as f:
        json.dump({"cached_keys": [f"{k[0]}__L{k[1]}" for k in embedding_cache]}, f)

    return results


# ============================================================================
# Group B: Multi-Layer Concat + Ensemble Exploration
# ============================================================================

def run_group_b(raw_cfg, include_none, device, seed, output_dir):
    """Multi-layer concat + soft-voting ensemble exploration."""
    logger.info("=" * 70)
    logger.info("GROUP B: Multi-Layer Concat + Ensemble Exploration (N=105)")
    logger.info("=" * 70)

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ---- Part 1: Multi-layer concat ----
    logger.info("--- Part 1: Multi-Layer Concatenation ---")

    # We use 2+1+2 context as the reference (the previous Phase 1 best context)
    base_ctx = {"before": 2, "after": 2, "mode": "bidirectional"}
    ml_results = []

    # First extract single-layer embeddings needed for combos
    layer_embeddings = {}
    unique_layers = set()
    for combo in MULTI_LAYER_COMBOS:
        unique_layers.update(combo["layers"])

    for layer in sorted(unique_layers):
        logger.info("Extracting L%d (2+1+2)...", layer)
        Z, y, class_names = _extract_embeddings(
            raw_cfg, include_none, device, layer,
            base_ctx["before"], base_ctx["after"], base_ctx["mode"])
        layer_embeddings[layer] = Z

    n_classes = len(np.unique(y))

    # Run single-layer baselines for comparison
    for layer in sorted(unique_layers):
        Z_single = layer_embeddings[layer]
        _, _, acc = _rf_loocv(Z_single, y, n_classes, seed)
        n_correct = int(round(acc * len(y)))
        ci_lo, ci_hi = compute_wilson_ci(n_correct, len(y))
        ml_results.append({
            "name": f"L{layer} (single)",
            "layers": f"L{layer}",
            "context": "2+1+2",
            "method": "single",
            "dim": Z_single.shape[1],
            "accuracy": round(acc, 4),
            "ci_95_lower": round(ci_lo, 4),
            "ci_95_upper": round(ci_hi, 4),
            "n_errors": len(y) - n_correct,
        })
        logger.info("  L%d single: Acc=%.4f (dim=%d)", layer, acc, Z_single.shape[1])

    # Multi-layer concat
    for combo in MULTI_LAYER_COMBOS:
        t0 = time.time()
        Z_cat = np.concatenate([layer_embeddings[l] for l in combo["layers"]], axis=1)
        _, _, acc = _rf_loocv(Z_cat, y, n_classes, seed)
        n_correct = int(round(acc * len(y)))
        ci_lo, ci_hi = compute_wilson_ci(n_correct, len(y))
        elapsed = time.time() - t0

        ml_results.append({
            "name": combo["name"],
            "layers": "+".join(f"L{l}" for l in combo["layers"]),
            "context": "2+1+2",
            "method": "concat",
            "dim": Z_cat.shape[1],
            "accuracy": round(acc, 4),
            "ci_95_lower": round(ci_lo, 4),
            "ci_95_upper": round(ci_hi, 4),
            "n_errors": len(y) - n_correct,
            "time_sec": round(elapsed, 1),
        })
        logger.info("  %s concat: Acc=%.4f (dim=%d, %.1fs)",
                    combo["name"], acc, Z_cat.shape[1], elapsed)

    ml_results.sort(key=lambda r: -r["accuracy"])
    _save_csv(ml_results, tables_dir / "group_b_multilayer.csv")

    # ---- Part 2: Ensemble (soft-voting) ----
    logger.info("--- Part 2: Soft-Voting Ensemble ---")

    # Extract embeddings for each context used in ensembles
    ctx_lookup = {c["name"]: c for c in CONTEXT_CONFIGS}
    needed_contexts = set()
    for pair in ENSEMBLE_2WAY:
        needed_contexts.update(pair)
    for triple in ENSEMBLE_3WAY:
        needed_contexts.update(triple)

    # We use L3 as the standard layer for ensemble (Phase 1 best)
    ensemble_layer = 3
    ctx_embeddings = {}
    ctx_probs = {}

    for ctx_name in sorted(needed_contexts):
        ctx_cfg = ctx_lookup.get(ctx_name)
        if ctx_cfg is None:
            logger.warning("Context %s not found, skipping", ctx_name)
            continue

        logger.info("Extracting L%d + %s for ensemble...", ensemble_layer, ctx_name)
        Z, y_ctx, _ = _extract_embeddings(
            raw_cfg, include_none, device, ensemble_layer,
            ctx_cfg["before"], ctx_cfg["after"], ctx_cfg["mode"])
        ctx_embeddings[ctx_name] = Z

        _, probs, acc = _rf_loocv(Z, y_ctx, n_classes, seed)
        ctx_probs[ctx_name] = probs
        logger.info("  %s single: Acc=%.4f", ctx_name, acc)

    ensemble_results = []

    # Single-context baselines (for comparison)
    for ctx_name, probs in ctx_probs.items():
        pred = probs.argmax(axis=1)
        acc = float((pred == y).mean())
        n_correct = int(round(acc * len(y)))
        ci_lo, ci_hi = compute_wilson_ci(n_correct, len(y))
        ensemble_results.append({
            "name": f"RF {ctx_name}",
            "contexts": ctx_name,
            "n_ways": 1,
            "accuracy": round(acc, 4),
            "ci_95_lower": round(ci_lo, 4),
            "ci_95_upper": round(ci_hi, 4),
            "n_errors": len(y) - n_correct,
        })

    # 2-way ensembles
    for c1, c2 in ENSEMBLE_2WAY:
        if c1 not in ctx_probs or c2 not in ctx_probs:
            continue
        avg_prob = (ctx_probs[c1] + ctx_probs[c2]) / 2
        pred = avg_prob.argmax(axis=1)
        acc = float((pred == y).mean())
        n_correct = int(round(acc * len(y)))
        ci_lo, ci_hi = compute_wilson_ci(n_correct, len(y))
        ensemble_results.append({
            "name": f"Ens2 {c1}+{c2}",
            "contexts": f"{c1}+{c2}",
            "n_ways": 2,
            "accuracy": round(acc, 4),
            "ci_95_lower": round(ci_lo, 4),
            "ci_95_upper": round(ci_hi, 4),
            "n_errors": len(y) - n_correct,
        })
        logger.info("  Ens2 %s+%s: Acc=%.4f", c1, c2, acc)

    # 3-way ensembles
    for c1, c2, c3 in ENSEMBLE_3WAY:
        if c1 not in ctx_probs or c2 not in ctx_probs or c3 not in ctx_probs:
            continue
        avg_prob = (ctx_probs[c1] + ctx_probs[c2] + ctx_probs[c3]) / 3
        pred = avg_prob.argmax(axis=1)
        acc = float((pred == y).mean())
        n_correct = int(round(acc * len(y)))
        ci_lo, ci_hi = compute_wilson_ci(n_correct, len(y))
        ensemble_results.append({
            "name": f"Ens3 {c1}+{c2}+{c3}",
            "contexts": f"{c1}+{c2}+{c3}",
            "n_ways": 3,
            "accuracy": round(acc, 4),
            "ci_95_lower": round(ci_lo, 4),
            "ci_95_upper": round(ci_hi, 4),
            "n_errors": len(y) - n_correct,
        })
        logger.info("  Ens3 %s+%s+%s: Acc=%.4f", c1, c2, c3, acc)

    ensemble_results.sort(key=lambda r: -r["accuracy"])
    _save_csv(ensemble_results, tables_dir / "group_b_ensemble.csv")

    # Combined report
    all_b = ml_results + ensemble_results
    all_b.sort(key=lambda r: -r["accuracy"])

    report = {
        "group": "B",
        "description": "Multi-Layer Concat + Ensemble Exploration (RF LOOCV, N=105)",
        "multilayer_experiments": len(ml_results),
        "ensemble_experiments": len(ensemble_results),
        "top_10_overall": all_b[:10],
        "best_multilayer": ml_results[0] if ml_results else None,
        "best_ensemble": ensemble_results[0] if ensemble_results else None,
    }
    with open(tables_dir / "group_b_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 70)
    logger.info("GROUP B SUMMARY — Top 10 (Multi-Layer + Ensemble):")
    logger.info("=" * 70)
    for i, r in enumerate(all_b[:10]):
        logger.info("  #%d: %s = %.2f%% (CI [%.2f%%, %.2f%%], %d errors)",
                    i + 1, r["name"], r["accuracy"] * 100,
                    r["ci_95_lower"] * 100, r["ci_95_upper"] * 100,
                    r["n_errors"])

    return all_b


# ============================================================================
# Group C: Neural Heads + Final Ranking
# ============================================================================

def run_group_c(raw_cfg, include_none, device, seed, output_dir):
    """Neural head experiments + final comprehensive ranking."""
    logger.info("=" * 70)
    logger.info("GROUP C: Neural Heads + Final Ranking (N=105)")
    logger.info("=" * 70)

    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_cfg_raw = raw_cfg.get("training", {})
    train_config = TrainConfig(
        epochs=train_cfg_raw.get("epochs", 200),
        lr=train_cfg_raw.get("lr", 1e-3),
        weight_decay=train_cfg_raw.get("weight_decay", 0.01),
        label_smoothing=train_cfg_raw.get("label_smoothing", 0.0),
        early_stopping_patience=train_cfg_raw.get("early_stopping_patience", 30),
        device=device,
    )

    # Extract key embeddings for neural experiments
    logger.info("Extracting embeddings for neural head experiments...")
    Z_2_2, y, class_names = _extract_embeddings(
        raw_cfg, include_none, device, 3, 2, 2)
    Z_2_4, _, _ = _extract_embeddings(
        raw_cfg, include_none, device, 3, 2, 4)
    Z_L0, _, _ = _extract_embeddings(
        raw_cfg, include_none, device, 0, 2, 2)

    n_classes = len(np.unique(y))
    n_events = len(y)

    neural_results = []

    # ---- Neural Head Configs ----
    head_configs = [
        {"name": "Linear (L3, 2+1+2)", "Z": Z_2_2, "head_type": "linear",
         "head_kwargs": {}},
        {"name": "MLP[64]-d0.5 (L3, 2+1+2)", "Z": Z_2_2, "head_type": "mlp",
         "head_kwargs": {"hidden_dims": [64], "dropout": 0.5}},
        {"name": "MLP[128]-d0.5 (L3, 2+1+2)", "Z": Z_2_2, "head_type": "mlp",
         "head_kwargs": {"hidden_dims": [128], "dropout": 0.5}},
        {"name": "Linear Concat L0+L3", "Z": np.concatenate([Z_L0, Z_2_2], axis=1),
         "head_type": "linear", "head_kwargs": {}},
        {"name": "MLP[64]-d0.5 Concat L0+L3",
         "Z": np.concatenate([Z_L0, Z_2_2], axis=1),
         "head_type": "mlp", "head_kwargs": {"hidden_dims": [64], "dropout": 0.5}},
        {"name": "Linear (L3, 2+1+4)", "Z": Z_2_4, "head_type": "linear",
         "head_kwargs": {}},
    ]

    for cfg in head_configs:
        logger.info("--- Neural: %s ---", cfg["name"])
        t0 = time.time()
        Z_input = cfg["Z"]

        def head_factory(ht=cfg["head_type"], hk=cfg["head_kwargs"],
                         dim=Z_input.shape[1], nc=n_classes):
            return build_head(ht, dim, nc, **hk)

        metrics, y_pred, y_prob = run_neural_loocv(
            Z_input, y, head_factory, train_config, class_names, seed)
        n_correct = int(round(metrics.accuracy * n_events))
        ci_lo, ci_hi = compute_wilson_ci(n_correct, n_events)
        elapsed = time.time() - t0

        neural_results.append({
            "name": cfg["name"],
            "head_type": cfg["head_type"],
            "accuracy": round(metrics.accuracy, 4),
            "f1_macro": round(metrics.f1_macro, 4),
            "roc_auc": round(metrics.roc_auc, 4) if metrics.roc_auc else None,
            "ci_95_lower": round(ci_lo, 4),
            "ci_95_upper": round(ci_hi, 4),
            "n_errors": n_events - n_correct,
            "input_dim": Z_input.shape[1],
            "time_sec": round(elapsed, 1),
        })
        logger.info("  %s: Acc=%.4f  F1=%.4f  AUC=%s  CI=[%.4f,%.4f] (%.1fs)",
                    cfg["name"], metrics.accuracy, metrics.f1_macro,
                    f"{metrics.roc_auc:.4f}" if metrics.roc_auc else "N/A",
                    ci_lo, ci_hi, elapsed)

    neural_results.sort(key=lambda r: -r["accuracy"])
    _save_csv(neural_results, tables_dir / "group_c_neural.csv")

    # ---- RF baselines for comparison ----
    logger.info("--- RF baselines for comparison ---")
    rf_results = []

    rf_configs = [
        ("RF L3 2+1+2", Z_2_2),
        ("RF L3 2+1+4", Z_2_4),
        ("RF Concat L0+L3", np.concatenate([Z_L0, Z_2_2], axis=1)),
    ]

    for name, Z_input in rf_configs:
        y_pred, y_prob, acc = _rf_loocv(Z_input, y, n_classes, seed)
        n_correct = int(round(acc * n_events))
        ci_lo, ci_hi = compute_wilson_ci(n_correct, n_events)

        metrics = aggregate_cv_predictions(y, y_pred, y_prob,
                                           cv_method="LOOCV", n_folds=n_events,
                                           class_names=class_names)
        rf_results.append({
            "name": name,
            "method": "RF",
            "accuracy": round(acc, 4),
            "f1_macro": round(metrics.f1_macro, 4),
            "roc_auc": round(metrics.roc_auc, 4) if metrics.roc_auc else None,
            "ci_95_lower": round(ci_lo, 4),
            "ci_95_upper": round(ci_hi, 4),
            "n_errors": n_events - n_correct,
            "input_dim": Z_input.shape[1],
        })
        logger.info("  %s: Acc=%.4f  F1=%.4f", name, acc, metrics.f1_macro)

    # ---- 2-way ensemble (best from N=109) ----
    logger.info("--- 2-way Ensemble (2+1+2 + 2+1+4) ---")
    _, p_22, _ = _rf_loocv(Z_2_2, y, n_classes, seed)
    _, p_24, _ = _rf_loocv(Z_2_4, y, n_classes, seed)
    ens_prob = (p_22 + p_24) / 2
    ens_pred = ens_prob.argmax(axis=1)
    ens_acc = float((ens_pred == y).mean())
    n_correct = int(round(ens_acc * n_events))
    ci_lo, ci_hi = compute_wilson_ci(n_correct, n_events)
    ens_metrics = aggregate_cv_predictions(y, ens_pred, ens_prob,
                                           cv_method="LOOCV", n_folds=n_events,
                                           class_names=class_names)

    # ---- Multi-seed stability ----
    logger.info("--- Multi-seed stability (5 seeds) ---")
    n_seeds = 5
    seed_list = list(range(seed, seed + n_seeds))
    stability = {}

    # Ensemble stability
    ens_accs = []
    for s in seed_list:
        _, p1 = _rf_loocv(Z_2_2, y, n_classes, s)[:2]
        _, p2 = _rf_loocv(Z_2_4, y, n_classes, s)[:2]
        pred = ((p1 + p2) / 2).argmax(axis=1)
        ens_accs.append(float((pred == y).mean()))
    stability["Ensemble 2-way"] = {
        "mean": round(float(np.mean(ens_accs)), 4),
        "std": round(float(np.std(ens_accs)), 4),
        "seeds": [round(a, 4) for a in ens_accs],
    }
    logger.info("  Ensemble: mean=%.4f ±%.4f", np.mean(ens_accs), np.std(ens_accs))

    # RF single stability
    rf_accs = []
    for s in seed_list:
        _, _, a = _rf_loocv(Z_2_2, y, n_classes, s)
        rf_accs.append(a)
    stability["RF single"] = {
        "mean": round(float(np.mean(rf_accs)), 4),
        "std": round(float(np.std(rf_accs)), 4),
        "seeds": [round(a, 4) for a in rf_accs],
    }
    logger.info("  RF single: mean=%.4f ±%.4f", np.mean(rf_accs), np.std(rf_accs))

    # ---- Final comprehensive ranking ----
    all_results = []
    for r in neural_results:
        all_results.append({**r, "method": "Neural"})
    for r in rf_results:
        all_results.append(r)
    all_results.append({
        "name": "Ensemble RF 2-way (2+1+2 + 2+1+4)",
        "method": "Ensemble",
        "accuracy": round(ens_acc, 4),
        "f1_macro": round(ens_metrics.f1_macro, 4),
        "roc_auc": round(ens_metrics.roc_auc, 4) if ens_metrics.roc_auc else None,
        "ci_95_lower": round(ci_lo, 4),
        "ci_95_upper": round(ci_hi, 4),
        "n_errors": n_events - n_correct,
    })

    all_results.sort(key=lambda r: -r["accuracy"])
    _save_csv(all_results, tables_dir / "group_c_final_ranking.csv")

    # ---- Final report ----
    report = {
        "group": "C",
        "description": "Neural Heads + Final Ranking (N=105, cleaned)",
        "dataset": {
            "n_events": n_events,
            "n_classes": n_classes,
            "class_names": list(class_names),
            "theoretical_ceiling": "100% (collisions removed)",
        },
        "neural_experiments": len(neural_results),
        "rf_baselines": len(rf_results),
        "final_ranking": all_results,
        "stability": stability,
        "best_overall": all_results[0] if all_results else None,
    }
    with open(reports_dir / "n105_sweep_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 70)
    logger.info("GROUP C FINAL RANKING — All Methods (N=105):")
    logger.info("=" * 70)
    for i, r in enumerate(all_results):
        logger.info("  #%d: [%s] %s = %.2f%% (CI [%.2f%%, %.2f%%], %d errors)",
                    i + 1, r.get("method", "?"), r["name"],
                    r["accuracy"] * 100, r["ci_95_lower"] * 100,
                    r["ci_95_upper"] * 100, r["n_errors"])

    logger.info("\nMulti-seed stability:")
    for name, st in stability.items():
        logger.info("  %s: %.4f ±%.4f  seeds=%s",
                    name, st["mean"], st["std"], st["seeds"])

    return report


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="N=105 Optimal Configuration Sweep",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--group", required=True,
                        choices=["A", "B", "C", "all"],
                        help="Group: A (context×layer), B (multilayer+ensemble), "
                             "C (neural+ranking), all")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--include-none", action="store_true",
                        help="Include NONE class (3-class)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    raw_cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else raw_cfg.get("seed", 42)
    device = (args.device if args.device is not None
              else raw_cfg.get("model", {}).get("device", "cuda"))
    include_none = (args.include_none
                    or raw_cfg.get("data", {}).get("include_none", False))

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        raw_cfg.get("output_dir", "results/n105_optimal_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    groups = [args.group] if args.group != "all" else ["A", "B", "C"]

    results = {}
    for group in groups:
        if group == "A":
            results["group_a"] = run_group_a(
                raw_cfg, include_none, device, seed, output_dir)
        elif group == "B":
            results["group_b"] = run_group_b(
                raw_cfg, include_none, device, seed, output_dir)
        elif group == "C":
            results["group_c"] = run_group_c(
                raw_cfg, include_none, device, seed, output_dir)

    t_total = time.time() - t_start
    logger.info("=" * 70)
    logger.info("Done! Total time: %.1fs (%.1f min)", t_total, t_total / 60)
    logger.info("Output directory: %s", output_dir)

    # Save combined results
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    combined_path = reports_dir / f"group_{'_'.join(groups)}_combined.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    logger.info("Saved: %s", combined_path)


if __name__ == "__main__":
    main()
