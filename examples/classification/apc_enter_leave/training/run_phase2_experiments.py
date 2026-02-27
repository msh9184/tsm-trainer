"""Phase 2 Tier 2 Experiments: PCA Reduction + Multi-Context Ensemble.

Three independent experiments designed for parallel GPU execution:

  Experiment A: Re-run Stages 2-5 with optimal 2+1+2 context
      → Use existing run_phase2_finetune.py with --context-before 2 --context-after 2

  Experiment B: PCA dimensionality reduction sweep
      → 1024-dim → {32, 64, 128, 256, 512} + RF/Neural LOOCV

  Experiment C: Multi-context soft-voting ensemble
      → Combine 2+1+2, 2+1+4, 3+1+3 RF predictions via soft voting

Usage:
    cd examples/classification/apc_enter_leave

    # Experiment B: PCA sweep (parallelizable process 2)
    python training/run_phase2_experiments.py \\
        --config training/configs/enter-leave-phase2.yaml \\
        --experiment pca --device cuda

    # Experiment C: Multi-context ensemble (parallelizable process 3)
    python training/run_phase2_experiments.py \\
        --config training/configs/enter-leave-phase2.yaml \\
        --experiment ensemble --device cuda

    # Both B + C
    python training/run_phase2_experiments.py \\
        --config training/configs/enter-leave-phase2.yaml \\
        --experiment all --device cuda

Parallel execution on GPU server:
    # Terminal 1 (Exp A):
    python training/run_phase2_finetune.py --config ... \\
        --stages 2 3 4 5 --context-before 2 --context-after 2 --device cuda

    # Terminal 2 (Exp B):
    python training/run_phase2_experiments.py --config ... --experiment pca --device cuda

    # Terminal 3 (Exp C):
    python training/run_phase2_experiments.py --config ... --experiment ensemble --device cuda
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

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from training.run_event_detection import (
    load_config,
    load_data,
    load_mantis_model,
    extract_all_embeddings,
    run_loocv,
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
    _log_top3,
    _serialize_stage_results,
)
from training.heads import build_head

logger = logging.getLogger(__name__)


# ============================================================================
# Experiment B: PCA Dimensionality Reduction
# ============================================================================

def run_experiment_pca(
    raw_cfg: dict,
    include_none: bool,
    device: str,
    seed: int = 42,
    context_before: int = 2,
    context_after: int = 2,
) -> dict:
    """PCA dimensionality reduction sweep.

    Reduces 1024-dim MantisV2 embeddings to {32, 64, 128, 256, 512} dimensions
    via PCA, then evaluates with RF LOOCV and Neural LOOCV.

    Rationale: N=109, D=1024 → N/D=0.106 (extremely underdetermined).
    PCA to D'=128 gives N/D'=0.85, much more favorable for both RF and neural.

    Returns dict with results for each PCA dimension.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    logger.info("=" * 60)
    logger.info("EXPERIMENT B: PCA Dimensionality Reduction Sweep")
    logger.info("=" * 60)

    output_dir = Path(raw_cfg.get("output_dir", "results/phase2_finetune"))
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load data with optimal context
    cfg_copy = copy.deepcopy(raw_cfg)
    cfg_copy["dataset"] = {
        "context_mode": "bidirectional",
        "context_before": context_before,
        "context_after": context_after,
    }

    sensor_array, sensor_ts, event_ts, event_labels, ch_names, class_names = (
        load_data(cfg_copy, include_none=include_none)
    )
    y = event_labels
    n_events = len(y)
    n_classes = len(np.unique(y))

    ds_cfg = build_dataset_config(cfg_copy["dataset"])
    dataset = EventDataset(sensor_array, sensor_ts, event_ts, event_labels, ds_cfg)

    # Extract L3 embeddings
    model_cfg = raw_cfg.get("model", {})
    pretrained_name = model_cfg.get("pretrained_name", "paris-noah/MantisV2")
    output_token = model_cfg.get("output_token", "combined")

    logger.info("Extracting L3 embeddings (context: %d+1+%d)...", context_before, context_after)
    network, model = load_mantis_model(pretrained_name, 3, output_token, device)
    Z = extract_all_embeddings(model, dataset)
    del model, network
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Embeddings: shape=%s", Z.shape)

    # Training config
    train_cfg_raw = raw_cfg.get("training", {})
    base_train_config = TrainConfig(
        epochs=train_cfg_raw.get("epochs", 200),
        lr=train_cfg_raw.get("lr", 1e-3),
        weight_decay=train_cfg_raw.get("weight_decay", 0.01),
        label_smoothing=train_cfg_raw.get("label_smoothing", 0.0),
        early_stopping_patience=train_cfg_raw.get("early_stopping_patience", 30),
        device=device,
    )

    # PCA sweep dimensions — cap at min(n_samples, n_features) - 1
    max_components = min(n_events, Z.shape[1]) - 1  # PCA requires n_components < min(N, D)
    pca_dims = [d for d in [32, 64, 128, 256, 512] if d <= max_components]
    logger.info("PCA dims (max_components=%d): %s", max_components, pca_dims)

    results_rf = []
    results_neural = []

    # First: no PCA baseline (RF)
    logger.info("[RF baseline] No PCA (D=%d)", Z.shape[1])
    t0 = time.time()
    metrics_rf_base = run_sklearn_loocv(Z, y, class_names, seed)
    elapsed = time.time() - t0
    row = _make_result_row(f"RF D={Z.shape[1]} (no PCA)", metrics_rf_base, n_events, elapsed)
    results_rf.append(row)
    logger.info("  Acc=%.4f  F1=%.4f  AUC=%.4f  (%.1fs)",
                metrics_rf_base.accuracy, metrics_rf_base.f1_macro, metrics_rf_base.roc_auc, elapsed)

    # First: no PCA baseline (Neural MLP[64])
    logger.info("[Neural baseline] No PCA (D=%d)", Z.shape[1])
    t0 = time.time()
    def head_factory_base():
        return build_head("mlp", Z.shape[1], n_classes, hidden_dims=[64], dropout=0.5)
    metrics_nn_base, _, _ = run_neural_loocv(Z, y, head_factory_base, base_train_config, class_names, seed)
    elapsed = time.time() - t0
    row = _make_result_row(f"Neural D={Z.shape[1]} (no PCA)", metrics_nn_base, n_events, elapsed)
    results_neural.append(row)
    logger.info("  Acc=%.4f  F1=%.4f  AUC=%.4f  (%.1fs)",
                metrics_nn_base.accuracy, metrics_nn_base.f1_macro, metrics_nn_base.roc_auc, elapsed)

    # PCA sweep
    for d in pca_dims:
        if d >= Z.shape[1]:
            continue

        logger.info("[PCA D=%d] Fitting PCA and evaluating...", d)

        # PCA within LOOCV to avoid leakage? For PCA we fit-transform globally
        # since PCA is unsupervised (no label leakage). This is standard practice.
        scaler = StandardScaler()
        Z_scaled = scaler.fit_transform(Z)
        pca = PCA(n_components=d, random_state=seed)
        Z_pca = pca.fit_transform(Z_scaled)
        explained_var = pca.explained_variance_ratio_.sum()

        logger.info("  PCA %d: explained variance = %.2f%%", d, explained_var * 100)

        # RF LOOCV
        t0 = time.time()
        metrics_rf = run_sklearn_loocv(Z_pca, y, class_names, seed)
        elapsed_rf = time.time() - t0
        row_rf = _make_result_row(
            f"RF PCA-{d} ({explained_var:.1%})", metrics_rf, n_events, elapsed_rf,
            extra={"pca_dim": d, "explained_variance": round(explained_var, 4)},
        )
        results_rf.append(row_rf)
        logger.info("  RF:     Acc=%.4f  F1=%.4f  AUC=%.4f  (%.1fs)",
                    metrics_rf.accuracy, metrics_rf.f1_macro, metrics_rf.roc_auc, elapsed_rf)

        # Neural LOOCV (MLP head: smaller because input is smaller)
        hidden_dim = min(64, d)
        t0 = time.time()
        def head_factory_pca(ed=d, hd=hidden_dim):
            return build_head("mlp", ed, n_classes, hidden_dims=[hd], dropout=0.5)
        metrics_nn, _, _ = run_neural_loocv(Z_pca, y, head_factory_pca, base_train_config, class_names, seed)
        elapsed_nn = time.time() - t0
        row_nn = _make_result_row(
            f"Neural PCA-{d} MLP[{hidden_dim}]", metrics_nn, n_events, elapsed_nn,
            extra={"pca_dim": d, "explained_variance": round(explained_var, 4)},
        )
        results_neural.append(row_nn)
        logger.info("  Neural: Acc=%.4f  F1=%.4f  AUC=%.4f  (%.1fs)",
                    metrics_nn.accuracy, metrics_nn.f1_macro, metrics_nn.roc_auc, elapsed_nn)

    _rank_results(results_rf)
    _rank_results(results_neural)

    logger.info("-" * 40)
    _log_top3(results_rf, "PCA RF")
    _log_top3(results_neural, "PCA Neural")

    # Save results
    all_results = results_rf + results_neural
    save_results_csv(_serialize_stage_results(all_results), tables_dir / "exp_b_pca_sweep.csv")
    save_results_txt(_serialize_stage_results(all_results), tables_dir / "exp_b_pca_sweep.txt")

    return {
        "rf_results": _serialize_stage_results(results_rf),
        "neural_results": _serialize_stage_results(results_neural),
    }


# ============================================================================
# Experiment C: Multi-Context Soft-Voting Ensemble
# ============================================================================

def run_experiment_ensemble(
    raw_cfg: dict,
    include_none: bool,
    device: str,
    seed: int = 42,
) -> dict:
    """Multi-context soft-voting ensemble.

    Extracts MantisV2 embeddings with 3 context windows:
      - 2+1+2 (5min, Stage 1 best)
      - 2+1+4 (7min, Stage 1 #2)
      - 3+1+3 (7min, Stage 1 #3)

    For each window, runs RF LOOCV and collects per-sample probabilities.
    Then combines via soft voting (average probabilities → argmax).

    Also tests PCA + ensemble and neural + ensemble variants.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    logger.info("=" * 60)
    logger.info("EXPERIMENT C: Multi-Context Soft-Voting Ensemble")
    logger.info("=" * 60)

    output_dir = Path(raw_cfg.get("output_dir", "results/phase2_finetune"))
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = raw_cfg.get("model", {})
    pretrained_name = model_cfg.get("pretrained_name", "paris-noah/MantisV2")
    output_token = model_cfg.get("output_token", "combined")

    # Context window configs for ensemble
    context_configs = [
        {"name": "2+1+2", "context_before": 2, "context_after": 2},
        {"name": "2+1+4", "context_before": 2, "context_after": 4},
        {"name": "3+1+3", "context_before": 3, "context_after": 3},
    ]

    # Extract embeddings for each context window
    embeddings = {}
    y = None
    class_names = None

    for ctx in context_configs:
        logger.info("Extracting embeddings for context %s...", ctx["name"])

        cfg_copy = copy.deepcopy(raw_cfg)
        cfg_copy["dataset"] = {
            "context_mode": "bidirectional",
            "context_before": ctx["context_before"],
            "context_after": ctx["context_after"],
        }

        sensor_array, sensor_ts, event_ts, event_labels, ch_names, class_names_local = (
            load_data(cfg_copy, include_none=include_none)
        )

        if y is None:
            y = event_labels
            class_names = class_names_local

        ds_cfg = build_dataset_config(cfg_copy["dataset"])
        dataset = EventDataset(sensor_array, sensor_ts, event_ts, event_labels, ds_cfg)

        network, model = load_mantis_model(pretrained_name, 3, output_token, device)
        Z = extract_all_embeddings(model, dataset)
        del model, network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        embeddings[ctx["name"]] = Z
        logger.info("  %s: shape=%s", ctx["name"], Z.shape)

    n_events = len(y)
    n_classes = len(np.unique(y))

    # --- Individual RF LOOCV with per-sample probabilities ---
    rf_probs = {}

    for ctx_name, Z_ctx in embeddings.items():
        logger.info("[RF] Context %s LOOCV...", ctx_name)

        y_prob_all = np.zeros((n_events, n_classes), dtype=np.float64)

        for i in range(n_events):
            train_mask = np.ones(n_events, dtype=bool)
            train_mask[i] = False

            clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
            clf.fit(Z_ctx[train_mask], y[train_mask])

            probs = clf.predict_proba(Z_ctx[i:i + 1])[0]
            # Handle case where not all classes appear in training fold
            prob_vec = np.zeros(n_classes, dtype=np.float64)
            for ci, c in enumerate(clf.classes_):
                prob_vec[c] = probs[ci]
            y_prob_all[i] = prob_vec

        y_pred = y_prob_all.argmax(axis=1)
        acc = (y_pred == y).mean()
        rf_probs[ctx_name] = y_prob_all
        logger.info("  %s: Acc=%.4f", ctx_name, acc)

    # --- Soft-voting ensemble ---
    results = []

    # Individual results (for reference)
    for ctx_name, probs in rf_probs.items():
        y_pred = probs.argmax(axis=1)
        acc = float((y_pred == y).mean())
        metrics = aggregate_cv_predictions(y, y_pred, probs, cv_method="LOOCV", n_folds=n_events, class_names=class_names)
        row = _make_result_row(f"RF {ctx_name}", metrics, n_events, 0)
        results.append(row)

    # 2-way ensembles
    for c1, c2 in [("2+1+2", "2+1+4"), ("2+1+2", "3+1+3"), ("2+1+4", "3+1+3")]:
        prob_avg = (rf_probs[c1] + rf_probs[c2]) / 2
        y_pred = prob_avg.argmax(axis=1)
        metrics = aggregate_cv_predictions(y, y_pred, prob_avg, cv_method="LOOCV", n_folds=n_events, class_names=class_names)
        row = _make_result_row(f"Ensemble RF {c1}+{c2}", metrics, n_events, 0)
        results.append(row)

    # 3-way ensemble
    prob_avg_3 = (rf_probs["2+1+2"] + rf_probs["2+1+4"] + rf_probs["3+1+3"]) / 3
    y_pred_3 = prob_avg_3.argmax(axis=1)
    metrics_3 = aggregate_cv_predictions(y, y_pred_3, prob_avg_3, cv_method="LOOCV", n_folds=n_events, class_names=class_names)
    row_3 = _make_result_row("Ensemble RF 3-way", metrics_3, n_events, 0)
    results.append(row_3)

    # --- PCA + ensemble variant ---
    logger.info("Testing PCA + ensemble variants...")
    max_components = n_events - 1  # PCA requires n_components < n_samples
    pca_ensemble_dims = [d for d in [64, 128] if d <= max_components]
    logger.info("PCA ensemble dims (max_components=%d): %s", max_components, pca_ensemble_dims)
    for pca_dim in pca_ensemble_dims:
        pca_probs = {}
        for ctx_name, Z_ctx in embeddings.items():
            scaler = StandardScaler()
            Z_scaled = scaler.fit_transform(Z_ctx)
            pca = PCA(n_components=pca_dim, random_state=seed)
            Z_pca = pca.fit_transform(Z_scaled)

            y_prob_pca = np.zeros((n_events, n_classes), dtype=np.float64)
            for i in range(n_events):
                train_mask = np.ones(n_events, dtype=bool)
                train_mask[i] = False
                clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
                clf.fit(Z_pca[train_mask], y[train_mask])
                probs = clf.predict_proba(Z_pca[i:i + 1])[0]
                prob_vec = np.zeros(n_classes, dtype=np.float64)
                for ci, c in enumerate(clf.classes_):
                    prob_vec[c] = probs[ci]
                y_prob_pca[i] = prob_vec
            pca_probs[ctx_name] = y_prob_pca

        # 3-way PCA ensemble
        prob_avg_pca = sum(pca_probs.values()) / len(pca_probs)
        y_pred_pca = prob_avg_pca.argmax(axis=1)
        metrics_pca = aggregate_cv_predictions(
            y, y_pred_pca, prob_avg_pca, cv_method="LOOCV", n_folds=n_events, class_names=class_names,
        )
        row_pca = _make_result_row(f"Ensemble RF 3-way PCA-{pca_dim}", metrics_pca, n_events, 0)
        results.append(row_pca)
        logger.info("  PCA-%d ensemble: Acc=%.4f", pca_dim, metrics_pca.accuracy)

    # --- Concatenated context embeddings ---
    logger.info("Testing concatenated context embeddings...")
    Z_concat = np.concatenate(list(embeddings.values()), axis=1)
    logger.info("  Concatenated shape: %s", Z_concat.shape)

    t0 = time.time()
    metrics_concat = run_sklearn_loocv(Z_concat, y, class_names, seed)
    elapsed = time.time() - t0
    row_concat = _make_result_row(f"RF Concat-3ctx (D={Z_concat.shape[1]})", metrics_concat, n_events, elapsed)
    results.append(row_concat)
    logger.info("  Concat RF: Acc=%.4f  (%.1fs)", metrics_concat.accuracy, elapsed)

    # PCA on concatenated — cap at min(n_samples, n_features) - 1
    max_components_concat = min(n_events, Z_concat.shape[1]) - 1
    pca_concat_dims = [d for d in [64, 128, 256] if d <= max_components_concat]
    logger.info("PCA concat dims (max_components=%d): %s", max_components_concat, pca_concat_dims)
    for pca_dim in pca_concat_dims:
        scaler = StandardScaler()
        Z_concat_s = scaler.fit_transform(Z_concat)
        pca = PCA(n_components=pca_dim, random_state=seed)
        Z_concat_pca = pca.fit_transform(Z_concat_s)
        metrics_cp = run_sklearn_loocv(Z_concat_pca, y, class_names, seed)
        row_cp = _make_result_row(f"RF Concat-3ctx PCA-{pca_dim}", metrics_cp, n_events, 0)
        results.append(row_cp)
        logger.info("  Concat PCA-%d RF: Acc=%.4f", pca_dim, metrics_cp.accuracy)

    _rank_results(results)
    _log_top3(results, "Multi-Context Ensemble")

    # Save
    save_results_csv(_serialize_stage_results(results), tables_dir / "exp_c_ensemble.csv")
    save_results_txt(_serialize_stage_results(results), tables_dir / "exp_c_ensemble.txt")

    return {"results": _serialize_stage_results(results)}


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Tier 2 Experiments: PCA + Multi-Context Ensemble",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--experiment", required=True, choices=["pca", "ensemble", "all"],
        help="Experiment to run: pca (Exp B), ensemble (Exp C), or all",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: cuda or cpu",
    )
    parser.add_argument(
        "--include-none", action="store_true",
        help="Include NONE as 3rd class",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--context-before", type=int, default=2,
        help="Context before for PCA experiment (default: 2, Stage 1 best)",
    )
    parser.add_argument(
        "--context-after", type=int, default=2,
        help="Context after for PCA experiment (default: 2, Stage 1 best)",
    )

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

    t_start = time.time()
    results = {}

    if args.experiment in ("pca", "all"):
        results["pca"] = run_experiment_pca(
            raw_cfg, include_none, device, seed,
            context_before=args.context_before,
            context_after=args.context_after,
        )

    if args.experiment in ("ensemble", "all"):
        results["ensemble"] = run_experiment_ensemble(
            raw_cfg, include_none, device, seed,
        )

    t_total = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Done! Total time: %.1fs (%.1f min)", t_total, t_total / 60)

    # Save combined summary
    output_dir = Path(raw_cfg.get("output_dir", "results/phase2_finetune"))
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "tier2_experiments.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved: %s", reports_dir / "tier2_experiments.json")


if __name__ == "__main__":
    main()
