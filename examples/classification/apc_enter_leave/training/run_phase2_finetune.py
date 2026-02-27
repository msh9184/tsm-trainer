"""Phase 2 v2: Focused Augmentation + Redesigned Stage Sweep.

Trains lightweight neural classification heads on pre-extracted MantisV2
embeddings to beat the Phase 1 zero-shot baseline (88.99% LOOCV accuracy).

Redesigned 5-Stage Sweep (v2):
  Stage 1: Context window exploration (4 configs)
  Stage 2: Augmentation techniques (8 configs)
  Stage 3: Head architecture (6 configs)
  Stage 4: Embedding processing — layer selection + concat (5 configs)
  Stage 5: Final combination + TTA (4 configs)

Key changes from v1:
  - Stage 1 explores context window sizes (the actual bottleneck)
  - 5 evidence-based augmentation techniques for N~100 regime
  - Test-Time Augmentation (TTA) for inference-time ensembling
  - Pre-training augmentation (DC/SMOTE) applied per-fold to avoid leakage
  - Total ~27 configs (down from ~50), each addressing a specific question

Usage:
    cd examples/classification/apc_enter_leave

    # Full 5-stage sweep
    python training/run_phase2_finetune.py \\
        --config training/configs/enter-leave-phase2.yaml

    # Stage 1 only: context window exploration
    python training/run_phase2_finetune.py --config ... --stages 1

    # Stages 2-3 only (uses default 9min context)
    python training/run_phase2_finetune.py --config ... --stages 2 3

    # CPU-only
    python training/run_phase2_finetune.py --config ... --device cpu

    # Specific layer
    python training/run_phase2_finetune.py --config ... --layer 3

    # Include NONE class
    python training/run_phase2_finetune.py --config ... --include-none
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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

# Reuse existing infrastructure (NO duplication)
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
from visualization.style import (
    CLASS_COLORS,
    CLASS_NAMES,
    FIGSIZE_SINGLE,
    setup_style,
    save_figure,
    configure_output,
)
from visualization.curves import plot_confusion_matrix
from visualization.embeddings import reduce_dimensions, reduce_dimensions_joint

# Phase 2 modules
from training.heads import build_head
from training.augmentation import (
    apply_augmentation,
    apply_pretrain_augmentation,
)

logger = logging.getLogger(__name__)

ALL_LAYERS = [0, 1, 2, 3, 4, 5]


# ============================================================================
# Training loop
# ============================================================================

@dataclass
class TrainConfig:
    """Training hyperparameters for a single neural head."""
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.01
    label_smoothing: float = 0.0
    early_stopping_patience: int = 30
    augmentation: dict | None = None
    device: str = "cpu"


def compute_wilson_ci(
    n_correct: int,
    n_total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for binomial accuracy (no scipy needed).

    Recommended by Brown, Cai & DasGupta (2001) for N~100.
    Returns (lower, upper) bounds.
    """
    z_map = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_map.get(confidence, 1.960)
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def train_head(
    head: nn.Module,
    Z_train: torch.Tensor,
    y_train: torch.Tensor,
    config: TrainConfig,
    n_classes: int,
    Z_train_std: torch.Tensor | None = None,
) -> nn.Module:
    """Train a classification head on embedding data.

    Full-batch training with AdamW + CosineAnnealingLR.

    Parameters
    ----------
    head : nn.Module
        Classification head (moved to device internally).
    Z_train : Tensor, shape (n_train, embed_dim)
        Training embeddings (already scaled).
    y_train : Tensor, shape (n_train,) int64
        Training labels.
    config : TrainConfig
    n_classes : int
        Number of classes (for augmentation soft labels).
    Z_train_std : Tensor, optional
        Per-dimension std for adaptive_noise augmentation.

    Returns
    -------
    nn.Module -- trained head (in eval mode).
    """
    device = torch.device(config.device)
    head = head.to(device)
    head.train()

    Z_train = Z_train.to(device)
    y_train = y_train.to(device)
    if Z_train_std is not None:
        Z_train_std = Z_train_std.to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs,
    )

    # Loss function
    aug_cfg = config.augmentation or {}
    strategy = aug_cfg.get("strategy", "")
    use_soft_labels = (
        strategy not in ("frofa", "adaptive_noise", "within_class_mixup", "")
        or aug_cfg.get("mixup_alpha", 0) > 0
    )

    if use_soft_labels:
        def loss_fn(logits, targets):
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            return -(targets * log_probs).sum(dim=1).mean()
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        # Apply augmentation each epoch (different noise/mixup each time)
        if config.augmentation is not None:
            Z_aug, y_aug = apply_augmentation(
                Z_train, y_train, config.augmentation,
                Z_train_std=Z_train_std,
            )
        else:
            Z_aug, y_aug = Z_train, y_train

        logits = head(Z_aug)
        loss = loss_fn(logits, y_aug)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()

        # Early stopping on training loss plateau
        if loss_val < best_loss - 1e-5:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            break

        # Converged
        if loss_val < 0.01:
            break

    head.eval()
    return head


# ============================================================================
# Neural LOOCV variants
# ============================================================================

def run_neural_loocv(
    Z: np.ndarray,
    y: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    class_names: list[str] | None = None,
    seed: int = 42,
) -> tuple[EventClassificationMetrics, np.ndarray, np.ndarray]:
    """Leave-One-Out CV with neural classification head.

    For each fold: scaler -> augment -> train head -> predict.
    Aggregate all N predictions for global metrics.

    Returns
    -------
    tuple of (metrics, y_pred_all, y_prob_all)
        y_prob_all has shape (n, n_classes) -- always multiclass probabilities.
    """
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    n_classes = len(np.unique(y))

    y_pred_all = np.zeros(n, dtype=np.int64)
    y_prob_all = np.zeros((n, n_classes), dtype=np.float64)

    device = torch.device(train_config.device)

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        Z_train_np, y_train_np = Z[train_mask], y[train_mask]
        Z_test_np = Z[i:i + 1]

        # Standard scaling
        scaler = StandardScaler()
        Z_train_s = scaler.fit_transform(Z_train_np)
        Z_test_s = scaler.transform(Z_test_np)

        # Convert to tensors
        Z_train_t = torch.from_numpy(Z_train_s).float()
        y_train_t = torch.from_numpy(y_train_np).long()
        Z_test_t = torch.from_numpy(Z_test_s).float().to(device)

        # Compute per-dim std for adaptive_noise
        Z_train_std = Z_train_t.std(dim=0)

        # Train fresh head
        torch.manual_seed(seed)
        head = head_factory()
        head = train_head(head, Z_train_t, y_train_t, train_config, n_classes, Z_train_std)

        # Predict
        with torch.no_grad():
            logits = head(Z_test_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        y_pred_all[i] = int(probs[0].argmax())
        y_prob_all[i] = probs[0]

    # Convert multiclass probs to binary if needed
    if n_classes == 2:
        y_prob_binary = y_prob_all[:, 1]
    else:
        y_prob_binary = y_prob_all

    metrics = aggregate_cv_predictions(
        y, y_pred_all, y_prob_binary,
        cv_method="LOOCV", n_folds=n, class_names=class_names,
    )
    return metrics, y_pred_all, y_prob_all


def run_neural_loocv_with_pretrain_aug(
    Z: np.ndarray,
    y: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    pretrain_aug_config: dict | None = None,
    epoch_aug_config: dict | None = None,
    class_names: list[str] | None = None,
    seed: int = 42,
) -> tuple[EventClassificationMetrics, np.ndarray, np.ndarray]:
    """LOOCV with pre-training augmentation (DC/SMOTE) applied per fold.

    Pre-training augmentation adds synthetic samples to the training set
    BEFORE tensor conversion and training. This is applied ONCE per fold
    (not per epoch) using only training data to avoid leakage.

    Epoch-level augmentation (frofa, adaptive_noise, etc.) is applied
    per-epoch during training via TrainConfig.augmentation.

    Returns
    -------
    tuple of (metrics, y_pred_all, y_prob_all)
    """
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    n_classes = len(np.unique(y))

    y_pred_all = np.zeros(n, dtype=np.int64)
    y_prob_all = np.zeros((n, n_classes), dtype=np.float64)

    device = torch.device(train_config.device)
    rng = np.random.default_rng(seed)

    # Build train config with epoch augmentation
    tc = TrainConfig(
        epochs=train_config.epochs,
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        label_smoothing=train_config.label_smoothing,
        early_stopping_patience=train_config.early_stopping_patience,
        augmentation=epoch_aug_config,
        device=train_config.device,
    )

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        Z_train_np, y_train_np = Z[train_mask], y[train_mask]
        Z_test_np = Z[i:i + 1]

        # Apply pre-training augmentation (DC/SMOTE) on numpy
        if pretrain_aug_config is not None:
            fold_rng = np.random.default_rng(rng.integers(0, 2**31))
            Z_train_aug, y_train_aug = apply_pretrain_augmentation(
                Z_train_np, y_train_np, pretrain_aug_config, fold_rng,
            )
        else:
            Z_train_aug, y_train_aug = Z_train_np, y_train_np

        # Standard scaling (fit on augmented training set)
        scaler = StandardScaler()
        Z_train_s = scaler.fit_transform(Z_train_aug)
        Z_test_s = scaler.transform(Z_test_np)

        # Convert to tensors
        Z_train_t = torch.from_numpy(Z_train_s).float()
        y_train_t = torch.from_numpy(y_train_aug).long()
        Z_test_t = torch.from_numpy(Z_test_s).float().to(device)

        Z_train_std = Z_train_t.std(dim=0)

        # Train fresh head
        torch.manual_seed(seed)
        head = head_factory()
        head = train_head(head, Z_train_t, y_train_t, tc, n_classes, Z_train_std)

        # Predict
        with torch.no_grad():
            logits = head(Z_test_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        y_pred_all[i] = int(probs[0].argmax())
        y_prob_all[i] = probs[0]

    if n_classes == 2:
        y_prob_binary = y_prob_all[:, 1]
    else:
        y_prob_binary = y_prob_all

    metrics = aggregate_cv_predictions(
        y, y_pred_all, y_prob_binary,
        cv_method="LOOCV", n_folds=n, class_names=class_names,
    )
    return metrics, y_pred_all, y_prob_all


def run_tta_loocv(
    Z: np.ndarray,
    y: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    tta_k: int = 5,
    tta_strategy: str = "frofa",
    tta_strength: float = 0.1,
    pretrain_aug_config: dict | None = None,
    class_names: list[str] | None = None,
    seed: int = 42,
) -> tuple[EventClassificationMetrics, np.ndarray, np.ndarray]:
    """Test-Time Augmentation LOOCV: average softmax over K augmented copies.

    At inference, the test sample is augmented K times and predictions
    are averaged. Uses global training statistics for augmentation
    (acceptable since TTA doesn't fit new parameters).

    Returns
    -------
    tuple of (metrics, y_pred_all, y_prob_all)
    """
    from sklearn.preprocessing import StandardScaler
    from training.augmentation import frofa_augmentation, adaptive_noise

    n = len(y)
    n_classes = len(np.unique(y))

    y_pred_all = np.zeros(n, dtype=np.int64)
    y_prob_all = np.zeros((n, n_classes), dtype=np.float64)

    device = torch.device(train_config.device)
    rng = np.random.default_rng(seed)

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        Z_train_np, y_train_np = Z[train_mask], y[train_mask]
        Z_test_np = Z[i:i + 1]

        # Pre-training augmentation
        if pretrain_aug_config is not None:
            fold_rng = np.random.default_rng(rng.integers(0, 2**31))
            Z_train_aug, y_train_aug = apply_pretrain_augmentation(
                Z_train_np, y_train_np, pretrain_aug_config, fold_rng,
            )
        else:
            Z_train_aug, y_train_aug = Z_train_np, y_train_np

        # Standard scaling
        scaler = StandardScaler()
        Z_train_s = scaler.fit_transform(Z_train_aug)
        Z_test_s = scaler.transform(Z_test_np)

        Z_train_t = torch.from_numpy(Z_train_s).float()
        y_train_t = torch.from_numpy(y_train_aug).long()
        Z_test_t = torch.from_numpy(Z_test_s).float().to(device)

        Z_train_std = Z_train_t.std(dim=0)

        # Train head (with epoch-level augmentation from train_config)
        torch.manual_seed(seed)
        head = head_factory()
        head = train_head(head, Z_train_t, y_train_t, train_config, n_classes, Z_train_std)

        # TTA: average predictions over K augmented copies of test sample
        prob_sum = np.zeros(n_classes, dtype=np.float64)

        with torch.no_grad():
            # Original prediction
            logits = head(Z_test_t)
            prob_sum += torch.softmax(logits, dim=1).cpu().numpy()[0]

            # K augmented predictions
            for k in range(tta_k):
                gen = torch.Generator(device=Z_test_t.device)
                gen.manual_seed(seed + k + 1)

                if tta_strategy == "frofa":
                    Z_aug = frofa_augmentation(Z_test_t, strength=tta_strength, generator=gen)
                elif tta_strategy == "adaptive_noise":
                    Z_aug = adaptive_noise(
                        Z_test_t, Z_train_std.to(device),
                        scale=tta_strength, generator=gen,
                    )
                else:
                    Z_aug = Z_test_t

                logits_aug = head(Z_aug)
                prob_sum += torch.softmax(logits_aug, dim=1).cpu().numpy()[0]

        # Average over 1 + K predictions
        probs_avg = prob_sum / (1 + tta_k)
        y_pred_all[i] = int(probs_avg.argmax())
        y_prob_all[i] = probs_avg

    if n_classes == 2:
        y_prob_binary = y_prob_all[:, 1]
    else:
        y_prob_binary = y_prob_all

    metrics = aggregate_cv_predictions(
        y, y_pred_all, y_prob_binary,
        cv_method=f"LOOCV-TTA{tta_k}", n_folds=n, class_names=class_names,
    )
    return metrics, y_pred_all, y_prob_all


def run_neural_loocv_multi_layer(
    all_Z: dict[int, np.ndarray],
    layer_indices: list[int],
    y: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    class_names: list[str] | None = None,
    seed: int = 42,
) -> tuple[EventClassificationMetrics, np.ndarray, np.ndarray]:
    """LOOCV with multi-layer fusion heads.

    Returns
    -------
    tuple of (metrics, y_pred_all, y_prob_all)
    """
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    n_classes = len(np.unique(y))

    y_pred_all = np.zeros(n, dtype=np.int64)
    y_prob_all = np.zeros((n, n_classes), dtype=np.float64)

    device = torch.device(train_config.device)

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        # Per-layer scaling
        train_tensors = []
        test_tensors = []
        for li in layer_indices:
            Z_layer = all_Z[li]
            scaler = StandardScaler()
            Z_train_s = scaler.fit_transform(Z_layer[train_mask])
            Z_test_s = scaler.transform(Z_layer[i:i + 1])
            train_tensors.append(torch.from_numpy(Z_train_s).float())
            test_tensors.append(torch.from_numpy(Z_test_s).float().to(device))

        y_train_t = torch.from_numpy(y[train_mask]).long()

        # Train: head expects list of per-layer tensors
        torch.manual_seed(seed)
        head = head_factory()
        head = head.to(device)
        head.train()

        optimizer = torch.optim.AdamW(
            head.parameters(), lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_config.epochs,
        )
        loss_fn = nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)

        train_inputs = [t.to(device) for t in train_tensors]
        y_train_dev = y_train_t.to(device)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(train_config.epochs):
            logits = head(train_inputs)
            loss = loss_fn(logits, y_train_dev)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            if loss_val < best_loss - 1e-5:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= train_config.early_stopping_patience:
                break
            if loss_val < 0.01:
                break

        # Predict
        head.eval()
        with torch.no_grad():
            logits = head(test_tensors)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        y_pred_all[i] = int(probs[0].argmax())
        y_prob_all[i] = probs[0]

    if n_classes == 2:
        y_prob_binary = y_prob_all[:, 1]
    else:
        y_prob_binary = y_prob_all

    metrics = aggregate_cv_predictions(
        y, y_pred_all, y_prob_binary,
        cv_method="LOOCV", n_folds=n, class_names=class_names,
    )
    return metrics, y_pred_all, y_prob_all


def run_neural_loocv_ensemble(
    Z: np.ndarray,
    y: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    n_seeds: int,
    base_seed: int = 42,
    class_names: list[str] | None = None,
) -> tuple[EventClassificationMetrics, np.ndarray, np.ndarray]:
    """Multi-seed ensemble LOOCV: average softmax across seeds.

    Returns
    -------
    tuple of (metrics, y_pred_all, y_prob_avg)
    """
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    n_classes = len(np.unique(y))

    y_prob_sum = np.zeros((n, n_classes), dtype=np.float64)
    device = torch.device(train_config.device)

    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset

        for i in range(n):
            train_mask = np.ones(n, dtype=bool)
            train_mask[i] = False

            scaler = StandardScaler()
            Z_train_s = scaler.fit_transform(Z[train_mask])
            Z_test_s = scaler.transform(Z[i:i + 1])

            Z_train_t = torch.from_numpy(Z_train_s).float()
            y_train_t = torch.from_numpy(y[train_mask]).long()
            Z_test_t = torch.from_numpy(Z_test_s).float().to(device)

            torch.manual_seed(seed)
            head = head_factory()
            head = train_head(head, Z_train_t, y_train_t, train_config, n_classes)

            with torch.no_grad():
                logits = head(Z_test_t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            y_prob_sum[i] += probs[0]

    # Average across seeds
    y_prob_avg = y_prob_sum / n_seeds
    y_pred_all = y_prob_avg.argmax(axis=1).astype(np.int64)

    if n_classes == 2:
        y_prob_out = y_prob_avg[:, 1]
    else:
        y_prob_out = y_prob_avg

    metrics = aggregate_cv_predictions(
        y, y_pred_all, y_prob_out,
        cv_method=f"LOOCV-Ensemble{n_seeds}", n_folds=n,
        class_names=class_names,
    )
    return metrics, y_pred_all, y_prob_avg


# ============================================================================
# sklearn LOOCV wrapper (for fast Stage 1 context evaluation)
# ============================================================================

def run_sklearn_loocv(
    Z: np.ndarray,
    y: np.ndarray,
    class_names: list[str] | None = None,
    seed: int = 42,
) -> EventClassificationMetrics:
    """Quick sklearn RF LOOCV for context window evaluation.

    Wraps the existing run_loocv() with a RandomForest classifier.
    Much faster than neural LOOCV — no training loop needed.
    """
    from sklearn.ensemble import RandomForestClassifier

    def clf_factory():
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)

    return run_loocv(Z, y, clf_factory, class_names=class_names)


# ============================================================================
# Stage runners (v2)
# ============================================================================

def _make_result_row(name, metrics, n_events, elapsed, extra=None):
    """Build a standardized result dict from metrics."""
    n_correct = int(round(metrics.accuracy * n_events))
    ci_lo, ci_hi = compute_wilson_ci(n_correct, n_events)
    row = {
        "rank": 0,
        "name": name,
        "accuracy": round(metrics.accuracy, 4),
        "ci_95_lower": round(ci_lo, 4),
        "ci_95_upper": round(ci_hi, 4),
        "f1_macro": round(metrics.f1_macro, 4),
        "roc_auc": round(metrics.roc_auc, 4) if not np.isnan(metrics.roc_auc) else None,
        "eer": round(metrics.eer, 4) if not np.isnan(metrics.eer) else None,
        "time_sec": round(elapsed, 1),
        "metrics": metrics,
    }
    if extra:
        row.update(extra)
    return row


def _rank_results(results):
    """Sort by accuracy (primary), F1 (secondary) and assign ranks."""
    results.sort(key=lambda r: (-r["accuracy"], -r["f1_macro"]))
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results


def _log_top3(results, stage_name):
    """Log top-3 results for a stage."""
    logger.info("-" * 40)
    logger.info("%s Top-3:", stage_name)
    for r in results[:3]:
        logger.info(
            "  #%d %s: Acc=%.4f, F1=%.4f, AUC=%s",
            r["rank"], r["name"], r["accuracy"], r["f1_macro"],
            f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "N/A",
        )


def run_stage1_context_window(
    raw_cfg: dict,
    include_none: bool,
    device: str,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """Stage 1: Context window exploration.

    Re-extracts embeddings with different context_before/context_after
    settings and evaluates with RF LOOCV (fast, no neural head needed).

    Returns
    -------
    results : list[dict]
        Sorted results.
    best_context : dict
        Best context window config for subsequent stages.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Context Window Exploration")
    logger.info("=" * 60)

    # Context configs to explore
    context_cfgs = raw_cfg.get("context_exploration", [
        {"name": "9min (4+1+4)", "context_before": 4, "context_after": 4, "context_mode": "bidirectional"},
        {"name": "21min (10+1+10)", "context_before": 10, "context_after": 10, "context_mode": "bidirectional"},
        {"name": "41min (20+1+20)", "context_before": 20, "context_after": 20, "context_mode": "bidirectional"},
        {"name": "31min backward (30+1+0)", "context_before": 30, "context_after": 0, "context_mode": "bidirectional"},
    ])

    model_cfg = raw_cfg.get("model", {})
    pretrained_name = model_cfg.get("pretrained_name", "paris-noah/MantisV2")
    output_token = model_cfg.get("output_token", "combined")
    eval_layer = 3  # Use L3 (Phase 1 best) for fair comparison

    results = []
    for i, ctx_cfg in enumerate(context_cfgs):
        name = ctx_cfg["name"]
        logger.info("[%d/%d] %s", i + 1, len(context_cfgs), name)

        # Override dataset config
        cfg_copy = copy.deepcopy(raw_cfg)
        ds_override = {
            "context_mode": ctx_cfg.get("context_mode", "bidirectional"),
            "context_before": ctx_cfg.get("context_before", 4),
            "context_after": ctx_cfg.get("context_after", 4),
        }
        cfg_copy["dataset"] = ds_override

        try:
            sensor_array, sensor_ts, event_ts, event_labels, ch_names, class_names = (
                load_data(cfg_copy, include_none=include_none)
            )
            ds_cfg = build_dataset_config(ds_override)
            dataset = EventDataset(sensor_array, sensor_ts, event_ts, event_labels, ds_cfg)

            # Extract embeddings
            network, model = load_mantis_model(pretrained_name, eval_layer, output_token, device)
            Z = extract_all_embeddings(model, dataset)
            del model, network
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            y = event_labels
            n_events = len(y)

            t0 = time.time()
            metrics = run_sklearn_loocv(Z, y, class_names, seed)
            elapsed = time.time() - t0

            row = _make_result_row(name, metrics, n_events, elapsed, extra={
                "context_cfg": ctx_cfg,
            })
            results.append(row)

            logger.info(
                "  Acc=%.4f  F1=%.4f  AUC=%s  (%.1fs)",
                metrics.accuracy, metrics.f1_macro,
                f"{metrics.roc_auc:.4f}" if not np.isnan(metrics.roc_auc) else "N/A",
                elapsed,
            )

        except Exception as e:
            logger.error("  Failed: %s", e)
            continue

    _rank_results(results)
    _log_top3(results, "Stage 1")

    # Determine best context for subsequent stages
    if results:
        best_ctx = results[0].get("context_cfg", context_cfgs[0])
    else:
        best_ctx = context_cfgs[0]

    return results, best_ctx


def run_stage2_augmentation(
    Z: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    train_config: TrainConfig,
    seed: int = 42,
) -> list[dict]:
    """Stage 2: Augmentation technique comparison.

    Tests 8 augmentation strategies using simple MLP[64]-d0.5 head.
    Includes both pre-training (DC, SMOTE) and per-epoch (FroFA,
    adaptive noise, within-class mixup) augmentations.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Augmentation Techniques")
    logger.info("=" * 60)

    embed_dim = Z.shape[1]
    n_classes = len(np.unique(y))
    n_events = len(y)

    # Fixed head: MLP[64]-d0.5 (simple, fast)
    def head_factory():
        return build_head("mlp", embed_dim, n_classes, hidden_dims=[64], dropout=0.5)

    configs = [
        # 1. Baseline (no augmentation)
        {"name": "baseline (no aug)", "pretrain_aug": None, "epoch_aug": None},
        # 2-3. Distribution Calibration variants
        {"name": "DC (a=0.5, n=50)", "pretrain_aug": {"strategy": "dc", "alpha": 0.5, "n_synthetic": 50}, "epoch_aug": None},
        {"name": "DC (a=0.3, n=100)", "pretrain_aug": {"strategy": "dc", "alpha": 0.3, "n_synthetic": 100}, "epoch_aug": None},
        # 4. SMOTE
        {"name": "SMOTE (k=5, n=30)", "pretrain_aug": {"strategy": "smote", "k": 5, "n_synthetic": 30}, "epoch_aug": None},
        # 5. FroFA
        {"name": "FroFA (s=0.1)", "pretrain_aug": None, "epoch_aug": {"strategy": "frofa", "strength": 0.1}},
        # 6. Adaptive noise
        {"name": "Adaptive noise (s=0.1)", "pretrain_aug": None, "epoch_aug": {"strategy": "adaptive_noise", "scale": 0.1}},
        # 7. Within-class mixup
        {"name": "Within-class mixup (a=0.3)", "pretrain_aug": None, "epoch_aug": {"strategy": "within_class_mixup", "alpha": 0.3}},
        # 8. DC + FroFA combined
        {"name": "DC + FroFA combined", "pretrain_aug": {"strategy": "dc", "alpha": 0.5, "n_synthetic": 50}, "epoch_aug": {"strategy": "frofa", "strength": 0.1}},
    ]

    results = []
    for i, cfg in enumerate(configs):
        logger.info("[%d/%d] %s", i + 1, len(configs), cfg["name"])

        t0 = time.time()

        has_pretrain = cfg["pretrain_aug"] is not None

        if has_pretrain or cfg["epoch_aug"] is not None:
            metrics, _, _ = run_neural_loocv_with_pretrain_aug(
                Z, y, head_factory, train_config,
                pretrain_aug_config=cfg["pretrain_aug"],
                epoch_aug_config=cfg["epoch_aug"],
                class_names=class_names, seed=seed,
            )
        else:
            metrics, _, _ = run_neural_loocv(
                Z, y, head_factory, train_config, class_names, seed,
            )

        elapsed = time.time() - t0

        row = _make_result_row(cfg["name"], metrics, n_events, elapsed, extra={
            "pretrain_aug": cfg["pretrain_aug"],
            "epoch_aug": cfg["epoch_aug"],
        })
        results.append(row)

        logger.info(
            "  Acc=%.4f  F1=%.4f  AUC=%s  (%.1fs)",
            metrics.accuracy, metrics.f1_macro,
            f"{metrics.roc_auc:.4f}" if not np.isnan(metrics.roc_auc) else "N/A",
            elapsed,
        )

    _rank_results(results)
    _log_top3(results, "Stage 2")

    return results


def run_stage3_head_architecture(
    Z: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    train_config: TrainConfig,
    best_aug: dict,
    seed: int = 42,
) -> list[dict]:
    """Stage 3: Head architecture selection.

    Tests 6 head configurations using the best augmentation from Stage 2.
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: Head Architecture")
    logger.info("=" * 60)

    embed_dim = Z.shape[1]
    n_classes = len(np.unique(y))
    n_events = len(y)

    best_pretrain = best_aug.get("pretrain_aug")
    best_epoch = best_aug.get("epoch_aug")

    configs = [
        {"name": "Linear", "type": "linear", "kwargs": {}},
        {"name": "MLP[64]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.5}},
        {"name": "MLP[128]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [128], "dropout": 0.5}},
        {"name": "MLP[128,64]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [128, 64], "dropout": 0.5}},
        {"name": "MLP[64]-d0.3", "type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.3}},
        {"name": "MLP[64]-d0.7", "type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.7}},
    ]

    results = []
    for i, cfg in enumerate(configs):
        logger.info("[%d/%d] %s", i + 1, len(configs), cfg["name"])

        def head_factory(c=cfg):
            return build_head(c["type"], embed_dim, n_classes, **c["kwargs"])

        t0 = time.time()

        if best_pretrain or best_epoch:
            metrics, _, _ = run_neural_loocv_with_pretrain_aug(
                Z, y, head_factory, train_config,
                pretrain_aug_config=best_pretrain,
                epoch_aug_config=best_epoch,
                class_names=class_names, seed=seed,
            )
        else:
            metrics, _, _ = run_neural_loocv(
                Z, y, head_factory, train_config, class_names, seed,
            )

        elapsed = time.time() - t0

        n_params = sum(p.numel() for p in head_factory().parameters())
        row = _make_result_row(cfg["name"], metrics, n_events, elapsed, extra={
            "head_type": cfg["type"],
            "n_params": n_params,
            "config": cfg,
        })
        results.append(row)

        logger.info(
            "  Acc=%.4f  F1=%.4f  AUC=%s  params=%d  (%.1fs)",
            metrics.accuracy, metrics.f1_macro,
            f"{metrics.roc_auc:.4f}" if not np.isnan(metrics.roc_auc) else "N/A",
            n_params, elapsed,
        )

    _rank_results(results)
    _log_top3(results, "Stage 3")

    return results


def run_stage4_embedding_processing(
    all_Z: dict[int, np.ndarray],
    y: np.ndarray,
    class_names: list[str],
    train_config: TrainConfig,
    best_aug: dict,
    best_head_cfg: dict,
    seed: int = 42,
) -> list[dict]:
    """Stage 4: Embedding processing -- layer selection and concatenation.

    Tests different layer combinations and fusion strategies using
    best head + augmentation from Stages 2-3.
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: Embedding Processing (Layer Selection)")
    logger.info("=" * 60)

    n_classes = len(np.unique(y))
    embed_dim = next(iter(all_Z.values())).shape[1]
    n_events = len(y)

    best_pretrain = best_aug.get("pretrain_aug")
    best_epoch = best_aug.get("epoch_aug")
    head_type = best_head_cfg.get("type", "mlp")
    head_kwargs = best_head_cfg.get("kwargs", {"hidden_dims": [64], "dropout": 0.5})

    configs = [
        {"name": "L3 only", "mode": "single", "layers": [3]},
        {"name": "L0 only", "mode": "single", "layers": [0]},
        {"name": "Concat L0+L3", "mode": "concat", "layers": [0, 3]},
        {"name": "Fusion L2+L3+L4", "mode": "fusion", "layers": [2, 3, 4]},
        {"name": "Concat L0+L3+L5", "mode": "concat", "layers": [0, 3, 5]},
    ]

    # Filter configs to only use available layers
    available_layers = set(all_Z.keys())
    configs = [c for c in configs if all(l in available_layers for l in c["layers"])]

    results = []
    for i, cfg in enumerate(configs):
        logger.info("[%d/%d] %s", i + 1, len(configs), cfg["name"])

        t0 = time.time()

        if cfg["mode"] == "single":
            Z_use = all_Z[cfg["layers"][0]]

            def head_factory(ht=head_type, kw=head_kwargs, ed=embed_dim):
                return build_head(ht, ed, n_classes, **kw)

            if best_pretrain or best_epoch:
                metrics, _, _ = run_neural_loocv_with_pretrain_aug(
                    Z_use, y, head_factory, train_config,
                    pretrain_aug_config=best_pretrain,
                    epoch_aug_config=best_epoch,
                    class_names=class_names, seed=seed,
                )
            else:
                metrics, _, _ = run_neural_loocv(
                    Z_use, y, head_factory, train_config, class_names, seed,
                )

        elif cfg["mode"] == "concat":
            Z_concat = np.concatenate([all_Z[l] for l in cfg["layers"]], axis=1)
            concat_dim = Z_concat.shape[1]
            concat_hidden = [min(256, concat_dim // 2), 128]

            def head_factory(cd=concat_dim, ch=concat_hidden, kw=head_kwargs):
                return build_head(
                    "mlp", cd, n_classes,
                    hidden_dims=ch, dropout=kw.get("dropout", 0.5),
                )

            if best_pretrain or best_epoch:
                metrics, _, _ = run_neural_loocv_with_pretrain_aug(
                    Z_concat, y, head_factory, train_config,
                    pretrain_aug_config=best_pretrain,
                    epoch_aug_config=best_epoch,
                    class_names=class_names, seed=seed,
                )
            else:
                metrics, _, _ = run_neural_loocv(
                    Z_concat, y, head_factory, train_config, class_names, seed,
                )

        elif cfg["mode"] == "fusion":
            n_layers = len(cfg["layers"])

            def head_factory(nl=n_layers, kw=head_kwargs):
                return build_head(
                    "multi_layer_fusion", embed_dim, n_classes,
                    n_layers=nl,
                    hidden_dims=kw.get("hidden_dims", [64]),
                    dropout=kw.get("dropout", 0.5),
                )

            tc = TrainConfig(
                epochs=train_config.epochs, lr=train_config.lr,
                weight_decay=train_config.weight_decay,
                label_smoothing=train_config.label_smoothing,
                early_stopping_patience=train_config.early_stopping_patience,
                device=train_config.device,
            )
            metrics, _, _ = run_neural_loocv_multi_layer(
                all_Z, cfg["layers"], y, head_factory, tc, class_names, seed,
            )
        else:
            continue

        elapsed = time.time() - t0

        row = _make_result_row(cfg["name"], metrics, n_events, elapsed, extra={
            "mode": cfg["mode"],
            "layers": cfg["layers"],
        })
        results.append(row)

        logger.info(
            "  Acc=%.4f  F1=%.4f  AUC=%s  (%.1fs)",
            metrics.accuracy, metrics.f1_macro,
            f"{metrics.roc_auc:.4f}" if not np.isnan(metrics.roc_auc) else "N/A",
            elapsed,
        )

    _rank_results(results)
    _log_top3(results, "Stage 4")

    return results


def run_stage5_final_tta(
    Z: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    train_config: TrainConfig,
    best_aug: dict,
    best_head_cfg: dict,
    seed: int = 42,
) -> list[dict]:
    """Stage 5: Final combination + Test-Time Augmentation.

    Combines best choices from Stages 1-4 and adds TTA.
    """
    logger.info("=" * 60)
    logger.info("STAGE 5: Final Combination + TTA")
    logger.info("=" * 60)

    embed_dim = Z.shape[1]
    n_classes = len(np.unique(y))
    n_events = len(y)

    head_type = best_head_cfg.get("type", "mlp")
    head_kwargs = best_head_cfg.get("kwargs", {"hidden_dims": [64], "dropout": 0.5})
    best_pretrain = best_aug.get("pretrain_aug")
    best_epoch = best_aug.get("epoch_aug")

    def head_factory(ht=head_type, kw=head_kwargs):
        return build_head(ht, embed_dim, n_classes, **kw)

    configs = [
        {"name": "Best combo (no TTA)", "tta": False, "multi_seed": None},
        {"name": "Best + TTA-5 (FroFA)", "tta": True, "tta_k": 5, "tta_strategy": "frofa"},
        {"name": "Best + TTA-10 (FroFA)", "tta": True, "tta_k": 10, "tta_strategy": "frofa"},
        {"name": "Best + multi-seed (5)", "tta": False, "multi_seed": 5},
    ]

    results = []
    for i, cfg in enumerate(configs):
        logger.info("[%d/%d] %s", i + 1, len(configs), cfg["name"])

        t0 = time.time()

        if cfg.get("tta"):
            metrics, _, _ = run_tta_loocv(
                Z, y, head_factory, train_config,
                tta_k=cfg["tta_k"],
                tta_strategy=cfg.get("tta_strategy", "frofa"),
                tta_strength=0.1,
                pretrain_aug_config=best_pretrain,
                class_names=class_names, seed=seed,
            )
        elif cfg.get("multi_seed"):
            metrics, _, _ = run_neural_loocv_ensemble(
                Z, y, head_factory, train_config,
                n_seeds=cfg["multi_seed"], base_seed=seed,
                class_names=class_names,
            )
        else:
            if best_pretrain or best_epoch:
                metrics, _, _ = run_neural_loocv_with_pretrain_aug(
                    Z, y, head_factory, train_config,
                    pretrain_aug_config=best_pretrain,
                    epoch_aug_config=best_epoch,
                    class_names=class_names, seed=seed,
                )
            else:
                metrics, _, _ = run_neural_loocv(
                    Z, y, head_factory, train_config, class_names, seed,
                )

        elapsed = time.time() - t0

        row = _make_result_row(cfg["name"], metrics, n_events, elapsed)
        results.append(row)

        logger.info(
            "  Acc=%.4f  F1=%.4f  AUC=%s  (%.1fs)",
            metrics.accuracy, metrics.f1_macro,
            f"{metrics.roc_auc:.4f}" if not np.isnan(metrics.roc_auc) else "N/A",
            elapsed,
        )

    _rank_results(results)
    _log_top3(results, "Stage 5")

    return results


# ============================================================================
# Multi-seed stability
# ============================================================================

def run_multi_seed_stability(
    Z: np.ndarray,
    y: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    n_seeds: int = 5,
    base_seed: int = 42,
    class_names: list[str] | None = None,
) -> dict:
    """Run LOOCV with multiple seeds and report stability.

    Returns dict with mean, std, min, max accuracy across seeds.
    """
    accuracies = []
    for i in range(n_seeds):
        seed = base_seed + i
        metrics, _, _ = run_neural_loocv(Z, y, head_factory, train_config, class_names, seed)
        accuracies.append(metrics.accuracy)
        logger.info("  Seed %d: Acc=%.4f", seed, metrics.accuracy)

    accs = np.array(accuracies)
    result = {
        "seeds": list(range(base_seed, base_seed + n_seeds)),
        "accuracies": [round(a, 4) for a in accuracies],
        "mean": round(float(accs.mean()), 4),
        "std": round(float(accs.std()), 4),
        "min": round(float(accs.min()), 4),
        "max": round(float(accs.max()), 4),
        "range": round(float(accs.max() - accs.min()), 4),
    }
    logger.info(
        "Stability: mean=%.4f +/- %.4f (range=%.4f, min=%.4f, max=%.4f)",
        result["mean"], result["std"], result["range"], result["min"], result["max"],
    )
    return result


# ============================================================================
# Plotting helpers
# ============================================================================

def plot_stage_bar(
    results: list[dict],
    name_key: str,
    title: str,
    output_path: Path,
) -> None:
    """Bar chart of accuracy across stage configs."""
    setup_style()

    names = [r[name_key] for r in results]
    accs = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), 5))
    colors = ["#DE8F05" if i == 0 else "#0173B2" for i in range(len(names))]
    bars = ax.barh(range(len(names)), accs, color=colors)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()

    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8,
        )

    save_figure(fig, output_path)
    plt.close(fig)


def plot_embedding_analysis(
    Z: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    all_Z: dict[int, np.ndarray] | None = None,
    learned_features: np.ndarray | None = None,
) -> None:
    """Comprehensive embedding visualization (7 plots).

    Parameters
    ----------
    Z : np.ndarray, shape (n, embed_dim)
        Primary layer embeddings.
    y : np.ndarray, shape (n,)
        True labels.
    y_pred : np.ndarray, shape (n,)
        Predicted labels from best config LOOCV.
    y_prob : np.ndarray, shape (n, n_classes)
        Softmax probabilities from best config LOOCV.
    class_names : list[str]
    output_dir : Path
        Directory for embedding plots (e.g. plots/embeddings/).
    all_Z : dict[int, np.ndarray], optional
        Per-layer embeddings for per-layer comparison plot.
    learned_features : np.ndarray, optional
        MLP hidden features (trained on ALL data) for learned features plot.
    """
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shared t-SNE coordinates for primary embeddings
    tsne_2d = reduce_dimensions(Z, method="tsne", random_state=42)
    pca_2d = reduce_dimensions(Z, method="pca", random_state=42)

    # --- Plot 1: Raw multi-method (PCA + t-SNE side-by-side) ---
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, emb_2d, method_label in [
            (axes[0], pca_2d, "PCA"),
            (axes[1], tsne_2d, "t-SNE"),
        ]:
            for cls in sorted(set(y)):
                mask = y == cls
                color = CLASS_COLORS.get(cls, "#333333")
                name = CLASS_NAMES.get(cls, f"Class {cls}")
                if cls < len(class_names):
                    name = class_names[cls]
                ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                           c=color, label=name, s=25, alpha=0.7, edgecolors="none")
            ax.set_title(method_label)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.legend(markerscale=2, fontsize=8)
        fig.suptitle("Raw MantisV2 Embeddings", fontsize=12, y=1.02)
        save_figure(fig, output_dir / "raw_multi_method")
        plt.close(fig)
    except Exception:
        logger.warning("Failed to plot raw_multi_method", exc_info=True)

    # --- Plot 2: Classification overlay (correct/incorrect) ---
    try:
        correct_mask = y_pred == y
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        for cls in sorted(set(y)):
            mask_cls = y == cls
            color = CLASS_COLORS.get(cls, "#333333")
            name = class_names[cls] if cls < len(class_names) else f"Class {cls}"

            mask_correct = mask_cls & correct_mask
            if mask_correct.any():
                ax.scatter(tsne_2d[mask_correct, 0], tsne_2d[mask_correct, 1],
                           c=color, label=f"{name} (correct)", s=30,
                           alpha=0.7, edgecolors="none", marker="o")

            mask_wrong = mask_cls & ~correct_mask
            if mask_wrong.any():
                ax.scatter(tsne_2d[mask_wrong, 0], tsne_2d[mask_wrong, 1],
                           c=color, label=f"{name} (wrong)", s=60,
                           alpha=0.9, edgecolors="black", linewidths=1.0, marker="X")

        ax.set_title(f"Classification Overlay ({int(correct_mask.sum())}/{len(y)} correct)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(markerscale=1.5, fontsize=7, loc="best")
        save_figure(fig, output_dir / "classification_overlay")
        plt.close(fig)
    except Exception:
        logger.warning("Failed to plot classification_overlay", exc_info=True)

    # --- Plot 3: Confidence map (softmax entropy) ---
    try:
        eps = 1e-10
        entropy = -(y_prob * np.log(y_prob + eps)).sum(axis=1)
        max_entropy = np.log(y_prob.shape[1])
        norm_entropy = entropy / max_entropy if max_entropy > 0 else entropy

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        scatter = ax.scatter(
            tsne_2d[:, 0], tsne_2d[:, 1],
            c=norm_entropy, cmap="RdYlBu_r", s=30, alpha=0.8,
            edgecolors="none", vmin=0, vmax=1,
        )
        cbar = fig.colorbar(scatter, ax=ax, label="Normalized Entropy")
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels(["Confident", "Medium", "Uncertain"])
        ax.set_title("Prediction Confidence Map")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        save_figure(fig, output_dir / "confidence_map")
        plt.close(fig)
    except Exception:
        logger.warning("Failed to plot confidence_map", exc_info=True)

    # --- Plot 4: Per-layer comparison (L0-L5) ---
    if all_Z and len(all_Z) > 1:
        try:
            n_layers = len(all_Z)
            n_cols = min(3, n_layers)
            n_rows = (n_layers + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes_flat = np.array(axes).flatten() if n_layers > 1 else [axes]

            for idx, layer_idx in enumerate(sorted(all_Z.keys())):
                if idx >= len(axes_flat):
                    break
                ax = axes_flat[idx]
                emb_2d = reduce_dimensions(all_Z[layer_idx], method="tsne", random_state=42)
                for cls in sorted(set(y)):
                    mask = y == cls
                    color = CLASS_COLORS.get(cls, "#333333")
                    name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
                    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                               c=color, label=name, s=20, alpha=0.6, edgecolors="none")
                ax.set_title(f"Layer {layer_idx}")
                ax.legend(markerscale=2, fontsize=7)

            for idx in range(n_layers, len(axes_flat)):
                axes_flat[idx].set_visible(False)

            fig.suptitle("Representation Evolution Across Layers", fontsize=12, y=1.02)
            save_figure(fig, output_dir / "per_layer_comparison")
            plt.close(fig)
        except Exception:
            logger.warning("Failed to plot per_layer_comparison", exc_info=True)

    # --- Plot 5: Learned features (MLP-transformed vs raw) ---
    if learned_features is not None:
        try:
            raw_2d, learned_2d = reduce_dimensions_joint(
                Z, learned_features, method="tsne", random_state=42,
            )
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, emb_2d, title in [
                (axes[0], raw_2d, "Raw Embeddings"),
                (axes[1], learned_2d, "MLP Hidden Features"),
            ]:
                for cls in sorted(set(y)):
                    mask = y == cls
                    color = CLASS_COLORS.get(cls, "#333333")
                    name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
                    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                               c=color, label=name, s=25, alpha=0.7, edgecolors="none")
                ax.set_title(title)
                ax.legend(markerscale=2, fontsize=8)
            fig.suptitle("Raw vs Learned Feature Space", fontsize=12, y=1.02)
            save_figure(fig, output_dir / "learned_features")
            plt.close(fig)
        except Exception:
            logger.warning("Failed to plot learned_features", exc_info=True)

    # --- Plot 6: Class separation (centroids + per-class scatter) ---
    centroids = {}
    try:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        for cls in sorted(set(y)):
            mask = y == cls
            color = CLASS_COLORS.get(cls, "#333333")
            name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
            pts = tsne_2d[mask]
            ax.scatter(pts[:, 0], pts[:, 1], c=color, label=name,
                       s=25, alpha=0.5, edgecolors="none")
            centroid = pts.mean(axis=0)
            centroids[cls] = centroid
            ax.scatter(*centroid, c=color, s=200, marker="*",
                       edgecolors="black", linewidths=1.0, zorder=5)

        cls_list = sorted(centroids.keys())
        for ci in range(len(cls_list)):
            for cj in range(ci + 1, len(cls_list)):
                c1, c2 = centroids[cls_list[ci]], centroids[cls_list[cj]]
                dist = np.linalg.norm(c1 - c2)
                mid = (c1 + c2) / 2
                ax.plot([c1[0], c2[0]], [c1[1], c2[1]],
                        "k--", alpha=0.3, linewidth=1)
                ax.text(mid[0], mid[1], f"{dist:.1f}", fontsize=7,
                        ha="center", va="bottom", alpha=0.6)

        ax.set_title("Class Separation (stars = centroids)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(markerscale=2, fontsize=8)
        save_figure(fig, output_dir / "class_separation")
        plt.close(fig)
    except Exception:
        logger.warning("Failed to plot class_separation", exc_info=True)

    # --- Plot 7: Analysis summary (2x3 dashboard) ---
    try:
        correct_mask = y_pred == y
        eps = 1e-10
        entropy = -(y_prob * np.log(y_prob + eps)).sum(axis=1)
        max_entropy = np.log(y_prob.shape[1])
        norm_entropy = entropy / max_entropy if max_entropy > 0 else entropy

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        # Panel 1: PCA
        ax = axes[0, 0]
        for cls in sorted(set(y)):
            mask = y == cls
            color = CLASS_COLORS.get(cls, "#333333")
            name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
            ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1],
                       c=color, label=name, s=15, alpha=0.6, edgecolors="none")
        ax.set_title("PCA", fontsize=10)
        ax.legend(markerscale=2, fontsize=6)

        # Panel 2: t-SNE
        ax = axes[0, 1]
        for cls in sorted(set(y)):
            mask = y == cls
            color = CLASS_COLORS.get(cls, "#333333")
            name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
            ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
                       c=color, label=name, s=15, alpha=0.6, edgecolors="none")
        ax.set_title("t-SNE", fontsize=10)
        ax.legend(markerscale=2, fontsize=6)

        # Panel 3: Classification overlay
        ax = axes[0, 2]
        for cls in sorted(set(y)):
            mask_cls = y == cls
            color = CLASS_COLORS.get(cls, "#333333")
            mask_ok = mask_cls & correct_mask
            mask_err = mask_cls & ~correct_mask
            if mask_ok.any():
                ax.scatter(tsne_2d[mask_ok, 0], tsne_2d[mask_ok, 1],
                           c=color, s=15, alpha=0.6, edgecolors="none")
            if mask_err.any():
                ax.scatter(tsne_2d[mask_err, 0], tsne_2d[mask_err, 1],
                           c=color, s=40, alpha=0.9, edgecolors="black",
                           linewidths=0.8, marker="X")
        ax.set_title(f"Correct/Wrong ({int(correct_mask.sum())}/{len(y)})", fontsize=10)

        # Panel 4: Confidence map
        ax = axes[1, 0]
        ax.scatter(tsne_2d[:, 0], tsne_2d[:, 1],
                   c=norm_entropy, cmap="RdYlBu_r", s=15, alpha=0.7,
                   edgecolors="none", vmin=0, vmax=1)
        ax.set_title("Confidence (entropy)", fontsize=10)

        # Panel 5: Class centroids
        ax = axes[1, 1]
        for cls in sorted(set(y)):
            mask = y == cls
            color = CLASS_COLORS.get(cls, "#333333")
            name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
            ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
                       c=color, label=name, s=15, alpha=0.4, edgecolors="none")
            if cls in centroids:
                ax.scatter(*centroids[cls], c=color, s=150, marker="*",
                           edgecolors="black", linewidths=0.8, zorder=5)
        ax.set_title("Class Centroids", fontsize=10)
        ax.legend(markerscale=2, fontsize=6)

        # Panel 6: Learned features or empty
        ax = axes[1, 2]
        if learned_features is not None:
            learned_2d_local = reduce_dimensions(learned_features, method="tsne", random_state=42)
            for cls in sorted(set(y)):
                mask = y == cls
                color = CLASS_COLORS.get(cls, "#333333")
                name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
                ax.scatter(learned_2d_local[mask, 0], learned_2d_local[mask, 1],
                           c=color, label=name, s=15, alpha=0.6, edgecolors="none")
            ax.set_title("MLP Features", fontsize=10)
            ax.legend(markerscale=2, fontsize=6)
        else:
            ax.text(0.5, 0.5, "N/A\n(LinearHead)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#999999")
            ax.set_title("Learned Features", fontsize=10)

        fig.suptitle("Embedding Analysis Summary", fontsize=13, y=1.01)
        save_figure(fig, output_dir / "analysis_summary")
        plt.close(fig)
    except Exception:
        logger.warning("Failed to plot analysis_summary", exc_info=True)

    logger.info("Saved %d embedding plots to %s", 7, output_dir)


# ============================================================================
# Result serialization
# ============================================================================

def _serialize_stage_results(results: list[dict], exclude_keys: set | None = None) -> list[dict]:
    """Convert stage results to JSON-serializable format."""
    if exclude_keys is None:
        exclude_keys = {"metrics", "config", "aug_config", "context_cfg"}

    serializable = []
    for r in results:
        row = {}
        for k, v in r.items():
            if k in exclude_keys:
                continue
            # Convert numpy types
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            row[k] = v
        serializable.append(row)
    return serializable


# ============================================================================
# Main sweep orchestrator
# ============================================================================

def run_full_sweep(
    raw_cfg: dict,
    stages: list[int],
    layer: int | None,
    include_none: bool,
    device: str,
    seed: int,
) -> dict:
    """Run the Phase 2 v2 sweep.

    5-Stage pipeline:
      1. Context window exploration (re-extracts embeddings per config)
      2. Augmentation techniques (DC, SMOTE, FroFA, etc.)
      3. Head architecture (Linear, MLP variants)
      4. Embedding processing (layer selection, concat, fusion)
      5. Final combination + TTA

    Results from each stage feed into the next (greedy forward selection).

    Parameters
    ----------
    raw_cfg : dict
        Loaded YAML config.
    stages : list[int]
        Which stages to run (1-5).
    layer : int or None
        Specific layer (None = default L3).
    include_none : bool
    device : str
    seed : int

    Returns
    -------
    dict -- comprehensive summary.
    """
    output_dir = Path(raw_cfg.get("output_dir", "results/phase2_finetune"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"
    for d in [tables_dir, plots_dir, reports_dir]:
        d.mkdir(exist_ok=True)

    configure_output(formats=["png"], dpi=200)

    # ----- Load data (default context) -----
    logger.info("Loading data...")
    sensor_array, sensor_ts, event_ts, event_labels, channel_names, class_names = (
        load_data(raw_cfg, include_none=include_none)
    )

    y = event_labels
    n_events = len(y)

    logger.info("Events: %d, Classes: %s, Channels: %s", n_events, class_names, channel_names)

    # ----- Model config -----
    model_cfg = raw_cfg.get("model", {})
    pretrained_name = model_cfg.get("pretrained_name", "paris-noah/MantisV2")
    output_token = model_cfg.get("output_token", "combined")

    # ----- Training config -----
    train_cfg_raw = raw_cfg.get("training", {})
    base_train_config = TrainConfig(
        epochs=train_cfg_raw.get("epochs", 200),
        lr=train_cfg_raw.get("lr", 1e-3),
        weight_decay=train_cfg_raw.get("weight_decay", 0.01),
        label_smoothing=train_cfg_raw.get("label_smoothing", 0.0),
        early_stopping_patience=train_cfg_raw.get("early_stopping_patience", 30),
        device=device,
    )

    # ----- Default layer -----
    primary_layer = layer if layer is not None else 3  # L3 = Phase 1 best

    # ----- Track best config across stages -----
    best_context = {
        "context_mode": "bidirectional",
        "context_before": raw_cfg.get("dataset", {}).get("context_before", 4),
        "context_after": raw_cfg.get("dataset", {}).get("context_after", 4),
    }
    best_aug = {"pretrain_aug": None, "epoch_aug": None}
    best_head_cfg = {"type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.5}}
    best_accuracy = 0.0

    summary = {
        "experiment": {
            "n_events": n_events,
            "n_channels": len(channel_names),
            "channel_names": channel_names,
            "class_names": class_names,
            "primary_layer": primary_layer,
            "include_none": include_none,
            "seed": seed,
            "device": device,
            "stages_run": stages,
        },
    }

    # ---- Stage 1: Context Window ----
    if 1 in stages:
        s1_results, best_ctx = run_stage1_context_window(
            raw_cfg, include_none, device, seed,
        )

        s1_csv = _serialize_stage_results(s1_results)
        save_results_csv(s1_csv, tables_dir / "stage1_context_window.csv")
        save_results_txt(s1_csv, tables_dir / "stage1_context_window.txt")

        try:
            plot_stage_bar(s1_results, "name", "Stage 1: Context Window", plots_dir / "stage1_context_bar")
        except Exception:
            logger.warning("Failed to plot Stage 1 bar", exc_info=True)

        if s1_results:
            best_context = best_ctx
            best_accuracy = s1_results[0]["accuracy"]
        summary["stage1"] = _serialize_stage_results(s1_results)
    else:
        s1_results = []

    # ----- Re-extract embeddings with best context for stages 2-5 -----
    # Build dataset with best context
    ds_override = {
        "context_mode": best_context.get("context_mode", "bidirectional"),
        "context_before": best_context.get("context_before", 4),
        "context_after": best_context.get("context_after", 4),
    }
    ds_cfg = build_dataset_config(ds_override)
    dataset = EventDataset(sensor_array, sensor_ts, event_ts, event_labels, ds_cfg)

    # Extract embeddings
    layers_needed = {primary_layer}
    if 4 in stages:
        layers_needed.update(ALL_LAYERS)

    all_embeddings = {}
    for l in sorted(layers_needed):
        logger.info("Extracting L%d embeddings...", l)
        network, model = load_mantis_model(pretrained_name, l, output_token, device)
        Z = extract_all_embeddings(model, dataset)
        all_embeddings[l] = Z
        del model, network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    Z_primary = all_embeddings[primary_layer]
    logger.info("Primary embeddings: L%d, shape=%s", primary_layer, Z_primary.shape)

    # ---- Stage 2: Augmentation ----
    if 2 in stages:
        s2_results = run_stage2_augmentation(
            Z_primary, y, class_names, base_train_config, seed,
        )

        s2_csv = _serialize_stage_results(s2_results)
        save_results_csv(s2_csv, tables_dir / "stage2_augmentation.csv")
        save_results_txt(s2_csv, tables_dir / "stage2_augmentation.txt")

        try:
            plot_stage_bar(s2_results, "name", "Stage 2: Augmentation", plots_dir / "stage2_augmentation_bar")
        except Exception:
            logger.warning("Failed to plot Stage 2 bar", exc_info=True)

        if s2_results:
            top = s2_results[0]
            if top["accuracy"] >= best_accuracy:
                best_accuracy = top["accuracy"]
            best_aug = {
                "pretrain_aug": top.get("pretrain_aug"),
                "epoch_aug": top.get("epoch_aug"),
            }
        summary["stage2"] = _serialize_stage_results(s2_results)
    else:
        s2_results = []

    # ---- Stage 3: Head Architecture ----
    if 3 in stages:
        s3_results = run_stage3_head_architecture(
            Z_primary, y, class_names, base_train_config, best_aug, seed,
        )

        s3_csv = _serialize_stage_results(s3_results)
        save_results_csv(s3_csv, tables_dir / "stage3_head_architecture.csv")
        save_results_txt(s3_csv, tables_dir / "stage3_head_architecture.txt")

        try:
            plot_stage_bar(s3_results, "name", "Stage 3: Head Architecture", plots_dir / "stage3_head_bar")
        except Exception:
            logger.warning("Failed to plot Stage 3 bar", exc_info=True)

        if s3_results:
            top = s3_results[0]
            if top["accuracy"] >= best_accuracy:
                best_accuracy = top["accuracy"]
            cfg = top.get("config", {})
            best_head_cfg = {"type": cfg.get("type", "mlp"), "kwargs": cfg.get("kwargs", {"hidden_dims": [64], "dropout": 0.5})}

            # Confusion matrix for best Stage 3
            try:
                fig, _ = plot_confusion_matrix(
                    top["metrics"].confusion_matrix,
                    class_names=class_names,
                    output_path=plots_dir / "confusion_matrix_stage3_best",
                )
                plt.close(fig)
            except Exception:
                logger.warning("Failed to plot Stage 3 confusion matrix", exc_info=True)

        summary["stage3"] = _serialize_stage_results(s3_results)
    else:
        s3_results = []

    # ---- Stage 4: Embedding Processing ----
    if 4 in stages:
        s4_results = run_stage4_embedding_processing(
            all_embeddings, y, class_names, base_train_config,
            best_aug, best_head_cfg, seed,
        )

        s4_csv = _serialize_stage_results(s4_results)
        save_results_csv(s4_csv, tables_dir / "stage4_embedding_processing.csv")
        save_results_txt(s4_csv, tables_dir / "stage4_embedding_processing.txt")

        try:
            plot_stage_bar(s4_results, "name", "Stage 4: Embedding Processing", plots_dir / "stage4_embedding_bar")
        except Exception:
            logger.warning("Failed to plot Stage 4 bar", exc_info=True)

        if s4_results and s4_results[0]["accuracy"] >= best_accuracy:
            best_accuracy = s4_results[0]["accuracy"]
        summary["stage4"] = _serialize_stage_results(s4_results)
    else:
        s4_results = []

    # ---- Stage 5: Final + TTA ----
    if 5 in stages:
        s5_results = run_stage5_final_tta(
            Z_primary, y, class_names, base_train_config,
            best_aug, best_head_cfg, seed,
        )

        s5_csv = _serialize_stage_results(s5_results)
        save_results_csv(s5_csv, tables_dir / "stage5_final_tta.csv")
        save_results_txt(s5_csv, tables_dir / "stage5_final_tta.txt")

        try:
            plot_stage_bar(s5_results, "name", "Stage 5: Final + TTA", plots_dir / "stage5_tta_bar")
        except Exception:
            logger.warning("Failed to plot Stage 5 bar", exc_info=True)

        if s5_results and s5_results[0]["accuracy"] >= best_accuracy:
            best_accuracy = s5_results[0]["accuracy"]
        summary["stage5"] = _serialize_stage_results(s5_results)
    else:
        s5_results = []

    # ----- Overall best table -----
    phase1_baseline = 0.8899  # Phase 1 best: M+C, L3, RF, LOOCV
    best_phase2 = best_accuracy

    all_stage_results = []
    for stage_name, stage_results in [
        ("stage1", s1_results), ("stage2", s2_results),
        ("stage3", s3_results), ("stage4", s4_results),
        ("stage5", s5_results),
    ]:
        for r in stage_results:
            all_stage_results.append({
                "stage": stage_name,
                "name": r.get("name", "?"),
                "accuracy": r["accuracy"],
                "ci_95_lower": r.get("ci_95_lower"),
                "ci_95_upper": r.get("ci_95_upper"),
                "f1_macro": r["f1_macro"],
                "roc_auc": r.get("roc_auc"),
            })

    all_stage_results.sort(key=lambda r: -r["accuracy"])
    top_overall = all_stage_results[:10]
    save_results_csv(top_overall, tables_dir / "overall_best.csv")
    save_results_txt(top_overall, tables_dir / "overall_best.txt")

    # ----- Best config re-evaluation -----
    logger.info("=" * 60)
    logger.info("FINAL ANALYSIS: Best config re-evaluation")
    logger.info("=" * 60)

    embed_dim = Z_primary.shape[1]
    n_classes = len(np.unique(y))

    def best_head_factory(ht=best_head_cfg["type"], kw=best_head_cfg["kwargs"]):
        return build_head(ht, embed_dim, n_classes, **kw)

    best_pretrain = best_aug.get("pretrain_aug")
    best_epoch = best_aug.get("epoch_aug")

    # Re-run best config LOOCV for per-sample predictions
    logger.info("Re-running best config LOOCV for per-sample predictions...")
    if best_pretrain or best_epoch:
        best_metrics, y_pred_best, y_prob_best = run_neural_loocv_with_pretrain_aug(
            Z_primary, y, best_head_factory, base_train_config,
            pretrain_aug_config=best_pretrain,
            epoch_aug_config=best_epoch,
            class_names=class_names, seed=seed,
        )
    else:
        best_metrics, y_pred_best, y_prob_best = run_neural_loocv(
            Z_primary, y, best_head_factory, base_train_config, class_names, seed,
        )

    # Wilson CI
    n_correct_best = int(round(best_metrics.accuracy * n_events))
    ci_lo_best, ci_hi_best = compute_wilson_ci(n_correct_best, n_events)
    logger.info(
        "Best config: Acc=%.4f  95%% CI=[%.4f, %.4f]",
        best_metrics.accuracy, ci_lo_best, ci_hi_best,
    )

    # Confusion matrix
    try:
        fig, _ = plot_confusion_matrix(
            best_metrics.confusion_matrix,
            class_names=class_names,
            output_path=plots_dir / "confusion_matrix_best",
        )
        plt.close(fig)
    except Exception:
        logger.warning("Failed to plot best confusion matrix", exc_info=True)

    # Multi-seed stability
    logger.info("Running multi-seed stability check (5 seeds)...")
    stability = run_multi_seed_stability(
        Z_primary, y, best_head_factory, base_train_config,
        n_seeds=5, base_seed=seed, class_names=class_names,
    )
    save_results_csv(
        [{"seed": s, "accuracy": a} for s, a in zip(stability["seeds"], stability["accuracies"])],
        tables_dir / "multi_seed_stability.csv",
    )

    # Embedding visualization
    logger.info("Generating embedding visualizations...")

    learned_features = None
    if best_head_cfg.get("type") == "mlp":
        try:
            from sklearn.preprocessing import StandardScaler
            torch.manual_seed(seed)
            full_head = best_head_factory()
            Z_scaled = StandardScaler().fit_transform(Z_primary)
            Z_t = torch.from_numpy(Z_scaled).float()
            y_t = torch.from_numpy(y).long()
            full_head = train_head(full_head, Z_t, y_t, base_train_config, n_classes)
            with torch.no_grad():
                feat_t = full_head.features(Z_t.to(torch.device(device)))
                learned_features = feat_t.cpu().numpy()
            logger.info("Extracted learned features: shape=%s", learned_features.shape)
        except Exception:
            logger.warning("Failed to extract learned features", exc_info=True)

    try:
        plot_embedding_analysis(
            Z_primary, y, y_pred_best, y_prob_best, class_names,
            output_dir=plots_dir / "embeddings",
            all_Z=all_embeddings if len(all_embeddings) > 1 else None,
            learned_features=learned_features,
        )
    except Exception:
        logger.warning("Failed to generate embedding plots", exc_info=True)

    # ----- Save JSON summary -----
    summary["best_overall"] = {
        "accuracy": best_phase2,
        "confidence_interval": {
            "level": 0.95,
            "method": "wilson",
            "lower": round(ci_lo_best, 4),
            "upper": round(ci_hi_best, 4),
            "width": round(ci_hi_best - ci_lo_best, 4),
        },
        "stability": stability,
        "phase1_baseline": phase1_baseline,
        "improvement": round(best_phase2 - phase1_baseline, 4),
        "best_context": best_context,
        "best_augmentation": best_aug,
        "best_head": best_head_cfg,
    }
    summary["top_overall"] = top_overall

    report_path = reports_dir / "summary.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved summary: %s", report_path)

    # ----- Text report -----
    report_lines = [
        "Phase 2 v2: Focused Augmentation + Redesigned Stage Sweep",
        "=" * 60,
        "",
        f"Events: {n_events} ({', '.join(f'{n}={int((y == i).sum())}' for i, n in enumerate(class_names))})",
        f"Channels: {len(channel_names)} ({', '.join(channel_names)})",
        f"Primary layer: L{primary_layer}",
        f"Best context: {best_context}",
        f"Best augmentation: {best_aug}",
        f"Best head: {best_head_cfg}",
        f"Device: {device}",
        "",
        f"Best accuracy: {best_phase2:.4f}  95% CI=[{ci_lo_best:.4f}, {ci_hi_best:.4f}]",
        f"Stability (5 seeds): mean={stability['mean']:.4f} +/- {stability['std']:.4f}"
        f"  (range={stability['range']:.4f})",
        f"Phase 1 reference: {phase1_baseline:.4f}  (improvement: {best_phase2 - phase1_baseline:+.4f})",
        "",
        "Top-10 Overall:",
    ]
    for i, r in enumerate(top_overall):
        ci_str = ""
        if r.get("ci_95_lower") is not None:
            ci_str = f"  CI=[{r['ci_95_lower']:.4f}, {r['ci_95_upper']:.4f}]"
        report_lines.append(
            f"  #{i + 1} [{r['stage']}] {r['name']}: Acc={r['accuracy']:.4f}{ci_str}"
        )

    report_txt_path = reports_dir / "phase2_report.txt"
    report_txt_path.write_text("\n".join(report_lines) + "\n")
    logger.info("Saved report: %s", report_txt_path)

    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 v2: Focused Augmentation + Redesigned Stage Sweep",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--stages", type=int, nargs="+", default=None,
        help="Stages to run (1-5, default: from config or all)",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Primary transformer layer (default: 3)",
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
    stages = args.stages if args.stages is not None else raw_cfg.get("sweep", {}).get("stages", [1, 2, 3, 4, 5])

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info("=" * 60)
    logger.info("Phase 2 v2: Focused Augmentation + Redesigned Stage Sweep")
    logger.info("=" * 60)
    logger.info("  Stages: %s", stages)
    logger.info("  Layer: %s", args.layer or "3 (default)")
    logger.info("  Device: %s", device)
    logger.info("  Seed: %d", seed)
    logger.info("  Include NONE: %s", include_none)

    t_start = time.time()

    summary = run_full_sweep(
        raw_cfg,
        stages=stages,
        layer=args.layer,
        include_none=include_none,
        device=device,
        seed=seed,
    )

    t_total = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Done! Total time: %.1fs (%.1f min)", t_total, t_total / 60)

    best = summary.get("best_overall", {})
    ci = best.get("confidence_interval", {})
    stab = best.get("stability", {})
    logger.info(
        "Best: %.4f  95%% CI=[%.4f, %.4f]  stability=%.4f +/- %.4f",
        best.get("accuracy", 0),
        ci.get("lower", 0), ci.get("upper", 0),
        stab.get("mean", 0), stab.get("std", 0),
    )


if __name__ == "__main__":
    main()
