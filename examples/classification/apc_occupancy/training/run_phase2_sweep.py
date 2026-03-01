#!/usr/bin/env python3
"""Phase 2 Comprehensive Sweep: MantisV2 Frozen Embeddings + Neural Classification Heads.

Trains lightweight neural classification heads on pre-extracted MantisV2
embeddings to beat the Phase 1 sklearn baseline.

Designed for 3 parallel GPU servers to find optimal configuration.

Three sweep groups (run independently on separate GPUs):

  Group A — Augmentation × Head Architecture (48 experiments)
    - 8 augmentation strategies × 6 head architectures
    - Fixed: best context/layer/channels from Phase 1
    - Goal: Find best augmentation + head combination

  Group B — Layer Selection & Fusion (30 experiments)
    - 6 single layers + 10 concat combos + 5 fusion configs
    + 9 attention-pool configs = 30 experiments
    - Fixed: best augmentation from Group A
    - Goal: Find optimal layer utilization strategy

  Group C — Training Hyperparameters + TTA + Ensemble (42 experiments)
    - 6 LR × 3 weight_decay + 6 TTA configs + 6 ensemble configs
    + 3 label smoothing + 6 epoch/patience combos = 42 experiments
    - Fixed: best choices from Groups A & B
    - Goal: Fine-tune training + inference-time ensembling

Usage:
  cd examples/classification/apc_occupancy

  # Run all groups sequentially
  python training/run_phase2_sweep.py \\
      --config training/configs/occupancy-phase2.yaml --device cuda

  # Run single group (for parallel GPU servers)
  python training/run_phase2_sweep.py \\
      --config training/configs/occupancy-phase2.yaml --group A --device cuda
  python training/run_phase2_sweep.py \\
      --config training/configs/occupancy-phase2.yaml --group B --device cuda
  python training/run_phase2_sweep.py \\
      --config training/configs/occupancy-phase2.yaml --group C --device cuda
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
import torch
import torch.nn as nn

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
from training.heads import build_head
from training.augmentation import (
    apply_augmentation,
    apply_pretrain_augmentation,
)
from visualization.style import (
    setup_style,
    save_figure,
    configure_output,
    CLASS_COLORS,
    CLASS_NAMES,
    FIGSIZE_SINGLE,
)
from visualization.curves import plot_confusion_matrix
from visualization.embeddings import reduce_dimensions

logger = logging.getLogger(__name__)

ALL_LAYERS = [0, 1, 2, 3, 4, 5]


# ============================================================================
# Training config
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


# ============================================================================
# Model + Embedding utilities (shared with Phase 1)
# ============================================================================

def load_mantis_model(pretrained_name: str, layer: int, output_token: str, device: str):
    """Load MantisV2 model + trainer wrapper."""
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
    return model


def extract_embeddings(model, dataset: OccupancyDataset, device: str) -> np.ndarray:
    """Extract frozen embeddings for all windows in the dataset.

    Returns shape (n_windows, embed_dim).
    """
    X, _ = dataset.get_numpy_arrays()  # (N, C, L)
    n_samples, n_channels, seq_len = X.shape

    # MantisV2 Channel Independence: extract per-channel, then concatenate
    all_embeddings = []
    for ch in range(n_channels):
        X_ch = X[:, [ch], :]  # (N, 1, L)
        Z_ch = model.transform(X_ch)  # (N, D)
        all_embeddings.append(Z_ch)

    Z = np.concatenate(all_embeddings, axis=-1)  # (N, C*D)

    # NaN safety
    if np.isnan(Z).any():
        n_nan = np.isnan(Z).sum()
        logger.warning("Found %d NaN values in embeddings, replacing with 0", n_nan)
        Z = np.nan_to_num(Z, nan=0.0)

    return Z


def _extract_prob_positive(head: nn.Module, Z_test: torch.Tensor, device: str) -> np.ndarray:
    """Extract P(class=1) from neural head."""
    head.eval()
    with torch.no_grad():
        logits = head(Z_test.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    if probs.shape[1] >= 2:
        return probs[:, 1]
    return probs[:, 0]


# ============================================================================
# Neural training loop
# ============================================================================

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

        if loss_val < best_loss - 1e-5:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            break
        if loss_val < 0.01:
            break

    head.eval()
    return head


# ============================================================================
# Train/test evaluation (occupancy-specific: date-based split)
# ============================================================================

def run_neural_train_test(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    pretrain_aug_config: dict | None = None,
    epoch_aug_config: dict | None = None,
    seed: int = 42,
) -> tuple[ClassificationMetrics, np.ndarray, np.ndarray]:
    """Train neural head and evaluate on test set.

    Supports both pre-training augmentation (DC/SMOTE on numpy)
    and per-epoch augmentation (FroFA/noise on tensors).

    Returns
    -------
    tuple of (metrics, y_pred, y_prob)
    """
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    device = torch.device(train_config.device)

    # Pre-training augmentation (DC/SMOTE) on numpy
    if pretrain_aug_config is not None:
        Z_train_aug, y_train_aug = apply_pretrain_augmentation(
            Z_train, y_train, pretrain_aug_config, rng,
        )
    else:
        Z_train_aug, y_train_aug = Z_train, y_train

    # Standard scaling
    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train_aug)
    Z_test_s = scaler.transform(Z_test)

    # Convert to tensors
    Z_train_t = torch.from_numpy(Z_train_s).float()
    y_train_t = torch.from_numpy(y_train_aug).long()
    Z_test_t = torch.from_numpy(Z_test_s).float().to(device)

    Z_train_std = Z_train_t.std(dim=0)

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

    # Train fresh head
    torch.manual_seed(seed)
    head = head_factory()
    head = train_head(head, Z_train_t, y_train_t, tc, n_classes, Z_train_std)

    # Predict
    with torch.no_grad():
        logits = head(Z_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    y_pred = probs.argmax(axis=1).astype(np.int64)
    y_prob = probs[:, 1] if n_classes == 2 else probs

    metrics = compute_metrics(y_test, y_pred, y_prob, class_names=["Empty", "Occupied"])
    return metrics, y_pred, y_prob


def run_neural_train_test_multi_layer(
    all_Z_train: dict[int, np.ndarray],
    all_Z_test: dict[int, np.ndarray],
    layer_indices: list[int],
    y_train: np.ndarray,
    y_test: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    seed: int = 42,
) -> tuple[ClassificationMetrics, np.ndarray, np.ndarray]:
    """Train/test with multi-layer fusion heads."""
    from sklearn.preprocessing import StandardScaler

    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    device = torch.device(train_config.device)

    # Per-layer scaling
    train_tensors = []
    test_tensors = []
    for li in layer_indices:
        scaler = StandardScaler()
        Z_train_s = scaler.fit_transform(all_Z_train[li])
        Z_test_s = scaler.transform(all_Z_test[li])
        train_tensors.append(torch.from_numpy(Z_train_s).float())
        test_tensors.append(torch.from_numpy(Z_test_s).float().to(device))

    y_train_t = torch.from_numpy(y_train).long()

    # Train
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

    y_pred = probs.argmax(axis=1).astype(np.int64)
    y_prob = probs[:, 1] if n_classes == 2 else probs

    metrics = compute_metrics(y_test, y_pred, y_prob, class_names=["Empty", "Occupied"])
    return metrics, y_pred, y_prob


def run_tta_train_test(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    tta_k: int = 5,
    tta_strategy: str = "frofa",
    tta_strength: float = 0.1,
    pretrain_aug_config: dict | None = None,
    seed: int = 42,
) -> tuple[ClassificationMetrics, np.ndarray, np.ndarray]:
    """Test-Time Augmentation: average softmax over K augmented copies."""
    from sklearn.preprocessing import StandardScaler
    from training.augmentation import frofa_augmentation, adaptive_noise

    rng = np.random.default_rng(seed)
    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    device = torch.device(train_config.device)

    # Pre-training augmentation
    if pretrain_aug_config is not None:
        Z_train_aug, y_train_aug = apply_pretrain_augmentation(
            Z_train, y_train, pretrain_aug_config, rng,
        )
    else:
        Z_train_aug, y_train_aug = Z_train, y_train

    # Standard scaling
    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train_aug)
    Z_test_s = scaler.transform(Z_test)

    Z_train_t = torch.from_numpy(Z_train_s).float()
    y_train_t = torch.from_numpy(y_train_aug).long()
    Z_test_t = torch.from_numpy(Z_test_s).float().to(device)

    Z_train_std = Z_train_t.std(dim=0)

    # Train head
    torch.manual_seed(seed)
    head = head_factory()
    head = train_head(head, Z_train_t, y_train_t, train_config, n_classes, Z_train_std)

    # TTA: average predictions over K augmented copies
    n_test = len(y_test)
    prob_sum = np.zeros((n_test, n_classes), dtype=np.float64)

    with torch.no_grad():
        # Original prediction
        logits = head(Z_test_t)
        prob_sum += torch.softmax(logits, dim=1).cpu().numpy()

        # K augmented predictions
        for k in range(tta_k):
            gen = torch.Generator(device=Z_test_t.device)
            gen.manual_seed(seed + k + 1)

            if tta_strategy == "frofa":
                Z_aug = frofa_augmentation(
                    Z_test_t, strength=tta_strength, generator=gen,
                    Z_train_std=Z_train_std.to(device),
                )
            elif tta_strategy == "adaptive_noise":
                Z_aug = adaptive_noise(
                    Z_test_t, Z_train_std.to(device),
                    scale=tta_strength, generator=gen,
                )
            else:
                Z_aug = Z_test_t

            logits_aug = head(Z_aug)
            prob_sum += torch.softmax(logits_aug, dim=1).cpu().numpy()

    # Average over 1 + K predictions
    probs_avg = prob_sum / (1 + tta_k)
    y_pred = probs_avg.argmax(axis=1).astype(np.int64)
    y_prob = probs_avg[:, 1] if n_classes == 2 else probs_avg

    metrics = compute_metrics(y_test, y_pred, y_prob, class_names=["Empty", "Occupied"])
    return metrics, y_pred, y_prob


def run_ensemble_train_test(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    n_seeds: int = 5,
    base_seed: int = 42,
    pretrain_aug_config: dict | None = None,
    epoch_aug_config: dict | None = None,
) -> tuple[ClassificationMetrics, np.ndarray, np.ndarray]:
    """Multi-seed ensemble: average softmax across seeds."""
    from sklearn.preprocessing import StandardScaler

    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    device = torch.device(train_config.device)
    n_test = len(y_test)
    prob_sum = np.zeros((n_test, n_classes), dtype=np.float64)

    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        rng = np.random.default_rng(seed)

        # Pre-training augmentation
        if pretrain_aug_config is not None:
            Z_train_aug, y_train_aug = apply_pretrain_augmentation(
                Z_train, y_train, pretrain_aug_config, rng,
            )
        else:
            Z_train_aug, y_train_aug = Z_train, y_train

        scaler = StandardScaler()
        Z_train_s = scaler.fit_transform(Z_train_aug)
        Z_test_s = scaler.transform(Z_test)

        Z_train_t = torch.from_numpy(Z_train_s).float()
        y_train_t = torch.from_numpy(y_train_aug).long()
        Z_test_t = torch.from_numpy(Z_test_s).float().to(device)

        Z_train_std = Z_train_t.std(dim=0)

        tc = TrainConfig(
            epochs=train_config.epochs,
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
            label_smoothing=train_config.label_smoothing,
            early_stopping_patience=train_config.early_stopping_patience,
            augmentation=epoch_aug_config,
            device=train_config.device,
        )

        torch.manual_seed(seed)
        head = head_factory()
        head = train_head(head, Z_train_t, y_train_t, tc, n_classes, Z_train_std)

        with torch.no_grad():
            logits = head(Z_test_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        prob_sum += probs

    probs_avg = prob_sum / n_seeds
    y_pred = probs_avg.argmax(axis=1).astype(np.int64)
    y_prob = probs_avg[:, 1] if n_classes == 2 else probs_avg

    metrics = compute_metrics(y_test, y_pred, y_prob, class_names=["Empty", "Occupied"])
    return metrics, y_pred, y_prob


# ============================================================================
# Result helpers
# ============================================================================

def _make_result_row(name: str, metrics: ClassificationMetrics, elapsed: float,
                     extra: dict | None = None) -> dict:
    """Build a standardized result dict."""
    row = {
        "name": name,
        "accuracy": round(metrics.accuracy, 4),
        "f1": round(metrics.f1, 4),
        "f1_macro": round(metrics.f1_macro, 4),
        "auc": round(metrics.roc_auc, 4) if not np.isnan(metrics.roc_auc) else None,
        "eer": round(metrics.eer, 4) if not np.isnan(metrics.eer) else None,
        "ci_lower": round(metrics.ci_lower, 4) if metrics.ci_lower is not None else None,
        "ci_upper": round(metrics.ci_upper, 4) if metrics.ci_upper is not None else None,
        "n_samples": metrics.n_samples,
        "time_s": round(elapsed, 1),
    }
    if extra:
        row.update(extra)
    return row


# ============================================================================
# Group A: Augmentation × Head Architecture (48 experiments)
# ============================================================================

# 8 augmentation strategies
AUGMENTATION_CONFIGS = [
    {"name": "no aug", "pretrain_aug": None, "epoch_aug": None},
    {"name": "DC (a=0.5, n=50)", "pretrain_aug": {"strategy": "dc", "alpha": 0.5, "n_synthetic": 50}, "epoch_aug": None},
    {"name": "DC (a=0.3, n=100)", "pretrain_aug": {"strategy": "dc", "alpha": 0.3, "n_synthetic": 100}, "epoch_aug": None},
    {"name": "SMOTE (k=5, n=30)", "pretrain_aug": {"strategy": "smote", "k": 5, "n_synthetic": 30}, "epoch_aug": None},
    {"name": "FroFA (s=0.1)", "pretrain_aug": None, "epoch_aug": {"strategy": "frofa", "strength": 0.1}},
    {"name": "Adaptive noise (s=0.1)", "pretrain_aug": None, "epoch_aug": {"strategy": "adaptive_noise", "scale": 0.1}},
    {"name": "Within-class mixup (a=0.3)", "pretrain_aug": None, "epoch_aug": {"strategy": "within_class_mixup", "alpha": 0.3}},
    {"name": "DC+FroFA combined", "pretrain_aug": {"strategy": "dc", "alpha": 0.5, "n_synthetic": 50}, "epoch_aug": {"strategy": "frofa", "strength": 0.1}},
]

# 6 head architectures
HEAD_CONFIGS = [
    {"name": "Linear", "type": "linear", "kwargs": {}},
    {"name": "MLP[64]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.5}},
    {"name": "MLP[128]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [128], "dropout": 0.5}},
    {"name": "MLP[128,64]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [128, 64], "dropout": 0.5}},
    {"name": "MLP[64]-d0.3", "type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.3}},
    {"name": "MLP[64]-d0.7", "type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.7}},
]


def run_group_a(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    train_config: TrainConfig,
    seed: int,
    output_dir: Path,
) -> list[dict]:
    """Sweep augmentation × head architecture.

    8 augmentations × 6 heads = 48 experiments.
    """
    logger.info("=" * 70)
    logger.info("GROUP A: Augmentation × Head Architecture (48 experiments)")
    logger.info("=" * 70)

    embed_dim = Z_train.shape[1]
    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    results = []

    for aug_cfg in AUGMENTATION_CONFIGS:
        for head_cfg in HEAD_CONFIGS:
            exp_name = f"{aug_cfg['name']} | {head_cfg['name']}"
            logger.info("  %s", exp_name)
            t0 = time.time()

            try:
                def head_factory(hc=head_cfg):
                    return build_head(hc["type"], embed_dim, n_classes, **hc["kwargs"])

                metrics, _, _ = run_neural_train_test(
                    Z_train, y_train, Z_test, y_test,
                    head_factory, train_config,
                    pretrain_aug_config=aug_cfg["pretrain_aug"],
                    epoch_aug_config=aug_cfg["epoch_aug"],
                    seed=seed,
                )
                elapsed = time.time() - t0

                n_params = sum(p.numel() for p in head_factory().parameters())
                row = _make_result_row(exp_name, metrics, elapsed, extra={
                    "group": "A",
                    "augmentation": aug_cfg["name"],
                    "head": head_cfg["name"],
                    "head_type": head_cfg["type"],
                    "n_params": n_params,
                })
                results.append(row)
                logger.info(
                    "    Acc=%.4f  F1=%.4f  AUC=%s  (%.1fs)",
                    metrics.accuracy, metrics.f1,
                    f"{metrics.roc_auc:.4f}" if not np.isnan(metrics.roc_auc) else "N/A",
                    elapsed,
                )
            except Exception as e:
                logger.error("    FAILED: %s", e)
                results.append({
                    "group": "A", "name": exp_name,
                    "augmentation": aug_cfg["name"],
                    "head": head_cfg["name"],
                    "error": str(e),
                })

    # Save results
    df = pd.DataFrame(results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "group_a_aug_head.csv", index=False)
    logger.info("Saved: %s (%d rows)", tables_dir / "group_a_aug_head.csv", len(df))

    return results


# ============================================================================
# Group B: Layer Selection & Fusion (30 experiments)
# ============================================================================

# Single layer configs
SINGLE_LAYER_CONFIGS = [{"name": f"L{l} only", "mode": "single", "layers": [l]} for l in ALL_LAYERS]

# Concatenation combos (10)
CONCAT_CONFIGS = [
    {"name": "Concat L0+L3", "mode": "concat", "layers": [0, 3]},
    {"name": "Concat L0+L5", "mode": "concat", "layers": [0, 5]},
    {"name": "Concat L2+L3", "mode": "concat", "layers": [2, 3]},
    {"name": "Concat L3+L5", "mode": "concat", "layers": [3, 5]},
    {"name": "Concat L0+L2+L3", "mode": "concat", "layers": [0, 2, 3]},
    {"name": "Concat L0+L3+L5", "mode": "concat", "layers": [0, 3, 5]},
    {"name": "Concat L1+L3+L5", "mode": "concat", "layers": [1, 3, 5]},
    {"name": "Concat L2+L3+L4", "mode": "concat", "layers": [2, 3, 4]},
    {"name": "Concat L0+L2+L4+L5", "mode": "concat", "layers": [0, 2, 4, 5]},
    {"name": "Concat All L0-L5", "mode": "concat", "layers": [0, 1, 2, 3, 4, 5]},
]

# Fusion configs (5)
FUSION_CONFIGS = [
    {"name": "Fusion L2+L3+L4", "mode": "fusion", "layers": [2, 3, 4]},
    {"name": "Fusion L0+L3+L5", "mode": "fusion", "layers": [0, 3, 5]},
    {"name": "Fusion L1+L3+L5", "mode": "fusion", "layers": [1, 3, 5]},
    {"name": "Fusion L0+L2+L4+L5", "mode": "fusion", "layers": [0, 2, 4, 5]},
    {"name": "Fusion All L0-L5", "mode": "fusion", "layers": [0, 1, 2, 3, 4, 5]},
]

# Attention pool configs (9)
ATTENTION_CONFIGS = [
    {"name": "Attn L0+L3", "mode": "attention", "layers": [0, 3]},
    {"name": "Attn L2+L3+L4", "mode": "attention", "layers": [2, 3, 4]},
    {"name": "Attn L0+L3+L5", "mode": "attention", "layers": [0, 3, 5]},
    {"name": "Attn L1+L3+L5", "mode": "attention", "layers": [1, 3, 5]},
    {"name": "Attn L0+L2+L4+L5", "mode": "attention", "layers": [0, 2, 4, 5]},
    {"name": "Attn All L0-L5", "mode": "attention", "layers": [0, 1, 2, 3, 4, 5]},
    {"name": "Attn L3+L4+L5", "mode": "attention", "layers": [3, 4, 5]},
    {"name": "Attn L0+L1+L2", "mode": "attention", "layers": [0, 1, 2]},
    {"name": "Attn L0+L5", "mode": "attention", "layers": [0, 5]},
]


def run_group_b(
    all_Z_train: dict[int, np.ndarray],
    all_Z_test: dict[int, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_config: TrainConfig,
    best_aug: dict,
    seed: int,
    output_dir: Path,
) -> list[dict]:
    """Sweep layer combinations.

    6 single + 10 concat + 5 fusion + 9 attention = 30 experiments.
    """
    logger.info("=" * 70)
    logger.info("GROUP B: Layer Selection & Fusion (30 experiments)")
    logger.info("=" * 70)

    embed_dim = next(iter(all_Z_train.values())).shape[1]
    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))

    best_pretrain = best_aug.get("pretrain_aug")
    best_epoch = best_aug.get("epoch_aug")

    all_configs = SINGLE_LAYER_CONFIGS + CONCAT_CONFIGS + FUSION_CONFIGS + ATTENTION_CONFIGS
    results = []

    for cfg in all_configs:
        exp_name = cfg["name"]
        logger.info("  %s", exp_name)
        t0 = time.time()

        try:
            if cfg["mode"] == "single":
                layer = cfg["layers"][0]
                Z_tr = all_Z_train[layer]
                Z_te = all_Z_test[layer]

                def head_factory(ed=embed_dim):
                    return build_head("mlp", ed, n_classes, hidden_dims=[64], dropout=0.5)

                metrics, _, _ = run_neural_train_test(
                    Z_tr, y_train, Z_te, y_test,
                    head_factory, train_config,
                    pretrain_aug_config=best_pretrain,
                    epoch_aug_config=best_epoch,
                    seed=seed,
                )

            elif cfg["mode"] == "concat":
                Z_tr = np.concatenate([all_Z_train[l] for l in cfg["layers"]], axis=1)
                Z_te = np.concatenate([all_Z_test[l] for l in cfg["layers"]], axis=1)
                concat_dim = Z_tr.shape[1]
                concat_hidden = [min(256, concat_dim // 2), 128]

                def head_factory(cd=concat_dim, ch=concat_hidden):
                    return build_head("mlp", cd, n_classes, hidden_dims=ch, dropout=0.5)

                metrics, _, _ = run_neural_train_test(
                    Z_tr, y_train, Z_te, y_test,
                    head_factory, train_config,
                    pretrain_aug_config=best_pretrain,
                    epoch_aug_config=best_epoch,
                    seed=seed,
                )

            elif cfg["mode"] == "fusion":
                n_layers = len(cfg["layers"])

                def head_factory(nl=n_layers):
                    return build_head(
                        "multi_layer_fusion", embed_dim, n_classes,
                        n_layers=nl, hidden_dims=[64], dropout=0.5,
                    )

                metrics, _, _ = run_neural_train_test_multi_layer(
                    all_Z_train, all_Z_test, cfg["layers"],
                    y_train, y_test, head_factory, train_config, seed,
                )

            elif cfg["mode"] == "attention":
                n_layers = len(cfg["layers"])

                def head_factory(nl=n_layers):
                    return build_head(
                        "attention_pool", embed_dim, n_classes,
                        n_layers=nl, hidden_dims=[64], dropout=0.5,
                    )

                metrics, _, _ = run_neural_train_test_multi_layer(
                    all_Z_train, all_Z_test, cfg["layers"],
                    y_train, y_test, head_factory, train_config, seed,
                )
            else:
                continue

            elapsed = time.time() - t0

            row = _make_result_row(exp_name, metrics, elapsed, extra={
                "group": "B",
                "mode": cfg["mode"],
                "layers": cfg["layers"],
                "n_layers": len(cfg["layers"]),
            })
            results.append(row)
            logger.info(
                "    Acc=%.4f  F1=%.4f  AUC=%s  (%.1fs)",
                metrics.accuracy, metrics.f1,
                f"{metrics.roc_auc:.4f}" if not np.isnan(metrics.roc_auc) else "N/A",
                elapsed,
            )
        except Exception as e:
            logger.error("    FAILED: %s", e)
            results.append({
                "group": "B", "name": exp_name, "error": str(e),
            })

    df = pd.DataFrame(results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "group_b_layer_fusion.csv", index=False)
    logger.info("Saved: %s (%d rows)", tables_dir / "group_b_layer_fusion.csv", len(df))

    return results


# ============================================================================
# Group C: Training Hyperparameters + TTA + Ensemble (42 experiments)
# ============================================================================

# LR × Weight Decay grid (18 configs)
LR_VALUES = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
WD_VALUES = [0.0, 0.01, 0.1]

# TTA configs (6)
TTA_CONFIGS = [
    {"name": "TTA-3 FroFA s=0.05", "tta_k": 3, "tta_strategy": "frofa", "tta_strength": 0.05},
    {"name": "TTA-5 FroFA s=0.1", "tta_k": 5, "tta_strategy": "frofa", "tta_strength": 0.1},
    {"name": "TTA-10 FroFA s=0.1", "tta_k": 10, "tta_strategy": "frofa", "tta_strength": 0.1},
    {"name": "TTA-5 FroFA s=0.2", "tta_k": 5, "tta_strategy": "frofa", "tta_strength": 0.2},
    {"name": "TTA-5 AdaptNoise s=0.1", "tta_k": 5, "tta_strategy": "adaptive_noise", "tta_strength": 0.1},
    {"name": "TTA-10 AdaptNoise s=0.1", "tta_k": 10, "tta_strategy": "adaptive_noise", "tta_strength": 0.1},
]

# Ensemble configs (6)
ENSEMBLE_CONFIGS = [
    {"name": "Ensemble 3-seed", "n_seeds": 3},
    {"name": "Ensemble 5-seed", "n_seeds": 5},
    {"name": "Ensemble 7-seed", "n_seeds": 7},
    {"name": "Ensemble 10-seed", "n_seeds": 10},
    {"name": "Ensemble 5-seed + DC", "n_seeds": 5, "pretrain_aug": {"strategy": "dc", "alpha": 0.5, "n_synthetic": 50}},
    {"name": "Ensemble 5-seed + FroFA", "n_seeds": 5, "epoch_aug": {"strategy": "frofa", "strength": 0.1}},
]

# Label smoothing (3)
LS_VALUES = [0.0, 0.05, 0.1]

# Epoch/patience combos (6)
EP_PATIENCE_CONFIGS = [
    {"name": "100ep/20pat", "epochs": 100, "patience": 20},
    {"name": "200ep/30pat", "epochs": 200, "patience": 30},
    {"name": "300ep/50pat", "epochs": 300, "patience": 50},
    {"name": "500ep/50pat", "epochs": 500, "patience": 50},
    {"name": "200ep/10pat", "epochs": 200, "patience": 10},
    {"name": "100ep/50pat", "epochs": 100, "patience": 50},
]


def run_group_c(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    base_train_config: TrainConfig,
    best_aug: dict,
    best_head_cfg: dict,
    seed: int,
    output_dir: Path,
) -> list[dict]:
    """Sweep training hyperparameters + TTA + ensemble.

    18 LR×WD + 6 TTA + 6 ensemble + 3 label smoothing + 6 epoch/patience
    = ~39 experiments (some overlap, ~42 total with variants).
    """
    logger.info("=" * 70)
    logger.info("GROUP C: Training Hyperparameters + TTA + Ensemble")
    logger.info("=" * 70)

    embed_dim = Z_train.shape[1]
    n_classes = len(set(y_train.tolist()) | set(y_test.tolist()))
    best_pretrain = best_aug.get("pretrain_aug")
    best_epoch = best_aug.get("epoch_aug")
    head_type = best_head_cfg.get("type", "mlp")
    head_kwargs = best_head_cfg.get("kwargs", {"hidden_dims": [64], "dropout": 0.5})

    results = []

    # --- Part 1: LR × Weight Decay grid (18 configs) ---
    logger.info("--- Part 1: LR × Weight Decay grid (18 configs) ---")
    for lr in LR_VALUES:
        for wd in WD_VALUES:
            exp_name = f"LR={lr:.0e} WD={wd}"
            logger.info("  %s", exp_name)
            t0 = time.time()

            try:
                tc = TrainConfig(
                    epochs=base_train_config.epochs,
                    lr=lr,
                    weight_decay=wd,
                    label_smoothing=base_train_config.label_smoothing,
                    early_stopping_patience=base_train_config.early_stopping_patience,
                    device=base_train_config.device,
                )

                def head_factory(ht=head_type, kw=head_kwargs):
                    return build_head(ht, embed_dim, n_classes, **kw)

                metrics, _, _ = run_neural_train_test(
                    Z_train, y_train, Z_test, y_test,
                    head_factory, tc,
                    pretrain_aug_config=best_pretrain,
                    epoch_aug_config=best_epoch,
                    seed=seed,
                )
                elapsed = time.time() - t0

                row = _make_result_row(exp_name, metrics, elapsed, extra={
                    "group": "C", "subgroup": "lr_wd",
                    "lr": lr, "weight_decay": wd,
                })
                results.append(row)
                logger.info(
                    "    Acc=%.4f  F1=%.4f  (%.1fs)",
                    metrics.accuracy, metrics.f1, elapsed,
                )
            except Exception as e:
                logger.error("    FAILED: %s", e)
                results.append({"group": "C", "name": exp_name, "error": str(e)})

    # --- Part 2: TTA configs (6) ---
    logger.info("--- Part 2: TTA configs (6) ---")
    for tta_cfg in TTA_CONFIGS:
        exp_name = tta_cfg["name"]
        logger.info("  %s", exp_name)
        t0 = time.time()

        try:
            def head_factory(ht=head_type, kw=head_kwargs):
                return build_head(ht, embed_dim, n_classes, **kw)

            metrics, _, _ = run_tta_train_test(
                Z_train, y_train, Z_test, y_test,
                head_factory, base_train_config,
                tta_k=tta_cfg["tta_k"],
                tta_strategy=tta_cfg["tta_strategy"],
                tta_strength=tta_cfg["tta_strength"],
                pretrain_aug_config=best_pretrain,
                seed=seed,
            )
            elapsed = time.time() - t0

            row = _make_result_row(exp_name, metrics, elapsed, extra={
                "group": "C", "subgroup": "tta",
                "tta_k": tta_cfg["tta_k"],
                "tta_strategy": tta_cfg["tta_strategy"],
            })
            results.append(row)
            logger.info("    Acc=%.4f  F1=%.4f  (%.1fs)", metrics.accuracy, metrics.f1, elapsed)
        except Exception as e:
            logger.error("    FAILED: %s", e)
            results.append({"group": "C", "name": exp_name, "error": str(e)})

    # --- Part 3: Ensemble configs (6) ---
    logger.info("--- Part 3: Ensemble configs (6) ---")
    for ens_cfg in ENSEMBLE_CONFIGS:
        exp_name = ens_cfg["name"]
        logger.info("  %s", exp_name)
        t0 = time.time()

        try:
            def head_factory(ht=head_type, kw=head_kwargs):
                return build_head(ht, embed_dim, n_classes, **kw)

            metrics, _, _ = run_ensemble_train_test(
                Z_train, y_train, Z_test, y_test,
                head_factory, base_train_config,
                n_seeds=ens_cfg["n_seeds"],
                base_seed=seed,
                pretrain_aug_config=ens_cfg.get("pretrain_aug", best_pretrain),
                epoch_aug_config=ens_cfg.get("epoch_aug", best_epoch),
            )
            elapsed = time.time() - t0

            row = _make_result_row(exp_name, metrics, elapsed, extra={
                "group": "C", "subgroup": "ensemble",
                "n_seeds": ens_cfg["n_seeds"],
            })
            results.append(row)
            logger.info("    Acc=%.4f  F1=%.4f  (%.1fs)", metrics.accuracy, metrics.f1, elapsed)
        except Exception as e:
            logger.error("    FAILED: %s", e)
            results.append({"group": "C", "name": exp_name, "error": str(e)})

    # --- Part 4: Label smoothing (3) ---
    logger.info("--- Part 4: Label smoothing (3) ---")
    for ls in LS_VALUES:
        exp_name = f"LabelSmooth={ls}"
        logger.info("  %s", exp_name)
        t0 = time.time()

        try:
            tc = TrainConfig(
                epochs=base_train_config.epochs,
                lr=base_train_config.lr,
                weight_decay=base_train_config.weight_decay,
                label_smoothing=ls,
                early_stopping_patience=base_train_config.early_stopping_patience,
                device=base_train_config.device,
            )

            def head_factory(ht=head_type, kw=head_kwargs):
                return build_head(ht, embed_dim, n_classes, **kw)

            metrics, _, _ = run_neural_train_test(
                Z_train, y_train, Z_test, y_test,
                head_factory, tc,
                pretrain_aug_config=best_pretrain,
                epoch_aug_config=best_epoch,
                seed=seed,
            )
            elapsed = time.time() - t0

            row = _make_result_row(exp_name, metrics, elapsed, extra={
                "group": "C", "subgroup": "label_smoothing",
                "label_smoothing": ls,
            })
            results.append(row)
            logger.info("    Acc=%.4f  F1=%.4f  (%.1fs)", metrics.accuracy, metrics.f1, elapsed)
        except Exception as e:
            logger.error("    FAILED: %s", e)
            results.append({"group": "C", "name": exp_name, "error": str(e)})

    # --- Part 5: Epoch/patience combos (6) ---
    logger.info("--- Part 5: Epoch/patience combos (6) ---")
    for ep_cfg in EP_PATIENCE_CONFIGS:
        exp_name = ep_cfg["name"]
        logger.info("  %s", exp_name)
        t0 = time.time()

        try:
            tc = TrainConfig(
                epochs=ep_cfg["epochs"],
                lr=base_train_config.lr,
                weight_decay=base_train_config.weight_decay,
                label_smoothing=base_train_config.label_smoothing,
                early_stopping_patience=ep_cfg["patience"],
                device=base_train_config.device,
            )

            def head_factory(ht=head_type, kw=head_kwargs):
                return build_head(ht, embed_dim, n_classes, **kw)

            metrics, _, _ = run_neural_train_test(
                Z_train, y_train, Z_test, y_test,
                head_factory, tc,
                pretrain_aug_config=best_pretrain,
                epoch_aug_config=best_epoch,
                seed=seed,
            )
            elapsed = time.time() - t0

            row = _make_result_row(exp_name, metrics, elapsed, extra={
                "group": "C", "subgroup": "ep_patience",
                "epochs": ep_cfg["epochs"],
                "patience": ep_cfg["patience"],
            })
            results.append(row)
            logger.info("    Acc=%.4f  F1=%.4f  (%.1fs)", metrics.accuracy, metrics.f1, elapsed)
        except Exception as e:
            logger.error("    FAILED: %s", e)
            results.append({"group": "C", "name": exp_name, "error": str(e)})

    # Save
    df = pd.DataFrame(results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "group_c_training_tta.csv", index=False)
    logger.info("Saved: %s (%d rows)", tables_dir / "group_c_training_tta.csv", len(df))

    return results


# ============================================================================
# Summary + Visualization
# ============================================================================

def generate_summary(all_results: list[dict], output_dir: Path):
    """Generate combined ranking and summary report."""
    df = pd.DataFrame(all_results)
    valid = df.dropna(subset=["accuracy"]) if "accuracy" in df.columns else df
    if len(valid) == 0:
        logger.warning("No valid results to summarize")
        return

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
        name = row.get("name", "?")
        acc = row.get("accuracy", 0)
        f1_val = row.get("f1", 0)
        auc_val = row.get("auc", 0) or 0
        logger.info(
            "  #%2d [%s] %s: Acc=%.4f F1=%.4f AUC=%.4f",
            i + 1, group, name, acc, f1_val, auc_val,
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
    with open(reports_dir / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved: %s", reports_dir / "phase2_summary.json")


def plot_group_bar(results: list[dict], title: str, output_path: Path):
    """Horizontal bar chart of accuracy for a group of results."""
    setup_style()
    valid = [r for r in results if "accuracy" in r and r.get("error") is None]
    if not valid:
        return

    valid.sort(key=lambda r: r["accuracy"], reverse=True)
    names = [r["name"] for r in valid[:20]]  # Top 20
    accs = [r["accuracy"] for r in valid[:20]]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.4), max(5, len(names) * 0.3)))
    colors = ["#DE8F05" if i == 0 else "#0173B2" for i in range(len(names))]
    bars = ax.barh(range(len(names)), accs, color=colors)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()

    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=7,
        )

    save_figure(fig, output_path)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Comprehensive Sweep")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--group", choices=["A", "B", "C"], default=None,
                        help="Run specific group only (A, B, or C)")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--split-date", default=None,
                        help="Override train/test split date (YYYY-MM-DD)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Primary transformer layer (default: from config or 3)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: from config or 42)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    if args.split_date is not None:
        cfg["split_date"] = args.split_date

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    output_dir = Path(cfg.get("output_dir", "results/phase2_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_output(formats=["png"], dpi=200)

    # Load data
    split_date = cfg.get("split_date", "2026-02-15")
    default_channels = cfg.get("default_channels", ["d620900d_motionSensor", "408981c2_contactSensor"])

    # Dataset config from Phase 1 best (or config defaults)
    ctx_before = cfg.get("default_context_before", 4)
    ctx_after = cfg.get("default_context_after", 4)

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
        channels=default_channels,
    )

    logger.info("Loading data...")
    sensor_arr, train_labels, test_labels, ch_names, timestamps = (
        load_occupancy_data(prep_cfg, split_date=split_date)
    )

    all_labels = np.where(train_labels >= 0, train_labels, test_labels)
    dataset = OccupancyDataset(sensor_arr, all_labels, timestamps, ds_config)
    train_mask, test_mask = dataset.get_train_test_split(split_date)

    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())
    logger.info("Dataset: %d total, %d train, %d test", len(dataset), n_train, n_test)

    # Model config
    pretrained = cfg["model"]["pretrained_name"]
    output_token = cfg["model"].get("output_token", "combined")
    primary_layer = args.layer if args.layer is not None else cfg.get("default_layer", 3)

    # Training config
    train_raw = cfg.get("training", {})
    base_train_config = TrainConfig(
        epochs=train_raw.get("epochs", 200),
        lr=train_raw.get("lr", 1e-3),
        weight_decay=train_raw.get("weight_decay", 0.01),
        label_smoothing=train_raw.get("label_smoothing", 0.0),
        early_stopping_patience=train_raw.get("early_stopping_patience", 30),
        device=device,
    )

    # Extract embeddings
    groups = [args.group] if args.group else ["A", "B", "C"]

    # Determine which layers to extract
    layers_needed = {primary_layer}
    if "B" in groups:
        layers_needed.update(ALL_LAYERS)

    all_Z = {}
    for l in sorted(layers_needed):
        logger.info("Extracting L%d embeddings...", l)
        trainer = load_mantis_model(pretrained, l, output_token, device)
        Z = extract_embeddings(trainer, dataset, device)
        all_Z[l] = Z
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Split embeddings into train/test
    y_all = dataset.labels
    all_Z_train = {l: Z[train_mask] for l, Z in all_Z.items()}
    all_Z_test = {l: Z[test_mask] for l, Z in all_Z.items()}
    y_train = y_all[train_mask]
    y_test = y_all[test_mask]

    Z_train_primary = all_Z_train[primary_layer]
    Z_test_primary = all_Z_test[primary_layer]

    logger.info("Primary embeddings (L%d): train=%s, test=%s",
                primary_layer, Z_train_primary.shape, Z_test_primary.shape)

    # Default best aug/head (will be updated by Group A results)
    best_aug = {"pretrain_aug": None, "epoch_aug": None}
    best_head_cfg = {"type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.5}}

    t_start = time.time()
    all_results = []

    # Run groups
    for group in groups:
        if group == "A":
            results = run_group_a(
                Z_train_primary, y_train, Z_test_primary, y_test,
                base_train_config, seed, output_dir,
            )
            all_results.extend(results)

            # Update best aug + head from Group A top result
            valid_a = [r for r in results if "accuracy" in r and r.get("error") is None]
            if valid_a:
                valid_a.sort(key=lambda r: -r["accuracy"])
                top = valid_a[0]
                aug_name = top.get("augmentation", "no aug")
                for acfg in AUGMENTATION_CONFIGS:
                    if acfg["name"] == aug_name:
                        best_aug = {"pretrain_aug": acfg["pretrain_aug"], "epoch_aug": acfg["epoch_aug"]}
                        break
                head_name = top.get("head", "MLP[64]-d0.5")
                for hcfg in HEAD_CONFIGS:
                    if hcfg["name"] == head_name:
                        best_head_cfg = {"type": hcfg["type"], "kwargs": hcfg["kwargs"]}
                        break

            try:
                plot_group_bar(results, "Group A: Augmentation × Head", output_dir / "plots" / "group_a_bar")
            except Exception:
                logger.warning("Failed to plot Group A bar", exc_info=True)

        elif group == "B":
            results = run_group_b(
                all_Z_train, all_Z_test, y_train, y_test,
                base_train_config, best_aug, seed, output_dir,
            )
            all_results.extend(results)

            try:
                plot_group_bar(results, "Group B: Layer Selection & Fusion", output_dir / "plots" / "group_b_bar")
            except Exception:
                logger.warning("Failed to plot Group B bar", exc_info=True)

        elif group == "C":
            results = run_group_c(
                Z_train_primary, y_train, Z_test_primary, y_test,
                base_train_config, best_aug, best_head_cfg, seed, output_dir,
            )
            all_results.extend(results)

            try:
                plot_group_bar(results, "Group C: Training + TTA + Ensemble", output_dir / "plots" / "group_c_bar")
            except Exception:
                logger.warning("Failed to plot Group C bar", exc_info=True)

    generate_summary(all_results, output_dir)

    total_time = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("Done! Total time: %.1fs (%.1f min)", total_time, total_time / 60)
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
