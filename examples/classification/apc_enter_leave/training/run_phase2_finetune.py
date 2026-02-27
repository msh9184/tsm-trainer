"""Phase 2: Neural Classification Head Fine-Tuning on Frozen MantisV2 Embeddings.

Trains lightweight neural classification heads on pre-extracted MantisV2
embeddings to beat the Phase 1 zero-shot baseline (88.99% LOOCV accuracy).

5-Stage Sweep:
  Stage 1: Head architecture selection (17 configs)
  Stage 2: Augmentation & regularization (~20 configs)
  Stage 3: Multi-layer fusion (6 configs)
  Stage 4: Sensor combination validation (4 configs)
  Stage 5: Ensemble & final (3 configs)

Key design:
  - MantisV2 backbone frozen -> embeddings extracted ONCE per layer
  - Only classification head trains -> tiny params (< 100K), fast LOOCV
  - N=109 samples -> aggressive regularization (dropout, label smoothing, mixup)
  - Full-batch training (108 samples per fold) -> no mini-batch overhead

Usage:
    cd examples/classification/apc_enter_leave

    # Full 5-stage sweep
    python training/run_phase2_finetune.py \\
        --config training/configs/enter-leave-phase2.yaml

    # Quick: Stage 1 only
    python training/run_phase2_finetune.py --config ... --stages 1

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
    save_results_csv,
    save_results_txt,
)
from data.dataset import EventDataset, build_dataset_config
from evaluation.metrics import (
    EventClassificationMetrics,
    aggregate_cv_predictions,
)
from visualization.style import (
    FIGSIZE_SINGLE,
    setup_style,
    save_figure,
    configure_output,
)
from visualization.curves import plot_confusion_matrix

# Phase 2 modules
from training.heads import build_head
from training.augmentation import apply_augmentation

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


def train_head(
    head: nn.Module,
    Z_train: torch.Tensor,
    y_train: torch.Tensor,
    config: TrainConfig,
    n_classes: int,
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

    Returns
    -------
    nn.Module — trained head (in eval mode).
    """
    device = torch.device(config.device)
    head = head.to(device)
    head.train()

    Z_train = Z_train.to(device)
    y_train = y_train.to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs,
    )

    # Loss function
    use_soft_labels = (
        config.augmentation is not None
        and config.augmentation.get("mixup_alpha", 0) > 0
    )

    if use_soft_labels:
        # Mixup produces soft labels -> use KLDiv or manual CE
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
            Z_aug, y_aug = apply_augmentation(Z_train, y_train, config.augmentation)
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
# Neural LOOCV
# ============================================================================

def run_neural_loocv(
    Z: np.ndarray,
    y: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    class_names: list[str] | None = None,
    seed: int = 42,
) -> EventClassificationMetrics:
    """Leave-One-Out CV with neural classification head.

    For each fold: scaler -> augment -> train head -> predict.
    Aggregate all N predictions for global metrics.

    Parameters
    ----------
    Z : np.ndarray, shape (n, embed_dim)
    y : np.ndarray, shape (n,) int64
    head_factory : callable
        Returns a fresh nn.Module for each fold.
    train_config : TrainConfig
    class_names : list[str], optional
    seed : int

    Returns
    -------
    EventClassificationMetrics
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

        # Train fresh head
        torch.manual_seed(seed)
        head = head_factory()
        head = train_head(head, Z_train_t, y_train_t, train_config, n_classes)

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

    return aggregate_cv_predictions(
        y, y_pred_all, y_prob_binary,
        cv_method="LOOCV", n_folds=n, class_names=class_names,
    )


def run_neural_loocv_multi_layer(
    all_Z: dict[int, np.ndarray],
    layer_indices: list[int],
    y: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    class_names: list[str] | None = None,
    seed: int = 42,
) -> EventClassificationMetrics:
    """LOOCV with multi-layer fusion heads.

    Parameters
    ----------
    all_Z : dict[int, np.ndarray]
        {layer_idx: embeddings array (n, embed_dim)}.
    layer_indices : list[int]
        Which layers to fuse.
    y : np.ndarray, shape (n,)
    head_factory : callable
        Returns fresh multi-layer head (MultiLayerFusionHead or AttentionPoolHead).
    train_config : TrainConfig
    class_names : list[str], optional
    seed : int
    """
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    n_classes = len(np.unique(y))
    n_layers = len(layer_indices)

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

    return aggregate_cv_predictions(
        y, y_pred_all, y_prob_binary,
        cv_method="LOOCV", n_folds=n, class_names=class_names,
    )


def run_neural_loocv_ensemble(
    Z: np.ndarray,
    y: np.ndarray,
    head_factory,
    train_config: TrainConfig,
    n_seeds: int,
    base_seed: int = 42,
    class_names: list[str] | None = None,
) -> EventClassificationMetrics:
    """Multi-seed ensemble LOOCV: average softmax across seeds.

    Parameters
    ----------
    n_seeds : int
        Number of random seeds to ensemble.
    base_seed : int
        Seeds will be base_seed, base_seed+1, ..., base_seed+n_seeds-1.
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

    return aggregate_cv_predictions(
        y, y_pred_all, y_prob_out,
        cv_method=f"LOOCV-Ensemble{n_seeds}", n_folds=n,
        class_names=class_names,
    )


# ============================================================================
# Stage runners
# ============================================================================

def run_stage1_architecture(
    Z: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    train_config: TrainConfig,
    seed: int = 42,
) -> list[dict]:
    """Stage 1: Head architecture selection.

    Tests 17 configurations of LinearHead and MLPHead variants.
    Returns sorted results (best first).
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Head Architecture Selection")
    logger.info("=" * 60)

    embed_dim = Z.shape[1]
    n_classes = len(np.unique(y))

    configs = [
        # 1. Linear baseline
        {"name": "Linear", "type": "linear", "kwargs": {}},
        # 2-4. MLP [64], dropout sweep
        {"name": "MLP[64]-d0.3", "type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.3}},
        {"name": "MLP[64]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.5}},
        {"name": "MLP[64]-d0.7", "type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.7}},
        # 5-7. MLP [128], dropout sweep
        {"name": "MLP[128]-d0.3", "type": "mlp", "kwargs": {"hidden_dims": [128], "dropout": 0.3}},
        {"name": "MLP[128]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [128], "dropout": 0.5}},
        {"name": "MLP[128]-d0.7", "type": "mlp", "kwargs": {"hidden_dims": [128], "dropout": 0.7}},
        # 8-10. MLP [128, 64], dropout sweep
        {"name": "MLP[128,64]-d0.3", "type": "mlp", "kwargs": {"hidden_dims": [128, 64], "dropout": 0.3}},
        {"name": "MLP[128,64]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [128, 64], "dropout": 0.5}},
        {"name": "MLP[128,64]-d0.7", "type": "mlp", "kwargs": {"hidden_dims": [128, 64], "dropout": 0.7}},
        # 11-12. MLP [256, 128], larger capacity
        {"name": "MLP[256,128]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [256, 128], "dropout": 0.5}},
        {"name": "MLP[256,128]-d0.7", "type": "mlp", "kwargs": {"hidden_dims": [256, 128], "dropout": 0.7}},
        # 13-14. No BatchNorm variants
        {"name": "MLP[64]-d0.5-noBN", "type": "mlp", "kwargs": {"hidden_dims": [64], "dropout": 0.5, "use_batchnorm": False}},
        {"name": "MLP[128,64]-d0.5-noBN", "type": "mlp", "kwargs": {"hidden_dims": [128, 64], "dropout": 0.5, "use_batchnorm": False}},
        # 15-16. Small MLP [32]
        {"name": "MLP[32]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [32], "dropout": 0.5}},
        {"name": "MLP[32]-d0.7", "type": "mlp", "kwargs": {"hidden_dims": [32], "dropout": 0.7}},
        # 17. MLP [64, 32]
        {"name": "MLP[64,32]-d0.5", "type": "mlp", "kwargs": {"hidden_dims": [64, 32], "dropout": 0.5}},
    ]

    results = []
    for i, cfg in enumerate(configs):
        logger.info("[%d/%d] %s", i + 1, len(configs), cfg["name"])

        def head_factory(c=cfg):
            return build_head(c["type"], embed_dim, n_classes, **c["kwargs"])

        t0 = time.time()
        metrics = run_neural_loocv(Z, y, head_factory, train_config, class_names, seed)
        elapsed = time.time() - t0

        n_params = sum(p.numel() for p in head_factory().parameters())

        row = {
            "rank": 0,
            "name": cfg["name"],
            "head_type": cfg["type"],
            "accuracy": round(metrics.accuracy, 4),
            "f1_macro": round(metrics.f1_macro, 4),
            "roc_auc": round(metrics.roc_auc, 4) if not np.isnan(metrics.roc_auc) else None,
            "eer": round(metrics.eer, 4) if not np.isnan(metrics.eer) else None,
            "n_params": n_params,
            "time_sec": round(elapsed, 1),
            "config": cfg,
            "metrics": metrics,
        }
        results.append(row)

        logger.info(
            "  Acc=%.4f  F1=%.4f  AUC=%s  params=%d  (%.1fs)",
            metrics.accuracy, metrics.f1_macro,
            f"{metrics.roc_auc:.4f}" if not np.isnan(metrics.roc_auc) else "N/A",
            n_params, elapsed,
        )

    # Sort by accuracy (primary), then F1 (secondary)
    results.sort(key=lambda r: (-r["accuracy"], -r["f1_macro"]))
    for i, r in enumerate(results):
        r["rank"] = i + 1

    logger.info("-" * 40)
    logger.info("Stage 1 Top-3:")
    for r in results[:3]:
        logger.info(
            "  #%d %s: Acc=%.4f, F1=%.4f, AUC=%s",
            r["rank"], r["name"], r["accuracy"], r["f1_macro"],
            f"{r['roc_auc']:.4f}" if r["roc_auc"] else "N/A",
        )

    return results


def run_stage2_augmentation(
    Z: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    top_heads: list[dict],
    base_train_config: TrainConfig,
    seed: int = 42,
) -> list[dict]:
    """Stage 2: Augmentation & regularization sweep.

    Tests augmentation strategies on top-3 heads from Stage 1.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Augmentation & Regularization")
    logger.info("=" * 60)

    embed_dim = Z.shape[1]
    n_classes = len(np.unique(y))

    # Build augmentation configs
    aug_configs = []

    # For top-3 heads: baseline + noise + mixup
    for head_cfg in top_heads[:3]:
        head_name = head_cfg["name"]
        # Baseline (no aug)
        aug_configs.append({
            "label": f"{head_name} | baseline",
            "head_cfg": head_cfg["config"],
            "aug": {},
            "weight_decay": base_train_config.weight_decay,
            "label_smoothing": 0.0,
        })
        # Noise variants
        for sigma in [0.05, 0.1]:
            aug_configs.append({
                "label": f"{head_name} | noise={sigma}",
                "head_cfg": head_cfg["config"],
                "aug": {"gaussian_noise_sigma": sigma},
                "weight_decay": base_train_config.weight_decay,
                "label_smoothing": 0.0,
            })
        # Mixup variants
        for alpha in [0.2, 0.5]:
            aug_configs.append({
                "label": f"{head_name} | mixup={alpha}",
                "head_cfg": head_cfg["config"],
                "aug": {"mixup_alpha": alpha},
                "weight_decay": base_train_config.weight_decay,
                "label_smoothing": 0.0,
            })

    # Top-1 only: combo tests
    top1 = top_heads[0]
    top1_name = top1["name"]
    top1_cfg = top1["config"]

    # Noise + Mixup combo
    aug_configs.append({
        "label": f"{top1_name} | noise=0.05+mixup=0.2",
        "head_cfg": top1_cfg,
        "aug": {"gaussian_noise_sigma": 0.05, "mixup_alpha": 0.2},
        "weight_decay": base_train_config.weight_decay,
        "label_smoothing": 0.0,
    })
    # Label smoothing
    aug_configs.append({
        "label": f"{top1_name} | ls=0.1",
        "head_cfg": top1_cfg,
        "aug": {},
        "weight_decay": base_train_config.weight_decay,
        "label_smoothing": 0.1,
    })
    # Weight decay variants
    for wd in [0.05, 0.1]:
        aug_configs.append({
            "label": f"{top1_name} | wd={wd}",
            "head_cfg": top1_cfg,
            "aug": {},
            "weight_decay": wd,
            "label_smoothing": 0.0,
        })
    # Full combo
    aug_configs.append({
        "label": f"{top1_name} | noise=0.05+mixup=0.2+ls=0.1+wd=0.05",
        "head_cfg": top1_cfg,
        "aug": {"gaussian_noise_sigma": 0.05, "mixup_alpha": 0.2},
        "weight_decay": 0.05,
        "label_smoothing": 0.1,
    })

    results = []
    for i, acfg in enumerate(aug_configs):
        logger.info("[%d/%d] %s", i + 1, len(aug_configs), acfg["label"])

        hcfg = acfg["head_cfg"]

        def head_factory(c=hcfg):
            return build_head(c["type"], embed_dim, n_classes, **c["kwargs"])

        tc = TrainConfig(
            epochs=base_train_config.epochs,
            lr=base_train_config.lr,
            weight_decay=acfg["weight_decay"],
            label_smoothing=acfg["label_smoothing"],
            early_stopping_patience=base_train_config.early_stopping_patience,
            augmentation=acfg["aug"] if acfg["aug"] else None,
            device=base_train_config.device,
        )

        t0 = time.time()
        metrics = run_neural_loocv(Z, y, head_factory, tc, class_names, seed)
        elapsed = time.time() - t0

        row = {
            "rank": 0,
            "label": acfg["label"],
            "accuracy": round(metrics.accuracy, 4),
            "f1_macro": round(metrics.f1_macro, 4),
            "roc_auc": round(metrics.roc_auc, 4) if not np.isnan(metrics.roc_auc) else None,
            "eer": round(metrics.eer, 4) if not np.isnan(metrics.eer) else None,
            "time_sec": round(elapsed, 1),
            "aug_config": acfg,
            "metrics": metrics,
        }
        results.append(row)

        logger.info(
            "  Acc=%.4f  F1=%.4f  AUC=%s  (%.1fs)",
            metrics.accuracy, metrics.f1_macro,
            f"{metrics.roc_auc:.4f}" if not np.isnan(metrics.roc_auc) else "N/A",
            elapsed,
        )

    results.sort(key=lambda r: (-r["accuracy"], -r["f1_macro"]))
    for i, r in enumerate(results):
        r["rank"] = i + 1

    logger.info("-" * 40)
    logger.info("Stage 2 Top-3:")
    for r in results[:3]:
        logger.info("  #%d %s: Acc=%.4f", r["rank"], r["label"], r["accuracy"])

    return results


def run_stage3_layer_fusion(
    all_Z: dict[int, np.ndarray],
    y: np.ndarray,
    class_names: list[str],
    best_head_cfg: dict,
    best_aug_cfg: dict,
    train_config: TrainConfig,
    seed: int = 42,
) -> list[dict]:
    """Stage 3: Multi-layer fusion experiments.

    Tests single best layer vs fusion of multiple layers.
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: Multi-Layer Fusion")
    logger.info("=" * 60)

    n_classes = len(np.unique(y))
    embed_dim = next(iter(all_Z.values())).shape[1]

    # Find best single layer from Stage 1 (default L3)
    best_layer = best_head_cfg.get("best_layer", 3)

    configs = [
        # 1. Best single layer baseline
        {"name": f"Single L{best_layer} (baseline)", "mode": "single",
         "layers": [best_layer]},
        # 2. Fusion all 6 layers
        {"name": "Fusion ALL (L0-L5)", "mode": "fusion",
         "layers": [0, 1, 2, 3, 4, 5]},
        # 3. Fusion top-3 (L2, L3, L4)
        {"name": "Fusion L2+L3+L4", "mode": "fusion",
         "layers": [2, 3, 4]},
        # 4. Concatenation L3+L1 (1024-dim MLP)
        {"name": "Concat L3+L1", "mode": "concat",
         "layers": [3, 1]},
        # 5. AttentionPool all 6
        {"name": "AttentionPool ALL", "mode": "attention",
         "layers": [0, 1, 2, 3, 4, 5]},
        # 6. AttentionPool L2+L3+L4
        {"name": "AttentionPool L2+L3+L4", "mode": "attention",
         "layers": [2, 3, 4]},
    ]

    # Filter configs to only use available layers
    available_layers = set(all_Z.keys())
    configs = [c for c in configs if all(l in available_layers for l in c["layers"])]

    hcfg = best_head_cfg.get("config", {"type": "mlp", "kwargs": {"hidden_dims": [128, 64], "dropout": 0.5}})
    head_kwargs = hcfg.get("kwargs", {})

    results = []
    for i, cfg in enumerate(configs):
        logger.info("[%d/%d] %s", i + 1, len(configs), cfg["name"])

        t0 = time.time()

        if cfg["mode"] == "single":
            # Standard single-layer LOOCV
            Z = all_Z[cfg["layers"][0]]

            def head_factory(c=hcfg):
                return build_head(c["type"], embed_dim, n_classes, **c["kwargs"])

            tc = TrainConfig(
                epochs=train_config.epochs, lr=train_config.lr,
                weight_decay=train_config.weight_decay,
                label_smoothing=train_config.label_smoothing,
                early_stopping_patience=train_config.early_stopping_patience,
                augmentation=best_aug_cfg.get("aug") if best_aug_cfg else None,
                device=train_config.device,
            )
            metrics = run_neural_loocv(Z, y, head_factory, tc, class_names, seed)

        elif cfg["mode"] == "concat":
            # Concatenate layer embeddings -> bigger MLP
            layer_list = cfg["layers"]
            Z_concat = np.concatenate([all_Z[l] for l in layer_list], axis=1)
            concat_dim = Z_concat.shape[1]

            # Use slightly larger hidden dims for larger input
            concat_hidden = [min(256, concat_dim // 2), 128]

            def head_factory(cd=concat_dim, ch=concat_hidden, kw=head_kwargs):
                return build_head(
                    "mlp", cd, n_classes,
                    hidden_dims=ch, dropout=kw.get("dropout", 0.5),
                )

            tc = TrainConfig(
                epochs=train_config.epochs, lr=train_config.lr,
                weight_decay=train_config.weight_decay,
                label_smoothing=train_config.label_smoothing,
                early_stopping_patience=train_config.early_stopping_patience,
                augmentation=best_aug_cfg.get("aug") if best_aug_cfg else None,
                device=train_config.device,
            )
            metrics = run_neural_loocv(
                Z_concat, y, head_factory, tc, class_names, seed,
            )

        elif cfg["mode"] == "fusion":
            # MultiLayerFusionHead
            layer_list = cfg["layers"]
            n_layers = len(layer_list)

            def head_factory(nl=n_layers, kw=head_kwargs):
                return build_head(
                    "multi_layer_fusion", embed_dim, n_classes,
                    n_layers=nl,
                    hidden_dims=kw.get("hidden_dims", [128, 64]),
                    dropout=kw.get("dropout", 0.5),
                )

            tc = TrainConfig(
                epochs=train_config.epochs, lr=train_config.lr,
                weight_decay=train_config.weight_decay,
                label_smoothing=train_config.label_smoothing,
                early_stopping_patience=train_config.early_stopping_patience,
                device=train_config.device,
            )
            metrics = run_neural_loocv_multi_layer(
                all_Z, layer_list, y, head_factory, tc, class_names, seed,
            )

        elif cfg["mode"] == "attention":
            layer_list = cfg["layers"]
            n_layers = len(layer_list)

            def head_factory(nl=n_layers, kw=head_kwargs):
                return build_head(
                    "attention_pool", embed_dim, n_classes,
                    n_layers=nl, n_heads=1,
                    hidden_dims=kw.get("hidden_dims", [128, 64]),
                    dropout=kw.get("dropout", 0.5),
                )

            tc = TrainConfig(
                epochs=train_config.epochs, lr=train_config.lr,
                weight_decay=train_config.weight_decay,
                label_smoothing=train_config.label_smoothing,
                early_stopping_patience=train_config.early_stopping_patience,
                device=train_config.device,
            )
            metrics = run_neural_loocv_multi_layer(
                all_Z, layer_list, y, head_factory, tc, class_names, seed,
            )
        else:
            continue

        elapsed = time.time() - t0

        row = {
            "rank": 0,
            "name": cfg["name"],
            "mode": cfg["mode"],
            "layers": cfg["layers"],
            "accuracy": round(metrics.accuracy, 4),
            "f1_macro": round(metrics.f1_macro, 4),
            "roc_auc": round(metrics.roc_auc, 4) if not np.isnan(metrics.roc_auc) else None,
            "eer": round(metrics.eer, 4) if not np.isnan(metrics.eer) else None,
            "time_sec": round(elapsed, 1),
            "metrics": metrics,
        }
        results.append(row)

        logger.info(
            "  Acc=%.4f  F1=%.4f  AUC=%s  (%.1fs)",
            metrics.accuracy, metrics.f1_macro,
            f"{metrics.roc_auc:.4f}" if not np.isnan(metrics.roc_auc) else "N/A",
            elapsed,
        )

    results.sort(key=lambda r: (-r["accuracy"], -r["f1_macro"]))
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


def run_stage4_sensors(
    raw_cfg: dict,
    best_overall_config: dict,
    train_config: TrainConfig,
    include_none: bool,
    device: str,
    seed: int = 42,
) -> list[dict]:
    """Stage 4: Sensor combination validation.

    Re-extracts embeddings for different sensor combos and runs best config.
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: Sensor Combination Validation")
    logger.info("=" * 60)

    sensor_combos = raw_cfg.get("sensor_combos", {})
    if not sensor_combos:
        logger.warning("No sensor_combos defined in config, skipping Stage 4")
        return []

    model_cfg = raw_cfg.get("model", {})
    pretrained_name = model_cfg.get("pretrained_name", "paris-noah/MantisV2")
    output_token = model_cfg.get("output_token", "combined")
    best_layer = best_overall_config.get("best_layer", 3)
    head_cfg = best_overall_config.get("config", {"type": "mlp", "kwargs": {"hidden_dims": [128, 64], "dropout": 0.5}})

    results = []
    for combo_name, channel_list in sensor_combos.items():
        logger.info("[%s] channels: %s", combo_name, channel_list)

        # Override channels in config
        cfg_copy = copy.deepcopy(raw_cfg)
        cfg_copy["data"]["channels"] = channel_list

        try:
            sensor_array, sensor_ts, event_ts, event_labels, ch_names, class_names = (
                load_data(cfg_copy, include_none=include_none)
            )
        except Exception as e:
            logger.error("Failed to load data for %s: %s", combo_name, e)
            continue

        ds_cfg = build_dataset_config(raw_cfg.get("dataset", {}))
        dataset = EventDataset(sensor_array, sensor_ts, event_ts, event_labels, ds_cfg)

        # Extract embeddings
        network, model = load_mantis_model(pretrained_name, best_layer, output_token, device)
        Z = extract_all_embeddings(model, dataset)
        del model, network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        y = event_labels
        n_classes = len(np.unique(y))
        embed_dim = Z.shape[1]

        def head_factory(c=head_cfg, ed=embed_dim, nc=n_classes):
            return build_head(c["type"], ed, nc, **c["kwargs"])

        aug = best_overall_config.get("aug")
        tc = TrainConfig(
            epochs=train_config.epochs, lr=train_config.lr,
            weight_decay=best_overall_config.get("weight_decay", train_config.weight_decay),
            label_smoothing=best_overall_config.get("label_smoothing", 0.0),
            early_stopping_patience=train_config.early_stopping_patience,
            augmentation=aug if aug else None,
            device=train_config.device,
        )

        t0 = time.time()
        metrics = run_neural_loocv(Z, y, head_factory, tc, class_names, seed)
        elapsed = time.time() - t0

        row = {
            "rank": 0,
            "combo": combo_name,
            "channels": len(channel_list),
            "accuracy": round(metrics.accuracy, 4),
            "f1_macro": round(metrics.f1_macro, 4),
            "roc_auc": round(metrics.roc_auc, 4) if not np.isnan(metrics.roc_auc) else None,
            "eer": round(metrics.eer, 4) if not np.isnan(metrics.eer) else None,
            "time_sec": round(elapsed, 1),
            "metrics": metrics,
        }
        results.append(row)

        logger.info(
            "  %s: Acc=%.4f  F1=%.4f  AUC=%s  (%.1fs)",
            combo_name, metrics.accuracy, metrics.f1_macro,
            f"{metrics.roc_auc:.4f}" if not np.isnan(metrics.roc_auc) else "N/A",
            elapsed,
        )

    results.sort(key=lambda r: (-r["accuracy"], -r["f1_macro"]))
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


def run_stage5_ensemble(
    Z: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    best_overall_config: dict,
    train_config: TrainConfig,
    seed: int = 42,
) -> list[dict]:
    """Stage 5: Multi-seed ensemble evaluation.

    Tests single seed vs 5-seed and 10-seed ensembles.
    """
    logger.info("=" * 60)
    logger.info("STAGE 5: Ensemble & Final")
    logger.info("=" * 60)

    embed_dim = Z.shape[1]
    n_classes = len(np.unique(y))
    head_cfg = best_overall_config.get("config", {"type": "mlp", "kwargs": {"hidden_dims": [128, 64], "dropout": 0.5}})

    def head_factory(c=head_cfg):
        return build_head(c["type"], embed_dim, n_classes, **c["kwargs"])

    aug = best_overall_config.get("aug")
    tc = TrainConfig(
        epochs=train_config.epochs, lr=train_config.lr,
        weight_decay=best_overall_config.get("weight_decay", train_config.weight_decay),
        label_smoothing=best_overall_config.get("label_smoothing", 0.0),
        early_stopping_patience=train_config.early_stopping_patience,
        augmentation=aug if aug else None,
        device=train_config.device,
    )

    ensemble_configs = [
        {"name": "Single seed (seed=42)", "n_seeds": 1},
        {"name": "5-seed ensemble", "n_seeds": 5},
        {"name": "10-seed ensemble", "n_seeds": 10},
    ]

    results = []
    for cfg in ensemble_configs:
        logger.info("[%s]", cfg["name"])

        t0 = time.time()
        if cfg["n_seeds"] == 1:
            metrics = run_neural_loocv(Z, y, head_factory, tc, class_names, seed)
        else:
            metrics = run_neural_loocv_ensemble(
                Z, y, head_factory, tc,
                n_seeds=cfg["n_seeds"], base_seed=seed,
                class_names=class_names,
            )
        elapsed = time.time() - t0

        row = {
            "rank": 0,
            "name": cfg["name"],
            "n_seeds": cfg["n_seeds"],
            "accuracy": round(metrics.accuracy, 4),
            "f1_macro": round(metrics.f1_macro, 4),
            "roc_auc": round(metrics.roc_auc, 4) if not np.isnan(metrics.roc_auc) else None,
            "eer": round(metrics.eer, 4) if not np.isnan(metrics.eer) else None,
            "time_sec": round(elapsed, 1),
            "metrics": metrics,
        }
        results.append(row)

        logger.info(
            "  Acc=%.4f  F1=%.4f  AUC=%s  (%.1fs)",
            metrics.accuracy, metrics.f1_macro,
            f"{metrics.roc_auc:.4f}" if not np.isnan(metrics.roc_auc) else "N/A",
            elapsed,
        )

    results.sort(key=lambda r: (-r["accuracy"], -r["f1_macro"]))
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


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


def plot_phase_comparison(
    phase1_acc: float,
    phase2_acc: float,
    phase1_label: str,
    phase2_label: str,
    output_path: Path,
) -> None:
    """Side-by-side comparison of Phase 1 vs Phase 2 best accuracy."""
    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    labels = [phase1_label, phase2_label]
    values = [phase1_acc, phase2_acc]
    colors = ["#999999", "#DE8F05"]

    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel("Accuracy")
    ax.set_title("Phase 1 (Zero-Shot) vs Phase 2 (Fine-Tuned)")
    ax.set_ylim(0, 1.05)

    save_figure(fig, output_path)
    plt.close(fig)


# ============================================================================
# Result serialization
# ============================================================================

def _serialize_stage_results(results: list[dict], exclude_keys: set | None = None) -> list[dict]:
    """Convert stage results to JSON-serializable format."""
    if exclude_keys is None:
        exclude_keys = {"metrics", "config", "aug_config"}

    serializable = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in exclude_keys}
        serializable.append(row)
    return serializable


# ============================================================================
# Main
# ============================================================================

def run_full_sweep(
    raw_cfg: dict,
    stages: list[int],
    layer: int | None,
    include_none: bool,
    device: str,
    seed: int,
) -> dict:
    """Run the full Phase 2 sweep.

    Parameters
    ----------
    raw_cfg : dict
        Loaded YAML config.
    stages : list[int]
        Which stages to run (1-5).
    layer : int or None
        Specific layer (None = default L3 for stages 1-2, all for stage 3).
    include_none : bool
    device : str
    seed : int

    Returns
    -------
    dict — comprehensive summary.
    """
    output_dir = Path(raw_cfg.get("output_dir", "results/phase2_finetune"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"
    for d in [tables_dir, plots_dir, reports_dir]:
        d.mkdir(exist_ok=True)

    configure_output(formats=["png"], dpi=200)

    # ----- Load data -----
    logger.info("Loading data...")
    sensor_array, sensor_ts, event_ts, event_labels, channel_names, class_names = (
        load_data(raw_cfg, include_none=include_none)
    )

    ds_cfg = build_dataset_config(raw_cfg.get("dataset", {}))
    dataset = EventDataset(sensor_array, sensor_ts, event_ts, event_labels, ds_cfg)

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

    # ----- Default layer for stages 1, 2, 4, 5 -----
    primary_layer = layer if layer is not None else 3  # L3 = Phase 1 best

    # ----- Extract embeddings -----
    # For stages 1-2, 4-5: only primary layer
    # For stage 3: all 6 layers
    layers_needed = {primary_layer}
    if 3 in stages:
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

    # ----- Run stages -----
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

    # Track best config across stages
    best_overall = {
        "accuracy": 0,
        "config": {"type": "mlp", "kwargs": {"hidden_dims": [128, 64], "dropout": 0.5}},
        "best_layer": primary_layer,
        "aug": None,
        "weight_decay": base_train_config.weight_decay,
        "label_smoothing": 0.0,
    }

    # ---- Stage 1 ----
    if 1 in stages:
        s1_results = run_stage1_architecture(
            Z_primary, y, class_names, base_train_config, seed,
        )

        # Save
        s1_csv = [
            {k: v for k, v in r.items() if k not in {"config", "metrics"}}
            for r in s1_results
        ]
        save_results_csv(s1_csv, tables_dir / "stage1_architecture.csv")
        save_results_txt(s1_csv, tables_dir / "stage1_architecture.txt")

        try:
            plot_stage_bar(s1_results, "name", "Stage 1: Head Architecture", plots_dir / "stage1_accuracy_bar")
        except Exception:
            logger.warning("Failed to plot Stage 1 bar", exc_info=True)

        # Update best
        if s1_results:
            top = s1_results[0]
            best_overall["accuracy"] = top["accuracy"]
            best_overall["config"] = top["config"]
            summary["stage1"] = _serialize_stage_results(s1_results)

            # Confusion matrix for best Stage 1
            best_metrics = top["metrics"]
            try:
                fig, _ = plot_confusion_matrix(
                    best_metrics.confusion_matrix,
                    class_names=class_names,
                    output_path=plots_dir / "confusion_matrix_stage1_best",
                )
                plt.close(fig)
            except Exception:
                logger.warning("Failed to plot Stage 1 confusion matrix", exc_info=True)
    else:
        s1_results = []

    # ---- Stage 2 ----
    if 2 in stages:
        top_heads = s1_results[:3] if s1_results else [
            {"name": "MLP[128,64]-d0.5", "config": {"type": "mlp", "kwargs": {"hidden_dims": [128, 64], "dropout": 0.5}}},
        ]

        s2_results = run_stage2_augmentation(
            Z_primary, y, class_names, top_heads, base_train_config, seed,
        )

        s2_csv = [
            {k: v for k, v in r.items() if k not in {"aug_config", "metrics"}}
            for r in s2_results
        ]
        save_results_csv(s2_csv, tables_dir / "stage2_augmentation.csv")
        save_results_txt(s2_csv, tables_dir / "stage2_augmentation.txt")

        try:
            plot_stage_bar(s2_results, "label", "Stage 2: Augmentation", plots_dir / "stage2_augmentation_bar")
        except Exception:
            logger.warning("Failed to plot Stage 2 bar", exc_info=True)

        if s2_results:
            top = s2_results[0]
            if top["accuracy"] >= best_overall["accuracy"]:
                best_overall["accuracy"] = top["accuracy"]
                acfg = top.get("aug_config", {})
                best_overall["aug"] = acfg.get("aug")
                best_overall["weight_decay"] = acfg.get("weight_decay", base_train_config.weight_decay)
                best_overall["label_smoothing"] = acfg.get("label_smoothing", 0.0)
            summary["stage2"] = _serialize_stage_results(s2_results)
    else:
        s2_results = []

    # ---- Stage 3 ----
    if 3 in stages:
        best_aug_cfg = {"aug": best_overall.get("aug")}

        s3_results = run_stage3_layer_fusion(
            all_embeddings, y, class_names,
            best_overall, best_aug_cfg, base_train_config, seed,
        )

        s3_csv = [
            {k: v for k, v in r.items() if k not in {"metrics"}}
            for r in s3_results
        ]
        save_results_csv(s3_csv, tables_dir / "stage3_layer_fusion.csv")
        save_results_txt(s3_csv, tables_dir / "stage3_layer_fusion.txt")

        if s3_results and s3_results[0]["accuracy"] >= best_overall["accuracy"]:
            best_overall["accuracy"] = s3_results[0]["accuracy"]
        summary["stage3"] = _serialize_stage_results(s3_results)
    else:
        s3_results = []

    # ---- Stage 4 ----
    if 4 in stages:
        s4_results = run_stage4_sensors(
            raw_cfg, best_overall, base_train_config,
            include_none, device, seed,
        )

        s4_csv = [
            {k: v for k, v in r.items() if k not in {"metrics"}}
            for r in s4_results
        ]
        save_results_csv(s4_csv, tables_dir / "stage4_sensors.csv")
        save_results_txt(s4_csv, tables_dir / "stage4_sensors.txt")

        if s4_results and s4_results[0]["accuracy"] >= best_overall["accuracy"]:
            best_overall["accuracy"] = s4_results[0]["accuracy"]
        summary["stage4"] = _serialize_stage_results(s4_results)
    else:
        s4_results = []

    # ---- Stage 5 ----
    if 5 in stages:
        s5_results = run_stage5_ensemble(
            Z_primary, y, class_names, best_overall, base_train_config, seed,
        )

        s5_csv = [
            {k: v for k, v in r.items() if k not in {"metrics"}}
            for r in s5_results
        ]
        save_results_csv(s5_csv, tables_dir / "stage5_ensemble.csv")
        save_results_txt(s5_csv, tables_dir / "stage5_ensemble.txt")

        if s5_results and s5_results[0]["accuracy"] >= best_overall["accuracy"]:
            best_overall["accuracy"] = s5_results[0]["accuracy"]
        summary["stage5"] = _serialize_stage_results(s5_results)

        # Best ensemble confusion matrix
        if s5_results:
            best_ens_metrics = s5_results[0]["metrics"]
            try:
                fig, _ = plot_confusion_matrix(
                    best_ens_metrics.confusion_matrix,
                    class_names=class_names,
                    output_path=plots_dir / "confusion_matrix_best",
                )
                plt.close(fig)
            except Exception:
                logger.warning("Failed to plot best confusion matrix", exc_info=True)
    else:
        s5_results = []

    # ----- Phase 1 vs Phase 2 comparison -----
    phase1_baseline = 0.8899  # Phase 1 best: M+C, L3, RF, LOOCV
    best_phase2 = best_overall["accuracy"]

    try:
        plot_phase_comparison(
            phase1_baseline, best_phase2,
            "Phase1 (RF Zero-Shot)", "Phase2 (Neural Fine-Tuned)",
            plots_dir / "phase1_vs_phase2_bar",
        )
    except Exception:
        logger.warning("Failed to plot phase comparison", exc_info=True)

    # ----- Overall best table -----
    all_stage_results = []
    for stage_name, stage_results in [
        ("stage1", s1_results), ("stage2", s2_results),
        ("stage3", s3_results), ("stage4", s4_results),
        ("stage5", s5_results),
    ]:
        for r in stage_results:
            all_stage_results.append({
                "stage": stage_name,
                "name": r.get("name", r.get("label", r.get("combo", "?"))),
                "accuracy": r["accuracy"],
                "f1_macro": r["f1_macro"],
                "roc_auc": r.get("roc_auc"),
            })

    all_stage_results.sort(key=lambda r: -r["accuracy"])
    top_overall = all_stage_results[:10]
    save_results_csv(top_overall, tables_dir / "overall_best.csv")
    save_results_txt(top_overall, tables_dir / "overall_best.txt")

    # ----- Save JSON summary -----
    summary["best_overall"] = {
        "accuracy": best_phase2,
        "phase1_baseline": phase1_baseline,
        "improvement": round(best_phase2 - phase1_baseline, 4),
    }
    summary["top_overall"] = top_overall

    report_path = reports_dir / "summary.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved summary: %s", report_path)

    # ----- Text report -----
    report_lines = [
        "Phase 2: Neural Classification Head Fine-Tuning",
        "=" * 60,
        "",
        f"Events: {n_events} ({', '.join(f'{n}={int((y == i).sum())}' for i, n in enumerate(class_names))})",
        f"Channels: {len(channel_names)} ({', '.join(channel_names)})",
        f"Primary layer: L{primary_layer}",
        f"Device: {device}",
        "",
        f"Phase 1 baseline (RF zero-shot): {phase1_baseline:.4f}",
        f"Phase 2 best (neural fine-tuned): {best_phase2:.4f}",
        f"Improvement: {best_phase2 - phase1_baseline:+.4f}",
        "",
        "Top-10 Overall:",
    ]
    for i, r in enumerate(top_overall):
        report_lines.append(
            f"  #{i + 1} [{r['stage']}] {r['name']}: Acc={r['accuracy']:.4f}"
        )

    report_txt_path = reports_dir / "phase2_report.txt"
    report_txt_path.write_text("\n".join(report_lines) + "\n")
    logger.info("Saved report: %s", report_txt_path)

    # Phase 1 vs Phase 2 comparison text
    comp_lines = [
        "Phase 1 vs Phase 2 Comparison",
        "=" * 50,
        "",
        f"Phase 1 (Zero-Shot):      {phase1_baseline:.4f}  (M+C, L3, RF, LOOCV)",
        f"Phase 2 (Fine-Tuned):     {best_phase2:.4f}",
        f"Absolute improvement:     {best_phase2 - phase1_baseline:+.4f}",
        f"Relative improvement:     {(best_phase2 - phase1_baseline) / phase1_baseline * 100:+.2f}%",
    ]
    comp_path = reports_dir / "phase1_vs_phase2.txt"
    comp_path.write_text("\n".join(comp_lines) + "\n")

    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Neural Classification Head Fine-Tuning",
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
    logger.info("Phase 2: Neural Classification Head Fine-Tuning")
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
    logger.info(
        "Phase 2 best: %.4f (vs Phase 1: %.4f, improvement: %+.4f)",
        best.get("accuracy", 0), best.get("phase1_baseline", 0),
        best.get("improvement", 0),
    )


if __name__ == "__main__":
    main()
