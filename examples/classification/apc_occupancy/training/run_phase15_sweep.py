#!/usr/bin/env python3
"""Phase 1.5 Exhaustive Sweep: MantisV2 Frozen Embeddings + Comprehensive Ablation.

Designed for 6 parallel GPU servers for exhaustive configuration search.
Informed by Phase 1 findings:
  - Context window is dominant: AUC peaks at 241-361min, U-shape beyond
  - SVM >> RF at long contexts: gap grows from ±0.01 at 121min to ±0.08 at 361min
  - M+C+T1 synergy: AUC=0.9196 at 241min RF (vs M+C=0.8391)
  - Best overall: 361min L2 SVM M+C AUC=0.9321
  - Critical gap: M+C+T1 + SVM + 361min NEVER tested together
  - E channel (408981c2_temperatureMeasurement) doesn't exist in data → removed

Six sweep groups (run independently on separate GPUs):

  Group D — Channel × Classifier × Context Cross (~240 experiments)
    - 8 channel combos × 5 classifiers × 6 contexts, L2 fixed
    - Goal: Full cross-product of top channels, classifiers, and contexts

  Group E — Layer × Channel Interaction (~288 experiments)
    - 4 channels × 6 layers × 3 classifiers × 4 contexts
    - Goal: Check if channel importance varies with layer selection

  Group F — Ultra-Fine-Grain Context Resolution (~180 experiments)
    - 30 context points (10min steps, 151-601min) × 3 channels × 2 classifiers
    - Goal: Find exact optimal context window with fine resolution

  Group G — SVM Hyperparameter Deep Tuning (~186 experiments)
    - 31 SVM configs (RBF/Linear/Poly × C × gamma) × 3 channels × 2 contexts
    - Goal: Optimize SVM hyperparameters at best context sizes

  Group H — Multi-Layer Fusion + Per-Channel-Layer (~180 experiments)
    - Part H1: 11 layer fusion combos × 3 channels × 2 contexts × 2 classifiers = 132
    - Part H2: 12 per-channel-layer configs × 2 contexts × 2 classifiers = 48
    - Goal: Test multi-layer information fusion strategies

  Group I — Alternative Classifiers + Special Configs (~294 experiments)
    - Part I1: 8 alt classifiers × 5 channels × 2 contexts = 80
    - Part I2: Extended contexts (481-1441min) × 4 channels × 2 classifiers = 40
    - Part I3: Backward-only mode (9 contexts) × 3 channels × 2 classifiers = 54
    - Part I4: Asymmetric past-heavy (18 configs) × 2 channels × 2 classifiers = 72
    - Part I5: Future-heavy (8 configs) × 3 channels × 2 classifiers = 48
    - Goal: Explore classifiers, context directionality, and edge-case configs

Total: ~1,368 experiments across 6 servers.

Usage:
  cd examples/classification/apc_occupancy

  # Run specific group (one per GPU server)
  python training/run_phase15_sweep.py \\
      --config training/configs/occupancy-phase1.yaml --group D --device cuda
  python training/run_phase15_sweep.py \\
      --config training/configs/occupancy-phase1.yaml --group E --device cuda
  python training/run_phase15_sweep.py \\
      --config training/configs/occupancy-phase1.yaml --group F --device cuda
  python training/run_phase15_sweep.py \\
      --config training/configs/occupancy-phase1.yaml --group G --device cuda
  python training/run_phase15_sweep.py \\
      --config training/configs/occupancy-phase1.yaml --group H --device cuda
  python training/run_phase15_sweep.py \\
      --config training/configs/occupancy-phase1.yaml --group I --device cuda
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Headless matplotlib
import matplotlib as mpl
if not os.environ.get("DISPLAY"):
    mpl.use("Agg")

# Local imports — run from apc_occupancy/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import PreprocessConfig, load_occupancy_data
from data.dataset import DatasetConfig, OccupancyDataset
from evaluation.metrics import compute_metrics, ClassificationMetrics

logger = logging.getLogger(__name__)

# ============================================================================
# Channel Map — verified available channels (E removed: doesn't exist in data)
# ============================================================================

CHANNEL_MAP = {
    "M": "d620900d_motionSensor",
    "C": "408981c2_contactSensor",
    "T1": "d620900d_temperatureMeasurement",
    "T2": "ccea734e_temperatureMeasurement",
    "P": "f2e891c6_powerMeter",
}

ALL_LAYERS = [0, 1, 2, 3, 4, 5]


def _resolve_channels(keys: list[str]) -> list[str]:
    """Convert short keys (M, C, T1, ...) to full channel names."""
    return [CHANNEL_MAP[k] for k in keys]


def _ch_label(keys: list[str]) -> str:
    """Generate human-readable channel label from keys."""
    return "+".join(keys)


# ============================================================================
# Group D: Channel × Classifier × Context Cross  (240 experiments)
# ============================================================================

GROUP_D_CHANNELS = [
    {"name": "M", "keys": ["M"]},
    {"name": "T1", "keys": ["T1"]},
    {"name": "M+C", "keys": ["M", "C"]},
    {"name": "M+T1", "keys": ["M", "T1"]},
    {"name": "T1+C", "keys": ["T1", "C"]},
    {"name": "M+C+T1", "keys": ["M", "C", "T1"]},
    {"name": "M+C+T2", "keys": ["M", "C", "T2"]},
    {"name": "T1+T2", "keys": ["T1", "T2"]},
]

GROUP_D_CLASSIFIERS = [
    {"name": "RF", "config": {"type": "random_forest"}},
    {"name": "SVM_rbf", "config": {"type": "svm", "kernel": "rbf", "C": 1.0}},
    {"name": "SVM_linear", "config": {"type": "svm", "kernel": "linear", "C": 1.0}},
    {"name": "LogReg", "config": {"type": "logistic_regression"}},
    {"name": "GradBoost", "config": {"type": "gradient_boosting"}},
]

GROUP_D_CONTEXTS = [
    {"name": "91min", "before": 45, "after": 45},
    {"name": "121min", "before": 60, "after": 60},
    {"name": "181min", "before": 90, "after": 90},
    {"name": "241min", "before": 120, "after": 120},
    {"name": "301min", "before": 150, "after": 150},
    {"name": "361min", "before": 180, "after": 180},
]

# ============================================================================
# Group E: Layer × Channel Interaction  (288 experiments)
# ============================================================================

GROUP_E_CHANNELS = [
    {"name": "M+C", "keys": ["M", "C"]},
    {"name": "M+C+T1", "keys": ["M", "C", "T1"]},
    {"name": "T1+C", "keys": ["T1", "C"]},
    {"name": "T1", "keys": ["T1"]},
]

GROUP_E_CLASSIFIERS = [
    {"name": "SVM_rbf", "config": {"type": "svm", "kernel": "rbf", "C": 1.0}},
    {"name": "RF", "config": {"type": "random_forest"}},
    {"name": "LogReg", "config": {"type": "logistic_regression"}},
]

GROUP_E_CONTEXTS = [
    {"name": "181min", "before": 90, "after": 90},
    {"name": "241min", "before": 120, "after": 120},
    {"name": "301min", "before": 150, "after": 150},
    {"name": "361min", "before": 180, "after": 180},
]

# ============================================================================
# Group F: Ultra-Fine-Grain Context Resolution  (180 experiments)
# ============================================================================


def _build_finegrain_contexts() -> list[dict]:
    """Build 30 context configs with 10min resolution in 151-401, wider beyond."""
    configs = []
    # 10-min steps: 151 to 401 (26 points)
    for total in range(151, 402, 10):
        half = (total - 1) // 2
        configs.append({"name": f"{total}min", "before": half, "after": half})
    # Wider steps beyond 401
    for total in [441, 481, 541, 601]:
        half = (total - 1) // 2
        configs.append({"name": f"{total}min", "before": half, "after": half})
    return configs


GROUP_F_CONTEXTS = _build_finegrain_contexts()  # 30 points

GROUP_F_CHANNELS = [
    {"name": "M+C", "keys": ["M", "C"]},
    {"name": "M+C+T1", "keys": ["M", "C", "T1"]},
    {"name": "T1+C", "keys": ["T1", "C"]},
]

GROUP_F_CLASSIFIERS = [
    {"name": "SVM_rbf", "config": {"type": "svm", "kernel": "rbf", "C": 1.0}},
    {"name": "RF", "config": {"type": "random_forest"}},
]

# ============================================================================
# Group G: SVM Hyperparameter Deep Tuning  (186 experiments)
# ============================================================================


def _build_svm_configs() -> list[dict]:
    """Build 31 SVM configurations for hyperparameter search."""
    configs = []

    # RBF kernel: C × gamma = 5 × 4 = 20
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        for gamma in [0.0001, 0.001, 0.01, "scale"]:
            g_str = gamma if isinstance(gamma, str) else f"{gamma}"
            configs.append({
                "name": f"RBF_C{C}_g{g_str}",
                "config": {"type": "svm", "kernel": "rbf", "C": C, "gamma": gamma},
            })

    # Linear kernel: C = 5
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        configs.append({
            "name": f"Linear_C{C}",
            "config": {"type": "svm", "kernel": "linear", "C": C},
        })

    # Poly kernel degree 2: C = 3
    for C in [0.1, 1.0, 10.0]:
        configs.append({
            "name": f"Poly2_C{C}",
            "config": {"type": "svm", "kernel": "poly", "C": C, "degree": 2},
        })

    # Poly kernel degree 3: C = 3
    for C in [0.1, 1.0, 10.0]:
        configs.append({
            "name": f"Poly3_C{C}",
            "config": {"type": "svm", "kernel": "poly", "C": C, "degree": 3},
        })

    return configs  # 20 + 5 + 3 + 3 = 31


GROUP_G_SVM_CONFIGS = _build_svm_configs()

GROUP_G_CHANNELS = [
    {"name": "M+C", "keys": ["M", "C"]},
    {"name": "M+C+T1", "keys": ["M", "C", "T1"]},
    {"name": "T1+C", "keys": ["T1", "C"]},
]

GROUP_G_CONTEXTS = [
    {"name": "241min", "before": 120, "after": 120},
    {"name": "361min", "before": 180, "after": 180},
]

# ============================================================================
# Group H: Multi-Layer Fusion + Per-Channel-Layer  (180 experiments)
# ============================================================================

# H1: Multi-layer fusion combos (11)
MULTILAYER_COMBOS = [
    {"name": "L0+L1", "layers": [0, 1]},
    {"name": "L0+L2", "layers": [0, 2]},
    {"name": "L0+L3", "layers": [0, 3]},
    {"name": "L0+L5", "layers": [0, 5]},
    {"name": "L1+L2", "layers": [1, 2]},
    {"name": "L2+L3", "layers": [2, 3]},
    {"name": "L2+L5", "layers": [2, 5]},
    {"name": "L0+L2+L5", "layers": [0, 2, 5]},
    {"name": "L0+L3+L5", "layers": [0, 3, 5]},
    {"name": "L1+L2+L3", "layers": [1, 2, 3]},
    {"name": "L0-L5_all", "layers": [0, 1, 2, 3, 4, 5]},
]

GROUP_H_CHANNELS = [
    {"name": "M+C", "keys": ["M", "C"]},
    {"name": "M+C+T1", "keys": ["M", "C", "T1"]},
    {"name": "T1+C", "keys": ["T1", "C"]},
]

GROUP_H_CONTEXTS = [
    {"name": "241min", "before": 120, "after": 120},
    {"name": "361min", "before": 180, "after": 180},
]

GROUP_H_CLASSIFIERS = [
    {"name": "SVM_rbf", "config": {"type": "svm", "kernel": "rbf", "C": 1.0}},
    {"name": "RF", "config": {"type": "random_forest"}},
]

# H2: Per-channel-layer configs — each channel extracted at a different layer (12)
#   channel_keys: channels to use (order matters)
#   layers: layer for each channel (same index correspondence)
PERCHANNEL_CONFIGS = [
    # M+C: 6 configs (test if M and C benefit from different layers)
    {"name": "M@L0+C@L2", "channel_keys": ["M", "C"], "layers": [0, 2]},
    {"name": "M@L0+C@L3", "channel_keys": ["M", "C"], "layers": [0, 3]},
    {"name": "M@L2+C@L0", "channel_keys": ["M", "C"], "layers": [2, 0]},
    {"name": "M@L2+C@L3", "channel_keys": ["M", "C"], "layers": [2, 3]},
    {"name": "M@L3+C@L0", "channel_keys": ["M", "C"], "layers": [3, 0]},
    {"name": "M@L3+C@L2", "channel_keys": ["M", "C"], "layers": [3, 2]},
    # M+C+T1: 6 configs (test best layer per channel)
    {"name": "M@L0+C@L2+T1@L0", "channel_keys": ["M", "C", "T1"], "layers": [0, 2, 0]},
    {"name": "M@L0+C@L0+T1@L2", "channel_keys": ["M", "C", "T1"], "layers": [0, 0, 2]},
    {"name": "M@L2+C@L0+T1@L0", "channel_keys": ["M", "C", "T1"], "layers": [2, 0, 0]},
    {"name": "M@L2+C@L2+T1@L0", "channel_keys": ["M", "C", "T1"], "layers": [2, 2, 0]},
    {"name": "M@L0+C@L2+T1@L2", "channel_keys": ["M", "C", "T1"], "layers": [0, 2, 2]},
    {"name": "M@L2+C@L0+T1@L2", "channel_keys": ["M", "C", "T1"], "layers": [2, 0, 2]},
]

# ============================================================================
# Group I: Alternative Classifiers + Special Configs  (192 experiments)
# ============================================================================

# I1: 8 alternative classifiers × 5 channels × 2 contexts = 80
GROUP_I1_CLASSIFIERS = [
    {"name": "GradBoost", "config": {"type": "gradient_boosting"}},
    {"name": "ExtraTrees", "config": {"type": "extra_trees"}},
    {"name": "AdaBoost", "config": {"type": "adaboost"}},
    {"name": "kNN_3", "config": {"type": "knn", "n_neighbors": 3}},
    {"name": "kNN_5", "config": {"type": "knn", "n_neighbors": 5}},
    {"name": "kNN_7", "config": {"type": "knn", "n_neighbors": 7}},
    {"name": "Ridge", "config": {"type": "ridge"}},
    {"name": "LDA", "config": {"type": "lda"}},
]

GROUP_I1_CHANNELS = [
    {"name": "M+C", "keys": ["M", "C"]},
    {"name": "M+C+T1", "keys": ["M", "C", "T1"]},
    {"name": "T1+C", "keys": ["T1", "C"]},
    {"name": "T1", "keys": ["T1"]},
    {"name": "M+C+P", "keys": ["M", "C", "P"]},
]

GROUP_I1_CONTEXTS = [
    {"name": "241min", "before": 120, "after": 120},
    {"name": "361min", "before": 180, "after": 180},
]

# I2: Extended context (very long) — 5 contexts × 4 channels × 2 classifiers = 40
GROUP_I2_CONTEXTS = [
    {"name": "481min", "before": 240, "after": 240},
    {"name": "601min", "before": 300, "after": 300},
    {"name": "721min", "before": 360, "after": 360},
    {"name": "961min", "before": 480, "after": 480},
    {"name": "1441min", "before": 720, "after": 720},
]

GROUP_I2_CHANNELS = [
    {"name": "M+C", "keys": ["M", "C"]},
    {"name": "M+C+T1", "keys": ["M", "C", "T1"]},
    {"name": "T1+C", "keys": ["T1", "C"]},
    {"name": "T1", "keys": ["T1"]},
]

GROUP_I2_CLASSIFIERS = [
    {"name": "SVM_rbf", "config": {"type": "svm", "kernel": "rbf", "C": 1.0}},
    {"name": "RF", "config": {"type": "random_forest"}},
]

# I3: Backward-only mode — 9 contexts × 3 channels × 2 classifiers = 54
GROUP_I3_CONTEXTS = [
    {"name": "bw_91min", "before": 90, "after": 0},
    {"name": "bw_121min", "before": 120, "after": 0},
    {"name": "bw_181min", "before": 180, "after": 0},
    {"name": "bw_241min", "before": 240, "after": 0},
    {"name": "bw_361min", "before": 360, "after": 0},
    {"name": "bw_481min", "before": 480, "after": 0},
    {"name": "bw_721min", "before": 720, "after": 0},
    {"name": "bw_961min", "before": 960, "after": 0},
    {"name": "bw_1441min", "before": 1440, "after": 0},
]

GROUP_I3_CHANNELS = [
    {"name": "M+C", "keys": ["M", "C"]},
    {"name": "M+C+T1", "keys": ["M", "C", "T1"]},
    {"name": "T1+C", "keys": ["T1", "C"]},
]

GROUP_I3_CLASSIFIERS = [
    {"name": "SVM_rbf", "config": {"type": "svm", "kernel": "rbf", "C": 1.0}},
    {"name": "RF", "config": {"type": "random_forest"}},
]

# I4: Asymmetric (past-heavy) — 18 configs × 2 channels × 2 classifiers = 72
GROUP_I4_CONTEXTS = [
    # Moderate asymmetry
    {"name": "asym_120p+5f", "before": 120, "after": 5},
    {"name": "asym_180p+60f", "before": 180, "after": 60},
    {"name": "asym_240p+30f", "before": 240, "after": 30},
    {"name": "asym_240p+60f", "before": 240, "after": 60},
    {"name": "asym_240p+120f", "before": 240, "after": 120},
    {"name": "asym_300p+60f", "before": 300, "after": 60},
    # Strong past-heavy
    {"name": "asym_360p+5f", "before": 360, "after": 5},
    {"name": "asym_360p+10f", "before": 360, "after": 10},
    {"name": "asym_360p+30f", "before": 360, "after": 30},
    {"name": "asym_360p+60f", "before": 360, "after": 60},
    {"name": "asym_360p+120f", "before": 360, "after": 120},
    # Large windows
    {"name": "asym_480p+10f", "before": 480, "after": 10},
    {"name": "asym_480p+30f", "before": 480, "after": 30},
    {"name": "asym_480p+60f", "before": 480, "after": 60},
    {"name": "asym_480p+120f", "before": 480, "after": 120},
    {"name": "asym_720p+60f", "before": 720, "after": 60},
    {"name": "asym_720p+120f", "before": 720, "after": 120},
    {"name": "asym_960p+60f", "before": 960, "after": 60},
]

GROUP_I4_CHANNELS = [
    {"name": "M+C", "keys": ["M", "C"]},
    {"name": "M+C+T1", "keys": ["M", "C", "T1"]},
]

GROUP_I4_CLASSIFIERS = [
    {"name": "SVM_rbf", "config": {"type": "svm", "kernel": "rbf", "C": 1.0}},
    {"name": "RF", "config": {"type": "random_forest"}},
]

# I5: Future-heavy (reversed asymmetry) — 8 configs × 3 channels × 2 classifiers = 48
# Tests whether future context is more informative than past for occupancy
GROUP_I5_CONTEXTS = [
    {"name": "fh_5p+120f", "before": 5, "after": 120},
    {"name": "fh_10p+240f", "before": 10, "after": 240},
    {"name": "fh_30p+180f", "before": 30, "after": 180},
    {"name": "fh_30p+360f", "before": 30, "after": 360},
    {"name": "fh_60p+180f", "before": 60, "after": 180},
    {"name": "fh_60p+360f", "before": 60, "after": 360},
    {"name": "fh_120p+360f", "before": 120, "after": 360},
    {"name": "fh_120p+720f", "before": 120, "after": 720},
]

GROUP_I5_CHANNELS = [
    {"name": "M+C", "keys": ["M", "C"]},
    {"name": "M+C+T1", "keys": ["M", "C", "T1"]},
    {"name": "T1+C", "keys": ["T1", "C"]},
]

GROUP_I5_CLASSIFIERS = [
    {"name": "SVM_rbf", "config": {"type": "svm", "kernel": "rbf", "C": 1.0}},
    {"name": "RF", "config": {"type": "random_forest"}},
]


# ============================================================================
# Model + Embedding Utilities
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
    """Extract frozen embeddings for all windows (Channel Independence mode).

    Returns shape (n_windows, n_channels * embed_dim).
    """
    X, _ = dataset.get_numpy_arrays()  # (N, C, L)
    n_samples, n_channels, seq_len = X.shape

    all_embeddings = []
    for ch in range(n_channels):
        X_ch = X[:, [ch], :]  # (N, 1, L)
        Z_ch = model.transform(X_ch)  # (N, D)
        all_embeddings.append(Z_ch)

    Z = np.concatenate(all_embeddings, axis=-1)  # (N, C*D)

    if np.isnan(Z).any():
        n_nan = np.isnan(Z).sum()
        logger.warning("Found %d NaN values in embeddings, replacing with 0", n_nan)
        Z = np.nan_to_num(Z, nan=0.0)

    return Z


def extract_multilayer_embeddings(
    pretrained: str,
    layers: list[int],
    output_token: str,
    dataset: OccupancyDataset,
    device: str,
) -> np.ndarray:
    """Extract and concatenate embeddings from multiple MantisV2 layers.

    Each layer produces (N, C*D) → concatenated to (N, len(layers)*C*D).
    """
    all_Z = []
    for layer in layers:
        model = load_mantis_model(pretrained, layer, output_token, device)
        Z_layer = extract_embeddings(model, dataset, device)  # (N, C*D)
        all_Z.append(Z_layer)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(all_Z, axis=-1)


def extract_perchannel_layer_embeddings(
    pretrained: str,
    channel_layers: list[int],
    output_token: str,
    dataset: OccupancyDataset,
    device: str,
) -> np.ndarray:
    """Extract embeddings where each channel uses a specific MantisV2 layer.

    channel_layers[i] = layer number for the i-th channel in the dataset.
    Returns shape (N, n_channels * embed_dim_per_channel).
    """
    X, _ = dataset.get_numpy_arrays()  # (N, C, L)
    n_samples, n_channels, seq_len = X.shape

    if len(channel_layers) != n_channels:
        raise ValueError(
            f"channel_layers has {len(channel_layers)} entries "
            f"but dataset has {n_channels} channels"
        )

    all_embeddings = []
    for ch_idx in range(n_channels):
        layer = channel_layers[ch_idx]
        model = load_mantis_model(pretrained, layer, output_token, device)
        X_ch = X[:, [ch_idx], :]  # (N, 1, L)
        Z_ch = model.transform(X_ch)  # (N, D)
        all_embeddings.append(Z_ch)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    Z = np.concatenate(all_embeddings, axis=-1)

    if np.isnan(Z).any():
        n_nan = np.isnan(Z).sum()
        logger.warning("Found %d NaN values, replacing with 0", n_nan)
        Z = np.nan_to_num(Z, nan=0.0)

    return Z


# ============================================================================
# Classifier Builder (Extended)
# ============================================================================

def build_classifier_v2(config: dict, seed: int = 42):
    """Build sklearn classifier from config dict.

    Supports: random_forest, svm (rbf/linear/poly), logistic_regression,
    gradient_boosting, extra_trees, adaboost, knn, ridge, lda, nearest_centroid.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import (
        AdaBoostClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
    from sklearn.svm import SVC

    clf_type = config["type"]

    if clf_type == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    elif clf_type == "svm":
        kernel = config.get("kernel", "rbf")
        C = config.get("C", 1.0)
        gamma = config.get("gamma", "scale")
        degree = config.get("degree", 3)
        return SVC(
            kernel=kernel, C=C, gamma=gamma, degree=degree,
            probability=True, random_state=seed,
        )
    elif clf_type == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=seed, n_jobs=-1)
    elif clf_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=seed,
        )
    elif clf_type == "extra_trees":
        return ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    elif clf_type == "adaboost":
        return AdaBoostClassifier(n_estimators=100, random_state=seed)
    elif clf_type == "knn":
        n_neighbors = config.get("n_neighbors", 5)
        return KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    elif clf_type == "ridge":
        from sklearn.calibration import CalibratedClassifierCV
        base = RidgeClassifier(alpha=1.0)
        return CalibratedClassifierCV(base, cv=3, method="sigmoid")
    elif clf_type == "lda":
        return LinearDiscriminantAnalysis()
    elif clf_type == "nearest_centroid":
        return NearestCentroid()
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")


# ============================================================================
# Evaluation Utilities
# ============================================================================

def _extract_prob_positive(clf, X_test) -> np.ndarray | None:
    """Extract P(class=1) from classifier, handling edge cases.

    Falls back to decision_function for classifiers without predict_proba.
    Returns None if probability extraction fails (AUC/EER will be NaN).
    """
    # Check single-class: if model only saw one class, AUC is undefined
    if hasattr(clf, "classes_") and len(clf.classes_) < 2:
        logger.debug(
            "Single-class classifier (classes=%s), returning None for probs",
            clf.classes_,
        )
        return None

    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(X_test)
            if hasattr(clf, "classes_"):
                classes = list(clf.classes_)
                if 1 in classes:
                    return proba[:, classes.index(1)]
                # Class 1 not seen → all probability is 0
                logger.warning(
                    "Class 1 not in clf.classes_=%s, returning zeros", classes,
                )
                return np.zeros(len(X_test))
            return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        except Exception as e:
            logger.debug("predict_proba failed: %s", e)

    if hasattr(clf, "decision_function"):
        try:
            scores = clf.decision_function(X_test)
            if scores.ndim == 1:
                return scores
            return scores[:, 1] if scores.shape[1] > 1 else scores[:, 0]
        except Exception as e:
            logger.debug("decision_function failed: %s", e)

    return None


def run_experiment(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    clf_config: dict,
    seed: int = 42,
) -> ClassificationMetrics:
    """Standard scaling + classifier training + evaluation."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_test_s = scaler.transform(Z_test)

    clf = build_classifier_v2(clf_config, seed)
    clf.fit(Z_train_s, y_train)

    y_pred = clf.predict(Z_test_s)
    y_prob = _extract_prob_positive(clf, Z_test_s)

    return compute_metrics(y_test, y_pred, y_prob, class_names=["Empty", "Occupied"])


def _make_result_row(
    group: str,
    subgroup: str,
    exp_name: str,
    ch_name: str,
    clf_name: str,
    layer_info: str,
    ctx_name: str,
    ctx_before: int,
    ctx_after: int,
    metrics: ClassificationMetrics,
    n_train: int,
    n_test: int,
    embed_dim: int,
    elapsed: float,
    **extra,
) -> dict:
    """Build a standardized result row dict."""
    def _safe_nan(v):
        """Convert NaN to None for JSON-safe output."""
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    row = {
        "group": group,
        "subgroup": subgroup,
        "experiment": exp_name,
        "channels": ch_name,
        "classifier": clf_name,
        "layer": layer_info,
        "context_name": ctx_name,
        "context_before": ctx_before,
        "context_after": ctx_after,
        "total_context_min": ctx_before + 1 + ctx_after,
        # Primary metrics (threshold-independent)
        "auc": _safe_nan(metrics.roc_auc),
        "eer": _safe_nan(metrics.eer),
        "eer_threshold": _safe_nan(metrics.eer_threshold),
        # Secondary metrics
        "accuracy": metrics.accuracy,
        "f1": metrics.f1,
        "f1_macro": metrics.f1_macro,
        "f1_weighted": metrics.f1_weighted,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "precision_macro": metrics.precision_macro,
        "recall_macro": metrics.recall_macro,
        # Confidence interval
        "ci_lower": metrics.ci_lower,
        "ci_upper": metrics.ci_upper,
        # Metadata
        "n_train": n_train,
        "n_test": n_test,
        "embed_dim": embed_dim,
        "time_s": round(elapsed, 1),
    }
    row.update(extra)
    return row


def _format_metrics_log(
    idx: int, total: int, label: str,
    metrics: ClassificationMetrics, elapsed: float,
    extra: str = "",
) -> str:
    """Format per-experiment log line with all key metrics.

    Output format (2 lines):
      [  1/240] M | 91min | RF                                          (1.9s)
               AUC=0.6841 EER=0.3159 | Acc=0.6773 F1=0.6512 P=0.6800 R=0.6234 | CI=[0.665,0.690]
    """
    def _v(val, fmt=".4f"):
        if isinstance(val, float) and np.isnan(val):
            return "  N/A "
        return f"{val:{fmt}}"

    auc = _v(metrics.roc_auc)
    eer = _v(metrics.eer)
    acc = _v(metrics.accuracy)
    f1 = _v(metrics.f1)
    prec = _v(metrics.precision)
    rec = _v(metrics.recall)

    ci_str = ""
    if metrics.ci_lower is not None and metrics.ci_upper is not None:
        ci_str = f" | CI=[{metrics.ci_lower:.3f},{metrics.ci_upper:.3f}]"

    extra_str = f" {extra}" if extra else ""
    line1 = f"  [{idx:3d}/{total}] {label}{extra_str}  ({elapsed:.1f}s)"
    line2 = (
        f"           AUC={auc} EER={eer}"
        f" | Acc={acc} F1={f1} P={prec} R={rec}{ci_str}"
    )
    return f"{line1}\n{line2}"


def _cleanup_gpu():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Data Loading Helper
# ============================================================================

def _load_sensor_data(
    cfg: dict,
    channel_keys: list[str],
    split_date: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Load sensor data for given channels, return arrays + metadata."""
    channels = _resolve_channels(channel_keys)
    prep_cfg = PreprocessConfig(
        sensor_csv=cfg["data"]["sensor_csv"],
        label_csv=cfg["data"]["label_csv"],
        label_format="events",
        initial_occupancy=cfg["data"].get("initial_occupancy", 0),
        binarize=True,
        channels=channels,
    )
    return load_occupancy_data(prep_cfg, split_date=split_date)


def _build_dataset_and_split(
    sensor_arr: np.ndarray,
    all_labels: np.ndarray,
    timestamps: pd.DatetimeIndex,
    ctx_before: int,
    ctx_after: int,
    stride: int,
    split_date: str,
) -> tuple[OccupancyDataset, np.ndarray, np.ndarray]:
    """Build sliding window dataset and return train/test split masks."""
    ds_config = DatasetConfig(
        context_mode="bidirectional",
        context_before=ctx_before,
        context_after=ctx_after,
        stride=stride,
    )
    dataset = OccupancyDataset(sensor_arr, all_labels, timestamps, ds_config)
    train_mask, test_mask = dataset.get_train_test_split(split_date)
    return dataset, train_mask, test_mask


# ============================================================================
# Group D: Channel × Classifier × Context Cross
# ============================================================================

def run_group_d(cfg: dict, device: str, output_dir: Path) -> list[dict]:
    """8 channels × 5 classifiers × 6 contexts = 240 experiments.

    Fixed: L2, combined token, bidirectional.
    Model loaded once (L2 only).
    """
    n_total = len(GROUP_D_CHANNELS) * len(GROUP_D_CLASSIFIERS) * len(GROUP_D_CONTEXTS)
    logger.info("=" * 70)
    logger.info("GROUP D: Channel × Classifier × Context Cross (%d experiments)", n_total)
    logger.info("=" * 70)

    results = []
    split_date = cfg.get("split_date", "2026-02-15")
    pretrained = cfg["model"]["pretrained_name"]
    seed = cfg.get("seed", 42)
    stride = cfg.get("stride", 1)
    layer = 2

    # Load model once (L2)
    model = load_mantis_model(pretrained, layer, "combined", device)
    exp_idx = 0

    for ch_cfg in GROUP_D_CHANNELS:
        ch_name = ch_cfg["name"]
        ch_keys = ch_cfg["keys"]
        logger.info("=== Channels: %s ===", ch_name)

        # Load sensor data once per channel combo
        try:
            sensor_arr, train_labels, test_labels, ch_names_loaded, timestamps = (
                _load_sensor_data(cfg, ch_keys, split_date)
            )
        except Exception as e:
            logger.error("Data load failed for %s: %s", ch_name, e)
            for ctx_cfg in GROUP_D_CONTEXTS:
                for clf_cfg in GROUP_D_CLASSIFIERS:
                    exp_idx += 1
                    results.append({"group": "D", "channels": ch_name, "error": str(e)})
            continue

        all_labels = np.where(train_labels >= 0, train_labels, test_labels)

        for ctx_cfg in GROUP_D_CONTEXTS:
            ctx_name = ctx_cfg["name"]
            ctx_before = ctx_cfg["before"]
            ctx_after = ctx_cfg["after"]

            try:
                dataset, train_mask, test_mask = _build_dataset_and_split(
                    sensor_arr, all_labels, timestamps,
                    ctx_before, ctx_after, stride, split_date,
                )
            except Exception as e:
                logger.error("Dataset build failed for %s|%s: %s", ch_name, ctx_name, e)
                for clf_cfg in GROUP_D_CLASSIFIERS:
                    exp_idx += 1
                    results.append({
                        "group": "D", "channels": ch_name, "context_name": ctx_name,
                        "error": str(e),
                    })
                continue

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                logger.warning("Skipping %s|%s: empty split", ch_name, ctx_name)
                continue

            # Extract embeddings (reusing model)
            Z = extract_embeddings(model, dataset, device)
            Z_train = Z[train_mask]
            y_train = dataset.labels[train_mask]
            Z_test = Z[test_mask]
            y_test = dataset.labels[test_mask]

            for clf_cfg in GROUP_D_CLASSIFIERS:
                clf_name = clf_cfg["name"]
                exp_idx += 1
                exp_label = f"{ch_name} | {ctx_name} | {clf_name}"
                t0 = time.time()

                try:
                    metrics = run_experiment(
                        Z_train, y_train, Z_test, y_test, clf_cfg["config"], seed,
                    )
                    elapsed = time.time() - t0

                    row = _make_result_row(
                        group="D", subgroup="ch_clf_ctx",
                        exp_name=exp_label, ch_name=ch_name, clf_name=clf_name,
                        layer_info=f"L{layer}", ctx_name=ctx_name,
                        ctx_before=ctx_before, ctx_after=ctx_after,
                        metrics=metrics, n_train=int(train_mask.sum()),
                        n_test=int(test_mask.sum()), embed_dim=Z.shape[1],
                        elapsed=elapsed,
                    )
                    results.append(row)
                    logger.info(
                        _format_metrics_log(exp_idx, n_total, exp_label, metrics, elapsed),
                    )
                except Exception as e:
                    logger.error("  [%d/%d] %s FAILED: %s", exp_idx, n_total, exp_label, e)
                    results.append({
                        "group": "D", "channels": ch_name, "classifier": clf_name,
                        "context_name": ctx_name, "error": str(e),
                    })

    del model
    _cleanup_gpu()

    # Save
    df = pd.DataFrame(results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "group_d_channel_clf_context.csv", index=False)
    logger.info("Saved: %s (%d rows)", tables_dir / "group_d_channel_clf_context.csv", len(df))

    return results


# ============================================================================
# Group E: Layer × Channel Interaction
# ============================================================================

def run_group_e(cfg: dict, device: str, output_dir: Path) -> list[dict]:
    """4 channels × 6 layers × 3 classifiers × 4 contexts = 288 experiments.

    Model loaded per layer (6 loads, reused across channels/classifiers/contexts).
    """
    n_total = (
        len(GROUP_E_CHANNELS) * len(ALL_LAYERS)
        * len(GROUP_E_CLASSIFIERS) * len(GROUP_E_CONTEXTS)
    )
    logger.info("=" * 70)
    logger.info("GROUP E: Layer × Channel Interaction (%d experiments)", n_total)
    logger.info("=" * 70)

    results = []
    split_date = cfg.get("split_date", "2026-02-15")
    pretrained = cfg["model"]["pretrained_name"]
    seed = cfg.get("seed", 42)
    stride = cfg.get("stride", 1)
    exp_idx = 0

    for layer in ALL_LAYERS:
        logger.info("=== Layer L%d ===", layer)
        model = load_mantis_model(pretrained, layer, "combined", device)

        for ch_cfg in GROUP_E_CHANNELS:
            ch_name = ch_cfg["name"]
            ch_keys = ch_cfg["keys"]

            try:
                sensor_arr, train_labels, test_labels, _, timestamps = (
                    _load_sensor_data(cfg, ch_keys, split_date)
                )
            except Exception as e:
                logger.error("Data load failed for L%d|%s: %s", layer, ch_name, e)
                skip_count = len(GROUP_E_CLASSIFIERS) * len(GROUP_E_CONTEXTS)
                for _ in range(skip_count):
                    exp_idx += 1
                    results.append({
                        "group": "E", "layer": layer, "channels": ch_name,
                        "error": str(e),
                    })
                continue

            all_labels = np.where(train_labels >= 0, train_labels, test_labels)

            for ctx_cfg in GROUP_E_CONTEXTS:
                ctx_name = ctx_cfg["name"]
                ctx_before = ctx_cfg["before"]
                ctx_after = ctx_cfg["after"]

                try:
                    dataset, train_mask, test_mask = _build_dataset_and_split(
                        sensor_arr, all_labels, timestamps,
                        ctx_before, ctx_after, stride, split_date,
                    )
                except Exception as e:
                    logger.error("Dataset failed L%d|%s|%s: %s", layer, ch_name, ctx_name, e)
                    for clf_cfg in GROUP_E_CLASSIFIERS:
                        exp_idx += 1
                        results.append({
                            "group": "E", "layer": layer, "channels": ch_name,
                            "context_name": ctx_name, "error": str(e),
                        })
                    continue

                if train_mask.sum() == 0 or test_mask.sum() == 0:
                    continue

                Z = extract_embeddings(model, dataset, device)
                Z_train = Z[train_mask]
                y_train = dataset.labels[train_mask]
                Z_test = Z[test_mask]
                y_test = dataset.labels[test_mask]

                for clf_cfg in GROUP_E_CLASSIFIERS:
                    clf_name = clf_cfg["name"]
                    exp_idx += 1
                    exp_label = f"L{layer} | {ch_name} | {ctx_name} | {clf_name}"
                    t0 = time.time()

                    try:
                        metrics = run_experiment(
                            Z_train, y_train, Z_test, y_test, clf_cfg["config"], seed,
                        )
                        elapsed = time.time() - t0

                        row = _make_result_row(
                            group="E", subgroup="layer_channel",
                            exp_name=exp_label, ch_name=ch_name, clf_name=clf_name,
                            layer_info=f"L{layer}", ctx_name=ctx_name,
                            ctx_before=ctx_before, ctx_after=ctx_after,
                            metrics=metrics, n_train=int(train_mask.sum()),
                            n_test=int(test_mask.sum()), embed_dim=Z.shape[1],
                            elapsed=elapsed,
                        )
                        results.append(row)
                        logger.info(
                            _format_metrics_log(exp_idx, n_total, exp_label, metrics, elapsed),
                        )
                    except Exception as e:
                        logger.error("  [%d/%d] %s FAILED: %s", exp_idx, n_total, exp_label, e)
                        results.append({
                            "group": "E", "layer": layer, "channels": ch_name,
                            "classifier": clf_name, "context_name": ctx_name,
                            "error": str(e),
                        })

        del model
        _cleanup_gpu()

    df = pd.DataFrame(results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "group_e_layer_channel.csv", index=False)
    logger.info("Saved: %s (%d rows)", tables_dir / "group_e_layer_channel.csv", len(df))

    return results


# ============================================================================
# Group F: Ultra-Fine-Grain Context Resolution
# ============================================================================

def run_group_f(cfg: dict, device: str, output_dir: Path) -> list[dict]:
    """30 contexts × 3 channels × 2 classifiers = 180 experiments.

    Fixed: L2, combined token. Model loaded once.
    10-min steps from 151-401min, wider steps beyond.
    """
    n_total = len(GROUP_F_CONTEXTS) * len(GROUP_F_CHANNELS) * len(GROUP_F_CLASSIFIERS)
    logger.info("=" * 70)
    logger.info("GROUP F: Ultra-Fine-Grain Context (%d experiments)", n_total)
    logger.info("=" * 70)

    results = []
    split_date = cfg.get("split_date", "2026-02-15")
    pretrained = cfg["model"]["pretrained_name"]
    seed = cfg.get("seed", 42)
    stride = cfg.get("stride", 1)
    layer = 2

    model = load_mantis_model(pretrained, layer, "combined", device)
    exp_idx = 0

    for ch_cfg in GROUP_F_CHANNELS:
        ch_name = ch_cfg["name"]
        ch_keys = ch_cfg["keys"]
        logger.info("=== Channels: %s ===", ch_name)

        try:
            sensor_arr, train_labels, test_labels, _, timestamps = (
                _load_sensor_data(cfg, ch_keys, split_date)
            )
        except Exception as e:
            logger.error("Data load failed for %s: %s", ch_name, e)
            skip_count = len(GROUP_F_CONTEXTS) * len(GROUP_F_CLASSIFIERS)
            for _ in range(skip_count):
                exp_idx += 1
                results.append({"group": "F", "channels": ch_name, "error": str(e)})
            continue

        all_labels = np.where(train_labels >= 0, train_labels, test_labels)

        for ctx_cfg in GROUP_F_CONTEXTS:
            ctx_name = ctx_cfg["name"]
            ctx_before = ctx_cfg["before"]
            ctx_after = ctx_cfg["after"]

            try:
                dataset, train_mask, test_mask = _build_dataset_and_split(
                    sensor_arr, all_labels, timestamps,
                    ctx_before, ctx_after, stride, split_date,
                )
            except Exception as e:
                logger.error("Dataset failed %s|%s: %s", ch_name, ctx_name, e)
                for clf_cfg in GROUP_F_CLASSIFIERS:
                    exp_idx += 1
                    results.append({
                        "group": "F", "channels": ch_name,
                        "context_name": ctx_name, "error": str(e),
                    })
                continue

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            Z = extract_embeddings(model, dataset, device)
            Z_train = Z[train_mask]
            y_train = dataset.labels[train_mask]
            Z_test = Z[test_mask]
            y_test = dataset.labels[test_mask]

            for clf_cfg in GROUP_F_CLASSIFIERS:
                clf_name = clf_cfg["name"]
                exp_idx += 1
                exp_label = f"{ch_name} | {ctx_name} | {clf_name}"
                t0 = time.time()

                try:
                    metrics = run_experiment(
                        Z_train, y_train, Z_test, y_test, clf_cfg["config"], seed,
                    )
                    elapsed = time.time() - t0

                    row = _make_result_row(
                        group="F", subgroup="finegrain_ctx",
                        exp_name=exp_label, ch_name=ch_name, clf_name=clf_name,
                        layer_info=f"L{layer}", ctx_name=ctx_name,
                        ctx_before=ctx_before, ctx_after=ctx_after,
                        metrics=metrics, n_train=int(train_mask.sum()),
                        n_test=int(test_mask.sum()), embed_dim=Z.shape[1],
                        elapsed=elapsed,
                    )
                    results.append(row)
                    logger.info(
                        _format_metrics_log(exp_idx, n_total, exp_label, metrics, elapsed),
                    )
                except Exception as e:
                    logger.error("  [%d/%d] %s FAILED: %s", exp_idx, n_total, exp_label, e)
                    results.append({
                        "group": "F", "channels": ch_name, "classifier": clf_name,
                        "context_name": ctx_name, "error": str(e),
                    })

    del model
    _cleanup_gpu()

    df = pd.DataFrame(results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "group_f_finegrain_context.csv", index=False)
    logger.info("Saved: %s (%d rows)", tables_dir / "group_f_finegrain_context.csv", len(df))

    return results


# ============================================================================
# Group G: SVM Hyperparameter Deep Tuning
# ============================================================================

def run_group_g(cfg: dict, device: str, output_dir: Path) -> list[dict]:
    """31 SVM configs × 3 channels × 2 contexts = 186 experiments.

    Fixed: L2, combined token. Model loaded once.
    """
    n_total = len(GROUP_G_SVM_CONFIGS) * len(GROUP_G_CHANNELS) * len(GROUP_G_CONTEXTS)
    logger.info("=" * 70)
    logger.info("GROUP G: SVM Hyperparameter Deep Tuning (%d experiments)", n_total)
    logger.info("=" * 70)

    results = []
    split_date = cfg.get("split_date", "2026-02-15")
    pretrained = cfg["model"]["pretrained_name"]
    seed = cfg.get("seed", 42)
    stride = cfg.get("stride", 1)
    layer = 2

    model = load_mantis_model(pretrained, layer, "combined", device)
    exp_idx = 0

    for ch_cfg in GROUP_G_CHANNELS:
        ch_name = ch_cfg["name"]
        ch_keys = ch_cfg["keys"]
        logger.info("=== Channels: %s ===", ch_name)

        try:
            sensor_arr, train_labels, test_labels, _, timestamps = (
                _load_sensor_data(cfg, ch_keys, split_date)
            )
        except Exception as e:
            logger.error("Data load failed for %s: %s", ch_name, e)
            skip_count = len(GROUP_G_CONTEXTS) * len(GROUP_G_SVM_CONFIGS)
            for _ in range(skip_count):
                exp_idx += 1
                results.append({"group": "G", "channels": ch_name, "error": str(e)})
            continue

        all_labels = np.where(train_labels >= 0, train_labels, test_labels)

        for ctx_cfg in GROUP_G_CONTEXTS:
            ctx_name = ctx_cfg["name"]
            ctx_before = ctx_cfg["before"]
            ctx_after = ctx_cfg["after"]

            try:
                dataset, train_mask, test_mask = _build_dataset_and_split(
                    sensor_arr, all_labels, timestamps,
                    ctx_before, ctx_after, stride, split_date,
                )
            except Exception as e:
                logger.error("Dataset failed %s|%s: %s", ch_name, ctx_name, e)
                for svm_cfg in GROUP_G_SVM_CONFIGS:
                    exp_idx += 1
                    results.append({
                        "group": "G", "channels": ch_name,
                        "context_name": ctx_name, "error": str(e),
                    })
                continue

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            Z = extract_embeddings(model, dataset, device)
            Z_train = Z[train_mask]
            y_train = dataset.labels[train_mask]
            Z_test = Z[test_mask]
            y_test = dataset.labels[test_mask]

            for svm_cfg in GROUP_G_SVM_CONFIGS:
                svm_name = svm_cfg["name"]
                exp_idx += 1
                exp_label = f"{ch_name} | {ctx_name} | {svm_name}"
                t0 = time.time()

                try:
                    metrics = run_experiment(
                        Z_train, y_train, Z_test, y_test, svm_cfg["config"], seed,
                    )
                    elapsed = time.time() - t0

                    row = _make_result_row(
                        group="G", subgroup="svm_tuning",
                        exp_name=exp_label, ch_name=ch_name, clf_name=svm_name,
                        layer_info=f"L{layer}", ctx_name=ctx_name,
                        ctx_before=ctx_before, ctx_after=ctx_after,
                        metrics=metrics, n_train=int(train_mask.sum()),
                        n_test=int(test_mask.sum()), embed_dim=Z.shape[1],
                        elapsed=elapsed,
                        svm_kernel=svm_cfg["config"].get("kernel", "rbf"),
                        svm_C=svm_cfg["config"].get("C", 1.0),
                        svm_gamma=str(svm_cfg["config"].get("gamma", "scale")),
                        svm_degree=svm_cfg["config"].get("degree", 3),
                    )
                    results.append(row)
                    logger.info(
                        _format_metrics_log(exp_idx, n_total, exp_label, metrics, elapsed),
                    )
                except Exception as e:
                    logger.error("  [%d/%d] %s FAILED: %s", exp_idx, n_total, exp_label, e)
                    results.append({
                        "group": "G", "channels": ch_name, "classifier": svm_name,
                        "context_name": ctx_name, "error": str(e),
                    })

    del model
    _cleanup_gpu()

    df = pd.DataFrame(results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "group_g_svm_tuning.csv", index=False)
    logger.info("Saved: %s (%d rows)", tables_dir / "group_g_svm_tuning.csv", len(df))

    return results


# ============================================================================
# Group H: Multi-Layer Fusion + Per-Channel-Layer
# ============================================================================

def run_group_h(cfg: dict, device: str, output_dir: Path) -> list[dict]:
    """Part H1: 11 multi-layer × 3 channels × 2 contexts × 2 classifiers = 132.
    Part H2: 12 per-channel-layer × 2 contexts × 2 classifiers = 48.
    Total: 180 experiments.
    """
    n_h1 = len(MULTILAYER_COMBOS) * len(GROUP_H_CHANNELS) * len(GROUP_H_CONTEXTS) * len(GROUP_H_CLASSIFIERS)
    n_h2 = len(PERCHANNEL_CONFIGS) * len(GROUP_H_CONTEXTS) * len(GROUP_H_CLASSIFIERS)
    n_total = n_h1 + n_h2
    logger.info("=" * 70)
    logger.info("GROUP H: Multi-Layer Fusion + Per-Channel-Layer (%d experiments)", n_total)
    logger.info("  H1: Multi-layer fusion (%d)", n_h1)
    logger.info("  H2: Per-channel-layer (%d)", n_h2)
    logger.info("=" * 70)

    results = []
    split_date = cfg.get("split_date", "2026-02-15")
    pretrained = cfg["model"]["pretrained_name"]
    seed = cfg.get("seed", 42)
    stride = cfg.get("stride", 1)
    exp_idx = 0

    # ── H1: Multi-layer fusion ──
    logger.info("--- H1: Multi-Layer Fusion ---")

    for ch_cfg in GROUP_H_CHANNELS:
        ch_name = ch_cfg["name"]
        ch_keys = ch_cfg["keys"]
        logger.info("=== Channels: %s ===", ch_name)

        try:
            sensor_arr, train_labels, test_labels, _, timestamps = (
                _load_sensor_data(cfg, ch_keys, split_date)
            )
        except Exception as e:
            logger.error("Data load failed for %s: %s", ch_name, e)
            skip = len(MULTILAYER_COMBOS) * len(GROUP_H_CONTEXTS) * len(GROUP_H_CLASSIFIERS)
            for _ in range(skip):
                exp_idx += 1
                results.append({"group": "H", "subgroup": "multilayer", "channels": ch_name, "error": str(e)})
            continue

        all_labels = np.where(train_labels >= 0, train_labels, test_labels)

        for ctx_cfg in GROUP_H_CONTEXTS:
            ctx_name = ctx_cfg["name"]
            ctx_before = ctx_cfg["before"]
            ctx_after = ctx_cfg["after"]

            try:
                dataset, train_mask, test_mask = _build_dataset_and_split(
                    sensor_arr, all_labels, timestamps,
                    ctx_before, ctx_after, stride, split_date,
                )
            except Exception as e:
                logger.error("Dataset failed %s|%s: %s", ch_name, ctx_name, e)
                skip = len(MULTILAYER_COMBOS) * len(GROUP_H_CLASSIFIERS)
                for _ in range(skip):
                    exp_idx += 1
                    results.append({
                        "group": "H", "subgroup": "multilayer",
                        "channels": ch_name, "context_name": ctx_name, "error": str(e),
                    })
                continue

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            for ml_cfg in MULTILAYER_COMBOS:
                ml_name = ml_cfg["name"]
                layers = ml_cfg["layers"]

                try:
                    Z = extract_multilayer_embeddings(
                        pretrained, layers, "combined", dataset, device,
                    )
                except Exception as e:
                    logger.error("Multi-layer extraction failed %s|%s|%s: %s",
                                 ch_name, ctx_name, ml_name, e)
                    for clf_cfg in GROUP_H_CLASSIFIERS:
                        exp_idx += 1
                        results.append({
                            "group": "H", "subgroup": "multilayer",
                            "channels": ch_name, "context_name": ctx_name,
                            "layer": ml_name, "error": str(e),
                        })
                    continue

                Z_train = Z[train_mask]
                y_train = dataset.labels[train_mask]
                Z_test = Z[test_mask]
                y_test = dataset.labels[test_mask]

                for clf_cfg in GROUP_H_CLASSIFIERS:
                    clf_name = clf_cfg["name"]
                    exp_idx += 1
                    exp_label = f"H1 {ch_name} | {ctx_name} | {ml_name} | {clf_name}"
                    t0 = time.time()

                    try:
                        metrics = run_experiment(
                            Z_train, y_train, Z_test, y_test, clf_cfg["config"], seed,
                        )
                        elapsed = time.time() - t0

                        row = _make_result_row(
                            group="H", subgroup="multilayer",
                            exp_name=exp_label, ch_name=ch_name, clf_name=clf_name,
                            layer_info=ml_name, ctx_name=ctx_name,
                            ctx_before=ctx_before, ctx_after=ctx_after,
                            metrics=metrics, n_train=int(train_mask.sum()),
                            n_test=int(test_mask.sum()), embed_dim=Z.shape[1],
                            elapsed=elapsed,
                            fusion_layers=str(layers),
                            n_layers=len(layers),
                        )
                        results.append(row)
                        logger.info(
                            _format_metrics_log(
                                exp_idx, n_total, exp_label, metrics, elapsed,
                                extra=f"dim={Z.shape[1]}",
                            ),
                        )
                    except Exception as e:
                        logger.error("  [%d/%d] %s FAILED: %s",
                                     exp_idx, n_total, exp_label, e)
                        results.append({
                            "group": "H", "subgroup": "multilayer",
                            "channels": ch_name, "classifier": clf_name,
                            "layer": ml_name, "context_name": ctx_name,
                            "error": str(e),
                        })

    # ── H2: Per-channel-layer ──
    logger.info("--- H2: Per-Channel-Layer ---")

    for pc_cfg in PERCHANNEL_CONFIGS:
        pc_name = pc_cfg["name"]
        ch_keys = pc_cfg["channel_keys"]
        ch_layers = pc_cfg["layers"]
        ch_label = "+".join(ch_keys)
        logger.info("=== %s ===", pc_name)

        try:
            sensor_arr, train_labels, test_labels, _, timestamps = (
                _load_sensor_data(cfg, ch_keys, split_date)
            )
        except Exception as e:
            logger.error("Data load failed for %s: %s", pc_name, e)
            skip = len(GROUP_H_CONTEXTS) * len(GROUP_H_CLASSIFIERS)
            for _ in range(skip):
                exp_idx += 1
                results.append({
                    "group": "H", "subgroup": "perchannel_layer",
                    "channels": ch_label, "layer": pc_name, "error": str(e),
                })
            continue

        all_labels = np.where(train_labels >= 0, train_labels, test_labels)

        for ctx_cfg in GROUP_H_CONTEXTS:
            ctx_name = ctx_cfg["name"]
            ctx_before = ctx_cfg["before"]
            ctx_after = ctx_cfg["after"]

            try:
                dataset, train_mask, test_mask = _build_dataset_and_split(
                    sensor_arr, all_labels, timestamps,
                    ctx_before, ctx_after, stride, split_date,
                )
            except Exception as e:
                logger.error("Dataset failed %s|%s: %s", pc_name, ctx_name, e)
                for clf_cfg in GROUP_H_CLASSIFIERS:
                    exp_idx += 1
                    results.append({
                        "group": "H", "subgroup": "perchannel_layer",
                        "channels": ch_label, "layer": pc_name,
                        "context_name": ctx_name, "error": str(e),
                    })
                continue

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            try:
                Z = extract_perchannel_layer_embeddings(
                    pretrained, ch_layers, "combined", dataset, device,
                )
            except Exception as e:
                logger.error("Per-channel-layer extraction failed %s|%s: %s",
                             pc_name, ctx_name, e)
                for clf_cfg in GROUP_H_CLASSIFIERS:
                    exp_idx += 1
                    results.append({
                        "group": "H", "subgroup": "perchannel_layer",
                        "channels": ch_label, "layer": pc_name,
                        "context_name": ctx_name, "error": str(e),
                    })
                continue

            Z_train = Z[train_mask]
            y_train = dataset.labels[train_mask]
            Z_test = Z[test_mask]
            y_test = dataset.labels[test_mask]

            for clf_cfg in GROUP_H_CLASSIFIERS:
                clf_name = clf_cfg["name"]
                exp_idx += 1
                exp_label = f"H2 {pc_name} | {ctx_name} | {clf_name}"
                t0 = time.time()

                try:
                    metrics = run_experiment(
                        Z_train, y_train, Z_test, y_test, clf_cfg["config"], seed,
                    )
                    elapsed = time.time() - t0

                    row = _make_result_row(
                        group="H", subgroup="perchannel_layer",
                        exp_name=exp_label, ch_name=ch_label, clf_name=clf_name,
                        layer_info=pc_name, ctx_name=ctx_name,
                        ctx_before=ctx_before, ctx_after=ctx_after,
                        metrics=metrics, n_train=int(train_mask.sum()),
                        n_test=int(test_mask.sum()), embed_dim=Z.shape[1],
                        elapsed=elapsed,
                        perchannel_layers=str(ch_layers),
                    )
                    results.append(row)
                    logger.info(
                        _format_metrics_log(exp_idx, n_total, exp_label, metrics, elapsed),
                    )
                except Exception as e:
                    logger.error("  [%d/%d] %s FAILED: %s",
                                 exp_idx, n_total, exp_label, e)
                    results.append({
                        "group": "H", "subgroup": "perchannel_layer",
                        "channels": ch_label, "classifier": clf_name,
                        "layer": pc_name, "context_name": ctx_name,
                        "error": str(e),
                    })

    _cleanup_gpu()

    df = pd.DataFrame(results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "group_h_multilayer_fusion.csv", index=False)
    logger.info("Saved: %s (%d rows)", tables_dir / "group_h_multilayer_fusion.csv", len(df))

    return results


# ============================================================================
# Group I: Alternative Classifiers + Special Configs
# ============================================================================

def run_group_i(cfg: dict, device: str, output_dir: Path) -> list[dict]:
    """I1: 8 alt classifiers × 5 channels × 2 contexts = 80.
    I2: 5 extended contexts × 4 channels × 2 classifiers = 40.
    I3: 9 backward contexts × 3 channels × 2 classifiers = 54.
    I4: 18 asymmetric × 2 channels × 2 classifiers = 72.
    I5: 8 future-heavy × 3 channels × 2 classifiers = 48.
    Total: 294 experiments.
    """
    n_i1 = len(GROUP_I1_CLASSIFIERS) * len(GROUP_I1_CHANNELS) * len(GROUP_I1_CONTEXTS)
    n_i2 = len(GROUP_I2_CONTEXTS) * len(GROUP_I2_CHANNELS) * len(GROUP_I2_CLASSIFIERS)
    n_i3 = len(GROUP_I3_CONTEXTS) * len(GROUP_I3_CHANNELS) * len(GROUP_I3_CLASSIFIERS)
    n_i4 = len(GROUP_I4_CONTEXTS) * len(GROUP_I4_CHANNELS) * len(GROUP_I4_CLASSIFIERS)
    n_i5 = len(GROUP_I5_CONTEXTS) * len(GROUP_I5_CHANNELS) * len(GROUP_I5_CLASSIFIERS)
    n_total = n_i1 + n_i2 + n_i3 + n_i4 + n_i5
    logger.info("=" * 70)
    logger.info("GROUP I: Alternative Classifiers + Special (%d experiments)", n_total)
    logger.info("  I1: Alt classifiers (%d)", n_i1)
    logger.info("  I2: Extended context (%d)", n_i2)
    logger.info("  I3: Backward mode (%d)", n_i3)
    logger.info("  I4: Asymmetric past-heavy (%d)", n_i4)
    logger.info("  I5: Future-heavy (%d)", n_i5)
    logger.info("=" * 70)

    results = []
    split_date = cfg.get("split_date", "2026-02-15")
    pretrained = cfg["model"]["pretrained_name"]
    seed = cfg.get("seed", 42)
    stride = cfg.get("stride", 1)
    layer = 2
    exp_idx = 0

    # Load model once (L2 for all subgroups)
    model = load_mantis_model(pretrained, layer, "combined", device)

    # ── I1: Alternative Classifiers ──
    logger.info("--- I1: Alternative Classifiers ---")

    for ch_cfg in GROUP_I1_CHANNELS:
        ch_name = ch_cfg["name"]
        ch_keys = ch_cfg["keys"]

        try:
            sensor_arr, train_labels, test_labels, _, timestamps = (
                _load_sensor_data(cfg, ch_keys, split_date)
            )
        except Exception as e:
            logger.error("Data load failed for %s: %s", ch_name, e)
            skip = len(GROUP_I1_CONTEXTS) * len(GROUP_I1_CLASSIFIERS)
            for _ in range(skip):
                exp_idx += 1
                results.append({"group": "I", "subgroup": "I1_alt_clf", "channels": ch_name, "error": str(e)})
            continue

        all_labels = np.where(train_labels >= 0, train_labels, test_labels)

        for ctx_cfg in GROUP_I1_CONTEXTS:
            ctx_name = ctx_cfg["name"]
            ctx_before = ctx_cfg["before"]
            ctx_after = ctx_cfg["after"]

            try:
                dataset, train_mask, test_mask = _build_dataset_and_split(
                    sensor_arr, all_labels, timestamps,
                    ctx_before, ctx_after, stride, split_date,
                )
            except Exception as e:
                logger.error("Dataset failed %s|%s: %s", ch_name, ctx_name, e)
                for clf_cfg in GROUP_I1_CLASSIFIERS:
                    exp_idx += 1
                    results.append({
                        "group": "I", "subgroup": "I1_alt_clf",
                        "channels": ch_name, "context_name": ctx_name, "error": str(e),
                    })
                continue

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            Z = extract_embeddings(model, dataset, device)
            Z_train = Z[train_mask]
            y_train = dataset.labels[train_mask]
            Z_test = Z[test_mask]
            y_test = dataset.labels[test_mask]

            for clf_cfg in GROUP_I1_CLASSIFIERS:
                clf_name = clf_cfg["name"]
                exp_idx += 1
                exp_label = f"I1 {ch_name} | {ctx_name} | {clf_name}"
                t0 = time.time()

                try:
                    metrics = run_experiment(
                        Z_train, y_train, Z_test, y_test, clf_cfg["config"], seed,
                    )
                    elapsed = time.time() - t0

                    row = _make_result_row(
                        group="I", subgroup="I1_alt_clf",
                        exp_name=exp_label, ch_name=ch_name, clf_name=clf_name,
                        layer_info=f"L{layer}", ctx_name=ctx_name,
                        ctx_before=ctx_before, ctx_after=ctx_after,
                        metrics=metrics, n_train=int(train_mask.sum()),
                        n_test=int(test_mask.sum()), embed_dim=Z.shape[1],
                        elapsed=elapsed,
                    )
                    results.append(row)
                    logger.info(
                        _format_metrics_log(exp_idx, n_total, exp_label, metrics, elapsed),
                    )
                except Exception as e:
                    logger.error("  [%d/%d] %s FAILED: %s", exp_idx, n_total, exp_label, e)
                    results.append({
                        "group": "I", "subgroup": "I1_alt_clf",
                        "channels": ch_name, "classifier": clf_name,
                        "context_name": ctx_name, "error": str(e),
                    })

    # ── I2: Extended Context ──
    logger.info("--- I2: Extended Context ---")

    for ch_cfg in GROUP_I2_CHANNELS:
        ch_name = ch_cfg["name"]
        ch_keys = ch_cfg["keys"]

        try:
            sensor_arr, train_labels, test_labels, _, timestamps = (
                _load_sensor_data(cfg, ch_keys, split_date)
            )
        except Exception as e:
            logger.error("Data load failed for %s: %s", ch_name, e)
            skip = len(GROUP_I2_CONTEXTS) * len(GROUP_I2_CLASSIFIERS)
            for _ in range(skip):
                exp_idx += 1
                results.append({"group": "I", "subgroup": "I2_extended_ctx", "channels": ch_name, "error": str(e)})
            continue

        all_labels = np.where(train_labels >= 0, train_labels, test_labels)

        for ctx_cfg in GROUP_I2_CONTEXTS:
            ctx_name = ctx_cfg["name"]
            ctx_before = ctx_cfg["before"]
            ctx_after = ctx_cfg["after"]

            try:
                dataset, train_mask, test_mask = _build_dataset_and_split(
                    sensor_arr, all_labels, timestamps,
                    ctx_before, ctx_after, stride, split_date,
                )
            except Exception as e:
                logger.error("Dataset failed %s|%s: %s", ch_name, ctx_name, e)
                for clf_cfg in GROUP_I2_CLASSIFIERS:
                    exp_idx += 1
                    results.append({
                        "group": "I", "subgroup": "I2_extended_ctx",
                        "channels": ch_name, "context_name": ctx_name, "error": str(e),
                    })
                continue

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            Z = extract_embeddings(model, dataset, device)
            Z_train = Z[train_mask]
            y_train = dataset.labels[train_mask]
            Z_test = Z[test_mask]
            y_test = dataset.labels[test_mask]

            for clf_cfg in GROUP_I2_CLASSIFIERS:
                clf_name = clf_cfg["name"]
                exp_idx += 1
                exp_label = f"I2 {ch_name} | {ctx_name} | {clf_name}"
                t0 = time.time()

                try:
                    metrics = run_experiment(
                        Z_train, y_train, Z_test, y_test, clf_cfg["config"], seed,
                    )
                    elapsed = time.time() - t0

                    row = _make_result_row(
                        group="I", subgroup="I2_extended_ctx",
                        exp_name=exp_label, ch_name=ch_name, clf_name=clf_name,
                        layer_info=f"L{layer}", ctx_name=ctx_name,
                        ctx_before=ctx_before, ctx_after=ctx_after,
                        metrics=metrics, n_train=int(train_mask.sum()),
                        n_test=int(test_mask.sum()), embed_dim=Z.shape[1],
                        elapsed=elapsed,
                    )
                    results.append(row)
                    logger.info(
                        _format_metrics_log(exp_idx, n_total, exp_label, metrics, elapsed),
                    )
                except Exception as e:
                    logger.error("  [%d/%d] %s FAILED: %s", exp_idx, n_total, exp_label, e)
                    results.append({
                        "group": "I", "subgroup": "I2_extended_ctx",
                        "channels": ch_name, "classifier": clf_name,
                        "context_name": ctx_name, "error": str(e),
                    })

    # ── I3: Backward-Only Mode ──
    logger.info("--- I3: Backward-Only Mode ---")

    for ch_cfg in GROUP_I3_CHANNELS:
        ch_name = ch_cfg["name"]
        ch_keys = ch_cfg["keys"]

        try:
            sensor_arr, train_labels, test_labels, _, timestamps = (
                _load_sensor_data(cfg, ch_keys, split_date)
            )
        except Exception as e:
            logger.error("Data load failed for %s: %s", ch_name, e)
            skip = len(GROUP_I3_CONTEXTS) * len(GROUP_I3_CLASSIFIERS)
            for _ in range(skip):
                exp_idx += 1
                results.append({"group": "I", "subgroup": "I3_backward", "channels": ch_name, "error": str(e)})
            continue

        all_labels = np.where(train_labels >= 0, train_labels, test_labels)

        for ctx_cfg in GROUP_I3_CONTEXTS:
            ctx_name = ctx_cfg["name"]
            ctx_before = ctx_cfg["before"]
            ctx_after = ctx_cfg["after"]  # always 0 for backward

            try:
                dataset, train_mask, test_mask = _build_dataset_and_split(
                    sensor_arr, all_labels, timestamps,
                    ctx_before, ctx_after, stride, split_date,
                )
            except Exception as e:
                logger.error("Dataset failed %s|%s: %s", ch_name, ctx_name, e)
                for clf_cfg in GROUP_I3_CLASSIFIERS:
                    exp_idx += 1
                    results.append({
                        "group": "I", "subgroup": "I3_backward",
                        "channels": ch_name, "context_name": ctx_name, "error": str(e),
                    })
                continue

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            Z = extract_embeddings(model, dataset, device)
            Z_train = Z[train_mask]
            y_train = dataset.labels[train_mask]
            Z_test = Z[test_mask]
            y_test = dataset.labels[test_mask]

            for clf_cfg in GROUP_I3_CLASSIFIERS:
                clf_name = clf_cfg["name"]
                exp_idx += 1
                exp_label = f"I3 {ch_name} | {ctx_name} | {clf_name}"
                t0 = time.time()

                try:
                    metrics = run_experiment(
                        Z_train, y_train, Z_test, y_test, clf_cfg["config"], seed,
                    )
                    elapsed = time.time() - t0

                    row = _make_result_row(
                        group="I", subgroup="I3_backward",
                        exp_name=exp_label, ch_name=ch_name, clf_name=clf_name,
                        layer_info=f"L{layer}", ctx_name=ctx_name,
                        ctx_before=ctx_before, ctx_after=ctx_after,
                        metrics=metrics, n_train=int(train_mask.sum()),
                        n_test=int(test_mask.sum()), embed_dim=Z.shape[1],
                        elapsed=elapsed,
                        context_mode="backward",
                    )
                    results.append(row)
                    logger.info(
                        _format_metrics_log(exp_idx, n_total, exp_label, metrics, elapsed),
                    )
                except Exception as e:
                    logger.error("  [%d/%d] %s FAILED: %s", exp_idx, n_total, exp_label, e)
                    results.append({
                        "group": "I", "subgroup": "I3_backward",
                        "channels": ch_name, "classifier": clf_name,
                        "context_name": ctx_name, "error": str(e),
                    })

    # ── I4: Asymmetric Optimized ──
    logger.info("--- I4: Asymmetric Optimized ---")

    for ch_cfg in GROUP_I4_CHANNELS:
        ch_name = ch_cfg["name"]
        ch_keys = ch_cfg["keys"]

        try:
            sensor_arr, train_labels, test_labels, _, timestamps = (
                _load_sensor_data(cfg, ch_keys, split_date)
            )
        except Exception as e:
            logger.error("Data load failed for %s: %s", ch_name, e)
            skip = len(GROUP_I4_CONTEXTS) * len(GROUP_I4_CLASSIFIERS)
            for _ in range(skip):
                exp_idx += 1
                results.append({"group": "I", "subgroup": "I4_asymmetric", "channels": ch_name, "error": str(e)})
            continue

        all_labels = np.where(train_labels >= 0, train_labels, test_labels)

        for ctx_cfg in GROUP_I4_CONTEXTS:
            ctx_name = ctx_cfg["name"]
            ctx_before = ctx_cfg["before"]
            ctx_after = ctx_cfg["after"]

            try:
                dataset, train_mask, test_mask = _build_dataset_and_split(
                    sensor_arr, all_labels, timestamps,
                    ctx_before, ctx_after, stride, split_date,
                )
            except Exception as e:
                logger.error("Dataset failed %s|%s: %s", ch_name, ctx_name, e)
                for clf_cfg in GROUP_I4_CLASSIFIERS:
                    exp_idx += 1
                    results.append({
                        "group": "I", "subgroup": "I4_asymmetric",
                        "channels": ch_name, "context_name": ctx_name, "error": str(e),
                    })
                continue

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            Z = extract_embeddings(model, dataset, device)
            Z_train = Z[train_mask]
            y_train = dataset.labels[train_mask]
            Z_test = Z[test_mask]
            y_test = dataset.labels[test_mask]

            for clf_cfg in GROUP_I4_CLASSIFIERS:
                clf_name = clf_cfg["name"]
                exp_idx += 1
                exp_label = f"I4 {ch_name} | {ctx_name} | {clf_name}"
                t0 = time.time()

                try:
                    metrics = run_experiment(
                        Z_train, y_train, Z_test, y_test, clf_cfg["config"], seed,
                    )
                    elapsed = time.time() - t0

                    row = _make_result_row(
                        group="I", subgroup="I4_asymmetric",
                        exp_name=exp_label, ch_name=ch_name, clf_name=clf_name,
                        layer_info=f"L{layer}", ctx_name=ctx_name,
                        ctx_before=ctx_before, ctx_after=ctx_after,
                        metrics=metrics, n_train=int(train_mask.sum()),
                        n_test=int(test_mask.sum()), embed_dim=Z.shape[1],
                        elapsed=elapsed,
                        context_mode="asymmetric",
                    )
                    results.append(row)
                    logger.info(
                        _format_metrics_log(exp_idx, n_total, exp_label, metrics, elapsed),
                    )
                except Exception as e:
                    logger.error("  [%d/%d] %s FAILED: %s", exp_idx, n_total, exp_label, e)
                    results.append({
                        "group": "I", "subgroup": "I4_asymmetric",
                        "channels": ch_name, "classifier": clf_name,
                        "context_name": ctx_name, "error": str(e),
                    })

    # ── I5: Future-Heavy Mode ──
    logger.info("--- I5: Future-Heavy Mode ---")

    for ch_cfg in GROUP_I5_CHANNELS:
        ch_name = ch_cfg["name"]
        ch_keys = ch_cfg["keys"]

        try:
            sensor_arr, train_labels, test_labels, _, timestamps = (
                _load_sensor_data(cfg, ch_keys, split_date)
            )
        except Exception as e:
            logger.error("Data load failed for %s: %s", ch_name, e)
            skip = len(GROUP_I5_CONTEXTS) * len(GROUP_I5_CLASSIFIERS)
            for _ in range(skip):
                exp_idx += 1
                results.append({"group": "I", "subgroup": "I5_future_heavy", "channels": ch_name, "error": str(e)})
            continue

        all_labels = np.where(train_labels >= 0, train_labels, test_labels)

        for ctx_cfg in GROUP_I5_CONTEXTS:
            ctx_name = ctx_cfg["name"]
            ctx_before = ctx_cfg["before"]
            ctx_after = ctx_cfg["after"]

            try:
                dataset, train_mask, test_mask = _build_dataset_and_split(
                    sensor_arr, all_labels, timestamps,
                    ctx_before, ctx_after, stride, split_date,
                )
            except Exception as e:
                logger.error("Dataset failed %s|%s: %s", ch_name, ctx_name, e)
                for clf_cfg in GROUP_I5_CLASSIFIERS:
                    exp_idx += 1
                    results.append({
                        "group": "I", "subgroup": "I5_future_heavy",
                        "channels": ch_name, "context_name": ctx_name, "error": str(e),
                    })
                continue

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            Z = extract_embeddings(model, dataset, device)
            Z_train = Z[train_mask]
            y_train = dataset.labels[train_mask]
            Z_test = Z[test_mask]
            y_test = dataset.labels[test_mask]

            for clf_cfg in GROUP_I5_CLASSIFIERS:
                clf_name = clf_cfg["name"]
                exp_idx += 1
                exp_label = f"I5 {ch_name} | {ctx_name} | {clf_name}"
                t0 = time.time()

                try:
                    metrics = run_experiment(
                        Z_train, y_train, Z_test, y_test, clf_cfg["config"], seed,
                    )
                    elapsed = time.time() - t0

                    row = _make_result_row(
                        group="I", subgroup="I5_future_heavy",
                        exp_name=exp_label, ch_name=ch_name, clf_name=clf_name,
                        layer_info=f"L{layer}", ctx_name=ctx_name,
                        ctx_before=ctx_before, ctx_after=ctx_after,
                        metrics=metrics, n_train=int(train_mask.sum()),
                        n_test=int(test_mask.sum()), embed_dim=Z.shape[1],
                        elapsed=elapsed,
                        context_mode="future_heavy",
                    )
                    results.append(row)
                    logger.info(
                        _format_metrics_log(exp_idx, n_total, exp_label, metrics, elapsed),
                    )
                except Exception as e:
                    logger.error("  [%d/%d] %s FAILED: %s", exp_idx, n_total, exp_label, e)
                    results.append({
                        "group": "I", "subgroup": "I5_future_heavy",
                        "channels": ch_name, "classifier": clf_name,
                        "context_name": ctx_name, "error": str(e),
                    })

    del model
    _cleanup_gpu()

    df = pd.DataFrame(results)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "group_i_alt_classifiers.csv", index=False)
    logger.info("Saved: %s (%d rows)", tables_dir / "group_i_alt_classifiers.csv", len(df))

    return results


# ============================================================================
# Visualization Generation
# ============================================================================

def generate_visualizations(all_results: list[dict], output_dir: Path):
    """Generate sweep-level analysis plots from experiment results.

    Produces:
      1. Top-K sweep bar chart (AUC ranking)
      2. Context performance curve (AUC/EER vs context window)
      3. Per-classifier AUC comparison across context windows
      4. Per-group best AUC comparison
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualization.curves import (
        plot_context_performance,
        plot_sweep_bar,
    )
    from visualization.style import save_figure, setup_style

    df = pd.DataFrame(all_results)
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()
    if len(df) == 0 or "accuracy" not in df.columns:
        logger.warning("No valid results for visualization")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Top-K AUC bar chart ---
    try:
        if "auc" in df.columns and df["auc"].notna().any():
            fig = plot_sweep_bar(
                df, group_col="experiment", metric_col="auc",
                hue_col="classifier", title="Phase 1.5 — Top 30 by AUC",
                top_k=30, output_path=plots_dir / "sweep_top30_auc",
            )
            plt.close(fig)
            logger.info("Saved: %s", plots_dir / "sweep_top30_auc.png")
    except Exception:
        logger.warning("Failed to plot sweep AUC bar chart", exc_info=True)

    # --- 2. Context performance curve (bidirectional symmetric only) ---
    try:
        bidir = df[
            (df["context_before"] == df["context_after"])
            & df["auc"].notna()
        ].copy()
        if len(bidir) > 0:
            bidir["total_ctx"] = bidir["context_before"] * 2 + 1
            # Per-classifier groups
            clf_groups = {}
            for clf_name in bidir["classifier"].unique():
                clf_df = bidir[bidir["classifier"] == clf_name]
                # Average AUC per context window per classifier
                avg = clf_df.groupby("total_ctx")["auc"].mean().sort_index()
                clf_groups[clf_name] = (avg.index.tolist(), avg.values.tolist())

            # Also gather EER (average across classifiers)
            eer_avg = bidir.groupby("total_ctx")["eer"].mean().sort_index()
            ctx_list = eer_avg.index.tolist()
            eer_list = eer_avg.values.tolist()

            fig = plot_context_performance(
                context_mins=ctx_list,
                auc_values=None,
                eer_values=eer_list,
                classifier_groups=clf_groups,
                output_path=plots_dir / "context_performance",
            )
            plt.close(fig)
            logger.info("Saved: %s", plots_dir / "context_performance.png")
    except Exception:
        logger.warning("Failed to plot context performance curve", exc_info=True)

    # --- 3. Per-group best AUC comparison ---
    try:
        if "group" in df.columns and "auc" in df.columns:
            setup_style()
            groups = sorted(df["group"].unique())
            best_aucs = []
            best_labels = []
            for grp in groups:
                grp_df = df[df["group"] == grp]
                if grp_df["auc"].notna().any():
                    best_idx = grp_df["auc"].idxmax()
                    best_aucs.append(grp_df.loc[best_idx, "auc"])
                    best_labels.append(f"Group {grp}")
                else:
                    best_aucs.append(0)
                    best_labels.append(f"Group {grp} (no AUC)")

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["#0173B2", "#DE8F05", "#029E73", "#CC3311", "#9467BD", "#8C564B"]
            bars = ax.bar(
                best_labels, best_aucs,
                color=[colors[i % len(colors)] for i in range(len(groups))],
                edgecolor="white", linewidth=0.5,
            )
            for bar, val in zip(bars, best_aucs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, val + 0.003,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9,
                )
            ax.set_ylabel("AUC")
            ax.set_title("Best AUC per Group")
            ax.set_ylim(0.5, 1.0)
            ax.grid(axis="y", alpha=0.3)
            save_figure(fig, plots_dir / "group_best_auc")
            plt.close(fig)
            logger.info("Saved: %s", plots_dir / "group_best_auc.png")
    except Exception:
        logger.warning("Failed to plot per-group AUC comparison", exc_info=True)

    # --- 4. Classifier comparison (aggregated) ---
    try:
        if "classifier" in df.columns and "auc" in df.columns:
            setup_style()
            clf_auc = df.groupby("classifier")["auc"].agg(["mean", "std", "max", "count"])
            clf_auc = clf_auc.sort_values("mean", ascending=False)

            fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(clf_auc))))
            y_pos = np.arange(len(clf_auc))
            ax.barh(
                y_pos, clf_auc["mean"],
                xerr=clf_auc["std"], height=0.6,
                color="#0173B2", alpha=0.8, edgecolor="white",
                capsize=3,
            )
            for i, (mean, mx) in enumerate(zip(clf_auc["mean"], clf_auc["max"])):
                ax.text(mean + 0.005, i, f"μ={mean:.4f} max={mx:.4f}", va="center", fontsize=8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(clf_auc.index, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("AUC (mean ± std)")
            ax.set_title("Classifier Comparison (all experiments)")
            save_figure(fig, plots_dir / "classifier_comparison")
            plt.close(fig)
            logger.info("Saved: %s", plots_dir / "classifier_comparison.png")
    except Exception:
        logger.warning("Failed to plot classifier comparison", exc_info=True)

    # --- 5. Channel combination comparison ---
    try:
        if "channels" in df.columns and "auc" in df.columns:
            setup_style()
            ch_auc = df.groupby("channels")["auc"].agg(["mean", "std", "max"])
            ch_auc = ch_auc.sort_values("mean", ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(ch_auc))))
            y_pos = np.arange(len(ch_auc))
            ax.barh(
                y_pos, ch_auc["mean"],
                xerr=ch_auc["std"], height=0.6,
                color="#029E73", alpha=0.8, edgecolor="white",
                capsize=3,
            )
            for i, (mean, mx) in enumerate(zip(ch_auc["mean"], ch_auc["max"])):
                ax.text(mean + 0.005, i, f"μ={mean:.4f} max={mx:.4f}", va="center", fontsize=8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(ch_auc.index, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("AUC (mean ± std)")
            ax.set_title("Channel Combination Comparison (Top 15)")
            save_figure(fig, plots_dir / "channel_comparison")
            plt.close(fig)
            logger.info("Saved: %s", plots_dir / "channel_comparison.png")
    except Exception:
        logger.warning("Failed to plot channel comparison", exc_info=True)

    logger.info("Visualization complete. Plots saved to: %s", plots_dir)


# ============================================================================
# Summary + Ranking
# ============================================================================

def generate_summary(all_results: list[dict], output_dir: Path):
    """Generate combined ranking and summary report.

    Primary ranking: AUC (threshold-independent, best for binary classification).
    Secondary: EER (lower is better), accuracy.
    Classifiers without predict_proba will have AUC=None → ranked at bottom.
    """
    df = pd.DataFrame(all_results)
    if "error" in df.columns:
        valid = df[df["error"].isna()].copy()
    else:
        valid = df.copy()
    if "accuracy" not in valid.columns or len(valid) == 0:
        logger.warning("No valid results to summarize")
        return

    valid = valid.dropna(subset=["accuracy"])

    # Split into AUC-available vs AUC-unavailable
    has_auc = valid[valid["auc"].notna()].copy() if "auc" in valid.columns else valid.iloc[:0].copy()
    no_auc = valid[valid["auc"].isna()].copy() if "auc" in valid.columns else valid.copy()

    # AUC-available: sort by AUC desc → EER asc (lower=better) → accuracy desc
    if len(has_auc) > 0:
        sort_keys = ["auc"]
        ascending = [False]
        if "eer" in has_auc.columns:
            sort_keys.append("eer")
            ascending.append(True)  # lower EER = better
        sort_keys.append("accuracy")
        ascending.append(False)
        has_auc = has_auc.sort_values(sort_keys, ascending=ascending)

    # AUC-unavailable: sort by accuracy desc
    if len(no_auc) > 0:
        no_auc = no_auc.sort_values("accuracy", ascending=False)

    ranked = pd.concat([has_auc, no_auc], ignore_index=True).head(30)

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(tables_dir / "top30_ranking.csv", index=False)

    # Also save full sorted results
    full_ranked = pd.concat([has_auc, no_auc], ignore_index=True)
    full_ranked.to_csv(tables_dir / "all_results_ranked.csv", index=False)

    logger.info("")
    logger.info("=" * 70)
    logger.info("TOP 30 CONFIGURATIONS (Primary: AUC, Secondary: EER):")
    logger.info("=" * 70)
    for i, (_, row) in enumerate(ranked.iterrows()):
        group = row.get("group", "?")
        sub = row.get("subgroup", "")
        exp = row.get("experiment", "?")
        acc = row.get("accuracy", 0)
        auc_val = row.get("auc")
        eer_val = row.get("eer")
        f1_val = row.get("f1_macro", 0)
        auc_str = f"AUC={auc_val:.4f}" if auc_val is not None else "AUC=N/A"
        eer_str = f" EER={eer_val:.4f}" if eer_val is not None else ""
        logger.info(
            "  #%2d [%s/%s] %s: %s%s Acc=%.4f F1m=%.4f",
            i + 1, group, sub, exp, auc_str, eer_str, acc, f1_val,
        )

    # Per-group summary
    logger.info("")
    logger.info("PER-GROUP BEST (by AUC):")
    for grp in sorted(valid["group"].unique()):
        grp_df = valid[valid["group"] == grp]
        if "auc" in grp_df.columns and grp_df["auc"].notna().any():
            best_idx = grp_df["auc"].idxmax()
            best = grp_df.loc[best_idx]
            logger.info(
                "  %s: %s  AUC=%.4f Acc=%.4f",
                grp, best.get("experiment", "?"), best["auc"], best["accuracy"],
            )
        else:
            best = grp_df.sort_values("accuracy", ascending=False).iloc[0]
            logger.info(
                "  %s: %s  AUC=N/A Acc=%.4f",
                grp, best.get("experiment", "?"), best["accuracy"],
            )

    # Save summary JSON
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    n_with_auc = len(has_auc)
    n_without_auc = len(no_auc)
    summary = {
        "total_experiments": len(all_results),
        "successful": len(valid),
        "failed": len(all_results) - len(valid),
        "with_auc": n_with_auc,
        "without_auc": n_without_auc,
        "groups_run": sorted(valid["group"].unique().tolist()) if "group" in valid.columns else [],
        "top10": ranked.head(10).to_dict("records"),
        "metric_note": "Primary: AUC (threshold-independent), Secondary: EER (balanced error rate)",
    }
    with open(reports_dir / "phase15_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved: %s", reports_dir / "phase15_summary.json")


# ============================================================================
# Main
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.5 Exhaustive Sweep (~1,368 experiments across 6 groups)"
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--group", choices=["D", "E", "F", "G", "H", "I"],
        required=True, help="Group to run (one per GPU server)",
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory (default: results/phase15_sweep)",
    )
    parser.add_argument(
        "--stride", type=int, default=None, help="Override stride",
    )
    parser.add_argument(
        "--split-date", default=None, help="Override split date (YYYY-MM-DD)",
    )
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

    output_dir = Path(args.output_dir or "results/phase15_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    group = args.group

    logger.info("=" * 70)
    logger.info("Phase 1.5 Exhaustive Sweep — Group %s", group)
    logger.info("Device: %s", device)
    logger.info("Config: %s", args.config)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 70)

    t_start = time.time()

    GROUP_RUNNERS = {
        "D": run_group_d,
        "E": run_group_e,
        "F": run_group_f,
        "G": run_group_g,
        "H": run_group_h,
        "I": run_group_i,
    }

    runner = GROUP_RUNNERS[group]
    results = runner(cfg, device, output_dir)
    generate_summary(results, output_dir)
    generate_visualizations(results, output_dir)

    total_time = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "Done! Group %s: %d experiments in %.1fs (%.1f min)",
        group, len(results), total_time, total_time / 60,
    )
    logger.info("Output: %s", output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
