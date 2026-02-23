"""Training script for APC occupancy detection using MantisV2.

Supports two modes:
  Phase A — Zero-shot: Extract embeddings with pretrained MantisV2,
            train sklearn classifiers (Nearest Centroid, Random Forest, SVM).
  Phase B — Fine-tuning: Fine-tune MantisV2 classification head, adapter,
            or the entire model using MantisTrainer.

Usage:
    # Zero-shot evaluation
    python training/train.py --config training/configs/zeroshot.yaml

    # Fine-tuning (head only)
    python training/train.py --config training/configs/finetune-head.yaml

    # Fine-tuning (full)
    python training/train.py --config training/configs/finetune-full.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

# Local imports (relative to examples/classification/apc_occupancy/)
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from data.preprocess import PreprocessConfig, load_and_preprocess
from data.dataset import DatasetConfig, OccupancyDataset, create_datasets
from evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """MantisV2 model configuration."""
    pretrained_name: str = "paris-noah/MantisV2"
    return_transf_layer: int = 2  # 2 = 3rd layer (recommended for zero-shot)
    output_token: str = "combined"  # 'cls_token', 'mean_token', 'combined'
    device: str = "cuda"

    @property
    def hidden_dim(self) -> int:
        return 512 if self.output_token == "combined" else 256


@dataclass
class ZeroShotConfig:
    """Zero-shot evaluation configuration."""
    classifiers: list[str] = field(
        default_factory=lambda: ["nearest_centroid", "random_forest", "svm"]
    )
    random_forest_n_estimators: int = 200
    random_forest_n_jobs: int = -1
    svm_kernel: str = "rbf"
    svm_C: float = 1.0
    random_seed: int = 42


@dataclass
class FineTuneConfig:
    """Fine-tuning configuration."""
    fine_tuning_type: str = "head"  # 'full', 'head', 'adapter_head', 'scratch'
    num_epochs: int = 500
    batch_size: int = 256
    learning_rate: float = 2e-4
    learning_rate_adjusting: bool = True
    label_smoothing: float = 0.1
    # Adapter settings (for adapter_head mode)
    adapter_type: str | None = None  # 'pca', 'svd', 'var', 'linear', None
    adapter_new_channels: int = 5
    # Custom head settings
    head_hidden_dim: int = 100


@dataclass
class TrainConfig:
    """Top-level training configuration."""
    mode: str = "zeroshot"  # 'zeroshot' or 'finetune'
    model: ModelConfig = field(default_factory=ModelConfig)
    data: dict = field(default_factory=dict)  # PreprocessConfig fields
    dataset: dict = field(default_factory=dict)  # DatasetConfig fields
    zeroshot: ZeroShotConfig = field(default_factory=ZeroShotConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    output_dir: str = "results"
    seed: int = 42


def load_config(config_path: str | Path) -> TrainConfig:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    config = TrainConfig()
    config.mode = raw.get("mode", "zeroshot")
    config.output_dir = raw.get("output_dir", "results")
    config.seed = raw.get("seed", 42)

    # Model config
    model_cfg = raw.get("model", {})
    config.model = ModelConfig(**{
        k: v for k, v in model_cfg.items() if k in ModelConfig.__dataclass_fields__
    })

    # Store raw data/dataset config dicts for flexible construction
    config.data = raw.get("data", {})
    config.dataset = raw.get("dataset", {})

    # Zero-shot config
    zs_cfg = raw.get("zeroshot", {})
    config.zeroshot = ZeroShotConfig(**{
        k: v for k, v in zs_cfg.items() if k in ZeroShotConfig.__dataclass_fields__
    })

    # Fine-tune config
    ft_cfg = raw.get("finetune", {})
    config.finetune = FineTuneConfig(**{
        k: v for k, v in ft_cfg.items() if k in FineTuneConfig.__dataclass_fields__
    })

    return config


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_mantis_model(cfg: ModelConfig):
    """Load pretrained MantisV2 model.

    Returns
    -------
    network : MantisV2
        Pretrained model.
    model : MantisTrainer
        Trainer wrapper.
    """
    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    logger.info(
        "Loading MantisV2 (pretrained=%s, layer=%d, token=%s, device=%s)",
        cfg.pretrained_name, cfg.return_transf_layer, cfg.output_token, device,
    )

    network = MantisV2(
        device=device,
        return_transf_layer=cfg.return_transf_layer,
        output_token=cfg.output_token,
    )
    network = network.from_pretrained(cfg.pretrained_name)

    model = MantisTrainer(device=device, network=network)

    n_params = sum(p.numel() for p in network.parameters())
    logger.info("Model loaded: %d parameters (%.2fM)", n_params, n_params / 1e6)

    return network, model


# ---------------------------------------------------------------------------
# Phase A: Zero-shot
# ---------------------------------------------------------------------------

def _build_sklearn_classifiers(cfg: ZeroShotConfig) -> dict:
    """Build sklearn classifiers for zero-shot evaluation."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import NearestCentroid
    from sklearn.svm import SVC

    classifiers = {}
    for name in cfg.classifiers:
        if name == "nearest_centroid":
            classifiers[name] = NearestCentroid()
        elif name == "random_forest":
            classifiers[name] = RandomForestClassifier(
                n_estimators=cfg.random_forest_n_estimators,
                n_jobs=cfg.random_forest_n_jobs,
                random_state=cfg.random_seed,
            )
        elif name == "svm":
            classifiers[name] = SVC(
                kernel=cfg.svm_kernel,
                C=cfg.svm_C,
                probability=True,
                random_state=cfg.random_seed,
            )
        else:
            logger.warning("Unknown classifier: %s, skipping", name)

    return classifiers


def run_zeroshot(
    model,
    train_dataset: OccupancyDataset,
    test_dataset: OccupancyDataset,
    config: TrainConfig,
) -> dict:
    """Run zero-shot classification with MantisV2 embeddings.

    1. Extract embeddings from train/test windows
    2. Train sklearn classifiers on train embeddings
    3. Evaluate on test embeddings

    Returns dict of {classifier_name: ClassificationMetrics}.
    """
    logger.info("=" * 60)
    logger.info("Phase A: Zero-shot MantisV2 Classification")
    logger.info("=" * 60)

    X_train, y_train = train_dataset.get_numpy_arrays()
    X_test, y_test = test_dataset.get_numpy_arrays()

    logger.info("Train: %s, Test: %s", X_train.shape, X_test.shape)

    # Extract embeddings
    t0 = time.time()
    logger.info("Extracting train embeddings...")
    Z_train = model.transform(X_train)  # (n_train, n_channels * hidden_dim)
    logger.info("Extracting test embeddings...")
    Z_test = model.transform(X_test)    # (n_test, n_channels * hidden_dim)
    embed_time = time.time() - t0

    logger.info(
        "Embeddings: train=%s, test=%s (%.1fs)",
        Z_train.shape, Z_test.shape, embed_time,
    )

    # Train and evaluate classifiers
    classifiers = _build_sklearn_classifiers(config.zeroshot)
    results = {}

    for name, clf in classifiers.items():
        logger.info("-" * 40)
        logger.info("Classifier: %s", name)
        t0 = time.time()

        clf.fit(Z_train, y_train)
        y_pred = clf.predict(Z_test)

        y_prob = None
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(Z_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        train_time = time.time() - t0

        logger.info("%s", metrics.summary())
        logger.info("Classifier training + eval time: %.1fs", train_time)

        results[name] = metrics

    return results


# ---------------------------------------------------------------------------
# Phase B: Fine-tuning
# ---------------------------------------------------------------------------

def _build_adapter(cfg: FineTuneConfig, n_channels: int):
    """Build a multivariate channel adapter if configured."""
    if cfg.adapter_type is None:
        return None

    if cfg.fine_tuning_type not in ("adapter_head", "full"):
        logger.warning(
            "Adapter specified but fine_tuning_type=%s; "
            "adapter only effective with 'adapter_head' or 'full'",
            cfg.fine_tuning_type,
        )

    if cfg.adapter_type == "linear":
        from mantis.adapters import LinearChannelCombiner
        adapter = LinearChannelCombiner(
            num_channels=n_channels,
            new_num_channels=cfg.adapter_new_channels,
        )
        logger.info(
            "LinearChannelCombiner: %d -> %d channels",
            n_channels, cfg.adapter_new_channels,
        )
    elif cfg.adapter_type in ("pca", "svd", "rand"):
        from mantis.adapters import MultichannelProjector
        adapter = MultichannelProjector(
            new_num_channels=cfg.adapter_new_channels,
            patch_window_size=1,
            base_projector=cfg.adapter_type,
        )
        logger.info(
            "MultichannelProjector (%s): %d -> %d channels",
            cfg.adapter_type, n_channels, cfg.adapter_new_channels,
        )
    elif cfg.adapter_type == "var":
        from mantis.adapters import VarianceBasedSelector
        adapter = VarianceBasedSelector(
            new_num_channels=cfg.adapter_new_channels,
        )
        logger.info(
            "VarianceBasedSelector: %d -> %d channels",
            n_channels, cfg.adapter_new_channels,
        )
    else:
        raise ValueError(f"Unknown adapter type: {cfg.adapter_type}")

    return adapter


def _build_head(cfg: FineTuneConfig, n_channels: int, hidden_dim: int):
    """Build a custom classification head."""
    import torch.nn as nn

    in_dim = hidden_dim * n_channels
    head = nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, cfg.head_hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(cfg.head_hidden_dim, 2),
    )
    logger.info(
        "Custom head: %d -> %d -> 2 (with LayerNorm, ReLU, Dropout)",
        in_dim, cfg.head_hidden_dim,
    )
    return head


def run_finetune(
    network,
    model,
    train_dataset: OccupancyDataset,
    test_dataset: OccupancyDataset,
    config: TrainConfig,
) -> dict:
    """Run fine-tuning with MantisTrainer.

    Returns dict with 'finetune' key -> ClassificationMetrics.
    """
    ft_cfg = config.finetune
    logger.info("=" * 60)
    logger.info("Phase B: Fine-tuning MantisV2 (%s)", ft_cfg.fine_tuning_type)
    logger.info("=" * 60)

    X_train, y_train = train_dataset.get_numpy_arrays()
    X_test, y_test = test_dataset.get_numpy_arrays()
    n_channels = X_train.shape[1]

    logger.info(
        "Train: %s (labels: {0:%d, 1:%d}), Test: %s",
        X_train.shape, (y_train == 0).sum(), (y_train == 1).sum(), X_test.shape,
    )

    # Build adapter and head
    adapter = _build_adapter(ft_cfg, n_channels)

    # Determine effective channels for head
    effective_channels = n_channels
    if adapter is not None and hasattr(adapter, "new_num_channels"):
        effective_channels = adapter.new_num_channels

    # For non-differentiable adapters (PCA/SVD/var), apply before training
    if adapter is not None and ft_cfg.adapter_type in ("pca", "svd", "var", "rand"):
        logger.info("Applying standalone adapter before training...")
        adapter.fit(X_train)
        X_train = adapter.transform(X_train)
        X_test = adapter.transform(X_test)
        n_channels = X_train.shape[1]
        effective_channels = n_channels
        adapter = None  # Already applied, don't pass to model.fit

    head = _build_head(ft_cfg, effective_channels, config.model.hidden_dim)

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=ft_cfg.label_smoothing,
    )

    logger.info(
        "Training: epochs=%d, batch_size=%d, lr=%g, label_smoothing=%g",
        ft_cfg.num_epochs, ft_cfg.batch_size, ft_cfg.learning_rate,
        ft_cfg.label_smoothing,
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        fine_tuning_type=ft_cfg.fine_tuning_type,
        adapter=adapter,
        head=head,
        num_epochs=ft_cfg.num_epochs,
        batch_size=ft_cfg.batch_size,
        base_learning_rate=ft_cfg.learning_rate,
        criterion=criterion,
        learning_rate_adjusting=ft_cfg.learning_rate_adjusting,
    )
    train_time = time.time() - t0
    logger.info("Training completed in %.1fs", train_time)

    # Evaluate
    logger.info("Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    logger.info("\n%s", metrics.summary())

    return {"finetune": metrics}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_results(results: dict, output_dir: Path, config: TrainConfig) -> None:
    """Save evaluation results and predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "mode": config.mode,
        "model": {
            "pretrained": config.model.pretrained_name,
            "return_transf_layer": config.model.return_transf_layer,
            "output_token": config.model.output_token,
        },
        "results": {},
    }

    for name, metrics in results.items():
        report["results"][name] = metrics.to_dict()

    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved report to %s", report_path)


def main():
    parser = argparse.ArgumentParser(
        description="APC Occupancy Detection with MantisV2",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (cuda/cpu)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    if args.device:
        config.model.device = args.device
    if args.seed is not None:
        config.seed = args.seed

    _set_seed(config.seed)
    logger.info("Configuration loaded from %s", args.config)
    logger.info("Mode: %s, Device: %s, Seed: %d", config.mode, config.model.device, config.seed)

    # Load data
    preprocess_cfg = PreprocessConfig(**config.data)
    train_data = load_and_preprocess(preprocess_cfg)
    train_sensor, train_labels, channel_names, train_timestamps = train_data

    # Load test data (separate config entries)
    test_data_cfg = config.data.copy()
    test_data_cfg["sensor_csv"] = config.data.get("test_sensor_csv", "")
    test_data_cfg["label_csv"] = config.data.get("test_label_csv", "")
    # Remove test-specific keys before creating PreprocessConfig
    for key in ("test_sensor_csv", "test_label_csv"):
        test_data_cfg.pop(key, None)
    test_preprocess_cfg = PreprocessConfig(**test_data_cfg)
    test_data = load_and_preprocess(test_preprocess_cfg)
    test_sensor, test_labels, test_channel_names, test_timestamps = test_data

    # Use common channels between train and test
    common_channels = [c for c in channel_names if c in test_channel_names]
    if len(common_channels) < len(channel_names):
        logger.warning(
            "Train has %d channels, test has %d, using %d common channels",
            len(channel_names), len(test_channel_names), len(common_channels),
        )
        train_idx = [channel_names.index(c) for c in common_channels]
        test_idx = [test_channel_names.index(c) for c in common_channels]
        train_sensor = train_sensor[:, train_idx]
        test_sensor = test_sensor[:, test_idx]
        channel_names = common_channels

    logger.info("Using %d channels: %s", len(channel_names), channel_names)

    # Create datasets
    dataset_cfg = DatasetConfig(**config.dataset)
    train_dataset, test_dataset = create_datasets(
        train_sensor, train_labels, test_sensor, test_labels, dataset_cfg,
    )

    # Load model
    network, model = load_mantis_model(config.model)

    # Run appropriate phase
    output_dir = Path(config.output_dir)
    if config.mode == "zeroshot":
        results = run_zeroshot(model, train_dataset, test_dataset, config)
    elif config.mode == "finetune":
        results = run_finetune(network, model, train_dataset, test_dataset, config)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    # Save results
    _save_results(results, output_dir, config)

    # Print final summary
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    for name, metrics in results.items():
        logger.info(
            "  %-20s  Acc=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f",
            name, metrics.accuracy, metrics.f1, metrics.precision, metrics.recall,
        )


if __name__ == "__main__":
    main()
