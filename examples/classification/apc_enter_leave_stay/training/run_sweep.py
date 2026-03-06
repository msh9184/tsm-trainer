"""Comprehensive sweep/ablation for Enter/Leave/Stay classification.

Systematically explores all combinations of:
  - Transformer layers (L0-L5)
  - Sklearn classifiers (6 types)
  - Neural heads (linear, mlp)
  - Context windows (bidirectional: various before/after)
  - Channel subsets
  - Label settings (1, 2, 3)
  - Neural hyperparameters (lr, dropout, epochs, hidden_dims)

Designed for GPU execution with A100 80GB.

Usage:
    cd examples/classification/apc_enter_leave_stay

    # Full sweep (Setting 3 only, all phases)
    python training/run_sweep.py --config training/configs/setting3-loocv.yaml

    # Single phase
    python training/run_sweep.py --config ... --phase A
    python training/run_sweep.py --config ... --phase B
    python training/run_sweep.py --config ... --phase C

    # Quick validation (2 experiments only)
    python training/run_sweep.py --config ... --dry-run

    # Resume from partial results
    python training/run_sweep.py --config ... --phase A --resume
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from data.preprocess import EventPreprocessConfig, load_sensor_and_events, CLASS_NAMES_BY_SETTING
from data.dataset import EventDatasetConfig, EventDataset, build_dataset_config
from training.run_experiment import (
    load_config,
    load_mantis_model,
    extract_all_embeddings,
    build_sklearn_classifier,
    run_loocv_sklearn,
    run_loocv_neural,
    save_results_csv,
    save_results_txt,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Sweep configuration
# ============================================================================

# All transformer layers
ALL_LAYERS = [0, 1, 2, 3, 4, 5]

# All sklearn classifiers
ALL_SKLEARN = [
    "random_forest", "svm", "logistic_regression",
    "extra_trees", "gradient_boosting", "nearest_centroid",
]

# Neural head types
ALL_HEADS = ["linear", "mlp"]

# Context window configurations (before, after)
CONTEXT_CONFIGS = {
    # Symmetric windows
    "sym_1_1": (1, 1),       # 3 min
    "sym_2_2": (2, 2),       # 5 min
    "sym_4_4": (4, 4),       # 9 min (default)
    "sym_8_8": (8, 8),       # 17 min
    "sym_16_16": (16, 16),   # 33 min
    "sym_32_32": (32, 32),   # 65 min
    # Asymmetric past-heavy
    "asym_8_2": (8, 2),      # 11 min
    "asym_16_4": (16, 4),    # 21 min
    "asym_32_4": (32, 4),    # 37 min
    # Asymmetric future-heavy
    "asym_2_8": (2, 8),      # 11 min
    "asym_4_16": (4, 16),    # 21 min
    # Past-only
    "past_8_0": (8, 0),      # 9 min
    "past_16_0": (16, 0),    # 17 min
    # Future-only
    "future_0_8": (0, 8),    # 9 min
    "future_0_16": (0, 16),  # 17 min
}

# Channel subsets to test
CHANNEL_SUBSETS = {
    "all_6": None,  # All 6 channels from config
    "motion_only": ["d620900d_motionSensor"],
    "power_only": ["f2e891c6_powerMeter"],
    "motion_power": ["d620900d_motionSensor", "f2e891c6_powerMeter"],
    "motion_temp": [
        "d620900d_motionSensor",
        "d620900d_temperatureMeasurement",
        "ccea734e_temperatureMeasurement",
    ],
    "motion_power_contact": [
        "d620900d_motionSensor",
        "f2e891c6_powerMeter",
        "408981c2_contactSensor",
    ],
    "no_energy": [
        "d620900d_motionSensor",
        "f2e891c6_powerMeter",
        "d620900d_temperatureMeasurement",
        "ccea734e_temperatureMeasurement",
        "408981c2_contactSensor",
    ],
}

# Neural hyperparameter sweep
NEURAL_LR = [0.0001, 0.0005, 0.001, 0.005]
NEURAL_DROPOUT = [0.1, 0.3, 0.5, 0.7]
NEURAL_EPOCHS = [30, 50, 100]
NEURAL_HIDDEN = [
    [64],
    [128, 64],
    [256, 128],
    [256, 128, 64],
]


@dataclass
class SweepExperiment:
    """Single experiment configuration."""
    exp_id: str
    phase: str
    layer: int
    classifier_type: str  # "sklearn" or "neural"
    classifier_name: str
    context_name: str
    context_before: int
    context_after: int
    channel_name: str
    channels: list[str] | None
    label_setting: int = 3
    # Neural-specific
    lr: float = 0.001
    dropout: float = 0.5
    epochs: int = 50
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])

    def to_dict(self) -> dict:
        return {
            "exp_id": self.exp_id,
            "phase": self.phase,
            "layer": self.layer,
            "classifier_type": self.classifier_type,
            "classifier_name": self.classifier_name,
            "context_name": self.context_name,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "channel_name": self.channel_name,
            "label_setting": self.label_setting,
            "lr": self.lr,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "hidden_dims": str(self.hidden_dims),
        }


# ============================================================================
# Phase definitions
# ============================================================================

def generate_phase_A(label_setting: int = 3) -> list[SweepExperiment]:
    """Phase A: Layer × Classifier sweep (default context, all channels).

    6 layers × (6 sklearn + 2 neural) = 48 experiments.
    """
    experiments = []
    ctx = CONTEXT_CONFIGS["sym_4_4"]

    for layer in ALL_LAYERS:
        for clf in ALL_SKLEARN:
            experiments.append(SweepExperiment(
                exp_id=f"A_{layer}_{clf}",
                phase="A",
                layer=layer,
                classifier_type="sklearn",
                classifier_name=clf,
                context_name="sym_4_4",
                context_before=ctx[0],
                context_after=ctx[1],
                channel_name="all_6",
                channels=None,
                label_setting=label_setting,
            ))
        for head in ALL_HEADS:
            experiments.append(SweepExperiment(
                exp_id=f"A_{layer}_neural_{head}",
                phase="A",
                layer=layer,
                classifier_type="neural",
                classifier_name=head,
                context_name="sym_4_4",
                context_before=ctx[0],
                context_after=ctx[1],
                channel_name="all_6",
                channels=None,
                label_setting=label_setting,
            ))

    return experiments


def generate_phase_B(label_setting: int = 3) -> list[SweepExperiment]:
    """Phase B: Context window sweep (best layers from Phase A: L0, L2, L5).

    3 layers × 15 contexts × 3 classifiers (RF, SVM, NearestCentroid) = 135 experiments.
    """
    experiments = []
    representative_layers = [0, 2, 5]
    representative_clfs = ["random_forest", "svm", "nearest_centroid"]

    for layer in representative_layers:
        for ctx_name, (ctx_b, ctx_a) in CONTEXT_CONFIGS.items():
            for clf in representative_clfs:
                experiments.append(SweepExperiment(
                    exp_id=f"B_{layer}_{ctx_name}_{clf}",
                    phase="B",
                    layer=layer,
                    classifier_type="sklearn",
                    classifier_name=clf,
                    context_name=ctx_name,
                    context_before=ctx_b,
                    context_after=ctx_a,
                    channel_name="all_6",
                    channels=None,
                    label_setting=label_setting,
                ))

    return experiments


def generate_phase_C(label_setting: int = 3) -> list[SweepExperiment]:
    """Phase C: Channel subset sweep.

    3 layers × 7 channel configs × 3 classifiers = 63 experiments.
    """
    experiments = []
    representative_layers = [0, 2, 5]
    representative_clfs = ["random_forest", "svm", "nearest_centroid"]
    ctx = CONTEXT_CONFIGS["sym_4_4"]

    for layer in representative_layers:
        for ch_name, ch_list in CHANNEL_SUBSETS.items():
            for clf in representative_clfs:
                experiments.append(SweepExperiment(
                    exp_id=f"C_{layer}_{ch_name}_{clf}",
                    phase="C",
                    layer=layer,
                    classifier_type="sklearn",
                    classifier_name=clf,
                    context_name="sym_4_4",
                    context_before=ctx[0],
                    context_after=ctx[1],
                    channel_name=ch_name,
                    channels=ch_list,
                    label_setting=label_setting,
                ))

    return experiments


def generate_phase_D(label_setting: int = 3) -> list[SweepExperiment]:
    """Phase D: Neural hyperparameter sweep (MLP head).

    3 layers × 4 lr × 4 dropout × 3 epochs × 4 hidden = 576 experiments.
    Reduced: 3 layers × top combos = ~216 experiments.
    """
    experiments = []
    representative_layers = [0, 2, 5]
    ctx = CONTEXT_CONFIGS["sym_4_4"]

    for layer in representative_layers:
        for lr in NEURAL_LR:
            for dropout in NEURAL_DROPOUT:
                for epochs in NEURAL_EPOCHS:
                    for hidden in NEURAL_HIDDEN:
                        experiments.append(SweepExperiment(
                            exp_id=f"D_{layer}_lr{lr}_do{dropout}_ep{epochs}_h{'x'.join(map(str, hidden))}",
                            phase="D",
                            layer=layer,
                            classifier_type="neural",
                            classifier_name="mlp",
                            context_name="sym_4_4",
                            context_before=ctx[0],
                            context_after=ctx[1],
                            channel_name="all_6",
                            channels=None,
                            label_setting=label_setting,
                            lr=lr,
                            dropout=dropout,
                            epochs=epochs,
                            hidden_dims=hidden,
                        ))

    return experiments


def generate_phase_E(label_setting: int = 3) -> list[SweepExperiment]:
    """Phase E: Cross-setting comparison (Settings 1, 2, 3).

    3 settings × 6 layers × 3 classifiers = 54 experiments.
    """
    experiments = []
    representative_clfs = ["random_forest", "svm", "nearest_centroid"]
    ctx = CONTEXT_CONFIGS["sym_4_4"]

    for setting in [1, 2, 3]:
        for layer in ALL_LAYERS:
            for clf in representative_clfs:
                experiments.append(SweepExperiment(
                    exp_id=f"E_s{setting}_{layer}_{clf}",
                    phase="E",
                    layer=layer,
                    classifier_type="sklearn",
                    classifier_name=clf,
                    context_name="sym_4_4",
                    context_before=ctx[0],
                    context_after=ctx[1],
                    channel_name="all_6",
                    channels=None,
                    label_setting=setting,
                ))

    return experiments


def generate_phase_F(label_setting: int = 3) -> list[SweepExperiment]:
    """Phase F: Best context × Best channels × All classifiers.

    Top 3 contexts × Top 3 channels × 6 layers × (6 sklearn + 2 neural) = 432 experiments.
    Reduced: 3 contexts × 3 channels × 3 layers × 8 classifiers = 216 experiments.
    """
    experiments = []
    representative_layers = [0, 2, 5]
    top_contexts = ["sym_4_4", "sym_8_8", "asym_16_4"]
    top_channels = ["all_6", "motion_power", "motion_power_contact"]
    ctx_map = CONTEXT_CONFIGS

    for layer in representative_layers:
        for ctx_name in top_contexts:
            ctx_b, ctx_a = ctx_map[ctx_name]
            for ch_name in top_channels:
                ch_list = CHANNEL_SUBSETS[ch_name]
                for clf in ALL_SKLEARN:
                    experiments.append(SweepExperiment(
                        exp_id=f"F_{layer}_{ctx_name}_{ch_name}_{clf}",
                        phase="F",
                        layer=layer,
                        classifier_type="sklearn",
                        classifier_name=clf,
                        context_name=ctx_name,
                        context_before=ctx_b,
                        context_after=ctx_a,
                        channel_name=ch_name,
                        channels=ch_list,
                        label_setting=label_setting,
                    ))
                for head in ALL_HEADS:
                    experiments.append(SweepExperiment(
                        exp_id=f"F_{layer}_{ctx_name}_{ch_name}_neural_{head}",
                        phase="F",
                        layer=layer,
                        classifier_type="neural",
                        classifier_name=head,
                        context_name=ctx_name,
                        context_before=ctx_b,
                        context_after=ctx_a,
                        channel_name=ch_name,
                        channels=ch_list,
                        label_setting=label_setting,
                    ))

    return experiments


PHASE_GENERATORS = {
    "A": generate_phase_A,
    "B": generate_phase_B,
    "C": generate_phase_C,
    "D": generate_phase_D,
    "E": generate_phase_E,
    "F": generate_phase_F,
}


# ============================================================================
# Sweep runner
# ============================================================================

def load_completed_ids(results_csv: Path) -> set[str]:
    """Load exp_ids already completed from results CSV."""
    if not results_csv.exists():
        return set()
    completed = set()
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "exp_id" in row:
                completed.add(row["exp_id"])
    return completed


def run_single_experiment(
    exp: SweepExperiment,
    raw_cfg: dict,
    device: str,
    seed: int,
) -> dict:
    """Run a single sweep experiment and return result dict."""
    import torch

    # Override config for this experiment
    cfg = dict(raw_cfg)
    cfg["label_setting"] = exp.label_setting

    data_cfg = cfg.get("data", {})
    if exp.channels is not None:
        data_cfg["channels"] = exp.channels
    cfg["data"] = data_cfg

    # Load data
    preprocess_cfg = EventPreprocessConfig(
        sensor_csv=data_cfg.get("sensor_csv", ""),
        events_csv=data_cfg.get("events_csv", ""),
        column_names_csv=data_cfg.get("column_names_csv"),
        column_names=data_cfg.get("column_names"),
        channels=exp.channels if exp.channels is not None else data_cfg.get("channels"),
        exclude_channels=data_cfg.get("exclude_channels", []),
        nan_threshold=data_cfg.get("nan_threshold", 0.3),
        label_setting=exp.label_setting,
        add_time_features=data_cfg.get("add_time_features", False),
    )

    sensor_array, sensor_timestamps, event_timestamps, event_labels, channel_names, class_names = (
        load_sensor_and_events(preprocess_cfg)
    )

    # Build dataset with experiment-specific context
    ds_config = EventDatasetConfig(
        context_mode="bidirectional",
        context_before=exp.context_before,
        context_after=exp.context_after,
    )
    dataset = EventDataset(
        sensor_array, sensor_timestamps, event_timestamps, event_labels, ds_config,
    )

    # Load model and extract embeddings
    model_cfg = cfg.get("model", {})
    pretrained_name = model_cfg.get("pretrained_name", "paris-noah/MantisV2")
    output_token = model_cfg.get("output_token", "combined")

    network, model = load_mantis_model(pretrained_name, exp.layer, output_token, device)
    Z = extract_all_embeddings(model, dataset)

    del model, network
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_classes = len(class_names)

    # Run experiment
    t0 = time.time()

    if exp.classifier_type == "sklearn":
        clf_factory = lambda s=seed, name=exp.classifier_name: build_sklearn_classifier(name, s)
        metrics = run_loocv_sklearn(Z, event_labels, clf_factory, class_names)
    else:
        metrics = run_loocv_neural(
            Z, event_labels,
            head_type=exp.classifier_name,
            n_classes=n_classes,
            epochs=exp.epochs,
            lr=exp.lr,
            weight_decay=0.01,
            dropout=exp.dropout,
            hidden_dims=exp.hidden_dims,
            device=device,
            class_names=class_names,
        )

    elapsed = time.time() - t0

    result = exp.to_dict()
    result.update({
        "accuracy": round(metrics.accuracy, 4),
        "f1_macro": round(metrics.f1_macro, 4),
        "f1_weighted": round(metrics.f1_weighted, 4),
        "precision_macro": round(metrics.precision_macro, 4),
        "recall_macro": round(metrics.recall_macro, 4),
        "roc_auc": round(metrics.roc_auc, 4) if not np.isnan(metrics.roc_auc) else None,
        "n_samples": metrics.n_samples,
        "n_channels": len(channel_names),
        "time_sec": round(elapsed, 1),
    })

    # Per-class F1
    for cls_name, f1_val in metrics.f1_per_class.items():
        result[f"f1_{cls_name}"] = round(f1_val, 4)

    return result


def append_result_csv(result: dict, output_path: Path) -> None:
    """Append a single result row to CSV (thread-safe-ish)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    fieldnames = list(result.keys())

    # If file exists, use its header order
    if file_exists:
        with open(output_path) as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
        new_fields = [k for k in result.keys() if k not in existing_fields]
        fieldnames = list(existing_fields) + new_fields

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def run_sweep(
    raw_cfg: dict,
    phases: list[str],
    device: str = "cuda",
    seed: int = 42,
    label_setting: int = 3,
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    """Run the full sweep across specified phases."""
    output_dir = Path(raw_cfg.get("output_dir", "results/enter_leave_stay_setting3"))
    sweep_dir = output_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiments
    all_experiments = []
    for phase in phases:
        gen = PHASE_GENERATORS.get(phase)
        if gen is None:
            logger.error("Unknown phase: %s (available: %s)", phase, list(PHASE_GENERATORS.keys()))
            continue
        if phase == "E":
            exps = gen()  # E generates all settings internally
        else:
            exps = gen(label_setting)
        all_experiments.extend(exps)
        logger.info("Phase %s: %d experiments", phase, len(exps))

    logger.info("Total experiments: %d", len(all_experiments))

    if dry_run:
        logger.info("DRY RUN: showing first 5 experiments")
        for exp in all_experiments[:5]:
            logger.info("  %s", exp.to_dict())
        logger.info("... and %d more", max(0, len(all_experiments) - 5))

        # Save experiment plan
        plan_path = sweep_dir / "experiment_plan.json"
        plan = [exp.to_dict() for exp in all_experiments]
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2)
        logger.info("Saved experiment plan: %s (%d experiments)", plan_path, len(plan))
        return

    # Resume support
    results_csv = sweep_dir / "sweep_results.csv"
    completed = set()
    if resume:
        completed = load_completed_ids(results_csv)
        logger.info("Resuming: %d experiments already completed", len(completed))

    remaining = [e for e in all_experiments if e.exp_id not in completed]
    logger.info("Experiments to run: %d (skipping %d completed)", len(remaining), len(completed))

    # Run experiments
    n_total = len(remaining)
    n_errors = 0

    for i, exp in enumerate(remaining):
        logger.info("=" * 60)
        logger.info("[%d/%d] %s", i + 1, n_total, exp.exp_id)
        logger.info(
            "  L%d / %s/%s / ctx=%s / ch=%s / setting=%d",
            exp.layer, exp.classifier_type, exp.classifier_name,
            exp.context_name, exp.channel_name, exp.label_setting,
        )

        try:
            result = run_single_experiment(exp, raw_cfg, device, seed)
            append_result_csv(result, results_csv)

            logger.info(
                "  -> Acc=%.4f  F1=%.4f  (%.1fs)",
                result["accuracy"], result["f1_macro"], result["time_sec"],
            )
        except Exception:
            n_errors += 1
            logger.error("  FAILED: %s", exp.exp_id, exc_info=True)
            # Save error marker
            error_result = exp.to_dict()
            error_result["accuracy"] = None
            error_result["f1_macro"] = None
            error_result["error"] = "FAILED"
            append_result_csv(error_result, results_csv)

    # Summary
    logger.info("=" * 60)
    logger.info("Sweep complete: %d/%d succeeded, %d errors", n_total - n_errors, n_total, n_errors)
    logger.info("Results: %s", results_csv)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Enter/Leave/Stay Ablation Sweep",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--phase", nargs="+", default=["A"],
        choices=list(PHASE_GENERATORS.keys()),
        help="Phases to run: A (layer×clf), B (context), C (channels), "
             "D (neural HP), E (cross-setting), F (interaction)",
    )
    parser.add_argument("--label-setting", type=int, default=3, help="Primary label setting (default: 3)")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from partial results")
    parser.add_argument("--dry-run", action="store_true", help="Show experiment plan without running")
    parser.add_argument("--count", action="store_true", help="Show experiment counts per phase and exit")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.count:
        total = 0
        for phase_name, gen in sorted(PHASE_GENERATORS.items()):
            if phase_name == "E":
                exps = gen()
            else:
                exps = gen(args.label_setting)
            logger.info("Phase %s: %d experiments", phase_name, len(exps))
            total += len(exps)
        logger.info("Grand total: %d experiments", total)
        return

    raw_cfg = load_config(args.config)

    logger.info("=" * 60)
    logger.info("Enter/Leave/Stay Comprehensive Sweep")
    logger.info("=" * 60)
    logger.info("  Phases: %s", args.phase)
    logger.info("  Label setting: %d", args.label_setting)
    logger.info("  Device: %s", args.device)
    logger.info("  Resume: %s", args.resume)
    logger.info("  Dry run: %s", args.dry_run)

    t_start = time.time()

    run_sweep(
        raw_cfg,
        phases=args.phase,
        device=args.device,
        seed=args.seed,
        label_setting=args.label_setting,
        resume=args.resume,
        dry_run=args.dry_run,
    )

    t_total = time.time() - t_start
    logger.info("Total sweep time: %.1fs (%.1f min)", t_total, t_total / 60)


if __name__ == "__main__":
    main()
