"""Comprehensive sweep/ablation for Enter/Leave/Stay classification.

Optimized for 6 independent GPU servers (1× A100 80GB each).
Key optimization: embedding caching — same (layer, context, channels)
group loads model once, extracts embeddings once, then runs all classifiers.

Strategy:
  Round 1: Independent factor sweeps (A/B/C/D/E/F on 6 servers)
  Round 2: Targeted interaction sweep based on Round 1 findings

Usage:
    cd examples/classification/apc_enter_leave_stay

    # Quick validation
    python training/run_sweep.py --config training/configs/setting3-loocv.yaml --phase A --quick

    # 6 servers in parallel (each runs one phase):
    # Server 1: python training/run_sweep.py --config ... --phase A
    # Server 2: python training/run_sweep.py --config ... --phase B
    # Server 3: python training/run_sweep.py --config ... --phase C
    # Server 4: python training/run_sweep.py --config ... --phase D
    # Server 5: python training/run_sweep.py --config ... --phase E
    # Server 6: python training/run_sweep.py --config ... --phase F

    # Resume interrupted sweep
    python training/run_sweep.py --config ... --phase A --resume

    # Experiment counts
    python training/run_sweep.py --config ... --count

    # Dry run (save plan only)
    python training/run_sweep.py --config ... --phase A B --dry-run
"""

from __future__ import annotations

import argparse
import csv
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
from data.dataset import EventDatasetConfig, EventDataset
from training.run_experiment import (
    load_config,
    load_mantis_model,
    extract_all_embeddings,
    build_sklearn_classifier,
    run_loocv_sklearn,
    run_loocv_neural,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Sweep configuration — informed by prior apc_occupancy & apc_enter_leave
# ============================================================================

ALL_LAYERS = [0, 1, 2, 3, 4, 5]

ALL_SKLEARN = [
    "random_forest", "svm", "logistic_regression",
    "extra_trees", "gradient_boosting", "nearest_centroid",
]

ALL_HEADS = ["linear", "mlp"]

# Context window configurations (before, after)
# Informed by prior results:
#   - Enter/Leave optimal: sym_2_2 (5min)
#   - Occupancy optimal: ~125+1+125 (251min)
#   - Enter/Leave/Stay is a hybrid → need to test short-to-medium range
CONTEXT_CONFIGS = {
    # Symmetric windows (short to long)
    "sym_1_1": (1, 1),       # 3 min
    "sym_2_2": (2, 2),       # 5 min  <- enter/leave optimal
    "sym_3_3": (3, 3),       # 7 min
    "sym_4_4": (4, 4),       # 9 min  <- default
    "sym_6_6": (6, 6),       # 13 min
    "sym_8_8": (8, 8),       # 17 min
    "sym_12_12": (12, 12),   # 25 min
    "sym_16_16": (16, 16),   # 33 min
    "sym_24_24": (24, 24),   # 49 min
    "sym_32_32": (32, 32),   # 65 min
    "sym_48_48": (48, 48),   # 97 min
    "sym_64_64": (64, 64),   # 129 min
    # Asymmetric past-heavy (event detection favors past context)
    "asym_4_2": (4, 2),      # 7 min
    "asym_8_2": (8, 2),      # 11 min
    "asym_8_4": (8, 4),      # 13 min
    "asym_16_4": (16, 4),    # 21 min
    "asym_32_4": (32, 4),    # 37 min
    "asym_32_8": (32, 8),    # 41 min
    # Asymmetric future-heavy
    "asym_2_4": (2, 4),      # 7 min
    "asym_2_8": (2, 8),      # 11 min
    "asym_4_8": (4, 8),      # 13 min
    "asym_4_16": (4, 16),    # 21 min
    # Past-only
    "past_4_0": (4, 0),      # 5 min
    "past_8_0": (8, 0),      # 9 min
    "past_16_0": (16, 0),    # 17 min
    # Future-only
    "future_0_4": (0, 4),    # 5 min
    "future_0_8": (0, 8),    # 9 min
    "future_0_16": (0, 16),  # 17 min
}

# Channel subsets — informed by prior experiments:
#   - M+C: best for enter/leave (instantaneous events)
#   - M+C+T1: best for occupancy (sustained patterns)
#   - T1 alone: strong occupancy signal
#   - M+C+T1 may be ideal for 3-class (Enter/Leave uses M+C, Stay uses T1)
CHANNEL_SUBSETS = {
    "all_6": None,  # All 6 channels from config
    "motion_only": ["d620900d_motionSensor"],
    "contact_only": ["408981c2_contactSensor"],
    "temp1_only": ["d620900d_temperatureMeasurement"],
    "power_only": ["f2e891c6_powerMeter"],
    "motion_contact": [  # best for enter/leave
        "d620900d_motionSensor",
        "408981c2_contactSensor",
    ],
    "motion_contact_temp1": [  # best for occupancy
        "d620900d_motionSensor",
        "408981c2_contactSensor",
        "d620900d_temperatureMeasurement",
    ],
    "motion_power": [
        "d620900d_motionSensor",
        "f2e891c6_powerMeter",
    ],
    "motion_power_contact": [
        "d620900d_motionSensor",
        "f2e891c6_powerMeter",
        "408981c2_contactSensor",
    ],
    "motion_temp": [
        "d620900d_motionSensor",
        "d620900d_temperatureMeasurement",
        "ccea734e_temperatureMeasurement",
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
NEURAL_LR = [0.0001, 0.0005, 0.001, 0.005, 0.01]
NEURAL_DROPOUT = [0.0, 0.1, 0.3, 0.5, 0.7]
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
    lr: float = 0.001
    dropout: float = 0.5
    epochs: int = 50
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])

    @property
    def embedding_key(self) -> str:
        """Key for embedding cache: same key → same embeddings."""
        return f"L{self.layer}_ctx{self.context_name}_ch{self.channel_name}_s{self.label_setting}"

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
    """Phase A: Layer × Classifier baseline (default context, all channels).

    6 layers × (6 sklearn + 2 neural) = 48 experiments.
    Embedding groups: 6 (one per layer).
    """
    experiments = []
    ctx = CONTEXT_CONFIGS["sym_4_4"]

    for layer in ALL_LAYERS:
        for clf in ALL_SKLEARN:
            experiments.append(SweepExperiment(
                exp_id=f"A_{layer}_{clf}",
                phase="A", layer=layer,
                classifier_type="sklearn", classifier_name=clf,
                context_name="sym_4_4", context_before=ctx[0], context_after=ctx[1],
                channel_name="all_6", channels=None,
                label_setting=label_setting,
            ))
        for head in ALL_HEADS:
            experiments.append(SweepExperiment(
                exp_id=f"A_{layer}_neural_{head}",
                phase="A", layer=layer,
                classifier_type="neural", classifier_name=head,
                context_name="sym_4_4", context_before=ctx[0], context_after=ctx[1],
                channel_name="all_6", channels=None,
                label_setting=label_setting,
            ))

    return experiments


def generate_phase_B(label_setting: int = 3) -> list[SweepExperiment]:
    """Phase B: Context window sweep.

    Informed by prior: Enter/Leave=5min, Occupancy=251min.
    3-class hybrid → sweep 3min to 129min range.

    4 layers × 27 contexts × 4 classifiers = 432 experiments.
    Embedding groups: 4×27 = 108.
    """
    experiments = []
    layers = [0, 2, 3, 5]  # L2-L3 sklearn optimal, L5 neural optimal, L0 baseline
    clfs = ["random_forest", "svm", "logistic_regression", "nearest_centroid"]

    for layer in layers:
        for ctx_name, (ctx_b, ctx_a) in CONTEXT_CONFIGS.items():
            for clf in clfs:
                experiments.append(SweepExperiment(
                    exp_id=f"B_{layer}_{ctx_name}_{clf}",
                    phase="B", layer=layer,
                    classifier_type="sklearn", classifier_name=clf,
                    context_name=ctx_name, context_before=ctx_b, context_after=ctx_a,
                    channel_name="all_6", channels=None,
                    label_setting=label_setting,
                ))

    return experiments


def generate_phase_C(label_setting: int = 3) -> list[SweepExperiment]:
    """Phase C: Channel subset sweep.

    Informed by prior: M+C for events, M+C+T1 for occupancy.
    Test all 11 subsets to find 3-class optimal.

    4 layers × 11 channel configs × 4 classifiers = 176 experiments.
    Embedding groups: 4×11 = 44.
    """
    experiments = []
    layers = [0, 2, 3, 5]
    clfs = ["random_forest", "svm", "logistic_regression", "nearest_centroid"]
    ctx = CONTEXT_CONFIGS["sym_4_4"]

    for layer in layers:
        for ch_name, ch_list in CHANNEL_SUBSETS.items():
            for clf in clfs:
                experiments.append(SweepExperiment(
                    exp_id=f"C_{layer}_{ch_name}_{clf}",
                    phase="C", layer=layer,
                    classifier_type="sklearn", classifier_name=clf,
                    context_name="sym_4_4", context_before=ctx[0], context_after=ctx[1],
                    channel_name=ch_name, channels=ch_list,
                    label_setting=label_setting,
                ))

    return experiments


def generate_phase_D(label_setting: int = 3) -> list[SweepExperiment]:
    """Phase D: Neural hyperparameter sweep (MLP head).

    Informed by prior: dropout=0.5, no BatchNorm best. lr=0.01 or 0.001.
    Test all combos to find 3-class optimal.

    3 layers × 5 lr × 5 dropout × 3 epochs × 4 hidden = 900 experiments.
    Embedding groups: 3 (one per layer, same context+channels).
    """
    experiments = []
    layers = [0, 2, 5]
    ctx = CONTEXT_CONFIGS["sym_4_4"]

    for layer in layers:
        for lr in NEURAL_LR:
            for dropout in NEURAL_DROPOUT:
                for epochs in NEURAL_EPOCHS:
                    for hidden in NEURAL_HIDDEN:
                        hid_str = "x".join(map(str, hidden))
                        experiments.append(SweepExperiment(
                            exp_id=f"D_{layer}_lr{lr}_do{dropout}_ep{epochs}_h{hid_str}",
                            phase="D", layer=layer,
                            classifier_type="neural", classifier_name="mlp",
                            context_name="sym_4_4", context_before=ctx[0], context_after=ctx[1],
                            channel_name="all_6", channels=None,
                            label_setting=label_setting,
                            lr=lr, dropout=dropout, epochs=epochs, hidden_dims=hidden,
                        ))

    return experiments


def generate_phase_E(label_setting: int = 3) -> list[SweepExperiment]:
    """Phase E: Cross-setting comparison (Settings 1, 2, 3).

    3 settings × 6 layers × 4 classifiers = 72 experiments.
    Embedding groups: 6 per setting × 3 = 18.
    """
    experiments = []
    clfs = ["random_forest", "svm", "logistic_regression", "nearest_centroid"]
    ctx = CONTEXT_CONFIGS["sym_4_4"]

    for setting in [1, 2, 3]:
        for layer in ALL_LAYERS:
            for clf in clfs:
                experiments.append(SweepExperiment(
                    exp_id=f"E_s{setting}_{layer}_{clf}",
                    phase="E", layer=layer,
                    classifier_type="sklearn", classifier_name=clf,
                    context_name="sym_4_4", context_before=ctx[0], context_after=ctx[1],
                    channel_name="all_6", channels=None,
                    label_setting=setting,
                ))

    return experiments


def generate_phase_F(label_setting: int = 3) -> list[SweepExperiment]:
    """Phase F: Interaction sweep (context × channels × classifiers).

    Test top channel subsets with multiple context windows.
    Informed by prior: M+C, M+C+T1 are candidates for best channels.

    4 layers × 6 contexts × 5 channels × (4 sklearn + 2 neural) = 720 experiments.
    Embedding groups: 4×6×5 = 120.
    """
    experiments = []
    layers = [0, 2, 3, 5]
    contexts = ["sym_2_2", "sym_4_4", "sym_8_8", "sym_16_16", "asym_8_4", "asym_4_8"]
    channels = ["all_6", "motion_contact", "motion_contact_temp1", "motion_power_contact", "no_energy"]
    clfs = ["random_forest", "svm", "logistic_regression", "nearest_centroid"]

    for layer in layers:
        for ctx_name in contexts:
            ctx_b, ctx_a = CONTEXT_CONFIGS[ctx_name]
            for ch_name in channels:
                ch_list = CHANNEL_SUBSETS[ch_name]
                for clf in clfs:
                    experiments.append(SweepExperiment(
                        exp_id=f"F_{layer}_{ctx_name}_{ch_name}_{clf}",
                        phase="F", layer=layer,
                        classifier_type="sklearn", classifier_name=clf,
                        context_name=ctx_name, context_before=ctx_b, context_after=ctx_a,
                        channel_name=ch_name, channels=ch_list,
                        label_setting=label_setting,
                    ))
                for head in ALL_HEADS:
                    experiments.append(SweepExperiment(
                        exp_id=f"F_{layer}_{ctx_name}_{ch_name}_neural_{head}",
                        phase="F", layer=layer,
                        classifier_type="neural", classifier_name=head,
                        context_name=ctx_name, context_before=ctx_b, context_after=ctx_a,
                        channel_name=ch_name, channels=ch_list,
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
# Embedding cache & group-based execution
# ============================================================================

def _group_experiments(experiments: list[SweepExperiment]) -> dict[str, list[SweepExperiment]]:
    """Group experiments by embedding key for efficient batch execution."""
    groups: dict[str, list[SweepExperiment]] = {}
    for exp in experiments:
        key = exp.embedding_key
        groups.setdefault(key, []).append(exp)
    return groups


def _load_data_for_experiment(
    exp: SweepExperiment,
    raw_cfg: dict,
) -> tuple[np.ndarray, "pd.DatetimeIndex", np.ndarray, np.ndarray, list[str], list[str]]:
    """Load preprocessed data for an experiment configuration."""
    data_cfg = raw_cfg.get("data", {})
    channels = exp.channels if exp.channels is not None else data_cfg.get("channels")

    preprocess_cfg = EventPreprocessConfig(
        sensor_csv=data_cfg.get("sensor_csv", ""),
        events_csv=data_cfg.get("events_csv", ""),
        column_names_csv=data_cfg.get("column_names_csv"),
        column_names=data_cfg.get("column_names"),
        channels=channels,
        exclude_channels=data_cfg.get("exclude_channels", []),
        nan_threshold=data_cfg.get("nan_threshold", 0.3),
        label_setting=exp.label_setting,
        add_time_features=data_cfg.get("add_time_features", False),
    )

    return load_sensor_and_events(preprocess_cfg)


def run_experiment_group(
    group_key: str,
    experiments: list[SweepExperiment],
    raw_cfg: dict,
    device: str,
    seed: int,
    results_csv: Path,
    completed: set[str],
) -> tuple[int, int]:
    """Run a group of experiments sharing the same embeddings.

    Returns (n_success, n_error).
    """
    import torch

    # Filter already completed
    remaining = [e for e in experiments if e.exp_id not in completed]
    if not remaining:
        return 0, 0

    # Use first experiment as representative for data/model loading
    rep = remaining[0]

    # Load data
    sensor_array, sensor_timestamps, event_timestamps, event_labels, channel_names, class_names = (
        _load_data_for_experiment(rep, raw_cfg)
    )

    # Build dataset
    ds_config = EventDatasetConfig(
        context_mode="bidirectional",
        context_before=rep.context_before,
        context_after=rep.context_after,
    )
    dataset = EventDataset(
        sensor_array, sensor_timestamps, event_timestamps, event_labels, ds_config,
    )

    # Load model and extract embeddings ONCE for this group
    model_cfg = raw_cfg.get("model", {})
    pretrained_name = model_cfg.get("pretrained_name", "paris-noah/MantisV2")
    output_token = model_cfg.get("output_token", "combined")

    logger.info("  Loading MantisV2 L%d for group %s (%d experiments)",
                rep.layer, group_key, len(remaining))
    network, model = load_mantis_model(pretrained_name, rep.layer, output_token, device)
    Z = extract_all_embeddings(model, dataset)

    del model, network
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_classes = len(class_names)
    n_success = 0
    n_error = 0

    # Run all classifiers on the cached embeddings
    for exp in remaining:
        try:
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
            for cls_name, f1_val in metrics.f1_per_class.items():
                result[f"f1_{cls_name}"] = round(f1_val, 4)

            append_result_csv(result, results_csv)
            n_success += 1

            logger.info(
                "    [%s] Acc=%.4f  F1=%.4f  (%.1fs)",
                exp.exp_id, result["accuracy"], result["f1_macro"], elapsed,
            )

        except Exception:
            n_error += 1
            logger.error("    FAILED: %s", exp.exp_id, exc_info=True)
            error_result = exp.to_dict()
            error_result.update({"accuracy": None, "f1_macro": None, "error": "FAILED"})
            append_result_csv(error_result, results_csv)

    return n_success, n_error


# ============================================================================
# CSV helpers
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


def append_result_csv(result: dict, output_path: Path) -> None:
    """Append a single result row to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    fieldnames = list(result.keys())
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


# ============================================================================
# Main sweep runner
# ============================================================================

def run_sweep(
    raw_cfg: dict,
    phases: list[str],
    device: str = "cuda",
    seed: int = 42,
    label_setting: int = 3,
    resume: bool = False,
    dry_run: bool = False,
    quick: bool = False,
) -> None:
    """Run the full sweep with embedding caching."""
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

    # Quick mode: limit to first 2 embedding groups
    if quick:
        groups = _group_experiments(all_experiments)
        quick_keys = list(groups.keys())[:2]
        all_experiments = []
        for k in quick_keys:
            all_experiments.extend(groups[k][:3])  # max 3 experiments per group
        logger.info("QUICK MODE: limited to %d experiments", len(all_experiments))

    logger.info("Total experiments: %d", len(all_experiments))

    # Group by embedding key
    groups = _group_experiments(all_experiments)
    logger.info("Embedding groups: %d (model loads saved: %d)",
                len(groups), len(all_experiments) - len(groups))

    if dry_run:
        logger.info("DRY RUN — Experiment plan:")
        for key, exps in sorted(groups.items()):
            logger.info("  Group %s: %d experiments", key, len(exps))
            for exp in exps[:3]:
                logger.info("    %s (%s/%s)", exp.exp_id, exp.classifier_type, exp.classifier_name)
            if len(exps) > 3:
                logger.info("    ... +%d more", len(exps) - 3)

        plan_path = sweep_dir / f"plan_{'_'.join(phases)}.json"
        plan = {
            "phases": phases,
            "total_experiments": len(all_experiments),
            "embedding_groups": len(groups),
            "model_loads_saved": len(all_experiments) - len(groups),
            "experiments": [e.to_dict() for e in all_experiments],
        }
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2)
        logger.info("Saved plan: %s", plan_path)
        return

    # Per-phase result file (avoids conflicts between 6 servers)
    phase_tag = "_".join(phases)
    results_csv = sweep_dir / f"results_{phase_tag}.csv"

    # Resume support
    completed = set()
    if resume:
        # Check both per-phase and global results
        completed = load_completed_ids(results_csv)
        global_csv = sweep_dir / "sweep_results.csv"
        if global_csv.exists():
            completed |= load_completed_ids(global_csv)
        logger.info("Resuming: %d experiments already completed", len(completed))

    # Run groups
    n_groups = len(groups)
    total_success = 0
    total_error = 0

    for gi, (group_key, group_exps) in enumerate(sorted(groups.items())):
        remaining_in_group = [e for e in group_exps if e.exp_id not in completed]
        if not remaining_in_group:
            continue

        logger.info("=" * 60)
        logger.info("[Group %d/%d] %s (%d experiments)",
                    gi + 1, n_groups, group_key, len(remaining_in_group))
        logger.info("=" * 60)

        n_ok, n_err = run_experiment_group(
            group_key, group_exps, raw_cfg, device, seed, results_csv, completed,
        )
        total_success += n_ok
        total_error += n_err

    # Summary
    logger.info("=" * 60)
    logger.info("Sweep complete: %d succeeded, %d errors", total_success, total_error)
    logger.info("Results: %s", results_csv)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enter/Leave/Stay Ablation Sweep (optimized for 6 GPU servers)",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--phase", nargs="+", default=["A"],
        choices=list(PHASE_GENERATORS.keys()),
        help="Phases to run (A/B/C/D/E/F). Each server runs one phase.",
    )
    parser.add_argument("--label-setting", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from partial results")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--quick", action="store_true", help="Quick validation (few experiments)")
    parser.add_argument("--count", action="store_true", help="Show experiment counts and exit")

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
            groups = _group_experiments(exps)
            logger.info("Phase %s: %4d experiments, %3d embedding groups",
                        phase_name, len(exps), len(groups))
            total += len(exps)
        logger.info("Grand total: %d experiments", total)
        return

    raw_cfg = load_config(args.config)

    logger.info("=" * 60)
    logger.info("Enter/Leave/Stay Sweep (embedding-cached)")
    logger.info("=" * 60)
    logger.info("  Phases: %s", args.phase)
    logger.info("  Label setting: %d", args.label_setting)
    logger.info("  Device: %s", args.device)
    logger.info("  Seed: %d", args.seed)

    t_start = time.time()

    run_sweep(
        raw_cfg,
        phases=args.phase,
        device=args.device,
        seed=args.seed,
        label_setting=args.label_setting,
        resume=args.resume,
        dry_run=args.dry_run,
        quick=args.quick,
    )

    t_total = time.time() - t_start
    logger.info("Total time: %.1fs (%.1f min)", t_total, t_total / 60)


if __name__ == "__main__":
    main()
