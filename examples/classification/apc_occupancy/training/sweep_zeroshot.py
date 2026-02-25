"""Zero-shot sweep: systematically evaluate (seq_len, channels, time_features) combinations.

Runs MantisV2 zero-shot classification across all specified configurations and
produces a summary CSV + console table for easy comparison.

Usage:
    cd examples/classification/apc_occupancy

    # Run full sweep (all 16 sensor combos × 4 seq_lens = 64 experiments)
    python training/sweep_zeroshot.py --config training/configs/p4-zeroshot.yaml

    # Quick sweep: Phase 1 only (5 key combos × 1 seq_len)
    python training/sweep_zeroshot.py --config training/configs/p4-zeroshot.yaml --quick

    # Custom seq_lens
    python training/sweep_zeroshot.py --config training/configs/p4-zeroshot.yaml \
        --seq-lens 64 288
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from data.preprocess import PreprocessConfig, load_sensor_and_labels
from data.dataset import DatasetConfig, create_datasets_from_splits
from evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensor combination definitions
# ---------------------------------------------------------------------------

# Mandatory base sensors
BASE_SENSORS = [
    "d620900d_motionSensor",
    "f2e891c6_powerMeter",
]

# Optional sensors (add in combinations)
OPTIONAL_SENSORS = [
    "d620900d_temperatureMeasurement",   # T1 (d620, hallway)
    "ccea734e_temperatureMeasurement",   # T2 (ccea, room)
    "408981c2_contactSensor",            # C (door)
    "f2e891c6_energyMeter",             # E (cumulative energy)
]

# Short names for readable output
SENSOR_SHORT = {
    "d620900d_motionSensor": "M",
    "f2e891c6_powerMeter": "P",
    "d620900d_temperatureMeasurement": "T1",
    "ccea734e_temperatureMeasurement": "T2",
    "408981c2_contactSensor": "C",
    "f2e891c6_energyMeter": "E",
}

# Phase 1: key combos for initial trend analysis (5 experiments)
PHASE1_COMBOS = [
    [],                          # S2: M+P (base only)
    ["d620900d_temperatureMeasurement"],  # S3a: M+P+T1
    ["ccea734e_temperatureMeasurement"],  # S3b: M+P+T2
    ["d620900d_temperatureMeasurement", "ccea734e_temperatureMeasurement"],  # S4a: M+P+T1+T2
    OPTIONAL_SENSORS,            # S6: M+P+T1+T2+C+E (all)
]


def generate_all_combos() -> list[list[str]]:
    """Generate all 16 sensor combinations (base + 0-4 optional sensors)."""
    combos = []
    for r in range(len(OPTIONAL_SENSORS) + 1):
        for combo in combinations(OPTIONAL_SENSORS, r):
            combos.append(list(combo))
    return combos


def combo_name(optional: list[str]) -> str:
    """Create readable name like 'M+P+T1+T2'."""
    all_sensors = BASE_SENSORS + optional
    shorts = [SENSOR_SHORT.get(s, s[:4]) for s in all_sensors]
    return "+".join(shorts)


# ---------------------------------------------------------------------------
# Core sweep logic
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    """Result from a single experiment."""
    combo_id: str
    n_channels: int
    channels: str
    seq_len: int
    time_features: bool
    classifier: str
    accuracy: float
    f1: float
    precision: float
    recall: float
    eer: float
    roc_auc: float
    n_train: int
    n_test: int
    embed_dim: int
    time_sec: float


def run_single_experiment(
    sensor_array: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    all_channel_names: list[str],
    selected_channels: list[str],
    seq_len: int,
    target_seq_len: int,
    model,
    classifiers_cfg: dict,
    seed: int = 42,
) -> list[SweepResult]:
    """Run one experiment configuration and return results for all classifiers."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import NearestCentroid
    from sklearn.svm import SVC

    # Select channels
    channel_indices = [all_channel_names.index(c) for c in selected_channels]
    sensor_subset = sensor_array[:, channel_indices]

    # Build datasets
    ds_cfg = DatasetConfig(seq_len=seq_len, stride=1, target_seq_len=target_seq_len)
    train_ds, test_ds = create_datasets_from_splits(
        sensor_subset, train_labels, test_labels, ds_cfg,
    )

    X_train, y_train = train_ds.get_numpy_arrays()
    X_test, y_test = test_ds.get_numpy_arrays()

    # Extract embeddings
    Z_train = model.transform(X_train)
    Z_test = model.transform(X_test)

    # Build classifiers
    clfs = {}
    for name in classifiers_cfg.get("classifiers", ["nearest_centroid", "random_forest", "svm"]):
        if name == "nearest_centroid":
            clfs[name] = NearestCentroid()
        elif name == "random_forest":
            clfs[name] = RandomForestClassifier(
                n_estimators=classifiers_cfg.get("random_forest_n_estimators", 200),
                n_jobs=classifiers_cfg.get("random_forest_n_jobs", -1),
                random_state=seed,
            )
        elif name == "svm":
            clfs[name] = SVC(
                kernel=classifiers_cfg.get("svm_kernel", "rbf"),
                C=classifiers_cfg.get("svm_C", 1.0),
                probability=True,
                random_state=seed,
            )

    optional = [c for c in selected_channels if c not in BASE_SENSORS]
    cid = combo_name(optional)
    time_features = any("hour_sin" in c or "hour_cos" in c for c in selected_channels)

    results = []
    for clf_name, clf in clfs.items():
        t0 = time.time()
        clf.fit(Z_train, y_train)
        y_pred = clf.predict(Z_test)
        y_prob = None
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(Z_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        elapsed = time.time() - t0

        results.append(SweepResult(
            combo_id=cid,
            n_channels=len(selected_channels),
            channels=", ".join(selected_channels),
            seq_len=seq_len,
            time_features=time_features,
            classifier=clf_name,
            accuracy=metrics.accuracy,
            f1=metrics.f1,
            precision=metrics.precision,
            recall=metrics.recall,
            eer=metrics.eer,
            roc_auc=metrics.roc_auc,
            n_train=len(y_train),
            n_test=len(y_test),
            embed_dim=Z_train.shape[1],
            time_sec=elapsed,
        ))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Zero-shot sweep for P4 experiments")
    parser.add_argument("--config", type=str, required=True, help="Base YAML config")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=None,
                        help="Override seq_len values to sweep (default: 64 128 288 512)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: Phase 1 combos only, primary seq_len only")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import yaml
    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)

    # Determine sweep parameters
    if args.seq_lens:
        seq_lens = args.seq_lens
    elif args.quick:
        seq_lens = [288]
    else:
        seq_lens = [64, 128, 288, 512]

    if args.quick:
        combos = PHASE1_COMBOS
    else:
        combos = generate_all_combos()

    target_seq_len = raw_cfg.get("dataset", {}).get("target_seq_len", 512)
    output_dir = Path(args.output or raw_cfg.get("output_dir", "results/p4-sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data once (with ALL 6 channels)
    data_cfg = raw_cfg.get("data", {})
    _preprocess_fields = set(PreprocessConfig.__dataclass_fields__)
    base_cfg = {k: v for k, v in data_cfg.items() if k in _preprocess_fields}
    base_cfg["channels"] = BASE_SENSORS + OPTIONAL_SENSORS  # All 6 channels

    train_cfg = PreprocessConfig(**base_cfg)
    sensor_array, train_labels, channel_names, _, _ = load_sensor_and_labels(train_cfg)

    test_cfg_dict = dict(base_cfg)
    test_cfg_dict["label_csv"] = data_cfg.get("test_label_csv", "")
    test_cfg = PreprocessConfig(**test_cfg_dict)
    _, test_labels, _, _, _ = load_sensor_and_labels(test_cfg)

    logger.info("Sensor data: %s, Channels: %s", sensor_array.shape, channel_names)
    logger.info("Train labels: %d, Test labels: %d",
                (train_labels >= 0).sum(), (test_labels >= 0).sum())

    # Load model once
    device = args.device or raw_cfg.get("model", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer

    model_cfg = raw_cfg.get("model", {})
    network = MantisV2(
        device=device,
        return_transf_layer=model_cfg.get("return_transf_layer", 2),
        output_token=model_cfg.get("output_token", "combined"),
    )
    network = network.from_pretrained(model_cfg.get("pretrained_name", "paris-noah/MantisV2"))
    model = MantisTrainer(device=device, network=network)
    logger.info("Model loaded on %s", device)

    classifiers_cfg = raw_cfg.get("zeroshot", {})

    # Run sweep
    all_results: list[SweepResult] = []
    total_experiments = len(combos) * len(seq_lens)
    logger.info("=" * 60)
    logger.info("SWEEP: %d combos × %d seq_lens = %d experiments", len(combos), len(seq_lens), total_experiments)
    logger.info("=" * 60)

    exp_idx = 0
    t_start = time.time()

    for optional in combos:
        selected = BASE_SENSORS + optional
        cname = combo_name(optional)

        for seq_len in seq_lens:
            exp_idx += 1
            logger.info("[%d/%d] %s, seq_len=%d", exp_idx, total_experiments, cname, seq_len)

            try:
                results = run_single_experiment(
                    sensor_array, train_labels, test_labels,
                    channel_names, selected,
                    seq_len=seq_len, target_seq_len=target_seq_len,
                    model=model, classifiers_cfg=classifiers_cfg, seed=args.seed,
                )
                all_results.extend(results)

                # Log best classifier for this config
                best = max(results, key=lambda r: r.f1)
                logger.info(
                    "  Best: %s Acc=%.4f F1=%.4f AUC=%.4f",
                    best.classifier, best.accuracy, best.f1, best.roc_auc,
                )
            except Exception:
                logger.error("  FAILED", exc_info=True)

    total_time = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Sweep completed: %d results in %.1fs", len(all_results), total_time)

    # Save CSV
    csv_path = output_dir / "sweep_results.csv"
    if all_results:
        fieldnames = list(SweepResult.__dataclass_fields__.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: getattr(r, k) for k in fieldnames})
        logger.info("Results saved to %s", csv_path)

    # Save JSON summary
    json_path = output_dir / "sweep_summary.json"
    summary = {
        "total_experiments": len(all_results),
        "seq_lens": seq_lens,
        "n_combos": len(combos),
        "total_time_sec": total_time,
        "best_per_metric": {},
    }
    if all_results:
        best_f1 = max(all_results, key=lambda r: r.f1)
        best_acc = max(all_results, key=lambda r: r.accuracy)
        best_auc = max(all_results, key=lambda r: r.roc_auc)
        summary["best_per_metric"] = {
            "best_f1": {"combo": best_f1.combo_id, "seq_len": best_f1.seq_len,
                        "clf": best_f1.classifier, "f1": best_f1.f1, "acc": best_f1.accuracy},
            "best_acc": {"combo": best_acc.combo_id, "seq_len": best_acc.seq_len,
                         "clf": best_acc.classifier, "acc": best_acc.accuracy, "f1": best_acc.f1},
            "best_auc": {"combo": best_auc.combo_id, "seq_len": best_auc.seq_len,
                         "clf": best_auc.classifier, "auc": best_auc.roc_auc, "f1": best_auc.f1},
        }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", json_path)

    # Print console summary table
    if all_results:
        _print_summary_table(all_results)


def _print_summary_table(results: list[SweepResult]) -> None:
    """Print a formatted summary table to console."""
    # Group by (combo_id, seq_len), show best classifier's metrics
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        groups[(r.combo_id, r.seq_len)].append(r)

    print("\n" + "=" * 100)
    print("SWEEP RESULTS SUMMARY (best classifier per configuration)")
    print("=" * 100)
    header = f"{'Combo':<18} {'#Ch':>3} {'SeqLen':>6} {'Clf':<16} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6} {'EER':>6}"
    print(header)
    print("-" * 100)

    # Sort by (n_channels, seq_len)
    sorted_keys = sorted(groups.keys(), key=lambda k: (len(k[0]), k[1]))

    for key in sorted_keys:
        group = groups[key]
        best = max(group, key=lambda r: r.f1)
        print(
            f"{best.combo_id:<18} {best.n_channels:>3} {best.seq_len:>6} "
            f"{best.classifier:<16} {best.accuracy:>6.4f} {best.f1:>6.4f} "
            f"{best.precision:>6.4f} {best.recall:>6.4f} {best.roc_auc:>6.4f} "
            f"{best.eer:>6.4f}"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
