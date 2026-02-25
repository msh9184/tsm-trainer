"""Ablation sweep for APC occupancy zero-shot classification.

Systematically evaluates combinations of:
  - Transformer layers (0-5)
  - Channel subsets (single, pairs, all)
  - Hybrid features (stat + MantisV2 embeddings)
  - Threshold optimization for class imbalance

Usage:
    cd examples/classification/apc_occupancy
    python training/ablation_sweep.py --config training/configs/p4-zeroshot.yaml 2>&1 | tee /tmp/ablation.log
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from data.preprocess import PreprocessConfig, load_sensor_and_labels
from data.dataset import DatasetConfig, OccupancyDataset

logger = logging.getLogger(__name__)


def extract_stat_features(X: np.ndarray) -> np.ndarray:
    """Per-channel statistical features: mean, std, min, max, range, median, zero_frac, energy."""
    n, c, l = X.shape
    feats = []
    for ch in range(c):
        data = X[:, ch, :]
        feats.extend([
            np.mean(data, axis=1),
            np.std(data, axis=1),
            np.min(data, axis=1),
            np.max(data, axis=1),
            np.max(data, axis=1) - np.min(data, axis=1),
            np.median(data, axis=1),
            (data == 0).mean(axis=1),
            np.mean(data ** 2, axis=1),
        ])
    return np.column_stack(feats)


def optimal_threshold_f1(y_true, y_prob):
    """Find threshold that maximizes F1 score."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t
    return best_t, best_f1


def run_sweep(config_path: str):
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # --- Load full sensor data ---
    _preprocess_fields = set(PreprocessConfig.__dataclass_fields__)
    base_cfg_dict = {k: v for k, v in raw.get("data", {}).items() if k in _preprocess_fields}

    # Remove channel filter to load ALL channels
    all_cfg = dict(base_cfg_dict)
    all_cfg.pop("channels", None)
    all_cfg["nan_threshold"] = 0.3  # Stricter to get cleaner channels

    train_cfg = PreprocessConfig(**all_cfg)
    sensor_all, train_labels, all_channels, sensor_ts, _ = load_sensor_and_labels(train_cfg)

    test_cfg_dict = dict(all_cfg)
    test_cfg_dict["label_csv"] = raw["data"].get("test_label_csv", "")
    test_cfg = PreprocessConfig(**test_cfg_dict)
    _, test_labels, _, _, _ = load_sensor_and_labels(test_cfg)

    ds_cfg = DatasetConfig(**raw.get("dataset", {}))

    print("=" * 80)
    print("APC OCCUPANCY ABLATION SWEEP")
    print("=" * 80)
    print(f"Available channels ({len(all_channels)}): {all_channels}")
    print(f"Sensor shape: {sensor_all.shape}")

    # --- Channel groups to test ---
    selected_4ch = [
        "d620900d_motionSensor",
        "f2e891c6_powerMeter",
        "d620900d_temperatureMeasurement",
        "ccea734e_temperatureMeasurement",
    ]

    channel_groups = {
        "motion_only": ["d620900d_motionSensor"],
        "power_only": ["f2e891c6_powerMeter"],
        "motion+power": ["d620900d_motionSensor", "f2e891c6_powerMeter"],
        "motion+power+temp1": [
            "d620900d_motionSensor", "f2e891c6_powerMeter",
            "d620900d_temperatureMeasurement",
        ],
        "4ch_baseline": selected_4ch,
    }

    # Map channel names to indices
    ch_name_to_idx = {name: i for i, name in enumerate(all_channels)}

    # --- MantisV2 setup ---
    device = raw.get("model", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    output_token = raw.get("model", {}).get("output_token", "combined")

    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer

    # --- Sweep: Layers × Channel Groups ---
    layers_to_test = [0, 1, 2, 3, 4, 5]

    results = []

    print(f"\n{'='*80}")
    print("PART 1: LAYER × CHANNEL SWEEP (RF classifier)")
    print(f"{'='*80}")
    print(f"\n{'Layer':>5} {'Channels':<25} {'#Ch':>4} {'Emb':>6} "
          f"{'TrainAcc':>9} {'TestAcc':>8} {'TestAUC':>8} {'OptF1':>6} {'OptThr':>7}")
    print("-" * 90)

    for layer in layers_to_test:
        # Load model for this layer
        network = MantisV2(
            device=device,
            return_transf_layer=layer,
            output_token=output_token,
        )
        network = network.from_pretrained(
            raw.get("model", {}).get("pretrained_name", "paris-noah/MantisV2")
        )
        model = MantisTrainer(device=device, network=network)

        for ch_name, ch_list in channel_groups.items():
            # Check all channels exist
            missing = [c for c in ch_list if c not in ch_name_to_idx]
            if missing:
                print(f"  {layer:>5} {ch_name:<25} SKIP (missing: {missing})")
                continue

            ch_indices = [ch_name_to_idx[c] for c in ch_list]
            sensor_subset = sensor_all[:, ch_indices]

            # Create datasets
            train_ds = OccupancyDataset(sensor_subset, train_labels, ds_cfg)
            test_ds = OccupancyDataset(sensor_subset, test_labels, ds_cfg)

            X_train, y_train = train_ds.get_numpy_arrays()
            X_test, y_test = test_ds.get_numpy_arrays()

            # Extract embeddings
            Z_train = model.transform(X_train)
            Z_test = model.transform(X_test)

            # RF classifier
            rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(Z_train, y_train)

            train_pred = rf.predict(Z_train)
            test_pred = rf.predict(Z_test)
            test_prob = rf.predict_proba(Z_test)[:, 1]

            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            test_auc = roc_auc_score(y_test, test_prob)

            opt_thr, opt_f1 = optimal_threshold_f1(y_test, test_prob)

            print(f"  {layer:>5} {ch_name:<25} {len(ch_list):>4} {Z_train.shape[1]:>6} "
                  f"{train_acc:>9.4f} {test_acc:>8.4f} {test_auc:>8.4f} {opt_f1:>6.4f} {opt_thr:>7.2f}")

            results.append({
                "layer": layer, "channels": ch_name, "n_ch": len(ch_list),
                "emb_dim": Z_train.shape[1], "train_acc": train_acc,
                "test_acc": test_acc, "test_auc": test_auc,
                "opt_f1": opt_f1, "opt_thr": opt_thr,
            })

        del network, model
        torch.cuda.empty_cache()

    # --- PART 2: Hybrid Features (Best Layer + Stat) ---
    print(f"\n{'='*80}")
    print("PART 2: HYBRID FEATURES (Stat + MantisV2 Embedding)")
    print(f"{'='*80}")

    # Find best layer from results
    best_result = max(results, key=lambda r: r["test_auc"])
    best_layer = best_result["layer"]
    print(f"\nBest layer from sweep: Layer {best_layer} (AUC={best_result['test_auc']:.4f})")

    # Reload best model
    network = MantisV2(
        device=device, return_transf_layer=best_layer, output_token=output_token,
    )
    network = network.from_pretrained(
        raw.get("model", {}).get("pretrained_name", "paris-noah/MantisV2")
    )
    model = MantisTrainer(device=device, network=network)

    print(f"\n{'Method':<40} {'TestAcc':>8} {'TestAUC':>8} {'OptF1':>6} {'OptThr':>7}")
    print("-" * 70)

    for ch_name, ch_list in channel_groups.items():
        ch_indices = [ch_name_to_idx[c] for c in ch_list if c in ch_name_to_idx]
        if not ch_indices:
            continue

        sensor_subset = sensor_all[:, ch_indices]
        train_ds = OccupancyDataset(sensor_subset, train_labels, ds_cfg)
        test_ds = OccupancyDataset(sensor_subset, test_labels, ds_cfg)
        X_train, y_train = train_ds.get_numpy_arrays()
        X_test, y_test = test_ds.get_numpy_arrays()

        # MantisV2 embeddings
        Z_train = model.transform(X_train)
        Z_test = model.transform(X_test)

        # Stat features (from raw windows before interpolation)
        S_train = extract_stat_features(train_ds.windows)
        S_test = extract_stat_features(test_ds.windows)

        # Normalize stat features
        scaler = StandardScaler()
        S_train_s = scaler.fit_transform(S_train)
        S_test_s = scaler.transform(S_test)

        # MantisV2 only
        rf_m = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf_m.fit(Z_train, y_train)
        prob_m = rf_m.predict_proba(Z_test)[:, 1]
        auc_m = roc_auc_score(y_test, prob_m)
        acc_m = accuracy_score(y_test, rf_m.predict(Z_test))
        thr_m, f1_m = optimal_threshold_f1(y_test, prob_m)
        print(f"  MantisV2 L{best_layer} {ch_name:<27} {acc_m:>8.4f} {auc_m:>8.4f} {f1_m:>6.4f} {thr_m:>7.2f}")

        # Stat only
        rf_s = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf_s.fit(S_train_s, y_train)
        prob_s = rf_s.predict_proba(S_test_s)[:, 1]
        auc_s = roc_auc_score(y_test, prob_s)
        acc_s = accuracy_score(y_test, rf_s.predict(S_test_s))
        thr_s, f1_s = optimal_threshold_f1(y_test, prob_s)
        print(f"  Stat only {ch_name:<29} {acc_s:>8.4f} {auc_s:>8.4f} {f1_s:>6.4f} {thr_s:>7.2f}")

        # Hybrid: concat stat + MantisV2
        H_train = np.hstack([S_train_s, Z_train])
        H_test = np.hstack([S_test_s, Z_test])
        rf_h = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf_h.fit(H_train, y_train)
        prob_h = rf_h.predict_proba(H_test)[:, 1]
        auc_h = roc_auc_score(y_test, prob_h)
        acc_h = accuracy_score(y_test, rf_h.predict(H_test))
        thr_h, f1_h = optimal_threshold_f1(y_test, prob_h)
        print(f"  Hybrid (Stat+MantisV2) {ch_name:<16} {acc_h:>8.4f} {auc_h:>8.4f} {f1_h:>6.4f} {thr_h:>7.2f}")
        print()

    del network, model
    torch.cuda.empty_cache()

    # --- PART 3: Top-5 Results ---
    print(f"\n{'='*80}")
    print("TOP 5 CONFIGURATIONS BY TEST AUC")
    print(f"{'='*80}")
    sorted_results = sorted(results, key=lambda r: r["test_auc"], reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        print(f"  #{i+1}  Layer={r['layer']}  Channels={r['channels']:<25}  "
              f"AUC={r['test_auc']:.4f}  Acc={r['test_acc']:.4f}  OptF1={r['opt_f1']:.4f}")

    # --- Save results ---
    import json
    out_path = Path("results/ablation_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APC Occupancy Ablation Sweep")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_sweep(args.config)
