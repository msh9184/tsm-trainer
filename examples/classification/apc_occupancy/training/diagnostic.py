"""Diagnostic script for APC occupancy classification pipeline.

Validates whether the data contains discriminative signal and whether
MantisV2 embeddings capture that signal. Compares against a simple
statistical baseline to isolate the bottleneck.

Usage:
    cd examples/classification/apc_occupancy
    python training/diagnostic.py --config training/configs/p4-zeroshot.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import yaml

import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from data.preprocess import PreprocessConfig, load_sensor_and_labels
from data.dataset import DatasetConfig, OccupancyDataset, create_datasets_from_splits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistical feature extraction (MantisV2-free baseline)
# ---------------------------------------------------------------------------

def extract_stat_features(X: np.ndarray) -> np.ndarray:
    """Extract simple statistical features per window.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_channels, seq_len)

    Returns
    -------
    features : np.ndarray, shape (n_samples, n_channels * 8)
        Per-channel: mean, std, min, max, range, median,
        zero_fraction, energy (mean of x^2).
    """
    n, c, l = X.shape
    feats = []
    for ch in range(c):
        data = X[:, ch, :]  # (n, l)
        feats.append(np.mean(data, axis=1))
        feats.append(np.std(data, axis=1))
        feats.append(np.min(data, axis=1))
        feats.append(np.max(data, axis=1))
        feats.append(np.max(data, axis=1) - np.min(data, axis=1))
        feats.append(np.median(data, axis=1))
        feats.append((data == 0).mean(axis=1))  # zero fraction
        feats.append(np.mean(data ** 2, axis=1))  # energy
    return np.column_stack(feats)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics(config_path: str):
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    print("=" * 70)
    print("APC OCCUPANCY DIAGNOSTIC REPORT")
    print("=" * 70)

    # --- Load data ---
    _preprocess_fields = set(PreprocessConfig.__dataclass_fields__)
    base_cfg = {k: v for k, v in raw.get("data", {}).items() if k in _preprocess_fields}

    train_cfg = PreprocessConfig(**base_cfg)
    sensor_array, train_labels, channel_names, sensor_ts, _ = load_sensor_and_labels(train_cfg)

    test_cfg_dict = dict(base_cfg)
    test_cfg_dict["label_csv"] = raw["data"].get("test_label_csv", "")
    test_cfg = PreprocessConfig(**test_cfg_dict)
    _, test_labels, _, _, _ = load_sensor_and_labels(test_cfg)

    ds_cfg = DatasetConfig(**raw.get("dataset", {}))
    train_ds = OccupancyDataset(sensor_array, train_labels, ds_cfg)
    test_ds = OccupancyDataset(sensor_array, test_labels, ds_cfg)

    X_train, y_train = train_ds.get_numpy_arrays()
    X_test, y_test = test_ds.get_numpy_arrays()

    print(f"\nData shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Train labels: {{0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}}}")
    print(f"Test labels:  {{0: {(y_test==0).sum()}, 1: {(y_test==1).sum()}}}")

    # --- Diagnostic 1: Statistical Feature Baseline ---
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1: STATISTICAL FEATURE BASELINE (no MantisV2)")
    print("=" * 70)

    # Use raw (non-interpolated) windows for stats
    X_train_raw = train_ds.windows  # (N, C, seq_len) before interpolation
    X_test_raw = test_ds.windows

    F_train = extract_stat_features(X_train_raw)
    F_test = extract_stat_features(X_test_raw)
    print(f"Stat features: train={F_train.shape}, test={F_test.shape}")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import NearestCentroid
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score

    scaler = StandardScaler()
    F_train_s = scaler.fit_transform(F_train)
    F_test_s = scaler.transform(F_test)

    stat_classifiers = {
        "NearestCentroid": NearestCentroid(),
        "RandomForest(200)": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "SVM(RBF)": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
    }

    print(f"\n{'Classifier':<25} {'Train Acc':>10} {'Test Acc':>10} {'Test AUC':>10}")
    print("-" * 60)
    for name, clf in stat_classifiers.items():
        clf.fit(F_train_s, y_train)
        train_pred = clf.predict(F_train_s)
        test_pred = clf.predict(F_test_s)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        if hasattr(clf, "predict_proba"):
            test_prob = clf.predict_proba(F_test_s)[:, 1]
            test_auc = roc_auc_score(y_test, test_prob)
        else:
            test_auc = float("nan")
        print(f"{name:<25} {train_acc:>10.4f} {test_acc:>10.4f} {test_auc:>10.4f}")

    # --- Diagnostic 2: MantisV2 Embeddings Analysis ---
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: MantisV2 EMBEDDING ANALYSIS")
    print("=" * 70)

    model_cfg = raw.get("model", {})
    device = model_cfg.get("device", "cuda")
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer

    network = MantisV2(
        device=device,
        return_transf_layer=model_cfg.get("return_transf_layer", 2),
        output_token=model_cfg.get("output_token", "combined"),
    )
    network = network.from_pretrained(model_cfg.get("pretrained_name", "paris-noah/MantisV2"))
    model = MantisTrainer(device=device, network=network)

    Z_train = model.transform(X_train)
    Z_test = model.transform(X_test)

    print(f"Embeddings: train={Z_train.shape}, test={Z_test.shape}")

    # 2a: Train accuracy with MantisV2 embeddings
    print(f"\n--- 2a: MantisV2 Embedding Classifier Performance ---")
    print(f"{'Classifier':<25} {'Train Acc':>10} {'Test Acc':>10} {'Test AUC':>10}")
    print("-" * 60)

    emb_classifiers = {
        "NearestCentroid": NearestCentroid(),
        "RandomForest(200)": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "SVM(RBF)": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
    }

    for name, clf in emb_classifiers.items():
        clf.fit(Z_train, y_train)
        train_pred = clf.predict(Z_train)
        test_pred = clf.predict(Z_test)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        if hasattr(clf, "predict_proba"):
            test_prob = clf.predict_proba(Z_test)[:, 1]
            test_auc = roc_auc_score(y_test, test_prob)
        else:
            test_auc = float("nan")
        print(f"{name:<25} {train_acc:>10.4f} {test_acc:>10.4f} {test_auc:>10.4f}")

    # 2b: Per-class embedding statistics
    print(f"\n--- 2b: Per-class Embedding Statistics ---")
    for cls in [0, 1]:
        cls_name = "Empty" if cls == 0 else "Occupied"
        mask_train = y_train == cls
        mask_test = y_test == cls
        z_tr = Z_train[mask_train]
        z_te = Z_test[mask_test]
        print(f"\n  {cls_name} (class={cls}):")
        print(f"    Train: n={mask_train.sum()}, mean_norm={np.linalg.norm(z_tr, axis=1).mean():.2f}, "
              f"std_norm={np.linalg.norm(z_tr, axis=1).std():.2f}")
        print(f"    Test:  n={mask_test.sum()}, mean_norm={np.linalg.norm(z_te, axis=1).mean():.2f}, "
              f"std_norm={np.linalg.norm(z_te, axis=1).std():.2f}")

    # 2c: Class centroid distance
    c0_train = Z_train[y_train == 0].mean(axis=0)
    c1_train = Z_train[y_train == 1].mean(axis=0)
    dist_l2 = np.linalg.norm(c0_train - c1_train)
    cosine_sim = np.dot(c0_train, c1_train) / (np.linalg.norm(c0_train) * np.linalg.norm(c1_train))
    print(f"\n--- 2c: Class Centroid Distance (Train) ---")
    print(f"  L2 distance:    {dist_l2:.4f}")
    print(f"  Cosine similarity: {cosine_sim:.6f}")
    print(f"  (cosine sim ≈ 1.0 means centroids nearly identical → no separability)")

    # 2d: Train-Test domain shift
    print(f"\n--- 2d: Train-Test Domain Shift ---")
    train_centroid = Z_train.mean(axis=0)
    test_centroid = Z_test.mean(axis=0)
    shift_l2 = np.linalg.norm(train_centroid - test_centroid)
    shift_cosine = np.dot(train_centroid, test_centroid) / (np.linalg.norm(train_centroid) * np.linalg.norm(test_centroid))
    print(f"  Train→Test centroid L2:     {shift_l2:.4f}")
    print(f"  Train→Test centroid cosine: {shift_cosine:.6f}")
    print(f"  Intra-class L2:             {dist_l2:.4f}")
    print(f"  (domain shift > intra-class distance = temporal position dominates class)")

    # 2e: Variance explained by class vs temporal position
    print(f"\n--- 2e: Variance Decomposition ---")
    # Between-class variance
    overall_mean = Z_train.mean(axis=0)
    bss = sum(
        (y_train == c).sum() * np.sum((Z_train[y_train == c].mean(axis=0) - overall_mean) ** 2)
        for c in [0, 1]
    )
    tss = np.sum((Z_train - overall_mean) ** 2)
    class_var_ratio = bss / tss if tss > 0 else 0
    print(f"  Between-class SS / Total SS = {class_var_ratio:.6f}")
    print(f"  (class explains {class_var_ratio*100:.4f}% of embedding variance)")
    print(f"  → Near 0% means class label is NOT reflected in embedding space")

    # --- Diagnostic 3: Per-channel Ablation ---
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3: PER-CHANNEL EMBEDDING ANALYSIS")
    print("=" * 70)

    n_channels = X_train.shape[1]
    channel_names_list = raw["data"].get("channels", [f"ch{i}" for i in range(n_channels)])

    per_ch_dim = Z_train.shape[1] // n_channels

    print(f"\nPer-channel embedding dim: {per_ch_dim}, Total: {Z_train.shape[1]}")
    print(f"\n{'Channel':<40} {'Centroid L2':>12} {'Cosine':>10} {'RF Test Acc':>12} {'RF Test AUC':>12}")
    print("-" * 90)

    for ch_idx in range(n_channels):
        ch_name = channel_names_list[ch_idx] if ch_idx < len(channel_names_list) else f"ch{ch_idx}"
        start = ch_idx * per_ch_dim
        end = start + per_ch_dim

        z_tr_ch = Z_train[:, start:end]
        z_te_ch = Z_test[:, start:end]

        c0 = z_tr_ch[y_train == 0].mean(axis=0)
        c1 = z_tr_ch[y_train == 1].mean(axis=0)
        ch_l2 = np.linalg.norm(c0 - c1)
        ch_cos = np.dot(c0, c1) / (np.linalg.norm(c0) * np.linalg.norm(c1) + 1e-8)

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(z_tr_ch, y_train)
        ch_test_pred = rf.predict(z_te_ch)
        ch_test_acc = accuracy_score(y_test, ch_test_pred)
        ch_test_prob = rf.predict_proba(z_te_ch)[:, 1]
        ch_test_auc = roc_auc_score(y_test, ch_test_prob)

        print(f"{ch_name:<40} {ch_l2:>12.4f} {ch_cos:>10.6f} {ch_test_acc:>12.4f} {ch_test_auc:>12.4f}")

    # --- Diagnostic 4: Different Layer Analysis ---
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 4: TRANSFORMER LAYER COMPARISON")
    print("=" * 70)

    print(f"\n{'Layer':>6} {'Emb Dim':>8} {'Centroid L2':>12} {'RF Test Acc':>12} {'RF Test AUC':>12}")
    print("-" * 55)

    for layer in [0, 1, 2, 3, 4, 5]:
        try:
            net_l = MantisV2(
                device=device,
                return_transf_layer=layer,
                output_token=model_cfg.get("output_token", "combined"),
            )
            net_l = net_l.from_pretrained(model_cfg.get("pretrained_name", "paris-noah/MantisV2"))
            model_l = MantisTrainer(device=device, network=net_l)

            z_tr_l = model_l.transform(X_train)
            z_te_l = model_l.transform(X_test)

            c0_l = z_tr_l[y_train == 0].mean(axis=0)
            c1_l = z_tr_l[y_train == 1].mean(axis=0)
            l2_l = np.linalg.norm(c0_l - c1_l)

            rf_l = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_l.fit(z_tr_l, y_train)
            pred_l = rf_l.predict(z_te_l)
            prob_l = rf_l.predict_proba(z_te_l)[:, 1]
            acc_l = accuracy_score(y_test, pred_l)
            auc_l = roc_auc_score(y_test, prob_l)

            print(f"{layer:>6} {z_tr_l.shape[1]:>8} {l2_l:>12.4f} {acc_l:>12.4f} {auc_l:>12.4f}")

            del net_l, model_l, z_tr_l, z_te_l
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{layer:>6} ERROR: {e}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print("""
Key questions this diagnostic answers:

1. STAT BASELINE vs MANTISV2:
   - If stat baseline >> MantisV2 → Data has signal, MantisV2 destroys it
   - If stat baseline ≈ MantisV2 ≈ 50% → Data itself lacks signal (unlikely)

2. TRAIN ACC:
   - If train acc ≈ 50% → Embeddings have zero class signal
   - If train acc >> test acc → Temporal domain shift problem

3. CLASS CENTROID DISTANCE:
   - Cosine sim ≈ 1.0 → Classes are indistinguishable in embedding space
   - Large L2 distance → Classes are separable (classifier issue)

4. PER-CHANNEL ANALYSIS:
   - Identifies which sensor channel has the most discriminative embedding
   - Helps select optimal channel combinations

5. LAYER COMPARISON:
   - Different transformer layers capture different abstraction levels
   - Earlier layers may retain more signal for out-of-distribution tasks
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APC Occupancy Diagnostic")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_diagnostics(args.config)
