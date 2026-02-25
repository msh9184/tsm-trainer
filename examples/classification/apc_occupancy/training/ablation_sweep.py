"""Comprehensive zero-shot ablation sweep for APC occupancy classification.

Five-part systematic evaluation:
  Part 1: Layer x Individual channel — per-channel discriminative power
  Part 2: Layer x Channel combinations — multi-channel fusion effect
  Part 3: Multi-layer fusion — embedding aggregation across transformer layers
  Part 4: Hybrid features — statistical + MantisV2 embedding concatenation
  Part 5: Threshold sensitivity — decision boundary robustness analysis

Output structure (results/ablation/):
  config.json                     Experiment configuration
  part1_single_channel/           Layer x Channel results + heatmaps + t-SNE
  part2_channel_combos/           Channel subset results + heatmaps + t-SNE
  part3_multilayer_fusion/        Multi-layer concat results + bar chart
  part4_hybrid/                   Stat vs MantisV2 vs Hybrid comparison
  part5_threshold/                Threshold sensitivity curves + ROC overlay
  summary/                        Top-20 results, full CSV, statistics

Usage:
    cd examples/classification/apc_occupancy

    # Full sweep (all parts, all plots)
    python training/ablation_sweep.py --config training/configs/p4-zeroshot.yaml

    # Skip plots (faster, no matplotlib)
    python training/ablation_sweep.py --config training/configs/p4-zeroshot.yaml --no-plots

    # Run specific parts only
    python training/ablation_sweep.py --config training/configs/p4-zeroshot.yaml --parts 1 3 5

    # Skip t-SNE only (heatmaps still generated)
    python training/ablation_sweep.py --config training/configs/p4-zeroshot.yaml --no-tsne
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from data.preprocess import PreprocessConfig, load_sensor_and_labels
from data.dataset import DatasetConfig, OccupancyDataset

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

LAYERS = [0, 1, 2, 3, 4, 5]

KEY_CHANNELS = [
    "d620900d_motionSensor",
    "f2e891c6_powerMeter",
    "d620900d_temperatureMeasurement",
    "ccea734e_temperatureMeasurement",
]

CH_SHORT: dict[str, str] = {
    "d620900d_motionSensor": "M",
    "f2e891c6_powerMeter": "P",
    "d620900d_temperatureMeasurement": "T1",
    "ccea734e_temperatureMeasurement": "T2",
}

RESULT_FIELDS = [
    "part", "layer", "channels", "n_ch", "emb_dim",
    "train_acc", "test_acc", "test_auc", "test_eer", "eer_threshold",
    "test_f1", "test_precision", "test_recall", "opt_f1", "opt_threshold",
    "n_train", "n_test",
]


def short_name(ch: str) -> str:
    """Short display name for a channel."""
    if ch in CH_SHORT:
        return CH_SHORT[ch]
    parts = ch.split("_", 1)
    return parts[-1][:10] if len(parts) == 2 else ch[:10]


def combo_label(channels: list[str]) -> str:
    """Create short combo name from channel list."""
    return "+".join(short_name(c) for c in channels)


# ============================================================================
# Metric computation
# ============================================================================

def compute_eer(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate and its threshold."""
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_thr = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, eer_thr


def optimal_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Find threshold maximizing F1."""
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1


def run_experiment(
    Z_train: np.ndarray, y_train: np.ndarray,
    Z_test: np.ndarray, y_test: np.ndarray,
    seed: int = 42,
) -> dict:
    """Run RF classifier, compute all metrics, return dict with raw predictions."""
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(Z_train, y_train)

    train_pred = rf.predict(Z_train)
    test_pred = rf.predict(Z_test)
    test_prob = rf.predict_proba(Z_test)[:, 1]

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    try:
        test_auc = roc_auc_score(y_test, test_prob)
    except ValueError:
        test_auc = float("nan")

    test_eer, eer_thr = compute_eer(y_test, test_prob)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    test_prec = precision_score(y_test, test_pred, zero_division=0)
    test_rec = recall_score(y_test, test_pred, zero_division=0)
    opt_thr, opt_f1 = optimal_threshold_f1(y_test, test_prob)

    try:
        fpr, tpr, thresholds = roc_curve(y_test, test_prob)
    except ValueError:
        fpr, tpr, thresholds = None, None, None

    return {
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "test_auc": float(test_auc),
        "test_eer": float(test_eer),
        "eer_threshold": float(eer_thr),
        "test_f1": float(test_f1),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "opt_f1": float(opt_f1),
        "opt_threshold": float(opt_thr),
        # Raw data (prefixed _ = not serialized to JSON)
        "_test_prob": test_prob,
        "_roc_fpr": fpr,
        "_roc_tpr": tpr,
        "_roc_thresholds": thresholds,
    }


def extract_stat_features(X: np.ndarray) -> np.ndarray:
    """Per-channel stat features: mean, std, min, max, range, median, zero_frac, energy.

    Args:
        X: (n_samples, n_channels, seq_len).

    Returns:
        (n_samples, n_channels * 8).
    """
    n, c, length = X.shape
    feats = []
    for ch in range(c):
        data = X[:, ch, :]
        feats.extend([
            np.mean(data, axis=1),
            np.std(data, axis=1),
            np.min(data, axis=1),
            np.max(data, axis=1),
            np.ptp(data, axis=1),
            np.median(data, axis=1),
            (data == 0).mean(axis=1),
            np.mean(data ** 2, axis=1),
        ])
    return np.column_stack(feats)


# ============================================================================
# Result I/O
# ============================================================================

def make_result(
    part: str, layer: str, channels: str, n_ch: int,
    emb_dim: int, metrics: dict, n_train: int, n_test: int,
) -> dict:
    """Create standardised result dict."""
    return {
        "part": part,
        "layer": layer,
        "channels": channels,
        "n_ch": n_ch,
        "emb_dim": emb_dim,
        **{k: metrics[k] for k in RESULT_FIELDS[5:]
           if k in metrics},
        "train_acc": metrics["train_acc"],
        "test_acc": metrics["test_acc"],
        "test_auc": metrics["test_auc"],
        "test_eer": metrics["test_eer"],
        "eer_threshold": metrics["eer_threshold"],
        "test_f1": metrics["test_f1"],
        "test_precision": metrics["test_precision"],
        "test_recall": metrics["test_recall"],
        "opt_f1": metrics["opt_f1"],
        "opt_threshold": metrics["opt_threshold"],
        "n_train": n_train,
        "n_test": n_test,
    }


def _nan_safe(v):
    """Convert NaN/Inf to None for JSON serialisation."""
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def save_json(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = [
        {k: _nan_safe(v) for k, v in r.items() if not k.startswith("_")}
        for r in results
    ]
    with open(path, "w") as f:
        json.dump(cleaned, f, indent=2)


def save_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in RESULT_FIELDS})


# ============================================================================
# Visualization helpers
# ============================================================================

_PLT = None


def _get_plt():
    global _PLT
    if _PLT is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "figure.dpi": 150, "savefig.dpi": 300,
            "font.size": 10, "axes.titlesize": 12,
            "axes.labelsize": 10, "xtick.labelsize": 8,
            "ytick.labelsize": 8, "legend.fontsize": 8,
            "figure.figsize": (8, 6),
            "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
        })
        _PLT = plt
    return _PLT


def plot_tsne(
    Z: np.ndarray, y: np.ndarray, title: str, output_path: Path,
) -> None:
    """t-SNE scatter plot for test embeddings."""
    plt = _get_plt()
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    n_components = min(50, Z.shape[1], Z.shape[0] - 1)
    Z_pca = PCA(n_components=n_components, random_state=42).fit_transform(Z) if Z.shape[1] > n_components else Z

    perp = min(30, max(5, len(Z) // 4))
    Z_2d = TSNE(n_components=2, random_state=42, max_iter=1000, perplexity=perp).fit_transform(Z_pca)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for cls, color, label in [(0, "#0173B2", "Empty"), (1, "#DE8F05", "Occupied")]:
        mask = y == cls
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=color, label=label,
                   alpha=0.5, s=8, edgecolors="none")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", markerscale=2)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_heatmap(
    matrix: np.ndarray, row_labels: list, col_labels: list,
    title: str, output_path: Path,
    metric_name: str = "AUC", cmap: str = "RdYlGn",
) -> None:
    """Layer x Channel heatmap."""
    plt = _get_plt()

    h = max(3.5, len(row_labels) * 0.55)
    w = max(6, len(col_labels) * 0.7 + 1.5)
    fig, ax = plt.subplots(figsize=(w, h))

    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            txt = "-" if np.isnan(val) else f"{val:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color="white" if np.isnan(val) or val < np.nanpercentile(matrix, 25) else "black")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_ylabel("Transformer Layer")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label=metric_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_roc_overlay(roc_items: list[dict], output_path: Path) -> None:
    """Overlay multiple ROC curves."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(7, 6))

    for item in roc_items:
        if item.get("fpr") is None:
            continue
        ax.plot(item["fpr"], item["tpr"],
                label=f"{item['name']} (AUC={item['auc']:.3f})", linewidth=1.3)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Top Configurations")
    ax.legend(loc="lower right", fontsize=6)
    ax.grid(alpha=0.15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_threshold_curves(items: list[dict], output_path: Path) -> None:
    """F1 / Accuracy / Precision / Recall vs threshold for multiple configs."""
    plt = _get_plt()
    thresholds = np.arange(0.05, 0.96, 0.02)

    metric_fns = [
        ("F1", lambda yt, yp, t: f1_score(yt, (yp >= t).astype(int), zero_division=0)),
        ("Accuracy", lambda yt, yp, t: accuracy_score(yt, (yp >= t).astype(int))),
        ("Precision", lambda yt, yp, t: precision_score(yt, (yp >= t).astype(int), zero_division=0)),
        ("Recall", lambda yt, yp, t: recall_score(yt, (yp >= t).astype(int), zero_division=0)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (mname, mfn) in zip(axes.flat, metric_fns):
        for item in items:
            vals = [mfn(item["y_true"], item["y_prob"], t) for t in thresholds]
            ax.plot(thresholds, vals, label=item["name"], linewidth=1.2)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(mname)
        ax.set_title(f"{mname} vs Threshold")
        ax.legend(fontsize=5, loc="best")
        ax.grid(alpha=0.15)
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.3)

    fig.suptitle("Threshold Sensitivity Analysis", fontsize=13)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_bar(data: list[dict], output_path: Path, metric: str = "test_auc") -> None:
    """Horizontal bar chart comparison."""
    plt = _get_plt()
    data_sorted = sorted(data, key=lambda d: d.get(metric, 0))

    names = [d["name"] for d in data_sorted]
    values = [d.get(metric, 0) for d in data_sorted]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.3)))
    bars = ax.barh(range(len(names)), values, color="#0173B2", alpha=0.8, height=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} Comparison")
    ax.grid(axis="x", alpha=0.15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


# ============================================================================
# Data loading
# ============================================================================

def load_data(config_path: str):
    """Load sensor data and labels, returning all numeric channels.

    Returns:
        sensor_all, train_labels, test_labels, all_channels, ds_cfg, raw_config
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    _fields = set(PreprocessConfig.__dataclass_fields__)
    base = {k: v for k, v in raw.get("data", {}).items() if k in _fields}

    # Load ALL numeric channels (remove explicit channel filter)
    all_cfg = dict(base)
    all_cfg.pop("channels", None)
    all_cfg["nan_threshold"] = 0.3

    train_cfg = PreprocessConfig(**all_cfg)
    sensor_all, train_labels, all_channels, sensor_ts, _ = load_sensor_and_labels(train_cfg)

    test_cfg_dict = dict(all_cfg)
    test_cfg_dict["label_csv"] = raw["data"].get("test_label_csv", "")
    test_cfg = PreprocessConfig(**test_cfg_dict)
    _, test_labels, _, _, _ = load_sensor_and_labels(test_cfg)

    # Resolve train/test overlap
    overlap = ((train_labels >= 0) & (test_labels >= 0)).sum()
    if overlap > 0:
        test_labels = test_labels.copy()
        test_labels[train_labels >= 0] = -1

    ds_cfg = DatasetConfig(**raw.get("dataset", {}))
    return sensor_all, train_labels, test_labels, all_channels, ds_cfg, raw


# ============================================================================
# Embedding extraction & caching
# ============================================================================

def extract_all_embeddings(
    sensor_all: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    all_channels: list[str],
    ch_name_to_idx: dict[str, int],
    ds_cfg: DatasetConfig,
    model_config: dict,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Extract single-channel embeddings for ALL layers x ALL channels.

    MantisV2 processes each channel independently, so multi-channel embeddings
    are just concatenation of single-channel embeddings. This makes caching
    single-channel results very efficient.

    Returns:
        cache: {(layer, channel_name): {"train": Z_train, "test": Z_test}}
        y_train, y_test: window labels (consistent across all channel configs)
    """
    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer

    device = model_config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    output_token = model_config.get("output_token", "combined")
    pretrained = model_config.get("pretrained_name", "paris-noah/MantisV2")

    cache = {}
    y_train = None
    y_test = None
    total = len(LAYERS) * len(all_channels)
    done = 0

    for layer in LAYERS:
        print(f"\n  Layer {layer}: Loading MantisV2...", end=" ", flush=True)
        t0 = time.time()

        network = MantisV2(device=device, return_transf_layer=layer, output_token=output_token)
        network = network.from_pretrained(pretrained)
        model = MantisTrainer(device=device, network=network)
        print(f"({time.time() - t0:.1f}s)")

        for ch_name in all_channels:
            done += 1
            ch_idx = ch_name_to_idx[ch_name]
            sensor_1ch = sensor_all[:, [ch_idx]]

            train_ds = OccupancyDataset(sensor_1ch, train_labels, ds_cfg)
            test_ds = OccupancyDataset(sensor_1ch, test_labels, ds_cfg)

            X_tr, y_tr = train_ds.get_numpy_arrays()
            X_te, y_te = test_ds.get_numpy_arrays()

            if y_train is None:
                y_train = y_tr
                y_test = y_te

            t1 = time.time()
            Z_tr = model.transform(X_tr)
            Z_te = model.transform(X_te)
            dt = time.time() - t1

            cache[(layer, ch_name)] = {"train": Z_tr, "test": Z_te}
            print(f"    [{done:>3}/{total}] L{layer} {short_name(ch_name):<12} "
                  f"emb={Z_tr.shape[1]}d  ({dt:.1f}s)")

        del network, model
        torch.cuda.empty_cache()

    return cache, y_train, y_test


def compose_embeddings(
    cache: dict, layers: list[int], channels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compose multi-channel / multi-layer embeddings by concatenation."""
    tr_parts, te_parts = [], []
    for layer in layers:
        for ch in channels:
            entry = cache[(layer, ch)]
            tr_parts.append(entry["train"])
            te_parts.append(entry["test"])
    return np.hstack(tr_parts), np.hstack(te_parts)


# ============================================================================
# Part 1: Single Channel x All Layers
# ============================================================================

def part1_single_channel(
    cache, y_train, y_test, all_channels, out_dir, do_plots, do_tsne,
):
    print(f"\n{'='*80}")
    print("PART 1: SINGLE CHANNEL x ALL LAYERS")
    print(f"{'='*80}")
    print(f"  Channels: {len(all_channels)}, Layers: {len(LAYERS)}")
    print(f"  Experiments: {len(all_channels) * len(LAYERS)}")

    results = []

    hdr = f"{'L':>2} {'Channel':<16} {'Dim':>5} {'TrAcc':>6} {'TeAcc':>6} {'AUC':>6} {'EER':>6} {'OptF1':>6} {'OptThr':>6}"
    print(f"\n  {hdr}")
    print("  " + "-" * len(hdr))

    for layer in LAYERS:
        for ch in all_channels:
            Z_tr, Z_te = compose_embeddings(cache, [layer], [ch])
            m = run_experiment(Z_tr, y_train, Z_te, y_test)
            r = make_result("part1", str(layer), short_name(ch), 1, Z_tr.shape[1],
                            m, len(y_train), len(y_test))
            r["_full_ch"] = ch
            r["_test_prob"] = m["_test_prob"]
            r["_roc_fpr"] = m["_roc_fpr"]
            r["_roc_tpr"] = m["_roc_tpr"]
            results.append(r)

            print(f"  {layer:>2} {short_name(ch):<16} {Z_tr.shape[1]:>5} "
                  f"{m['train_acc']:>6.3f} {m['test_acc']:>6.3f} "
                  f"{m['test_auc']:>6.3f} {m['test_eer']:>6.3f} "
                  f"{m['opt_f1']:>6.3f} {m['opt_threshold']:>6.2f}")

    # Save
    d = out_dir / "part1_single_channel"
    save_json(results, d / "results.json")
    save_csv(results, d / "results.csv")

    if do_plots:
        ch_labels = [short_name(c) for c in all_channels]
        layer_labels = [f"L{l}" for l in LAYERS]

        for mname, mkey, cm in [
            ("AUC", "test_auc", "RdYlGn"),
            ("EER", "test_eer", "RdYlGn_r"),
            ("Opt-F1", "opt_f1", "RdYlGn"),
        ]:
            mat = np.full((len(LAYERS), len(all_channels)), np.nan)
            for r in results:
                li = LAYERS.index(int(r["layer"]))
                ci = ch_labels.index(r["channels"])
                mat[li, ci] = r[mkey]
            plot_heatmap(mat, layer_labels, ch_labels,
                         f"Part 1: {mname} — Single Channel x Layer",
                         d / f"heatmap_{mkey}.png", mname, cm)

    if do_tsne:
        tsne_d = d / "tsne"
        for layer in LAYERS:
            for ch in all_channels:
                _, Z_te = compose_embeddings(cache, [layer], [ch])
                r_match = [r for r in results
                           if r["layer"] == str(layer) and r["_full_ch"] == ch]
                auc_val = r_match[0]["test_auc"] if r_match else 0
                plot_tsne(Z_te, y_test,
                          f"L{layer} {short_name(ch)} (AUC={auc_val:.3f})",
                          tsne_d / f"L{layer}_{short_name(ch)}.png")

    return results


# ============================================================================
# Part 2: Channel Combinations x All Layers
# ============================================================================

def part2_channel_combos(
    cache, y_train, y_test, key_channels, out_dir, do_plots, do_tsne,
):
    print(f"\n{'='*80}")
    print("PART 2: CHANNEL COMBINATIONS x ALL LAYERS")
    print(f"{'='*80}")

    combos = []
    for r in range(1, len(key_channels) + 1):
        for subset in itertools.combinations(key_channels, r):
            combos.append(list(subset))

    print(f"  Key channels: {[short_name(c) for c in key_channels]}")
    print(f"  Combinations: {len(combos)}, Layers: {len(LAYERS)}")
    print(f"  Experiments: {len(combos) * len(LAYERS)}")

    results = []

    hdr = f"{'L':>2} {'Channels':<18} {'#':>2} {'Dim':>5} {'TrAcc':>6} {'TeAcc':>6} {'AUC':>6} {'EER':>6} {'OptF1':>6} {'OptThr':>6}"
    print(f"\n  {hdr}")
    print("  " + "-" * len(hdr))

    for layer in LAYERS:
        for ch_list in combos:
            cl = combo_label(ch_list)
            try:
                Z_tr, Z_te = compose_embeddings(cache, [layer], ch_list)
            except KeyError:
                print(f"  {layer:>2} {cl:<18} SKIP")
                continue

            m = run_experiment(Z_tr, y_train, Z_te, y_test)
            r = make_result("part2", str(layer), cl, len(ch_list), Z_tr.shape[1],
                            m, len(y_train), len(y_test))
            r["_ch_list"] = ch_list
            r["_test_prob"] = m["_test_prob"]
            r["_roc_fpr"] = m["_roc_fpr"]
            r["_roc_tpr"] = m["_roc_tpr"]
            results.append(r)

            print(f"  {layer:>2} {cl:<18} {len(ch_list):>2} {Z_tr.shape[1]:>5} "
                  f"{m['train_acc']:>6.3f} {m['test_acc']:>6.3f} "
                  f"{m['test_auc']:>6.3f} {m['test_eer']:>6.3f} "
                  f"{m['opt_f1']:>6.3f} {m['opt_threshold']:>6.2f}")

    d = out_dir / "part2_channel_combos"
    save_json(results, d / "results.json")
    save_csv(results, d / "results.csv")

    if do_plots:
        combo_labels = [combo_label(c) for c in combos]
        layer_labels = [f"L{l}" for l in LAYERS]
        for mname, mkey, cm in [
            ("AUC", "test_auc", "RdYlGn"),
            ("EER", "test_eer", "RdYlGn_r"),
            ("Opt-F1", "opt_f1", "RdYlGn"),
        ]:
            mat = np.full((len(LAYERS), len(combos)), np.nan)
            for r in results:
                li = LAYERS.index(int(r["layer"]))
                ci = combo_labels.index(r["channels"])
                mat[li, ci] = r[mkey]
            plot_heatmap(mat, layer_labels, combo_labels,
                         f"Part 2: {mname} — Channel Combos x Layer",
                         d / f"heatmap_{mkey}.png", mname, cm)

    if do_tsne:
        top10 = sorted(results, key=lambda x: x.get("test_auc", 0), reverse=True)[:10]
        tsne_d = d / "tsne"
        for r in top10:
            _, Z_te = compose_embeddings(cache, [int(r["layer"])], r["_ch_list"])
            ch_tag = r["channels"].replace("+", "_")
            plot_tsne(Z_te, y_test,
                      f"L{r['layer']} {r['channels']} (AUC={r['test_auc']:.3f})",
                      tsne_d / f"L{r['layer']}_{ch_tag}.png")

    return results


# ============================================================================
# Part 3: Multi-Layer Fusion
# ============================================================================

def part3_multilayer_fusion(
    cache, y_train, y_test, key_channels, p2_results, out_dir, do_plots, do_tsne,
):
    print(f"\n{'='*80}")
    print("PART 3: MULTI-LAYER FUSION")
    print(f"{'='*80}")

    # Best channel config from Part 2
    if p2_results:
        best_p2 = max(p2_results, key=lambda r: r.get("test_auc", 0))
        best_ch = best_p2.get("_ch_list", key_channels[:1])
    else:
        best_ch = key_channels[:1]
    best_cl = combo_label(best_ch)

    # Also test motionSensor alone if it's a key channel
    channel_configs = [(best_ch, best_cl)]
    motion_ch = [c for c in key_channels if "motionSensor" in c]
    if motion_ch and motion_ch != best_ch:
        channel_configs.append((motion_ch, combo_label(motion_ch)))

    print(f"  Channel configs: {[cl for _, cl in channel_configs]}")

    # Layer fusion strategies
    layer_combos = [
        ([0, 1], "L0+L1"), ([0, 3], "L0+L3"), ([1, 3], "L1+L3"),
        ([1, 5], "L1+L5"), ([2, 3], "L2+L3"), ([3, 5], "L3+L5"),
        ([0, 1, 3], "L0+L1+L3"), ([1, 3, 5], "L1+L3+L5"),
        ([0, 3, 5], "L0+L3+L5"),
        ([0, 2, 4], "Leven"),
        ([1, 3, 5], "Lodd"),
        ([0, 1, 2, 3, 4, 5], "Lall"),
    ]

    results = []

    hdr = f"{'Layers':<14} {'Ch':<10} {'Dim':>6} {'TrAcc':>6} {'TeAcc':>6} {'AUC':>6} {'EER':>6} {'OptF1':>6}"
    print(f"\n  {hdr}")
    print("  " + "-" * len(hdr))

    for ch_list, cl in channel_configs:
        # Single-layer baselines
        for layer in LAYERS:
            try:
                Z_tr, Z_te = compose_embeddings(cache, [layer], ch_list)
            except KeyError:
                continue
            m = run_experiment(Z_tr, y_train, Z_te, y_test)
            r = make_result("part3_base", f"L{layer}", cl, len(ch_list),
                            Z_tr.shape[1], m, len(y_train), len(y_test))
            r["_layer_list"] = [layer]
            r["_ch_list"] = ch_list
            r["_test_prob"] = m["_test_prob"]
            r["_roc_fpr"] = m["_roc_fpr"]
            r["_roc_tpr"] = m["_roc_tpr"]
            results.append(r)
            print(f"  L{layer:<13} {cl:<10} {Z_tr.shape[1]:>6} "
                  f"{m['train_acc']:>6.3f} {m['test_acc']:>6.3f} "
                  f"{m['test_auc']:>6.3f} {m['test_eer']:>6.3f} {m['opt_f1']:>6.3f}")

        print("  --- fusions ---")

        for layer_list, ll in layer_combos:
            try:
                Z_tr, Z_te = compose_embeddings(cache, layer_list, ch_list)
            except KeyError:
                continue
            m = run_experiment(Z_tr, y_train, Z_te, y_test)
            r = make_result("part3_fusion", ll, cl, len(ch_list) * len(layer_list),
                            Z_tr.shape[1], m, len(y_train), len(y_test))
            r["_layer_list"] = layer_list
            r["_ch_list"] = ch_list
            r["_test_prob"] = m["_test_prob"]
            r["_roc_fpr"] = m["_roc_fpr"]
            r["_roc_tpr"] = m["_roc_tpr"]
            results.append(r)
            print(f"  {ll:<14} {cl:<10} {Z_tr.shape[1]:>6} "
                  f"{m['train_acc']:>6.3f} {m['test_acc']:>6.3f} "
                  f"{m['test_auc']:>6.3f} {m['test_eer']:>6.3f} {m['opt_f1']:>6.3f}")

        print()

    d = out_dir / "part3_multilayer_fusion"
    save_json(results, d / "results.json")
    save_csv(results, d / "results.csv")

    if do_plots:
        bar_data = [
            {"name": f"{r['layer']}|{r['channels']}", "test_auc": r["test_auc"]}
            for r in results
        ]
        plot_bar(bar_data, d / "bar_chart_auc.png", "test_auc")

    if do_tsne:
        fusions = [r for r in results if r["part"] == "part3_fusion"]
        top5 = sorted(fusions, key=lambda x: x.get("test_auc", 0), reverse=True)[:5]
        tsne_d = d / "tsne"
        for r in top5:
            _, Z_te = compose_embeddings(cache, r["_layer_list"], r["_ch_list"])
            safe_name = r["layer"].replace("+", "_")
            plot_tsne(Z_te, y_test,
                      f"{r['layer']} {r['channels']} (AUC={r['test_auc']:.3f})",
                      tsne_d / f"{safe_name}_{r['channels']}.png")

    return results


# ============================================================================
# Part 4: Hybrid Features
# ============================================================================

def part4_hybrid_features(
    cache, y_train, y_test,
    sensor_all, train_labels, test_labels,
    key_channels, ch_name_to_idx, ds_cfg,
    out_dir, do_plots,
):
    print(f"\n{'='*80}")
    print("PART 4: HYBRID FEATURES (Stat + MantisV2)")
    print(f"{'='*80}")

    best_layer = 3  # From diagnostic analysis

    combos = []
    for r in range(1, len(key_channels) + 1):
        for subset in itertools.combinations(key_channels, r):
            combos.append(list(subset))

    results = []

    hdr = f"{'Method':<30} {'Ch':<12} {'Dim':>6} {'TeAcc':>6} {'AUC':>6} {'EER':>6} {'OptF1':>6}"
    print(f"\n  {hdr}")
    print("  " + "-" * len(hdr))

    for ch_list in combos:
        cl = combo_label(ch_list)
        ch_indices = [ch_name_to_idx[c] for c in ch_list if c in ch_name_to_idx]
        if not ch_indices:
            continue

        sensor_sub = sensor_all[:, ch_indices]
        tr_ds = OccupancyDataset(sensor_sub, train_labels, ds_cfg)
        te_ds = OccupancyDataset(sensor_sub, test_labels, ds_cfg)

        S_tr = extract_stat_features(tr_ds.windows)
        S_te = extract_stat_features(te_ds.windows)
        scaler = StandardScaler()
        S_tr_s = scaler.fit_transform(S_tr)
        S_te_s = scaler.transform(S_te)

        Z_tr, Z_te = compose_embeddings(cache, [best_layer], ch_list)

        # Stat only
        ms = run_experiment(S_tr_s, y_train, S_te_s, y_test)
        results.append(make_result("part4_stat", "stat", cl, len(ch_list),
                                   S_tr_s.shape[1], ms, len(y_train), len(y_test)))
        print(f"  {'Stat':<30} {cl:<12} {S_tr_s.shape[1]:>6} "
              f"{ms['test_acc']:>6.3f} {ms['test_auc']:>6.3f} "
              f"{ms['test_eer']:>6.3f} {ms['opt_f1']:>6.3f}")

        # MantisV2 only
        mm = run_experiment(Z_tr, y_train, Z_te, y_test)
        results.append(make_result("part4_mantis", f"L{best_layer}", cl, len(ch_list),
                                   Z_tr.shape[1], mm, len(y_train), len(y_test)))
        print(f"  {f'MantisV2 L{best_layer}':<30} {cl:<12} {Z_tr.shape[1]:>6} "
              f"{mm['test_acc']:>6.3f} {mm['test_auc']:>6.3f} "
              f"{mm['test_eer']:>6.3f} {mm['opt_f1']:>6.3f}")

        # Hybrid
        H_tr = np.hstack([S_tr_s, Z_tr])
        H_te = np.hstack([S_te_s, Z_te])
        mh = run_experiment(H_tr, y_train, H_te, y_test)
        results.append(make_result("part4_hybrid", f"stat+L{best_layer}", cl, len(ch_list),
                                   H_tr.shape[1], mh, len(y_train), len(y_test)))
        print(f"  {f'Hybrid (stat+L{best_layer})':<30} {cl:<12} {H_tr.shape[1]:>6} "
              f"{mh['test_acc']:>6.3f} {mh['test_auc']:>6.3f} "
              f"{mh['test_eer']:>6.3f} {mh['opt_f1']:>6.3f}")
        print()

    d = out_dir / "part4_hybrid"
    save_json(results, d / "results.json")
    save_csv(results, d / "results.csv")

    if do_plots:
        plt = _get_plt()

        methods = ["part4_stat", "part4_mantis", "part4_hybrid"]
        method_labels = ["Stat", f"MantisV2 L{best_layer}", "Hybrid"]
        ch_labels = sorted(set(r["channels"] for r in results))

        fig, ax = plt.subplots(figsize=(max(10, len(ch_labels) * 2.2), 5))
        x = np.arange(len(ch_labels))
        w = 0.25
        colors = ["#0173B2", "#DE8F05", "#029E73"]

        for i, (meth, ml, col) in enumerate(zip(methods, method_labels, colors)):
            vals = []
            for cl in ch_labels:
                match = [r for r in results if r["part"] == meth and r["channels"] == cl]
                vals.append(match[0]["test_auc"] if match else 0)
            bars = ax.bar(x + i * w, vals, w, label=ml, color=col, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=5)

        ax.set_xticks(x + w)
        ax.set_xticklabels(ch_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Test AUC")
        ax.set_title("Part 4: Stat vs MantisV2 vs Hybrid — AUC")
        ax.legend()
        ax.grid(axis="y", alpha=0.15)
        fig.savefig(d / "comparison_auc.png")
        plt.close(fig)

    return results


# ============================================================================
# Part 5: Threshold Sensitivity
# ============================================================================

def part5_threshold_sensitivity(
    cache, y_train, y_test, all_results, out_dir, do_plots,
):
    print(f"\n{'='*80}")
    print("PART 5: THRESHOLD SENSITIVITY ANALYSIS")
    print(f"{'='*80}")

    # Collect configs that have probability predictions
    scored = sorted(
        [r for r in all_results if r.get("_test_prob") is not None],
        key=lambda r: r.get("test_auc", 0), reverse=True,
    )

    seen = set()
    top_configs = []
    for r in scored:
        key = (r.get("part", ""), r["layer"], r["channels"])
        if key not in seen and len(top_configs) < 10:
            seen.add(key)
            top_configs.append(r)

    print(f"  Analyzing top {len(top_configs)} configurations\n")

    results = []
    threshold_data = []
    roc_items = []
    thresholds = np.arange(0.05, 0.96, 0.01)

    hdr = f"{'Config':<35} {'MaxF1':>6} {'@Thr':>5} {'Stable':>6} {'F1std':>6} {'AUC':>6} {'EER':>6}"
    print(f"  {hdr}")
    print("  " + "-" * len(hdr))

    for r in top_configs:
        name = f"{r.get('part','')[:5]}|L{r['layer']}|{r['channels']}"
        prob = r["_test_prob"]

        f1_vals = np.array([
            f1_score(y_test, (prob >= t).astype(int), zero_division=0)
            for t in thresholds
        ])

        max_f1 = float(np.max(f1_vals))
        max_f1_thr = float(thresholds[np.argmax(f1_vals)])
        f1_std = float(np.std(f1_vals))

        # Stable range: where F1 >= 95% of max
        good = thresholds[f1_vals >= 0.95 * max_f1]
        stable_range = float(good[-1] - good[0]) if len(good) > 1 else 0.0

        results.append({
            "part": "part5", "name": name,
            "layer": r["layer"], "channels": r["channels"],
            "test_auc": r["test_auc"],
            "test_eer": r.get("test_eer", float("nan")),
            "max_f1": max_f1, "max_f1_threshold": max_f1_thr,
            "f1_std": f1_std, "f1_stable_range": stable_range,
            "eer_threshold": r.get("eer_threshold", float("nan")),
        })

        threshold_data.append({
            "name": name, "y_true": y_test, "y_prob": prob,
        })

        if r.get("_roc_fpr") is not None:
            roc_items.append({
                "name": name, "fpr": r["_roc_fpr"],
                "tpr": r["_roc_tpr"], "auc": r["test_auc"],
            })

        print(f"  {name:<35} {max_f1:>6.3f} {max_f1_thr:>5.2f} "
              f"{stable_range:>6.2f} {f1_std:>6.3f} "
              f"{r['test_auc']:>6.3f} {r.get('test_eer', float('nan')):>6.3f}")

    d = out_dir / "part5_threshold"
    save_json(results, d / "results.json")

    if do_plots:
        if threshold_data:
            plot_threshold_curves(threshold_data, d / "threshold_sensitivity.png")
        if roc_items:
            plot_roc_overlay(roc_items, d / "roc_overlay.png")

    return results


# ============================================================================
# Summary
# ============================================================================

def generate_summary(all_results, out_dir):
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    main = [r for r in all_results if "part" in r and not r["part"].startswith("part5")]

    by_auc = sorted(main, key=lambda r: r.get("test_auc", 0), reverse=True)

    print(f"\n  TOP 20 BY AUC:")
    print(f"  {'#':>3} {'Part':<14} {'Layer':<10} {'Ch':<16} {'AUC':>6} {'EER':>6} {'OptF1':>6} {'Acc':>6}")
    print("  " + "-" * 80)
    for i, r in enumerate(by_auc[:20]):
        print(f"  {i+1:>3} {r['part']:<14} {r['layer']:<10} {r['channels']:<16} "
              f"{r.get('test_auc',0):>6.3f} {r.get('test_eer', float('nan')):>6.3f} "
              f"{r.get('opt_f1',0):>6.3f} {r.get('test_acc',0):>6.3f}")

    by_eer = sorted(
        [r for r in main if not np.isnan(r.get("test_eer", float("nan")))],
        key=lambda r: r.get("test_eer", 1.0),
    )

    print(f"\n  TOP 20 BY EER (lower=better):")
    print(f"  {'#':>3} {'Part':<14} {'Layer':<10} {'Ch':<16} {'EER':>6} {'EER_Thr':>8} {'AUC':>6}")
    print("  " + "-" * 80)
    for i, r in enumerate(by_eer[:20]):
        print(f"  {i+1:>3} {r['part']:<14} {r['layer']:<10} {r['channels']:<16} "
              f"{r.get('test_eer',0):>6.3f} {r.get('eer_threshold',0):>8.3f} "
              f"{r.get('test_auc',0):>6.3f}")

    # Save
    sd = out_dir / "summary"
    save_json(by_auc[:20], sd / "top20_by_auc.json")
    save_json(by_eer[:20], sd / "top20_by_eer.json")
    save_csv(main, sd / "all_experiments.csv")
    save_json(main, sd / "all_experiments.json")

    best = by_auc[0] if by_auc else {}
    best_eer = by_eer[0] if by_eer else {}

    stats = {
        "total_experiments": len(main),
        "best_auc": best.get("test_auc"),
        "best_auc_config": f"L{best.get('layer')} {best.get('channels')} ({best.get('part')})",
        "best_eer": best_eer.get("test_eer"),
        "best_eer_config": f"L{best_eer.get('layer')} {best_eer.get('channels')} ({best_eer.get('part')})",
    }
    with open(sd / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Total: {stats['total_experiments']} experiments")
    if best:
        print(f"  Best AUC: {best.get('test_auc', 0):.4f}  ({stats['best_auc_config']})")
    if best_eer:
        print(f"  Best EER: {best_eer.get('test_eer', 0):.4f}  ({stats['best_eer_config']})")


# ============================================================================
# Main orchestrator
# ============================================================================

def run_sweep(config_path: str, parts: list[int] | None = None,
              no_plots: bool = False, no_tsne: bool = False):
    """Run the complete ablation sweep."""
    print("=" * 80)
    print("APC OCCUPANCY — COMPREHENSIVE ZERO-SHOT ABLATION SWEEP")
    print("=" * 80)

    t_start = time.time()

    # --- Load data ---
    print("\n[STEP 1] Loading data...")
    sensor_all, train_labels, test_labels, all_channels, ds_cfg, raw = load_data(config_path)
    ch_to_idx = {name: i for i, name in enumerate(all_channels)}
    avail_key = [c for c in KEY_CHANNELS if c in ch_to_idx]

    print(f"  Sensor: {sensor_all.shape}")
    print(f"  Channels ({len(all_channels)}): {[short_name(c) for c in all_channels]}")
    print(f"  Key channels: {[short_name(c) for c in avail_key]}")
    print(f"  Dataset: seq_len={ds_cfg.seq_len}, stride={ds_cfg.stride}, "
          f"target_seq_len={ds_cfg.target_seq_len}")
    tr_n = (train_labels >= 0).sum()
    te_n = (test_labels >= 0).sum()
    print(f"  Train labels: {tr_n} (0:{(train_labels==0).sum()}, 1:{(train_labels==1).sum()})")
    print(f"  Test labels: {te_n} (0:{(test_labels==0).sum()}, 1:{(test_labels==1).sum()})")

    out_dir = Path("results/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    config_record = {
        "config_path": config_path,
        "sensor_shape": list(sensor_all.shape),
        "all_channels": all_channels,
        "key_channels": avail_key,
        "dataset": {"seq_len": ds_cfg.seq_len, "stride": ds_cfg.stride,
                     "target_seq_len": ds_cfg.target_seq_len},
        "model": raw.get("model", {}),
        "n_train_labels": int(tr_n), "n_test_labels": int(te_n),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    do_plots = not no_plots
    do_tsne = not no_tsne and do_plots
    run_parts = set(parts or [1, 2, 3, 4, 5])

    # --- Extract embeddings ---
    print("\n[STEP 2] Extracting single-channel embeddings (all layers x all channels)...")
    t_emb = time.time()
    cache, y_train, y_test = extract_all_embeddings(
        sensor_all, train_labels, test_labels, all_channels,
        ch_to_idx, ds_cfg, raw.get("model", {}),
    )
    print(f"\n  Embedding extraction: {time.time() - t_emb:.1f}s")
    print(f"  Cache: {len(cache)} entries, Train: {len(y_train)}, Test: {len(y_test)}")

    all_results = []

    if 1 in run_parts:
        print("\n[STEP 3] Part 1: Single Channel x All Layers")
        p1 = part1_single_channel(cache, y_train, y_test, all_channels, out_dir, do_plots, do_tsne)
        all_results.extend(p1)

    p2 = []
    if 2 in run_parts:
        print("\n[STEP 4] Part 2: Channel Combinations x All Layers")
        p2 = part2_channel_combos(cache, y_train, y_test, avail_key, out_dir, do_plots, do_tsne)
        all_results.extend(p2)

    if 3 in run_parts:
        print("\n[STEP 5] Part 3: Multi-Layer Fusion")
        p3 = part3_multilayer_fusion(cache, y_train, y_test, avail_key, p2, out_dir, do_plots, do_tsne)
        all_results.extend(p3)

    if 4 in run_parts:
        print("\n[STEP 6] Part 4: Hybrid Features")
        p4 = part4_hybrid_features(
            cache, y_train, y_test, sensor_all, train_labels, test_labels,
            avail_key, ch_to_idx, ds_cfg, out_dir, do_plots,
        )
        all_results.extend(p4)

    if 5 in run_parts:
        print("\n[STEP 7] Part 5: Threshold Sensitivity")
        p5 = part5_threshold_sensitivity(cache, y_train, y_test, all_results, out_dir, do_plots)
        all_results.extend(p5)

    generate_summary(all_results, out_dir)

    total = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"SWEEP COMPLETE — {total:.1f}s ({total/60:.1f} min)")
    print(f"Results: {out_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APC Occupancy Comprehensive Ablation Sweep")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--parts", type=int, nargs="+", default=None,
                        help="Run specific parts (1-5). Default: all.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip all visualization")
    parser.add_argument("--no-tsne", action="store_true",
                        help="Skip t-SNE plots only (heatmaps still generated)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_sweep(args.config, parts=args.parts,
              no_plots=args.no_plots, no_tsne=args.no_tsne)
