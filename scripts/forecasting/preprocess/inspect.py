#!/usr/bin/env python3
"""
inspect.py — Unified Dataset Inspection Tool
=============================================

Two subcommands:

  dataset      Analyse a single dataset (Arrow or .npz) and produce statistics + plots.
  benchmarks   Aggregate statistics across multiple benchmark datasets, optionally
               comparing against training datasets.

USAGE
-----
  # Single dataset
  python inspect.py dataset /path/to/kernel_synth_3d
  python inspect.py dataset /tmp/kernel_synth_3d_tmp                  # .npz dir
  python inspect.py dataset /path/to/training_corpus_tsmixup_10m \\
      --max_samples 2000 --seed 7

  # Benchmark aggregation
  python inspect.py benchmarks \\
      --benchmarks_root /group-volume/ts-dataset/benchmarks_unified \\
      --train_paths /path/to/kernel_synth_3d /path/to/composite_synth_3d \\
      --max_per_dataset 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Shared: data loading
# ──────────────────────────────────────────────────────────────────────────────

def _is_hf_dataset_dir(p: Path) -> bool:
    return p.is_dir() and (
        (p / "dataset_dict.json").exists() or
        (p / "dataset_info.json").exists()
    )


def _load_arrow(folder: Path, split: str = "train"):
    """Return (HF Dataset, format_label).  Handles DatasetDict and bare Dataset."""
    import datasets as hf
    raw = hf.load_from_disk(str(folder))
    if isinstance(raw, hf.DatasetDict):
        if split in raw:
            return raw[split], f"arrow_datasetdict[{split}]"
        first = next(iter(raw))
        log.warning("Split '%s' not found; using '%s'.", split, first)
        return raw[first], f"arrow_datasetdict[{first}]"
    return raw, "arrow_dataset"


def load_dataset(folder: Path, split: str = "train"):
    """
    Returns:
        samples  -- list of np.ndarray (npz) or HF Dataset object (arrow)
        fmt      -- human-readable format string
        total    -- total rows in the underlying dataset
    """
    npz_files = sorted(folder.glob("sample_*.npz"))
    if npz_files:
        log.info("Detected .npz directory: %d files", len(npz_files))
        samples = [np.load(f)["target"] for f in npz_files]
        return samples, "npz_directory", len(npz_files)

    if _is_hf_dataset_dir(folder):
        ds, fmt = _load_arrow(folder, split)
        return ds, fmt, len(ds)

    raise ValueError(
        f"Cannot determine dataset format for {folder}.\n"
        "Expected either sample_*.npz files or an HF Arrow dataset directory."
    )


def _get_target(ds, idx: int) -> np.ndarray:
    row = ds[int(idx)]
    return np.array(row["target"], dtype=np.float64)


def collect_samples(ds, total: int, max_samples: int,
                    rng: np.random.Generator) -> list[np.ndarray]:
    """Draw up to max_samples rows from an HF Dataset."""
    n = min(total, max_samples)
    indices = rng.choice(total, size=n, replace=False)
    samples = []
    for idx in indices:
        try:
            samples.append(_get_target(ds, idx))
        except Exception as exc:
            log.debug("Row %d failed: %s", idx, exc)
    return samples


def _load_samples_from_dir(ds_dir: Path, max_n: int,
                           rng: np.random.Generator) -> list[np.ndarray]:
    """Load up to max_n target arrays from an HF Arrow dataset, filtering NaN/Inf."""
    import datasets as hf
    try:
        raw = hf.load_from_disk(str(ds_dir))
        ds = raw["train"] if isinstance(raw, hf.DatasetDict) else raw
        total = len(ds)
        n = min(total, max_n)
        indices = rng.choice(total, size=n, replace=False)
        out = []
        for idx in indices:
            try:
                arr = np.array(ds[int(idx)]["target"], dtype=np.float64)
                if arr.size > 0 and not (np.any(np.isnan(arr)) or np.any(np.isinf(arr))):
                    out.append(arr)
            except Exception:
                pass
        return out
    except Exception as exc:
        log.debug("  [skip] %s: %s", ds_dir.name, exc)
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Shared: statistics computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(samples: list[np.ndarray]) -> dict:
    """Compute per-sample and aggregate statistics."""
    lengths, n_variates_list = [], []
    per_var_stds: list[list[float]] = []
    per_var_means: list[list[float]] = []
    cross_corrs: list[float] = []
    nan_inf_count = 0
    zero_var_count = 0
    empty_count = 0

    for arr in samples:
        if arr is None or arr.size == 0:
            empty_count += 1
            continue
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            nan_inf_count += 1
            continue

        arr2d = arr.reshape(-1, arr.shape[-1])
        k, L = arr2d.shape
        lengths.append(L)
        n_variates_list.append(k)

        stds = np.std(arr2d, axis=1)
        means = np.mean(arr2d, axis=1)

        if np.any(stds < 1e-12):
            zero_var_count += 1

        per_var_stds.append(stds.tolist())
        per_var_means.append(means.tolist())

        if k > 1:
            corr = np.corrcoef(arr2d)
            n = corr.shape[0]
            off = [corr[i, j] for i in range(n) for j in range(n) if i != j]
            cross_corrs.append(float(np.mean(np.abs(off))))

    lengths_arr = np.array(lengths)
    all_stds = np.concatenate(per_var_stds) if per_var_stds else np.array([])
    all_means = np.concatenate(per_var_means) if per_var_means else np.array([])

    n_valid = len(lengths)
    mode_variates = int(np.bincount(n_variates_list).argmax()) if n_variates_list else 0

    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else float("nan")

    return {
        "n_valid":          n_valid,
        "n_empty":          empty_count,
        "n_nan_inf":        nan_inf_count,
        "n_zero_variance":  zero_var_count,
        "mode_variates":    mode_variates,
        "n_variates_list":  n_variates_list,
        "lengths": {
            "min":  int(lengths_arr.min()) if len(lengths_arr) else 0,
            "max":  int(lengths_arr.max()) if len(lengths_arr) else 0,
            "mean": float(lengths_arr.mean()) if len(lengths_arr) else 0,
            "std":  float(lengths_arr.std()) if len(lengths_arr) else 0,
            "p25":  pct(lengths_arr, 25),
            "p50":  pct(lengths_arr, 50),
            "p75":  pct(lengths_arr, 75),
        },
        "std_per_variate": {
            "min":  float(all_stds.min()) if len(all_stds) else 0,
            "max":  float(all_stds.max()) if len(all_stds) else 0,
            "mean": float(all_stds.mean()) if len(all_stds) else 0,
            "p50":  pct(all_stds, 50),
        },
        "cross_corr": {
            "mean": float(np.mean(cross_corrs)) if cross_corrs else float("nan"),
            "std":  float(np.std(cross_corrs)) if cross_corrs else float("nan"),
            "p10":  pct(cross_corrs, 10),
            "p50":  pct(cross_corrs, 50),
            "p90":  pct(cross_corrs, 90),
        } if cross_corrs else None,
        # Raw arrays for plotting
        "per_var_stds":  per_var_stds,
        "per_var_means": per_var_means,
        "cross_corrs":   cross_corrs,
        "lengths_raw":   lengths,
        "all_stds":      all_stds,
        "all_means":     all_means,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Shared: plot constants
# ──────────────────────────────────────────────────────────────────────────────

_VARIATE_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
_VARIATE_LABELS = ["v\u2081", "v\u2082", "v\u2083", "v\u2084", "v\u2085"]

_GROUP_COLORS = {
    "chronos":            "#2196F3",
    "fev_bench":          "#FF5722",
    "gift_eval":          "#4CAF50",
    "ltsf":               "#9C27B0",
    "kernel_synth_3d":    "#FF9800",
    "composite_synth_3d": "#795548",
}


def _vlabels(k: int) -> list[str]:
    return [_VARIATE_LABELS[j] if j < len(_VARIATE_LABELS) else f"v{j+1}"
            for j in range(k)]


# ──────────────────────────────────────────────────────────────────────────────
# Subcommand: dataset
# ──────────────────────────────────────────────────────────────────────────────

def _print_dataset_stats(stats: dict, folder: Path, fmt: str,
                         total: int, max_samples: int) -> None:
    n = stats["n_valid"]
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Dataset Statistics")
    print(f"  Path   : {folder}")
    print(f"  Format : {fmt}")
    print(sep)
    print(f"  Total rows in dataset  : {total:,}")
    print(f"  Analysed (random sample): {n + stats['n_nan_inf'] + stats['n_empty']:,}"
          f"  (max_samples={max_samples})")
    print(f"  Valid samples          : {n:,}")
    print(f"  Skipped -- NaN/Inf     : {stats['n_nan_inf']}")
    print(f"  Skipped -- Empty       : {stats['n_empty']}")
    print(f"  Samples w/ zero var    : {stats['n_zero_variance']}")
    print()
    print(f"  Dimensionality (mode)  : {stats['mode_variates']}D")
    if len(set(stats["n_variates_list"])) > 1:
        print(f"  Variate counts         : {dict(Counter(stats['n_variates_list']))}")
    print()
    L = stats["lengths"]
    print("  Time-series length")
    print(f"    min / mean +- std / max : {L['min']:,} / {L['mean']:,.1f} +- {L['std']:,.1f} / {L['max']:,}")
    print(f"    p25 / p50 / p75        : {L['p25']:,.0f} / {L['p50']:,.0f} / {L['p75']:,.0f}")
    print()
    S = stats["std_per_variate"]
    print("  Std dev per variate  (across all samples and variates)")
    print(f"    min / mean / p50 / max : {S['min']:.4f} / {S['mean']:.4f} / {S['p50']:.4f} / {S['max']:.4f}")
    print()
    if stats["cross_corr"] is not None:
        C = stats["cross_corr"]
        print("  Mean |cross-variate correlation|  (per sample, then aggregated)")
        print(f"    mean +- std : {C['mean']:.4f} +- {C['std']:.4f}")
        print(f"    p10 / p50 / p90 : {C['p10']:.4f} / {C['p50']:.4f} / {C['p90']:.4f}")
    else:
        print("  Cross-variate correlation : N/A  (univariate data)")
    print(sep + "\n")


def _plot_one_sample(ax_ts, ax_corr, arr: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    arr2d = arr.reshape(-1, arr.shape[-1])
    k, L = arr2d.shape
    t = np.arange(L)

    for j in range(k):
        label = _vlabels(k)[j] if k > 1 else "target"
        color = _VARIATE_COLORS[j % len(_VARIATE_COLORS)]
        ax_ts.plot(t, arr2d[j], lw=0.8, alpha=0.85, color=color, label=label)

    ax_ts.set_title(title, fontsize=9, pad=3)
    ax_ts.set_xlabel("time step", fontsize=7)
    ax_ts.set_ylabel("value", fontsize=7)
    ax_ts.tick_params(labelsize=6)
    if k > 1:
        ax_ts.legend(fontsize=6, loc="upper right", framealpha=0.6)
    ax_ts.grid(True, lw=0.3, alpha=0.4)

    if ax_corr is None:
        return
    if k > 1:
        corr = np.corrcoef(arr2d)
        im = ax_corr.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
        ax_corr.set_xticks(range(k))
        ax_corr.set_yticks(range(k))
        ax_corr.set_xticklabels(_vlabels(k), fontsize=7)
        ax_corr.set_yticklabels(_vlabels(k), fontsize=7)
        ax_corr.set_title("corr", fontsize=8, pad=2)
        for i in range(k):
            for j2 in range(k):
                ax_corr.text(j2, i, f"{corr[i, j2]:.2f}",
                             ha="center", va="center", fontsize=6,
                             color="white" if abs(corr[i, j2]) > 0.5 else "black")
    else:
        ax_corr.axis("off")
        ax_corr.text(0.5, 0.5, "univariate", ha="center", va="center",
                     fontsize=8, transform=ax_corr.transAxes)


def _plot_dataset_samples(chosen: list[np.ndarray], folder: Path,
                          output_path: Path, show: bool) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_samples = len(chosen)
    multivariate = any(a.ndim == 2 and a.shape[0] > 1 for a in chosen)
    n_rows = 2 if multivariate else 1

    fig, axes = plt.subplots(
        n_rows, n_samples,
        figsize=(5 * n_samples, 3.5 * n_rows),
        squeeze=False,
    )
    fig.suptitle(f"Random Sample Viewer -- {folder.name}", fontsize=11, y=1.01)

    for col, arr in enumerate(chosen):
        arr2d = arr.reshape(-1, arr.shape[-1])
        k, L = arr2d.shape
        title = f"sample {col + 1}  (shape {k}x{L})"
        ax_ts = axes[0][col]
        ax_corr = axes[1][col] if multivariate else None
        _plot_one_sample(ax_ts, ax_corr, arr, title)

    fig.tight_layout()
    if show:
        plt.show()
    else:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info("Saved sample plot -> %s", output_path)
    plt.close(fig)


def _plot_dataset_stats(samples: list[np.ndarray], stats: dict,
                        folder: Path, output_path: Path, show: bool) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    multivariate = stats["mode_variates"] > 1
    n_cols = 4 if multivariate else 3
    fig = plt.figure(figsize=(4.5 * n_cols, 7))
    gs = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(f"Dataset Statistics -- {folder.name}", fontsize=12)

    lengths = np.array(stats["lengths_raw"])
    all_stds = stats["all_stds"]
    all_means = stats["all_means"]

    # Row 0, Col 0: Length distribution
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(lengths, bins=min(40, max(10, len(lengths) // 10)),
            color="#2196F3", edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(np.mean(lengths), color="red", lw=1.2, linestyle="--",
               label=f"mean={np.mean(lengths):.0f}")
    ax.set_title("Series Length Distribution", fontsize=9)
    ax.set_xlabel("length (timesteps)", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # Row 0, Col 1: Std distribution
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(all_stds, bins=min(40, max(10, len(all_stds) // 5)),
            color="#FF5722", edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(np.median(all_stds), color="darkred", lw=1.2, linestyle="--",
               label=f"median={np.median(all_stds):.3f}")
    ax.set_title("Std Dev Distribution (all variates)", fontsize=9)
    ax.set_xlabel("std dev", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # Row 0, Col 2: Mean distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(all_means, bins=min(40, max(10, len(all_means) // 5)),
            color="#4CAF50", edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(np.mean(all_means), color="darkgreen", lw=1.2, linestyle="--",
               label=f"mean={np.mean(all_means):.3f}")
    ax.set_title("Mean Distribution (all variates)", fontsize=9)
    ax.set_xlabel("mean value", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # Row 0, Col 3 (multivariate): Cross-corr distribution
    if multivariate and n_cols == 4:
        ax = fig.add_subplot(gs[0, 3])
        corrs = stats["cross_corrs"]
        if corrs:
            ax.hist(corrs, bins=min(30, max(5, len(corrs) // 5)),
                    color="#9C27B0", edgecolor="white", linewidth=0.4, alpha=0.85)
            ax.axvline(np.mean(corrs), color="purple", lw=1.2, linestyle="--",
                       label=f"mean={np.mean(corrs):.3f}")
            ax.set_title("Mean |Cross-Variate Corr|", fontsize=9)
            ax.set_xlabel("|correlation|", fontsize=8)
            ax.set_ylabel("count (samples)", fontsize=8)
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "no multivariate samples", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title("Mean |Cross-Variate Corr|", fontsize=9)
        ax.tick_params(labelsize=7)

    # Row 1, Col 0: Per-variate std box plot
    ax = fig.add_subplot(gs[1, 0])
    n_var = stats["mode_variates"]
    if n_var > 1 and stats["per_var_stds"]:
        by_var = [[] for _ in range(n_var)]
        for row_stds in stats["per_var_stds"]:
            for j, s in enumerate(row_stds[:n_var]):
                by_var[j].append(s)
        bp = ax.boxplot(by_var, labels=_vlabels(n_var),
                        patch_artist=True, medianprops={"color": "white", "lw": 1.5})
        for j, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(_VARIATE_COLORS[j % len(_VARIATE_COLORS)])
        ax.set_title("Std Dev per Variate", fontsize=9)
        ax.set_ylabel("std dev", fontsize=8)
    else:
        ax.hist(all_stds, bins=20, color="#FF5722", alpha=0.8, edgecolor="white")
        ax.set_title("Std Dev Distribution", fontsize=9)
        ax.set_ylabel("count", fontsize=8)
    ax.tick_params(labelsize=7)

    # Row 1, Col 1: Value range distribution
    ax = fig.add_subplot(gs[1, 1])
    subset_vals = []
    for arr in samples[:200]:
        arr2d = arr.reshape(-1, arr.shape[-1])
        subset_vals.extend(arr2d.flatten().tolist())
    subset_vals = np.array(subset_vals)
    if len(subset_vals) > 50000:
        subset_vals = np.random.choice(subset_vals, size=50000, replace=False)
    ax.hist(np.clip(subset_vals, np.percentile(subset_vals, 1),
                    np.percentile(subset_vals, 99)),
            bins=60, color="#607D8B", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.set_title("Value Distribution (clipped p1-p99)", fontsize=9)
    ax.set_xlabel("value", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.tick_params(labelsize=7)

    # Row 1, Col 2: Summary text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    L = stats["lengths"]
    S = stats["std_per_variate"]
    lines = [
        f"Dataset: {folder.name}",
        "",
        f"Total rows      : {stats['n_valid'] + stats['n_nan_inf'] + stats['n_empty']:,}  (analysed)",
        f"Valid samples   : {stats['n_valid']:,}",
        f"Skipped NaN/Inf : {stats['n_nan_inf']}",
        f"Skipped empty   : {stats['n_empty']}",
        f"Zero variance   : {stats['n_zero_variance']}",
        "",
        f"Dimensionality  : {stats['mode_variates']}D",
        "",
        "Length (timesteps)",
        f"  min  : {L['min']:,}",
        f"  mean : {L['mean']:,.1f}",
        f"  max  : {L['max']:,}",
        "",
        "Std Dev (all variates)",
        f"  min  : {S['min']:.4f}",
        f"  mean : {S['mean']:.4f}",
        f"  max  : {S['max']:.4f}",
    ]
    if stats["cross_corr"]:
        C = stats["cross_corr"]
        lines += [
            "",
            "Cross-Variate |corr|",
            f"  mean : {C['mean']:.4f}",
            f"  p50  : {C['p50']:.4f}",
        ]
    ax.text(0.05, 0.97, "\n".join(lines), transform=ax.transAxes,
            fontsize=7.5, va="top", ha="left", family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F5F5F5",
                  "edgecolor": "#BDBDBD"})

    # Row 1, Col 3 (multivariate): mean correlation heatmap
    if multivariate and n_cols == 4:
        ax = fig.add_subplot(gs[1, 3])
        multivar_samples = [a for a in samples if a.ndim == 2 and a.shape[0] > 1]
        if multivar_samples:
            k_ref = stats["mode_variates"]
            corr_sum = np.zeros((k_ref, k_ref))
            count = 0
            for arr in multivar_samples[:500]:
                arr2d = arr.reshape(-1, arr.shape[-1])
                if arr2d.shape[0] == k_ref:
                    corr_sum += np.corrcoef(arr2d)
                    count += 1
            if count > 0:
                mean_corr = corr_sum / count
                im = ax.imshow(mean_corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xticks(range(k_ref))
                ax.set_yticks(range(k_ref))
                ax.set_xticklabels(_vlabels(k_ref), fontsize=8)
                ax.set_yticklabels(_vlabels(k_ref), fontsize=8)
                ax.set_title(f"Mean Corr Matrix\n(n={count} samples)", fontsize=9)
                for i in range(k_ref):
                    for j2 in range(k_ref):
                        ax.text(j2, i, f"{mean_corr[i, j2]:.2f}",
                                ha="center", va="center", fontsize=7,
                                color="white" if abs(mean_corr[i, j2]) > 0.5 else "black")

    if show:
        plt.show()
    else:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info("Saved stats plot -> %s", output_path)
    plt.close(fig)


def cmd_dataset(args: argparse.Namespace) -> None:
    """Subcommand: inspect a single dataset."""
    folder = Path(args.folder).resolve()
    if not folder.exists():
        log.error("Path does not exist: %s", folder)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else folder
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    log.info("Loading dataset from: %s", folder)
    raw, fmt, total = load_dataset(folder, args.split)
    log.info("Format: %s | Total rows: %d", fmt, total)

    if isinstance(raw, list):
        all_arrays = raw
        n_to_analyse = min(len(all_arrays), args.max_samples)
        chosen_indices = rng.choice(len(all_arrays), size=n_to_analyse, replace=False)
        samples = [all_arrays[i] for i in chosen_indices]
    else:
        log.info("Sampling up to %d rows for analysis ...", args.max_samples)
        samples = collect_samples(raw, total, args.max_samples, rng)

    log.info("Collected %d samples for analysis.", len(samples))
    if not samples:
        log.error("No valid samples found.")
        sys.exit(1)

    stats = compute_stats(samples)
    _print_dataset_stats(stats, folder, fmt, total, args.max_samples)

    n_plot = min(3, len(samples))
    plot_indices = rng.choice(len(samples), size=n_plot, replace=False)
    chosen = [samples[i] for i in plot_indices]

    _plot_dataset_samples(chosen, folder, output_dir / "samples.png", show=args.show)
    _plot_dataset_stats(samples, stats, folder, output_dir / "stats.png", show=args.show)

    if not args.show:
        log.info("Done.  Figures saved to: %s", output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Subcommand: benchmarks
# ──────────────────────────────────────────────────────────────────────────────

def _compute_benchmarks_stats(samples: list[np.ndarray]) -> dict:
    """Compact stats dict used by the benchmarks summary table and plots."""
    lengths, stds, means, n_vars, cross_corrs = [], [], [], [], []
    for arr in samples:
        arr2d = arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 2 else arr[np.newaxis, :]
        k, L = arr2d.shape
        lengths.append(L)
        n_vars.append(k)
        s = np.std(arr2d, axis=1)
        m = np.mean(arr2d, axis=1)
        stds.extend(s.tolist())
        means.extend(m.tolist())
        if k > 1:
            corr = np.corrcoef(arr2d)
            n = corr.shape[0]
            off = [abs(corr[i, j]) for i in range(n) for j in range(n) if i != j]
            cross_corrs.append(float(np.mean(off)))

    if not lengths:
        return {}

    la = np.array(lengths)
    sa = np.array(stds)
    return {
        "n":           len(lengths),
        "lengths":     la,
        "stds":        sa,
        "means":       np.array(means),
        "n_vars":      np.array(n_vars),
        "cross_corrs": np.array(cross_corrs) if cross_corrs else None,
        "len_min":     int(la.min()),
        "len_mean":    float(la.mean()),
        "len_median":  float(np.median(la)),
        "len_max":     int(la.max()),
        "std_median":  float(np.median(sa)),
        "std_max":     float(sa.max()),
        "pct_multi":   100.0 * np.mean(np.array(n_vars) > 1),
        "corr_mean":   float(np.mean(cross_corrs)) if cross_corrs else float("nan"),
    }


def _collect_group(root: Path, max_per: int,
                   rng: np.random.Generator) -> list[np.ndarray]:
    samples = []
    dirs = sorted(d for d in root.iterdir() if _is_hf_dataset_dir(d))
    for d in dirs:
        s = _load_samples_from_dir(d, max_per, rng)
        samples.extend(s)
    log.info("  %s: %d datasets -> %d samples", root.name, len(dirs), len(samples))
    return samples


def _print_benchmarks_table(groups: dict[str, dict]) -> None:
    header = (f"{'Group':<22} {'N':>7} {'%Multi':>7} {'L_min':>7} {'L_med':>8} "
              f"{'L_max':>8} {'std_med':>9} {'|corr|_mean':>12}")
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("  Benchmark Statistics Summary")
    print(sep)
    print(header)
    print(sep)
    for name, st in groups.items():
        if not st:
            print(f"  {name:<20}  (no samples)")
            continue
        corr_str = f"{st['corr_mean']:.3f}" if not np.isnan(st['corr_mean']) else "  n/a "
        print(
            f"  {name:<20} {st['n']:>7,} {st['pct_multi']:>7.1f}"
            f" {st['len_min']:>7,} {st['len_median']:>8,.0f}"
            f" {st['len_max']:>8,} {st['std_median']:>9.3f} {corr_str:>12}"
        )
    print(sep + "\n")


def _plot_benchmarks_aggregate(groups: dict[str, dict], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Benchmark + Training Data: Aggregate Statistics", fontsize=13)

    ax_len, ax_std, ax_mean = axes[0]
    ax_corr, ax_nvar, ax_summary = axes[1]

    def _hist(ax, key, title, xlabel, log_scale=False, xlim_pct=None):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)
        for name, st in groups.items():
            if not st or key not in st or st[key] is None:
                continue
            data = st[key]
            if len(data) == 0:
                continue
            color = _GROUP_COLORS.get(name, "#607D8B")
            if xlim_pct:
                lo = np.percentile(data, xlim_pct[0])
                hi = np.percentile(data, xlim_pct[1])
                data = data[(data >= lo) & (data <= hi)]
            ax.hist(data, bins=50, density=True, alpha=0.55,
                    color=color, edgecolor="none", label=name)
        if log_scale:
            ax.set_yscale("log")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, lw=0.3, alpha=0.4)

    _hist(ax_len, "lengths", "Series Length Distribution", "length (timesteps)")
    _hist(ax_std, "stds", "Std Dev Distribution (all variates)", "std dev",
          xlim_pct=(1, 99))
    _hist(ax_mean, "means", "Mean Distribution (all variates)", "mean value",
          xlim_pct=(1, 99))

    # Cross-variate correlation
    ax_corr.set_title("Mean |Cross-Variate Corr| (multivariate only)", fontsize=9)
    ax_corr.set_xlabel("|correlation|", fontsize=8)
    ax_corr.set_ylabel("density", fontsize=8)
    ax_corr.tick_params(labelsize=7)
    has_corr = False
    for name, st in groups.items():
        if not st or st.get("cross_corrs") is None or len(st["cross_corrs"]) == 0:
            continue
        color = _GROUP_COLORS.get(name, "#607D8B")
        ax_corr.hist(st["cross_corrs"], bins=30, density=True, alpha=0.55,
                     color=color, edgecolor="none", label=name)
        has_corr = True
    if not has_corr:
        ax_corr.text(0.5, 0.5, "no multivariate data", ha="center", va="center",
                     transform=ax_corr.transAxes)
    ax_corr.legend(fontsize=6)
    ax_corr.grid(True, lw=0.3, alpha=0.4)

    # Variate count distribution
    ax_nvar.set_title("Variate Count Distribution", fontsize=9)
    ax_nvar.set_xlabel("n_variates", fontsize=8)
    ax_nvar.set_ylabel("fraction", fontsize=8)
    ax_nvar.tick_params(labelsize=7)
    all_n_vars = []
    for st in groups.values():
        if st and "n_vars" in st:
            all_n_vars.extend(st["n_vars"].tolist())
    if all_n_vars:
        cnt = Counter(all_n_vars)
        k_vals = sorted(cnt)
        totals = sum(cnt.values())
        ax_nvar.bar(k_vals, [cnt[k] / totals for k in k_vals], color="#607D8B", alpha=0.8)

    # Summary text
    ax_summary.axis("off")
    lines = ["Scalar Summary\n"]
    for name, st in groups.items():
        if not st:
            continue
        lines.append(f"[{name}]")
        lines.append(f"  N={st['n']:,}  %multi={st['pct_multi']:.0f}%")
        lines.append(f"  L: {st['len_min']:,}-{st['len_median']:.0f}-{st['len_max']:,}")
        lines.append(f"  std_median={st['std_median']:.3f}")
        if not np.isnan(st["corr_mean"]):
            lines.append(f"  |corr|_mean={st['corr_mean']:.3f}")
        lines.append("")
    ax_summary.text(0.04, 0.98, "\n".join(lines), transform=ax_summary.transAxes,
                    fontsize=7, va="top", ha="left", family="monospace",
                    bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F5F5F5",
                          "edgecolor": "#BDBDBD"})

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Saved -> %s", output_path)
    plt.close(fig)


def _plot_benchmarks_length_cdf(groups: dict[str, dict], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Series Length Distribution by Source", fontsize=12)

    for ax, use_log in zip(axes, (False, True)):
        for name, st in groups.items():
            if not st or "lengths" not in st or len(st["lengths"]) == 0:
                continue
            data = np.sort(st["lengths"])
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data, cdf, label=f"{name} (n={len(data):,})",
                    color=_GROUP_COLORS.get(name, "#607D8B"), lw=1.6)
        ax.set_ylabel("CDF", fontsize=9)
        ax.set_xlabel("series length (timesteps)", fontsize=9)
        if use_log:
            ax.set_xscale("log")
            ax.set_title("CDF (log x-axis)", fontsize=9)
        else:
            ax.set_title("CDF (linear)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, lw=0.3, alpha=0.4)
        for L, lbl in [(128, "128"), (512, "512"), (2048, "2k"), (4096, "4k"), (8192, "8k")]:
            ax.axvline(L, lw=0.8, linestyle="--", color="#BDBDBD", alpha=0.7)
            ax.text(L, 0.02, lbl, fontsize=6, color="#888888", ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Saved -> %s", output_path)
    plt.close(fig)


def cmd_benchmarks(args: argparse.Namespace) -> None:
    """Subcommand: aggregate benchmark statistics."""
    broot = Path(args.benchmarks_root)
    output_dir = Path(args.output_dir) if args.output_dir else broot
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    groups: dict[str, dict] = {}

    for bench_name in ["chronos", "fev_bench", "gift_eval", "ltsf"]:
        src = broot / bench_name
        if not src.exists():
            log.warning("[skip] %s not found", src)
            continue
        log.info("Loading %s ...", bench_name)
        samples = _collect_group(src, args.max_per_dataset, rng)
        groups[bench_name] = _compute_benchmarks_stats(samples) if samples else {}

    for train_path in (args.train_paths or []):
        tp = Path(train_path)
        name = tp.name
        log.info("Loading training: %s ...", name)
        samples = _load_samples_from_dir(tp, args.max_per_dataset * 5, rng)
        groups[name] = _compute_benchmarks_stats(samples) if samples else {}
        log.info("  %s: %d samples", name, groups[name].get("n", 0))

    _print_benchmarks_table(groups)
    _plot_benchmarks_aggregate(groups, output_dir / "benchmarks_stats.png")
    _plot_benchmarks_length_cdf(groups, output_dir / "benchmarks_length_cdf.png")
    log.info("Done. Output at: %s", output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified dataset inspection tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- dataset subcommand --
    p_ds = sub.add_parser("dataset",
                          help="Inspect a single dataset (Arrow or .npz directory).")
    p_ds.add_argument("folder", help="Dataset folder path")
    p_ds.add_argument("--max_samples", type=int, default=1000,
                      help="Max rows to analyse (default: 1000)")
    p_ds.add_argument("--seed", type=int, default=42)
    p_ds.add_argument("--output_dir", default=None,
                      help="Output directory for PNGs (default: dataset folder)")
    p_ds.add_argument("--show", action="store_true",
                      help="Display figures interactively instead of saving")
    p_ds.add_argument("--split", default="train",
                      help="HF Dataset split to use (default: train)")

    # -- benchmarks subcommand --
    p_bm = sub.add_parser("benchmarks",
                          help="Aggregate statistics across benchmark datasets.")
    p_bm.add_argument("--benchmarks_root",
                       default="/group-volume/ts-dataset/benchmarks_unified",
                       help="Root of unified benchmark datasets")
    p_bm.add_argument("--train_paths", nargs="*", default=[],
                       help="Optional training dataset paths for comparison")
    p_bm.add_argument("--max_per_dataset", type=int, default=200,
                       help="Max samples per individual dataset (default: 200)")
    p_bm.add_argument("--output_dir", default=None,
                       help="Where to save PNGs (default: benchmarks_root)")
    p_bm.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "dataset":
        cmd_dataset(args)
    elif args.command == "benchmarks":
        cmd_benchmarks(args)


if __name__ == "__main__":
    main()
