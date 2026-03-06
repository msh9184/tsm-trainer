"""Analyze sweep results and generate summary tables + plots.

Reads per-phase CSV files (results_A.csv, results_B.csv, ...) or a merged
sweep_results.csv and produces:
  - Top-N results per phase
  - Best configuration per factor (layer, context, channel, classifier)
  - Heatmaps: layer × classifier, context × classifier
  - Bar charts: per-factor ablation

Usage:
    cd examples/classification/apc_enter_leave_stay

    # Analyze a single phase result
    python training/analyze_sweep.py --results results/.../sweep/results_A.csv

    # Merge all per-phase CSVs and analyze
    python training/analyze_sweep.py --results-dir results/.../sweep/

    # Analyze pre-merged file
    python training/analyze_sweep.py --results results/.../sweep/sweep_results.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

import matplotlib as mpl
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    mpl.use("Agg")
import matplotlib.pyplot as plt

from visualization.style import setup_style, save_figure, FIGSIZE_SINGLE

logger = logging.getLogger(__name__)


def load_results(csv_path: str | Path):
    """Load sweep results CSV into pandas DataFrame."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["f1_macro"])
    df["f1_macro"] = df["f1_macro"].astype(float)
    df["accuracy"] = df["accuracy"].astype(float)
    logger.info("Loaded %d valid results from %s", len(df), csv_path)
    return df


def merge_phase_results(sweep_dir: str | Path) -> "pd.DataFrame":
    """Merge all per-phase result CSVs (results_A.csv, results_B.csv, etc.)."""
    import pandas as pd
    import glob as globmod

    sweep_dir = Path(sweep_dir)
    csv_files = sorted(sweep_dir.glob("results_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No results_*.csv files found in {sweep_dir}")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            logger.info("  %s: %d rows", f.name, len(df))
            dfs.append(df)
        except pd.errors.ParserError:
            # Handle CSVs with mixed column counts (e.g., Phase E with
            # different label settings producing different per-class F1 columns)
            logger.warning("  %s: mixed columns, using python engine with skip", f.name)
            try:
                df = pd.read_csv(f, engine="python", on_bad_lines="warn")
                logger.info("  %s: %d rows (recovered)", f.name, len(df))
                dfs.append(df)
            except Exception:
                logger.warning("Failed to read %s", f, exc_info=True)
        except Exception:
            logger.warning("Failed to read %s", f, exc_info=True)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["exp_id"], keep="last")
    merged = merged.dropna(subset=["f1_macro"])
    merged["f1_macro"] = merged["f1_macro"].astype(float)
    merged["accuracy"] = merged["accuracy"].astype(float)

    # Save merged file
    merged_path = sweep_dir / "sweep_results.csv"
    merged.to_csv(merged_path, index=False)
    logger.info("Merged %d valid results → %s", len(merged), merged_path)
    return merged


def top_n_results(df, n: int = 20, metric: str = "f1_macro"):
    """Return top N results sorted by metric."""
    return df.nlargest(n, metric)


def best_per_factor(df, factor: str, metric: str = "f1_macro"):
    """Find best result per unique value of factor."""
    return df.loc[df.groupby(factor)[metric].idxmax()].sort_values(metric, ascending=False)


def plot_layer_heatmap(df, output_path: Path) -> None:
    """Heatmap of F1 macro: layer × classifier."""
    import pandas as pd
    import seaborn as sns

    setup_style()

    # Combine classifier_type and classifier_name
    df = df.copy()
    df["clf_label"] = df["classifier_type"] + "/" + df["classifier_name"]

    pivot = df.pivot_table(
        values="f1_macro", index="layer", columns="clf_label", aggfunc="mean",
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.8), 5))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.5, linecolor="white", ax=ax,
        vmin=0, vmax=1,
    )
    ax.set_title("F1 Macro: Layer × Classifier")
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Layer")

    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_context_comparison(df, output_path: Path) -> None:
    """Bar chart of best F1 per context configuration."""
    setup_style()

    best = df.groupby("context_name")["f1_macro"].max().sort_values(ascending=False)
    if best.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(best) * 0.5), 5))
    colors = ["#DE8F05" if i == 0 else "#0173B2" for i in range(len(best))]
    bars = ax.bar(range(len(best)), best.values, color=colors)

    ax.set_xticks(range(len(best)))
    ax.set_xticklabels(best.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1 Macro (best)")
    ax.set_title("Best F1 by Context Window")
    ax.set_ylim(0, 1.05)

    for bar, val in zip(bars, best.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8,
        )

    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_channel_comparison(df, output_path: Path) -> None:
    """Bar chart of best F1 per channel subset."""
    setup_style()

    best = df.groupby("channel_name")["f1_macro"].max().sort_values(ascending=False)
    if best.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(best) * 0.8), 5))
    colors = ["#DE8F05" if i == 0 else "#029E73" for i in range(len(best))]
    bars = ax.bar(range(len(best)), best.values, color=colors)

    ax.set_xticks(range(len(best)))
    ax.set_xticklabels(best.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1 Macro (best)")
    ax.set_title("Best F1 by Channel Subset")
    ax.set_ylim(0, 1.05)

    for bar, val in zip(bars, best.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8,
        )

    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def generate_report(df, output_dir: Path) -> None:
    """Generate comprehensive text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("SWEEP ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Total experiments: {len(df)}")
    lines.append(f"Phases: {sorted(df['phase'].unique().tolist())}")
    lines.append("")

    # Overall best
    best_idx = df["f1_macro"].idxmax()
    best = df.loc[best_idx]
    lines.append("OVERALL BEST:")
    lines.append(f"  Exp ID: {best['exp_id']}")
    lines.append(f"  F1 Macro: {best['f1_macro']:.4f}")
    lines.append(f"  Accuracy: {best['accuracy']:.4f}")
    lines.append(f"  Layer: L{best['layer']}")
    lines.append(f"  Classifier: {best['classifier_type']}/{best['classifier_name']}")
    lines.append(f"  Context: {best['context_name']}")
    lines.append(f"  Channels: {best['channel_name']}")
    lines.append("")

    # Per-phase summary
    if "phase" in df.columns:
        lines.append("PER-PHASE SUMMARY:")
        lines.append("-" * 70)
        for phase in sorted(df["phase"].unique()):
            pdf = df[df["phase"] == phase]
            best_row = pdf.loc[pdf["f1_macro"].idxmax()]
            lines.append(
                f"  Phase {phase}: {len(pdf):4d} exps  "
                f"Best F1={best_row['f1_macro']:.4f}  "
                f"({best_row['classifier_type']}/{best_row['classifier_name']}, "
                f"L{best_row['layer']}, {best_row.get('context_name', 'N/A')}, "
                f"{best_row.get('channel_name', 'N/A')})"
            )
        lines.append("")

    # Top 10
    lines.append("TOP 10 RESULTS:")
    lines.append("-" * 70)
    top10 = df.nlargest(10, "f1_macro")
    for _, row in top10.iterrows():
        lines.append(
            f"  {row['exp_id']:50s}  F1={row['f1_macro']:.4f}  Acc={row['accuracy']:.4f}"
        )
    lines.append("")

    # Best per factor
    for factor in ["layer", "classifier_name", "context_name", "channel_name"]:
        if factor not in df.columns:
            continue
        lines.append(f"BEST PER {factor.upper()}:")
        lines.append("-" * 70)
        best_per = df.loc[df.groupby(factor)["f1_macro"].idxmax()].sort_values("f1_macro", ascending=False)
        for _, row in best_per.iterrows():
            lines.append(
                f"  {str(row[factor]):25s}  F1={row['f1_macro']:.4f}  "
                f"({row['classifier_type']}/{row['classifier_name']}, {row.get('context_name', 'N/A')})"
            )
        lines.append("")

    report_path = output_dir / "sweep_analysis_report.txt"
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Saved report: %s", report_path)


def main():
    parser = argparse.ArgumentParser(description="Analyze sweep results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results", help="Path to a single results CSV")
    group.add_argument("--results-dir", help="Path to sweep dir with per-phase CSVs (auto-merges)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as results)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.results_dir:
        results_dir = Path(args.results_dir)
        df = merge_phase_results(results_dir)
        output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    else:
        results_path = Path(args.results)
        df = load_results(results_path)
        output_dir = Path(args.output_dir) if args.output_dir else results_path.parent / "analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        logger.error("No valid results found!")
        return

    # Generate report
    generate_report(df, output_dir)

    # Plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    try:
        plot_layer_heatmap(df, plots_dir / "heatmap_layer_clf")
    except Exception:
        logger.warning("Failed to generate layer heatmap", exc_info=True)

    if "context_name" in df.columns and df["context_name"].nunique() > 1:
        try:
            plot_context_comparison(df, plots_dir / "context_comparison")
        except Exception:
            logger.warning("Failed to generate context comparison", exc_info=True)

    if "channel_name" in df.columns and df["channel_name"].nunique() > 1:
        try:
            plot_channel_comparison(df, plots_dir / "channel_comparison")
        except Exception:
            logger.warning("Failed to generate channel comparison", exc_info=True)

    logger.info("Analysis complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
