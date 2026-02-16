#!/usr/bin/env python3
"""Model comparison table generator for TSM-Trainer.

Generates publication-quality comparison tables (Markdown, LaTeX, CSV)
from benchmark results across multiple models.

Usage:
    # Compare two experiments on Chronos Benchmark II
    python compare_models.py \
        --results results/experiments/exp001 results/experiments/exp002 \
        --baseline results/baselines/seasonal-naive \
        --benchmark chronos_bench_zero_shot \
        --format markdown

    # Generate LaTeX table for paper
    python compare_models.py \
        --results results/experiments/exp001 \
        --baseline results/baselines/seasonal-naive \
        --benchmark chronos_bench_zero_shot \
        --format latex

    # Include pre-computed baseline models
    python compare_models.py \
        --results results/experiments/exp001 \
        --precomputed-baselines chronos-t5-base chronos-bolt-base \
        --benchmark chronos_bench_zero_shot \
        --format markdown
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("ModelComparator")

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"


def load_results(path: str | Path) -> pd.DataFrame:
    """Load results from path (CSV file or directory)."""
    path = Path(path)

    if path.is_file() and path.suffix == ".csv":
        return pd.read_csv(path)

    # Try common patterns in directory
    for pattern in ["*.csv"]:
        csvs = sorted(path.glob(pattern))
        if csvs:
            return pd.read_csv(csvs[0])

    raise FileNotFoundError(f"No CSV results found at: {path}")


def load_precomputed_baseline(model_name: str, benchmark_type: str) -> pd.DataFrame | None:
    """Load pre-computed baseline results from the results/ directory."""
    # Try exact match first
    path = RESULTS_DIR / f"{model_name}-{benchmark_type}.csv"
    if path.exists():
        return pd.read_csv(path)

    # Try variations
    for suffix in [benchmark_type, benchmark_type.replace("_", "-")]:
        path = RESULTS_DIR / f"{model_name}-{suffix}.csv"
        if path.exists():
            return pd.read_csv(path)

    return None


def generate_comparison(
    model_results: dict[str, pd.DataFrame],
    baseline_df: pd.DataFrame | None = None,
    metrics: list[str] | None = None,
    format: str = "markdown",
) -> str:
    """Generate model comparison table."""
    from engine.aggregator import Aggregator

    if metrics is None:
        metrics = ["WQL", "MASE"]

    rows = []
    for model_name, results_df in model_results.items():
        row = {"Model": model_name}

        for metric in metrics:
            if metric not in results_df.columns:
                row[f"Avg {metric}"] = float("nan")
                continue

            values = results_df[metric].dropna().values
            row[f"Avg {metric}"] = float(np.mean(values)) if len(values) > 0 else float("nan")

            # Relative score vs baseline
            if baseline_df is not None and metric in baseline_df.columns:
                try:
                    agg = Aggregator.aggregate_gmean(results_df, baseline_df, [metric])
                    key = f"agg_rel_{metric.lower()}"
                    row[f"Rel {metric}"] = agg.get(key, float("nan"))
                except Exception:
                    row[f"Rel {metric}"] = float("nan")

        rows.append(row)

    table_df = pd.DataFrame(rows)

    if format == "markdown":
        return _to_markdown(table_df)
    elif format == "latex":
        return _to_latex(table_df)
    elif format == "csv":
        return table_df.to_csv(index=False)
    else:
        return table_df.to_string(index=False)


def _to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    # Alignment: left for Model, center for metrics
    aligns = [":---"] + [":-:"] * (len(headers) - 1)
    lines.append("| " + " | ".join(aligns) + " |")

    for _, row in df.iterrows():
        cells = []
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                if np.isnan(val):
                    cells.append("—")
                elif val < 1.0:
                    cells.append(f"{val:.4f}")
                else:
                    cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _to_latex(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    n_cols = len(headers)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{'l' + 'c' * (n_cols - 1)}}}",
        "\\toprule",
        " & ".join([f"\\textbf{{{h}}}" for h in headers]) + " \\\\",
        "\\midrule",
    ]

    for _, row in df.iterrows():
        cells = []
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                if np.isnan(val):
                    cells.append("—")
                else:
                    cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Model comparison on time series forecasting benchmark.}",
        "\\label{tab:benchmark}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate model comparison tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results", nargs="+", required=True,
        help="Paths to experiment result directories or CSV files",
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Path to baseline results (e.g., seasonal-naive)",
    )
    parser.add_argument(
        "--precomputed-baselines", nargs="*", default=[],
        help="Names of pre-computed baseline models (from results/ dir)",
    )
    parser.add_argument(
        "--benchmark", type=str, default="zero-shot",
        help="Benchmark type for loading pre-computed baselines",
    )
    parser.add_argument(
        "--metrics", nargs="*", default=["WQL", "MASE"],
        help="Metrics to compare",
    )
    parser.add_argument(
        "--format", type=str, default="markdown",
        choices=["markdown", "latex", "csv"],
        help="Output format",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: stdout)",
    )

    args = parser.parse_args()

    # Load model results
    model_results = {}
    for path in args.results:
        path = Path(path)
        name = path.stem if path.is_file() else path.name
        try:
            model_results[name] = load_results(path)
            logger.info(f"Loaded: {name} ({len(model_results[name])} rows)")
        except FileNotFoundError as e:
            logger.error(f"Failed to load {path}: {e}")

    # Load pre-computed baselines
    for baseline_name in args.precomputed_baselines:
        df = load_precomputed_baseline(baseline_name, args.benchmark)
        if df is not None:
            model_results[baseline_name] = df
            logger.info(f"Loaded pre-computed: {baseline_name} ({len(df)} rows)")
        else:
            logger.warning(f"Pre-computed baseline not found: {baseline_name}")

    # Load comparison baseline
    baseline_df = None
    if args.baseline:
        try:
            baseline_df = load_results(args.baseline)
            logger.info(f"Loaded baseline: {args.baseline} ({len(baseline_df)} rows)")
        except FileNotFoundError:
            # Try pre-computed
            baseline_df = load_precomputed_baseline("seasonal-naive", args.benchmark)
            if baseline_df is not None:
                logger.info(f"Using pre-computed seasonal-naive baseline")

    if not model_results:
        logger.error("No results loaded. Exiting.")
        sys.exit(1)

    # Generate table
    table = generate_comparison(
        model_results, baseline_df, args.metrics, args.format
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(table)
        logger.info(f"Table saved: {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    sys.path.insert(0, str(SCRIPT_DIR))
    main()
