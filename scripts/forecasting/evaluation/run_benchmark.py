#!/usr/bin/env python3
"""Formal benchmark evaluation runner for TSM-Trainer.

Unified entry point for running all supported benchmarks against a trained model.
Produces structured results in CSV, JSON, and human-readable report formats.

Usage:
    # Evaluate on Chronos Benchmark II (zero-shot) — local datasets
    python run_benchmark.py \
        --model-path /path/to/checkpoint \
        --benchmarks chronos_ii \
        --datasets-root /path/to/local/eval_datasets/ \
        --output-dir results/experiments/

    # Quick evaluation with lite benchmark (~3 min on A100)
    python run_benchmark.py \
        --model-path /path/to/checkpoint \
        --benchmarks lite \
        --datasets-root /path/to/local/eval_datasets/ \
        --device cuda --torch-dtype bfloat16

    # Evaluate on multiple benchmarks
    python run_benchmark.py \
        --model-path /path/to/checkpoint \
        --benchmarks chronos_ii chronos_i \
        --datasets-root /path/to/local/eval_datasets/ \
        --output-dir results/experiments/

    # CPU evaluation (no GPU required)
    python run_benchmark.py \
        --model-path /path/to/checkpoint \
        --benchmarks lite \
        --datasets-root /path/to/local/eval_datasets/ \
        --device cpu --torch-dtype float32

Supported benchmarks:
    chronos_i    — Chronos Benchmark I (15 in-domain datasets)
    chronos_ii   — Chronos Benchmark II (27 zero-shot datasets)
    lite         — Lite Benchmark (5 datasets, ~3 min on A100)
    extended     — Extended Benchmark (15 datasets, ~15 min on A100)
    gift_eval    — GIFT-Eval (~98 configs, requires gift-eval library)
    fev_bench    — fev-bench (100 tasks, requires fev library)
"""

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("BenchmarkRunner")

# Resolve paths for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # tsm-trainer/
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SCRIPT_DIR))
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Environment & system info
# ---------------------------------------------------------------------------

def collect_environment_info() -> dict:
    """Collect system and runtime environment information."""
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "timestamp": datetime.now().isoformat(),
    }

    # PyTorch info
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            info["gpu_memory_gb"] = [
                round(torch.cuda.get_device_properties(i).total_mem / (1024**3), 1)
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        info["torch_version"] = "not installed"

    # Key library versions
    for lib in ["transformers", "datasets", "gluonts", "numpy", "pandas", "scipy"]:
        try:
            mod = __import__(lib)
            info[f"{lib}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    return info


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def resolve_device(requested: str) -> str:
    """Resolve device, falling back to CPU if CUDA unavailable."""
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            f"Requested device '{requested}' but CUDA is not available. "
            f"Falling back to CPU."
        )
        return "cpu"
    return requested


def load_forecaster(args):
    """Load a forecaster from model path."""
    from engine.forecaster import Chronos2Forecaster, ChronosBoltForecaster

    model_path = args.model_path
    device = resolve_device(args.device)

    # Auto-detect model type
    if "bolt" in model_path.lower():
        logger.info(f"Loading Chronos-Bolt model: {model_path} (device={device})")
        return ChronosBoltForecaster(
            model_path=model_path,
            device=device,
            torch_dtype=args.torch_dtype,
        )
    else:
        logger.info(f"Loading Chronos-2 model: {model_path} (device={device})")
        return Chronos2Forecaster(
            model_path=model_path,
            device=device,
            torch_dtype=args.torch_dtype,
        )


# ---------------------------------------------------------------------------
# Benchmark adapter factory
# ---------------------------------------------------------------------------

def get_adapter(benchmark_name: str, args):
    """Create a benchmark adapter by name."""
    configs_dir = SCRIPT_DIR / "configs"

    if benchmark_name == "chronos_ii":
        from benchmarks.chronos_bench import ChronosBenchmarkAdapter
        return ChronosBenchmarkAdapter(
            config_path=configs_dir / "zero-shot.yaml",
            benchmark_type="zero-shot",
            datasets_root=args.datasets_root,
            batch_size=args.batch_size,
        )
    elif benchmark_name == "chronos_i":
        from benchmarks.chronos_bench import ChronosBenchmarkAdapter
        return ChronosBenchmarkAdapter(
            config_path=configs_dir / "in-domain.yaml",
            benchmark_type="in-domain",
            datasets_root=args.datasets_root,
            batch_size=args.batch_size,
        )
    elif benchmark_name == "lite":
        from benchmarks.chronos_bench import ChronosLiteBenchmarkAdapter
        return ChronosLiteBenchmarkAdapter(
            config_path=configs_dir / "lite-benchmark.yaml",
            datasets_root=args.datasets_root,
            batch_size=args.batch_size,
        )
    elif benchmark_name == "extended":
        from benchmarks.chronos_bench import ChronosLiteBenchmarkAdapter
        return ChronosLiteBenchmarkAdapter(
            config_path=configs_dir / "extended-benchmark.yaml",
            datasets_root=args.datasets_root,
            batch_size=args.batch_size,
        )
    elif benchmark_name == "gift_eval":
        from benchmarks.gift_eval import GiftEvalAdapter
        return GiftEvalAdapter(
            data_dir=args.gift_eval_data,
            batch_size=args.batch_size,
        )
    elif benchmark_name == "fev_bench":
        from benchmarks.fev_bench import FevBenchAdapter
        return FevBenchAdapter(
            data_dir=args.fev_data,
            batch_size=args.batch_size,
        )
    else:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. "
            f"Available: chronos_i, chronos_ii, lite, extended, gift_eval, fev_bench"
        )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_benchmarks(args):
    """Run all selected benchmarks and save results."""
    start_time = time.time()

    # Collect environment info
    env_info = collect_environment_info()

    # Create output directory
    output_dir = Path(args.output_dir)
    if not args.experiment_name:
        args.experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = output_dir / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Print banner
    logger.info("")
    logger.info("=" * 72)
    logger.info("  TSM-Trainer Benchmark Evaluation")
    logger.info("=" * 72)
    logger.info(f"  Experiment : {args.experiment_name}")
    logger.info(f"  Model      : {args.model_path}")
    logger.info(f"  Benchmarks : {', '.join(args.benchmarks)}")
    logger.info(f"  Device     : {args.device} ({args.torch_dtype})")
    logger.info(f"  Batch size : {args.batch_size}")
    if args.datasets_root:
        logger.info(f"  Data root  : {args.datasets_root}")
    logger.info(f"  Output     : {experiment_dir}")
    logger.info("=" * 72)
    logger.info("")

    # Save experiment config
    config = {
        "experiment_id": args.experiment_name,
        "timestamp": datetime.now().isoformat(),
        "model": {
            "path": args.model_path,
            "device": args.device,
            "dtype": args.torch_dtype,
        },
        "evaluation": {
            "benchmarks": args.benchmarks,
            "batch_size": args.batch_size,
            "datasets_root": args.datasets_root,
        },
        "environment": env_info,
    }
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load model
    model_load_start = time.time()
    forecaster = load_forecaster(args)
    model_load_time = time.time() - model_load_start
    logger.info(f"Model loaded in {model_load_time:.1f}s")

    # Run each benchmark
    all_results = {}
    all_summaries = {}
    benchmark_timings = {}

    for idx, benchmark_name in enumerate(args.benchmarks, 1):
        logger.info("")
        logger.info("=" * 72)
        logger.info(f"  [{idx}/{len(args.benchmarks)}] BENCHMARK: {benchmark_name}")
        logger.info("=" * 72)

        try:
            adapter = get_adapter(benchmark_name, args)
            bm_start = time.time()

            # Evaluate
            results = adapter.evaluate(forecaster)
            summary = adapter.aggregate(results)

            bm_elapsed = time.time() - bm_start
            summary["elapsed_seconds"] = round(bm_elapsed, 1)
            benchmark_timings[benchmark_name] = bm_elapsed

            # Save per-benchmark results
            adapter.save_results(results, summary, experiment_dir, args.experiment_name)

            all_results[benchmark_name] = results
            all_summaries[benchmark_name] = summary

            # Print summary
            logger.info("")
            logger.info(f"  {benchmark_name} completed in {bm_elapsed:.1f}s")
            _log_summary(summary, indent=4)

        except Exception as e:
            logger.error(f"  {benchmark_name} FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            benchmark_timings[benchmark_name] = None

    # Overall timing
    total_elapsed = time.time() - start_time

    # Save overall summary
    overall_summary = {
        "experiment_id": args.experiment_name,
        "model_path": args.model_path,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "model_load_seconds": round(model_load_time, 1),
        "benchmark_timings": {
            k: round(v, 1) if v is not None else "FAILED"
            for k, v in benchmark_timings.items()
        },
        "benchmarks": all_summaries,
        "environment": env_info,
    }
    with open(experiment_dir / "summary.json", "w") as f:
        json.dump(overall_summary, f, indent=2, default=str)

    # Generate comprehensive report
    report = generate_report(
        all_results, all_summaries, args,
        total_elapsed, model_load_time, benchmark_timings, env_info,
    )
    report_path = experiment_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)

    # Final banner
    n_ok = sum(1 for v in benchmark_timings.values() if v is not None)
    n_fail = sum(1 for v in benchmark_timings.values() if v is None)
    logger.info("")
    logger.info("=" * 72)
    logger.info("  EVALUATION COMPLETE")
    logger.info(f"  Benchmarks : {n_ok} passed, {n_fail} failed")
    logger.info(f"  Total time : {total_elapsed:.1f}s")
    logger.info(f"  Results    : {experiment_dir}")
    logger.info(f"  Report     : {report_path}")
    logger.info("=" * 72)
    logger.info("")


def _log_summary(summary: dict, indent: int = 4):
    """Log summary metrics to console."""
    prefix = " " * indent
    for key, val in sorted(summary.items()):
        if key in ("benchmark_type", "terms", "subset", "elapsed_seconds"):
            continue
        if isinstance(val, float):
            logger.info(f"{prefix}{key}: {val:.4f}")
        elif isinstance(val, dict):
            continue
        elif isinstance(val, (int, str)):
            logger.info(f"{prefix}{key}: {val}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    all_results: dict,
    all_summaries: dict,
    args,
    total_elapsed: float,
    model_load_time: float,
    benchmark_timings: dict,
    env_info: dict,
) -> str:
    """Generate comprehensive markdown evaluation report."""
    lines = []

    # ---- Header ----
    lines.extend([
        f"# TSM-Trainer Benchmark Report",
        "",
        f"> Experiment: **{args.experiment_name}**",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ])

    # ---- Executive Summary ----
    lines.extend([
        "## 1. Executive Summary",
        "",
    ])

    n_ok = sum(1 for v in benchmark_timings.values() if v is not None)
    n_fail = sum(1 for v in benchmark_timings.values() if v is None)
    lines.append(f"| Item | Value |")
    lines.append(f"|------|-------|")
    lines.append(f"| Model | `{args.model_path}` |")
    lines.append(f"| Device | {args.device} ({args.torch_dtype}) |")
    lines.append(f"| Benchmarks run | {n_ok} passed / {n_fail} failed |")
    lines.append(f"| Total time | {_fmt_time(total_elapsed)} |")
    lines.append(f"| Model load time | {_fmt_time(model_load_time)} |")
    lines.append("")

    # Key metrics summary table
    if all_summaries:
        lines.extend([
            "### Key Metrics",
            "",
            "| Benchmark | Datasets | WQL | MASE | Time |",
            "|-----------|----------|-----|------|------|",
        ])
        for bm_name, summary in all_summaries.items():
            n_ds = summary.get("n_datasets", summary.get("n_tasks", "N/A"))
            wql = _fmt_metric(summary.get("avg_wql"))
            mase = _fmt_metric(summary.get("avg_mase"))
            bm_time = _fmt_time(benchmark_timings.get(bm_name))
            lines.append(f"| {bm_name} | {n_ds} | {wql} | {mase} | {bm_time} |")
        lines.append("")

    # ---- Per-benchmark details ----
    lines.extend([
        "---",
        "",
        "## 2. Per-Benchmark Results",
        "",
    ])

    for bm_name, summary in all_summaries.items():
        bm_results = all_results.get(bm_name)
        lines.extend(_render_benchmark_section(bm_name, summary, bm_results))

    # ---- Per-dataset breakdown ----
    if any(r is not None for r in all_results.values()):
        lines.extend([
            "---",
            "",
            "## 3. Per-Dataset Results",
            "",
        ])
        for bm_name, results_df in all_results.items():
            if results_df is None or results_df.empty:
                continue
            lines.extend(_render_dataset_table(bm_name, results_df))

    # ---- Environment ----
    lines.extend([
        "---",
        "",
        "## 4. Environment",
        "",
        "| Component | Version |",
        "|-----------|---------|",
    ])
    lines.append(f"| Python | {env_info.get('python_version', 'N/A')} |")
    lines.append(f"| PyTorch | {env_info.get('torch_version', 'N/A')} |")
    lines.append(f"| Transformers | {env_info.get('transformers_version', 'N/A')} |")
    lines.append(f"| Datasets | {env_info.get('datasets_version', 'N/A')} |")
    lines.append(f"| GluonTS | {env_info.get('gluonts_version', 'N/A')} |")
    lines.append(f"| Platform | {env_info.get('platform', 'N/A')} |")
    lines.append(f"| Hostname | {env_info.get('hostname', 'N/A')} |")

    if env_info.get("cuda_available"):
        gpu_devices = env_info.get("gpu_devices", [])
        gpu_mem = env_info.get("gpu_memory_gb", [])
        for i, (name, mem) in enumerate(zip(gpu_devices, gpu_mem)):
            lines.append(f"| GPU {i} | {name} ({mem} GB) |")
        lines.append(f"| CUDA | {env_info.get('cuda_version', 'N/A')} |")
    else:
        lines.append(f"| GPU | None (CPU mode) |")

    lines.extend(["", ""])

    # ---- Timing breakdown ----
    lines.extend([
        "---",
        "",
        "## 5. Timing Breakdown",
        "",
        "| Phase | Time |",
        "|-------|------|",
        f"| Model loading | {_fmt_time(model_load_time)} |",
    ])
    for bm_name, bm_time in benchmark_timings.items():
        status = _fmt_time(bm_time) if bm_time is not None else "FAILED"
        lines.append(f"| {bm_name} | {status} |")
    lines.append(f"| **Total** | **{_fmt_time(total_elapsed)}** |")
    lines.extend(["", ""])

    # ---- Reproduction ----
    lines.extend([
        "---",
        "",
        "## 6. Reproduction",
        "",
        "```bash",
        f"python run_benchmark.py \\",
        f"    --model-path {args.model_path} \\",
        f"    --benchmarks {' '.join(args.benchmarks)} \\",
    ])
    if args.datasets_root:
        lines.append(f"    --datasets-root {args.datasets_root} \\")
    lines.extend([
        f"    --output-dir {args.output_dir} \\",
        f"    --device {args.device} \\",
        f"    --torch-dtype {args.torch_dtype} \\",
        f"    --batch-size {args.batch_size}",
        "```",
        "",
    ])

    return "\n".join(lines)


def _render_benchmark_section(bm_name: str, summary: dict, results_df) -> list[str]:
    """Render a single benchmark section for the report."""
    lines = [
        f"### {bm_name}",
        "",
    ]

    # Metadata
    bm_type = summary.get("benchmark_type", "")
    n_items = summary.get("n_datasets", summary.get("n_tasks", "N/A"))
    elapsed = summary.get("elapsed_seconds", 0)

    lines.append(f"- **Type**: {bm_type or bm_name}")
    lines.append(f"- **Datasets/Tasks**: {n_items}")
    lines.append(f"- **Evaluation time**: {_fmt_time(elapsed)}")
    lines.append("")

    # Metrics table
    metric_keys = [
        k for k in sorted(summary.keys())
        if k not in ("benchmark_type", "n_datasets", "n_tasks", "elapsed_seconds",
                      "terms", "subset")
        and isinstance(summary[k], (int, float))
    ]
    if metric_keys:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key in metric_keys:
            val = summary[key]
            if isinstance(val, float):
                lines.append(f"| {key} | {val:.4f} |")
            else:
                lines.append(f"| {key} | {val} |")
        lines.append("")

    # Relative scores (if available)
    for key in ["agg_rel_wql", "agg_rel_mase"]:
        if key in summary:
            val = summary[key]
            direction = "better" if val < 1.0 else "worse"
            pct = abs(1.0 - val) * 100
            lines.append(
                f"- **{key}**: {val:.4f} "
                f"({pct:.1f}% {direction} than Seasonal Naive)"
            )

    if any(k.startswith("agg_rel_") for k in summary):
        lines.append("")

    return lines


def _render_dataset_table(bm_name: str, results_df) -> list[str]:
    """Render per-dataset results table."""
    import numpy as np

    lines = [
        f"### {bm_name} — Per-Dataset",
        "",
    ]

    # Determine columns to show
    display_cols = ["dataset"]
    for col in ["model", "WQL", "MASE", "sql", "crps", "smape"]:
        if col in results_df.columns:
            display_cols.append(col)

    # Header
    lines.append("| " + " | ".join(display_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(display_cols)) + "|")

    # Rows
    for _, row in results_df.iterrows():
        cells = []
        for col in display_cols:
            val = row.get(col, "")
            if isinstance(val, float):
                if np.isnan(val):
                    cells.append("FAILED")
                else:
                    cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    # Footer with averages
    lines.append("|" + "|".join(["---"] * len(display_cols)) + "|")
    avg_cells = ["**Average**"]
    for col in display_cols[1:]:
        if col in results_df.columns and results_df[col].dtype in ("float64", "float32"):
            avg = results_df[col].dropna().mean()
            avg_cells.append(f"**{avg:.4f}**")
        else:
            avg_cells.append("")
    lines.append("| " + " | ".join(avg_cells) + " |")

    lines.extend(["", ""])
    return lines


def _fmt_metric(val) -> str:
    """Format a metric value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        import math
        if math.isnan(val):
            return "N/A"
        return f"{val:.4f}"
    return str(val)


def _fmt_time(seconds) -> str:
    """Format time duration for display."""
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="TSM-Trainer Formal Benchmark Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to model checkpoint or HuggingFace model ID",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", required=True,
        choices=["chronos_i", "chronos_ii", "lite", "extended", "gift_eval", "fev_bench"],
        help="Benchmark(s) to run",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str,
        default="results/experiments/",
        help="Base directory for experiment results",
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help="Experiment identifier (default: auto-generated timestamp)",
    )

    # Model settings
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference: cuda, cuda:0, cpu (default: cuda)")
    parser.add_argument("--torch-dtype", type=str, default="float32",
                        help="Model dtype: float32, bfloat16 (default: float32)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Inference batch size (default: 32)")

    # Data settings
    parser.add_argument(
        "--datasets-root", type=str, default=None,
        help="Root directory for local datasets (enables local-only mode)",
    )
    parser.add_argument(
        "--gift-eval-data", type=str, default=None,
        help="Local path to GIFT-Eval data",
    )
    parser.add_argument(
        "--fev-data", type=str, default=None,
        help="Local path to fev-bench data",
    )

    # Comparison
    parser.add_argument(
        "--compare-with", type=str, nargs="*", default=None,
        help="Result directories to compare against",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmarks(args)
