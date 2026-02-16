#!/usr/bin/env python3
"""Formal benchmark evaluation runner for TSM-Trainer.

Unified entry point for running all supported benchmarks against a trained model.
Produces structured results in CSV, JSON, and human-readable report formats.

Usage:
    # Evaluate on Chronos Benchmark II (zero-shot)
    python run_benchmark.py \
        --model-path /path/to/checkpoint \
        --benchmarks chronos_ii \
        --output-dir results/experiments/exp001/

    # Evaluate on multiple benchmarks
    python run_benchmark.py \
        --model-path /path/to/checkpoint \
        --benchmarks chronos_ii chronos_i \
        --output-dir results/experiments/exp001/

    # With local dataset paths (offline mode)
    python run_benchmark.py \
        --model-path /path/to/checkpoint \
        --benchmarks chronos_ii \
        --datasets-root /path/to/local/eval_datasets/ \
        --output-dir results/experiments/exp001/

    # GIFT-Eval (requires gift-eval library)
    python run_benchmark.py \
        --model-path /path/to/checkpoint \
        --benchmarks gift_eval \
        --gift-eval-data /path/to/local/gift-eval/ \
        --output-dir results/experiments/exp001/

    # fev-bench (requires fev library)
    python run_benchmark.py \
        --model-path /path/to/checkpoint \
        --benchmarks fev_bench \
        --output-dir results/experiments/exp001/

    # All benchmarks with comparison
    python run_benchmark.py \
        --model-path /path/to/checkpoint \
        --benchmarks chronos_ii gift_eval fev_bench \
        --output-dir results/experiments/exp001/ \
        --compare-with results/baselines/seasonal-naive/

Supported benchmarks:
    chronos_i    — Chronos Benchmark I (15 in-domain datasets)
    chronos_ii   — Chronos Benchmark II (27 zero-shot datasets)
    lite         — Lite Benchmark (5 datasets, ~3 min)
    extended     — Extended Benchmark (15 datasets, ~15 min)
    gift_eval    — GIFT-Eval (~98 configs, requires gift-eval library)
    fev_bench    — fev-bench (100 tasks, requires fev library)
"""

import argparse
import json
import logging
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
sys.path.insert(0, str(SCRIPT_DIR))


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


def load_forecaster(args):
    """Load a forecaster from model path."""
    from engine.forecaster import Chronos2Forecaster, ChronosBoltForecaster

    model_path = args.model_path

    # Auto-detect model type
    if "bolt" in model_path.lower():
        logger.info(f"Loading Chronos-Bolt model: {model_path}")
        return ChronosBoltForecaster(
            model_path=model_path,
            device=args.device,
            torch_dtype=args.torch_dtype,
        )
    else:
        logger.info(f"Loading Chronos-2 model: {model_path}")
        return Chronos2Forecaster(
            model_path=model_path,
            device=args.device,
            torch_dtype=args.torch_dtype,
        )


def run_benchmarks(args):
    """Run all selected benchmarks and save results."""
    start_time = time.time()

    # Create output directory
    output_dir = Path(args.output_dir)
    if not args.experiment_name:
        args.experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = output_dir / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

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
    }
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load model
    forecaster = load_forecaster(args)

    # Run each benchmark
    all_results = {}
    all_summaries = {}

    for benchmark_name in args.benchmarks:
        logger.info(f"\n{'=' * 72}")
        logger.info(f"  BENCHMARK: {benchmark_name}")
        logger.info(f"{'=' * 72}")

        try:
            adapter = get_adapter(benchmark_name, args)
            bm_start = time.time()

            # Evaluate
            results = adapter.evaluate(forecaster)
            summary = adapter.aggregate(results)

            bm_elapsed = time.time() - bm_start
            summary["elapsed_seconds"] = bm_elapsed

            # Save
            adapter.save_results(results, summary, experiment_dir, args.experiment_name)

            all_results[benchmark_name] = results
            all_summaries[benchmark_name] = summary

            logger.info(f"\n  {benchmark_name} completed in {bm_elapsed:.0f}s")
            for key, val in summary.items():
                if isinstance(val, float):
                    logger.info(f"    {key}: {val:.4f}")
                elif isinstance(val, (int, str)):
                    logger.info(f"    {key}: {val}")

        except Exception as e:
            logger.error(f"  {benchmark_name} FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Save overall summary
    total_elapsed = time.time() - start_time
    overall_summary = {
        "experiment_id": args.experiment_name,
        "model_path": args.model_path,
        "total_elapsed_seconds": total_elapsed,
        "benchmarks": all_summaries,
    }
    with open(experiment_dir / "summary.json", "w") as f:
        json.dump(overall_summary, f, indent=2, default=str)

    # Generate human-readable report
    report = generate_report(all_summaries, args, total_elapsed)
    report_path = experiment_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"\n{'=' * 72}")
    logger.info(f"  ALL BENCHMARKS COMPLETE")
    logger.info(f"  Total time: {total_elapsed:.0f}s")
    logger.info(f"  Results: {experiment_dir}")
    logger.info(f"  Report:  {report_path}")
    logger.info(f"{'=' * 72}")


def generate_report(summaries: dict, args, elapsed: float) -> str:
    """Generate a human-readable markdown report."""
    lines = [
        f"# Benchmark Report: {args.experiment_name}",
        "",
        f"**Model**: `{args.model_path}`",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Device**: {args.device} ({args.torch_dtype})",
        f"**Total time**: {elapsed:.0f}s",
        "",
        "---",
        "",
    ]

    for bm_name, summary in summaries.items():
        lines.append(f"## {bm_name}")
        lines.append("")

        n_tasks = summary.get("n_tasks", "N/A")
        lines.append(f"- **Tasks**: {n_tasks}")

        for key, val in sorted(summary.items()):
            if key in ("n_tasks", "benchmark_type", "terms", "subset"):
                continue
            if isinstance(val, float):
                lines.append(f"- **{key}**: {val:.4f}")
            elif isinstance(val, dict):
                continue  # Skip nested dicts in summary
            else:
                lines.append(f"- **{key}**: {val}")

        lines.append("")

    return "\n".join(lines)


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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-dtype", type=str, default="float32")
    parser.add_argument("--batch-size", type=int, default=32)

    # Data settings
    parser.add_argument(
        "--datasets-root", type=str, default=None,
        help="Root directory for local datasets (offline mode)",
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
