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
    chronos_i        — Chronos Benchmark I (15 in-domain datasets)
    chronos_ii       — Chronos Benchmark II (27 zero-shot datasets)
    chronos_lite     — Chronos Lite (5 datasets, ~3 min on A100)
    chronos_extended — Chronos Extended (15 datasets, ~15 min on A100)
    chronos_full     — Chronos Full (42 datasets, ~90 min on A100)
    gift_eval        — GIFT-Eval (~98 configs, requires gift-eval library)
    fev_bench        — fev-bench (100 tasks, requires fev library)
    lite             — Alias for chronos_lite
    extended         — Alias for chronos_extended
"""

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

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
    for lib in ["transformers", "datasets", "gluonts", "numpy", "pandas", "scipy", "pyarrow"]:
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


def validate_model_path(model_path: str) -> list[str]:
    """Validate model path before loading.

    Checks that the model directory exists and contains expected config files.
    Returns list of warnings (empty = all good).
    """
    warnings = []
    path = Path(model_path)

    if not path.exists():
        # Could be a HuggingFace model ID — skip local validation
        return []

    if not path.is_dir():
        warnings.append(f"Model path is not a directory: {path}")
        return warnings

    # Check for expected config files
    expected_configs = ["config.json"]
    for cfg in expected_configs:
        if not (path / cfg).exists():
            warnings.append(f"Missing expected config file: {cfg}")

    # Check for model weights
    weight_patterns = ["*.safetensors", "*.bin", "pytorch_model*"]
    has_weights = False
    for pattern in weight_patterns:
        if list(path.glob(pattern)):
            has_weights = True
            break
    if not has_weights:
        warnings.append("No model weight files found (*.safetensors, *.bin)")

    return warnings


def get_model_info(model_path: str) -> dict:
    """Extract model metadata from config files."""
    info = {"path": model_path}
    path = Path(model_path)

    if path.exists() and path.is_dir():
        config_path = path / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                info["model_type"] = config.get("model_type", "unknown")
                info["hidden_size"] = config.get("hidden_size") or config.get("d_model")
                info["num_layers"] = config.get("num_hidden_layers") or config.get("num_layers")
                info["num_heads"] = config.get("num_attention_heads") or config.get("num_heads")
            except Exception:
                pass

        # Count parameters from safetensors metadata
        safetensors_files = list(path.glob("*.safetensors"))
        if safetensors_files:
            try:
                from safetensors import safe_open
                total_params = 0
                for sf in safetensors_files:
                    with safe_open(str(sf), framework="pt") as f:
                        for key in f.keys():
                            tensor = f.get_tensor(key)
                            total_params += tensor.numel()
                info["total_params"] = total_params
                info["total_params_m"] = round(total_params / 1e6, 1)
            except Exception:
                pass

        # Get total size on disk
        total_size = sum(
            f.stat().st_size
            for f in path.rglob("*")
            if f.is_file()
        )
        info["disk_size_gb"] = round(total_size / (1024 ** 3), 2)

    return info


def get_gpu_memory_stats() -> dict | None:
    """Get current GPU memory usage stats."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "allocated_gb": round(torch.cuda.memory_allocated() / (1024 ** 3), 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / (1024 ** 3), 2),
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024 ** 3), 2),
            }
    except Exception:
        pass
    return None


def reset_gpu_memory_stats():
    """Reset GPU peak memory tracking."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def load_forecaster(args):
    """Load a forecaster from model path with pre-validation."""
    from engine.forecaster import Chronos2Forecaster, ChronosBoltForecaster

    model_path = args.model_path
    device = resolve_device(args.device)

    # Validate model path
    model_warnings = validate_model_path(model_path)
    for w in model_warnings:
        logger.warning(f"  Model validation: {w}")

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
    """Create a benchmark adapter by name.

    Available benchmarks:
        chronos_i    — Chronos Benchmark I (15 in-domain datasets)
        chronos_ii   — Chronos Benchmark II (27 zero-shot datasets)
        chronos_lite  — Quick validation subset (5 datasets, ~3 min)
        chronos_extended — Thorough validation (15 datasets, ~15 min)
        chronos_full  — All Chronos datasets combined (42 datasets)
        gift_eval    — GIFT-Eval (~98 tasks, requires gift-eval library)
        fev_bench    — fev-bench (100 tasks, requires fev library)
    """
    configs_dir = SCRIPT_DIR / "configs"

    if benchmark_name == "chronos_ii":
        from benchmarks.chronos_bench import ChronosBenchmarkAdapter
        return ChronosBenchmarkAdapter(
            config_path=configs_dir / "chronos-ii.yaml",
            benchmark_type="zero-shot",
            datasets_root=args.datasets_root,
            batch_size=args.batch_size,
        )
    elif benchmark_name == "chronos_i":
        from benchmarks.chronos_bench import ChronosBenchmarkAdapter
        return ChronosBenchmarkAdapter(
            config_path=configs_dir / "chronos-i.yaml",
            benchmark_type="in-domain",
            datasets_root=args.datasets_root,
            batch_size=args.batch_size,
        )
    elif benchmark_name in ("lite", "chronos_lite"):
        from benchmarks.chronos_bench import ChronosLiteBenchmarkAdapter
        return ChronosLiteBenchmarkAdapter(
            config_path=configs_dir / "chronos-lite.yaml",
            datasets_root=args.datasets_root,
            batch_size=args.batch_size,
        )
    elif benchmark_name in ("extended", "chronos_extended"):
        from benchmarks.chronos_bench import ChronosLiteBenchmarkAdapter
        return ChronosLiteBenchmarkAdapter(
            config_path=configs_dir / "chronos-extended.yaml",
            datasets_root=args.datasets_root,
            batch_size=args.batch_size,
        )
    elif benchmark_name == "chronos_full":
        from benchmarks.chronos_bench import ChronosBenchmarkAdapter
        return ChronosBenchmarkAdapter(
            config_path=configs_dir / "chronos-full.yaml",
            benchmark_type="full",
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
            f"Available: chronos_i, chronos_ii, chronos_lite, chronos_extended, "
            f"chronos_full, lite, extended, gift_eval, fev_bench"
        )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_benchmarks(args):
    """Run all selected benchmarks and save results."""
    start_time = time.time()

    # Set random seed for reproducibility
    if args.seed is not None:
        import numpy as np
        import torch
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    # Collect environment info
    env_info = collect_environment_info()

    # Get model metadata
    model_info = get_model_info(args.model_path)

    # Create output directory
    output_dir = Path(args.output_dir)
    if not args.experiment_name:
        args.experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = output_dir / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Pre-validate benchmark configs
    from engine.evaluator import validate_config
    configs_dir = SCRIPT_DIR / "configs"
    config_map = {
        "chronos_ii": "chronos-ii.yaml",
        "chronos_i": "chronos-i.yaml",
        "lite": "chronos-lite.yaml",
        "chronos_lite": "chronos-lite.yaml",
        "extended": "chronos-extended.yaml",
        "chronos_extended": "chronos-extended.yaml",
        "chronos_full": "chronos-full.yaml",
    }
    for benchmark_name in args.benchmarks:
        config_file = config_map.get(benchmark_name)
        if config_file:
            config_path = configs_dir / config_file
            errors = validate_config(config_path)
            if errors:
                logger.error(f"Config validation failed for {benchmark_name}:")
                for err in errors:
                    logger.error(f"  - {err}")
                raise ValueError(f"Invalid benchmark config: {benchmark_name}")

    # Print banner
    logger.info("")
    logger.info("=" * 72)
    logger.info("  TSM-Trainer Benchmark Evaluation")
    logger.info("=" * 72)
    logger.info(f"  Experiment : {args.experiment_name}")
    logger.info(f"  Model      : {args.model_path}")
    if model_info.get("total_params_m"):
        logger.info(f"  Parameters : {model_info['total_params_m']}M")
    if model_info.get("model_type"):
        logger.info(f"  Model type : {model_info['model_type']}")
    logger.info(f"  Benchmarks : {', '.join(args.benchmarks)}")
    logger.info(f"  Device     : {args.device} ({args.torch_dtype})")
    logger.info(f"  Batch size : {args.batch_size}")
    if args.seed is not None:
        logger.info(f"  Seed       : {args.seed}")
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
            **{k: v for k, v in model_info.items() if k != "path"},
        },
        "evaluation": {
            "benchmarks": args.benchmarks,
            "batch_size": args.batch_size,
            "datasets_root": args.datasets_root,
            "seed": args.seed,
        },
        "environment": env_info,
    }
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Reset GPU memory tracking
    reset_gpu_memory_stats()

    # Load model
    model_load_start = time.time()
    forecaster = load_forecaster(args)
    model_load_time = time.time() - model_load_start
    gpu_after_load = get_gpu_memory_stats()
    logger.info(f"Model loaded in {model_load_time:.1f}s")
    if gpu_after_load:
        logger.info(
            f"  GPU memory after load: "
            f"{gpu_after_load['allocated_gb']:.1f} GB allocated, "
            f"{gpu_after_load['reserved_gb']:.1f} GB reserved"
        )

    # Pre-validate datasets for all benchmarks
    if args.datasets_root:
        from engine.evaluator import validate_datasets
        logger.info("Pre-validating datasets...")
        all_valid = True
        for benchmark_name in args.benchmarks:
            config_map = {
                "chronos_ii": "zero-shot.yaml",
                "chronos_i": "in-domain.yaml",
                "lite": "lite-benchmark.yaml",
                "extended": "extended-benchmark.yaml",
            }
            config_file = config_map.get(benchmark_name)
            if config_file:
                config_path = SCRIPT_DIR / "configs" / config_file
                if config_path.exists():
                    report = validate_datasets(config_path, args.datasets_root)
                    if report["missing"] > 0:
                        all_valid = False
                        missing_names = [
                            name for name, info in report["datasets"].items()
                            if info["status"] == "missing"
                        ]
                        logger.warning(
                            f"  {benchmark_name}: {report['missing']}/{report['total']} "
                            f"datasets MISSING: {missing_names}"
                        )
                    else:
                        logger.info(
                            f"  {benchmark_name}: {report['found']}/{report['total']} "
                            f"datasets OK"
                        )
        if not all_valid:
            logger.warning("Some datasets are missing. Evaluation will proceed but may fail.")
        logger.info("")

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

            # Determine checkpoint path for resume support
            checkpoint_path = None
            if getattr(args, "resume", False):
                checkpoint_path = experiment_dir / f".checkpoint_{benchmark_name}.json"

            # Evaluate (with optional checkpointing)
            if checkpoint_path is not None:
                # Use Evaluator directly with checkpoint support
                from engine.forecaster import Chronos2Forecaster, ChronosBoltForecaster
                from engine.evaluator import Evaluator

                evaluator = Evaluator(
                    forecaster=forecaster,
                    batch_size=args.batch_size,
                    datasets_root=args.datasets_root,
                )
                config_map = {
                    "chronos_ii": "zero-shot.yaml",
                    "chronos_i": "in-domain.yaml",
                    "lite": "lite-benchmark.yaml",
                    "extended": "extended-benchmark.yaml",
                }
                config_file = config_map.get(benchmark_name)
                if config_file:
                    cfg_path = SCRIPT_DIR / "configs" / config_file
                    results = evaluator.evaluate_benchmark(
                        cfg_path,
                        checkpoint_path=checkpoint_path,
                    )
                    summary = adapter.aggregate(results)
                else:
                    results = adapter.evaluate(forecaster)
                    summary = adapter.aggregate(results)
            else:
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

    # Capture peak GPU memory
    gpu_peak = get_gpu_memory_stats()

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
        "model_info": model_info,
        "gpu_memory": gpu_peak,
        "environment": env_info,
    }
    with open(experiment_dir / "summary.json", "w") as f:
        json.dump(overall_summary, f, indent=2, default=str)

    # Generate comprehensive report
    report = generate_report(
        all_results, all_summaries, args,
        total_elapsed, model_load_time, benchmark_timings, env_info,
        model_info=model_info,
        gpu_memory=gpu_peak,
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
    if gpu_peak:
        logger.info(f"  GPU peak   : {gpu_peak['peak_allocated_gb']:.1f} GB allocated")
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
    model_info: dict | None = None,
    gpu_memory: dict | None = None,
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
    if model_info and model_info.get("total_params_m"):
        lines.append(f"| Parameters | {model_info['total_params_m']}M |")
    if model_info and model_info.get("model_type"):
        lines.append(f"| Architecture | {model_info['model_type']} |")
    lines.append(f"| Device | {args.device} ({args.torch_dtype}) |")
    lines.append(f"| Benchmarks run | {n_ok} passed / {n_fail} failed |")
    lines.append(f"| Total time | {_fmt_time(total_elapsed)} |")
    lines.append(f"| Model load time | {_fmt_time(model_load_time)} |")
    if gpu_memory and gpu_memory.get("peak_allocated_gb"):
        lines.append(f"| GPU peak memory | {gpu_memory['peak_allocated_gb']:.1f} GB |")
    if args.seed is not None:
        lines.append(f"| Random seed | {args.seed} |")
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

    # ---- GPU Memory ----
    if gpu_memory:
        lines.extend([
            "---",
            "",
            "## 5b. GPU Memory",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Peak allocated | {gpu_memory.get('peak_allocated_gb', 'N/A')} GB |",
            f"| Peak reserved | {gpu_memory.get('peak_reserved_gb', 'N/A')} GB |",
            f"| Current allocated | {gpu_memory.get('allocated_gb', 'N/A')} GB |",
            f"| Current reserved | {gpu_memory.get('reserved_gb', 'N/A')} GB |",
            "",
        ])

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
    ])
    if args.seed is not None:
        # Replace last line to add continuation
        lines[-1] = lines[-1] + " \\"
        lines.append(f"    --seed {args.seed}")
    lines.extend([
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
    for col in ["WQL", "MASE", "n_series", "elapsed_s", "sql", "crps", "smape"]:
        if col in results_df.columns:
            display_cols.append(col)

    # Column display names
    col_names = {
        "dataset": "Dataset",
        "WQL": "WQL",
        "MASE": "MASE",
        "n_series": "Series",
        "elapsed_s": "Time(s)",
        "sql": "SQL",
        "crps": "CRPS",
        "smape": "sMAPE",
    }

    # Header
    header = [col_names.get(c, c) for c in display_cols]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(display_cols)) + "|")

    # Rows
    for _, row in results_df.iterrows():
        cells = []
        for col in display_cols:
            val = row.get(col, "")
            if isinstance(val, float):
                if np.isnan(val):
                    cells.append("FAILED")
                elif col == "elapsed_s":
                    cells.append(f"{val:.1f}")
                else:
                    cells.append(f"{val:.4f}")
            elif isinstance(val, int):
                cells.append(f"{val:,}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    # Footer with averages
    lines.append("|" + "|".join(["---"] * len(display_cols)) + "|")
    avg_cells = ["**Average**"]
    for col in display_cols[1:]:
        if col in results_df.columns and results_df[col].dtype in ("float64", "float32"):
            if col == "elapsed_s":
                total = results_df[col].dropna().sum()
                avg_cells.append(f"**{total:.1f}** (total)")
            elif col == "n_series":
                total = int(results_df[col].dropna().sum())
                avg_cells.append(f"**{total:,}** (total)")
            else:
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
        choices=[
            "chronos_i", "chronos_ii", "chronos_lite", "chronos_extended",
            "chronos_full", "lite", "extended", "gift_eval", "fev_bench",
        ],
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

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (sets numpy + torch seeds)",
    )

    # Diagnostics
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check data availability and model loading without running evaluation",
    )

    # Resume / checkpointing
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if previous run was interrupted",
    )

    # Logging
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug-level logging for detailed output",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce logging to warnings and errors only",
    )

    return parser.parse_args()


def run_dry_run(args):
    """Check data availability and model loading without running evaluation."""
    import yaml
    from engine.evaluator import validate_config

    logger.info("")
    logger.info("=" * 72)
    logger.info("  DRY RUN — Checking data and model availability")
    logger.info("=" * 72)
    logger.info("")

    # Check model
    model_path = Path(args.model_path)
    if model_path.exists():
        logger.info(f"  Model path: {model_path} [EXISTS]")
        config_files = list(model_path.glob("config*.json"))
        logger.info(f"    Config files: {[f.name for f in config_files]}")
        model_warnings = validate_model_path(args.model_path)
        for w in model_warnings:
            logger.warning(f"    {w}")
        if not model_warnings:
            logger.info(f"    Model validation: OK")
        # Show model info
        info = get_model_info(args.model_path)
        if info.get("total_params_m"):
            logger.info(f"    Parameters: {info['total_params_m']}M")
        if info.get("disk_size_gb"):
            logger.info(f"    Disk size: {info['disk_size_gb']} GB")
    else:
        logger.warning(f"  Model path: {model_path} [NOT FOUND]")

    # Check each benchmark's datasets
    configs_dir = SCRIPT_DIR / "configs"
    for benchmark_name in args.benchmarks:
        logger.info("")
        logger.info(f"  --- Benchmark: {benchmark_name} ---")

        config_map = {
            "chronos_ii": "zero-shot.yaml",
            "chronos_i": "in-domain.yaml",
            "lite": "lite-benchmark.yaml",
            "extended": "extended-benchmark.yaml",
        }
        config_file = config_map.get(benchmark_name)
        if not config_file:
            logger.info(f"    (external benchmark, skipping dataset check)")
            continue

        config_path = configs_dir / config_file
        if not config_path.exists():
            logger.warning(f"    Config file NOT FOUND: {config_path}")
            continue

        # Validate config schema
        config_errors = validate_config(config_path)
        if config_errors:
            logger.warning(f"    Config validation FAILED:")
            for err in config_errors:
                logger.warning(f"      - {err}")
        else:
            logger.info(f"    Config validation: OK")

        with open(config_path) as f:
            configs = yaml.safe_load(f)

        for ds_config in configs:
            ds_name = ds_config["name"]
            found = False
            checked_paths = []

            if args.datasets_root:
                local_path = Path(args.datasets_root) / ds_name
                checked_paths.append(str(local_path))
                if local_path.exists():
                    arrow_files = list(local_path.glob("*.arrow"))
                    parquet_files = list(local_path.glob("*.parquet"))
                    data_files = arrow_files + parquet_files
                    total_size = sum(f.stat().st_size for f in data_files)
                    logger.info(
                        f"    {ds_name}: [FOUND] "
                        f"{len(data_files)} data file(s), "
                        f"{total_size / (1024*1024):.1f} MB"
                    )
                    found = True

            if not found:
                logger.warning(
                    f"    {ds_name}: [NOT FOUND] "
                    f"Checked: {checked_paths or ['(no datasets_root set)']}"
                )

    # Environment info
    env_info = collect_environment_info()
    logger.info("")
    logger.info("  --- Environment ---")
    logger.info(f"    Python: {env_info.get('python_version', 'N/A')}")
    logger.info(f"    PyTorch: {env_info.get('torch_version', 'N/A')}")
    logger.info(f"    CUDA: {env_info.get('cuda_available', False)}")
    logger.info(f"    PyArrow: {env_info.get('pyarrow_version', 'N/A')}")
    logger.info(f"    Datasets: {env_info.get('datasets_version', 'N/A')}")
    logger.info(f"    GluonTS: {env_info.get('gluonts_version', 'N/A')}")
    logger.info("")
    logger.info("  Dry run complete. No evaluation was performed.")
    logger.info("=" * 72)


if __name__ == "__main__":
    args = parse_args()

    # Configure logging level
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=log_level,
    )

    if args.dry_run:
        run_dry_run(args)
    else:
        run_benchmarks(args)
