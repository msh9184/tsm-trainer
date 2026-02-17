"""Enhanced Benchmark Callback for training-time validation.

Multi-tier benchmark evaluation with composite checkpoint selection
and hierarchical TensorBoard logging. Replaces LiteBenchmarkCallback
with a more capable and extensible architecture.

Features:
- Configurable benchmark configs from scripts/forecasting/evaluation/configs/
- Available configs: chronos-lite, chronos-extended, chronos-i, chronos-ii, chronos-full
- Composite checkpoint metric: Weighted WQL + MASE
- Per-dataset TensorBoard logging under benchmark/
- Top-K checkpoint management by configurable metric
- Distributed evaluation: ALL ranks participate in forward passes (FSDP-compatible)
- Evaluation timeout to prevent training hangs
- Uses shared evaluation engine (no code duplication)
- Persistent file reports: per-eval JSON + cumulative CSV
- NeMo-style metric-encoded checkpoint filenames in best_checkpoints/
- Resume-safe state recovery from disk

Usage in training config YAML:
    benchmark_config: configs/chronos-lite.yaml
    benchmark_eval_steps: 200
    benchmark_top_k_checkpoints: 3
    benchmark_batch_size: 256
    benchmark_checkpoint_metric: composite  # "wql" | "mase" | "composite"
    benchmark_composite_weights:
      wql: 0.6
      mase: 0.4
    benchmark_datasets_root: /group-volume/ts-dataset/benchmarks/chronos
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from transformers import TrainerCallback

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)

# Ensure evaluation engine is importable
# Path: callbacks/benchmark_callback.py → training/ → forecasting/ → forecasting/evaluation/
_EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "evaluation"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

# Regex for parsing metric-encoded checkpoint filenames
_METRIC_CKPT_RE = re.compile(
    r"model-step=(\d+)-wql=([\d.]+)-mase=([\d.]+)-composite=([\d.]+)\.safetensors"
)

# Box-drawing constants for consistent formatting
_BOX_W = 74
_HLINE = "\u2500" * _BOX_W
_TOP = f"\u250c{_HLINE}\u2510"
_BOT = f"\u2514{_HLINE}\u2518"
_SEP = f"\u251c{_HLINE}\u2524"
_L = "\u2502"  # left/right border


def _boxline(text: str) -> str:
    """Format a single line inside a box (left-aligned, padded to BOX_W)."""
    return f"{_L} {text.ljust(_BOX_W - 1)}{_L}"


def _safe_float(v) -> float | None:
    """Convert value to float, returning None for NaN/None/non-numeric."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def is_main_process() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


class EnhancedBenchmarkCallback(TrainerCallback):
    """Multi-tier benchmark evaluation callback for training.

    Provides tiered validation with configurable evaluation frequency,
    composite checkpoint selection, and enhanced TensorBoard logging.

    Parameters
    ----------
    tier1_config : str
        Path to Tier 1 benchmark YAML (chronos-lite.yaml, 5 datasets).
    tier2_config : str, optional
        Path to Tier 2 benchmark YAML (chronos-extended.yaml, 15 datasets).
    tier1_eval_steps : int
        Steps between Tier 1 evaluations.
    tier2_eval_steps : int
        Steps between Tier 2 evaluations.
    top_k_checkpoints : int
        Number of best checkpoints to retain.
    checkpoint_metric : str
        Metric for checkpoint selection: "wql", "mase", or "composite".
    composite_weights : dict
        Weights for composite metric: {"wql": 0.7, "mase": 0.3}.
    eval_batch_size : int
        Batch size for evaluation.
    datasets_root : str, optional
        Root directory for local datasets (offline mode).
    eval_timeout : float
        Maximum seconds for a single tier evaluation. 0 = no timeout (default).
    """

    def __init__(
        self,
        tier1_config: str,
        tier2_config: str | None = None,
        tier1_eval_steps: int = 5000,
        tier2_eval_steps: int = 20000,
        top_k_checkpoints: int = 3,
        checkpoint_metric: str = "wql",
        composite_weights: dict | None = None,
        eval_batch_size: int = 32,
        datasets_root: str | None = None,
        eval_timeout: float = 0,
    ):
        self._tier1_config = tier1_config
        self._tier2_config = tier2_config
        self._tier1_eval_steps = tier1_eval_steps
        self._tier2_eval_steps = tier2_eval_steps
        self._top_k = top_k_checkpoints
        self._checkpoint_metric = checkpoint_metric
        self._composite_weights = composite_weights or {"wql": 0.7, "mase": 0.3}
        self._eval_batch_size = eval_batch_size
        self._datasets_root = datasets_root
        self._eval_timeout = eval_timeout

        # State
        self._last_tier1_step: int = 0
        self._last_tier2_step: int = 0
        self._best_checkpoints: list[tuple[float, int, str, dict]] = []
        # (score, step, filepath, results)  — filepath points to best_checkpoints/ .safetensors
        self._pending_eval_results: dict[int, dict] = {}  # step → results
        self.last_benchmark_results: dict | None = None
        self._previous_results: dict | None = None  # for delta tracking
        self._eval_count: int = 0
        self._tb_writer = None

        # Track evaluation engine availability
        self._engine_available: bool | None = None

        # Lazy state restoration flag
        self._state_restored: bool = False

    # ─── Trainer Hooks ───────────────────────────────────────────────────

    def on_step_end(self, args, state, control, **kwargs):
        """Check if evaluation should be triggered."""
        step = state.global_step
        if step <= 0:
            return

        model = kwargs.get("model")
        if model is None:
            return

        # Lazy state restoration on first evaluation step
        if not self._state_restored:
            self._restore_state_from_disk(args)

        # Tier 1: Frequent, lightweight evaluation
        if (
            self._tier1_eval_steps > 0
            and step - self._last_tier1_step >= self._tier1_eval_steps
        ):
            self._last_tier1_step = step
            self._run_evaluation(
                model=model,
                config_path=self._tier1_config,
                tier="tier1",
                step=step,
                args=args,
                state=state,
            )

        # Tier 2: Less frequent, deeper evaluation
        if (
            self._tier2_config
            and self._tier2_eval_steps > 0
            and step - self._last_tier2_step >= self._tier2_eval_steps
        ):
            self._last_tier2_step = step
            self._run_evaluation(
                model=model,
                config_path=self._tier2_config,
                tier="tier2",
                step=step,
                args=args,
                state=state,
            )

    def on_save(self, args, state, control, **kwargs):
        """Process pending eval results after HF Trainer writes checkpoint-{step}/."""
        if not is_main_process():
            return

        step = state.global_step

        # Process current step's pending results (primary case)
        if step in self._pending_eval_results:
            results = self._pending_eval_results.pop(step)
            self._process_metric_checkpoint(step, results, args)

        # Clean up stale pending results from earlier steps
        # (occurs when save_steps > eval_steps — those checkpoints were never saved)
        stale_steps = [s for s in self._pending_eval_results if s < step]
        for stale_step in stale_steps:
            self._pending_eval_results.pop(stale_step)

    # ─── Evaluation Dispatch ─────────────────────────────────────────────

    def _run_evaluation(
        self,
        model,
        config_path: str,
        tier: str,
        step: int,
        args,
        state,
    ):
        """Dispatch benchmark evaluation: distributed (multi-GPU) or single-GPU."""
        if dist.is_initialized() and dist.get_world_size() > 1:
            self._run_evaluation_distributed(model, config_path, tier, step, args, state)
        else:
            self._run_evaluation_single(model, config_path, tier, step, args, state)

    def _run_evaluation_single(
        self,
        model,
        config_path: str,
        tier: str,
        step: int,
        args,
        state,
    ):
        """Original rank-0-only evaluation path for single-GPU training."""
        tier_label = "LITE (Tier 1)" if tier == "tier1" else "EXTENDED (Tier 2)"

        model.eval()
        device = next(model.parameters()).device

        eval_start = time.time()
        try:
            results = self._evaluate_benchmark_single(
                model=model,
                config_path=config_path,
                device=device,
            )

            eval_elapsed = time.time() - eval_start
            results["step"] = step
            results["tier"] = tier
            results["elapsed_seconds"] = round(eval_elapsed, 1)

            self._log_results(results, tier, tier_label, step, eval_elapsed, args)

        except Exception as e:
            eval_elapsed = time.time() - eval_start
            logger.error(f"  Benchmark evaluation failed after {eval_elapsed:.1f}s: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            model.train()

    def _run_evaluation_distributed(
        self,
        model,
        config_path: str,
        tier: str,
        step: int,
        args,
        state,
    ):
        """Distributed evaluation: ALL ranks participate in forward passes.

        Series are sharded across ranks via round-robin. All ranks call
        model.forward() the same number of times (required by FSDP).
        Predictions are gathered to rank-0 for metric computation.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        tier_label = "LITE (Tier 1)" if tier == "tier1" else "EXTENDED (Tier 2)"

        model.eval()
        device = next(model.parameters()).device

        eval_start = time.time()
        try:
            results = self._evaluate_benchmark_distributed(
                model=model,
                config_path=config_path,
                device=device,
                rank=rank,
                world_size=world_size,
            )

            eval_elapsed = time.time() - eval_start

            # Only rank-0 handles logging, checkpointing, and result tracking
            if rank == 0 and results is not None:
                results["step"] = step
                results["tier"] = tier
                results["elapsed_seconds"] = round(eval_elapsed, 1)

                self._log_results(results, tier, tier_label, step, eval_elapsed, args)

        except Exception as e:
            eval_elapsed = time.time() - eval_start
            if rank == 0:
                logger.error(f"  Benchmark evaluation failed after {eval_elapsed:.1f}s: {e}")
                import traceback
                logger.error(traceback.format_exc())
        finally:
            model.train()

        # Brief final sync
        dist.barrier()

    # ─── Result Logging (Box-Formatted Console) ──────────────────────────

    def _log_results(
        self,
        results: dict,
        tier: str,
        tier_label: str,
        step: int,
        eval_elapsed: float,
        args,
    ):
        """Log evaluation results with rich box-formatted console output.

        Includes: summary metrics, improvement/regression tracking,
        per-dataset breakdown table, best-ever indicators, and top-K status.
        """
        self._eval_count += 1

        # Check timeout
        if self._eval_timeout > 0 and eval_elapsed > self._eval_timeout:
            logger.warning(
                f"  Evaluation took {eval_elapsed:.0f}s "
                f"(timeout={self._eval_timeout:.0f}s)"
            )

        avg_wql = results.get("avg_wql")
        avg_mase = results.get("avg_mase")
        composite = self._compute_checkpoint_score(results)
        per_dataset = results.get("per_dataset", {})

        # Compute dataset success/failure counts
        n_total = len(per_dataset)
        n_ok = sum(
            1 for ds_m in per_dataset.values()
            if _safe_float(ds_m.get("WQL")) is not None
        )
        n_fail = n_total - n_ok

        # Compute deltas from previous evaluation
        prev = self._previous_results
        wql_delta = None
        mase_delta = None
        if prev is not None and avg_wql is not None and prev.get("avg_wql") is not None:
            wql_delta = avg_wql - prev["avg_wql"]
        if prev is not None and avg_mase is not None and prev.get("avg_mase") is not None:
            mase_delta = avg_mase - prev["avg_mase"]

        # Check if this is a new best (before updating _best_checkpoints)
        is_new_best = False
        if composite is not None:
            if not self._best_checkpoints or composite < self._best_checkpoints[0][0]:
                is_new_best = True

        # Distributed info
        gpu_info = ""
        if dist.is_initialized() and dist.get_world_size() > 1:
            gpu_info = f" ({dist.get_world_size()} GPUs)"

        # ── Build Box Output ──
        lines = []
        lines.append("")
        lines.append(_TOP)

        # Header
        header = f"BENCHMARK EVALUATION \u2014 {tier_label} \u2014 Step {step:,}{gpu_info}"
        lines.append(_boxline(header))
        lines.append(_SEP)

        # Summary Metrics
        lines.append(_boxline(""))
        lines.append(_boxline("[Summary]"))

        if avg_wql is not None:
            delta_str = ""
            if wql_delta is not None:
                arrow = "\u2193" if wql_delta < 0 else ("\u2191" if wql_delta > 0 else "\u2192")
                delta_str = f"  ({arrow} {abs(wql_delta):.4f})"
            lines.append(_boxline(f"  Avg WQL (val_loss):  {avg_wql:.4f}{delta_str}"))

        if avg_mase is not None:
            delta_str = ""
            if mase_delta is not None:
                arrow = "\u2193" if mase_delta < 0 else ("\u2191" if mase_delta > 0 else "\u2192")
                delta_str = f"  ({arrow} {abs(mase_delta):.4f})"
            lines.append(_boxline(f"  Avg MASE:            {avg_mase:.4f}{delta_str}"))

        if composite is not None:
            w = self._composite_weights
            best_marker = "  << NEW BEST" if is_new_best else ""
            lines.append(_boxline(
                f"  Composite:           {composite:.4f}  "
                f"(wql*{w.get('wql', 0.7):.1f} + mase*{w.get('mase', 0.3):.1f})"
                f"{best_marker}"
            ))

        lines.append(_boxline(f"  Elapsed:             {eval_elapsed:.1f}s"))
        lines.append(_boxline(f"  Datasets:            {n_ok}/{n_total} OK" + (
            f", {n_fail} failed" if n_fail > 0 else ""
        )))

        # Best-ever tracking
        if self._best_checkpoints:
            best_score, best_step, _, best_res = self._best_checkpoints[0]
            lines.append(_boxline(""))
            lines.append(_boxline("[Best So Far]"))
            best_wql = best_res.get("avg_wql")
            best_mase = best_res.get("avg_mase")
            if best_wql is not None:
                lines.append(_boxline(f"  Best WQL:            {best_wql:.4f}  (step {best_step:,})"))
            if best_mase is not None:
                lines.append(_boxline(f"  Best MASE:           {best_mase:.4f}  (step {best_step:,})"))
            lines.append(_boxline(f"  Best Composite:      {best_score:.4f}  (step {best_step:,})"))

        # Per-dataset breakdown table
        if per_dataset:
            lines.append(_boxline(""))
            lines.append(_boxline("[Per-Dataset Results]"))
            # Table header
            lines.append(_boxline(f"  {'Dataset':<24s} {'WQL':>10s} {'MASE':>10s}"))
            lines.append(_boxline(f"  {'─' * 24} {'─' * 10} {'─' * 10}"))
            for ds_name, ds_m in per_dataset.items():
                wql_v = _safe_float(ds_m.get("WQL"))
                mase_v = _safe_float(ds_m.get("MASE"))
                wql_str = f"{wql_v:.4f}" if wql_v is not None else "  FAIL"
                mase_str = f"{mase_v:.4f}" if mase_v is not None else "  FAIL"
                lines.append(_boxline(f"  {ds_name:<24s} {wql_str:>10s} {mase_str:>10s}"))

        # Top-K checkpoint status
        if self._best_checkpoints:
            lines.append(_boxline(""))
            lines.append(_boxline(f"[Top-{self._top_k} Checkpoints by {self._checkpoint_metric}]"))
            for rank_i, (s, st, p, _) in enumerate(self._best_checkpoints, 1):
                fname = Path(p).name if p else f"step-{st}"
                lines.append(_boxline(f"  #{rank_i}: {s:.4f}  (step {st:,})  {fname}"))

        lines.append(_boxline(""))
        lines.append(_BOT)
        lines.append("")

        logger.info("\n".join(lines))

        # ── TensorBoard logging ──
        self._log_to_tensorboard(results, tier, step, args)

        # ── Update state ──
        self._previous_results = results
        self.last_benchmark_results = results

        # ── File-based reporting (JSON + CSV) ──
        self._save_eval_report(results, step, args, n_ok, n_fail)

        # ── Checkpoint management (only on Tier 1 for fast feedback) ──
        if tier == "tier1" and composite is not None:
            self._pending_eval_results[step] = results

    # ─── Evaluation Engine ───────────────────────────────────────────────

    def _evaluate_benchmark_single(
        self,
        model,
        config_path: str,
        device: torch.device,
    ) -> dict:
        """Run benchmark evaluation on a single GPU using the shared evaluation engine."""
        if self._engine_available is False:
            raise ImportError("Evaluation engine not available (checked previously)")

        try:
            from engine.forecaster import TrainingModelForecaster
            from engine.evaluator import Evaluator

            self._engine_available = True

            forecaster = TrainingModelForecaster(model, device)
            evaluator = Evaluator(
                forecaster=forecaster,
                batch_size=self._eval_batch_size,
                datasets_root=self._datasets_root,
            )

            return evaluator.evaluate_quick(config_path)

        except ImportError as e:
            self._engine_available = False
            raise ImportError(
                f"Evaluation engine not importable: {e}\n"
                f"Ensure scripts/forecasting/evaluation/engine/ is accessible.\n"
                f"Current sys.path includes: {_EVAL_DIR}"
            ) from e

    def _evaluate_benchmark_distributed(
        self,
        model,
        config_path: str,
        device: torch.device,
        rank: int,
        world_size: int,
    ) -> dict | None:
        """Run distributed benchmark evaluation across all ranks.

        All ranks load datasets and participate in forward passes.
        Only rank-0 computes metrics and returns results.

        Returns
        -------
        dict or None
            Evaluation results on rank-0, None on other ranks.
        """
        if self._engine_available is False:
            raise ImportError("Evaluation engine not available (checked previously)")

        try:
            import yaml as yaml_loader
            from engine.forecaster import TrainingModelForecaster
            from engine.evaluator import (
                EVAL_QUANTILES,
                load_dataset_from_config,
                validate_config,
            )
            from engine.distributed import (
                broadcast_error,
                generate_forecasts_distributed,
                gather_forecasts_to_rank0,
            )

            self._engine_available = True
        except ImportError as e:
            self._engine_available = False
            raise ImportError(
                f"Evaluation engine not importable: {e}\n"
                f"Ensure scripts/forecasting/evaluation/engine/ is accessible.\n"
                f"Current sys.path includes: {_EVAL_DIR}"
            ) from e

        # Validate config (all ranks — identical local files)
        config_errors = validate_config(config_path)
        if config_errors:
            error_msg = "\n".join(f"  - {e}" for e in config_errors)
            raise ValueError(
                f"Benchmark config validation failed:\n{error_msg}"
            )

        with open(config_path) as f:
            configs = yaml_loader.safe_load(f)

        forecaster = TrainingModelForecaster(model, device)
        quantile_levels = EVAL_QUANTILES
        n_quantiles = len(quantile_levels)
        per_dataset_results = {}

        for ds_config in configs:
            ds_name = ds_config["name"]
            prediction_length = ds_config["prediction_length"]

            # Phase 1: Load dataset on ALL ranks (local disk, fast).
            # Use error synchronization to prevent NCCL deadlock if any rank fails.
            load_failed = False
            test_data = None
            all_inputs = []
            n_series = 0
            pred_len = prediction_length

            try:
                _, test_data, pred_len, _ = load_dataset_from_config(
                    ds_config, datasets_root=self._datasets_root
                )
                all_inputs = list(test_data.input)
                n_series = len(all_inputs)
            except Exception as e:
                load_failed = True
                if rank == 0:
                    logger.warning(f"  Dataset {ds_name} load failed: {e}")

            # Synchronize: if ANY rank failed, ALL ranks skip this dataset.
            # Without this, succeeding ranks would enter all_reduce while
            # failing ranks skip it → NCCL deadlock.
            if broadcast_error(load_failed, device):
                if rank == 0:
                    logger.warning(f"  Skipping {ds_name}: load failed on one or more ranks")
                    per_dataset_results[ds_name] = {
                        "WQL": float("nan"),
                        "MASE": float("nan"),
                    }
                continue

            # Phase 2: Distributed forecasting (all ranks participate in forward passes)
            try:
                local_preds, local_indices = generate_forecasts_distributed(
                    all_inputs,
                    forecaster,
                    prediction_length=pred_len,
                    quantile_levels=quantile_levels,
                    batch_size=self._eval_batch_size,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                )

                # Phase 3: Gather predictions to rank-0
                all_preds = gather_forecasts_to_rank0(
                    local_preds, local_indices, n_series, n_quantiles, pred_len,
                    rank, world_size, device,
                )

                # Phase 4: Rank-0 computes metrics
                if rank == 0 and all_preds is not None:
                    from gluonts.model.forecast import QuantileForecast
                    from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
                    from gluonts.model.evaluation import evaluate_forecasts

                    # Build QuantileForecast objects (use all_inputs, not
                    # test_data.input, to avoid re-iterating a potential generator)
                    forecasts = []
                    for item, ts in zip(all_preds, all_inputs):
                        forecast_start = ts["start"] + len(ts["target"])
                        forecasts.append(
                            QuantileForecast(
                                forecast_arrays=item,
                                forecast_keys=list(map(str, quantile_levels)),
                                start_date=forecast_start,
                            )
                        )

                    metrics = (
                        evaluate_forecasts(
                            forecasts,
                            test_data=test_data,
                            metrics=[
                                MASE(),
                                MeanWeightedSumQuantileLoss(quantile_levels),
                            ],
                            batch_size=5000,
                        )
                        .reset_index(drop=True)
                        .to_dict(orient="records")
                    )

                    result = metrics[0]
                    if "MASE[0.5]" in result:
                        result["MASE"] = result.pop("MASE[0.5]")
                    if "mean_weighted_sum_quantile_loss" in result:
                        result["WQL"] = result.pop("mean_weighted_sum_quantile_loss")

                    per_dataset_results[ds_name] = {
                        "WQL": result.get("WQL", float("nan")),
                        "MASE": result.get("MASE", float("nan")),
                    }

            except Exception as e:
                if rank == 0:
                    logger.warning(f"  Dataset {ds_name} evaluation failed: {e}")
                    per_dataset_results[ds_name] = {
                        "WQL": float("nan"),
                        "MASE": float("nan"),
                    }

        # Rank-0: aggregate results
        if rank == 0:
            import numpy as np

            wql_values = [
                v["WQL"] for v in per_dataset_results.values()
                if isinstance(v.get("WQL"), float) and not np.isnan(v["WQL"])
            ]
            mase_values = [
                v["MASE"] for v in per_dataset_results.values()
                if isinstance(v.get("MASE"), float) and not np.isnan(v["MASE"])
            ]

            return {
                "avg_wql": float(np.mean(wql_values)) if wql_values else None,
                "avg_mase": float(np.mean(mase_values)) if mase_values else None,
                "per_dataset": per_dataset_results,
            }

        return None

    # ─── Scoring ─────────────────────────────────────────────────────────

    def _compute_checkpoint_score(self, results: dict) -> float | None:
        """Compute the score used for checkpoint ranking.

        Returns None if score cannot be computed.
        Lower score is better.
        """
        avg_wql = results.get("avg_wql")
        avg_mase = results.get("avg_mase")

        if self._checkpoint_metric == "wql":
            return avg_wql
        elif self._checkpoint_metric == "mase":
            return avg_mase
        elif self._checkpoint_metric == "composite":
            if avg_wql is None or avg_mase is None:
                return avg_wql  # fallback to WQL
            w = self._composite_weights
            return w.get("wql", 0.7) * avg_wql + w.get("mase", 0.3) * avg_mase
        else:
            return avg_wql  # default

    # ─── TensorBoard ─────────────────────────────────────────────────────

    def _log_to_tensorboard(
        self,
        results: dict,
        tier: str,
        step: int,
        args,
    ):
        """Log benchmark results to TensorBoard with hierarchical tags.

        Tag structure:
            benchmark/{tier}/avg_wql
            benchmark/{tier}/avg_mase
            benchmark/{tier}/validation_loss
            benchmark/{tier}/{dataset}/wql
            benchmark/{tier}/{dataset}/mase
            benchmark/{tier}/composite_score
            benchmark/{tier}/elapsed_seconds
            benchmark/checkpoint/best_composite
            benchmark/checkpoint/best_wql
            benchmark/checkpoint/best_mase
            benchmark/checkpoint/n_top_k
        """
        try:
            from torch.utils.tensorboard import SummaryWriter

            if self._tb_writer is None:
                tb_dir = Path(args.output_dir) / "runs"
                if tb_dir.exists():
                    tb_dirs = sorted(tb_dir.iterdir(), key=lambda p: p.stat().st_mtime)
                    if tb_dirs:
                        self._tb_writer = SummaryWriter(log_dir=str(tb_dirs[-1]))

            if self._tb_writer is None:
                return

            avg_wql = results.get("avg_wql")
            avg_mase = results.get("avg_mase")

            # Tier-level averages
            if avg_wql is not None:
                self._tb_writer.add_scalar(f"benchmark/{tier}/avg_wql", avg_wql, step)
                # Validation loss proxy (WQL for quantile regression)
                self._tb_writer.add_scalar(f"benchmark/{tier}/validation_loss", avg_wql, step)
            if avg_mase is not None:
                self._tb_writer.add_scalar(f"benchmark/{tier}/avg_mase", avg_mase, step)

            # Per-dataset metrics
            for ds_name, ds_m in results.get("per_dataset", {}).items():
                wql_v = _safe_float(ds_m.get("WQL"))
                mase_v = _safe_float(ds_m.get("MASE"))
                if wql_v is not None:
                    self._tb_writer.add_scalar(f"benchmark/{tier}/{ds_name}/wql", wql_v, step)
                if mase_v is not None:
                    self._tb_writer.add_scalar(f"benchmark/{tier}/{ds_name}/mase", mase_v, step)

            # Composite score
            composite = self._compute_checkpoint_score(results)
            if composite is not None:
                self._tb_writer.add_scalar(f"benchmark/{tier}/composite_score", composite, step)

            # Evaluation timing
            elapsed = results.get("elapsed_seconds")
            if elapsed is not None:
                self._tb_writer.add_scalar(f"benchmark/{tier}/elapsed_seconds", elapsed, step)

            # Checkpoint tracking — best metrics across all evaluations
            if self._best_checkpoints:
                best_score, _, _, best_res = self._best_checkpoints[0]
                self._tb_writer.add_scalar("benchmark/checkpoint/best_composite", best_score, step)
                best_wql = best_res.get("avg_wql")
                best_mase = best_res.get("avg_mase")
                if best_wql is not None:
                    self._tb_writer.add_scalar("benchmark/checkpoint/best_wql", best_wql, step)
                if best_mase is not None:
                    self._tb_writer.add_scalar("benchmark/checkpoint/best_mase", best_mase, step)
                self._tb_writer.add_scalar(
                    "benchmark/checkpoint/n_top_k", len(self._best_checkpoints), step
                )

            self._tb_writer.flush()

        except Exception:
            pass  # TensorBoard logging is best-effort

    # ─── File Reporting (JSON + CSV) ─────────────────────────────────────

    def _save_eval_report(
        self, results: dict, step: int, args, n_ok: int = 0, n_fail: int = 0
    ):
        """Save per-evaluation JSON report and append to cumulative CSV.

        Uses _safe_float() to sanitize NaN values into null for valid JSON.
        """
        if not is_main_process():
            return

        eval_dir = Path(args.output_dir) / "eval_results"
        try:
            eval_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"  Could not create eval_results directory: {e}")
            return

        avg_wql = _safe_float(results.get("avg_wql"))
        avg_mase = _safe_float(results.get("avg_mase"))
        composite = _safe_float(self._compute_checkpoint_score(results))
        tier = results.get("tier", "tier1")
        elapsed = results.get("elapsed_seconds")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        # Build top-K checkpoint list for JSON
        top_k_list = []
        for s, st, fp, _ in self._best_checkpoints:
            top_k_list.append({
                "step": st,
                "composite": round(s, 4),
                "filename": Path(fp).name,
            })

        # ── JSON Report ──
        report = {
            "step": step,
            "timestamp": timestamp,
            "tier": tier,
            "eval_number": self._eval_count,
            "elapsed_seconds": elapsed,
            "n_datasets": n_ok + n_fail,
            "n_datasets_ok": n_ok,
            "n_datasets_failed": n_fail,
            "avg_wql": round(avg_wql, 4) if avg_wql is not None else None,
            "avg_mase": round(avg_mase, 4) if avg_mase is not None else None,
            "composite_score": round(composite, 4) if composite is not None else None,
            "composite_weights": dict(self._composite_weights),
            "checkpoint_metric": self._checkpoint_metric,
            "validation_loss": round(avg_wql, 4) if avg_wql is not None else None,
            "per_dataset": {},
            "top_k_checkpoints": top_k_list,
        }
        for ds_name, ds_m in results.get("per_dataset", {}).items():
            report["per_dataset"][ds_name] = {
                "WQL": _safe_float(ds_m.get("WQL")),
                "MASE": _safe_float(ds_m.get("MASE")),
            }

        json_path = eval_dir / f"eval_step_{step:06d}.json"
        try:
            with open(json_path, "w") as f:
                json.dump(report, f, indent=2)
        except IOError as e:
            logger.warning(f"  Could not write JSON report: {e}")

        # ── CSV Append ──
        csv_path = eval_dir / "validation_history.csv"
        try:
            row: dict = {
                "step": step,
                "timestamp": timestamp,
                "tier": tier,
                "avg_wql": avg_wql,
                "avg_mase": avg_mase,
                "composite": composite,
                "validation_loss": avg_wql,
                "elapsed_seconds": elapsed,
            }
            # Add per-dataset columns
            for ds_name, ds_m in results.get("per_dataset", {}).items():
                row[f"{ds_name}_wql"] = _safe_float(ds_m.get("WQL"))
                row[f"{ds_name}_mase"] = _safe_float(ds_m.get("MASE"))

            write_header = not csv_path.exists()
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except IOError as e:
            logger.warning(f"  Could not append to validation_history.csv: {e}")

    # ─── Metric-Encoded Checkpoints ──────────────────────────────────────

    def _process_metric_checkpoint(self, step: int, results: dict, args):
        """Create metric-encoded checkpoint and manage top-K."""
        composite = self._compute_checkpoint_score(results)
        if composite is None:
            return

        filepath = self._create_metric_checkpoint(composite, step, results, args)
        if filepath is None:
            return

        # Track and manage top-K
        self._best_checkpoints.append((composite, step, filepath, results))
        self._best_checkpoints.sort(key=lambda x: x[0])  # ascending (lower=better)

        if len(self._best_checkpoints) > self._top_k:
            worst = self._best_checkpoints.pop()
            self._remove_metric_checkpoint(worst[2])

        logger.info(f"  Saved metric checkpoint: {Path(filepath).name}")

    def _create_metric_checkpoint(
        self,
        score: float,
        step: int,
        results: dict,
        args,
    ) -> str | None:
        """Create a metric-encoded checkpoint file in best_checkpoints/.

        Creates a hardlink (or copy) of model.safetensors from checkpoint-{step}/
        with a descriptive filename encoding the evaluation metrics.

        Returns the filepath of the metric-encoded checkpoint, or None on failure.
        """
        best_dir = Path(args.output_dir) / "best_checkpoints"
        try:
            best_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"  Could not create best_checkpoints directory: {e}")
            return None

        # Source: HF Trainer's checkpoint
        src_model = Path(args.output_dir) / f"checkpoint-{step}" / "model.safetensors"
        if not src_model.exists():
            logger.warning(f"  Source checkpoint not found: {src_model}")
            return None

        # Copy config.json once (shared across all best checkpoints)
        config_dst = best_dir / "config.json"
        if not config_dst.exists():
            config_src = Path(args.output_dir) / f"checkpoint-{step}" / "config.json"
            if config_src.exists():
                try:
                    shutil.copy2(str(config_src), str(config_dst))
                except OSError:
                    pass

        # Build metric-encoded filename
        avg_wql = results.get("avg_wql", 0.0) or 0.0
        avg_mase = results.get("avg_mase", 0.0) or 0.0
        filename = (
            f"model-step={step:06d}"
            f"-wql={avg_wql:.4f}"
            f"-mase={avg_mase:.4f}"
            f"-composite={score:.4f}"
            f".safetensors"
        )
        dst_path = best_dir / filename

        # Skip if already exists (e.g., from state recovery)
        if dst_path.exists():
            return str(dst_path)

        # Try hardlink first (same filesystem, no extra disk space)
        try:
            os.link(str(src_model), str(dst_path))
            return str(dst_path)
        except OSError:
            pass

        # Fallback to copy
        try:
            shutil.copy2(str(src_model), str(dst_path))
            return str(dst_path)
        except OSError as e:
            logger.warning(f"  Could not create metric checkpoint: {e}")
            return None

    def _remove_metric_checkpoint(self, filepath: str):
        """Remove a metric-encoded checkpoint file."""
        p = Path(filepath)
        if p.exists():
            try:
                p.unlink()
                logger.info(f"  Removed worst checkpoint: {p.name}")
            except OSError:
                pass

    # ─── State Recovery ──────────────────────────────────────────────────

    def _restore_state_from_disk(self, args):
        """Restore top-K state and last eval step from disk for training resume.

        Scans best_checkpoints/ for metric-encoded filenames and rebuilds
        _best_checkpoints list. Scans eval_results/ for JSON reports to
        restore _last_tier1_step and last_benchmark_results.
        """
        self._state_restored = True

        if not is_main_process():
            return

        best_dir = Path(args.output_dir) / "best_checkpoints"
        if best_dir.exists():
            for f in best_dir.iterdir():
                if not f.is_file():
                    continue
                m = _METRIC_CKPT_RE.match(f.name)
                if m:
                    ckpt_step = int(m.group(1))
                    wql = float(m.group(2))
                    mase = float(m.group(3))
                    composite = float(m.group(4))
                    restored_results = {
                        "avg_wql": wql,
                        "avg_mase": mase,
                    }
                    self._best_checkpoints.append(
                        (composite, ckpt_step, str(f), restored_results)
                    )

            if self._best_checkpoints:
                self._best_checkpoints.sort(key=lambda x: x[0])
                logger.info(
                    f"  Restored {len(self._best_checkpoints)} metric checkpoint(s) "
                    f"from {best_dir}"
                )

        # Restore state from eval JSON reports
        eval_dir = Path(args.output_dir) / "eval_results"
        if eval_dir.exists():
            max_tier1_step = 0
            max_tier2_step = 0
            latest_report = None
            latest_step = 0

            for f in sorted(eval_dir.glob("eval_step_*.json")):
                try:
                    with open(f) as fp:
                        report = json.load(fp)
                    report_step = report.get("step", 0)
                    report_tier = report.get("tier", "tier1")
                    if report_tier == "tier1" and report_step > max_tier1_step:
                        max_tier1_step = report_step
                    if report_tier == "tier2" and report_step > max_tier2_step:
                        max_tier2_step = report_step
                    if report_step > latest_step:
                        latest_step = report_step
                        latest_report = report
                    self._eval_count += 1
                except (json.JSONDecodeError, IOError):
                    continue

            if max_tier1_step > 0:
                self._last_tier1_step = max_tier1_step
                logger.info(f"  Restored last tier1 eval step: {max_tier1_step}")
            if max_tier2_step > 0:
                self._last_tier2_step = max_tier2_step
                logger.info(f"  Restored last tier2 eval step: {max_tier2_step}")

            # Restore last_benchmark_results for health report
            if latest_report is not None:
                self.last_benchmark_results = {
                    "avg_wql": latest_report.get("avg_wql"),
                    "avg_mase": latest_report.get("avg_mase"),
                    "per_dataset": latest_report.get("per_dataset", {}),
                }
                self._previous_results = self.last_benchmark_results
                logger.info(f"  Restored last eval results from step {latest_step}")

    # ─── Public API ──────────────────────────────────────────────────────

    def get_best_checkpoint(self) -> tuple[str, float, int] | None:
        """Get the best checkpoint path, score, and step.

        Returns
        -------
        tuple[str, float, int] or None
            (checkpoint_path, best_score, best_step)
        """
        if self._best_checkpoints:
            score, step, path, _ = self._best_checkpoints[0]
            return path, score, step
        return None

    def __del__(self):
        """Close TensorBoard writer on cleanup."""
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass
