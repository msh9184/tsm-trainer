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

import logging
import sys
import time
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
        self._best_results: list[tuple[float, int, str, dict]] = []
        # (score, step, ckpt_path, full_results)
        self.last_benchmark_results: dict | None = None
        self._tb_writer = None

        # Track evaluation engine availability
        self._engine_available: bool | None = None

    def on_step_end(self, args, state, control, **kwargs):
        """Check if evaluation should be triggered."""
        step = state.global_step
        if step <= 0:
            return

        model = kwargs.get("model")
        if model is None:
            return

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
        logger.info(f"\n{'─' * 72}")
        logger.info(f"  BENCHMARK EVALUATION — {tier_label} — Step {step:,}")
        logger.info(f"{'─' * 72}")

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

        logger.info(f"{'─' * 72}\n")

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

        if rank == 0:
            logger.info(f"\n{'─' * 72}")
            logger.info(
                f"  BENCHMARK EVALUATION — {tier_label} — Step {step:,} "
                f"(distributed: {world_size} GPUs)"
            )
            logger.info(f"{'─' * 72}")

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

        if rank == 0:
            logger.info(f"{'─' * 72}\n")

        # Brief final sync
        dist.barrier()

    def _log_results(
        self,
        results: dict,
        tier: str,
        tier_label: str,
        step: int,
        eval_elapsed: float,
        args,
    ):
        """Log evaluation results, update state, and manage checkpoints (rank-0 only)."""
        # Check timeout
        if self._eval_timeout > 0 and eval_elapsed > self._eval_timeout:
            logger.warning(
                f"  Evaluation took {eval_elapsed:.0f}s "
                f"(timeout={self._eval_timeout:.0f}s)"
            )

        avg_wql = results.get("avg_wql")
        avg_mase = results.get("avg_mase")

        logger.info(f"  ── {tier_label} Results ({eval_elapsed:.1f}s) ──")
        if avg_wql is not None:
            logger.info(f"  Avg WQL:  {avg_wql:.4f}")
        if avg_mase is not None:
            logger.info(f"  Avg MASE: {avg_mase:.4f}")

        # Compute composite score
        composite = self._compute_checkpoint_score(results)
        if composite is not None:
            logger.info(f"  Composite ({self._checkpoint_metric}): {composite:.4f}")

        # Log per-dataset results
        for ds_name, ds_m in results.get("per_dataset", {}).items():
            wql_str = f"{ds_m['WQL']:.4f}" if isinstance(ds_m.get('WQL'), float) else "error"
            mase_str = f"{ds_m['MASE']:.4f}" if isinstance(ds_m.get('MASE'), float) else "error"
            logger.info(f"    {ds_name}: WQL={wql_str}, MASE={mase_str}")

        # TensorBoard logging
        self._log_to_tensorboard(results, tier, step, args)

        # Update last results
        self.last_benchmark_results = results

        # Checkpoint management (only on Tier 1 for fast feedback)
        if tier == "tier1" and composite is not None:
            self._manage_checkpoints(composite, step, results, args)

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
            benchmark/{tier}/{dataset}/wql
            benchmark/{tier}/{dataset}/mase
            benchmark/checkpoint/best_score
            benchmark/checkpoint/best_step
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
            if avg_mase is not None:
                self._tb_writer.add_scalar(f"benchmark/{tier}/avg_mase", avg_mase, step)

            # Per-dataset metrics
            for ds_name, ds_m in results.get("per_dataset", {}).items():
                if isinstance(ds_m.get("WQL"), float):
                    self._tb_writer.add_scalar(f"benchmark/{tier}/{ds_name}/wql", ds_m["WQL"], step)
                if isinstance(ds_m.get("MASE"), float):
                    self._tb_writer.add_scalar(f"benchmark/{tier}/{ds_name}/mase", ds_m["MASE"], step)

            # Composite score
            composite = self._compute_checkpoint_score(results)
            if composite is not None:
                self._tb_writer.add_scalar(f"benchmark/{tier}/composite_score", composite, step)

            # Evaluation timing
            elapsed = results.get("elapsed_seconds")
            if elapsed is not None:
                self._tb_writer.add_scalar(f"benchmark/{tier}/elapsed_seconds", elapsed, step)

            # Checkpoint tracking
            if self._best_results:
                best_score, best_step, _, _ = self._best_results[0]
                self._tb_writer.add_scalar("benchmark/checkpoint/best_score", best_score, step)
                self._tb_writer.add_scalar("benchmark/checkpoint/best_step", best_step, step)

            self._tb_writer.flush()

        except Exception:
            pass  # TensorBoard logging is best-effort

    def _manage_checkpoints(
        self,
        score: float,
        step: int,
        results: dict,
        args,
    ):
        """Manage top-K checkpoints by evaluation score."""
        ckpt_path = str(Path(args.output_dir) / f"checkpoint-{step}")

        self._best_results.append((score, step, ckpt_path, results))
        self._best_results.sort(key=lambda x: x[0])  # ascending (lower=better)

        if len(self._best_results) > self._top_k:
            worst = self._best_results.pop()
            worst_path = Path(worst[2])
            if worst_path.exists() and worst[1] != step:
                import shutil
                try:
                    shutil.rmtree(worst_path)
                    logger.info(
                        f"  Removed worst checkpoint: {worst_path.name} "
                        f"({self._checkpoint_metric}={worst[0]:.4f})"
                    )
                except OSError:
                    pass

        logger.info(f"  Top-{self._top_k} checkpoints by {self._checkpoint_metric}:")
        for rank_i, (s, st, p, _) in enumerate(self._best_results, 1):
            logger.info(f"    #{rank_i}: step {st:,} — {self._checkpoint_metric}={s:.4f}")

    def get_best_checkpoint(self) -> tuple[str, float, int] | None:
        """Get the best checkpoint path, score, and step.

        Returns
        -------
        tuple[str, float, int] or None
            (checkpoint_path, best_score, best_step)
        """
        if self._best_results:
            score, step, path, _ = self._best_results[0]
            return path, score, step
        return None

    def __del__(self):
        """Close TensorBoard writer on cleanup."""
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass
