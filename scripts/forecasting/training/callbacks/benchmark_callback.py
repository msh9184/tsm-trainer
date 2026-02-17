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
- Distributed training support (rank-0 only evaluation)
- Evaluation timeout to prevent training hangs
- Uses shared evaluation engine (no code duplication)

Usage in training config YAML:
    benchmark_config: configs/chronos-lite.yaml
    benchmark_eval_steps: 200
    benchmark_top_k_checkpoints: 3
    benchmark_batch_size: 32
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
        """Run benchmark evaluation for a given tier."""
        if is_main_process():
            tier_label = "LITE (Tier 1)" if tier == "tier1" else "EXTENDED (Tier 2)"
            logger.info(f"\n{'─' * 72}")
            logger.info(f"  BENCHMARK EVALUATION — {tier_label} — Step {step:,}")
            logger.info(f"{'─' * 72}")

            model.eval()
            device = next(model.parameters()).device

            eval_start = time.time()
            try:
                results = self._evaluate_benchmark(
                    model=model,
                    config_path=config_path,
                    device=device,
                )

                eval_elapsed = time.time() - eval_start

                # Check timeout
                if self._eval_timeout > 0 and eval_elapsed > self._eval_timeout:
                    logger.warning(
                        f"  Evaluation took {eval_elapsed:.0f}s "
                        f"(timeout={self._eval_timeout:.0f}s)"
                    )

                results["step"] = step
                results["tier"] = tier
                results["elapsed_seconds"] = round(eval_elapsed, 1)

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

                # TensorBoard logging (hierarchical)
                self._log_to_tensorboard(results, tier, step, args)

                # Update last results (always use latest regardless of tier)
                self.last_benchmark_results = results

                # Checkpoint management (only on Tier 1 for fast feedback)
                if tier == "tier1" and composite is not None:
                    self._manage_checkpoints(composite, step, results, args)

            except Exception as e:
                eval_elapsed = time.time() - eval_start
                logger.error(f"  Benchmark evaluation failed after {eval_elapsed:.1f}s: {e}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                model.train()

            logger.info(f"{'─' * 72}\n")

        # Barrier for distributed training
        if dist.is_initialized():
            dist.barrier()

    def _evaluate_benchmark(
        self,
        model,
        config_path: str,
        device: torch.device,
    ) -> dict:
        """Run benchmark evaluation using the shared evaluation engine.

        The evaluation engine (scripts/forecasting/evaluation/engine/) provides
        all data loading, forecasting, and metric computation. This callback
        wraps the training model in a TrainingModelForecaster and delegates
        to the engine's Evaluator.
        """
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
