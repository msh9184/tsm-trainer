"""Enhanced Benchmark Callback for training-time validation.

Multi-tier benchmark evaluation with composite checkpoint selection
and hierarchical TensorBoard logging. Replaces LiteBenchmarkCallback
with a more capable and extensible architecture.

Features:
- Tier 1 (Lite, 5 datasets, ~3 min): Every N steps for rapid quality signal
- Tier 2 (Extended, 15 datasets, ~15 min): Every M steps for deeper validation
- Composite checkpoint metric: Weighted WQL + MASE
- Per-dataset TensorBoard logging under benchmark/tier{N}/
- Top-K checkpoint management by configurable metric
- Distributed training support (rank-0 only evaluation)

Usage in training config YAML:
    benchmark_config: configs/lite-benchmark.yaml
    benchmark_eval_steps: 5000
    benchmark_tier2_config: configs/extended-benchmark.yaml
    benchmark_tier2_eval_steps: 20000
    benchmark_top_k_checkpoints: 3
    benchmark_checkpoint_metric: composite  # "wql" | "mase" | "composite"
    benchmark_composite_weights:
      wql: 0.7
      mase: 0.3
"""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


def is_main_process() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


class EnhancedBenchmarkCallback:
    """Multi-tier benchmark evaluation callback for training.

    Provides tiered validation with configurable evaluation frequency,
    composite checkpoint selection, and enhanced TensorBoard logging.

    Parameters
    ----------
    tier1_config : str
        Path to Tier 1 benchmark YAML (lite-benchmark.yaml, 5 datasets).
    tier2_config : str, optional
        Path to Tier 2 benchmark YAML (extended-benchmark.yaml, 15 datasets).
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

        # State
        self._last_tier1_step: int = 0
        self._last_tier2_step: int = 0
        self._best_results: list[tuple[float, int, str, dict]] = []
        # (score, step, ckpt_path, full_results)
        self.last_benchmark_results: dict | None = None
        self._tb_writer = None

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

            try:
                results = self._evaluate_benchmark(
                    model=model,
                    config_path=config_path,
                    device=device,
                )
                results["step"] = step
                results["tier"] = tier

                avg_wql = results.get("avg_wql")
                avg_mase = results.get("avg_mase")

                logger.info(f"  ── {tier_label} Results ──")
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
                logger.error(f"  Benchmark evaluation failed: {e}")
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
        """Run benchmark evaluation using the engine module.

        Falls back to the legacy inline evaluation if the engine module
        is not available.
        """
        try:
            # Use the new evaluation engine
            sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))
            from engine.forecaster import TrainingModelForecaster
            from engine.evaluator import Evaluator

            forecaster = TrainingModelForecaster(model, device)
            evaluator = Evaluator(
                forecaster=forecaster,
                batch_size=self._eval_batch_size,
                datasets_root=self._datasets_root,
            )

            return evaluator.evaluate_quick(config_path)

        except ImportError:
            logger.warning("Engine module not available, falling back to legacy evaluation")
            return self._evaluate_benchmark_legacy(model, config_path, device)

    def _evaluate_benchmark_legacy(
        self,
        model,
        config_path: str,
        device: torch.device,
    ) -> dict:
        """Legacy evaluation (same as original _run_lite_benchmark).

        Kept for backward compatibility when engine module is not in path.
        Supports both local dataset loading and HuggingFace download.
        """
        import datasets as hf_datasets
        import pandas as pd
        import yaml
        from gluonts.dataset.split import split as gluonts_split
        from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
        from gluonts.itertools import batcher
        from gluonts.model.evaluation import evaluate_forecasts
        from gluonts.model.forecast import QuantileForecast

        from chronos import Chronos2Pipeline

        eval_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        with open(config_path) as f:
            configs = yaml.safe_load(f)

        pipeline = Chronos2Pipeline.__new__(Chronos2Pipeline)
        pipeline.model = model
        pipeline.device = device
        pipeline.chronos_config = model.chronos_config

        results_per_dataset = {}
        all_wql = []
        all_mase = []

        for bm_config in configs:
            ds_name = bm_config["name"]
            pred_len = bm_config["prediction_length"]
            hf_repo = bm_config["hf_repo"]
            offset = bm_config["offset"]
            num_rolls = bm_config.get("num_rolls", 1)
            split_name = bm_config.get("split", "train")

            try:
                entries = None
                series_fields = None

                # Try local dataset loading first
                if self._datasets_root:
                    local_path = Path(self._datasets_root) / ds_name
                    if local_path.exists():
                        logger.info(f"  Loading {ds_name} from local: {local_path}")
                        try:
                            ds = hf_datasets.load_from_disk(str(local_path))
                            if isinstance(ds, hf_datasets.DatasetDict):
                                ds = ds[split_name]
                            ds.set_format("numpy")
                            series_fields = [
                                col for col in ds.features
                                if isinstance(ds.features[col], hf_datasets.Sequence)
                                and col != "timestamp"
                            ]
                            entries = ds
                        except Exception:
                            # Fallback to Arrow IPC
                            entries, series_fields = self._load_arrow_fallback(local_path)

                # Fallback to HuggingFace download
                if entries is None:
                    trust_remote_code = hf_repo == "autogluon/chronos_datasets_extra"
                    ds = hf_datasets.load_dataset(
                        hf_repo, ds_name, split=split_name,
                        trust_remote_code=trust_remote_code,
                    )
                    ds.set_format("numpy")
                    series_fields = [
                        col for col in ds.features
                        if isinstance(ds.features[col], hf_datasets.Sequence)
                        and col != "timestamp"
                    ]
                    entries = ds

                # Convert to GluonTS format
                first_entry = entries[0]
                dataset_freq = pd.DatetimeIndex(first_entry["timestamp"]).to_period()[0].freqstr

                gts_dataset = []
                for entry in entries:
                    for field in series_fields:
                        gts_dataset.append({
                            "start": pd.Period(entry["timestamp"][0], freq=dataset_freq),
                            "target": entry[field],
                        })

                _, test_template = gluonts_split(gts_dataset, offset=offset)
                test_data = test_template.generate_instances(pred_len, windows=num_rolls)

                # Generate forecasts — pipeline returns (N, H, Q), swap to (N, Q, H)
                forecast_outputs = []
                for batch in batcher(test_data.input, batch_size=self._eval_batch_size):
                    context = [torch.tensor(e["target"]) for e in batch]
                    with torch.no_grad():
                        quantiles, _ = pipeline.predict_quantiles(
                            context, prediction_length=pred_len,
                            quantile_levels=eval_quantiles,
                        )
                    if isinstance(quantiles, list):
                        quantiles = np.stack(quantiles).squeeze(axis=1)
                    elif isinstance(quantiles, torch.Tensor):
                        quantiles = quantiles.cpu().numpy()
                    # Pipeline raw output is (N, H, Q) → swap to (N, Q, H)
                    if quantiles.ndim == 3 and quantiles.shape[-1] == len(eval_quantiles):
                        quantiles = quantiles.swapaxes(-1, -2)
                    forecast_outputs.append(quantiles)
                forecast_outputs = np.concatenate(forecast_outputs)

                # Convert to QuantileForecast — each item is (Q, H)
                forecasts = []
                for item, ts in zip(forecast_outputs, test_data.input):
                    start = ts["start"] + len(ts["target"])
                    forecasts.append(QuantileForecast(
                        forecast_arrays=item,
                        forecast_keys=list(map(str, eval_quantiles)),
                        start_date=start,
                    ))

                metrics = (
                    evaluate_forecasts(
                        forecasts, test_data=test_data,
                        metrics=[MASE(), MeanWeightedSumQuantileLoss(eval_quantiles)],
                        batch_size=5000,
                    )
                    .reset_index(drop=True)
                    .to_dict(orient="records")
                )
                wql = metrics[0].get("mean_weighted_sum_quantile_loss", float("nan"))
                mase = metrics[0].get("MASE[0.5]", float("nan"))

                results_per_dataset[ds_name] = {"WQL": wql, "MASE": mase}
                if not math.isnan(wql):
                    all_wql.append(wql)
                if not math.isnan(mase):
                    all_mase.append(mase)

            except Exception as e:
                logger.warning(f"  Benchmark {ds_name} FAILED: {e}")
                results_per_dataset[ds_name] = {"WQL": "error", "MASE": "error"}

        return {
            "avg_wql": float(np.mean(all_wql)) if all_wql else None,
            "avg_mase": float(np.mean(all_mase)) if all_mase else None,
            "per_dataset": results_per_dataset,
        }

    @staticmethod
    def _load_arrow_fallback(data_path: Path) -> tuple[list[dict], list[str]]:
        """Load dataset directly from Arrow/Parquet files (version-mismatch fallback)."""
        import pyarrow as pa
        from pyarrow import ipc

        arrow_files = sorted(data_path.glob("*.arrow"))
        parquet_files = sorted(data_path.glob("*.parquet"))

        if not arrow_files and not parquet_files:
            raise FileNotFoundError(f"No .arrow or .parquet files in {data_path}")

        tables = []
        if arrow_files:
            for f in arrow_files:
                try:
                    reader = ipc.open_stream(str(f))
                    tables.append(reader.read_all())
                except Exception:
                    tables.append(ipc.open_file(str(f)).read_all())
        elif parquet_files:
            import pyarrow.parquet as pq
            for f in parquet_files:
                tables.append(pq.read_table(str(f)))

        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

        series_fields = [
            field.name for field in table.schema
            if field.name != "timestamp"
            and isinstance(field.type, (pa.ListType, pa.LargeListType))
        ]

        data = table.to_pydict()
        rows = []
        for i in range(len(table)):
            row = {"timestamp": np.array(data["timestamp"][i])}
            for field in series_fields:
                row[field] = np.asarray(data[field][i], dtype=np.float32)
            rows.append(row)

        return rows, series_fields

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
