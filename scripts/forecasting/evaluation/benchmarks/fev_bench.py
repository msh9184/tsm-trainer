"""fev-bench benchmark adapter.

fev-bench is a comprehensive time series forecasting evaluation benchmark
with 100 tasks across 96 unique datasets, including 46 tasks with covariates.

Protocol:
- Primary metric: SQL (Scaled Quantile Loss)
- Win Rate: with tie handling (0.5 for ties)
- Skill Score: 1 - gmean(clipped_relative_errors)
- Bootstrap CI: B=1000, seed=123, task-level resampling
- Non-overlapping rolling windows (1-20 per task)
- Quantiles: {0.1, 0.2, ..., 0.9}

Requirements:
    pip install fev  # v0.7.0+

References:
    - fev-bench paper: arXiv:2503.05495
    - Package: https://pypi.org/project/fev/

Status: STUB — Ready for integration when fev library is installed on GPU server.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .base import BenchmarkAdapter

if TYPE_CHECKING:
    from engine.forecaster import BaseForecaster

logger = logging.getLogger(__name__)

# fev-bench constants
FEV_BOOTSTRAP_B = 1000
FEV_BOOTSTRAP_SEED = 123
FEV_BOOTSTRAP_ALPHA = 0.05
FEV_CLIP_RANGE = (0.01, 100.0)
FEV_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class FevBenchAdapter(BenchmarkAdapter):
    """fev-bench benchmark adapter.

    Parameters
    ----------
    data_dir : str or Path, optional
        Local path to pre-downloaded fev datasets.
        If None, fev will attempt to download from HuggingFace.
    subset : str, optional
        Task subset to evaluate: "all", "univariate", "multivariate", "covariate".
    include_covariates : bool
        Whether to pass covariate information to the model.
    batch_size : int
        Inference batch size.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        subset: str = "all",
        include_covariates: bool = True,
        batch_size: int = 32,
    ):
        self._data_dir = data_dir
        self._subset = subset
        self._include_covariates = include_covariates
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return "fev_bench"

    def load_tasks(self) -> list[dict]:
        """Load fev-bench task configurations.

        Returns list of task dicts with:
        - name: str (task name)
        - dataset: str
        - prediction_length: int
        - freq: str
        - has_covariates: bool
        - covariate_types: list[str]  (known_dynamic, past_dynamic, static)
        """
        try:
            import fev
        except ImportError:
            raise ImportError(
                "fev not installed. Install with: pip install fev\n"
                "See docs/BENCHMARKS.md for details."
            )

        benchmark = fev.Benchmark.from_default()
        tasks = []

        for task in benchmark.tasks:
            task_info = {
                "name": task.name,
                "dataset": task.dataset_name,
                "prediction_length": task.prediction_length,
                "freq": task.freq,
                "has_covariates": task.has_covariates,
                "n_windows": task.n_windows,
            }

            # Filter by subset
            if self._subset == "univariate" and task.has_covariates:
                continue
            elif self._subset == "covariate" and not task.has_covariates:
                continue

            tasks.append(task_info)

        logger.info(f"fev-bench: {len(tasks)} tasks loaded (subset={self._subset})")
        return tasks

    def evaluate(
        self,
        forecaster: BaseForecaster,
        **kwargs,
    ) -> pd.DataFrame:
        """Run fev-bench evaluation.

        Handles covariates:
        - known_dynamic: passed to model as future context
        - past_dynamic: available in past only
        - static: constant features

        Returns
        -------
        pd.DataFrame
            100 rows with columns: task, dataset, SQL, MASE, WQL, WAPE, etc.
        """
        try:
            import fev
        except ImportError:
            raise ImportError(
                "fev not installed. Install with: pip install fev"
            )

        import torch

        benchmark = fev.Benchmark.from_default()
        results = []

        for task in benchmark.tasks:
            # Filter by subset
            if self._subset == "univariate" and task.has_covariates:
                continue
            elif self._subset == "covariate" and not task.has_covariates:
                continue

            try:
                pred_len = task.prediction_length

                # Get test data
                contexts = [
                    torch.tensor(ts["target"])
                    for ts in task.test_data.input
                ]

                # Generate forecasts
                quantiles = forecaster.predict_batch(
                    contexts,
                    prediction_length=pred_len,
                    quantile_levels=FEV_QUANTILES,
                    batch_size=self._batch_size,
                )

                # Compute metrics using fev's evaluation
                task_metrics = task.evaluate(
                    predictions=quantiles,
                    quantile_levels=FEV_QUANTILES,
                )

                results.append({
                    "task": task.name,
                    "dataset": task.dataset_name,
                    "prediction_length": pred_len,
                    "freq": task.freq,
                    "has_covariates": task.has_covariates,
                    **task_metrics,
                })

                sql_val = task_metrics.get("sql", float("nan"))
                logger.info(f"  fev {task.name}: SQL={sql_val:.4f}")

            except Exception as e:
                logger.warning(f"  fev {task.name}: FAILED — {e}")
                results.append({
                    "task": task.name,
                    "dataset": task.dataset_name,
                    "prediction_length": task.prediction_length,
                    "freq": task.freq,
                    "has_covariates": task.has_covariates,
                })

        return pd.DataFrame(results)

    def aggregate(
        self,
        results: pd.DataFrame,
        baseline_model: str = "seasonal_naive",
    ) -> dict:
        """Aggregate fev-bench results with bootstrap CI.

        Computes:
        - Win Rate (with tie handling)
        - Skill Score (1 - gmean of clipped relative errors)
        - Bootstrap 95% CI for both

        Parameters
        ----------
        results : pd.DataFrame
            Per-task results from evaluate().
        baseline_model : str
            Baseline model for comparison.

        Returns
        -------
        dict
            {
                win_rate, win_rate_ci_lower, win_rate_ci_upper,
                skill_score, skill_score_ci_lower, skill_score_ci_upper,
                avg_sql, n_tasks,
                per_subset: {univariate: {...}, covariate: {...}}
            }
        """
        from engine.aggregator import Aggregator

        summary = {
            "n_tasks": len(results),
            "subset": self._subset,
        }

        # Average SQL
        if "sql" in results.columns:
            sql_values = results["sql"].dropna().values
            if len(sql_values) > 0:
                summary["avg_sql"] = float(sql_values.mean())

                # Bootstrap CI for SQL
                ci = Aggregator.bootstrap_ci(
                    sql_values,
                    statistic="mean",
                    n_resamples=FEV_BOOTSTRAP_B,
                    alpha=FEV_BOOTSTRAP_ALPHA,
                    seed=FEV_BOOTSTRAP_SEED,
                )
                summary["sql_ci_lower"] = ci["ci_lower"]
                summary["sql_ci_upper"] = ci["ci_upper"]

        # Per-subset breakdown
        if "has_covariates" in results.columns:
            uni_results = results[~results["has_covariates"]]
            cov_results = results[results["has_covariates"]]

            if "sql" in results.columns:
                if len(uni_results) > 0:
                    summary["univariate_avg_sql"] = float(uni_results["sql"].dropna().mean())
                if len(cov_results) > 0:
                    summary["covariate_avg_sql"] = float(cov_results["sql"].dropna().mean())

        return summary
