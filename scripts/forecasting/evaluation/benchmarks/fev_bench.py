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
    - fev-bench paper: arXiv:2509.26468
    - Package: https://pypi.org/project/fev/
    - Leaderboard: https://huggingface.co/spaces/autogluon/fev-bench
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .base import BenchmarkAdapter

if TYPE_CHECKING:
    from engine.forecaster import BaseForecaster

logger = logging.getLogger(__name__)

# fev-bench constants
FEV_TASKS_YAML_URL = (
    "https://github.com/autogluon/fev/raw/refs/heads/main/benchmarks/fev_bench/tasks.yaml"
)
FEV_BOOTSTRAP_B = 1000
FEV_BOOTSTRAP_SEED = 123
FEV_BOOTSTRAP_ALPHA = 0.05
FEV_CLIP_RANGE = (0.01, 100.0)
FEV_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _has_covariates(task) -> bool:
    """Check whether a fev task has any covariates."""
    return (
        (task.known_dynamic_columns is not None and len(task.known_dynamic_columns) > 0)
        or (task.past_dynamic_columns is not None and len(task.past_dynamic_columns) > 0)
        or (task.static_columns is not None and len(task.static_columns) > 0)
    )


def _covariate_types(task) -> list[str]:
    """List covariate types present in a fev task."""
    types = []
    if task.known_dynamic_columns is not None and len(task.known_dynamic_columns) > 0:
        types.append("known_dynamic")
    if task.past_dynamic_columns is not None and len(task.past_dynamic_columns) > 0:
        types.append("past_dynamic")
    if task.static_columns is not None and len(task.static_columns) > 0:
        types.append("static")
    return types


def _window_to_contexts(past_data, target_col: str | list[str]) -> list[np.ndarray]:
    """Extract past target values from a fev window's past_data.

    Parameters
    ----------
    past_data : list[dict]
        Per-series dicts from ``window.get_input_data()[0]``.
    target_col : str or list[str]
        Target column name(s).

    Returns
    -------
    list[np.ndarray]
        One 1-D array per series (univariate targets are flattened).
    """
    if isinstance(target_col, str):
        target_col = [target_col]

    contexts: list[np.ndarray] = []
    for item in past_data:
        # For univariate: single column; for multivariate: treat each target
        # column as a separate univariate series (channel-independent).
        for col in target_col:
            values = np.asarray(item[col], dtype=np.float32)
            # Replace NaN with 0.0 for robustness
            values = np.nan_to_num(values, nan=0.0)
            contexts.append(values)
    return contexts


def _format_predictions(
    quantile_array: np.ndarray,
    quantile_levels: list[float],
    n_series: int,
    target_columns: list[str],
    horizon: int,
) -> list[dict]:
    """Convert (N, Q, H) numpy array into fev prediction format.

    fev expects ``list[dict]`` per window where each dict has:
      - ``"predictions"`` : list[float] of length H (point forecast = median)
      - ``"0.1"`` ... ``"0.9"`` : list[float] of length H

    For multivariate tasks, predictions are wrapped in a ``DatasetDict``
    keyed by target column name — but we handle single-target (univariate)
    first, and multivariate by repeating per column.

    Parameters
    ----------
    quantile_array : np.ndarray, shape (N_total, Q, H)
        N_total = n_series * len(target_columns).
    quantile_levels : list[float]
    n_series : int
        Number of original series (before multivariate expansion).
    target_columns : list[str]
    horizon : int

    Returns
    -------
    list[dict] or DatasetDict
    """
    import fev

    n_targets = len(target_columns)
    median_idx = quantile_levels.index(0.5) if 0.5 in quantile_levels else len(quantile_levels) // 2

    if n_targets == 1:
        # Univariate: straightforward list of dicts
        predictions = []
        for i in range(n_series):
            pred_dict: dict = {
                "predictions": quantile_array[i, median_idx, :horizon].tolist(),
            }
            for q_idx, q in enumerate(quantile_levels):
                pred_dict[str(q)] = quantile_array[i, q_idx, :horizon].tolist()
            predictions.append(pred_dict)
        return predictions
    else:
        # Multivariate: group by target column
        import datasets as hf_datasets

        per_target = {}
        for t_idx, col in enumerate(target_columns):
            col_preds = []
            for s_idx in range(n_series):
                flat_idx = s_idx * n_targets + t_idx
                pred_dict = {
                    "predictions": quantile_array[flat_idx, median_idx, :horizon].tolist(),
                }
                for q_idx, q in enumerate(quantile_levels):
                    pred_dict[str(q)] = quantile_array[flat_idx, q_idx, :horizon].tolist()
                col_preds.append(pred_dict)
            per_target[col] = hf_datasets.Dataset.from_list(col_preds)

        return fev.combine_univariate_predictions_to_multivariate(
            hf_datasets.DatasetDict(per_target),
            target_columns=target_columns,
        )


class FevBenchAdapter(BenchmarkAdapter):
    """fev-bench benchmark adapter.

    Parameters
    ----------
    data_dir : str or Path, optional
        Local path to pre-downloaded fev datasets (sets ``HF_DATASETS_CACHE``).
        If None, fev will attempt to download from HuggingFace.
    subset : str, optional
        Task subset to evaluate: ``"all"``, ``"univariate"``, ``"covariate"``.
    batch_size : int
        Inference batch size.
    tasks_yaml : str, optional
        URL or local path to tasks.yaml defining the fev-bench tasks.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        subset: str = "all",
        batch_size: int = 32,
        tasks_yaml: str | None = None,
    ):
        self._data_dir = data_dir
        self._subset = subset
        self._batch_size = batch_size
        self._tasks_yaml = tasks_yaml or FEV_TASKS_YAML_URL

        # If data_dir is set, configure HF cache so fev finds local data
        if data_dir is not None:
            import os
            os.environ.setdefault("HF_DATASETS_CACHE", str(data_dir))

    @property
    def name(self) -> str:
        return "fev_bench"

    def _load_benchmark(self):
        """Load fev Benchmark from YAML."""
        try:
            import fev
        except ImportError:
            raise ImportError(
                "fev not installed. Install with: pip install fev\n"
                "See docs/BENCHMARKS.md for details."
            )
        return fev.Benchmark.from_yaml(self._tasks_yaml)

    def _filter_task(self, task) -> bool:
        """Return True if the task passes the subset filter."""
        if self._subset == "all":
            return True
        has_cov = _has_covariates(task)
        if self._subset == "univariate":
            return not has_cov
        if self._subset == "covariate":
            return has_cov
        return True

    def load_tasks(self) -> list[dict]:
        """Load fev-bench task configurations.

        Returns list of task dicts with metadata for reporting.
        """
        benchmark = self._load_benchmark()
        tasks = []

        for task in benchmark.tasks:
            if not self._filter_task(task):
                continue
            tasks.append({
                "dataset_config": task.dataset_config,
                "target": task.target if isinstance(task.target, str) else ", ".join(task.target),
                "horizon": task.horizon,
                "seasonality": task.seasonality,
                "num_windows": task.num_windows,
                "has_covariates": _has_covariates(task),
                "covariate_types": _covariate_types(task),
                "quantile_levels": task.quantile_levels,
            })

        logger.info(f"fev-bench: {len(tasks)} tasks loaded (subset={self._subset})")
        return tasks

    def evaluate(
        self,
        forecaster: BaseForecaster,
        **kwargs,
    ) -> pd.DataFrame:
        """Run fev-bench evaluation.

        Follows the official fev evaluation protocol:
        1. Iterate over tasks
        2. For each task, iterate over rolling windows
        3. For each window, extract context → predict → format predictions
        4. Call task.evaluation_summary() for official metric computation

        Returns
        -------
        pd.DataFrame
            Per-task results with SQL, MASE, WQL, WAPE columns.
        """
        import torch

        benchmark = self._load_benchmark()
        results = []

        n_total = sum(1 for t in benchmark.tasks if self._filter_task(t))
        logger.info(f"fev-bench: evaluating {n_total} tasks")

        task_idx = 0
        for task in benchmark.tasks:
            if not self._filter_task(task):
                continue
            task_idx += 1

            task_start = time.time()
            target_columns = task.target_columns  # always a list
            horizon = task.horizon
            quantile_levels = task.quantile_levels or FEV_QUANTILES

            try:
                predictions_per_window = []

                for window in task.iter_windows(trust_remote_code=True):
                    past_data, future_data = window.get_input_data()

                    # Extract context arrays from past_data
                    contexts_np = _window_to_contexts(past_data, target_columns)
                    contexts = [torch.tensor(c, dtype=torch.float32) for c in contexts_np]
                    n_series = len(past_data)

                    # Generate quantile forecasts
                    quantile_array = forecaster.predict_batch(
                        contexts,
                        prediction_length=horizon,
                        quantile_levels=quantile_levels,
                        batch_size=self._batch_size,
                    )

                    # Format predictions in fev's expected format
                    preds = _format_predictions(
                        quantile_array, quantile_levels,
                        n_series, target_columns, horizon,
                    )
                    predictions_per_window.append(preds)

                # Compute metrics via fev's official evaluation
                elapsed = time.time() - task_start
                summary = task.evaluation_summary(
                    predictions_per_window,
                    model_name=forecaster.name,
                    inference_time_s=elapsed,
                )

                # Build result row
                row = {
                    "task": task.dataset_config,
                    "target": task.target if isinstance(task.target, str) else ", ".join(task.target),
                    "horizon": horizon,
                    "seasonality": task.seasonality,
                    "num_windows": task.num_windows,
                    "has_covariates": _has_covariates(task),
                    "elapsed_s": round(elapsed, 1),
                }
                # Merge fev summary metrics into row
                if isinstance(summary, dict):
                    row.update(summary)
                elif isinstance(summary, pd.DataFrame):
                    # evaluation_summary may return a DataFrame
                    for col in summary.columns:
                        row[col] = summary[col].iloc[0] if len(summary) > 0 else None
                elif isinstance(summary, pd.Series):
                    row.update(summary.to_dict())

                results.append(row)

                sql_val = row.get("SQL", row.get("sql", float("nan")))
                logger.info(
                    f"  [{task_idx}/{n_total}] {task.dataset_config}: "
                    f"SQL={sql_val:.4f}, {elapsed:.1f}s"
                )

            except Exception as e:
                elapsed = time.time() - task_start
                logger.warning(
                    f"  [{task_idx}/{n_total}] {task.dataset_config}: FAILED — {e}"
                )
                results.append({
                    "task": task.dataset_config,
                    "target": task.target if isinstance(task.target, str) else ", ".join(task.target),
                    "horizon": horizon,
                    "seasonality": task.seasonality,
                    "num_windows": task.num_windows,
                    "has_covariates": _has_covariates(task),
                    "elapsed_s": round(elapsed, 1),
                })

        return pd.DataFrame(results)

    def aggregate(
        self,
        results: pd.DataFrame,
    ) -> dict:
        """Aggregate fev-bench results.

        Computes average SQL across tasks and per-subset breakdowns.
        For full Win Rate and Skill Score, use fev.leaderboard() directly
        with multiple models' results.

        Returns
        -------
        dict
            Summary metrics including avg_sql, n_tasks, per-subset breakdowns.
        """
        from engine.aggregator import Aggregator

        summary: dict = {
            "n_tasks": len(results),
            "subset": self._subset,
        }

        # Identify SQL column (fev may use "SQL" or "sql")
        sql_col = None
        for candidate in ["SQL", "sql"]:
            if candidate in results.columns:
                sql_col = candidate
                break

        if sql_col is not None:
            sql_values = results[sql_col].dropna().values
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

        # MASE average
        for metric in ["MASE", "mase"]:
            if metric in results.columns:
                vals = results[metric].dropna().values
                if len(vals) > 0:
                    summary["avg_mase"] = float(vals.mean())
                break

        # Per-subset breakdown
        if "has_covariates" in results.columns and sql_col is not None:
            uni_results = results[~results["has_covariates"]]
            cov_results = results[results["has_covariates"]]

            uni_sql = uni_results[sql_col].dropna().values
            cov_sql = cov_results[sql_col].dropna().values

            if len(uni_sql) > 0:
                summary["univariate_avg_sql"] = float(uni_sql.mean())
                summary["univariate_n_tasks"] = len(uni_results)
            if len(cov_sql) > 0:
                summary["covariate_avg_sql"] = float(cov_sql.mean())
                summary["covariate_n_tasks"] = len(cov_results)

        return summary

    def generate_leaderboard(
        self,
        *result_csvs: str | Path,
        baseline_name: str = "SeasonalNaive",
    ) -> pd.DataFrame:
        """Generate fev-bench leaderboard from multiple model results.

        This is a convenience wrapper around ``fev.leaderboard()``.

        Parameters
        ----------
        *result_csvs : str or Path
            Paths to per-model result CSVs from evaluate().
        baseline_name : str
            Name of the baseline model for skill score computation.

        Returns
        -------
        pd.DataFrame
            Leaderboard with Win Rate, Skill Score, and bootstrap CIs.
        """
        import fev

        dfs = [pd.read_csv(p) for p in result_csvs]
        combined = pd.concat(dfs, ignore_index=True)

        return fev.leaderboard(
            combined,
            baseline=baseline_name,
            bootstrap_b=FEV_BOOTSTRAP_B,
            bootstrap_seed=FEV_BOOTSTRAP_SEED,
        )
