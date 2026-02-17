"""GIFT-Eval benchmark adapter.

GIFT-Eval (General Time Series Forecasting Model Evaluation) is a comprehensive
benchmark from Salesforce AI Research with ~98 task configurations across
23+ base datasets, 7 domains, and 10 frequencies.

Protocol:
- TEST_SPLIT = 0.1 (last 10% of each series)
- Non-overlapping rolling windows (MAX_WINDOW = 20)
- 11 metrics per configuration
- Primary ranking metric: CRPS (via MeanWeightedSumQuantileLoss)
- Quantiles: {0.1, 0.2, ..., 0.9}
- Terms: short (1x), medium (10x), long (15x) of base prediction length

Requirements:
    pip install gift-eval
    # or: git clone https://github.com/SalesforceAIResearch/gift-eval.git
    #     cd gift-eval && pip install -e .
    # Download: huggingface-cli download Salesforce/GiftEval --repo-type=dataset

References:
    - GIFT-Eval paper: arXiv:2410.10393 (NeurIPS 2024 Datasets & Benchmarks)
    - GitHub: https://github.com/SalesforceAIResearch/gift-eval
    - Leaderboard: https://huggingface.co/spaces/Salesforce/GIFT-Eval
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .base import BenchmarkAdapter

if TYPE_CHECKING:
    from engine.forecaster import BaseForecaster

logger = logging.getLogger(__name__)

# GIFT-Eval constants
GIFT_EVAL_TEST_SPLIT = 0.1
GIFT_EVAL_MAX_WINDOW = 20
GIFT_EVAL_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Datasets with only short term (no medium/long)
# These are low-frequency datasets where medium/long would be too large
GIFT_EVAL_SHORT_ONLY = {
    "m4_yearly", "m4_quarterly", "m4_monthly", "m4_weekly", "m4_daily",
    "m4_hourly", "hospital", "covid_deaths", "car_parts_with_missing",
    "restaurant", "temperature_rain_with_missing", "saugeenday",
    "us_births",
}

# All known GIFT-Eval dataset names (short-term configs)
GIFT_EVAL_DATASETS = [
    "m4_yearly", "m4_quarterly", "m4_monthly", "m4_weekly", "m4_daily",
    "m4_hourly",
    "electricity/15T", "electricity/H", "electricity/D", "electricity/W",
    "solar/10T", "solar/H", "solar/D", "solar/W",
    "hospital", "covid_deaths",
    "us_births/D", "us_births/M", "us_births/W",
    "saugeenday/D", "saugeenday/M", "saugeenday/W",
    "temperature_rain_with_missing",
    "kdd_cup_2018_with_missing/H", "kdd_cup_2018_with_missing/D",
    "car_parts_with_missing", "restaurant",
    "hierarchical_sales/D", "hierarchical_sales/W",
    "LOOP_SEATTLE/5T", "LOOP_SEATTLE/H", "LOOP_SEATTLE/D",
    "SZ_TAXI/15T", "SZ_TAXI/H",
    "M_DENSE/H", "M_DENSE/D",
    "ett1/15T", "ett1/H", "ett1/D", "ett1/W",
    "ett2/15T", "ett2/H", "ett2/D", "ett2/W",
    "jena_weather/10T", "jena_weather/H", "jena_weather/D",
    "bitbrains_fast_storage/5T", "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T", "bitbrains_rnd/H",
    "bizitobs_application", "bizitobs_service",
    "bizitobs_l2c/5T", "bizitobs_l2c/H",
]

# ── Metric display infrastructure ──────────────────────────────────────────
# Ordered metric columns for GIFT-Eval reporting.
# Each entry: (list_of_column_name_candidates, display_name)
# The 11 metrics match the official GIFT-Eval leaderboard.
_METRIC_COLUMNS = [
    (["eval_metrics/mean_weighted_sum_quantile_loss", "mean_weighted_sum_quantile_loss"], "CRPS"),
    (["eval_metrics/MASE[0.5]", "MASE[0.5]"], "MASE"),
    (["eval_metrics/sMAPE[0.5]", "sMAPE[0.5]"], "sMAPE"),
    (["eval_metrics/MAPE[0.5]", "MAPE[0.5]"], "MAPE"),
    (["eval_metrics/MSE[0.5]", "MSE[0.5]"], "MSE"),
    (["eval_metrics/MAE[0.5]", "MAE[0.5]"], "MAE"),
    (["eval_metrics/RMSE[0.5]", "RMSE[0.5]"], "RMSE"),
    (["eval_metrics/NRMSE[0.5]", "NRMSE[0.5]"], "NRMSE"),
    (["eval_metrics/ND[0.5]", "ND[0.5]"], "ND"),
    (["eval_metrics/MSIS", "MSIS"], "MSIS"),
    (["eval_metrics/MSE[mean]", "MSE[mean]"], "MSE*"),
]


def _fmt_metric(value: float) -> str:
    """Format a metric value with adaptive precision (right-aligned, 8 chars)."""
    abs_val = abs(value)
    if abs_val == 0:
        return "  0.0000"
    if abs_val >= 1000:
        return f"{value:>8.1f}"
    if abs_val >= 100:
        return f"{value:>8.2f}"
    if abs_val >= 10:
        return f"{value:>8.3f}"
    return f"{value:>8.4f}"


def _extract_metrics(row: dict) -> list[tuple[str, float | None]]:
    """Extract ordered metric values from a GIFT-Eval result row."""
    result = []
    for candidates, name in _METRIC_COLUMNS:
        val = None
        for key in candidates:
            if key in row and row[key] is not None:
                try:
                    val = float(row[key])
                except (ValueError, TypeError):
                    pass
                break
        result.append((name, val))
    return result


def _format_task_report(
    row: dict,
    idx: int,
    n_total: int,
    ds_name: str,
    term: str,
    domain: str,
    elapsed: float,
) -> str:
    """Format per-task GIFT-Eval metrics as a clean multi-line log block.

    Example::

      ─── [01/97] electricity/15T/short ──────────── Energy │ 12.3s ───
        CRPS   0.0234 │ MASE   0.8912 │ sMAPE  0.1045 │ MAPE   0.0891
        MSE    0.0123 │ MAE    0.0567 │ RMSE   0.1234 │ NRMSE  0.0456
        ND     0.0678 │ MSIS   1.2345 │ MSE*   0.0145
    """
    metrics = _extract_metrics(row)
    valid = [(n, v) for n, v in metrics if v is not None]

    # Header with task info
    idx_w = len(str(n_total))
    label = f"{ds_name}/{term}"
    left = f"─── [{idx:0{idx_w}d}/{n_total}] {label} "
    right = f" {domain} │ {elapsed:.1f}s ───"
    W = 75
    pad = max(0, W - len(left) - len(right))
    header = left + "─" * pad + right

    # Metric lines (4 per line, pipe-separated)
    lines = [header]
    for start in range(0, len(valid), 4):
        chunk = valid[start : start + 4]
        parts = [f"{name:<5s}{_fmt_metric(val)}" for name, val in chunk]
        lines.append("  " + " │ ".join(parts))

    return "\n".join(lines)


def _format_eval_summary(results: list[dict], total_elapsed: float) -> str:
    """Build a comprehensive summary box for GIFT-Eval evaluation results.

    Displays average metrics across all tasks and per-domain CRPS breakdown.
    """
    n = len(results)
    if n == 0:
        return ""

    # Compute average for each metric
    metric_avgs: list[tuple[str, float | None, int]] = []
    for candidates, name in _METRIC_COLUMNS:
        vals = []
        for r in results:
            for key in candidates:
                if key in r and r[key] is not None:
                    try:
                        vals.append(float(r[key]))
                    except (ValueError, TypeError):
                        pass
                    break
        avg = sum(vals) / len(vals) if vals else None
        metric_avgs.append((name, avg, len(vals)))

    # Format elapsed time
    h, rem = divmod(int(total_elapsed), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        time_str = f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        time_str = f"{m}m {s:02d}s"
    else:
        time_str = f"{s}s"

    W = 75
    sep = "═" * W

    lines = [
        f"╔{sep}╗",
        f"║{'GIFT-Eval Evaluation Complete':^{W}}║",
        f"║{f'{n} tasks │ {time_str}':^{W}}║",
        f"╠{sep}╣",
        f"║{' Average Metrics (across all tasks)':<{W}}║",
    ]

    # Metric averages (4 per line)
    valid_avgs = [(name, avg) for name, avg, _ in metric_avgs if avg is not None]
    for start in range(0, len(valid_avgs), 4):
        chunk = valid_avgs[start : start + 4]
        parts = [f"{name:<5s}{_fmt_metric(val)}" for name, val in chunk]
        content = "   " + " │ ".join(parts)
        lines.append(f"║{content:<{W}}║")

    # Per-domain CRPS breakdown
    crps_keys = _METRIC_COLUMNS[0][0]  # CRPS column candidates
    domain_data: dict[str, list[float]] = {}
    for r in results:
        domain = r.get("domain", "Other")
        for key in crps_keys:
            if key in r and r[key] is not None:
                try:
                    domain_data.setdefault(domain, []).append(float(r[key]))
                except (ValueError, TypeError):
                    pass
                break

    if domain_data:
        lines.append(f"╠{sep}╣")
        lines.append(f"║{' Per-Domain CRPS':<{W}}║")
        # Sort by average CRPS (best first)
        sorted_domains = sorted(
            domain_data.items(),
            key=lambda x: sum(x[1]) / len(x[1]),
        )
        for dom, vals in sorted_domains:
            avg = sum(vals) / len(vals)
            entry = f"   {dom:<16s}{avg:>8.4f}  ({len(vals):>2d} tasks)"
            lines.append(f"║{entry:<{W}}║")

    lines.append(f"╚{sep}╝")
    return "\n".join(lines)


# Datasets with sub-daily frequency that support medium/long terms
GIFT_EVAL_MED_LONG_DATASETS = [
    "electricity/15T", "electricity/H",
    "solar/10T", "solar/H",
    "kdd_cup_2018_with_missing/H",
    "LOOP_SEATTLE/5T", "LOOP_SEATTLE/H",
    "SZ_TAXI/15T",
    "M_DENSE/H",
    "ett1/15T", "ett1/H",
    "ett2/15T", "ett2/H",
    "jena_weather/10T", "jena_weather/H",
    "bitbrains_fast_storage/5T",
    "bitbrains_rnd/5T",
    "bizitobs_application", "bizitobs_service",
    "bizitobs_l2c/5T", "bizitobs_l2c/H",
]


class GiftEvalAdapter(BenchmarkAdapter):
    """GIFT-Eval benchmark adapter.

    Parameters
    ----------
    data_dir : str or Path, optional
        Local path to GIFT-Eval data. Falls back to ``GIFT_EVAL`` environment
        variable, then HuggingFace download.
    terms : list[str], optional
        Forecast terms to evaluate: ``"short"``, ``"medium"``, ``"long"``.
        Default: all three.
    datasets : list[str], optional
        Specific dataset names to evaluate. Default: all available.
    batch_size : int
        Inference batch size.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        terms: list[str] | None = None,
        datasets: list[str] | None = None,
        batch_size: int = 32,
    ):
        import os

        self._data_dir = data_dir or os.environ.get("GIFT_EVAL", None)
        self._terms = terms or ["short", "medium", "long"]
        self._datasets = datasets
        self._batch_size = batch_size

        # Set GIFT_EVAL env var for gift-eval library
        if self._data_dir is not None:
            os.environ["GIFT_EVAL"] = str(self._data_dir)

        if self._data_dir is None:
            logger.warning(
                "GIFT-Eval data directory not set. "
                "Set GIFT_EVAL env var or pass data_dir parameter. "
                "See docs/BENCHMARKS.md for setup instructions."
            )

    @property
    def name(self) -> str:
        return "gift_eval"

    def _get_task_configs(self) -> list[tuple[str, str]]:
        """Build list of (dataset_name, term) pairs to evaluate.

        Returns
        -------
        list[tuple[str, str]]
            Each tuple is (dataset_name, term).
        """
        configs = []
        dataset_list = self._datasets or GIFT_EVAL_DATASETS

        for ds_name in dataset_list:
            for term in self._terms:
                # Only short-only datasets skip medium/long
                base_name = ds_name.split("/")[0]
                if term != "short" and ds_name not in GIFT_EVAL_MED_LONG_DATASETS:
                    continue
                configs.append((ds_name, term))

        return configs

    def load_tasks(self) -> list[dict]:
        """Load GIFT-Eval task configurations.

        Returns list of task dicts with dataset, term, prediction_length info.
        """
        try:
            from gift_eval.data import Dataset
        except ImportError:
            raise ImportError(
                "gift-eval not installed. Install with:\n"
                "  git clone https://github.com/SalesforceAIResearch/gift-eval.git\n"
                "  cd gift-eval && pip install -e .\n"
                "See docs/BENCHMARKS.md for details."
            )

        task_configs = self._get_task_configs()
        tasks = []

        for ds_name, term in task_configs:
            try:
                ds = Dataset(name=ds_name, term=term)
                tasks.append({
                    "dataset": ds_name,
                    "term": term,
                    "freq": ds.freq,
                    "prediction_length": ds.prediction_length,
                    "target_dim": ds.target_dim,
                    "windows": ds.windows,
                })
            except Exception as e:
                logger.debug(f"Skipping {ds_name}/{term}: {e}")

        logger.info(f"GIFT-Eval: {len(tasks)} task configurations loaded")
        return tasks

    def evaluate(
        self,
        forecaster: BaseForecaster,
        **kwargs,
    ) -> pd.DataFrame:
        """Run GIFT-Eval evaluation with fast vectorized metrics.

        Optimized pipeline that bypasses GluonTS ``evaluate_model()``:
        1. Load Dataset once with ``to_univariate=True``
        2. Extract test inputs/labels directly from GluonTS TestData
        3. Batch GPU inference → (N, Q, H) numpy array
        4. Compute all 11 metrics via vectorized numpy (no QuantileForecast objects)

        This eliminates the primary CPU bottleneck:
        - No QuantileForecast Python object creation (thousands per task)
        - No GluonTS per-item metric iteration
        - No pandas DataFrame manipulation in evaluate_model

        Returns
        -------
        pd.DataFrame
            ~98 rows with 15 columns matching leaderboard format.
        """
        import torch

        try:
            from gift_eval.data import Dataset
            from gluonts.time_feature import get_seasonality
        except ImportError:
            raise ImportError(
                "gift-eval or gluonts not installed. Install with:\n"
                "  pip install gift-eval gluonts"
            )

        from engine.metrics import MetricRegistry

        task_configs = self._get_task_configs()
        results = []

        n_total = len(task_configs)
        logger.info(f"GIFT-Eval: evaluating {n_total} task configurations (fast mode)")

        # Suppress pandas FutureWarning from gluonts/gift-eval internals
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"(gluonts|gift_eval)",
        )

        eval_start = time.time()

        for idx, (ds_name, term) in enumerate(task_configs, 1):
            task_start = time.time()

            try:
                # Load dataset ONCE — always to_univariate=True for Chronos-2
                ds = Dataset(name=ds_name, term=term, to_univariate=True)
                num_variates = ds.target_dim  # After to_univariate=True (always 1 for Chronos-2)

                pred_len = ds.prediction_length
                freq = ds.freq
                season_length = get_seasonality(freq)

                # ── Extract test data directly (bypass GluonTS evaluate_model) ──
                test_data = ds.test_data
                contexts = []
                y_pasts = []
                y_trues = []

                for inp, lbl in zip(test_data.input, test_data.label):
                    past = np.asarray(inp["target"], dtype=np.float32)
                    future = np.asarray(lbl["target"], dtype=np.float32)

                    if past.ndim == 2:
                        # Multivariate after to_univariate=True shouldn't happen,
                        # but handle defensively: split each variate
                        for v in range(past.shape[0]):
                            contexts.append(torch.tensor(past[v], dtype=torch.float32))
                            y_pasts.append(past[v])
                            fut_v = future[v] if future.ndim == 2 else future
                            y_trues.append(fut_v[:pred_len])
                    else:
                        contexts.append(torch.tensor(past, dtype=torch.float32))
                        y_pasts.append(past)
                        y_trues.append(future[:pred_len])

                if not contexts:
                    raise ValueError(f"No test entries for {ds_name}/{term}")

                y_true = np.stack(y_trues)  # (N, H)
                n_series = len(contexts)

                # ── Batch GPU inference ──
                y_pred_q = forecaster.predict_batch(
                    contexts,
                    prediction_length=pred_len,
                    quantile_levels=GIFT_EVAL_QUANTILES,
                    batch_size=self._batch_size,
                )  # (N, Q, H)

                # ── Fast vectorized metric computation (all 11 metrics) ──
                metric_dict = MetricRegistry.compute_gift_eval_metrics_fast(
                    y_pred_q, y_true, y_pasts,
                    GIFT_EVAL_QUANTILES, season_length,
                )

                elapsed = time.time() - task_start

                # Build result row matching leaderboard format
                row = {
                    "dataset": f"{ds_name}/{term}",
                    "model": forecaster.name,
                    **metric_dict,
                    "domain": self._get_domain(ds_name),
                    "num_variates": num_variates,
                    "n_series": n_series,
                    "elapsed_s": round(elapsed, 1),
                }
                results.append(row)

                # Log all 11 metrics for this task
                logger.info(
                    _format_task_report(
                        row, idx, n_total, ds_name, term,
                        row.get("domain", "Other"), elapsed,
                    )
                )

            except Exception as e:
                elapsed = time.time() - task_start
                idx_w = len(str(n_total))
                logger.warning(
                    f"─── [{idx:0{idx_w}d}/{n_total}] {ds_name}/{term} "
                    f"─── FAILED │ {elapsed:.1f}s ───\n  {e}"
                )
                results.append({
                    "dataset": f"{ds_name}/{term}",
                    "model": forecaster.name,
                    "domain": self._get_domain(ds_name),
                    "elapsed_s": round(elapsed, 1),
                })

        # Log comprehensive evaluation summary
        total_elapsed = time.time() - eval_start
        summary_text = _format_eval_summary(results, total_elapsed)
        if summary_text:
            logger.info(f"\n{summary_text}")

        return pd.DataFrame(results)

    def aggregate(self, results: pd.DataFrame) -> dict:
        """Aggregate GIFT-Eval results.

        Computes average CRPS (primary), MASE, and per-domain breakdowns.

        Returns
        -------
        dict
            Summary metrics.
        """
        summary: dict = {
            "n_tasks": len(results),
            "terms": self._terms,
        }

        # Primary metric: CRPS (MeanWeightedSumQuantileLoss)
        crps_col = None
        for candidate in [
            "eval_metrics/mean_weighted_sum_quantile_loss",
            "mean_weighted_sum_quantile_loss",
        ]:
            if candidate in results.columns:
                crps_col = candidate
                break

        if crps_col is not None:
            crps_values = results[crps_col].dropna().values
            if len(crps_values) > 0:
                summary["avg_crps"] = float(crps_values.mean())

        # Secondary metrics
        for metric_name, col_candidates in [
            ("avg_mase", ["eval_metrics/MASE[0.5]", "MASE[0.5]", "MASE"]),
            ("avg_smape", ["eval_metrics/sMAPE[0.5]", "sMAPE[0.5]", "smape"]),
            ("avg_mse", ["eval_metrics/MSE[0.5]", "MSE[0.5]"]),
        ]:
            for col in col_candidates:
                if col in results.columns:
                    vals = results[col].dropna().values
                    if len(vals) > 0:
                        summary[metric_name] = float(vals.mean())
                    break

        # Per-domain breakdown
        if "domain" in results.columns and crps_col is not None:
            domain_avgs = {}
            for domain in results["domain"].dropna().unique():
                domain_df = results[results["domain"] == domain]
                vals = domain_df[crps_col].dropna().values
                if len(vals) > 0:
                    domain_avgs[domain] = float(vals.mean())
            if domain_avgs:
                summary["per_domain_crps"] = domain_avgs

        return summary

    def export_for_leaderboard(
        self,
        results: pd.DataFrame,
        output_dir: str | Path,
        model_name: str = "chronos-2",
        model_type: str = "pretrained",
    ) -> Path:
        """Export results in GIFT-Eval leaderboard submission format.

        Creates ``all_results.csv`` and ``config.json`` matching the
        leaderboard PR submission format.

        Parameters
        ----------
        results : pd.DataFrame
        output_dir : str or Path
        model_name : str
        model_type : str

        Returns
        -------
        Path
            Path to the output directory.
        """
        import json

        output_dir = Path(output_dir) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results CSV
        results_path = output_dir / "all_results.csv"
        results.to_csv(results_path, index=False)

        # Save config.json
        config = {
            "model": model_name,
            "model_type": model_type,
            "model_dtype": "bfloat16",
            "model_link": "",
            "code_link": "",
            "org": "",
            "testdata_leakage": "No",
            "replication_code_available": "Yes",
        }
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"GIFT-Eval leaderboard submission exported: {output_dir}")
        return output_dir

    @staticmethod
    def _get_domain(ds_name: str) -> str:
        """Map dataset name to domain category."""
        base = ds_name.split("/")[0].lower()
        domain_map = {
            "m4_yearly": "Econ/Fin", "m4_quarterly": "Econ/Fin",
            "m4_monthly": "Econ/Fin", "m4_weekly": "Econ/Fin",
            "m4_daily": "Econ/Fin", "m4_hourly": "Econ/Fin",
            "electricity": "Energy", "solar": "Energy",
            "ett1": "Energy", "ett2": "Energy",
            "hospital": "Healthcare", "covid_deaths": "Healthcare",
            "us_births": "Healthcare",
            "saugeenday": "Nature", "temperature_rain_with_missing": "Nature",
            "kdd_cup_2018_with_missing": "Nature", "jena_weather": "Nature",
            "car_parts_with_missing": "Sales", "restaurant": "Sales",
            "hierarchical_sales": "Sales",
            "loop_seattle": "Transport", "sz_taxi": "Transport",
            "m_dense": "Transport",
            "bitbrains_fast_storage": "Web/CloudOps",
            "bitbrains_rnd": "Web/CloudOps",
            "bizitobs_application": "Web/CloudOps",
            "bizitobs_service": "Web/CloudOps",
            "bizitobs_l2c": "Web/CloudOps",
        }
        return domain_map.get(base, "Other")
