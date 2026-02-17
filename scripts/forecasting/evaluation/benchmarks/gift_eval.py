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


def _build_gift_eval_metrics():
    """Build the standard 11 GIFT-Eval GluonTS metrics."""
    from gluonts.ev.metrics import (
        MAE,
        MAPE,
        MASE,
        MSE,
        MSIS,
        ND,
        NRMSE,
        RMSE,
        SMAPE,
        MeanWeightedSumQuantileLoss,
    )

    return [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=GIFT_EVAL_QUANTILES,
        ),
    ]


class GiftEvalPredictor:
    """GluonTS-compatible predictor wrapping a BaseForecaster.

    This adapts our forecaster interface to produce ``QuantileForecast``
    objects that gift-eval's ``evaluate_model()`` expects.
    """

    def __init__(
        self,
        forecaster: BaseForecaster,
        prediction_length: int,
        freq: str,
        quantile_levels: list[float] | None = None,
        batch_size: int = 32,
    ):
        self.forecaster = forecaster
        self.prediction_length = prediction_length
        self.freq = freq
        self.quantile_levels = quantile_levels or GIFT_EVAL_QUANTILES
        self.batch_size = batch_size

    def predict(self, dataset, **kwargs):
        """Generate QuantileForecast objects from dataset entries.

        Yields one ``QuantileForecast`` per entry in the dataset.
        """
        import torch
        from gluonts.model.forecast import QuantileForecast

        # Collect all contexts
        entries = list(dataset)
        contexts = []
        start_dates = []
        for entry in entries:
            target = np.asarray(entry["target"], dtype=np.float32)
            # For multivariate targets, each variate is treated independently
            if target.ndim == 2:
                # Multivariate: shape (n_vars, T) — each row is a univariate series
                for var_idx in range(target.shape[0]):
                    contexts.append(torch.tensor(target[var_idx], dtype=torch.float32))
            else:
                contexts.append(torch.tensor(target, dtype=torch.float32))
            start_dates.append(entry.get("start", pd.Timestamp("1970-01-01")))

        if not contexts:
            return

        # Predict in batches
        all_quantiles = self.forecaster.predict_batch(
            contexts,
            prediction_length=self.prediction_length,
            quantile_levels=self.quantile_levels,
            batch_size=self.batch_size,
        )
        # all_quantiles shape: (N, Q, H)

        # Yield QuantileForecast objects
        ctx_idx = 0
        for entry_idx, entry in enumerate(entries):
            target = np.asarray(entry["target"], dtype=np.float32)

            if target.ndim == 2:
                n_vars = target.shape[0]
                # Collect quantile forecasts for all variates
                forecast_arrays = []
                for _ in range(n_vars):
                    # Shape: (Q, H)
                    forecast_arrays.append(all_quantiles[ctx_idx])
                    ctx_idx += 1
                # Stack to (n_vars, Q, H) then transpose to (Q, H, n_vars)
                # But QuantileForecast expects (Q, H) for univariate
                # For multivariate, stack into (Q, H) per variate
                # gift-eval with to_univariate=True already splits, so this
                # branch handles the multivariate case when to_univariate=False
                stacked = np.stack(forecast_arrays, axis=0)  # (n_vars, Q, H)
                # Rearrange to (Q, H) if single variate, otherwise (Q, H*n_vars)?
                # Actually for GIFT-Eval, we handle this per variate
                for var_idx in range(n_vars):
                    yield QuantileForecast(
                        forecast_arrays=stacked[var_idx],  # (Q, H)
                        forecast_keys=[str(q) for q in self.quantile_levels],
                        start_date=start_dates[entry_idx],
                    )
            else:
                yield QuantileForecast(
                    forecast_arrays=all_quantiles[ctx_idx],  # (Q, H)
                    forecast_keys=[str(q) for q in self.quantile_levels],
                    start_date=start_dates[entry_idx],
                )
                ctx_idx += 1


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
        """Run GIFT-Eval evaluation.

        Follows the official GIFT-Eval protocol from chronos-2.ipynb:
        1. For each (dataset, term), load Dataset
        2. Convert multivariate to univariate if needed
        3. Create GluonTS-compatible predictor
        4. Call evaluate_model() with 11 metrics, axis=None

        Returns
        -------
        pd.DataFrame
            ~98 rows with 15 columns matching leaderboard format.
        """
        try:
            from gift_eval.data import Dataset
            from gluonts.model import evaluate_model
            from gluonts.time_feature import get_seasonality
        except ImportError:
            raise ImportError(
                "gift-eval or gluonts not installed. Install with:\n"
                "  pip install gift-eval gluonts"
            )

        metrics = _build_gift_eval_metrics()
        task_configs = self._get_task_configs()
        results = []

        n_total = len(task_configs)
        logger.info(f"GIFT-Eval: evaluating {n_total} task configurations")

        for idx, (ds_name, term) in enumerate(task_configs, 1):
            task_start = time.time()

            try:
                # Load dataset — convert multivariate to univariate
                ds_raw = Dataset(name=ds_name, term=term, to_univariate=False)
                to_univariate = ds_raw.target_dim > 1
                ds = Dataset(name=ds_name, term=term, to_univariate=to_univariate)

                pred_len = ds.prediction_length
                freq = ds.freq
                season_length = get_seasonality(freq)

                # Create GluonTS-compatible predictor
                predictor = GiftEvalPredictor(
                    forecaster=forecaster,
                    prediction_length=pred_len,
                    freq=freq,
                    quantile_levels=GIFT_EVAL_QUANTILES,
                    batch_size=self._batch_size,
                )

                # Evaluate using official GIFT-Eval protocol
                res = evaluate_model(
                    predictor,
                    test_data=ds.test_data,
                    metrics=metrics,
                    batch_size=1024,
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=season_length,
                )

                elapsed = time.time() - task_start

                # Build result row matching leaderboard CSV format
                row = {
                    "dataset": f"{ds_name}/{term}" if "/" not in ds_name else f"{ds_name}/{term}",
                    "model": forecaster.name,
                }

                # Extract metric values from evaluate_model result
                # The result is a dict or DataFrame with metric column names
                if isinstance(res, dict):
                    row.update(res)
                elif isinstance(res, pd.DataFrame):
                    for col in res.columns:
                        row[col] = res[col].iloc[0] if len(res) > 0 else None

                # Add metadata columns
                row["domain"] = self._get_domain(ds_name)
                row["num_variates"] = ds_raw.target_dim
                row["elapsed_s"] = round(elapsed, 1)

                results.append(row)

                # Log progress
                crps = row.get(
                    "eval_metrics/mean_weighted_sum_quantile_loss",
                    row.get("mean_weighted_sum_quantile_loss", "N/A"),
                )
                crps_str = f"{crps:.4f}" if isinstance(crps, (int, float)) else str(crps)
                logger.info(
                    f"  [{idx}/{n_total}] {ds_name}/{term}: "
                    f"CRPS={crps_str}, {elapsed:.1f}s"
                )

            except Exception as e:
                elapsed = time.time() - task_start
                logger.warning(
                    f"  [{idx}/{n_total}] {ds_name}/{term}: FAILED — {e}"
                )
                results.append({
                    "dataset": f"{ds_name}/{term}" if "/" not in ds_name else f"{ds_name}/{term}",
                    "model": forecaster.name,
                    "domain": self._get_domain(ds_name),
                    "elapsed_s": round(elapsed, 1),
                })

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
