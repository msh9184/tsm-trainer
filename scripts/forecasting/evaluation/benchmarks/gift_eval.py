"""GIFT-Eval benchmark adapter.

GIFT-Eval (General Time Series Forecasting Model Evaluation) is a comprehensive
benchmark from Salesforce AI Research with ~98 task configurations across
23 base datasets.

Protocol:
- TEST_SPLIT = 0.1 (last 10% of each series)
- Non-overlapping rolling windows (MAX_WINDOW = 20)
- 11 metrics per configuration
- Primary metrics: WQL (CRPS proxy) + MASE
- Quantiles: {0.1, 0.2, ..., 0.9}

Requirements:
    pip install gift-eval
    # or: git clone https://github.com/SalesforceAIResearch/gift-eval.git
    #     cd gift-eval && pip install -e .
    # Download: huggingface-cli download Salesforce/GiftEval --repo-type=dataset

References:
    - GIFT-Eval paper: arXiv:2501.xxxxx (Salesforce AI Research)
    - Leaderboard: https://huggingface.co/spaces/Salesforce/GiftEval

Status: STUB — Ready for integration when gift-eval library is installed on GPU server.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .base import BenchmarkAdapter

if TYPE_CHECKING:
    from engine.forecaster import BaseForecaster

logger = logging.getLogger(__name__)

# GIFT-Eval constants
GIFT_EVAL_TEST_SPLIT = 0.1
GIFT_EVAL_MAX_WINDOW = 20
GIFT_EVAL_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class GiftEvalAdapter(BenchmarkAdapter):
    """GIFT-Eval benchmark adapter.

    Parameters
    ----------
    data_dir : str or Path, optional
        Local path to GIFT-Eval data. Falls back to GIFT_EVAL environment
        variable, then HuggingFace download.
    terms : list[str], optional
        Forecast horizons to evaluate: "short", "medium", "long".
        Default: all three.
    datasets : str, optional
        Dataset subset: "all", "short", "med_long".
    batch_size : int
        Inference batch size.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        terms: list[str] | None = None,
        datasets: str = "all",
        batch_size: int = 32,
    ):
        import os

        self._data_dir = data_dir or os.environ.get("GIFT_EVAL", None)
        self._terms = terms or ["short", "medium", "long"]
        self._datasets = datasets
        self._batch_size = batch_size

        if self._data_dir is None:
            logger.warning(
                "GIFT-Eval data directory not set. "
                "Set GIFT_EVAL env var or pass data_dir parameter. "
                "See docs/BENCHMARKS.md for setup instructions."
            )

    @property
    def name(self) -> str:
        return "gift_eval"

    def load_tasks(self) -> list[dict]:
        """Load GIFT-Eval task configurations.

        Returns list of task dicts with:
        - dataset: str
        - term: str (short/medium/long)
        - freq: str
        - prediction_length: int
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

        # Load available dataset configs from gift-eval
        tasks = []
        for ds in Dataset.available():
            for term in self._terms:
                tasks.append({
                    "dataset": ds.name,
                    "term": term,
                    "freq": ds.freq,
                    "prediction_length": ds.prediction_length(term),
                })

        logger.info(f"GIFT-Eval: {len(tasks)} task configurations loaded")
        return tasks

    def evaluate(
        self,
        forecaster: BaseForecaster,
        **kwargs,
    ) -> pd.DataFrame:
        """Run GIFT-Eval evaluation.

        Uses the official gift-eval evaluate_model() function with:
        - axis=None
        - mask_invalid_label=True
        - allow_nan_forecast=False

        Returns
        -------
        pd.DataFrame
            ~98 rows with columns: dataset, term, freq, prediction_length,
            plus 11 metric columns (CRPS, MASE, WQL, SMAPE, etc.)
        """
        try:
            from gift_eval.data import Dataset
            from gift_eval.metrics import evaluate_model
        except ImportError:
            raise ImportError(
                "gift-eval not installed. See load_tasks() docstring for install instructions."
            )

        tasks = self.load_tasks()
        results = []

        for task in tasks:
            try:
                ds = Dataset(task["dataset"], term=task["term"])
                pred_len = task["prediction_length"]

                # Generate forecasts
                import torch
                contexts = [torch.tensor(ts["target"]) for ts in ds.test_data.input]
                quantiles = forecaster.predict_batch(
                    contexts,
                    prediction_length=pred_len,
                    quantile_levels=GIFT_EVAL_QUANTILES,
                    batch_size=self._batch_size,
                )

                # Evaluate using official function
                metrics = evaluate_model(
                    predictions=quantiles,
                    test_data=ds.test_data,
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                )

                results.append({
                    "dataset": task["dataset"],
                    "term": task["term"],
                    "freq": task["freq"],
                    "prediction_length": pred_len,
                    **metrics,
                })

                logger.info(
                    f"  GIFT-Eval {task['dataset']}/{task['term']}: "
                    f"CRPS={metrics.get('crps', 'N/A'):.4f}"
                )

            except Exception as e:
                logger.warning(f"  GIFT-Eval {task['dataset']}/{task['term']}: FAILED — {e}")
                results.append({
                    "dataset": task["dataset"],
                    "term": task["term"],
                    "freq": task["freq"],
                    "prediction_length": task["prediction_length"],
                })

        return pd.DataFrame(results)

    def aggregate(self, results: pd.DataFrame) -> dict:
        """Aggregate GIFT-Eval results.

        Returns
        -------
        dict
            avg_rank_crps, avg_rank_mase, normalized_crps, normalized_mase
        """
        summary = {
            "n_tasks": len(results),
            "terms": self._terms,
        }

        for metric in ["crps", "mase", "wql", "smape"]:
            if metric in results.columns:
                values = results[metric].dropna().values
                if len(values) > 0:
                    summary[f"avg_{metric}"] = float(values.mean())

        return summary

    def export_for_leaderboard(
        self,
        results: pd.DataFrame,
        output_dir: str | Path,
    ) -> Path:
        """Export results in GIFT-Eval leaderboard submission format.

        Parameters
        ----------
        results : pd.DataFrame
        output_dir : str or Path

        Returns
        -------
        Path
            Path to the exported file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "gift_eval_submission.csv"
        results.to_csv(output_path, index=False)
        logger.info(f"GIFT-Eval leaderboard submission exported: {output_path}")
        return output_path
