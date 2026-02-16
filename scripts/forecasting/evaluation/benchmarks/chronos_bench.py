"""Chronos Benchmark I (in-domain) & II (zero-shot) adapter.

Implements the evaluation protocol from the original Chronos paper
(arXiv:2403.07815) and Chronos-2 paper (arXiv:2510.15821).

Benchmark structure:
- Chronos Bench I (in-domain): 15 datasets from training corpus
- Chronos Bench II (zero-shot): 27 datasets NOT in training corpus
- Metrics: WQL (probabilistic) + MASE (point)
- Aggregation: Geometric mean of relative scores vs Seasonal Naive

Data loading supports:
- Local dataset paths (for offline GPU servers)
- HuggingFace Hub datasets (with cache)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

from .base import BenchmarkAdapter

if TYPE_CHECKING:
    from engine.forecaster import BaseForecaster

logger = logging.getLogger(__name__)


class ChronosBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for Chronos Benchmark I (in-domain) and II (zero-shot).

    Parameters
    ----------
    config_path : str or Path
        Path to benchmark YAML config (e.g., zero-shot.yaml, in-domain.yaml).
    benchmark_type : str
        "in-domain" or "zero-shot". Used for naming and aggregation.
    baseline_results_path : str or Path, optional
        Path to baseline (Seasonal Naive) results CSV for relative score computation.
    datasets_root : str or Path, optional
        Root directory for local datasets. If set, datasets are loaded from
        {datasets_root}/{dataset_name}/ instead of HuggingFace Hub.
    batch_size : int
        Inference batch size.
    quantile_levels : list[float], optional
        Quantile levels for WQL computation.
    """

    def __init__(
        self,
        config_path: str | Path,
        benchmark_type: str = "zero-shot",
        baseline_results_path: str | Path | None = None,
        datasets_root: str | Path | None = None,
        batch_size: int = 32,
        quantile_levels: list[float] | None = None,
    ):
        self._config_path = Path(config_path)
        self._benchmark_type = benchmark_type
        self._baseline_path = Path(baseline_results_path) if baseline_results_path else None
        self._datasets_root = datasets_root
        self._batch_size = batch_size
        self._quantile_levels = quantile_levels or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Auto-detect baseline path from results directory
        if self._baseline_path is None:
            results_dir = self._config_path.parent.parent / "results"
            candidate = results_dir / f"seasonal-naive-{benchmark_type}.csv"
            if candidate.exists():
                self._baseline_path = candidate

    @property
    def name(self) -> str:
        return f"chronos_bench_{self._benchmark_type.replace('-', '_')}"

    def load_tasks(self) -> list[dict]:
        with open(self._config_path) as f:
            return yaml.safe_load(f)

    def evaluate(
        self,
        forecaster: BaseForecaster,
        **kwargs,
    ) -> pd.DataFrame:
        """Run Chronos benchmark evaluation.

        Parameters
        ----------
        forecaster : BaseForecaster
            Model to evaluate.

        Returns
        -------
        pd.DataFrame
            Per-dataset results with columns: dataset, model, WQL, MASE.
        """
        from engine.evaluator import Evaluator

        evaluator = Evaluator(
            forecaster=forecaster,
            quantile_levels=self._quantile_levels,
            batch_size=self._batch_size,
            datasets_root=self._datasets_root,
        )

        return evaluator.evaluate_benchmark(
            config_path=self._config_path,
            **kwargs,
        )

    def aggregate(self, results: pd.DataFrame) -> dict:
        """Compute aggregated relative scores.

        Uses geometric mean of per-dataset relative scores vs Seasonal Naive,
        following the exact protocol from the Chronos paper.

        Parameters
        ----------
        results : pd.DataFrame
            Per-dataset results from evaluate().

        Returns
        -------
        dict
            {
                agg_rel_wql: float,    # Geometric mean relative WQL
                agg_rel_mase: float,   # Geometric mean relative MASE
                avg_wql: float,        # Simple average WQL
                avg_mase: float,       # Simple average MASE
                n_datasets: int,
                benchmark_type: str,
            }
        """
        from engine.aggregator import Aggregator

        summary = {
            "benchmark_type": self._benchmark_type,
            "n_datasets": len(results),
            "avg_wql": float(results["WQL"].dropna().mean()),
            "avg_mase": float(results["MASE"].dropna().mean()),
        }

        # Compute relative scores if baseline available
        if self._baseline_path is not None and self._baseline_path.exists():
            baseline_df = pd.read_csv(self._baseline_path)
            agg_scores = Aggregator.aggregate_gmean(results, baseline_df)
            summary.update(agg_scores)
        else:
            logger.warning(
                f"Baseline results not found at {self._baseline_path}. "
                f"Relative scores will not be computed."
            )

        return summary


class ChronosLiteBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for the Lite Benchmark (training-time quick evaluation).

    Wraps 5 datasets for fast validation during training (~3 minutes on A100).
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        datasets_root: str | Path | None = None,
        batch_size: int = 32,
    ):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "lite-benchmark.yaml"
        self._config_path = Path(config_path)
        self._datasets_root = datasets_root
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return "chronos_lite"

    def load_tasks(self) -> list[dict]:
        with open(self._config_path) as f:
            return yaml.safe_load(f)

    def evaluate(self, forecaster: BaseForecaster, **kwargs) -> pd.DataFrame:
        from engine.evaluator import Evaluator

        evaluator = Evaluator(
            forecaster=forecaster,
            batch_size=self._batch_size,
            datasets_root=self._datasets_root,
        )
        return evaluator.evaluate_benchmark(self._config_path, **kwargs)

    def aggregate(self, results: pd.DataFrame) -> dict:
        return {
            "avg_wql": float(results["WQL"].dropna().mean()),
            "avg_mase": float(results["MASE"].dropna().mean()),
            "n_datasets": len(results),
        }
