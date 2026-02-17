"""Abstract base class for benchmark adapters.

Each benchmark suite (Chronos, GIFT-Eval, fev-bench, LTSF) has its own
data format, evaluation protocol, and aggregation method. The BenchmarkAdapter
ABC provides a unified interface for all benchmarks.

Usage:
    adapter = ChronosBenchmarkAdapter(config_path="configs/chronos-ii.yaml")
    results = adapter.evaluate(forecaster)
    summary = adapter.aggregate(results)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from engine.forecaster import BaseForecaster

logger = logging.getLogger(__name__)


class BenchmarkAdapter(ABC):
    """Abstract base class for benchmark evaluation adapters.

    Subclasses must implement:
    - name: Human-readable benchmark name
    - load_tasks(): Load task configurations
    - evaluate(): Run evaluation on all tasks
    - aggregate(): Compute summary metrics from per-task results
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark suite name (e.g., 'chronos_bench_ii', 'gift_eval')."""
        ...

    @abstractmethod
    def load_tasks(self) -> list[dict]:
        """Load task configurations.

        Returns
        -------
        list[dict]
            List of task config dicts. Each dict contains at minimum:
            - name: str
            - prediction_length: int
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        forecaster: BaseForecaster,
        **kwargs,
    ) -> pd.DataFrame:
        """Run benchmark evaluation.

        Parameters
        ----------
        forecaster : BaseForecaster
            Model to evaluate.

        Returns
        -------
        pd.DataFrame
            Per-task results. Columns depend on the benchmark.
        """
        ...

    @abstractmethod
    def aggregate(self, results: pd.DataFrame) -> dict:
        """Aggregate per-task results into summary metrics.

        Parameters
        ----------
        results : pd.DataFrame
            Per-task results from evaluate().

        Returns
        -------
        dict
            Summary metrics. Keys depend on the benchmark.
        """
        ...

    def save_results(
        self,
        results: pd.DataFrame,
        summary: dict,
        output_dir: str | Path,
        experiment_name: str | None = None,
    ) -> Path:
        """Save evaluation results to disk.

        Parameters
        ----------
        results : pd.DataFrame
            Per-task results.
        summary : dict
            Aggregated summary metrics.
        output_dir : str or Path
            Directory to save results.
        experiment_name : str, optional
            Experiment identifier.

        Returns
        -------
        Path
            Path to the output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save per-task results
        results_path = output_dir / f"{self.name}.csv"
        results.to_csv(results_path, index=False)

        # Save summary
        summary_path = output_dir / f"{self.name}_summary.json"
        summary_data = {
            "benchmark": self.name,
            "timestamp": datetime.now().isoformat(),
            "n_tasks": len(results),
            "metrics": summary,
        }
        if experiment_name:
            summary_data["experiment"] = experiment_name

        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)

        logger.info(f"Results saved: {results_path}")
        logger.info(f"Summary saved: {summary_path}")

        return output_dir
