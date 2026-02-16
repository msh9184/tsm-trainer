"""Result aggregation for time series forecasting benchmarks.

Provides methods for aggregating per-dataset results into summary metrics:
- Geometric mean of relative scores (Chronos benchmarks)
- Win rate with tie handling (fev-bench)
- Skill score (fev-bench)
- Bootstrap confidence intervals (fev-bench)
- Ranking across models

References:
    - Geometric mean for ratio aggregation: Fleming & Wallace (1986)
    - Win rate: fev-bench protocol (arXiv:2503.05495)
    - Skill score: 1 - gmean(clipped_relative_errors)
    - Bootstrap CI: Efron & Tibshirani (1993), task-level resampling
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Aggregator:
    """Aggregate per-dataset benchmark results into summary scores."""

    @staticmethod
    def relative_scores(
        model_results: pd.DataFrame,
        baseline_results: pd.DataFrame,
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute per-dataset relative scores (model / baseline).

        Parameters
        ----------
        model_results : pd.DataFrame
            Per-dataset results with 'dataset' column and metric columns.
        baseline_results : pd.DataFrame
            Baseline results with same structure.
        metrics : list[str], optional
            Metric columns to compute ratios for. Default: ["WQL", "MASE"].

        Returns
        -------
        pd.DataFrame
            Per-dataset relative scores.
        """
        if metrics is None:
            metrics = ["WQL", "MASE"]

        model_df = model_results.set_index("dataset")[metrics]
        baseline_df = baseline_results.set_index("dataset")[metrics]

        # Align datasets
        common_datasets = model_df.index.intersection(baseline_df.index)
        if len(common_datasets) < len(model_df):
            missing = set(model_df.index) - set(common_datasets)
            logger.warning(f"Datasets missing from baseline: {missing}")

        relative = model_df.loc[common_datasets] / baseline_df.loc[common_datasets]
        return relative

    @staticmethod
    def aggregate_gmean(
        model_results: pd.DataFrame,
        baseline_results: pd.DataFrame,
        metrics: list[str] | None = None,
    ) -> dict:
        """Compute geometric mean of relative scores.

        This is the standard aggregation method used in Chronos benchmarks.
        Geometric mean is the only mathematically valid way to aggregate
        ratio scores (Fleming & Wallace, 1986).

        Parameters
        ----------
        model_results : pd.DataFrame
        baseline_results : pd.DataFrame
        metrics : list[str], optional

        Returns
        -------
        dict
            {metric_name: gmean_relative_score, ...}
        """
        from scipy.stats import gmean

        relative = Aggregator.relative_scores(
            model_results, baseline_results, metrics
        )

        result = {}
        for col in relative.columns:
            values = relative[col].dropna().values
            # Filter to positive values (required for gmean; negative ratios
            # indicate data issues)
            positive = values[values > 0]
            if len(positive) > 0:
                if len(positive) < len(values):
                    logger.warning(
                        f"  Filtered {len(values) - len(positive)} non-positive "
                        f"relative scores for {col} before gmean"
                    )
                # Use log-space computation for numerical stability
                log_mean = np.mean(np.log(positive))
                result[f"agg_rel_{col.lower()}"] = float(np.exp(log_mean))
            else:
                result[f"agg_rel_{col.lower()}"] = float("nan")

        return result

    @staticmethod
    def win_rate(
        model_errors: np.ndarray,
        baseline_errors: np.ndarray,
    ) -> float:
        """Compute win rate with tie handling.

        Win Rate = mean over tasks of:
            1.0  if model < baseline (win)
            0.5  if model == baseline (tie)
            0.0  if model > baseline (loss)

        Parameters
        ----------
        model_errors : np.ndarray, shape (n_tasks,)
            Model error per task (lower is better).
        baseline_errors : np.ndarray, shape (n_tasks,)
            Baseline error per task.

        Returns
        -------
        float
            Win rate in [0, 1].
        """
        wins = (model_errors < baseline_errors).astype(float)
        ties = (model_errors == baseline_errors).astype(float) * 0.5
        return float(np.mean(wins + ties))

    @staticmethod
    def skill_score(
        model_errors: np.ndarray,
        baseline_errors: np.ndarray,
        clip_range: tuple[float, float] = (0.01, 100.0),
    ) -> float:
        """Compute skill score (fev-bench style).

        Skill Score = 1 - gmean(clip(model_error / baseline_error, clip_range))

        Positive values indicate the model is better than baseline.

        Parameters
        ----------
        model_errors : np.ndarray, shape (n_tasks,)
        baseline_errors : np.ndarray, shape (n_tasks,)
        clip_range : tuple[float, float]
            Range for clipping relative errors to prevent extreme values.

        Returns
        -------
        float
            Skill score. Positive = better than baseline.
        """
        from scipy.stats import gmean

        # Avoid division by zero
        mask = baseline_errors > 0
        if not mask.any():
            return float("nan")

        relative = model_errors[mask] / baseline_errors[mask]
        clipped = np.clip(relative, clip_range[0], clip_range[1])

        return float(1.0 - gmean(clipped))

    @staticmethod
    def bootstrap_ci(
        values: np.ndarray,
        statistic: str = "mean",
        n_resamples: int = 1000,
        alpha: float = 0.05,
        seed: int = 123,
    ) -> dict:
        """Bootstrap confidence interval.

        Uses task-level resampling following the fev-bench protocol:
        resample tasks (not individual predictions), then compute statistic.

        Parameters
        ----------
        values : np.ndarray, shape (n_tasks,)
            Per-task metric values.
        statistic : str
            Statistic to compute: "mean", "gmean", or "median".
        n_resamples : int
            Number of bootstrap resamples (fev-bench: 1000).
        alpha : float
            Significance level (fev-bench: 0.05 → 95% CI).
        seed : int
            Random seed for reproducibility (fev-bench: 123).

        Returns
        -------
        dict
            {point_estimate, ci_lower, ci_upper, std_error}
        """
        from scipy.stats import gmean

        rng = np.random.RandomState(seed)

        stat_funcs = {
            "mean": np.mean,
            "gmean": lambda x: gmean(x[x > 0]) if (x > 0).any() else float("nan"),
            "median": np.median,
        }
        stat_func = stat_funcs.get(statistic, np.mean)

        point_estimate = float(stat_func(values))

        # Bootstrap
        boot_stats = []
        n = len(values)
        for _ in range(n_resamples):
            indices = rng.randint(0, n, size=n)
            sample = values[indices]
            boot_stats.append(stat_func(sample))

        boot_stats = np.array(boot_stats)
        ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        std_error = float(np.std(boot_stats))

        return {
            "point_estimate": point_estimate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "std_error": std_error,
        }

    @staticmethod
    def generate_comparison_table(
        results: dict[str, pd.DataFrame],
        baseline_name: str = "seasonal-naive",
        metrics: list[str] | None = None,
        format: str = "markdown",
    ) -> str:
        """Generate a model comparison table.

        Parameters
        ----------
        results : dict[str, pd.DataFrame]
            {model_name: per_dataset_results_df, ...}
        baseline_name : str
            Name of the baseline model in the results dict.
        metrics : list[str], optional
            Metrics to include. Default: ["WQL", "MASE"].
        format : str
            Output format: "markdown", "latex", or "csv".

        Returns
        -------
        str
            Formatted comparison table.
        """
        if metrics is None:
            metrics = ["WQL", "MASE"]

        if baseline_name not in results:
            raise ValueError(f"Baseline '{baseline_name}' not found in results")

        baseline_df = results[baseline_name]
        rows = []

        for model_name, model_df in results.items():
            row = {"Model": model_name}
            for metric in metrics:
                # Per-dataset average
                values = model_df[metric].dropna().values
                row[f"Avg {metric}"] = float(np.mean(values)) if len(values) > 0 else float("nan")

                # Relative score vs baseline
                agg = Aggregator.aggregate_gmean(model_df, baseline_df, [metric])
                key = f"agg_rel_{metric.lower()}"
                row[f"Rel {metric}"] = agg.get(key, float("nan"))

            rows.append(row)

        table_df = pd.DataFrame(rows)

        # Add ranking columns based on relative scores (lower is better)
        for metric in metrics:
            rel_col = f"Rel {metric}"
            if rel_col in table_df.columns:
                rank_col = f"Rank {metric}"
                table_df[rank_col] = table_df[rel_col].rank(method="min").astype(int)

        if format == "markdown":
            return _format_markdown_table(table_df)
        elif format == "latex":
            return _format_latex_table(table_df)
        elif format == "csv":
            return table_df.to_csv(index=False)
        else:
            raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def load_results(results_dir: str | Path) -> pd.DataFrame:
        """Load results from a CSV file in a results directory.

        Parameters
        ----------
        results_dir : str or Path
            Path to directory or CSV file.

        Returns
        -------
        pd.DataFrame
        """
        results_dir = Path(results_dir)
        if results_dir.is_file() and results_dir.suffix == ".csv":
            return pd.read_csv(results_dir)

        # Try common filenames
        for name in ["results.csv", "chronos_bench_ii.csv", "zero-shot.csv", "in-domain.csv"]:
            path = results_dir / name
            if path.exists():
                return pd.read_csv(path)

        # Find any CSV
        csvs = list(results_dir.glob("*.csv"))
        if csvs:
            return pd.read_csv(csvs[0])

        raise FileNotFoundError(f"No CSV results found in {results_dir}")


def _format_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table."""
    lines = []
    headers = list(df.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for _, row in df.iterrows():
        cells = []
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                cells.append(f"{val:.4f}" if not np.isnan(val) else "—")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _format_latex_table(df: pd.DataFrame) -> str:
    """Format DataFrame as LaTeX table."""
    headers = list(df.columns)
    n_cols = len(headers)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular}}{{{'l' + 'c' * (n_cols - 1)}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]

    for _, row in df.iterrows():
        cells = []
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                cells.append(f"{val:.4f}" if not np.isnan(val) else "—")
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Model comparison on benchmark.}",
        "\\end{table}",
    ])

    return "\n".join(lines)
