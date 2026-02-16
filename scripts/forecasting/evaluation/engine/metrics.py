"""Metric computation for time series forecasting evaluation.

Provides unified access to all metrics used across benchmarks:
- WQL (Weighted Quantile Loss) — Chronos benchmarks, GIFT-Eval
- MASE (Mean Absolute Scaled Error) — Chronos benchmarks, GIFT-Eval
- SQL (Scaled Quantile Loss) — fev-bench primary metric
- MSE / MAE — LTSF benchmark
- CRPS (approximated via quantiles) — GIFT-Eval
- SMAPE — GIFT-Eval supplementary

Mathematical definitions follow the exact formulations used in each
benchmark's official implementation to ensure reproducible results.

References:
    - WQL: Chronos paper (arXiv:2403.07815), GluonTS implementation
    - SQL: fev-bench (arXiv:2503.05495)
    - CRPS: Matheson & Winkler (1976), quantile approximation
    - MASE: Hyndman & Koehler (2006)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gluonts.ev.metrics import Metric as GluonTSMetric

logger = logging.getLogger(__name__)

# Standard quantile levels used across benchmarks
QUANTILES_9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QUANTILES_21 = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
]

# Seasonal periods for MASE computation (from GluonTS / Chronos convention)
SEASONAL_PERIODS = {
    "T": 1440,   # minutely → 1 day
    "5T": 288,   # 5-minutely → 1 day
    "10T": 144,  # 10-minutely → 1 day
    "15T": 96,   # 15-minutely → 1 day
    "H": 24,     # hourly → 1 day
    "D": 7,      # daily → 1 week
    "W": 52,     # weekly → 1 year
    "M": 12,     # monthly → 1 year
    "MS": 12,    # month-start → 1 year
    "Q": 4,      # quarterly → 1 year
    "QS": 4,     # quarter-start → 1 year
    "A": 1,      # yearly → itself
    "Y": 1,      # yearly → itself
    "B": 5,      # business-daily → 1 week
}


class MetricRegistry:
    """Registry for evaluation metrics.

    Provides factory methods for creating GluonTS-compatible metric instances
    and standalone metric computation functions for benchmarks that don't
    use GluonTS (e.g., fev-bench, LTSF).
    """

    STANDARD_QUANTILES = QUANTILES_9

    @staticmethod
    def get_gluonts_metrics(
        names: list[str],
        quantile_levels: list[float] | None = None,
    ) -> list[GluonTSMetric]:
        """Create GluonTS metric instances by name.

        Parameters
        ----------
        names : list[str]
            Metric names: "wql", "mase", or both.
        quantile_levels : list[float], optional
            Quantile levels for WQL. Default: [0.1, ..., 0.9].

        Returns
        -------
        list[GluonTSMetric]
            GluonTS metric objects for use with evaluate_forecasts().
        """
        from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss

        if quantile_levels is None:
            quantile_levels = QUANTILES_9

        metrics = []
        for name in names:
            name_lower = name.lower()
            if name_lower == "wql":
                metrics.append(MeanWeightedSumQuantileLoss(quantile_levels))
            elif name_lower == "mase":
                metrics.append(MASE())
            else:
                raise ValueError(
                    f"Unknown GluonTS metric: {name}. "
                    f"Available: 'wql', 'mase'"
                )
        return metrics

    # ------------------------------------------------------------------
    # Standalone metric functions (no GluonTS dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def quantile_loss(
        y_true: np.ndarray,
        y_pred_quantiles: np.ndarray,
        quantile_levels: list[float],
    ) -> np.ndarray:
        """Compute quantile loss (pinball loss) per quantile level.

        QL_q(y, ŷ_q) = q * max(y - ŷ_q, 0) + (1-q) * max(ŷ_q - y, 0)

        Parameters
        ----------
        y_true : np.ndarray, shape (N, H)
            Ground truth values.
        y_pred_quantiles : np.ndarray, shape (N, Q, H)
            Predicted quantile values.
        quantile_levels : list[float]
            Quantile levels corresponding to axis 1 of y_pred_quantiles.

        Returns
        -------
        np.ndarray, shape (Q,)
            Mean quantile loss per quantile level.
        """
        y = y_true[:, np.newaxis, :]  # (N, 1, H)
        q = np.array(quantile_levels)[np.newaxis, :, np.newaxis]  # (1, Q, 1)

        errors = y - y_pred_quantiles  # (N, Q, H)
        loss = np.where(errors >= 0, q * errors, (q - 1) * errors)

        return loss.mean(axis=(0, 2))  # mean over samples and horizon

    @staticmethod
    def wql(
        y_true: np.ndarray,
        y_pred_quantiles: np.ndarray,
        quantile_levels: list[float] | None = None,
    ) -> float:
        """Weighted Quantile Loss (WQL).

        WQL = (1/|Q|) * sum_q [ sum_{i,t} QL_q(y_{i,t}, ŷ_{q,i,t})
                                 / sum_{i,t} |y_{i,t}| ]

        This is the normalized version used in Chronos benchmarks.

        Parameters
        ----------
        y_true : np.ndarray, shape (N, H)
        y_pred_quantiles : np.ndarray, shape (N, Q, H)
        quantile_levels : list[float], optional

        Returns
        -------
        float
            WQL score (lower is better).
        """
        if quantile_levels is None:
            quantile_levels = QUANTILES_9

        y = y_true[:, np.newaxis, :]  # (N, 1, H)
        q = np.array(quantile_levels)[np.newaxis, :, np.newaxis]  # (1, Q, 1)

        errors = y - y_pred_quantiles  # (N, Q, H)
        ql = np.where(errors >= 0, q * errors, (q - 1) * errors)

        # Sum over all items and timesteps, then normalize
        abs_sum = np.abs(y_true).sum()
        if abs_sum == 0:
            return float("nan")

        per_quantile = ql.sum(axis=(0, 2)) / abs_sum  # (Q,)
        return float(per_quantile.mean())

    @staticmethod
    def mase(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        seasonal_period: int = 1,
    ) -> float:
        """Mean Absolute Scaled Error (MASE).

        MASE = mean_i [ mean_t |y_{i,t} - ŷ_{i,t}|  /  a_i ]

        where a_i = (1/(T-m)) * sum_{t=m+1}^{T} |y_{i,t} - y_{i,t-m}|
        is the in-sample seasonal naive MAE for series i with season m.

        Parameters
        ----------
        y_true : np.ndarray, shape (N, H)
            Ground truth future values.
        y_pred : np.ndarray, shape (N, H)
            Point forecast (typically median / q=0.5).
        y_past : np.ndarray, shape (N, T)
            Historical (context) values for computing seasonal naive scale.
        seasonal_period : int
            Seasonal period m.

        Returns
        -------
        float
            MASE score (lower is better).
        """
        n_series = y_true.shape[0]
        mase_values = []

        n_skipped = 0
        for i in range(n_series):
            past = y_past[i]
            # Seasonal naive in-sample error
            if len(past) <= seasonal_period:
                # Series too short for seasonal differencing — use non-seasonal naive (m=1)
                if len(past) > 1:
                    naive_errors = np.abs(past[1:] - past[:-1])
                    scale = naive_errors.mean()
                else:
                    n_skipped += 1
                    continue
            else:
                naive_errors = np.abs(past[seasonal_period:] - past[:-seasonal_period])
                scale = naive_errors.mean()

            if scale == 0 or np.isnan(scale):
                n_skipped += 1
                continue  # skip constant series

            forecast_error = np.abs(y_true[i] - y_pred[i]).mean()
            mase_values.append(forecast_error / scale)

        if n_skipped > 0:
            logger.debug(
                f"MASE: skipped {n_skipped}/{n_series} series "
                f"(constant or too short for seasonal_period={seasonal_period})"
            )

        if not mase_values:
            return float("nan")

        return float(np.mean(mase_values))

    @staticmethod
    def sql(
        y_true: np.ndarray,
        y_pred_quantiles: np.ndarray,
        y_past: np.ndarray,
        quantile_levels: list[float] | None = None,
        seasonal_period: int = 1,
    ) -> float:
        """Scaled Quantile Loss (SQL) — fev-bench primary metric.

        SQL = mean_items [ mean_q ( QL_q_per_timestep / seasonal_error_i ) ]

        Per-item scaled version that normalizes by each series' seasonal
        naive error, similar to how MASE normalizes by seasonal naive MAE.

        Parameters
        ----------
        y_true : np.ndarray, shape (N, H)
        y_pred_quantiles : np.ndarray, shape (N, Q, H)
        y_past : np.ndarray, shape (N, T)
        quantile_levels : list[float], optional
        seasonal_period : int

        Returns
        -------
        float
            SQL score (lower is better).
        """
        if quantile_levels is None:
            quantile_levels = QUANTILES_9

        n_series = y_true.shape[0]
        q = np.array(quantile_levels)[np.newaxis, :, np.newaxis]  # (1, Q, 1)

        sql_values = []
        for i in range(n_series):
            past = y_past[i]
            # Seasonal naive scale
            if len(past) <= seasonal_period:
                scale = np.abs(past).mean()
            else:
                naive_errors = np.abs(past[seasonal_period:] - past[:-seasonal_period])
                scale = naive_errors.mean()

            if scale == 0 or np.isnan(scale):
                continue

            y_i = y_true[i][np.newaxis, np.newaxis, :]  # (1, 1, H)
            yhat_i = y_pred_quantiles[i][np.newaxis, :, :]  # (1, Q, H)

            errors = y_i - yhat_i
            ql = np.where(errors >= 0, q * errors, (q - 1) * errors)

            # Mean over timesteps, then over quantiles, scaled by seasonal error
            ql_per_q = ql.mean(axis=2).squeeze(0)  # (Q,)
            sql_values.append(float(ql_per_q.mean() / scale))

        if not sql_values:
            return float("nan")

        return float(np.mean(sql_values))

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error.

        Parameters
        ----------
        y_true : np.ndarray, shape (N, H)
        y_pred : np.ndarray, shape (N, H)

        Returns
        -------
        float
        """
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error.

        Parameters
        ----------
        y_true : np.ndarray, shape (N, H)
        y_pred : np.ndarray, shape (N, H)

        Returns
        -------
        float
        """
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error.

        sMAPE = 200 * mean( |y - ŷ| / (|y| + |ŷ|) )

        Parameters
        ----------
        y_true : np.ndarray, shape (N, H)
        y_pred : np.ndarray, shape (N, H)

        Returns
        -------
        float
        """
        denom = np.abs(y_true) + np.abs(y_pred)
        # Avoid division by zero
        mask = denom > 0
        if not mask.any():
            return float("nan")
        return float(200.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))

    @staticmethod
    def crps_quantile(
        y_true: np.ndarray,
        y_pred_quantiles: np.ndarray,
        quantile_levels: list[float] | None = None,
    ) -> float:
        """CRPS approximated from quantile forecasts.

        CRPS ≈ (2 / |Q|) * sum_q QL_q(y, ŷ_q)

        This is the standard quantile-based approximation of CRPS,
        equivalent to the approach used in GIFT-Eval.

        Parameters
        ----------
        y_true : np.ndarray, shape (N, H)
        y_pred_quantiles : np.ndarray, shape (N, Q, H)
        quantile_levels : list[float], optional

        Returns
        -------
        float
        """
        if quantile_levels is None:
            quantile_levels = QUANTILES_9

        y = y_true[:, np.newaxis, :]  # (N, 1, H)
        q = np.array(quantile_levels)[np.newaxis, :, np.newaxis]

        errors = y - y_pred_quantiles
        ql = np.where(errors >= 0, q * errors, (q - 1) * errors)

        return float(2.0 * ql.mean())

    @staticmethod
    def get_seasonal_period(freq: str) -> int:
        """Get the standard seasonal period for a given frequency string.

        Parameters
        ----------
        freq : str
            Pandas-style frequency string (e.g., "H", "D", "M").

        Returns
        -------
        int
            Seasonal period. Defaults to 1 if frequency is unknown.
        """
        # Normalize frequency string
        freq_clean = freq.strip().replace("-", "")

        # Remove numeric prefix (e.g., "1H" → "H", "5T" → "5T")
        # But keep it if it's a multiplier like "5T", "10T", "15T"
        if len(freq_clean) > 1 and freq_clean[0] == "1" and not freq_clean[1].isdigit():
            freq_clean = freq_clean[1:]

        freq_upper = freq_clean.upper()

        # Handle common aliases
        if freq_upper.endswith("MIN"):
            freq_upper = freq_upper.replace("MIN", "T")
        # Handle pandas offset aliases (added in newer pandas versions)
        alias_map = {
            "ME": "M",   # MonthEnd
            "YE": "Y",   # YearEnd
            "QE": "Q",   # QuarterEnd
            "BME": "M",  # BusinessMonthEnd
            "BQE": "Q",  # BusinessQuarterEnd
            "BYE": "Y",  # BusinessYearEnd
            "SME": "MS",  # SemiMonthEnd
            "BH": "H",   # BusinessHour
        }
        freq_upper = alias_map.get(freq_upper, freq_upper)

        period = SEASONAL_PERIODS.get(freq_upper, None)
        if period is None:
            logger.debug(f"Unknown frequency '{freq}' (normalized: '{freq_upper}'), defaulting to period=1")
            return 1
        return period
