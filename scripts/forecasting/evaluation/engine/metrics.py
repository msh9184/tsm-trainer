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

    # ------------------------------------------------------------------
    # Fast batch metric computation (bypass GluonTS)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_chronos_metrics_fast(
        y_pred_quantiles: np.ndarray,
        y_true: np.ndarray,
        y_past: list[np.ndarray],
        quantile_levels: list[float],
        seasonal_period: int = 1,
    ) -> dict:
        """Compute WQL + MASE matching GluonTS evaluate_forecasts output.

        Replaces the entire GluonTS evaluation pipeline for Chronos benchmarks:
        - No QuantileForecast object creation
        - No evaluate_forecasts() call
        - Pure numpy vectorized computation

        The WQL computation matches GluonTS ``MeanWeightedSumQuantileLoss``:
        per-item normalized quantile loss, then averaged across items.

        Parameters
        ----------
        y_pred_quantiles : np.ndarray, shape (N, Q, H)
            Predicted quantile values.
        y_true : np.ndarray, shape (N, H)
            Ground truth future values.
        y_past : list of np.ndarray
            Historical (context) values per series, variable length.
        quantile_levels : list[float]
            Quantile levels corresponding to Q axis.
        seasonal_period : int
            Seasonal period for MASE naive scale.

        Returns
        -------
        dict
            {"WQL": float, "MASE": float}
        """
        N = y_true.shape[0]
        Q = len(quantile_levels)

        # ── WQL (matching GluonTS MeanWeightedSumQuantileLoss) ──
        # Per-item: 2 * sum(QL over Q,H) / (Q * sum|y|)
        q_arr = np.asarray(quantile_levels, dtype=np.float64)[np.newaxis, :, np.newaxis]
        y_exp = y_true[:, np.newaxis, :].astype(np.float64)
        yq = y_pred_quantiles.astype(np.float64)

        errors = y_exp - yq  # (N, Q, H)
        ql = np.where(errors >= 0, q_arr * errors, (q_arr - 1) * errors)

        ql_sum_per_item = ql.sum(axis=(1, 2))  # (N,)
        abs_y_sum = np.abs(y_true.astype(np.float64)).sum(axis=1)  # (N,)

        valid_wql = abs_y_sum > 0
        wql_values = np.full(N, np.nan)
        if valid_wql.any():
            wql_values[valid_wql] = (
                2.0 * ql_sum_per_item[valid_wql] / (Q * abs_y_sum[valid_wql])
            )
        wql = float(np.nanmean(wql_values))

        # ── MASE (median point forecast) ──
        median_idx = min(range(Q), key=lambda i: abs(quantile_levels[i] - 0.5))
        y_median = y_pred_quantiles[:, median_idx, :]  # (N, H)

        # Compute seasonal naive scales for all items (nanmean for missing values)
        scales = np.empty(N)
        for i in range(N):
            past = y_past[i]
            if len(past) <= seasonal_period:
                if len(past) > 1:
                    scales[i] = float(np.nanmean(np.abs(np.diff(past))))
                else:
                    scales[i] = np.nan
            else:
                diffs = past[seasonal_period:] - past[:-seasonal_period]
                scales[i] = float(np.nanmean(np.abs(diffs)))

        valid_scale = (scales > 0) & ~np.isnan(scales)
        mae_per_item = np.nanmean(np.abs(y_true - y_median), axis=1)  # (N,)
        mase_values = np.full(N, np.nan)
        if valid_scale.any():
            mase_values[valid_scale] = mae_per_item[valid_scale] / scales[valid_scale]
        mase = float(np.nanmean(mase_values))

        n_skipped = int((~valid_scale).sum())
        if n_skipped > 0:
            logger.debug(
                f"MASE fast: skipped {n_skipped}/{N} series "
                f"(constant or too short for seasonal_period={seasonal_period})"
            )

        return {"WQL": wql, "MASE": mase}

    @staticmethod
    def compute_gift_eval_metrics_fast(
        y_pred_quantiles: np.ndarray,
        y_true: np.ndarray,
        y_past: list[np.ndarray],
        quantile_levels: list[float],
        seasonal_period: int = 1,
    ) -> dict:
        """Compute all 11 GIFT-Eval metrics from raw numpy arrays.

        Replaces GluonTS ``evaluate_model()`` entirely, computing all metrics
        in vectorized numpy operations where possible. Only the seasonal scale
        computation for MASE/MSIS requires a per-item loop.

        Metrics computed (matching GluonTS/GIFT-Eval leaderboard):
        1. CRPS (MeanWeightedSumQuantileLoss) — quantile-based
        2. MASE[0.5] — point forecast, seasonal-scaled
        3. sMAPE[0.5] — symmetric percentage error
        4. MAPE[0.5] — mean absolute percentage error
        5. MSE[0.5] — mean squared error
        6. MAE[0.5] — mean absolute error
        7. RMSE[0.5] — root mean squared error
        8. NRMSE[0.5] — normalized RMSE
        9. ND[0.5] — normalized deviation
        10. MSIS — mean scaled interval score (alpha=0.05)
        11. MSE[mean] — equals MSE[0.5] for quantile models

        Parameters
        ----------
        y_pred_quantiles : np.ndarray, shape (N, Q, H)
        y_true : np.ndarray, shape (N, H)
        y_past : list of np.ndarray, each (T_i,)
        quantile_levels : list[float], length Q
        seasonal_period : int

        Returns
        -------
        dict
            Keys match GluonTS evaluate_model column names for compatibility
            with existing reporting infrastructure.
        """
        N = y_true.shape[0]
        Q = len(quantile_levels)

        # Find key quantile indices
        median_idx = min(range(Q), key=lambda i: abs(quantile_levels[i] - 0.5))
        q01_idx = min(range(Q), key=lambda i: abs(quantile_levels[i] - 0.1))
        q09_idx = min(range(Q), key=lambda i: abs(quantile_levels[i] - 0.9))

        y_median = y_pred_quantiles[:, median_idx, :]  # (N, H)

        # ── CRPS (= GluonTS MeanWeightedSumQuantileLoss) ──
        q_arr = np.asarray(quantile_levels, dtype=np.float64)[np.newaxis, :, np.newaxis]
        y_exp = y_true[:, np.newaxis, :].astype(np.float64)
        yq = y_pred_quantiles.astype(np.float64)

        errors = y_exp - yq  # (N, Q, H)
        ql = np.where(errors >= 0, q_arr * errors, (q_arr - 1) * errors)

        ql_sum_per_item = ql.sum(axis=(1, 2))  # (N,)
        abs_y_sum = np.abs(y_true.astype(np.float64)).sum(axis=1)  # (N,)

        valid_crps = abs_y_sum > 0
        crps_vals = np.full(N, np.nan)
        if valid_crps.any():
            crps_vals[valid_crps] = (
                2.0 * ql_sum_per_item[valid_crps] / (Q * abs_y_sum[valid_crps])
            )
        crps = float(np.nanmean(crps_vals))

        # ── Point forecast metrics (vectorized) ──
        diff = (y_true - y_median).astype(np.float64)  # (N, H)
        abs_diff = np.abs(diff)

        # MSE[0.5]: per-item nanmean, then nanmean (handles missing values)
        mse_per_item = np.nanmean(diff ** 2, axis=1)  # (N,)
        mse = float(np.nanmean(mse_per_item))

        # MAE[0.5]: per-item nanmean, then nanmean (handles missing values)
        mae_per_item = np.nanmean(abs_diff, axis=1)  # (N,)
        mae = float(np.nanmean(mae_per_item))

        # RMSE[0.5]: per-item RMSE, then nanmean (handles missing values)
        rmse_per_item = np.sqrt(mse_per_item)  # (N,)
        rmse = float(np.nanmean(rmse_per_item))

        # sMAPE[0.5]: 200 * mean(|y-ŷ| / (|y|+|ŷ|)) per item
        denom_smape = np.abs(y_true.astype(np.float64)) + np.abs(y_median.astype(np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_smape = np.where(denom_smape > 0, abs_diff / denom_smape, np.nan)
        with np.errstate(all="ignore"):
            smape_per_item = 200.0 * np.nanmean(ratio_smape, axis=1)  # (N,)
            smape = float(np.nanmean(smape_per_item))

        # MAPE[0.5]: 100 * mean(|y-ŷ| / |y|) per item
        abs_y_point = np.abs(y_true.astype(np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_mape = np.where(abs_y_point > 0, abs_diff / abs_y_point, np.nan)
        with np.errstate(all="ignore"):
            mape_per_item = 100.0 * np.nanmean(ratio_mape, axis=1)  # (N,)
            mape = float(np.nanmean(mape_per_item))

        # NRMSE[0.5]: RMSE / mean(|y|) per item
        mean_abs_y = np.nanmean(np.abs(y_true.astype(np.float64)), axis=1)  # (N,)
        nrmse_valid = mean_abs_y > 0
        nrmse_vals = np.full(N, np.nan)
        if nrmse_valid.any():
            nrmse_vals[nrmse_valid] = rmse_per_item[nrmse_valid] / mean_abs_y[nrmse_valid]
        nrmse = float(np.nanmean(nrmse_vals))

        # ND[0.5]: sum|y-ŷ| / sum|y| per item (nansum for missing values)
        sum_abs_diff = np.nansum(abs_diff, axis=1)  # (N,)
        sum_abs_y = np.nansum(np.abs(y_true.astype(np.float64)), axis=1)  # (N,)
        nd_valid = sum_abs_y > 0
        nd_vals = np.full(N, np.nan)
        if nd_valid.any():
            nd_vals[nd_valid] = sum_abs_diff[nd_valid] / sum_abs_y[nd_valid]
        nd = float(np.nanmean(nd_vals))

        # ── MSIS: Mean Scaled Interval Score (alpha=0.05) ──
        # GluonTS uses quantile(0.025) and quantile(0.975), which clip to
        # q(0.1) and q(0.9) respectively via np.interp boundary behavior.
        lower = y_pred_quantiles[:, q01_idx, :].astype(np.float64)  # (N, H)
        upper = y_pred_quantiles[:, q09_idx, :].astype(np.float64)  # (N, H)
        y_true_f64 = y_true.astype(np.float64)

        alpha = 0.05
        interval_width = upper - lower
        penalty_lower = np.maximum(lower - y_true_f64, 0)
        penalty_upper = np.maximum(y_true_f64 - upper, 0)
        interval_score = interval_width + (2.0 / alpha) * (penalty_lower + penalty_upper)
        is_mean_per_item = np.nanmean(interval_score, axis=1)  # (N,)

        # ── Seasonal scales for MASE + MSIS ──
        # Use nanmean to handle missing values in past data (e.g., *_with_missing datasets)
        scales = np.empty(N)
        for i in range(N):
            past = y_past[i]
            if len(past) <= seasonal_period:
                if len(past) > 1:
                    scales[i] = float(np.nanmean(np.abs(np.diff(past))))
                else:
                    scales[i] = np.nan
            else:
                diffs = past[seasonal_period:] - past[:-seasonal_period]
                scales[i] = float(np.nanmean(np.abs(diffs)))

        valid_scale = (scales > 0) & ~np.isnan(scales)

        # MASE
        mase_vals = np.full(N, np.nan)
        if valid_scale.any():
            mase_vals[valid_scale] = mae_per_item[valid_scale] / scales[valid_scale]
        with np.errstate(all="ignore"):
            mase = float(np.nanmean(mase_vals))

        # MSIS
        msis_vals = np.full(N, np.nan)
        if valid_scale.any():
            msis_vals[valid_scale] = is_mean_per_item[valid_scale] / scales[valid_scale]
        with np.errstate(all="ignore"):
            msis = float(np.nanmean(msis_vals))

        # MSE[mean] = MSE[0.5] for quantile models (QuantileForecast.mean → median)
        mse_mean = mse

        return {
            "mean_weighted_sum_quantile_loss": crps,
            "MASE[0.5]": mase,
            "sMAPE[0.5]": smape,
            "MAPE[0.5]": mape,
            "MSE[0.5]": mse,
            "MAE[0.5]": mae,
            "RMSE[0.5]": rmse,
            "NRMSE[0.5]": nrmse,
            "ND[0.5]": nd,
            "MSIS": msis,
            "MSE[mean]": mse_mean,
        }

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
