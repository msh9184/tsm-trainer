# Evaluation Engine — Core metrics, forecasting interface, and aggregation
#
# This module provides model-agnostic evaluation infrastructure for
# time series forecasting benchmarks. It separates concerns into:
#
#   metrics.py     — Metric computation (WQL, MASE, SQL, MSE, MAE, etc.)
#   forecaster.py  — Abstract forecasting interface (model-agnostic)
#   evaluator.py   — Unified evaluation loop (dataset → forecasts → metrics)
#   aggregator.py  — Result aggregation (gmean, bootstrap CI, win rate, skill score)

from .metrics import MetricRegistry
from .forecaster import BaseForecaster, Chronos2Forecaster
from .evaluator import Evaluator
from .aggregator import Aggregator

__all__ = [
    "MetricRegistry",
    "BaseForecaster",
    "Chronos2Forecaster",
    "Evaluator",
    "Aggregator",
]
