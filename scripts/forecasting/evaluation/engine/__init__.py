# Evaluation Engine — Core metrics, forecasting interface, and aggregation
#
# This module provides model-agnostic evaluation infrastructure for
# time series forecasting benchmarks. It separates concerns into:
#
#   metrics.py      — Metric computation (WQL, MASE, SQL, MSE, MAE, etc.)
#   forecaster.py   — Abstract forecasting interface (model-agnostic)
#   evaluator.py    — Unified evaluation loop (dataset → forecasts → metrics)
#   aggregator.py   — Result aggregation (gmean, bootstrap CI, win rate, skill score)
#   distributed.py  — Distributed evaluation utilities (FSDP-compatible)

from .metrics import MetricRegistry
from .forecaster import (
    BaseForecaster,
    Chronos2Forecaster,
    ChronosBoltForecaster,
    TrainingModelForecaster,
)
from .evaluator import Evaluator, validate_datasets, validate_config
from .aggregator import Aggregator
from .distributed import (
    get_dist_info,
    broadcast_error,
    generate_forecasts_distributed,
    gather_forecasts_to_rank0,
)

__all__ = [
    "MetricRegistry",
    "BaseForecaster",
    "Chronos2Forecaster",
    "ChronosBoltForecaster",
    "TrainingModelForecaster",
    "Evaluator",
    "Aggregator",
    "validate_datasets",
    "validate_config",
    "get_dist_info",
    "broadcast_error",
    "generate_forecasts_distributed",
    "gather_forecasts_to_rank0",
]
