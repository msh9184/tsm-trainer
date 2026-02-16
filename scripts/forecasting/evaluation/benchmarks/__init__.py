# Benchmark Adapters — Pluggable benchmark evaluation protocols
#
# Each adapter encapsulates the specific data loading, evaluation protocol,
# and aggregation logic for a benchmark suite:
#
#   base.py          — BenchmarkAdapter abstract base class
#   chronos_bench.py — Chronos Benchmark I (in-domain) & II (zero-shot)
#   gift_eval.py     — GIFT-Eval (~98 tasks, 23 datasets)
#   fev_bench.py     — fev-bench (100 tasks, covariates)
#   ltsf.py          — LTSF-Benchmark (ETT, Weather, Traffic, Electricity)

from .base import BenchmarkAdapter
from .chronos_bench import ChronosBenchmarkAdapter, ChronosLiteBenchmarkAdapter

__all__ = [
    "BenchmarkAdapter",
    "ChronosBenchmarkAdapter",
    "ChronosLiteBenchmarkAdapter",
]
