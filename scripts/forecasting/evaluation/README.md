# Evaluation Framework

A modular, production-ready evaluation framework for time series foundation models.
Supports multiple benchmark suites, local-first data loading, and comprehensive result reporting.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Supported Benchmarks](#supported-benchmarks)
- [Metrics](#metrics)
- [Step-by-Step Guide](#step-by-step-guide)
  - [1. Prepare Datasets](#1-prepare-datasets)
  - [2. Verify Data Availability](#2-verify-data-availability-dry-run)
  - [3. Run Evaluation](#3-run-evaluation)
  - [4. View Results](#4-view-results)
- [CLI Reference](#cli-reference)
- [Example Output](#example-output)
- [Reference Performance](#reference-performance)
- [Architecture](#architecture)
- [Output Structure](#output-structure)

---

## Quick Start

```bash
cd scripts/forecasting/evaluation

# Step 1: Verify data and model availability
python run_benchmark.py \
    --model-path /path/to/chronos-2 \
    --benchmarks lite \
    --datasets-root /path/to/datasets/ \
    --dry-run

# Step 2: Run evaluation
python run_benchmark.py \
    --model-path /path/to/chronos-2 \
    --benchmarks lite \
    --datasets-root /path/to/datasets/ \
    --output-dir results/experiments/ \
    --device cuda --torch-dtype bfloat16
```

---

## Supported Benchmarks

| Benchmark | Datasets | Description | Est. Time (A100) | Est. Time (CPU) |
|-----------|----------|-------------|-------------------|-----------------|
| `lite` | 5 | Quick validation (diverse frequency mix) | ~3 min | ~40 min |
| `extended` | 15 | Thorough validation (frequency + domain diversity) | ~15 min | ~3 hr |
| `chronos_i` | 15 | Chronos Benchmark I — in-domain datasets | ~30 min | ~5 hr |
| `chronos_ii` | 27 | Chronos Benchmark II — zero-shot datasets | ~60 min | ~10 hr |
| `gift_eval` | ~98 | GIFT-Eval (23 datasets, multi-config) | ~2 hr | N/A |
| `fev_bench` | 100 | fev-bench (task-level with bootstrap CI) | ~3 hr | N/A |

### Benchmark Details

#### `lite` — Quick Validation (5 datasets)

Designed for rapid sanity checks during training. Covers hourly, daily, and monthly frequencies across different domains.

| Dataset | Series | Horizon (H) | Frequency | Domain |
|---------|--------|-------------|-----------|--------|
| m4_hourly | 414 | 48 | Hourly | Competition |
| m4_monthly | 48,000 | 18 | Monthly | Competition |
| monash_weather | 3,010 | 30 | Daily | Weather |
| nn5 | 111 | 56 | Daily | Finance (ATM) |
| exchange_rate | 8 | 30 | Business-daily | Finance (FX) |

#### `extended` — Thorough Validation (15 datasets)

Includes all `lite` datasets plus 10 additional datasets covering transport, healthcare, energy, retail, tourism, and macro-economics.

#### `chronos_i` / `chronos_ii` — Chronos Paper Benchmarks

The official benchmark suites from the Chronos papers:
- **Chronos I (in-domain)**: 15 datasets that were part of the training corpus. Measures how well the model fits seen data distributions.
- **Chronos II (zero-shot)**: 27 datasets NOT in the training corpus. Measures generalization to unseen domains and frequencies.

Both use **geometric mean of relative scores vs. Seasonal Naive** for aggregation (lower = better; <1.0 means better than baseline).

---

## Metrics

### Primary Metrics

| Metric | Full Name | Type | Description |
|--------|-----------|------|-------------|
| **WQL** | Weighted Quantile Loss | Probabilistic | Measures the quality of quantile predictions across 9 levels (0.1–0.9). Lower = better. |
| **MASE** | Mean Absolute Scaled Error | Point | Measures point forecast accuracy scaled by the seasonal naive baseline. MASE < 1.0 means better than seasonal naive. |

### Additional Metrics (benchmark-specific)

| Metric | Full Name | Used By | Description |
|--------|-----------|---------|-------------|
| **SQL** | Scaled Quantile Loss | fev-bench | Quantile loss scaled by per-series seasonal naive error |
| **CRPS** | Continuous Ranked Probability Score | GIFT-Eval | Approximated from quantile predictions |
| **MSE** | Mean Squared Error | LTSF | Standard squared error metric |
| **MAE** | Mean Absolute Error | LTSF | Standard absolute error metric |
| **sMAPE** | Symmetric Mean Absolute Percentage Error | GIFT-Eval | Percentage-based accuracy metric |

### How WQL Works

WQL (Weighted Quantile Loss) evaluates the full predictive distribution by computing the pinball loss at each quantile level, then averaging:

```
WQL = (2 / Q) * sum_q [ sum_i,t  rho_q(y_it - y_hat_it^q) ] / sum_i,t |y_it|
```

Where `rho_q` is the quantile loss (pinball loss) function. The 9 quantile levels used are `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`.

### How MASE Works

MASE (Mean Absolute Scaled Error) scales the forecast error by the in-sample seasonal naive error:

```
MASE = mean|y_t - y_hat_t| / mean|y_t - y_{t-m}|
```

Where `m` is the seasonal period (e.g., 24 for hourly, 7 for daily, 12 for monthly). A MASE value of 1.0 means the model performs equally to the seasonal naive baseline.

### Aggregation Methods

| Method | Used By | Description |
|--------|---------|-------------|
| **Geometric Mean** | Chronos I/II | `gmean(model_metric / baseline_metric)` per dataset. Ratio < 1.0 = model is better. |
| **Simple Average** | Lite, Extended | `mean(metric)` across datasets |
| **Win Rate** | fev-bench | Fraction of tasks where model beats baseline (0.5 for ties) |
| **Skill Score** | fev-bench | `1 - gmean(clip(model/baseline, [0.01, 100]))`. Positive = better. |

---

## Step-by-Step Guide

### 1. Prepare Datasets

Download datasets to a local directory. All datasets are loaded from local disk — **no internet connection is required at evaluation time**.

```bash
cd scripts/forecasting/evaluation

# Download datasets for lite/extended benchmarks
python download_eval_datasets.py \
    --config configs/lite-benchmark.yaml \
    --output-dir /path/to/datasets/ \
    --dry-run   # Preview first

python download_eval_datasets.py \
    --config configs/lite-benchmark.yaml \
    --output-dir /path/to/datasets/

# Download for full Chronos benchmarks
python download_eval_datasets.py \
    --config configs/zero-shot.yaml configs/in-domain.yaml \
    --output-dir /path/to/datasets/
```

The datasets will be stored in Arrow IPC format:
```
/path/to/datasets/
├── m4_hourly/
│   └── data-00000-of-00001.arrow
├── m4_monthly/
│   └── data-00000-of-00001.arrow
├── monash_weather/
│   ├── data-00000-of-00002.arrow
│   └── data-00001-of-00002.arrow
├── nn5/
│   └── data-00000-of-00001.arrow
└── exchange_rate/
    └── data-00000-of-00001.arrow
```

### 2. Verify Data Availability (Dry Run)

Before running evaluation, verify that all datasets and the model are accessible:

```bash
python run_benchmark.py \
    --model-path /path/to/chronos-2 \
    --benchmarks lite \
    --datasets-root /path/to/datasets/ \
    --dry-run
```

Expected output:
```
========================================================================
  DRY RUN — Checking data and model availability
========================================================================

  Model path: /path/to/chronos-2 [EXISTS]
    Config files: ['config.json']

  --- Benchmark: lite ---
    m4_hourly: [FOUND] 1 data file(s), 5.7 MB
    m4_monthly: [FOUND] 1 data file(s), 173.0 MB
    monash_weather: [FOUND] 2 data file(s), 656.7 MB
    nn5: [FOUND] 1 data file(s), 1.0 MB
    exchange_rate: [FOUND] 1 data file(s), 0.7 MB

  --- Environment ---
    Python: 3.10.12
    PyTorch: 2.6.0
    CUDA: True
    PyArrow: 16.1.0
    Datasets: 2.17.1
    GluonTS: 0.15.1

  Dry run complete. No evaluation was performed.
========================================================================
```

### 3. Run Evaluation

#### Option A: GPU Evaluation (Recommended)

```bash
python run_benchmark.py \
    --model-path /path/to/chronos-2 \
    --benchmarks lite \
    --datasets-root /path/to/datasets/ \
    --output-dir results/experiments/ \
    --device cuda \
    --torch-dtype bfloat16 \
    --batch-size 32
```

#### Option B: CPU Evaluation (No GPU Required)

```bash
python run_benchmark.py \
    --model-path /path/to/chronos-2 \
    --benchmarks lite \
    --datasets-root /path/to/datasets/ \
    --output-dir results/experiments/ \
    --device cpu \
    --torch-dtype float32
```

> **Note**: When `--device cuda` is specified but CUDA is unavailable, the framework automatically falls back to CPU with a warning.

#### Option C: Multiple Benchmarks

```bash
python run_benchmark.py \
    --model-path /path/to/chronos-2 \
    --benchmarks chronos_i chronos_ii \
    --datasets-root /path/to/datasets/ \
    --output-dir results/experiments/ \
    --device cuda --torch-dtype bfloat16
```

#### Option D: Chronos-Bolt Models

```bash
python run_benchmark.py \
    --model-path /path/to/chronos-bolt-base \
    --benchmarks lite \
    --datasets-root /path/to/datasets/ \
    --device cuda --torch-dtype bfloat16
```

The framework automatically detects the model type (Chronos-2 vs. Chronos-Bolt) from the model path.

### 4. View Results

After evaluation completes, results are stored in the experiment directory:

```bash
# View the markdown report
cat results/experiments/exp_20260216_124836/report.md

# View per-dataset results
cat results/experiments/exp_20260216_124836/chronos_lite.csv

# View summary metrics
cat results/experiments/exp_20260216_124836/chronos_lite_summary.json

# Compare multiple models
python compare_models.py \
    --results-dirs results/experiments/exp_A results/experiments/exp_B \
    --model-names "Chronos-2" "Our Model" \
    --format markdown
```

---

## CLI Reference

```
python run_benchmark.py [OPTIONS]

Required:
  --model-path PATH        Path to model checkpoint or HuggingFace model ID
  --benchmarks NAME [...]  Benchmark(s) to run:
                           chronos_i, chronos_ii, lite, extended, gift_eval, fev_bench

Optional:
  --output-dir DIR         Base directory for results (default: results/experiments/)
  --experiment-name NAME   Custom experiment identifier (default: auto-timestamp)
  --device DEVICE          cuda, cuda:0, cpu (default: cuda)
  --torch-dtype DTYPE      float32, bfloat16 (default: float32)
  --batch-size N           Inference batch size (default: 32)
  --datasets-root DIR      Root directory for local datasets (enables local-only mode)
  --dry-run                Check data/model availability without running evaluation
  --compare-with DIR [..] Result directories to compare against
```

---

## Example Output

### Console Output (Lite Benchmark, CPU)

```
========================================================================
  TSM-Trainer Benchmark Evaluation
========================================================================
  Experiment : exp_20260216_124836
  Model      : /path/to/chronos-2
  Benchmarks : lite
  Device     : cpu (float32)
  Batch size : 32
  Data root  : /path/to/datasets/
  Output     : results/experiments/exp_20260216_124836
========================================================================

Loading Chronos-2 model: /path/to/chronos-2 (device=cpu)
Model loaded in 19.9s

========================================================================
  [1/1] BENCHMARK: lite
========================================================================
Evaluating benchmark: lite-benchmark (5 datasets, model=chronos-2)
  [1/5] m4_hourly:      WQL=0.0265, MASE=0.8106 (17.6s)
  [2/5] m4_monthly:     WQL=0.0926, MASE=0.9224 (1079.0s)
  [3/5] monash_weather: WQL=0.1253, MASE=0.7741 (1196.3s)
  [4/5] nn5:            WQL=0.1488, MASE=0.5768 (4.2s)
  [5/5] exchange_rate:  WQL=0.0121, MASE=1.8520 (2.7s)

Results saved: results/experiments/exp_20260216_124836/chronos_lite.csv
Summary saved: results/experiments/exp_20260216_124836/chronos_lite_summary.json

  lite completed in 2299.8s
    avg_mase: 0.9872
    avg_wql: 0.0811
    n_datasets: 5

========================================================================
  EVALUATION COMPLETE
  Benchmarks : 1 passed, 0 failed
  Total time : 2322.4s
  Results    : results/experiments/exp_20260216_124836
  Report     : results/experiments/exp_20260216_124836/report.md
========================================================================
```

### Per-Dataset CSV (`chronos_lite.csv`)

```csv
dataset,model,MASE,WQL
exchange_rate,chronos-2,1.8520,0.0121
m4_hourly,chronos-2,0.8106,0.0265
m4_monthly,chronos-2,0.9224,0.0926
monash_weather,chronos-2,0.7741,0.1253
nn5,chronos-2,0.5768,0.1488
```

### Summary JSON (`chronos_lite_summary.json`)

```json
{
  "avg_wql": 0.0811,
  "avg_mase": 0.9872,
  "n_datasets": 5
}
```

---

## Reference Performance

Published and verified baseline model performance for comparison.

### Lite Benchmark — Chronos-2 (Verified)

Results from evaluating `amazon/chronos-2` on the lite benchmark.
Our framework produces results that **exactly match** the reference:

| Dataset | WQL | MASE |
|---------|-----|------|
| m4_hourly | 0.0265 | 0.8105 |
| m4_monthly | 0.0926 | 0.9224 |
| monash_weather | 0.1253 | 0.7741 |
| nn5 | 0.1488 | 0.5768 |
| exchange_rate | 0.0121 | 1.8509 |
| **Average** | **0.0811** | **0.9870** |

### Chronos Benchmark I (In-domain) — Aggregated Relative Scores

Geometric mean of per-dataset relative scores vs. Seasonal Naive baseline.
**Lower is better** (< 1.0 means outperforming the baseline).

| Model | Params | Rel. WQL | Rel. MASE |
|-------|--------|----------|-----------|
| Chronos-Bolt Base | 205M | **0.534** | **0.680** |
| Chronos-Bolt Small | 48M | 0.544 | 0.703 |
| Chronos-T5 Large | 710M | 0.560 | 0.694 |
| Chronos-Bolt Mini | 9M | 0.565 | 0.727 |
| Chronos-Bolt Tiny | 3M | 0.573 | 0.740 |
| Chronos-T5 Base | 200M | 0.579 | 0.701 |
| Chronos-T5 Mini | 20M | 0.597 | 0.725 |
| Chronos-T5 Small | 46M | 0.609 | 0.730 |
| Chronos-T5 Tiny | 8M | 0.629 | 0.765 |

### Chronos Benchmark II (Zero-shot) — Aggregated Relative Scores

Geometric mean of per-dataset relative scores vs. Seasonal Naive baseline.
**Lower is better** (< 1.0 means outperforming the baseline).

| Model | Params | Rel. WQL | Rel. MASE |
|-------|--------|----------|-----------|
| Chronos-Bolt Base | 205M | **0.624** | **0.791** |
| Chronos-Bolt Small | 48M | 0.636 | 0.819 |
| Chronos-T5 Base | 200M | 0.642 | 0.816 |
| Chronos-Bolt Mini | 9M | 0.644 | 0.822 |
| Chronos-T5 Large | 710M | 0.650 | 0.821 |
| Chronos-T5 Small | 46M | 0.665 | 0.830 |
| Chronos-Bolt Tiny | 3M | 0.668 | 0.845 |
| Chronos-T5 Mini | 20M | 0.689 | 0.841 |
| Chronos-T5 Tiny | 8M | 0.711 | 0.870 |

> **Source**: Results extracted from the Chronos/Chronos-2 papers and verified against the official evaluation scripts. Reference CSV files are available in the `results/` directory.

---

## Architecture

```
scripts/forecasting/evaluation/
│
├── run_benchmark.py              # CLI entry point (benchmark runner)
├── compare_models.py             # Multi-model comparison tool
├── README.md                     # This file
│
├── engine/                       # Core evaluation infrastructure
│   ├── metrics.py                #   MetricRegistry (WQL, MASE, SQL, CRPS, ...)
│   ├── forecaster.py             #   BaseForecaster ABC + model adapters
│   │                             #     ├── Chronos2Forecaster
│   │                             #     ├── ChronosBoltForecaster
│   │                             #     └── TrainingModelForecaster
│   ├── evaluator.py              #   Unified evaluation loop
│   │                             #     dataset → forecast → metrics
│   └── aggregator.py             #   Result aggregation
│                                 #     gmean, bootstrap CI, win rate, skill score
│
├── benchmarks/                   # Pluggable benchmark adapters
│   ├── base.py                   #   BenchmarkAdapter ABC (interface)
│   ├── chronos_bench.py          #   Chronos I/II, Lite, Extended
│   ├── gift_eval.py              #   GIFT-Eval adapter
│   └── fev_bench.py              #   fev-bench adapter
│
├── configs/                      # Benchmark configuration (YAML)
│   ├── lite-benchmark.yaml       #   5 datasets for quick validation
│   ├── extended-benchmark.yaml   #   15 datasets for thorough validation
│   ├── in-domain.yaml            #   Chronos Benchmark I (15 datasets)
│   └── zero-shot.yaml            #   Chronos Benchmark II (27 datasets)
│
├── results/                      # Reference baseline results (CSV)
│   ├── seasonal-naive-*.csv      #   Seasonal Naive baseline scores
│   ├── chronos-t5-*-*.csv        #   Chronos-T5 model scores
│   ├── chronos-bolt-*-*.csv      #   Chronos-Bolt model scores
│   └── chronos-2-lite-*.csv      #   Chronos-2 lite benchmark reference
│
├── download_eval_datasets.py     # Dataset download (HuggingFace → local)
├── download_gift_eval.py         # GIFT-Eval data download
└── download_fev_bench.py         # fev-bench data download
```

### Design Principles

- **Local-first data loading**: All datasets loaded from local disk. If a dataset is not found locally, a clear error is raised — no silent downloads.
- **Arrow IPC fallback**: Handles `datasets` library version mismatches gracefully via direct PyArrow reader.
- **Shape contract**: `BaseForecaster.predict_quantiles()` returns `(N, Q, H)`. Assertions enforce this at every stage.
- **Pluggable adapters**: Each benchmark is an independent adapter implementing `BenchmarkAdapter`. Adding a new benchmark = one new file.

---

## Output Structure

Each evaluation creates a timestamped experiment directory:

```
results/experiments/
└── exp_20260216_124836/
    ├── config.json                    # Experiment config (model, device, env)
    ├── summary.json                   # Overall results + timing
    ├── report.md                      # Human-readable report (6 sections)
    │                                  #   1. Executive Summary
    │                                  #   2. Per-Benchmark Results
    │                                  #   3. Per-Dataset Results
    │                                  #   4. Environment Info
    │                                  #   5. Timing Breakdown
    │                                  #   6. Reproduction Command
    ├── chronos_lite.csv               # Per-dataset results (CSV)
    └── chronos_lite_summary.json      # Aggregated summary (JSON)
```

For multiple benchmarks:
```
results/experiments/
└── exp_20260216_150000/
    ├── config.json
    ├── summary.json
    ├── report.md
    ├── chronos_bench_in_domain.csv
    ├── chronos_bench_in_domain_summary.json
    ├── chronos_bench_zero_shot.csv
    └── chronos_bench_zero_shot_summary.json
```

---

## Estimated Runtime

| Benchmark | A100 (bfloat16) | A100 (float32) | CPU (72 cores) |
|-----------|-----------------|----------------|----------------|
| `lite` (5 ds) | ~3 min | ~5 min | ~40 min |
| `extended` (15 ds) | ~15 min | ~25 min | ~3 hr |
| `chronos_i` (15 ds) | ~30 min | ~50 min | ~5 hr |
| `chronos_ii` (27 ds) | ~60 min | ~90 min | ~10 hr |

> **Tip**: Use `--torch-dtype bfloat16` on GPU for faster inference with minimal accuracy impact.
> The lite benchmark produces identical results with bfloat16 and float32 (verified).
