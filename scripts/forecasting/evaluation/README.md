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
- [Training-Time Validation](#training-time-validation)
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
    --benchmarks chronos_lite \
    --datasets-root /group-volume/ts-dataset/benchmarks/chronos/ \
    --dry-run

# Step 2: Run evaluation
python run_benchmark.py \
    --model-path /path/to/chronos-2 \
    --benchmarks chronos_lite \
    --datasets-root /group-volume/ts-dataset/benchmarks/chronos/ \
    --output-dir results/experiments/ \
    --device cuda --torch-dtype bfloat16
```

---

## Supported Benchmarks

### Chronos Benchmarks (Paper-Aligned)

| Benchmark | Config | Datasets | Description | Est. Time (A100) |
|-----------|--------|----------|-------------|-------------------|
| `chronos_lite` | `chronos-lite.yaml` | 5 | Quick validation (diverse frequency mix) | ~3 min |
| `chronos_extended` | `chronos-extended.yaml` | 15 | Thorough validation (frequency + domain diversity) | ~15 min |
| `chronos_i` | `chronos-i.yaml` | 15 | Chronos Benchmark I — in-domain datasets | ~30 min |
| `chronos_ii` | `chronos-ii.yaml` | 27 | Chronos Benchmark II — zero-shot datasets | ~60 min |
| `chronos_full` | `chronos-full.yaml` | 42 | All Chronos datasets (I + II combined) | ~90 min |

### External Benchmarks (Library-Based)

| Benchmark | Config | Tasks | Description | Est. Time (A100) |
|-----------|--------|-------|-------------|-------------------|
| `gift_eval` | `gift-eval.yaml` | ~98 | GIFT-Eval (23 datasets, multi-config) | ~2 hr |
| `fev_bench` | `fev-bench.yaml` | 100 | fev-bench (task-level with bootstrap CI) | ~3 hr |

### Backward-Compatible Aliases

| Alias | Maps to |
|-------|---------|
| `lite` | `chronos_lite` |
| `extended` | `chronos_extended` |

### Benchmark Details

#### `chronos_lite` — Quick Validation (5 datasets)

Designed for rapid sanity checks during training. Covers hourly, daily, and monthly frequencies across different domains.

| Dataset | Series | Horizon (H) | Frequency | Domain |
|---------|--------|-------------|-----------|--------|
| m4_hourly | 414 | 48 | Hourly | Competition |
| m4_monthly | 48,000 | 18 | Monthly | Competition |
| monash_weather | 3,010 | 30 | Daily | Weather |
| nn5 | 111 | 56 | Daily | Finance (ATM) |
| exchange_rate | 8 | 30 | Business-daily | Finance (FX) |

#### `chronos_extended` — Thorough Validation (15 datasets)

Includes all `chronos_lite` datasets plus 10 additional datasets covering transport, healthcare, energy, retail, tourism, and macro-economics.

#### `chronos_i` — Chronos Benchmark I (In-Domain)

The official in-domain benchmark from the Chronos paper:
- **15 datasets** that were part of the training corpus.
- Measures how well the model fits seen data distributions.
- Aggregation: **Geometric mean of relative scores vs. Seasonal Naive** (lower = better; <1.0 means better than baseline).

#### `chronos_ii` — Chronos Benchmark II (Zero-Shot)

The official zero-shot benchmark from the Chronos-2 paper:
- **27 datasets** NOT in the training corpus.
- Measures generalization to unseen domains and frequencies.
- Aggregation: **Geometric mean of relative scores vs. Seasonal Naive**.

#### `chronos_full` — All Chronos Datasets

Combined: Chronos I (15) + Chronos II (27) = **42 unique datasets**.
Use for final model evaluation before release.

#### `gift_eval` — GIFT-Eval

Comprehensive benchmark from Salesforce AI Research:
- 23 base datasets, ~98 task configurations (short/medium/long horizons).
- Requires: `pip install gift-eval`
- Local data: `/group-volume/ts-dataset/benchmarks/gift_eval/`

#### `fev_bench` — fev-bench

Comprehensive forecasting evaluation benchmark:
- 100 tasks across 96 unique datasets, including 46 with covariates.
- Requires: `pip install fev`
- Local data: `/group-volume/ts-dataset/benchmarks/fev_bench/`

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
| **Simple Average** | Chronos Lite/Extended | `mean(metric)` across datasets |
| **Win Rate** | fev-bench | Fraction of tasks where model beats baseline (0.5 for ties) |
| **Skill Score** | fev-bench | `1 - gmean(clip(model/baseline, [0.01, 100]))`. Positive = better. |

---

## Step-by-Step Guide

### 1. Prepare Datasets

All benchmark datasets are pre-downloaded to local disk on the GPU server:

```
/group-volume/ts-dataset/benchmarks/
├── chronos/       # 42 datasets (Chronos I + II)
├── fev_bench/     # ~48 datasets
└── gift_eval/     # ~30 datasets
```

For a fresh setup, download datasets to a local directory:

```bash
cd scripts/forecasting/evaluation

# Download for Chronos benchmarks
python download_eval_datasets.py \
    --config configs/chronos-lite.yaml \
    --output-dir /path/to/datasets/

python download_eval_datasets.py \
    --config configs/chronos-i.yaml configs/chronos-ii.yaml \
    --output-dir /path/to/datasets/
```

The datasets will be stored in Arrow IPC format:
```
/path/to/datasets/
├── m4_hourly/
│   └── data-00000-of-00001.arrow
├── m4_monthly/
│   └── data-00000-of-00001.arrow
└── ...
```

### 2. Verify Data Availability (Dry Run)

Before running evaluation, verify that all datasets and the model are accessible:

```bash
python run_benchmark.py \
    --model-path /path/to/chronos-2 \
    --benchmarks chronos_lite \
    --datasets-root /group-volume/ts-dataset/benchmarks/chronos/ \
    --dry-run
```

### 3. Run Evaluation

#### Option A: GPU Evaluation (Recommended)

```bash
python run_benchmark.py \
    --model-path /path/to/chronos-2 \
    --benchmarks chronos_lite \
    --datasets-root /group-volume/ts-dataset/benchmarks/chronos/ \
    --output-dir results/experiments/ \
    --device cuda \
    --torch-dtype bfloat16 \
    --batch-size 32
```

#### Option B: Multiple Benchmarks

```bash
python run_benchmark.py \
    --model-path /path/to/chronos-2 \
    --benchmarks chronos_i chronos_ii \
    --datasets-root /group-volume/ts-dataset/benchmarks/chronos/ \
    --output-dir results/experiments/ \
    --device cuda --torch-dtype bfloat16
```

#### Option C: Full Evaluation Suite

```bash
python run_benchmark.py \
    --model-path /path/to/chronos-2 \
    --benchmarks chronos_full \
    --datasets-root /group-volume/ts-dataset/benchmarks/chronos/ \
    --device cuda --torch-dtype bfloat16
```

### 4. View Results

```bash
# View the markdown report
cat results/experiments/exp_20260216_124836/report.md

# View per-dataset results
cat results/experiments/exp_20260216_124836/chronos_lite.csv

# Compare multiple models
python compare_models.py \
    --results-dirs results/experiments/exp_A results/experiments/exp_B \
    --model-names "Chronos-2" "Our Model" \
    --format markdown
```

---

## Training-Time Validation

During training, benchmark evaluation runs periodically to monitor model quality.

### Configuration (in training YAML)

```yaml
# Benchmark config — choose from evaluation/configs/:
#   chronos-lite.yaml      — 5 datasets, ~3 min (quick validation)
#   chronos-extended.yaml  — 15 datasets, ~15 min (thorough validation)
#   chronos-i.yaml         — 15 datasets, ~30 min (Chronos Bench I)
#   chronos-ii.yaml        — 27 datasets, ~60 min (Chronos Bench II)
#   chronos-full.yaml      — 42 datasets, ~90 min (all Chronos datasets)
benchmark_config: "configs/chronos-lite.yaml"
benchmark_eval_steps: 200
benchmark_top_k_checkpoints: 3
benchmark_batch_size: 32

# Checkpoint selection metric: "wql" | "mase" | "composite"
benchmark_checkpoint_metric: "composite"
benchmark_composite_weights:
  wql: 0.6
  mase: 0.4

# Local datasets root (Chronos benchmarks)
benchmark_datasets_root: "/group-volume/ts-dataset/benchmarks/chronos"

# Evaluation timeout (seconds, 0 = unlimited)
benchmark_eval_timeout: 600
```

### Path Resolution

The training script resolves benchmark config paths in order:
1. `scripts/forecasting/training/{benchmark_config}` (training dir)
2. `scripts/forecasting/evaluation/{benchmark_config}` (evaluation dir)
3. Project root

### Available Metrics for Checkpoint Selection

| Metric | Description | Best For |
|--------|-------------|----------|
| `wql` | Weighted Quantile Loss | Probabilistic forecast quality |
| `mase` | Mean Absolute Scaled Error | Point forecast accuracy |
| `composite` | Weighted WQL + MASE | Balanced quality signal |

---

## CLI Reference

```
python run_benchmark.py [OPTIONS]

Required:
  --model-path PATH        Path to model checkpoint or HuggingFace model ID
  --benchmarks NAME [...]  Benchmark(s) to run:
                           chronos_i, chronos_ii, chronos_lite, chronos_extended,
                           chronos_full, lite, extended, gift_eval, fev_bench

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

### Console Output (Chronos Lite Benchmark)

```
========================================================================
  TSM-Trainer Benchmark Evaluation
========================================================================
  Experiment : exp_20260216_124836
  Model      : /path/to/chronos-2
  Benchmarks : chronos_lite
  Device     : cuda (bfloat16)
  Batch size : 32
  Data root  : /group-volume/ts-dataset/benchmarks/chronos/
  Output     : results/experiments/exp_20260216_124836
========================================================================

Loading Chronos-2 model: /path/to/chronos-2 (device=cuda)
Model loaded in 2.3s

========================================================================
  [1/1] BENCHMARK: chronos_lite
========================================================================
Evaluating benchmark: chronos-lite (5 datasets, model=chronos-2)
  [1/5] m4_hourly:      WQL=0.0265, MASE=0.8106 (3.2s)
  [2/5] m4_monthly:     WQL=0.0926, MASE=0.9224 (45.1s)
  [3/5] monash_weather: WQL=0.1253, MASE=0.7741 (38.7s)
  [4/5] nn5:            WQL=0.1488, MASE=0.5768 (1.2s)
  [5/5] exchange_rate:  WQL=0.0121, MASE=1.8520 (0.8s)

  chronos_lite completed in 89.0s
    avg_mase: 0.9872
    avg_wql: 0.0811
    n_datasets: 5
```

---

## Reference Performance

Published and verified baseline model performance for comparison.

### Chronos Lite Benchmark — Chronos-2 (Verified)

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
│   ├── chronos_bench.py          #   Chronos I/II, Lite, Extended, Full
│   ├── gift_eval.py              #   GIFT-Eval adapter
│   └── fev_bench.py              #   fev-bench adapter
│
├── configs/                      # Benchmark configuration (YAML)
│   ├── chronos-lite.yaml         #   5 datasets — quick validation
│   ├── chronos-extended.yaml     #   15 datasets — thorough validation
│   ├── chronos-i.yaml            #   15 datasets — Chronos Bench I (in-domain)
│   ├── chronos-ii.yaml           #   27 datasets — Chronos Bench II (zero-shot)
│   ├── chronos-full.yaml         #   42 datasets — all Chronos datasets
│   ├── gift-eval.yaml            #   GIFT-Eval reference config
│   └── fev-bench.yaml            #   fev-bench reference config
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

- **Local-first data loading**: All datasets loaded from local disk via `datasets_root`. No network access required at evaluation time.
- **Paper-aligned benchmarks**: Config names match paper benchmark names (chronos_i, chronos_ii, gift_eval, fev_bench).
- **Arrow IPC fallback**: Handles `datasets` library version mismatches gracefully via direct PyArrow reader.
- **Shape contract**: `BaseForecaster.predict_quantiles()` returns `(N, Q, H)`. Assertions enforce this at every stage.
- **Pluggable adapters**: Each benchmark is an independent adapter implementing `BenchmarkAdapter`. Adding a new benchmark = one new file.
- **hf_repo optional**: Dataset configs no longer require `hf_repo`; `datasets_root` resolves all local paths.

---

## Output Structure

Each evaluation creates a timestamped experiment directory:

```
results/experiments/
└── exp_20260216_124836/
    ├── config.json                    # Experiment config (model, device, env)
    ├── summary.json                   # Overall results + timing
    ├── report.md                      # Human-readable report
    ├── chronos_lite.csv               # Per-dataset results (CSV)
    └── chronos_lite_summary.json      # Aggregated summary (JSON)
```

---

## Estimated Runtime

| Benchmark | A100 (bfloat16) | A100 (float32) | CPU (72 cores) |
|-----------|-----------------|----------------|----------------|
| `chronos_lite` (5 ds) | ~3 min | ~5 min | ~40 min |
| `chronos_extended` (15 ds) | ~15 min | ~25 min | ~3 hr |
| `chronos_i` (15 ds) | ~30 min | ~50 min | ~5 hr |
| `chronos_ii` (27 ds) | ~60 min | ~90 min | ~10 hr |
| `chronos_full` (42 ds) | ~90 min | ~120 min | ~12 hr |

> **Tip**: Use `--torch-dtype bfloat16` on GPU for faster inference with minimal accuracy impact.
