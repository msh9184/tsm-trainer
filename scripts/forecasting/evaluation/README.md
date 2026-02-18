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

| Benchmark | Config | Tasks | Primary Metrics | Description | Est. Time (A100) |
|-----------|--------|-------|-----------------|-------------|-------------------|
| `gift_eval` | `gift-eval.yaml` | 97 | CRPS, MASE, WQL | GIFT-Eval: 28 datasets × 3 terms | ~6-7 hr |
| `fev_bench` | `fev-bench.yaml` | 100 | SQL, Win Rate | fev-bench: 96 datasets, 46 with covariates | ~3 hr |
| `ltsf` | `ltsf.yaml` | 36 | MSE, MAE | LTSF: 9 datasets × 4 horizons | ~30 min |

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

> **Paper**: [arXiv:2403.07815](https://arxiv.org/abs/2403.07815) (Chronos, TMLR 2024)
> **Data**: `autogluon/chronos_datasets` on HuggingFace

The official in-domain benchmark from the Chronos paper. These **15 datasets** were part of the training corpus — evaluates how well the model fits seen data distributions.

**Protocol**:
- Single evaluation window per dataset (`num_rolls=1`)
- Quantile levels: {0.1, 0.2, ..., 0.9} (9 levels)
- Aggregation: **Geometric mean of relative scores vs. Seasonal Naive** (< 1.0 = beats baseline)
- Context: full series history up to the prediction window

**Dataset Catalog (~97K total series)**:

| # | Dataset | Domain | Freq | Series | H | Source |
|---|---------|--------|------|-------:|--:|--------|
| 1 | electricity_15min | Energy | 15T | 370 | 24 | UCI ML Repository |
| 2 | monash_electricity_hourly | Energy | H | 321 | 24 | Monash Archive |
| 3 | monash_electricity_weekly | Energy | W | 321 | 8 | Monash Archive |
| 4 | monash_kdd_cup_2018 | Nature | H | 270 | 48 | Monash Archive |
| 5 | m4_daily | Competition | D | 4,227 | 14 | M4 Competition |
| 6 | m4_hourly | Competition | H | 414 | 48 | M4 Competition |
| 7 | m4_monthly | Competition | M | 48,000 | 18 | M4 Competition |
| 8 | m4_weekly | Competition | W | 359 | 13 | M4 Competition |
| 9 | monash_pedestrian_counts | Transport | H | 66 | 48 | Monash Archive |
| 10 | taxi_30min | Transport | 30T | 2,428 | 48 | NYC TLC |
| 11 | uber_tlc_hourly | Transport | H | 262 | 24 | Uber TLC |
| 12 | uber_tlc_daily | Transport | D | 262 | 7 | Uber TLC |
| 13 | monash_rideshare | Transport | H | 2,340 | 24 | Monash Archive |
| 14 | monash_temperature_rain | Nature | D | 32,072 | 30 | Monash Archive |
| 15 | monash_london_smart_meters | Energy | 30T | 5,560 | 48 | Monash Archive |

**Domain breakdown**: Energy (4), Transport (5), Competition (4), Nature (2)

#### `chronos_ii` — Chronos Benchmark II (Zero-Shot)

> **Paper**: [arXiv:2510.15821](https://arxiv.org/abs/2510.15821) (Chronos-2, 2025)
> **Data**: `autogluon/chronos_datasets` + `autogluon/chronos_datasets_extra` on HuggingFace

The official zero-shot benchmark. These **27 datasets** were NOT in the training corpus — evaluates generalization to unseen domains and frequencies.

**Protocol**: Same as Benchmark I (single window, gmean vs Seasonal Naive)

**Dataset Catalog (~191K total series)**:

| # | Dataset | Domain | Freq | Series | H | Source |
|---|---------|--------|------|-------:|--:|--------|
| 1 | m4_quarterly | Competition | Q | 24,000 | 8 | M4 Competition |
| 2 | m4_yearly | Competition | Y | 23,000 | 6 | M4 Competition |
| 3 | m5 | Retail | D | 30,490 | 28 | M5 Competition |
| 4 | dominick | Retail | D | 100,014 | 8 | Dominick's Grocery |
| 5 | monash_m1_yearly | Competition | Y | 181 | 6 | Monash (M1) |
| 6 | monash_m1_quarterly | Competition | Q | 203 | 8 | Monash (M1) |
| 7 | monash_m1_monthly | Competition | M | 617 | 18 | Monash (M1) |
| 8 | monash_m3_monthly | Competition | M | 1,428 | 18 | Monash (M3) |
| 9 | monash_m3_yearly | Competition | Y | 645 | 6 | Monash (M3) |
| 10 | monash_m3_quarterly | Competition | Q | 756 | 8 | Monash (M3) |
| 11 | monash_traffic | Transport | H | 862 | 24 | Monash Archive |
| 12 | monash_australian_electricity | Energy | 30T | 5 | 48 | Monash Archive |
| 13 | ercot | Energy | H | 8 | 24 | ERCOT Texas |
| 14 | monash_weather | Nature | D | 3,010 | 30 | Monash Archive |
| 15 | ETTm | Energy | 15T | 14 | 24 | ETDataset |
| 16 | ETTh | Energy | H | 14 | 24 | ETDataset |
| 17 | exchange_rate | Finance | B | 8 | 30 | Lai et al. (2018) |
| 18 | nn5 | Finance | D | 111 | 56 | NN5 Competition |
| 19 | monash_nn5_weekly | Finance | W | 111 | 8 | Monash Archive |
| 20 | monash_covid_deaths | Healthcare | D | 266 | 30 | Monash Archive |
| 21 | monash_fred_md | Economics | M | 107 | 12 | Monash (FRED-MD) |
| 22 | monash_tourism_monthly | Tourism | M | 366 | 24 | Monash Archive |
| 23 | monash_tourism_quarterly | Tourism | Q | 427 | 8 | Monash Archive |
| 24 | monash_tourism_yearly | Tourism | Y | 518 | 4 | Monash Archive |
| 25 | monash_car_parts | Retail | M | 2,674 | 12 | Monash Archive |
| 26 | monash_hospital | Healthcare | M | 767 | 12 | Monash Archive |
| 27 | monash_cif_2016 | Banking | M | 72 | 12 | Monash Archive |

**Domain breakdown**: Competition (8), Energy (4), Transport (1), Retail (3), Finance (3), Tourism (3), Nature (1), Healthcare (2), Economics (1), Banking (1)

**Seasonal periods used for MASE scaling**:

| Frequency | Period (m) | Meaning |
|-----------|-----------|---------|
| 15T / 30T | 96 / 48 | 1 day |
| Hourly (H) | 24 | 1 day |
| Daily (D) | 7 | 1 week |
| Business-daily (B) | 5 | 1 week |
| Weekly (W) | 52 | 1 year |
| Monthly (M) | 12 | 1 year |
| Quarterly (Q) | 4 | 1 year |
| Yearly (Y/A) | 1 | Non-seasonal (Naive) |

#### `chronos_full` — All Chronos Datasets

Combined: Chronos I (15) + Chronos II (27) = **42 unique datasets** (~288K series).
Use for final model evaluation before release.

#### `gift_eval` — GIFT-Eval (Salesforce AI Research)

> **Paper**: [arXiv:2410.10393](https://arxiv.org/abs/2410.10393) (NeurIPS 2024 Datasets & Benchmarks)
> **GitHub**: [SalesforceAIResearch/gift-eval](https://github.com/SalesforceAIResearch/gift-eval)
> **Leaderboard**: [huggingface.co/spaces/Salesforce/GiftEval](https://huggingface.co/spaces/Salesforce/GiftEval)
> **Data**: `Salesforce/GiftEval` on HuggingFace Hub
> **Requires**: `pip install gift-eval` (or clone + `pip install -e .`)

Comprehensive multi-domain benchmark with 28 datasets (55 dataset/frequency combinations) evaluated across 3 forecast terms (short/medium/long), producing **97 task configurations**. The primary ranking metric is **average CRPS rank** across all 97 tasks.

| Property | Value |
|----------|-------|
| Datasets | 28 base (55 dataset/freq combos), 97 task configs |
| Domains | 7 (Econ/Finance, Energy, Healthcare, Nature, Sales, Transport, Web/CloudOps) |
| Terms | Short (1x), Medium (10x), Long (15x) of base prediction length |
| Quantiles | {0.1, 0.2, ..., 0.9} (9 levels) |
| Test Split | Last 10% of each series (`TEST_SPLIT = 0.1`) |
| Rolling Windows | Non-overlapping, max 20 (`MAX_WINDOW = 20`) |
| Metrics | 11 per task (see below) |
| Ranking | Average rank across 97 tasks (lower = better) |
| Multivariate | 8 datasets (channel-independent evaluation) |

**Evaluation Protocol Details**:

1. **Test region**: Last 10% of the minimum series length in each dataset
2. **Window count**: `windows = min(max(1, ceil(0.1 * min_length / pred_length)), 20)`. M4 datasets always use 1 window.
3. **Prediction length**: Determined by frequency × term multiplier:

| Frequency | Base H (M4) | Base H (Standard) | Short | Medium | Long |
|-----------|:-----------:|:-----------------:|------:|-------:|-----:|
| Yearly (A) | 6 | — | 6 | 60 | 90 |
| Quarterly (Q) | 8 | — | 8 | 80 | 120 |
| Monthly (M) | 18 | 12 | 12–18 | 120–180 | 180–270 |
| Weekly (W) | 13 | 8 | 8–13 | 80–130 | 120–195 |
| Daily (D) | 14 | 30 | 14–30 | 140–300 | 210–450 |
| Hourly (H) | 48 | 48 | 48 | 480 | 720 |
| Minutely (T) | — | 48 | 48 | 480 | 720 |
| Secondly (S) | — | 60 | 60 | 600 | 900 |

**11 Metrics (exact CSV column names)**:

| # | Column Name | Display | Description |
|---|-------------|---------|-------------|
| 1 | `mean_weighted_sum_quantile_loss` | CRPS | WQL over 9 quantiles (primary ranking metric) |
| 2 | `MASE[0.5]` | MASE | Mean Absolute Scaled Error (median forecast) |
| 3 | `sMAPE[0.5]` | sMAPE | Symmetric Mean Absolute Percentage Error |
| 4 | `MAPE[0.5]` | MAPE | Mean Absolute Percentage Error |
| 5 | `MSE[0.5]` | MSE | Mean Squared Error (median forecast) |
| 6 | `MAE[0.5]` | MAE | Mean Absolute Error (median forecast) |
| 7 | `RMSE[mean]` | RMSE | Root Mean Squared Error (mean forecast) |
| 8 | `NRMSE[mean]` | NRMSE | Normalized RMSE (mean forecast) |
| 9 | `ND[0.5]` | ND | Normalized Deviation (median forecast) |
| 10 | `MSIS` | MSIS | Mean Scaled Interval Score |
| 11 | `MSE[mean]` | MSE* | Mean Squared Error (mean forecast) |

**Dataset Catalog by Domain (28 datasets, 97 task configs)**:

| Domain | Dataset | #Var | Frequencies | Terms | Configs |
|--------|---------|-----:|-------------|-------|--------:|
| **Econ/Finance** | m4_yearly | 1 | A | short | 1 |
| | m4_quarterly | 1 | Q | short | 1 |
| | m4_monthly | 1 | M | short | 1 |
| | m4_weekly | 1 | W | short | 1 |
| | m4_daily | 1 | D | short | 1 |
| | m4_hourly | 1 | H | short | 1 |
| **Energy** | electricity | 1 | 15T, H, D, W | short/med/long | 8 |
| | ett1 | 7 | 15T, H, D, W | short/med/long | 8 |
| | ett2 | 7 | 15T, H, D, W | short/med/long | 8 |
| | solar | 1 | 10T, H, D, W | short/med/long | 8 |
| **Healthcare** | covid_deaths | 1 | D | short | 1 |
| | hospital | 1 | M | short | 1 |
| | us_births | 1 | D, M, W | short | 3 |
| **Nature** | jena_weather | 21 | 10T, H, D | short/med/long | 7 |
| | kdd_cup_2018 | 1 | H, D | short/med/long | 4 |
| | saugeenday | 1 | D, M, W | short | 3 |
| | temperature_rain | 1 | D | short | 1 |
| **Sales** | car_parts | 1 | M | short | 1 |
| | hierarchical_sales | 1 | D, W | short | 2 |
| | restaurant | 1 | D | short | 1 |
| **Transport** | LOOP_SEATTLE | 1 | 5T, H, D | short/med/long | 7 |
| | M_DENSE | 1 | H, D | short/med/long | 4 |
| | SZ_TAXI | 1 | 15T, H | short/med/long | 4 |
| **Web/CloudOps** | bitbrains_fast_storage | 2 | 5T, H | short/med/long | 4 |
| | bitbrains_rnd | 2 | 5T, H | short/med/long | 4 |
| | bizitobs_application | 2 | 10S | short/med/long | 3 |
| | bizitobs_l2c | 7 | 5T, H | short/med/long | 6 |
| | bizitobs_service | 2 | 10S | short/med/long | 3 |
| | | | | **Total** | **97** |

**Task distribution**: 55 short + 21 medium + 21 long = 97. M4 and low-frequency datasets have short-term only.

**Leaderboard Top-15 (by avg CRPS Rank, 64 models total, Feb 2026)**:

| Rank | Model | Type | Org | CRPS Rank | MASE Rank |
|-----:|-------|------|-----|----------:|----------:|
| 1 | TSOrchestra | Agentic | USC | 8.75 | 9.34 |
| 2 | DeOSAlpha-TimeGPT | Zero-shot | Vencortex | 8.95 | 9.57 |
| 3 | Credence | Agentic | ContinualIST | 10.32 | 11.38 |
| 4 | TSOrchestra-test | Fine-tuned* | USC | 11.20 | 15.45 |
| 5 | Samay | Agentic | Kairosity | 11.36 | 13.09 |
| 6 | Synapse | Agentic | Google Cloud AI | 11.66 | 13.46 |
| 7 | MoiraiAgent* | Agentic | Salesforce | 12.54 | 10.50 |
| 8 | TimeCopilot | Agentic | — | 12.75 | 13.71 |
| 9 | MoiraiAgent | Agentic | Salesforce | 12.77 | 11.25 |
| **10** | **Chronos-2** | **Pretrained** | **AWS** | **15.21** | **14.64** |
| 11 | PatchTST-FM-r1 | Zero-shot | IBM/RPI | 15.29 | 17.24 |
| 12 | TiRex | Zero-shot | NX-AI | 15.50 | 18.73 |
| 13 | TimesFM-2.5 | Zero-shot | Google | 17.19 | 17.59 |
| 14 | Xihe-ultra | Zero-shot | Ant | 19.43 | 20.01 |
| 15 | FlowState-9.1M | Zero-shot | IBM | 20.14 | 22.62 |

*\*Asterisk = data leaking (used test data for training/selection)*

**Key observation**: Top ranks are dominated by **agentic** models (LLM-based multi-step reasoning that selects and ensembles forecasters). Among **non-agentic, non-leaking pretrained/zero-shot** models:

| Effective Rank | Model | CRPS Rank | MASE Rank |
|:-:|-------|----------:|----------:|
| 1 | **Chronos-2** | 15.21 | 14.64 |
| 2 | PatchTST-FM-r1 | 15.29 | 17.24 |
| 3 | TiRex | 15.50 | 18.73 |
| 4 | TimesFM-2.5 | 17.19 | 17.59 |
| 5 | Xihe-ultra | 19.43 | 20.01 |

```bash
# Download data
python utils/download_gift_eval.py \
    --output-dir /group-volume/ts-dataset/benchmarks/gift_eval/

# Run evaluation (~6-7 hr on 1x A100 bfloat16)
python run_benchmark.py \
    --model-path /path/to/model \
    --benchmarks gift_eval \
    --gift-eval-data /group-volume/ts-dataset/benchmarks/gift_eval/
```

#### `fev_bench` — fev-bench (AutoGluon / Amazon)

> **Paper**: [arXiv:2509.26468](https://arxiv.org/abs/2509.26468)
> **Package**: [pypi.org/project/fev](https://pypi.org/project/fev/) (v0.7.0+)
> **Leaderboard**: [huggingface.co/spaces/autogluon/fev-bench](https://huggingface.co/spaces/autogluon/fev-bench)
> **Requires**: `pip install fev`

100 tasks across 96 unique datasets spanning 7 domains, with 46 tasks featuring covariates (known future, past dynamic, static).

| Property | Value |
|----------|-------|
| Tasks | 100 (54 univariate + 46 with covariates) |
| Datasets | 96 unique |
| Quantiles | {0.1, 0.2, ..., 0.9} |
| Evaluation | Non-overlapping rolling windows (1–20 per task) |
| Primary metric | SQL (Scaled Quantile Loss) |
| Aggregation | Win Rate (tie=0.5) + Skill Score + Bootstrap CI (B=1000) |
| Covariates | known_dynamic, past_dynamic, static |

**Representative baselines for comparison**:
| Model | Type | Reference |
|-------|------|-----------|
| Chronos-2 (120M) | Foundation model | Amazon (2025) |
| TimesFM (200M) | Foundation model | Google (2024) |
| AutoGluon-TimeSeries | AutoML | Amazon (2024) |
| SeasonalNaive | Statistical baseline | — |
| ETS / AutoARIMA | Statistical | — |

```bash
# Download data
python utils/download_fev_bench.py \
    --output-dir /group-volume/ts-dataset/benchmarks/fev_bench/

# Run evaluation
python run_benchmark.py \
    --model-path /path/to/model \
    --benchmarks fev_bench \
    --fev-data /group-volume/ts-dataset/benchmarks/fev_bench/
```

#### `ltsf` — LTSF Benchmark (DLinear / PatchTST / iTransformer)

> **Origin**: DLinear paper ([arXiv:2205.13504](https://arxiv.org/abs/2205.13504), AAAI 2023)
> **Standardized by**: PatchTST, iTransformer, TimesNet, Crossformer, FEDformer

The standard long-term series forecasting benchmark used by virtually all supervised time series transformer papers. Evaluates **point forecast accuracy** on z-normalized multivariate data.

| Property | Value |
|----------|-------|
| Datasets | 9 (ETTh1, ETTh2, ETTm1, ETTm2, Weather, Traffic, Electricity, Exchange, ILI) |
| Horizons | {96, 192, 336, 720} (ILI: {24, 36, 48, 60}) |
| Tasks | 9 × 4 = 36 |
| Lookback | 96 timesteps (configurable) |
| Normalization | Per-variable z-score using train-set statistics only |
| Evaluation | Stride-1 sliding window over test split |
| Metrics | MSE, MAE on z-normalized values (flat mean) |
| Splits | ETT: 6:2:2, Others: 7:1:2 (chronological) |
| Forecasting | Channel-independent (each variable as univariate) |

**Note**: Chronos-2 does NOT evaluate on LTSF in the paper. This benchmark enables cross-comparison with supervised baselines like PatchTST, iTransformer, DLinear that are primarily evaluated on LTSF.

**LTSF datasets**:
| Dataset | Variables | Frequency | Total Timesteps | Domain |
|---------|-----------|-----------|-----------------|--------|
| ETTh1/h2 | 7 | Hourly | 17,420 | Energy (Electricity Transformer Temperature) |
| ETTm1/m2 | 7 | 15-min | 69,680 | Energy (Electricity Transformer Temperature) |
| Weather | 21 | 10-min | 52,696 | Meteorology |
| Traffic | 862 | Hourly | 17,544 | Transportation |
| Electricity | 321 | Hourly | 26,304 | Energy (consumption) |
| Exchange | 8 | Daily | 7,588 | Finance (exchange rates) |
| ILI | 7 | Weekly | 966 | Healthcare (influenza-like illness) |

**Representative baselines for comparison**:
| Model | Type | MSE (avg) | Reference |
|-------|------|-----------|-----------|
| PatchTST | Supervised transformer | ~0.35 | Nie et al. (2023) |
| iTransformer | Supervised transformer | ~0.36 | Liu et al. (2024) |
| DLinear | Linear model | ~0.38 | Zeng et al. (2023) |
| TimesNet | Supervised CNN | ~0.37 | Wu et al. (2023) |
| Crossformer | Supervised transformer | ~0.40 | Zhang & Yan (2023) |
| FEDformer | Supervised transformer | ~0.42 | Zhou et al. (2022) |

```bash
# Download data
python utils/download_ltsf.py \
    --output-dir /group-volume/ts-dataset/benchmarks/ltsf/

# Run evaluation
python run_benchmark.py \
    --model-path /path/to/model \
    --benchmarks ltsf \
    --ltsf-data /group-volume/ts-dataset/benchmarks/ltsf/
```

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
├── chronos/       # 42 datasets (Chronos I + II) — Arrow IPC format
├── gift_eval/     # ~30 datasets — HuggingFace snapshot
├── fev_bench/     # ~48 datasets — HuggingFace Parquet
└── ltsf/          # 9 datasets — CSV files
```

For a fresh setup, download datasets to a local directory:

```bash
cd scripts/forecasting/evaluation

# Chronos benchmarks (Arrow format)
python utils/download_eval_datasets.py \
    --config configs/chronos-full.yaml \
    --output-dir /path/to/benchmarks/chronos/

# GIFT-Eval (from Salesforce/GiftEval HuggingFace)
python utils/download_gift_eval.py \
    --output-dir /path/to/benchmarks/gift_eval/

# fev-bench (from autogluon/fev_datasets HuggingFace)
python utils/download_fev_bench.py \
    --output-dir /path/to/benchmarks/fev_bench/

# LTSF (9 CSV files)
python utils/download_ltsf.py \
    --output-dir /path/to/benchmarks/ltsf/
```

The Chronos datasets are stored in Arrow IPC format:
```
/path/to/benchmarks/chronos/
├── m4_hourly/
│   └── data-00000-of-00001.arrow
├── m4_monthly/
│   └── data-00000-of-00001.arrow
└── ...

/path/to/benchmarks/ltsf/
├── ETTh1.csv
├── ETTh2.csv
├── Weather.csv
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
                           chronos_full, lite, extended, gift_eval, fev_bench, ltsf

Data paths (benchmark-specific):
  --datasets-root DIR      Root directory for Chronos datasets (Arrow IPC format)
  --gift-eval-data DIR     GIFT-Eval data directory (Salesforce/GiftEval HF snapshot)
  --fev-data DIR           fev-bench data directory (autogluon/fev_datasets HF snapshot)
  --ltsf-data DIR          LTSF data directory (9 CSV files)

Optional:
  --output-dir DIR         Base directory for results (default: results/experiments/)
  --experiment-name NAME   Custom experiment identifier (default: auto-timestamp)
  --device DEVICE          cuda, cuda:0, cpu (default: cuda)
  --torch-dtype DTYPE      float32, bfloat16 (default: float32)
  --batch-size N           Inference batch size (default: 32)
  --seed N                 Random seed for reproducibility (default: 42)
  --dry-run                Check data/model availability without running evaluation
  --resume                 Resume interrupted evaluation from partial results
  --verbose                Enable verbose logging (per-series progress)
  --quiet                  Suppress per-dataset progress output
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

### Chronos-2 vs Foundation Models (from Chronos-2 Paper, arXiv:2510.15821)

Win Rate (%) and Skill Score (%) vs. Seasonal Naive baseline.
Higher is better. Win Rate > 50% = beats more models than it loses to.

**Chronos Benchmark II — WQL (27 datasets)**:

| Model | Win Rate (%) | Skill Score (%) |
|-------|:------------:|:---------------:|
| **Chronos-2 (120M)** | **79.8** | **46.6** |
| TiRex (35M) | 70.4 | 41.7 |
| TimesFM-2.5 (200M) | 70.0 | 42.4 |
| Toto-1.0 (151M) | 60.9 | 41.9 |
| Moirai-2.0 (305M) | 56.0 | 40.9 |
| Chronos-Bolt (205M) | 49.4 | 39.3 |
| TabPFN-TS (11M) | 46.3 | 32.6 |
| COSMIC | 42.8 | 36.7 |
| Seasonal Naive | 10.1 | 0.0 |

**Chronos Benchmark II — MASE (27 datasets)**:

| Model | Win Rate (%) | Skill Score (%) |
|-------|:------------:|:---------------:|
| **Chronos-2 (120M)** | **81.5** | **26.5** |
| TimesFM-2.5 (200M) | 71.6 | 23.3 |
| TiRex (35M) | 67.1 | 22.2 |
| Toto-1.0 (151M) | 58.0 | 22.3 |
| Moirai-2.0 (305M) | 53.5 | 19.8 |
| Chronos-Bolt (205M) | 50.6 | 20.4 |
| Seasonal Naive | 13.8 | 0.0 |

**GIFT-Eval — WQL (97 tasks)**:

| Model | Win Rate (%) | Skill Score (%) |
|-------|:------------:|:---------------:|
| **Chronos-2 (120M)** | **81.9** | **51.4** |
| TimesFM-2.5 (200M) | 77.5 | 51.0 |
| TiRex (35M) | 76.5 | 50.2 |
| Toto-1.0 (151M) | 67.4 | 48.6 |
| Moirai-2.0 (305M) | 64.4 | 48.4 |
| COSMIC | 56.4 | 44.5 |
| Chronos-Bolt (205M) | 53.8 | 42.6 |

**fev-bench — SQL (100 tasks)**:

| Model | Win Rate (%) | Skill Score (%) | Runtime (s) |
|-------|:------------:|:---------------:|:-----------:|
| **Chronos-2 (120M)** | **90.7** | **47.3** | 3.6 |
| TiRex (35M) | 80.8 | 42.6 | 1.4 |
| TimesFM-2.5 (200M) | 75.9 | 42.3 | 16.9 |
| Toto-1.0 (151M) | 66.6 | 40.7 | 90.7 |
| Moirai-2.0 (305M) | 61.1 | 39.3 | 2.5 |
| Chronos-Bolt (205M) | 60.3 | 38.9 | 1.0 |
| TabPFN-TS (11M) | 59.3 | 39.6 | 305.5 |
| Seasonal Naive | 14.5 | 0.0 | 2.3 |

**Metric relationship** (from Chronos-2 paper):
- Skill Score S and Geometric Mean Relative Error G: `G = 1 - S/100`
- Win Rate W and Average Rank R (over N models): `R = 1 + (1 - W/100)(N - 1)`

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
│   │                             #     + compute_chronos_metrics_fast()
│   │                             #     + compute_gift_eval_metrics_fast()
│   ├── forecaster.py             #   BaseForecaster ABC + model adapters
│   │                             #     ├── Chronos2Forecaster (inference_mode)
│   │                             #     ├── ChronosBoltForecaster (inference_mode)
│   │                             #     └── TrainingModelForecaster (inference_mode)
│   ├── evaluator.py              #   Unified evaluation loop
│   │                             #     dataset → batch GPU inference → fast metrics
│   └── aggregator.py             #   Result aggregation
│                                 #     gmean, bootstrap CI, win rate, skill score
│
├── benchmarks/                   # Pluggable benchmark adapters
│   ├── base.py                   #   BenchmarkAdapter ABC (interface)
│   ├── chronos_bench.py          #   Chronos I/II, Lite, Extended, Full
│   ├── gift_eval.py              #   GIFT-Eval adapter (Salesforce)
│   ├── fev_bench.py              #   fev-bench adapter (AutoGluon)
│   └── ltsf.py                   #   LTSF adapter (DLinear/PatchTST)
│
├── configs/                      # Benchmark configuration (YAML)
│   ├── chronos-lite.yaml         #   5 datasets — quick validation
│   ├── chronos-extended.yaml     #   15 datasets — thorough validation
│   ├── chronos-i.yaml            #   15 datasets — Chronos Bench I (in-domain)
│   ├── chronos-ii.yaml           #   27 datasets — Chronos Bench II (zero-shot)
│   ├── chronos-full.yaml         #   42 datasets — all Chronos datasets
│   ├── gift-eval.yaml            #   GIFT-Eval reference config
│   ├── fev-bench.yaml            #   fev-bench reference config
│   └── ltsf.yaml                 #   LTSF benchmark config (9 datasets)
│
├── results/                      # Reference baseline results (CSV)
│   ├── seasonal-naive-*.csv      #   Seasonal Naive baseline scores
│   ├── chronos-t5-*-*.csv        #   Chronos-T5 model scores
│   ├── chronos-bolt-*-*.csv      #   Chronos-Bolt model scores
│   └── chronos-2-lite-*.csv      #   Chronos-2 lite benchmark reference
│
└── utils/                       # Dataset download utilities
    ├── download_eval_datasets.py #   Chronos benchmarks (HuggingFace → local)
    ├── download_failed_datasets.py # Recover failed/incompatible downloads
    ├── download_gift_eval.py     #   GIFT-Eval data download
    ├── download_fev_bench.py     #   fev-bench data download
    └── download_ltsf.py          #   LTSF CSV data download
```

### Design Principles

- **Local-first data loading**: All datasets loaded from local disk via `datasets_root`. No network access required at evaluation time.
- **Paper-aligned benchmarks**: Config names match paper benchmark names (chronos_i, chronos_ii, gift_eval, fev_bench).
- **Arrow IPC fallback**: Handles `datasets` library version mismatches gracefully via direct PyArrow reader.
- **Shape contract**: `BaseForecaster.predict_quantiles()` returns `(N, Q, H)`. Assertions enforce this at every stage.
- **Pluggable adapters**: Each benchmark is an independent adapter implementing `BenchmarkAdapter`. Adding a new benchmark = one new file.
- **hf_repo optional**: Dataset configs no longer require `hf_repo`; `datasets_root` resolves all local paths.
- **GPU-first inference**: `torch.inference_mode()` for all model forward passes (faster than `torch.no_grad()`).
- **Vectorized metrics**: All metric computation uses direct numpy operations, bypassing GluonTS Python-level iteration.

### Performance Optimization

The evaluation pipeline uses a **fast vectorized computation path** that bypasses GluonTS `evaluate_model()` / `evaluate_forecasts()` entirely. This eliminates the primary CPU bottleneck observed during large-scale evaluation (3000%+ CPU utilization with 0% GPU).

#### Problem

GluonTS evaluation creates `QuantileForecast` Python objects per series, then iterates over them in Python loops to compute metrics. For datasets with thousands of series (e.g., Electricity: 5,261 series × 7 rolling windows = 36,827 items), this creates tens of thousands of Python objects and per-item metric calls, causing CPU-bound numpy BLAS operations to dominate runtime while the GPU sits idle.

#### Solution

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| **Chronos metrics** | GluonTS `evaluate_forecasts()` + QuantileForecast | `MetricRegistry.compute_chronos_metrics_fast()` — vectorized numpy | ~100x |
| **GIFT-Eval metrics** | GluonTS `evaluate_model()` + 11 metric evaluators | `MetricRegistry.compute_gift_eval_metrics_fast()` — all 11 metrics vectorized | ~100x |
| **Inference context** | `torch.no_grad()` | `torch.inference_mode()` — disables view tracking & version counting | ~10-20% |
| **LTSF tensor creation** | Nested Python loop (27K+ `torch.tensor()` calls) | `np.ascontiguousarray().reshape()` + `torch.from_numpy()` | ~50x |
| **GIFT-Eval data loading** | Double `Dataset()` load per task | Single `Dataset(to_univariate=True)` load | 2x |

#### Fast Metric Functions

**`MetricRegistry.compute_chronos_metrics_fast()`** — Computes WQL + MASE matching GluonTS output:
- Input: `y_pred_quantiles (N, Q, H)`, `y_true (N, H)`, `y_past [list of 1D arrays]`
- WQL: per-item normalized quantile loss, then mean (matches `MeanWeightedSumQuantileLoss`)
- MASE: seasonal naive scale, NaN-safe for constant/short series

**`MetricRegistry.compute_gift_eval_metrics_fast()`** — Computes all 11 GIFT-Eval metrics:
- CRPS, MASE, sMAPE, MAPE, MSE, MAE, RMSE, NRMSE, ND, MSIS, MSE[mean]
- Output keys match GluonTS column names for report compatibility
- Uses `float64` precision for numerical stability

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

| Benchmark | Tasks | A100 (bfloat16) | A100 (float32) | CPU (72 cores) |
|-----------|-------|-----------------|----------------|----------------|
| `chronos_lite` | 5 | ~3 min | ~5 min | ~40 min |
| `chronos_extended` | 15 | ~15 min | ~25 min | ~3 hr |
| `chronos_i` | 15 | ~30 min | ~50 min | ~5 hr |
| `chronos_ii` | 27 | ~60 min | ~90 min | ~10 hr |
| `chronos_full` | 42 | ~90 min | ~120 min | ~12 hr |
| `gift_eval` | 97 | ~6-7 hr | ~10 hr | ~48+ hr |
| `fev_bench` | 100 | ~3 hr | ~5 hr | ~30 hr |
| `ltsf` | 36 | ~30 min | ~45 min | ~4 hr |

> **Tip**: Use `--torch-dtype bfloat16` on GPU for faster inference with minimal accuracy impact.
> **Note**: GIFT-Eval and fev-bench times depend on dataset download speed if not pre-cached.
