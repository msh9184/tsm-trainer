<p align="center">
  <h1 align="center">TSM-Trainer</h1>
  <p align="center"><b>Time Series Model Trainer</b> — A unified framework for training, evaluating, and deploying time series foundation models.</p>
</p>

<p align="center">
  <a href="#supported-models">Models</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#project-structure">Structure</a> &bull;
  <a href="#distributed-training">Distributed Training</a> &bull;
  <a href="#roadmap">Roadmap</a>
</p>

---

## Overview

**TSM-Trainer** is a research framework for training and evaluating time series foundation models at scale. Built on top of [HuggingFace Transformers](https://github.com/huggingface/transformers) and [PyTorch](https://pytorch.org/), it provides production-ready distributed training pipelines, comprehensive monitoring, and standardized evaluation benchmarks.

The first target model is **Chronos-2** (120M params), an encoder-only transformer with group attention that supports univariate forecasting, multivariate forecasting, and covariate-informed forecasting through in-context learning.

> **Reference**: [Chronos-2: From Univariate to Universal Forecasting](https://arxiv.org/abs/2510.15821) (arXiv:2510.15821)

## Supported Models

| Model | Architecture | Params | Approach | Status |
|-------|-------------|--------|----------|--------|
| **Chronos-2 Base** | Encoder-only + Group Attention | 120M | Quantile regression, patch-based | Active |
| **Chronos-2 Small** | Encoder-only + Group Attention | 28M | Quantile regression, patch-based | Active |
| **Chronos (T5)** | T5 Seq2Seq | 8M–710M | Token-level cross-entropy | Supported |
| **Chronos-Bolt** | Encoder-only, patch-based | — | Direct quantile output | Supported |

## Features

### Training

| Feature | Description |
|---------|-------------|
| **Multi-Node Distributed** | mpirun-based multi-node training (tested on 64x A100 80GB) |
| **FSDP** | Fully Sharded Data Parallel with auto-wrapping per encoder block |
| **DDP** | Standard Distributed Data Parallel for single-node multi-GPU |
| **Gradient Checkpointing** | Monkey-patched activation checkpointing (~30-40% memory reduction) |
| **Two-Stage Training** | Context extension from 2048 to 8192 tokens with automatic config transition |
| **Lazy Data Loading** | Arrow-native on-demand reads — zero upfront memory, LRU cached |
| **Mixed Precision** | bf16 / tf32 on Ampere+ GPUs |
| **Fused Optimizer** | `adamw_torch_fused` for reduced kernel launch overhead |

### Monitoring & Evaluation

| Feature | Description |
|---------|-------------|
| **Health Reports** | Periodic boxed reports with loss analysis, gradient health, GPU utilization, throughput |
| **Sparkline Plots** | Unicode sparklines for loss/LR trends directly in terminal |
| **Distributed Benchmark** | WQL/MASE evaluation across all GPUs (FSDP-compatible) |
| **Fast Vectorized Metrics** | Bypasses GluonTS with direct numpy computation (~100x faster metric phase) |
| **Metric-Encoded Checkpoints** | NeMo-style filenames with step/WQL/MASE/composite scores |
| **Top-K Checkpoint Management** | Automatic retention of best K models with worst-removal |
| **Validation Reports** | Per-eval JSON + cumulative CSV for offline analysis |
| **Resume-Safe State** | Rebuilds top-K state from disk artifacts on training resume |
| **TensorBoard** | Hierarchical metrics: per-dataset WQL/MASE, validation_loss, best tracking |

### Data

| Feature | Description |
|---------|-------------|
| **KernelSynth** | Gaussian Process synthetic data with 35 kernel types |
| **HuggingFace Datasets** | Native Arrow IPC format, memory-mapped loading |
| **Offline Support** | Pre-download utility for air-gapped / proxy-restricted environments |

## Quick Start

### Installation

```bash
pip install -e ".[dev]"
```

### Single-GPU Test

```bash
python scripts/forecasting/training/train_chronos2.py \
    --config scripts/forecasting/training/configs/chronos2-test.yaml
```

### Multi-GPU (Single Node)

```bash
torchrun --nproc_per_node=8 scripts/forecasting/training/train_chronos2.py \
    --config scripts/forecasting/training/configs/chronos2-base.yaml
```

### Multi-Node (mpirun)

```bash
bash scripts/forecasting/training/train.sh \
    --config configs/chronos2-base.yaml
```

> See [`scripts/forecasting/README.md`](scripts/forecasting/README.md) for the full training, evaluation, and monitoring guide.

## Project Structure

```
tsm-trainer/
├── README.md                          # This file
├── ci/
│   └── evaluate/
│       └── backtest_config.yaml       # CI evaluation config
├── scripts/
│   └── forecasting/                   # Forecasting task scripts
│       ├── README.md                  # Detailed forecasting guide
│       ├── training/
│       │   ├── train_chronos2.py      # Chronos-2 training script
│       │   ├── train.py               # Chronos (T5/GPT2) training
│       │   ├── train.sh               # mpirun launcher
│       │   ├── callbacks/             # Training callbacks
│       │   │   └── benchmark_callback.py  # Validation + checkpoint management
│       │   └── configs/               # Training YAML configs
│       ├── evaluation/
│       │   ├── run_benchmark.py       # Unified benchmark CLI
│       │   ├── compare_models.py      # Model comparison tables
│       │   ├── engine/                # Core evaluation engine
│       │   ├── benchmarks/            # Benchmark adapters
│       │   ├── configs/               # Benchmark dataset configs
│       │   ├── results/               # Pre-computed baselines
│       │   └── utils/                 # Dataset downloaders
│       └── kernel-synth.py            # Synthetic data generator
└── src/
    └── chronos/                       # Core model library
        ├── chronos2/                  # Chronos-2 (encoder-only)
        │   ├── config.py              # Model & forecasting configs
        │   ├── layers.py              # RoPE, MHA, GroupAttention, etc.
        │   ├── model.py               # Chronos2Model
        │   ├── pipeline.py            # Inference & fine-tuning API
        │   ├── dataset.py             # Chronos2Dataset (batching)
        │   └── trainer.py             # Custom HF Trainer
        ├── chronos.py                 # Chronos (T5-based)
        └── chronos_bolt.py            # Chronos-Bolt
```

The `scripts/` directory is organized by **task type** (`forecasting/`, and future `classification/`, `anomaly_detection/`, etc.), enabling clean extension to new time series tasks while sharing the core `src/chronos/` library.

## Distributed Training

### Supported Configurations

| Mode | Command | Use Case |
|------|---------|----------|
| Single GPU | `python train_chronos2.py --config ...` | Development & debugging |
| Single Node | `torchrun --nproc_per_node=8 train_chronos2.py --config ...` | 8x GPU training |
| Multi-Node | `bash train.sh --config ...` | 64+ GPU production training |

### FSDP (Fully Sharded Data Parallel)

FSDP shards model parameters, gradients, and optimizer states across all GPUs. Configured in YAML:

```yaml
fsdp: "full_shard auto_wrap"
fsdp_config:
  fsdp_transformer_layer_cls_to_wrap: "Chronos2EncoderBlock"
  fsdp_backward_prefetch: "backward_pre"
  fsdp_use_orig_params: true
  fsdp_sync_module_states: true
  fsdp_state_dict_type: "FULL_STATE_DICT"
```

### Gradient Checkpointing

Trades compute for memory by recomputing activations during backward pass. Enabled per config:

```yaml
gradient_checkpointing: true
```

Wraps each `Chronos2EncoderBlock.forward()` with `torch.utils.checkpoint.checkpoint()`.

## Training Configurations

| Config | Model | GPUs | Context | Batch | Steps | Use Case |
|--------|-------|------|---------|-------|-------|----------|
| `chronos2-test` | 2L/256d | 1–64 | 512 | 16 | 200 | Pipeline verification |
| `chronos2-base` | 12L/768d (120M) | 64 | 2048 | 12 | 200K | Stage 1 pretraining |
| `chronos2-base-stage2` | 12L/768d (120M) | 64 | 8192 | 4 | 50K | Stage 2 context extension |
| `chronos2-small` | 6L/512d (28M) | 64 | 2048 | 16 | 200K | Small model pretraining |
| `chronos2-finetune` | 12L/768d (120M) | 8–64 | 2048 | 8 | 50K | Fine-tuning from pretrained |

## Evaluation Benchmarks

### Supported Benchmark Suites

| Suite | Tasks | Metrics | Forecast Type | Use Case |
|-------|-------|---------|---------------|----------|
| **Chronos Lite** | 5 | WQL, MASE | Probabilistic | Quick training-time validation (~3 min) |
| **Chronos Extended** | 15 | WQL, MASE | Probabilistic | Thorough validation (~15 min) |
| **Chronos Bench I** | 15 | WQL, MASE | Probabilistic | In-domain evaluation (paper) |
| **Chronos Bench II** | 27 | WQL, MASE | Probabilistic | Zero-shot evaluation (paper) |
| **GIFT-Eval** | 97 | CRPS, MASE, WQL, sMAPE + 7 more | Probabilistic | Multi-domain comprehensive eval |
| **fev-bench** | 100 | SQL, Win Rate, Skill Score | Probabilistic | Covariate-capable evaluation |
| **LTSF** | 36 | MSE, MAE | Point (median) | Cross-comparison with supervised baselines |

### Benchmark Comparison

| | Chronos I/II | GIFT-Eval | fev-bench | LTSF |
|---|---|---|---|---|
| **Origin** | Amazon (Chronos paper) | Salesforce AI Research | AutoGluon (Amazon) | DLinear (AAAI 2023) |
| **Evaluation** | Rolling windows | Rolling windows (max 20) | Rolling windows (1–20) | Stride-1 sliding window |
| **Normalization** | None (raw values) | None (raw values) | None (raw values) | z-score (train stats) |
| **Covariates** | No | Some tasks | 46 of 100 tasks | No |
| **Multivariate** | Univariate only | Some tasks | Some tasks | Channel-independent |
| **Baseline** | Seasonal Naive | Naive predictor | SeasonalNaive | N/A (supervised models) |
| **Key baselines** | Chronos-T5/Bolt, ARIMA | Chronos-2, TimesFM, Moirai | Chronos-2, TimesFM, AutoGluon | PatchTST, iTransformer, DLinear |

> See [`scripts/forecasting/evaluation/README.md`](scripts/forecasting/evaluation/README.md) for detailed benchmark protocols, metric formulas, and baseline performance.

## Proxy / Offline Setup

For GPU servers with network restrictions:

```bash
# Set proxy and cache
export HTTPS_PROXY=http://proxy:8080
export HF_HOME=/group-volume/hf_cache

# Chronos benchmarks (42 datasets, Arrow format)
python scripts/forecasting/evaluation/utils/download_eval_datasets.py \
    --config scripts/forecasting/evaluation/configs/chronos-full.yaml \
    --output-dir /group-volume/ts-dataset/benchmarks/chronos/

# GIFT-Eval (Salesforce/GiftEval from HuggingFace)
python scripts/forecasting/evaluation/utils/download_gift_eval.py \
    --output-dir /group-volume/ts-dataset/benchmarks/gift_eval/

# fev-bench (autogluon/fev_datasets from HuggingFace)
python scripts/forecasting/evaluation/utils/download_fev_bench.py \
    --output-dir /group-volume/ts-dataset/benchmarks/fev_bench/

# LTSF (9 CSV files — ETT, Weather, Traffic, etc.)
python scripts/forecasting/evaluation/utils/download_ltsf.py \
    --output-dir /group-volume/ts-dataset/benchmarks/ltsf/
```

## Roadmap

- [x] **Chronos-2 training pipeline** — mpirun, FSDP, gradient checkpointing, lazy loading
- [x] **Training monitoring** — health reports, sparklines, TensorBoard
- [x] **Distributed benchmark** — multi-GPU WQL/MASE evaluation during training
- [x] **Validation infrastructure** — metric-encoded checkpoints, JSON/CSV reports, resume-safe state
- [x] **Benchmark suite** — GIFT-Eval, fev-bench, LTSF adapters integrated with unified CLI
- [x] **Evaluation optimization** — vectorized numpy metrics, GPU-first inference, GluonTS bypass
- [ ] **Data generation** — TSI, TCM generators, multivariatizers
- [ ] **Classification task** — time series classification framework
- [ ] **Anomaly detection task** — time series anomaly detection framework

## License

Apache-2.0

## Acknowledgments

Built on the [Chronos](https://github.com/amazon-science/chronos-forecasting) framework by Amazon Science.
