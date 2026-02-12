# Forecasting Guide

Comprehensive guide for training, evaluating, and monitoring **Chronos-2** time series foundation models.

---

## Table of Contents

- [Architecture](#architecture)
- [Training](#training)
  - [Quick Test](#quick-test)
  - [Pretraining](#pretraining-from-scratch)
  - [Stage 2 Context Extension](#stage-2-context-extension)
  - [Fine-Tuning](#fine-tuning)
- [Data Pipeline](#data-pipeline)
- [Evaluation](#evaluation)
- [Monitoring](#monitoring)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Architecture

### Chronos-2 Encoder Block

Each encoder block processes the input through three sub-layers:

```
Input
  │
  ├──► TimeSelfAttention (RoPE)    ─── temporal patterns within each series
  │
  ├──► GroupSelfAttention           ─── cross-series ICL (batch-axis attention)
  │
  ├──► FeedForward (Pre-Norm + MLP)
  │
  ▼
Output
```

### End-to-End Pipeline

```
Raw Series ──► InstanceNorm (arcsinh) ──► Patching (size P, non-overlapping)
           ──► Embed [time_enc, patch, mask] ──► [REG] token
           ──► N × EncoderBlock ──► QuantileHead
           ──► Q quantile forecasts (direct multi-step)
```

### Model Variants

| | Base (120M) | Small (28M) |
|---|---|---|
| Layers | 12 | 6 |
| d_model | 768 | 512 |
| Heads | 12 | 8 |
| d_kv | 64 | 64 |
| d_ff | 3072 | 2048 |
| Quantiles | 21 | 13 |

### Inference Modes

```python
# Univariate — each series independent
pipeline.predict([series_a, series_b], prediction_length=24)

# Multivariate — group related series for ICL
pipeline.predict([{"target": multivariate_tensor}], prediction_length=24)

# Cross-learning — all series share context
pipeline.predict([s1, s2, s3], prediction_length=24, cross_learning=True)
```

---

## Training

### Quick Test

Verify the full pipeline works before committing to long runs:

```bash
# Single GPU
python training/train_chronos2.py --config training/configs/chronos2-test.yaml

# Multi-node (64 GPU)
bash training/train.sh --config configs/chronos2-test.yaml
```

**Expected**: ~200 steps in 5–10 minutes. Loss starts ~14 and should decrease.

### Pretraining (from scratch)

```bash
# 120M Base model, 64x A100
bash training/train.sh --config configs/chronos2-base.yaml
```

**Config highlights** (`chronos2-base.yaml`):

```yaml
random_init: true
d_model: 768
num_layers: 12
context_length: 2048                       # Stage 1
max_steps: 200_000
per_device_train_batch_size: 12            # × 64 GPUs = 768 effective
learning_rate: 1.0e-4
lr_scheduler_type: "cosine"

training_data_paths:
  - ".../training_corpus_tsmixup_10m"      # 90%
  - ".../training_corpus_kernel_synth_1m"  # 10%

fsdp: "full_shard auto_wrap"
gradient_checkpointing: true
```

### Stage 2 (Context Extension)

Extends context from 2048 → 8192 tokens. Loads from Stage 1 checkpoint:

```bash
bash training/train.sh --config configs/chronos2-base-stage2.yaml
```

**Key differences from Stage 1**:

| Parameter | Stage 1 | Stage 2 |
|-----------|---------|---------|
| `context_length` | 2048 | 8192 |
| `prediction_length` | 64 | 256 |
| `learning_rate` | 1e-4 | 3e-5 |
| `per_device_train_batch_size` | 12 | 4 |
| `gradient_accumulation_steps` | 1 | 2 |
| `max_steps` | 200K | 50K |
| `model_id` | — | `./output/chronos2-base-stage1/final-checkpoint` |

### Fine-Tuning

From a pretrained model (e.g., `amazon/chronos-2`):

```bash
bash training/train.sh --config configs/chronos2-finetune.yaml
```

```yaml
random_init: false
model_id: "amazon/chronos-2"     # or local checkpoint path
learning_rate: 1.0e-5            # 10x lower than pretraining
max_steps: 50_000
```

### CLI Overrides

Any config value can be overridden from the command line:

```bash
python training/train_chronos2.py \
    --config training/configs/chronos2-base.yaml \
    --max-steps 50000 \
    --learning-rate 5e-5 \
    --per-device-train-batch-size 8 \
    --output-dir ./output/experiment-1
```

---

## Data Pipeline

### Lazy Arrow Loading

Training data must be in HuggingFace `datasets` format (Arrow IPC):

```python
import datasets
ds = datasets.Dataset.from_dict({"target": [series1, series2, ...]})
ds.save_to_disk("/path/to/dataset")
```

The training script uses **LazyHFTaskSource** for memory-efficient loading:

```
┌─────────────────────────────────────────────────────────┐
│  LazyHFTaskSource                                       │
│                                                         │
│  Arrow datasets  ──(memory-mapped)──► Random access     │
│  No upfront load    LRU cache (50K)   Per-batch reads   │
│                                                         │
│  Each rank reads independently — no broadcast needed    │
└─────────────────────────────────────────────────────────┘
```

### Training Data Sources

| Dataset | Size | Description |
|---------|------|-------------|
| `training_corpus_tsmixup_10m` | ~10M series | TSMixup augmented real data |
| `training_corpus_kernel_synth_1m` | ~1M series | KernelSynth GP-generated data |

### Synthetic Data Generation

```bash
# Generate KernelSynth data (35 GP kernel types)
python kernel-synth.py
```

KernelSynth composes 1–5 random kernels (periodic, RBF, trend, noise) via `+`/`×` operators to produce diverse synthetic time series of 1024 time points each.

### Download Datasets for Offline Use

```bash
# Lite benchmark (5 datasets)
python evaluation/download_eval_datasets.py \
    --config evaluation/configs/lite-benchmark.yaml

# Full zero-shot benchmark (27 datasets)
HTTPS_PROXY=http://proxy:8080 python evaluation/download_eval_datasets.py \
    --config evaluation/configs/zero-shot.yaml

# Custom cache directory
python evaluation/download_eval_datasets.py \
    --config evaluation/configs/lite-benchmark.yaml \
    --cache-dir /group-volume/hf_cache
```

### Validate Datasets

```bash
python validate_datasets.py                    # Check default datasets
python validate_datasets.py --list-all         # List available datasets
python validate_datasets.py --count-all        # Count series (slow)
```

---

## Evaluation

### Standalone Evaluation

```bash
python evaluation/evaluate.py \
    evaluation/configs/zero-shot.yaml \
    results/my-model-zero-shot.csv \
    --chronos-model-id ./output/chronos2-base-stage1/final-checkpoint \
    --device cuda:0 \
    --batch-size 32
```

### Benchmark Configs

| Config | Datasets | Purpose |
|--------|----------|---------|
| `lite-benchmark.yaml` | 5 | Quick check during training (~3 min) |
| `zero-shot.yaml` | 27 | Unseen datasets (ETT, Traffic, COVID, Tourism, ...) |
| `in-domain.yaml` | 15 | Datasets from the training corpus |

### Lite Benchmark Datasets

| Dataset | Domain | Prediction Length |
|---------|--------|-------------------|
| M4 Hourly | Mixed | 48 |
| M4 Monthly | Mixed | 18 |
| Monash Weather | Weather | 30 |
| NN5 | Finance | 56 |
| Exchange Rate | Finance | 30 |

### Metrics

- **WQL** (Weighted Quantile Loss): Evaluates probabilistic forecasts across quantiles [0.1, 0.2, ..., 0.9]. Lower is better.
- **MASE** (Mean Absolute Scaled Error): Scale-free accuracy metric using the naive seasonal baseline. Lower is better.

### Training-Time Evaluation

When `benchmark_config` is set in the training YAML, the **LiteBenchmarkCallback** runs evaluation every N steps:

```yaml
benchmark_config: "configs/lite-benchmark.yaml"
benchmark_eval_steps: 10000
benchmark_top_k_checkpoints: 3
benchmark_batch_size: 32
```

Results are logged to TensorBoard under `benchmark/avg_wql`, `benchmark/avg_mase`, and per-dataset tags.

---

## Monitoring

### Per-Step Log Line

Printed every `log_steps` (default: 100) on rank-0:

```
Step 1,000/200,000 | loss=9.6887 | ema=9.7102 | best=9.6432 | lr=4.99e-05 | mem=42.1/80GB | 9.3 stp/s | 7131 smp/s | eta=5h 52m | gnorm=1.243
```

| Field | Meaning |
|-------|---------|
| `loss` | Current step loss |
| `ema` | Exponential moving average (alpha=0.05) |
| `best` | Lowest loss seen so far |
| `lr` | Current learning rate |
| `mem` | GPU memory allocated / total |
| `stp/s` | Training steps per second |
| `smp/s` | Samples per second (steps/s × effective_batch) |
| `eta` | Estimated time to completion |
| `gnorm` | Gradient norm (pre-clipping) |

### Health Report

Printed every `health_report_interval` steps (default: 1000):

```
╔════════════════════════════════════════════════════════════════════════╗
║  TRAINING HEALTH REPORT — Step 1,000 / 200,000 (0.5%)               ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  [Loss]                                                              ║
║    Current:       9.6887                                             ║
║    EMA (fast):    9.7102                                             ║
║    Best:          9.6432 (step 850)                                  ║
║    Convergence:   -0.31 per 1K steps (improving)                     ║
║    Variance:      0.0042                                             ║
║    Sparkline:     ▇▆▅▅▄▄▃▃▃▂▂▂▂▂▁▁▁▁▁▁                             ║
║                                                                      ║
║  [Gradient Health]                                                   ║
║    Mean norm:     1.842                                              ║
║    Max norm:      4.186                                              ║
║    Clip rate:     20.0% (2/10)                                       ║
║    Explosions:    0                                                  ║
║                                                                      ║
║  [GPU Memory (rank 0)]                                               ║
║    Allocated:     42.1 GB                                            ║
║    Peak:          45.8 GB                                            ║
║    Total:         80.0 GB                                            ║
║    [████████████████████████░░░░░░] 52.6%                            ║
║                                                                      ║
║  [Throughput]                                                        ║
║    Steps/sec:     9.31                                               ║
║    Samples/sec:   7,131                                              ║
║    Avg step:      0.107s (p95: 0.124s)                               ║
║    ETA:           5h 52m                                             ║
║                                                                      ║
║  [Data Pipeline]                                                     ║
║    Cache hit rate: 84.2% (50K capacity)                              ║
║    Short skips:    12                                                ║
║                                                                      ║
╚════════════════════════════════════════════════════════════════════════╝
```

### TensorBoard

```bash
tensorboard --logdir output/chronos2-base-stage1
```

Logged metrics:
- `train/loss`, `train/learning_rate`, `train/grad_norm`
- `benchmark/avg_wql`, `benchmark/avg_mase`
- `benchmark/{dataset}/wql`, `benchmark/{dataset}/mase`

### Training Configuration Summary

Printed once at the start of training:

```
╔════════════════════════════════════════════════════════════════════════╗
║                   TRAINING CONFIGURATION                             ║
╠════════════════════════════════════════════════════════════════════════╣
║  ┌─ Model ─────────────────────────────────────────────────────────┐ ║
║  │  Mode:          Pretraining (random init)                       │ ║
║  │  Parameters:    119.5M (119,488,277)                            │ ║
║  │  Context:       2048 -> 64 (patch=16)                           │ ║
║  │  Quantiles:     21   Max output patches: 64                     │ ║
║  │  Precision:     bf16=True, tf32=True                            │ ║
║  │  Grad ckpt:     True (12 blocks wrapped)                        │ ║
║  └─────────────────────────────────────────────────────────────────┘ ║
║  ┌─ Optimization ──────────────────────────────────────────────────┐ ║
║  │  Max steps:     200,000                                         │ ║
║  │  Batch/device:  12 x 64 GPUs x 1 accum = 768 effective         │ ║
║  │  Learning rate: 1.0e-04 (cosine)                                │ ║
║  │  Optimizer:     adamw_torch_fused                               │ ║
║  └─────────────────────────────────────────────────────────────────┘ ║
║  ┌─ Distributed ───────────────────────────────────────────────────┐ ║
║  │  Strategy:      FSDP (full_shard auto_wrap)                     │ ║
║  │  Wrap layer:    Chronos2EncoderBlock                            │ ║
║  │  World size:    64 GPUs                                         │ ║
║  └─────────────────────────────────────────────────────────────────┘ ║
║  ┌─ Data ──────────────────────────────────────────────────────────┐ ║
║  │  Source:        lazy Arrow (on-demand reads)                     │ ║
║  │  Train series:  1,000,000                                       │ ║
║  │    [0] training_corpus_tsmixup_10m (weight=0.90)                │ ║
║  │    [1] training_corpus_kernel_synth_1m (weight=0.10)            │ ║
║  └─────────────────────────────────────────────────────────────────┘ ║
║  ┌─ Monitoring ────────────────────────────────────────────────────┐ ║
║  │  Health reports: every 1000 steps                               │ ║
║  │  Benchmark:      every 10000 steps (lite-benchmark.yaml)        │ ║
║  │  Top-K ckpts:    3 (by WQL)                                     │ ║
║  │  Output dir:     ./output/chronos2-base-stage1                  │ ║
║  └─────────────────────────────────────────────────────────────────┘ ║
╚════════════════════════════════════════════════════════════════════════╝
```

---

## Configuration Reference

### Model Architecture

```yaml
d_model: 768              # Hidden dimension
d_kv: 64                  # Key/value head dimension
d_ff: 3072                # Feed-forward intermediate
num_layers: 12            # Encoder blocks
num_heads: 12             # Attention heads
dropout_rate: 0.1
feed_forward_proj: "relu"
rope_theta: 10000.0       # RoPE base frequency
attn_implementation: "sdpa"
```

### Forecasting

```yaml
context_length: 2048         # Input context window
prediction_length: 64        # Default prediction horizon
input_patch_size: 16         # Patch size (non-overlapping)
output_patch_size: 16
max_output_patches: 64       # Max prediction patches
use_reg_token: true          # [REG] separator between context and future
use_arcsinh: true            # Robust scaling
time_encoding_scale: 8192    # Time encoding denominator

quantiles:                   # 21 quantile levels
  - 0.01
  - 0.05
  - 0.1
  # ... (0.15 to 0.85 in 0.05 steps)
  - 0.9
  - 0.95
  - 0.99
```

### Training Hyperparameters

```yaml
max_steps: 200_000
per_device_train_batch_size: 12
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
lr_scheduler_type: "cosine"
warmup_ratio: 0.01
weight_decay: 0.01
max_grad_norm: 1.0
optim: "adamw_torch_fused"
seed: 42
```

### Data

```yaml
training_data_paths:
  - "/path/to/dataset1"
  - "/path/to/dataset2"
probability:
  - 0.9
  - 0.1
max_train_series: 1_000_000   # Subset size (random, deterministic)
min_past: 60                   # Minimum history length
```

### Distributed

```yaml
# FSDP (recommended for multi-node)
fsdp: "full_shard auto_wrap"
fsdp_config:
  fsdp_transformer_layer_cls_to_wrap: "Chronos2EncoderBlock"
  fsdp_backward_prefetch: "backward_pre"
  fsdp_use_orig_params: true
  fsdp_sync_module_states: true
  fsdp_state_dict_type: "FULL_STATE_DICT"

# DDP fallback
ddp_bucket_cap_mb: 25

# Performance
gradient_checkpointing: true
dataloader_num_workers: 4
```

### Monitoring

```yaml
output_dir: "./output/chronos2-base-stage1"
save_steps: 10_000
log_steps: 100
save_total_limit: 3
report_to: "tensorboard"
health_report_interval: 1000

# Lite benchmark (optional)
benchmark_config: "configs/lite-benchmark.yaml"
benchmark_eval_steps: 10_000
benchmark_top_k_checkpoints: 3
benchmark_batch_size: 32
```

---

## Troubleshooting

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Training hangs at data loading | Eager loading of millions of series | Use LazyHFTaskSource (default in `train_chronos2.py`) |
| `NotImplementedError: get_input_embeddings` | Chronos2 uses patch embedding, not token embedding | Already fixed — uses `register_forward_hook` instead |
| "Could not estimate the number of tokens" | HF Trainer can't compute FLOPs for patch-based model | Harmless — ignore |
| OOM on Stage 2 (8192 context) | 4x memory vs Stage 1 | Enable FSDP + gradient checkpointing, reduce batch to 4 |
| Benchmark config not found | Path resolves relative to training dir | Script auto-searches `evaluation/configs/` as fallback |

### Environment Variables

```bash
# HuggingFace
export HF_HOME=/group-volume/hf_cache       # Custom cache directory
export HF_HUB_OFFLINE=1                      # Offline mode

# Proxy (for restricted networks)
export HTTPS_PROXY=http://proxy:8080
export HTTP_PROXY=http://proxy:8080

# NCCL (for multi-node debugging)
export NCCL_DEBUG=INFO                        # Verbose NCCL logs
export NCCL_SOCKET_IFNAME=eth0               # Network interface
```

### mpirun Launcher

The `train.sh` launcher auto-detects the environment:

| Condition | Mode | Command |
|-----------|------|---------|
| Hostfile exists | Multi-node mpirun | `mpirun -np {TOTAL_GPUS} ... python3 train_chronos2.py` |
| Multiple GPUs | Single-node torchrun | `torchrun --nproc_per_node={N} train_chronos2.py` |
| Single GPU | Direct | `python3 train_chronos2.py` |

Hostfile location: `/horovod/generated/hostfile` (default, customizable via `HOSTFILE` env var).

---

## File Reference

```
scripts/forecasting/
├── training/
│   ├── train_chronos2.py        # Main Chronos-2 training script (~1900 lines)
│   ├── train.py                 # Chronos (T5/GPT2) training script
│   ├── train.sh                 # mpirun/torchrun auto-launcher
│   └── configs/
│       ├── chronos2-test.yaml          # Quick test (2L/256d, 200 steps)
│       ├── chronos2-base.yaml          # 120M Stage 1 (2048 ctx)
│       ├── chronos2-base-stage2.yaml   # 120M Stage 2 (8192 ctx)
│       ├── chronos2-small.yaml         # 28M pretraining
│       ├── chronos2-finetune.yaml      # Fine-tune from pretrained
│       ├── chronos-t5-*.yaml           # Original Chronos configs
│       └── chronos-gpt2.yaml
├── evaluation/
│   ├── evaluate.py              # Standalone evaluation (WQL/MASE)
│   ├── download_eval_datasets.py # Offline dataset downloader
│   ├── agg-relative-score.py    # Aggregate relative scoring
│   ├── configs/
│   │   ├── lite-benchmark.yaml  # 5 datasets (training-time eval)
│   │   ├── zero-shot.yaml       # 27 unseen datasets
│   │   └── in-domain.yaml       # 15 in-training datasets
│   └── results/                 # Pre-computed baseline results
├── kernel-synth.py              # KernelSynth GP data generator
└── validate_datasets.py         # Dataset integrity checker
```
