# TSM-Trainer: Chronos-2 Training Framework

Training framework for **Chronos-2** time series foundation models on multi-node GPU clusters.

## Features

- **Multi-node distributed training** via mpirun + DDP/FSDP (tested on 64x A100 80GB)
- **Lazy Arrow data loading** — zero upfront loading, on-demand reads from HuggingFace datasets
- **FSDP support** with auto-wrapping of `Chronos2EncoderBlock`
- **Gradient checkpointing** via monkey-patched activation checkpointing
- **Two-stage training** with automatic context length transition (2048 -> 8192)
- **Lite benchmark evaluation** during training (WQL/MASE on Monash datasets)
- **Advanced monitoring** — sparklines, convergence analysis, gradient health, GPU utilization
- **Top-K checkpoint management** by benchmark WQL score
- **TensorBoard integration** for loss, learning rate, and benchmark metrics

## Quick Start

### 1. Setup

```bash
# Install from project root
pip install -e ".[dev]"
```

### 2. Download Evaluation Datasets (for benchmark during training)

```bash
# Download lite benchmark datasets (run on a machine with internet access)
python scripts/evaluation/download_eval_datasets.py \
    --config scripts/evaluation/configs/lite-benchmark.yaml

# With proxy (for restricted networks)
HTTPS_PROXY=http://proxy:8080 python scripts/evaluation/download_eval_datasets.py \
    --config scripts/evaluation/configs/lite-benchmark.yaml
```

### 3. Run Training

```bash
# Quick test (single GPU)
python scripts/training/train_chronos2.py \
    --config scripts/training/configs/chronos2-test.yaml

# Multi-GPU (8x A100, single node via torchrun)
torchrun --nproc_per_node=8 scripts/training/train_chronos2.py \
    --config scripts/training/configs/chronos2-base.yaml

# Multi-node (64x A100 via mpirun)
bash scripts/training/train.sh --config configs/chronos2-base.yaml
```

### 4. Monitor Training

```bash
# TensorBoard
tensorboard --logdir output/chronos2-base-stage1

# Tail logs (rank 0 only prints detailed info)
tail -f output/chronos2-base-stage1/training.log
```

## Configurations

| Config | Model | GPUs | Context | Batch | Steps | Use Case |
|--------|-------|------|---------|-------|-------|----------|
| `chronos2-test.yaml` | 2L/256d (tiny) | 1-64 | 512 | 16 | 200 | Pipeline verification |
| `chronos2-base.yaml` | 12L/768d (120M) | 64 | 2048 | 12 | 200K | Stage 1 pretraining |
| `chronos2-base-stage2.yaml` | 12L/768d (120M) | 64 | 8192 | 4 | 50K | Stage 2 (from Stage 1 checkpoint) |
| `chronos2-small.yaml` | 8L/512d (28M) | 64 | 2048 | 16 | 200K | Small model pretraining |
| `chronos2-finetune.yaml` | 12L/768d (120M) | 8-64 | 2048 | 8 | 50K | Fine-tuning from pretrained |

## Training Modes

### Pretraining (from scratch)

```yaml
random_init: true
```

### Fine-tuning (from pretrained)

```yaml
random_init: false
model_id: "amazon/chronos-2"  # or local path
```

### Stage 2 (context extension)

```yaml
random_init: false
model_id: "./output/chronos2-base-stage1/final-checkpoint"
context_length: 8192
```

The training script automatically detects context length transitions and updates the model's internal config to prevent silent truncation.

## Data Loading

The training pipeline uses **LazyHFTaskSource** for memory-efficient data loading:

- HuggingFace Arrow datasets are opened via memory-mapping (no data loaded into RAM)
- Each training sample is read on-demand from disk
- LRU cache (50K entries) provides locality for frequently accessed series
- Short series are automatically skipped with retry logic
- All 64 ranks read independently — no rank-0 bottleneck

```
Data Loading Evolution:
  v0: All ranks each loading full dataset -> hang (N-fold redundant I/O)
  v1: Rank-0 only + broadcast -> still hangs (900K individual conversions)
  v2: LazyHFTaskSource -> zero upfront loading, on-demand Arrow reads
```

### Training Data Format

Training data must be in HuggingFace `datasets` format (Arrow IPC):

```python
# Expected columns:
#   - target: Sequence[float]         (required)
#   - past_feat_dynamic_real: Sequence[Sequence[float]]  (optional, covariates)

# Save as HF dataset:
import datasets
ds = datasets.Dataset.from_dict({"target": [series1, series2, ...]})
ds.save_to_disk("/path/to/dataset")
```

## Monitoring & Evaluation

### Per-Step Log Line

```
Step 1,000/200,000 | loss=9.6887 | ema=9.7102 | best=9.6432 | lr=4.99e-05 | mem=42.1/80GB | 9.3 stp/s | 7131 smp/s | eta=5h 52m | gnorm=1.243
```

### Periodic Health Report

Every `health_report_interval` steps, a comprehensive boxed report is printed with:

- Loss analysis (EMA fast/slow, convergence rate, variance, sparkline)
- Gradient health (norms, clip rate, explosion detection)
- Learning rate schedule with sparkline
- GPU memory with visual utilization bar
- Throughput (steps/sec, samples/sec, percentile latencies, ETA)
- Data pipeline stats (cache hit rate, short series skips)
- Lite benchmark results (if enabled)

### Lite Benchmark

When `benchmark_config` is set, the training script periodically evaluates forecast quality:

```yaml
benchmark_config: "configs/lite-benchmark.yaml"
benchmark_eval_steps: 10000      # Evaluate every 10K steps
benchmark_top_k_checkpoints: 3   # Keep top 3 checkpoints by WQL
```

Metrics logged to TensorBoard:
- `benchmark/avg_wql` — Average Weighted Quantile Loss
- `benchmark/avg_mase` — Average Mean Absolute Scaled Error
- `benchmark/{dataset}/wql` — Per-dataset WQL
- `benchmark/{dataset}/mase` — Per-dataset MASE

### Checkpoint Management

Three checkpoint retention strategies work together:
1. **HF Trainer** `save_total_limit`: Keeps N most recent checkpoints
2. **Benchmark top-K**: Keeps top K checkpoints by WQL score
3. **Final checkpoint**: Always saved at training end

## CLI Arguments

```
--config CONFIG           Path to YAML config (required)
--model-id MODEL_ID       HuggingFace model ID or local path
--random-init             Train from scratch
--no-random-init          Fine-tune from pretrained
--context-length N        Override context length
--prediction-length N     Override prediction length
--max-steps N             Override max training steps
--learning-rate LR        Override learning rate
--per-device-train-batch-size N  Override batch size
--gradient-accumulation-steps N  Override gradient accumulation
--output-dir DIR          Override output directory
--resume-from-checkpoint PATH    Resume from checkpoint
--max-train-series N      Limit training series (for testing)
--seed N                  Override random seed
```

## Architecture Reference

Chronos-2 is an encoder-only transformer with:

```
Input -> RobustScaling (arcsinh) -> Patching -> Embedding
  -> N x [TimeSelfAttention (RoPE) -> GroupSelfAttention -> FeedForward]
  -> QuantileHead -> Q quantile forecasts
```

| Component | Description |
|-----------|-------------|
| TimeSelfAttention | Standard self-attention along temporal axis with RoPE |
| GroupSelfAttention | Attention along batch axis (enables ICL across related series) |
| QuantileHead | Direct multi-step quantile regression output |
| RobustScaling | sinh^{-1} normalization for outlier robustness |

### Model Sizes

| Variant | Params | Layers | d_model | Heads | Quantiles |
|---------|--------|--------|---------|-------|-----------|
| Base | 120M | 12 | 768 | 12 | 21 |
| Small | 28M | 8 | 512 | 8 | 13 |

## File Structure

```
scripts/training/
  train_chronos2.py          Main training script
  train.sh                   mpirun launcher (auto-detects hostfile)
  TRAINING_GUIDE_CHRONOS2.md Detailed training guide
  README.md                  This file
  configs/
    chronos2-test.yaml       Quick verification test
    chronos2-base.yaml       120M Stage 1 pretraining
    chronos2-base-stage2.yaml 120M Stage 2 (context extension)
    chronos2-small.yaml      28M pretraining
    chronos2-finetune.yaml   Fine-tuning from pretrained

scripts/evaluation/
  evaluate.py                Standalone evaluation script
  download_eval_datasets.py  Download datasets for offline use
  configs/
    lite-benchmark.yaml      5 datasets for training-time evaluation
    zero-shot.yaml           27 zero-shot benchmark datasets
    in-domain.yaml           15 in-domain benchmark datasets
```
