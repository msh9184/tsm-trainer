# APC Occupancy Detection with MantisV2

Binary occupancy detection (occupied/empty) from SmartThings IoT sensor data
using the [MantisV2](https://arxiv.org/abs/2602.17868) pretrained time series
classification foundation model (4.2M parameters).

## Task Overview

| Item | Detail |
|------|--------|
| **Task** | Binary classification: occupied (1) vs empty (0) |
| **Input** | Multivariate 5-min binned sensor time series |
| **Sensors** | Motion, power, temperature (hallway + room), contact, energy |
| **Model** | MantisV2 (pretrained on CauKer synthetic data, 4.2M params) |
| **Data** | P4 period: Feb 09–23, 2026 (13.3 days, 3,840 timesteps) |
| **Labels** | Event-based: Feb 10–19 (9.1 days, 2,724 labeled timesteps) |

## Directory Structure

```
apc_occupancy/
├── README.md                          # This file
├── data/
│   ├── __init__.py
│   ├── preprocess.py                  # Load CSVs, NaN handling, channel filtering
│   └── dataset.py                     # Sliding window dataset for MantisV2
├── training/
│   ├── train.py                       # Main training/eval script
│   ├── train.sh                       # GPU launcher
│   ├── sweep_zeroshot.py              # Systematic zero-shot sweep
│   └── configs/
│       ├── p4-zeroshot.yaml           # P4: zero-shot (primary)
│       ├── p4-finetune-head.yaml      # P4: head-only fine-tuning
│       ├── p4-finetune-full.yaml      # P4: full fine-tuning
│       ├── zeroshot.yaml              # Legacy: zero-shot
│       ├── finetune-head.yaml         # Legacy: head-only
│       ├── finetune-adapter.yaml      # Legacy: adapter + head
│       └── finetune-full.yaml         # Legacy: full
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                     # Accuracy, F1, Precision, Recall, EER, AUC
│   └── evaluate.py                    # Standalone evaluation runner
└── visualization/
    ├── __init__.py
    ├── style.py                       # Publication-quality plot styling
    ├── embeddings.py                  # t-SNE, UMAP, PCA embedding plots
    ├── curves.py                      # ROC, DET, confusion matrix plots
    └── animation.py                   # Training progression GIF animation
```

## Setup

### 1. Install Dependencies

```bash
# MantisV2 (from source)
pip install git+https://github.com/vfeofanov/mantis.git

# Core dependencies
pip install scikit-learn pyyaml matplotlib seaborn scipy

# Optional: UMAP for embedding visualization (falls back to t-SNE if absent)
pip install umap-learn

# Optional: GIF animation for training progression
pip install imageio
```

### 2. Data

#### P4 Pipeline (Recommended)

Single sensor CSV + separate train/test label CSVs on the GPU server:

```
/group-volume/workspace/haeri.kim/Time-Series/data/SmartThings/Samsung_QST_Data/enter_leave/
├── merged_data_with_motion_count_0209_0223.csv   # Sensor data (02/09~02/23)
├── occupancy_events_0210_0219_processed_train.csv # Train labels (events)
└── occupancy_events_0210_0219_processed_test.csv  # Test labels (events)
```

**Key design**: The sensor file covers the full 13-day range. Labels only cover
the middle ~9 days. Context windows can reference sensor data outside the
labeled range (e.g., pre-label sensor data for context), preventing information
loss at label boundaries.

**Label format**: Event-based CSV with `time, Status, Head-count, At-home count`
columns. The preprocessing pipeline replays ENTER_HOME/LEAVE_HOME events
chronologically to generate per-timestep binary labels.

**Selected sensors** (6 channels):

| Short | CSV Column | Type |
|-------|------------|------|
| M | `d620900d_motionSensor` | Continuous (count) |
| P | `f2e891c6_powerMeter` | Continuous (W) |
| T1 | `d620900d_temperatureMeasurement` | Continuous (°C) |
| T2 | `ccea734e_temperatureMeasurement` | Continuous (°C) |
| C | `408981c2_contactSensor` | Binary (0/1) |
| E | `f2e891c6_energyMeter` | Continuous (Wh) |

#### Legacy Pipeline

Separate train/test sensor+label CSV pairs (per-timestep counts format).
Set `data_mode: legacy` in the config. See legacy config files for paths.

### 3. Download Pretrained Model

MantisV2 weights are hosted on HuggingFace Hub (`paris-noah/MantisV2`).
The model will be downloaded automatically on first use.

For offline environments (GPU server with network restrictions):

```bash
# Pre-download on a machine with internet
python -c "
from mantis.architecture import MantisV2
net = MantisV2(device='cpu')
net = net.from_pretrained('paris-noah/MantisV2')
print('Downloaded successfully')
"

# Set HuggingFace cache to a shared volume
export HF_HOME=/group-volume/hf_cache
```

## Quick Start

```bash
cd examples/classification/apc_occupancy

# P4 Zero-shot (primary experiment: 24h window, 4 sensors)
python training/train.py --config training/configs/p4-zeroshot.yaml

# P4 Head-only fine-tuning
python training/train.py --config training/configs/p4-finetune-head.yaml

# P4 Full fine-tuning
python training/train.py --config training/configs/p4-finetune-full.yaml

# Using the launcher script
bash training/train.sh --config training/configs/p4-zeroshot.yaml
bash training/train.sh --config training/configs/p4-finetune-head.yaml --device cpu
```

### CLI Override Options

```bash
# Override sensor channels (2-channel baseline)
python training/train.py --config training/configs/p4-zeroshot.yaml \
    --channels d620900d_motionSensor f2e891c6_powerMeter

# Override window length (must be multiple of 32)
python training/train.py --config training/configs/p4-zeroshot.yaml --seq-len 64

# Add hour-of-day cyclical features
python training/train.py --config training/configs/p4-zeroshot.yaml --add-time-features

# Disable visualization for faster iteration
python training/train.py --config training/configs/p4-zeroshot.yaml --no-viz

# Override device and seed
python training/train.py --config training/configs/p4-zeroshot.yaml --device cpu --seed 123
```

## Zero-shot Sweep

Systematically evaluate combinations of (seq_len, channels, time_features):

```bash
# Full sweep: 16 sensor combos × 4 seq_lens = 64 experiments
python training/sweep_zeroshot.py --config training/configs/p4-zeroshot.yaml

# Quick sweep: 5 key combos × 1 seq_len (phase 1 only)
python training/sweep_zeroshot.py --config training/configs/p4-zeroshot.yaml --quick

# Custom seq_lens
python training/sweep_zeroshot.py --config training/configs/p4-zeroshot.yaml \
    --seq-lens 64 288 512
```

**Outputs** (in `results/p4-sweep/`):
- `sweep_results.csv`: All experiment results (one row per classifier per config)
- `sweep_summary.json`: Best configurations per metric
- Console summary table with sorted results

**Sensor combinations**: M+P (base), then adding T1, T2, C, E in all
C(4,0)+C(4,1)+C(4,2)+C(4,3)+C(4,4) = 16 combinations.

## Approach

### Phase A: Zero-shot Classification

1. Load pretrained MantisV2 (layer 2, combined token → 512-dim per channel)
2. Process each sensor channel independently through the backbone
3. Concatenate per-channel embeddings → feature vector of `512 × n_channels` dimensions
4. Train simple classifiers (Nearest Centroid, Random Forest, SVM) on frozen features
5. Evaluate on held-out test period

This requires **no gradient computation** and runs in seconds.

#### How Zero-shot Classification Works

The term "zero-shot" refers to the **backbone** being used without any gradient
updates (pretrained weights only). The classification itself is **fully
supervised** — sklearn classifiers learn from labeled training embeddings:

```
1. Z_train = MantisV2.transform(X_train)   # Frozen backbone → (N_train, 7680)
2. Z_test  = MantisV2.transform(X_test)    # Frozen backbone → (N_test, 7680)
3. clf.fit(Z_train, y_train)               # Learns decision boundary from labels
4. y_pred  = clf.predict(Z_test)           # Classifies test embeddings
```

Each classifier learns to distinguish "occupied" from "empty" embeddings:
- **NearestCentroid**: Computes the centroid (mean) of each class in embedding
  space, assigns test points to the nearest class center.
- **RandomForest**: Builds an ensemble of decision trees that split on embedding
  features to separate classes.
- **SVM**: Finds the optimal hyperplane that maximally separates the two classes
  in embedding space (with RBF kernel for nonlinear boundaries).

The label information (`y_train`: 0=empty, 1=occupied) is explicitly provided
during `clf.fit()`. No unsupervised clustering or threshold tuning is involved.

### Phase B: Fine-tuning

Three fine-tuning strategies in order of parameter count:

| Mode | Backbone | Head | Adapter | Description |
|------|----------|------|---------|-------------|
| `head` | Frozen | Trained | — | Fastest, lowest risk of overfitting |
| `adapter_head` | Frozen | Trained | Trained | Learnable channel mixing |
| `full` | Trained | Trained | — | Highest capacity, needs careful LR |

The classification head architecture:
```
LayerNorm(in_dim) → Linear(in_dim, 100) → ReLU → Dropout(0.1) → Linear(100, 2)
```

### Window Length Selection

| seq_len | Time Span | Interpolation | Use Case |
|---------|-----------|---------------|----------|
| 64 | 5h 20m | 8× to 512 | Short context, time features required |
| 128 | 10h 40m | 4× to 512 | Half-day patterns |
| **288** | **24h** | **1.78× to 512** | **Full daily cycle (recommended)** |
| 512 | 42h 40m | None (native) | Maximum context, no interpolation |

The primary configuration uses `seq_len=288` (24 hours), which captures the
complete daily occupancy cycle (night sleep → morning departure → daytime
absence → evening return) with minimal interpolation distortion.

### Hour-of-Day Features

Optional cyclical time encoding (`add_time_features: true`) appends two
extra channels:
- `hour_sin = sin(2π × hour / 24)`
- `hour_cos = cos(2π × hour / 24)`

**When to use**:
- `seq_len=64` (5h): **Required** — window is too short to infer time of day
- `seq_len=128` (10h): Recommended — partially captures daily patterns
- `seq_len=288` (24h): Optional — daily cycle is inherent in the window
- `seq_len=512` (42h): Not needed — time information is implicit

## Configuration Reference

### Data Section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sensor_csv` | str | — | Path to sensor data CSV |
| `label_csv` | str | — | Path to train label CSV |
| `test_label_csv` | str | — | Path to test label CSV |
| `label_format` | str | counts | Label format: "events" or "counts" |
| `initial_occupancy` | int | 0 | Initial occupancy for event-based labels |
| `nan_threshold` | float | 0.5 | Drop channels with NaN fraction above this |
| `binarize` | bool | true | Convert counts to binary (>0 → 1) |
| `channels` | list | null | Explicit channel list (null = auto-select) |
| `exclude_channels` | list | [] | Channel names to always exclude |
| `add_time_features` | bool | false | Append hour_sin/hour_cos channels |

### Top-Level

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | str | zeroshot | Training mode: "zeroshot" or "finetune" |
| `data_mode` | str | p4 | Data pipeline: "p4" (new) or "legacy" |
| `seed` | int | 42 | Random seed |
| `output_dir` | str | results | Output directory |

### Dataset Section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `seq_len` | int | 288 | Window length (must be multiple of 32) |
| `stride` | int | 1 | Stride between windows |
| `target_seq_len` | int | 512 | Resize windows to this length (multiple of 32) |

### Model Section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `pretrained_name` | str | paris-noah/MantisV2 | HuggingFace model ID |
| `return_transf_layer` | int | 2 | Transformer layer for features (-1 = last) |
| `output_token` | str | combined | Token type: cls_token, mean_token, combined |
| `device` | str | cuda | Device: cuda or cpu |

### Fine-tuning Section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fine_tuning_type` | str | head | full, head, adapter_head, scratch |
| `num_epochs` | int | 500 | Training epochs |
| `batch_size` | int | 256 | Batch size |
| `learning_rate` | float | 2e-4 | Base learning rate |
| `label_smoothing` | float | 0.1 | Cross-entropy label smoothing |
| `adapter_type` | str | null | Channel adapter: linear, pca, svd, var, null |
| `adapter_new_channels` | int | 5 | Target channel count for adapter |

### Visualization Section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | true | Enable/disable all visualizations |
| `methods` | list | ["tsne", "pca"] | Dimensionality reduction methods |
| `save_format` | list | ["png", "pdf"] | Output formats for saved plots |
| `dpi` | int | 300 | Resolution for raster outputs |
| `snapshot_interval` | int | 0 | Embedding snapshots every N epochs (0=disabled) |
| `create_gif` | bool | true | Create animated GIF from snapshots |
| `gif_fps` | int | 2 | GIF frames per second |

> **Note on `snapshot_interval`**: When set to a value > 0, the fine-tuning
> loop calls `model.fit(num_epochs=1)` per epoch to capture intermediate
> embeddings. This causes MantisTrainer to **recreate the optimizer each call**,
> resetting momentum and LR schedule. For this reason, `snapshot_interval`
> defaults to `0` in all fine-tuning configs. Enable it only when you need
> training progression visualization and understand the trade-off (constant LR
> is forced automatically in snapshot mode to reduce the impact). For production
> training, always keep `snapshot_interval: 0`.

## Multivariate and Covariate Handling

MantisV2 processes each sensor channel **independently** through a shared backbone
(channel independence / CI strategy). For `n_channels` sensors with hidden dimension
`d`, the final embedding is `n_channels × d` dimensional.

### Embedding Dimensions by Channel Count

| Channels | output_token | Per-Channel | Total Embedding | Head Input |
|----------|-------------|-------------|-----------------|------------|
| 2 (M+P) | combined | 512 | 1,024 | LayerNorm(1024) |
| 4 (M+P+T1+T2) | combined | 512 | 2,048 | LayerNorm(2048) |
| 6 (all) | combined | 512 | 3,072 | LayerNorm(3072) |

## Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification accuracy |
| **F1 Score** | Harmonic mean of precision and recall |
| **Precision** | True positives / (true positives + false positives) |
| **Recall** | True positives / (true positives + false negatives) |
| **EER** | Equal Error Rate (where FPR = FNR). Requires probability output. |
| **AUC** | Area Under the ROC Curve. Requires probability output. |

## Output

### Evaluation Report

Results are saved to `{output_dir}/eval_report.json`:

```json
{
  "mode": "zeroshot",
  "data_mode": "p4",
  "model": {
    "pretrained": "paris-noah/MantisV2",
    "return_transf_layer": 2,
    "output_token": "combined"
  },
  "dataset": {"seq_len": 288, "stride": 1, "target_seq_len": 512},
  "results": {
    "svm": {
      "accuracy": 0.85,
      "f1": 0.72,
      ...
    }
  }
}
```

### Predictions

Saved to `{output_dir}/predictions_{classifier}.npz` containing `y_true`,
`y_pred`, and (when available) `y_prob`. These can be used for standalone
re-evaluation:

```bash
python evaluation/evaluate.py --predictions results/p4-zeroshot/predictions_svm.npz
```

### Visualization Outputs

When `visualization.enabled: true`, the following plots are generated in
`{output_dir}/plots/`:

| File | Description |
|------|-------------|
| `embeddings_train_test_tsne.{png,pdf}` | Train vs test embedding comparison (shared t-SNE space) |
| `embeddings_methods.{png,pdf}` | Side-by-side PCA / t-SNE (/ UMAP) of test embeddings |
| `roc_curve_{clf}.{png,pdf}` | ROC curve with AUC annotation (per classifier) |
| `det_curve_{clf}.{png,pdf}` | DET curve with EER point on normal-deviate scale (per classifier) |
| `confusion_matrix_{clf}.{png,pdf}` | Confusion matrix heatmap with counts and percentages (per classifier) |

For fine-tuning with `snapshot_interval > 0`:

| File | Description |
|------|-------------|
| `snapshots/epoch_NNNN.png` | PCA embedding snapshot at epoch N |
| `training_progression.gif` | Animated GIF of embedding evolution |

## Troubleshooting

### CUDA Out of Memory
Reduce `batch_size` in the config or use `--device cpu` for debugging.

### MantisV2 Import Error
Ensure `mantis-tsfm` is installed:
```bash
pip install git+https://github.com/vfeofanov/mantis.git
```

### No Overlapping Timestamps
Check that `sensor_csv` and `label_csv` cover the same time range.
The preprocessing performs an inner join on timestamps.

### High NaN Fraction
If too many channels are dropped, lower `nan_threshold` or explicitly
list desired channels in `channels`.

### Headless Server (No Display)
The visualization module auto-detects headless environments and uses the
`Agg` backend. No manual configuration is needed on GPU servers.
