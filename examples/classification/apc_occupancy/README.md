# APC Occupancy Detection with MantisV2

Binary occupancy detection (occupied/empty) from SmartThings IoT sensor data
using the [MantisV2](https://arxiv.org/abs/2602.17868) pretrained time series
classification foundation model (4.2M parameters).

## Task Overview

| Item | Detail |
|------|--------|
| **Task** | Binary classification: occupied (1) vs empty (0) |
| **Input** | Multivariate 5-min binned sensor time series |
| **Sensors** | Temperature, humidity, illuminance, motion, contact, presence, switch, power, etc. |
| **Model** | MantisV2 (pretrained on CauKer synthetic data, 4.2M params) |
| **Train** | Jan 31 – Feb 02, 2026 (3 days, ~864 time steps) |
| **Test** | Jan 26, 2026 (1 day, ~288 time steps) |

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
│   └── configs/
│       ├── zeroshot.yaml              # Phase A: zero-shot with sklearn classifiers
│       ├── finetune-head.yaml         # Phase B: head-only fine-tuning
│       ├── finetune-adapter.yaml      # Phase B: adapter + head
│       └── finetune-full.yaml         # Phase B: full fine-tuning
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

The data is expected at the following paths on the GPU server:

```
/group-volume/workspace/haeri.kim/Time-Series/data/SmartThings/Samsung_QST_Data/
├── user01_0125_0126/output/
│   ├── merged_processed_data.csv      # Test sensor data
│   └── occupancy_counts.csv           # Test labels
└── user01_0131_0202/output/
    ├── merged_processed_data.csv      # Train sensor data
    └── occupancy_counts.csv           # Train labels
```

**Data format**:
- `merged_processed_data.csv`: 5-min binned sensor readings with `time` column + sensor columns
- `occupancy_counts.csv`: 5-min binned occupancy counts with `time,occupancy` columns

To use different data paths, edit the `data` section in the YAML config files.

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

# Phase A: Zero-shot evaluation
python training/train.py --config training/configs/zeroshot.yaml

# Phase B: Head-only fine-tuning
python training/train.py --config training/configs/finetune-head.yaml

# Phase B: Full fine-tuning
python training/train.py --config training/configs/finetune-full.yaml

# Using the launcher script
bash training/train.sh --config training/configs/zeroshot.yaml
bash training/train.sh --config training/configs/finetune-head.yaml --device cpu
```

### Override Options

```bash
# Override device
python training/train.py --config training/configs/zeroshot.yaml --device cpu

# Override random seed
python training/train.py --config training/configs/zeroshot.yaml --seed 123
```

## Approach

### Phase A: Zero-shot Classification

1. Load pretrained MantisV2 (layer 2, combined token → 512-dim per channel)
2. Process each sensor channel independently through the backbone
3. Concatenate per-channel embeddings → feature vector of `512 × n_channels` dimensions
4. Train simple classifiers (Nearest Centroid, Random Forest, SVM) on frozen features
5. Evaluate on held-out test day

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

### Phase C: Advanced (Future Work)

- MOMENT (385M) + LoRA fine-tuning
- Forecasting-as-extractor (arXiv:2510.26777)
- Multi-scale self-ensembling

## Configuration Reference

### Data Section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sensor_csv` | str | — | Path to train sensor CSV |
| `label_csv` | str | — | Path to train occupancy counts CSV |
| `test_sensor_csv` | str | — | Path to test sensor CSV |
| `test_label_csv` | str | — | Path to test occupancy counts CSV |
| `nan_threshold` | float | 0.5 | Drop channels with NaN fraction above this |
| `binarize` | bool | true | Convert counts to binary (>0 → 1) |
| `channels` | list | null | Explicit channel list (null = auto-select) |
| `exclude_channels` | list | [] | Channel names to always exclude |

### Dataset Section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `seq_len` | int | 64 | Window length (must be multiple of 32) |
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

### Multivariate Mode (Default)
All sensors are treated as equal input channels. No distinction between "target"
and "auxiliary" variables.

### Covariate Support (Future)
Some sensor channels may have future values available at inference time (e.g.,
scheduled events, weather forecasts). This can be handled by:
1. Designating covariate channels in the config (`channels` field)
2. Using a custom head that weights covariate features differently
3. Extending to a model that natively supports covariates (e.g., Chronos-2)

Currently, all channels are treated identically by MantisV2. Covariate-aware
classification is planned for Phase C.

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
  "model": {
    "pretrained": "paris-noah/MantisV2",
    "return_transf_layer": 2,
    "output_token": "combined"
  },
  "results": {
    "svm": {
      "accuracy": 0.85,
      "f1": 0.72,
      "precision": 0.78,
      "recall": 0.67,
      "eer": 0.18,
      "eer_threshold": 0.42,
      "roc_auc": 0.92,
      "confusion_matrix": [[150, 30], [20, 88]],
      "n_samples": 288,
      "class_distribution": {"0": 180, "1": 108}
    }
  }
}
```

### Predictions

Saved to `{output_dir}/predictions_{classifier}.npz` containing `y_true`,
`y_pred`, and (when available) `y_prob`. These can be used for standalone
re-evaluation:

```bash
python evaluation/evaluate.py --predictions results/zeroshot/predictions_svm.npz
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
