# Chronos-2 Multi-Node Training Script
# Supports: mpirun (OpenMPI), torchrun, single-GPU
#
# Usage:
#   Single GPU:   python scripts/training/train_chronos2.py --config configs/chronos2-test.yaml
#   Multi-GPU:    torchrun --nproc_per_node=8 scripts/training/train_chronos2.py --config configs/chronos2-base.yaml
#   Multi-Node:   bash scripts/training/train.sh --config configs/chronos2-base.yaml
#                 (uses mpirun internally, see train.sh)
#
# Modes:
#   Pretraining (from scratch):  set random_init: true in config
#   Fine-tuning (from pretrained): set random_init: false, model_id: "amazon/chronos-2"

# ===========================================================================
# CRITICAL: OpenMPI environment variable mapping
# Must happen BEFORE any torch/transformers imports so that HuggingFace
# Trainer correctly detects the distributed environment.
# ===========================================================================
import os
import socket

def _setup_mpi_env():
    """Map OpenMPI environment variables to PyTorch/HuggingFace expected names.

    OpenMPI sets:
        OMPI_COMM_WORLD_RANK, OMPI_COMM_WORLD_SIZE, OMPI_COMM_WORLD_LOCAL_RANK
    PyTorch/HuggingFace expects:
        RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    """
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        os.environ.setdefault("RANK", os.environ["OMPI_COMM_WORLD_RANK"])
        os.environ.setdefault("WORLD_SIZE", os.environ["OMPI_COMM_WORLD_SIZE"])
        os.environ.setdefault("LOCAL_RANK", os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

        # MASTER_ADDR: resolve from hostfile or use first node hostname
        if "MASTER_ADDR" not in os.environ:
            # Try reading from hostfile
            hostfile = os.environ.get("HOSTFILE", "/horovod/generated/hostfile")
            if os.path.exists(hostfile):
                with open(hostfile) as f:
                    first_host = f.readline().strip().split()[0]
                    try:
                        os.environ["MASTER_ADDR"] = socket.gethostbyname(first_host)
                    except socket.gaierror:
                        os.environ["MASTER_ADDR"] = first_host
            else:
                os.environ["MASTER_ADDR"] = "localhost"

        os.environ.setdefault("MASTER_PORT", "29500")

_setup_mpi_env()

# ===========================================================================
# Standard imports (after env setup)
# ===========================================================================
import argparse
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import yaml
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
from chronos.chronos2.model import Chronos2Model
from chronos.chronos2.trainer import Chronos2Trainer, EvaluateAndSaveFinalStepCallback

_is_rank_zero = int(os.environ.get("RANK", 0)) == 0
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO if _is_rank_zero else logging.WARNING,
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Distributed utilities
# ===========================================================================

def get_rank() -> int:
    return int(os.environ.get("RANK", 0))

def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))

def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))

def is_main_process() -> bool:
    
    return get_rank() == 0

def setup_distributed_seeds(seed: int = 42):
    """Set rank-aware random seeds to ensure each DDP process samples different data.

    This is CRITICAL for correctness: Chronos2Dataset uses np.random.randint()
    for random task sampling. Without rank-aware seeding, all DDP ranks would
    produce identical batches, wasting computation.
    """
    rank = get_rank()
    rank_seed = seed + rank

    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)

    if is_main_process():
        logger.info(f"Set rank-aware seeds: base={seed}, rank_seed={rank_seed}")


def setup_cuda_device():
    """Assign CUDA device based on LOCAL_RANK for multi-GPU training."""
    local_rank = get_local_rank()
    if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
        torch.cuda.set_device(local_rank)
        if is_main_process():
            logger.info(f"CUDA device set to {local_rank} "
                        f"({torch.cuda.get_device_name(local_rank)})")


def log_distributed_info():
    """Log distributed training configuration."""
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    master_addr = os.environ.get("MASTER_ADDR", "N/A")
    master_port = os.environ.get("MASTER_PORT", "N/A")

    if is_main_process():
        logger.info("=" * 70)
        logger.info("Distributed Training Configuration")
        logger.info("=" * 70)
        logger.info(f"  World size:    {world_size}")
        logger.info(f"  Num nodes:     {world_size // max(torch.cuda.device_count(), 1)}")
        logger.info(f"  GPUs per node: {torch.cuda.device_count()}")
        logger.info(f"  Master addr:   {master_addr}")
        logger.info(f"  Master port:   {master_port}")
        logger.info(f"  Backend:       nccl (expected)")
        logger.info("=" * 70)

    # All ranks log their assignment
    logger.info(f"[Rank {rank}] local_rank={local_rank}, "
                f"host={socket.gethostname()}, "
                f"cuda_device={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}")


# ===========================================================================
# Data Loading Utilities
# ===========================================================================

def _open_hf_dataset(dataset_path: str, split: str = "train"):
    """Open a HuggingFace dataset and return (dataset, format_type).

    Returns the HF Dataset object without loading data into memory.
    This allows efficient random-access via Arrow columnar reads.
    """
    import datasets as hf_datasets

    path = Path(dataset_path)

    if (path / "dataset_dict.json").exists():
        ds_dict = hf_datasets.load_from_disk(str(path))
        if split in ds_dict:
            return ds_dict[split], "hf"
        available = list(ds_dict.keys())
        logger.warning(f"Split '{split}' not found in {path}. "
                       f"Available: {available}. Using '{available[0]}'.")
        return ds_dict[available[0]], "hf"
    elif (path / "dataset_info.json").exists():
        return hf_datasets.load_from_disk(str(path)), "hf"
    else:
        return path, "gluonts"


# ---------------------------------------------------------------------------
# LazyHFTaskSource: Arrow-native lazy task provider (ASR-style)
# ---------------------------------------------------------------------------
# Instead of materializing 900K+ series into list[dict] → list[tuple] upfront,
# this class wraps HF Arrow datasets and provides on-demand task conversion.
# Only the tasks actually sampled during training are read from disk.
# ---------------------------------------------------------------------------

class LazyHFTaskSource:
    """Lazy task provider backed by HuggingFace Arrow datasets.

    Provides list-like random access to tasks (tuples as expected by
    Chronos2Dataset._construct_slice) without loading all data upfront.

    Each task is read from Arrow and converted on first access, then cached
    in an LRU-style cache for locality (training often re-samples recent tasks).
    """

    def __init__(
        self,
        hf_datasets_list: list,
        index_maps: list[np.ndarray],
        prediction_length: int,
        min_past: int,
    ):
        """
        Parameters
        ----------
        hf_datasets_list : list of HF Dataset objects
            The opened Arrow datasets (not loaded into memory).
        index_maps : list of np.ndarray
            For each HF dataset, the array of global→local index mappings.
            index_maps[ds_idx][local_pos] = row index into hf_datasets_list[ds_idx].
        prediction_length : int
        min_past : int
        """
        self._hf_datasets = hf_datasets_list
        self._index_maps = index_maps
        self._prediction_length = prediction_length
        self._min_past = min_past

        # Build cumulative offsets for global→(ds_idx, local_pos) mapping
        self._offsets = []  # (start, end, ds_idx)
        total = 0
        for ds_idx, idx_map in enumerate(index_maps):
            n = len(idx_map)
            self._offsets.append((total, total + n, ds_idx))
            total += n
        self._total_len = total

        # LRU cache for converted tasks
        self._cache_max = 50000  # cache up to 50K tasks (~reasonable memory)
        self._cache: dict[int, tuple] = {}
        self._cache_order: list[int] = []  # tracks access order for LRU
        self._lock = threading.Lock()  # protects cache mutations

        # Stats for logging
        self._cache_hits = 0
        self._cache_misses = 0
        self._short_series_skips = 0

    def __len__(self):
        return self._total_len

    def _global_to_local(self, global_idx: int):
        """Map global index to (ds_idx, row_in_hf_dataset)."""
        for start, end, ds_idx in self._offsets:
            if start <= global_idx < end:
                local_pos = global_idx - start
                row_idx = self._index_maps[ds_idx][local_pos]
                return ds_idx, int(row_idx)
        raise IndexError(f"Global index {global_idx} out of range [0, {self._total_len})")

    def _read_and_convert(self, ds_idx: int, row_idx: int) -> tuple:
        """Read a single row from Arrow and convert to task tuple.

        This is the on-demand equivalent of:
        1. load_hf_dataset_as_dicts (one row) → dict
        2. validate_and_prepare_single_dict_task → tuple
        """
        from chronos.chronos2.dataset import validate_and_prepare_single_dict_task

        ds = self._hf_datasets[ds_idx]
        row = ds[row_idx]  # Single Arrow row read — fast for one row

        target = np.array(row["target"], dtype=np.float32)
        task_dict: dict = {"target": target}

        # Handle covariates if present
        if "past_feat_dynamic_real" in ds.column_names:
            cov_data = row.get("past_feat_dynamic_real")
            if cov_data is not None:
                covariates = np.array(cov_data, dtype=np.float32)
                if covariates.ndim == 2 and covariates.shape[0] > 0:
                    past_covariates = {}
                    for j in range(covariates.shape[0]):
                        past_covariates[f"covariate_{j}"] = covariates[j]
                    task_dict["past_covariates"] = past_covariates

        # Convert to the tuple format expected by Chronos2Dataset
        task_tuple = validate_and_prepare_single_dict_task(
            task_dict, idx=row_idx, prediction_length=self._prediction_length
        )
        return task_tuple

    def __getitem__(self, global_idx: int) -> tuple:
        """Get a task tuple by global index, with LRU caching.

        Uses iterative retry (max 50) instead of recursion for short series.
        Cache mutations are thread-safe via self._lock.
        """
        # Lock-free cache read (dict 'in' check is thread-safe in CPython)
        if global_idx in self._cache:
            self._cache_hits += 1
            return self._cache[global_idx]

        self._cache_misses += 1
        min_required = self._min_past + self._prediction_length

        # Iterative retry loop (replaces recursive call)
        current_idx = global_idx
        for _attempt in range(50):
            ds_idx, row_idx = self._global_to_local(current_idx)
            task_tuple = self._read_and_convert(ds_idx, row_idx)

            task_context = task_tuple[0]
            if task_context.shape[-1] >= min_required:
                break  # valid series found
            # Series too short — try a random different one
            self._short_series_skips += 1
            current_idx = np.random.randint(self._total_len)
        else:
            # All 50 attempts got short series (should be extremely rare)
            logger.warning(
                f"LazyHFTaskSource: all 50 retries got short series "
                f"(need {min_required}, got {task_context.shape[-1]}). "
                f"Using last attempt anyway."
            )

        # Thread-safe cache mutation
        with self._lock:
            if len(self._cache) >= self._cache_max:
                evict_count = self._cache_max // 4
                for old_key in self._cache_order[:evict_count]:
                    self._cache.pop(old_key, None)
                self._cache_order = self._cache_order[evict_count:]

            self._cache[current_idx] = task_tuple
            self._cache_order.append(current_idx)

        return task_tuple

    def log_stats(self):
        """Log cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(total, 1) * 100
        logger.info(f"LazyHFTaskSource: {self._total_len} series, "
                     f"cache={len(self._cache)}/{self._cache_max}, "
                     f"hit_rate={hit_rate:.1f}% ({self._cache_hits}/{total}), "
                     f"short_series_skips={self._short_series_skips}")


def _build_lazy_task_source(
    data_paths: list[str],
    probabilities: list[float],
    split: str,
    max_total_series: Optional[int],
    prediction_length: int,
    min_past: int,
    shuffle_seed: int = 42,
) -> LazyHFTaskSource:
    """Build a LazyHFTaskSource from multiple HF datasets.

    All ranks open the same Arrow datasets independently (memory-mapped,
    no data loaded into RAM). Index shuffling is deterministic so all
    ranks see the same global task ordering.
    """
    hf_datasets_list = []
    index_maps = []

    rng = np.random.default_rng(shuffle_seed)

    for ds_idx, (path, prob) in enumerate(zip(data_paths, probabilities)):
        ds, fmt = _open_hf_dataset(path, split)

        if fmt != "hf":
            raise ValueError(
                f"Lazy loading requires HuggingFace format datasets. "
                f"Got '{fmt}' for {path}. Convert to HF format first."
            )

        ds_size = len(ds)
        if max_total_series is not None:
            max_for_ds = min(int(max_total_series * prob), ds_size)
        else:
            max_for_ds = ds_size

        # Generate shuffled indices (deterministic across all ranks)
        perm = rng.permutation(ds_size)[:max_for_ds]

        hf_datasets_list.append(ds)
        index_maps.append(perm)

        if is_main_process():
            logger.info(f"  Dataset '{Path(path).name}': {max_for_ds}/{ds_size} series "
                         f"(weight={prob:.2f}) [lazy Arrow access]")

    source = LazyHFTaskSource(
        hf_datasets_list=hf_datasets_list,
        index_maps=index_maps,
        prediction_length=prediction_length,
        min_past=min_past,
    )

    if is_main_process():
        logger.info(f"LazyHFTaskSource ready: {len(source)} total series "
                     f"(zero upfront loading, on-demand Arrow reads)")

    return source


# ---------------------------------------------------------------------------
# Legacy eager loading (kept for validation data which is typically small)
# ---------------------------------------------------------------------------

def load_hf_dataset_as_dicts(
    dataset_path: str,
    split: str = "train",
    max_series: Optional[int] = None,
    prediction_length: int = 64,
) -> list[dict]:
    """Load a HuggingFace dataset into Chronos-2 dict format (eager).

    Used for validation data (small, loaded once). For training data,
    use _build_lazy_task_source() instead.
    """
    ds_or_path, fmt = _open_hf_dataset(dataset_path, split)

    if fmt == "gluonts":
        try:
            from gluonts.dataset.common import FileDataset
            gluonts_ds = FileDataset(path=ds_or_path, freq="h")
            result = []
            for i, entry in enumerate(gluonts_ds):
                if max_series is not None and i >= max_series:
                    break
                target = np.array(entry["target"], dtype=np.float32)
                result.append({"target": target})
            logger.info(f"Loaded {len(result)} series from GluonTS: {ds_or_path}")
            return result
        except Exception as e:
            raise ValueError(f"Cannot load dataset at {ds_or_path}: {e}")

    ds = ds_or_path
    ds_total = len(ds)
    total = min(ds_total, max_series) if max_series is not None else ds_total

    if is_main_process():
        logger.info(f"  Loading {total} series from {Path(dataset_path).name} "
                     f"(total available: {ds_total})...")

    # For small datasets (validation), batch conversion is fine
    ds_subset = ds.select(range(total)) if total < ds_total else ds

    # Read entire target column at once
    load_start = time.time()
    all_targets = ds_subset["target"]
    if is_main_process():
        logger.info(f"    Target column read: {time.time() - load_start:.1f}s")

    has_covariates = "past_feat_dynamic_real" in ds_subset.column_names
    all_covariates = None
    if has_covariates:
        all_covariates = ds_subset["past_feat_dynamic_real"]

    # Convert to list of dicts (acceptable for small validation sets)
    result = []
    for i in range(total):
        target = np.array(all_targets[i], dtype=np.float32)
        task_dict: dict = {"target": target}

        if all_covariates is not None and all_covariates[i] is not None:
            covariates = np.array(all_covariates[i], dtype=np.float32)
            if covariates.ndim == 2 and covariates.shape[0] > 0:
                past_covariates = {}
                for j in range(covariates.shape[0]):
                    past_covariates[f"covariate_{j}"] = covariates[j]
                task_dict["past_covariates"] = past_covariates

        result.append(task_dict)

        if is_main_process() and total > 1000 and (i + 1) % (total // 10) == 0:
            logger.info(f"    Converting: {i + 1}/{total}")

    del all_targets
    if all_covariates is not None:
        del all_covariates

    logger.info(f"Loaded {len(result)} series from HF dataset: "
                f"{Path(dataset_path).name} (split={split})")
    return result


# ===========================================================================
# Model Creation
# ===========================================================================

def create_model_from_config(config: dict) -> Chronos2Model:
    """Create a Chronos2Model from scratch (random initialization)."""
    chronos_config = {
        "context_length": config["context_length"],
        "input_patch_size": config.get("input_patch_size", 16),
        "input_patch_stride": config.get("input_patch_stride", 16),
        "output_patch_size": config.get("output_patch_size", 16),
        "quantiles": config.get("quantiles", [
            0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
            0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
        ]),
        "use_reg_token": config.get("use_reg_token", True),
        "use_arcsinh": config.get("use_arcsinh", True),
        "max_output_patches": config.get("max_output_patches", 64),
        "time_encoding_scale": config.get("time_encoding_scale", config["context_length"]),
    }

    core_config = Chronos2CoreConfig(
        d_model=config.get("d_model", 768),
        d_kv=config.get("d_kv", 64),
        d_ff=config.get("d_ff", 3072),
        num_layers=config.get("num_layers", 12),
        num_heads=config.get("num_heads", 12),
        dropout_rate=config.get("dropout_rate", 0.1),
        layer_norm_epsilon=config.get("layer_norm_epsilon", 1e-6),
        initializer_factor=config.get("initializer_factor", 0.05),
        feed_forward_proj=config.get("feed_forward_proj", "relu"),
        rope_theta=config.get("rope_theta", 10000.0),
        attn_implementation=config.get("attn_implementation", "sdpa"),
        chronos_config=chronos_config,
    )

    model = Chronos2Model(core_config)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main_process():
        logger.info(f"Created Chronos2Model: {n_params / 1e6:.1f}M params (random init)")
        logger.info(f"  d_model={core_config.d_model}, layers={core_config.num_layers}, "
                     f"heads={core_config.num_heads}, d_ff={core_config.d_ff}")
        logger.info(f"  context_length={chronos_config['context_length']}, "
                     f"patch_size={chronos_config['input_patch_size']}, "
                     f"quantiles={len(chronos_config['quantiles'])}")
    return model


def load_pretrained_model(model_id: str) -> Chronos2Model:
    """Load a pretrained Chronos-2 model from HuggingFace or local path.

    Always loads to CPU first for memory-efficient distributed init.
    HF Trainer / FSDP will handle device placement.
    """
    from chronos import BaseChronosPipeline

    if is_main_process():
        logger.info(f"Loading pretrained model: {model_id}")

    pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu")
    model = pipeline.model
    n_params = sum(p.numel() for p in model.parameters())

    if is_main_process():
        logger.info(f"Loaded pretrained: {n_params / 1e6:.1f}M params")
        logger.info(f"  context_length={model.chronos_config.context_length}, "
                     f"quantiles={len(model.chronos_config.quantiles)}")
    return model


# ===========================================================================
# Gradient Checkpointing
# ===========================================================================

def enable_gradient_checkpointing(model: Chronos2Model):
    """Wrap each Chronos2EncoderBlock.forward() with activation checkpointing.

    Since Chronos2Model does not implement _supports_gradient_checkpointing,
    we monkey-patch each block's forward to use torch.utils.checkpoint.
    This trades ~20% speed for ~30-40% memory reduction.

    Must be called BEFORE Trainer wraps model with DDP/FSDP.
    """
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    blocks = model.encoder.block  # nn.ModuleList of Chronos2EncoderBlock
    n_wrapped = 0

    for i, block in enumerate(blocks):
        original_forward = block.forward

        # Use a factory to capture `original_forward` correctly in closure
        def make_wrapper(orig_fn):
            def wrapper(hidden_states, *, position_ids, attention_mask,
                        group_time_mask, output_attentions=False):
                # checkpoint requires a function that takes positional args.
                # We wrap keyword-only args into a helper.
                def run_block(hs, pos_ids, attn_mask, gt_mask, out_attn):
                    return orig_fn(
                        hs,
                        position_ids=pos_ids,
                        attention_mask=attn_mask,
                        group_time_mask=gt_mask,
                        output_attentions=out_attn,
                    )
                return torch_checkpoint(
                    run_block,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    group_time_mask,
                    output_attentions,
                    use_reentrant=False,
                )
            return wrapper

        block.forward = make_wrapper(original_forward)
        n_wrapped += 1

    if is_main_process():
        logger.info(f"Gradient checkpointing: wrapped {n_wrapped} Chronos2EncoderBlock(s)")

    # Enable gradient computation for inputs (required by checkpoint).
    # Chronos2Model doesn't implement get_input_embeddings() (it uses
    # input_patch_embedding, not standard token embeddings), so the default
    # model.enable_input_require_grads() fails with NotImplementedError.
    # Instead, we register a forward hook on input_patch_embedding to ensure
    # its output has requires_grad=True before entering checkpointed blocks.
    def _make_inputs_require_grads(module, input, output):
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)
        elif isinstance(output, tuple):
            for o in output:
                if isinstance(o, torch.Tensor) and o.is_floating_point():
                    o.requires_grad_(True)

    model.input_patch_embedding.register_forward_hook(_make_inputs_require_grads)


# ===========================================================================
# Training Callbacks
# ===========================================================================

class TrainingHealthMonitor:
    """Comprehensive training metrics tracker with advanced analytics.

    Tracks loss (multi-window EMA, variance, convergence), gradient health
    (norms, clipping, explosion detection), GPU utilization, throughput,
    learning rate schedule, and data pipeline stats.
    """

    def __init__(self, ema_alpha: float = 0.05):
        # Loss tracking
        self.loss_history: list[tuple[int, float]] = []  # (step, loss)
        self.loss_ema: float | None = None
        self.loss_ema_slow: float | None = None  # slower EMA for trend
        self._ema_alpha = ema_alpha
        self._ema_alpha_slow = 0.01
        self.best_loss: float = float("inf")
        self.best_loss_step: int = 0
        self.worst_loss: float = 0.0
        self.worst_loss_step: int = 0

        # Loss milestones
        self._loss_milestones: list[tuple[int, float]] = []  # (step, loss) at each 10% improvement

        # Gradient health
        self.grad_norms: list[float] = []
        self.grad_clip_count: int = 0
        self.grad_total_count: int = 0
        self.grad_nan_count: int = 0
        self.grad_explosion_count: int = 0  # norms > 10x average

        # Learning rate tracking
        self.lr_history: list[tuple[int, float]] = []

        # Throughput
        self.step_times: list[float] = []
        self._last_step_time: float | None = None
        self.total_samples: int = 0
        self._samples_per_step: int = 0  # effective_batch_size

        # GPU memory tracking over time
        self.gpu_mem_history: list[tuple[int, float]] = []  # (step, allocated_gb)

    def set_samples_per_step(self, effective_batch: int):
        self._samples_per_step = effective_batch

    def update_loss(self, loss: float, step: int):
        self.loss_history.append((step, loss))

        # Fast EMA
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            self.loss_ema = self._ema_alpha * loss + (1 - self._ema_alpha) * self.loss_ema

        # Slow EMA for macro trend
        if self.loss_ema_slow is None:
            self.loss_ema_slow = loss
        else:
            self.loss_ema_slow = self._ema_alpha_slow * loss + (1 - self._ema_alpha_slow) * self.loss_ema_slow

        if loss < self.best_loss:
            # Check for milestone (10% improvement)
            if self.best_loss < float("inf"):
                improvement = (self.best_loss - loss) / self.best_loss * 100
                if improvement >= 10:
                    self._loss_milestones.append((step, loss))
            self.best_loss = loss
            self.best_loss_step = step
        if loss > self.worst_loss:
            self.worst_loss = loss
            self.worst_loss_step = step

    def update_lr(self, lr: float, step: int):
        self.lr_history.append((step, lr))

    def update_grad_norm(self, grad_norm: float, max_grad_norm: float):
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            self.grad_nan_count += 1
            return
        self.grad_norms.append(grad_norm)
        self.grad_total_count += 1
        if grad_norm >= max_grad_norm * 0.99:
            self.grad_clip_count += 1
        # Explosion detection: norm > 10x recent average
        if len(self.grad_norms) > 10:
            recent_avg = np.mean(self.grad_norms[-11:-1])
            if recent_avg > 0 and grad_norm > recent_avg * 10:
                self.grad_explosion_count += 1

    def mark_step(self):
        now = time.time()
        if self._last_step_time is not None:
            self.step_times.append(now - self._last_step_time)
        self._last_step_time = now
        self.total_samples += self._samples_per_step

    def get_loss_trend(self, window: int = 10) -> tuple[str, float]:
        """Compare last `window` losses vs previous `window`. Returns (label, pct)."""
        if len(self.loss_history) < window * 2:
            return "insufficient data", 0.0
        recent = np.mean([l for _, l in self.loss_history[-window:]])
        previous = np.mean([l for _, l in self.loss_history[-2 * window:-window]])
        pct = (recent - previous) / abs(previous) * 100 if previous != 0 else 0
        if pct < -0.5:
            return "converging", pct
        elif pct < -0.1:
            return "decreasing", pct
        elif pct > 1.0:
            return "DIVERGING", pct
        elif pct > 0.1:
            return "increasing", pct
        return "stable", pct

    def get_loss_variance(self, window: int = 20) -> float:
        """Recent loss variance (indicator of training stability)."""
        if len(self.loss_history) < window:
            return 0.0
        return float(np.var([l for _, l in self.loss_history[-window:]]))

    def get_convergence_rate(self) -> float:
        """Loss reduction per 1000 steps (negative = improving)."""
        if len(self.loss_history) < 2:
            return 0.0
        first_step, first_loss = self.loss_history[0]
        last_step, last_loss = self.loss_history[-1]
        step_diff = last_step - first_step
        if step_diff == 0:
            return 0.0
        return (last_loss - first_loss) / step_diff * 1000

    def get_gpu_memory(self) -> dict:
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }

    def get_throughput(self) -> dict:
        if not self.step_times:
            return {"steps_per_sec": 0, "avg_step_time": 0, "samples_per_sec": 0}
        recent = self.step_times[-50:]
        avg_time = np.mean(recent)
        sps = 1.0 / avg_time if avg_time > 0 else 0
        return {
            "steps_per_sec": sps,
            "avg_step_time": avg_time,
            "samples_per_sec": sps * self._samples_per_step,
            "p50_step_time": float(np.median(recent)),
            "p95_step_time": float(np.percentile(recent, 95)) if len(recent) >= 5 else avg_time,
        }

    def get_grad_health_summary(self) -> dict:
        if not self.grad_norms:
            return {}
        recent = self.grad_norms[-50:]
        return {
            "mean": float(np.mean(recent)),
            "max": float(np.max(recent)),
            "min": float(np.min(recent)),
            "std": float(np.std(recent)),
            "clip_rate": self.grad_clip_count / max(self.grad_total_count, 1) * 100,
            "nan_count": self.grad_nan_count,
            "explosion_count": self.grad_explosion_count,
        }

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.0f}m {seconds % 60:.0f}s"
        elif seconds < 86400:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h {m}m"
        else:
            d = int(seconds // 86400)
            h = int((seconds % 86400) // 3600)
            return f"{d}d {h}h"

    def _make_progress_bar(self, pct: float, width: int = 30) -> str:
        filled = int(width * pct / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {pct:.1f}%"

    def _make_sparkline(self, values: list[float], width: int = 20) -> str:
        """Create a Unicode sparkline from values."""
        if not values or len(values) < 2:
            return ""
        blocks = " ▁▂▃▄▅▆▇█"
        mn, mx = min(values), max(values)
        rng = mx - mn if mx != mn else 1.0
        # Downsample if too many values
        if len(values) > width:
            step = len(values) / width
            sampled = [values[int(i * step)] for i in range(width)]
        else:
            sampled = values
        return "".join(blocks[min(int((v - mn) / rng * 8), 8)] for v in sampled)

    def format_health_report(
        self,
        step: int,
        max_steps: int,
        elapsed_time: float,
        lazy_source: "LazyHFTaskSource | None" = None,
        benchmark_results: "dict | None" = None,
    ) -> str:
        pct = step / max_steps * 100 if max_steps > 0 else 0

        lines = [
            "",
            "╔" + "═" * 72 + "╗",
            "║" + f"  TRAINING HEALTH REPORT — Step {step:,}/{max_steps:,}".ljust(72) + "║",
            "║" + f"  {self._make_progress_bar(pct, 40)}  Elapsed: {self._format_time(elapsed_time)}".ljust(72) + "║",
            "╠" + "═" * 72 + "╣",
        ]

        # ━━━ LOSS ANALYSIS ━━━
        lines.append("║" + "  ┌─ Loss Analysis ─────────────────────────────────────────────────┐".ljust(72) + "║")
        if self.loss_history:
            current = self.loss_history[-1][1]
            lines.append("║" + f"  │  Current:     {current:.6f}".ljust(72) + "║")
        if self.loss_ema is not None:
            lines.append("║" + f"  │  EMA (fast):  {self.loss_ema:.6f}".ljust(72) + "║")
        if self.loss_ema_slow is not None:
            lines.append("║" + f"  │  EMA (slow):  {self.loss_ema_slow:.6f}".ljust(72) + "║")
        lines.append("║" + f"  │  Best:        {self.best_loss:.6f} (step {self.best_loss_step:,})".ljust(72) + "║")

        trend_label, trend_pct = self.get_loss_trend()
        trend_icon = {"converging": "↘↘", "decreasing": "↘", "stable": "→",
                      "increasing": "↗", "DIVERGING": "↗↗ ⚠️"}.get(trend_label, "?")
        lines.append("║" + f"  │  Trend:       {trend_icon} {trend_label} ({trend_pct:+.2f}%)".ljust(72) + "║")

        loss_var = self.get_loss_variance()
        conv_rate = self.get_convergence_rate()
        lines.append("║" + f"  │  Variance:    {loss_var:.6f}  (stability indicator)".ljust(72) + "║")
        lines.append("║" + f"  │  Conv. rate:  {conv_rate:+.4f} per 1K steps".ljust(72) + "║")

        # Sparkline of recent losses
        if len(self.loss_history) >= 5:
            recent_losses = [l for _, l in self.loss_history[-40:]]
            spark = self._make_sparkline(recent_losses)
            lines.append("║" + f"  │  History:     {spark}".ljust(72) + "║")

        if self._loss_milestones:
            last_ms = self._loss_milestones[-1]
            lines.append("║" + f"  │  Milestones:  {len(self._loss_milestones)} (last: step {last_ms[0]:,} = {last_ms[1]:.4f})".ljust(72) + "║")

        lines.append("║" + "  └─────────────────────────────────────────────────────────────────┘".ljust(72) + "║")

        # ━━━ GRADIENT HEALTH ━━━
        grad = self.get_grad_health_summary()
        if grad:
            lines.append("║" + "  ┌─ Gradient Health ───────────────────────────────────────────────┐".ljust(72) + "║")
            lines.append("║" + f"  │  Norm (avg):  {grad['mean']:.4f}   Norm (max): {grad['max']:.4f}   Std: {grad['std']:.4f}".ljust(72) + "║")

            clip_str = f"{self.grad_clip_count}/{self.grad_total_count} ({grad['clip_rate']:.1f}%)"
            lines.append("║" + f"  │  Clip rate:   {clip_str}".ljust(72) + "║")

            # Warnings
            if grad['nan_count'] > 0:
                lines.append("║" + f"  │  ⚠️  NaN gradients detected: {grad['nan_count']}".ljust(72) + "║")
            if grad['explosion_count'] > 0:
                lines.append("║" + f"  │  ⚠️  Gradient explosions (>10x avg): {grad['explosion_count']}".ljust(72) + "║")
            if grad['clip_rate'] > 50:
                lines.append("║" + f"  │  ⚠️  High clip rate! Consider increasing max_grad_norm".ljust(72) + "║")

            if len(self.grad_norms) >= 5:
                spark = self._make_sparkline(self.grad_norms[-40:])
                lines.append("║" + f"  │  History:     {spark}".ljust(72) + "║")

            lines.append("║" + "  └─────────────────────────────────────────────────────────────────┘".ljust(72) + "║")

        # ━━━ LEARNING RATE ━━━
        if self.lr_history:
            lines.append("║" + "  ┌─ Learning Rate ─────────────────────────────────────────────────┐".ljust(72) + "║")
            current_lr = self.lr_history[-1][1]
            peak_lr = max(lr for _, lr in self.lr_history)
            lr_ratio = current_lr / peak_lr if peak_lr > 0 else 0
            lines.append("║" + f"  │  Current:     {current_lr:.2e}  (peak: {peak_lr:.2e}, ratio: {lr_ratio:.1%})".ljust(72) + "║")

            if len(self.lr_history) >= 5:
                lr_vals = [lr for _, lr in self.lr_history[-40:]]
                spark = self._make_sparkline(lr_vals)
                lines.append("║" + f"  │  Schedule:    {spark}".ljust(72) + "║")

            lines.append("║" + "  └─────────────────────────────────────────────────────────────────┘".ljust(72) + "║")

        # ━━━ GPU MEMORY ━━━
        gpu_mem = self.get_gpu_memory()
        if gpu_mem:
            util_pct = gpu_mem["allocated_gb"] / gpu_mem["total_gb"] * 100
            lines.append("║" + "  ┌─ GPU Memory (Rank 0) ───────────────────────────────────────────┐".ljust(72) + "║")

            # Visual memory bar
            mem_bar_width = 30
            alloc_filled = int(mem_bar_width * gpu_mem["allocated_gb"] / gpu_mem["total_gb"])
            peak_filled = int(mem_bar_width * gpu_mem["peak_gb"] / gpu_mem["total_gb"])
            bar = "█" * alloc_filled + "▓" * max(0, peak_filled - alloc_filled) + "░" * (mem_bar_width - peak_filled)
            lines.append("║" + f"  │  [{bar}] {gpu_mem['allocated_gb']:.1f}/{gpu_mem['total_gb']:.0f} GB ({util_pct:.0f}%)".ljust(72) + "║")
            lines.append("║" + f"  │  Allocated: {gpu_mem['allocated_gb']:.2f} GB  Reserved: {gpu_mem['reserved_gb']:.2f} GB  Peak: {gpu_mem['peak_gb']:.2f} GB".ljust(72) + "║")

            if util_pct < 30:
                lines.append("║" + f"  │  �� Low utilization — consider increasing batch_size".ljust(72) + "║")
            elif util_pct > 90:
                lines.append("║" + f"  │  ⚠️  High utilization — risk of OOM".ljust(72) + "║")

            lines.append("║" + "  └─────────────────────────────────────────────────────────────────┘".ljust(72) + "║")

        # ━━━ THROUGHPUT ━━━
        tp = self.get_throughput()
        if tp["steps_per_sec"] > 0:
            remaining = max_steps - step
            eta_sec = remaining / tp["steps_per_sec"]
            lines.append("║" + "  ┌─ Throughput ────────────────────────────────────────────────────┐".ljust(72) + "║")
            lines.append("║" + f"  │  Steps/sec:   {tp['steps_per_sec']:.2f}    Samples/sec: {tp['samples_per_sec']:.0f}".ljust(72) + "║")
            lines.append("║" + f"  │  Step time:   avg={tp['avg_step_time']:.3f}s  p50={tp['p50_step_time']:.3f}s  p95={tp['p95_step_time']:.3f}s".ljust(72) + "║")
            lines.append("║" + f"  │  ETA:         {self._format_time(eta_sec)}".ljust(72) + "║")
            lines.append("║" + f"  │  Samples:     {self.total_samples:,} total processed".ljust(72) + "║")
            lines.append("║" + "  └─────────────────────────────────────────────────────────────────┘".ljust(72) + "║")

        # ━━━ DATA PIPELINE ━━━
        if lazy_source is not None:
            total_accesses = lazy_source._cache_hits + lazy_source._cache_misses
            hit_rate = lazy_source._cache_hits / max(total_accesses, 1) * 100
            lines.append("║" + "  ┌─ Data Pipeline (LazyHFTaskSource) ──────────────────────────────┐".ljust(72) + "║")
            lines.append("║" + f"  │  Total series: {lazy_source._total_len:,}".ljust(72) + "║")
            lines.append("║" + f"  │  Cache:        {len(lazy_source._cache):,}/{lazy_source._cache_max:,} ({len(lazy_source._cache)/lazy_source._cache_max*100:.0f}% full)".ljust(72) + "║")
            lines.append("║" + f"  │  Hit rate:     {hit_rate:.1f}% ({lazy_source._cache_hits:,}/{total_accesses:,})".ljust(72) + "║")
            lines.append("║" + f"  │  Short skips:  {lazy_source._short_series_skips:,}".ljust(72) + "║")
            lines.append("║" + "  └─────────────────────────────────────────────────────────────────┘".ljust(72) + "║")

        # ━━━ BENCHMARK RESULTS (if available) ━━━
        if benchmark_results:
            lines.append("║" + "  ┌─ Lite Benchmark (last evaluation) ─────────────────────────────┐".ljust(72) + "║")
            avg_wql = benchmark_results.get("avg_wql", None)
            avg_mase = benchmark_results.get("avg_mase", None)
            bm_step = benchmark_results.get("step", "?")
            if avg_wql is not None:
                lines.append("║" + f"  │  Avg WQL:     {avg_wql:.4f}  (step {bm_step:,})".ljust(72) + "║")
            if avg_mase is not None:
                lines.append("║" + f"  │  Avg MASE:    {avg_mase:.4f}".ljust(72) + "║")
            per_ds = benchmark_results.get("per_dataset", {})
            for ds_name, ds_metrics in per_ds.items():
                wql = ds_metrics.get("WQL", "N/A")
                mase = ds_metrics.get("MASE", "N/A")
                wql_str = f"{wql:.4f}" if isinstance(wql, float) else str(wql)
                mase_str = f"{mase:.4f}" if isinstance(mase, float) else str(mase)
                lines.append("║" + f"  │    {ds_name:<25s} WQL={wql_str}  MASE={mase_str}".ljust(72) + "║")
            lines.append("║" + "  └─────────────────────────────────────────────────────────────────┘".ljust(72) + "║")

        lines.append("╚" + "═" * 72 + "╝")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lite Benchmark Evaluation (during training)
# ---------------------------------------------------------------------------

def _run_lite_benchmark(
    model: Chronos2Model,
    config_path: str,
    device: torch.device,
    eval_batch_size: int = 32,
) -> dict:
    """Run lite benchmark evaluation and return metrics dict.

    Uses the same evaluation logic as scripts/evaluation/evaluate.py
    but runs on rank-0 only during training.

    Returns dict with:
        avg_wql, avg_mase, per_dataset: {name: {WQL, MASE}}
    """
    import datasets as hf_datasets
    import pandas as pd
    from gluonts.dataset.split import split as gluonts_split
    from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
    from gluonts.model.evaluation import evaluate_forecasts
    from gluonts.model.forecast import QuantileForecast
    from gluonts.itertools import batcher

    from chronos import Chronos2Pipeline

    eval_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Load benchmark configs
    with open(config_path) as f:
        configs = yaml.safe_load(f)

    # Build a temporary pipeline wrapper around the training model
    pipeline = Chronos2Pipeline.__new__(Chronos2Pipeline)
    pipeline.model = model
    pipeline.device = device
    pipeline.chronos_config = model.chronos_config

    results_per_dataset = {}
    all_wql = []
    all_mase = []

    for bm_config in configs:
        ds_name = bm_config["name"]
        pred_len = bm_config["prediction_length"]
        hf_repo = bm_config["hf_repo"]
        offset = bm_config["offset"]
        num_rolls = bm_config["num_rolls"]

        try:
            trust_remote = hf_repo == "autogluon/chronos_datasets_extra"
            ds = hf_datasets.load_dataset(hf_repo, ds_name, split="train",
                                           trust_remote_code=trust_remote)
            ds.set_format("numpy")

            # Convert to GluonTS univariate format
            series_fields = [
                col for col in ds.features
                if isinstance(ds.features[col], hf_datasets.Sequence) and col != "timestamp"
            ]
            gts_dataset = []
            dataset_freq = pd.DatetimeIndex(ds[0]["timestamp"]).to_period()[0].freqstr
            for entry in ds:
                for field in series_fields:
                    gts_dataset.append({
                        "start": pd.Period(entry["timestamp"][0], freq=dataset_freq),
                        "target": entry[field],
                    })

            _, test_template = gluonts_split(gts_dataset, offset=offset)
            test_data = test_template.generate_instances(pred_len, windows=num_rolls)

            # Generate forecasts
            forecast_outputs = []
            for batch in batcher(test_data.input, batch_size=eval_batch_size):
                context = [torch.tensor(e["target"]) for e in batch]
                with torch.no_grad():
                    quantiles, _ = pipeline.predict_quantiles(
                        context, prediction_length=pred_len,
                        quantile_levels=eval_quantiles,
                    )
                if isinstance(quantiles, list):
                    quantiles = np.stack(quantiles).squeeze(axis=1)
                quantiles = quantiles.swapaxes(-1, -2)
                forecast_outputs.append(quantiles)
            forecast_outputs = np.concatenate(forecast_outputs)

            forecasts = []
            for item, ts in zip(forecast_outputs, test_data.input):
                start = ts["start"] + len(ts["target"])
                forecasts.append(QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=list(map(str, eval_quantiles)),
                    start_date=start,
                ))

            metrics = (
                evaluate_forecasts(
                    forecasts, test_data=test_data,
                    metrics=[MASE(), MeanWeightedSumQuantileLoss(eval_quantiles)],
                    batch_size=5000,
                )
                .reset_index(drop=True)
                .to_dict(orient="records")
            )
            wql = metrics[0].get("mean_weighted_sum_quantile_loss", float("nan"))
            mase = metrics[0].get("MASE[0.5]", float("nan"))

            results_per_dataset[ds_name] = {"WQL": wql, "MASE": mase}
            if not math.isnan(wql):
                all_wql.append(wql)
            if not math.isnan(mase):
                all_mase.append(mase)

            logger.info(f"  Benchmark {ds_name}: WQL={wql:.4f}, MASE={mase:.4f}")

        except Exception as e:
            logger.warning(f"  Benchmark {ds_name} FAILED: {e}")
            results_per_dataset[ds_name] = {"WQL": "error", "MASE": "error"}

    return {
        "avg_wql": float(np.mean(all_wql)) if all_wql else None,
        "avg_mase": float(np.mean(all_mase)) if all_mase else None,
        "per_dataset": results_per_dataset,
    }


class LiteBenchmarkCallback(TrainerCallback):
    """Periodic lite benchmark evaluation during training.

    Runs a small set of forecast benchmarks (WQL/MASE) every N steps on rank-0.
    Logs results to TensorBoard and manages top-K checkpoint retention by WQL.

    The model is temporarily moved to eval mode, benchmarks are run, then
    training resumes. All other ranks wait at a barrier.
    """

    def __init__(
        self,
        benchmark_config_path: str,
        eval_steps: int = 10000,
        top_k_checkpoints: int = 3,
        eval_batch_size: int = 32,
    ):
        self._config_path = benchmark_config_path
        self._eval_steps = eval_steps
        self._top_k = top_k_checkpoints
        self._eval_batch_size = eval_batch_size
        self._last_eval_step: int = 0
        self._best_results: list[tuple[float, int, str]] = []  # (wql, step, ckpt_path)
        self.last_benchmark_results: dict | None = None

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if (
            self._eval_steps > 0
            and step > 0
            and step - self._last_eval_step >= self._eval_steps
        ):
            self._last_eval_step = step
            model = kwargs.get("model")
            if model is None:
                return

            if is_main_process():
                logger.info(f"\n{'─' * 72}")
                logger.info(f"  LITE BENCHMARK EVALUATION — Step {step:,}")
                logger.info(f"{'─' * 72}")

                model.eval()
                device = next(model.parameters()).device
                try:
                    results = _run_lite_benchmark(
                        model=model,
                        config_path=self._config_path,
                        device=device,
                        eval_batch_size=self._eval_batch_size,
                    )
                    results["step"] = step
                    self.last_benchmark_results = results

                    avg_wql = results.get("avg_wql")
                    avg_mase = results.get("avg_mase")

                    logger.info(f"  ── Results ──")
                    if avg_wql is not None:
                        logger.info(f"  Avg WQL:  {avg_wql:.4f}")
                    if avg_mase is not None:
                        logger.info(f"  Avg MASE: {avg_mase:.4f}")

                    # Log to TensorBoard
                    if state.is_world_process_zero and hasattr(state, "log_history"):
                        tb_logs = {}
                        if avg_wql is not None:
                            tb_logs["benchmark/avg_wql"] = avg_wql
                        if avg_mase is not None:
                            tb_logs["benchmark/avg_mase"] = avg_mase
                        for ds_name, ds_m in results.get("per_dataset", {}).items():
                            if isinstance(ds_m.get("WQL"), float):
                                tb_logs[f"benchmark/{ds_name}/wql"] = ds_m["WQL"]
                            if isinstance(ds_m.get("MASE"), float):
                                tb_logs[f"benchmark/{ds_name}/mase"] = ds_m["MASE"]

                        # Write via trainer's log method if available
                        try:
                            from torch.utils.tensorboard import SummaryWriter
                            tb_dir = Path(args.output_dir) / "runs"
                            if tb_dir.exists():
                                # Find the most recent events directory
                                tb_dirs = sorted(tb_dir.iterdir(), key=lambda p: p.stat().st_mtime)
                                if tb_dirs:
                                    writer = SummaryWriter(log_dir=str(tb_dirs[-1]))
                                    for key, val in tb_logs.items():
                                        writer.add_scalar(key, val, step)
                                    writer.flush()
                                    writer.close()
                        except Exception:
                            pass  # TensorBoard logging is best-effort

                    # Track top-K checkpoints by WQL
                    if avg_wql is not None:
                        ckpt_path = str(Path(args.output_dir) / f"checkpoint-{step}")
                        self._best_results.append((avg_wql, step, ckpt_path))
                        self._best_results.sort(key=lambda x: x[0])  # sort by WQL ascending (lower=better)

                        if len(self._best_results) > self._top_k:
                            # Remove worst checkpoint
                            worst = self._best_results.pop()
                            worst_path = Path(worst[2])
                            if worst_path.exists() and worst[1] != step:
                                import shutil
                                try:
                                    shutil.rmtree(worst_path)
                                    logger.info(f"  Removed worst checkpoint: {worst_path.name} (WQL={worst[0]:.4f})")
                                except OSError:
                                    pass

                        logger.info(f"  Top-{self._top_k} checkpoints by WQL:")
                        for rank_i, (wql, s, p) in enumerate(self._best_results, 1):
                            logger.info(f"    #{rank_i}: step {s:,} — WQL={wql:.4f}")

                except Exception as e:
                    logger.error(f"  Lite benchmark failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                finally:
                    model.train()

                logger.info(f"{'─' * 72}\n")

            # Barrier so other ranks wait for evaluation to finish
            if dist.is_initialized():
                dist.barrier()


class ComprehensiveTrainingCallback(TrainerCallback):
    """Advanced training callback with rich monitoring and periodic health reports.

    Features:
    - Compact single-line per log step with key metrics
    - Periodic boxed health reports with sparklines and visual indicators
    - Loss convergence analysis with multi-window EMA
    - Gradient health monitoring with explosion/NaN detection
    - GPU memory utilization with visual bars
    - Throughput tracking with percentile latencies
    - Data pipeline statistics
    - Benchmark results integration
    - Final comprehensive training summary
    """

    def __init__(
        self,
        health_report_interval: int = 1000,
        lazy_source: "LazyHFTaskSource | None" = None,
        max_grad_norm: float = 1.0,
        effective_batch_size: int = 1,
        benchmark_callback: "LiteBenchmarkCallback | None" = None,
    ):
        self.monitor = TrainingHealthMonitor()
        self.monitor.set_samples_per_step(effective_batch_size)
        self._health_report_interval = health_report_interval
        self._lazy_source = lazy_source
        self._max_grad_norm = max_grad_norm
        self._benchmark_cb = benchmark_callback
        self._train_start_time: float | None = None
        self._last_report_step: int = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start_time = time.time()
        if is_main_process():
            logger.info(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}. "
                        f"Max steps: {state.max_steps:,}")

    def on_step_end(self, args, state, control, **kwargs):
        self.monitor.mark_step()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not is_main_process() or not logs:
            return

        step = state.global_step
        loss = logs.get("loss")
        grad_norm = logs.get("grad_norm")
        lr = logs.get("learning_rate", 0)

        # Update monitor
        if loss is not None:
            self.monitor.update_loss(loss, step)
        if grad_norm is not None:
            self.monitor.update_grad_norm(grad_norm, self._max_grad_norm)
        if lr > 0:
            self.monitor.update_lr(lr, step)

        # ── Compact per-step log line ──
        if loss is not None:
            gpu_mem = self.monitor.get_gpu_memory()
            tp = self.monitor.get_throughput()

            parts = [f"Step {step:,}/{state.max_steps:,}"]
            parts.append(f"loss={loss:.4f}")

            if self.monitor.loss_ema is not None:
                parts.append(f"ema={self.monitor.loss_ema:.4f}")
            parts.append(f"best={self.monitor.best_loss:.4f}")
            parts.append(f"lr={lr:.2e}")

            if gpu_mem:
                parts.append(f"mem={gpu_mem['allocated_gb']:.1f}/{gpu_mem['total_gb']:.0f}GB")

            if tp["steps_per_sec"] > 0:
                remaining = state.max_steps - step
                eta_sec = remaining / tp["steps_per_sec"]
                eta_str = self.monitor._format_time(eta_sec)
                parts.append(f"{tp['steps_per_sec']:.1f} stp/s")
                parts.append(f"{tp['samples_per_sec']:.0f} smp/s")
                parts.append(f"eta={eta_str}")

            if grad_norm is not None:
                gnorm_str = f"gnorm={grad_norm:.3f}"
                if self.monitor.grad_norms and grad_norm > np.mean(self.monitor.grad_norms[-10:]) * 5:
                    gnorm_str += " ⚠️"
                parts.append(gnorm_str)

            logger.info(" | ".join(parts))

        # ── Periodic health report ──
        if (
            self._health_report_interval > 0
            and step > 0
            and step - self._last_report_step >= self._health_report_interval
        ):
            self._last_report_step = step
            elapsed = time.time() - self._train_start_time if self._train_start_time else 0
            bm_results = None
            if self._benchmark_cb is not None:
                bm_results = self._benchmark_cb.last_benchmark_results
            report = self.monitor.format_health_report(
                step, state.max_steps, elapsed,
                self._lazy_source, bm_results,
            )
            logger.info(report)

    def on_train_end(self, args, state, control, **kwargs):
        if not is_main_process():
            return

        total_time = time.time() - self._train_start_time if self._train_start_time else 0
        gpu_mem = self.monitor.get_gpu_memory()
        tp = self.monitor.get_throughput()

        lines = [
            "",
            "╔" + "═" * 72 + "╗",
            "║" + "  TRAINING COMPLETE".ljust(72) + "║",
            "║" + f"  {time.strftime('%Y-%m-%d %H:%M:%S')}".ljust(72) + "║",
            "╠" + "═" * 72 + "╣",
        ]

        lines.append("║" + f"  Total time:       {self.monitor._format_time(total_time)}".ljust(72) + "║")
        lines.append("║" + f"  Total steps:      {state.global_step:,}".ljust(72) + "║")
        lines.append("║" + f"  Total samples:    {self.monitor.total_samples:,}".ljust(72) + "║")

        if self.monitor.loss_history:
            final_loss = self.monitor.loss_history[-1][1]
            first_loss = self.monitor.loss_history[0][1]
            improvement = ((first_loss - final_loss) / first_loss * 100) if first_loss > 0 else 0
            lines.append("║" + "".ljust(72) + "║")
            lines.append("║" + f"  Final loss:       {final_loss:.6f}".ljust(72) + "║")
            lines.append("║" + f"  Best loss:        {self.monitor.best_loss:.6f} (step {self.monitor.best_loss_step:,})".ljust(72) + "║")
            lines.append("║" + f"  First loss:       {first_loss:.6f}".ljust(72) + "║")
            lines.append("║" + f"  Improvement:      {improvement:.1f}% from start".ljust(72) + "║")

            if self.monitor._loss_milestones:
                lines.append("║" + f"  Milestones:       {len(self.monitor._loss_milestones)} (10%+ improvements)".ljust(72) + "║")

        if gpu_mem:
            lines.append("║" + "".ljust(72) + "║")
            lines.append("║" + f"  Peak GPU memory:  {gpu_mem['peak_gb']:.2f} GB / {gpu_mem['total_gb']:.0f} GB ({gpu_mem['peak_gb']/gpu_mem['total_gb']*100:.0f}%)".ljust(72) + "║")

        if tp["steps_per_sec"] > 0:
            lines.append("║" + f"  Avg throughput:   {tp['steps_per_sec']:.2f} steps/s, {tp['samples_per_sec']:.0f} samples/s".ljust(72) + "║")

        grad = self.monitor.get_grad_health_summary()
        if grad:
            lines.append("║" + "".ljust(72) + "║")
            lines.append("║" + f"  Grad clip rate:   {grad['clip_rate']:.1f}%".ljust(72) + "║")
            if grad['nan_count'] > 0:
                lines.append("║" + f"  Grad NaN count:   {grad['nan_count']}".ljust(72) + "║")
            if grad['explosion_count'] > 0:
                lines.append("║" + f"  Grad explosions:  {grad['explosion_count']}".ljust(72) + "║")

        # Benchmark summary
        bm_results = None
        if self._benchmark_cb is not None:
            bm_results = self._benchmark_cb.last_benchmark_results
            best_results = self._benchmark_cb._best_results
        if bm_results:
            lines.append("║" + "".ljust(72) + "║")
            lines.append("║" + f"  Last benchmark WQL:  {bm_results.get('avg_wql', 'N/A')}".ljust(72) + "║")
            lines.append("║" + f"  Last benchmark MASE: {bm_results.get('avg_mase', 'N/A')}".ljust(72) + "║")
            if best_results:
                lines.append("║" + f"  Best benchmark WQL:  {best_results[0][0]:.4f} (step {best_results[0][1]:,})".ljust(72) + "║")

        if self._lazy_source is not None:
            lines.append("║" + "".ljust(72) + "║")
            self._lazy_source.log_stats()

        lines.append("╚" + "═" * 72 + "╝")
        logger.info("\n".join(lines))


# ===========================================================================
# Core Training Logic
# ===========================================================================

def _build_training_args(config: dict, eval_dataset, world_size: int) -> TrainingArguments:
    """Build HuggingFace TrainingArguments from config dict."""
    batch_size = config["per_device_train_batch_size"]
    output_dir = config.get("output_dir",
                            f"./output/chronos2-{time.strftime('%Y%m%d-%H%M%S')}")
    has_sm80 = (torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 8)
    grad_accum = config.get("gradient_accumulation_steps", 1)

    training_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=config.get("learning_rate", 1e-4),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.01),
        warmup_steps=config.get("warmup_steps", 0),
        optim=config.get("optim", "adamw_torch_fused"),
        weight_decay=config.get("weight_decay", 0.01),
        adam_beta1=config.get("adam_beta1", 0.9),
        adam_beta2=config.get("adam_beta2", 0.999),
        adam_epsilon=config.get("adam_epsilon", 1e-8),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        logging_strategy="steps",
        logging_steps=config.get("log_steps", 100),
        disable_tqdm=not is_main_process(),
        report_to=config.get("report_to", "tensorboard"),
        max_steps=config["max_steps"],
        gradient_accumulation_steps=grad_accum,
        dataloader_num_workers=config.get("dataloader_num_workers", 0),
        dataloader_pin_memory=True,
        tf32=has_sm80,
        bf16=has_sm80,
        save_only_model=True,
        prediction_loss_only=True,
        save_total_limit=config.get("save_total_limit", 3),
        # Logging: suppress replica rank output
        log_level="info",
        log_level_replica="warning",
        # DDP settings
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=config.get("ddp_bucket_cap_mb", 25),
        # FSDP settings
        fsdp=config.get("fsdp", ""),
        fsdp_config=config.get("fsdp_config", None),
        # Distributed
        local_rank=get_local_rank(),
    )

    if eval_dataset is not None:
        save_steps = config.get("save_steps", 1000)
        training_kwargs.update(
            save_strategy="steps",
            save_steps=save_steps,
            eval_strategy="steps",
            eval_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            label_names=["future_target"],
        )
    else:
        save_steps = config.get("save_steps", 5000)
        training_kwargs.update(
            save_strategy="steps",
            save_steps=save_steps,
            eval_strategy="no",
            load_best_model_at_end=False,
        )

    if config.get("torch_compile", False):
        training_kwargs["torch_compile"] = True

    # Warn if multi-node without FSDP
    if world_size > 8 and not config.get("fsdp", ""):
        logger.warning(
            f"Running {world_size} GPUs without FSDP. "
            f"Consider enabling FSDP for better memory efficiency: "
            f"set fsdp: 'full_shard auto_wrap' in your config."
        )

    return TrainingArguments(**training_kwargs)


def run_training(config: dict):
    """Main training function supporting pretraining and fine-tuning.

    Uses lazy Arrow-based data loading (ASR-style) for training data:
    - Each rank opens the HF Arrow datasets independently (memory-mapped)
    - Tasks are read on-demand during training, not materialized upfront
    - Zero upfront data loading time — training starts immediately
    - Memory usage proportional to cache size, not dataset size

    Validation data uses eager loading (small datasets).
    """

    world_size = get_world_size()

    # -------------------------------------------------------------------
    # 1. Setup CUDA and seeds
    # -------------------------------------------------------------------
    setup_cuda_device()
    setup_distributed_seeds(seed=config.get("seed", 42))
    log_distributed_info()

    # -------------------------------------------------------------------
    # 2. Create or load model
    # -------------------------------------------------------------------
    model_id = config.get("model_id", None)
    random_init = config.get("random_init", True)

    if random_init:
        model = create_model_from_config(config)
    else:
        if model_id is None:
            raise ValueError("model_id is required when random_init=False (fine-tuning mode)")
        model = load_pretrained_model(model_id)

        # CRITICAL: Update model's forecasting config for stage transitions.
        # Without this, loading a Stage 1 checkpoint (context_length=2048) for
        # Stage 2 training (context_length=8192) would silently truncate all
        # inputs to 2048 at model.py:_prepare_patched_context().
        # Chronos2ForecastingConfig.editable_fields() defines which fields are
        # safe to modify: ["context_length", "max_output_patches"]
        old_ctx = model.chronos_config.context_length
        old_max_patches = model.chronos_config.max_output_patches
        model.chronos_config.context_length = config["context_length"]
        model.chronos_config.max_output_patches = config.get(
            "max_output_patches", model.chronos_config.max_output_patches
        )
        # Sync back to model.config so it's saved correctly in checkpoints
        model.config.chronos_config = model.chronos_config.__dict__

        if is_main_process():
            if old_ctx != config["context_length"]:
                logger.info(
                    f"  Updated context_length: {old_ctx} -> {config['context_length']} "
                    f"(stage transition detected)"
                )
            if old_max_patches != model.chronos_config.max_output_patches:
                logger.info(
                    f"  Updated max_output_patches: {old_max_patches} -> "
                    f"{model.chronos_config.max_output_patches}"
                )

    # -------------------------------------------------------------------
    # 2b. Gradient checkpointing (optional, before DDP/FSDP wrapping)
    # -------------------------------------------------------------------
    use_grad_ckpt = config.get("gradient_checkpointing", False)
    if use_grad_ckpt:
        enable_gradient_checkpointing(model)

    # -------------------------------------------------------------------
    # 3. Build lazy training data source (ASR-style)
    # -------------------------------------------------------------------
    # All ranks open the same Arrow datasets independently via memory-mapping.
    # No rank-0 broadcast needed — Arrow files are read directly from NFS.
    # Index shuffling is deterministic (same seed on all ranks).
    training_data_paths = config["training_data_paths"]
    probabilities = config.get("probability",
                               [1.0 / len(training_data_paths)] * len(training_data_paths))
    context_length = config["context_length"]
    prediction_length = config["prediction_length"]
    batch_size = config["per_device_train_batch_size"]
    output_patch_size = config.get("output_patch_size", 16)
    min_past = config.get("min_past", prediction_length)
    base_seed = config.get("seed", 42)

    if is_main_process():
        logger.info("Building lazy training data source (Arrow memory-mapped)...")

    lazy_task_source = _build_lazy_task_source(
        data_paths=training_data_paths,
        probabilities=probabilities,
        split="train",
        max_total_series=config.get("max_train_series", None),
        prediction_length=prediction_length,
        min_past=min_past,
        shuffle_seed=base_seed,
    )

    # -------------------------------------------------------------------
    # 4. Load validation data (optional, eager — small dataset)
    # -------------------------------------------------------------------
    validation_data_paths = config.get("validation_data_paths", None)
    val_dicts = None
    if validation_data_paths:
        if is_main_process():
            logger.info("Loading validation data (eager)...")
        val_dicts = []
        for vpath in validation_data_paths:
            vdicts = load_hf_dataset_as_dicts(
                dataset_path=vpath,
                split="train",
                max_series=config.get("max_val_series", 10000),
                prediction_length=prediction_length,
            )
            val_dicts.extend(vdicts)

    # -------------------------------------------------------------------
    # 5. Create Chronos2Datasets
    # -------------------------------------------------------------------
    grad_accum = config.get("gradient_accumulation_steps", 1)
    effective_batch = batch_size * world_size * grad_accum

    if is_main_process():
        logger.info(f"Creating datasets: ctx={context_length}, pred={prediction_length}, "
                     f"batch/device={batch_size}")

    # Create training dataset with a minimal dummy input, then replace
    # self.tasks with the lazy task source. This avoids the O(N) upfront
    # _prepare_tasks call while keeping all batch construction logic intact.
    dummy_target = np.zeros(min_past + prediction_length, dtype=np.float32)
    train_dataset = Chronos2Dataset(
        inputs=[{"target": dummy_target}],
        context_length=context_length,
        prediction_length=prediction_length,
        batch_size=batch_size,
        output_patch_size=output_patch_size,
        min_past=min_past,
        mode=DatasetMode.TRAIN,
    )
    # Replace the pre-computed tasks list with our lazy source.
    # LazyHFTaskSource has the same interface: __len__() and __getitem__(idx)
    # returning the same tuple format as _prepare_tasks produces.
    train_dataset.tasks = lazy_task_source

    if is_main_process():
        logger.info(f"Training dataset: {len(train_dataset.tasks)} series (lazy Arrow access)")

    eval_dataset = None
    if val_dicts is not None:
        eval_dataset = Chronos2Dataset.convert_inputs(
            inputs=val_dicts,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=output_patch_size,
            mode=DatasetMode.VALIDATION,
        )
        del val_dicts

    # -------------------------------------------------------------------
    # 6. Configure TrainingArguments
    # -------------------------------------------------------------------
    output_dir = config.get("output_dir",
                            f"./output/chronos2-{time.strftime('%Y%m%d-%H%M%S')}")
    training_args = _build_training_args(config, eval_dataset, world_size)

    # -------------------------------------------------------------------
    # 7. Create Callbacks
    # -------------------------------------------------------------------
    health_report_interval = config.get("health_report_interval", 1000)
    max_grad_norm = config.get("max_grad_norm", 1.0)

    # Lite Benchmark Callback (optional — requires config path)
    benchmark_config_path = config.get("benchmark_config", None)
    benchmark_eval_steps = config.get("benchmark_eval_steps", 10000)
    benchmark_top_k = config.get("benchmark_top_k_checkpoints", 3)
    benchmark_batch_size = config.get("benchmark_batch_size", 32)
    benchmark_cb = None

    if benchmark_config_path:
        # Resolve path: try relative to script dir, then evaluation dir, then project root
        bm_path = Path(benchmark_config_path)
        if not bm_path.is_absolute():
            script_dir = Path(__file__).parent
            candidates = [
                script_dir / benchmark_config_path,                          # scripts/training/configs/...
                script_dir.parent / "evaluation" / benchmark_config_path,    # scripts/evaluation/configs/...
                script_dir.parent.parent / benchmark_config_path,            # project root/configs/...
            ]
            bm_path = next((c for c in candidates if c.exists()), candidates[0])
        if bm_path.exists():
            benchmark_cb = LiteBenchmarkCallback(
                benchmark_config_path=str(bm_path),
                eval_steps=benchmark_eval_steps,
                top_k_checkpoints=benchmark_top_k,
                eval_batch_size=benchmark_batch_size,
            )
            if is_main_process():
                logger.info(f"Lite benchmark enabled: {bm_path.name} every {benchmark_eval_steps} steps")
        elif is_main_process():
            logger.warning(f"Benchmark config not found at any of: {[str(c) for c in candidates]}. Benchmark evaluation disabled.")

    # Comprehensive Training Callback (always active)
    comprehensive_cb = ComprehensiveTrainingCallback(
        health_report_interval=health_report_interval,
        lazy_source=lazy_task_source,
        max_grad_norm=max_grad_norm,
        effective_batch_size=effective_batch,
        benchmark_callback=benchmark_cb,
    )

    callbacks = [comprehensive_cb]
    if benchmark_cb is not None:
        callbacks.append(benchmark_cb)
    if eval_dataset is not None:
        callbacks.append(EvaluateAndSaveFinalStepCallback())

    # -------------------------------------------------------------------
    # 8. Create Trainer
    # -------------------------------------------------------------------
    trainer = Chronos2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters())
        has_sm80 = (torch.cuda.is_available()
                    and torch.cuda.get_device_capability()[0] >= 8)
        n_quantiles = len(config.get("quantiles", [0.1, 0.5, 0.9]))
        max_out_patches = config.get("max_output_patches", 64)
        warmup_ratio = config.get("warmup_ratio", 0.01)
        warmup_steps = config.get("warmup_steps", 0)
        expected_warmup = warmup_steps if warmup_steps > 0 else int(config["max_steps"] * warmup_ratio)
        total_tokens_est = effective_batch * config["max_steps"] * context_length

        logger.info("")
        logger.info("╔" + "═" * 72 + "╗")
        logger.info("║" + "  TRAINING CONFIGURATION".ljust(72) + "║")
        logger.info("╠" + "═" * 72 + "╣")

        # Model
        logger.info("║" + "  ┌─ Model ─────────────────────────────────────────────────────────┐".ljust(72) + "║")
        mode_str = "Pretraining (random init)" if random_init else f"Fine-tuning ({model_id})"
        logger.info("║" + f"  │  Mode:           {mode_str}".ljust(72) + "║")
        logger.info("║" + f"  │  Parameters:     {n_params / 1e6:.1f}M ({n_params:,})".ljust(72) + "║")
        logger.info("║" + f"  │  Context:        {context_length} -> {prediction_length} (patch={output_patch_size})".ljust(72) + "║")
        logger.info("║" + f"  │  Quantiles:      {n_quantiles}  Max output patches: {max_out_patches}".ljust(72) + "║")
        logger.info("║" + f"  │  Precision:      bf16={has_sm80}, tf32={has_sm80}".ljust(72) + "║")
        logger.info("║" + f"  │  Grad ckpt:      {use_grad_ckpt}".ljust(72) + "║")
        logger.info("║" + "  └─────────────────────────────────────────────────────────────────┘".ljust(72) + "║")

        # Optimization
        logger.info("║" + "  ┌─ Optimization ──────────────────────────────────────────────────┐".ljust(72) + "║")
        logger.info("║" + f"  │  Max steps:      {config['max_steps']:,}".ljust(72) + "║")
        logger.info("║" + f"  │  Batch/device:   {batch_size}  x  {world_size} GPUs  x  {grad_accum} accum  =  {effective_batch} effective".ljust(72) + "║")
        logger.info("║" + f"  │  Learning rate:  {config.get('learning_rate', 1e-4):.1e} ({config.get('lr_scheduler_type', 'cosine')})".ljust(72) + "║")
        logger.info("║" + f"  │  Warmup steps:   {expected_warmup:,}".ljust(72) + "║")
        logger.info("║" + f"  │  Optimizer:      {config.get('optim', 'adamw_torch_fused')}".ljust(72) + "║")
        logger.info("║" + f"  │  Weight decay:   {config.get('weight_decay', 0.01)}".ljust(72) + "║")
        logger.info("║" + f"  │  Max grad norm:  {max_grad_norm}".ljust(72) + "║")
        logger.info("║" + f"  │  Est. tokens:    {total_tokens_est / 1e9:.1f}B total".ljust(72) + "║")
        logger.info("║" + "  └─────────────────────────────────────────────────────────────────┘".ljust(72) + "║")

        # Distributed
        logger.info("║" + "  ┌─ Distributed ───────────────────────────────────────────────────┐".ljust(72) + "║")
        fsdp_str = config.get("fsdp", "")
        if fsdp_str:
            fsdp_cfg = config.get("fsdp_config", {})
            wrap_cls = fsdp_cfg.get("fsdp_transformer_layer_cls_to_wrap", "N/A") if fsdp_cfg else "N/A"
            logger.info("║" + f"  │  Strategy:      FSDP ({fsdp_str})".ljust(72) + "║")
            logger.info("║" + f"  │  Wrap layer:    {wrap_cls}".ljust(72) + "║")
        else:
            logger.info("║" + f"  │  Strategy:      DDP (no FSDP)".ljust(72) + "║")
        logger.info("║" + f"  │  World size:    {world_size} GPUs".ljust(72) + "║")
        logger.info("║" + "  └─────────────────────────────────────────────────────────────────┘".ljust(72) + "║")

        # Data
        logger.info("║" + "  ┌─ Data ──────────────────────────────────────────────────────────┐".ljust(72) + "║")
        logger.info("║" + f"  │  Source:         lazy Arrow (on-demand reads)".ljust(72) + "║")
        logger.info("║" + f"  │  Train series:   {len(train_dataset.tasks):,}".ljust(72) + "║")
        for i, path in enumerate(training_data_paths):
            prob = probabilities[i] if i < len(probabilities) else 0
            logger.info("║" + f"  │    [{i}] {Path(path).name} (weight={prob:.2f})".ljust(72) + "║")
        if eval_dataset:
            logger.info("║" + f"  │  Val series:    {len(eval_dataset.tasks):,} (eager)".ljust(72) + "║")
        logger.info("║" + "  └─────────────────────────────────────────────────────────────────┘".ljust(72) + "║")

        # Monitoring
        logger.info("║" + "  ┌─ Monitoring ────────────────────────────────────────────────────┐".ljust(72) + "║")
        logger.info("║" + f"  │  Health reports: every {health_report_interval} steps".ljust(72) + "║")
        logger.info("║" + f"  │  Log steps:      every {config.get('log_steps', 100)} steps".ljust(72) + "║")
        logger.info("║" + f"  │  Save steps:     every {config.get('save_steps', 5000)} steps".ljust(72) + "║")
        if benchmark_cb:
            logger.info("║" + f"  │  Benchmark:     every {benchmark_eval_steps} steps ({Path(benchmark_config_path).name})".ljust(72) + "║")
            logger.info("║" + f"  │  Top-K ckpts:   {benchmark_top_k} (by WQL)".ljust(72) + "║")
        else:
            logger.info("║" + f"  │  Benchmark:     disabled (set benchmark_config in yaml)".ljust(72) + "║")
        logger.info("║" + f"  │  Output dir:    {output_dir}".ljust(72) + "║")
        logger.info("║" + f"  │  Report to:     {config.get('report_to', 'tensorboard')}".ljust(72) + "║")
        logger.info("║" + "  └─────────────────────────────────────────────────────────────────┘".ljust(72) + "║")

        logger.info("╚" + "═" * 72 + "╝")
        logger.info("")

    # -------------------------------------------------------------------
    # 9. Start training
    # -------------------------------------------------------------------
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint", None))

    # -------------------------------------------------------------------
    # 10. Save final model (rank 0 only)
    # -------------------------------------------------------------------
    if is_main_process():
        final_path = Path(output_dir) / "final-checkpoint"

        # Update config for inference (max context used during training)
        model.chronos_config.context_length = max(
            model.chronos_config.context_length, context_length
        )
        model.chronos_config.max_output_patches = max(
            model.chronos_config.max_output_patches,
            math.ceil(prediction_length / output_patch_size),
        )
        model.config.chronos_config = model.chronos_config.__dict__

        trainer.save_model(str(final_path))
        logger.info(f"Final model saved to {final_path}")

    # Synchronize all processes before exit
    if dist.is_initialized():
        dist.barrier()


# ===========================================================================
# Entry Point
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chronos-2 Training Script (pretraining + fine-tuning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (single GPU)
  python train_chronos2.py --config configs/chronos2-test.yaml --max-steps 50

  # Pretraining (8x A100, via train.sh)
  bash train.sh --config configs/chronos2-base.yaml

  # Fine-tuning from pretrained model
  bash train.sh --config configs/chronos2-base.yaml --no-random-init --model-id amazon/chronos-2

  # Resume from checkpoint
  bash train.sh --config configs/chronos2-base.yaml --resume-from-checkpoint output/checkpoint-50000
        """,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    # Override config values via CLI
    parser.add_argument("--model-id", type=str, default=None,
                        help="HuggingFace model ID or local path for fine-tuning")
    parser.add_argument("--random-init", action="store_true", default=None,
                        help="Train from scratch (random initialization)")
    parser.add_argument("--no-random-init", dest="random_init", action="store_false",
                        help="Fine-tune from pretrained model (requires --model-id)")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--prediction-length", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--max-train-series", type=int, default=None,
                        help="Limit number of training series (for quick tests)")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override config with CLI args
    cli_overrides = {
        "model_id": args.model_id,
        "random_init": args.random_init,
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "output_dir": args.output_dir,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "max_train_series": args.max_train_series,
        "seed": args.seed,
    }

    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value

    run_training(config)


if __name__ == "__main__":
    main()
