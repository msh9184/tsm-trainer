"""Distributed evaluation utilities for FSDP-compatible benchmark evaluation.

When training with FSDP across multiple GPUs, all ranks must call model.forward()
the same number of times (FSDP triggers internal all-gather collectives). This module
provides utilities to distribute evaluation series across ranks, synchronize batch
counts, and gather predictions back to rank-0 for metric computation.

Architecture:
    1. ALL ranks: Load dataset, shard series by round-robin
    2. ALL ranks: Synchronize batch count via all_reduce(MAX)
    3. ALL ranks: Forward pass loop (real + padding batches)
    4. ALL ranks: Gather predictions to rank-0
    5. Rank-0: Reconstruct correct order, compute metrics

Key design decisions:
- Round-robin sharding (not contiguous) for balanced load across varying series lengths
- all_reduce(MAX) ensures identical loop count across ranks (prevents FSDP deadlock)
- Padding batches use real tensor shapes with dummy data; predictions are discarded
- all_gather with fixed-size tensors for cross-rank communication
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from .forecaster import BaseForecaster

logger = logging.getLogger(__name__)


def get_dist_info() -> tuple[int, int, bool]:
    """Return (rank, world_size, is_distributed)."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), True
    return 0, 1, False


def shard_series_indices(total: int, rank: int, world_size: int) -> list[int]:
    """Round-robin assignment: rank k gets indices k, k+world_size, k+2*world_size, ...

    This ensures balanced load across ranks even when series have varying lengths,
    since adjacent series in a benchmark config tend to come from the same dataset
    and have similar properties.
    """
    return list(range(rank, total, world_size))


def synchronized_batch_count(local_count: int, device: torch.device) -> int:
    """Synchronize batch count across all ranks via all_reduce(MAX).

    This is CRITICAL for FSDP: all ranks must call model.forward() the exact
    same number of times. The rank with the most batches determines the count
    for all ranks. Ranks with fewer real batches run padding batches.
    """
    count_tensor = torch.tensor([local_count], dtype=torch.long, device=device)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.MAX)
    return count_tensor.item()


def is_fsdp_model(model) -> bool:
    """Check if model is wrapped in FSDP."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        return isinstance(model, FSDP)
    except ImportError:
        return False


def unwrap_model(model):
    """Unwrap FSDP/DDP wrappers to access the inner model attributes.

    Traverses _fsdp_wrapped_module and .module attributes to reach the
    underlying Chronos2Model. Used for accessing chronos_config while
    keeping the wrapper for forward() calls.
    """
    inner = model
    while hasattr(inner, "_fsdp_wrapped_module"):
        inner = inner._fsdp_wrapped_module
    while hasattr(inner, "module"):
        inner = inner.module
    return inner


def broadcast_error(local_error: bool, device: torch.device) -> bool:
    """Synchronize error status across all ranks via all_reduce(MAX).

    If ANY rank has an error, returns True on ALL ranks. This prevents
    NCCL deadlocks where some ranks proceed to distributed operations
    while others skip them due to local exceptions.

    Parameters
    ----------
    local_error : bool
        Whether this rank encountered an error.
    device : torch.device
        Device for the synchronization tensor.

    Returns
    -------
    bool
        True if any rank had an error, False if all ranks succeeded.
    """
    error_flag = torch.tensor([int(local_error)], dtype=torch.long, device=device)
    dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
    return error_flag.item() > 0


def generate_forecasts_distributed(
    test_data_input,
    forecaster: BaseForecaster,
    prediction_length: int,
    quantile_levels: list[float],
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[np.ndarray | None, list[int]]:
    """Generate forecasts with series distributed across ranks.

    1. Shard test_data_input by series index (round-robin)
    2. Compute local batch count, synchronize via all_reduce(MAX)
    3. Run forward passes (real + padding batches)
    4. Return (local_predictions, local_indices)

    Parameters
    ----------
    test_data_input : Iterable
        Test data input from GluonTS split (list of dicts with "target" key).
    forecaster : BaseForecaster
        Model forecaster (must be FSDP-wrapped for distributed).
    prediction_length : int
        Forecast horizon.
    quantile_levels : list[float]
        Quantile levels to predict.
    batch_size : int
        Batch size per rank for inference.
    rank : int
        Current rank.
    world_size : int
        Total number of ranks.
    device : torch.device
        Device for tensors.

    Returns
    -------
    tuple[np.ndarray | None, list[int]]
        (local_predictions with shape (n_local, Q, H), local_series_indices)
        local_predictions is None if this rank has 0 series.
    """
    # Materialize test_data_input to list for indexing
    all_inputs = list(test_data_input)
    total_series = len(all_inputs)

    # Shard series indices
    local_indices = shard_series_indices(total_series, rank, world_size)
    local_inputs = [all_inputs[i] for i in local_indices]
    n_local = len(local_inputs)

    # Compute local batch count and synchronize
    local_batch_count = math.ceil(n_local / batch_size) if n_local > 0 else 0
    global_batch_count = synchronized_batch_count(local_batch_count, device)

    if rank == 0:
        logger.info(
            f"  Distributed eval: {total_series} series across {world_size} ranks, "
            f"batch_size={batch_size}, global_batches={global_batch_count}"
        )

    # Dummy context for padding batches — must be long enough for patching
    # (at least min_past=60 typically, use 128 as safe minimum)
    dummy_context_len = max(prediction_length * 2, 128)

    # Run forward passes
    local_predictions = []
    for batch_idx in range(global_batch_count):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_local)

        if start < n_local:
            # Real batch: convert targets to tensors
            batch_inputs = local_inputs[start:end]
            context = [torch.tensor(entry["target"], dtype=torch.float32) for entry in batch_inputs]
            preds = forecaster.predict_quantiles(
                context,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                batch_size=len(context),  # Force single internal batch (FSDP safety)
            )
            local_predictions.append(preds)
        else:
            # Padding batch: run a dummy forward pass to keep FSDP collectives in sync.
            # Uses the same prediction_length so autoregressive unrolling step count matches.
            dummy_context = [torch.zeros(dummy_context_len, dtype=torch.float32)]
            _ = forecaster.predict_quantiles(
                dummy_context,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                batch_size=1,  # Single item, single internal batch
            )
            # Do NOT append — padding predictions are discarded

    if local_predictions:
        local_predictions = np.concatenate(local_predictions, axis=0)
    else:
        local_predictions = None

    return local_predictions, local_indices


def gather_forecasts_to_rank0(
    local_predictions: np.ndarray | None,
    local_indices: list[int],
    total_series: int,
    n_quantiles: int,
    prediction_length: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> np.ndarray | None:
    """Gather predictions from all ranks to rank-0.

    Uses torch.distributed.all_gather with padding for uneven splits.
    Rank-0 reconstructs correct order using indices.

    Parameters
    ----------
    local_predictions : np.ndarray or None
        Shape (n_local, Q, H) on this rank. None if rank has 0 series.
    local_indices : list[int]
        Series indices assigned to this rank.
    total_series : int
        Total number of series across all ranks.
    n_quantiles : int
        Number of quantile levels.
    prediction_length : int
        Forecast horizon.
    rank : int
        Current rank.
    world_size : int
        Total number of ranks.
    device : torch.device
        Device for tensors.

    Returns
    -------
    np.ndarray or None
        Full (N, Q, H) array on rank-0, None on other ranks.
    """
    n_local = len(local_indices)

    # Gather local counts to determine max size for padding
    local_count = torch.tensor([n_local], dtype=torch.long, device=device)
    all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_counts, local_count)
    max_count = max(c.item() for c in all_counts)

    if max_count == 0:
        # No series at all
        return np.zeros((0, n_quantiles, prediction_length)) if rank == 0 else None

    # Pad local predictions to max_count for uniform all_gather
    padded = torch.zeros(max_count, n_quantiles, prediction_length, device=device)
    if local_predictions is not None and n_local > 0:
        padded[:n_local] = torch.from_numpy(local_predictions).to(device)

    # Gather all padded predictions
    gathered = [
        torch.zeros(max_count, n_quantiles, prediction_length, device=device)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered, padded)

    # Gather indices
    padded_indices = torch.full((max_count,), -1, dtype=torch.long, device=device)
    if n_local > 0:
        padded_indices[:n_local] = torch.tensor(local_indices, dtype=torch.long, device=device)

    gathered_indices = [
        torch.full((max_count,), -1, dtype=torch.long, device=device)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered_indices, padded_indices)

    if rank == 0:
        # Reconstruct in correct order
        full_predictions = np.zeros((total_series, n_quantiles, prediction_length), dtype=np.float32)
        for r in range(world_size):
            count_r = all_counts[r].item()
            preds_r = gathered[r][:count_r].cpu().numpy()
            indices_r = gathered_indices[r][:count_r].cpu().numpy()
            for local_idx, global_idx in enumerate(indices_r):
                if global_idx >= 0:
                    full_predictions[global_idx] = preds_r[local_idx]

        return full_predictions
    return None
