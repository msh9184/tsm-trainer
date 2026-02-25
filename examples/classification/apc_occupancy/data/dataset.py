"""Occupancy detection dataset with sliding window construction.

Converts preprocessed (n_timesteps, n_channels) arrays into sliding window
samples shaped (n_channels, seq_len) suitable for MantisV2 input.

The label for each window is the label at the *last* time step of the window,
representing the occupancy state at the time of classification.

Supports two construction modes:
  - Legacy: sensor and label arrays are same length (inner-joined).
  - P4 pipeline: full sensor array + label array with -1 for unlabeled
    timesteps. Context windows can reference sensor data outside the
    labeled time range while only using labeled timesteps as targets.

MantisV2 constraints:
  - seq_len must be a multiple of 32 (num_patches)
  - The model's Conv1d expects in_channels=1 and processes each channel
    independently. The trainer concatenates channel embeddings internally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for sliding window dataset construction.

    Attributes:
        seq_len: Window length in time steps. Must be a multiple of 32.
            Default 288 = 24h at 5-min bins.
        stride: Stride between consecutive windows. Default 1 (maximum
            overlap for dense predictions).
        target_seq_len: If set, interpolate each window to this length
            for MantisV2. Must be multiple of 32. Useful when seq_len
            differs from MantisV2's pretrained length (512).
    """
    seq_len: int = 288
    stride: int = 1
    target_seq_len: int | None = None

    def __post_init__(self):
        if self.seq_len % 32 != 0:
            raise ValueError(
                f"seq_len must be a multiple of 32 for MantisV2, got {self.seq_len}"
            )
        if self.target_seq_len is not None and self.target_seq_len % 32 != 0:
            raise ValueError(
                f"target_seq_len must be a multiple of 32, got {self.target_seq_len}"
            )


class OccupancyDataset(Dataset):
    """Sliding window dataset for occupancy detection.

    Each sample is a (n_channels, seq_len) window of sensor readings
    with a binary label (0=empty, 1=occupied) at the window's end.

    Supports cross-boundary context: sensor_array may be longer than the
    labeled region. Only timesteps with label >= 0 are used as window
    targets, but the sensor context window can extend into unlabeled regions.

    Parameters
    ----------
    sensor_array : np.ndarray
        Shape (n_timesteps, n_channels). Float32 sensor readings.
    label_array : np.ndarray
        Shape (n_timesteps,). Int64 labels. -1 = unlabeled (skipped).
    config : DatasetConfig
        Window construction parameters.
    """

    def __init__(
        self,
        sensor_array: np.ndarray,
        label_array: np.ndarray,
        config: DatasetConfig | None = None,
    ):
        if config is None:
            config = DatasetConfig()

        self.config = config
        n_timesteps, n_channels = sensor_array.shape

        if n_timesteps != len(label_array):
            raise ValueError(
                f"sensor_array ({n_timesteps}) and label_array ({len(label_array)}) "
                f"must have the same length"
            )

        if n_timesteps < config.seq_len:
            raise ValueError(
                f"Not enough timesteps ({n_timesteps}) for window size "
                f"({config.seq_len})"
            )

        # Find valid target indices: timesteps with label >= 0 AND
        # enough preceding sensor data for a full context window.
        labeled_mask = label_array >= 0
        labeled_indices = np.where(labeled_mask)[0]

        # Filter: only indices where window [idx - seq_len + 1, idx] fits
        min_idx = config.seq_len - 1
        valid_indices = labeled_indices[labeled_indices >= min_idx]

        # Apply stride: take every stride-th from the valid targets
        if config.stride > 1:
            valid_indices = valid_indices[::config.stride]

        self.n_windows = len(valid_indices)
        self.n_channels = n_channels

        if self.n_windows == 0:
            raise ValueError(
                f"No valid windows: {labeled_mask.sum()} labeled timesteps, "
                f"seq_len={config.seq_len}, min valid index={min_idx}"
            )

        # Pre-extract all windows: (n_windows, n_channels, seq_len)
        windows = np.stack([
            sensor_array[i - config.seq_len + 1 : i + 1].T  # (n_channels, seq_len)
            for i in valid_indices
        ])
        labels = label_array[valid_indices]

        self.windows = windows.astype(np.float32)
        self.labels = labels.astype(np.int64)

        n_occ = (labels == 1).sum()
        n_emp = (labels == 0).sum()
        logger.info(
            "OccupancyDataset: %d windows (seq_len=%d, stride=%d, channels=%d), "
            "class dist: {0: %d, 1: %d}",
            self.n_windows, config.seq_len, config.stride, n_channels,
            n_emp, n_occ,
        )

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """Return (window, label) pair.

        Returns
        -------
        window : np.ndarray
            Shape (n_channels, seq_len) or (n_channels, target_seq_len)
            if interpolation is configured.
        label : int
            Binary label (0 or 1).
        """
        window = self.windows[idx]  # (n_channels, seq_len)

        # Optionally resize to target_seq_len via linear interpolation
        if self.config.target_seq_len is not None:
            tensor = torch.from_numpy(window).unsqueeze(0)  # (1, C, L)
            tensor = F.interpolate(
                tensor, size=self.config.target_seq_len,
                mode="linear", align_corners=False,
            )
            window = tensor.squeeze(0).numpy()  # (C, target_seq_len)

        return window, self.labels[idx]

    def get_numpy_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return all data as numpy arrays for sklearn-style training.

        Returns
        -------
        X : np.ndarray
            Shape (n_windows, n_channels, effective_seq_len).
        y : np.ndarray
            Shape (n_windows,). Binary labels.
        """
        if self.config.target_seq_len is not None:
            # Resize all windows in one batch
            tensor = torch.from_numpy(self.windows)  # (N, C, L)
            tensor = F.interpolate(
                tensor, size=self.config.target_seq_len,
                mode="linear", align_corners=False,
            )
            return tensor.numpy(), self.labels.copy()

        return self.windows.copy(), self.labels.copy()


def create_datasets_from_splits(
    sensor_array: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    config: DatasetConfig | None = None,
) -> tuple[OccupancyDataset, OccupancyDataset]:
    """Create train/test datasets from a single sensor array with split labels.

    This is the P4 pipeline entry point. Both label arrays are aligned to
    sensor_array (same length), with -1 for timesteps outside the split's
    labeled region.

    Parameters
    ----------
    sensor_array : np.ndarray
        Shape (n_timesteps, n_channels). Full sensor data.
    train_labels : np.ndarray
        Shape (n_timesteps,). Train labels (-1 for non-train timesteps).
    test_labels : np.ndarray
        Shape (n_timesteps,). Test labels (-1 for non-test timesteps).
    config : DatasetConfig, optional
        Shared window configuration.

    Returns
    -------
    train_dataset, test_dataset : OccupancyDataset
    """
    if config is None:
        config = DatasetConfig()

    train_ds = OccupancyDataset(sensor_array, train_labels, config)
    test_ds = OccupancyDataset(sensor_array, test_labels, config)

    logger.info(
        "Created P4 datasets: train=%d, test=%d windows "
        "(shared sensor array: %d timesteps, %d channels)",
        len(train_ds), len(test_ds),
        sensor_array.shape[0], sensor_array.shape[1],
    )
    return train_ds, test_ds


def create_datasets(
    train_sensor: np.ndarray,
    train_labels: np.ndarray,
    test_sensor: np.ndarray,
    test_labels: np.ndarray,
    config: DatasetConfig | None = None,
) -> tuple[OccupancyDataset, OccupancyDataset]:
    """Create train and test OccupancyDataset instances (legacy API).

    Parameters
    ----------
    train_sensor, test_sensor : np.ndarray
        Shape (n_timesteps, n_channels).
    train_labels, test_labels : np.ndarray
        Shape (n_timesteps,).
    config : DatasetConfig, optional
        Shared window configuration.

    Returns
    -------
    train_dataset, test_dataset : OccupancyDataset
    """
    if config is None:
        config = DatasetConfig()

    train_ds = OccupancyDataset(train_sensor, train_labels, config)
    test_ds = OccupancyDataset(test_sensor, test_labels, config)

    logger.info(
        "Created datasets: train=%d, test=%d windows",
        len(train_ds), len(test_ds),
    )
    return train_ds, test_ds
