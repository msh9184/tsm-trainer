"""Occupancy detection dataset with sliding window construction.

Converts preprocessed (n_timesteps, n_channels) arrays into sliding window
samples shaped (n_channels, seq_len) suitable for MantisV2 input.

The label for each window is the label at the *last* time step of the window,
representing the occupancy state at the time of classification.

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
            Default 64 = 5h20m at 5-min bins.
        stride: Stride between consecutive windows. Default 1 (maximum
            overlap for dense predictions).
        target_seq_len: If set, interpolate each window to this length
            for MantisV2. Must be multiple of 32. Useful when seq_len
            differs from MantisV2's pretrained length (512).
    """
    seq_len: int = 64
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

    Parameters
    ----------
    sensor_array : np.ndarray
        Shape (n_timesteps, n_channels). Float32 sensor readings.
    label_array : np.ndarray
        Shape (n_timesteps,). Int64 binary labels.
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

        if n_timesteps < config.seq_len:
            raise ValueError(
                f"Not enough timesteps ({n_timesteps}) for window size "
                f"({config.seq_len})"
            )

        # Build window indices
        starts = list(range(0, n_timesteps - config.seq_len + 1, config.stride))
        self.n_windows = len(starts)
        self.n_channels = n_channels

        # Pre-extract all windows: (n_windows, n_channels, seq_len)
        windows = np.stack([
            sensor_array[s : s + config.seq_len].T  # (n_channels, seq_len)
            for s in starts
        ])
        # Labels: value at the last timestep of each window
        labels = np.array([label_array[s + config.seq_len - 1] for s in starts])

        self.windows = windows.astype(np.float32)
        self.labels = labels.astype(np.int64)

        logger.info(
            "OccupancyDataset: %d windows (seq_len=%d, stride=%d, channels=%d), "
            "class dist: {0: %d, 1: %d}",
            self.n_windows, config.seq_len, config.stride, n_channels,
            (labels == 0).sum(), (labels == 1).sum(),
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


def create_datasets(
    train_sensor: np.ndarray,
    train_labels: np.ndarray,
    test_sensor: np.ndarray,
    test_labels: np.ndarray,
    config: DatasetConfig | None = None,
) -> tuple[OccupancyDataset, OccupancyDataset]:
    """Create train and test OccupancyDataset instances.

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
