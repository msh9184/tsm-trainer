"""Event dataset for enter/leave classification.

Unlike the occupancy pipeline (dense sliding windows over thousands of
timesteps), this module creates exactly ONE context window per event.
For an event at time t, the window contains sensor data from
[t - seq_len + 1, t] — the preceding ``seq_len`` minutes of sensor readings.

Events with insufficient preceding sensor data are zero-padded from the left.

MantisV2 constraints:
  - seq_len must be a multiple of 32 (num_patches)
  - At native 1-min resolution with seq_len=512, each window covers 8.5 hours
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class EventDatasetConfig:
    """Configuration for event dataset construction.

    Attributes:
        seq_len: Context window length in timesteps (minutes at 1-min resolution).
            Must be a multiple of 32 for MantisV2. Default 512 = 8.5h.
        target_seq_len: If set, interpolate each window to this length.
            Must be multiple of 32. None means no interpolation.
    """

    seq_len: int = 512
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


class EventDataset(Dataset):
    """One context window per event for enter/leave classification.

    For an event at time t, extracts the backward context window:
        sensor_data[t - seq_len + 1 : t + 1]

    Events with insufficient preceding sensor data (near the start of the
    sensor array) are zero-padded from the left.

    Parameters
    ----------
    sensor_array : np.ndarray
        Shape (n_timesteps, n_channels). Float32 sensor readings.
    sensor_timestamps : pd.DatetimeIndex
        Timestamps for each row of sensor_array.
    event_timestamps : np.ndarray
        Datetime64 timestamps of each event.
    event_labels : np.ndarray
        Integer class labels for each event.
    config : EventDatasetConfig, optional
        Window configuration.
    """

    def __init__(
        self,
        sensor_array: np.ndarray,
        sensor_timestamps: pd.DatetimeIndex,
        event_timestamps: np.ndarray,
        event_labels: np.ndarray,
        config: EventDatasetConfig | None = None,
    ):
        if config is None:
            config = EventDatasetConfig()
        self.config = config
        self.n_channels = sensor_array.shape[1]

        n_events = len(event_timestamps)
        if n_events != len(event_labels):
            raise ValueError(
                f"event_timestamps ({n_events}) and event_labels ({len(event_labels)}) "
                f"must have the same length"
            )

        # Map each event timestamp to the nearest sensor index
        sensor_ts_ns = sensor_timestamps.values.astype("int64")
        event_ts_ns = pd.DatetimeIndex(event_timestamps).values.astype("int64")

        event_indices = np.searchsorted(sensor_ts_ns, event_ts_ns, side="right") - 1
        event_indices = np.clip(event_indices, 0, len(sensor_array) - 1)

        # Extract backward context window per event
        seq_len = config.seq_len
        windows = []
        n_padded = 0

        for i, idx in enumerate(event_indices):
            start = idx - seq_len + 1
            if start >= 0:
                window = sensor_array[start : idx + 1]  # (seq_len, n_channels)
            else:
                # Zero-pad from the left
                available = sensor_array[0 : idx + 1]  # (idx+1, n_channels)
                pad_len = seq_len - len(available)
                padding = np.zeros((pad_len, self.n_channels), dtype=np.float32)
                window = np.concatenate([padding, available], axis=0)
                n_padded += 1

            windows.append(window.T)  # (n_channels, seq_len)

        self.windows = np.stack(windows).astype(np.float32)  # (n_events, n_channels, seq_len)
        self.labels = np.asarray(event_labels, dtype=np.int64)
        self.event_timestamps = event_timestamps
        self._event_indices = event_indices

        if n_padded > 0:
            logger.warning(
                "%d/%d events required zero-padding (insufficient preceding sensor data)",
                n_padded, n_events,
            )

        unique, counts = np.unique(self.labels, return_counts=True)
        dist_str = ", ".join(f"{int(u)}:{int(c)}" for u, c in zip(unique, counts))
        logger.info(
            "EventDataset: %d events (seq_len=%d, channels=%d), class dist: {%s}",
            n_events, seq_len, self.n_channels, dist_str,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """Return (window, label) pair.

        Returns
        -------
        window : np.ndarray
            Shape (n_channels, seq_len) or (n_channels, target_seq_len)
            if interpolation is configured.
        label : int
            Class label.
        """
        window = self.windows[idx]  # (n_channels, seq_len)

        if self.config.target_seq_len is not None:
            tensor = torch.from_numpy(window).unsqueeze(0)  # (1, C, L)
            tensor = F.interpolate(
                tensor, size=self.config.target_seq_len,
                mode="linear", align_corners=False,
            )
            window = tensor.squeeze(0).numpy()  # (C, target_seq_len)

        return window, int(self.labels[idx])

    def get_numpy_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return all data as numpy arrays for sklearn-style training.

        Returns
        -------
        X : np.ndarray
            Shape (n_events, n_channels, effective_seq_len).
        y : np.ndarray
            Shape (n_events,). Class labels.
        """
        if self.config.target_seq_len is not None:
            tensor = torch.from_numpy(self.windows)  # (N, C, L)
            tensor = F.interpolate(
                tensor, size=self.config.target_seq_len,
                mode="linear", align_corners=False,
            )
            return tensor.numpy(), self.labels.copy()

        return self.windows.copy(), self.labels.copy()

    def get_day_groups(self) -> np.ndarray:
        """Return calendar day index per event (for Leave-One-Day-Out CV).

        Returns
        -------
        day_groups : np.ndarray
            Shape (n_events,). Integer index of the calendar day for each event.
            Days are numbered 0, 1, ... in chronological order.
        """
        event_dates = pd.DatetimeIndex(self.event_timestamps).date
        unique_dates = sorted(set(event_dates))
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        return np.array([date_to_idx[d] for d in event_dates], dtype=np.int64)
