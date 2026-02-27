"""Event dataset for enter/leave classification.

Unlike the occupancy pipeline (dense sliding windows over thousands of
timesteps), this module creates exactly ONE context window per event.

Supports two context modes:

**Backward mode** (legacy):
    For an event at time t, the window contains sensor data from
    [t - seq_len + 1, t] -- the preceding ``seq_len`` minutes.
    Events with insufficient preceding data are zero-padded from the left.

**Bidirectional mode** (default):
    For an event at time t, the window contains sensor data from
    [t - context_before, t + context_after] -- past and future context.
    Events near sensor array boundaries are zero-padded from both sides.

MantisV2 constraints:
  - The input length must be a multiple of 32 (patch_size)
  - In backward mode: seq_len must be a multiple of 32
  - In bidirectional mode: effective window is auto-interpolated to
    the nearest multiple of 32 (via target_seq_len)
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

    Supports two context modes:

    **Backward mode** (``context_mode="backward"``):
        Window covers ``seq_len`` timesteps before the event: [t - seq_len + 1, t].
        seq_len must be a multiple of 32 for MantisV2. Default: 512 min (8.5h).

    **Bidirectional mode** (``context_mode="bidirectional"``):
        Window covers past and future context around the event:
        [t - context_before, t + context_after]. Total window length =
        context_before + 1 + context_after timesteps.
        Auto-interpolated to MantisV2-compatible length (multiple of 32).

    Attributes:
        context_mode: "backward" or "bidirectional".
        seq_len: Backward window length in timesteps (backward mode only).
            Must be a multiple of 32 for MantisV2. Default 512 = 8.5h.
        context_before: Minutes before event (bidirectional mode). Default 4.
        context_after: Minutes after event (bidirectional mode). Default 4.
        target_seq_len: If set, interpolate each window to this length.
            Must be multiple of 32. Auto-computed for bidirectional if not set.
    """

    context_mode: str = "bidirectional"
    seq_len: int = 512
    context_before: int = 4
    context_after: int = 4
    target_seq_len: int | None = None

    @property
    def effective_seq_len(self) -> int:
        """Actual number of timesteps in each raw context window."""
        if self.context_mode == "bidirectional":
            return self.context_before + 1 + self.context_after
        return self.seq_len

    def describe(self) -> str:
        """Human-readable description of the context window configuration."""
        if self.context_mode == "bidirectional":
            eff = self.effective_seq_len
            tgt = self.target_seq_len
            desc = (
                f"bidirectional: {self.context_before}+1+{self.context_after}"
                f"={eff} timesteps"
            )
            if tgt is not None and tgt != eff:
                desc += f" -> interp to {tgt}"
            return desc
        tgt = self.target_seq_len
        desc = f"backward: {self.seq_len} min ({self.seq_len / 60:.1f}h)"
        if tgt is not None and tgt != self.seq_len:
            desc += f" -> interp to {tgt}"
        return desc

    def __post_init__(self):
        if self.context_mode not in ("backward", "bidirectional"):
            raise ValueError(
                f"context_mode must be 'backward' or 'bidirectional', "
                f"got {self.context_mode!r}"
            )

        if self.context_mode == "backward":
            if self.seq_len % 32 != 0:
                raise ValueError(
                    f"seq_len must be a multiple of 32 for MantisV2, "
                    f"got {self.seq_len}"
                )
        else:  # bidirectional
            if self.context_before < 0 or self.context_after < 0:
                raise ValueError(
                    f"context_before and context_after must be >= 0, "
                    f"got before={self.context_before}, after={self.context_after}"
                )
            # Auto-compute target_seq_len if effective length isn't MantisV2-compatible
            eff = self.effective_seq_len
            if self.target_seq_len is None and eff % 32 != 0:
                self.target_seq_len = ((eff + 31) // 32) * 32

        if self.target_seq_len is not None and self.target_seq_len % 32 != 0:
            raise ValueError(
                f"target_seq_len must be a multiple of 32, got {self.target_seq_len}"
            )


def build_dataset_config(ds_cfg: dict) -> EventDatasetConfig:
    """Build EventDatasetConfig from YAML dict with smart mode detection.

    Detection logic:
      - If ``context_mode`` is explicit -> use it.
      - Elif ``context_before`` or ``context_after`` present -> bidirectional.
      - Elif ``seq_len`` present -> backward.
      - Else -> bidirectional with defaults (before=4, after=4).

    Parameters
    ----------
    ds_cfg : dict
        The ``dataset:`` section from the YAML config.

    Returns
    -------
    EventDatasetConfig
    """
    if "context_mode" in ds_cfg:
        mode = ds_cfg["context_mode"]
    elif "context_before" in ds_cfg or "context_after" in ds_cfg:
        mode = "bidirectional"
    elif "seq_len" in ds_cfg:
        mode = "backward"
    else:
        mode = "bidirectional"

    if mode == "backward":
        return EventDatasetConfig(
            context_mode="backward",
            seq_len=ds_cfg.get("seq_len", 512),
            target_seq_len=ds_cfg.get("target_seq_len"),
        )
    return EventDatasetConfig(
        context_mode="bidirectional",
        context_before=ds_cfg.get("context_before", 4),
        context_after=ds_cfg.get("context_after", 4),
        target_seq_len=ds_cfg.get("target_seq_len"),
    )


class EventDataset(Dataset):
    """One context window per event for enter/leave classification.

    Supports two context modes:

    **Backward**: For an event at time t, extracts
    ``sensor_data[t - seq_len + 1 : t + 1]``.
    Events with insufficient preceding data are zero-padded from the left.

    **Bidirectional**: For an event at time t, extracts
    ``sensor_data[t - context_before : t + context_after + 1]``.
    Events near sensor array boundaries are zero-padded from both sides.

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

        # Extract context windows per event
        n_sensor = len(sensor_array)
        windows = []
        n_padded = 0

        if config.context_mode == "bidirectional":
            ctx_before = config.context_before
            ctx_after = config.context_after

            for i, idx in enumerate(event_indices):
                start = idx - ctx_before
                end = idx + ctx_after + 1  # exclusive

                if start >= 0 and end <= n_sensor:
                    window = sensor_array[start:end]
                else:
                    # Zero-pad from left and/or right as needed
                    left_pad = max(0, -start)
                    right_pad = max(0, end - n_sensor)
                    actual_start = max(0, start)
                    actual_end = min(n_sensor, end)
                    available = sensor_array[actual_start:actual_end]

                    parts = []
                    if left_pad > 0:
                        parts.append(
                            np.zeros((left_pad, self.n_channels), dtype=np.float32)
                        )
                    parts.append(available)
                    if right_pad > 0:
                        parts.append(
                            np.zeros((right_pad, self.n_channels), dtype=np.float32)
                        )
                    window = np.concatenate(parts, axis=0)
                    n_padded += 1

                windows.append(window.T)  # (n_channels, eff_len)

        else:  # backward mode
            seq_len = config.seq_len

            for i, idx in enumerate(event_indices):
                start = idx - seq_len + 1
                if start >= 0:
                    window = sensor_array[start : idx + 1]  # (seq_len, n_channels)
                else:
                    # Zero-pad from the left
                    available = sensor_array[0 : idx + 1]  # (idx+1, n_channels)
                    pad_len = seq_len - len(available)
                    padding = np.zeros(
                        (pad_len, self.n_channels), dtype=np.float32
                    )
                    window = np.concatenate([padding, available], axis=0)
                    n_padded += 1

                windows.append(window.T)  # (n_channels, seq_len)

        self.windows = np.stack(windows).astype(np.float32)
        self.labels = np.asarray(event_labels, dtype=np.int64)
        self.event_timestamps = event_timestamps
        self._event_indices = event_indices

        if n_padded > 0:
            logger.warning(
                "%d/%d events required zero-padding (insufficient sensor data at boundary)",
                n_padded, n_events,
            )

        unique, counts = np.unique(self.labels, return_counts=True)
        dist_str = ", ".join(f"{int(u)}:{int(c)}" for u, c in zip(unique, counts))
        logger.info(
            "EventDataset: %d events (%s, channels=%d), class dist: {%s}",
            n_events, config.describe(), self.n_channels, dist_str,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """Return (window, label) pair.

        Returns
        -------
        window : np.ndarray
            Shape (n_channels, effective_seq_len) or (n_channels, target_seq_len)
            if interpolation is configured.
        label : int
            Class label.
        """
        window = self.windows[idx]  # (n_channels, effective_seq_len)

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
