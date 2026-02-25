"""Data preprocessing for SmartThings sensor data.

Supports two label formats:
  - "counts": Per-timestep occupancy_counts.csv (time, occupancy)
  - "events": Event-based occupancy_events CSV (time, Status, Head-count, At-home count)

The preprocessing pipeline:
  1. Load sensor CSV (full time range)
  2. Load label CSV (per-timestep counts or event-based)
  3. For events: replay ENTER/LEAVE events to generate per-timestep labels
  4. Align sensor data to label timestamps
  5. Filter/select channels, handle NaN
  6. Optionally inject hour-of-day cyclical features
  7. Return (sensor_array, label_array, channel_names, timestamps)

Extended mode (load_sensor_and_labels):
  Loads the full sensor array separately from labels, allowing context
  windows to reference sensor data outside the labeled time range.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing.

    Attributes:
        sensor_csv: Path to merged sensor data CSV.
        label_csv: Path to label CSV (counts or events format).
        label_format: Label file format. "counts" for per-timestep
            occupancy_counts.csv, "events" for event-based CSV.
        initial_occupancy: Initial occupancy state for event-based labels.
            Used for timesteps before the first event.
        nan_threshold: Drop channels with NaN fraction above this value.
        channels: Explicit list of channel names to use. If None, all
            channels passing nan_threshold are used.
        exclude_channels: Channels to always exclude (substring match).
        binarize: If True, convert occupancy count to binary (>0 -> 1).
        add_time_features: If True, append hour_sin and hour_cos channels
            derived from the timestamp of each row.
    """
    sensor_csv: str | Path = ""
    label_csv: str | Path = ""
    label_format: str = "counts"  # "counts" or "events"
    initial_occupancy: int = 0
    nan_threshold: float = 0.5
    channels: list[str] | None = None
    exclude_channels: list[str] = field(default_factory=list)
    binarize: bool = True
    add_time_features: bool = False


# ---------------------------------------------------------------------------
# Sensor loading
# ---------------------------------------------------------------------------

def _load_sensor_data(path: str | Path) -> pd.DataFrame:
    """Load sensor CSV with timestamp index."""
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index.name = "time"
    df = df.sort_index()
    logger.info(
        "Loaded sensor data: %d rows, %d channels, range [%s, %s]",
        len(df), len(df.columns), df.index.min(), df.index.max(),
    )
    return df


# ---------------------------------------------------------------------------
# Label loading — counts format (legacy)
# ---------------------------------------------------------------------------

def _load_label_counts(path: str | Path, binarize: bool = True) -> pd.Series:
    """Load per-timestep occupancy_counts.csv."""
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index.name = "time"
    df = df.sort_index()

    occupancy = df["occupancy"]
    if binarize:
        occupancy = (occupancy > 0).astype(np.int64)
        logger.info(
            "Loaded labels (counts, binarized): %d rows, "
            "class dist: {0: %d, 1: %d}",
            len(occupancy), (occupancy == 0).sum(), (occupancy == 1).sum(),
        )
    else:
        logger.info(
            "Loaded labels (counts, raw): %d rows, range [%d, %d]",
            len(occupancy), occupancy.min(), occupancy.max(),
        )
    return occupancy


# ---------------------------------------------------------------------------
# Label loading — events format (new)
# ---------------------------------------------------------------------------

def _load_label_events(
    path: str | Path,
    sensor_timestamps: pd.DatetimeIndex,
    initial_occupancy: int = 0,
    binarize: bool = True,
) -> pd.Series:
    """Convert event-based label CSV to per-timestep labels.

    Replays ENTER_HOME/LEAVE_HOME events chronologically and assigns
    an occupancy state to each sensor timestep.

    Parameters
    ----------
    path : str or Path
        Path to event CSV with columns: time, Status, Head-count, At-home count.
    sensor_timestamps : pd.DatetimeIndex
        Timestamps from the sensor data to align labels to.
    initial_occupancy : int
        Occupancy count before the first event.
    binarize : bool
        If True, convert counts to binary (>0 -> 1).

    Returns
    -------
    pd.Series
        Per-timestep labels indexed by sensor_timestamps.
        Timesteps after the last event are marked as -1 (unlabeled).
    """
    events_df = pd.read_csv(path, parse_dates=["time"])
    events_df = events_df.sort_values("time").reset_index(drop=True)

    # Determine the occupancy count column name (flexible naming)
    count_col = None
    for candidate in ["At-home count", "At_home_count", "at_home_count", "occupancy"]:
        if candidate in events_df.columns:
            count_col = candidate
            break

    # Filter out NONE events if Status column exists
    if "Status" in events_df.columns:
        events_df = events_df[events_df["Status"] != "NONE"].copy()

    if count_col is None:
        # If no count column, infer from Status
        logger.info("No count column found; inferring from Status column")
        state = initial_occupancy
        counts = []
        for _, row in events_df.iterrows():
            status = str(row.get("Status", "")).upper()
            if "ENTER" in status:
                state += 1
            elif "LEAVE" in status:
                state = max(0, state - 1)
            counts.append(state)
        events_df["_count"] = counts
        count_col = "_count"

    # Assign labels to each sensor timestep
    event_times = events_df["time"].values
    event_counts = events_df[count_col].values.astype(np.int64)

    labels = np.full(len(sensor_timestamps), -1, dtype=np.int64)
    current_count = initial_occupancy

    ts_array = sensor_timestamps.values
    event_idx = 0
    n_events = len(event_times)

    for i, ts in enumerate(ts_array):
        # Advance event pointer to the latest event at or before this timestep
        while event_idx < n_events and event_times[event_idx] <= ts:
            current_count = int(event_counts[event_idx])
            event_idx += 1

        # Only label timesteps within the event range
        if n_events > 0 and ts >= event_times[0]:
            if event_idx >= n_events and ts > event_times[-1]:
                # After last event: mark as unlabeled
                labels[i] = -1
            else:
                labels[i] = current_count
        else:
            # Before first event: use initial_occupancy
            labels[i] = initial_occupancy

    if binarize:
        # Keep -1 as unlabeled, binarize the rest
        labeled_mask = labels >= 0
        labels[labeled_mask] = (labels[labeled_mask] > 0).astype(np.int64)

    n_labeled = (labels >= 0).sum()
    n_unlabeled = (labels == -1).sum()
    n_occupied = (labels == 1).sum()
    n_empty = (labels == 0).sum()

    logger.info(
        "Loaded labels (events): %d total, %d labeled "
        "(occupied=%d, empty=%d), %d unlabeled",
        len(labels), n_labeled, n_occupied, n_empty, n_unlabeled,
    )

    return pd.Series(labels, index=sensor_timestamps, name="occupancy")


# ---------------------------------------------------------------------------
# Channel filtering and NaN handling
# ---------------------------------------------------------------------------

def _filter_channels(
    df: pd.DataFrame,
    nan_threshold: float,
    channels: list[str] | None,
    exclude_channels: list[str],
) -> pd.DataFrame:
    """Filter sensor channels by NaN fraction and explicit lists."""
    if exclude_channels:
        cols_before = set(df.columns)
        for pattern in exclude_channels:
            matched = [c for c in df.columns if pattern in c]
            df = df.drop(columns=matched, errors="ignore")
        dropped = cols_before - set(df.columns)
        if dropped:
            logger.info("Excluded channels: %s", sorted(dropped))

    if channels is not None:
        available = [c for c in channels if c in df.columns]
        missing = [c for c in channels if c not in df.columns]
        if missing:
            logger.warning("Requested channels not found: %s", missing)
        df = df[available]
        logger.info("Using %d explicitly specified channels", len(available))
        return df

    nan_fracs = df.isna().mean()
    keep = nan_fracs[nan_fracs <= nan_threshold].index.tolist()
    dropped_cols = nan_fracs[nan_fracs > nan_threshold].index.tolist()

    if dropped_cols:
        logger.info(
            "Dropped %d channels exceeding %.0f%% NaN threshold: %s",
            len(dropped_cols), nan_threshold * 100, dropped_cols,
        )

    df = df[keep]
    logger.info("Retained %d channels (NaN threshold=%.2f)", len(keep), nan_threshold)
    return df


def _fill_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill -> backward-fill -> zero-fill remaining NaN."""
    n_nan_before = df.isna().sum().sum()
    df = df.ffill().bfill().fillna(0.0)
    n_nan_after = df.isna().sum().sum()
    logger.info(
        "NaN fill: %d -> %d (filled %d values)",
        n_nan_before, n_nan_after, n_nan_before - n_nan_after,
    )
    return df


# ---------------------------------------------------------------------------
# Time feature injection
# ---------------------------------------------------------------------------

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append hour_sin and hour_cos cyclical features as extra columns.

    Encodes hour-of-day as two cyclical features using sine/cosine
    transformation, ensuring continuity at midnight (23:55 -> 00:00).

    The features range from -1 to +1:
      hour_sin = sin(2*pi*hour/24)
      hour_cos = cos(2*pi*hour/24)
    """
    hour_frac = df.index.hour + df.index.minute / 60.0
    hour_rad = 2.0 * np.pi * hour_frac / 24.0
    df = df.copy()
    df["hour_sin"] = np.sin(hour_rad).astype(np.float32)
    df["hour_cos"] = np.cos(hour_rad).astype(np.float32)
    logger.info("Added time features: hour_sin, hour_cos")
    return df


# ---------------------------------------------------------------------------
# Public API — legacy (backward-compatible)
# ---------------------------------------------------------------------------

def load_and_preprocess(
    config: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Load and preprocess sensor + label data (legacy API).

    Labels and sensors are inner-joined on timestamp. For the new P4
    pipeline with separate train/test labels, use load_sensor_and_labels()
    instead.

    Returns
    -------
    sensor_array : np.ndarray, shape (n_timesteps, n_channels)
    label_array : np.ndarray, shape (n_timesteps,)
    channel_names : list[str]
    timestamps : pd.DatetimeIndex
    """
    sensor_df = _load_sensor_data(config.sensor_csv)

    if config.label_format == "events":
        labels = _load_label_events(
            config.label_csv, sensor_df.index,
            initial_occupancy=config.initial_occupancy,
            binarize=config.binarize,
        )
        # Keep only labeled rows for legacy mode
        labeled_mask = labels >= 0
        sensor_df = sensor_df.loc[labeled_mask]
        labels = labels.loc[labeled_mask]
    else:
        labels = _load_label_counts(config.label_csv, binarize=config.binarize)
        # Inner join on timestamp
        common_idx = sensor_df.index.intersection(labels.index)
        if len(common_idx) == 0:
            raise ValueError(
                f"No overlapping timestamps between sensor data "
                f"[{sensor_df.index.min()}, {sensor_df.index.max()}] "
                f"and labels [{labels.index.min()}, {labels.index.max()}]"
            )
        sensor_df = sensor_df.loc[common_idx]
        labels = labels.loc[common_idx]

    logger.info(
        "Aligned %d rows (range: %s to %s)",
        len(sensor_df), sensor_df.index.min(), sensor_df.index.max(),
    )

    # Channel filtering
    sensor_df = _filter_channels(
        sensor_df, config.nan_threshold, config.channels, config.exclude_channels,
    )
    if len(sensor_df.columns) == 0:
        raise ValueError("No channels remaining after filtering")

    # NaN handling
    sensor_df = _fill_nan(sensor_df)

    # Time features (appended after NaN fill so they are never NaN)
    if config.add_time_features:
        sensor_df = _add_time_features(sensor_df)

    sensor_array = sensor_df.values.astype(np.float32)
    label_array = labels.values.astype(np.int64)
    channel_names = list(sensor_df.columns)
    timestamps = sensor_df.index

    logger.info(
        "Final preprocessed data: %d timesteps, %d channels, "
        "label dist: {0: %d, 1: %d}",
        len(sensor_array), len(channel_names),
        (label_array == 0).sum(), (label_array == 1).sum(),
    )

    return sensor_array, label_array, channel_names, timestamps


# ---------------------------------------------------------------------------
# Public API — new (P4 pipeline)
# ---------------------------------------------------------------------------

def load_sensor_and_labels(
    config: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex, pd.DatetimeIndex]:
    """Load full sensor data and aligned labels separately.

    Unlike load_and_preprocess(), this function returns the FULL sensor
    array (not trimmed to label range), allowing context windows to
    reference sensor data outside the labeled time range.

    Returns
    -------
    sensor_array : np.ndarray, shape (n_sensor_timesteps, n_channels)
        Full sensor data (entire CSV time range).
    label_array : np.ndarray, shape (n_sensor_timesteps,)
        Per-timestep labels aligned to sensor timestamps.
        -1 = unlabeled (outside label range or after last event).
    channel_names : list[str]
    sensor_timestamps : pd.DatetimeIndex
        Full sensor time range.
    labeled_timestamps : pd.DatetimeIndex
        Timestamps where label >= 0 (for dataset construction).
    """
    sensor_df = _load_sensor_data(config.sensor_csv)

    # Generate per-timestep labels aligned to sensor timestamps
    if config.label_format == "events":
        labels = _load_label_events(
            config.label_csv, sensor_df.index,
            initial_occupancy=config.initial_occupancy,
            binarize=config.binarize,
        )
    else:
        raw_labels = _load_label_counts(config.label_csv, binarize=config.binarize)
        # Map counts to sensor timestamps; default -1 for unmatched
        labels = pd.Series(
            np.full(len(sensor_df), -1, dtype=np.int64),
            index=sensor_df.index, name="occupancy",
        )
        common_idx = sensor_df.index.intersection(raw_labels.index)
        labels.loc[common_idx] = raw_labels.loc[common_idx].values

    # Channel filtering (on full sensor data)
    sensor_df = _filter_channels(
        sensor_df, config.nan_threshold, config.channels, config.exclude_channels,
    )
    if len(sensor_df.columns) == 0:
        raise ValueError("No channels remaining after filtering")

    # NaN handling
    sensor_df = _fill_nan(sensor_df)

    # Time features
    if config.add_time_features:
        sensor_df = _add_time_features(sensor_df)

    sensor_array = sensor_df.values.astype(np.float32)
    label_array = labels.values.astype(np.int64)
    channel_names = list(sensor_df.columns)
    sensor_timestamps = sensor_df.index
    labeled_timestamps = sensor_timestamps[label_array >= 0]

    n_labeled = (label_array >= 0).sum()
    n_unlabeled = (label_array == -1).sum()

    logger.info(
        "Full sensor: %d timesteps, %d channels. "
        "Labels: %d labeled, %d unlabeled. "
        "Labeled range: [%s, %s]",
        len(sensor_array), len(channel_names),
        n_labeled, n_unlabeled,
        labeled_timestamps.min() if len(labeled_timestamps) > 0 else "N/A",
        labeled_timestamps.max() if len(labeled_timestamps) > 0 else "N/A",
    )

    return sensor_array, label_array, channel_names, sensor_timestamps, labeled_timestamps
