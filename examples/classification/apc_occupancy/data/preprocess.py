"""Data preprocessing for SmartThings sensor data.

Loads merged_processed_data.csv (5-min binned sensor readings) and
occupancy_counts.csv (5-min binned occupancy counts), aligns them by
timestamp, applies NaN handling, and produces numpy arrays suitable
for MantisV2 consumption.

The preprocessing pipeline:
  1. Load sensor CSV and occupancy CSV
  2. Inner-join on timestamp (only overlapping rows)
  3. Drop channels exceeding NaN threshold
  4. Forward-fill -> backward-fill -> zero-fill remaining NaN
  5. Binarize occupancy labels (count > 0 -> 1)
  6. Return (sensor_array, label_array, channel_names, timestamps)
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
        sensor_csv: Path to merged_processed_data.csv
        label_csv: Path to occupancy_counts.csv
        nan_threshold: Drop channels with NaN fraction above this value.
            E.g. 0.5 means channels with >50% NaN are excluded.
        channels: Explicit list of channel names to use. If None, all
            channels passing nan_threshold are used.
        exclude_channels: Channels to always exclude (substring match on
            channel names).
        binarize: If True, convert occupancy count to binary (>0 -> 1).
    """
    sensor_csv: str | Path = ""
    label_csv: str | Path = ""
    nan_threshold: float = 0.5
    channels: list[str] | None = None
    exclude_channels: list[str] = field(default_factory=list)
    binarize: bool = True


def _load_sensor_data(path: str | Path) -> pd.DataFrame:
    """Load sensor CSV with timestamp index."""
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index.name = "time"
    # Ensure sorted by time
    df = df.sort_index()
    logger.info(
        "Loaded sensor data: %d rows, %d channels, range [%s, %s]",
        len(df), len(df.columns), df.index.min(), df.index.max(),
    )
    return df


def _load_label_data(path: str | Path, binarize: bool = True) -> pd.Series:
    """Load occupancy counts CSV, optionally binarize."""
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index.name = "time"
    df = df.sort_index()

    occupancy = df["occupancy"]
    if binarize:
        occupancy = (occupancy > 0).astype(np.int64)
        logger.info(
            "Loaded labels (binarized): %d rows, class dist: {0: %d, 1: %d}",
            len(occupancy),
            (occupancy == 0).sum(),
            (occupancy == 1).sum(),
        )
    else:
        logger.info(
            "Loaded labels (raw counts): %d rows, range [%d, %d]",
            len(occupancy), occupancy.min(), occupancy.max(),
        )
    return occupancy


def _filter_channels(
    df: pd.DataFrame,
    nan_threshold: float,
    channels: list[str] | None,
    exclude_channels: list[str],
) -> pd.DataFrame:
    """Filter sensor channels by NaN fraction and explicit lists."""
    # Apply explicit exclusion first
    if exclude_channels:
        cols_before = set(df.columns)
        for pattern in exclude_channels:
            matched = [c for c in df.columns if pattern in c]
            df = df.drop(columns=matched, errors="ignore")
        dropped = cols_before - set(df.columns)
        if dropped:
            logger.info("Excluded channels: %s", sorted(dropped))

    # If explicit channel list is given, use only those
    if channels is not None:
        available = [c for c in channels if c in df.columns]
        missing = [c for c in channels if c not in df.columns]
        if missing:
            logger.warning("Requested channels not found: %s", missing)
        df = df[available]
        logger.info("Using %d explicitly specified channels", len(available))
        return df

    # Auto-filter by NaN threshold
    nan_fracs = df.isna().mean()
    keep = nan_fracs[nan_fracs <= nan_threshold].index.tolist()
    dropped = nan_fracs[nan_fracs > nan_threshold].index.tolist()

    if dropped:
        logger.info(
            "Dropped %d channels exceeding %.0f%% NaN threshold: %s",
            len(dropped), nan_threshold * 100, dropped,
        )

    df = df[keep]
    logger.info(
        "Retained %d channels (NaN threshold=%.2f)",
        len(keep), nan_threshold,
    )
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


def load_and_preprocess(
    config: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Load and preprocess sensor + label data.

    Parameters
    ----------
    config : PreprocessConfig
        Preprocessing configuration.

    Returns
    -------
    sensor_array : np.ndarray
        Shape (n_timesteps, n_channels). Float32 sensor readings.
    label_array : np.ndarray
        Shape (n_timesteps,). Int64 labels (binary if binarize=True).
    channel_names : list[str]
        Ordered list of channel names.
    timestamps : pd.DatetimeIndex
        Aligned timestamps.
    """
    sensor_df = _load_sensor_data(config.sensor_csv)
    labels = _load_label_data(config.label_csv, binarize=config.binarize)

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
        "Aligned %d rows (sensor range: %s-%s, label range: %s-%s)",
        len(common_idx),
        sensor_df.index.min(), sensor_df.index.max(),
        labels.index.min(), labels.index.max(),
    )

    # Channel filtering
    sensor_df = _filter_channels(
        sensor_df, config.nan_threshold, config.channels, config.exclude_channels,
    )

    if len(sensor_df.columns) == 0:
        raise ValueError("No channels remaining after filtering")

    # NaN handling
    sensor_df = _fill_nan(sensor_df)

    # Convert to numpy
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
