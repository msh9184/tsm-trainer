"""Unified data loader for all 4 sensor + label period files.

Loads raw CSV files, converts event-based occupancy labels to
per-timestep binary labels, and handles cross-period schema differences.

Key operations:
  1. Load sensor CSV → pd.DataFrame (time-indexed, numeric columns only)
  2. Load occupancy event CSV → reconstruct At-home count per 5-min bin
  3. Inner-join sensor + labels by timestamp → aligned arrays
  4. Aggregate all periods into a single multi-period dataset
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PERIODS, AnalysisConfig, PeriodDef

logger = logging.getLogger(__name__)


@dataclass
class PeriodData:
    """Loaded and aligned data for a single period."""
    name: str
    sensor_df: pd.DataFrame        # (n_timesteps, n_channels) time-indexed
    label_series: pd.Series         # (n_timesteps,) binary labels, time-indexed
    occupancy_counts: pd.Series     # (n_timesteps,) raw At-home counts
    events_df: pd.DataFrame         # Raw occupancy events
    sensor_raw_df: pd.DataFrame     # Raw sensor DataFrame before filtering
    date_range: str


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to numeric, coercing errors to NaN.

    Non-numeric string values (e.g. 'critical', 'off', 'unlocked')
    are replaced with NaN. This is necessary because some SmartThings
    sensors report string states mixed with numeric values.
    """
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col == "time":
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        out[col] = converted
    return out


def load_sensor_csv(path: Path) -> pd.DataFrame:
    """Load a merged sensor CSV with time index.

    Returns
    -------
    pd.DataFrame
        Time-indexed sensor readings. All columns numeric (NaN for
        non-numeric entries). Sorted by time.
    """
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.set_index("time").sort_index()

    # Drop any completely empty columns
    df = df.dropna(axis=1, how="all")

    # Store raw column set before numeric coercion
    raw_df = df.copy()

    # Coerce to numeric
    df = _coerce_numeric(df)

    n_non_numeric = 0
    for col in raw_df.columns:
        if col not in df.columns:
            continue
        mask = raw_df[col].notna() & df[col].isna()
        n_non_numeric += mask.sum()

    if n_non_numeric > 0:
        logger.info(
            "%s: coerced %d non-numeric values to NaN",
            path.name, n_non_numeric,
        )

    logger.info(
        "Loaded sensor: %s (%d rows, %d cols, %s ~ %s)",
        path.name, len(df), len(df.columns),
        df.index.min(), df.index.max(),
    )
    return df, raw_df


def load_events_csv(path: Path) -> pd.DataFrame:
    """Load occupancy events CSV.

    Expected columns: time, Status, Head-count, At-home count

    Returns
    -------
    pd.DataFrame
        Parsed events with DatetimeIndex.
    """
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    logger.info(
        "Loaded events: %s (%d events, %s ~ %s)",
        path.name, len(df),
        df["time"].min(), df["time"].max(),
    )
    return df


def events_to_per_timestep_labels(
    events_df: pd.DataFrame,
    time_index: pd.DatetimeIndex,
    initial_occupancy: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """Convert event-based occupancy to per-timestep labels.

    For each 5-min timestep in ``time_index``, determine the At-home count
    by replaying ENTER_HOME / LEAVE_HOME events chronologically.

    NONE events (Head-count=0, no state change) are skipped.

    Parameters
    ----------
    events_df : pd.DataFrame
        Must have columns: time, Status, Head-count, At-home count
    time_index : pd.DatetimeIndex
        Target timestamps (5-min bins from sensor data).
    initial_occupancy : int
        Initial At-home count before the first event. Inferred from
        the first event's context.

    Returns
    -------
    binary_labels : pd.Series
        Binary (0=empty, 1=occupied) per timestep.
    occupancy_counts : pd.Series
        Raw At-home count per timestep.
    """
    # Build occupancy state timeline from events
    events = events_df.copy()
    events["time"] = pd.to_datetime(events["time"])

    # Filter out NONE events (sensor noise)
    events = events[events["Status"] != "NONE"].copy()

    # Use the At-home count column directly — it's the ground truth
    # state AFTER each event
    event_times = events["time"].values
    at_home_after = events["At-home count"].values.astype(int)

    # Create occupancy count for each sensor timestep
    occupancy = np.full(len(time_index), -1, dtype=int)  # -1 = unlabeled

    for i, ts in enumerate(time_index):
        # Find the latest event at or before this timestep
        mask = event_times <= ts
        if mask.any():
            last_event_idx = np.where(mask)[0][-1]
            occupancy[i] = at_home_after[last_event_idx]
        else:
            # Before first event: use initial occupancy estimate
            occupancy[i] = initial_occupancy

    occupancy_counts = pd.Series(occupancy, index=time_index, name="occupancy_count")
    binary_labels = pd.Series(
        (occupancy > 0).astype(np.int64), index=time_index, name="label"
    )

    # Mark truly unlabeled timesteps (after last event in label file
    # but before sensor data ends) — keep as -1 in occupancy
    if len(event_times) > 0:
        last_event_time = pd.Timestamp(event_times[-1])
        unlabeled_mask = time_index > last_event_time
        n_unlabeled = unlabeled_mask.sum()
        if n_unlabeled > 0:
            binary_labels.loc[unlabeled_mask] = -1
            occupancy_counts.loc[unlabeled_mask] = -1
            logger.info(
                "%d timesteps after last event (%s) marked unlabeled",
                n_unlabeled, last_event_time,
            )

    n_occupied = (binary_labels == 1).sum()
    n_empty = (binary_labels == 0).sum()
    n_unlabeled_total = (binary_labels == -1).sum()
    logger.info(
        "Labels: occupied=%d, empty=%d, unlabeled=%d (total=%d)",
        n_occupied, n_empty, n_unlabeled_total, len(binary_labels),
    )
    return binary_labels, occupancy_counts


def load_period(
    period: PeriodDef,
    data_root: Path,
) -> PeriodData:
    """Load and align sensor + label data for a single period.

    Parameters
    ----------
    period : PeriodDef
        Period definition from config.
    data_root : Path
        Root directory containing all CSV files.

    Returns
    -------
    PeriodData
        Aligned sensor + label data.
    """
    sensor_path = data_root / period.sensor_file
    label_path = data_root / period.label_file

    if not sensor_path.exists():
        raise FileNotFoundError(f"Sensor file not found: {sensor_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    sensor_df, sensor_raw_df = load_sensor_csv(sensor_path)
    events_df = load_events_csv(label_path)

    # Generate per-timestep labels
    binary_labels, occupancy_counts = events_to_per_timestep_labels(
        events_df, sensor_df.index, initial_occupancy=period.initial_occupancy,
    )

    return PeriodData(
        name=period.name,
        sensor_df=sensor_df,
        label_series=binary_labels,
        occupancy_counts=occupancy_counts,
        events_df=events_df,
        sensor_raw_df=sensor_raw_df,
        date_range=period.date_range,
    )


def load_all_periods(config: AnalysisConfig) -> list[PeriodData]:
    """Load all 4 periods.

    Parameters
    ----------
    config : AnalysisConfig
        Must have ``data_root`` pointing to directory with CSV files.

    Returns
    -------
    list[PeriodData]
        One entry per period (P1-P4).
    """
    data_list = []
    for period in PERIODS:
        try:
            pdata = load_period(period, config.data_root)
            data_list.append(pdata)
            logger.info(
                "Loaded %s: %d sensor timesteps, %d events",
                period.name, len(pdata.sensor_df), len(pdata.events_df),
            )
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", period.name, e)
    return data_list


def get_labeled_only(pdata: PeriodData) -> tuple[pd.DataFrame, pd.Series]:
    """Return sensor and binary labels for labeled timesteps only.

    Filters out timesteps where label == -1 (unlabeled).
    """
    mask = pdata.label_series != -1
    return pdata.sensor_df.loc[mask], pdata.label_series.loc[mask]
