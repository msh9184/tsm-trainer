"""Data preprocessing for 5-class occupancy event classification.

Loads SmartThings sensor data and the 5-class event CSV
(ENTER_HOME_NEW, ENTER_HOME_ADD, LEAVE_HOME_LAST, LEAVE_HOME_REDUCE, STAY),
then maps to one of 3 label settings via ``apply_label_setting()``.

Label Settings:
  Setting 1 (5-class): All 5 classes independently
  Setting 2 (3-class event): ENTER=NEW+ADD, LEAVE=LAST+REDUCE, STAY
  Setting 3 (3-class occupancy): ENTER=NEW, LEAVE=LAST, STAY=ADD+REDUCE+STAY

Preprocessing pipeline:
  1. Load sensor CSV (auto-detect headerless format)
  2. Load 5-class events CSV
  3. Apply label setting to map to target classes
  4. Filter/select channels, handle NaN
  5. Optionally inject hour-of-day cyclical features
  6. Return (sensor_array, sensor_timestamps, event_timestamps,
     event_labels, channel_names, class_names)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label setting definitions
# ---------------------------------------------------------------------------

# Maps 5-class status strings to integer labels for each setting
LABEL_SETTINGS: dict[int, dict[str, int]] = {
    1: {  # 5-class (all independent)
        "ENTER_HOME_NEW": 0,
        "ENTER_HOME_ADD": 1,
        "LEAVE_HOME_LAST": 2,
        "LEAVE_HOME_REDUCE": 3,
        "STAY": 4,
    },
    2: {  # 3-class event-based (merge by direction)
        "ENTER_HOME_NEW": 0,
        "ENTER_HOME_ADD": 0,
        "LEAVE_HOME_LAST": 1,
        "LEAVE_HOME_REDUCE": 1,
        "STAY": 2,
    },
    3: {  # 3-class occupancy-based (merge by state change)
        "ENTER_HOME_NEW": 0,
        "LEAVE_HOME_LAST": 1,
        "ENTER_HOME_ADD": 2,
        "LEAVE_HOME_REDUCE": 2,
        "STAY": 2,
    },
}

CLASS_NAMES_BY_SETTING: dict[int, list[str]] = {
    1: ["Enter_New", "Enter_Add", "Leave_Last", "Leave_Reduce", "Stay"],
    2: ["Enter", "Leave", "Stay"],
    3: ["Enter", "Leave", "Stay"],
}


@dataclass
class EventPreprocessConfig:
    """Configuration for event detection preprocessing.

    Attributes:
        sensor_csv: Path to merged sensor data CSV (may be headerless).
        events_csv: Path to 5-class event CSV with Status column.
        column_names_csv: Path to a reference CSV whose header row provides
            column names for ``sensor_csv`` when it has no header.
        column_names: Explicit list of column names (overrides column_names_csv).
        channels: Explicit list of channel names to use.
        exclude_channels: Channels to exclude (substring match).
        nan_threshold: Drop channels with NaN fraction above this value.
        label_setting: Which label setting to use (1, 2, or 3). Default: 3.
        add_time_features: If True, append hour_sin/hour_cos channels.
    """

    sensor_csv: str | Path = ""
    events_csv: str | Path = ""
    column_names_csv: str | Path | None = None
    column_names: list[str] | None = None
    channels: list[str] | None = None
    exclude_channels: list[str] = field(default_factory=list)
    nan_threshold: float = 0.3
    label_setting: int = 3
    add_time_features: bool = False


# ---------------------------------------------------------------------------
# Sensor loading (reused from apc_enter_leave)
# ---------------------------------------------------------------------------

def _detect_headerless(path: str | Path) -> bool:
    """Return True if the CSV appears to have no header row."""
    with open(path) as f:
        first_line = f.readline().strip()
    if not first_line:
        return False
    first_field = first_line.split(",")[0].strip().strip('"')
    try:
        pd.Timestamp(first_field)
        return True
    except (ValueError, TypeError):
        return False


def _resolve_column_names(
    path: str | Path,
    column_names: list[str] | None,
    column_names_csv: str | Path | None,
    n_columns: int,
) -> list[str]:
    """Determine column names for a headerless CSV."""
    if column_names is not None:
        if len(column_names) >= n_columns:
            return column_names[:n_columns]
        return column_names + [f"col_{i}" for i in range(len(column_names), n_columns)]

    if column_names_csv is not None:
        ref_path = Path(column_names_csv)
        if ref_path.exists():
            ref_header = pd.read_csv(ref_path, nrows=0).columns.tolist()
            if len(ref_header) >= n_columns:
                return ref_header[:n_columns]
            return ref_header + [f"col_{i}" for i in range(len(ref_header), n_columns)]
        logger.warning("Reference CSV not found: %s", ref_path)

    logger.warning("No column names provided for headerless CSV; using generated names")
    return [f"col_{i}" for i in range(n_columns)]


def _load_sensor_data(
    path: str | Path,
    column_names: list[str] | None = None,
    column_names_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Load sensor CSV with timestamp index (auto-detects headerless)."""
    headerless = _detect_headerless(path)

    if headerless:
        df = pd.read_csv(path, header=None)
        names = _resolve_column_names(
            path, column_names, column_names_csv, len(df.columns),
        )
        df.columns = names

        time_col = None
        for col in df.columns:
            try:
                pd.to_datetime(df[col].head(5))
                time_col = col
                break
            except (ValueError, TypeError):
                continue

        if time_col is None:
            raise ValueError(f"Could not identify timestamp column in headerless CSV: {path}")

        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
        df.index.name = "time"
        logger.info(
            "Loaded headerless sensor CSV: %d rows, %d columns",
            len(df), len(df.columns),
        )
    else:
        df = pd.read_csv(path, parse_dates=["time"], index_col="time")
        df.index.name = "time"

    df = df.sort_index()
    logger.info(
        "Sensor data: %d rows, %d channels, range [%s, %s]",
        len(df), len(df.columns), df.index.min(), df.index.max(),
    )
    return df


# ---------------------------------------------------------------------------
# Event loading (5-class)
# ---------------------------------------------------------------------------

VALID_STATUSES_5CLASS = {
    "ENTER_HOME_NEW", "ENTER_HOME_ADD",
    "LEAVE_HOME_LAST", "LEAVE_HOME_REDUCE", "STAY",
}


def load_events(
    path: str | Path,
    label_setting: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load 5-class event CSV and map to the specified label setting.

    Parameters
    ----------
    path : str or Path
        Path to event CSV with columns: time, Status, Head-count, At-home count.
    label_setting : int
        Label setting (1, 2, or 3). See LABEL_SETTINGS for mapping.

    Returns
    -------
    timestamps : np.ndarray of datetime64
    labels : np.ndarray of int64
    class_names : list[str]
    """
    if label_setting not in LABEL_SETTINGS:
        raise ValueError(
            f"Unknown label_setting={label_setting}. "
            f"Available: {list(LABEL_SETTINGS)}"
        )

    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Normalize status strings
    df["Status"] = df["Status"].str.strip().str.upper()

    # Validate known statuses
    unknown_raw = set(df["Status"].unique()) - VALID_STATUSES_5CLASS
    if unknown_raw:
        logger.warning("Unknown raw statuses found: %s", unknown_raw)

    # Get label mapping for this setting
    label_map = LABEL_SETTINGS[label_setting]
    class_names = CLASS_NAMES_BY_SETTING[label_setting]

    # Map status to label, drop unknown statuses
    df["label"] = df["Status"].map(label_map)
    unknown = df["label"].isna()
    if unknown.any():
        unknown_statuses = df.loc[unknown, "Status"].unique().tolist()
        logger.warning(
            "Dropping %d events with unknown status: %s",
            unknown.sum(), unknown_statuses,
        )
        df = df[~unknown].copy()

    timestamps = df["time"].values.astype("datetime64[ns]")
    labels = df["label"].values.astype(np.int64)

    if len(labels) == 0:
        raise ValueError(f"No valid events after filtering in {path}")

    # Log class distribution
    logger.info("Label setting %d: %d classes", label_setting, len(class_names))
    for i, name in enumerate(class_names):
        count = int((labels == i).sum())
        pct = 100.0 * count / len(labels)
        logger.info("  %s (label=%d): %d events (%.1f%%)", name, i, count, pct)

    # Log 5-class raw distribution for reference
    raw_dist = df["Status"].value_counts()
    logger.info("Raw 5-class distribution:")
    for status, count in raw_dist.items():
        logger.info("  %s: %d", status, count)

    # Check same-timestamp collisions
    ts_series = pd.Series(labels, index=pd.DatetimeIndex(timestamps))
    dup_times = ts_series.index[ts_series.index.duplicated(keep=False)]
    if len(dup_times) > 0:
        unique_dup = dup_times.unique()
        logger.warning(
            "Same-timestamp collisions at %d unique times (%d events total).",
            len(unique_dup), len(dup_times),
        )

    logger.info(
        "Loaded %d events from %s, range [%s, %s]",
        len(labels), path, timestamps[0], timestamps[-1],
    )
    return timestamps, labels, class_names


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
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = sorted(set(df.columns) - set(numeric_cols))
    if non_numeric:
        logger.info("Dropped %d non-numeric channels: %s", len(non_numeric), non_numeric)
        df = df[numeric_cols]

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
    logger.info("NaN fill: %d -> 0 (filled %d values)", n_nan_before, n_nan_before)
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append hour_sin and hour_cos cyclical features."""
    hour_frac = df.index.hour + df.index.minute / 60.0
    hour_rad = 2.0 * np.pi * hour_frac / 24.0
    df = df.copy()
    df["hour_sin"] = np.sin(hour_rad).astype(np.float32)
    df["hour_cos"] = np.cos(hour_rad).astype(np.float32)
    logger.info("Added time features: hour_sin, hour_cos")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sensor_and_events(
    config: EventPreprocessConfig,
) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load full sensor data and event list with label setting applied.

    Returns
    -------
    sensor_array : np.ndarray, shape (n_timesteps, n_channels)
    sensor_timestamps : pd.DatetimeIndex
    event_timestamps : np.ndarray of datetime64
    event_labels : np.ndarray of int64
    channel_names : list[str]
    class_names : list[str]
    """
    # Load sensor data
    sensor_df = _load_sensor_data(
        config.sensor_csv,
        column_names=config.column_names,
        column_names_csv=config.column_names_csv,
    )

    # Load events with label setting
    event_timestamps, event_labels, class_names = load_events(
        config.events_csv, label_setting=config.label_setting,
    )

    # Channel filtering
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
    sensor_timestamps = sensor_df.index
    channel_names = list(sensor_df.columns)

    # Verify events fall within sensor range
    sensor_start = sensor_timestamps[0]
    sensor_end = sensor_timestamps[-1]
    event_ts_pd = pd.DatetimeIndex(event_timestamps)
    in_range = (event_ts_pd >= sensor_start) & (event_ts_pd <= sensor_end)
    n_out = (~in_range).sum()
    if n_out > 0:
        logger.warning(
            "%d events fall outside sensor range [%s, %s]",
            n_out, sensor_start, sensor_end,
        )

    logger.info(
        "Loaded: %d sensor timesteps (%d channels), %d events (%s)",
        len(sensor_array), len(channel_names), len(event_labels),
        ", ".join(f"{name}={int((event_labels == i).sum())}" for i, name in enumerate(class_names)),
    )

    return sensor_array, sensor_timestamps, event_timestamps, event_labels, channel_names, class_names
