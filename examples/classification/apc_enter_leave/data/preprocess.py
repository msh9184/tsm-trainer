"""Data preprocessing for enter/leave event detection.

Loads SmartThings sensor data (including headerless CSVs) and event logs.
Unlike the occupancy pipeline which generates per-timestep labels, this module
loads individual ENTER_HOME / LEAVE_HOME events as discrete classification
targets with associated timestamps.

Preprocessing pipeline:
  1. Load sensor CSV (auto-detect headerless format)
  2. Load events CSV (ENTER_HOME / LEAVE_HOME / NONE)
  3. Filter/select channels, handle NaN
  4. Optionally inject hour-of-day cyclical features
  5. Return (sensor_array, sensor_timestamps, event_timestamps,
     event_labels, channel_names, class_names)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EventPreprocessConfig:
    """Configuration for event detection preprocessing.

    Attributes:
        sensor_csv: Path to merged sensor data CSV (may be headerless).
        events_csv: Path to event CSV with Status column.
        column_names_csv: Path to a reference CSV whose header row provides
            column names for ``sensor_csv`` when it has no header.
        column_names: Explicit list of column names (overrides column_names_csv).
        channels: Explicit list of channel names to use.
        exclude_channels: Channels to exclude (substring match).
        nan_threshold: Drop channels with NaN fraction above this value.
        include_none: If True, include NONE events as a third class.
        add_time_features: If True, append hour_sin/hour_cos channels.
    """

    sensor_csv: str | Path = ""
    events_csv: str | Path = ""
    column_names_csv: str | Path | None = None
    column_names: list[str] | None = None
    channels: list[str] | None = None
    exclude_channels: list[str] = field(default_factory=list)
    nan_threshold: float = 0.3
    include_none: bool = False
    add_time_features: bool = False


# ---------------------------------------------------------------------------
# Sensor loading — with headerless CSV support
# ---------------------------------------------------------------------------

def _detect_headerless(path: str | Path) -> bool:
    """Return True if the CSV appears to have no header row.

    Heuristic: if the first field of the first line parses as a datetime,
    the file is headerless (data starts immediately).
    """
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
    """Determine column names for a headerless CSV.

    Priority: explicit ``column_names`` > ``column_names_csv`` header >
    generated fallback names (col_0, col_1, ...).
    """
    if column_names is not None:
        if len(column_names) >= n_columns:
            return column_names[:n_columns]
        logger.warning(
            "Provided %d column names but CSV has %d columns; "
            "padding with generated names",
            len(column_names), n_columns,
        )
        return column_names + [f"col_{i}" for i in range(len(column_names), n_columns)]

    if column_names_csv is not None:
        ref_path = Path(column_names_csv)
        if ref_path.exists():
            ref_header = pd.read_csv(ref_path, nrows=0).columns.tolist()
            if len(ref_header) >= n_columns:
                return ref_header[:n_columns]
            logger.warning(
                "Reference CSV has %d columns but sensor CSV has %d; "
                "padding with generated names",
                len(ref_header), n_columns,
            )
            return ref_header + [f"col_{i}" for i in range(len(ref_header), n_columns)]
        logger.warning("Reference CSV not found: %s", ref_path)

    logger.warning(
        "No column names provided for headerless CSV %s; "
        "using generated names (col_0, col_1, ...)",
        path,
    )
    return [f"col_{i}" for i in range(n_columns)]


def _load_sensor_data(
    path: str | Path,
    column_names: list[str] | None = None,
    column_names_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Load sensor CSV with timestamp index.

    Auto-detects headerless CSVs: if the first field is a datetime value,
    the file is treated as headerless and column names are resolved from
    ``column_names`` or ``column_names_csv``.
    """
    headerless = _detect_headerless(path)

    if headerless:
        df = pd.read_csv(path, header=None)
        names = _resolve_column_names(
            path, column_names, column_names_csv, len(df.columns),
        )
        df.columns = names

        # Find the timestamp column (first column parseable as datetime)
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
            "Loaded headerless sensor CSV: %d rows, %d columns, "
            "timestamp column: '%s'",
            len(df), len(df.columns), time_col,
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
# Event loading
# ---------------------------------------------------------------------------

def load_events(
    path: str | Path,
    include_none: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load event CSV and return timestamps, integer labels, and class names.

    Label encoding:
      - ENTER_HOME → 0
      - LEAVE_HOME → 1
      - NONE → 2 (only if ``include_none=True``)

    Parameters
    ----------
    path : str or Path
        Path to event CSV with columns: time, Status, [Head-count, At-home count].
    include_none : bool
        If True, keep NONE events as a third class.

    Returns
    -------
    timestamps : np.ndarray of datetime64
        Event timestamps.
    labels : np.ndarray of int64
        Integer class labels.
    class_names : list[str]
        Human-readable class names indexed by label value.
    """
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Normalize status strings
    df["Status"] = df["Status"].str.strip().str.upper()

    if not include_none:
        df = df[df["Status"] != "NONE"].copy()

    # Encode labels
    label_map = {"ENTER_HOME": 0, "LEAVE_HOME": 1}
    class_names = ["Enter", "Leave"]
    if include_none:
        label_map["NONE"] = 2
        class_names.append("None")

    # Map status to label, drop unknown statuses
    df["label"] = df["Status"].map(label_map)
    unknown = df["label"].isna()
    if unknown.any():
        unknown_statuses = df.loc[unknown, "Status"].unique().tolist()
        logger.warning("Dropping %d events with unknown status: %s", unknown.sum(), unknown_statuses)
        df = df[~unknown].copy()

    timestamps = df["time"].values.astype("datetime64[ns]")
    labels = df["label"].values.astype(np.int64)

    if len(labels) == 0:
        raise ValueError(f"No valid events after filtering in {path}")

    # Log class distribution
    for i, name in enumerate(class_names):
        count = (labels == i).sum()
        pct = 100.0 * count / len(labels)
        logger.info("  %s (label=%d): %d events (%.1f%%)", name, i, count, pct)

    # Log same-timestamp collisions
    ts_series = pd.Series(labels, index=pd.DatetimeIndex(timestamps))
    dup_times = ts_series.index[ts_series.index.duplicated(keep=False)]
    if len(dup_times) > 0:
        unique_dup = dup_times.unique()
        logger.warning(
            "Same-timestamp collisions at %d unique times (%d events total). "
            "These events share identical context windows but may have different labels.",
            len(unique_dup), len(dup_times),
        )
        for t in unique_dup:
            collision_labels = ts_series.loc[[t]]  # always returns Series
            label_strs = [class_names[l] for l in collision_labels.values]
            logger.warning("  %s: %s", t, label_strs)

    logger.info(
        "Loaded %d events from %s, range [%s, %s]",
        len(labels), path, timestamps[0], timestamps[-1],
    )
    return timestamps, labels, class_names


# ---------------------------------------------------------------------------
# Channel filtering and NaN handling (reused from occupancy pipeline)
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
    n_nan_after = df.isna().sum().sum()
    logger.info(
        "NaN fill: %d -> %d (filled %d values)",
        n_nan_before, n_nan_after, n_nan_before - n_nan_after,
    )
    return df


# ---------------------------------------------------------------------------
# Time feature injection (reused from occupancy pipeline)
# ---------------------------------------------------------------------------

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append hour_sin and hour_cos cyclical features as extra columns."""
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
    """Load full sensor data and event list.

    Returns
    -------
    sensor_array : np.ndarray, shape (n_timesteps, n_channels)
        Full preprocessed sensor data.
    sensor_timestamps : pd.DatetimeIndex
        Timestamps for each row of sensor_array.
    event_timestamps : np.ndarray of datetime64
        Timestamps of each event.
    event_labels : np.ndarray of int64
        Event class labels (0=Enter, 1=Leave, [2=None]).
    channel_names : list[str]
        Sensor channel names.
    class_names : list[str]
        Human-readable class names indexed by label.
    """
    # Load sensor data
    sensor_df = _load_sensor_data(
        config.sensor_csv,
        column_names=config.column_names,
        column_names_csv=config.column_names_csv,
    )

    # Load events
    event_timestamps, event_labels, class_names = load_events(
        config.events_csv, include_none=config.include_none,
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
