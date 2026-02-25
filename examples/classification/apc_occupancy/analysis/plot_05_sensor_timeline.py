"""05. Sensor Timeline — Time series plots with occupancy label overlay.

Generates:
  - Multi-panel sensor time series with occupancy background shading
  - Core sensor dashboard (8 key sensors in one figure)
  - Event markers on sensor timeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .config import (
    AnalysisConfig, CLASS_COLORS, CLASS_NAMES, CORE_SENSORS,
    PERIOD_COLORS, SENSOR_CATEGORIES, DEVICE_MAP,
)
from .data_loader import PeriodData

logger = logging.getLogger(__name__)


def _get_device_label(col_name: str) -> str:
    """Convert column name to human-readable label with device info."""
    parts = col_name.split("_", 1)
    if len(parts) == 2:
        device_id, capability = parts
        info = DEVICE_MAP.get(device_id, {})
        device_name = info.get("name", device_id)
        location = info.get("location", "")
        if location:
            return f"{capability}\n({device_name}, {location})"
        return f"{capability}\n({device_name})"
    return col_name


def _add_occupancy_background(
    ax: plt.Axes,
    labels: pd.Series,
    y_min: float,
    y_max: float,
) -> None:
    """Add occupancy state background shading to an axis."""
    for cls, color in CLASS_COLORS.items():
        mask = labels == cls
        if mask.any():
            ax.fill_between(
                labels.index, y_min, y_max,
                where=mask.values,
                alpha=0.08, color=color,
                step="post",
            )


def plot_core_sensor_dashboard(
    period: PeriodData,
    config: AnalysisConfig,
) -> plt.Figure:
    """Dashboard of 8 core sensors for a single period with occupancy overlay.

    Each sensor gets its own subplot with:
      - Sensor value time series
      - Occupancy state background shading
      - Event markers (ENTER/LEAVE)
    """
    available_core = [c for c in CORE_SENSORS if c in period.sensor_df.columns]
    n_sensors = len(available_core)

    if n_sensors == 0:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, f"No core sensors available in {period.name}",
                ha="center", va="center")
        return fig

    fig, axes = plt.subplots(
        n_sensors, 1, figsize=(14, 2.2 * n_sensors),
        sharex=True,
    )
    if n_sensors == 1:
        axes = [axes]

    mask = period.label_series != -1
    labels = period.label_series[mask]

    for i, (ax, col) in enumerate(zip(axes, available_core)):
        series = period.sensor_df[col]

        # Only plot labeled region
        series_labeled = series.loc[mask]

        if len(series_labeled) == 0:
            ax.text(0.5, 0.5, "No labeled data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        y_min = series_labeled.min()
        y_max = series_labeled.max()
        if np.isnan(y_min) or np.isnan(y_max) or y_min == y_max:
            y_min, y_max = 0, 1

        # Occupancy background
        _add_occupancy_background(ax, labels, y_min - (y_max - y_min) * 0.1,
                                  y_max + (y_max - y_min) * 0.1)

        # Sensor line
        ax.plot(
            series_labeled.index, series_labeled.values,
            color="#333333", linewidth=0.8, alpha=0.9,
        )

        # NaN regions (if any)
        nan_mask = series_labeled.isna()
        if nan_mask.any():
            nan_frac = nan_mask.mean()
            ax.text(
                0.99, 0.95, f"NaN: {nan_frac:.1%}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="#CC3311",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

        ax.set_ylabel(_get_device_label(col), fontsize=7, labelpad=5)
        ax.tick_params(axis="y", labelsize=7)

        # Add event markers on first subplot
        if i == 0:
            for _, event in period.events_df.iterrows():
                t = pd.Timestamp(event["time"])
                if t >= series_labeled.index.min() and t <= series_labeled.index.max():
                    color = "#CC3311" if event["Status"] == "LEAVE_HOME" else "#009E73"
                    marker = "v" if event["Status"] == "LEAVE_HOME" else "^"
                    if event["Status"] != "NONE":
                        ax.axvline(t, color=color, linewidth=0.3, alpha=0.5)

    # Format x-axis on bottom
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, fontsize=7)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=CLASS_COLORS[0], alpha=0.15, label="Empty"),
        mpatches.Patch(facecolor=CLASS_COLORS[1], alpha=0.15, label="Occupied"),
        plt.Line2D([0], [0], color="#009E73", linewidth=0.5, label="ENTER event"),
        plt.Line2D([0], [0], color="#CC3311", linewidth=0.5, label="LEAVE event"),
    ]
    axes[0].legend(
        handles=legend_elements, loc="upper right",
        fontsize=7, ncol=4, framealpha=0.9,
    )

    fig.suptitle(
        f"Core Sensor Dashboard — {period.name} ({period.date_range})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    return fig


def plot_all_sensors_overview(
    period: PeriodData,
    config: AnalysisConfig,
) -> plt.Figure:
    """Overview of ALL sensor columns for a single period.

    Compact multi-panel plot with normalized sensor values
    and occupancy overlay. Good for spotting anomalies.
    """
    cols = list(period.sensor_df.columns)
    n_cols = len(cols)

    if n_cols == 0:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No sensor data", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(
        n_cols, 1, figsize=(14, max(6, n_cols * 1.2)),
        sharex=True,
    )
    if n_cols == 1:
        axes = [axes]

    mask = period.label_series != -1
    labels = period.label_series[mask]

    for ax, col in zip(axes, cols):
        series = period.sensor_df[col].loc[mask]

        # Normalize to [0, 1] for consistent display
        s_min = series.min()
        s_max = series.max()
        if pd.notna(s_min) and pd.notna(s_max) and s_min != s_max:
            normalized = (series - s_min) / (s_max - s_min)
        else:
            normalized = series * 0  # Flat or all NaN

        # Occupancy background
        _add_occupancy_background(ax, labels, -0.1, 1.1)

        # Sensor line
        ax.plot(
            normalized.index, normalized.values,
            color=PERIOD_COLORS[period.name], linewidth=0.6, alpha=0.8,
        )

        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["", "", ""], fontsize=5)

        # Short label
        short_label = col.split("_", 1)[1] if "_" in col else col
        is_core = col in CORE_SENSORS
        weight = "bold" if is_core else "normal"
        ax.set_ylabel(short_label, fontsize=6, fontweight=weight, rotation=0,
                      labelpad=60, ha="right")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, fontsize=7)

    fig.suptitle(
        f"All Sensors (Normalized) — {period.name} ({period.date_range})\n"
        "(Bold = core sensor, Background: blue=empty, orange=occupied)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    return fig


def generate_sensor_timeline(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> list[Path]:
    """Generate sensor timeline plots for all periods."""
    output_dir = config.output_dir / "05_sensor_timeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for p in periods:
        # Core sensor dashboard
        fig = plot_core_sensor_dashboard(p, config)
        path = output_dir / f"core_dashboard_{p.name}.png"
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        logger.info("Saved: %s", path)

        # All sensors overview
        fig = plot_all_sensors_overview(p, config)
        path = output_dir / f"all_sensors_{p.name}.png"
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        logger.info("Saved: %s", path)

    return saved
