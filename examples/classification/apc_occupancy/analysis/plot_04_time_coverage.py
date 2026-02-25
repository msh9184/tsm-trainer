"""04. Time Coverage Analysis — Gantt chart and overlap visualization.

Generates:
  - Gantt-style chart: sensor vs label time ranges per period
  - Inter-period gap analysis (gaps between consecutive periods)
  - Sensor-label overlap percentage matrix
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .config import AnalysisConfig, PERIOD_COLORS
from .data_loader import PeriodData

logger = logging.getLogger(__name__)


def plot_gantt_chart(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Gantt-style chart showing sensor and label time ranges.

    For each period, draws two horizontal bars:
      - Sensor data range (filled)
      - Label events range (hatched)
    Uses broken_barh with date2num for proper matplotlib date handling.
    """
    fig, ax = plt.subplots(figsize=(16, 5))

    y_positions = []
    y_labels = []
    y_counter = 0

    for p in periods:
        # Sensor range
        s_start = p.sensor_df.index.min()
        s_end = p.sensor_df.index.max()
        s_start_num = mdates.date2num(s_start)
        s_duration = mdates.date2num(s_end) - s_start_num

        # Label range
        if len(p.events_df) > 0:
            l_start = pd.Timestamp(p.events_df["time"].min())
            l_end = pd.Timestamp(p.events_df["time"].max())
        else:
            l_start = l_end = s_start
        l_start_num = mdates.date2num(l_start)
        l_duration = mdates.date2num(l_end) - l_start_num

        color = PERIOD_COLORS[p.name]

        # Sensor bar (using broken_barh: dates in date2num units)
        ax.broken_barh(
            [(s_start_num, s_duration)],
            (y_counter - 0.17, 0.35),
            facecolors=color, alpha=0.7,
            edgecolors="black", linewidths=0.5,
        )
        y_labels.append(f"{p.name} Sensor")
        y_positions.append(y_counter)

        # Label bar
        y_counter += 0.5
        ax.broken_barh(
            [(l_start_num, max(l_duration, 0.01))],
            (y_counter - 0.17, 0.35),
            facecolors=color, alpha=0.4,
            edgecolors="black", linewidths=0.5, hatch="///",
        )
        y_labels.append(f"{p.name} Labels")
        y_positions.append(y_counter)

        # Annotate time ranges (right side)
        text_x = s_start_num + s_duration + 0.3
        ax.text(
            text_x, y_counter - 0.25,
            f"Sensor: {s_start.strftime('%m/%d %H:%M')} - {s_end.strftime('%m/%d %H:%M')}",
            fontsize=6.5, va="center", color="#333333",
        )
        ax.text(
            text_x, y_counter + 0.0,
            f"Label: {l_start.strftime('%m/%d %H:%M')} - {l_end.strftime('%m/%d %H:%M')}",
            fontsize=6.5, va="center", color="#666666",
        )

        y_counter += 1.0

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=8)
    ax.set_xlabel("Date (2026)")
    ax.set_title(
        "Sensor vs Label Time Coverage (Gantt Chart)\n"
        "(Solid = sensor data, Hatched = label events)",
        fontsize=12, fontweight="bold",
    )
    ax.invert_yaxis()

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor="#888888", alpha=0.7, label="Sensor Data"),
        mpatches.Patch(facecolor="#888888", alpha=0.4, hatch="///", label="Label Events"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    return fig


def plot_inter_period_gaps(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Visualize gaps between consecutive data collection periods.

    Shows a timeline with periods as blocks and gaps as annotated arrows.
    """
    if len(periods) < 2:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "Need at least 2 periods for gap analysis",
                ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(16, 3))

    for i, p in enumerate(periods):
        s_start = p.sensor_df.index.min()
        s_end = p.sensor_df.index.max()
        s_start_num = mdates.date2num(s_start)
        s_duration = mdates.date2num(s_end) - s_start_num

        # Draw period block
        ax.broken_barh(
            [(s_start_num, s_duration)],
            (0.2, 0.6),
            facecolors=PERIOD_COLORS[p.name],
            edgecolors="black", linewidths=0.8,
        )
        duration_h = (s_end - s_start).total_seconds() / 3600
        ax.text(
            s_start_num + s_duration / 2, 0.5,
            f"{p.name}\n{duration_h:.0f}h",
            ha="center", va="center", fontsize=9, fontweight="bold", color="white",
        )

        # Draw gap arrow to next period
        if i < len(periods) - 1:
            next_start = periods[i + 1].sensor_df.index.min()
            next_start_num = mdates.date2num(next_start)
            s_end_num = mdates.date2num(s_end)
            gap_h = (next_start - s_end).total_seconds() / 3600
            gap_mid = (s_end_num + next_start_num) / 2

            ax.annotate(
                "", xy=(next_start_num, 0.5),
                xytext=(s_end_num, 0.5),
                arrowprops=dict(arrowstyle="<->", color="#CC3311", lw=1.5),
            )
            ax.text(
                gap_mid, 0.9,
                f"Gap: {gap_h:.1f}h\n({gap_h / 24:.1f}d)",
                ha="center", va="bottom", fontsize=8, color="#CC3311",
                fontweight="bold",
            )

    ax.set_ylim(0, 1.3)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=8)
    ax.set_xlabel("Date (2026)")
    ax.set_title(
        "Inter-Period Gaps Between Data Collection Windows",
        fontsize=12, fontweight="bold",
    )
    ax.set_yticks([])

    return fig


def plot_overlap_matrix(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Matrix showing sensor-label overlap and non-overlap statistics.

    For each period:
      - Total sensor timesteps
      - Labeled timesteps (within label event range)
      - Unlabeled timesteps (after last label event)
      - Overlap percentage
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    rows = []
    for p in periods:
        s_start = p.sensor_df.index.min()
        s_end = p.sensor_df.index.max()
        total = len(p.sensor_df)

        mask_labeled = p.label_series != -1
        n_labeled = mask_labeled.sum()
        n_unlabeled = (~mask_labeled).sum()

        # Events time range
        if len(p.events_df) > 0:
            l_start = pd.Timestamp(p.events_df["time"].min())
            l_end = pd.Timestamp(p.events_df["time"].max())
            # Overlap: sensor timesteps within label range
            overlap_mask = (p.sensor_df.index >= l_start) & (p.sensor_df.index <= l_end)
            n_overlap = overlap_mask.sum()
        else:
            l_start = l_end = s_start
            n_overlap = 0

        overlap_pct = n_overlap / max(total, 1) * 100

        rows.append([
            p.name,
            f"{s_start.strftime('%m/%d %H:%M')} - {s_end.strftime('%m/%d %H:%M')}",
            f"{l_start.strftime('%m/%d %H:%M')} - {l_end.strftime('%m/%d %H:%M')}",
            f"{total:,}",
            f"{n_labeled:,}",
            f"{n_unlabeled:,}",
            f"{n_overlap:,}",
            f"{overlap_pct:.1f}%",
        ])

    col_labels = [
        "Period", "Sensor Range", "Label Range",
        "Total Steps", "Labeled", "Unlabeled",
        "Overlap Steps", "Overlap %",
    ]

    table = ax.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)

    # Color-code overlap percentage (last column)
    last_col = len(col_labels) - 1
    for i, row in enumerate(rows, 1):
        pct = float(row[-1].replace("%", ""))
        if pct >= 90:
            table[i, last_col].set_facecolor("#D5F5E3")
        elif pct >= 70:
            table[i, last_col].set_facecolor("#FEF9E7")
        else:
            table[i, last_col].set_facecolor("#FADBD8")

    ax.set_title(
        "Sensor-Label Time Overlap Analysis",
        fontsize=13, fontweight="bold", pad=20,
    )
    return fig


def generate_time_coverage(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> list[Path]:
    """Generate all time coverage plots."""
    output_dir = config.output_dir / "04_time_coverage"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    plots = [
        ("gantt_chart", plot_gantt_chart),
        ("inter_period_gaps", plot_inter_period_gaps),
        ("overlap_matrix", plot_overlap_matrix),
    ]

    for name, func in plots:
        fig = func(periods, config)
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        logger.info("Saved: %s", path)

    return saved
