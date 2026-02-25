"""07. Time Gap Analysis — Interval consistency and gap detection.

Generates:
  - Histogram of time intervals between consecutive timesteps
  - Gap detection: identify intervals > expected 5 minutes
  - Event timing analysis: time between consecutive occupancy events
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .config import AnalysisConfig, PERIOD_COLORS
from .data_loader import PeriodData

logger = logging.getLogger(__name__)


def plot_interval_histogram(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Histogram of time intervals between consecutive sensor timesteps.

    Expected: all intervals = 5 minutes. Any deviation indicates gaps.
    """
    fig, axes = plt.subplots(1, len(periods), figsize=(4 * len(periods), 5))
    if len(periods) == 1:
        axes = [axes]

    for ax, p in zip(axes, periods):
        # Compute intervals in minutes
        diffs = pd.Series(p.sensor_df.index).diff().dropna()
        intervals_min = diffs.dt.total_seconds() / 60

        # Histogram
        bins = np.arange(0, max(intervals_min.max() + 1, 12), 1)
        counts, edges, patches = ax.hist(
            intervals_min, bins=bins,
            color=PERIOD_COLORS[p.name], edgecolor="white", linewidth=0.5,
            alpha=0.8,
        )

        # Highlight the expected 5-min bin
        for patch, left_edge in zip(patches, edges[:-1]):
            if 4.5 <= left_edge < 5.5:
                patch.set_facecolor("#2ECC71")
                patch.set_edgecolor("black")

        # Annotate statistics
        n_perfect = (intervals_min == 5.0).sum()
        n_total = len(intervals_min)
        n_gaps = (intervals_min > 5.0).sum()

        stats_text = (
            f"Total: {n_total}\n"
            f"= 5 min: {n_perfect} ({n_perfect / n_total * 100:.1f}%)\n"
            f"> 5 min: {n_gaps}"
        )
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
        )

        ax.set_xlabel("Interval (minutes)")
        ax.set_ylabel("Count")
        ax.set_title(f"{p.name} ({p.date_range})", fontsize=10, fontweight="bold")
        ax.axvline(5.0, color="#CC3311", linewidth=1, linestyle="--",
                   label="Expected (5 min)")
        ax.legend(fontsize=7)

    fig.suptitle(
        "Time Interval Distribution Between Consecutive Timesteps",
        fontsize=12, fontweight="bold", y=1.02,
    )
    return fig


def plot_gap_detection(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Detect and visualize gaps (intervals > 5 min) in sensor data.

    If gaps exist: scatter plot of gap locations and sizes.
    If no gaps: summary table showing interval consistency per period.
    """
    all_gaps = []

    for p in periods:
        timestamps = p.sensor_df.index
        diffs = pd.Series(timestamps).diff().dropna()
        intervals_min = diffs.dt.total_seconds() / 60

        gap_mask = intervals_min > 5.0
        if gap_mask.any():
            gap_indices = gap_mask[gap_mask].index
            for idx in gap_indices:
                gap_start = timestamps[idx - 1]
                gap_end = timestamps[idx]
                gap_size = intervals_min.iloc[idx]
                all_gaps.append({
                    "period": p.name,
                    "start": gap_start,
                    "end": gap_end,
                    "size_min": gap_size,
                })

    if all_gaps:
        fig, ax = plt.subplots(figsize=(14, 5))
        for g in all_gaps:
            ax.scatter(
                mdates.date2num(g["start"]), g["size_min"],
                color=PERIOD_COLORS[g["period"]],
                s=max(20, min(g["size_min"] * 2, 200)),
                alpha=0.7, edgecolor="black", linewidth=0.5,
                zorder=3,
            )
        ax.axhline(5.0, color="#2ECC71", linewidth=1, linestyle="--",
                   label="Expected (5 min)", zorder=1)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Gap Size (minutes)")
        ax.set_title(
            "Gap Detection: Time Intervals Exceeding 5 Minutes\n"
            "(Bubble size proportional to gap duration)",
            fontsize=12, fontweight="bold",
        )
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PERIOD_COLORS[p.name],
                   markersize=8, label=p.name)
            for p in periods
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        logger.info("Detected %d gaps across all periods:", len(all_gaps))
        for g in all_gaps:
            logger.info(
                "  %s: %s -> %s (%.1f min)",
                g["period"], g["start"], g["end"], g["size_min"],
            )
    else:
        # No gaps: show a clean summary table instead of empty scatter
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")

        rows = []
        for p in periods:
            timestamps = p.sensor_df.index
            diffs = pd.Series(timestamps).diff().dropna()
            intervals_min = diffs.dt.total_seconds() / 60
            n_intervals = len(intervals_min)
            n_exact_5 = (intervals_min == 5.0).sum()
            pct_exact = n_exact_5 / max(n_intervals, 1) * 100

            rows.append([
                p.name, p.date_range,
                f"{n_intervals:,}", f"{n_exact_5:,}",
                f"{pct_exact:.1f}%", "0",
                "PERFECT" if pct_exact == 100 else f"{100 - pct_exact:.1f}% irregular",
            ])

        col_labels = [
            "Period", "Date Range", "Total Intervals",
            "Exact 5 min", "Consistency", "Gaps (>5min)", "Status",
        ]
        table = ax.table(
            cellText=rows, colLabels=col_labels,
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)

        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)

        # Green status for perfect (last column)
        last_col = len(col_labels) - 1
        for i, row in enumerate(rows, 1):
            if "PERFECT" in row[-1]:
                table[i, last_col].set_facecolor("#D5F5E3")

        ax.set_title(
            "Gap Detection Summary\n"
            "All sensor files have perfect 5-minute intervals with zero gaps!",
            fontsize=12, fontweight="bold", pad=20, color="#2ECC71",
        )

    return fig


def plot_event_timing(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Analyze timing patterns of occupancy events.

    Shows:
      - Distribution of time between consecutive events
      - Event density over time (events per hour)
      - Event type breakdown (ENTER vs LEAVE vs NONE)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Left: Inter-event time distribution ---
    ax = axes[0]
    all_inter_event = []
    for p in periods:
        events = p.events_df.copy()
        events["time"] = pd.to_datetime(events["time"])
        if len(events) > 1:
            diffs_min = events["time"].diff().dropna().dt.total_seconds() / 60
            ax.hist(
                diffs_min, bins=30, alpha=0.6,
                color=PERIOD_COLORS[p.name], label=p.name,
                edgecolor="white", linewidth=0.3,
            )
            all_inter_event.extend(diffs_min.tolist())

    if all_inter_event:
        median_gap = np.median(all_inter_event)
        ax.axvline(
            median_gap, color="#CC3311", linewidth=1.5, linestyle="--",
            label=f"Median: {median_gap:.0f} min",
        )
    ax.set_xlabel("Minutes Between Events")
    ax.set_ylabel("Count")
    ax.set_title("Inter-Event Time Distribution")
    ax.legend(fontsize=7)
    ax.set_xlim(left=0)

    # --- Center: Event type breakdown ---
    ax2 = axes[1]
    event_types = {"ENTER_HOME": [], "LEAVE_HOME": [], "NONE": []}
    for p in periods:
        for etype in event_types:
            count = (p.events_df["Status"] == etype).sum()
            event_types[etype].append(count)

    x = np.arange(len(periods))
    width = 0.25
    type_colors = {"ENTER_HOME": "#009E73", "LEAVE_HOME": "#CC3311", "NONE": "#999999"}

    for i, (etype, counts) in enumerate(event_types.items()):
        offset = (i - 1) * width
        ax2.bar(x + offset, counts, width, label=etype,
                color=type_colors[etype], edgecolor="white", linewidth=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels([p.name for p in periods])
    ax2.set_ylabel("Count")
    ax2.set_title("Event Type Breakdown per Period")
    ax2.legend(fontsize=8)

    # --- Right: Events per hour of day ---
    ax3 = axes[2]
    hourly_counts = np.zeros(24)
    for p in periods:
        events = p.events_df.copy()
        events["time"] = pd.to_datetime(events["time"])
        hours = events["time"].dt.hour
        for h in hours:
            hourly_counts[h] += 1

    ax3.bar(
        range(24), hourly_counts, color="#4E79A7",
        edgecolor="white", linewidth=0.3,
    )
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("Total Events (All Periods)")
    ax3.set_title("Event Frequency by Hour")
    ax3.set_xticks(range(0, 24, 2))
    ax3.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)], fontsize=8)

    # Annotate peak hours
    peak_hour = int(np.argmax(hourly_counts))
    ax3.annotate(
        f"Peak: {peak_hour:02d}h\n({int(hourly_counts[peak_hour])} events)",
        xy=(peak_hour, hourly_counts[peak_hour]),
        xytext=(peak_hour + 3, hourly_counts[peak_hour]),
        fontsize=8, arrowprops=dict(arrowstyle="->", color="#CC3311"),
        color="#CC3311",
    )

    fig.suptitle(
        "Occupancy Event Timing Analysis",
        fontsize=13, fontweight="bold", y=1.02,
    )
    return fig


def generate_gap_analysis(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> list[Path]:
    """Generate all gap analysis plots."""
    output_dir = config.output_dir / "07_gap_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    plots = [
        ("interval_histogram", plot_interval_histogram),
        ("gap_detection", plot_gap_detection),
        ("event_timing", plot_event_timing),
    ]

    for name, func in plots:
        fig = func(periods, config)
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        logger.info("Saved: %s", path)

    return saved
