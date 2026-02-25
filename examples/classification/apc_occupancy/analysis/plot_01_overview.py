"""01. Dataset Overview — Summary statistics and data inventory.

Generates:
  - Summary statistics table (PNG image of table)
  - Column inventory heatmap (which columns exist in which period)
  - Per-period data volume bar chart
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import AnalysisConfig, PERIOD_COLORS, COMMON_COLUMNS
from .data_loader import PeriodData

logger = logging.getLogger(__name__)


def plot_data_volume_bar(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Stacked bar chart: sensor timesteps (labeled vs unlabeled) per period.

    Shows total timestep count per period, with labeled portion in one
    color and unlabeled portion (after last event) in gray.
    """
    names = [p.name for p in periods]
    n_labeled = []
    n_unlabeled = []
    n_events = []

    for p in periods:
        mask_labeled = p.label_series != -1
        n_labeled.append(mask_labeled.sum())
        n_unlabeled.append((~mask_labeled).sum())
        n_events.append(len(p.events_df))

    fig, axes = plt.subplots(1, 2, figsize=config.figsize_wide)

    # --- Left: Sensor timestep volume ---
    ax = axes[0]
    x = np.arange(len(names))
    width = 0.5

    bars_labeled = ax.bar(
        x, n_labeled, width, label="Labeled",
        color=[PERIOD_COLORS[n] for n in names], edgecolor="white", linewidth=0.5,
    )
    bars_unlabeled = ax.bar(
        x, n_unlabeled, width, bottom=n_labeled, label="Unlabeled",
        color="#CCCCCC", edgecolor="white", linewidth=0.5,
    )

    # Annotate bar values
    for bar_l, bar_u, nl, nu in zip(bars_labeled, bars_unlabeled, n_labeled, n_unlabeled):
        total = nl + nu
        ax.text(
            bar_l.get_x() + bar_l.get_width() / 2, total + 20,
            f"{total:,}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
        if nu > 0:
            ax.text(
                bar_u.get_x() + bar_u.get_width() / 2,
                nl + nu / 2,
                f"{nu:,}", ha="center", va="center", fontsize=8, color="#666666",
            )

    ax.set_xlabel("Period")
    ax.set_ylabel("Number of 5-min Timesteps")
    ax.set_title("Sensor Data Volume per Period")
    ax.set_xticks(x)
    date_labels = [f"{p.name}\n({p.date_range})" for p in periods]
    ax.set_xticklabels(date_labels, fontsize=8)
    ax.legend(loc="upper left", framealpha=0.9)

    # --- Right: Event count per period ---
    ax2 = axes[1]
    colors = [PERIOD_COLORS[n] for n in names]
    bars = ax2.bar(x, n_events, width, color=colors, edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, n_events):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, count + 0.5,
            str(count), ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax2.set_xlabel("Period")
    ax2.set_ylabel("Number of Occupancy Events")
    ax2.set_title("Occupancy Events per Period")
    ax2.set_xticks(x)
    ax2.set_xticklabels(date_labels, fontsize=8)

    fig.suptitle("Dataset Overview: Data Volume", fontsize=13, fontweight="bold", y=1.02)
    return fig


def plot_column_inventory(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Heatmap showing which sensor columns are present in each period.

    Rows = sensor columns (sorted), columns = periods.
    Cell value: 1 = present, 0 = absent. Color-coded for clarity.
    """
    # Collect all columns across periods
    all_cols = sorted(set().union(*(set(p.sensor_df.columns) for p in periods)))
    names = [p.name for p in periods]

    # Build presence matrix
    matrix = np.zeros((len(all_cols), len(names)), dtype=int)
    for j, p in enumerate(periods):
        for i, col in enumerate(all_cols):
            if col in p.sensor_df.columns:
                matrix[i, j] = 1

    fig, ax = plt.subplots(figsize=(6, max(8, len(all_cols) * 0.28)))

    # Use imshow for heatmap (no seaborn needed)
    im = ax.imshow(
        matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1,
        interpolation="nearest",
    )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(all_cols)))

    # Highlight common columns vs period-specific
    ytick_labels = []
    for col in all_cols:
        if col in COMMON_COLUMNS:
            ytick_labels.append(col)
        else:
            ytick_labels.append(f"* {col}")
    ax.set_yticklabels(ytick_labels, fontsize=7, fontfamily="monospace")

    # Add cell text
    for i in range(len(all_cols)):
        for j in range(len(names)):
            color = "white" if matrix[i, j] == 1 else "#999999"
            symbol = "Y" if matrix[i, j] == 1 else "-"
            ax.text(j, i, symbol, ha="center", va="center", fontsize=7, color=color)

    ax.set_title(
        "Sensor Column Inventory\n(* = period-specific, unstarred = common to all)",
        fontsize=11, fontweight="bold",
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.3, aspect=10, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Absent", "Present"])

    return fig


def plot_summary_table(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Render a publication-quality summary statistics table as an image."""
    rows = []
    for p in periods:
        mask_labeled = p.label_series != -1
        n_labeled = mask_labeled.sum()
        n_unlabeled = (~mask_labeled).sum()
        n_occupied = (p.label_series == 1).sum()
        n_empty = (p.label_series == 0).sum()

        # Time range
        t_start = p.sensor_df.index.min().strftime("%m/%d %H:%M")
        t_end = p.sensor_df.index.max().strftime("%m/%d %H:%M")

        # Duration in days
        duration = (p.sensor_df.index.max() - p.sensor_df.index.min())
        duration_days = duration.total_seconds() / 86400

        occ_pct = n_occupied / max(n_labeled, 1) * 100

        rows.append([
            p.name,
            f"{t_start} - {t_end}",
            f"{duration_days:.1f}d",
            f"{len(p.sensor_df):,}",
            str(len(p.sensor_df.columns)),
            str(len(p.events_df)),
            f"{n_labeled:,}",
            f"{n_unlabeled:,}",
            f"{n_occupied:,} ({occ_pct:.0f}%)",
            f"{n_empty:,} ({100 - occ_pct:.0f}%)",
        ])

    # Add total row
    total_ts = sum(len(p.sensor_df) for p in periods)
    total_events = sum(len(p.events_df) for p in periods)
    total_labeled = sum((p.label_series != -1).sum() for p in periods)
    total_unlabeled = sum((p.label_series == -1).sum() for p in periods)
    total_occ = sum((p.label_series == 1).sum() for p in periods)
    total_empty = sum((p.label_series == 0).sum() for p in periods)
    total_occ_pct = total_occ / max(total_labeled, 1) * 100

    rows.append([
        "TOTAL", "-", "-",
        f"{total_ts:,}", "-",
        str(total_events),
        f"{total_labeled:,}",
        f"{total_unlabeled:,}",
        f"{total_occ:,} ({total_occ_pct:.0f}%)",
        f"{total_empty:,} ({100 - total_occ_pct:.0f}%)",
    ])

    col_labels = [
        "Period", "Time Range", "Duration",
        "Timesteps", "Columns", "Events",
        "Labeled", "Unlabeled",
        "Occupied", "Empty",
    ]

    fig, ax = plt.subplots(figsize=(16, 3.5))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)

    # Style total row
    total_row_idx = len(rows)
    for j in range(len(col_labels)):
        cell = table[total_row_idx, j]
        cell.set_facecolor("#ECF0F1")
        cell.set_text_props(fontweight="bold")

    # Alternate row colors
    for i in range(1, len(rows)):
        color = "#F8F9FA" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    ax.set_title(
        "APC Occupancy Dataset — Summary Statistics",
        fontsize=13, fontweight="bold", pad=20,
    )
    return fig


def generate_overview(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> list[Path]:
    """Generate all overview plots and return saved file paths."""
    output_dir = config.output_dir / "01_overview"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1. Summary table
    fig = plot_summary_table(periods, config)
    path = output_dir / "summary_table.png"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    logger.info("Saved: %s", path)

    # 2. Data volume bar chart
    fig = plot_data_volume_bar(periods, config)
    path = output_dir / "data_volume.png"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    logger.info("Saved: %s", path)

    # 3. Column inventory heatmap
    fig = plot_column_inventory(periods, config)
    path = output_dir / "column_inventory.png"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    logger.info("Saved: %s", path)

    return saved
