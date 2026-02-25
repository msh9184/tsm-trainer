"""02. Class Balance Analysis — Distribution, imbalance, and temporal patterns.

Generates:
  - Per-period class distribution bars (occupied vs empty)
  - Overall class ratio pie chart with imbalance ratio annotation
  - Hourly occupancy heatmap (hour-of-day vs period)
  - Occupancy rate over time (rolling window)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .config import AnalysisConfig, CLASS_COLORS, CLASS_NAMES, PERIOD_COLORS
from .data_loader import PeriodData

logger = logging.getLogger(__name__)


def plot_class_distribution(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Side-by-side class distribution bars and pie charts per period."""
    n_periods = len(periods)
    fig, axes = plt.subplots(2, n_periods, figsize=(4 * n_periods, 8))

    if n_periods == 1:
        axes = axes.reshape(2, 1)

    for j, p in enumerate(periods):
        mask = p.label_series != -1
        labels = p.label_series[mask]
        n_occ = (labels == 1).sum()
        n_empty = (labels == 0).sum()
        total = n_occ + n_empty

        # --- Top: Bar chart ---
        ax = axes[0, j]
        bars = ax.bar(
            ["Empty", "Occupied"], [n_empty, n_occ],
            color=[CLASS_COLORS[0], CLASS_COLORS[1]],
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, [n_empty, n_occ]):
            pct = val / max(total, 1) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=8,
            )
        ax.set_title(f"{p.name} ({p.date_range})", fontsize=10, fontweight="bold")
        ax.set_ylabel("Count" if j == 0 else "")

        # Imbalance ratio
        imbalance_ratio = max(n_occ, n_empty) / max(min(n_occ, n_empty), 1)
        ax.text(
            0.95, 0.95, f"IR: {imbalance_ratio:.1f}:1",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#666666",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0", alpha=0.8),
        )

        # --- Bottom: Pie chart ---
        ax2 = axes[1, j]
        if n_occ == 0 or n_empty == 0:
            # Degenerate case
            ax2.pie(
                [max(n_occ, 1), max(n_empty, 1)],
                colors=[CLASS_COLORS[1], CLASS_COLORS[0]],
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 9},
            )
        else:
            ax2.pie(
                [n_occ, n_empty],
                labels=["Occupied", "Empty"],
                colors=[CLASS_COLORS[1], CLASS_COLORS[0]],
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 9},
                wedgeprops={"edgecolor": "white", "linewidth": 0.5},
            )

    fig.suptitle(
        "Class Distribution per Period (Binary: Occupied vs Empty)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    return fig


def plot_overall_balance(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Overall aggregate class balance with annotation."""
    total_occ = sum((p.label_series == 1).sum() for p in periods)
    total_empty = sum((p.label_series == 0).sum() for p in periods)
    total = total_occ + total_empty

    fig, axes = plt.subplots(1, 2, figsize=config.figsize_wide)

    # --- Left: Stacked horizontal bar per period ---
    ax = axes[0]
    names = [p.name for p in periods]
    y_pos = np.arange(len(names))

    occ_counts = [(p.label_series == 1).sum() for p in periods]
    empty_counts = [(p.label_series == 0).sum() for p in periods]

    ax.barh(y_pos, empty_counts, color=CLASS_COLORS[0], label="Empty", height=0.6)
    ax.barh(y_pos, occ_counts, left=empty_counts, color=CLASS_COLORS[1],
            label="Occupied", height=0.6)

    # Annotate percentages
    for i, (ec, oc) in enumerate(zip(empty_counts, occ_counts)):
        total_p = ec + oc
        if total_p > 0:
            ax.text(
                ec / 2, i, f"{ec / total_p * 100:.0f}%",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold",
            )
            ax.text(
                ec + oc / 2, i, f"{oc / total_p * 100:.0f}%",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Number of Timesteps")
    ax.set_title("Class Balance per Period")
    ax.legend(loc="lower right")

    # --- Right: Overall donut chart ---
    ax2 = axes[1]
    sizes = [total_occ, total_empty]
    colors = [CLASS_COLORS[1], CLASS_COLORS[0]]
    explode = (0.03, 0.03)

    wedges, texts, autotexts = ax2.pie(
        sizes, labels=["Occupied", "Empty"], colors=colors,
        autopct="%1.1f%%", startangle=90, explode=explode,
        pctdistance=0.75, textprops={"fontsize": 10},
        wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 1},
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")

    # Center text
    ir = max(total_occ, total_empty) / max(min(total_occ, total_empty), 1)
    ax2.text(
        0, 0, f"Total\n{total:,}\nIR: {ir:.1f}:1",
        ha="center", va="center", fontsize=10, fontweight="bold",
    )
    ax2.set_title("Overall Class Distribution")

    fig.suptitle(
        "Class Balance Summary (All Periods Combined)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    return fig


def plot_hourly_occupancy_heatmap(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Heatmap: occupancy rate by hour-of-day across periods.

    Each cell shows the fraction of timesteps that are occupied
    for that (period, hour) combination. Reveals daily patterns.
    """
    names = [p.name for p in periods]
    hours = list(range(24))

    # Build matrix: (periods × hours)
    matrix = np.full((len(periods), 24), np.nan)

    for i, p in enumerate(periods):
        mask = p.label_series != -1
        df = pd.DataFrame({
            "label": p.label_series[mask],
            "hour": p.label_series[mask].index.hour,
        })
        for h in hours:
            hour_mask = df["hour"] == h
            if hour_mask.sum() > 0:
                matrix[i, h] = df.loc[hour_mask, "label"].mean()

    fig, ax = plt.subplots(figsize=(14, 4))

    im = ax.imshow(
        matrix, aspect="auto", cmap="RdYlGn_r",
        vmin=0, vmax=1, interpolation="nearest",
    )

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in hours], fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} ({p.date_range})" for n, p in zip(names, periods)],
                       fontsize=9)
    ax.set_xlabel("Hour of Day")
    ax.set_title(
        "Occupancy Rate by Hour of Day\n"
        "(Green = mostly empty, Red = mostly occupied, White = no data)",
        fontsize=11, fontweight="bold",
    )

    # Add cell text
    for i in range(len(periods)):
        for h in range(24):
            val = matrix[i, h]
            if not np.isnan(val):
                color = "white" if val > 0.6 or val < 0.3 else "black"
                ax.text(
                    h, i, f"{val:.0%}",
                    ha="center", va="center", fontsize=7, color=color,
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Occupancy Rate", fontsize=9)

    return fig


def plot_occupancy_timeline(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Rolling occupancy rate over time for each period (line chart).

    Shows how occupancy rate varies over time with a 1-hour rolling window.
    """
    fig, axes = plt.subplots(
        len(periods), 1, figsize=(14, 3 * len(periods)),
        sharex=False,
    )
    if len(periods) == 1:
        axes = [axes]

    for ax, p in zip(axes, periods):
        mask = p.label_series != -1
        labels = p.label_series[mask].copy()

        # Rolling 1-hour window (12 × 5-min steps)
        rolling_rate = labels.rolling(window=12, min_periods=1).mean()

        # Fill background based on binary label
        ax.fill_between(
            labels.index, 0, 1,
            where=(labels == 1),
            alpha=0.15, color=CLASS_COLORS[1], label="Occupied",
            step="post",
        )
        ax.fill_between(
            labels.index, 0, 1,
            where=(labels == 0),
            alpha=0.15, color=CLASS_COLORS[0], label="Empty",
            step="post",
        )

        # Rolling rate line
        ax.plot(
            rolling_rate.index, rolling_rate.values,
            color="#333333", linewidth=1.2, label="1h rolling rate",
        )

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Occ. Rate")
        ax.set_title(f"{p.name} ({p.date_range})", fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=7)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.axhline(0.5, color="#999999", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.suptitle(
        "Occupancy State Timeline (1-hour Rolling Average)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    return fig


def generate_class_balance(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> list[Path]:
    """Generate all class balance plots."""
    output_dir = config.output_dir / "02_class_balance"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    plots = [
        ("class_distribution", plot_class_distribution),
        ("overall_balance", plot_overall_balance),
        ("hourly_heatmap", plot_hourly_occupancy_heatmap),
        ("occupancy_timeline", plot_occupancy_timeline),
    ]

    for name, func in plots:
        fig = func(periods, config)
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        logger.info("Saved: %s", path)

    return saved
