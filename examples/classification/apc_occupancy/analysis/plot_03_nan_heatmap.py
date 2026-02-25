"""03. Missing Value (NaN) Analysis — Heatmaps and patterns.

Generates:
  - NaN fraction heatmap per sensor × period
  - NaN fraction bar chart (sorted) for common columns
  - Temporal NaN pattern: NaN rate over time per sensor
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from .config import AnalysisConfig, COMMON_COLUMNS, CORE_SENSORS, NAN_CMAP, PERIOD_COLORS
from .data_loader import PeriodData

logger = logging.getLogger(__name__)


def plot_nan_heatmap(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Heatmap: NaN fraction for each sensor column in each period.

    Rows = sensor columns (sorted), columns = periods.
    Color intensity shows NaN fraction (0% green → 100% red).
    """
    # Collect all columns
    all_cols = sorted(set().union(*(set(p.sensor_df.columns) for p in periods)))
    names = [p.name for p in periods]

    # Build NaN fraction matrix
    matrix = np.full((len(all_cols), len(names)), np.nan)
    for j, p in enumerate(periods):
        nan_frac = p.sensor_df.isna().mean()
        for i, col in enumerate(all_cols):
            if col in nan_frac.index:
                matrix[i, j] = nan_frac[col]
            else:
                matrix[i, j] = 1.0  # Column doesn't exist = 100% missing

    fig, ax = plt.subplots(figsize=(7, max(8, len(all_cols) * 0.28)))

    # Custom colormap: green (0%) -> yellow (50%) -> red (100%)
    cmap = plt.colormaps.get_cmap(NAN_CMAP)
    im = ax.imshow(
        matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1,
        interpolation="nearest",
    )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(all_cols)))

    # Mark core sensors with a marker
    ytick_labels = []
    for col in all_cols:
        prefix = ">> " if col in CORE_SENSORS else "   "
        ytick_labels.append(f"{prefix}{col}")
    ax.set_yticklabels(ytick_labels, fontsize=6.5, fontfamily="monospace")

    # Cell text: NaN percentage
    for i in range(len(all_cols)):
        for j in range(len(names)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.6 else "black"
                text = f"{val:.0%}" if val < 1.0 else "N/A"
                ax.text(
                    j, i, text,
                    ha="center", va="center", fontsize=6, color=color,
                )

    ax.set_title(
        "NaN Fraction per Sensor Column × Period\n"
        "(>> = core sensor, N/A = column absent in period)",
        fontsize=11, fontweight="bold",
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    cbar = fig.colorbar(im, ax=ax, shrink=0.4, aspect=15, pad=0.02)
    cbar.set_label("NaN Fraction", fontsize=9)

    return fig


def plot_nan_bars_common(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Bar chart: NaN fraction for the 15 common columns, grouped by period."""
    names = [p.name for p in periods]
    n_groups = len(COMMON_COLUMNS)
    n_bars = len(periods)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_groups)

    for j, p in enumerate(periods):
        nan_fracs = []
        for col in COMMON_COLUMNS:
            if col in p.sensor_df.columns:
                nan_fracs.append(p.sensor_df[col].isna().mean())
            else:
                nan_fracs.append(1.0)

        offset = (j - n_bars / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, nan_fracs, width,
            label=p.name, color=PERIOD_COLORS[p.name],
            edgecolor="white", linewidth=0.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.split("_", 1)[1] if "_" in c else c for c in COMMON_COLUMNS],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_ylabel("NaN Fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "NaN Fraction for 15 Common Sensor Columns (Grouped by Period)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0.5, color="#CC3311", linestyle="--", linewidth=0.8,
               alpha=0.7, label="50% threshold")

    # Add device ID prefixes as secondary x-axis labels
    device_ids = [c.split("_")[0] for c in COMMON_COLUMNS]
    ax2 = ax.secondary_xaxis("bottom")
    ax2.set_xticks(x)
    ax2.set_xticklabels(device_ids, fontsize=5.5, color="#999999")
    ax2.tick_params(axis="x", pad=35)

    return fig


def plot_nan_temporal(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Temporal NaN density: fraction of NaN columns at each timestep.

    Shows how data completeness varies over time for each period.
    """
    fig, axes = plt.subplots(
        len(periods), 1,
        figsize=(14, 2.5 * len(periods)),
        sharex=False,
    )
    if len(periods) == 1:
        axes = [axes]

    for ax, p in enumerate(periods):
        pdata = periods[p] if isinstance(p, int) else p
        ax_obj = axes[p] if isinstance(p, int) else ax

    # Re-do loop properly
    for idx, (ax, p) in enumerate(zip(axes, periods)):
        # NaN fraction per timestep (across all columns)
        nan_per_row = p.sensor_df.isna().mean(axis=1)

        ax.fill_between(
            nan_per_row.index, nan_per_row.values,
            alpha=0.6, color=PERIOD_COLORS[p.name],
        )
        ax.plot(
            nan_per_row.index, nan_per_row.values,
            color=PERIOD_COLORS[p.name], linewidth=0.5,
        )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("NaN Frac.", fontsize=8)
        ax.set_title(f"{p.name} ({p.date_range})", fontsize=9, fontweight="bold")

        # Annotate average NaN rate
        avg_nan = nan_per_row.mean()
        ax.axhline(avg_nan, color="#333333", linestyle="--", linewidth=0.5)
        ax.text(
            0.98, 0.85, f"Avg: {avg_nan:.1%}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#333333",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=7)

    fig.suptitle(
        "Temporal NaN Density (Fraction of Missing Columns per Timestep)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    return fig


def generate_nan_analysis(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> list[Path]:
    """Generate all NaN analysis plots."""
    output_dir = config.output_dir / "03_nan_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    plots = [
        ("nan_heatmap", plot_nan_heatmap),
        ("nan_bars_common", plot_nan_bars_common),
        ("nan_temporal", plot_nan_temporal),
    ]

    for name, func in plots:
        fig = func(periods, config)
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        logger.info("Saved: %s", path)

    return saved
