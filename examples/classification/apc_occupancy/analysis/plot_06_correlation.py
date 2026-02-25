"""06. Correlation Analysis — Sensor inter-correlation and class separability.

Generates:
  - Sensor-sensor correlation matrix heatmap
  - Sensor-label point-biserial correlation bar chart
  - Class-conditional distribution (box plots) for top sensors
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .config import AnalysisConfig, CLASS_COLORS, CLASS_NAMES, CORE_SENSORS, PERIOD_COLORS
from .data_loader import PeriodData, get_labeled_only

logger = logging.getLogger(__name__)


def _point_biserial(sensor_values: np.ndarray, labels: np.ndarray) -> float:
    """Compute point-biserial correlation between sensor and binary label.

    Handles NaN by dropping pairs where either value is NaN.
    """
    mask = ~(np.isnan(sensor_values) | np.isnan(labels.astype(float)))
    if mask.sum() < 10:
        return np.nan
    corr, _ = sp_stats.pointbiserialr(labels[mask], sensor_values[mask])
    return corr


def plot_correlation_matrix(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Correlation matrix of all common sensor columns using combined data.

    Uses Pearson correlation on numeric sensor values aggregated
    across all periods.
    """
    # Combine all period sensor data (common columns only)
    from .config import COMMON_COLUMNS
    all_dfs = []
    for p in periods:
        available = [c for c in COMMON_COLUMNS if c in p.sensor_df.columns]
        df_subset = p.sensor_df[available].copy()
        all_dfs.append(df_subset)

    if not all_dfs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    combined = pd.concat(all_dfs, axis=0)
    corr_matrix = combined.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 9))

    # Plot heatmap
    n = len(corr_matrix)
    im = ax.imshow(
        corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1,
        aspect="auto", interpolation="nearest",
    )

    # Short column names
    short_names = []
    for col in corr_matrix.columns:
        parts = col.split("_", 1)
        short = parts[1] if len(parts) == 2 else col
        prefix = parts[0][:4] if len(parts) == 2 else ""
        short_names.append(f"{prefix}_{short}" if prefix else short)

    ax.set_xticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=7)

    # Add correlation values in cells
    for i in range(n):
        for j in range(n):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=5, color=color,
                )

    ax.set_title(
        "Sensor-Sensor Pearson Correlation (15 Common Columns, All Periods)",
        fontsize=11, fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Pearson r", fontsize=9)

    return fig


def plot_sensor_label_correlation(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Point-biserial correlation between each sensor and the binary label.

    Sorted by absolute correlation. Higher = more discriminative.
    """
    from .config import COMMON_COLUMNS

    # Collect all labeled data
    all_sensors = []
    all_labels = []
    for p in periods:
        sensor_df, label_series = get_labeled_only(p)
        available = [c for c in COMMON_COLUMNS if c in sensor_df.columns]
        all_sensors.append(sensor_df[available])
        all_labels.append(label_series)

    combined_sensors = pd.concat(all_sensors, axis=0)
    combined_labels = pd.concat(all_labels, axis=0)

    # Compute point-biserial correlation for each sensor
    correlations = {}
    for col in combined_sensors.columns:
        corr = _point_biserial(
            combined_sensors[col].values,
            combined_labels.values,
        )
        correlations[col] = corr

    # Sort by absolute value
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0,
                          reverse=True)
    names = [c[0] for c in sorted_corrs]
    values = [c[1] for c in sorted_corrs]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Short names
    short_names = []
    for col in names:
        parts = col.split("_", 1)
        short = parts[1] if len(parts) == 2 else col
        is_core = col in CORE_SENSORS
        short_names.append(f">> {short}" if is_core else f"   {short}")

    y_pos = np.arange(len(names))
    colors = ["#CC3311" if v < 0 else "#0173B2" for v in values]

    ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.3, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=7, fontfamily="monospace")
    ax.set_xlabel("Point-Biserial Correlation with Occupancy Label")
    ax.set_title(
        "Sensor Discriminative Power\n"
        "(Point-Biserial r with Binary Occupancy, >> = core sensor)\n"
        "Positive = higher when occupied, Negative = lower when occupied",
        fontsize=10, fontweight="bold",
    )
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.1, color="#999999", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(-0.1, color="#999999", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.invert_yaxis()

    # Annotate values
    for i, v in enumerate(values):
        if not np.isnan(v):
            ax.text(
                v + 0.005 * np.sign(v), i, f"{v:.3f}",
                va="center", fontsize=6,
            )

    return fig


def plot_class_conditional_boxplots(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> plt.Figure:
    """Box plots of core sensor values conditioned on class label.

    Shows distribution of sensor readings when occupied vs empty,
    revealing which sensors have the best class separation.
    """
    # Collect all labeled data
    all_sensors = []
    all_labels = []
    for p in periods:
        sensor_df, label_series = get_labeled_only(p)
        all_sensors.append(sensor_df)
        all_labels.append(label_series)

    combined_sensors = pd.concat(all_sensors, axis=0)
    combined_labels = pd.concat(all_labels, axis=0)

    # Use core sensors that are available
    available_core = [c for c in CORE_SENSORS if c in combined_sensors.columns]
    n = len(available_core)

    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No core sensors available", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(
        2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 8),
    )
    axes = axes.flatten()

    for i, col in enumerate(available_core):
        ax = axes[i]
        values = combined_sensors[col].values
        labels = combined_labels.values

        # Separate by class
        empty_vals = values[(labels == 0) & ~np.isnan(values)]
        occ_vals = values[(labels == 1) & ~np.isnan(values)]

        if len(empty_vals) == 0 and len(occ_vals) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        bp = ax.boxplot(
            [empty_vals, occ_vals],
            labels=["Empty", "Occupied"],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )

        for patch, color in zip(bp["boxes"], [CLASS_COLORS[0], CLASS_COLORS[1]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        # Compute effect size (Cohen's d)
        if len(empty_vals) > 1 and len(occ_vals) > 1:
            pooled_std = np.sqrt(
                ((len(empty_vals) - 1) * np.var(empty_vals, ddof=1) +
                 (len(occ_vals) - 1) * np.var(occ_vals, ddof=1)) /
                (len(empty_vals) + len(occ_vals) - 2)
            )
            if pooled_std > 0:
                cohen_d = (np.mean(occ_vals) - np.mean(empty_vals)) / pooled_std
                ax.text(
                    0.95, 0.95, f"d={cohen_d:.2f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="#333333",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#F0F0F0", alpha=0.8),
                )

        short_name = col.split("_", 1)[1] if "_" in col else col
        ax.set_title(short_name, fontsize=9, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Class-Conditional Sensor Distributions (Core Sensors)\n"
        "d = Cohen's d effect size (|d|>0.5 = medium, |d|>0.8 = large)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    return fig


def generate_correlation(
    periods: list[PeriodData],
    config: AnalysisConfig,
) -> list[Path]:
    """Generate all correlation analysis plots."""
    output_dir = config.output_dir / "06_correlation"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    plots = [
        ("correlation_matrix", plot_correlation_matrix),
        ("sensor_label_correlation", plot_sensor_label_correlation),
        ("class_conditional_boxplots", plot_class_conditional_boxplots),
    ]

    for name, func in plots:
        fig = func(periods, config)
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        logger.info("Saved: %s", path)

    return saved
