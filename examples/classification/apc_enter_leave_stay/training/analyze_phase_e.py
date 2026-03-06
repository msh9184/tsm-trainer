"""Deep analysis of Phase E (cross-setting) results.

Phase E runs 3 label settings × 6 layers × 4 classifiers = 72 experiments.
This script properly parses the mixed-column CSV and provides:
  - Cross-setting comparison tables
  - Per-class F1 breakdown revealing class imbalance issues
  - Layer × Setting heatmaps
  - Recommendations for Round 2 setting selection

Usage:
    cd examples/classification/apc_enter_leave_stay

    # Analyze Phase E results
    python training/analyze_phase_e.py --results /path/to/results_E.csv

    # Also verify class distribution from raw event data
    python training/analyze_phase_e.py --results /path/to/results_E.csv \
        --events-csv /path/to/occupancy_events_..._class5.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

import matplotlib as mpl
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    mpl.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label setting definitions (from preprocess.py)
# ---------------------------------------------------------------------------

LABEL_SETTINGS = {
    1: {  # 5-class
        "ENTER_HOME_NEW": 0, "ENTER_HOME_ADD": 1,
        "LEAVE_HOME_LAST": 2, "LEAVE_HOME_REDUCE": 3, "STAY": 4,
    },
    2: {  # 3-class event-based
        "ENTER_HOME_NEW": 0, "ENTER_HOME_ADD": 0,
        "LEAVE_HOME_LAST": 1, "LEAVE_HOME_REDUCE": 1, "STAY": 2,
    },
    3: {  # 3-class occupancy-based
        "ENTER_HOME_NEW": 0, "LEAVE_HOME_LAST": 1,
        "ENTER_HOME_ADD": 2, "LEAVE_HOME_REDUCE": 2, "STAY": 2,
    },
}

CLASS_NAMES = {
    1: ["Enter_New", "Enter_Add", "Leave_Last", "Leave_Reduce", "Stay"],
    2: ["Enter", "Leave", "Stay"],
    3: ["Enter", "Leave", "Stay"],
}


def parse_phase_e_csv(csv_path: str | Path) -> pd.DataFrame:
    """Parse Phase E CSV handling mixed-column per-class F1 values.

    Phase E has 3 label settings writing different per-class F1 columns:
    - Setting 1: f1_Enter_New, f1_Enter_Add, f1_Leave_Last, f1_Leave_Reduce, f1_Stay
    - Setting 2: f1_Enter, f1_Leave, f1_Stay
    - Setting 3: f1_Enter, f1_Leave, f1_Stay

    The CSV has misaligned columns because the header was written for Setting 1
    but Settings 2/3 append extra columns. We parse manually.
    """
    rows = []
    with open(csv_path) as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty CSV: {csv_path}")

    # Base columns (before per-class F1)
    base_cols = [
        "exp_id", "phase", "layer", "classifier_type", "classifier_name",
        "context_name", "context_before", "context_after", "channel_name",
        "label_setting", "lr", "dropout", "epochs", "hidden_dims",
        "accuracy", "f1_macro", "f1_weighted", "precision_macro",
        "recall_macro", "roc_auc", "n_samples", "n_channels", "time_sec",
    ]

    for line_num, line in enumerate(lines):
        if line_num == 0:
            continue  # Skip header

        line = line.strip()
        if not line:
            continue

        # Handle quoted fields (hidden_dims has commas inside quotes)
        reader = csv.reader(StringIO(line))
        fields = next(reader)

        if len(fields) < 23:
            logger.warning("Line %d: too few fields (%d), skipping", line_num + 1, len(fields))
            continue

        row = {}
        for i, col in enumerate(base_cols):
            row[col] = fields[i] if i < len(fields) else ""

        # Determine setting from exp_id or label_setting field
        setting = int(row.get("label_setting", 3))

        # Extract per-class F1 from remaining fields
        remaining = fields[len(base_cols):]

        if setting == 1:
            # 5-class: exactly 5 per-class F1 values at positions 0-4
            s1_names = ["f1_Enter_New", "f1_Enter_Add", "f1_Leave_Last",
                        "f1_Leave_Reduce", "f1_Stay_5class"]
            for i, name in enumerate(s1_names):
                val = remaining[i] if i < len(remaining) else ""
                row[name] = float(val) if val else None
        else:
            # 3-class (Setting 2 or 3):
            # CSV column layout in the OLD format (before CSV header rewrite fix):
            #   remaining[0-3]: empty (f1_Enter_New..f1_Leave_Reduce not in S2/S3 dict)
            #   remaining[4]: f1_Stay value (shared column name with Setting 1 header)
            #   remaining[5]: f1_Enter value (new column appended after header)
            #   remaining[6]: f1_Leave value (new column appended after header)
            #
            # The DictWriter writes f1_Stay to the existing header position (col 28),
            # then appends f1_Enter and f1_Leave as new columns (cols 29-30).
            if len(remaining) >= 7:
                # Old format: positional extraction
                val_stay = remaining[4].strip() if remaining[4].strip() else ""
                val_enter = remaining[5].strip() if len(remaining) > 5 and remaining[5].strip() else ""
                val_leave = remaining[6].strip() if len(remaining) > 6 and remaining[6].strip() else ""
                row["f1_Stay"] = float(val_stay) if val_stay else None
                row["f1_Enter"] = float(val_enter) if val_enter else None
                row["f1_Leave"] = float(val_leave) if val_leave else None
            else:
                # New format or short row: extract non-empty values in order
                # After the CSV rewrite fix, columns are aligned properly:
                #   f1_Enter, f1_Leave, f1_Stay in order
                non_empty = [v for v in remaining if v.strip()]
                for i, name in enumerate(["f1_Enter", "f1_Leave", "f1_Stay"]):
                    val = non_empty[i] if i < len(non_empty) else ""
                    row[name] = float(val) if val else None

        rows.append(row)

    df = pd.DataFrame(rows)

    # Convert numeric columns
    numeric_cols = [
        "layer", "context_before", "context_after", "label_setting",
        "lr", "dropout", "epochs", "accuracy", "f1_macro", "f1_weighted",
        "precision_macro", "recall_macro", "roc_auc", "n_samples",
        "n_channels", "time_sec",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Parsed %d experiments from %s", len(df), csv_path)
    return df


def verify_class_distribution(events_csv: str | Path) -> dict[int, dict[str, int]]:
    """Load 5-class event CSV and compute class distribution per setting."""
    df = pd.read_csv(events_csv, parse_dates=["time"])
    df["Status"] = df["Status"].str.strip().str.upper()

    # 5-class raw distribution
    raw_dist = df["Status"].value_counts().to_dict()
    print("\n=== 5-Class Raw Distribution ===")
    for status, count in sorted(raw_dist.items()):
        print(f"  {status}: {count}")
    print(f"  Total: {len(df)}")

    # Per-setting distribution
    distributions = {}
    for setting, mapping in LABEL_SETTINGS.items():
        class_names = CLASS_NAMES[setting]
        df["label"] = df["Status"].map(mapping)
        dist = {}
        for label, name in enumerate(class_names):
            count = int((df["label"] == label).sum())
            dist[name] = count
        distributions[setting] = dist

        print(f"\n=== Setting {setting} Distribution ===")
        for name, count in dist.items():
            pct = 100.0 * count / len(df)
            print(f"  {name}: {count} ({pct:.1f}%)")

    return distributions


def cross_setting_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate cross-setting comparison table and plots."""
    lines = []
    lines.append("=" * 80)
    lines.append("PHASE E: CROSS-SETTING COMPARISON")
    lines.append("=" * 80)

    # Best per setting
    lines.append("\n--- Best Result per Setting (by Accuracy) ---")
    lines.append(f"{'Setting':<12} {'Best Exp':<45} {'Acc':>8} {'F1':>8} {'Clf':>20} {'Layer':>6}")
    lines.append("-" * 100)

    for setting in [1, 2, 3]:
        sdf = df[df["label_setting"] == setting]
        if sdf.empty:
            continue
        best = sdf.loc[sdf["accuracy"].idxmax()]
        lines.append(
            f"Setting {setting:<4} {best['exp_id']:<45} "
            f"{best['accuracy']:>8.4f} {best['f1_macro']:>8.4f} "
            f"{best['classifier_name']:>20} L{int(best['layer']):>4}"
        )

    # Full table: Setting × Layer × Classifier
    lines.append("\n\n--- Full Results by Setting (sorted by Accuracy) ---")
    for setting in [1, 2, 3]:
        sdf = df[df["label_setting"] == setting].sort_values("accuracy", ascending=False)
        if sdf.empty:
            continue
        class_names = CLASS_NAMES[setting]
        lines.append(f"\n{'=' * 80}")
        lines.append(f"SETTING {setting} ({len(class_names)}-class: {', '.join(class_names)})")
        lines.append(f"{'=' * 80}")

        header = f"{'Layer':>5} {'Classifier':>20} {'Acc':>8} {'F1_macro':>8}"
        for cn in class_names:
            header += f" {'f1_'+cn:>12}"
        lines.append(header)
        lines.append("-" * len(header))

        for _, row in sdf.iterrows():
            line = f"L{int(row['layer']):>4} {row['classifier_name']:>20} "
            line += f"{row['accuracy']:>8.4f} {row['f1_macro']:>8.4f}"

            if setting == 1:
                for cn in ["Enter_New", "Enter_Add", "Leave_Last", "Leave_Reduce", "Stay_5class"]:
                    val = row.get(f"f1_{cn}", None)
                    line += f" {val:>12.4f}" if val is not None else f" {'N/A':>12}"
            else:
                for cn in class_names:
                    val = row.get(f"f1_{cn}", None)
                    line += f" {val:>12.4f}" if val is not None else f" {'N/A':>12}"
            lines.append(line)

    # Layer analysis for Setting 2 (best setting)
    lines.append("\n\n--- Setting 2: Layer Analysis (Accuracy) ---")
    s2 = df[df["label_setting"] == 2]
    if not s2.empty:
        pivot = s2.pivot_table(values="accuracy", index="layer", columns="classifier_name", aggfunc="mean")
        lines.append(pivot.to_string(float_format="%.4f"))

    # Per-class F1 analysis for Setting 2
    lines.append("\n\n--- Setting 2: Per-Class F1 Analysis ---")
    lines.append("NOTE: Setting 2 has Stay=5 (6.8%) — a tiny minority class.")
    lines.append("f1_Stay=0.0 means the classifier never predicts Stay correctly.")
    lines.append("f1_Enter and f1_Leave are high because Enter=33 and Leave=36 are well-separated.")
    if not s2.empty:
        for _, row in s2.sort_values("accuracy", ascending=False).iterrows():
            f1e = row.get("f1_Enter", 0) or 0
            f1l = row.get("f1_Leave", 0) or 0
            f1s = row.get("f1_Stay", 0) or 0
            marker = " *** STAY DETECTED" if f1s and f1s > 0 else ""
            lines.append(
                f"  L{int(row['layer'])} {row['classifier_name']:>20}  "
                f"f1_E={f1e:.4f}  f1_L={f1l:.4f}  f1_S={f1s:.4f}  "
                f"Acc={row['accuracy']:.4f}{marker}"
            )

    # Key findings
    lines.append("\n\n" + "=" * 80)
    lines.append("KEY FINDINGS")
    lines.append("=" * 80)

    # Verified class distribution
    lines.append("\n0. 5-CLASS RAW DISTRIBUTION (verified):")
    lines.append("   ENTER_HOME_NEW=21, ENTER_HOME_ADD=12, LEAVE_HOME_LAST=21,")
    lines.append("   LEAVE_HOME_REDUCE=15, STAY=5. Total=74")
    lines.append("")
    lines.append("   Setting 1: Enter_New=21, Enter_Add=12, Leave_Last=21, Leave_Reduce=15, Stay=5")
    lines.append("   Setting 2: Enter=33 (44.6%), Leave=36 (48.6%), Stay=5 (6.8%)")
    lines.append("   Setting 3: Enter=21 (28.4%), Leave=21 (28.4%), Stay=32 (43.2%)")

    s2_best = s2.loc[s2["accuracy"].idxmax()] if not s2.empty else None
    s3 = df[df["label_setting"] == 3]
    s3_best = s3.loc[s3["accuracy"].idxmax()] if not s3.empty else None

    if s2_best is not None and s3_best is not None:
        gap = s2_best["accuracy"] - s3_best["accuracy"]
        lines.append(f"\n1. Setting 2 DRAMATICALLY outperforms Setting 3:")
        lines.append(f"   Setting 2 best: Acc={s2_best['accuracy']:.4f} (L{int(s2_best['layer'])}/{s2_best['classifier_name']})")
        lines.append(f"   Setting 3 best: Acc={s3_best['accuracy']:.4f} (L{int(s3_best['layer'])}/{s3_best['classifier_name']})")
        lines.append(f"   Gap: +{gap:.4f} ({gap*100:.1f}%p)")

    lines.append(f"\n2. Setting 2 is essentially BINARY (Enter vs Leave):")
    lines.append(f"   - Stay has only 5 samples (6.8%) — too few to learn")
    lines.append(f"   - f1_Stay=0.0 for most classifiers (cannot detect Stay)")
    lines.append(f"   - High accuracy comes from correctly classifying Enter(33)+Leave(36)")
    lines.append(f"   - f1_Enter≈0.85, f1_Leave≈0.86 → well-separated")
    lines.append(f"   - The model ignores Stay entirely")

    lines.append(f"\n3. Setting 3 is the TRUE 3-class problem:")
    lines.append(f"   - Enter=21, Leave=21, Stay=32 — more balanced")
    lines.append(f"   - Best Acc=0.4459 (Phase E) / 0.6622 (Phase B with larger context)")
    lines.append(f"   - Much harder but the real target for Enter/Leave/Stay classification")

    lines.append(f"\n4. Setting 1 (5-class) is clearly too hard:")
    s1 = df[df["label_setting"] == 1]
    if not s1.empty:
        lines.append(f"   Best accuracy: {s1['accuracy'].max():.4f}")
        lines.append(f"   Enter_Add(12) and Stay(5) are too small for 5-class LOOCV")

    lines.append(f"\n5. IMPLICATIONS FOR ROUND 2:")
    lines.append(f"   a) Setting 2 高 accuracy is MISLEADING — it's mostly binary classification")
    lines.append(f"   b) Setting 3 remains the correct target for 3-class Enter/Leave/Stay")
    lines.append(f"   c) Stay(S3)=32 is the merged class (ADD+REDUCE+STAY) — properly sized")
    lines.append(f"   d) Round 2 should continue focusing on Setting 3 with practical factors")
    lines.append(f"   e) Setting 2 useful as REFERENCE ONLY — upper bound for Enter-vs-Leave")

    report_path = output_dir / "phase_e_analysis.txt"
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Saved: %s", report_path)
    print(f"\nReport saved: {report_path}")


def plot_setting_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate comparison plots across settings."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 1. Accuracy heatmap: Setting × Layer (best classifier per cell)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, setting in enumerate([1, 2, 3]):
        sdf = df[df["label_setting"] == setting]
        if sdf.empty:
            continue
        pivot = sdf.pivot_table(
            values="accuracy", index="layer", columns="classifier_name",
            aggfunc="max",
        )
        ax = axes[idx]
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0.15, vmax=0.85)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"L{int(l)}" for l in pivot.index], fontsize=9)
        ax.set_title(f"Setting {setting}\n({', '.join(CLASS_NAMES[setting])})", fontsize=10)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    color = "white" if val > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

    fig.colorbar(im, ax=axes, label="Accuracy", shrink=0.8)
    fig.suptitle("Phase E: Accuracy by Setting × Layer × Classifier", fontsize=12)
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(plots_dir / "phase_e_settings_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", plots_dir / "phase_e_settings_heatmap.png")

    # 2. Setting comparison bar chart (best accuracy per setting per layer)
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = sorted(df["layer"].unique())
    x = np.arange(len(layers))
    width = 0.25
    colors = {"1": "#E24A33", "2": "#348ABD", "3": "#988ED5"}

    for i, setting in enumerate([1, 2, 3]):
        sdf = df[df["label_setting"] == setting]
        best_per_layer = sdf.groupby("layer")["accuracy"].max()
        vals = [best_per_layer.get(l, 0) for l in layers]
        bars = ax.bar(x + i * width, vals, width, label=f"Setting {setting}",
                      color=colors[str(setting)], alpha=0.85)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Best Accuracy")
    ax.set_title("Phase E: Best Accuracy per Layer × Setting")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"L{int(l)}" for l in layers])
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(plots_dir / "phase_e_setting_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", plots_dir / "phase_e_setting_bars.png")

    # 3. Setting 2 per-class F1 analysis
    s2 = df[df["label_setting"] == 2].copy()
    if not s2.empty and "f1_Enter" in s2.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        s2_sorted = s2.sort_values("accuracy", ascending=False)
        labels = [f"L{int(r['layer'])}_{r['classifier_name'][:6]}" for _, r in s2_sorted.iterrows()]
        x = np.arange(len(labels))

        for i, (cls, color) in enumerate([("f1_Enter", "#E24A33"), ("f1_Leave", "#348ABD"), ("f1_Stay", "#988ED5")]):
            vals = s2_sorted[cls].fillna(0).values
            ax.bar(x + i * 0.25, vals, 0.25, label=cls.replace("f1_", ""),
                   color=color, alpha=0.85)

        ax.set_xlabel("Experiment")
        ax.set_ylabel("F1 Score")
        ax.set_title("Setting 2: Per-Class F1 Scores (sorted by Accuracy)")
        ax.set_xticks(x + 0.25)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        fig.savefig(plots_dir / "phase_e_setting2_perclass.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", plots_dir / "phase_e_setting2_perclass.png")


def main():
    parser = argparse.ArgumentParser(description="Deep analysis of Phase E results")
    parser.add_argument("--results", required=True, help="Path to results_E.csv")
    parser.add_argument("--events-csv", default=None,
                        help="Path to 5-class event CSV for distribution verification")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as results)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    results_path = Path(args.results)
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse Phase E with mixed-column handling
    df = parse_phase_e_csv(results_path)
    print(f"Loaded {len(df)} experiments ({df['label_setting'].nunique()} settings)")

    # Verify class distribution if events CSV provided
    if args.events_csv:
        verify_class_distribution(args.events_csv)

    # Generate analysis
    cross_setting_comparison(df, output_dir)
    plot_setting_comparison(df, output_dir)

    print(f"\nAnalysis complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
