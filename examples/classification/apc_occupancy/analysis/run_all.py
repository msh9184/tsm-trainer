"""Main entry point: Run all APC occupancy data analysis modules.

Generates a comprehensive EDA report with publication-quality visualizations
in the specified output directory.

Usage (GPU server — 사내)::

    cd /group-volume/workspace/sunghwan.mun/mygit3/tsm-trainer
    python -m examples.classification.apc_occupancy.analysis.run_all \\
        --data-root /group-volume/workspace/haeri.kim/Time-Series/data/SmartThings/Samsung_QST_Data/enter_leave/ \\
        --output-dir results/apc_analysis/

Usage (WSL — local sample data for development)::

    cd /mnt/c/Users/User/workspace/samsung_research/sunghwan.mun/mygit3/tsm-trainer
    python -m examples.classification.apc_occupancy.analysis.run_all \\
        --data-root docs/sample_downstream_task/ \\
        --output-dir results/apc_analysis/

Options::

    --dpi 300          Output resolution (default: 300)
    --formats png      Output format(s): png, pdf, svg (default: png)
    --modules all      Which modules to run (comma-separated):
                       01_overview, 02_class_balance, 03_nan,
                       04_coverage, 05_sensor, 06_correlation, 07_gap
                       (default: all)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure non-interactive matplotlib backend before any pyplot import
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with timestamps."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="APC Occupancy Data Analysis & EDA Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="Root directory containing sensor + label CSV files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/apc_analysis"),
        help="Output directory for plots (default: results/apc_analysis/)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Output resolution (default: 300)",
    )
    parser.add_argument(
        "--formats", type=str, default="png",
        help="Output format(s), comma-separated (default: png)",
    )
    parser.add_argument(
        "--modules", type=str, default="all",
        help="Modules to run, comma-separated (default: all). "
             "Options: 01_overview, 02_class_balance, 03_nan, "
             "04_coverage, 05_sensor, 06_correlation, 07_gap",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    """Run all analysis modules."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("apc_analysis")

    # Import analysis modules
    from .config import AnalysisConfig
    from .data_loader import load_all_periods

    # Configure
    config = AnalysisConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        dpi=args.dpi,
        formats=args.formats.split(","),
    )

    # Apply matplotlib style
    plt.rcParams.update({
        "figure.dpi": config.dpi,
        "savefig.dpi": config.dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })

    logger.info("=" * 70)
    logger.info("APC Occupancy Data Analysis")
    logger.info("=" * 70)
    logger.info("Data root: %s", config.data_root)
    logger.info("Output dir: %s", config.output_dir)
    logger.info("DPI: %d, Formats: %s", config.dpi, config.formats)

    # Verify data root exists
    if not config.data_root.exists():
        logger.error("Data root does not exist: %s", config.data_root)
        logger.info(
            "Hint: Use --data-root docs/sample_downstream_task/ for local development"
        )
        sys.exit(1)

    # Load all periods
    t0 = time.time()
    periods = load_all_periods(config)
    load_time = time.time() - t0

    if not periods:
        logger.error("No periods loaded! Check data files in %s", config.data_root)
        sys.exit(1)

    logger.info(
        "Loaded %d periods in %.1fs: %s",
        len(periods), load_time,
        ", ".join(f"{p.name} ({len(p.sensor_df)} steps)" for p in periods),
    )

    # Determine which modules to run
    if args.modules == "all":
        modules_to_run = [
            "01_overview", "02_class_balance", "03_nan",
            "04_coverage", "05_sensor", "06_correlation", "07_gap",
        ]
    else:
        modules_to_run = [m.strip() for m in args.modules.split(",")]

    # Module registry
    module_map = {}

    if "01_overview" in modules_to_run:
        from .plot_01_overview import generate_overview
        module_map["01_overview"] = generate_overview

    if "02_class_balance" in modules_to_run:
        from .plot_02_class_balance import generate_class_balance
        module_map["02_class_balance"] = generate_class_balance

    if "03_nan" in modules_to_run:
        from .plot_03_nan_heatmap import generate_nan_analysis
        module_map["03_nan"] = generate_nan_analysis

    if "04_coverage" in modules_to_run:
        from .plot_04_time_coverage import generate_time_coverage
        module_map["04_coverage"] = generate_time_coverage

    if "05_sensor" in modules_to_run:
        from .plot_05_sensor_timeline import generate_sensor_timeline
        module_map["05_sensor"] = generate_sensor_timeline

    if "06_correlation" in modules_to_run:
        from .plot_06_correlation import generate_correlation
        module_map["06_correlation"] = generate_correlation

    if "07_gap" in modules_to_run:
        from .plot_07_gap_analysis import generate_gap_analysis
        module_map["07_gap"] = generate_gap_analysis

    # Run each module
    all_saved = []
    total_start = time.time()

    for module_name, func in module_map.items():
        logger.info("-" * 50)
        logger.info("Running: %s", module_name)
        t_start = time.time()

        try:
            saved = func(periods, config)
            elapsed = time.time() - t_start
            all_saved.extend(saved)
            logger.info(
                "Completed %s: %d plots in %.1fs",
                module_name, len(saved), elapsed,
            )
        except Exception:
            logger.exception("Error in %s", module_name)
            continue

    total_elapsed = time.time() - total_start

    # Summary
    logger.info("=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info("Total plots generated: %d", len(all_saved))
    logger.info("Total time: %.1fs", total_elapsed)
    logger.info("Output directory: %s", config.output_dir.resolve())
    logger.info("")
    logger.info("Generated plots:")
    for p in all_saved:
        logger.info("  %s", p)


if __name__ == "__main__":
    main()
