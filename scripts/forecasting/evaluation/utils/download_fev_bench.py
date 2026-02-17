#!/usr/bin/env python3
"""Pre-download fev-bench data for offline evaluation.

Downloads fev-bench datasets from HuggingFace for use in network-restricted
environments. The fev library normally auto-downloads data on first use,
but this script pre-caches everything for fully offline operation.

Usage:
    # Pre-download all fev-bench data
    python download_fev_bench.py \
        --output-dir /path/to/benchmarks/fev_bench/

    # Verify existing download
    python download_fev_bench.py \
        --output-dir /path/to/benchmarks/fev_bench/ \
        --verify

    # Pre-cache via fev library (if fev is installed)
    python download_fev_bench.py \
        --output-dir /path/to/benchmarks/fev_bench/ \
        --precache-tasks

Setup:
    1. Install fev library:
       pip install fev

    2. Download data (this script):
       python download_fev_bench.py --output-dir /path/to/benchmarks/fev_bench/

    3. Run evaluation:
       python run_benchmark.py --model-path /path/to/model \
           --benchmarks fev_bench \
           --fev-data /path/to/benchmarks/fev_bench/

Data sources:
    - autogluon/fev_datasets (80+ configs, Parquet format)
    - autogluon/chronos_datasets (shared with Chronos benchmarks)

References:
    - PyPI: https://pypi.org/project/fev/
    - GitHub: https://github.com/autogluon/fev
    - Paper: arXiv:2509.26468
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

FEV_DATASETS_REPO = "autogluon/fev_datasets"


def download_fev_datasets(output_dir: Path) -> bool:
    """Download fev_datasets repo using huggingface_hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error(
            "huggingface_hub not installed. "
            "Install with: pip install huggingface_hub"
        )
        return False

    logger.info(f"  Downloading {FEV_DATASETS_REPO} → {output_dir}")
    logger.info("  This may take a while (80+ dataset configs)...")

    try:
        local_path = snapshot_download(
            FEV_DATASETS_REPO,
            repo_type="dataset",
            local_dir=str(output_dir),
        )
        logger.info(f"  OK: Downloaded to {local_path}")
        return True
    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return False


def precache_fev_tasks(output_dir: Path) -> bool:
    """Pre-cache fev benchmark tasks (triggers HF dataset downloads).

    This loads each fev task, which causes the fev library to download
    and cache the underlying datasets via HuggingFace.
    """
    try:
        import fev
    except ImportError:
        logger.warning(
            "fev not installed. Skipping task pre-caching.\n"
            "Install with: pip install fev"
        )
        return False

    logger.info("  Pre-caching fev benchmark tasks...")
    tasks_yaml_url = (
        "https://github.com/autogluon/fev/raw/refs/heads/main/"
        "benchmarks/fev_bench/tasks.yaml"
    )
    benchmark = fev.Benchmark.from_yaml(tasks_yaml_url)

    cached = 0
    failed = 0
    for task in benchmark.tasks:
        try:
            # Iterating windows triggers the dataset download
            for window in task.iter_windows(trust_remote_code=True):
                _ = window.get_input_data()
                break  # Only need to trigger download, not iterate all windows
            cached += 1
            logger.info(f"    Cached: {task.dataset_config}")
        except Exception as e:
            failed += 1
            logger.warning(f"    Failed: {task.dataset_config} — {e}")

    logger.info(f"  Pre-cached: {cached} | Failed: {failed}")
    return failed == 0


def verify_download(output_dir: Path) -> dict:
    """Verify downloaded fev-bench data."""
    stats = {"exists": output_dir.exists(), "files": 0, "size_mb": 0, "configs": 0}

    if not output_dir.exists():
        return stats

    total_size = 0
    file_count = 0
    parquet_count = 0
    for f in output_dir.rglob("*"):
        if f.is_file():
            file_count += 1
            total_size += f.stat().st_size
            if f.suffix == ".parquet":
                parquet_count += 1

    stats["files"] = file_count
    stats["size_mb"] = total_size / (1024 * 1024)
    stats["configs"] = parquet_count
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download fev-bench data for offline evaluation",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for fev-bench data",
    )
    parser.add_argument(
        "--precache-tasks", action="store_true",
        help="Also pre-cache fev tasks (requires fev library)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify existing download",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    logger.info(f"{'=' * 60}")
    logger.info(f"  fev-bench Data Downloader")
    logger.info(f"  Source: {FEV_DATASETS_REPO}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"{'=' * 60}")

    # Show environment
    for env_var in ["HTTPS_PROXY", "HTTP_PROXY", "HF_HOME", "HF_TOKEN"]:
        val = os.environ.get(env_var, "")
        if val:
            display = val[:20] + "..." if len(val) > 20 else val
            if "TOKEN" in env_var:
                display = val[:4] + "****"
            logger.info(f"  {env_var}={display}")

    if args.verify:
        stats = verify_download(output_dir)
        if stats["exists"]:
            logger.info(
                f"  Current state: {stats['files']} files "
                f"({stats['configs']} parquet), "
                f"{stats['size_mb']:.1f} MB"
            )
        else:
            logger.info("  Not yet downloaded")
        return

    if args.dry_run:
        logger.info(f"\n  DRY RUN — would download:")
        logger.info(f"    {FEV_DATASETS_REPO} → {output_dir}")
        if args.precache_tasks:
            logger.info(f"    + pre-cache all 100 fev benchmark tasks")
        return

    # Download
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    ok = download_fev_datasets(output_dir)

    if args.precache_tasks:
        precache_fev_tasks(output_dir)

    elapsed = time.time() - start

    if ok:
        stats = verify_download(output_dir)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Download complete ({elapsed:.0f}s)")
        logger.info(
            f"  Files: {stats['files']} ({stats['configs']} parquet) | "
            f"Size: {stats['size_mb']:.1f} MB"
        )
        logger.info(f"  Output: {output_dir}")
        logger.info(f"")
        logger.info(f"  Next steps:")
        logger.info(f"    python run_benchmark.py --benchmarks fev_bench \\")
        logger.info(f"        --fev-data {output_dir} ...")
        logger.info(f"{'=' * 60}")
    else:
        logger.error(f"\n  Download failed after {elapsed:.0f}s")
        logger.error(f"  Check network/proxy settings or HF_TOKEN")
        sys.exit(1)


if __name__ == "__main__":
    main()
