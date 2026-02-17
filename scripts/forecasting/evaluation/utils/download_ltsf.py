#!/usr/bin/env python3
"""Download LTSF benchmark CSV files for offline evaluation.

Downloads the 9 standard LTSF (Long-Term Series Forecasting) datasets used by
DLinear, PatchTST, iTransformer, and other long-term forecasting baselines.

Usage:
    # Download all LTSF datasets
    python download_ltsf.py \
        --output-dir /path/to/benchmarks/ltsf/

    # Download specific datasets only
    python download_ltsf.py \
        --output-dir /path/to/benchmarks/ltsf/ \
        --datasets ETTh1 ETTm1 Weather

    # Dry run (show what would be downloaded)
    python download_ltsf.py \
        --output-dir /path/to/benchmarks/ltsf/ \
        --dry-run

    # Verify existing download
    python download_ltsf.py \
        --output-dir /path/to/benchmarks/ltsf/ \
        --verify

Data sources:
    - ETTh1, ETTh2, ETTm1, ETTm2: ETDataset GitHub repo
    - Weather, Traffic, Electricity, Exchange, ILI:
      Autoformer dataset collection (Google Drive)

References:
    - DLinear paper: arXiv:2205.13504 (AAAI 2023)
    - ETDataset: https://github.com/zhouhaoyi/ETDataset
    - Autoformer: https://github.com/thuml/Autoformer
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

# ETT datasets from GitHub raw URLs (most reliable)
ETT_BASE_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
)

# Dataset download URLs — primary sources
LTSF_DOWNLOAD_URLS = {
    "ETTh1": f"{ETT_BASE_URL}/ETTh1.csv",
    "ETTh2": f"{ETT_BASE_URL}/ETTh2.csv",
    "ETTm1": f"{ETT_BASE_URL}/ETTm1.csv",
    "ETTm2": f"{ETT_BASE_URL}/ETTm2.csv",
    # Non-ETT datasets: Autoformer/PatchTST/iTransformer shared collection
    # These are typically distributed via Google Drive by the Autoformer team
    # Fallback: users can manually download from the Autoformer repo
    "Weather": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/weather.csv",
    "Exchange": None,      # Requires manual download
    "Traffic": None,       # Requires manual download (large, ~160MB)
    "Electricity": None,   # Requires manual download (~120MB)
    "ILI": None,           # Requires manual download (CDC ILI data)
}

# Alternative HuggingFace-hosted URLs (community uploads)
LTSF_HF_FALLBACK = {
    "Weather": "thuml/weather",
    "Exchange": "thuml/exchange_rate",
    "Traffic": "thuml/traffic",
    "Electricity": "thuml/electricity",
    "ILI": "thuml/illness",
}

# Expected properties for verification
LTSF_EXPECTED = {
    "ETTh1": {"rows": 17420, "cols": 8},
    "ETTh2": {"rows": 17420, "cols": 8},
    "ETTm1": {"rows": 69680, "cols": 8},
    "ETTm2": {"rows": 69680, "cols": 8},
    "Weather": {"rows": 52696, "cols": 22},
    "Traffic": {"rows": 17544, "cols": 863},
    "Electricity": {"rows": 26304, "cols": 322},
    "Exchange": {"rows": 7588, "cols": 9},
    "ILI": {"rows": 966, "cols": 8},
}

ALL_DATASETS = list(LTSF_DOWNLOAD_URLS.keys())


def download_csv(url: str, save_path: Path, overwrite: bool = False) -> bool:
    """Download a CSV file from a URL."""
    if save_path.exists() and not overwrite:
        logger.info(f"  SKIP (exists): {save_path.name}")
        return True

    try:
        import urllib.request
        import urllib.error

        logger.info(f"  Downloading: {url}")
        start = time.time()

        # Set up proxy from environment
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        if proxy:
            proxy_handler = urllib.request.ProxyHandler({"https": proxy, "http": proxy})
            opener = urllib.request.build_opener(proxy_handler)
        else:
            opener = urllib.request.build_opener()

        req = urllib.request.Request(url, headers={"User-Agent": "tsm-trainer/1.0"})
        with opener.open(req, timeout=300) as response:
            data = response.read()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(data)

        elapsed = time.time() - start
        size_mb = len(data) / (1024 * 1024)
        logger.info(f"  OK: {save_path.name} — {size_mb:.1f} MB ({elapsed:.1f}s)")
        return True

    except Exception as e:
        logger.error(f"  FAILED: {save_path.name} — {e}")
        return False


def download_from_hf(dataset_id: str, save_path: Path) -> bool:
    """Download a dataset from HuggingFace and save as CSV (fallback)."""
    try:
        import datasets
    except ImportError:
        logger.warning("  datasets library not installed. Cannot use HF fallback.")
        return False

    try:
        logger.info(f"  HF fallback: {dataset_id}")
        start = time.time()

        ds = datasets.load_dataset(dataset_id, split="train")
        df = ds.to_pandas()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

        elapsed = time.time() - start
        logger.info(
            f"  OK (HF): {save_path.name} — "
            f"{len(df)} rows, {len(df.columns)} cols ({elapsed:.1f}s)"
        )
        return True
    except Exception as e:
        logger.error(f"  HF FAILED: {dataset_id} — {e}")
        return False


def verify_csv(csv_path: Path, dataset_name: str) -> dict:
    """Verify a downloaded LTSF CSV file."""
    result = {"exists": csv_path.exists(), "valid": False}

    if not csv_path.exists():
        return result

    try:
        import pandas as pd

        df = pd.read_csv(csv_path, nrows=5)
        full_rows = sum(1 for _ in open(csv_path)) - 1  # subtract header

        result["rows"] = full_rows
        result["cols"] = len(df.columns)
        result["size_mb"] = csv_path.stat().st_size / (1024 * 1024)
        result["columns"] = list(df.columns)

        # Check against expected
        expected = LTSF_EXPECTED.get(dataset_name, {})
        if expected:
            result["expected_rows"] = expected["rows"]
            result["expected_cols"] = expected["cols"]
            result["valid"] = (
                abs(full_rows - expected["rows"]) < 10
                and len(df.columns) == expected["cols"]
            )
        else:
            result["valid"] = full_rows > 0 and len(df.columns) > 1

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Download LTSF benchmark CSV files",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for LTSF CSV files",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        choices=ALL_DATASETS,
        help=f"Specific datasets to download (default: all)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download and overwrite existing files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify existing downloads",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    datasets = args.datasets or ALL_DATASETS

    logger.info(f"{'=' * 60}")
    logger.info(f"  LTSF Benchmark Data Downloader")
    logger.info(f"  Datasets: {len(datasets)}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"{'=' * 60}")

    # Show environment
    for env_var in ["HTTPS_PROXY", "HTTP_PROXY"]:
        val = os.environ.get(env_var, "")
        if val:
            display = val[:30] + "..." if len(val) > 30 else val
            logger.info(f"  {env_var}={display}")

    # Verify mode
    if args.verify:
        logger.info("")
        all_valid = True
        for name in datasets:
            csv_path = output_dir / f"{name}.csv"
            result = verify_csv(csv_path, name)
            if result["exists"] and result.get("valid"):
                logger.info(
                    f"  {name}: OK — {result['rows']} rows, "
                    f"{result['cols']} cols, {result.get('size_mb', 0):.1f} MB"
                )
            elif result["exists"]:
                logger.warning(
                    f"  {name}: MISMATCH — {result.get('rows', '?')} rows "
                    f"(expected {result.get('expected_rows', '?')}), "
                    f"{result.get('cols', '?')} cols "
                    f"(expected {result.get('expected_cols', '?')})"
                )
                all_valid = False
            else:
                logger.warning(f"  {name}: NOT FOUND")
                all_valid = False
        return

    # Dry run mode
    if args.dry_run:
        logger.info("")
        logger.info("DRY RUN — datasets to download:")
        for name in datasets:
            csv_path = output_dir / f"{name}.csv"
            status = "EXISTS" if csv_path.exists() else "MISSING"
            url = LTSF_DOWNLOAD_URLS.get(name)
            source = url if url else f"HF: {LTSF_HF_FALLBACK.get(name, 'manual')}"
            logger.info(f"  [{status}] {name} — {source}")
        return

    # Download
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    success = 0
    skipped = 0
    failed = 0
    manual_needed = []

    for name in datasets:
        csv_path = output_dir / f"{name}.csv"

        if csv_path.exists() and not args.overwrite:
            logger.info(f"  SKIP (exists): {name}")
            skipped += 1
            continue

        url = LTSF_DOWNLOAD_URLS.get(name)
        ok = False

        # Try direct URL first
        if url:
            ok = download_csv(url, csv_path, args.overwrite)

        # Try HuggingFace fallback
        if not ok and name in LTSF_HF_FALLBACK:
            ok = download_from_hf(LTSF_HF_FALLBACK[name], csv_path)

        if ok:
            success += 1
        else:
            failed += 1
            manual_needed.append(name)

    total_time = time.time() - start

    logger.info("")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Download complete ({total_time:.0f}s)")
    logger.info(
        f"  Downloaded: {success}  |  Skipped: {skipped}  |  Failed: {failed}"
    )
    logger.info(f"  Output: {output_dir}")

    if manual_needed:
        logger.info("")
        logger.info("  Manual download needed for:")
        for name in manual_needed:
            logger.info(f"    - {name}")
        logger.info("")
        logger.info("  These datasets are typically distributed via Google Drive.")
        logger.info("  Download from one of these sources:")
        logger.info("    - Autoformer: https://github.com/thuml/Autoformer")
        logger.info("    - PatchTST: https://github.com/yuqinie98/PatchTST")
        logger.info("    - iTransformer: https://github.com/thuml/iTransformer")
        logger.info(f"  Place CSV files in: {output_dir}/")

    logger.info("")
    logger.info("  Next steps:")
    logger.info(f"    python run_benchmark.py --benchmarks ltsf \\")
    logger.info(f"        --ltsf-data {output_dir} ...")
    logger.info(f"{'=' * 60}")

    if failed > 0 and not manual_needed:
        sys.exit(1)


if __name__ == "__main__":
    main()
