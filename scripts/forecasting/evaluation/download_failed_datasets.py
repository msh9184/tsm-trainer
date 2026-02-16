#!/usr/bin/env python3
"""Recover failed benchmark dataset downloads.

Handles two categories of download failures:

1. ETTm/ETTh (autogluon/chronos_datasets_extra):
   The custom loading script (chronos_datasets_extra.py) is incompatible
   with datasets >= 4.0.0. This script downloads ETT data from original
   sources and converts to Arrow format matching the evaluation pipeline.

2. Standard datasets (e.g., electricity_15min):
   Retries download from autogluon/chronos_datasets for transient failures.

Usage:
    # Check which datasets are missing
    python download_failed_datasets.py \
        --output-dir /path/to/benchmarks/chronos/ \
        --check-only

    # Download missing datasets
    python download_failed_datasets.py \
        --output-dir /path/to/benchmarks/chronos/

    # Force re-download specific datasets
    python download_failed_datasets.py \
        --output-dir /path/to/benchmarks/chronos/ \
        --datasets ETTm ETTh electricity_15min
"""

import argparse
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

import yaml

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent / "configs"

# ETT dataset definitions
ETT_DATASETS = {
    "ETTh": {
        "subsets": ["ETTh1", "ETTh2"],
        "description": "Electricity Transformer Temperature (hourly)",
    },
    "ETTm": {
        "subsets": ["ETTm1", "ETTm2"],
        "description": "Electricity Transformer Temperature (15-minute)",
    },
}

ETT_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def setup_proxy():
    """Configure urllib proxy from environment variables."""
    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy_url:
        proxy = urllib.request.ProxyHandler({
            "https": proxy_url,
            "http": proxy_url,
        })
        opener = urllib.request.build_opener(proxy)
        urllib.request.install_opener(opener)
        logger.info(f"  Proxy configured: {proxy_url[:30]}...")


def collect_expected_datasets(config_dir: Path) -> dict[str, dict]:
    """Collect all expected datasets from config files."""
    configs = {}
    for yaml_path in sorted(config_dir.glob("*.yaml")):
        with open(yaml_path) as f:
            entries = yaml.safe_load(f)
        if entries is None:
            continue
        for entry in entries:
            name = entry["name"]
            if name not in configs:
                configs[name] = entry
    return configs


def check_missing(output_dir: Path, config_dir: Path) -> list[str]:
    """Identify datasets that are expected but not present."""
    expected = collect_expected_datasets(config_dir)
    missing = []
    for name in sorted(expected):
        ds_path = output_dir / name
        if not ds_path.exists():
            missing.append(name)
    return missing


def _download_csv(url: str) -> "pd.DataFrame":
    """Download CSV from URL with proxy support and return DataFrame."""
    import pandas as pd
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urllib.request.urlretrieve(url, tmp_path)
        df = pd.read_csv(tmp_path)
        return df
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def download_ett_dataset(ds_name: str, output_dir: Path) -> bool:
    """Download ETT dataset from original source and convert to Arrow format.

    The chronos_datasets_extra repo only contains a loading script (no raw
    data files). This function downloads from the original ETT source and
    creates an Arrow dataset matching the format expected by the evaluator.

    Expected format:
    - N rows (one per ETT subset, e.g., ETTh1, ETTh2)
    - Columns: timestamp (Sequence[timestamp]), HUFL..OT (Sequence[float64])
    - Each variable column becomes a separate univariate series in evaluation
    """
    import datasets
    import pandas as pd

    config = ETT_DATASETS[ds_name]
    save_path = output_dir / ds_name

    logger.info(f"  Processing {ds_name} ({config['description']})...")

    # GitHub raw URLs for ETT data (original academic source)
    github_base = (
        "https://raw.githubusercontent.com/"
        "zhouhaoyi/ETDataset/main/ETT-small"
    )

    all_timestamps = []
    all_data = {col: [] for col in ETT_COLUMNS}

    for subset_name in config["subsets"]:
        df = None

        # Strategy 1: Direct URL download via urllib (proxy-aware)
        url = f"{github_base}/{subset_name}.csv"
        logger.info(f"    [{subset_name}] Trying: {url}")
        try:
            df = _download_csv(url)
            logger.info(f"    [{subset_name}] OK — {len(df)} rows")
        except Exception as e:
            logger.warning(f"    [{subset_name}] URL failed: {e}")

        # Strategy 2: Try pandas read_csv (may use different network stack)
        if df is None:
            try:
                import pandas as pd_direct
                df = pd_direct.read_csv(url)
                logger.info(f"    [{subset_name}] OK (pandas) — {len(df)} rows")
            except Exception as e:
                logger.warning(f"    [{subset_name}] pandas failed: {e}")

        # Strategy 3: Try huggingface_hub download
        if df is None:
            try:
                from huggingface_hub import hf_hub_download
                local_path = hf_hub_download(
                    repo_id="zhouhaoyi/ETDataset",
                    filename=f"ETT-small/{subset_name}.csv",
                    repo_type="dataset",
                )
                df = pd.read_csv(local_path)
                logger.info(f"    [{subset_name}] OK (hf_hub) — {len(df)} rows")
            except Exception as e:
                logger.warning(f"    [{subset_name}] hf_hub failed: {e}")

        if df is None:
            logger.error(
                f"    [{subset_name}] All download strategies failed. "
                f"Consider manual download: {url}"
            )
            return False

        # Parse and validate
        date_col = None
        for col_name in ["date", "Date", "datetime", "timestamp"]:
            if col_name in df.columns:
                date_col = col_name
                break

        if date_col is None:
            logger.error(f"    [{subset_name}] No date column found")
            return False

        timestamps = pd.to_datetime(df[date_col]).tolist()
        all_timestamps.append(timestamps)

        missing_cols = [c for c in ETT_COLUMNS if c not in df.columns]
        if missing_cols:
            logger.error(f"    [{subset_name}] Missing columns: {missing_cols}")
            return False

        for col in ETT_COLUMNS:
            all_data[col].append(df[col].tolist())

    # Build Arrow dataset
    logger.info(f"  Creating Arrow dataset for {ds_name}...")

    feature_dict = {
        "timestamp": datasets.Sequence(datasets.Value("timestamp[s]")),
    }
    for col in ETT_COLUMNS:
        feature_dict[col] = datasets.Sequence(datasets.Value("float64"))

    data_dict = {"timestamp": all_timestamps}
    for col in ETT_COLUMNS:
        data_dict[col] = all_data[col]

    features = datasets.Features(feature_dict)
    ds = datasets.Dataset.from_dict(data_dict, features=features)

    save_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(save_path))

    # Validate
    loaded = datasets.load_from_disk(str(save_path))
    loaded.set_format("numpy")
    n_fields = sum(
        1 for col in loaded.features
        if isinstance(loaded.features[col], datasets.Sequence)
        and col != "timestamp"
    )
    logger.info(
        f"  OK: {ds_name} — {len(loaded)} rows, "
        f"{n_fields} series fields → {save_path}"
    )
    return True


def retry_standard_download(
    name: str,
    hf_repo: str,
    output_dir: Path,
    cache_dir: str | None = None,
) -> bool:
    """Retry downloading a standard dataset from HuggingFace."""
    import datasets

    save_path = output_dir / name

    logger.info(f"  Retrying download: {hf_repo}/{name}...")
    start = time.time()

    try:
        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        ds = datasets.load_dataset(
            hf_repo, name,
            split="train",
            **kwargs,
        )

        save_path.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(save_path))

        elapsed = time.time() - start
        logger.info(
            f"  OK: {name} — {len(ds)} rows → {save_path} ({elapsed:.1f}s)"
        )
        return True

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"  FAILED: {name} — {e} ({elapsed:.1f}s)")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Recover failed benchmark dataset downloads",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for benchmark datasets",
    )
    parser.add_argument(
        "--config-dir", type=str, default=None,
        help="Directory containing benchmark config YAMLs",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="*", default=None,
        help="Specific dataset names to download (default: auto-detect missing)",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only check which datasets are missing",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config_dir = Path(args.config_dir) if args.config_dir else CONFIGS_DIR
    cache_dir = args.cache_dir or os.environ.get("HF_HOME")

    setup_proxy()

    # Determine which datasets to process
    expected = collect_expected_datasets(config_dir)

    if args.datasets:
        # User specified specific datasets
        target_names = args.datasets
    else:
        # Auto-detect missing
        target_names = check_missing(output_dir, config_dir)

    logger.info(f"{'=' * 60}")
    logger.info(f"  Failed Dataset Recovery")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(
        f"  Expected: {len(expected)} | "
        f"Downloaded: {len(expected) - len(target_names)} | "
        f"To recover: {len(target_names)}"
    )
    logger.info(f"{'=' * 60}")

    if not target_names:
        logger.info("  All datasets present!")
        return

    for name in target_names:
        is_ett = name in ETT_DATASETS
        tag = " (ETT — alternative download)" if is_ett else ""
        logger.info(f"  TARGET: {name}{tag}")

    if args.check_only:
        return

    # Download
    success = 0
    failed = 0
    total_start = time.time()

    for name in target_names:
        if name in ETT_DATASETS:
            ok = download_ett_dataset(name, output_dir)
        else:
            hf_repo = expected.get(name, {}).get(
                "hf_repo", "autogluon/chronos_datasets"
            )
            ok = retry_standard_download(name, hf_repo, output_dir, cache_dir)

        if ok:
            success += 1
        else:
            failed += 1

    total_time = time.time() - total_start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Recovery complete ({total_time:.0f}s)")
    logger.info(f"  Recovered: {success} | Still failed: {failed}")
    logger.info(f"{'=' * 60}")

    if failed > 0:
        logger.warning(
            "Some datasets still failed. Check error messages above.\n"
            "For ETT datasets, you may need to manually download CSV files\n"
            "and place them in the output directory."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
