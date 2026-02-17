#!/usr/bin/env python3
"""Download evaluation datasets and save to structured local directory.

Downloads HuggingFace datasets and saves them in Arrow format (save_to_disk)
for offline evaluation on network-restricted GPU servers.

Usage:
    # Download lite benchmark datasets
    python download_eval_datasets.py \
        --config configs/chronos-lite.yaml \
        --output-dir /path/to/benchmarks/chronos/

    # Download all Chronos Benchmark II (zero-shot) datasets
    python download_eval_datasets.py \
        --config configs/chronos-ii.yaml \
        --output-dir /path/to/benchmarks/chronos/

    # Download ALL Chronos benchmark datasets at once
    python download_eval_datasets.py \
        --config configs/chronos-full.yaml \
        --output-dir /path/to/benchmarks/chronos/

    # Dry run (show what would be downloaded)
    python download_eval_datasets.py \
        --config configs/chronos-ii.yaml \
        --output-dir /path/to/benchmarks/chronos/ \
        --dry-run

Environment variables:
    HF_HOME              HuggingFace cache directory
    HF_TOKEN             HuggingFace authentication token
    HTTPS_PROXY          HTTPS proxy URL
    HTTP_PROXY           HTTP proxy URL
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def download_and_save(
    hf_repo: str,
    name: str,
    output_dir: Path,
    cache_dir: str | None = None,
    overwrite: bool = False,
) -> bool:
    """Download a dataset from HuggingFace and save to local disk."""
    import datasets

    save_path = output_dir / name

    # Skip if already exists
    if save_path.exists() and not overwrite:
        logger.info(f"  SKIP (exists): {name} → {save_path}")
        return True

    trust_remote_code = hf_repo in ("autogluon/chronos_datasets_extra",)
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    logger.info(f"  Downloading: {hf_repo}/{name} ...")
    start = time.time()

    try:
        ds = datasets.load_dataset(
            hf_repo, name,
            split="train",
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        # Save to structured directory
        save_path.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(save_path))

        elapsed = time.time() - start
        logger.info(f"  OK: {name} — {len(ds)} rows → {save_path} ({elapsed:.1f}s)")
        return True

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"  FAILED: {name} — {e} ({elapsed:.1f}s)")
        return False


def collect_datasets(config_paths: list[Path]) -> list[dict]:
    """Collect unique datasets from multiple config files."""
    seen = set()
    datasets = []

    for config_path in config_paths:
        with open(config_path) as f:
            configs = yaml.safe_load(f)

        for config in configs:
            name = config["name"]
            if name not in seen:
                seen.add(name)
                datasets.append(config)
            else:
                logger.debug(f"  Skipping duplicate: {name}")

    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Download evaluation datasets to structured local directory",
    )
    parser.add_argument(
        "--config", type=str, nargs="+", required=True,
        help="Path(s) to evaluation YAML config(s)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for saved datasets (e.g., /path/to/benchmarks/chronos/)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="HuggingFace cache directory for intermediate downloads",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download and overwrite existing datasets",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without downloading",
    )
    args = parser.parse_args()

    # Resolve config paths
    config_paths = []
    for cfg in args.config:
        p = Path(cfg)
        if not p.exists():
            p = Path(__file__).parent / cfg
        if not p.exists():
            logger.error(f"Config not found: {cfg}")
            sys.exit(1)
        config_paths.append(p)

    output_dir = Path(args.output_dir)
    cache_dir = args.cache_dir or os.environ.get("HF_HOME", None)

    # Collect unique datasets
    all_datasets = collect_datasets(config_paths)

    logger.info(f"{'=' * 60}")
    logger.info(f"  Benchmark Dataset Downloader")
    logger.info(f"  Configs: {[p.name for p in config_paths]}")
    logger.info(f"  Unique datasets: {len(all_datasets)}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"{'=' * 60}")

    # Show environment
    for env_var in ["HTTPS_PROXY", "HTTP_PROXY", "HF_HOME", "HF_TOKEN"]:
        val = os.environ.get(env_var, "")
        if val:
            display = val[:20] + "..." if len(val) > 20 else val
            if "TOKEN" in env_var:
                display = val[:4] + "****"
            logger.info(f"  {env_var}={display}")

    if args.dry_run:
        logger.info("")
        logger.info("DRY RUN — datasets to download:")
        for ds in all_datasets:
            target = output_dir / ds["name"]
            status = "EXISTS" if target.exists() else "MISSING"
            hf_repo = ds.get('hf_repo', 'autogluon/chronos_datasets')
            logger.info(f"  [{status}] {hf_repo}/{ds['name']} → {target}")
        return

    # Download
    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    failed = 0
    total_start = time.time()

    for config in all_datasets:
        name = config["name"]
        hf_repo = config.get("hf_repo", "autogluon/chronos_datasets")

        save_path = output_dir / name
        if save_path.exists() and not args.overwrite:
            logger.info(f"  SKIP (exists): {name}")
            skipped += 1
            continue

        if download_and_save(hf_repo, name, output_dir, cache_dir, args.overwrite):
            success += 1
        else:
            failed += 1

    total_time = time.time() - total_start
    logger.info("")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Download complete ({total_time:.0f}s)")
    logger.info(f"  Downloaded: {success}  |  Skipped: {skipped}  |  Failed: {failed}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"{'=' * 60}")

    if failed > 0:
        logger.warning(
            "Some datasets failed. Check network/proxy settings "
            "or try setting HTTPS_PROXY and HF_TOKEN."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
