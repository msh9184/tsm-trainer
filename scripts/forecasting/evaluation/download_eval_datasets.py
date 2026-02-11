#!/usr/bin/env python3
"""Download evaluation datasets for offline use.

This script pre-downloads HuggingFace datasets used in evaluation configs
so they are available on GPU servers that may have network/proxy restrictions.

Usage:
    # Download all lite-benchmark datasets
    python download_eval_datasets.py --config configs/lite-benchmark.yaml

    # Download all zero-shot datasets
    python download_eval_datasets.py --config configs/zero-shot.yaml

    # Download to custom cache directory
    python download_eval_datasets.py --config configs/lite-benchmark.yaml --cache-dir /data/hf_cache

    # With proxy support
    HTTPS_PROXY=http://proxy:8080 python download_eval_datasets.py --config configs/lite-benchmark.yaml

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


def download_dataset(hf_repo: str, name: str, cache_dir: str | None = None):
    """Download a single HuggingFace dataset."""
    import datasets

    kwargs = {"split": "train"}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    trust_remote_code = hf_repo == "autogluon/chronos_datasets_extra"

    logger.info(f"  Downloading: {hf_repo}/{name} ...")
    start = time.time()
    try:
        ds = datasets.load_dataset(
            hf_repo, name, trust_remote_code=trust_remote_code, **kwargs
        )
        elapsed = time.time() - start
        logger.info(
            f"  OK: {name} — {len(ds)} rows, {elapsed:.1f}s"
        )
        return True
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"  FAILED: {name} — {e} ({elapsed:.1f}s)")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download evaluation datasets for offline use",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to evaluation YAML config (e.g., configs/lite-benchmark.yaml)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Custom HuggingFace cache directory (default: HF_HOME or ~/.cache/huggingface)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to script directory
        config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(config_path) as f:
        configs = yaml.safe_load(f)

    cache_dir = args.cache_dir or os.environ.get("HF_HOME", None)

    logger.info(f"Downloading {len(configs)} datasets from {config_path.name}")
    if cache_dir:
        logger.info(f"Cache directory: {cache_dir}")

    # Show proxy info
    for env_var in ["HTTPS_PROXY", "HTTP_PROXY", "HF_HOME", "HF_TOKEN"]:
        val = os.environ.get(env_var, "")
        if val:
            display = val[:20] + "..." if len(val) > 20 else val
            if "TOKEN" in env_var:
                display = val[:4] + "****"
            logger.info(f"  {env_var}={display}")

    success = 0
    failed = 0
    total_start = time.time()

    for config in configs:
        name = config["name"]
        hf_repo = config["hf_repo"]
        if download_dataset(hf_repo, name, cache_dir):
            success += 1
        else:
            failed += 1

    total_time = time.time() - total_start
    logger.info("")
    logger.info(f"Download complete: {success} succeeded, {failed} failed ({total_time:.0f}s)")

    if failed > 0:
        logger.warning(
            "Some datasets failed to download. Check network/proxy settings "
            "or try setting HTTPS_PROXY and HF_TOKEN environment variables."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
