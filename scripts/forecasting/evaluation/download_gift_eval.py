#!/usr/bin/env python3
"""Download GIFT-Eval benchmark data for offline evaluation.

Downloads the Salesforce/GiftEval dataset from HuggingFace for use in
network-restricted environments.

Usage:
    python download_gift_eval.py \
        --output-dir /path/to/benchmarks/gift_eval/

    # Dry run (show what would be downloaded)
    python download_gift_eval.py \
        --output-dir /path/to/benchmarks/gift_eval/ \
        --dry-run

Setup:
    1. Install gift-eval library:
       git clone https://github.com/SalesforceAIResearch/gift-eval.git
       cd gift-eval && pip install -e .

    2. Download data (this script):
       python download_gift_eval.py --output-dir /path/to/benchmarks/gift_eval/

    3. Set environment variable:
       export GIFT_EVAL=/path/to/benchmarks/gift_eval/

    4. Run evaluation:
       python run_benchmark.py --model-path /path/to/model \
           --benchmarks gift_eval \
           --gift-eval-data /path/to/benchmarks/gift_eval/

References:
    - HuggingFace: https://huggingface.co/datasets/Salesforce/GiftEval
    - GitHub: https://github.com/SalesforceAIResearch/gift-eval
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

GIFT_EVAL_REPO = "Salesforce/GiftEval"


def download_via_snapshot(output_dir: Path) -> bool:
    """Download GIFT-Eval using huggingface_hub snapshot_download."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error(
            "huggingface_hub not installed. "
            "Install with: pip install huggingface_hub"
        )
        return False

    logger.info(f"  Downloading {GIFT_EVAL_REPO} → {output_dir}")
    logger.info("  This may take a while (dataset is several GB)...")

    try:
        local_path = snapshot_download(
            GIFT_EVAL_REPO,
            repo_type="dataset",
            local_dir=str(output_dir),
        )
        logger.info(f"  OK: Downloaded to {local_path}")
        return True
    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return False


def download_via_cli(output_dir: Path) -> bool:
    """Download using huggingface-cli (fallback method)."""
    import subprocess

    logger.info("  Trying huggingface-cli download...")
    try:
        result = subprocess.run(
            [
                "huggingface-cli", "download",
                GIFT_EVAL_REPO,
                "--repo-type=dataset",
                f"--local-dir={output_dir}",
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode == 0:
            logger.info(f"  OK: Downloaded to {output_dir}")
            return True
        else:
            logger.error(f"  CLI failed: {result.stderr}")
            return False
    except FileNotFoundError:
        logger.warning("  huggingface-cli not found")
        return False
    except Exception as e:
        logger.error(f"  CLI error: {e}")
        return False


def verify_download(output_dir: Path) -> dict:
    """Verify downloaded GIFT-Eval data."""
    stats = {"exists": output_dir.exists(), "files": 0, "size_mb": 0}

    if not output_dir.exists():
        return stats

    total_size = 0
    file_count = 0
    for f in output_dir.rglob("*"):
        if f.is_file():
            file_count += 1
            total_size += f.stat().st_size

    stats["files"] = file_count
    stats["size_mb"] = total_size / (1024 * 1024)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download GIFT-Eval benchmark data",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for GIFT-Eval data",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify existing download",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    logger.info(f"{'=' * 60}")
    logger.info(f"  GIFT-Eval Data Downloader")
    logger.info(f"  Source: {GIFT_EVAL_REPO}")
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

    if args.verify or args.dry_run:
        stats = verify_download(output_dir)
        if stats["exists"]:
            logger.info(
                f"  Current state: {stats['files']} files, "
                f"{stats['size_mb']:.1f} MB"
            )
        else:
            logger.info("  Not yet downloaded")

        if args.dry_run:
            logger.info(f"\n  DRY RUN — would download {GIFT_EVAL_REPO} to {output_dir}")
            return

    # Download
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    ok = download_via_snapshot(output_dir)
    if not ok:
        logger.info("  Trying fallback method (CLI)...")
        ok = download_via_cli(output_dir)

    elapsed = time.time() - start

    if ok:
        stats = verify_download(output_dir)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Download complete ({elapsed:.0f}s)")
        logger.info(f"  Files: {stats['files']} | Size: {stats['size_mb']:.1f} MB")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"")
        logger.info(f"  Next steps:")
        logger.info(f"    export GIFT_EVAL={output_dir}")
        logger.info(f"    python run_benchmark.py --benchmarks gift_eval \\")
        logger.info(f"        --gift-eval-data {output_dir} ...")
        logger.info(f"{'=' * 60}")
    else:
        logger.error(f"\n  Download failed after {elapsed:.0f}s")
        logger.error(f"  Check network/proxy settings or HF_TOKEN")
        sys.exit(1)


if __name__ == "__main__":
    main()
