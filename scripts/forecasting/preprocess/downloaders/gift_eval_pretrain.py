"""GiftEvalPretrain downloader: HuggingFace snapshot → Arrow DatasetDict.

Stage 1: Download Salesforce/GiftEvalPretrain via snapshot_download
         → {root}/gift_eval_pretrain_raw/

Stage 2: Load from snapshot and save as HF Arrow DatasetDict (train split)
         → {root}/chronos_datasets/gift_eval_pretrain/

The Stage 2 output is training-ready and consumed by LazyHFTaskSource
in the same way as chronos_train corpora (DatasetDict with a "train" split).

Loading strategies (tried in order during Stage 2):
  1. load_from_disk       — if snapshot is already in HF Arrow format
  2. load_dataset(parquet) — if snapshot contains .parquet files
  3. load_dataset(HF_REPO) — direct HF load (fallback, no snapshot required)
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.forecasting.preprocess.configs import DatasetSpec

logger = logging.getLogger(__name__)

HF_REPO = "Salesforce/GiftEvalPretrain"


def _raw_dir(root: Path) -> Path:
    return root / "gift_eval_pretrain_raw"


def _out_dir(root: Path) -> Path:
    return root / "chronos_datasets" / "gift_eval_pretrain"


def _is_downloaded(raw_dir: Path) -> bool:
    if not raw_dir.exists():
        return False
    return any(raw_dir.rglob("*"))


def _check_arrow_ready(out_dir: Path) -> bool:
    return (out_dir / "dataset_dict.json").exists() and (out_dir / "train").exists()


def _fmt_size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ["B", "KB", "MB", "GB"]:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


# ---------------------------------------------------------------------------
# Stage 1: Download
# ---------------------------------------------------------------------------

def download(
    spec: "DatasetSpec",
    root: Path,
    force: bool = False,
    resume: bool = False,
    retries: int = 3,
    dry_run: bool = False,
    **kwargs,
) -> bool:
    """Download GiftEvalPretrain snapshot from HuggingFace."""
    raw_dir = _raw_dir(root)

    if _is_downloaded(raw_dir) and not force:
        logger.info(f"  [Stage 1 SKIP] gift-eval-pretrain — snapshot exists at {raw_dir}")
        return True

    if dry_run:
        logger.info(f"  [Stage 1 DRY RUN] Would download {HF_REPO} → {raw_dir}")
        return True

    logger.info(f"  [Stage 1] gift-eval-pretrain — downloading {HF_REPO} → {raw_dir}")
    logger.info("    Large dataset — this may take a while …")
    raw_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        start = time.time()
        ok = _snapshot_download(raw_dir)
        elapsed = time.time() - start
        if ok:
            logger.info(f"    OK: {raw_dir} ({elapsed:.0f}s)")
            return True
        logger.warning(f"    Attempt {attempt} FAILED ({elapsed:.1f}s)")
        if attempt == 1:
            logger.info("    Trying huggingface-cli fallback …")
            if _cli_download(raw_dir):
                return True
        if attempt < retries:
            time.sleep(5.0)

    logger.error("  [Stage 1 FAILED] gift-eval-pretrain — all attempts exhausted")
    return False


def _snapshot_download(raw_dir: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(HF_REPO, repo_type="dataset", local_dir=str(raw_dir))
        return True
    except ImportError:
        logger.error("    huggingface_hub not installed")
        return False
    except Exception as e:
        logger.warning(f"    snapshot_download failed: {e}")
        return False


def _cli_download(raw_dir: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "huggingface-cli", "download",
                HF_REPO,
                "--repo-type=dataset",
                f"--local-dir={raw_dir}",
            ],
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if result.returncode == 0:
            return True
        logger.warning(f"    CLI failed: {result.stderr[:300]}")
        return False
    except FileNotFoundError:
        logger.warning("    huggingface-cli not found")
        return False
    except Exception as e:
        logger.warning(f"    CLI error: {e}")
        return False


# ---------------------------------------------------------------------------
# Stage 2: Convert → Arrow DatasetDict
# ---------------------------------------------------------------------------

def convert(
    spec: "DatasetSpec",
    root: Path,
    num_proc: int = 1,
    max_series: int | None = None,
    seed: int = 42,
    force: bool = False,
    dry_run: bool = False,
    **kwargs,
) -> bool:
    """Convert GiftEvalPretrain snapshot to HF Arrow DatasetDict.

    Output at {root}/chronos_datasets/gift_eval_pretrain/ is a DatasetDict
    with a single "train" split, directly consumable by LazyHFTaskSource.
    """
    import datasets as hf_datasets

    raw_dir = _raw_dir(root)
    out_dir = _out_dir(root)

    if _check_arrow_ready(out_dir) and not force:
        logger.info(f"  [Stage 2 SKIP] gift-eval-pretrain — Arrow exists at {out_dir}")
        return True

    if dry_run:
        logger.info(f"  [Stage 2 DRY RUN] Would convert {raw_dir} → {out_dir}")
        return True

    logger.info(f"  [Stage 2] gift-eval-pretrain — converting → {out_dir}")
    start = time.time()

    ds = _load_from_raw(raw_dir, num_proc)
    if ds is None:
        logger.error(
            "  [Stage 2 FAILED] gift-eval-pretrain — could not load from snapshot.\n"
            f"    Run Stage 1 first, or verify snapshot integrity at {raw_dir}."
        )
        return False

    logger.info(f"    Loaded {len(ds):,} rows, columns: {ds.column_names}")

    if max_series and len(ds) > max_series:
        import numpy as np
        rng = np.random.default_rng(seed)
        indices = sorted(rng.permutation(len(ds))[:max_series].tolist())
        ds = ds.select(indices)
        logger.info(f"    Subsampled to {len(ds):,} rows (max_series={max_series})")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_dict = hf_datasets.DatasetDict({"train": ds})
        ds_dict.save_to_disk(str(out_dir), num_proc=num_proc if num_proc > 1 else None)
        elapsed = time.time() - start
        size = _fmt_size(out_dir)
        logger.info(f"    OK: {len(ds):,} rows → {out_dir} ({size}, {elapsed:.1f}s)")
        return True
    except Exception as e:
        logger.error(f"  [Stage 2 FAILED] gift-eval-pretrain: {e}")
        return False


def _load_from_raw(raw_dir: Path, num_proc: int):
    """Detect and load the downloaded snapshot as a HF Dataset.

    Tries three strategies in order so the downloader works regardless of
    whether HF stored the dataset as Arrow, Parquet, or something else.
    """
    import datasets as hf_datasets

    # Strategy 1: already HF Arrow format (dataset_info.json / dataset_dict.json)
    if (raw_dir / "dataset_info.json").exists() or (raw_dir / "dataset_dict.json").exists():
        try:
            logger.info("    Detected HF Arrow format — using load_from_disk …")
            raw = hf_datasets.load_from_disk(str(raw_dir))
            if isinstance(raw, hf_datasets.DatasetDict):
                split = "train" if "train" in raw else next(iter(raw.keys()))
                logger.info(f"    Using split '{split}'")
                return raw[split]
            return raw
        except Exception as e:
            logger.warning(f"    load_from_disk failed: {e}")

    # Strategy 2: Parquet files in snapshot
    parquet_files = sorted(raw_dir.rglob("*.parquet"))
    if parquet_files:
        try:
            logger.info(f"    Detected {len(parquet_files)} Parquet file(s) — loading …")
            ds = hf_datasets.load_dataset(
                "parquet",
                data_files={"train": [str(f) for f in parquet_files]},
                split="train",
                num_proc=num_proc if num_proc > 1 else None,
            )
            return ds
        except Exception as e:
            logger.warning(f"    Parquet load failed: {e}")

    # Strategy 3: Direct HF load (works without a prior snapshot download)
    try:
        logger.info(f"    Falling back to direct HF streaming ({HF_REPO}) …")
        raw = hf_datasets.load_dataset(HF_REPO)
        if isinstance(raw, hf_datasets.DatasetDict):
            split = "train" if "train" in raw else next(iter(raw.keys()))
            logger.info(f"    Using split '{split}'")
            return raw[split]
        return raw
    except Exception as e:
        logger.warning(f"    Direct HF load failed: {e}")

    return None
