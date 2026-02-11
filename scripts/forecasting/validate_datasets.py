# Copyright 2024 - Validation script for pre-downloaded Chronos training datasets
# Usage: python scripts/validate_datasets.py --data-root /group-volume/ts-dataset/chronos_datasets

import argparse
import sys
from pathlib import Path

import numpy as np


def validate_arrow_dataset(path: Path, max_samples: int = 100):
    """Validate a GluonTS arrow dataset."""
    from gluonts.dataset.common import FileDataset

    results = {
        "path": str(path),
        "format": None,
        "num_series_checked": 0,
        "total_series": None,
        "lengths": [],
        "has_nan": False,
        "nan_count": 0,
        "sample_stats": {},
        "errors": [],
    }

    try:
        # Check for arrow files
        arrow_files = list(path.glob("*.arrow"))
        parquet_files = list(path.glob("*.parquet"))
        json_files = list(path.glob("*.json"))
        metadata_file = path / "metadata.json"

        if arrow_files:
            results["format"] = f"arrow ({len(arrow_files)} files)"
        elif parquet_files:
            results["format"] = f"parquet ({len(parquet_files)} files)"
        elif json_files:
            results["format"] = f"json ({len(json_files)} files)"
        else:
            # Try listing contents
            contents = list(path.iterdir())
            results["format"] = f"unknown ({len(contents)} items: {[c.name for c in contents[:5]]})"

        if metadata_file.exists():
            import json

            with open(metadata_file) as f:
                meta = json.load(f)
            results["metadata"] = meta

        # Try loading as FileDataset
        ds = FileDataset(path=path, freq="h")

        lengths = []
        nan_total = 0
        value_min = float("inf")
        value_max = float("-inf")
        value_sum = 0.0
        value_count = 0

        for i, entry in enumerate(ds):
            target = entry["target"]
            if isinstance(target, np.ndarray):
                length = len(target)
            else:
                length = len(target)

            lengths.append(length)
            arr = np.array(target, dtype=np.float64)
            nan_count = int(np.isnan(arr).sum())
            nan_total += nan_count

            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                value_min = min(value_min, float(valid.min()))
                value_max = max(value_max, float(valid.max()))
                value_sum += float(valid.sum())
                value_count += len(valid)

            if i + 1 >= max_samples:
                break

        results["num_series_checked"] = len(lengths)
        results["lengths"] = lengths
        results["nan_count"] = nan_total
        results["has_nan"] = nan_total > 0
        results["sample_stats"] = {
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "mean_length": float(np.mean(lengths)) if lengths else 0,
            "value_range": [value_min, value_max],
            "mean_value": value_sum / value_count if value_count > 0 else 0,
        }

    except Exception as e:
        results["errors"].append(f"{type(e).__name__}: {str(e)}")

    return results


def count_total_series(path: Path):
    """Count total series (can be slow for large datasets)."""
    try:
        from gluonts.dataset.common import FileDataset

        ds = FileDataset(path=path, freq="h")
        count = sum(1 for _ in ds)
        return count
    except Exception as e:
        return f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Validate Chronos training datasets")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/group-volume/ts-dataset/chronos_datasets",
        help="Root directory of downloaded datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Specific datasets to validate (default: kernel_synth and tsmixup)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Max series to check per dataset",
    )
    parser.add_argument(
        "--count-all",
        action="store_true",
        help="Count total series (slow for large datasets)",
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="List all available datasets in data-root",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if not data_root.exists():
        print(f"ERROR: Data root does not exist: {data_root}")
        sys.exit(1)

    # List all datasets
    if args.list_all:
        print(f"\n{'='*70}")
        print(f"Available datasets in: {data_root}")
        print(f"{'='*70}")
        for d in sorted(data_root.iterdir()):
            if d.is_dir():
                size_info = ""
                try:
                    import subprocess

                    result = subprocess.run(
                        ["du", "-sh", str(d)],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        size_info = result.stdout.split()[0]
                except Exception:
                    pass
                print(f"  {d.name:<50} {size_info}")
        print()

    # Determine which datasets to validate
    if args.datasets:
        targets = args.datasets
    else:
        targets = ["training_corpus_kernel_synth_1m", "training_corpus_tsmixup_10m"]

    print(f"\n{'='*70}")
    print("DATASET VALIDATION REPORT")
    print(f"{'='*70}\n")

    for dataset_name in targets:
        dataset_path = data_root / dataset_name

        print(f"\n{'─'*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Path:    {dataset_path}")
        print(f"{'─'*60}")

        if not dataset_path.exists():
            print("  STATUS: NOT FOUND")
            continue

        # Run validation
        results = validate_arrow_dataset(dataset_path, max_samples=args.max_samples)

        if results["errors"]:
            print(f"  STATUS: ERROR")
            for err in results["errors"]:
                print(f"  Error: {err}")
        else:
            print(f"  STATUS: OK")

        print(f"  Format: {results['format']}")

        if "metadata" in results:
            print(f"  Metadata: {results['metadata']}")

        print(f"  Series checked: {results['num_series_checked']}")

        if results["sample_stats"]:
            stats = results["sample_stats"]
            print(f"  Length range: [{stats['min_length']}, {stats['max_length']}]")
            print(f"  Mean length: {stats['mean_length']:.1f}")
            print(f"  Value range: [{stats['value_range'][0]:.4f}, {stats['value_range'][1]:.4f}]")
            print(f"  Mean value: {stats['mean_value']:.4f}")
            print(f"  Has NaN: {results['has_nan']} (count: {results['nan_count']})")

        # Show length distribution
        if results["lengths"]:
            lengths = results["lengths"]
            print(f"\n  Length distribution (first {len(lengths)} series):")
            unique_lengths = sorted(set(lengths))
            if len(unique_lengths) <= 10:
                for l in unique_lengths:
                    count = lengths.count(l)
                    print(f"    length={l}: {count} series")
            else:
                percentiles = [0, 25, 50, 75, 100]
                for p in percentiles:
                    val = np.percentile(lengths, p)
                    print(f"    {p}th percentile: {val:.0f}")

        # Count total series if requested
        if args.count_all:
            print(f"\n  Counting total series (this may take a while)...")
            total = count_total_series(dataset_path)
            print(f"  Total series: {total}")

    # Check Chronos-2 specific compatibility
    print(f"\n\n{'='*70}")
    print("CHRONOS-2 TRAINING COMPATIBILITY CHECK")
    print(f"{'='*70}\n")

    print("Training config expects:")
    print("  - training_data_paths: list of paths to arrow datasets")
    print("  - Data format: GluonTS FileDataset (arrow)")
    print("  - Fields: 'start' (datetime), 'target' (1-D array)")
    print()
    print("Recommended config paths for GPU server:")
    print("  training_data_paths:")
    print(f'  - "{data_root}/training_corpus_tsmixup_10m"')
    print(f'  - "{data_root}/training_corpus_kernel_synth_1m"')
    print("  probability:")
    print("  - 0.9")
    print("  - 0.1")


if __name__ == "__main__":
    main()
