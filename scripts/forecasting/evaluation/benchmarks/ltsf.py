"""LTSF (Long-Term Series Forecasting) benchmark adapter.

Implements the standard LTSF benchmark protocol established by DLinear
(AAAI 2023) and followed by PatchTST, iTransformer, and other models.

Datasets: 9 (ETTh1, ETTh2, ETTm1, ETTm2, Weather, Traffic, Electricity,
          Exchange, ILI)
Horizons: {96, 192, 336, 720}  (ILI: {24, 36, 48, 60})
Protocol:
- z-normalization per-variable using train-set mean/std
- Stride-1 sliding window evaluation over test portion
- Point forecasts (median q=0.5 from quantile output)
- MSE and MAE computed on z-normalized values
- Flat mean over ALL (windows x timesteps x variables)

Setting: Multivariate "M" — all variables predicted from all variables.
Each variable is treated as an independent univariate channel for Chronos-2
(channel-independent, unique group_ids).

References:
    - DLinear: arXiv:2205.13504 (AAAI 2023)
    - PatchTST: arXiv:2211.14730 (ICLR 2023)
    - iTransformer: arXiv:2310.06625 (ICLR 2024)

NOTE: Chronos-2 does NOT evaluate on LTSF in the paper. This adapter
enables cross-comparison with PatchTST/iTransformer/DLinear baselines.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .base import BenchmarkAdapter

if TYPE_CHECKING:
    from engine.forecaster import BaseForecaster

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LTSF dataset specifications
# ---------------------------------------------------------------------------

LTSF_DATASETS = {
    "ETTh1": {"n_vars": 7, "freq": "H", "split_ratio": (0.6, 0.2, 0.2), "total": 14400},
    "ETTh2": {"n_vars": 7, "freq": "H", "split_ratio": (0.6, 0.2, 0.2), "total": 14400},
    "ETTm1": {"n_vars": 7, "freq": "15T", "split_ratio": (0.6, 0.2, 0.2), "total": 57600},
    "ETTm2": {"n_vars": 7, "freq": "15T", "split_ratio": (0.6, 0.2, 0.2), "total": 57600},
    "Weather": {"n_vars": 21, "freq": "10T", "split_ratio": (0.7, 0.1, 0.2), "total": 52696},
    "Traffic": {"n_vars": 862, "freq": "H", "split_ratio": (0.7, 0.1, 0.2), "total": 17544},
    "Electricity": {"n_vars": 321, "freq": "H", "split_ratio": (0.7, 0.1, 0.2), "total": 26304},
    "Exchange": {"n_vars": 8, "freq": "D", "split_ratio": (0.7, 0.1, 0.2), "total": 7588},
    "ILI": {"n_vars": 7, "freq": "W", "split_ratio": (0.7, 0.1, 0.2), "total": 966},
}

# Standard prediction horizons
LTSF_HORIZONS = [96, 192, 336, 720]
LTSF_ILI_HORIZONS = [24, 36, 48, 60]

# Default lookback window
LTSF_DEFAULT_SEQ_LEN = 96

# CSV file name mapping (some datasets use different names on disk)
LTSF_CSV_NAMES = {
    "ETTh1": "ETTh1.csv",
    "ETTh2": "ETTh2.csv",
    "ETTm1": "ETTm1.csv",
    "ETTm2": "ETTm2.csv",
    "Weather": "weather.csv",
    "Traffic": "traffic.csv",
    "Electricity": "electricity.csv",
    "Exchange": "exchange_rate.csv",
    "ILI": "national_illness.csv",
}


def _compute_split_indices(
    total_len: int,
    split_ratio: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Compute train/val/test split indices.

    Parameters
    ----------
    total_len : int
        Total number of timesteps in the dataset.
    split_ratio : tuple[float, float, float]
        (train_ratio, val_ratio, test_ratio).

    Returns
    -------
    tuple[int, int, int]
        (train_end, val_end, test_end) indices (exclusive boundaries).
    """
    train_ratio, val_ratio, _ = split_ratio
    num_train = int(total_len * train_ratio)
    num_test = int(total_len * (1 - train_ratio - val_ratio))
    num_val = total_len - num_train - num_test

    train_end = num_train
    val_end = num_train + num_val
    test_end = total_len

    return train_end, val_end, test_end


def _z_normalize(
    data: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-normalize data using train-set statistics (per-variable).

    Parameters
    ----------
    data : np.ndarray, shape (T, C)
        Full dataset values.
    train_end : int
        End index of training portion.

    Returns
    -------
    tuple
        (normalized_data, mean, std) — mean and std are shape (C,).
    """
    train_data = data[:train_end]
    mean = train_data.mean(axis=0)  # (C,)
    std = train_data.std(axis=0)    # (C,)

    # Prevent division by zero
    std = np.where(std == 0, 1.0, std)

    normalized = (data - mean) / std
    return normalized, mean, std


class LTSFAdapter(BenchmarkAdapter):
    """LTSF benchmark adapter.

    Evaluates a model using the standard LTSF protocol from DLinear/PatchTST.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing CSV files for the 9 LTSF datasets.
    datasets : list[str], optional
        Subset of datasets to evaluate. Default: all 9.
    horizons : list[int], optional
        Prediction horizons. Default: [96, 192, 336, 720].
    seq_len : int
        Lookback window length. Default: 96.
    batch_size : int
        Inference batch size (number of sliding windows per batch).
    stride : int
        Sliding window stride. Default: 1 (standard LTSF protocol).
    max_windows : int or None
        Maximum windows per (dataset, horizon) pair. None = no limit.
        Set to a smaller value (e.g., 100) for faster evaluation.
    """

    def __init__(
        self,
        data_dir: str | Path,
        datasets: list[str] | None = None,
        horizons: list[int] | None = None,
        seq_len: int = LTSF_DEFAULT_SEQ_LEN,
        batch_size: int = 32,
        stride: int = 1,
        max_windows: int | None = None,
    ):
        self._data_dir = Path(data_dir)
        self._datasets = datasets or list(LTSF_DATASETS.keys())
        self._horizons = horizons  # None = use defaults per dataset
        self._seq_len = seq_len
        self._batch_size = batch_size
        self._stride = stride
        self._max_windows = max_windows

    @property
    def name(self) -> str:
        return "ltsf"

    def _get_horizons(self, ds_name: str) -> list[int]:
        """Get prediction horizons for a dataset."""
        if self._horizons is not None:
            return self._horizons
        if ds_name == "ILI":
            return LTSF_ILI_HORIZONS
        return LTSF_HORIZONS

    def _load_csv(self, ds_name: str) -> pd.DataFrame:
        """Load a dataset CSV file.

        Parameters
        ----------
        ds_name : str
            Dataset name (e.g., "ETTh1", "Weather").

        Returns
        -------
        pd.DataFrame
            DataFrame with date column and feature columns.
        """
        csv_name = LTSF_CSV_NAMES.get(ds_name, f"{ds_name}.csv")
        csv_path = self._data_dir / csv_name

        if not csv_path.exists():
            # Try lowercase
            csv_path = self._data_dir / csv_name.lower()
        if not csv_path.exists():
            raise FileNotFoundError(
                f"LTSF dataset not found: {csv_path}. "
                f"Download with: python utils/download_ltsf.py --output-dir {self._data_dir}"
            )

        df = pd.read_csv(csv_path)

        # Drop the date column — we only need feature values
        if "date" in df.columns:
            df = df.drop(columns=["date"])
        elif df.columns[0].lower() in ("date", "datetime", "timestamp"):
            df = df.iloc[:, 1:]

        return df

    def _generate_windows(
        self,
        data: np.ndarray,
        test_start: int,
        test_end: int,
        seq_len: int,
        pred_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate sliding-window contexts and targets from test portion.

        Parameters
        ----------
        data : np.ndarray, shape (T, C)
            Z-normalized full dataset.
        test_start : int
            Start index of test portion (accounting for lookback).
        test_end : int
            End index of dataset.
        seq_len : int
            Lookback window length.
        pred_len : int
            Prediction horizon.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            contexts: shape (N, seq_len, C)
            targets: shape (N, pred_len, C)
        """
        contexts = []
        targets = []

        # The test portion starts at test_start, but we need seq_len before it
        # for the first window's context.
        start = test_start - seq_len
        end = test_end - pred_len

        for i in range(start, end + 1, self._stride):
            ctx_start = i
            ctx_end = i + seq_len
            tgt_start = ctx_end
            tgt_end = tgt_start + pred_len

            if tgt_end > test_end:
                break

            contexts.append(data[ctx_start:ctx_end])
            targets.append(data[tgt_start:tgt_end])

            if self._max_windows is not None and len(contexts) >= self._max_windows:
                break

        return np.array(contexts), np.array(targets)

    def load_tasks(self) -> list[dict]:
        """Load LTSF task configurations.

        Returns list of dicts with dataset name, horizon, and dataset info.
        """
        tasks = []
        for ds_name in self._datasets:
            if ds_name not in LTSF_DATASETS:
                logger.warning(f"Unknown LTSF dataset: {ds_name}, skipping")
                continue

            info = LTSF_DATASETS[ds_name]
            for horizon in self._get_horizons(ds_name):
                tasks.append({
                    "dataset": ds_name,
                    "horizon": horizon,
                    "n_vars": info["n_vars"],
                    "freq": info["freq"],
                    "seq_len": self._seq_len,
                })

        logger.info(f"LTSF: {len(tasks)} tasks loaded ({len(self._datasets)} datasets)")
        return tasks

    def evaluate(
        self,
        forecaster: BaseForecaster,
        **kwargs,
    ) -> pd.DataFrame:
        """Run LTSF benchmark evaluation.

        For each (dataset, horizon):
        1. Load CSV, compute splits, z-normalize
        2. Generate stride-1 sliding windows over test portion
        3. For each window, predict per-variable point forecasts (median)
        4. Compute MSE and MAE on z-normalized values

        Returns
        -------
        pd.DataFrame
            Per-(dataset, horizon) results with MSE and MAE columns.
        """
        import torch

        tasks = self.load_tasks()
        results = []
        n_total = len(tasks)

        logger.info(f"LTSF: evaluating {n_total} tasks (seq_len={self._seq_len})")

        for idx, task in enumerate(tasks, 1):
            ds_name = task["dataset"]
            pred_len = task["horizon"]
            n_vars = task["n_vars"]
            task_start = time.time()

            try:
                # Load and prepare data
                df = self._load_csv(ds_name)
                data = df.values.astype(np.float32)  # (T, C)
                total_len = len(data)

                info = LTSF_DATASETS[ds_name]
                train_end, val_end, test_end = _compute_split_indices(
                    total_len, info["split_ratio"]
                )

                # Z-normalize using train-set statistics
                data_norm, _, _ = _z_normalize(data, train_end)

                # Generate sliding windows from test portion
                # Test starts at val_end (the border between val and test)
                contexts, targets = self._generate_windows(
                    data_norm, val_end, test_end, self._seq_len, pred_len
                )
                n_windows = len(contexts)

                if n_windows == 0:
                    logger.warning(
                        f"  [{idx}/{n_total}] {ds_name}/H={pred_len}: "
                        f"No valid windows (test too short)"
                    )
                    continue

                # Predict: for Chronos-2, each variable is an independent series
                # Flatten (N, seq_len, C) → list of N*C 1D tensors
                all_preds = []

                for batch_start in range(0, n_windows, self._batch_size):
                    batch_end = min(batch_start + self._batch_size, n_windows)
                    batch_contexts = contexts[batch_start:batch_end]  # (B, seq_len, C)
                    n_batch = len(batch_contexts)

                    # Flatten: each variable in each window becomes a separate series
                    flat_contexts = []
                    for w in range(n_batch):
                        for v in range(n_vars):
                            flat_contexts.append(
                                torch.tensor(batch_contexts[w, :, v], dtype=torch.float32)
                            )

                    # Predict point forecasts (median)
                    point_preds = forecaster.predict_point(
                        flat_contexts,
                        prediction_length=pred_len,
                    )
                    # point_preds shape: (n_batch * n_vars, pred_len)

                    # Reshape back to (n_batch, pred_len, n_vars)
                    point_preds = point_preds.reshape(n_batch, n_vars, pred_len)
                    point_preds = np.transpose(point_preds, (0, 2, 1))  # (n_batch, pred_len, n_vars)

                    all_preds.append(point_preds)

                predictions = np.concatenate(all_preds, axis=0)  # (n_windows, pred_len, n_vars)

                # Compute metrics on z-normalized values
                # Flat mean over ALL elements: (windows * timesteps * variables)
                mse = float(np.mean((predictions - targets) ** 2))
                mae = float(np.mean(np.abs(predictions - targets)))

                elapsed = time.time() - task_start

                results.append({
                    "dataset": ds_name,
                    "horizon": pred_len,
                    "MSE": mse,
                    "MAE": mae,
                    "n_vars": n_vars,
                    "n_windows": n_windows,
                    "seq_len": self._seq_len,
                    "elapsed_s": round(elapsed, 1),
                })

                logger.info(
                    f"  [{idx}/{n_total}] {ds_name}/H={pred_len}: "
                    f"MSE={mse:.4f}, MAE={mae:.4f}, "
                    f"{n_windows} windows, {elapsed:.1f}s"
                )

            except Exception as e:
                elapsed = time.time() - task_start
                logger.warning(
                    f"  [{idx}/{n_total}] {ds_name}/H={pred_len}: FAILED — {e}"
                )
                results.append({
                    "dataset": ds_name,
                    "horizon": pred_len,
                    "n_vars": n_vars,
                    "seq_len": self._seq_len,
                    "elapsed_s": round(elapsed, 1),
                })

        return pd.DataFrame(results)

    def aggregate(self, results: pd.DataFrame) -> dict:
        """Aggregate LTSF results.

        Computes:
        - Overall average MSE and MAE
        - Per-horizon averages
        - Per-dataset averages

        Returns
        -------
        dict
            Summary metrics.
        """
        summary: dict = {
            "n_tasks": len(results),
            "seq_len": self._seq_len,
        }

        # Overall averages
        if "MSE" in results.columns:
            mse_vals = results["MSE"].dropna().values
            if len(mse_vals) > 0:
                summary["avg_mse"] = float(mse_vals.mean())

        if "MAE" in results.columns:
            mae_vals = results["MAE"].dropna().values
            if len(mae_vals) > 0:
                summary["avg_mae"] = float(mae_vals.mean())

        # Per-horizon averages
        if "horizon" in results.columns and "MSE" in results.columns:
            per_horizon = {}
            for h in sorted(results["horizon"].unique()):
                h_df = results[results["horizon"] == h]
                mse = h_df["MSE"].dropna().values
                mae = h_df["MAE"].dropna().values
                if len(mse) > 0:
                    per_horizon[int(h)] = {
                        "avg_mse": float(mse.mean()),
                        "avg_mae": float(mae.mean()),
                        "n_datasets": len(mse),
                    }
            if per_horizon:
                summary["per_horizon"] = per_horizon

        # Per-dataset averages
        if "dataset" in results.columns and "MSE" in results.columns:
            per_dataset = {}
            for ds in sorted(results["dataset"].unique()):
                ds_df = results[results["dataset"] == ds]
                mse = ds_df["MSE"].dropna().values
                mae = ds_df["MAE"].dropna().values
                if len(mse) > 0:
                    per_dataset[ds] = {
                        "avg_mse": float(mse.mean()),
                        "avg_mae": float(mae.mean()),
                    }
            if per_dataset:
                summary["per_dataset"] = per_dataset

        return summary
