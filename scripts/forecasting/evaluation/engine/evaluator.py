"""Unified evaluation runner for time series forecasting.

Provides a single Evaluator class that:
1. Loads datasets from local paths or HuggingFace cache
2. Converts to GluonTS format for splitting
3. Generates forecasts using any BaseForecaster
4. Computes metrics (WQL, MASE, etc.) via GluonTS
5. Returns structured results as DataFrames

Supports both standalone evaluation and training-time validation.

Key design: All data loading supports local file paths to work in
network-restricted environments (GPU server offline mode).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd
import torch
import yaml

if TYPE_CHECKING:
    from .forecaster import BaseForecaster

logger = logging.getLogger(__name__)

# Default quantile levels for evaluation
EVAL_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _read_arrow_table(file_path: str | Path):
    """Read a single Arrow file, handling both IPC stream and file formats.

    datasets.save_to_disk() uses Arrow IPC *streaming* format (.arrow files),
    NOT the random-access file format. We try streaming first, then file format.
    """
    import pyarrow as pa
    from pyarrow import ipc

    f = str(file_path)

    # 1) Arrow IPC streaming format (datasets library default)
    try:
        reader = ipc.open_stream(f)
        return reader.read_all()
    except Exception:
        pass

    # 2) Arrow IPC file/random-access format
    try:
        return ipc.open_file(f).read_all()
    except Exception:
        pass

    raise ValueError(f"Cannot read as Arrow stream or file: {file_path}")


def _load_dataset_arrow_fallback(
    data_path: str | Path,
) -> tuple[list[dict], list[str]]:
    """Load dataset directly from Arrow/Parquet files (version-mismatch fallback).

    When datasets.load_from_disk() fails due to library version incompatibility
    (e.g., saved with datasets>=4.0 but loading with datasets<3.0), this reads
    the raw data files directly via PyArrow — bypassing metadata validation.

    Supports:
    - Arrow IPC streaming format (.arrow) — datasets library default
    - Arrow IPC file format (.arrow) — older datasets versions
    - Parquet format (.parquet) — newer datasets versions (3.x+)

    Parameters
    ----------
    data_path : str or Path
        Directory containing data files from datasets.save_to_disk().

    Returns
    -------
    tuple : (rows, series_fields)
        rows: list of dicts mapping column names to numpy arrays
        series_fields: column names containing time series data (List type)
    """
    import pyarrow as pa

    data_path = Path(data_path)
    arrow_files = sorted(data_path.glob("*.arrow"))
    parquet_files = sorted(data_path.glob("*.parquet"))

    if not arrow_files and not parquet_files:
        raise FileNotFoundError(
            f"No .arrow or .parquet files found in {data_path}.\n"
            f"  Contents: {[p.name for p in data_path.iterdir()]}"
        )

    # Read data files into PyArrow Table
    tables = []
    if arrow_files:
        for f in arrow_files:
            tables.append(_read_arrow_table(f))
    elif parquet_files:
        import pyarrow.parquet as pq
        for f in parquet_files:
            tables.append(pq.read_table(str(f)))

    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

    logger.info(f"  Arrow fallback: read {len(table)} rows from {len(tables)} file(s)")
    logger.info(f"  Schema: {table.schema}")

    # Identify series fields: List-type columns excluding 'timestamp'
    series_fields = []
    for field in table.schema:
        if field.name == "timestamp":
            continue
        if isinstance(field.type, (pa.ListType, pa.LargeListType)):
            series_fields.append(field.name)

    if not series_fields:
        raise ValueError(
            f"No series columns (List type) found in {data_path}.\n"
            f"  Schema: {table.schema}"
        )

    # Convert to row-oriented dicts with numpy arrays
    data = table.to_pydict()
    n_rows = len(table)

    rows = []
    for i in range(n_rows):
        row = {"timestamp": np.array(data["timestamp"][i])}
        for field in series_fields:
            row[field] = np.asarray(data[field][i], dtype=np.float32)
        rows.append(row)

    logger.info(f"  Arrow fallback OK: {n_rows} rows, fields={series_fields}")
    return rows, series_fields


def load_dataset_from_config(
    config: dict,
    datasets_root: str | Path | None = None,
) -> tuple:
    """Load a dataset from a benchmark config entry.

    Supports loading from:
    1. Local file path (if hf_repo is a local directory)
    2. HuggingFace cache (if hf_repo is a HuggingFace repo ID)

    Parameters
    ----------
    config : dict
        Dataset config with keys: name, hf_repo, offset, prediction_length, num_rolls.
        Optional: split (default "train"), data_dir (local path override).
    datasets_root : str or Path, optional
        Root directory for local datasets. If set, hf_repo is treated as
        a subdirectory under this root.

    Returns
    -------
    tuple : (gts_dataset, test_data, prediction_length, dataset_freq)
    """
    from gluonts.dataset.split import split as gluonts_split

    ds_name = config["name"]
    hf_repo = config["hf_repo"]
    offset = config["offset"]
    prediction_length = config["prediction_length"]
    num_rolls = config.get("num_rolls", 1)
    split_name = config.get("split", "train")

    # Resolve data path: local override → datasets_root → hf_repo as-is
    data_path = config.get("data_dir", None)
    if data_path is None and datasets_root is not None:
        candidate = Path(datasets_root) / ds_name
        if candidate.exists():
            data_path = str(candidate)

    # Load dataset
    local_only = config.get("local_only", datasets_root is not None)

    entries = None       # Iterable of row dicts (Dataset or list[dict])
    series_fields = None

    if data_path and Path(data_path).exists():
        logger.info(f"  Loading {ds_name} from local: {data_path}")

        # Try datasets.load_from_disk() first
        try:
            import datasets as hf_datasets
            ds = hf_datasets.load_from_disk(data_path)
            if isinstance(ds, hf_datasets.DatasetDict):
                ds = ds[split_name]
            ds.set_format("numpy")
            series_fields = [
                col for col in ds.features
                if isinstance(ds.features[col], hf_datasets.Sequence)
                and col != "timestamp"
            ]
            entries = ds
        except Exception as e:
            # Fallback: read Arrow IPC files directly (version-mismatch safe)
            logger.warning(
                f"  load_from_disk failed: {e}\n"
                f"  Falling back to direct Arrow file loading..."
            )
            rows, series_fields = _load_dataset_arrow_fallback(data_path)
            entries = rows

    elif local_only:
        searched = data_path or (Path(datasets_root) / ds_name if datasets_root else "N/A")
        raise FileNotFoundError(
            f"Dataset '{ds_name}' not found at local path: {searched}\n"
            f"  datasets_root: {datasets_root}\n"
            f"  Expected: {datasets_root}/{ds_name}/\n"
            f"  Hint: Run download_eval_datasets.py first, or check the dataset name."
        )
    else:
        # Load from HuggingFace (uses cache) — only when local_only is False
        import datasets as hf_datasets
        trust_remote_code = hf_repo == "autogluon/chronos_datasets_extra"
        logger.info(f"  Loading {ds_name} from HuggingFace: {hf_repo}")
        ds = hf_datasets.load_dataset(
            hf_repo, ds_name, split=split_name,
            trust_remote_code=trust_remote_code,
        )
        ds.set_format("numpy")
        series_fields = [
            col for col in ds.features
            if isinstance(ds.features[col], hf_datasets.Sequence)
            and col != "timestamp"
        ]
        entries = ds

    # Convert to GluonTS univariate format
    first_entry = entries[0]
    try:
        ts_index = pd.DatetimeIndex(first_entry["timestamp"])
        dataset_freq = ts_index.to_period()[0].freqstr
    except Exception:
        # Fallback: infer frequency from timestamp differences
        try:
            ts_index = pd.DatetimeIndex(first_entry["timestamp"])
            inferred = pd.infer_freq(ts_index)
            if inferred is None:
                # Use median difference as frequency estimate
                diffs = np.diff(ts_index.asi8)
                median_ns = np.median(diffs)
                hours = median_ns / 3.6e12
                if hours < 1:
                    dataset_freq = "T"
                elif hours < 24:
                    dataset_freq = "H"
                elif hours < 168:
                    dataset_freq = "D"
                elif hours < 720:
                    dataset_freq = "W"
                else:
                    dataset_freq = "M"
                logger.warning(f"  Could not detect exact frequency, estimated: {dataset_freq}")
            else:
                dataset_freq = inferred
        except Exception as freq_err:
            dataset_freq = "D"  # Safe default
            logger.warning(f"  Frequency detection failed ({freq_err}), defaulting to '{dataset_freq}'")

    gts_dataset = []
    for entry in entries:
        for field in series_fields:
            gts_dataset.append({
                "start": pd.Period(entry["timestamp"][0], freq=dataset_freq),
                "target": entry[field],
            })

    # Split for evaluation
    _, test_template = gluonts_split(gts_dataset, offset=offset)
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls)

    return gts_dataset, test_data, prediction_length, dataset_freq


def generate_quantile_forecasts(
    test_data_input: Iterable,
    forecaster: BaseForecaster,
    prediction_length: int,
    quantile_levels: list[float],
    batch_size: int = 32,
    **predict_kwargs,
) -> list:
    """Generate quantile forecasts and convert to GluonTS Forecast objects.

    Parameters
    ----------
    test_data_input : Iterable
        Test data input from GluonTS split.
    forecaster : BaseForecaster
        Model forecaster instance.
    prediction_length : int
        Forecast horizon.
    quantile_levels : list[float]
        Quantile levels to predict.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    list[QuantileForecast]
        GluonTS-compatible forecast objects.
    """
    from gluonts.itertools import batcher
    from gluonts.model.forecast import QuantileForecast

    forecast_outputs = []
    n_batches = 0
    n_series_processed = 0
    for batch in batcher(test_data_input, batch_size=batch_size):
        context = [torch.tensor(entry["target"]) for entry in batch]
        quantiles = forecaster.predict_quantiles(
            context,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            **predict_kwargs,
        )
        # predict_quantiles() returns (N, Q, H) per BaseForecaster contract.
        # QuantileForecast expects per-item (Q, H), so keep as-is.
        if quantiles.ndim != 3:
            raise ValueError(
                f"predict_quantiles() returned {quantiles.ndim}D array "
                f"(shape={quantiles.shape}), expected 3D (N, Q, H). "
                f"Check the forecaster's predict_quantiles() implementation."
            )
        forecast_outputs.append(quantiles)
        n_batches += 1
        n_series_processed += len(batch)
        if n_batches % 10 == 0:
            logger.info(f"  Forecasting progress: {n_series_processed} series done")

    forecast_outputs = np.concatenate(forecast_outputs)

    # Validate: forecast_outputs shape should be (total_N, Q, H)
    n_quantiles = len(quantile_levels)
    if forecast_outputs.shape[1] != n_quantiles:
        raise ValueError(
            f"Shape mismatch: forecast_outputs has shape {forecast_outputs.shape} "
            f"but expected axis 1 = {n_quantiles} (number of quantile levels). "
            f"Check that predict_quantiles() returns (N, Q, H)."
        )

    # Convert to GluonTS QuantileForecast objects
    # Each item has shape (Q, H) matching QuantileForecast expectation
    forecasts = []
    for item, ts in zip(forecast_outputs, test_data_input):
        forecast_start = ts["start"] + len(ts["target"])
        forecasts.append(
            QuantileForecast(
                forecast_arrays=item,
                forecast_keys=list(map(str, quantile_levels)),
                start_date=forecast_start,
            )
        )

    return forecasts


class Evaluator:
    """Unified evaluation runner for time series forecasting.

    Handles the full evaluation pipeline:
    dataset loading → forecast generation → metric computation → result aggregation.

    Parameters
    ----------
    forecaster : BaseForecaster
        Model to evaluate.
    quantile_levels : list[float], optional
        Quantile levels for evaluation.
    batch_size : int
        Inference batch size.
    datasets_root : str or Path, optional
        Root directory for local datasets.
    """

    def __init__(
        self,
        forecaster: BaseForecaster,
        quantile_levels: list[float] | None = None,
        batch_size: int = 32,
        datasets_root: str | Path | None = None,
    ):
        self.forecaster = forecaster
        self.quantile_levels = quantile_levels or EVAL_QUANTILES
        self.batch_size = batch_size
        self.datasets_root = datasets_root

    def evaluate_dataset(
        self,
        config: dict,
        **predict_kwargs,
    ) -> dict:
        """Evaluate a single dataset from config.

        Parameters
        ----------
        config : dict
            Dataset config entry with name, hf_repo, offset, etc.

        Returns
        -------
        dict
            {dataset, model, WQL, MASE, ...}
        """
        from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
        from gluonts.model.evaluation import evaluate_forecasts

        ds_name = config["name"]
        prediction_length = config["prediction_length"]

        logger.info(f"Loading dataset: {ds_name}")
        _, test_data, pred_len, _ = load_dataset_from_config(
            config, datasets_root=self.datasets_root
        )

        n_series = len(test_data.input)
        logger.info(f"Generating forecasts: {ds_name} ({n_series} series, H={pred_len})")

        forecasts = generate_quantile_forecasts(
            test_data.input,
            self.forecaster,
            prediction_length=pred_len,
            quantile_levels=self.quantile_levels,
            batch_size=self.batch_size,
            **predict_kwargs,
        )

        logger.info(f"Computing metrics: {ds_name}")
        metrics = (
            evaluate_forecasts(
                forecasts,
                test_data=test_data,
                metrics=[
                    MASE(),
                    MeanWeightedSumQuantileLoss(self.quantile_levels),
                ],
                batch_size=5000,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )

        result = {
            "dataset": ds_name,
            "model": self.forecaster.name,
            **metrics[0],
        }

        # Rename to standard names
        if "MASE[0.5]" in result:
            result["MASE"] = result.pop("MASE[0.5]")
        if "mean_weighted_sum_quantile_loss" in result:
            result["WQL"] = result.pop("mean_weighted_sum_quantile_loss")

        return result

    def evaluate_benchmark(
        self,
        config_path: str | Path,
        output_path: str | Path | None = None,
        **predict_kwargs,
    ) -> pd.DataFrame:
        """Evaluate all datasets in a benchmark config.

        Parameters
        ----------
        config_path : str or Path
            Path to benchmark YAML config.
        output_path : str or Path, optional
            Path to save results CSV. If None, results are not saved.

        Returns
        -------
        pd.DataFrame
            Per-dataset results with columns: dataset, model, WQL, MASE.
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            configs = yaml.safe_load(f)

        logger.info(
            f"Evaluating benchmark: {config_path.stem} "
            f"({len(configs)} datasets, model={self.forecaster.name})"
        )

        import time

        result_rows = []
        for idx, config in enumerate(configs, 1):
            ds_start = time.time()
            try:
                result = self.evaluate_dataset(config, **predict_kwargs)
                result_rows.append(result)
                wql = result.get("WQL", float("nan"))
                mase = result.get("MASE", float("nan"))
                elapsed = time.time() - ds_start
                logger.info(
                    f"  [{idx}/{len(configs)}] {config['name']}: "
                    f"WQL={wql:.4f}, MASE={mase:.4f} ({elapsed:.1f}s)"
                )
            except Exception as e:
                elapsed = time.time() - ds_start
                logger.warning(
                    f"  [{idx}/{len(configs)}] {config['name']}: "
                    f"FAILED ({elapsed:.1f}s) — {e}"
                )
                result_rows.append({
                    "dataset": config["name"],
                    "model": self.forecaster.name,
                    "WQL": float("nan"),
                    "MASE": float("nan"),
                })

        results_df = pd.DataFrame(result_rows).sort_values(by="dataset")

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved: {output_path}")

        return results_df

    def evaluate_quick(
        self,
        config_path: str | Path,
        **predict_kwargs,
    ) -> dict:
        """Quick evaluation returning summary metrics only.

        Used by training callbacks for fast validation.

        Parameters
        ----------
        config_path : str or Path
            Path to benchmark YAML config.

        Returns
        -------
        dict
            {avg_wql, avg_mase, per_dataset: {name: {WQL, MASE}}}
        """
        results_df = self.evaluate_benchmark(config_path, **predict_kwargs)

        wql_values = results_df["WQL"].dropna().values
        mase_values = results_df["MASE"].dropna().values

        per_dataset = {}
        for _, row in results_df.iterrows():
            per_dataset[row["dataset"]] = {
                "WQL": row.get("WQL", float("nan")),
                "MASE": row.get("MASE", float("nan")),
            }

        return {
            "avg_wql": float(np.mean(wql_values)) if len(wql_values) > 0 else None,
            "avg_mase": float(np.mean(mase_values)) if len(mase_values) > 0 else None,
            "per_dataset": per_dataset,
        }
