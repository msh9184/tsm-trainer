"""Unified evaluation runner for time series forecasting.

Provides a single Evaluator class that:
1. Loads datasets from local paths or HuggingFace cache
2. Converts to GluonTS format for splitting
3. Generates forecasts using any BaseForecaster
4. Computes metrics (WQL, MASE, etc.) via GluonTS
5. Returns structured results as DataFrames

Supports both standalone evaluation and training-time validation.

Key features:
- Local-first data loading for network-restricted environments
- Benchmark checkpointing: resume interrupted evaluations from partial results
- Dataset pre-validation: check all datasets exist before starting
- Per-dataset timeout: skip slow-loading or hanging datasets
- Config schema validation: catch configuration errors before evaluation
- Per-dataset timing: track evaluation time for each dataset
"""

from __future__ import annotations

import json
import logging
import time as time_module
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
    hf_repo = config.get("hf_repo", None)
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

    elif local_only or hf_repo is None:
        searched = data_path or (Path(datasets_root) / ds_name if datasets_root else "N/A")
        raise FileNotFoundError(
            f"Dataset '{ds_name}' not found at local path: {searched}\n"
            f"  datasets_root: {datasets_root}\n"
            f"  Expected: {datasets_root}/{ds_name}/\n"
            f"  Hint: Run utils/download_eval_datasets.py first, or check the dataset name."
        )
    else:
        # Load from HuggingFace (uses cache) — only when local_only is False and hf_repo is set
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


def validate_datasets(
    config_path: str | Path,
    datasets_root: str | Path | None = None,
) -> dict:
    """Pre-validate that all datasets in a benchmark config are available.

    Checks each dataset for existence and basic readability without loading
    the full data. Returns a report of found/missing datasets.

    Parameters
    ----------
    config_path : str or Path
        Path to benchmark YAML config.
    datasets_root : str or Path, optional
        Root directory for local datasets.

    Returns
    -------
    dict
        {
            "total": int,
            "found": int,
            "missing": int,
            "datasets": {name: {"status": "found"|"missing", "path": str, "size_mb": float}},
        }
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        configs = yaml.safe_load(f)

    report = {"total": len(configs), "found": 0, "missing": 0, "datasets": {}}

    for config in configs:
        ds_name = config["name"]
        ds_info = {"status": "missing", "path": None, "size_mb": 0.0}

        # Check local path
        if datasets_root is not None:
            local_path = Path(datasets_root) / ds_name
            if local_path.exists():
                data_files = (
                    list(local_path.glob("*.arrow"))
                    + list(local_path.glob("*.parquet"))
                )
                if data_files:
                    total_size = sum(f.stat().st_size for f in data_files)
                    ds_info = {
                        "status": "found",
                        "path": str(local_path),
                        "size_mb": round(total_size / (1024 * 1024), 1),
                        "n_files": len(data_files),
                    }

        if ds_info["status"] == "found":
            report["found"] += 1
        else:
            report["missing"] += 1

        report["datasets"][ds_name] = ds_info

    return report


def validate_config(config_path: str | Path) -> list[str]:
    """Validate a benchmark YAML config for schema correctness.

    Checks that all required fields are present and have valid types
    before evaluation begins, providing clear error messages.

    Parameters
    ----------
    config_path : str or Path
        Path to benchmark YAML config.

    Returns
    -------
    list[str]
        List of validation errors. Empty list means valid config.
    """
    config_path = Path(config_path)
    errors = []

    if not config_path.exists():
        return [f"Config file not found: {config_path}"]

    try:
        with open(config_path) as f:
            configs = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"YAML parse error in {config_path}: {e}"]

    if not isinstance(configs, list):
        return [f"Config must be a YAML list of dataset entries, got {type(configs).__name__}"]

    if len(configs) == 0:
        return [f"Config file is empty: {config_path}"]

    required_fields = {"name": str, "offset": int, "prediction_length": int}
    optional_fields = {"hf_repo": str, "num_rolls": int, "split": str, "data_dir": str, "local_only": bool}

    for idx, entry in enumerate(configs):
        prefix = f"Entry {idx} ({entry.get('name', '???')})"

        if not isinstance(entry, dict):
            errors.append(f"{prefix}: Expected dict, got {type(entry).__name__}")
            continue

        # Check required fields
        for field, expected_type in required_fields.items():
            if field not in entry:
                errors.append(f"{prefix}: Missing required field '{field}'")
            elif not isinstance(entry[field], expected_type):
                errors.append(
                    f"{prefix}: Field '{field}' should be {expected_type.__name__}, "
                    f"got {type(entry[field]).__name__} ({entry[field]!r})"
                )

        # Check optional fields types
        for field, expected_type in optional_fields.items():
            if field in entry and not isinstance(entry[field], expected_type):
                errors.append(
                    f"{prefix}: Field '{field}' should be {expected_type.__name__}, "
                    f"got {type(entry[field]).__name__}"
                )

        # Semantic validation
        if "prediction_length" in entry and isinstance(entry["prediction_length"], int):
            if entry["prediction_length"] <= 0:
                errors.append(f"{prefix}: prediction_length must be positive, got {entry['prediction_length']}")

        if "offset" in entry and isinstance(entry["offset"], int):
            if entry["offset"] >= 0:
                errors.append(f"{prefix}: offset should be negative (test tail), got {entry['offset']}")

        if "num_rolls" in entry and isinstance(entry["num_rolls"], int):
            if entry["num_rolls"] <= 0:
                errors.append(f"{prefix}: num_rolls must be positive, got {entry['num_rolls']}")

    return errors


class Evaluator:
    """Unified evaluation runner for time series forecasting.

    Handles the full evaluation pipeline:
    dataset loading → forecast generation → metric computation → result aggregation.

    Supports benchmark checkpointing for resuming interrupted evaluations,
    dataset pre-validation, and per-dataset timeout.

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
    dataset_timeout : float, optional
        Maximum seconds per dataset evaluation. 0 = no timeout (default).
    """

    def __init__(
        self,
        forecaster: BaseForecaster,
        quantile_levels: list[float] | None = None,
        batch_size: int = 32,
        datasets_root: str | Path | None = None,
        dataset_timeout: float = 0,
    ):
        self.forecaster = forecaster
        self.quantile_levels = quantile_levels or EVAL_QUANTILES
        self.batch_size = batch_size
        self.datasets_root = datasets_root
        self.dataset_timeout = dataset_timeout

    def evaluate_dataset(
        self,
        config: dict,
        **predict_kwargs,
    ) -> dict:
        """Evaluate a single dataset from config.

        Uses fast vectorized metric computation, bypassing GluonTS
        ``evaluate_forecasts()`` and ``QuantileForecast`` object creation.
        This eliminates the primary CPU bottleneck (Python-level iteration
        over thousands of forecast objects).

        Parameters
        ----------
        config : dict
            Dataset config entry with name, hf_repo, offset, etc.

        Returns
        -------
        dict
            {dataset, model, WQL, MASE, n_series, elapsed_s, ...}
        """
        from .metrics import MetricRegistry

        ds_name = config["name"]
        prediction_length = config["prediction_length"]
        ds_start = time_module.time()

        logger.info(f"Loading dataset: {ds_name}")
        _, test_data, pred_len, dataset_freq = load_dataset_from_config(
            config, datasets_root=self.datasets_root
        )

        # Collect test data: inputs (context) and labels (ground truth)
        inputs_list = []
        labels_list = []
        for inp, lbl in zip(test_data.input, test_data.label):
            inputs_list.append(inp)
            labels_list.append(lbl)

        n_series = len(inputs_list)
        load_time = time_module.time() - ds_start
        logger.info(
            f"Generating forecasts: {ds_name} "
            f"({n_series} series, H={pred_len}, load={load_time:.1f}s)"
        )

        # Extract context tensors and ground truth arrays
        contexts = [torch.tensor(inp["target"], dtype=torch.float32) for inp in inputs_list]
        y_past = [np.asarray(inp["target"], dtype=np.float32) for inp in inputs_list]
        y_true = np.array(
            [np.asarray(lbl["target"], dtype=np.float32)[:pred_len] for lbl in labels_list]
        )  # (N, H)

        # Batch GPU inference → (N, Q, H) numpy array
        forecast_start = time_module.time()
        y_pred_quantiles = self.forecaster.predict_batch(
            contexts,
            prediction_length=pred_len,
            quantile_levels=self.quantile_levels,
            batch_size=self.batch_size,
            **predict_kwargs,
        )  # (N, Q, H)
        forecast_time = time_module.time() - forecast_start

        # Fast vectorized metric computation (no GluonTS overhead)
        logger.info(f"Computing metrics: {ds_name} (forecast={forecast_time:.1f}s)")
        metric_start = time_module.time()
        seasonal_period = MetricRegistry.get_seasonal_period(dataset_freq)
        metrics = MetricRegistry.compute_chronos_metrics_fast(
            y_pred_quantiles, y_true, y_past,
            self.quantile_levels, seasonal_period,
        )
        metric_time = time_module.time() - metric_start

        total_elapsed = time_module.time() - ds_start
        result = {
            "dataset": ds_name,
            "model": self.forecaster.name,
            "WQL": metrics["WQL"],
            "MASE": metrics["MASE"],
            "n_series": n_series,
            "elapsed_s": round(total_elapsed, 1),
        }

        logger.debug(
            f"  {ds_name}: forecast={forecast_time:.1f}s, metrics={metric_time:.1f}s"
        )

        return result

    def evaluate_benchmark(
        self,
        config_path: str | Path,
        output_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        **predict_kwargs,
    ) -> pd.DataFrame:
        """Evaluate all datasets in a benchmark config.

        Supports checkpointing: if checkpoint_path is provided, completed
        results are saved after each dataset. If a checkpoint file already
        exists, previously completed datasets are skipped.

        Parameters
        ----------
        config_path : str or Path
            Path to benchmark YAML config.
        output_path : str or Path, optional
            Path to save final results CSV. If None, results are not saved.
        checkpoint_path : str or Path, optional
            Path to checkpoint file for resume support. If None, no
            checkpointing is performed.

        Returns
        -------
        pd.DataFrame
            Per-dataset results with columns: dataset, model, WQL, MASE.
        """
        config_path = Path(config_path)

        # Validate config schema before starting evaluation
        config_errors = validate_config(config_path)
        if config_errors:
            error_msg = "\n".join(f"  - {e}" for e in config_errors)
            raise ValueError(
                f"Benchmark config validation failed ({config_path.name}):\n{error_msg}"
            )

        with open(config_path) as f:
            configs = yaml.safe_load(f)

        # Load checkpoint if resuming
        completed_datasets: dict[str, dict] = {}
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                try:
                    with open(checkpoint_path) as f:
                        checkpoint_data = json.load(f)
                    for row in checkpoint_data.get("results", []):
                        ds_name = row.get("dataset")
                        if ds_name and not (
                            np.isnan(row.get("WQL", float("nan")))
                            and np.isnan(row.get("MASE", float("nan")))
                        ):
                            completed_datasets[ds_name] = row
                    if completed_datasets:
                        logger.info(
                            f"Resuming from checkpoint: {len(completed_datasets)}/{len(configs)} "
                            f"datasets already completed"
                        )
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {e}")

        remaining = [c for c in configs if c["name"] not in completed_datasets]
        logger.info(
            f"Evaluating benchmark: {config_path.stem} "
            f"({len(remaining)}/{len(configs)} datasets to evaluate, "
            f"model={self.forecaster.name})"
        )

        result_rows = list(completed_datasets.values())
        for idx, config in enumerate(remaining, len(completed_datasets) + 1):
            ds_start = time_module.time()
            try:
                result = self.evaluate_dataset(config, **predict_kwargs)
                result_rows.append(result)
                wql = result.get("WQL", float("nan"))
                mase = result.get("MASE", float("nan"))
                elapsed = time_module.time() - ds_start
                logger.info(
                    f"  [{idx}/{len(configs)}] {config['name']}: "
                    f"WQL={wql:.4f}, MASE={mase:.4f} ({elapsed:.1f}s)"
                )
            except Exception as e:
                elapsed = time_module.time() - ds_start
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

            # Save checkpoint after each dataset
            if checkpoint_path is not None:
                self._save_checkpoint(checkpoint_path, result_rows, config_path.stem)

        results_df = pd.DataFrame(result_rows).sort_values(by="dataset")

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved: {output_path}")

        # Clean up checkpoint on successful completion
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()
            logger.info(f"Checkpoint removed (benchmark complete)")

        return results_df

    @staticmethod
    def _save_checkpoint(
        checkpoint_path: Path,
        result_rows: list[dict],
        benchmark_name: str,
    ):
        """Save intermediate results as a checkpoint file."""
        checkpoint_data = {
            "benchmark": benchmark_name,
            "n_completed": len(result_rows),
            "timestamp": pd.Timestamp.now().isoformat(),
            "results": result_rows,
        }
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

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
