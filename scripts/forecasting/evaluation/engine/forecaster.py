"""Model-agnostic forecasting interface.

Provides an abstract base class that all forecasting models must implement,
enabling benchmark evaluation code to work with any model architecture.

Supported models:
- Chronos-2 (encoder-only, patch-based, quantile regression)
- Chronos-Bolt (encoder-only, patch-based)
- Chronos T5 (seq2seq, token-level)
- Custom models (extend BaseForecaster)

The interface supports both quantile-based forecasting (for WQL, CRPS, SQL)
and point forecasting (for MSE, MAE in LTSF).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from chronos import BaseChronosPipeline

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for time series forecasting models.

    All models must implement predict_quantiles(). Point forecasts
    are derived from the median (q=0.5) by default.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name for reporting."""
        ...

    @abstractmethod
    def predict_quantiles(
        self,
        context: list[torch.Tensor],
        prediction_length: int,
        quantile_levels: list[float],
        **kwargs,
    ) -> np.ndarray:
        """Generate quantile forecasts.

        Parameters
        ----------
        context : list[torch.Tensor]
            List of 1D tensors, each containing historical values for one series.
        prediction_length : int
            Number of future time steps to forecast.
        quantile_levels : list[float]
            Quantile levels to predict (e.g., [0.1, 0.2, ..., 0.9]).
        **kwargs
            Model-specific parameters.

        Returns
        -------
        np.ndarray, shape (N, Q, H)
            Quantile forecasts: N series, Q quantile levels, H horizon steps.
        """
        ...

    def predict_point(
        self,
        context: list[torch.Tensor],
        prediction_length: int,
        **kwargs,
    ) -> np.ndarray:
        """Generate point forecasts (median, q=0.5).

        Default implementation extracts median from quantile predictions.
        Override for models with native point forecast support.

        Parameters
        ----------
        context : list[torch.Tensor]
            Historical values per series.
        prediction_length : int
            Forecast horizon.

        Returns
        -------
        np.ndarray, shape (N, H)
            Point forecasts.
        """
        quantiles = self.predict_quantiles(
            context, prediction_length, [0.5], **kwargs
        )
        return quantiles[:, 0, :]  # Extract q=0.5

    def predict_batch(
        self,
        contexts: list[torch.Tensor],
        prediction_length: int,
        quantile_levels: list[float],
        batch_size: int = 32,
        **kwargs,
    ) -> np.ndarray:
        """Generate quantile forecasts in batches.

        Parameters
        ----------
        contexts : list[torch.Tensor]
            All series contexts.
        prediction_length : int
            Forecast horizon.
        quantile_levels : list[float]
            Quantile levels.
        batch_size : int
            Batch size for inference.

        Returns
        -------
        np.ndarray, shape (N, Q, H)
        """
        all_outputs = []
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i : i + batch_size]
            outputs = self.predict_quantiles(
                batch, prediction_length, quantile_levels, **kwargs
            )
            all_outputs.append(outputs)
        return np.concatenate(all_outputs, axis=0)


def _normalize_pipeline_output(
    quantiles,
    n_quantiles: int,
    prediction_length: int,
) -> np.ndarray:
    """Convert raw pipeline output to standardized (N, Q, H) numpy array.

    Chronos pipelines return quantile predictions in (N, H, Q) format.
    This function handles type conversion and axis reordering to produce
    the (N, Q, H) format expected by BaseForecaster callers.

    Uses both n_quantiles and prediction_length to disambiguate the output
    shape, which is critical when H == Q (e.g., 9 quantiles with H=9).

    Parameters
    ----------
    quantiles : list, torch.Tensor, or np.ndarray
        Raw pipeline output.
    n_quantiles : int
        Number of quantile levels requested.
    prediction_length : int
        Expected forecast horizon length.

    Returns
    -------
    np.ndarray, shape (N, Q, H)
    """
    # Convert to numpy
    if isinstance(quantiles, list):
        # Chronos-2 returns list of tensors for multivariate
        quantiles = np.stack(quantiles).squeeze(axis=1)
    elif isinstance(quantiles, torch.Tensor):
        quantiles = quantiles.cpu().numpy()

    if quantiles.ndim != 3:
        raise ValueError(
            f"Pipeline returned {quantiles.ndim}D array (shape={quantiles.shape}), "
            f"expected 3D (N, H, Q) or (N, Q, H)."
        )

    # Detect and swap: pipeline returns (N, H, Q), we need (N, Q, H).
    # Use both prediction_length and n_quantiles for robust disambiguation.
    dim1, dim2 = quantiles.shape[1], quantiles.shape[2]

    if dim1 == n_quantiles and dim2 == prediction_length:
        # Already in (N, Q, H) format — no swap needed
        pass
    elif dim1 == prediction_length and dim2 == n_quantiles:
        # Pipeline convention (N, H, Q) — swap to (N, Q, H)
        quantiles = quantiles.swapaxes(-1, -2)
    elif dim2 == n_quantiles and dim1 != n_quantiles:
        # Last dim matches Q but first dim doesn't match H exactly
        # (may happen with variable-length outputs) — swap
        quantiles = quantiles.swapaxes(-1, -2)
    elif dim1 == n_quantiles and dim2 != prediction_length:
        # First dim matches Q — already (N, Q, H) with different H
        pass
    else:
        # Ambiguous: neither dimension clearly matches.
        # Default to pipeline convention (N, H, Q) and swap.
        logger.warning(
            f"Ambiguous pipeline output shape {quantiles.shape}: "
            f"expected Q={n_quantiles}, H={prediction_length}. "
            f"Assuming (N, H, Q) convention and swapping."
        )
        quantiles = quantiles.swapaxes(-1, -2)

    return quantiles


class Chronos2Forecaster(BaseForecaster):
    """Forecaster wrapper for Chronos-2 models.

    Wraps Chronos2Pipeline to conform to the BaseForecaster interface.
    Supports loading from local checkpoints or HuggingFace model IDs.

    Parameters
    ----------
    model_path : str
        Path to local checkpoint directory or HuggingFace model ID.
    device : str
        Device for inference (e.g., "cuda", "cuda:0", "cpu").
    torch_dtype : str or torch.dtype
        Model data type (e.g., "float32", "bfloat16").
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: str | torch.dtype = "float32",
    ):
        from chronos import BaseChronosPipeline

        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)

        self._model_path = model_path
        self._device = device
        self._pipeline = BaseChronosPipeline.from_pretrained(
            model_path, device_map=device, dtype=torch_dtype
        )
        self._model_name = Path(model_path).name if Path(model_path).exists() else model_path

    @property
    def name(self) -> str:
        return f"chronos-2:{self._model_name}"

    @property
    def pipeline(self) -> BaseChronosPipeline:
        return self._pipeline

    def predict_quantiles(
        self,
        context: list[torch.Tensor],
        prediction_length: int,
        quantile_levels: list[float],
        **kwargs,
    ) -> np.ndarray:
        with torch.no_grad():
            quantiles, _ = self._pipeline.predict_quantiles(
                context,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                **kwargs,
            )

        quantiles = _normalize_pipeline_output(
            quantiles, len(quantile_levels), prediction_length
        )
        return quantiles


class ChronosBoltForecaster(BaseForecaster):
    """Forecaster wrapper for Chronos-Bolt models."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: str | torch.dtype = "float32",
    ):
        from chronos import BaseChronosPipeline

        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)

        self._model_path = model_path
        self._device = device
        self._pipeline = BaseChronosPipeline.from_pretrained(
            model_path, device_map=device, dtype=torch_dtype
        )
        self._model_name = Path(model_path).name if Path(model_path).exists() else model_path

    @property
    def name(self) -> str:
        return f"chronos-bolt:{self._model_name}"

    def predict_quantiles(
        self,
        context: list[torch.Tensor],
        prediction_length: int,
        quantile_levels: list[float],
        **kwargs,
    ) -> np.ndarray:
        with torch.no_grad():
            quantiles, _ = self._pipeline.predict_quantiles(
                context,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                **kwargs,
            )

        quantiles = _normalize_pipeline_output(
            quantiles, len(quantile_levels), prediction_length
        )
        return quantiles


class TrainingModelForecaster(BaseForecaster):
    """Forecaster that wraps a Chronos2Model during training.

    Used by the EnhancedBenchmarkCallback to evaluate the model being trained
    without loading a separate pipeline. Constructs a temporary pipeline wrapper
    around the live training model.

    Parameters
    ----------
    model : Chronos2Model
        The model currently being trained.
    device : torch.device
        Device the model is on.
    """

    def __init__(self, model, device: torch.device):
        from chronos import Chronos2Pipeline

        self._model = model
        self._device = device

        # Build a temporary pipeline wrapper
        self._pipeline = Chronos2Pipeline.__new__(Chronos2Pipeline)
        self._pipeline.model = model
        self._pipeline.device = device
        self._pipeline.chronos_config = model.chronos_config

    @property
    def name(self) -> str:
        return "training-model"

    def predict_quantiles(
        self,
        context: list[torch.Tensor],
        prediction_length: int,
        quantile_levels: list[float],
        **kwargs,
    ) -> np.ndarray:
        with torch.no_grad():
            quantiles, _ = self._pipeline.predict_quantiles(
                context,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                **kwargs,
            )

        quantiles = _normalize_pipeline_output(
            quantiles, len(quantile_levels), prediction_length
        )
        return quantiles
