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

        if isinstance(quantiles, list):
            # Chronos-2 returns list of tensors for multivariate
            quantiles = np.stack(quantiles).squeeze(axis=1)
        elif isinstance(quantiles, torch.Tensor):
            quantiles = quantiles.cpu().numpy()

        # Shape: (N, Q, H) — swap if needed
        # Chronos pipeline returns (N, H, Q), need (N, Q, H)
        if quantiles.ndim == 3 and quantiles.shape[-1] == len(quantile_levels):
            quantiles = quantiles.swapaxes(-1, -2)

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

        if isinstance(quantiles, torch.Tensor):
            quantiles = quantiles.cpu().numpy()

        if quantiles.ndim == 3 and quantiles.shape[-1] == len(quantile_levels):
            quantiles = quantiles.swapaxes(-1, -2)

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

        if isinstance(quantiles, list):
            quantiles = np.stack(quantiles).squeeze(axis=1)
        elif isinstance(quantiles, torch.Tensor):
            quantiles = quantiles.cpu().numpy()

        if quantiles.ndim == 3 and quantiles.shape[-1] == len(quantile_levels):
            quantiles = quantiles.swapaxes(-1, -2)

        return quantiles
