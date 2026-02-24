"""Visualization module for APC occupancy detection.

Provides embedding space visualization, evaluation curves, and training
progression animations.
"""

from .animation import EmbeddingTracker
from .curves import plot_all_curves, plot_confusion_matrix, plot_det_curve, plot_roc_curve
from .embeddings import (
    plot_embeddings,
    plot_embeddings_multi_method,
    plot_train_test_comparison,
    reduce_dimensions,
)
from .style import CLASS_COLORS, CLASS_NAMES, configure_output, save_figure, setup_style

__all__ = [
    "CLASS_COLORS",
    "CLASS_NAMES",
    "configure_output",
    "EmbeddingTracker",
    "plot_all_curves",
    "plot_confusion_matrix",
    "plot_det_curve",
    "plot_embeddings_multi_method",
    "plot_embeddings",
    "plot_roc_curve",
    "plot_train_test_comparison",
    "reduce_dimensions",
    "save_figure",
    "setup_style",
]
