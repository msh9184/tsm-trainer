"""Classification metrics for binary occupancy detection.

Provides a unified metrics computation function returning accuracy, F1,
precision, recall, EER, and a confusion matrix as a structured dataclass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Container for binary classification metrics."""

    accuracy: float
    f1: float
    precision: float
    recall: float
    eer: float  # Equal Error Rate
    eer_threshold: float  # Threshold at EER
    confusion_matrix: np.ndarray  # Shape (2, 2): [[TN, FP], [FN, TP]]
    n_samples: int
    class_distribution: dict[int, int]  # {0: count, 1: count}

    def summary(self) -> str:
        """Return a formatted summary string."""
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        lines = [
            "Classification Results",
            "=" * 40,
            f"  Accuracy:    {self.accuracy:.4f}",
            f"  F1 Score:    {self.f1:.4f}",
            f"  Precision:   {self.precision:.4f}",
            f"  Recall:      {self.recall:.4f}",
            f"  EER:         {self.eer:.4f} (threshold={self.eer_threshold:.4f})",
            "",
            "Confusion Matrix:",
            f"              Pred=0  Pred=1",
            f"  Actual=0    {tn:5d}   {fp:5d}",
            f"  Actual=1    {fn:5d}   {tp:5d}",
            "",
            f"  Samples: {self.n_samples}  "
            f"(empty={self.class_distribution.get(0, 0)}, "
            f"occupied={self.class_distribution.get(1, 0)})",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "accuracy": float(self.accuracy),
            "f1": float(self.f1),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "eer": float(self.eer),
            "eer_threshold": float(self.eer_threshold),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "n_samples": self.n_samples,
            "class_distribution": self.class_distribution,
        }


def _compute_eer(
    y_true: np.ndarray, y_prob: np.ndarray | None,
) -> tuple[float, float]:
    """Compute Equal Error Rate from binary labels and probabilities.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels.
    y_prob : np.ndarray or None
        Predicted probabilities for the positive class. If None, returns
        (nan, nan) as EER requires probability scores.

    Returns
    -------
    eer : float
        Equal Error Rate.
    threshold : float
        Threshold at EER.
    """
    if y_prob is None:
        return float("nan"), float("nan")

    # Ensure we have both classes
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        logger.warning(
            "Only one class present in y_true (%s), EER is undefined",
            unique_labels,
        )
        return float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1.0 - tpr

    # Find the point where FPR ≈ FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5

    return eer, threshold


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute all classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Shape (n_samples,). Binary ground truth (0 or 1).
    y_pred : np.ndarray
        Shape (n_samples,). Predicted labels (0 or 1).
    y_prob : np.ndarray, optional
        Shape (n_samples,). Predicted probability for class 1.
        Required for EER computation.

    Returns
    -------
    ClassificationMetrics
        Container with all computed metrics.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    # Use zero_division=0 to handle degenerate cases gracefully
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    eer, eer_thresh = _compute_eer(y_true, y_prob)

    unique, counts = np.unique(y_true, return_counts=True)
    class_dist = dict(zip(unique.tolist(), counts.tolist()))

    return ClassificationMetrics(
        accuracy=float(acc),
        f1=float(f1),
        precision=float(prec),
        recall=float(rec),
        eer=eer,
        eer_threshold=eer_thresh,
        confusion_matrix=cm,
        n_samples=len(y_true),
        class_distribution=class_dist,
    )
