"""Classification metrics for enter/leave event detection.

Supports both binary (Enter vs Leave) and 3-class (+ None) evaluation.
Provides per-class precision/recall/F1, macro/weighted averages, ROC AUC,
EER, and cross-validation aggregation utilities.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc as sklearn_auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class EventClassificationMetrics:
    """Container for event classification metrics."""

    accuracy: float
    f1_macro: float
    f1_weighted: float
    f1_per_class: dict[str, float]
    precision_macro: float
    recall_macro: float
    confusion_matrix: np.ndarray
    n_samples: int
    class_distribution: dict[str, int]
    roc_auc: float = float("nan")
    eer: float = float("nan")
    eer_threshold: float = float("nan")
    roc_fpr: np.ndarray | None = None
    roc_tpr: np.ndarray | None = None
    cv_method: str = ""
    n_folds: int = 0

    def summary(self, class_names: list[str] | None = None) -> str:
        """Return a formatted summary string."""
        lines = [
            "Event Classification Results",
            "=" * 50,
            f"  Accuracy:       {self.accuracy:.4f}",
            f"  F1 (macro):     {self.f1_macro:.4f}",
            f"  F1 (weighted):  {self.f1_weighted:.4f}",
            f"  Precision:      {self.precision_macro:.4f}",
            f"  Recall:         {self.recall_macro:.4f}",
            f"  AUC:            {self.roc_auc:.4f}",
            f"  EER:            {self.eer:.4f}",
        ]

        if self.cv_method:
            lines.append(f"  CV method:      {self.cv_method} ({self.n_folds} folds)")

        lines.append("")
        lines.append("Per-Class F1:")
        for cls_name, f1_val in self.f1_per_class.items():
            lines.append(f"  {cls_name}: {f1_val:.4f}")

        lines.append("")
        lines.append("Confusion Matrix:")
        n_classes = self.confusion_matrix.shape[0]
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
        header = "           " + "  ".join(f"{name:>7s}" for name in class_names)
        lines.append(header)
        for i in range(n_classes):
            row_vals = "  ".join(f"{int(self.confusion_matrix[i, j]):7d}" for j in range(n_classes))
            lines.append(f"  {class_names[i]:>7s}  {row_vals}")

        lines.append("")
        dist_str = ", ".join(f"{k}={v}" for k, v in self.class_distribution.items())
        lines.append(f"  Samples: {self.n_samples} ({dist_str})")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""

        def _safe(v: float) -> float | None:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v

        return {
            "accuracy": float(self.accuracy),
            "f1_macro": float(self.f1_macro),
            "f1_weighted": float(self.f1_weighted),
            "f1_per_class": {k: float(v) for k, v in self.f1_per_class.items()},
            "precision_macro": float(self.precision_macro),
            "recall_macro": float(self.recall_macro),
            "roc_auc": _safe(float(self.roc_auc)),
            "eer": _safe(float(self.eer)),
            "eer_threshold": _safe(float(self.eer_threshold)),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "n_samples": self.n_samples,
            "class_distribution": self.class_distribution,
            "cv_method": self.cv_method,
            "n_folds": self.n_folds,
        }


# ---------------------------------------------------------------------------
# EER computation (binary only)
# ---------------------------------------------------------------------------

def _compute_eer_and_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray | None,
) -> tuple[float, float, np.ndarray | None, np.ndarray | None, float]:
    """Compute EER and ROC curve data for binary classification.

    Returns (eer, eer_threshold, fpr, tpr, auc_score).
    """
    nan = float("nan")
    if y_prob is None:
        return nan, nan, None, None, nan

    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        logger.warning("Only one class present; EER/AUC undefined")
        return nan, nan, None, None, nan

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1.0 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5

    auc_score = float(sklearn_auc(fpr, tpr))

    return eer, eer_threshold, fpr, tpr, auc_score


# ---------------------------------------------------------------------------
# Main metrics computation
# ---------------------------------------------------------------------------

def compute_event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> EventClassificationMetrics:
    """Compute all classification metrics, auto-detecting binary vs multiclass.

    Parameters
    ----------
    y_true : np.ndarray
        Shape (n_samples,). Ground truth labels.
    y_pred : np.ndarray
        Shape (n_samples,). Predicted labels.
    y_prob : np.ndarray, optional
        For binary: shape (n_samples,) predicted prob for class 1.
        For multiclass: shape (n_samples, n_classes) predicted probs.
    class_names : list[str], optional
        Human-readable class names.

    Returns
    -------
    EventClassificationMetrics
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    n_classes = len(unique_classes)
    is_binary = n_classes <= 2

    if class_names is None:
        class_names = [str(c) for c in unique_classes]

    labels = list(range(n_classes))

    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    prec_mac = precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    rec_mac = recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)

    # Per-class F1
    f1_per = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1_per_class = {}
    for i, label in enumerate(labels):
        name = class_names[label] if label < len(class_names) else str(label)
        f1_per_class[name] = float(f1_per[i])

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Class distribution
    unique, counts = np.unique(y_true, return_counts=True)
    class_dist = {}
    for u, c in zip(unique.tolist(), counts.tolist()):
        name = class_names[u] if u < len(class_names) else str(u)
        class_dist[name] = c

    # ROC AUC and EER
    eer = float("nan")
    eer_threshold = float("nan")
    roc_fpr = None
    roc_tpr = None
    roc_auc = float("nan")

    if is_binary and y_prob is not None:
        # Binary: y_prob is probability for class 1
        prob_1d = y_prob.ravel() if y_prob.ndim > 1 else y_prob
        eer, eer_threshold, roc_fpr, roc_tpr, roc_auc = _compute_eer_and_roc(y_true, prob_1d)
    elif not is_binary and y_prob is not None:
        # Multiclass: one-vs-rest AUC
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == n_classes:
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except ValueError:
            logger.warning("Could not compute multiclass AUC (may need all classes in predictions)")

    return EventClassificationMetrics(
        accuracy=float(acc),
        f1_macro=float(f1_mac),
        f1_weighted=float(f1_wt),
        f1_per_class=f1_per_class,
        precision_macro=float(prec_mac),
        recall_macro=float(rec_mac),
        confusion_matrix=cm,
        n_samples=len(y_true),
        class_distribution=class_dist,
        roc_auc=roc_auc,
        eer=eer,
        eer_threshold=eer_threshold,
        roc_fpr=roc_fpr,
        roc_tpr=roc_tpr,
    )


# ---------------------------------------------------------------------------
# Cross-validation aggregation
# ---------------------------------------------------------------------------

def aggregate_cv_predictions(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    y_prob_all: np.ndarray | None,
    cv_method: str,
    n_folds: int,
    class_names: list[str] | None = None,
) -> EventClassificationMetrics:
    """Aggregate all CV fold predictions and compute GLOBAL metrics.

    For LOOCV: each fold produces 1 prediction. Collect all N predictions
    into one array and compute metrics once (NOT per-fold averages).

    For K-fold with repeats: if a sample appears in multiple folds,
    average its predicted probabilities across appearances.

    Parameters
    ----------
    y_true_all : np.ndarray
        Shape (n_predictions,). Ground truth labels for all fold predictions.
    y_pred_all : np.ndarray
        Shape (n_predictions,). Predicted labels for all fold predictions.
    y_prob_all : np.ndarray or None
        Shape (n_predictions,) or (n_predictions, n_classes). Predicted
        probabilities for all fold predictions.
    cv_method : str
        Name of the CV method (e.g., "LOOCV", "StratifiedKFold5x10").
    n_folds : int
        Total number of folds.
    class_names : list[str], optional
        Human-readable class names.
    """
    metrics = compute_event_metrics(y_true_all, y_pred_all, y_prob_all, class_names)
    metrics.cv_method = cv_method
    metrics.n_folds = n_folds
    return metrics


def aggregate_repeated_cv_predictions(
    sample_indices_all: np.ndarray,
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    y_prob_all: np.ndarray | None,
    n_samples: int,
    cv_method: str,
    n_folds: int,
    class_names: list[str] | None = None,
) -> EventClassificationMetrics:
    """Aggregate repeated K-fold predictions by averaging per-sample probabilities.

    When a sample appears in multiple test folds (from repeated K-fold),
    average its predicted probabilities and take the argmax as the final
    prediction for a single global evaluation.

    Parameters
    ----------
    sample_indices_all : np.ndarray
        Shape (n_predictions,). Original sample index for each prediction.
    y_true_all : np.ndarray
        Shape (n_predictions,). True labels.
    y_pred_all : np.ndarray
        Shape (n_predictions,). Per-fold predicted labels (used as fallback).
    y_prob_all : np.ndarray or None
        Shape (n_predictions,) or (n_predictions, n_classes).
    n_samples : int
        Total number of unique samples.
    cv_method, n_folds, class_names : as above.
    """
    if y_prob_all is not None and y_prob_all.ndim == 1:
        # Binary: convert to 2-class probability matrix
        y_prob_all = np.column_stack([1.0 - y_prob_all, y_prob_all])

    if y_prob_all is not None:
        n_classes = y_prob_all.shape[1]
        avg_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
        counts = np.zeros(n_samples, dtype=np.int64)
        avg_true = np.full(n_samples, -1, dtype=np.int64)

        for i, sample_idx in enumerate(sample_indices_all):
            avg_probs[sample_idx] += y_prob_all[i]
            counts[sample_idx] += 1
            avg_true[sample_idx] = y_true_all[i]

        # Average probabilities
        valid = counts > 0
        avg_probs[valid] /= counts[valid, np.newaxis]

        y_true_final = avg_true[valid]
        y_pred_final = np.argmax(avg_probs[valid], axis=1).astype(np.int64)

        # For binary, extract prob of class 1
        n_classes_actual = len(np.unique(y_true_final))
        if n_classes_actual <= 2:
            y_prob_final = avg_probs[valid, 1]
        else:
            y_prob_final = avg_probs[valid]
    else:
        # Fallback: majority vote on predictions
        from collections import Counter
        vote_preds = {}
        vote_true = {}
        for i, sample_idx in enumerate(sample_indices_all):
            vote_preds.setdefault(sample_idx, []).append(y_pred_all[i])
            vote_true[sample_idx] = y_true_all[i]

        y_true_final = np.array([vote_true[k] for k in sorted(vote_true)], dtype=np.int64)
        y_pred_final = np.array(
            [Counter(vote_preds[k]).most_common(1)[0][0] for k in sorted(vote_preds)],
            dtype=np.int64,
        )
        y_prob_final = None

    metrics = compute_event_metrics(y_true_final, y_pred_final, y_prob_final, class_names)
    metrics.cv_method = cv_method
    metrics.n_folds = n_folds
    return metrics
