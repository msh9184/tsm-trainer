"""Classification metrics for occupancy detection.

Enhanced metrics matching the apc_enter_leave evaluation framework:
  - Binary and multiclass support
  - Per-class F1, macro/weighted aggregation
  - Wilson confidence intervals
  - ROC/AUC, EER computation
  - CV aggregation helpers
  - NaN-safe JSON serialization
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
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Container for classification evaluation results.

    Supports both legacy binary-only fields (``f1``, ``precision``, ``recall``)
    and enhanced multiclass-ready fields (``f1_macro``, ``f1_per_class``, etc.).
    """

    accuracy: float = 0.0
    f1: float = 0.0              # Binary F1 (backward compat)
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    f1_per_class: dict[str, float] = field(default_factory=dict)
    precision: float = 0.0       # Binary precision (backward compat)
    recall: float = 0.0          # Binary recall (backward compat)
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    eer: float = float("nan")
    eer_threshold: float = float("nan")
    confusion_matrix: np.ndarray | None = None
    n_samples: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)

    # ROC curve data (large arrays, excluded from JSON)
    roc_fpr: np.ndarray | None = None
    roc_tpr: np.ndarray | None = None
    roc_thresholds: np.ndarray | None = None
    roc_auc: float = float("nan")

    # Confidence intervals
    ci_lower: float | None = None
    ci_upper: float | None = None

    # CV metadata
    cv_method: str | None = None
    n_folds: int | None = None

    def summary(self) -> str:
        """Return a formatted summary string.

        Display order: threshold-independent metrics FIRST (AUC, EER),
        then threshold-dependent metrics (accuracy, F1, precision, recall).
        """
        lines = [
            "Classification Results",
            "=" * 50,
        ]
        # Primary: threshold-independent metrics
        if not math.isnan(self.roc_auc):
            lines.append(f"  AUC:           {self.roc_auc:.4f}")
        if not math.isnan(self.eer):
            lines.append(f"  EER:           {self.eer:.4f} (threshold={self.eer_threshold:.4f})")
        # Secondary: threshold-dependent metrics
        lines.extend([
            f"  Accuracy:      {self.accuracy:.4f}",
            f"  F1 (binary):   {self.f1:.4f}",
            f"  F1 (macro):    {self.f1_macro:.4f}",
            f"  Precision:     {self.precision:.4f}",
            f"  Recall:        {self.recall:.4f}",
        ])
        if self.ci_lower is not None:
            lines.append(f"  Wilson CI:     [{self.ci_lower:.4f}, {self.ci_upper:.4f}]")

        if self.f1_per_class:
            lines.append("  Per-class F1:")
            for name, f1_val in self.f1_per_class.items():
                lines.append(f"    {name}: {f1_val:.4f}")

        if self.confusion_matrix is not None:
            lines.append("")
            lines.append("  Confusion Matrix:")
            cm = self.confusion_matrix
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                lines.append(f"              Pred=0  Pred=1")
                lines.append(f"    Actual=0  {tn:5d}   {fp:5d}")
                lines.append(f"    Actual=1  {fn:5d}   {tp:5d}")
            else:
                lines.append(str(cm))

        lines.append("")
        dist = ", ".join(f"{k}={v}" for k, v in self.class_distribution.items())
        lines.append(f"  Samples: {self.n_samples}  ({dist})")

        if self.cv_method:
            lines.append(f"  CV: {self.cv_method} ({self.n_folds} folds)")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary.

        Excludes large ROC arrays.  NaN → None for valid JSON (RFC 7159).
        """
        def _safe(v: float) -> float | None:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v

        return {
            "accuracy": float(self.accuracy),
            "f1": _safe(float(self.f1)),
            "f1_macro": _safe(float(self.f1_macro)),
            "f1_weighted": _safe(float(self.f1_weighted)),
            "f1_per_class": dict(self.f1_per_class),
            "precision": _safe(float(self.precision)),
            "recall": _safe(float(self.recall)),
            "precision_macro": _safe(float(self.precision_macro)),
            "recall_macro": _safe(float(self.recall_macro)),
            "eer": _safe(float(self.eer)),
            "eer_threshold": _safe(float(self.eer_threshold)),
            "roc_auc": _safe(float(self.roc_auc)),
            "confusion_matrix": self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            "n_samples": self.n_samples,
            "class_distribution": dict(self.class_distribution),
            "ci_lower": _safe(self.ci_lower) if self.ci_lower is not None else None,
            "ci_upper": _safe(self.ci_upper) if self.ci_upper is not None else None,
            "cv_method": self.cv_method,
            "n_folds": self.n_folds,
        }


def _compute_eer_and_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray | None,
) -> tuple[float, float, np.ndarray | None, np.ndarray | None, np.ndarray | None, float]:
    """Compute Equal Error Rate and full ROC curve data.

    Returns (eer, eer_threshold, fpr, tpr, thresholds, auc_score).
    """
    nan = float("nan")
    if y_prob is None:
        return nan, nan, None, None, None, nan

    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        logger.warning(
            "Only one class present in y_true (%s), EER is undefined",
            unique_labels,
        )
        return nan, nan, None, None, None, nan

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1.0 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    auc_score = float(sklearn_auc(fpr, tpr))

    return eer, eer_threshold, fpr, tpr, thresholds, auc_score


def compute_wilson_ci(
    accuracy: float,
    n: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute Wilson score interval for a proportion.

    Parameters
    ----------
    accuracy : float
        Observed accuracy (0 to 1).
    n : int
        Number of samples.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    lower, upper : float
    """
    if n == 0:
        return 0.0, 1.0

    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    p = accuracy
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom

    return max(0.0, center - margin), min(1.0, center + margin)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
    compute_ci: bool = True,
) -> ClassificationMetrics:
    """Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (0 or 1 for binary).
    y_pred : np.ndarray
        Predicted labels.
    y_prob : np.ndarray or None
        Predicted probability for class 1 (binary, shape (n,)),
        or full probability matrix (multiclass, shape (n, C)).
    class_names : list[str] or None
        Human-readable class names.
    compute_ci : bool
        Whether to compute Wilson confidence interval.

    Returns
    -------
    ClassificationMetrics
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    n_classes = len(labels)
    n_samples = len(y_true)

    if class_names is None:
        class_names = ["Empty", "Occupied"] if n_classes <= 2 else [f"Class_{i}" for i in labels]

    # Core metrics
    acc = accuracy_score(y_true, y_pred)

    # Binary F1/precision/recall (backward compat)
    f1_bin = f1_score(y_true, y_pred, zero_division=0)
    prec_bin = precision_score(y_true, y_pred, zero_division=0)
    rec_bin = recall_score(y_true, y_pred, zero_division=0)

    # Macro/weighted
    f1_mac = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_wt = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    prec_mac = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec_mac = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    # Per-class F1
    f1_per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_per_class = {}
    for i, label in enumerate(labels):
        name = class_names[i] if i < len(class_names) else f"Class_{label}"
        f1_per_class[name] = float(f1_per[i])

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Class distribution
    unique, counts = np.unique(y_true, return_counts=True)
    class_dist = {}
    for u, c in zip(unique, counts):
        idx = labels.index(int(u)) if int(u) in labels else -1
        name = class_names[idx] if 0 <= idx < len(class_names) else f"Class_{u}"
        class_dist[name] = int(c)

    # ROC / EER
    eer_val, eer_thresh, roc_fpr, roc_tpr, roc_thresholds, roc_auc_val = (
        _compute_eer_and_roc(y_true, y_prob)
    )

    # Wilson CI
    ci_lower = None
    ci_upper = None
    if compute_ci and n_samples > 0:
        try:
            ci_lower, ci_upper = compute_wilson_ci(acc, n_samples)
        except ImportError:
            pass

    return ClassificationMetrics(
        accuracy=float(acc),
        f1=float(f1_bin),
        f1_macro=float(f1_mac),
        f1_weighted=float(f1_wt),
        f1_per_class=f1_per_class,
        precision=float(prec_bin),
        recall=float(rec_bin),
        precision_macro=float(prec_mac),
        recall_macro=float(rec_mac),
        eer=eer_val,
        eer_threshold=eer_thresh,
        confusion_matrix=cm,
        n_samples=n_samples,
        class_distribution=class_dist,
        roc_fpr=roc_fpr,
        roc_tpr=roc_tpr,
        roc_thresholds=roc_thresholds,
        roc_auc=roc_auc_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def aggregate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
    cv_method: str | None = None,
    n_folds: int | None = None,
) -> ClassificationMetrics:
    """Compute metrics from aggregated CV predictions."""
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    metrics.cv_method = cv_method
    metrics.n_folds = n_folds
    return metrics
