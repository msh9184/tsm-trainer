"""Standalone evaluation runner for occupancy detection models.

Usage:
    python evaluation/evaluate.py \\
        --config training/configs/zeroshot.yaml \\
        --predictions results/predictions.npz \\
        --output results/eval_report.json

Or programmatically:
    from evaluation.evaluate import evaluate_predictions
    metrics = evaluate_predictions(y_true, y_pred, y_prob)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from .metrics import ClassificationMetrics, compute_metrics

logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Evaluate predictions and return metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred : np.ndarray
        Predicted labels.
    y_prob : np.ndarray, optional
        Predicted probabilities for the positive class.

    Returns
    -------
    ClassificationMetrics
    """
    metrics = compute_metrics(y_true, y_pred, y_prob)
    logger.info("\n%s", metrics.summary())
    return metrics


def save_report(
    metrics: ClassificationMetrics,
    output_path: str | Path,
    extra_info: dict | None = None,
) -> None:
    """Save evaluation report as JSON.

    Parameters
    ----------
    metrics : ClassificationMetrics
        Computed metrics.
    output_path : str or Path
        JSON output file path.
    extra_info : dict, optional
        Additional metadata to include in the report.
    """
    report = metrics.to_dict()
    if extra_info:
        report["metadata"] = extra_info

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved evaluation report to %s", output_path)


def main():
    """CLI entry point for evaluating saved predictions."""
    parser = argparse.ArgumentParser(
        description="Evaluate occupancy detection predictions",
    )
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to .npz file containing y_true, y_pred, and optionally y_prob",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON evaluation report",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        logger.error("Predictions file not found: %s", pred_path)
        sys.exit(1)

    data = np.load(pred_path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    y_prob = data.get("y_prob", None)

    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    if args.output:
        save_report(metrics, args.output, extra_info={"source": str(pred_path)})

    print(metrics.summary())


if __name__ == "__main__":
    main()
