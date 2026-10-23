"""Evaluation metrics used by the CMERC reproduction pipeline."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def calibration_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 15,
) -> dict:
    """Compute confidence calibration and success-prediction metrics.

    ECE/MCE follow Guo et al. with equal-width confidence bins. RMSCE is
    the weighted root-mean-square calibration error. AUROC and AUPRC treat
    correct predictions as the positive class and maximum probability as
    the score. Values are returned as percentages.
    """
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities must have shape [num_samples, num_classes]")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predictions == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Guo et al. use ((m-1)/M, m/M]; maximum softmax confidence is > 0.
    bin_ids = np.searchsorted(bin_edges[1:], confidence, side="left")
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    weighted_abs_error = 0.0
    weighted_square_error = 0.0
    maximum_error = 0.0
    bins = []
    for bin_index in range(n_bins):
        selected = bin_ids == bin_index
        count = int(selected.sum())
        if count == 0:
            continue
        accuracy = float(correct[selected].mean())
        mean_confidence = float(confidence[selected].mean())
        gap = abs(accuracy - mean_confidence)
        weight = count / labels.size
        weighted_abs_error += weight * gap
        weighted_square_error += weight * gap * gap
        maximum_error = max(maximum_error, gap)
        bins.append(
            {
                "index": bin_index,
                "lower": float(bin_edges[bin_index]),
                "upper": float(bin_edges[bin_index + 1]),
                "count": count,
                "accuracy": accuracy,
                "confidence": mean_confidence,
                "gap": gap,
            }
        )

    if np.unique(correct).size == 2:
        auroc = float(roc_auc_score(correct, confidence))
        auprc = float(average_precision_score(correct, confidence))
    else:
        auroc = float("nan")
        auprc = float("nan")

    return {
        "n_bins": n_bins,
        "ece": round(100.0 * weighted_abs_error, 12),
        "mce": round(100.0 * maximum_error, 12),
        "rmsce": round(100.0 * np.sqrt(weighted_square_error), 12),
        "auroc_success": round(100.0 * auroc, 12),
        "auprc_success": round(100.0 * auprc, 12),
        "bins": bins,
    }


def confidence_enhancement_level(
    labels: np.ndarray,
    full_log_probabilities: np.ndarray,
    removed_log_probabilities: Mapping[str, np.ndarray],
) -> dict:
    """Reproduce the repository's CEL scale using true-class log probability."""
    labels = np.asarray(labels, dtype=np.int64)
    full = np.asarray(full_log_probabilities, dtype=np.float64)
    row_ids = np.arange(labels.size)
    full_true = full[row_ids, labels]
    by_removal = {}

    for name, values in removed_log_probabilities.items():
        removed = np.asarray(values, dtype=np.float64)
        if removed.shape != full.shape:
            raise ValueError(f"removal {name!r} has shape {removed.shape}, expected {full.shape}")
        increases = np.maximum(0.0, removed[row_ids, labels] - full_true)
        by_removal[name] = float(increases.sum())

    return {
        "total": float(sum(by_removal.values())),
        "by_removal": by_removal,
        "scale": "sum of positive true-class log-probability differences",
    }
