import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation_metrics import calibration_metrics, confidence_enhancement_level


def test_calibration_metrics_use_max_probability_and_correctness():
    labels = np.array([0, 1, 1, 0])
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.3, 0.7],
        ]
    )

    metrics = calibration_metrics(labels, probabilities, n_bins=2)

    # All samples fall in (0.5, 1.0]: accuracy=0.5, mean confidence=0.75.
    assert metrics["ece"] == 25.0
    assert metrics["mce"] == 25.0
    assert metrics["rmsce"] == 25.0
    assert metrics["auroc_success"] == 100.0


def test_cel_sums_only_confidence_increases_for_true_classes():
    labels = np.array([0, 1])
    full = np.log(np.array([[0.6, 0.4], [0.3, 0.7]]))
    removed = {
        "audio": np.log(np.array([[0.8, 0.2], [0.4, 0.6]])),
        "visual": np.log(np.array([[0.5, 0.5], [0.2, 0.8]])),
    }

    cel = confidence_enhancement_level(labels, full, removed)

    expected = np.log(0.8) - np.log(0.6) + np.log(0.8) - np.log(0.7)
    assert np.isclose(cel["total"], expected)
    assert np.isclose(cel["by_removal"]["audio"], np.log(0.8) - np.log(0.6))
    assert np.isclose(cel["by_removal"]["visual"], np.log(0.8) - np.log(0.7))
