"""
The CaliDist calibration itself (paper Section 3.3, Eq. 1-6).

For each sample:
    mu     = (1/k) * sum over distracted answers that differ from the original
             prediction of their confidence                          (Eq. 1)
    delta  = |original confidence - mean distracted confidence|      (Eq. 2)
    lambda = (1 - mu) / (delta + epsilon)                            (Eq. 3)
lambda is min-max normalized to [0, 10] with the dev set's min and max (Eq. 4),
and the calibrated confidence is

    sigmoid(lambda, alpha, beta) * original confidence               (Eq. 5)
    sigmoid(lambda, alpha, beta) = 1 / (1 + exp(-beta * (lambda - alpha)))   (Eq. 6)

alpha and beta are chosen by grid search for the lowest Brier score on dev.
"""
import numpy as np

EPSILON = 1e-10

ALPHA_CANDIDATES = np.linspace(-5.0, 5.0, 100)
BETA_CANDIDATES = np.linspace(0.1, 5.0, 100)


def get_mu(original_pred, distracted_preds, distracted_confs):
    """Prediction instability (Eq. 1). Averages over the parsed distracted answers."""
    if len(distracted_preds) == 0:
        raise ValueError("mu needs at least one distracted answer")
    confidence_tally = 0.0
    for i in range(len(distracted_preds)):
        if str(distracted_preds[i]) != str(original_pred):
            confidence_tally += distracted_confs[i]
    return confidence_tally / len(distracted_preds)


def get_delta(original_conf, distracted_confs):
    """Confidence instability (Eq. 2)."""
    mean_distracted_conf = np.mean(distracted_confs)
    return float(np.abs(original_conf - mean_distracted_conf))


def get_raw_lambda(mu, delta):
    """Reliability score before normalization (Eq. 3)."""
    return (1 - mu) / (delta + EPSILON)


def get_lambda_bounds(raw_lambdas_dev):
    """Min and max of raw lambda on the dev set; reused unchanged for test."""
    return float(np.min(raw_lambdas_dev)), float(np.max(raw_lambdas_dev))


def normalize_lambdas(raw_lambdas, lambda_min, lambda_max):
    """
    Min-max normalization to [0, 10] using the dev bounds (Eq. 4).
    Test values are not clipped, so they can fall outside [0, 10].
    """
    raw_lambdas = np.array(raw_lambdas, dtype=float)
    return (raw_lambdas - lambda_min) * 10 / (lambda_max - lambda_min + 1e-15)


def sigmoid(lambdas, alpha, beta):
    """Scaling factor (Eq. 6)."""
    return 1.0 / (1.0 + np.exp(-beta * (lambdas - alpha)))


def calibrate(original_confs, lambdas, alpha, beta):
    """Calibrated confidence (Eq. 5)."""
    original_confs = np.array(original_confs, dtype=float)
    return original_confs * sigmoid(np.array(lambdas, dtype=float), alpha, beta)


def find_alpha_beta(lambdas_dev, original_confs_dev, is_correct_dev):
    """
    Grid search over 100 alpha values in [-5, 5] and 100 beta values in [0.1, 5]
    for the lowest Brier score on dev. Returns (alpha, beta, best Brier score).
    """
    lambdas_dev = np.array(lambdas_dev, dtype=float)
    original_confs_dev = np.array(original_confs_dev, dtype=float)
    is_correct_dev = np.array(is_correct_dev, dtype=float)

    best_brier = float("inf")
    best_alpha = None
    best_beta = None
    for alpha in ALPHA_CANDIDATES:
        for beta in BETA_CANDIDATES:
            calibrated = calibrate(original_confs_dev, lambdas_dev, alpha, beta)
            brier = np.mean((calibrated - is_correct_dev) ** 2)
            if brier < best_brier:
                best_brier = brier
                best_alpha = float(alpha)
                best_beta = float(beta)
    return best_alpha, best_beta, float(best_brier)
