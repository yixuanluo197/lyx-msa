import numpy as np

from analyze_cross_seed import binary_metrics, bootstrap_differences, logo_scores, probability_features


def toy():
    dialogues = np.repeat(["d0", "d1", "d2", "d3"], 4)
    probabilities = np.array([[.8, .2], [.6, .4], [.3, .7], [.45, .55]] * 4)
    labels = np.array([1, 1, 0, 1] * 4)
    correctness = (probabilities.argmax(1) == labels).astype(int)
    return dialogues, probabilities, correctness


def test_probability_features():
    _, p, _ = toy()
    features = probability_features(p)
    assert features.shape == (len(p), 3)
    assert np.allclose(features[:, 0], p.max(1))
    assert np.all(features[:, 1] >= 0)
    assert np.all(features[:, 2] >= 0)


def test_logo_scores_are_complete_and_deterministic():
    dialogues, p, y = toy()
    x = probability_features(p)
    first = logo_scores(x, y, dialogues)
    second = logo_scores(x, y, dialogues)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all() and np.all((first > 0) & (first < 1))


def test_binary_metrics_and_fixed_oof_bootstrap():
    dialogues, _, y = toy()
    baseline = np.linspace(.2, .8, len(y))
    candidate = np.clip(baseline + np.where(y, .05, -.05), 1e-4, 1-1e-4)
    values = binary_metrics(y, candidate)
    assert set(values) == {"auroc_success", "auprc_success", "binary_brier", "binary_nll"}
    intervals = bootstrap_differences(y, dialogues, baseline, candidate, 50, 7)
    assert intervals["binary_brier"][2] == 50
