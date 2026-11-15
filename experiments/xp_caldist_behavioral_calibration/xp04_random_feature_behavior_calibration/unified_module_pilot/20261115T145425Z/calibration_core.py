"""Frozen-base, L2 offset correction with dialogue-nested selection; CPU only."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


EPS = 1e-15
LAMBDAS = ["off", 1.0, 0.1, 0.01]


def metrics(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, int)
    bins = np.minimum((p * 15).astype(int), 14)
    ece = sum(np.mean(bins == k) * abs(p[bins == k].mean() - y[bins == k].mean())
              for k in range(15) if np.any(bins == k))
    return {"ece15": float(ece), "binary_brier": float(np.mean((p - y) ** 2)),
            "binary_nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))}


def auxiliary(p, y, original, labels, predictions):
    p, y, original = np.asarray(p), np.asarray(y), np.asarray(original)
    return {"accuracy": float(np.mean(np.asarray(labels) == np.asarray(predictions))),
            "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "correctness_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else np.nan,
            "mean_confidence": float(p.mean()),
            "correct_mean_confidence": float(p[y == 1].mean()) if np.any(y == 1) else np.nan,
            "wrong_mean_confidence": float(p[y == 0].mean()) if np.any(y == 0) else np.nan,
            "raised_fraction": float(np.mean(p > original + 1e-12)),
            "lowered_fraction": float(np.mean(p < original - 1e-12))}


def base_design(c, name):
    x = np.asarray(c, float)
    if name == "L0":
        x = logit(np.clip(x, 1e-6, 1 - 1e-6))
    assert name in ("E1", "L0")
    return x.reshape(-1, 1)


def fit_base(c, y, name):
    x = base_design(c, name)
    mean, scale = np.zeros(1), np.ones(1)
    if name == "L0":
        s = StandardScaler().fit(x)
        mean, scale = s.mean_, s.scale_
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit((x - mean) / scale, y)
    assert int(model.n_iter_[0]) < 1000
    return {"name": name, "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficient": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0]),
            "iterations": int(model.n_iter_[0])}


def base_logits(c, params):
    x = base_design(c, params["name"])
    x = (x - np.asarray(params["mean"])) / np.asarray(params["scale"])
    return params["intercept"] + params["coefficient"] * x[:, 0]


def offset_objective(w, x, offset, y, penalty):
    t = offset + x @ w
    value = np.mean(np.logaddexp(0, t) - y * t) + penalty * np.dot(w, w) / 2
    gradient = x.T @ (expit(t) - y) / len(y) + penalty * w
    return value, gradient


def fit_correction(b, offset, y, penalty):
    scaler = StandardScaler().fit(b)
    x = scaler.transform(b)
    w = np.zeros(x.shape[1])
    iterations = 0
    objective = None
    gradient_norm = 0.0
    if penalty != "off":
        # Four parameters and strictly positive L2 permit a stable exact Hessian.
        # Damped Newton keeps the registered objective unchanged.
        target = np.asarray(y, float)
        lam = float(penalty)
        for iterations in range(100):
            value, gradient = offset_objective(w, x, offset, target, lam)
            if np.max(np.abs(gradient)) < 1e-8:
                break
            p = expit(offset + x @ w)
            hessian = (x.T * (p * (1 - p))) @ x / len(target) + lam * np.eye(len(w))
            step = np.linalg.solve(hessian, gradient)
            rate = 1.0
            directional = gradient @ step
            while rate > 2 ** -40:
                trial = w - rate * step
                trial_value = offset_objective(trial, x, offset, target, lam)[0]
                if trial_value <= value - 1e-4 * rate * directional:
                    w = trial
                    break
                rate /= 2
            else:
                raise RuntimeError(f"Newton line search stalled; gradient={np.max(np.abs(gradient))}")
        objective, gradient = offset_objective(w, x, offset, target, lam)
        gradient_norm = float(np.max(np.abs(gradient)))
        assert gradient_norm < 1e-7, gradient_norm
        assert np.isfinite(w).all()
    return {"penalty": penalty, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "weights": w.tolist(), "iterations": iterations,
            "gradient_max_abs": gradient_norm, "objective": float(objective) if objective is not None else None,
            "solver": "damped_newton_exact_hessian"}


def correction_logits(b, base_offset, params):
    z = (np.asarray(b) - np.asarray(params["mean"])) / np.asarray(params["scale"])
    return base_offset + z @ np.asarray(params["weights"])


def fit_nested_fold(c_train, b_train, y_train, groups_train, base_name):
    """Input is only outer training rows; no outer held labels enter this API."""
    c_train, b_train = np.asarray(c_train), np.asarray(b_train)
    y_train, groups_train = np.asarray(y_train), np.asarray(groups_train)
    inner = list(GroupKFold(n_splits=min(5, len(set(groups_train))))
                 .split(c_train, groups=groups_train))
    inner_predictions = {str(lam): np.full(len(c_train), np.nan) for lam in LAMBDAS}
    fold_records = []
    for fold, (tr, ho) in enumerate(inner):
        assert set(groups_train[tr]).isdisjoint(groups_train[ho])
        assert len(np.unique(y_train[tr])) == 2
        bp = fit_base(c_train[tr], y_train[tr], base_name)
        offset_train, offset_held = base_logits(c_train[tr], bp), base_logits(c_train[ho], bp)
        corrections = {}
        for lam in LAMBDAS:
            cp = fit_correction(b_train[tr], offset_train, y_train[tr], lam)
            inner_predictions[str(lam)][ho] = expit(correction_logits(b_train[ho], offset_held, cp))
            corrections[str(lam)] = cp
        fold_records.append({"fold": fold, "train_dialogues": sorted(set(groups_train[tr])),
                             "held_dialogues": sorted(set(groups_train[ho])), "train_n": len(tr),
                             "held_n": len(ho), "base": bp, "corrections": corrections})
    selected = "off"
    best = np.inf
    candidates = []
    for lam in LAMBDAS:
        scores = inner_predictions[str(lam)]
        assert np.isfinite(scores).all()
        ms = metrics(scores, y_train)
        candidates.append({"penalty": lam, **ms})
        if ms["binary_brier"] < best - 1e-12:
            selected, best = lam, ms["binary_brier"]
    base = fit_base(c_train, y_train, base_name)
    correction = fit_correction(b_train, base_logits(c_train, base), y_train, selected)
    return {"base": base, "correction": correction, "selected_penalty": selected,
            "candidates": candidates, "inner_folds": fold_records}


def predict(c, b, fit):
    return expit(correction_logits(b, base_logits(c, fit["base"]), fit["correction"]))
