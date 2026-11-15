"""Registered conditional calibration. Every fitting API receives training rows only."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('frozen_correction_core', ROOT / 'provenance/frozen_correction_core.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
C_GRID = [0.01, 0.1, 1.0, 10.0]
LAMBDAS = ['off', 1.0, 0.1, 0.01]
BEHAVIOR = ['gaussian_D', 'gaussian_A', 'dropout_D', 'dropout_A']


def feature_columns(dataset):
    return {
        'T': ['logit_C'],
        'S': ['logit_C', 'margin', 'normalized_entropy'] + [f'pred_class_{k}' for k in range(1, 6 if dataset == 'IEMOCAP' else 7)],
    }


def fit_logistic(x, y, c):
    x, y = np.asarray(x, float), np.asarray(y, int)
    assert np.isfinite(x).all() and set(np.unique(y)) == {0, 1}
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=c, max_iter=2000, solver='lbfgs', tol=1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(scaler.transform(x), y)
    assert int(model.n_iter_[0]) < 2000
    return {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'weights': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]),
            'iterations': int(model.n_iter_[0]), 'train_n': len(y),
            'parameter_count': x.shape[1] + 1}


def logistic_logits(x, params):
    return ((np.asarray(x, float) - params['mean']) / params['scale']) @ np.asarray(params['weights']) + params['intercept']


def choose_candidate(candidates, scores, tolerance=1e-12):
    """Input order is the preregistered tie order: off, strong lambda, small C."""
    assert len(candidates) == len(scores) and np.isfinite(scores).all()
    index, best = 0, np.inf
    for j, value in enumerate(scores):
        if value < best - tolerance:
            index, best = j, value
    return index


def candidates_for(method, dataset):
    cols = feature_columns(dataset)
    if method in ['T', 'S']:
        return [{'kind': 'joint', 'columns': cols[method], 'C': c, 'mode': 'base'} for c in C_GRID]
    if method == 'S_DA_OFFSET':
        # off first globally, then stronger correction shrinkage, then smaller C.
        return [{'kind': 'offset', 'columns': cols['S'], 'C': c, 'mode': 'base' if lam == 'off' else 'augmented', 'penalty': lam}
                for lam in LAMBDAS for c in C_GRID]
    base = 'T' if method == 'T_DA' else 'S'
    extra = [f'rand_{k}' for k in range(4)] if method == 'S_RAND' else BEHAVIOR
    return ([{'kind': 'joint', 'columns': cols[base], 'C': c, 'mode': 'base'} for c in C_GRID]
            + [{'kind': 'joint', 'columns': cols[base] + extra, 'C': c, 'mode': 'augmented'} for c in C_GRID])


def fit_candidate(frame, y, candidate, cache=None):
    columns = candidate['columns']
    key = (tuple(columns), candidate['C'])
    if cache is not None and key in cache:
        base = cache[key]
    else:
        base = fit_logistic(frame[columns].to_numpy(), y, candidate['C'])
        if cache is not None:
            cache[key] = base
    fit = {**candidate, 'base': base}
    if candidate['kind'] == 'offset':
        fit['correction'] = frozen.fit_correction(frame[BEHAVIOR].to_numpy(), logistic_logits(frame[columns].to_numpy(), base), y, candidate['penalty'])
    return fit


def predict_logits(frame, fit):
    offset = logistic_logits(frame[fit['columns']].to_numpy(), fit['base'])
    if fit['kind'] == 'offset':
        offset = frozen.correction_logits(frame[BEHAVIOR].to_numpy(), offset, fit['correction'])
    return offset


def predict(frame, fit):
    return expit(predict_logits(frame, fit))


def fit_nested_all(frame, y, groups, dataset):
    """No outer holdout frame/labels accepted, preventing selection leakage by API."""
    methods = ['T', 'T_DA', 'S', 'S_DA', 'S_DA_OFFSET', 'S_RAND']
    y, groups = np.asarray(y, int), np.asarray(groups, str)
    candidates = {m: candidates_for(m, dataset) for m in methods}
    predictions = {m: np.full((len(v), len(y)), np.nan) for m, v in candidates.items()}
    inner_records = []
    for inner_fold, (tr, ho) in enumerate(GroupKFold(5).split(frame, groups=groups)):
        assert set(groups[tr]).isdisjoint(groups[ho])
        inner = {'fold': inner_fold, 'train_indices': tr.tolist(), 'held_indices': ho.tolist(),
                 'train_dialogues': sorted(set(groups[tr])), 'held_dialogues': sorted(set(groups[ho])), 'models': {}}
        cache = {}
        for method in methods:
            fits = []
            for j, candidate in enumerate(candidates[method]):
                fit = fit_candidate(frame.iloc[tr], y[tr], candidate, cache)
                predictions[method][j, ho] = predict(frame.iloc[ho], fit)
                fits.append(fit)
            inner['models'][method] = fits
        inner_records.append(inner)
    result = {'inner_folds': inner_records, 'methods': {}}
    cache = {}
    for method in methods:
        ps = predictions[method]
        assert np.isfinite(ps).all()
        scores = ((ps - y[None, :]) ** 2).mean(axis=1)
        selected = choose_candidate(candidates[method], scores)
        fit = fit_candidate(frame, y, candidates[method][selected], cache)
        train_p = predict(frame, fit)
        result['methods'][method] = {'selected_index': selected, 'candidates': [dict(c, binary_brier=float(s)) for c, s in zip(candidates[method], scores)],
                                     'fit': fit, 'inner_metrics': frozen.metrics(ps[selected], y),
                                     'train_metrics': frozen.metrics(train_p, y)}
    return result
