"""Training-only confidence-conditioned mu/delta transforms and nested calibration."""
import sys
import warnings
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor'))
from mainline_core import (METRICS, PENALTIES, calibration_metrics, dump,
    markdown_table, metrics, select_penalty, sha)

FEATURES = ['gaussian_mu', 'gaussian_delta', 'dropout_mu', 'dropout_delta']
METHODS = ['M_joint', 'M_interaction']
OPTIMIZER = {'C': 1., 'solver': 'lbfgs', 'max_iter': 2000, 'tol': 1e-8}


def fit_transforms(frame, indices):
    """Fit C center and both scalers using exactly the supplied training rows."""
    train = frame.iloc[np.asarray(indices, dtype=int)]
    b = train[FEATURES].to_numpy(float)
    confidence = train.confidence.to_numpy(float)
    c0 = float(confidence.mean())
    scaler = StandardScaler().fit(b)
    # Preserve exact zero for truly constant columns despite summation roundoff.
    constant = np.ptp(b, axis=0) == 0
    scaler.mean_[constant] = b[0, constant]
    z = scaler.transform(b)
    h = (confidence - c0)[:, None] * z
    interaction_scaler = StandardScaler().fit(h)
    return {'c0': c0, 'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'interaction_scaler_mean': interaction_scaler.mean_.tolist(),
        'interaction_scaler_scale': interaction_scaler.scale_.tolist()}


def transform(frame, fitted):
    z = (frame[FEATURES].to_numpy(float) - fitted['scaler_mean']) / fitted['scaler_scale']
    h = (frame.confidence.to_numpy(float) - fitted['c0'])[:, None] * z
    t = (h - fitted['interaction_scaler_mean']) / fitted['interaction_scaler_scale']
    return z, t


def fit_model(frame, indices, method, penalty):
    """Raw C plus U/sqrt(n*lambda): shared behavioral L2, unpenalized intercept."""
    indices = np.asarray(indices, dtype=int)
    assert len(indices) and len(np.unique(indices)) == len(indices)
    train = frame.iloc[indices]; n = len(train)
    confidence = train.confidence.to_numpy(float)[:, None]
    if method == 'E1':
        assert penalty == 'off'
        transforms = {'c0': float(confidence.mean()), 'scaler_mean': [], 'scaler_scale': [],
            'interaction_scaler_mean': [], 'interaction_scaler_scale': []}
        features = []; divisor = 1.; design = confidence
    else:
        assert method in METHODS and penalty in PENALTIES[1:]
        transforms = fit_transforms(frame, indices)
        z, t = transform(train, transforms)
        behavior = np.column_stack([z, t]) if method == 'M_interaction' else z
        divisor = float(np.sqrt(n * float(penalty)))
        design = np.column_stack([confidence, behavior / divisor]); features = list(FEATURES)
    model = LogisticRegression(penalty='l2', **OPTIMIZER)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(design, train.correct.to_numpy(int))
    coefficients = model.coef_[0]
    effective = coefficients[1:] / divisor
    return dict(transforms, features=features, interaction=method == 'M_interaction',
        penalty=penalty, n_train=n, train_indices=indices.tolist(),
        train_dialogues=sorted(train.dialogue_id.unique().tolist()),
        design_divisor=divisor, raw_coefficients=coefficients.tolist(),
        intercept=float(model.intercept_[0]), confidence_coefficient=float(coefficients[0]),
        effective_behavior_coefficients=effective[:4].tolist(),
        effective_interaction_coefficients=effective[4:].tolist(),
        iterations=int(model.n_iter_[0]), optimizer=dict(OPTIMIZER))


def score(frame, fitted):
    base = fitted['intercept'] + fitted['confidence_coefficient'] * frame.confidence.to_numpy(float)
    additive = np.zeros(len(frame)); interaction = np.zeros(len(frame))
    if fitted['features']:
        z, t = transform(frame, fitted)
        additive = z @ np.asarray(fitted['effective_behavior_coefficients'])
        if fitted['interaction']:
            interaction = t @ np.asarray(fitted['effective_interaction_coefficients'])
    behavior = additive + interaction
    reliability = base + behavior
    q = expit(reliability)
    assert np.isfinite(q).all() and np.isfinite(reliability).all()
    return {'base_score': base, 'additive_score': additive, 'interaction_score': interaction,
        'behavior_score': behavior, 'reliability_score': reliability, 'oof_confidence': q}


def fit_outer(frame, train, held):
    """Nested grouped folds; cache E1 so each off path reuses its exact fit."""
    train = np.asarray(train, dtype=int); held = np.asarray(held, dtype=int)
    assert not set(train) & set(held)
    assert not set(frame.iloc[train].dialogue_id) & set(frame.iloc[held].dialogue_id)
    cache = {}
    def base(indices):
        key = tuple(int(i) for i in indices)
        if key not in cache:
            cache[key] = fit_model(frame, indices, 'E1', 'off')
        return cache[key]
    outer_base = base(train)
    result = {'train_indices': train.tolist(), 'held_indices': held.tolist(),
        'train_dialogues': sorted(frame.iloc[train].dialogue_id.unique().tolist()),
        'held_dialogues': sorted(frame.iloc[held].dialogue_id.unique().tolist()),
        'E1': outer_base, 'methods': {}}
    inner = []
    for k, (a, b) in enumerate(GroupKFold(5).split(train, groups=frame.iloc[train].dialogue_id)):
        it, iv = train[a], train[b]
        assert not set(frame.iloc[it].dialogue_id) & set(frame.iloc[iv].dialogue_id)
        inner.append({'inner_fold': k, 'train_indices': it.tolist(), 'held_indices': iv.tolist(),
            'train_dialogues': sorted(frame.iloc[it].dialogue_id.unique().tolist()),
            'held_dialogues': sorted(frame.iloc[iv].dialogue_id.unique().tolist()), 'E1': base(it)})
    result['inner_folds'] = inner
    outputs = {'E1': score(frame.iloc[held], outer_base)}
    for method in METHODS:
        losses = {}; candidates = {}
        for penalty in PENALTIES:
            squared = []; fits = []
            for fold in inner:
                fitted = fold['E1'] if penalty == 'off' else fit_model(frame, fold['train_indices'], method, penalty)
                check = frame.iloc[fold['held_indices']]
                q = score(check, fitted)['oof_confidence']
                residual = (q - check.correct.to_numpy(int))**2
                squared.extend(residual.tolist())
                fits.append({'inner_fold': fold['inner_fold'], 'fit': fitted,
                    'held_indices': fold['held_indices'], 'held_dialogues': fold['held_dialogues'],
                    'held_brier': float(np.mean(residual))})
            losses[str(penalty)] = float(np.mean(squared)); candidates[str(penalty)] = fits
        selected = select_penalty(losses)
        fitted = outer_base if selected == 'off' else fit_model(frame, train, method, selected)
        result['methods'][method] = {'features': list(FEATURES), 'selected_penalty': selected,
            'candidate_brier': losses, 'inner_candidate_fits': candidates, 'outer_fit': fitted}
        outputs[method] = score(frame.iloc[held], fitted)
        if selected == 'off':
            assert np.array_equal(outputs[method]['oof_confidence'], outputs['E1']['oof_confidence'])
    return outputs, result
