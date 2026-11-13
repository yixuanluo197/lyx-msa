"""Dialogue-held-out confidence-only and exact CaliDist calibration for Pilot."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

OUT = Path(__file__).resolve().parent
ALPHAS = np.linspace(-5.0, 5.0, 100)
BETAS = np.linspace(0.1, 5.0, 100)
GRID_ALPHA, GRID_BETA = np.meshgrid(ALPHAS, BETAS, indexing='ij')
GRID_ALPHA, GRID_BETA = GRID_ALPHA.ravel(), GRID_BETA.ravel()
EPSILON = 1e-10


def metrics(confidence, correct):
    p = np.clip(np.asarray(confidence, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(correct, dtype=float)
    bins = np.minimum(np.floor(p * 15).astype(int), 14)
    ece = sum((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean())
              for b in range(15) if np.any(bins == b))
    return dict(ece15=float(ece), binary_brier=float(np.mean((p - y) ** 2)),
                binary_nll=float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))))


def fit_calidist(train_lambda, train_conf, train_correct):
    best = (float('inf'), None)
    for begin in range(0, len(GRID_ALPHA), 250):
        end = begin + 250
        alpha = GRID_ALPHA[begin:end, None]
        beta = GRID_BETA[begin:end, None]
        scaled = train_conf[None, :] * expit(beta * (train_lambda[None, :] - alpha))
        loss = np.mean((scaled - train_correct[None, :]) ** 2, axis=1)
        index = int(np.argmin(loss))
        if float(loss[index]) < best[0]:
            best = (float(loss[index]), begin + index)
    i = best[1]
    assert i is not None
    return float(GRID_ALPHA[i]), float(GRID_BETA[i]), best[0]


def analyze_dataset(name):
    folder = OUT / name
    assert json.loads((folder / 'validation_parity.json').read_text())['completed']
    original = pd.read_csv(folder / 'original_predictions.csv', dtype={'dialogue_id': str})
    behavior = pd.read_csv(folder / 'behavior_features.csv', dtype={'dialogue_id': str})
    assert len(original) == (527 if name == 'IEMOCAP' else 1109)
    groups = original.dialogue_id.to_numpy()
    splits = list((LeaveOneGroupOut() if name == 'IEMOCAP' else GroupKFold(n_splits=5))
                  .split(original, groups=groups))
    assert len(splits) == (12 if name == 'IEMOCAP' else 5)
    for train, held in splits:
        assert not (set(groups[train]) & set(groups[held]))
    conf = original.confidence.to_numpy(float)
    correct = (original.prediction == original.label).to_numpy(int)
    e1 = np.full(len(original), np.nan)
    rows = []
    fold_records = []
    for fold, (train, held) in enumerate(splits):
        model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        model.fit(conf[train, None], correct[train])
        e1[held] = model.predict_proba(conf[held, None])[:, 1]
        fold_records.append(dict(dataset=name, fold=fold, method='E1',
                                 train_dialogues=sorted(set(groups[train])),
                                 held_out_dialogues=sorted(set(groups[held])),
                                 intercept=float(model.intercept_[0]),
                                 coefficient=float(model.coef_[0, 0])))
    assert np.isfinite(e1).all()
    for i in range(len(original)):
        rows.extend([dict(dataset=name, dialogue_id=groups[i], utterance_index=int(original.utterance_index[i]),
                          label=int(original.label[i]), prediction=int(original.prediction[i]),
                          correct=int(correct[i]), method=method, corruption='none', modality='none',
                          confidence=float(score)) for method, score in [('E0', conf[i]), ('E1', e1[i])]])
    reliability_rows = []
    for (kind, modality), subset in behavior.groupby(['corruption', 'modality'], sort=True):
        merged = original.merge(subset, on=['dataset', 'seed', 'dialogue_id', 'utterance_index'],
                                validate='one_to_one', suffixes=('', '_behavior'))
        assert len(merged) == len(original)
        assert np.allclose(merged.original_confidence.to_numpy(float), conf)
        mu = merged.mu.to_numpy(float)
        delta = merged.delta.to_numpy(float)
        raw = (1 - mu) / (delta + EPSILON)
        e2 = np.full(len(original), np.nan)
        lambdas = np.full(len(original), np.nan)
        for fold, (train, held) in enumerate(splits):
            minimum, maximum = float(raw[train].min()), float(raw[train].max())
            denominator = maximum - minimum + 1e-15
            ltrain = 10 * (raw[train] - minimum) / denominator
            lheld = 10 * (raw[held] - minimum) / denominator
            alpha, beta, train_brier = fit_calidist(ltrain, conf[train], correct[train])
            e2[held] = conf[held] * expit(beta * (lheld - alpha))
            lambdas[held] = lheld
            fold_records.append(dict(dataset=name, fold=fold, method='E2', corruption=kind,
                                     modality=modality, train_dialogues=sorted(set(groups[train])),
                                     held_out_dialogues=sorted(set(groups[held])),
                                     lambda_min=minimum, lambda_max=maximum,
                                     alpha=alpha, beta=beta, training_brier=train_brier))
        assert np.isfinite(e2).all() and np.all(e2 <= conf + 1e-12)
        for i in range(len(original)):
            common = dict(dataset=name, dialogue_id=groups[i], utterance_index=int(original.utterance_index[i]),
                          label=int(original.label[i]), prediction=int(original.prediction[i]),
                          correct=int(correct[i]), method='E2', corruption=kind, modality=modality)
            rows.append({**common, 'confidence': float(e2[i])})
            reliability_rows.append({**common, 'original_confidence': float(conf[i]),
                                     'mu': float(mu[i]), 'delta': float(delta[i]),
                                     'lambda_raw': float(raw[i]), 'lambda_oof': float(lambdas[i]),
                                     'calibrated_confidence': float(e2[i])})
    predictions = pd.DataFrame(rows)
    predictions.to_csv(folder / 'oof_calibration_predictions.csv', index=False)
    pd.DataFrame(reliability_rows).to_csv(folder / 'reliability_scores.csv', index=False)
    (folder / 'fold_parameters.json').write_text(json.dumps(fold_records, indent=2) + '\n')
    accuracy = accuracy_score(original.label, original.prediction)
    weighted = f1_score(original.label, original.prediction, average='weighted')
    macro = f1_score(original.label, original.prediction, average='macro')
    summaries = []
    for (method, kind, modality), subset in predictions.groupby(['method', 'corruption', 'modality'], sort=True):
        assert len(subset) == len(original)
        values = metrics(subset.confidence, subset.correct)
        summaries.append(dict(dataset=name, method=method, corruption=kind, modality=modality,
                              utterances=len(subset), dialogues=len(set(groups)),
                              accuracy=accuracy, weighted_f1=weighted, macro_f1=macro, **values))
    pd.DataFrame(summaries).to_csv(folder / 'calibration_metrics.csv', index=False)
    return summaries


def main():
    all_summaries = []
    for name in ('IEMOCAP', 'MELD'):
        all_summaries.extend(analyze_dataset(name))
    pd.DataFrame(all_summaries).to_csv(OUT / 'calibration_metrics.csv', index=False)
    print(pd.DataFrame(all_summaries).to_string(index=False))


if __name__ == '__main__':
    main()
