"""CPU analysis of the preregistered Seed1475 strength sensitivity experiment.

No inference, new random perturbations, CMERC fitting, or Test data access.
All repeated perturbations are aggregated before any reliability fitting.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

for _thread in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_thread] = '1'

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import rel_entr
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from mainline_core import FEATURES, METRICS, calibration_metrics, dump, fit_outer, metrics, sha

KEY = ['dataset', 'seed', 'dialogue_id', 'utterance_index']
CONDITION = ['dataset', 'seed', 'modality', 'strength']
GROUP = CONDITION + ['corruption']
METHODS = ['E0', 'E1', 'M_joint', 'M_gaussian', 'M_dropout']
BEHAVIOR_METHODS = METHODS[2:]
MEASURES = ['mu', 'delta', 'js', 'flip_rate']
RESPONSE_MEASURES = ['mu', 'delta', 'js', 'flip_rate', 'mu_zero_fraction', 'nonzero_mu_fraction']


def read(path, **kwargs):
    return pd.read_csv(path, dtype={'dataset': str, 'dialogue_id': str}, float_precision='round_trip', **kwargs)


def checked_bool(values):
    if values.isna().any() or not values.isin([True, False]).all():
        raise ValueError('Missing or invalid boolean flag')
    return values.astype(bool)


def unique(frame, columns):
    if frame.duplicated(columns).any():
        raise ValueError(f'duplicate identity: {columns}')


def response(confidence, prediction, probabilities, original_probabilities):
    """Original CaliDist definitions; confidence SD uses population ddof=0."""
    probabilities = np.asarray(probabilities, float)
    original_probabilities = np.asarray(original_probabilities, float)
    maxima = probabilities.max(axis=1)
    flips = probabilities.argmax(axis=1) != prediction
    midpoint = (probabilities + original_probabilities[None, :]) / 2
    js = .5 * np.sum(rel_entr(probabilities, midpoint) + rel_entr(original_probabilities[None, :], midpoint), axis=1)
    result = dict(mu=float(np.mean(flips * maxima)), delta=float(abs(confidence - maxima.mean())),
                  js=float(js.mean()), flip_rate=float(flips.mean()),
                  confidence_change_mean=float(maxima.mean() - confidence),
                  confidence_change_std=float(maxima.std(ddof=0)))
    for suffix, indices in [('left', slice(0, 5)), ('right', slice(5, 10))]:
        if len(maxima) == 10:
            result[f'mu_{suffix}'] = float(np.mean((flips * maxima)[indices]))
            result[f'delta_{suffix}'] = float(abs(confidence - maxima[indices].mean()))
            result[f'js_{suffix}'] = float(js[indices].mean())
            result[f'flip_rate_{suffix}'] = float(flips[indices].mean())
    return result


def validate_probabilities(probabilities, prediction, confidence, classes):
    p = np.asarray(probabilities, float)
    if p.ndim != 2 or p.shape[1] != classes or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('invalid full probability vector')
    if not np.allclose(p.sum(axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError('probabilities do not sum to one')
    if not np.array_equal(p.argmax(axis=1), np.asarray(prediction)):
        raise ValueError('prediction is not argmax')
    if not np.allclose(p.max(axis=1), confidence, atol=1e-12, rtol=0):
        raise ValueError('confidence is not probability maximum')
    return p


def load_originals(config):
    frames = []
    historical = read(Path(config['source_mainline']) / 'inputs/modality_features.csv')
    flags = historical[historical.modality.eq('joint')][KEY + ['visual_missing']]
    unique(flags, KEY)
    for ds, spec in config['datasets'].items():
        f = read(Path(config['source_pilot']) / ds / 'original_predictions.csv')
        unique(f, KEY)
        if len(f) != spec['utterances'] or f.dialogue_id.nunique() != spec['dialogues']:
            raise ValueError(f'{ds}: original coverage mismatch')
        if set(f.dataset) != {ds} or set(f.seed) != {1475}:
            raise ValueError('original identity mismatch')
        p = validate_probabilities([json.loads(v) for v in f.probabilities], f.prediction, f.confidence,
                                   6 if ds == 'IEMOCAP' else 7)
        if not np.all((f.label >= 0) & (f.label < p.shape[1])):
            raise ValueError('invalid original class label')
        f['correct'] = f.prediction.eq(f.label).astype(int)
        f = f.merge(flags, on=KEY, how='left', validate='one_to_one')
        f['visual_missing'] = checked_bool(f.visual_missing)
        if int(f.visual_missing.sum()) != spec['visual_missing']:
            raise ValueError('visual missing count differs from protocol')
        frames.append(f)
    return pd.concat(frames, ignore_index=True), historical


def validate_original_parity(path, original):
    f = read(path)
    unique(f, KEY)
    merged = original.merge(f, on=KEY, validate='one_to_one', suffixes=('_old', '_new'))
    if len(merged) != len(original) or len(f) != len(original):
        raise ValueError('new-run original coverage mismatch')
    for col in ['label', 'prediction']:
        if not merged[f'{col}_old'].equals(merged[f'{col}_new']):
            raise ValueError(f'original {col} changed')
    for old, new in zip(merged.probabilities_old, merged.probabilities_new):
        if not np.allclose(json.loads(old), json.loads(new), atol=1e-12, rtol=0):
            raise ValueError('new-run original probabilities changed')
    if not np.allclose(merged.confidence_old, merged.confidence_new, atol=1e-12, rtol=0):
        raise ValueError('new-run original confidence changed')


def build_behavior(root, config, include_new=True):
    originals, historical = load_originals(config)
    sources = [root / 'preregistration.json', root / 'EXPERIMENT_PLAN.md',
               Path(config['source_mainline']) / 'inputs/modality_features.csv']
    records, source_counts = [], []
    for ds, spec in config['datasets'].items():
        origin = originals[originals.dataset.eq(ds)].set_index(KEY)
        original_path = Path(config['source_pilot']) / ds / 'original_predictions.csv'
        sources.append(original_path)
        source_specs = [(Path(config['source_pilot']) / ds / 'perturbed_predictions.csv', .05, None)]
        if include_new:
            source_specs += [(root / 'runs' / ds / mod / 'perturbed_predictions.csv', .10, mod)
                             for mod in config['modalities']]
        for path, strength, expected_modality in source_specs:
            if expected_modality:
                completion = path.parent / 'complete.json'
                if not completion.exists():
                    raise FileNotFoundError(f'Inference not complete: {completion}')
                sources.append(completion)
                run_original = path.parent / 'original_predictions.csv'
                validate_original_parity(run_original, originals[originals.dataset.eq(ds)])
                sources.append(run_original)
            sources.append(path)
            raw = read(path, usecols=KEY + ['modality', 'corruption', 'strength', 'repeat_id',
                                         'visual_missing_skipped', 'prediction', 'confidence', 'probabilities'])
            expected_modalities = set(config['modalities']) if expected_modality is None else {expected_modality}
            unique(raw, KEY + ['modality', 'corruption', 'strength', 'repeat_id'])
            if set(raw.dataset) != {ds} or set(raw.seed) != {1475}:
                raise ValueError('perturbed dataset/seed mismatch')
            if set(raw.modality) != expected_modalities or set(raw.corruption) != {'gaussian', 'dropout'}:
                raise ValueError('condition coverage mismatch')
            if not raw.strength.eq(strength).all() or len(raw) != len(origin) * 20 * len(expected_modalities):
                raise ValueError('strength/repeat row count mismatch')
            if set(map(tuple, raw[KEY].drop_duplicates().to_numpy())) != set(origin.index):
                raise ValueError('perturbed utterance IDs differ from originals')
            raw['visual_missing_skipped'] = checked_bool(raw.visual_missing_skipped)
            for group_key, group in raw.groupby(KEY + ['modality', 'corruption'], sort=False):
                group = group.sort_values('repeat_id')
                if group.repeat_id.tolist() != list(range(10)):
                    raise ValueError('each utterance/condition requires distinct repeats 0..9')
                source = origin.loc[group_key[:4]]
                probs = validate_probabilities([json.loads(v) for v in group.probabilities],
                                               group.prediction, group.confidence, 6 if ds == 'IEMOCAP' else 7)
                missing, modality = bool(source.visual_missing), group_key[4]
                skipped = modality == 'visual' and missing
                raw_visual_skipped = missing and modality in ['visual', 'joint']
                if not group.visual_missing_skipped.eq(raw_visual_skipped).all():
                    raise ValueError('visual skip flag mismatch')
                original_probs = np.asarray(json.loads(source.probabilities), float)
                if skipped and not np.allclose(probs, original_probs[None, :], atol=1e-12, rtol=0):
                    raise ValueError('missing visual must preserve original probabilities')
                row = dict(zip(KEY + ['modality', 'corruption'], group_key))
                row.update(strength=strength, label=int(source.label), prediction=int(source.prediction),
                           confidence=float(source.confidence), correct=int(source.correct), visual_missing=missing,
                           perturbation_skipped=skipped, original_confidence=float(source.confidence), repeat_count=10)
                row.update(response(float(source.confidence), int(source.prediction), probs, original_probs))
                records.append(row)
            source_counts.append(dict(dataset=ds, strength=strength, modality=expected_modality or 'all',
                                      path=str(path), raw_rows=len(raw)))
            print(f'behavior verified {ds} strength={strength} modality={expected_modality or "all"}: {len(raw)} repeats', flush=True)
    behavior = pd.DataFrame(records)
    unique(behavior, KEY + ['modality', 'strength', 'corruption'])
    expected = 26176 if include_new else 13088
    if len(behavior) != expected:
        raise ValueError('aggregated behavior row count mismatch')
    feature_records = []
    for key, group in behavior.groupby(KEY + ['modality', 'strength'], sort=False):
        row = dict(zip(KEY + ['modality', 'strength'], key))
        for col in ['label', 'prediction', 'correct', 'confidence', 'visual_missing', 'perturbation_skipped']:
            if group[col].nunique() != 1:
                raise ValueError('original data changed across corruption')
            row[col] = group[col].iloc[0]
        for _, part in group.iterrows():
            for col in MEASURES + ['confidence_change_mean', 'confidence_change_std']:
                row[f'{part.corruption}_{col}'] = part[col]
        feature_records.append(row)
    features = pd.DataFrame(feature_records)
    # Preserve exact historical fitting order; GroupKFold tie behavior depends on order.
    historical_order = historical[KEY + ['modality']].copy()
    historical_order['_historical_order'] = np.arange(len(historical_order))
    features = features.merge(historical_order, on=KEY + ['modality'], validate='many_to_one').sort_values(
        ['strength', '_historical_order'], kind='stable').drop(columns='_historical_order').reset_index(drop=True)
    parity = features[features.strength.eq(.05)].merge(historical, on=KEY + ['modality'],
                   suffixes=('_new', '_old'), validate='one_to_one')
    feature_error = {}
    for col in FEATURES['M_joint']:
        error = float(np.max(np.abs(parity[f'{col}_new'] - parity[f'{col}_old'])))
        if error >= 1e-12:
            raise ValueError(f'historical feature derivation mismatch: {col} {error}')
        feature_error[col] = error
    return originals, behavior, features, sources, dict(source_rows=source_counts,
            historical_feature_max_errors=feature_error, raw_rows=sum(v['raw_rows'] for v in source_counts))


def populations(frame):
    yield 'all_retained', frame
    if frame.modality.iloc[0] in ['visual', 'joint']:
        yield 'valid_visual', frame[~frame.visual_missing]


def safe_spearman(left, right):
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return np.nan
    return float(spearmanr(left, right).statistic)


def match_confidences(correct, wrong, caliper=.02):
    """Hungarian max-cardinality matching, then minimal raw C distance."""
    correct, wrong = np.asarray(correct, float), np.asarray(wrong, float)
    if not len(correct) or not len(wrong):
        return []
    distance = np.abs(correct[:, None] - wrong[None, :])
    dummy = 1 + (min(len(correct), len(wrong)) + 1) * caliper
    cost = np.full((len(correct), len(wrong) + len(correct)), dummy)
    cost[:, :len(wrong)] = np.where(distance <= caliper, distance, 1e6)
    i, j = linear_sum_assignment(cost)
    return [(int(a), int(b)) for a, b in zip(i, j) if b < len(wrong) and distance[a, b] <= caliper]


def matching_indices(originals, valid_visual=False):
    result = []
    selected = originals[~originals.visual_missing] if valid_visual else originals
    for (ds, dialogue), f in selected.groupby(['dataset', 'dialogue_id'], sort=True):
        correct = f[f.correct.eq(1)].sort_values('utterance_index')
        wrong = f[f.correct.eq(0)].sort_values('utterance_index')
        for i, j in match_confidences(correct.confidence, wrong.confidence):
            a, b = correct.iloc[i], wrong.iloc[j]
            result.append(dict(dataset=ds, dialogue_id=dialogue, correct_utterance_index=int(a.utterance_index),
                               wrong_utterance_index=int(b.utterance_index), correct_confidence=float(a.confidence),
                               wrong_confidence=float(b.confidence), confidence_distance=abs(float(a.confidence-b.confidence))))
    return pd.DataFrame(result)


def descriptive_tables(originals, behavior):
    summaries, stability, bins, pairs, matched_summary = [], [], [], [], []
    match_sets = {name: matching_indices(originals, name == 'valid_visual') for name in ['all_retained', 'valid_visual']}
    for keys, condition in behavior.groupby(GROUP, sort=True):
        common = dict(zip(GROUP, keys))
        for population, f in populations(condition):
            base = dict(common, population=population, utterances=len(f), dialogues=f.dialogue_id.nunique(),
                        n_correct=int(f.correct.sum()), n_wrong=int((f.correct == 0).sum()),
                        missing_visual=int(f.visual_missing.sum()), skipped_visual=int(f.perturbation_skipped.sum()))
            summaries.append(dict(base, mu_zero_fraction=float(f.mu.eq(0).mean()), nonzero_mu_fraction=float(f.mu.ne(0).mean()),
                mean_mu=float(f.mu.mean()), mean_delta=float(f.delta.mean()), mean_js=float(f.js.mean()),
                mean_flip_rate=float(f.flip_rate.mean()), confidence_change_mean=float(f.confidence_change_mean.mean()),
                mean_confidence_change_std_ddof0=float(f.confidence_change_std.mean())))
            for measure in MEASURES:
                left, right = f[f'{measure}_left'].to_numpy(float), f[f'{measure}_right'].to_numpy(float)
                auc = float(roc_auc_score(1-f.correct, f[measure])) if f.correct.nunique() == 2 else np.nan
                stability.append(dict(base, measure=measure, split='repeat_0_to_4_vs_5_to_9',
                    spearman=safe_spearman(left, right), left_zero_fraction=float(np.mean(left == 0)),
                    right_zero_fraction=float(np.mean(right == 0)), full_zero_fraction=float(f[measure].eq(0).mean()),
                    auc_larger_is_error=auc, auc_direction='fixed_larger_is_error', n_repeats_each_half=5))
                ids = np.minimum(np.searchsorted([0, .4, .6, .8, 1.], f.confidence, side='right')-1, 3)
                for bin_id, (lower, upper) in enumerate(zip([0, .4, .6, .8], [.4, .6, .8, 1.])):
                    part = f[ids == bin_id]
                    good, bad = part[part.correct.eq(1)], part[part.correct.eq(0)]
                    bins.append(dict(common, population=population, measure=measure, bin_left=lower, bin_right=upper,
                        n_correct=len(good), n_wrong=len(bad), dialogues=part.dialogue_id.nunique(),
                        correct_mean=good[measure].mean(), wrong_mean=bad[measure].mean(),
                        wrong_minus_correct=bad[measure].mean()-good[measure].mean(),
                        correct_rawC_mean=good.confidence.mean(), wrong_rawC_mean=bad.confidence.mean(),
                        insufficient_support=len(good)<10 or len(bad)<10 or part.dialogue_id.nunique()<3))
            selected = match_sets[population]
            selected = selected[selected.dataset.eq(common['dataset'])]
            lookup = f.set_index(['dialogue_id', 'utterance_index'])
            local = []
            for pair in selected.to_dict('records'):
                a = lookup.loc[(pair['dialogue_id'], pair['correct_utterance_index'])]
                b = lookup.loc[(pair['dialogue_id'], pair['wrong_utterance_index'])]
                row = dict(pair, **{k:v for k,v in common.items() if k != 'dataset'}, population=population)
                for measure in MEASURES:
                    row.update({f'{measure}_correct':float(a[measure]), f'{measure}_wrong':float(b[measure]),
                                f'{measure}_wrong_minus_correct':float(b[measure]-a[measure])})
                local.append(row)
            pairs.extend(local)
            for measure in MEASURES:
                diff = np.array([r[f'{measure}_wrong_minus_correct'] for r in local], float)
                matched_summary.append(dict(base, measure=measure, n_pairs=len(local),
                    n_pair_dialogues=len(set(r['dialogue_id'] for r in local)),
                    correct_matched_fraction=len(local)/base['n_correct'] if base['n_correct'] else np.nan,
                    wrong_matched_fraction=len(local)/base['n_wrong'] if base['n_wrong'] else np.nan,
                    mean_wrong_minus_correct=float(diff.mean()) if len(diff) else np.nan,
                    positive_fraction=float(np.mean(diff>0)) if len(diff) else np.nan,
                    tie_fraction=float(np.mean(diff==0)) if len(diff) else np.nan,
                    mean_confidence_distance=float(np.mean([r['confidence_distance'] for r in local])) if local else np.nan,
                    max_confidence_distance=max((r['confidence_distance'] for r in local), default=np.nan),
                    insufficient_support=len(local)<10 or len(set(r['dialogue_id'] for r in local))<3))
    return dict(behavior_summary=pd.DataFrame(summaries), split_half_stability=pd.DataFrame(stability),
                conditional_diagnostic=pd.DataFrame(bins), matched_correct_wrong=pd.DataFrame(pairs),
                matched_statistics=pd.DataFrame(matched_summary))


def load_draws(config, ds):
    path = Path(config['source_five_seed']) / f'baseline/bootstrap/{ds}_dialogue_draws.npz'
    with np.load(path) as data:
        ids, draws = data['dialogue_ids'].astype(str), data['draws']
    if draws.shape != (2000, config['datasets'][ds]['dialogues']) or len(np.unique(ids)) != len(ids):
        raise ValueError('historical dialogue bootstrap draws mismatch')
    multiplicities = np.stack([np.bincount(row, minlength=len(ids)) for row in draws])
    return ids, multiplicities


def dialogue_means(frame, values, ids, multiplicities):
    idx = pd.Index(ids).get_indexer(frame.dialogue_id)
    if (idx < 0).any():
        raise ValueError('dialogue absent from frozen bootstrap draw registry')
    counts = np.bincount(idx, minlength=len(ids))
    sums = np.bincount(idx, weights=values, minlength=len(ids))
    denominator = multiplicities @ counts
    return np.divide(multiplicities@sums, denominator, out=np.full(len(multiplicities), np.nan), where=denominator>0)


def interval_row(common, point, distribution):
    distribution = np.asarray(distribution)
    valid = np.isfinite(distribution)
    low, high = np.quantile(distribution[valid], [.025, .975]) if valid.any() else (np.nan, np.nan)
    return dict(common, mean_difference=float(point), ci_low=float(low), ci_high=float(high),
                resamples=len(distribution), valid_resamples=int(valid.sum()), invalid_resamples=int((~valid).sum()),
                unit='shared_historical_dialogue', refit=False, exploratory=True, multiple_testing_adjusted=False)


def response_bootstrap(config, behavior, pairs):
    rows = []
    for ds in config['datasets']:
        ids, multiplicities = load_draws(config, ds)
        subset = behavior[behavior.dataset.eq(ds)]
        for (modality, corruption), condition in subset.groupby(['modality', 'corruption'], sort=True):
            for population, f in populations(condition):
                old, new = [f[f.strength.eq(s)].sort_values(KEY) for s in [.05, .1]]
                if len(old) != len(new) or not old[KEY].equals(new[KEY].set_axis(old.index)):
                    raise ValueError('strength response pairing mismatch')
                common = dict(dataset=ds, seed=1475, modality=modality, corruption=corruption,
                              population=population, comparison='strength_0.10_minus_0.05')
                for measure in RESPONSE_MEASURES:
                    col = 'mu' if measure.endswith('fraction') else measure
                    a, b = old[col].to_numpy(float), new[col].to_numpy(float)
                    if measure == 'mu_zero_fraction':
                        a, b = (a == 0).astype(float), (b == 0).astype(float)
                    elif measure == 'nonzero_mu_fraction':
                        a, b = (a != 0).astype(float), (b != 0).astype(float)
                    difference = b-a
                    rows.append(interval_row(dict(common, measure=measure), difference.mean(),
                                dialogue_means(old, difference, ids, multiplicities)))
                matched = pairs[pairs.dataset.eq(ds)&pairs.modality.eq(modality)&pairs.corruption.eq(corruption)&pairs.population.eq(population)]
                match_keys = ['dialogue_id', 'correct_utterance_index', 'wrong_utterance_index']
                a, b = [matched[matched.strength.eq(s)].sort_values(match_keys).reset_index(drop=True) for s in [.05, .1]]
                if not a[match_keys].equals(b[match_keys]):
                    raise ValueError('strength matching IDs differ')
                for measure in MEASURES:
                    col = f'{measure}_wrong_minus_correct'
                    for strength, part in [(.05, a), (.1, b)]:
                        vals = part[col].to_numpy(float)
                        rows.append(interval_row(dict(common, measure=measure, comparison='matched_wrong_minus_correct',
                                   strength=strength), vals.mean() if len(vals) else np.nan,
                                   dialogue_means(part, vals, ids, multiplicities)))
                    diff = b[col].to_numpy(float)-a[col].to_numpy(float)
                    rows.append(interval_row(dict(common, measure=f'matched_{measure}',
                               comparison='matched_difference_strength_0.10_minus_0.05'),
                               diff.mean() if len(diff) else np.nan, dialogue_means(a, diff, ids, multiplicities)))
    return pd.DataFrame(rows)


def run_calibration(root, config, features):
    outer = json.loads((Path(config['source_five_seed'])/'baseline/fold_parameters.json').read_text())
    prediction_parts, metric_rows, shifts, selections, parity_rows = [], [], [], [], []
    historical_predictions = read(Path(config['source_mainline'])/'modality/oof_predictions.csv')
    for keys, frame in features.groupby(CONDITION, sort=False):
        common = dict(zip(CONDITION, keys))
        common['seed'], common['strength'] = int(common['seed']), float(common['strength'])
        frame = frame.reset_index(drop=True)
        coverage = np.zeros(len(frame), int)
        parts, params = [], []
        folds = [f for f in outer if f['dataset']==common['dataset'] and f['seed']==1475]
        for fold in folds:
            train = np.flatnonzero(frame.dialogue_id.isin(fold['train_dialogues']))
            held = np.flatnonzero(frame.dialogue_id.isin(fold['held_dialogues']))
            if len(train)!=fold['train_n'] or len(held)!=fold['held_n'] or len(train)+len(held)!=len(frame):
                raise ValueError('outer fold coverage mismatch')
            outputs, fitted = fit_outer(frame, train, held, BEHAVIOR_METHODS)
            fitted.update(common, fold=fold['fold'])
            params.append(fitted)
            coverage[held] += 1
            outputs['E0'] = {'oof_confidence':frame.iloc[held].confidence.to_numpy(float),
                             'base_score':np.full(len(held), np.nan), 'behavior_score':np.zeros(len(held)),
                             'reliability_score':np.full(len(held), np.nan)}
            for method in METHODS:
                part = frame.iloc[held][KEY + ['modality', 'strength', 'label', 'prediction', 'correct',
                                              'confidence', 'visual_missing', 'perturbation_skipped']].copy()
                part['method'], part['fold'] = method, fold['fold']
                part['selected_penalty'] = fitted['methods'][method]['selected_penalty'] if method in BEHAVIOR_METHODS else ('off' if method=='E1' else 'none')
                for col, vals in outputs[method].items():
                    part[col] = vals
                parts.append(part)
            for method, selection in fitted['methods'].items():
                for penalty, loss in selection['candidate_brier'].items():
                    selections.append(dict(common, fold=fold['fold'], method=method, penalty=penalty,
                                      pooled_inner_brier=loss, selected=str(selection['selected_penalty'])==penalty))
        if not (coverage==1).all():
            raise ValueError('each utterance must be held out once')
        predictions = pd.concat(parts, ignore_index=True)
        prediction_parts.append(predictions)
        tag = f"{common['dataset']}_{common['modality']}_strength{common['strength']:.2f}"
        dump(root/'results/fold_parameters'/f'{tag}.json', params)
        for method, f in predictions.groupby('method', sort=False):
            metric_rows.append(dict(common, method=method, **metrics(f, f.oof_confidence.to_numpy(float))))
            for correct, part in f.groupby('correct'):
                diff = part.oof_confidence-part.confidence
                shifts.append(dict(common, method=method, correct=int(correct), utterances=len(part),
                    original_mean_confidence=float(part.confidence.mean()), calibrated_mean_confidence=float(part.oof_confidence.mean()),
                    mean_change=float(diff.mean()), raised_fraction=float((diff>1e-12).mean()), lowered_fraction=float((diff< -1e-12).mean())))
        reference = historical_predictions[historical_predictions.dataset.eq(common['dataset']) &
            historical_predictions.modality.eq(common['modality']) &
            historical_predictions.method.isin(['E1','M_joint'] if common['strength']==.05 else ['E1'])]
        joined = predictions.merge(reference[KEY+['method','oof_confidence']], on=KEY+['method'],
                                   validate='one_to_one', suffixes=('_new','_old'))
        for method, group in joined.groupby('method'):
            error = float(np.max(np.abs(group.oof_confidence_new-group.oof_confidence_old)))
            if len(group)!=len(frame) or error>1e-8:
                raise ValueError(f'historical {method} parity failed: {common} error {error}')
            parity_rows.append(dict(common, method=method, rows=len(group), max_absolute_error=error, tolerance=1e-8))
        print(f'OOF complete {tag}: {len(predictions)} rows', flush=True)
    return dict(oof_predictions=pd.concat(prediction_parts, ignore_index=True), calibration_metrics=pd.DataFrame(metric_rows),
                confidence_shift=pd.DataFrame(shifts), inner_cv_results=pd.DataFrame(selections)), parity_rows


def calibration_bootstrap(config, predictions):
    rows = []
    for ds in config['datasets']:
        ids, multiplicities = load_draws(config, ds)
        for modality in config['modalities']:
            subset = predictions[predictions.dataset.eq(ds)&predictions.modality.eq(modality)]
            distributions, points = {}, {}
            for (strength, method), f in subset.groupby(['strength','method'], sort=True):
                f = f.sort_values(KEY)
                idx = pd.Index(ids).get_indexer(f.dialogue_id)
                if (idx<0).any():
                    raise ValueError('OOF dialogue absent from frozen draws')
                counts = np.bincount(idx, minlength=len(ids))
                totals = multiplicities@counts
                q, y = f.oof_confidence.to_numpy(float), f.correct.to_numpy(int)
                bins = np.minimum(np.floor(q*15).astype(int), 14)
                residual = np.zeros((len(ids), 15))
                np.add.at(residual, (idx,bins), q-y)
                clip = np.clip(q,1e-15,1-1e-15)
                distributions[strength,method] = dict(
                    ece15=np.abs(multiplicities@residual).sum(axis=1)/totals,
                    binary_brier=multiplicities@np.bincount(idx,weights=(q-y)**2,minlength=len(ids))/totals,
                    binary_nll=multiplicities@np.bincount(idx,weights=-y*np.log(clip)-(1-y)*np.log1p(-clip),minlength=len(ids))/totals)
                points[strength,method] = calibration_metrics(q,y)
            comparisons = [(s,m,s,'E1','behavior_minus_E1') for s in [.05,.1] for m in BEHAVIOR_METHODS]
            comparisons += [(.1,m,.05,m,'strength_0.10_minus_0.05') for m in BEHAVIOR_METHODS]
            for candidate_s, candidate, reference_s, reference, label in comparisons:
                for metric in METRICS:
                    diff = distributions[candidate_s,candidate][metric]-distributions[reference_s,reference][metric]
                    rows.append(interval_row(dict(dataset=ds,seed=1475,modality=modality,comparison=label,
                        candidate=candidate, candidate_strength=candidate_s, reference=reference, reference_strength=reference_s,
                        metric=metric, bootstrap_fraction_improving=float((diff<0).mean())),
                        points[candidate_s,candidate][metric]-points[reference_s,reference][metric], diff))
    return pd.DataFrame(rows)


def save_tables(destination, tables):
    destination.mkdir(parents=True, exist_ok=True)
    for name, f in tables.items():
        f.to_csv(destination/f'{name}.csv', index=False, float_format='%.17g', na_rep='NA')
        print(f'{name}.csv: {len(f)} rows', flush=True)


def analysis_sources(root, config, include_new):
    """Enumerate every input before reading; detect mutations during derivation too."""
    paths = [root/'preregistration.json', root/'EXPERIMENT_PLAN.md',
             Path(config['source_mainline'])/'inputs/modality_features.csv',
             Path(config['source_mainline'])/'modality/oof_predictions.csv',
             Path(config['source_five_seed'])/'baseline/fold_parameters.json']
    paths += sorted((root/'scripts').glob('*.py'))
    for ds in config['datasets']:
        paths += [Path(config['source_pilot'])/ds/name for name in
                  ['original_predictions.csv','perturbed_predictions.csv']]
        paths.append(Path(config['source_five_seed'])/f'baseline/bootstrap/{ds}_dialogue_draws.npz')
        if include_new:
            for mod in config['modalities']:
                paths += [root/'runs'/ds/mod/name for name in
                          ['original_predictions.csv','perturbed_predictions.csv','complete.json']]
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--prepare', action='store_true', help='Validate historical 0.05 only, save under prepared/; no fitting')
    parser.add_argument('--behavior-only', action='store_true', help='Complete both-strength behavior analysis without fitting')
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root/'preregistration.json').read_text())
    if config['methods']!=METHODS or config['penalties']!=['off',1.,.1,.01] or config['seed']!=1475:
        raise ValueError('frozen protocol differs from implemented method contract')
    started = time.time()
    before = {str(p):sha(p) for p in analysis_sources(root,config,not args.prepare)}
    originals, behavior, features, sources, verification = build_behavior(root, config, not args.prepare)
    destination = root/('prepared' if args.prepare else 'results')
    tables = dict(original_predictions=originals, behavior_features=behavior, feature_matrix=features)
    tables.update(descriptive_tables(originals, behavior))
    if not args.prepare:
        tables['response_bootstrap'] = response_bootstrap(config, behavior, tables['matched_correct_wrong'])
    if not args.prepare and not args.behavior_only:
        fitted, parity = run_calibration(root, config, features)
        tables.update(fitted)
        tables['bootstrap_results'] = calibration_bootstrap(config,tables['oof_predictions'])
        verification['historical_oof_parity'] = parity
        if len(tables['oof_predictions'])!=65440 or len(tables['calibration_metrics'])!=80:
            raise ValueError('OOF/metric result coverage mismatch')
    save_tables(destination, tables)
    if not all(sha(p)==v for p,v in before.items()):
        raise ValueError('input/code changed during analysis')
    dump(destination/'source_hashes.json', before)
    verification.update(status='historical_prepared' if args.prepare else ('behavior_complete' if args.behavior_only else 'complete'),
        all_passed=True, independent_verification_pending=True,
        output_rows={name:len(frame) for name,frame in tables.items()},
        output_sha256={name+'.csv':sha(destination/(name+'.csv')) for name in tables},
        one_utterance_per_condition=True, input_code_unchanged=True, cpu_only=True, test_labels_used=False,
        cmerc_training=False, no_js_scoring=True, no_interactions=True, original_prediction_argmax_preserved=True,
        confidence_change_std_ddof=0, js_log_base='natural', bootstrap='2000 shared historical dialogue draws; exploratory; no refit',
        matching='same dialogue; abs rawC distance <=0.02; maximum cardinality then minimum distance; behavior independent',
        strengths_paired_at_utterance_level_only=True, independent_noise_streams_across_strengths=True,
        fitting_order='exact historical modality_features.csv order within dataset/modality',
        elapsed_seconds=time.time()-started,
        environment=dict(python=sys.executable, host=platform.node(),numpy=np.__version__,pandas=pd.__version__))
    dump(destination/'analysis_verification.json', verification)
    if not args.prepare and not args.behavior_only:
        from build_reports import build_reports
        build_reports(root)
    print(f'Analysis {verification["status"]}; elapsed {time.time()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
