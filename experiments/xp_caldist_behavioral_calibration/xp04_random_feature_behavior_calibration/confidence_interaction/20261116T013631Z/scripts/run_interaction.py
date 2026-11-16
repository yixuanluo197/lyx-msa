"""Fixed CPU-only five-seed confidence interaction evaluation; no new inference."""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from interaction_core import (FEATURES, METRICS, METHODS as BEHAVIOR_METHODS,
    calibration_metrics, dump, fit_outer, metrics, sha)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'preregistration.json').read_text())
KEY = ['dataset', 'seed', 'dialogue_id', 'utterance_index']
METHODS = ['E0', 'E1', *BEHAVIOR_METHODS]


def read(path):
    return pd.read_csv(path, dtype={'dataset': str, 'dialogue_id': str})


def checked_input():
    gate = json.loads((ROOT / 'inputs/verification.json').read_text())
    assert gate['all_passed'] and gate['source_files_unchanged'], 'Input integrity gate required'
    frame = read(ROOT / 'inputs/features.csv')
    assert len(frame) == CONFIG['expected']['input_rows']
    assert not frame.duplicated(KEY).any()
    assert set(frame.dataset) == set(CONFIG['datasets'])
    assert set(frame.seed) == set(CONFIG['seeds']) and set(frame.modality) == {'joint'}
    assert np.isfinite(frame[['confidence', *FEATURES]].to_numpy(float)).all()
    assert frame.confidence.between(0, 1).all()
    assert ((frame.prediction == frame.label).astype(int) == frame.correct).all()
    for (dataset, seed), group in frame.groupby(['dataset', 'seed']):
        expected = CONFIG['datasets'][dataset]
        assert seed in CONFIG['seeds']
        assert len(group) == expected['utterances'] and group.dialogue_id.nunique() == expected['dialogues']
    return frame


def shift_summary(group):
    change = group.oof_confidence.to_numpy(float) - group.confidence.to_numpy(float)
    return {'n': len(group), 'original_mean_confidence': float(group.confidence.mean()),
        'calibrated_mean_confidence': float(group.oof_confidence.mean()),
        'mean_change': float(change.mean()) if len(group) else np.nan,
        'raised_fraction': float(np.mean(change > 1e-12)) if len(group) else np.nan,
        'lowered_fraction': float(np.mean(change < -1e-12)) if len(group) else np.nan}


def confidence_diagnostics(predicted):
    shifts = []; bins = []
    edges = np.asarray(CONFIG['confidence_diagnostic_bins'], float)
    for key, group in predicted.groupby(['dataset', 'seed', 'modality', 'method'], sort=False):
        common = dict(zip(['dataset', 'seed', 'modality', 'method'], key))
        assignment = np.minimum(np.searchsorted(edges, group.confidence.to_numpy(float), side='right') - 1, len(edges) - 2)
        assert ((assignment >= 0) & (assignment < len(edges) - 1)).all()
        for correct in [0, 1]:
            subset = group[group.correct.eq(correct)]
            shifts.append(dict(common, correct=correct, **shift_summary(subset)))
            for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
                subset = group[(assignment == index) & group.correct.eq(correct)]
                bins.append(dict(common, bin_index=index, bin_low=float(low), bin_high=float(high),
                    correct=correct, **shift_summary(subset)))
    return pd.DataFrame(shifts), pd.DataFrame(bins)


def run_group(frame, definitions, out):
    frame = frame.reset_index(drop=True)
    dataset = str(frame.dataset.iloc[0]); seed = int(frame.seed.iloc[0])
    output = []; params = []; selections = []; assignments = np.zeros(len(frame), int)
    folds = [fold for fold in definitions if fold['dataset'] == dataset and fold['seed'] == seed]
    assert len(folds) == (12 if dataset == 'IEMOCAP' else 5)
    for fold in folds:
        train = np.flatnonzero(frame.dialogue_id.isin(fold['train_dialogues']))
        held = np.flatnonzero(frame.dialogue_id.isin(fold['held_dialogues']))
        assert len(train) == fold['train_n'] and len(held) == fold['held_n']
        assert len(train) + len(held) == len(frame)
        scores, param = fit_outer(frame, train, held)
        param.update({'dataset': dataset, 'seed': seed, 'modality': 'joint', 'fold': fold['fold']})
        params.append(param); assignments[held] += 1
        e0 = frame.iloc[held].copy()
        e0['method'] = 'E0'; e0['fold'] = fold['fold']; e0['selected_penalty'] = 'none'
        e0['base_score'] = np.nan; e0['additive_score'] = 0.; e0['interaction_score'] = 0.
        e0['behavior_score'] = 0.; e0['reliability_score'] = np.nan; e0['oof_confidence'] = e0.confidence
        output.append(e0)
        for method, values in scores.items():
            rows = frame.iloc[held].copy(); rows['method'] = method; rows['fold'] = fold['fold']
            rows['selected_penalty'] = param['methods'][method]['selected_penalty'] if method != 'E1' else 'off'
            for column, value in values.items():
                rows[column] = value
            output.append(rows)
        for method, item in param['methods'].items():
            for penalty, value in item['candidate_brier'].items():
                selections.append({'dataset': dataset, 'seed': seed, 'modality': 'joint',
                    'fold': fold['fold'], 'method': method, 'penalty': penalty,
                    'pooled_inner_brier': value, 'selected': str(item['selected_penalty']) == penalty})
        print(f'{dataset} {seed} joint fold {fold["fold"]} completed', flush=True)
    assert (assignments == 1).all()
    predicted = pd.concat(output, ignore_index=True).sort_values(KEY + ['method']).reset_index(drop=True)
    dump(out / f'fold_parameters/{dataset}_seed{seed}_joint.json', params)
    rows = []; dialogues = []
    for method, group in predicted.groupby('method', sort=False):
        common = {'dataset': dataset, 'seed': seed, 'modality': 'joint', 'method': method}
        rows.append(dict(common, **metrics(group, group.oof_confidence.to_numpy(float))))
        for dialogue, subset in group.groupby('dialogue_id'):
            dialogues.append(dict(common, dialogue_id=str(dialogue),
                **calibration_metrics(subset.oof_confidence, subset.correct)))
    return predicted, pd.DataFrame(rows), pd.DataFrame(dialogues), pd.DataFrame(selections), params


def bootstrap(predictions):
    """Reuse frozen shared Dialogue draws; average metric differences across seeds."""
    rows = []
    for dataset, expected in CONFIG['datasets'].items():
        with np.load(ROOT / f'inputs/{dataset}_dialogue_draws.npz') as source:
            dialogues = source['dialogue_ids'].astype(str); draws = source['draws']
        assert np.array_equal(dialogues, np.sort(dialogues))
        multiplicities = np.stack([np.bincount(draw, minlength=len(dialogues)) for draw in draws])
        assert multiplicities.shape == (CONFIG['bootstrap']['repeats'], expected['dialogues'])
        distributions = {}; point = {}
        for seed in CONFIG['seeds']:
            subset = predictions[predictions.dataset.eq(dataset) & predictions.seed.eq(seed)]
            for method, group in subset.groupby('method', sort=False):
                group = group.sort_values(KEY).reset_index(drop=True)
                index = np.searchsorted(dialogues, group.dialogue_id.to_numpy(str))
                assert np.array_equal(dialogues[index], group.dialogue_id.to_numpy(str))
                counts = np.bincount(index, minlength=len(dialogues)); total = multiplicities @ counts
                q = group.oof_confidence.to_numpy(float); y = group.correct.to_numpy(int)
                clip = np.clip(q, 1e-15, 1 - 1e-15)
                bins = np.minimum(np.floor(q * 15).astype(int), 14)
                residual = np.zeros((len(dialogues), 15)); np.add.at(residual, (index, bins), q - y)
                distributions[seed, method] = {
                    'ece15': np.abs(multiplicities @ residual).sum(axis=1) / total,
                    'binary_brier': multiplicities @ np.bincount(index, weights=(q-y)**2, minlength=len(dialogues)) / total,
                    'binary_nll': multiplicities @ np.bincount(index, weights=-y*np.log(clip)-(1-y)*np.log1p(-clip), minlength=len(dialogues)) / total}
                point[seed, method] = calibration_metrics(q, y)
        for candidate, reference in CONFIG['comparisons']:
            for metric in METRICS:
                delta = np.array([point[seed, candidate][metric] - point[seed, reference][metric] for seed in CONFIG['seeds']])
                distribution = np.mean([distributions[seed, candidate][metric] - distributions[seed, reference][metric]
                    for seed in CONFIG['seeds']], axis=0)
                rows.append({'dataset': dataset, 'modality': 'joint', 'candidate': candidate,
                    'reference': reference, 'metric': metric, 'mean_difference': float(delta.mean()),
                    'seed_difference_sample_sd': float(delta.std(ddof=1)),
                    'improved_seeds': int((delta < 0).sum()), 'seeds': len(CONFIG['seeds']),
                    'ci_low': float(np.quantile(distribution, .025)), 'ci_high': float(np.quantile(distribution, .975)),
                    'bootstrap_fraction_improving': float((distribution < 0).mean()),
                    'resamples': CONFIG['bootstrap']['repeats'], 'unit': 'shared_dialogue_across_seeds', 'refit': False})
    return pd.DataFrame(rows)


def aggregate(per_seed):
    groups = ['dataset', 'modality', 'method']
    numeric = [column for column in per_seed.select_dtypes(include=np.number)
        if column not in ['seed', 'utterances', 'dialogues']]
    rows = []
    for key, group in per_seed.groupby(groups, sort=False):
        row = dict(zip(groups, key)); row['seeds'] = len(group)
        for column in numeric:
            row[column + '_mean'] = float(group[column].mean())
            row[column + '_sample_sd'] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def historical_parity(predicted):
    previous = read(ROOT / 'inputs/reference_oof.csv')
    rows = []; all_passed = True
    for method in ['E1', 'M_joint']:
        current = predicted[predicted.method.eq(method)]
        old = previous[previous.method.eq(method)]
        matched = current.merge(old[KEY + ['fold', 'label', 'prediction', 'correct', 'oof_confidence']],
            on=KEY, validate='one_to_one', suffixes=('_new', '_old'))
        assert len(matched) == len(current) == len(old) == CONFIG['expected']['input_rows']
        for column in ['fold', 'label', 'prediction', 'correct']:
            assert (matched[column + '_new'] == matched[column + '_old']).all()
        for (dataset, seed), group in matched.groupby(['dataset', 'seed']):
            difference = np.abs(group.oof_confidence_new - group.oof_confidence_old)
            passed = bool((difference <= 1e-10).all()); all_passed &= passed
            rows.append({'dataset': dataset, 'seed': int(seed), 'method': method, 'rows': len(group),
                'max_absolute_confidence_difference': float(difference.max()),
                'mean_absolute_confidence_difference': float(difference.mean()),
                'rows_exceeding_target': int((difference > 1e-10).sum()), 'passed': passed})
    return {'target_absolute_tolerance': 1e-10, 'all_passed': bool(all_passed),
        'note': 'Historical optimizer differences are reported separately and cannot count as interaction gains.',
        'comparisons': rows}


def snapshots(out):
    paths = [ROOT / name for name in ['preregistration.json', 'INTERACTION_PROTOCOL.md', 'IMPLEMENTATION_PLAN.md']]
    paths += [path for directory in ['inputs', 'scripts', 'vendor']
        for path in sorted((ROOT / directory).rglob('*')) if path.is_file() and '__pycache__' not in path.parts]
    sources = {str(path): sha(path) for path in paths}
    dump(out / 'provenance/source_hashes.json', sources)
    dump(out / 'provenance/source_hashes_before.json', sources)
    dump(out / 'provenance/environment.json', {'python': sys.executable, 'python_version': sys.version,
        'host': platform.node(), 'platform': platform.platform(), 'numpy': np.__version__,
        'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
        'argv': sys.argv, 'device': 'CPU', 'pid': os.getpid(),
        'thread_environment': {key: os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}})
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true'); args = parser.parse_args()
    assert args.run, 'Use --run only after parent review and independent verifier freeze'
    assert CONFIG['methods'] == METHODS and CONFIG['behavior_features'] == FEATURES
    out = ROOT / 'results'
    assert not (out / 'oof_predictions.csv').exists() and not (out / 'fold_parameters').exists(), 'Refuse duplicate execution or overwrite'
    assert (ROOT / 'scripts/verify_interaction.py').is_file(), 'Independent verifier must exist before execution'
    started = time.time(); frame = checked_input(); sources = snapshots(out)
    definitions = json.loads((ROOT / 'inputs/fold_definitions.json').read_text())
    assert len(definitions) == CONFIG['expected']['outer_folds']
    outputs = [[] for _ in range(4)]; parameters = []
    for dataset in CONFIG['datasets']:
        for seed in CONFIG['seeds']:
            subset = frame[frame.dataset.eq(dataset) & frame.seed.eq(seed)].reset_index(drop=True)
            result = run_group(subset, definitions, out)
            for target, value in zip(outputs, result[:4]):
                target.append(value)
            parameters.extend(result[4])
    predicted, per_seed, dialogues, selections = [pd.concat(parts, ignore_index=True) for parts in outputs]
    assert len(predicted) == CONFIG['expected']['oof_rows']
    assert len(per_seed) == CONFIG['expected']['per_seed_metrics']
    assert len(parameters) == CONFIG['expected']['outer_folds']
    assert not predicted.duplicated(KEY + ['method']).any()
    assert np.isfinite(predicted.oof_confidence).all() and predicted.oof_confidence.between(0, 1).all()
    matched = predicted.merge(frame[KEY + ['label', 'prediction', 'correct']], on=KEY,
        validate='many_to_one', suffixes=('', '_input'))
    assert len(matched) == len(predicted)
    for column in ['label', 'prediction', 'correct']:
        assert (matched[column] == matched[column + '_input']).all()
    agg = aggregate(per_seed); boots = bootstrap(predicted)
    assert len(boots) == CONFIG['expected']['bootstrap_rows']
    shifts, bins = confidence_diagnostics(predicted)
    selected = [dict(dataset=param['dataset'], seed=param['seed'], fold=param['fold'], method=method,
        selected_penalty=item['selected_penalty']) for param in parameters for method, item in param['methods'].items()]
    tables = {'oof_predictions.csv': predicted, 'per_seed_metrics.csv': per_seed,
        'per_dialogue_results.csv': dialogues, 'inner_cv_results.csv': selections,
        'aggregate_metrics.csv': agg, 'bootstrap_results.csv': boots,
        'correct_wrong_adjustment.csv': shifts, 'confidence_bins.csv': bins,
        'selected_penalties.csv': pd.DataFrame(selected)}
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
    parity = historical_parity(predicted)
    dump(out / 'historical_parity.json', parity)
    after = {path: sha(path) for path in sources}
    dump(out / 'provenance/source_hashes_after.json', after)
    assert sources == after, 'Input/protocol/implementation changed during run'
    assert parity['all_passed'], 'Historical parity target exceeded; review explicit differences before interpretation'
    dump(out / 'verification.json', {'status': 'completed', 'independent_verification_pending': True,
        'cpu_only': True, 'feature_rows': len(frame), 'oof_rows': len(predicted),
        'metric_rows': len(per_seed), 'outer_folds': len(parameters), 'bootstrap_rows': len(boots),
        'input_and_code_hashes_unchanged': True, 'historical_parity_passed': parity['all_passed'],
        'confidence_finite': True, 'predicted_classes_unchanged': True,
        'elapsed_seconds': time.time() - started, 'methods': METHODS, 'primary_method': 'M_interaction',
        'training_record_per_utterance': 1, 'test_labels_used': False, 'cmerc_training': False,
        'new_perturbations': False, 'raw_data_modified': False})
    print(agg[['dataset', 'method'] + [metric + '_mean' for metric in METRICS]].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
