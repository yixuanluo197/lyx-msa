#!/usr/bin/env python3
"""Independent CPU-only audit; does not import the experiment statistics code."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EMOTIONS = ('happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated')
THRESHOLDS = (0.8, 0.9, 0.95)
TOLERANCE = 1e-11


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    return math.isclose(float(actual), float(expected), rel_tol=TOLERANCE, abs_tol=TOLERANCE)


def key(row):
    return row['dataset'], row['dialogue_id'], int(row['utterance_index'])


def recompute(root, out):
    checks = []

    def check(name, condition):
        checks.append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(name)

    sources = {
        'baseline': root / 'checkpoints/analysis/m3net_official_checkpoint_verification/all_predictions.csv',
        'cmerc': root / 'checkpoints/analysis/official_checkpoint_verification/IEMOCAP_predictions.csv',
    }
    rows = {name: read_csv(path) for name, path in sources.items()}
    arrays, metrics, bins = {}, {}, {}
    for name, records in rows.items():
        check(f'{name}:1623 effective utterances', len(records) == 1623)
        check(f'{name}:unique stable IDs', len(set(map(key, records))) == len(records))
        check(f'{name}:unique source utterance IDs', len({r['source_utterance_id'] for r in records}) == len(records))
        check(f'{name}:dataset and full input', all(r['dataset'] == 'IEMOCAP' and r['input_condition'] == 'full' for r in records))
        labels = np.array([int(r['true_label']) for r in records])
        predicted = np.array([int(r['predicted_label']) for r in records])
        probabilities = np.array([[float(r['probability_' + e]) for e in EMOTIONS] for r in records])
        check(f'{name}:finite bounded probabilities', np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all())
        check(f'{name}:probability rows sum to one', np.allclose(probabilities.sum(1), 1, rtol=0, atol=2e-6))
        check(f'{name}:argmax agrees', np.array_equal(predicted, probabilities.argmax(1)))
        check(f'{name}:class mapping agrees', all(r['true_emotion'] == EMOTIONS[int(r['true_label'])] and r['predicted_emotion'] == EMOTIONS[int(r['predicted_label'])] for r in records))
        confidence = probabilities.max(1)
        check(f'{name}:stored confidence agrees', all(close(r['predicted_confidence'], confidence[i]) for i, r in enumerate(records)))
        check(f'{name}:stored true probability agrees', all(close(r['true_label_probability'], probabilities[i, labels[i]]) for i, r in enumerate(records)))
        correct = labels == predicted
        if 'correct' in records[0]:
            check(f'{name}:stored correctness agrees', all(str(r['correct']).lower() == str(bool(correct[i])).lower() for i, r in enumerate(records)))
        for dialogue in {r['dialogue_id'] for r in records}:
            indices = [int(r['utterance_index']) for r in records if r['dialogue_id'] == dialogue]
            check(f'{name}:padding excluded and positions consecutive:{dialogue}', indices == list(range(len(indices))))
        arrays[name] = {'labels': labels, 'predicted': predicted, 'probabilities': probabilities, 'confidence': confidence, 'correct': correct}
        observed_bins, weighted_gap, maximum_gap = [], 0.0, 0.0
        for i in range(10):
            mask = (confidence >= i / 10) & ((confidence < (i + 1) / 10) if i < 9 else (confidence <= 1))
            n = int(mask.sum())
            mean = float(confidence[mask].mean()) if n else None
            accuracy = float(correct[mask].mean()) if n else None
            gap = mean - accuracy if n else None
            if n:
                weighted_gap += n / len(records) * abs(gap)
                maximum_gap = max(maximum_gap, abs(gap))
            observed_bins.append({'bin_index': i, 'count': n, 'mean_confidence': mean, 'accuracy': accuracy, 'signed_gap': gap})
        check(f'{name}:ten bins partition all samples', sum(b['count'] for b in observed_bins) == len(records))
        bins[name] = observed_bins
        confusion = Counter(zip(labels.tolist(), predicted.tolist()))
        weighted_f1, class_f1 = 0.0, []
        for c in range(6):
            tp = confusion[c, c]
            fp = sum(confusion[t, c] for t in range(6) if t != c)
            fn = sum(confusion[c, p] for p in range(6) if p != c)
            support = int((labels == c).sum())
            score = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
            class_f1.append(score)
            weighted_f1 += support * score / len(records)
        metrics[name] = {
            'valid_samples': len(records), 'total_errors': int((~correct).sum()),
            'accuracy': float(correct.mean()), 'weighted_f1': weighted_f1,
            'macro_f1': sum(class_f1) / len(class_f1),
            'mean_confidence': float(confidence.mean()), 'ece': weighted_gap,
            'mce': maximum_gap,
            'brier_score': float(np.square(probabilities - np.eye(6)[labels]).sum(1).mean()),
            'nll': float(-np.log(probabilities[np.arange(len(records)), labels]).mean()),
            'mean_error_confidence': float(confidence[~correct].mean()),
        }
    check('models have identical original sample order', list(map(key, rows['baseline'])) == list(map(key, rows['cmerc'])))
    for field in ('source_utterance_id', 'utterance_text', 'true_label', 'true_emotion'):
        check(f'models align:{field}', all(a[field] == b[field] for a, b in zip(rows['baseline'], rows['cmerc'])))
    a, b = arrays['baseline'], arrays['cmerc']
    ao, bo = a['correct'], b['correct']
    ac, bc = a['confidence'], b['confidence']
    transitions = {
        'both_correct': int((ao & bo).sum()), 'baseline_wrong_cmerc_correct': int((~ao & bo).sum()),
        'baseline_correct_cmerc_wrong': int((ao & ~bo).sum()), 'both_wrong': int((~ao & ~bo).sum()),
    }
    check('four transitions partition all utterances', sum(transitions.values()) == 1623)
    thresholds = []
    for threshold in THRESHOLDS:
        ah, bh = ~ao & (ac >= threshold), ~bo & (bc >= threshold)
        item = {'threshold': threshold}
        for name, confidence, hce in [('baseline', ac, ah), ('cmerc', bc, bh)]:
            count = int((confidence >= threshold).sum())
            item[name + '_high_confidence_predictions'] = count
            item[name + '_high_confidence_errors'] = int(hce.sum())
            item[name + '_high_confidence_error_rate'] = int(hce.sum()) / count if count else None
        item.update({
            'baseline_hce_corrected': int((ah & bo).sum()),
            'baseline_hce_still_wrong_confidence_lower': int((ah & ~bo & (bc < ac)).sum()),
            'baseline_hce_still_wrong_confidence_not_lower': int((ah & ~bo & (bc >= ac)).sum()),
            'baseline_hce_still_wrong_above_threshold': int((ah & bh).sum()),
            'new_cmerc_hce': int((~ah & bh).sum()),
            'new_cmerc_hce_baseline_correct': int((ao & bh).sum()),
        })
        check(f'{threshold}:baseline HCE partition', sum(item[k] for k in ('baseline_hce_corrected', 'baseline_hce_still_wrong_confidence_lower', 'baseline_hce_still_wrong_confidence_not_lower')) == int(ah.sum()))
        check(f'{threshold}:CMERC HCE new/persistent partition', item['new_cmerc_hce'] + item['baseline_hce_still_wrong_above_threshold'] == int(bh.sum()))
        thresholds.append(item)
    classwise = []
    for label, emotion in enumerate(EMOTIONS):
        mask = a['labels'] == label
        item = {'true_label': label, 'true_emotion': emotion, 'support': int(mask.sum()),
                'baseline_errors': int((mask & ~ao).sum()), 'cmerc_errors': int((mask & ~bo).sum()),
                'baseline_wrong_cmerc_correct': int((mask & ~ao & bo).sum()),
                'baseline_correct_cmerc_wrong': int((mask & ao & ~bo).sum()),
                'both_wrong': int((mask & ~ao & ~bo).sum())}
        item['net_error_change_cmerc_minus_baseline'] = item['cmerc_errors'] - item['baseline_errors']
        check(f'{emotion}:net errors agrees with discordant transitions', item['net_error_change_cmerc_minus_baseline'] == item['baseline_correct_cmerc_wrong'] - item['baseline_wrong_cmerc_correct'])
        classwise.append(item)
    preserved = json.loads((out / 'preserved_inputs.json').read_text())
    for relative, expected in preserved.items():
        check('preserved:' + relative, sha256(root / relative) == expected)
    return {'schema_version': 1, 'status': 'source_checks_passed_outputs_pending',
            'created_at': datetime.now(timezone.utc).isoformat(), 'engine': 'independent standard-library CSV/Counter/math and NumPy; no project statistics imports; CPU only',
            'tolerances': {'derived_numeric_absolute_and_relative': TOLERANCE, 'probability_sum_absolute': 2e-6},
            'inputs': {name: {'path': str(path.relative_to(root)), 'sha256': sha256(path)} for name, path in sources.items()},
            'preserved_old_files': len(preserved), 'preserved_manifest_sha256': sha256(out / 'preserved_inputs.json'),
            'independent_script_sha256': sha256(Path(__file__)), 'metrics': metrics, 'classification_transitions': transitions,
            'high_confidence_transitions': thresholds, 'classwise_transitions': classwise,
            'calibration_bins': bins, 'checks': checks}, rows, arrays


def audit_outputs(out, result, sources, arrays):
    """Compare each saved row to values independently derived from original CSVs."""
    checks = result['checks']

    def check(name, condition):
        checks.append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(name)

    def numeric(name, actual, expected):
        if expected is None:
            check(name, actual in ('N/A', ''))
        else:
            check(name, close(actual, expected))

    paired = read_csv(out / 'paired_predictions.csv')
    check('paired:exactly 1623 unique rows', len(paired) == 1623 and len(set(map(key, paired))) == 1623)
    check('paired:original order retained', list(map(key, paired)) == list(map(key, sources['baseline'])))
    a, b = arrays['baseline'], arrays['cmerc']
    ao, bo, ac, bc = a['correct'], b['correct'], a['confidence'], b['confidence']
    for i, row in enumerate(paired):
        name = f'paired row {i}'
        shared_fields = ('dataset', 'dialogue_id', 'utterance_index', 'source_utterance_id', 'utterance_text', 'true_label', 'true_emotion')
        check(name + ':identity/text/labels', all(row[f] == sources['baseline'][i][f] == sources['cmerc'][i][f] for f in shared_fields))
        for model in ('baseline', 'cmerc'):
            original = sources[model][i]
            string_fields = ('model', 'predicted_label', 'predicted_emotion')
            number_fields = ('predicted_confidence', 'true_label_probability') + tuple('probability_' + e for e in EMOTIONS)
            check(name + ':' + model + ':source fields exact', all(row[model + '_' + f] == original[f] for f in string_fields + number_fields))
            check(name + ':' + model + ':correctness', row[model + '_correct'].lower() == str(bool(arrays[model]['correct'][i])).lower())
        expected_transition = ('both_correct' if bo[i] else 'baseline_correct_cmerc_wrong') if ao[i] else ('baseline_wrong_cmerc_correct' if bo[i] else 'both_wrong')
        check(name + ':transition', row['transition'] == expected_transition)
        check(name + ':predicted class changed', row['predicted_class_changed'].lower() == str(bool(a['predicted'][i] != b['predicted'][i])).lower())
        numeric(name + ':maximum probability delta', row['confidence_delta'], bc[i] - ac[i])
        numeric(name + ':true label probability delta', row['true_label_probability_delta'], b['probabilities'][i, b['labels'][i]] - a['probabilities'][i, a['labels'][i]])
        probability_of_original_class = b['probabilities'][i, a['predicted'][i]]
        numeric(name + ':same class probability in CMERC', row['cmerc_probability_of_baseline_predicted_class'], probability_of_original_class)
        numeric(name + ':same class probability delta', row['baseline_predicted_class_probability_delta'], probability_of_original_class - ac[i])
        for threshold in THRESHOLDS:
            suffix = str(threshold).replace('.', 'p')
            ah, bh = not ao[i] and ac[i] >= threshold, not bo[i] and bc[i] >= threshold
            for prefix, expected in [('baseline_hc_error_ge_', ah), ('cmerc_hc_error_ge_', bh), ('new_cmerc_hc_error_ge_', bh and not ah)]:
                check(name + ':' + prefix + suffix, row[prefix + suffix].lower() == str(bool(expected)).lower())
            outcome = 'not_baseline_hc_error' if not ah else 'corrected' if bo[i] else 'still_wrong_confidence_lower' if bc[i] < ac[i] else 'still_wrong_confidence_not_lower'
            check(name + ':outcome ' + suffix, row['baseline_hc_outcome_ge_' + suffix] == outcome)
    transitions = read_csv(out / 'classification_transitions.csv')
    check('transition table has four rows', len(transitions) == 4)
    check('transition table unique keys', len({r['transition'] for r in transitions}) == 4)
    for row in transitions:
        count = result['classification_transitions'][row['transition']]
        numeric('transition:' + row['transition'] + ':count', row['count'], count)
        numeric('transition:' + row['transition'] + ':ratio', row['ratio'], count / 1623)
    hc = read_csv(out / 'high_confidence_transitions.csv')
    check('HC table has three unique thresholds', len(hc) == 3 and {float(r['threshold']) for r in hc} == set(THRESHOLDS))
    for row in hc:
        threshold = float(row['threshold'])
        expected = next(r for r in result['high_confidence_transitions'] if r['threshold'] == threshold)
        for field in ('baseline_high_confidence_predictions', 'baseline_high_confidence_errors', 'baseline_high_confidence_error_rate', 'cmerc_high_confidence_predictions', 'cmerc_high_confidence_errors', 'cmerc_high_confidence_error_rate'):
            numeric(f'HC {threshold}:{field}', row[field], expected[field])
        mapping = {'corrected': 'baseline_hce_corrected', 'still_wrong_confidence_lower': 'baseline_hce_still_wrong_confidence_lower', 'still_wrong_confidence_not_lower': 'baseline_hce_still_wrong_confidence_not_lower', 'still_wrong_above_threshold': 'baseline_hce_still_wrong_above_threshold', 'new_cmerc_hc_errors': 'new_cmerc_hce', 'new_cmerc_hc_errors_from_baseline_correct': 'new_cmerc_hce_baseline_correct'}
        for field, independent in mapping.items():
            numeric(f'HC {threshold}:{field}', row[field], expected[independent])
            if field + '_ratio' in row:
                numeric(f'HC {threshold}:{field}:ratio', row[field + '_ratio'], expected[independent] / expected['baseline_high_confidence_errors'])
        eligible = 1623 - expected['baseline_high_confidence_errors']
        numeric(f'HC {threshold}:eligible denominator', row['eligible_non_baseline_hc_errors'], eligible)
        numeric(f'HC {threshold}:new ratio', row['new_cmerc_hc_error_ratio_of_eligible'], expected['new_cmerc_hce'] / eligible)
        numeric(f'HC {threshold}:new baseline already wrong', row['new_cmerc_hc_errors_from_baseline_wrong_below_threshold'], expected['new_cmerc_hce'] - expected['new_cmerc_hce_baseline_correct'])
    classwise = read_csv(out / 'classwise_transitions.csv')
    check('classwise table six unique classes', len(classwise) == 6 and {int(r['true_label']) for r in classwise} == set(range(6)))
    for row in classwise:
        label = int(row['true_label'])
        expected = result['classwise_transitions'][label]
        check('classwise emotion mapping:' + str(label), row['true_emotion'] == EMOTIONS[label])
        mapping = {'support': 'support', 'baseline_errors': 'baseline_errors', 'cmerc_errors': 'cmerc_errors', 'corrected': 'baseline_wrong_cmerc_correct', 'regressed': 'baseline_correct_cmerc_wrong', 'both_wrong': 'both_wrong', 'net_error_change': 'net_error_change_cmerc_minus_baseline'}
        for field, independent in mapping.items():
            numeric(f'class {label}:{field}', row[field], expected[independent])
        numeric(f'class {label}:both correct', row['both_correct'], expected['support'] - expected['baseline_errors'] - expected['baseline_correct_cmerc_wrong'])
    comparison = read_csv(out / 'calibration_comparison.csv')
    check('calibration table two unique models', len(comparison) == 2 and {r['model_key'] for r in comparison} == {'baseline', 'cmerc'})
    for row in comparison:
        model = row['model_key']
        for field, expected in result['metrics'][model].items():
            numeric(model + ':calibration:' + field, row['valid_utterances' if field == 'valid_samples' else field], expected)
        check(model + ':calibration original file SHA', row['prediction_sha256'] == result['inputs'][model]['sha256'])
        numeric(model + ':calibration 10 bins', row['n_bins'], 10)
        numeric(model + ':no zero true probability', row['zero_true_probability_count'], 0)
    cbins = read_csv(out / 'calibration_bins.csv')
    check('calibration bins exactly 20 unique entries', len(cbins) == 20 and len({(r['model'], r['bin_index']) for r in cbins}) == 20)
    for row in cbins:
        model = 'baseline' if row['model'] == 'Official M³Net' else 'cmerc'
        index = int(row['bin_index'])
        expected = result['calibration_bins'][model][index]
        name = model + ':bin ' + str(index)
        for field in ('count', 'mean_confidence', 'accuracy', 'signed_gap'):
            numeric(name + ':' + field, row[field], expected[field])
        numeric(name + ':lower', row['lower_bound'], index / 10)
        numeric(name + ':upper', row['upper_bound'], (index + 1) / 10)
        gap = expected['signed_gap']
        numeric(name + ':absolute gap', row['absolute_gap'], abs(gap) if gap is not None else None)
        numeric(name + ':weight', row['weight'], expected['count'] / 1623)
        numeric(name + ':ECE contribution', row['ece_contribution'], expected['count'] / 1623 * abs(gap) if gap is not None else 0)
    cases = read_csv(out / 'typical_cases.csv')
    expected_cases = Counter()
    for threshold in THRESHOLDS:
        ah, bh = ~ao & (ac >= threshold), ~bo & (bc >= threshold)
        masks = {'corrected_baseline_hc_error': ah & bo,
                 'still_wrong_confidence_lower': ah & ~bo & (bc < ac),
                 'baseline_correct_new_cmerc_hc_error': ao & bh,
                 'both_high_confidence_wrong': ah & bh}
        for case_type, mask in masks.items():
            expected_cases.update((case_type, threshold, key(paired[i])) for i in np.flatnonzero(mask))
    actual_cases = Counter((r['case_type'], float(r['threshold']), key(r)) for r in cases)
    check('typical cases retain ALL four categories at ALL thresholds', expected_cases == actual_cases)
    paired_by_key = {key(r): r for r in paired}
    for i, row in enumerate(cases):
        original = paired_by_key[key(row)]
        check(f'case {i}:all paired fields unchanged', all(row[field] == value for field, value in original.items()))
        if row['case_type'] == 'both_high_confidence_wrong':
            score = min(float(original['baseline_predicted_confidence']), float(original['cmerc_predicted_confidence']))
        else:
            score_field = 'cmerc_predicted_confidence' if row['case_type'] == 'baseline_correct_new_cmerc_hc_error' else 'baseline_predicted_confidence'
            score = original[score_field]
        numeric(f'case {i}:selection score', row['selection_score'], score)
    result['typical_case_rows'] = len(cases)
    result['output_csv_files_checked'] = ['paired_predictions.csv', 'classification_transitions.csv', 'high_confidence_transitions.csv', 'classwise_transitions.csv', 'calibration_comparison.csv', 'calibration_bins.csv', 'typical_cases.csv']
    result['output_csv_sha256'] = {name: sha256(out / name) for name in result['output_csv_files_checked']}
    result['status'] = 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[6])
    parser.add_argument('--sources-only', action='store_true')
    args = parser.parse_args()
    out = Path(__file__).resolve().parents[1]
    result, rows, arrays = recompute(args.root.resolve(), out)
    if not args.sources_only:
        audit_outputs(out, result, rows, arrays)
    target = out / 'independent_verification.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'checks': len(result['checks']), 'output': str(target)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
