#!/usr/bin/env python3
"""Independent CPU-only audit; does not import the experiment statistics code."""
import argparse
import csv
import hashlib
import json
import math
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EMOTIONS = ('neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger')
PRIMARY = (0.8, 0.9, 0.95)
AUXILIARY = (0.4, 0.5, 0.6, 0.7)
THRESHOLDS = PRIMARY + AUXILIARY
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
        'baseline': out / 'baseline_predictions.csv',
        'cmerc': root / 'checkpoints/analysis/official_checkpoint_verification/MELD_predictions.csv',
    }
    rows = {name: read_csv(path) for name, path in sources.items()}
    arrays, metrics, bins = {}, {}, {}
    for name, records in rows.items():
        check(f'{name}:2610 effective utterances', len(records) == 2610)
        check(f'{name}:unique stable IDs', len(set(map(key, records))) == len(records))
        check(f'{name}:unique dialogue/source IDs', len({(r['dialogue_id'], r['source_utterance_id']) for r in records}) == len(records))
        check(f'{name}:dataset and full input', all(r['dataset'] == 'MELD' and r['input_condition'] == 'full' for r in records))
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
        for c in range(7):
            tp = confusion[c, c]
            fp = sum(confusion[t, c] for t in range(7) if t != c)
            fn = sum(confusion[c, p] for p in range(7) if p != c)
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
            'brier_score': float(np.square(probabilities - np.eye(7)[labels]).sum(1).mean()),
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
    check('four transitions partition all utterances', sum(transitions.values()) == 2610)
    thresholds = []
    for threshold in THRESHOLDS:
        ah, bh = ~ao & (ac >= threshold), ~bo & (bc >= threshold)
        item = {'threshold': threshold, 'threshold_group': 'primary' if threshold in PRIMARY else 'auxiliary'}
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

    evaluation = read_csv(out / 'baseline_evaluation_results.csv')
    check('evaluation table two fixed models', len(evaluation) == 2 and {r['model'] for r in evaluation} == {records[0]['model'] for records in sources.values()} and all(r['dataset'] == 'MELD' for r in evaluation))
    for model in ('baseline', 'cmerc'):
        row = next(r for r in evaluation if r['model'] == sources[model][0]['model'])
        for field in ('accuracy', 'weighted_f1', 'macro_f1'):
            numeric(model + ' evaluation independent:' + field, row[field], result['metrics'][model][field] * 100)
        numeric(model + ' evaluation independent:sample count', row['valid_samples'], 2610)

    paired = read_csv(out / 'paired_predictions.csv')
    check('paired:exactly 2610 unique rows', len(paired) == 2610 and len(set(map(key, paired))) == 2610)
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
        numeric('transition:' + row['transition'] + ':ratio', row['ratio'], count / 2610)
    primary_hc = read_csv(out / 'high_confidence_transitions.csv')
    auxiliary_hc = read_csv(out / 'auxiliary_threshold_transitions.csv')
    check('primary HC table has exactly original three thresholds', len(primary_hc) == 3 and {float(r['threshold']) for r in primary_hc} == set(PRIMARY))
    check('auxiliary HC table has exactly four predeclared thresholds', len(auxiliary_hc) == 4 and {float(r['threshold']) for r in auxiliary_hc} == set(AUXILIARY))
    hc = primary_hc + auxiliary_hc
    for row in hc:
        threshold = float(row['threshold'])
        expected = next(r for r in result['high_confidence_transitions'] if r['threshold'] == threshold)
        for field in ('baseline_high_confidence_predictions', 'baseline_high_confidence_errors', 'baseline_high_confidence_error_rate', 'cmerc_high_confidence_predictions', 'cmerc_high_confidence_errors', 'cmerc_high_confidence_error_rate'):
            numeric(f'HC {threshold}:{field}', row[field], expected[field])
        mapping = {'corrected': 'baseline_hce_corrected', 'still_wrong_confidence_lower': 'baseline_hce_still_wrong_confidence_lower', 'still_wrong_confidence_not_lower': 'baseline_hce_still_wrong_confidence_not_lower', 'still_wrong_above_threshold': 'baseline_hce_still_wrong_above_threshold', 'new_cmerc_hc_errors': 'new_cmerc_hce', 'new_cmerc_hc_errors_from_baseline_correct': 'new_cmerc_hce_baseline_correct'}
        for field, independent in mapping.items():
            numeric(f'HC {threshold}:{field}', row[field], expected[independent])
            if field + '_ratio' in row:
                numeric(f'HC {threshold}:{field}:ratio', row[field + '_ratio'], expected[independent] / expected['baseline_high_confidence_errors'] if expected['baseline_high_confidence_errors'] else None)
        eligible = 2610 - expected['baseline_high_confidence_errors']
        numeric(f'HC {threshold}:eligible denominator', row['eligible_non_baseline_hc_errors'], eligible)
        numeric(f'HC {threshold}:new ratio', row['new_cmerc_hc_error_ratio_of_eligible'], expected['new_cmerc_hce'] / eligible)
        numeric(f'HC {threshold}:new baseline already wrong', row['new_cmerc_hc_errors_from_baseline_wrong_below_threshold'], expected['new_cmerc_hce'] - expected['new_cmerc_hce_baseline_correct'])
    classwise = read_csv(out / 'classwise_transitions.csv')
    check('classwise table seven unique classes', len(classwise) == 7 and {int(r['true_label']) for r in classwise} == set(range(7)))
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
        model = 'cmerc' if row['model'] == 'Official CMERC' else 'baseline'
        index = int(row['bin_index'])
        expected = result['calibration_bins'][model][index]
        name = model + ':bin ' + str(index)
        for field in ('count', 'mean_confidence', 'accuracy', 'signed_gap'):
            numeric(name + ':' + field, row[field], expected[field])
        numeric(name + ':lower', row['lower_bound'], index / 10)
        numeric(name + ':upper', row['upper_bound'], (index + 1) / 10)
        gap = expected['signed_gap']
        numeric(name + ':absolute gap', row['absolute_gap'], abs(gap) if gap is not None else None)
        numeric(name + ':weight', row['weight'], expected['count'] / 2610)
        numeric(name + ':ECE contribution', row['ece_contribution'], expected['count'] / 2610 * abs(gap) if gap is not None else 0)
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
    check('case primary and auxiliary threshold groups match', all(r['threshold_group'] == ('primary' if float(r['threshold']) in PRIMARY else 'auxiliary') for r in cases))
    result['typical_case_rows'] = len(cases)
    result['output_csv_files_checked'] = ['baseline_evaluation_results.csv', 'baseline_classification_report.csv', 'baseline_confusion_matrix.csv', 'paired_predictions.csv', 'classification_transitions.csv', 'high_confidence_transitions.csv', 'auxiliary_threshold_transitions.csv', 'classwise_transitions.csv', 'calibration_comparison.csv', 'calibration_bins.csv', 'typical_cases.csv']
    result['output_csv_sha256'] = {name: sha256(out / name) for name in result['output_csv_files_checked']}
    result['status'] = 'passed'


def audit_raw_and_protocol(root, out, result, rows, arrays):
    """Check immutable evidence and raw features without importing inference code."""
    def check(name, condition):
        result['checks'].append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(name)

    audit = json.loads((out / 'baseline_checkpoint_audit.json').read_text())
    baseline = json.loads((out / 'baseline_inference/MELD_run_manifest.json').read_text())
    official_dir = root / 'checkpoints/analysis/official_checkpoint_verification'
    official = json.loads((official_dir / 'MELD_run_manifest.json').read_text())
    inventory = json.loads((out / 'baseline_inference/MELD_checkpoint_inventory.json').read_text())
    check('fixed first compatible candidate before inference', audit['selection_fixed_before_new_inference'])
    check('baseline local historical epoch5 test selection disclosed',
          audit['checkpoint']['epoch_1based'] == 5 and
          audit['checkpoint']['selection_protocol'] == 'official_test_selection' and
          audit['checkpoint']['args']['seed'] == 67137)
    for flag in ('calibrate', 'contrastlearning', 'courselearning'):
        check('baseline historical training flag disabled:' + flag, audit['checkpoint']['args'][flag] is False)
    for label, path, expected in [
        ('historical full checkpoint', Path(audit['checkpoint']['path']), audit['checkpoint']['sha256']),
        ('exported inference state', Path(audit['export']['path']), audit['export']['sha256']),
        ('official CMERC state', Path(audit['official_cmerc_reference']['path']), audit['official_cmerc_reference']['sha256']),
    ]:
        check(label + ' SHA256 unchanged', path.exists() and sha256(path) == expected)
    check('exported state source tensor fingerprint identical', audit['state']['tensor_content_fingerprint_sha256'] == audit['export']['tensor_content_fingerprint_sha256'] and audit['export']['source_state_identical'])
    check('strict83 tensor model compatibility complete', inventory['tensor_keys'] == 83 and len(inventory['parameters']) == 83 and all(not v for v in inventory['compatibility'].values()))
    check('fresh forward model state unchanged', baseline['model_state_sha256_before'] == baseline['model_state_sha256_after'])
    check('cached CMERC CSV SHA is actual source', baseline['cached_cmerc_prediction_sha256'] == result['inputs']['cmerc']['sha256'])
    check('zero repeated CMERC inference', baseline['cmerc_inference_calls'] == 0)
    for name, manifest in [('baseline', baseline), ('cmerc', official)]:
        check(name + ':completed on official environment', manifest['status'] == 'completed' and manifest['host'] == 'user22')
        for field in ('model_state_unchanged', 'all_parameter_grads_none', 'original_inputs_unchanged', 'cudnn_enabled', 'cudnn_deterministic'):
            check(name + ':' + field, manifest[field] is True)
        for field in ('training', 'modality_removal'):
            check(name + ':' + field + ' false', manifest[field] is False)
        run_audit = manifest['audit']
        check(name + ':strict load,18 full eval/no-grad forwards,no updates',
              run_audit['strict_load_calls'] == 1 and run_audit['forward_calls'] == 18 and
              run_audit['optimizer_steps'] == 0 and run_audit['masking_forward_calls'] == 0 and
              run_audit['gradient_enabled_observations'] == [False] * 18 and
              run_audit['model_training_observations'] == [False] * 18)
        check(name + ':no forbidden GPU4', manifest['actual_gpu_uuids'] == ['GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049'])
    for field in ('python', 'torch', 'numpy', 'sklearn', 'cuda_runtime', 'cudnn_version', 'effective_args', 'official_argv'):
        check('same official runtime protocol:' + field, baseline[field] == official[field])
    check('same actual batch IDs/padding/lengths/order', baseline['audit']['batches'] == official['audit']['batches'])
    for relative, expected in baseline['runtime_sha256'].items():
        check('same frozen runtime SHA:' + relative, sha256(official_dir / 'code_runtime' / relative) == expected)
    for fname, evidence in audit['data'].items():
        path = Path(evidence['current_path'])
        check('raw feature SHA:' + fname, path.exists() and sha256(path) == evidence['sha256'])
    # Trusted local feature file, exact previously recorded SHA checked above.
    raw_path = Path(audit['data']['MELD_features_raw1.pkl']['current_path'])
    with raw_path.open('rb') as stream:
        raw = pickle.load(stream, encoding='latin1')
    ids, labels, texts, test_dialogues = raw[0], raw[2], raw[6], list(raw[8])
    expected_keys = [('MELD', str(dialogue), i) for dialogue in test_dialogues for i in range(len(labels[dialogue]))]
    check('raw official test split has 280 dialogues and2610 valid utterances', len(test_dialogues) == 280 and len(expected_keys) == 2610)
    for name, records in rows.items():
        check(name + ':raw test dialogue-major flatten order exact', list(map(key, records)) == expected_keys)
        for i, row in enumerate(records):
            dialogue, utterance = int(row['dialogue_id']), int(row['utterance_index'])
            check(f'{name}:raw source labels/text/source ID:{i}',
                  int(row['true_label']) == int(labels[dialogue][utterance]) and
                  row['utterance_text'] == str(texts[dialogue][utterance]) and
                  row['source_utterance_id'] == str(ids[dialogue][utterance]))
        npz_path = out / 'baseline_predictions.npz' if name == 'baseline' else official_dir / 'MELD_predictions.npz'
        with np.load(npz_path, allow_pickle=False) as data:
            for field, expected in [('labels', arrays[name]['labels']), ('predictions', arrays[name]['predicted']), ('probabilities', arrays[name]['probabilities'])]:
                check(name + ':CSV exact numeric NPZ agreement:' + field, np.array_equal(data[field], expected))
            check(name + ':NPZ finite logsoftmax', data['log_probabilities'].shape == (2610, 7) and np.isfinite(data['log_probabilities']).all())
            check(name + ':raw logsoftmax exp no renormalization (cross-platform exp rounding)', np.allclose(np.exp(data['log_probabilities'].astype(np.float64)), data['probabilities'], rtol=0, atol=1e-15))
        for metric in ('accuracy', 'weighted_f1', 'macro_f1'):
            check(name + ':official recorded classification ' + metric, close(result['metrics'][name][metric] * 100, (baseline if name == 'baseline' else official)['classification_metrics'][metric]))
        report_path = out / 'baseline_classification_report.csv' if name == 'baseline' else official_dir / 'MELD_classification_report.csv'
        matrix_path = out / 'baseline_confusion_matrix.csv' if name == 'baseline' else official_dir / 'MELD_confusion_matrix.csv'
        report, matrix = read_csv(report_path), read_csv(matrix_path)
        check(name + ':classification seven report rows', len(report) == 7)
        check(name + ':confusion seven rows', len(matrix) == 7)
        table = Counter(zip(arrays[name]['labels'].tolist(), arrays[name]['predicted'].tolist()))
        for label, emotion in enumerate(EMOTIONS):
            tp = table[label, label]
            support = sum(table[label, j] for j in range(7))
            predicted_support = sum(table[j, label] for j in range(7))
            precision = tp / predicted_support if predicted_support else 0
            recall = tp / support if support else 0
            f1 = 2 * tp / (support + predicted_support) if support + predicted_support else 0
            row = next(r for r in report if int(r['class_id']) == label)
            check(name + ':classification class mapping:' + emotion, row['class_name'] == emotion)
            check(name + ':classification support:' + emotion, int(row['support']) == support)
            for field, expected in [('precision', precision), ('recall', recall), ('f1', f1)]:
                check(name + ':classification ' + field + ':' + emotion, close(row[field], expected * 100))
            mrow = next(r for r in matrix if r['true_emotion'] == emotion)
            check(name + ':confusion true row:' + emotion, all(int(mrow[e]) == table[label, j] for j, e in enumerate(EMOTIONS)))
    result['evidence'] = {
        'historical_full_checkpoint_sha256': audit['checkpoint']['sha256'],
        'inference_export_sha256': audit['export']['sha256'],
        'identical_tensor_fingerprint_sha256': audit['export']['tensor_content_fingerprint_sha256'],
        'baseline_selection_protocol': 'official_test_selection, epoch 5, training seed 67137',
        'strict_training_pairing': False,
        'warning': 'Local README baseline and author official CMERC; same inference protocol, unknown training pairing; no module causal attribution.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[6])
    parser.add_argument('--sources-only', action='store_true')
    args = parser.parse_args()
    out = Path(__file__).resolve().parents[1]
    result, rows, arrays = recompute(args.root.resolve(), out)
    audit_raw_and_protocol(args.root.resolve(), out, result, rows, arrays)
    if not args.sources_only:
        audit_outputs(out, result, rows, arrays)
    target = out / 'independent_verification.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'checks': len(result['checks']), 'output': str(target)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
