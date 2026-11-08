#!/usr/bin/env python
"""Validation-only representation sensitivity; paired dialogue bootstrap, no model fit."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
BEHAVIOR_NAMES = ('flip_rate', 'true_probability_change', 'confidence_change', 'js_divergence',
                  'correct_to_wrong_rate', 'wrong_to_correct_rate')


def safe_divide(numerator, denominator):
    a, b = np.broadcast_arrays(np.asarray(numerator, float), np.asarray(denominator, float))
    return np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)


def classification_from_confusion(confusion):
    """Sklearn's default union-of-present-labels macro F1; all values are fractions."""
    cm = np.asarray(confusion, float)
    support = cm.sum(axis=-1)
    predicted = cm.sum(axis=-2)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    total = support.sum(axis=-1)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom != 0)
    return dict(accuracy=safe_divide(tp.sum(-1), total),
                weighted_f1=safe_divide((f1 * support).sum(-1), total),
                macro_f1=safe_divide(f1.sum(-1), (denom > 0).sum(-1)))


def dialogue_bootstrap_weights(dialogues, repeats, seed):
    """Draw from the ORIGINAL dialogue pool; reuse these weights for every subgroup."""
    _, indices = np.unique(np.asarray(dialogues), return_inverse=True)
    n = int(indices.max()) + 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(repeats, n))
    weights = np.zeros((repeats, n), dtype=np.float64)
    np.add.at(weights, (np.repeat(np.arange(repeats), n), draws.ravel()), 1)
    return indices, weights


def probability_behavior(labels, original, perturbed):
    p, q = np.asarray(original, float), np.asarray(perturbed, float)
    y = np.asarray(labels, int)
    before, after = p.argmax(-1), q.argmax(-1)
    correct, changed_correct = before == y, after == y[:, None]
    flip = before[:, None] != after
    midpoint = (p[:, None] + q) / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        left = np.where(p[:, None] > 0, p[:, None] * np.log(p[:, None] / midpoint), 0).sum(-1)
        right = np.where(q > 0, q * np.log(q / midpoint), 0).sum(-1)
    return np.stack([flip, q[np.arange(len(y)), :, y] - p[np.arange(len(y)), y][:, None],
                     q.max(-1) - p.max(-1)[:, None], (left + right) / 2,
                     correct[:, None] & ~changed_correct, ~correct[:, None] & changed_correct], axis=-1)


def _interval(result, name, estimate, samples):
    result[name] = float(estimate)
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    low, high = np.quantile(samples, [.025, .975]) if len(samples) else (np.nan, np.nan)
    result[name + '_ci_low'] = float(low)
    result[name + '_ci_high'] = float(high)
    result[name + '_valid_bootstraps'] = len(samples)


def summarize_condition(labels, original, perturbed, selected, dialogue_indices, weights, *, f1=True):
    """Aggregate to dialogue sums/confusion matrices before all bootstrap draws.

    Random repeats are evaluated individually and then averaged. Subgroup rows
    are filtered AFTER the original-pool bootstrap weights have been generated.
    Draws containing no subgroup utterances remain NA and are excluded from CIs.
    """
    y, p, q = np.asarray(labels, int), np.asarray(original, float), np.asarray(perturbed, float)
    selected, di = np.asarray(selected, bool), np.asarray(dialogue_indices, int)
    d, r, c = weights.shape[1], q.shape[1], p.shape[1]
    count = np.bincount(di[selected], minlength=d).astype(float)
    # Compress zero-support dialogues only AFTER original-pool draws exist.
    live = np.flatnonzero(count)
    w = weights[:, live]
    boot_n = w @ count[live]
    result = dict(n_utterances=int(selected.sum()), n_dialogues_present=len(live), repeats=r,
                  bootstrap_unit='original_dialogue_pool', metric_scale='fraction; natural-log JS',
                  aggregation='per-repeat classification metric then mean; no probability ensemble')
    behavior = probability_behavior(y, p, q)
    sums = np.zeros((d, r, len(BEHAVIOR_NAMES)), float)
    np.add.at(sums, di[selected], behavior[selected])
    draws = (w @ sums[live].reshape(len(live), r * len(BEHAVIOR_NAMES))).reshape(len(weights), r, -1) if len(live) else np.zeros((len(weights), r, len(BEHAVIOR_NAMES)))
    mean_draws = safe_divide(draws, boot_n[:, None, None]).mean(axis=1)
    point = behavior[selected].mean(axis=(0, 1)) if selected.any() else np.full(len(BEHAVIOR_NAMES), np.nan)
    for k, name in enumerate(BEHAVIOR_NAMES):
        _interval(result, name, point[k], mean_draws[:, k])
    before, after = p.argmax(1), q.argmax(2)
    original_correct, perturbed_correct = before == y, after == y[:, None]
    correct_sums = np.zeros((d, 1 + r), float)
    np.add.at(correct_sums, di[selected], np.column_stack([original_correct, perturbed_correct])[selected])
    accuracy_draws = safe_divide(w @ correct_sums[live], boot_n[:, None])
    original_accuracy = float(original_correct[selected].mean()) if selected.any() else np.nan
    perturbed_accuracy = float(perturbed_correct[selected].mean()) if selected.any() else np.nan
    _interval(result, 'original_accuracy', original_accuracy, accuracy_draws[:, 0])
    _interval(result, 'perturbed_accuracy', perturbed_accuracy, accuracy_draws[:, 1:].mean(1))
    _interval(result, 'delta_accuracy', perturbed_accuracy - original_accuracy,
              accuracy_draws[:, 1:].mean(1) - accuracy_draws[:, 0])
    if f1:
        original_cm = np.zeros((d, c, c), float)
        perturbed_cm = np.zeros((d, r, c, c), float)
        np.add.at(original_cm, (di[selected], y[selected], before[selected]), 1)
        for repeat in range(r):
            np.add.at(perturbed_cm[:, repeat], (di[selected], y[selected], after[selected, repeat]), 1)
        base_point = classification_from_confusion(original_cm.sum(0))
        pert_point = classification_from_confusion(perturbed_cm.sum(0))
        base_draws = classification_from_confusion((w @ original_cm[live].reshape(len(live), c*c)).reshape(-1, c, c)) if len(live) else {k: np.full(len(weights), np.nan) for k in ('weighted_f1', 'macro_f1')}
        pert_draws = classification_from_confusion((w @ perturbed_cm[live].reshape(len(live), r*c*c)).reshape(-1, r, c, c)) if len(live) else {k: np.full((len(weights), r), np.nan) for k in ('weighted_f1', 'macro_f1')}
        for name in ('weighted_f1', 'macro_f1'):
            b, a = base_point[name], pert_point[name].mean()
            _interval(result, 'original_' + name, b, base_draws[name])
            _interval(result, 'perturbed_' + name, a, pert_draws[name].mean(1))
            _interval(result, 'delta_' + name, a - b, pert_draws[name].mean(1) - base_draws[name])
    return result


def correctness_contrasts(labels, original, perturbed, dialogue_indices, weights):
    """Direct wrong-minus-correct difference CIs using identical cluster draws.

    These are descriptive associations, unadjusted for original confidence and
    class composition. Fixed mask draws are not resampled by this bootstrap.
    """
    correct = original.argmax(1) == labels
    di = np.asarray(dialogue_indices, int)
    behavior = probability_behavior(labels, original, perturbed)
    indices = [0, 3, 2, 1]
    names = ['flip_rate', 'js_divergence', 'confidence_change', 'true_probability_change']
    behavior = behavior[:, :, indices].mean(axis=1)
    points, draws = {}, {}
    for name, selected in [('correct', correct), ('wrong', ~correct)]:
        counts = np.bincount(di[selected], minlength=weights.shape[1])
        sums = np.zeros((weights.shape[1], len(indices)), float)
        np.add.at(sums, di[selected], behavior[selected])
        points[name] = behavior[selected].mean(axis=0) if selected.any() else np.full(len(indices), np.nan)
        draws[name] = safe_divide(weights @ sums, (weights @ counts)[:, None])
    differences = draws['wrong'] - draws['correct']
    result = {}
    for index, name in enumerate(names):
        values = differences[:, index]
        values = values[np.isfinite(values)]
        low, high = np.quantile(values, [.025, .975]) if len(values) else (np.nan, np.nan)
        result[name] = dict(correct_mean=float(points['correct'][index]), wrong_mean=float(points['wrong'][index]),
                            difference_wrong_minus_correct=float(points['wrong'][index]-points['correct'][index]),
                            ci_low=float(low), ci_high=float(high), valid_bootstraps=len(values),
                            original_correct_count=int(correct.sum()), original_wrong_count=int((~correct).sum()),
                            interpretation='unadjusted association; paired original-dialogue draws; fixed masks')
    return result


def validate_payload(baseline, payload, repeats, target):
    for key in ('labels', 'dialogue_ids', 'utterance_indices', 'speaker_ids', 'batch_indices', 'dialogue_lengths'):
        if not np.array_equal(baseline[key], payload[key]):
            raise ValueError('target metadata mismatch: ' + key)
    p, q = baseline['probabilities'], payload['probabilities']
    if q.shape != (len(p), repeats, p.shape[1]):
        raise ValueError('probability shape/repeats mismatch')
    if not np.isfinite(q).all() or (q < 0).any() or not np.allclose(q.sum(-1), 1, atol=1e-6, rtol=1e-5):
        raise ValueError('invalid perturbed distribution')
    if not np.allclose(np.exp(payload['log_probabilities']), q, atol=1e-6, rtol=1e-5):
        raise ValueError('log probability disagreement')
    if not np.array_equal(payload['original_probabilities'], p) or not np.array_equal(payload['original_predictions'], p.argmax(1)):
        raise ValueError('original distribution/prediction changed')
    if not np.array_equal(payload['perturbed_predictions'], q.argmax(2)):
        raise ValueError('perturbed prediction disagreement')
    active, missing = np.asarray(payload['active']), np.asarray(payload['visual_missing'])
    if active.shape != (len(p),) or active.dtype != bool or missing.shape != (len(p),) or missing.dtype != bool:
        raise ValueError('missing/active masks must be boolean N vectors')
    if not np.array_equal(active, ~missing if target == 'V' else np.ones(len(p), bool)):
        raise ValueError('active mask disagrees with missingness protocol')
    if not np.array_equal(q[~active], np.repeat(p[~active, None, :], repeats, axis=1)):
        raise ValueError('inactive visual probability was modified')
    if 'log_probabilities' in baseline and not np.array_equal(payload['log_probabilities'][~active],
                                         np.repeat(baseline['log_probabilities'][~active, None, :], repeats, axis=1)):
        raise ValueError('inactive visual log probability was modified')


def read_npz(path):
    with np.load(path, allow_pickle=False) as file:
        return {k: file[k] for k in file.files}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('empty table: ' + str(path))
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar(payload, key):
    return np.asarray(payload[key]).item()


def compare_previous_pilot(previous_rows, dataset, target, baseline, detail):
    modality = {'T-All':'text_representation', 'A':'audio', 'V':'visual'}[target]
    records = {}
    for row in previous_rows:
        if row['dataset'] == dataset and row['modality'] == modality:
            key = (row['dialogue_id'], int(row['utterance_index']))
            records.setdefault(key, {})[row['perturbation_type']] = float(row['flip_rate'])
    stable = []
    for dialogue, ui in zip(baseline['dialogue_ids'], baseline['utterance_indices']):
        values = records.get((str(dialogue), int(ui)), {})
        if set(values) != {'gaussian','dropout','attenuation'}:
            raise ValueError('incomplete prior validation pilot identity')
        stable.append(all(v == 0 for v in values.values()))
    stable = np.asarray(stable, bool)
    wrong = baseline['predictions'] != baseline['labels']
    strong_flip = np.asarray(detail['flip'])[:,0]
    return dict(dataset=dataset, target=target, n_utterances=len(stable),
                weak_stable_count=int(stable.sum()), strong_flip_among_weak_stable=int((stable & strong_flip).sum()),
                strong_trueprob_drop_ge_point1_among_weak_stable=int((stable & (detail['true_probability_change'][:,0] <= -.1)).sum()),
                weak_stable_wrong_count=int((stable & wrong).sum()),
                strong_flip_among_weak_stable_wrong=int((stable & wrong & strong_flip).sum()),
                weak_definition='zero flips in all three prior 0.05/0.05/0.95 conditions; 5/5/1 draws',
                strong_definition='current overall target-only mean contraction rho1; other utterances unchanged')


def _tag(dataset, seed, payload):
    return dict(dataset=dataset, seed=seed, phase=str(scalar(payload, 'phase')),
                condition_id=str(scalar(payload, 'condition_id')), target=str(scalar(payload, 'target')),
                kind=str(scalar(payload, 'kind')), strength=float(scalar(payload, 'strength')),
                repeats=int(scalar(payload, 'repeats')))


def run(root):
    from evidence_core import condition_specs, sensitivity_metrics
    config = json.loads((root / 'configs/evidence_sensitivity_config.json').read_text())
    if config['inference']['split'] != 'validation' or config['analysis']['bootstrap_unit'] != 'dialogue_id':
        raise ValueError('outside validation-only protocol')
    output = root / 'results'; output.mkdir(parents=True, exist_ok=True)
    summaries, classes, missingness, high_cases, contrasts, previous_comparisons = [], [], [], [], [], []
    previous_path = Path(config['previous_pilot']) / 'results/utterance_behavior_metrics.csv'
    previous_rows = list(csv.DictReader(previous_path.open())) if previous_path.exists() else []
    sample_path = output / 'per_utterance_sensitivity.csv'
    total_vectors = 0
    with sample_path.open('w', newline='', encoding='utf-8') as file:
        writer = None
        for dataset, seed in config['anchors'].items():
            directory = root / 'runs' / dataset / f'seed{seed}'
            base = read_npz(directory / 'baseline_validation.npz')
            y, p = base['labels'], base['probabilities']
            expected = config['validation_sizes'][dataset]
            if len(y) != expected['utterances'] or len(np.unique(base['dialogue_ids'])) != expected['dialogues']:
                raise ValueError('incorrect validation size')
            if not np.array_equal(base['predictions'], p.argmax(1)):
                raise ValueError('incorrect baseline predictions')
            di, weights = dialogue_bootstrap_weights(base['dialogue_ids'], config['analysis']['bootstrap_repeats'], config['analysis']['bootstrap_seed'])
            np.savez_compressed(output / f'{dataset}_paired_bootstrap_weights.npz',
                                dialogue_ids=np.unique(base['dialogue_ids']), weights=weights.astype(np.uint16))
            correct = base['predictions'] == y
            visual_reference = None
            dataset_conditions = 0
            for phase in config['phases']:
                for spec in condition_specs(phase):
                    condition_id = spec['id']
                    data = read_npz(directory / 'conditions' / f'{condition_id}.npz')
                    tag = _tag(dataset, seed, data)
                    if tag['condition_id'] != condition_id or tag['phase'] != phase:
                        raise ValueError('condition identity mismatch')
                    for key in ('target', 'kind', 'strength', 'repeats'):
                        if tag[key] != spec[key]:
                            raise ValueError('condition configuration mismatch: ' + key)
                    validate_payload(base, data, tag['repeats'], tag['target'])
                    missing = data['visual_missing']
                    if visual_reference is None:
                        visual_reference = missing.copy()
                        if int(missing.sum()) != config['visual_missingness']['expected_counts'][dataset]:
                            raise ValueError('visual missingness count mismatch')
                    elif not np.array_equal(missing, visual_reference):
                        raise ValueError('visual missingness differs across conditions')
                    q = data['probabilities']
                    detail = sensitivity_metrics(y, p, q)
                    if previous_rows and phase == 'overall_attenuation' and tag['strength'] == 1:
                        previous_comparisons.append(compare_previous_pilot(previous_rows, dataset, tag['target'], base, detail))
                    for metric, values in correctness_contrasts(y, p, q, di, weights).items():
                        contrasts.append(dict(tag, metric=metric, **values))
                    groups = dict(all=np.ones(len(y), bool), original_correct=correct,
                                  original_wrong=~correct, visual_present=~missing, visual_missing=missing)
                    for group, selected in groups.items():
                        row = dict(tag, subgroup=group, active_utterances=int(data['active'][selected].sum()))
                        row.update(summarize_condition(y, p, q, selected, di, weights))
                        summaries.append(row)
                    for basis, labels in [('true_class', y), ('original_prediction_class', base['predictions'])]:
                        for class_index in range(p.shape[1]):
                            row = dict(tag, class_basis=basis, class_index=class_index)
                            row.update(summarize_condition(y, p, q, labels == class_index, di, weights, f1=False))
                            classes.append(row)
                    for index in range(len(y)):
                        for repeat in range(tag['repeats']):
                            row = dict(tag, dialogue_id=str(base['dialogue_ids'][index]),
                                       utterance_index=int(base['utterance_indices'][index]),
                                       speaker_id=int(base['speaker_ids'][index]), batch_index=int(base['batch_indices'][index]),
                                       dialogue_length=int(base['dialogue_lengths'][index]), repeat=repeat,
                                       label=int(y[index]), active=bool(data['active'][index]), visual_missing=bool(missing[index]),
                                       original_correct=bool(correct[index]),
                                       original_prediction=int(detail['original_prediction'][index]),
                                       perturbed_prediction=int(detail['perturbed_prediction'][index, repeat]),
                                       original_confidence=float(detail['original_confidence'][index]),
                                       perturbed_confidence=float(detail['perturbed_confidence'][index, repeat]),
                                       original_true_probability=float(detail['original_true_probability'][index]),
                                       perturbed_true_probability=float(detail['perturbed_true_probability'][index, repeat]))
                            for key in ('flip', 'true_probability_change', 'confidence_change', 'js_divergence',
                                        'correct_to_wrong', 'wrong_to_correct'):
                                row[key] = float(detail[key][index, repeat])
                            if writer is None:
                                writer = csv.DictWriter(file, fieldnames=list(row)); writer.writeheader()
                            writer.writerow(row)
                            if (tag['kind'] == 'attenuation' and tag['strength'] == 1 and not correct[index]
                                    and row['original_confidence'] >= config['analysis']['high_confidence_threshold']
                                    and (row['flip'] or row['true_probability_change'] <= -.1)):
                                high_cases.append(row)
                    total_vectors += len(y) * tag['repeats']; dataset_conditions += 1
            for group, selected in [('visual_present', ~visual_reference), ('visual_missing', visual_reference)]:
                values = summarize_condition(y, p, p[:, None], selected, di, weights, f1=False)
                missingness.append(dict(dataset=dataset, seed=seed, subgroup=group,
                                        n_total=len(y), n_utterances=int(selected.sum()),
                                        original_correct_count=int(correct[selected].sum()),
                                        original_accuracy=values['original_accuracy'],
                                        original_accuracy_ci_low=values['original_accuracy_ci_low'],
                                        original_accuracy_ci_high=values['original_accuracy_ci_high'],
                                        visual_intervention_applied=(group != 'visual_missing'),
                                        handling='missing rows retained; V missing inputs/probabilities unchanged'))
            if dataset_conditions != 56:
                raise ValueError('expected 56 conditions per dataset')
    write_csv(output / 'modality_sensitivity_summary.csv', summaries)
    # Keep the earlier reader interface byte-identical to the requested filename.
    (output / 'sensitivity_summary.csv').write_bytes((output / 'modality_sensitivity_summary.csv').read_bytes())
    write_csv(output / 'classwise_sensitivity.csv', classes)
    write_csv(output / 'correctness_group_contrasts.csv', contrasts)
    if previous_comparisons:
        write_csv(output / 'previous_pilot_sensitivity_comparison.csv', previous_comparisons)
    write_csv(output / 'visual_missingness_analysis.csv', missingness)
    if high_cases:
        write_csv(output / 'high_confidence_wrong_sensitive_cases.csv', high_cases)
    else:
        (output / 'high_confidence_wrong_sensitive_cases.csv').write_text('dataset,condition_id,dialogue_id,utterance_index\n')
    monotonicity = monotonicity_rows(summaries)
    write_csv(output / 'monotonicity_analysis.csv', monotonicity)
    write_reports(root, config, summaries, classes, missingness, high_cases, monotonicity, contrasts, previous_comparisons)
    return dict(conditions=len(summaries)//5, summary_rows=len(summaries),
                per_repeat_utterance_rows=total_vectors, classwise_rows=len(classes),
                high_confidence_wrong_sensitive_rows=len(high_cases), analysis_split='validation')


def monotonicity_rows(summaries):
    grouped = {}
    for row in summaries:
        if row['kind'] == 'attenuation':
            grouped.setdefault((row['dataset'], row['target'], row['subgroup']), []).append(row)
    result = []
    for (dataset, target, subgroup), rows in grouped.items():
        rows.sort(key=lambda row: row['strength'])
        accuracy = np.array([rows[0]['original_accuracy']] + [r['perturbed_accuracy'] for r in rows])
        truechange = np.array([0.] + [r['true_probability_change'] for r in rows])
        flip = np.array([0.] + [r['flip_rate'] for r in rows])
        da, dt, df = np.diff(accuracy), np.diff(truechange), np.diff(flip)
        result.append(dict(dataset=dataset, target=target, subgroup=subgroup,
                           rho=json.dumps([0.] + [r['strength'] for r in rows]),
                           accuracy=json.dumps(accuracy.tolist()), true_probability_change=json.dumps(truechange.tolist()),
                           flip_rate=json.dumps(flip.tolist()), accuracy_increases=int((da > 1e-12).sum()),
                           true_probability_increases=int((dt > 1e-12).sum()), flip_rate_decreases=int((df < -1e-12).sum()),
                           interpretation='descriptive reversals, no monotonicity assumption; exploratory'))
    return result


def fmt(value):
    return f'{value:.5g}' if np.isfinite(value) else 'NA'


def condition_table(rows, strength_label='rho/ratio'):
    lines = [f'| 数据集 | 目标 | {strength_label} | 子组 | n | ΔAcc | ΔWeighted-F1 | ΔMacro-F1 | Flip | ΔP(true) | ΔConfidence | JS |',
             '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        fields = [r['dataset'], r['target'], str(r['strength']), r['subgroup'], str(r['n_utterances'])]
        fields += [fmt(r[k]) for k in ('delta_accuracy','delta_weighted_f1','delta_macro_f1','flip_rate',
                                       'true_probability_change','confidence_change','js_divergence')]
        lines.append('| ' + ' | '.join(fields) + ' |')
    return lines


CLASS_NAMES = {'IEMOCAP':['happy','sad','neutral','angry','excited','frustrated'],
               'MELD':['neutral','surprise','fear','sadness','joy','disgust','anger']}


def interval_text(row, key):
    return f"{fmt(row[key])} [{fmt(row[key+'_ci_low'])}, {fmt(row[key+'_ci_high'])}]"


def contrast_table(rows):
    lines = ['| 数据集 | 目标 | 指标 | 正确组均值 | 错误组均值 | 错误−正确及95%CI |',
             '|---|---|---|---:|---:|---:|']
    for r in rows:
        lines.append('| ' + ' | '.join([r['dataset'],r['target'],r['metric'],fmt(r['correct_mean']),fmt(r['wrong_mean']),
                     f"{fmt(r['difference_wrong_minus_correct'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]"]) + ' |')
    return lines


def class_table(rows):
    lines = ['| 数据集 | 目标 | 真实类别 | n | 原始Acc | 干预Acc | ΔAcc及95%CI | ΔP(true)及95%CI |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        name = CLASS_NAMES[r['dataset']][r['class_index']]
        lines.append('| ' + ' | '.join([r['dataset'],r['target'],name,str(r['n_utterances']),fmt(r['original_accuracy']),
                     fmt(r['perturbed_accuracy']),interval_text(r,'delta_accuracy'),interval_text(r,'true_probability_change')]) + ' |')
    return lines


def unique_cases(cases):
    return {(r['dataset'],r['dialogue_id'],r['utterance_index']) for r in cases}


def substantive_answers(config, summaries, classes, missingness, cases, monotonicity, contrasts, previous):
    """Answer the ten requested questions from observed values without inference claims."""
    strong = [r for r in summaries if r['kind']=='attenuation' and r['strength']==1 and r['subgroup']=='all']
    whole = [r for r in strong if r['target'] in {'T-All','A','V'}]
    contrast = [r for r in contrasts if r['kind']=='attenuation' and r['strength']==1]
    lines = ['## 1. CMERC最依赖哪个模态？', '']
    for dataset in config['anchors']:
        rows = [r for r in whole if r['dataset']==dataset]
        drop = sorted(rows,key=lambda r:r['delta_accuracy'])
        flip = sorted(rows,key=lambda r:r['flip_rate'],reverse=True)
        lines.append(f"{dataset}: rho=1下ΔAcc最小为{drop[0]['target']}，{interval_text(drop[0],'delta_accuracy')}；翻转最多为{flip[0]['target']}，{interval_text(flip[0],'flip_rate')}。")
        for row in rows:
            lines.append(f"- {row['target']}: Acc {fmt(row['original_accuracy'])}→{fmt(row['perturbed_accuracy'])}；ΔP(true) {interval_text(row,'true_probability_change')}。")
    lines += ['', '这里只替换目标话语表征，保留其他话语同模态上下文、其他模态及完整对话前向；rho=1不是删除整段对话的某个模态。训练均值也未被证实情绪中性或位于数据流形内，排序仅为该干预定义下的依赖点估计。', '',
              '## 2. IEMOCAP与MELD是否有不同依赖结构？', '']
    rankings=[]
    for dataset in config['anchors']:
        rows=sorted([r for r in whole if r['dataset']==dataset],key=lambda r:r['delta_accuracy'])
        order=[r['target'] for r in rows];rankings.append(order)
        lines.append(f"- {dataset} 按rho=1准确率变化由小到大：{' → '.join(order)}。")
    lines += ['两个锚点的排序'+('不同' if len(rankings)>1 and rankings[0]!=rankings[1] else '相同')+'，但不同数据集/特征提取器/类别结构和单seed共同变化；没有对跨数据集排序差作确证检验。', '',
              '## 3. 文本四个存储数组哪个影响最大？', '']
    for dataset in config['anchors']:
        rows=sorted([r for r in strong if r['dataset']==dataset and r['target'] in {'T1','T2','T3','T4'}],key=lambda r:r['delta_accuracy'])
        lines.append(f"{dataset} rho=1单数组ΔAcc由小到大："+'；'.join(f"{r['target']} {interval_text(r,'delta_accuracy')}" for r in rows)+'。')
        by_flip=sorted(rows,key=lambda r:r['flip_rate'],reverse=True)
        by_probability=sorted(rows,key=lambda r:r['true_probability_change'])
        lines.append('按Flip由大到小：'+'；'.join(f"{r['target']} {fmt(r['flip_rate'])}" for r in by_flip)+
                     '；按ΔP(true)由小到大：'+'；'.join(f"{r['target']} {fmt(r['true_probability_change'])}" for r in by_probability)+'。')
    lines += ['“最重要”取决于准确率、翻转还是概率变化，不能给出不附指标的单一数组winner；很小的Acc点差也不证明数组间差异。原模型先分别归一化四个输入数组再求均值（冻结model.py:722–744）。单数组与T-All干预改变该归一化/融合输入，不能从点排序推断编码器层语义或因果重要性；数组对应具体encoder层号尚未证实。', '',
              '## 4. 音频与视觉逐渐削弱时预测如何变化？', '']
    for dataset in config['anchors']:
        for target in ['A','V']:
            rows=sorted([r for r in summaries if r['dataset']==dataset and r['target']==target and r['kind']=='attenuation' and r['subgroup']=='all'],key=lambda r:r['strength'])
            lines.append(f"- {dataset} {target} rho0/.25/.5/.75/1：Acc="+str([round(rows[0]['original_accuracy'],5)]+[round(r['perturbed_accuracy'],5) for r in rows])+
                         '；Flip='+str([0.]+[round(r['flip_rate'],5) for r in rows])+
                         '；ΔP(true)='+str([0.]+[round(r['true_probability_change'],5) for r in rows])+'。')
    reversals=[r for r in monotonicity if r['subgroup']=='all' and r['target'] in {'A','V'}]
    lines += [f"音视频all曲线中{sum(r['accuracy_increases']>0 for r in reversals)}/{len(reversals)}出现至少一次Acc相邻上升，未强制单调。强削弱可以使某些错误变正确，平均Acc变化不是唯一敏感性指标。", '',
              '## 5. 全零视觉对整体V实验有多大影响？', '']
    for dataset in config['anchors']:
        missing=next(r for r in missingness if r['dataset']==dataset and r['subgroup']=='visual_missing')
        rows={r['subgroup']:r for r in summaries if r['dataset']==dataset and r['target']=='V' and r['kind']=='attenuation' and r['strength']==1}
        lines.append(f"{dataset}: {missing['n_utterances']}/{missing['n_total']}={missing['n_utterances']/missing['n_total']:.2%}全零视觉；这组原始Acc={fmt(missing['original_accuracy'])}，原始错误{missing['n_utterances']-missing['original_correct_count']}条，在V干预下输入和概率均原样保留。rho1 V ΔAcc(all)={fmt(rows['all']['delta_accuracy'])}，ΔAcc(present)={fmt(rows['visual_present']['delta_accuracy'])}；Flip(all)={fmt(rows['all']['flip_rate'])}，Flip(present)={fmt(rows['visual_present']['flip_rate'])}。")
    lines += ['missing组无干预使all效应按present占比加权稀释；其稳定可能只反映零输入被保留，不能当作可靠。其他目标T/A仍对这些话语照常干预。', '',
              '## 6. 原始正确与错误预测的敏感性是否不同？', '',
              '下表直接计算同一原始对话bootstrap draw内错误组−正确组差值，而不是比较两个组CI是否重叠。随机mask的5次随机实现固定；区间仅覆盖对话抽样，不覆盖整个mask分布的不确定性。']
    lines += contrast_table([r for r in contrast if r['target'] in {'T-All','A','V'}])
    lines += ['', '差异为未经调整的关联，受原始置信度、类别构成及样本难度混杂；此处没有拟合评分或原始置信度模型，不能声称具有独立正确性信息。完全削弱可能合理损害正确预测，方向不预设为错误组更敏感。', '',
              '## 7. 哪些情绪类别更依赖非语言表征？', '',
              '按真实类别完整列出rho1音频/视觉变化及CI，避免只展示最大降幅。小n类别的区间可能很宽，类别差未经多重比较校正；不推断具体音调、音色或表情机制。']
    lines += class_table([r for r in classes if r['class_basis']=='true_class' and r['phase']=='overall_attenuation' and r['strength']==1 and r['target'] in {'A','V'}])
    lines += ['', '## 8. 是否有对某表征高度敏感的高置信错误？', '']
    for dataset in config['anchors']:
        selected=[r for r in cases if r['dataset']==dataset]
        lines.append(f"{dataset}: 按预注册定义（原始confidence≥.9且错误、rho1翻转或真实类概率下降≥.1），unique话语={len(unique_cases(selected))}，逐条件记录={len(selected)}。")
        seen=set()
        for row in sorted(selected,key=lambda r:r['true_probability_change']):
            identity=(row['dialogue_id'],row['utterance_index'])
            if identity in seen:continue
            seen.add(identity)
            lines.append(f"- {row['dialogue_id']} / u{row['utterance_index']}，{row['target']}：label={row['label']}，预测{row['original_prediction']}→{row['perturbed_prediction']}，confidence={fmt(row['original_confidence'])}，ΔP(true)={fmt(row['true_probability_change'])}。")
            if len(seen)==5:break
    lines += ['这些案例说明错误预测也可能依赖目标表征；翻转不保证纠错。未满足该阈值也不能认定没有表征依赖。', '',
              '## 9. 是否足以支持进一步研究模态依赖与校准的关系？', '']
    for row in previous:
        lines.append(f"- {row['dataset']} {row['target']}：前批三种轻扰全部零翻转{row['weak_stable_count']}/{row['n_utterances']}条，其中本批rho1翻转{row['strong_flip_among_weak_stable']}条、ΔP(true)≤−.1为{row['strong_trueprob_drop_ge_point1_among_weak_stable']}条；前批稳定且原始错误{row['weak_stable_wrong_count']}条，其中强削弱后翻转{row['strong_flip_among_weak_stable_wrong']}条。")
    lines += ['上述V all交叉计数包含全零视觉话语：本批missing按规则不干预，其零变化是执行策略而不是观测到视觉鲁棒。旧阶段Gaussian可能把全零视觉输入变成非零值，而旧Dropout/乘法衰减保持零，因此两个阶段的missing处理与干预语义不同。不能把旧V all的mild-stable直接类比为真实视觉依赖证据；本批视觉的实质响应应以visual_present分组为主、all为保留全部样本的辅助结果。',
              '轻度扰动稳定不等于不依赖该表征。当前证据支持将“依赖程度”和“无关干扰稳定性”分开研究；可提出进一步关联研究，但没有证明依赖可提升校准，更不能把移除证据后的变化直接并入可靠性评分。前批独立信息诊断也未确认稳定增益。', '',
              '## 10. 是否值得进入单模态扰动强度扩展？', '',
              '本批已在预注册范围覆盖rho=.25/.5/.75/1及mask=.1/.3/.5/.8；首先利用完整曲线与缺失/类别/正确性分组定位现象。rho1已替换为训练均值，不应继续外推rho>1。',
              '建议把已观察到较大目标表征依赖的模态作为后续独立预注册的研究候选，优先核验训练均值参考、表征尺度及不同seed的可重复性；视觉低响应须结合missing和present结果解释。是否增加其他单模态剂量/参照需要新协议，不能从本批结果自动进入联合模态、校准或正式测试。']
    return lines


def write_reports(root, config, summaries, classes, missingness, high_cases, monotonicity, contrasts, previous_comparisons):
    reports = root / 'reports'; reports.mkdir(parents=True, exist_ok=True)
    shared = ['仅使用两个锚点seed的真正验证集；未训练、未拟合可靠性评分/温度、未执行测试集或联合模态。',
              '干预对象是预提取表征。T1–T4为四个存储数组，尚不能认定为编码器的具体层；不声称移除了语义、音调、音色或面部表情。',
              'rho表示向训练集均值收缩的程度；rho=1表示替换目标话语相应特征为训练均值。随机掩盖按固定比例将维度替换为训练均值，不是零向量删除。',
              '只干预目标话语，其他话语同模态上下文与所有其他模态仍保留；rho1不是整段对话模态删除。训练均值未被证实情绪中性或位于流形内。',
              '所有指标为0–1比例，JS使用自然对数。Δ为干预后减原始。随机掩盖先逐repeat计算分类指标再平均，不对概率ensemble。',
              '95%区间来自2000次对话级配对bootstrap，同一数据集所有条件共用原对话池抽样。子组由固定原始属性选择，保留抽样中空子组为NA；macro-F1采用每个评估样本中真实/预测类别的并集。',
              '随机掩盖固定5次实现，bootstrap只重采样对话，不覆盖全体随机mask分布不确定性。正确/错误组差使用同draw直接差值CI，未控制原始置信度或类别混杂。',
              '全部条件与类别比较均为探索性，未做多重比较校正。敏感性表示模型依赖这些表征，不能推出原始预测错误或不可靠。']
    previous = Path(config['previous_pilot']) / 'results/modality_pilot_summary.csv'
    previous_rows = list(csv.DictReader(previous.open())) if previous.exists() else []
    for target_name, filename, targets in [('文本','TEXT_EVIDENCE_SENSITIVITY.md',{'T1','T2','T3','T4','T-All'}),
                                         ('音频','AUDIO_EVIDENCE_SENSITIVITY.md',{'A'}),
                                         ('视觉','VISUAL_EVIDENCE_SENSITIVITY.md',{'V'})]:
        selected = [r for r in summaries if r['target'] in targets and r['subgroup'] == 'all']
        lines = [f'# {target_name}表征证据敏感性', ''] + ['\n'.join(shared), '']
        lines += ['## 训练均值收缩（强度rho）', '']
        lines += condition_table([r for r in selected if r['kind']=='attenuation'], strength_label='rho')
        lines += ['', '## Random Feature Masking（掩盖比例ratio）', '',
                  '这是与训练均值收缩分开报告的操作：选择固定数量的维度替换为训练均值，5次固定随机实现逐次计算分类指标后平均。ratio不是rho，不能按同一剂量尺度解释。', '']
        lines += condition_table([r for r in selected if r['kind']=='random_masking'], strength_label='ratio')
        lines += ['', 'rho1原始正确/错误组的直接差值诊断：', '']
        lines += contrast_table([r for r in contrasts if r['target'] in targets and r['kind']=='attenuation' and r['strength']==1])
        if target_name=='文本':
            lines += ['', '四个数组在原模型中先归一化再求均值（model.py:722–744），这里不能将T1–T4点排序解释为具体encoder层的语义因果重要性。']
            for dataset in config['anchors']:
                ranked=sorted([r for r in summaries if r['dataset']==dataset and r['target'] in {'T1','T2','T3','T4'} and r['kind']=='attenuation' and r['strength']==1 and r['subgroup']=='all'],key=lambda r:r['delta_accuracy'])
                lines.append(f"{dataset} rho1 ΔAcc由小到大："+'；'.join(f"{r['target']} {interval_text(r,'delta_accuracy')}" for r in ranked)+'。')
                lines.append('Flip由大到小：'+'；'.join(f"{r['target']} {fmt(r['flip_rate'])}" for r in sorted(ranked,key=lambda r:r['flip_rate'],reverse=True))+
                             '；ΔP(true)由小到大：'+'；'.join(f"{r['target']} {fmt(r['true_probability_change'])}" for r in sorted(ranked,key=lambda r:r['true_probability_change']))+'。')
            lines.append('单数组“重要性”排序依赖指标；小的Acc点差及单独CI不能证明数组间差异，不能选一个不附指标的winner。')
        else:
            lines += ['', '真实类别完整rho1结果（探索性）：', '']
            lines += class_table([r for r in classes if r['target'] in targets and r['class_basis']=='true_class' and r['kind']=='attenuation' and r['strength']==1])
        if target_name == '视觉':
            lines += ['', '全零视觉话语不做V干预，输入及保存概率精确保留；all表包含这些样本，主解释还需查看visual_present。缺失与存在是观察性分组，不是随机干预对照。', '']
            lines += ['### visual_present：训练均值收缩（rho）', '']
            lines += condition_table([r for r in summaries if r['target'] == 'V' and r['subgroup'] == 'visual_present' and r['kind']=='attenuation'], strength_label='rho')
            lines += ['', '### visual_present：Random Feature Masking（ratio）', '']
            lines += condition_table([r for r in summaries if r['target'] == 'V' and r['subgroup'] == 'visual_present' and r['kind']=='random_masking'], strength_label='ratio')
            for row in missingness:
                lines.append(f"- {row['dataset']} {row['subgroup']}: n={row['n_utterances']}, 原始准确率={fmt(row['original_accuracy'])}。")
        family = {'文本':'text_representation','音频':'audio','视觉':'visual'}[target_name]
        lines += ['', '前批轻度行为扰动仅作横向背景；其零均值噪声/乘0.95操作与本批训练均值收缩不可当作同一剂量尺度：']
        for r in previous_rows:
            if r['modality'] == family:
                lines.append(f"- {r['dataset']} {r['perturbation_type']}: Flip={fmt(float(r['flip_rate']))}, Δconfidence={fmt(float(r['confidence_change']))}。")
        if target_name=='视觉':
            lines.append('旧V all包括missing输入：Gaussian可能给零向量制造非零值，而本批missing全部跳过且概率保留。两阶段V all不能直接当作同一种物理/视觉依赖比较；以本批visual_present为主解释。')
        lines += ['', '完整置信区间、正确/错误子组、真实类别及原始预测类别分组分别见sensitivity_summary.csv和classwise_sensitivity.csv。不根据敏感性重新定义可靠性。']
        (reports / filename).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    strongest = [r for r in summaries if r['phase'] == 'overall_attenuation' and r['strength'] == 1 and r['subgroup'] == 'all']
    lines = ['# 验证集表征证据敏感性总结', '', '\n'.join(shared), '']
    lines += substantive_answers(config,summaries,classes,missingness,high_cases,monotonicity,contrasts,previous_comparisons)
    lines += ['', '## Random Feature Masking结果（独立于收缩）', '',
              '固定展示最高预注册掩盖比例ratio=.8的全部7个目标，不根据结果选择最佳条件。每条记录来自已有summary，5次随机实现逐repeat计算指标后平均，区间仅覆盖对话抽样。ratio=.1/.3/.5/.8完整结果分别见三份模态报告的Random Feature Masking小节及modality_sensitivity_summary.csv；与rho收缩操作分开解释。', '']
    mask_rows=[r for r in summaries if r['kind']=='random_masking' and r['strength']==.8 and r['subgroup']=='all']
    lines += condition_table(mask_rows, strength_label='ratio')
    lines += ['', '上述固定ratio=.8条件的主要配对95%区间：', '',
              '| 数据集 | 目标 | ΔAcc及95%CI | Flip及95%CI | ΔP(true)及95%CI |',
              '|---|---|---:|---:|---:|']
    for row in mask_rows:
        lines.append('| '+' | '.join([row['dataset'],row['target'],interval_text(row,'delta_accuracy'),interval_text(row,'flip_rate'),interval_text(row,'true_probability_change')])+' |')
    lines += ['', '## 完全收缩至训练均值的总表（rho=1）', ''] + condition_table(strongest, strength_label='rho')
    lines += ['', '## 逐数据集描述', '']
    for dataset in config['anchors']:
        values = [r for r in strongest if r['dataset'] == dataset]
        by_drop = sorted(values, key=lambda r: r['delta_accuracy'])
        by_flip = sorted(values, key=lambda r: r['flip_rate'], reverse=True)
        lines.append(f"{dataset}: rho=1下ΔAcc最小为{by_drop[0]['target']}（ΔAcc={fmt(by_drop[0]['delta_accuracy'])}），翻转最多为{by_flip[0]['target']}（Flip={fmt(by_flip[0]['flip_rate'])}）。这是该验证集点估计排序，不是总体重要性或显著性排序。")
        for row in values:
            lines.append(f"- {row['target']}: ΔAcc 95%CI [{fmt(row['delta_accuracy_ci_low'])}, {fmt(row['delta_accuracy_ci_high'])}]；ΔP(true) 95%CI [{fmt(row['true_probability_change_ci_low'])}, {fmt(row['true_probability_change_ci_high'])}]。")
    lines += ['', '## 非单调性与失败案例', '',
              f"所有衰减曲线中出现准确率向上相邻变化的曲线数：{sum(r['accuracy_increases'] > 0 for r in monotonicity)}/{len(monotonicity)}（含子组，不独立）。完整相邻变化见monotonicity_analysis.csv。",
              f"原始置信度≥0.9且错误，在rho=1出现翻转或真实类概率下降≥0.1的逐条件案例数：{len(high_cases)}。同一话语可出现在多个条件；这不是唯一话语数量或总体失败率。", '',
              '视觉缺失保留在所有样本结果中；全零输入不做视觉干预，不能将无变化解释为视觉稳定性或可靠性证据。', '',
              '本批回答表征依赖程度；前批回答轻度特征扰动反应。两类实验均不能直接证明情绪证据的因果作用，不能把敏感性混入Reliability Score，也不能据此宣称校准改善。', '',
              '后续应先区分表征失效、缺失、尺度和已存编码层的影响，再另行预注册可靠性或联合模态实验。本脚本不启动任何后续实验。']
    (reports / 'FINAL_EVIDENCE_SENSITIVITY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve()), indent=2))


if __name__ == '__main__':
    main()
