#!/usr/bin/env python3
"""Frozen-cohort MELD matched analysis. CPU CSV only; no model/score fitting.

The cohort is never rematched. Bootstrap units are the same 80 frozen Dialogues
for every family, scope, metric and fixed confidence-gap sensitivity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FAMILIES = {
    'neutral_append': ('neutral_a', 'neutral_b'),
    'semantic_paraphrase': ('paraphrase_a', 'paraphrase_b'),
}
METRICS = ('mu', 'delta', 'js_mean', 'flip_rate')
SCOPES = ('AUTOMATIC_INPUT_QC_DIAGNOSTIC',
          'AI_REVIEWED_EXPLORATORY', 'HUMAN_CONFIRMED')
CALIPERS = (('all', None), ('gap_le_0.02', .02), ('gap_le_0.01', .01))
SEED = 20261120
REPETITIONS = 2000
NEAR_ZERO = 1e-6


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: '' if row.get(key) is None else row.get(key, '') for key in fields})


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def truth(value):
    return value is True or str(value).lower() in ('true', '1')


def probability(value):
    data = np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=np.float64)
    if data.shape != (7,) or not np.isfinite(data).all() or (data < 0).any():
        raise ValueError('MELD probability must contain seven finite nonnegative values')
    if not np.isclose(data.sum(), 1., atol=1e-5, rtol=0):
        raise ValueError('Probability sum is invalid')
    return data


def js_divergence(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    middle = .5 * (p + q)
    a, b = p > 0, q > 0
    return float(.5 * ((p[a] * np.log(p[a] / middle[a])).sum()
                       + (q[b] * np.log(q[b] / middle[b])).sum()))


def behavior(p, variants):
    p = probability(p)
    q = np.stack([probability(item) for item in variants])
    if len(q) != 2:
        raise ValueError('Frozen family requires exactly K=2')
    confidence = q.max(axis=1)
    flip = q.argmax(axis=1) != p.argmax()
    return dict(mu=float(np.mean(flip * confidence)),
                delta=float(abs(p.max() - confidence.mean())),
                js_mean=float(np.mean([js_divergence(p, item) for item in q])),
                flip_rate=float(np.mean(flip)),
                mean_confidence_change=float(confidence.mean() - p.max()),
                mean_abs_confidence_change=float(np.abs(confidence - p.max()).mean()),
                max_abs_confidence_change=float(np.abs(confidence - p.max()).max()),
                mean_original_class_probability_change=float((q[:, p.argmax()] - p.max()).mean()))


def concordance(difference):
    """Higher behavior means error; exact floating-point ties receive 0.5."""
    return float(difference > 0) + .5 * float(difference == 0)


def gap_group(signed_gap):
    if abs(signed_gap) <= NEAR_ZERO:
        return 'near_zero'
    return 'error_confidence_higher' if signed_gap > 0 else 'error_confidence_lower'


def validate_cohort(pairs, samples):
    by_id = {row['sample_id']: row for row in samples}
    if len(by_id) != len(samples):
        raise ValueError('Duplicate sample ID')
    used = set()
    pair_ids = set()
    for pair in pairs:
        if pair['pair_id'] in pair_ids:
            raise ValueError('Duplicate pair ID')
        pair_ids.add(pair['pair_id'])
        correct = by_id[pair['correct_sample_id']]
        error = by_id[pair['error_sample_id']]
        for row, expected in ((correct, 1), (error, 0)):
            if row['sample_id'] in used:
                raise ValueError('Cohort reuses a sample')
            used.add(row['sample_id'])
            p = probability(row['original_probabilities'])
            if int(row['original_correct']) != expected or int(p.argmax() == int(row['true_label'])) != expected:
                raise ValueError('Pair correctness or label conflicts with original probability')
            if row['dialogue_id'] != pair['dialogue_id'] or row['confidence_bin'] != pair['confidence_bin']:
                raise ValueError('Pair crosses frozen Dialogue/confidence cell')
            if abs(float(p.max()) - float(row['original_confidence'])) > 1e-7:
                raise ValueError('Stored original confidence conflicts with probability')
            index = min(9, int(float(p.max()) * 10))
            if row['confidence_bin'] != f'{index/10:.1f}-{(index+1)/10:.1f}':
                raise ValueError('Confidence-bin boundary mismatch')
            if row.get('pair_id', pair['pair_id']) != pair['pair_id']:
                raise ValueError('Sample references a different frozen pair')
        signed = float(error['original_confidence']) - float(correct['original_confidence'])
        for key, value in [('confidence_signed_gap', signed), ('confidence_abs_gap', abs(signed)),
                           ('correct_confidence', float(correct['original_confidence'])),
                           ('error_confidence', float(error['original_confidence']))]:
            if abs(float(pair[key]) - value) > 1e-12:
                raise ValueError('Frozen pair confidence field mismatch')
    if used != set(by_id):
        raise ValueError('Manifest sample set does not equal disjoint pair endpoints')
    return by_id


def aggregate_targets(predictions, samples):
    """Retain all rows; complete K, automatic QC, AI and human are separate gates.

    Historical reused versions retain their original AI records; the AI scope
    does not imply every version received two reviews in this follow-up run."""
    by_id = {row['sample_id']: row for row in samples}
    grouped = defaultdict(dict)
    for row in predictions:
        sid, family, variant = row['sample_id'], row['perturbation_type'], row['variant_id']
        if sid not in by_id or family not in FAMILIES or variant not in FAMILIES[family]:
            raise ValueError('Prediction references a non-cohort sample/family/variant')
        if variant in grouped[(sid, family)]:
            raise ValueError('Duplicate prediction condition')
        original = by_id[sid]
        for key in ('dialogue_id', 'utterance_index', 'original_correct'):
            if str(row[key]) != str(original[key]):
                raise ValueError('Prediction source metadata differs from frozen sample')
        if 'true_label' in row and str(row['true_label']) != str(original['true_label']):
            raise ValueError('Prediction label differs from frozen sample')
        if row.get('modality', 'text') != 'text':
            raise ValueError('Matched experiment only permits text modality')
        p = probability(row['original_probabilities'])
        if not np.array_equal(p, probability(original['original_probabilities'])):
            raise ValueError('Original probability changed across conditions')
        if abs(float(row['original_confidence']) - float(original['original_confidence'])) > 1e-12:
            raise ValueError('Original confidence changed across conditions')
        if row['status'] not in ('SUCCESS', 'FAILED'):
            raise ValueError('Unknown prediction status')
        if row['status'] == 'SUCCESS':
            probability(row['perturbed_probabilities'])
        grouped[(sid, family)][variant] = row
    result = []
    for sample in samples:
        for family, variants in FAMILIES.items():
            available = grouped[(sample['sample_id'], family)]
            ordered = [available[v] for v in variants if v in available]
            complete = len(ordered) == 2 and all(row['status'] == 'SUCCESS' for row in ordered)
            automatic = complete and all(truth(row.get('automatic_qc_pass', False)) for row in ordered)
            ai = automatic and all(
                truth(row.get('label_preserving_main_eligible', False))
                and row.get('label_review_status', '').upper() == 'PASS'
                and 'AI' in row.get('review_method', '').upper()
                and bool(row.get('reviewer', '').strip()) for row in ordered)
            # No field or blank field is PENDING. Never inherit human PASS from AI.
            human = automatic and all(
                row.get('human_reviewer_1_status', '').upper() == 'PASS'
                and row.get('human_reviewer_2_status', '').upper() == 'PASS'
                and bool(row.get('human_reviewer_1_id', '').strip())
                and bool(row.get('human_reviewer_2_id', '').strip())
                and row.get('human_reviewer_1_id') != row.get('human_reviewer_2_id')
                for row in ordered)
            record = dict(sample_id=sample['sample_id'], pair_id=sample.get('pair_id', ''),
                          dialogue_id=sample['dialogue_id'], utterance_index=sample['utterance_index'],
                          original_correct=int(sample['original_correct']),
                          original_confidence=float(sample['original_confidence']),
                          confidence_bin=sample['confidence_bin'], perturbation_type=family,
                          planned_k=2, recorded_k=len(ordered), successful_k=sum(row['status'] == 'SUCCESS' for row in ordered),
                          complete_planned_family=complete, automatic_qc_eligible=automatic,
                          ai_reviewed_exploratory_eligible=ai, human_confirmed_eligible=human,
                          label_review_statuses='|'.join(sorted({row.get('label_review_status', 'PENDING') for row in ordered})),
                          missing_variants='|'.join(v for v in variants if v not in available),
                          failed_variants='|'.join(row['variant_id'] for row in ordered if row['status'] != 'SUCCESS'))
            record.update({metric: None for metric in METRICS})
            if complete:
                record.update(behavior(sample['original_probabilities'], [row['perturbed_probabilities'] for row in ordered]))
            result.append(record)
    return result


def build_pair_records(pairs, target_rows):
    targets = {(row['sample_id'], row['perturbation_type']): row for row in target_rows}
    eligibility = dict(zip(SCOPES, ('automatic_qc_eligible', 'ai_reviewed_exploratory_eligible', 'human_confirmed_eligible')))
    records = []
    for pair in pairs:
        for family in FAMILIES:
            c = targets[(pair['correct_sample_id'], family)]
            e = targets[(pair['error_sample_id'], family)]
            for scope, gate in eligibility.items():
                eligible = bool(c[gate] and e[gate])
                reasons = []
                for name, target in [('correct', c), ('error', e)]:
                    if not target['complete_planned_family']:
                        reasons.append(name + ':INCOMPLETE_FIXED_K')
                    elif not target['automatic_qc_eligible']:
                        reasons.append(name + ':AUTOMATIC_QC_FAILED')
                    elif not target[gate]:
                        reasons.append(name + ':' + ('HUMAN_NOT_CONFIRMED' if scope == SCOPES[2] else 'AI_NOT_PASS'))
                signed = float(pair['confidence_signed_gap'])
                row = dict(pair_id=pair['pair_id'], dialogue_id=pair['dialogue_id'],
                           confidence_bin=pair['confidence_bin'], correct_sample_id=pair['correct_sample_id'],
                           error_sample_id=pair['error_sample_id'], perturbation_type=family, analysis_scope=scope,
                           eligible=eligible, exclusion_reason='|'.join(reasons),
                           correct_confidence=float(pair['correct_confidence']), error_confidence=float(pair['error_confidence']),
                           confidence_signed_gap=signed, confidence_abs_gap=abs(signed),
                           confidence_gap_group=gap_group(signed),
                           confidence_baseline_concordance=concordance(-signed))
                for metric in METRICS:
                    row[metric + '_correct'] = c[metric]
                    row[metric + '_error'] = e[metric]
                    difference = e[metric] - c[metric] if eligible else None
                    row[metric + '_difference'] = difference
                    rank = concordance(difference) if difference is not None else None
                    row[metric + '_concordance'] = rank
                    row[metric + '_concordance_minus_confidence'] = rank - row['confidence_baseline_concordance'] if rank is not None else None
                records.append(row)
    return records


def bootstrap_draws(dialogues, repetitions=REPETITIONS, seed=SEED):
    return np.random.RandomState(seed).randint(0, len(dialogues), (repetitions, len(dialogues)))


def summarize_values(rows, field, dialogues, draws):
    """Return equal-Dialogue and equal-pair estimates with the SAME cluster draws."""
    positions = {did: i for i, did in enumerate(dialogues)}
    sums = np.zeros(len(dialogues), dtype=np.float64)
    counts = np.zeros(len(dialogues), dtype=np.int64)
    for row in rows:
        i = positions[row['dialogue_id']]
        sums[i] += float(row[field])
        counts[i] += 1
    supported = counts > 0
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=supported)
    d_den = supported[draws].sum(axis=1)
    p_den = counts[draws].sum(axis=1)
    d = np.divide(means[draws].sum(axis=1), d_den,
                  out=np.full(len(draws), np.nan), where=d_den > 0)
    p = np.divide(sums[draws].sum(axis=1), p_den,
                  out=np.full(len(draws), np.nan), where=p_den > 0)
    return {'equal_dialogue': (float(means[supported].mean()) if supported.any() else None, d),
            'equal_pair': (float(sums.sum() / counts.sum()) if counts.sum() else None, p)}


def interval(values, level=.95):
    good = np.asarray(values, dtype=np.float64)
    good = good[np.isfinite(good)]
    if not len(good):
        return None, None
    alpha = (1. - level) / 2
    low, high = np.quantile(good, [alpha, 1 - alpha])
    return float(low), float(high)


def support_summary(rows, candidate_pairs, all_pairs, dialogues):
    signed = np.array([float(row['confidence_signed_gap']) for row in rows], dtype=np.float64)
    absolute = np.abs(signed)
    gaps_c = np.array([row['correct_confidence'] for row in rows], dtype=np.float64)
    gaps_e = np.array([row['error_confidence'] for row in rows], dtype=np.float64)
    counts = Counter(row['dialogue_id'] for row in rows)
    return dict(candidate_pairs=len(candidate_pairs), eligible_pairs=len(rows), eligible_targets=2 * len(rows),
                eligible_dialogues=len(counts), frozen_pairs=len(all_pairs), frozen_dialogues=len(dialogues),
                candidate_pair_coverage=len(rows) / len(candidate_pairs) if candidate_pairs else 0.,
                frozen_pair_coverage=len(rows) / len(all_pairs) if all_pairs else 0.,
                validation_target_coverage=2 * len(rows) / 1109,
                dialogue_coverage=len(counts) / len(dialogues),
                maximum_pairs_per_dialogue=max(counts.values(), default=0),
                confidence_signed_gap_mean=float(signed.mean()) if len(signed) else None,
                confidence_abs_gap_mean=float(absolute.mean()) if len(absolute) else None,
                confidence_abs_gap_median=float(np.median(absolute)) if len(absolute) else None,
                confidence_abs_gap_q90=float(np.quantile(absolute, .9)) if len(absolute) else None,
                confidence_abs_gap_max=float(absolute.max()) if len(absolute) else None,
                correct_confidence_mean=float(gaps_c.mean()) if len(gaps_c) else None,
                error_confidence_mean=float(gaps_e.mean()) if len(gaps_e) else None,
                residual_error_higher_pairs=int((signed > NEAR_ZERO).sum()),
                residual_error_lower_pairs=int((signed < -NEAR_ZERO).sum()),
                residual_near_zero_pairs=int((np.abs(signed) <= NEAR_ZERO).sum()),
                descriptive_support_floor_pass=len(rows) >= 30 and len(counts) >= 20,
                support_floor_is_not_power_proof=True)


def analyze_pairs(pair_rows, pairs, dialogues, draws):
    supports, contrasts, ranks, replicates = [], [], [], {}
    condition_num = 0
    for family in FAMILIES:
        for scope in SCOPES:
            scope_rows = [row for row in pair_rows if row['perturbation_type'] == family and row['analysis_scope'] == scope]
            for caliper_name, limit in CALIPERS:
                candidates = [pair for pair in pairs if limit is None or float(pair['confidence_abs_gap']) <= limit]
                candidate_ids = {pair['pair_id'] for pair in candidates}
                # Residual direction strata are only the unrestricted sensitivity;
                # fixed .02/.01 calipers retain the overall contrast.
                gap_groups = ('all', 'error_confidence_higher', 'error_confidence_lower', 'near_zero') if limit is None else ('all',)
                for group in gap_groups:
                    candidate_group = [pair for pair in candidates if group == 'all' or gap_group(float(pair['confidence_signed_gap'])) == group]
                    selected = [row for row in scope_rows if row['eligible'] and row['pair_id'] in candidate_ids
                                and (group == 'all' or row['confidence_gap_group'] == group)]
                    condition_num += 1
                    condition = dict(condition_id=f'C{condition_num:03d}', perturbation_type=family,
                                     analysis_scope=scope, confidence_caliper=caliper_name,
                                     residual_confidence_group=group)
                    support = support_summary(selected, candidate_group, pairs, dialogues)
                    supports.append(dict(**condition, **support))
                    if not selected:
                        status = 'NO_HUMAN_CONFIRMED_PAIRS' if scope == SCOPES[2] else 'NO_ELIGIBLE_PAIRS'
                    elif not support['descriptive_support_floor_pass']:
                        status = 'LOW_DESCRIPTIVE_SUPPORT'
                    else:
                        status = 'OK_EXPLORATORY'
                    for metric in METRICS:
                        for estimand, (estimate, samples) in summarize_values(selected, metric + '_difference', dialogues, draws).items():
                            key = condition['condition_id'] + '__' + metric + '__' + estimand + '__difference'
                            replicates[key] = samples
                            valid = int(np.isfinite(samples).sum())
                            low, high = interval(samples)
                            priority = metric == 'js_mean' and scope == SCOPES[1] and caliper_name == 'all' and group == 'all' and estimand == 'equal_dialogue'
                            adjusted = interval(samples, .975) if priority else (None, None)
                            contrasts.append(dict(**condition, metric=metric, estimand=estimand,
                                                  error_minus_correct=estimate, ci95_low=low, ci95_high=high,
                                                  priority_two_family_js=priority,
                                                  bonferroni_ci97_5_low=adjusted[0], bonferroni_ci97_5_high=adjusted[1],
                                                  eligible_pairs=len(selected), eligible_dialogues=support['eligible_dialogues'],
                                                  status=status, bootstrap_requested=len(draws), bootstrap_valid=valid,
                                                  low_valid_bootstrap_warning=bool(selected) and valid < 1900,
                                                  replicate_key=key))
                        # Ranking is prespecified equal-Dialogue, conditional on the
                        # SAME matched pairs; it is not a population ROC-AUC or gain.
                        for ranking_name, field in [('behavior', metric + '_concordance'),
                                                    ('confidence_only', 'confidence_baseline_concordance'),
                                                    ('behavior_minus_confidence', metric + '_concordance_minus_confidence')]:
                            estimate, samples = summarize_values(selected, field, dialogues, draws)['equal_dialogue']
                            key = condition['condition_id'] + '__' + metric + '__rank_' + ranking_name
                            replicates[key] = samples
                            low, high = interval(samples)
                            valid = int(np.isfinite(samples).sum())
                            ranks.append(dict(**condition, metric=metric, ranking=ranking_name,
                                              estimand='equal_dialogue_conditional_pair_concordance',
                                              estimate=estimate, ci95_low=low, ci95_high=high,
                                              eligible_pairs=len(selected), eligible_dialogues=support['eligible_dialogues'],
                                              status=status, bootstrap_requested=len(draws), bootstrap_valid=valid,
                                              low_valid_bootstrap_warning=bool(selected) and valid < 1900,
                                              replicate_key=key, population_roc_auc=False,
                                              fitted_incremental_prediction_gain=False))
    return supports, contrasts, ranks, replicates


def validate_locks(run_dir):
    protocol_path = run_dir / 'configs/EXPLORATORY_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text())
    lock = json.loads((run_dir / 'configs/COHORT_LOCK.json').read_text())
    if lock['sources_sha256'][str(protocol_path)] != sha256(protocol_path):
        raise ValueError('Frozen protocol SHA changed')
    for name in ('matched_pair_manifest.csv', 'matched_sample_manifest.csv', 'all_validation_matching_status.csv'):
        path = run_dir / 'results' / name
        if lock['outputs_sha256'][str(path)] != sha256(path):
            raise ValueError('Frozen cohort manifest SHA changed')
    stats = protocol['statistics']
    if stats['bootstrap_repetitions'] != REPETITIONS or stats['bootstrap_seed'] != SEED:
        raise ValueError('Frozen bootstrap parameters do not match implementation')
    if protocol['matching']['predeclared_caliper_sensitivities'] != [.02, .01]:
        raise ValueError('Frozen calipers changed')
    return protocol, lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    predictions_path = args.predictions or run_dir / 'results/perturbed_predictions.csv'
    output = args.output or run_dir / 'results/matched_analysis'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Analysis output must be empty; historical results cannot be overwritten')
    protocol, lock = validate_locks(run_dir)
    pairs = read_csv(run_dir / 'results/matched_pair_manifest.csv')
    samples = read_csv(run_dir / 'results/matched_sample_manifest.csv')
    validate_cohort(pairs, samples)
    dialogues = sorted({pair['dialogue_id'] for pair in pairs})
    if (len(pairs), len(samples), len(dialogues)) != (203, 406, 80):
        raise ValueError('Actual locked cohort must have 203 pairs/406 targets/80 Dialogues')
    predictions = read_csv(predictions_path)
    targets = aggregate_targets(predictions, samples)
    # Full experiment accounting is mandatory even when some conditions failed.
    if len(predictions) != 1624 or any(row['recorded_k'] != 2 for row in targets):
        raise ValueError('Expected exactly all 1624 fixed conditions, including failures')
    pair_rows = build_pair_records(pairs, targets)
    draws = bootstrap_draws(dialogues)
    support, contrasts, ranks, replicates = analyze_pairs(pair_rows, pairs, dialogues, draws)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'target_behavior.csv', targets)
    write_csv(output / 'pair_behavior_and_eligibility.csv', pair_rows)
    write_csv(output / 'condition_support.csv', support)
    write_csv(output / 'paired_metric_contrasts.csv', contrasts)
    write_csv(output / 'conditional_pair_ranking.csv', ranks)
    # These contain only resampling/statistic values, never embeddings or text.
    np.savez_compressed(output / 'bootstrap_draw_indices.npz',
                        indices=draws.astype(np.int16), dialogue_ids=np.asarray(dialogues, dtype='U64'))
    np.savez_compressed(output / 'bootstrap_replicates.npz', **replicates)
    failures = [row for row in predictions if row['status'] != 'SUCCESS'
                or not truth(row.get('automatic_qc_pass')) or row.get('label_review_status', '').upper() != 'PASS']
    # The source CSV already excludes raw text; copy only safe audit fields here.
    audit_fields = ['sample_id', 'dialogue_id', 'perturbation_type', 'variant_id', 'status',
                    'automatic_qc_pass', 'label_review_status', 'reviewer', 'review_method',
                    'label_preserving_main_eligible', 'error']
    write_csv(output / 'failed_uncertain_review_records.csv', failures, audit_fields)
    verification = dict(status='PASS_ANALYSIS_ACCOUNTING_NOT_SCIENTIFIC_CONFIRMATION',
                        created_at=datetime.now(timezone.utc).isoformat(),
                        protocol_sha256=sha256(run_dir / 'configs/EXPLORATORY_PROTOCOL.json'),
                        cohort_lock_sha256=sha256(run_dir / 'configs/COHORT_LOCK.json'),
                        predictions_sha256=sha256(predictions_path), script_sha256=sha256(__file__),
                        python=platform.python_version(), numpy=np.__version__,
                        matched_pairs=len(pairs), target_samples=len(samples),
                        target_families=len(targets), prediction_conditions=len(predictions),
                        prediction_status_counts=dict(Counter(row['status'] for row in predictions)),
                        label_review_counts=dict(Counter(row.get('label_review_status', 'PENDING') for row in predictions)),
                        preserved_failed_or_nonpass_records=len(failures),
                        pair_family_scope_rows=len(pair_rows), condition_rows=len(support),
                        contrast_rows=len(contrasts), ranking_rows=len(ranks),
                        human_confirmed_target_families=sum(row['human_confirmed_eligible'] for row in targets),
                        human_confirmation_must_not_be_inferred_from_ai=True,
                        bootstrap_seed=SEED, bootstrap_repetitions=REPETITIONS,
                        bootstrap_dialogues=dialogues,
                        shared_draws_shape=list(draws.shape),
                        shared_draws_raw_sha256=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest(),
                        shared_bootstrap_identity='One stored master index array reused verbatim for every family/metric/scope/caliper/estimand/ranking; no rematching',
                        replicate_arrays=len(replicates),
                        negative_and_null_results_retained=True,
                        validation_reused_exploratory=True, independent_confirmation=False,
                        population_accuracy_or_auc_estimated=False, calibration_module_fitted=False,
                        incremental_prediction_gain_established=False,
                        outputs_sha256={path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()})
    save_json(output / 'analysis_verification.json', verification)
    print(json.dumps({key: verification[key] for key in ('status', 'matched_pairs', 'target_samples',
                       'prediction_conditions', 'prediction_status_counts', 'human_confirmed_target_families',
                       'bootstrap_repetitions', 'condition_rows')}, indent=2))


if __name__ == '__main__':
    main()
