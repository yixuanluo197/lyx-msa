#!/usr/bin/env python3
"""Formal mainline mu/delta and Dialogue-cluster paired bootstrap (no fitted score)."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from raw_pilot import TYPES, confidence_bin, read_csv, save_json, write_csv

METRICS = ['mu', 'delta', 'flip_rate', 'js_mean', 'mean_confidence_change',
           'mean_abs_confidence_change', 'max_abs_confidence_change',
           'mean_original_class_probability_change']


def is_true(value):
    return value is True or str(value).lower() in ['true', '1']


def probability(value):
    p = np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-5):
        raise ValueError('Invalid full probability vector')
    return p


def js_divergence(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    middle = (p + q) / 2
    a, b = p > 0, q > 0
    return float(.5 * (np.sum(p[a] * np.log(p[a] / middle[a])) +
                       np.sum(q[b] * np.log(q[b] / middle[b]))))


def behavior_features(original, perturbed):
    """mainline_mu_delta/20261115T165025Z/scripts/build_inputs.py response().

    mu=mean(flip_k*max(p'_k)); delta=abs(max(p)-mean(max(p'_k))).
    delta is NOT mean absolute change and mu is NOT the unweighted flip rate.
    """
    p = probability(original)
    q = np.stack([probability(row) for row in perturbed])
    if q.shape[1] != len(p): raise ValueError('Class order/dimension mismatch')
    prediction = int(p.argmax()); confidence = float(p.max())
    changed_conf = q.max(axis=1); flip = q.argmax(axis=1) != prediction
    return dict(mu=float(np.mean(flip * changed_conf)),
                delta=float(abs(confidence - changed_conf.mean())),
                flip_rate=float(flip.mean()), original_class_retention=float(1 - flip.mean()),
                js_mean=float(np.mean([js_divergence(p, row) for row in q])),
                js_max=float(max(js_divergence(p, row) for row in q)),
                mean_confidence_change=float(np.mean(changed_conf - confidence)),
                mean_abs_confidence_change=float(np.mean(np.abs(changed_conf - confidence))),
                max_abs_confidence_change=float(np.max(np.abs(changed_conf - confidence))),
                mean_original_class_probability_change=float(np.mean(q[:, prediction] - p[prediction])))


def per_variant(row):
    p, q = probability(row['original_probabilities']), probability(row['perturbed_probabilities'])
    label = int(row['true_label']); original = int(p.argmax()); changed = int(q.argmax())
    eligible = is_true(row.get('label_preserving_main_eligible', False))
    return dict(dataset=row['dataset'], sample_id=row['sample_id'], dialogue_id=row['dialogue_id'],
                utterance_index=int(row['utterance_index']), modality=row['modality'],
                perturbation_type=row['perturbation_type'], variant_id=row['variant_id'],
                original_correct=int(original == label), original_confidence=float(p.max()),
                confidence_bin=confidence_bin(float(p.max())), flip=int(original != changed),
                correct_to_wrong=int(original == label and changed != label) if eligible else None,
                wrong_to_correct=int(original != label and changed == label) if eligible else None,
                confidence_change=float(q.max() - p.max()), abs_confidence_change=float(abs(q.max() - p.max())),
                original_class_probability_change=float(q[original] - p[original]),
                original_label_probability_change=float(q[label] - p[label]), js=js_divergence(p, q),
                label_preserving_main_eligible=eligible,
                label_confirmed_for_perturbed_accuracy=eligible,
                automatic_qc_pass=is_true(row.get('automatic_qc_pass', False)),
                label_review_status=row.get('label_review_status', 'PENDING'))


def aggregate_samples(rows):
    groups = defaultdict(list)
    for row in rows: groups[(row['dataset'], row['sample_id'], row['perturbation_type'])].append(row)
    result, missing = [], []
    for (dataset, sid, kind), variants in groups.items():
        expected = set(TYPES[kind]); observed = {r['variant_id'] for r in variants}
        success = [r for r in variants if r['status'] == 'SUCCESS']
        complete = observed == expected and len(variants) == len(expected) and len(success) == len(expected)
        base = variants[0]
        information = dict(dataset=dataset, sample_id=sid, dialogue_id=base['dialogue_id'],
                           utterance_index=int(base['utterance_index']), modality=base['modality'],
                           perturbation_type=kind, planned_k=len(expected), successful_k=len(success),
                           original_correct=int(base['original_correct']), true_label=int(base['true_label']),
                           original_confidence=float(base['original_confidence']), confidence_bin=base['confidence_bin'],
                           complete_planned_family=complete,
                           all_variants_automatic_qc_pass=complete and all(is_true(r.get('automatic_qc_pass')) for r in success),
                           label_review_statuses='|'.join(sorted({r.get('label_review_status', 'PENDING').upper() for r in variants})),
                           all_variants_label_preserving=complete and all(is_true(r.get('label_preserving_main_eligible')) and is_true(r.get('automatic_qc_pass')) for r in success))
        if not complete:
            missing.append(dict(**information, exclusion_reason='One or more fixed variants missing/failed; do not change K or hide this family'))
            continue
        if len({r['original_probabilities'] for r in success}) != 1: raise ValueError('Original baseline changed across variants')
        values = behavior_features(base['original_probabilities'], [r['perturbed_probabilities'] for r in success])
        label = int(base['true_label']); prediction = int(base['original_prediction'])
        preds = [int(probability(r['perturbed_probabilities']).argmax()) for r in success]
        values.update(correct_to_wrong_rate=float(np.mean([prediction == label and x != label for x in preds])) if information['all_variants_label_preserving'] else None,
                      wrong_to_correct_rate=float(np.mean([prediction != label and x == label for x in preds])) if information['all_variants_label_preserving'] else None)
        result.append(dict(**information, **values))
    return result, missing


def cluster_paired_bootstrap(rows, metric, seed, repetitions=2000, dialogue_universe=None):
    """Same resampled Dialogues supply BOTH correct/error means in each draw.

    Utterance-stratified selection is not inverse-weighted into a population claim.
    Confidence intervals describe the selected pilot conditional on its selection.
    """
    ids = sorted(set(map(str, dialogue_universe))) if dialogue_universe is not None else sorted({str(r['dialogue_id']) for r in rows})
    correct = [r for r in rows if int(r['original_correct']) == 1]
    wrong = [r for r in rows if int(r['original_correct']) == 0]
    correct_ids = {str(r['dialogue_id']) for r in correct}; wrong_ids = {str(r['dialogue_id']) for r in wrong}
    output = dict(metric=metric, correct_samples=len(correct), error_samples=len(wrong), dialogues=len(ids),
                  observed_dialogues=len({str(r['dialogue_id']) for r in rows}),
                  correct_dialogues=len(correct_ids), error_dialogues=len(wrong_ids),
                  shared_correct_error_dialogues=len(correct_ids & wrong_ids),
                  correct_mean=float(np.mean([r[metric] for r in correct])) if correct else None,
                  error_mean=float(np.mean([r[metric] for r in wrong])) if wrong else None,
                  estimand='Selected-pilot error minus correct utterance mean; clusters resampled as paired Dialogues',
                  limitation='Cluster resampling controls dependence; this between-group contrast does not itself eliminate Dialogue confounding')
    output['error_minus_correct'] = output['error_mean'] - output['correct_mean'] if correct and wrong else None
    if len(correct_ids) < 2 or len(wrong_ids) < 2:
        output.update(status='INSUFFICIENT_DIALOGUE_SUPPORT', bootstrap_valid=0, ci_low=None, ci_high=None)
        return output
    # Per-Dialogue sums/counts yield exact cluster resampling with duplicate draws.
    stats = np.zeros((len(ids), 4), dtype=np.float64)
    positions = {did: i for i, did in enumerate(ids)}
    for row in rows:
        index = positions[str(row['dialogue_id'])]; group = int(row['original_correct'])
        offset = 0 if group == 1 else 2
        stats[index, offset] += float(row[metric]); stats[index, offset + 1] += 1
    rng = np.random.RandomState(seed); differences = []
    for _ in range(repetitions):
        total = stats[rng.randint(0, len(ids), len(ids))].sum(axis=0)
        if total[1] and total[3]: differences.append(total[2] / total[3] - total[0] / total[1])
    low, high = np.quantile(differences, [.025, .975]) if differences else (None, None)
    output.update(status='OK' if len(differences) >= .9 * repetitions else 'LOW_VALID_BOOTSTRAP_SUPPORT',
                  bootstrap_requested=repetitions, bootstrap_valid=len(differences),
                  ci_low=float(low) if low is not None else None, ci_high=float(high) if high is not None else None)
    return output


def matched_dialogue_confidence(rows, metric, seed, repetitions=2000, dialogue_universe=None):
    """Equal fixed cell weights, matching on exact Dialogue AND original confidence bin.

    Cell inclusion uses clean baseline correct/error support, never perturbed effect.
    The same full-cohort Dialogue bootstrap draws are reused across conditions.
    """
    ids = sorted(set(map(str, dialogue_universe))) if dialogue_universe is not None else sorted({str(r['dialogue_id']) for r in rows})
    cells = defaultdict(list)
    for row in rows: cells[(str(row['dialogue_id']), row['confidence_bin'])].append(row)
    valid = []
    for (did, bin_name), values in sorted(cells.items()):
        correct = [r for r in values if int(r['original_correct']) == 1]
        wrong = [r for r in values if int(r['original_correct']) == 0]
        if not correct or not wrong: continue
        valid.append(dict(dialogue_id=did, confidence_bin=bin_name, correct_n=len(correct), error_n=len(wrong),
                          difference=float(np.mean([r[metric] for r in wrong]) - np.mean([r[metric] for r in correct])),
                          fixed_weight=1.))
    supported_dialogues = {r['dialogue_id'] for r in valid}
    supported_samples = sum(r['correct_n'] + r['error_n'] for r in valid)
    result = dict(metric=metric, all_cells=len(cells), eligible_cells=len(valid),
                  all_dialogues=len(ids), matched_dialogues=len(supported_dialogues),
                  candidate_samples=len(rows), matched_samples=supported_samples,
                  sample_support_fraction=supported_samples / len(rows) if rows else 0.,
                  cell_support_fraction=len(valid) / len(cells) if cells else 0.,
                  fixed_weights='Equal weight 1 per eligible Dialogue x original confidence-bin cell',
                  confidence_bin_width=.1,
                  error_minus_correct=float(np.mean([r['difference'] for r in valid])) if valid else None,
                  interpretation='Within-Dialogue/within-confidence-bin selected-cohort association; not calibration gain or proof of incremental predictive performance')
    if len(supported_dialogues) < 2:
        result.update(status='INSUFFICIENT_MATCHED_DIALOGUE_SUPPORT', bootstrap_valid=0, ci_low=None, ci_high=None)
        return result, valid
    # Fixed equal cell weights remain unchanged; resampling only changes each
    # Dialogue's multiplicity, including zero-support Dialogues from the cohort.
    stats = np.zeros((len(ids), 2), dtype=np.float64); positions = {did:i for i,did in enumerate(ids)}
    for row in valid:
        index=positions[row['dialogue_id']]
        stats[index,0] += row['fixed_weight'] * row['difference']; stats[index,1] += row['fixed_weight']
    rng=np.random.RandomState(seed); draws=[]
    for _ in range(repetitions):
        total=stats[rng.randint(0,len(ids),len(ids))].sum(axis=0)
        if total[1]: draws.append(total[0]/total[1])
    low,high=np.quantile(draws,[.025,.975]) if draws else (None,None)
    result.update(status='OK' if len(draws)>=.9*repetitions else 'LOW_VALID_BOOTSTRAP_SUPPORT',
                  bootstrap_requested=repetitions,bootstrap_valid=len(draws),
                  ci_low=float(low) if low is not None else None,ci_high=float(high) if high is not None else None)
    return result,valid


def condition_comparisons(rows, dataset, kind, scope, args, universe):
    comparisons=[]; matches=[]; cells=[]
    bin_names=[f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)]
    for bin_name in ['all'] + bin_names:
        subset=rows if bin_name=='all' else [r for r in rows if r['confidence_bin']==bin_name]
        # Same master seed/universe for every type/modality/metric/scope in this
        # dataset/bin. This supports paired draws without treating types as K draws.
        seed=int(hashlib.sha256((str(args.seed)+dataset+bin_name).encode()).hexdigest()[:8],16)
        for metric in METRICS:
            comparison=cluster_paired_bootstrap(subset,metric,seed,args.bootstrap,universe)
            comparisons.append(dict(dataset=dataset,perturbation_type=kind,analysis_scope=scope,
                 confidence_bin=bin_name,confidence_control='Fixed width0.1 original-confidence bin; between-group cluster CI alone does not remove Dialogue confounding',**comparison))
    seed=int(hashlib.sha256((str(args.seed)+dataset+'matched_cells').encode()).hexdigest()[:8],16)
    for metric in METRICS:
        matched,detail=matched_dialogue_confidence(rows,metric,seed,args.bootstrap,universe)
        matches.append(dict(dataset=dataset,perturbation_type=kind,analysis_scope=scope,**matched))
        cells.extend(dict(dataset=dataset,perturbation_type=kind,analysis_scope=scope,metric=metric,**r) for r in detail)
    return comparisons,matches,cells


def write_table_even_empty(path, rows, fields):
    if rows: write_csv(path,rows)
    else:
        with Path(path).open('w',newline='') as stream:
            csv.DictWriter(stream,fieldnames=list(dict.fromkeys(fields))).writeheader()


def analysis(args):
    rows = []
    for path in args.predictions: rows.extend(read_csv(path))
    keys = [(r['dataset'], r['sample_id'], r['variant_id']) for r in rows]
    if len(keys) != len(set(keys)): raise ValueError('Duplicate variant record across input files')
    all_groups = {(r['dataset'], r['sample_id'], r['perturbation_type']) for r in rows}
    variants = [per_variant(row) for row in rows if row['status'] == 'SUCCESS']
    samples, missing = aggregate_samples(rows)
    # Preserve diagnostics for all valid complete families independently of QA.
    # Only reviewed label-preserving families enter supervised reliability analyses.
    main = [row for row in samples if row['all_variants_label_preserving']]
    exploratory = [row for row in samples if row['modality']=='audio' and not row['all_variants_label_preserving']
                   and row['all_variants_automatic_qc_pass'] and 'FAIL' not in row['label_review_statuses']]
    for row in exploratory:
        row['exploratory_analysis_scope']='AUDIO_SIGNAL_QC_ONLY_NOT_LABEL_CONFIRMED'
    comparisons, summaries, dialogues, matches, match_cells = [], [], [], [], []
    exploratory_comparisons, exploratory_summaries, exploratory_matches, exploratory_cells = [], [], [], []
    families = sorted({(r['dataset'], r['perturbation_type']) for r in rows})
    for dataset, kind in families:
        complete = [r for r in samples if r['dataset'] == dataset and r['perturbation_type'] == kind]
        selected = [r for r in main if r['dataset'] == dataset and r['perturbation_type'] == kind]
        universe=sorted({str(r['dialogue_id']) for r in rows if r['dataset']==dataset})
        summary = dict(dataset=dataset, perturbation_type=kind, analysis_scope='STRICT_ATTRIBUTED_LABEL_REVIEW_MAIN', planned_k=len(TYPES[kind]),
                       planned_sample_families=sum(x[0] == dataset and x[2] == kind for x in all_groups),
                       complete_families=len(complete), label_preserving_main_families=len(selected),
                       incomplete_families=sum(r['dataset']==dataset and r['perturbation_type']==kind for r in missing),
                       unconfirmed_or_failed_review_families=len(complete)-len(selected),
                       population_claim=False, high_confidence_error_threshold=.9)
        for correct in [0, 1]:
            group = [r for r in selected if r['original_correct'] == correct]
            summary[('correct' if correct else 'error') + '_samples'] = len(group)
            for metric in METRICS:
                summary[('correct' if correct else 'error') + '_' + metric] = float(np.mean([r[metric] for r in group])) if group else None
        summaries.append(summary)
        c,m,cells=condition_comparisons(selected,dataset,kind,'STRICT_ATTRIBUTED_LABEL_REVIEW_MAIN',args,universe)
        comparisons.extend(c);matches.extend(m);match_cells.extend(cells)
        if kind in ['pitch_shift','timbre_tilt']:
            unconfirmed=[r for r in exploratory if r['dataset']==dataset and r['perturbation_type']==kind]
            exploratory_summary=dict(dataset=dataset,perturbation_type=kind,
                 analysis_scope='AUDIO_SIGNAL_QC_ONLY_NOT_LABEL_CONFIRMED',samples=len(unconfirmed),
                 perturbed_accuracy_or_calibration_computed=False,
                 original_correctness_is_clean_baseline_only=True)
            for correct in [0,1]:
                group=[r for r in unconfirmed if r['original_correct']==correct]
                prefix='correct' if correct else 'error'
                exploratory_summary[prefix+'_samples']=len(group)
                for metric in METRICS:
                    exploratory_summary[prefix+'_'+metric]=float(np.mean([r[metric] for r in group])) if group else None
            exploratory_summaries.append(exploratory_summary)
            c,m,cells=condition_comparisons(unconfirmed,dataset,kind,'AUDIO_SIGNAL_QC_ONLY_NOT_LABEL_CONFIRMED',args,universe)
            exploratory_comparisons.extend(c);exploratory_matches.extend(m);exploratory_cells.extend(cells)
        for did in sorted({r['dialogue_id'] for r in complete}):
            group = [r for r in complete if r['dialogue_id'] == did]
            for correct in [0, 1]:
                subset = [r for r in group if r['original_correct'] == correct]
                record = dict(dataset=dataset, perturbation_type=kind, dialogue_id=did, original_correct=correct,
                              analysis_scope='ALL_COMPLETE_DIAGNOSTIC_NOT_LABEL_PRESERVING_MAIN',
                              complete_samples=len(subset), main_eligible_samples=sum(r['all_variants_label_preserving'] for r in subset))
                record.update({metric:float(np.mean([r[metric] for r in subset])) if subset else None for metric in METRICS})
                dialogues.append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    feature_fields=list(samples[0]) if samples else ['dataset','sample_id','perturbation_type','planned_k','mu','delta','flip_rate','js_mean']
    write_table_even_empty(args.output / 'behavior_features_all_complete.csv',samples,feature_fields)
    write_table_even_empty(args.output / 'behavior_features_main.csv',main,feature_fields)
    write_table_even_empty(args.output / 'audio_exploratory_behavior_features.csv',exploratory,feature_fields+['exploratory_analysis_scope'])
    if missing: write_csv(args.output / 'incomplete_families.csv', missing)
    failed_qc=[r for r in rows if r['status']!='SUCCESS' or not is_true(r.get('automatic_qc_pass'))]
    if failed_qc: write_csv(args.output / 'failed_qc_records.csv',failed_qc)
    if variants:
        write_csv(args.output / 'confidence_changes.csv', variants)
        write_csv(args.output / 'prediction_flips.csv', variants)
    if comparisons: write_csv(args.output / 'correct_wrong_comparison.csv', comparisons)
    if matches: write_csv(args.output / 'within_dialogue_confidence_comparison.csv',matches)
    write_table_even_empty(args.output / 'within_dialogue_confidence_cells.csv',match_cells,
                          ['dataset','perturbation_type','analysis_scope','metric','dialogue_id','confidence_bin','correct_n','error_n','difference','fixed_weight'])
    if exploratory_comparisons: write_csv(args.output / 'audio_exploratory_correct_wrong_comparison.csv',exploratory_comparisons)
    if exploratory_matches: write_csv(args.output / 'audio_exploratory_within_dialogue_confidence_comparison.csv',exploratory_matches)
    if exploratory_cells: write_csv(args.output / 'audio_exploratory_within_dialogue_confidence_cells.csv',exploratory_cells)
    if exploratory_summaries: write_csv(args.output / 'audio_exploratory_condition_summary.csv',exploratory_summaries)
    if summaries: write_csv(args.output / 'condition_summary.csv', summaries)
    if dialogues: write_csv(args.output / 'dialogue_summary.csv', dialogues)
    save_json(args.output / 'analysis_verification.json', dict(status='COMPLETE', input_records=len(rows),
              successful_records=len(variants), failed_records=len(rows)-len(variants), complete_sample_families=len(samples),
              main_eligible_sample_families=len(main), incomplete_sample_families=len(missing),
              audio_signal_qc_only_exploratory_families=len(exploratory),
              failed_qc_records=len(failed_qc),
              unconfirmed_audio_not_used_for_perturbed_accuracy_or_calibration=True,
              formal_definitions=dict(mu='mean(1[argmax(p_prime_k)!=argmax(p)] * max(p_prime_k))',
                                      delta='abs(max(p) - mean(max(p_prime_k)))', flip_rate='mean(argmax(p_prime_k)!=argmax(p))',
                                      js='0.5*KL(p||m)+0.5*KL(p_prime||m), m=(p+p_prime)/2, natural log, 0log0=0'),
              formal_source='xp04_random_feature_behavior_calibration/mainline_mu_delta/20261115T165025Z/scripts/build_inputs.py:34-42',
              repetitions=args.bootstrap, seed=args.seed, bootstrap_unit='Dialogue, paired draws for correct and error',
              shared_bootstrap='Full-cohort Dialogue universe; identical seed by dataset and confidence bin across types/modalities/metrics/scopes',
              dialogue_effect_adjustment='Separate exact Dialogue x width0.1 confidence-bin cells with both clean-correct/error; fixed equal cell weights; full-cohort Dialogue bootstrap',
              cluster_ci_alone_does_not_remove_dialogue_confounding=True,
              no_mixed_k=True, no_fitted_reliability_or_calibration=True, test_accessed=False,
              confidence_bins=[i/10 for i in range(11)], selected_sample_population_metrics=False,
              selection_bias_and_limited_confidence_overlap_require_cautious_interpretation=True,
              input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in args.predictions},
              code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261120)
    args = parser.parse_args()
    if args.bootstrap != 2000: raise ValueError('Frozen pilot uses 2000 paired Dialogue bootstrap draws')
    analysis(args)


if __name__ == '__main__': main()
