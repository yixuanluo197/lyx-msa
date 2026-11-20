#!/usr/bin/env python3
"""Engineering repeatability of 172 inherited fixed text conditions only.

Run only after the owner confirms inference is complete. No models, raw text,
features, effect subgroups, fitted scores or scientific hypothesis tests are used.
This is reused-input engineering replay, never independent confirmation.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


RUN = Path(__file__).resolve().parents[1]
PARENT = RUN.parent / '20261120T143131Z'
FAMILIES = {'neutral_append': {'neutral_a', 'neutral_b'},
            'semantic_paraphrase': {'paraphrase_a', 'paraphrase_b'}}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def truth(value):
    return value is True or str(value).lower() in {'true', '1'}


def probability(value):
    p = np.asarray(json.loads(value), dtype=np.float64)
    if p.shape != (7,) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Expected a finite seven-class MELD softmax vector')
    if not np.isclose(p.sum(), 1., atol=1e-5, rtol=0.):
        raise ValueError('Invalid softmax normalization')
    return p


def key(row):
    return row['sample_id'], row['variant_id']


def main():
    output = RUN / 'provenance/overlap_reinference.json'
    if output.exists():
        raise FileExistsError('Preserve prior engineering replay report')
    status_path = RUN / 'provenance/text_engine_infer.json'
    state = json.loads(status_path.read_text())
    if state.get('status') != 'COMPLETE' or state.get('planned') != 1624:
        raise RuntimeError('Frozen inference is not complete')
    new_path = RUN / 'results/perturbed_predictions.csv'
    old_path = PARENT / 'results/pilot/MELD/perturbed_predictions_both.csv'
    if sha(new_path) != state['output_sha256']:
        raise RuntimeError('Completed inference output changed')
    lock = json.loads((RUN / 'configs/COHORT_LOCK.json').read_text())
    if lock.get('reused_variants') != 172:
        raise ValueError('Expected the predeclared 172 reused text conditions')
    current = read_csv(new_path)
    overlap = [row for row in current if truth(row.get('is_previous_pilot', False))]
    if len(overlap) != 172 or len({key(row) for row in overlap}) != 172:
        raise ValueError('All 172 fixed overlap conditions must be retained exactly once')
    old_rows = [row for row in read_csv(old_path) if row['modality'] == 'text']
    old = {key(row): row for row in old_rows}
    if len(old) != len(old_rows):
        raise ValueError('Ambiguous parent text condition keys')
    records, deltas, predicted_equal, bound_errors = [], [], [], []
    statuses = Counter()
    for row in sorted(overlap, key=key):
        prior = old.get(key(row))
        record = dict(sample_id=row['sample_id'], variant_id=row['variant_id'],
                      perturbation_type=row['perturbation_type'],
                      current_status=row['status'],
                      parent_status=prior['status'] if prior else 'MISSING_PARENT_CONDITION',
                      current_automatic_qc_pass=row.get('automatic_qc_pass', ''),
                      parent_automatic_qc_pass=prior.get('automatic_qc_pass', '') if prior else '',
                      current_label_review_status=row.get('label_review_status', ''),
                      parent_label_review_status=prior.get('label_review_status', '') if prior else '',
                      current_error=row.get('error', ''),
                      parent_error=prior.get('error', '') if prior else '')
        if row['perturbation_type'] not in FAMILIES or row['variant_id'] not in FAMILIES[row['perturbation_type']]:
            raise ValueError('Overlap contains an unexpected condition')
        if prior is None:
            bound_errors.append(key(row))
            record['comparison_category'] = 'MISSING_PARENT_CONDITION'
        else:
            fields = ['dialogue_id', 'utterance_index', 'utterance_id', 'speaker_id',
                      'perturbation_type', 'original_input_sha256', 'processed_input_sha256',
                      'original_feature_sha256']
            identity = {name: row.get(name) == prior.get(name) for name in fields}
            record['same_fixed_condition_identity'] = identity
            # Parent producer hashes int64 array bytes+shape; follow-up hashes
            # the JSON token list. Different digest schemes must not be compared
            # as if they proved different token IDs. The frozen raw strings and
            # tokenizer remain bound independently by the two engine locks.
            record['token_digest_provenance'] = dict(
                parent=prior.get('actual_token_ids_sha256', ''),
                current=row.get('actual_token_ids_sha256', ''),
                comparable=False,
                reason='Parent array_hash(int64 IDs) versus follow-up json_hash(token list)')
            record['qc_and_review_status_unchanged'] = all(
                row.get(name, '') == prior.get(name, '')
                for name in ['automatic_qc_pass', 'label_review_status'])
            if not all(identity.values()):
                bound_errors.append(key(row))
            successful = row['status'] == 'SUCCESS' and prior['status'] == 'SUCCESS'
            record['comparison_category'] = ('SUCCESS_BOTH' if successful else
                'PARENT_SUCCESS_CURRENT_FAILED' if prior['status'] == 'SUCCESS' else
                'CURRENT_SUCCESS_PARENT_FAILED' if row['status'] == 'SUCCESS' else 'FAILED_BOTH')
            if successful:
                p, q = probability(prior['perturbed_probabilities']), probability(row['perturbed_probabilities'])
                delta = np.abs(q-p)
                identical_class = int(p.argmax()) == int(q.argmax())
                record.update(max_probability_abs_error=float(delta.max()),
                              mean_probability_abs_error=float(delta.mean()),
                              prediction_unchanged=identical_class,
                              parent_prediction=int(p.argmax()), current_prediction=int(q.argmax()),
                              probability_allclose_parent_control_tolerance=bool(np.allclose(p,q,atol=1e-4,rtol=1e-4)))
                deltas.append(delta)
                predicted_equal.append(identical_class)
        statuses[record['comparison_category']] += 1
        records.append(record)
    errors = np.stack(deltas) if deltas else None
    result = dict(status='ENGINEERING_REPLAY_RECORDED' if not bound_errors else 'FIXED_CONDITION_IDENTITY_MISMATCH',
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  meaning='Reinference of reused inputs on the same frozen model; engineering repeatability only',
                  independent_validation=False, scientific_effect_subgroups=False,
                  expected_fixed_conditions=172, recorded_fixed_conditions=len(records),
                  comparison_counts=dict(statuses), successful_both=len(deltas),
                  maximum_probability_absolute_error=float(errors.max()) if errors is not None else None,
                  mean_probability_absolute_error=float(errors.mean()) if errors is not None else None,
                  prediction_agreement=float(np.mean(predicted_equal)) if predicted_equal else None,
                  all_successful_pairs_within_parent_control_tolerance=(all(r.get('probability_allclose_parent_control_tolerance', True) for r in records)
                                                                        if deltas else None),
                  tolerance=dict(atol=1e-4,rtol=1e-4,origin='Frozen parent clean-control engineering tolerance; not a new scientific threshold'),
                  identity_mismatch_keys=bound_errors, all_QC_failures_and_status_differences_retained=True,
                  input_sha256={str(p):sha(p) for p in [new_path,old_path,status_path,RUN/'configs/COHORT_LOCK.json',Path(__file__)]},
                  per_condition=records, models_loaded=False, original_media_or_text_read=False,
                  training=False, new_seed=False, test_access=False, fitted_score=False)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=True,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_condition','input_sha256']},indent=2))
    if bound_errors:
        raise RuntimeError('Fixed overlap condition identity mismatch; report preserved')


if __name__ == '__main__':
    main()
