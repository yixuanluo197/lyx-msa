#!/usr/bin/env python3
"""Freeze clean-only maximal matching and outcome-blind generation templates."""
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from transformers import RobertaTokenizer

D = Path(__file__).resolve().parents[1]
P = Path(json.loads((D / 'configs/PARENT_RUN.json').read_text())['parent_run'])
sys.path.insert(0, str(P / 'scripts'))
import raw_pilot
import text_encoder


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def save_json(path, obj):
    with path.open('x') as stream:
        stream.write(json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False) + '\n')


def save_csv(path, values):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def save_jsonl(path, values):
    with path.open('x', encoding='utf-8') as stream:
        for row in values:
            stream.write(json.dumps(row, ensure_ascii=True) + '\n')


def matching(rows):
    cells = defaultdict(lambda: [[], []])
    for r in rows:
        c = float(r['original_confidence'])
        b = min(9, math.floor(c * 10))
        name = f'{b / 10:.1f}-{(b + 1) / 10:.1f}'
        if r['confidence_bin'] != name:
            raise ValueError('Source fixed confidence bin does not match FP64 confidence')
        cells[(str(r['dialogue_id']), name)][int(r['original_correct'])].append(r)
    pairs = []
    for (did, b), groups in sorted(cells.items()):
        wrong, correct = [sorted(g, key=lambda x: x['sample_id']) for g in groups]
        if not wrong or not correct:
            continue
        cost = np.abs(np.asarray([float(x['original_confidence']) for x in correct])[:, None]
                      - np.asarray([float(x['original_confidence']) for x in wrong])[None, :])
        ci, ei = linear_sum_assignment(cost)
        for cidx, eidx in zip(ci, ei):
            c, e = correct[cidx], wrong[eidx]
            identity = json.dumps([did, b, c['sample_id'], e['sample_id']])
            pairs.append(dict(pair_id='P_' + hashlib.sha256(identity.encode()).hexdigest()[:16],
                              dialogue_id=did, confidence_bin=b,
                              correct_sample_id=c['sample_id'], error_sample_id=e['sample_id'],
                              correct_utterance_index=int(c['utterance_index']),
                              error_utterance_index=int(e['utterance_index']),
                              correct_confidence=float(c['original_confidence']),
                              error_confidence=float(e['original_confidence']),
                              confidence_signed_gap=float(e['original_confidence']) - float(c['original_confidence']),
                              confidence_abs_gap=float(cost[cidx, eidx])))
    return pairs, cells


def self_test():
    # Exact minimum cost and maximum-cardinality result against tiny brute force.
    for correct, wrong in [([.31, .34, .38], [.32, .39]), ([.31], [.32, .35])]:
        cost = np.abs(np.asarray(correct)[:, None] - np.asarray(wrong)[None, :])
        a, b = linear_sum_assignment(cost)
        k = min(len(correct), len(wrong))
        brute = min(sum(cost[i, j] for i, j in zip(ci, ei))
                    for ci in itertools.permutations(range(len(correct)), k)
                    for ei in itertools.permutations(range(len(wrong)), k))
        assert len(a) == k and np.isclose(cost[a, b].sum(), brute)
    assert min(9, math.floor(1.0 * 10)) == 9
    return True


def main():
    raw_pilot.require_runtime('encoder')
    self_test()
    protocol = D / 'configs/EXPLORATORY_PROTOCOL.json'
    baseline_lock = P / 'configs/BASELINE_LOCK_MELD.json'
    lock = json.loads(baseline_lock.read_text())
    data_dir = P / 'local_only/data/MELD'
    paths = {'dataset_sha256': data_dir / 'dataset.pt',
             'text_features_sha256': P / 'local_only/features/MELD/text_features.pt',
             'audio_features_sha256': P / 'local_only/features/MELD/audio_features.pt',
             'checkpoint_sha256': data_dir / 'runs/cmerc/best.pt'}
    for key, path in paths.items():
        if sha(path) != lock[key]:
            raise ValueError('Frozen parent asset hash differs: ' + key)
    prediction_source = P / 'results/pilot/MELD/original_predictions.csv'
    previous_samples = read_csv(P / 'results/pilot/MELD/sample_manifest.csv')
    previous_ids = {r['sample_id'] for r in previous_samples}
    rows = read_csv(prediction_source)
    assert len(rows) == 1109 and len({r['sample_id'] for r in rows}) == 1109
    data = torch.load(data_dir / 'dataset.pt', map_location='cpu', weights_only=False)
    assert data['dataset'] == 'MELD' and set(data['splits']) == {'train', 'validation'}
    dialogues = {str(d['dialogue_id']): d for d in data['splits']['validation']}
    for r in rows:
        dialogue = dialogues[r['dialogue_id']]
        index = int(r['utterance_index'])
        probs = np.asarray(json.loads(r['original_probabilities']), dtype=np.float64)
        assert probs.shape == (7,) and np.isfinite(probs).all() and np.isclose(probs.sum(), 1, atol=1e-5)
        assert int(r['true_label']) == int(dialogue['labels'][index])
        assert int(r['original_prediction']) == int(probs.argmax())
        assert int(r['original_correct']) == int(probs.argmax() == int(r['true_label']))
        assert float(r['original_confidence']) == float(probs.max())
        assert r['utterance_id'] == str(dialogue['utterance_ids'][index])
        assert int(r['speaker_id']) == int(dialogue['speaker'][index].argmax())
    pairs, cells = matching(rows)
    assert len(pairs) == 203
    assert matching(list(reversed(rows)))[0] == pairs
    lookup = {r['sample_id']: r for r in rows}
    selected, by_id = [], {}
    for pair in pairs:
        for group, key in [('correct', 'correct_sample_id'), ('error', 'error_sample_id')]:
            sid = pair[key]
            assert sid not in by_id
            sample = dict(lookup[sid], pair_id=pair['pair_id'], cohort_group=group,
                          is_previous_pilot=sid in previous_ids)
            selected.append(sample)
            by_id[sid] = sample
    assert len(selected) == 406 and len({r['dialogue_id'] for r in selected}) == 80
    status = []
    for r in rows:
        selected_here = r['sample_id'] in by_id
        has_opposite = all(cells[(r['dialogue_id'], r['confidence_bin'])])
        reason = 'MATCHED' if selected_here else ('EXCESS_MEMBER_WITHIN_ELIGIBLE_CELL' if has_opposite else 'NO_OPPOSITE_CORRECTNESS_IN_CELL')
        status.append(dict(r, matched=selected_here, match_status=reason,
                           pair_id=by_id[r['sample_id']]['pair_id'] if selected_here else '',
                           is_previous_pilot=r['sample_id'] in previous_ids))
    save_csv(D / 'results/matched_pair_manifest.csv', pairs)
    save_csv(D / 'results/matched_sample_manifest.csv', selected)
    save_csv(D / 'results/all_validation_matching_status.csv', status)
    with (P / 'local_only/pilot/MELD/text_reviewed.jsonl').open() as stream:
        old_reviews = [json.loads(line) for line in stream if line.strip()]
    reused = [r for r in old_reviews if r['sample_id'] in by_id]
    save_jsonl(D / 'local_only/reused_matched_text.jsonl', reused)
    model_dir = P / 'local_only/text_models/MELD/best_classifier'
    tokenizer = RobertaTokenizer.from_pretrained(model_dir, local_files_only=True)
    templates = []
    for sample in sorted(selected, key=lambda r: r['sample_id']):
        if sample['sample_id'] in previous_ids:
            continue
        dialogue = dialogues[sample['dialogue_id']]
        index = int(sample['utterance_index'])
        original = dialogue['texts'][index]
        clean_ids = tokenizer.encode(text_encoder.preprocessing(original, 'MELD'), add_special_tokens=True, truncation=False)
        context = [dict(utterance_index=i, speaker_id=int(dialogue['speaker'][i].argmax()), text=text)
                   for i, text in enumerate(dialogue['texts'])]
        for family in ['neutral_append', 'semantic_paraphrase']:
            for variant in raw_pilot.TYPES[family]:
                templates.append(dict(sample_id=sample['sample_id'], dialogue_id=sample['dialogue_id'],
                    utterance_index=index, perturbation_type=family, variant_id=variant,
                    original_text=original, original_text_sha256=hashlib.sha256(original.encode()).hexdigest(),
                    original_total_tokens=len(clean_ids),
                    maximum_neutral_added_tokens=min(24, max(4, math.floor(.5 * max(1, len(clean_ids)-2)))),
                    dialogue_context=context, processed_text='', review_status='PENDING', reviewer='',
                    review_method='', review_notes='',
                    instruction='Input-only semantic review; no confidence/correctness/pair/label available. Keep attitude, negation, intensity, references and speech act; neutral additions must not complete a fragment. No fabricated paraphrases for pure interjections. Retain FAIL/UNCERTAIN; actual tokenizer must change, no truncation, same family variants distinct.'))
    save_jsonl(D / 'local_only/new_text_template.jsonl', templates)
    source_files = [protocol, baseline_lock, prediction_source,
                    P / 'results/pilot/MELD/sample_manifest.csv',
                    P / 'local_only/pilot/MELD/text_reviewed.jsonl', Path(__file__), *paths.values()]
    output_files = [D / 'results/matched_pair_manifest.csv', D / 'results/matched_sample_manifest.csv',
                    D / 'results/all_validation_matching_status.csv', D / 'local_only/new_text_template.jsonl',
                    D / 'local_only/reused_matched_text.jsonl']
    gaps = np.asarray([r['confidence_abs_gap'] for r in pairs])
    result = dict(status='FROZEN_BEFORE_NEW_PERTURBED_INFERENCE', frozen_at=datetime.now(timezone.utc).isoformat(),
        pairs=len(pairs), targets=len(selected), dialogues=80, validation_rows=1109, validation_dialogues=114,
        eligible_cells=sum(all(v) for v in cells.values()), all_cells=len(cells),
        target_coverage=len(selected)/len(rows), pair_capacity=203,
        previous_pilot_overlap=sum(r['is_previous_pilot'] for r in selected),
        new_targets=len(templates)//4, reused_variants=len(reused), new_variants=len(templates),
        matched_text_variants=len(selected)*4, total_blind_package_variants=480+len(templates),
        confidence_gap=dict(mean=float(gaps.mean()), median=float(np.median(gaps)),
                            q90=float(np.quantile(gaps,.9)), maximum=float(gaps.max())),
        calipers={str(t):dict(pairs=sum(r['confidence_abs_gap']<=t for r in pairs),
                    dialogues=len({r['dialogue_id'] for r in pairs if r['confidence_abs_gap']<=t})) for t in [.02,.01]},
        no_test=True, no_training=True, no_new_perturbed_outputs_inspected=True,
        sources_sha256={str(p.resolve()):sha(p) for p in source_files},
        outputs_sha256={str(p.resolve()):sha(p) for p in output_files},
        matching_engine=dict(numpy=np.__version__, scipy=__import__('scipy').__version__),
        tiny_exhaustive_matching_test=True, input_order_invariance=True)
    save_json(D / 'configs/COHORT_LOCK.json', result)
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('sha256')}, indent=2))


if __name__ == '__main__':
    main()
