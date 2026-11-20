#!/usr/bin/env python3
"""Assemble immutable AI records and a genuinely unfilled human blind-review CSV.

Full text is local only. No prediction or outcome is read by this program.
"""
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
PARENT = RUN.parent / '20261120T143131Z'
PARTS = ['part_000_091', 'part_091_182', 'part_182_273', 'part_273_363']


def read(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return row['sample_id'], row['variant_id']


def jsonl(path, rows):
    with path.open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, allow_nan=False)+'\n')


def csvfile(path, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    template = {key(r): r for r in read(RUN/'local_only/new_text_template.jsonl')}
    generated, overlays, source_files = [], [], []
    for stem in PARTS:
        p = RUN/'local_only/generated_parts'/f'{stem}.jsonl'
        q = RUN/'local_only/generated_parts'/f'{stem}.second_review.jsonl'
        generated.extend(read(p)); overlays.extend(read(q)); source_files.extend([p, q])
    assert len(generated) == 1452 and len({key(r) for r in generated}) == 1452
    assert {key(r) for r in generated} == set(template)
    assert len(overlays) == 1452 and len({key(r) for r in overlays}) == 1452
    second = {key(r): r for r in overlays}
    assert set(second) == set(template)
    editable = {'processed_text','review_status','reviewer','review_method','review_notes',
                'initial_review_status','automatic_qc','automatic_qc_pass'}
    for row in generated:
        for field, value in template[key(row)].items():
            if field not in editable:
                assert row[field] == value, (key(row), field)
        assert row['review_status'] in ('PASS','FAIL','UNCERTAIN')
        qa = second[key(row)]
        assert qa['review_status'] in ('PASS','FAIL','UNCERTAIN')
        assert qa['reviewer'] and qa['reviewer'] != row['reviewer']
        assert qa['processed_text_sha256'] == hashlib.sha256(row['processed_text'].encode()).hexdigest()
        row['generation_review_status'] = row['review_status']
        row['generation_reviewer'] = row['reviewer']
        row['generation_review_notes'] = row['review_notes']
        row['second_ai_review'] = qa
        # A second AI review can veto PASS, never upgrade a rejected/uncertain version.
        if row['review_status'] == 'PASS' and qa['review_status'] != 'PASS':
            row['review_status'] = qa['review_status']
        row['reviewer'] += ' + ' + qa['reviewer']
        # Parent eligibility uses an exact, frozen method whitelist. Independent
        # QA attribution stays in second_ai_review and reviewer/notes, not a new
        # method name that the frozen parent would incorrectly reject.
        row['review_method'] = 'AI_CONTEXT_SEMANTIC_REVIEW'
        row['review_notes'] += ' | Independent AI QA: '+qa['review_notes']
    reused = read(RUN/'local_only/reused_matched_text.jsonl')
    old = read(PARENT/'local_only/pilot/MELD/text_reviewed.jsonl')
    old_lookup = {key(r): r for r in old}
    assert len(old) == 480 and len(reused) == 172
    assert all(old_lookup[key(r)] == r for r in reused)
    matched = sorted(reused+generated, key=key)
    assert len(matched) == 1624 and len({key(r) for r in matched}) == 1624
    combined = old+generated
    assert len(combined) == 1932 and len({key(r) for r in combined}) == 1932
    jsonl(RUN/'local_only/matched_text_reviewed.jsonl', matched)
    jsonl(RUN/'local_only/ai_review_records.jsonl', combined)
    # Randomized review IDs contain no source ID, family, label, correctness or AI decision.
    shuffled = sorted(combined, key=key)
    random.Random(20261120).shuffle(shuffled)
    dids = sorted({str(r['dialogue_id']) for r in combined})
    random.Random(20261121).shuffle(dids)
    anonymous_dialogues = {d: f'D{i+1:04d}' for i, d in enumerate(dids)}
    public_rows, private_rows = [], []
    for i, row in enumerate(shuffled):
        rid = f'R{i+1:05d}'
        assert all(set(turn) == {'utterance_index','speaker_id','text'}
                   for turn in row['dialogue_context']), 'Unexpected unblinded context metadata'
        record = dict(review_id=rid, blind_dialogue_id=anonymous_dialogues[str(row['dialogue_id'])],
            full_original_dialogue_context=json.dumps(row['dialogue_context'],ensure_ascii=True),
            target_utterance_index=row['utterance_index'], original_target_utterance=row['original_text'],
            perturbed_target_utterance=row['processed_text'])
        for reviewer in ('reviewer_1','reviewer_2'):
            for field in ('id','emotion_meaning_preserved','negation_changed','tone_changed',
                          'emotion_intensity_changed','other_meaning_changed','PASS_FAIL_UNCERTAIN','notes'):
                record[f'{reviewer}_{field}'] = ''
        record['human_consensus_status'] = 'PENDING'
        public_rows.append(record)
        private_rows.append(dict(review_id=rid,sample_id=row['sample_id'],dialogue_id=row['dialogue_id'],
            variant_id=row['variant_id'],perturbation_type=row['perturbation_type'],
            ai_review_status=row['review_status'],source='previous_pilot' if key(row) in old_lookup else 'new_matched'))
    csvfile(RUN/'TEXT_REVIEW_PACKAGE.csv', public_rows)
    csvfile(RUN/'local_only/review_key.csv', private_rows)
    sources = source_files + [RUN/'local_only/new_text_template.jsonl',
        RUN/'local_only/reused_matched_text.jsonl', PARENT/'local_only/pilot/MELD/text_reviewed.jsonl',Path(__file__)]
    outputs = [RUN/'local_only/matched_text_reviewed.jsonl',RUN/'local_only/ai_review_records.jsonl',
               RUN/'TEXT_REVIEW_PACKAGE.csv',RUN/'local_only/review_key.csv']
    audit = dict(status='FROZEN_BEFORE_NEW_MODEL_INFERENCE',created_at=datetime.now(timezone.utc).isoformat(),
        matched_rows=len(matched),blind_rows=len(public_rows),human_completed_rows=0,
        matched_ai_status_counts=dict(Counter(r['review_status'] for r in matched)),
        new_generation_status_counts=dict(Counter(r['generation_review_status'] for r in generated)),
        new_final_ai_status_counts=dict(Counter(r['review_status'] for r in generated)),
        unchanged_previous_rows=172,previous_blind_rows=480,new_rows=1452,
        sources_sha256={str(p):digest(p) for p in sources},outputs_sha256={str(p):digest(p) for p in outputs},
        blind_hidden_fields=['true_label','original_correct','confidence','probabilities','pair_id','AI review status/notes'],
        warning='Full raw dialogue text: LOCAL ONLY. AI reviews do not constitute human label validation.')
    with (RUN/'provenance/TEXT_INPUT_LOCK.json').open('x') as f:
        json.dump(audit,f,indent=2,ensure_ascii=True);f.write('\n')
    print(json.dumps({k:v for k,v in audit.items() if not k.endswith('sha256')},indent=2))


if __name__ == '__main__':
    main()
