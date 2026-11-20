#!/usr/bin/env python3
"""Freeze attributed input-only reviews; preserve each original template field."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDITABLE = {'processed_text', 'review_status', 'reviewer', 'review_method', 'review_notes'}


def read(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['IEMOCAP', 'MELD'])
    args = parser.parse_args()
    local = ROOT / 'local_only/pilot' / args.dataset
    destination = local / 'text_reviewed.jsonl'
    if destination.exists():
        raise FileExistsError('Do not overwrite frozen input reviews')
    template_path = local / 'text_review_template.jsonl'
    template = read(template_path)
    template_keys = [(r['sample_id'], r['variant_id']) for r in template]
    if len(template) != 480 or len(set(template_keys)) != 480:
        raise ValueError('Template must contain exactly 120 x four fixed text variants')
    parts = [local / 'generated_parts' / name for name in
             ['part_000_060.jsonl', 'part_060_120.jsonl']]
    reviewed = {}
    for part in parts:
        rows = read(part)
        if len(rows) != 240:
            raise ValueError('Each assigned partition must preserve all 240 variants')
        for row in rows:
            key = row['sample_id'], row['variant_id']
            if key in reviewed:
                raise ValueError('Duplicate reviewed condition')
            reviewed[key] = row
    if set(template_keys) != set(reviewed):
        raise ValueError('Review differs from fixed template condition set')
    second_path = local / 'independent_semantic_review.json'
    second = json.loads(second_path.read_text())
    if second['dataset'] != args.dataset:
        raise ValueError('Independent review belongs to a different dataset')
    for part in parts:
        if second['source_sha256'].get(str(part)) != sha(part):
            raise ValueError('Independent review is not bound to these completed candidates')
    second_rows = second['records']
    second_lookup = {(r['sample_id'], r['variant_id']): r for r in second_rows}
    if len(second_rows) != 480 or set(second_lookup) != set(template_keys):
        raise ValueError('Independent input-only semantic QA must cover all variants')
    final = []
    for original in template:
        row = reviewed[(original['sample_id'], original['variant_id'])]
        for key, value in original.items():
            if key not in EDITABLE and row.get(key) != value:
                raise ValueError('Review changed immutable original/context field: ' + key)
        if row.get('review_status') not in ['PASS', 'FAIL', 'UNCERTAIN', 'PENDING']:
            raise ValueError('Unknown semantic review status')
        if row.get('review_method') != 'AI_CONTEXT_SEMANTIC_REVIEW':
            raise ValueError('Do not attribute generated reviews to human review')
        if not all(isinstance(row.get(key), str) and row[key].strip()
                   for key in ['processed_text', 'reviewer', 'review_notes']):
            raise ValueError('Each candidate requires attributed text and concrete notes')
        check = second_lookup[(row['sample_id'], row['variant_id'])]
        if (check.get('review_status') not in ['PASS', 'FAIL', 'UNCERTAIN', 'PENDING']
                or (row['review_status'] != 'PASS' and check['review_status'] != row['review_status'])):
            raise ValueError('Independent QA may downgrade PASS only; no upgrades or uncertainty replacement')
        if (check.get('review_method') != 'AI_CONTEXT_SEMANTIC_REVIEW'
                or not check.get('reviewer', '').strip() or not check.get('review_notes', '').strip()):
            raise ValueError('Independent review requires actual AI attribution and notes')
        row = dict(row, initial_review_status=row['review_status'],
                   review_status=check['review_status'],
                   reviewer=row['reviewer'] + '; independent: ' + check['reviewer'],
                   review_notes=row['review_notes'] + ' | Independent AI QA: ' + check['review_notes'])
        final.append(row)
    with destination.open('x', encoding='utf-8') as stream:
        for row in final:
            stream.write(json.dumps(row, ensure_ascii=True) + '\n')
    report = dict(dataset=args.dataset, status='INPUT_REVIEW_FROZEN',
                  frozen_at=datetime.now(timezone.utc).isoformat(), variants=len(final),
                  unique_targets=len({r['sample_id'] for r in final}),
                  semantic_status_counts=dict(Counter(r['review_status'] for r in final)),
                  by_family={kind: dict(Counter(r['review_status'] for r in final if r['perturbation_type'] == kind))
                             for kind in sorted({r['perturbation_type'] for r in final})},
                  attributed_ai_review=True, human_review=False,
                  independent_ai_second_review=True,
                  downgraded_pass_candidates=sum(r['initial_review_status'] == 'PASS' and r['review_status'] != 'PASS' for r in final),
                  no_perturbed_predictions_examined=True,
                  not_eligible_until_independent_automatic_token_qc=True,
                  all_uncertain_failed_candidates_retained=True,
                  immutable_template_fields_preserved=True,
                  source_sha256={str(p): sha(p) for p in [template_path, *parts, second_path, Path(__file__)]},
                  final_sha256=sha(destination))
    output = ROOT / 'provenance' / ('text_input_review_' + args.dataset + '.json')
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
