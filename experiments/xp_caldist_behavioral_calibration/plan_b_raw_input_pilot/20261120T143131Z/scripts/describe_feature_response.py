#!/usr/bin/env python3
"""Post-hoc descriptive distances of already saved features; no forward/fitting."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def write(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distances(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    difference = a - b
    denominator = float(np.linalg.norm(a))
    return dict(rmse=float(np.sqrt(np.mean(difference ** 2))),
                relative_l2=float(np.linalg.norm(difference) / max(denominator, 1e-12)),
                cosine=float(np.dot(a, b) / max(denominator * float(np.linalg.norm(b)), 1e-12)))


def main():
    values = []
    controls = []
    for ds in ['IEMOCAP', 'MELD']:
        cache = torch.load(ROOT / 'local_only/pilot' / ds / 'features.pt', map_location='cpu', weights_only=False)
        originals = {kind: torch.load(ROOT / 'local_only/features' / ds / (kind + '_features.pt'),
                                     map_location='cpu', weights_only=False)['validation']
                     for kind in ['text', 'audio']}
        for variant in cache['variants']:
            if variant['status'] != 'EXTRACTED':
                continue
            did = str(variant['dialogue_id'])
            index = int(variant['utterance_index'])
            kind = variant['modality']
            base = dict(dataset=ds, sample_id=variant['sample_id'], modality=kind,
                        perturbation_type=variant['perturbation_type'], variant_id=variant['variant_id'],
                        label_review_status=variant['label_review_status'])
            values.append(dict(**base, **distances(originals[kind][did][index], variant['feature'])))
        for sid, modalities in cache['clean_controls'].items():
            for kind, control in modalities.items():
                if control['status'] != 'EXTRACTED':
                    continue
                controls.append(dict(dataset=ds, sample_id=sid, modality=kind,
                    **distances(originals[kind][str(control['dialogue_id'])][int(control['utterance_index'])], control['feature'])))
    write(ROOT / 'results/analysis/feature_response_per_variant.csv', values)
    write(ROOT / 'results/analysis/feature_clean_repeat_distances.csv', controls)
    summary = []
    for ds, kind in sorted({(r['dataset'], r['perturbation_type']) for r in values}):
        selected = [r for r in values if r['dataset'] == ds and r['perturbation_type'] == kind]
        row = dict(dataset=ds, perturbation_type=kind, extracted_variants=len(selected),
                   scope='POST_HOC_DESCRIPTIVE_ALL_EXTRACTED_INPUT_QC_PASS_NOT_LABEL_CONFIRMED')
        for metric in ['rmse', 'relative_l2', 'cosine']:
            data = [r[metric] for r in selected]
            row[metric + '_median'] = float(np.median(data))
            row[metric + '_min'] = float(np.min(data))
            row[metric + '_max'] = float(np.max(data))
        summary.append(row)
    write(ROOT / 'results/analysis/feature_response_summary.csv', summary)
    (ROOT / 'provenance/feature_response_description.json').write_text(json.dumps(dict(
        status='COMPLETE', no_forward_or_fitting=True, test_indexed=False,
        official_validation_saved_features_only=True, post_hoc_descriptive=True,
        note='Does not distinguish encoder invariance from downstream modality importance causally; no parameter or condition selection based on these distances.',
        variants=len(values), clean_controls=len(controls),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
