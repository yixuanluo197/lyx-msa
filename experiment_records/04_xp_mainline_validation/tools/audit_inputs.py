#!/usr/bin/env python3
"""CPU-only audit of the frozen batch's trusted feature/difficulty inputs."""
import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

BATCH = Path(__file__).resolve().parents[1]
CODE = BATCH / 'code'
ARCHIVE = CODE.parents[3] / 'experiment_records/03_v100_multiseed'
sys.path.insert(0, str(CODE))
from dataloader import resolve_meld_split, split_iemocap_dialogues


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def load(path):
    with path.open('rb') as handle:
        return pickle.load(handle, encoding='latin1')


def numeric_feature(values, labels, width, name):
    require(set(values) == set(labels), name + ': dialogue key mismatch')
    dtypes = Counter()
    count = 0
    for key, label in labels.items():
        array = np.asarray(values[key])
        require(array.shape == (len(label), width), name + ': shape mismatch at ' + str(key))
        require(np.issubdtype(array.dtype, np.number), name + ': nonnumeric feature')
        require(bool(np.isfinite(array).all()), name + ': nonfinite feature at ' + str(key))
        dtypes[str(array.dtype)] += 1
        count += array.size
    return {'dialogues': len(values), 'width': width, 'finite_values_checked': count,
            'dtypes_by_dialogue': dict(dtypes), 'utterance_length_matches_labels': True}


def audit_dataset(dataset, manifests, hashes):
    iemocap = dataset == 'IEMOCAP'
    raw_path = CODE / 'data' / ('IEMOCAP_features.pkl' if iemocap else 'MELD_features_raw1.pkl')
    roberta_path = CODE / 'data' / ('iemocap_features_roberta.pkl' if iemocap else 'meld_features_roberta.pkl')
    raw, roberta = load(raw_path), load(roberta_path)
    labels = raw[2]
    keys = set(labels)
    require(len(raw) == (9 if iemocap else 10), 'unexpected raw schema')
    require(len(roberta) == (10 if iemocap else 11), 'unexpected roberta schema')
    label_count, lengths = Counter(), []
    all_utterance_ids = []
    for key, values in labels.items():
        array = np.asarray(values)
        require(array.ndim == 1 and len(array) > 0, 'empty or invalid labels')
        require(np.issubdtype(array.dtype, np.integer), 'noninteger labels')
        require(bool(((array >= 0) & (array < (6 if iemocap else 7))).all()), 'label out of range')
        label_count.update(array.tolist())
        lengths.append(len(array))
        for index in (0, 1, 6):
            require(key in raw[index] and len(raw[index][key]) == len(array), 'raw sequence alignment failure')
        ids = list(raw[0][key])
        require(len(ids) == len(set(ids)), 'duplicate utterance ID within dialogue')
        all_utterance_ids.extend(ids)
        if iemocap:
            require(all(str(item).startswith(str(key) + '_') for item in ids), 'utterance/dialogue ID mismatch')
            require(set(raw[1][key]) <= {'M', 'F'}, 'invalid IEMOCAP speaker')
        else:
            speakers = np.asarray(raw[1][key])
            require(speakers.shape == (len(array), 9), 'invalid MELD speaker shape')
            require(bool(((speakers == 0) | (speakers == 1)).all()) and bool((speakers.sum(1) == 1).all()),
                    'invalid MELD one-hot speakers')
        for raw_index, roberta_index in ((1, 0), (2, 1), (6, 6 if iemocap else 7)):
            require(key in roberta[roberta_index], 'missing roberta metadata dialogue')
            require(np.array_equal(np.asarray(raw[raw_index][key]), np.asarray(roberta[roberta_index][key])),
                    'raw/roberta metadata ordering mismatch at ' + str(key))
    for index in (0, 1, 6):
        require(set(raw[index]) == keys, 'raw metadata key mismatch')
    for index in (0, 1, 6 if iemocap else 7):
        require(set(roberta[index]) == keys, 'roberta metadata key mismatch')
    if iemocap:
        require(len(all_utterance_ids) == len(set(all_utterance_ids)), 'global IEMOCAP utterance ID duplicate')
        train_ids, validation_ids = split_iemocap_dialogues(raw[7], 1475)
        test_ids = list(raw[8])
    else:
        train_ids = resolve_meld_split(roberta[8], roberta[10], roberta[9], 'train')
        validation_ids = resolve_meld_split(roberta[8], roberta[10], roberta[9], 'validation')
        test_ids = resolve_meld_split(roberta[8], roberta[10], roberta[9], 'test')
    split = {'train_ids': train_ids, 'validation_ids': validation_ids, 'test_ids': test_ids}
    for name, ids in split.items():
        require(len(ids) == len(set(ids)), 'duplicate split dialogue ID: ' + name)
    sets = [set(ids) for ids in split.values()]
    require(not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), 'overlapping splits')
    require(set.union(*sets) == keys, 'split union does not cover every dialogue')
    split['hash'] = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    relevant = [(path, data) for path, data in manifests if data['args']['Dataset'] == dataset]
    require(len(relevant) == 10, 'expected ten V2 run manifests per dataset')
    for path, data in relevant:
        require(data['split'] == split, 'split mismatch versus ' + str(path))
        require(data['args']['split_seed'] == 1475, 'V2 split seed mismatch')
    features = {}
    for index, name, width in ((3, 'legacy_text', 100 if iemocap else 600),
                                (4, 'audio', 1582 if iemocap else 300), (5, 'visual', 342)):
        features[name] = numeric_feature(raw[index], labels, width, dataset + '/' + name)
    for number, index in enumerate(range(2 if iemocap else 3, 6 if iemocap else 7), 1):
        name = 'roberta' + str(number)
        features[name] = numeric_feature(roberta[index], labels, 1024, dataset + '/' + name)
    difficulty = load(CODE / 'caldiff' / (dataset + '_diff.pkl'))
    require(isinstance(difficulty, dict) and all(key in difficulty for key in train_ids), 'missing training difficulty')
    require(all(np.asarray(value).shape == () and np.isfinite(value) for value in difficulty.values()),
            'nonfinite or nonscalar difficulty')
    require(set(difficulty) <= keys, 'difficulty contains unknown dialogue IDs')
    result = {
        'status': 'passed', 'dialogues': len(labels), 'utterances': sum(lengths),
        'dialogue_length_min': min(lengths), 'dialogue_length_max': max(lengths),
        'label_histogram': dict(sorted(label_count.items())), 'features': features,
        'dialogue_keys_match_all_features_and_metadata': True,
        'raw_roberta_labels_speakers_sentences_identical_in_order': True,
        'utterance_ids_unique_within_dialogue': True,
        'utterance_ids_globally_unique_required': iemocap,
        'feature_semantic_alignment_limit': 'Feature rows are structurally aligned to matching labels, speakers and sentences; semantic correctness of upstream feature extraction is not re-derived.',
        'split': split, 'split_disjoint': True, 'split_union_all_dialogues': True,
        'split_seed': 1475, 'same_split_for_all_five_seeds_and_both_variants': True,
        'v2_manifests_compared': [str(path) for path, _ in relevant],
        'split_counts': {name: {'dialogues': len(ids), 'utterances': sum(len(labels[key]) for key in ids)}
                         for name, ids in split.items() if name != 'hash'},
        'difficulty': {'source': 'fixed_author', 'entries': len(difficulty),
                       'coverage': {name: len(set(ids) & set(difficulty)) for name, ids in split.items() if name != 'hash'},
                       'all_training_ids_covered': True, 'generation_process_known': False,
                       'train_only_generation_claim_supported': False,
                       'limitation': 'Author fixed difficulty files are unchanged. Generation process and training-only information provenance are unknown; selection is restricted by V2 code to current training IDs.'},
        'data_sha256': {str(path.relative_to(CODE)): hashes[str(path.relative_to(CODE))]['sha256']
                        for path in (raw_path, roberta_path, CODE / 'caldiff' / (dataset + '_diff.pkl'))},
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BATCH / 'preflight/data_and_capacity/input_audit.json')
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden for this audit')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized')
    manifest_paths = sorted((ARCHIVE / 'runs/multiseed_validation_v2').glob('*/*/run_manifest.json'))
    require(len(manifest_paths) == 20, 'expected twenty V2 archive manifests')
    manifests = [(path, json.loads(path.read_text())) for path in manifest_paths]
    expected = {}
    for path, data in manifests:
        for name, sha in data['data'].items():
            require(name not in expected or expected[name] == sha, 'V2 data SHA conflict: ' + str(path))
            expected[name] = sha
    require(len(expected) == 6, 'expected four feature and two difficulty SHA entries')
    hashes = {}
    for name, sha in expected.items():
        path = CODE / name
        actual = digest(path)
        require(actual == sha, 'DATA CONFLICT: ' + name)
        hashes[name] = {'sha256': actual, 'matches_all_v2_manifests': True,
                        'resolved_path': str(path.resolve()), 'bytes': path.stat().st_size,
                        'is_symlink': path.is_symlink()}
    payload = {'status': 'passed', 'kind': 'cpu_input_and_split_preflight_not_training',
               'created_at_utc': datetime.now(timezone.utc).isoformat(), 'batch': str(BATCH),
               'python': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__,
               'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'cuda_initialized': False,
               'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
               'v2_archive': str(ARCHIVE), 'v2_manifests': len(manifests), 'inputs': hashes,
               'split_code_sha256': digest(CODE / 'dataloader.py'), 'audit_script_sha256': digest(Path(__file__)),
               'datasets': {dataset: audit_dataset(dataset, manifests, hashes) for dataset in ('IEMOCAP', 'MELD')}}
    require(not torch.cuda.is_initialized(), 'CUDA initialized during CPU audit')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write('\n')
    print(json.dumps({'status': 'passed', 'inputs': len(hashes), 'manifests': len(manifests),
                      'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
