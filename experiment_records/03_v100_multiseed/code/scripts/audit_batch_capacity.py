#!/usr/bin/env python3
"""Audit actual dialogue lengths for every pre-registered formal batch plan.

This is a CPU sampling/protocol audit, not model training or measured scores.
Only read the trusted official feature and difficulty files in this workspace.
"""
import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dataloader import resolve_meld_split, select_curriculum_keys, split_iemocap_dialogues
from train_cm import CapacitySafeBatchSampler, file_sha256


def load_inputs(dataset, split_seed):
    if dataset == 'IEMOCAP':
        feature_paths = ['data/IEMOCAP_features.pkl', 'data/iemocap_features_roberta.pkl']
        with (REPO / feature_paths[0]).open('rb') as handle:
            feature = pickle.load(handle, encoding='latin1')
        labels = feature[2]
        train_ids, validation_ids = split_iemocap_dialogues(feature[7], split_seed)
        test_ids = list(feature[8])
        epochs, ratio = 80, .15
    else:
        feature_paths = ['data/MELD_features_raw1.pkl', 'data/meld_features_roberta.pkl']
        with (REPO / feature_paths[0]).open('rb') as handle:
            labels = pickle.load(handle)[2]
        with (REPO / feature_paths[1]).open('rb') as handle:
            roberta = pickle.load(handle, encoding='latin1')
        train_ids, test_ids, validation_ids = [list(ids) for ids in roberta[8:]]
        del roberta
        train_ids = resolve_meld_split(train_ids, validation_ids, test_ids, 'train')
        epochs, ratio = 40, .4
    split = {'train_ids': train_ids, 'validation_ids': validation_ids, 'test_ids': test_ids}
    for name, ids in split.items():
        if len(ids) != len(set(ids)):
            raise ValueError('{} {} has duplicate dialogue IDs'.format(dataset, name))
    if (set(train_ids) & set(validation_ids) or set(train_ids) & set(test_ids)
            or set(validation_ids) & set(test_ids)):
        raise ValueError('{} split contains overlapping dialogue IDs'.format(dataset))
    split['hash'] = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    difficulty_path = 'caldiff/{}_diff.pkl'.format(dataset)
    with (REPO / difficulty_path).open('rb') as handle:
        difficulty = pickle.load(handle)
    data_hashes = {path: file_sha256(REPO / path) for path in feature_paths + [difficulty_path]}
    return labels, split, difficulty, epochs, ratio, data_hashes


def batch_counts(lengths, batches):
    sums = [sum(lengths[index] for index in batch) for batch in batches]
    return (max((total + 3 * len(batch) for total, batch in zip(sums, batches)), default=0),
            max((3 * total for total in sums), default=0))


def fixed_order_check(labels, ids, options):
    lengths = [len(labels[key]) for key in ids]
    batches = [list(range(start, min(start + options.batch_size, len(ids))))
               for start in range(0, len(ids), options.batch_size)]
    edges, nodes = batch_counts(lengths, batches)
    if edges > options.edge_capacity or nodes > options.node_capacity:
        raise ValueError('fixed validation/test order exceeds official graph capacity')
    return {'dialogues': len(ids), 'batches': len(batches), 'max_batch_edges': edges,
            'max_batch_nodes': nodes, 'fixed_source_order': True, 'status': 'passed'}


def scan_run(labels, train_ids, difficulty, epochs, ratio_step, seed, variant, options):
    records = []
    for epoch in range(epochs):
        ratio = -1 if variant == 'baseline' else min(1., round(ratio_step * (epoch + 1), 2))
        ids = train_ids if ratio == -1 else select_curriculum_keys(train_ids, difficulty, ratio)
        lengths = [len(labels[key]) for key in ids]
        generator = torch.Generator().manual_seed(seed + epoch)
        # DataLoader's _BaseDataLoaderIter consumes a worker/base seed first.
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        first_generator = torch.Generator()
        first_generator.set_state(generator.get_state())
        first = torch.randperm(len(ids), generator=first_generator).tolist()
        first_batches = [first[start:start + options.batch_size]
                         for start in range(0, len(first), options.batch_size)]
        first_edges, first_nodes = batch_counts(lengths, first_batches)
        sampler = CapacitySafeBatchSampler(lengths, options.batch_size, generator,
                                           options.edge_capacity, options.node_capacity,
                                           options.batch_max_retries)
        batches = list(sampler)
        emitted = sum(batches, [])
        if sorted(emitted) != list(range(len(ids))):
            raise ValueError('sampler duplicated, omitted, or replaced eligible dialogue IDs')
        if any(len(batch) != options.batch_size for batch in batches[:-1]):
            raise ValueError('sampler changed a full batch size')
        if not batches or not 0 < len(batches[-1]) <= options.batch_size:
            raise ValueError('sampler changed the final partial batch rule')
        emitted_ids = [ids[index] for index in emitted]
        records.append({
            'epoch': epoch + 1, 'sample_ratio': 1. if ratio == -1 else ratio,
            'eligible_dialogues': len(ids), 'batches': len(batches),
            'sampler_retries': sampler.retries, 'max_batch_edges': sampler.max_batch_edges,
            'max_batch_nodes': sampler.max_batch_nodes, 'each_eligible_id_once': True,
            'first_candidate_max_edges': first_edges, 'first_candidate_max_nodes': first_nodes,
            'permutation_sha256': hashlib.sha256(json.dumps(emitted_ids).encode()).hexdigest(),
        })
    return {'seed': seed, 'variant': variant, 'epochs': epochs, 'status': 'passed',
            'retry_histogram': dict(sorted(Counter(str(item['sampler_retries']) for item in records).items())),
            'max_retries': max(item['sampler_retries'] for item in records),
            'max_batch_edges': max(item['max_batch_edges'] for item in records),
            'max_batch_nodes': max(item['max_batch_nodes'] for item in records),
            'epochs_requiring_rejection': sum(item['sampler_retries'] > 0 for item in records),
            'all_eligible_ids_once_every_epoch': True, 'epoch_records': records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1475, 67137, 2024, 3407, 42])
    parser.add_argument('--split-seed', type=int, default=1475)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--edge-capacity', type=int, default=1000)
    parser.add_argument('--node-capacity', type=int, default=5200)
    parser.add_argument('--batch-max-retries', type=int, default=10000)
    options = parser.parse_args(argv)
    if options.output.exists():
        raise FileExistsError('refusing to overwrite capacity audit {}'.format(options.output))
    if len(set(options.seeds)) != len(options.seeds):
        parser.error('seeds must be unique')
    if (options.batch_size <= 0 or not 0 < options.edge_capacity <= 1000
            or not 0 < options.node_capacity <= 5200 or options.batch_max_retries < 0):
        parser.error('capacity settings must fit the official 1000 hyperedges / 5200 nodes')
    options.output.parent.mkdir(parents=True, exist_ok=True)
    before_rng = torch.get_rng_state().clone()
    payload = {'status': 'passed', 'kind': 'real_dialogue_length_sampling_audit_not_training',
               'torch_version': torch.__version__, 'configuration': {
                   key: value for key, value in vars(options).items() if key != 'output'},
               'batch_policy': 'capacity_safe', 'selection_protocol': 'validation',
               'rng_policy': 'continuous', 'causal': False, 'datasets': {}}
    for dataset in ('IEMOCAP', 'MELD'):
        labels, split, difficulty, epochs, ratio, data_hashes = load_inputs(dataset, options.split_seed)
        runs = [scan_run(labels, split['train_ids'], difficulty, epochs, ratio, seed, variant, options)
                for seed in options.seeds for variant in ('baseline', 'cmerc')]
        payload['datasets'][dataset] = {
            'split': split, 'split_disjoint': True, 'data_sha256': data_hashes,
            'fixed_validation': fixed_order_check(labels, split['validation_ids'], options),
            'fixed_test': fixed_order_check(labels, split['test_ids'], options), 'runs': runs,
        }
        del labels, difficulty
    if not torch.equal(torch.get_rng_state(), before_rng):
        raise ValueError('capacity audit unexpectedly advanced the global Torch RNG')
    origin_path = REPO / 'source_origin.json'
    payload['source'] = {
        'base_commit': json.loads(origin_path.read_text()).get('base_commit') if origin_path.exists() else None,
        'files': {str(path.relative_to(REPO)): file_sha256(path) for path in sorted(REPO.glob('*.py'))},
        'audit_script_sha256': file_sha256(Path(__file__)),
    }
    payload['total_epoch_permutations'] = sum(
        run['epochs'] for entry in payload['datasets'].values() for run in entry['runs'])
    with options.output.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({'status': payload['status'], 'epoch_permutations': payload['total_epoch_permutations'],
                      'output': str(options.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
