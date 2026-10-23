#!/usr/bin/env python3
"""CPU-only sampler regression; never import the training module or use CUDA."""
import argparse
import ast
import hashlib
import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sampling_surface(path):
    tree = ast.parse(path.read_bytes())
    names = {
        'CapacitySafeBatchSampler', 'SourceOrderBatchSampler',
        'validation_training_loader',
    }
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    scope = {'torch': torch, 'DataLoader': DataLoader, 'Sampler': Sampler}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope, {node.name: ast.dump(node, include_attributes=False) for node in nodes}


class DialogueIndices(Dataset):
    """Real dialogue lengths, deterministic integer items, ordinary DataLoader."""
    def __init__(self, ids, labels):
        self.keys = list(ids)
        self.videoLabels = labels

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return index

    @staticmethod
    def collate_fn(items):
        return list(items)


def make_loader(scope, ids, labels, seed, policy):
    generator = torch.Generator().manual_seed(seed)
    loader = scope['validation_training_loader'](
        DialogueIndices(ids, labels), 16, {'generator': generator}, policy,
        1000, 5200, 10000, 0, False,
    )
    return loader, generator


def capacity(ids, labels, batches):
    totals = [sum(len(labels[ids[index]]) for index in batch) for batch in batches]
    return {
        'max_batch_edges': max((n + 3 * len(b) for n, b in zip(totals, batches)), default=0),
        'max_batch_nodes': max((3 * n for n in totals), default=0),
        'max_padded_tokens': max((len(b) * max(len(labels[ids[i]]) for i in b)
                                 for b in batches), default=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'preflight/source_order_cpu.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('refusing to overwrite existing CPU evidence')
    old_path = ROOT / 'code_xp04/train_cm.py'
    new_path = ROOT / 'code_source_order/train_cm.py'
    old, old_ast = load_sampling_surface(old_path)
    new, new_ast = load_sampling_surface(new_path)
    assert old_ast['CapacitySafeBatchSampler'] == new_ast['CapacitySafeBatchSampler']
    new_bytes = new_path.read_bytes()
    assert new_bytes.count(b'\r\n') == new_bytes.count(b'\n')

    features_path = REPO / 'data/IEMOCAP_features.pkl'
    difficulty_path = REPO / 'caldiff/IEMOCAP_diff.pkl'
    with features_path.open('rb') as handle:
        features = pickle.load(handle, encoding='latin1')
    labels, source_ids = features[2], list(features[7])
    with difficulty_path.open('rb') as handle:
        difficulty = pickle.load(handle)
    records = []
    before_rng = torch.get_rng_state().clone()
    for variant in ('baseline', 'cmerc'):
        manifest_path = (REPO / 'experiment_records/04_xp_mainline_validation/runs'
                         / variant / 'IEMOCAP_validation_seed1475/run_manifest.json')
        manifest = json.loads(manifest_path.read_text())
        train_ids = manifest['split']['train_ids']
        validation_ids = manifest['split']['validation_ids']
        assert train_ids == [key for key in source_ids if key not in set(validation_ids)]
        assert manifest['args']['seed'] == manifest['args']['split_seed'] == 1475
        for epoch in range(80):
            ratio = 1.0 if variant == 'baseline' else min(1.0, round(.15 * (epoch + 1), 2))
            chosen = set(sorted(train_ids, key=lambda key: difficulty[key])[:int(ratio * len(train_ids))])
            ids = [key for key in train_ids if key in chosen]
            old_loader, old_rng = make_loader(old, ids, labels, 1475 + epoch, 'capacity_safe')
            regression_loader, regression_rng = make_loader(new, ids, labels, 1475 + epoch, 'capacity_safe')
            fixed_loader, fixed_rng = make_loader(new, ids, labels, 1475 + epoch, 'source_order')
            old_batches = list(old_loader)
            regression_batches = list(regression_loader)
            fixed_batches = list(fixed_loader)
            assert old_batches == regression_batches
            assert torch.equal(old_rng.get_state(), regression_rng.get_state())
            assert old_loader.batch_sampler.retries == regression_loader.batch_sampler.retries == 0
            assert torch.equal(old_rng.get_state(), fixed_rng.get_state())
            expected = [list(range(start, min(start + 16, len(ids))))
                        for start in range(0, len(ids), 16)]
            assert fixed_batches == expected
            assert sorted(sum(old_batches, [])) == sum(fixed_batches, []) == list(range(len(ids)))
            assert all(len(batch) == 16 for batch in fixed_batches[:-1])
            assert 0 < len(fixed_batches[-1]) <= 16
            fixed_capacity = capacity(ids, labels, fixed_batches)
            assert fixed_capacity['max_batch_edges'] <= 1000
            assert fixed_capacity['max_batch_nodes'] <= 5200
            assert fixed_loader.batch_sampler.retries == 0
            for field in ('max_batch_edges', 'max_batch_nodes'):
                assert getattr(fixed_loader.batch_sampler, field) == fixed_capacity[field]
                assert getattr(old_loader.batch_sampler, field) == getattr(regression_loader.batch_sampler, field)
            records.append({
                'variant': variant, 'epoch': epoch + 1, 'ratio': ratio,
                'dialogues': len(ids), 'batches': len(fixed_batches),
                'each_eligible_id_once': True, 'source_order_exact': True,
                'historical_capacity_safe_unchanged': True,
                'private_rng_end_equal': True, 'reference_retries': 0,
                'fixed_capacity': fixed_capacity,
                'reference_capacity': capacity(ids, labels, old_batches),
                'fixed_order_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                'reference_order_sha256': hashlib.sha256(json.dumps(
                    [ids[index] for index in sum(old_batches, [])]).encode()).hexdigest(),
            })

    # Feasible repartitioning exists, but the fixed first pair itself is unsafe.
    unsafe_rng = torch.Generator().manual_seed(1475)
    unsafe_state = unsafe_rng.get_state().clone()
    sampler = new['SourceOrderBatchSampler']([3, 3, 1, 1], 2, unsafe_rng,
                                            edge_capacity=10, node_capacity=100)
    try:
        next(iter(sampler))
    except RuntimeError as error:
        assert 'no batches were emitted' in str(error)
    else:
        raise AssertionError('unsafe fixed partition was emitted')
    assert torch.equal(unsafe_state, unsafe_rng.get_state())
    for lengths in ([], [1]):
        generator = torch.Generator().manual_seed(5)
        result = list(new['SourceOrderBatchSampler'](lengths, 16, generator))
        assert result == ([] if not lengths else [[0]])
    assert torch.equal(before_rng, torch.get_rng_state())
    assert not torch.cuda.is_initialized()

    evidence = {
        'status': 'passed', 'kind': 'cpu_real_length_sampler_regression_not_training',
        'torch_version': torch.__version__, 'seed': 1475, 'split_seed': 1475,
        'epochs_checked': len(records), 'global_cpu_rng_unchanged': True,
        'cuda_never_initialized': True, 'original_capacity_sampler_ast_unchanged': True,
        'capacity_safe_real_loader_regression_passed': True,
        'source_order_private_rng_matches_zero_retry_reference': True,
        'unsafe_fixed_partition_rejected_before_any_emit_or_rng_draw': True,
        'empty_and_single_item_cases_passed': True, 'new_source_crlf_preserved': True,
        'limitations': [
            'Same RNG policy does not imply identical model dropout draws after batch tensor shapes change.',
            'CPU capacity coverage does not establish GPU memory safety.',
            'Private RNG equality is asserted only for this registered zero-retry reference.',
        ],
        'sources': {str(path): sha256(path) for path in (
            old_path, new_path, features_path, difficulty_path, Path(__file__).resolve())},
        'records': records,
    }
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'status': 'passed', 'epochs_checked': len(records),
                      'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
