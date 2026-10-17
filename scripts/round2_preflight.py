#!/usr/bin/env python3
"""Finite real-training checks on train/validation data; never a formal result."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch_geometric
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_cm
from causal_provenance import data_provenance_from_files, hash_ids
from dataloader import IEMOCAPDataset, split_iemocap_dialogues
from model_cm import Model, Model_Intra, Model_Inter
from scripts.evaluate_checkpoint import dataset_config, _model_arguments


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--mode', choices=('inventory', 'baseline', 'cmerc'), required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    provenance = data_provenance_from_files('IEMOCAP', 1475)
    dataset = IEMOCAPDataset(causal=True)
    train_ids, valid_ids = split_iemocap_dialogues(dataset.keys, 1475, .1)
    checked = []
    for key in train_ids + valid_ids:
        tensors = dataset[dataset.keys.index(key)]
        n = len(tensors[-2])
        assert n > 0
        for value in tensors[:-1]:
            assert len(value) == n and torch.isfinite(value).all()
        assert all(t.shape[1] == 1024 for t in tensors[:4])
        assert tensors[4].shape[1] == 342 and tensors[5].shape[1] == 1582
        assert tensors[6].shape == (n, 2) and (tensors[6].sum(-1) == 1).all()
        assert (tensors[7] == 1).all() and tensors[8].min() >= 0 and tensors[8].max() < 6
        checked.append({'id': key, 'utterances': n, 'split': 'train' if key in train_ids else 'validation'})
    inventory = {'data_provenance': provenance, 'train_ids': train_ids, 'validation_ids': valid_ids,
                 'checks': checked, 'within_dialogue_order': 'stored feature row order; extraction process unreviewed',
                 'test_features_inspected': False, 'environment': {
                     'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
                     'pyg': torch_geometric.__version__, 'cuda_runtime': torch.version.cuda,
                     'hostname': platform.node(), 'platform': platform.platform(),
                     'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                     'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    if args.mode == 'inventory':
        dump(args.output_dir / 'DATA_ENVIRONMENT.json', inventory)
        print(json.dumps({'train_dialogues': len(train_ids), 'validation_dialogues': len(valid_ids),
                          'max_train_length': max(len(dataset.videoLabels[k]) for k in train_ids),
                          'feature_provenance': provenance}, indent=2))
        return
    assert torch.cuda.is_available(), 'GPU required for production-dimension preflight'
    device = torch.device('cuda:0')
    seed = 1937
    train_cm.seed_everything(seed)
    kwargs = _model_arguments(dataset_config('IEMOCAP'), device, causal=True)
    models = tuple(cls(**kwargs).to(device) for cls in (Model, Model, Model_Intra, Model_Inter))
    full = args.mode == 'cmerc'
    train_cm.args = SimpleNamespace(seed=seed, causal=True, norm='LN', multi_modal=True,
        mm_fusion_mthd='concat_DHT', modals='avl', calibrate=full, contrastlearning=full,
        rank_coff=.005 if full else 0., mscl_coff=.05 if full else 0., cscl_coff=.05 if full else 0.,
        auxiliary_policy='legacy_epoch_snapshot', tensorboard=False)
    initial_sha = fingerprint(models[0])
    # Inspect the actual fixed-seed first-epoch batches on CPU, then execute
    # the batch with the largest graph edge count. No invented worst cohort.
    trainset = IEMOCAPDataset(causal=True, keys=train_ids)
    generator = torch.Generator().manual_seed(seed)
    candidates = list(DataLoader(trainset, batch_size=16, shuffle=True, generator=generator,
                                collate_fn=trainset.collate_fn, num_workers=0))
    batch = max(candidates, key=lambda b: sum(len(dataset.videoLabels[k]) ** 2 for k in b[-1]))
    selected = batch[-1]
    loader = [batch]
    weights = torch.tensor([1/.086747, 1/.144406, 1/.227883, 1/.160585, 1/.127711, 1/.252668], device=device)
    optimizer = torch.optim.Adam(models[0].parameters(), lr=.0001, weight_decay=.00003)
    report = {'status': 'running', 'mode': args.mode, 'formal_result': False, 'seed': seed,
              'initial_state_sha256': initial_sha, 'data_provenance': provenance,
              'batch_size': 16, 'dialogue_lengths': [len(dataset.videoLabels[k]) for k in selected],
              'selected_train_ids_sha256': hash_ids(selected), 'device': str(device),
              'gpu': torch.cuda.get_device_name(0), 'dtype': 'float32',
              'CL': False, 'CC': full, 'HSCL': full,
              'note': 'largest edge-count actual first-epoch batch; full-loss preflight has no curriculum teacher'}
    path = args.output_dir / ('PREFLIGHT_' + args.mode.upper() + '.json')
    dump(path, report)
    started = time.monotonic()
    try:
        torch.cuda.reset_peak_memory_stats()
        records = []
        result = train_cm.train_or_eval_graph_model(*models, torch.nn.NLLLoss(weights), loader,
            0, True, 'avl', optimizer, True, return_components=True, diagnostic_callback=records.append)
        assert all(np.isfinite(v) for v in result[-1].values()), 'nonfinite loss'
        assert records and all(r['finite_gradients'] for r in records)
        assert any(v > 0 for v in records[0]['parameter_update_norms'].values()), 'no parameter updates'
        assert all(p.grad is None for m in models[1:] for p in m.parameters()), 'auxiliary optimized'
        report.update(status='passed', losses=result[-1], diagnostics=records)
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
                      peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                      peak_reserved_mb=torch.cuda.max_memory_reserved()/1024**2)
        dump(path, report)


if __name__ == '__main__':
    main()
