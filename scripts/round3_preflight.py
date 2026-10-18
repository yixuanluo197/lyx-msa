#!/usr/bin/env python3
"""Inventory and bounded real train/validation checks; no formal results/test use."""
import argparse
import json
import os
import platform
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
from dataloader import IEMOCAPDataset, MELDDataset, split_iemocap_dialogues
from model_cm import Model, Model_Intra, Model_Inter, FocalLoss
from scripts.evaluate_checkpoint import dataset_config, _model_arguments


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def datasets(name):
    if name == 'IEMOCAP':
        full = IEMOCAPDataset(causal=True)
        train_ids, valid_ids = split_iemocap_dialogues(full.keys, 1475, .1)
        return IEMOCAPDataset(keys=train_ids, causal=True), IEMOCAPDataset(keys=valid_ids, causal=True)
    return (MELDDataset('data/MELD_features_raw1.pkl', split='train', causal=True),
            MELDDataset('data/MELD_features_raw1.pkl', split='validation', causal=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['IEMOCAP', 'MELD'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['inventory', 'baseline', 'cmerc'], required=True)
    parser.add_argument('--seed', type=int, default=1937)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / ('DATA_ENVIRONMENT.json' if args.mode == 'inventory'
                                else f'PREFLIGHT_{args.mode.upper()}_{args.seed}.json')
    if target.exists():
        raise FileExistsError('Preserve existing preflight evidence: ' + str(target))
    torch.set_num_threads(4)
    config = dataset_config(args.dataset)
    provenance = data_provenance_from_files(args.dataset, 1475)
    trainset, validset = datasets(args.dataset)
    assert set(trainset.keys).isdisjoint(validset.keys)
    checks = {}
    for split, dataset in [('train', trainset), ('validation', validset)]:
        lengths, labels, speakers = [], [], set()
        for index in range(len(dataset)):
            sample = dataset[index]
            length = len(sample[8])
            assert length > 0 and all(len(value) == length for value in sample[:-1])
            assert all(torch.isfinite(value).all() for value in sample[:-1])
            assert all(value.shape[1] == 1024 for value in sample[:4])
            assert sample[4].shape[1] == config.visual_dim and sample[5].shape[1] == config.audio_dim
            assert sample[6].shape == (length, 2 if args.dataset == 'IEMOCAP' else 9)
            assert (sample[6].sum(-1) == 1).all() and ((sample[6] == 0) | (sample[6] == 1)).all()
            assert (sample[7] == 1).all() and sample[8].min() >= 0 and sample[8].max() < config.num_classes
            lengths.append(length); labels.extend(sample[8].tolist()); speakers.update(sample[6].argmax(-1).tolist())
        checks[split] = dict(dialogues=len(dataset), utterances=sum(lengths), min_length=min(lengths),
                             max_length=max(lengths), speakers=sorted(speakers),
                             class_counts=np.bincount(labels, minlength=config.num_classes).tolist())
    if args.mode == 'inventory':
        write(target, dict(data_provenance=provenance, train_ids=trainset.keys, validation_ids=validset.keys,
            checks=checks, test_inference=False, order='stored feature row order; upstream extraction unreviewed',
            environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                pyg=torch_geometric.__version__, cuda_runtime=torch.version.cuda, hostname=platform.node(),
                CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))))
        print(json.dumps(dict(status='passed', dataset=args.dataset, checks=checks)))
        return
    assert torch.cuda.is_available(), 'Production GPU preflight requires CUDA'
    train_cm.seed_everything(args.seed)
    kwargs = _model_arguments(config, torch.device('cuda:0'), causal=True)
    models = tuple(kind(**kwargs).cuda() for kind in (Model, Model, Model_Intra, Model_Inter))
    full = args.mode == 'cmerc'
    coefficients = (.005, .05, .05) if args.dataset == 'IEMOCAP' else (.002, .15, .15)
    train_cm.args = SimpleNamespace(seed=args.seed, causal=True, norm='LN', multi_modal=True,
        mm_fusion_mthd='concat_DHT', modals='avl', calibrate=full, contrastlearning=full,
        rank_coff=coefficients[0] if full else 0., mscl_coff=coefficients[1] if full else 0.,
        cscl_coff=coefficients[2] if full else 0., auxiliary_policy='legacy_epoch_snapshot',
        tensorboard=False, fail_on_nonfinite=True)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(trainset, batch_size=16, shuffle=True, generator=generator,
                        collate_fn=trainset.collate_fn, num_workers=0)
    # Keep only two largest actual batches; do not retain an entire dataset copy.
    candidates = []
    for batch in loader:
        score = sum(int(length)**2 for length in batch[7].sum(-1).tolist())
        candidates.append((score, batch))
        candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = candidates[:2]
    batches = [item[1] for item in candidates]
    criterion = FocalLoss().cuda() if args.dataset == 'MELD' else torch.nn.NLLLoss(
        torch.tensor([1/.086747, 1/.144406, 1/.227883, 1/.160585, 1/.127711, 1/.252668], device='cuda:0'))
    optimizer = torch.optim.Adam(models[0].parameters(), lr=.0001, weight_decay=.00003)
    report = dict(status='running', formal_result=False, dataset=args.dataset, seed=args.seed,
        batch_size=16, dtype='float32', device='cuda:0',
        initial_model_sha256=train_cm.model_state_sha256(models[0]),
        selected_batch_ids_sha256=[hash_ids(batch[-1]) for batch in batches],
        graph_dimensions=kwargs, CL=False, CC=full, HSCL=full, test_inference=False)
    write(target, report)
    started = time.monotonic()
    try:
        torch.cuda.reset_peak_memory_stats()
        records = []
        result = train_cm.train_or_eval_graph_model(*models, criterion, batches, 0, True, 'avl',
            optimizer, True, dataset=args.dataset, return_components=True, diagnostic_callback=records.append)
        assert all(np.isfinite(value) for value in result[-1].values())
        assert all(item['finite_gradients'] and item['finite_losses'] for item in records)
        assert all(any(value > 0 for value in item['parameter_update_norms'].values()) for item in records)
        assert records[0]['main_aux_max_abs_diff'] == 0 and records[1]['main_aux_max_abs_diff'] > 0
        assert all(p.grad is None for auxiliary in models[1:] for p in auxiliary.parameters())
        # Exercise production evaluation on all validation dialogues, never test.
        validation = DataLoader(validset, batch_size=16, collate_fn=validset.collate_fn, num_workers=0)
        evaluated = train_cm.train_or_eval_graph_model(*models, criterion, validation, 0, True, 'avl',
                                                      dataset=args.dataset, return_components=True)
        assert all(np.isfinite(value) for value in evaluated[-1].values())
        report.update(status='passed', diagnostics=records, train_losses=result[-1], validation_losses=evaluated[-1])
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
        write(target, report)


if __name__ == '__main__':
    main()
