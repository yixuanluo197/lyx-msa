#!/usr/bin/env python3
"""Fail-closed verification of completed, trusted-local round-two run artifacts.

This never trains or performs inference. Reports contain hashes and aggregates,
not dialogue IDs, labels, predictions, checkpoints, or absolute server paths.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from causal_provenance import architecture_from_args, resolve_causal_difficulty
from causal_run import run_config_digest, scientific_config, sha256
from train_cm import model_state_sha256


LOSSES = ('classification_loss', 'cc_loss', 'mscl_loss', 'cscl_loss', 'total_loss')
PROTOCOL = dict(Dataset='IEMOCAP', causal=True, selection_protocol='validation',
    seed=1937, split_seed=1475, base_model='GRU', batch_size=16, lr=.0001,
    dropout=.5, num_L=3, num_K=4, norm='LN', graph_type='hyper',
    mm_fusion_mthd='concat_DHT', modals='avl', multi_modal=True,
    graph_construct='direct', use_residue=False, av_using_lstm=False,
    use_speaker=True, use_modal=False, l2=.00003, fail_on_nonfinite=True,
    defer_final_test=True,
    auxiliary_policy='legacy_epoch_snapshot', replacement_policy='prefix_mean_v1')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_jsonl(path):
    require(path.is_file(), 'missing artifact: ' + path.name)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def state_hash(state):
    return model_state_sha256(SimpleNamespace(state_dict=lambda: state))


def verify_run(run_dir, expected_provenance, variant, epochs=80, paired_run=None, difficulty=None):
    """Verify a completed run; raise ValueError for incomplete/inconsistent evidence."""
    require(variant in ('baseline', 'cmerc'), 'unknown experiment variant')
    require(isinstance(epochs, int) and epochs > 0, 'epoch count must be positive')
    run = Path(run_dir)
    expected = json.loads(Path(expected_provenance).read_text())
    expected = expected.get('data_provenance', expected)
    require(not (run / 'final_test.json').exists(), 'test results exposed before validation analysis freeze')
    records = read_jsonl(run / 'metrics.jsonl')
    require([r.get('epoch') for r in records] == list(range(1, epochs + 1)),
            'metrics must contain contiguous completed epochs exactly once')
    for record in records:
        require(record.get('selection_protocol') == 'validation' and record.get('causal') is True
                and record.get('selection_split') == 'validation', 'metrics protocol mismatch')
        for key in LOSSES + ('validation_accuracy', 'validation_weighted_f1',
                'selection_weighted_f1', 'train_accuracy', 'train_weighted_f1',
                'elapsed_seconds', 'max_gpu_memory_mb'):
            require(finite(record.get(key)), 'nonfinite or missing epoch metric: ' + key)
        require(record['selection_weighted_f1'] == record['validation_weighted_f1'],
                'selection metric differs from validation W-F1')
        for key in ('test_accuracy', 'test_weighted_f1'):
            require(record.get(key) is None or
                    isinstance(record.get(key), float) and math.isnan(record[key]),
                    'test metrics exposed before frozen validation analysis')
    best_index = max(range(epochs), key=lambda i: records[i]['validation_weighted_f1'])
    best_metric = records[best_index]['validation_weighted_f1']
    manifest_path = run / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256(manifest_path)
    checkpoints = {name: torch.load(run / (name + '.pt'), map_location='cpu')
                   for name in ('best', 'latest')}
    for name, checkpoint in checkpoints.items():
        saved = checkpoint.get('args', {})
        for key, value in {**PROTOCOL, 'epochs': epochs}.items():
            require(saved.get(key) == value, 'checkpoint protocol mismatch: ' + key)
        require(checkpoint.get('causal') is True, 'checkpoint temporal protocol is not causal')
        require(saved.get('data_provenance') == expected, 'checkpoint data provenance mismatch')
        require(all(saved.get(key) is (variant == 'cmerc') for key in
                    ('calibrate', 'contrastlearning', 'courselearning')), 'component flags mismatch')
        if variant == 'cmerc':
            for key, value in dict(rank_coff=.005, mscl_coff=.05, cscl_coff=.05,
                                   epoch_ratio=.15, scheduler_steps=1).items():
                require(saved.get(key) == value, 'full CMERC coefficient mismatch: ' + key)
        require(checkpoint.get('epoch') == (best_index if name == 'best' else epochs - 1),
                'checkpoint completed epoch disagrees with metrics/best selection')
        require(checkpoint.get('best_epoch') == best_index and checkpoint.get('best_metric') == best_metric,
                'checkpoint best epoch/metric disagrees with earliest maximum validation W-F1')
        require(checkpoint.get('run_manifest_sha256') == manifest_sha, 'checkpoint run manifest SHA256 mismatch')
        require(manifest.get('config_sha256') == run_config_digest(saved), 'scientific configuration hash mismatch')
        require(manifest.get('config') == scientific_config(saved), 'manifest scientific configuration differs from checkpoint')
        require(checkpoint.get('optimizer_state_dict', {}).get('state')
                and checkpoint['optimizer_state_dict'].get('param_groups'), 'missing populated optimizer state')
        require(all(key in checkpoint.get('rng_state', {}) for key in ('python', 'numpy', 'torch')),
                'missing random state required for recovery')
        require('cuda' in checkpoint['rng_state'], 'missing CUDA random state for the formal GPU run')
        require(checkpoint.get('curriculum', {}).get('completed_epoch') == checkpoint['epoch'],
                'missing or incorrect curriculum recovery progress')
        next_ratio = (min(1., round(saved['epoch_ratio'] *
            int((checkpoint['epoch'] + 1) / saved['scheduler_steps'] + 1), 2))
            if saved.get('courselearning') and saved.get('scheduler_steps', 0) > 0 else -1)
        require(checkpoint['curriculum'].get('next_epoch_ratio') == next_ratio,
                'checkpoint next curriculum ratio differs from the saved protocol')
        require(all(torch.isfinite(value).all().item() for value in checkpoint['model_state_dict'].values()),
                'checkpoint contains nonfinite model parameters/buffers')
    best, latest = checkpoints['best'], checkpoints['latest']
    saved = latest['args']
    initial = json.loads((run / 'initial_state.json').read_text()).get('model_state_sha256')
    require(isinstance(initial, str) and len(initial) == 64 and
            all(checkpoint['args'].get('initial_model_sha256') == initial for checkpoint in checkpoints.values()),
            'initial parameter fingerprint missing or inconsistent')
    final_hash = state_hash(latest['model_state_dict'])
    require(initial != final_hash, 'trained model is unchanged from actual initialization')
    best_sha = sha256(run / 'best.pt')
    require(latest.get('best_checkpoint', {}).get('sha256') == best_sha,
            'latest checkpoint does not pin the available historical best')
    diagnostics = read_jsonl(run / 'training_diagnostics.jsonl')
    first = [record for record in diagnostics if record.get('batch') == 0]
    require([r.get('epoch') for r in first] == list(range(epochs)),
            'first-batch diagnostics missing/duplicated for completed epochs')
    coefficients = dict(classification_loss=1., cc_loss=saved.get('rank_coff', 0.) if variant == 'cmerc' else 0.,
        mscl_loss=saved.get('mscl_coff', 0.) if variant == 'cmerc' else 0.,
        cscl_loss=saved.get('cscl_coff', 0.) if variant == 'cmerc' else 0.)
    for epoch, record in enumerate(records):
        ratio = min(1., round(saved['epoch_ratio'] * int(epoch / saved['scheduler_steps'] + 1), 2)) \
            if variant == 'cmerc' else 1.
        require(record.get('sample_ratio') == ratio and
                record.get('train_dialogues') == int(ratio * expected['train_count']),
                'epoch curriculum sample ratio/dialogue count mismatch')
        reconstructed = sum(coefficients[key] * record[key] for key in coefficients)
        require(math.isclose(record['total_loss'], reconstructed, rel_tol=1e-6, abs_tol=1e-6),
                'weighted epoch losses do not reconstruct the optimized total')
    raw_norms, weighted_norms = {}, {}
    for diagnostic in diagnostics:
        require(diagnostic.get('finite_losses') is True and diagnostic.get('finite_gradients') is True,
                'nonfinite diagnostic loss/gradient')
        require(diagnostic.get('auxiliary_policy') == 'legacy_epoch_snapshot', 'diagnostic auxiliary policy mismatch')
        require(diagnostic.get('main_training') is True and diagnostic.get('aux_training') is False and
                diagnostic.get('aux_modes') == dict(mm=False, intra=False, inter=False),
                'diagnostic main/auxiliary training modes differ from legacy epoch snapshot')
        if diagnostic.get('batch') == 0:
            require(diagnostic.get('main_aux_max_abs_diff') == 0.,
                    'first-batch auxiliary snapshot differs from main weights')
        require(diagnostic.get('gradient_coefficients') == coefficients, 'gradient coefficient mismatch')
        require(all(finite(diagnostic.get('losses', {}).get(key)) for key in LOSSES),
                'diagnostic losses missing or nonfinite')
        require(math.isclose(diagnostic['losses']['total_loss'],
                sum(coefficients[key] * diagnostic['losses'][key] for key in coefficients),
                rel_tol=1e-6, abs_tol=1e-6), 'weighted diagnostic losses do not reconstruct total')
        raw = diagnostic.get('gradient_norms', {})
        weighted = diagnostic.get('weighted_gradient_norms', {})
        require(set(raw) == {key for key, coefficient in coefficients.items() if coefficient != 0.},
                'missing actual active loss gradient contribution')
        for key, value in raw.items():
            require(finite(value) and value >= 0 and finite(weighted.get(key)) and
                    math.isclose(weighted[key], value * abs(coefficients[key]), rel_tol=1e-7, abs_tol=1e-10),
                    'invalid actual weighted gradient contribution')
            raw_norms.setdefault(key, []).append(value)
            weighted_norms.setdefault(key, []).append(weighted[key])
        require(raw['classification_loss'] > 0, 'classification gradient is zero')
        updates = diagnostic.get('parameter_update_norms', {})
        require(updates and all(finite(value) and value >= 0 for value in updates.values())
                and any(value > 0 for value in updates.values()), 'no finite actual parameter update')
    result = dict(status='passed', variant=variant, completed_epochs=epochs, best_epoch=best_index + 1,
        best_validation_weighted_f1=best_metric, best_sha256=best_sha,
        latest_sha256=sha256(run / 'latest.pt'), run_manifest_sha256=manifest_sha,
        config_sha256=manifest['config_sha256'], initial_model_sha256=initial,
        final_model_state_sha256=final_hash, data_provenance=expected,
        total_elapsed_seconds=sum(r['elapsed_seconds'] for r in records),
        peak_gpu_memory_mb=max(r['max_gpu_memory_mb'] for r in records),
        gradient_coefficients=coefficients,
        raw_loss_epoch_means={key: sum(r[key] for r in records) / epochs for key in LOSSES},
        weighted_loss_epoch_means={key: coefficient * sum(r[key] for r in records) / epochs
                                  for key, coefficient in coefficients.items()},
        test_evaluation_performed=False, paired_initialization_verified=False,
        diagnostics=dict(epochs_verified=len(first), batches_verified=len(diagnostics),
            raw_gradient_norm_ranges={key: dict(min=min(values), max=max(values)) for key, values in raw_norms.items()},
            weighted_gradient_norm_ranges={key: dict(min=min(values), max=max(values)) for key, values in weighted_norms.items()}))
    if paired_run:
        paired = verify_run(paired_run, expected_provenance, 'baseline', epochs=epochs)
        paired_checkpoint = torch.load(Path(paired_run) / 'best.pt', map_location='cpu')
        require(initial == paired['initial_model_sha256'], 'paired initial model weights differ')
        require(architecture_from_args(saved) == architecture_from_args(paired_checkpoint['args']),
                'paired backbone architectures differ')
        result.update(paired_initialization_verified=True, paired_best_sha256=paired['best_sha256'])
    if variant == 'cmerc':
        require(paired_run is not None, 'full CMERC verification requires the paired baseline run')
        require(difficulty is not None, 'full CMERC verification requires an explicit difficulty artifact')
        pinned = (saved.get('difficulty_artifact') or {}).get('sha256')
        require(pinned and all(checkpoint.get('difficulty_sha256') == pinned for checkpoint in checkpoints.values()),
                'checkpoint difficulty SHA256 mismatch or missing')
        _, artifact = resolve_causal_difficulty(difficulty, 'IEMOCAP', expected,
            architecture_from_args(saved), expected_sha256=pinned, expected_teacher_seed=1937)
        teacher = artifact['manifest']['source_checkpoint']
        require(teacher['epoch'] == teacher['best_epoch'] and teacher['sha256'] == result['paired_best_sha256'],
                'difficulty teacher must be the exact paired validation-selected baseline best')
        result['difficulty'] = dict(sha256=artifact['sha256'], teacher_best_sha256=teacher['sha256'],
                                    replacement_policy=artifact['manifest']['replacement_policy'])
    else:
        require(all(checkpoint.get('difficulty_sha256') is None and not checkpoint['args'].get('difficulty_artifact')
                    for checkpoint in checkpoints.values()), 'baseline unexpectedly bound to a difficulty artifact')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--expected-provenance', type=Path, required=True)
    parser.add_argument('--variant', choices=('baseline', 'cmerc'), required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--paired-run', type=Path)
    parser.add_argument('--difficulty', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify_run(args.run_dir, args.expected_provenance, args.variant, args.epochs,
                        args.paired_run, args.difficulty)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
