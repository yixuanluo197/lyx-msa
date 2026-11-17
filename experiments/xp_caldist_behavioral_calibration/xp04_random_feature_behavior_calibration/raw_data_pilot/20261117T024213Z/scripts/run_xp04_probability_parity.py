"""Gate-protected, validation-only frozen XP04 audio probability parity.

No CMERC import, checkpoint load, or feature pickle load occurs until a full
independent clean-feature gate is accepted. This adapter must be run only on
LOCAL_HOST after the lead researcher approves the feature gate. The historical
official model requires CUDA; its unconditional CUDA calls are not patched.

Example (after a successful feature gate, with a host GPU explicitly selected):
  CUDA_VISIBLE_DEVICES=<host-GPU> multimodalerc310/bin/python THIS_SCRIPT \
    --dataset IEMOCAP --rebuilt-validation-audio local_only/IEMOCAP/rebuilt_validation_audio.npy \
    --manifest local_only/IEMOCAP/manifest_validation.csv \
    --feature-gate provenance/feature_gate.json --device cuda:0

The feature gate schema is documented in xp04_probability_adapter_audit.json.
Test arrays/labels are never indexed, no labels are exported, and no training or
perturbation API is called. Legacy feature pickle containers include all splits;
official loaders deserialize the containers, then access validation keys only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

WORKSPACE = Path('/LOCAL/lyx-msa')
ROOT = Path(__file__).resolve().parents[1]
HISTORY = WORKSPACE / 'experiments/xp_caldist_behavioral_calibration'
HISTORICAL_AUDIT = HISTORY / 'evidence_sensitivity/20261108T095843Z/provenance/audit.json'
HISTORICAL_AUDIT_SHA256 = '01a4cd48c9f592ec237d735ec8572cd1f79a2a2de38a45c2ff354d6802f80409'
HELPER = HISTORY / '20261107T172302Z/scripts/run_pilot.py'
HELPER_SHA256 = 'de0b49182745b71efca7a59882fbbeb61e3c676b5ec86cf914caca4924665a80'
SPECS = {'IEMOCAP': {'seed': 1475, 'rows': 527, 'width': 1582, 'dialogues': 12},
         'MELD': {'seed': 67137, 'rows': 1109, 'width': 300, 'dialogues': 114}}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def feature_gate(dataset, rebuilt, manifest, gate_path):
    """Fail closed before importing torch or opening historical data."""
    require(dataset in SPECS, 'Unknown dataset')
    gate = read_json(gate_path)
    item = gate.get('datasets', {}).get(dataset, {})
    for field in ('passed', 'clean_feature_parity_passed', 'validation_independent',
                  'full_validation_coverage', 'fit_effective_train_only',
                  'train_dialogue_group_cv_only', 'model_locked_before_validation'):
        require(item.get(field) is True, f'Feature gate required: {dataset}.{field}=true')
    require(item.get('test_used') is False, 'Feature gate must explicitly exclude Test')
    require(item.get('validation_rows') == SPECS[dataset]['rows'], 'Incomplete validation gate')
    require(item.get('output_width') == SPECS[dataset]['width'], 'Incorrect audio width')
    require(item.get('rebuilt_audio_sha256') == sha(rebuilt), 'Rebuilt audio hash mismatch')
    require(item.get('manifest_sha256') == sha(manifest), 'Manifest hash mismatch')
    if dataset == 'MELD':
        for field in ('selected1611_parity_passed', 'context300_parity_passed',
                      'full_dialogue_bigru_context', 'real_lengths_and_order_preserved',
                      'padding33_and_masking_preserved'):
            require(item.get(field) is True, f'MELD feature gate required: {field}=true')
    return item


def manifest_rows(path, dataset):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == SPECS[dataset]['rows'], 'Manifest must cover complete fixed validation')
    seen = set()
    for index, row in enumerate(rows):
        require(row['dataset'] == dataset and row['split'] == 'validation', 'Validation rows only')
        require(int(row['row_index']) == index, 'Manifest row_index must be contiguous')
        key = (row['dialogue_id'], int(row['utterance_index']))
        require(key not in seen, 'Duplicate manifest identity')
        seen.add(key)
    return rows


def frozen_inputs(dataset):
    require(sha(HISTORICAL_AUDIT) == HISTORICAL_AUDIT_SHA256, 'Historical audit changed')
    require(sha(HELPER) == HELPER_SHA256, 'Historical inference helper changed')
    audit = read_json(HISTORICAL_AUDIT)
    require(audit['status'] == 'passed', 'Historical audit did not pass')
    item = audit['datasets'][dataset]
    expected = dict(audit['frozen_source_sha256'])
    expected.update(item['data_sha256'])
    for kind in ('checkpoint', 'effective_config', 'validation_baseline'):
        expected[item[kind + '_path']] = item[kind + '_sha256']
    for path, digest in expected.items():
        require(sha(path) == digest, f'Frozen source/input changed: {path}')
    return item, expected


def import_helper():
    spec = importlib.util.spec_from_file_location('xp04_validation_helper', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_manifest_order(rows, keys, baseline, np):
    expected_ids = []
    for row in rows:
        if not expected_ids or expected_ids[-1] != row['dialogue_id']:
            expected_ids.append(row['dialogue_id'])
    require(expected_ids == [str(key) for key in keys], 'Manifest must preserve effective validation dialogue order')
    require(len(expected_ids) == len(set(expected_ids)), 'Interleaved dialogue rows are not permitted')
    for name, values in (
        ('dialogue_ids', [row['dialogue_id'] for row in rows]),
        ('utterance_indices', [int(row['utterance_index']) for row in rows]),
        ('speaker_ids', [int(row['speaker_id']) for row in rows]),
    ):
        require(np.array_equal(np.asarray(values), baseline[name]), f'Historical validation ordering differs: {name}')


def metric_summary(original, rebuilt, np):
    require(original.shape == rebuilt.shape and original.ndim == 2, 'Probability shapes differ')
    require(np.isfinite(original).all() and np.isfinite(rebuilt).all(), 'Nonfinite probability')
    for array in (original, rebuilt):
        require(np.allclose(array.sum(-1), 1, atol=1e-6, rtol=1e-5), 'Invalid softmax sums')
    error = np.abs(original - rebuilt)
    confidence_error = np.abs(original.max(-1) - rebuilt.max(-1))
    metrics = dict(rows=len(original), classes=original.shape[1],
                   max_probability_absolute_error=float(error.max()),
                   mean_probability_absolute_error=float(error.mean()),
                   predicted_class_agreement=float(np.mean(original.argmax(-1) == rebuilt.argmax(-1))),
                   max_confidence_absolute_error=float(confidence_error.max()),
                   mean_confidence_absolute_error=float(confidence_error.mean()))
    metrics['passed'] = bool(metrics['max_probability_absolute_error'] <= 1e-4
                             and metrics['mean_probability_absolute_error'] <= 1e-5
                             and metrics['predicted_class_agreement'] == 1
                             and metrics['max_confidence_absolute_error'] <= 1e-4)
    return metrics


def run(args):
    dataset = args.dataset
    # All checks here precede CMERC import, checkpoint deserialization, and data load.
    accepted_gate = feature_gate(dataset, args.rebuilt_validation_audio, args.manifest, args.feature_gate)
    rows = manifest_rows(args.manifest, dataset)
    reconstruction_hashes = {str(path): sha(path) for path in
                             (args.rebuilt_validation_audio, args.manifest, args.feature_gate)}
    require(platform.node().split('.')[0] == 'LOCAL_HOST', 'XP04 ownership is LOCAL_HOST; cross-server execution prohibited')
    require(args.device == 'cuda:0', 'Unmodified baseline requires explicitly selected CUDA device')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(visible.strip() and ',' not in visible and visible.strip() != '-1', 'Select one physical GPU explicitly via CUDA_VISIBLE_DEVICES')
    item, expected_hashes = frozen_inputs(dataset)
    destination = ROOT / 'local_only' / dataset / 'probability_parity'
    require(not destination.exists(), 'Refusing to overwrite probability parity run')
    import numpy as np
    rebuilt_audio = np.load(args.rebuilt_validation_audio, allow_pickle=False)
    require(rebuilt_audio.shape == (SPECS[dataset]['rows'], SPECS[dataset]['width']), 'Rebuilt audio shape differs')
    require(np.isfinite(rebuilt_audio).all(), 'Nonfinite reconstructed audio')
    helper = import_helper()
    api = helper.imports()
    torch, _, dl, _, _ = api
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one selected healthy GPU is required')
    torch.cuda.set_device(0)
    seed = SPECS[dataset]['seed']
    config = read_json(item['effective_config_path'])
    keys = config['provenance']['split']['validation_ids']
    require(len(keys) == SPECS[dataset]['dialogues'], 'Effective validation dialogue count differs')
    # Do not call helper.dataset_for: its config lookup is hard-coded to seed1475.
    os.chdir(helper.CODE)
    selected = (dl.IEMOCAPDataset(keys=keys) if dataset == 'IEMOCAP'
                else dl.MELDDataset('data/MELD_features_raw1.pkl', keys=keys))
    require(list(selected.keys) == keys, 'Official loader changed validation IDs/order')
    with np.load(item['validation_baseline_path'], allow_pickle=False) as archive:
        # Never request labels or any historical Test record/array.
        baseline = {key: archive[key] for key in ('dialogue_ids', 'utterance_indices', 'speaker_ids', 'probabilities')}
    validate_manifest_order(rows, keys, baseline, np)
    model = helper.load_model(dataset, seed, api)
    require(not model.training and all(not p.requires_grad for p in model.parameters()), 'Model not frozen')
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    guard = helper.GraphGuard(model, torch)
    original_parts, rebuilt_parts = [], []
    offset = 0
    started = time.monotonic()
    try:
        with torch.no_grad():
            for batch in helper.loader_for(selected, torch):
                tensors = [value.cuda() for value in batch[:-1]]
                unchanged = [value.clone() for value in tensors]
                lengths = [int(value.sum().item()) for value in tensors[7]]
                count = sum(lengths)
                expected_rows = []
                for dialogue, length in zip(batch[-1], lengths):
                    expected_rows.extend((str(dialogue), ui) for ui in range(length))
                require(expected_rows == [(row['dialogue_id'], int(row['utterance_index']))
                                          for row in rows[offset:offset + count]], 'Batch context/row order changed')
                guard.reset()
                original_parts.append(np.exp(helper.forward(model, tensors, lengths).cpu().numpy()))
                local = 0
                for di, length in enumerate(lengths):
                    block = rebuilt_audio[offset + local:offset + local + length]
                    tensors[5][:length, di].copy_(torch.as_tensor(block, dtype=torch.float32, device='cuda:0'))
                    local += length
                rebuilt_parts.append(np.exp(helper.forward(model, tensors, lengths).cpu().numpy()))
                for index, (actual, before) in enumerate(zip(tensors, unchanged)):
                    if index != 5:
                        require(torch.equal(actual, before), f'Non-audio input changed: {index}')
                # Reconstructed real rows do not change audio padding.
                for di, length in enumerate(lengths):
                    require(torch.equal(tensors[5][length:, di], unchanged[5][length:, di]), 'Audio padding changed')
                for name, value in model.state_dict().items():
                    require(torch.equal(value.cpu(), state[name]), f'Model parameter or buffer changed: {name}')
                require(all(parameter.grad is None for parameter in model.parameters()), 'Gradient unexpectedly exists')
                offset += count
    finally:
        guard.close()
    require(offset == SPECS[dataset]['rows'], 'Incomplete probability run')
    original = np.concatenate(original_parts)
    reconstructed = np.concatenate(rebuilt_parts)
    require(np.allclose(original, baseline['probabilities'], atol=1e-6, rtol=1e-5), 'Frozen original probabilities drifted from historical Validation baseline')
    require(np.array_equal(original.argmax(-1), baseline['probabilities'].argmax(-1)), 'Original baseline class drift')
    for path, digest in expected_hashes.items():
        require(sha(path) == digest, f'Frozen input changed during inference: {path}')
    for path, digest in reconstruction_hashes.items():
        require(sha(path) == digest, f'Reconstruction input or gate changed during inference: {path}')
    metrics = metric_summary(original, reconstructed, np)
    destination.mkdir(parents=True, mode=0o700)
    np.savez_compressed(destination / 'full_probabilities.npz', original_probabilities=original,
                        reconstructed_probabilities=reconstructed,
                        dialogue_ids=np.asarray([row['dialogue_id'] for row in rows]),
                        utterance_indices=np.asarray([int(row['utterance_index']) for row in rows]))
    fields = ['row_index', 'dataset', 'dialogue_id', 'utterance_index', 'speaker_id',
              'original_prediction', 'rebuilt_prediction', 'original_confidence', 'rebuilt_confidence',
              'max_probability_absolute_error', 'original_probabilities', 'rebuilt_probabilities']
    with (destination / 'per_utterance_probability_parity.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, p, q in zip(rows, original, reconstructed):
            record = {key: row[key] for key in fields[:5]}
            record.update(original_prediction=int(p.argmax()), rebuilt_prediction=int(q.argmax()),
                          original_confidence=float(p.max()), rebuilt_confidence=float(q.max()),
                          max_probability_absolute_error=float(np.abs(p-q).max()),
                          original_probabilities=json.dumps(p.tolist()), rebuilt_probabilities=json.dumps(q.tolist()))
            writer.writerow(record)
    result = dict(dataset=dataset, seed=seed, metrics=metrics, feature_gate=accepted_gate,
                  original_baseline_max_error=float(np.abs(original-baseline['probabilities']).max()),
                  frozen_input_hashes=expected_hashes, elapsed_seconds=time.monotonic()-started,
                  reconstruction_input_hashes=reconstruction_hashes,
                  full_dialogue_context=True, preserved_non_audio_inputs=True,
                  parameters_and_buffers_unchanged=True, graph_indices_unchanged=True,
                  batch_size=16, dtype='float32', eval=True, gradients_absent=True,
                  host=platform.node(), device=args.device, cuda_visible_devices=visible,
                  test_dataset_constructed=False, test_records_indexed=False, labels_exported=False,
                  checkpoint_sha256=item['checkpoint_sha256'], runner_sha256=sha(__file__),
                  feature_gate_sha256=sha(args.feature_gate), manifest_sha256=sha(args.manifest),
                  rebuilt_audio_sha256=sha(args.rebuilt_validation_audio),
                  outputs={path.name: sha(path) for path in destination.iterdir() if path.is_file()})
    (destination / 'probability_parity.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for path in destination.iterdir():
        if path.is_file():
            path.chmod(0o600)
    print(json.dumps({'dataset': dataset, 'metrics': metrics, 'output': str(destination)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(SPECS), required=True)
    parser.add_argument('--rebuilt-validation-audio', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--feature-gate', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda:0'], required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
