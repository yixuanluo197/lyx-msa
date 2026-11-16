"""Frozen seed1475 / strength0.10 target-only true-Validation inference."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / 'provenance/inference_helpers'
sys.path.insert(0, str(HELPERS))
import run_pilot as xp
from pilot_core import rng_for

SEED, STRENGTH, REPEATS = 1475, 0.1, 10
MODALITIES = {'text': [0, 1, 2, 3], 'audio': [5], 'visual': [4],
              'joint': [0, 1, 2, 3, 5, 4]}
NAMES = {0: 'text0', 1: 'text1', 2: 'text2', 3: 'text3', 5: 'audio', 4: 'visual'}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def key_for(dataset, dialogue, utterance, name, kind, repeat):
    return [str(dataset), SEED, str(dialogue), int(utterance), name, kind, STRENGTH, int(repeat)]


def canonical_key(key):
    return json.dumps(key, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def active_indices(modality, visual):
    missing = 4 in MODALITIES[modality] and bool(np.all(np.asarray(visual) == 0))
    return [j for j in MODALITIES[modality] if not (j == 4 and missing)], missing


def changed_vectors(vectors, stats, dataset, dialogue, utterance, kind, repeat):
    """Historical independent feature RNG and exact-count means replacement."""
    if kind not in ('gaussian', 'dropout'):
        raise ValueError(kind)
    output, audits = {}, []
    for index, vector in vectors.items():
        name = NAMES[index]
        original = np.asarray(vector, dtype=np.float32)
        x = original.astype(np.float64)
        std = np.asarray(stats[name + '_std'], dtype=np.float64)
        mean = np.asarray(stats[name + '_mean'], dtype=np.float64)
        assert x.ndim == 1 and std.shape == x.shape and mean.shape == x.shape
        assert np.isfinite(x).all() and np.isfinite(std).all() and np.isfinite(mean).all()
        assert np.all(std >= 0)
        key = key_for(dataset, dialogue, utterance, name, kind, repeat)
        rng = rng_for(*key)
        eligible = np.flatnonzero(std > 0)
        if kind == 'gaussian':
            requested = len(eligible)
            value = x + STRENGTH * std * rng.normal(size=x.shape)
        else:
            requested = int(round(STRENGTH * len(eligible)))
            selected = rng.choice(eligible, size=requested, replace=False)
            value = x.copy()
            value[selected] = mean[selected]
            assert len(np.unique(selected)) == requested
        value = value.astype(np.float32)
        assert value.shape == original.shape and np.isfinite(value).all()
        zero_unchanged = bool(np.array_equal(value[std == 0], original[std == 0]))
        assert zero_unchanged
        output[index] = value
        canonical = canonical_key(key)
        audits.append({'feature_name': name, 'feature_index': index, 'rng_key': canonical,
            'rng_seed': str(int.from_bytes(hashlib.sha256(canonical.encode('utf-8')).digest()[:16], 'big')),
            'eligible_count': len(eligible), 'requested_count': requested,
            'actual_changed_count': int(np.count_nonzero(value != original)),
            'vector_sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest(),
            'original_vector_sha256': hashlib.sha256(original.tobytes(order='C')).hexdigest(),
            'delta_l2': float(np.linalg.norm(value.astype(np.float64) - x)),
            'stdzero_count': int(np.count_nonzero(std == 0)), 'stdzero_unchanged': zero_unchanged})
    return output, audits


def assert_target_only(tensors, baseline, values, ui, di, torch):
    """Check full context, qmask, umask, labels and padded storage."""
    assert tensors[7][di, ui].item() == 1
    for j, (actual, original) in enumerate(zip(tensors, baseline)):
        expected = original.clone()
        if j in values:
            expected[ui, di].copy_(torch.from_numpy(values[j]).to(actual.device))
        assert torch.equal(actual, expected), 'Non-target feature/mask/padding changed: ' + str(j)


def freeze_inputs(dataset, reference_path, stats_path):
    config_path = xp.BASE / f'configs/effective/{dataset}_cmerc_validation_seed{SEED}.json'
    config = xp.config_for(dataset, SEED)
    assert config['batch_size'] == 16 and config['rnn_backend'] == 'aten'
    assert not config['causal']
    split = config['provenance']['split']
    valid = set(map(str, split['validation_ids']))
    assert valid.isdisjoint(map(str, split['train_ids']))
    assert valid.isdisjoint(map(str, split['test_ids']))
    checkpoint = xp.BASE / f'runs/cmerc/{dataset}_validation_seed{SEED}/best.pt'
    paths = [Path(__file__), ROOT / 'preregistration.json', ROOT / 'EXPERIMENT_PLAN.md',
             config_path, checkpoint, reference_path, stats_path]
    paths += list(HELPERS.glob('*.py')) + list(xp.CODE.rglob('*.py'))
    for relative, expected in config['provenance']['data'].items():
        path = xp.CODE / relative
        assert xp.sha(path) == expected, ('Historical data hash mismatch', path)
        paths.append(path)
    # Only hashes are retained for large data/checkpoints; no originals are copied.
    hashes = {str(p.resolve()): xp.sha(p) for p in paths}
    return config, hashes


def imported_sources(hashes):
    origin = json.loads((xp.BASE / 'provenance/origin.json').read_text())['source_files_sha256']
    paths = set()
    for module in tuple(sys.modules.values()):
        location = getattr(module, '__file__', None)
        if not location:
            continue
        p = Path(location).resolve()
        if p.suffix == '.pyc':
            p = Path(str(p).replace('/__pycache__/', '/').split('.cpython-')[0] + '.py')
        if p.is_relative_to(xp.CODE.resolve()) or p.is_relative_to(HELPERS.resolve()):
            assert str(p) in hashes, ('Unfrozen imported code', p)
            if p.is_relative_to(xp.CODE.resolve()):
                relative = str(p.relative_to(xp.CODE.resolve()))
                if relative in origin:
                    assert hashes[str(p)] == origin[relative], ('Historical imported source changed', p)
            paths.add(str(p))
    return {p: hashes[p] for p in sorted(paths)}


def run(args):
    from graph_index_cache import install_graph_cache
    dest = ROOT / 'runs' / args.dataset / args.modality
    if args.smoke:
        dest /= 'smoke'
    dest.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in dest.iterdir() if p.name != 'smoke']
    assert not existing, 'Run exists; archive partial artifacts explicitly before recovery: ' + str(existing)
    reference_path = xp.WORKSPACE / f'CMERC/experiments/xp_caldist_behavioral_calibration/20261107T172302Z/runs/{args.dataset}/seed{SEED}/validation_predictions.npz'
    stats_path = ROOT / f'provenance/{args.dataset}_train_statistics.npz'
    config, initial_hashes = freeze_inputs(args.dataset, reference_path, stats_path)
    prereg = json.loads((ROOT / 'preregistration.json').read_text())
    assert prereg['seed'] == SEED and prereg['new_strength'] == STRENGTH and prereg['repeats'] == REPEATS
    assert prereg['corruptions'] == ['gaussian', 'dropout']
    with np.load(reference_path) as z:
        reference = {k: z[k] for k in z.files}
    with np.load(stats_path) as z:
        stats = {k: z[k] for k in z.files}
    expected_n, expected_d, expected_missing, expected_train = ((527, 12, 1, 5283) if args.dataset == 'IEMOCAP' else (1109, 114, 95, 9989))
    assert all(int(stats[n + '_count']) == expected_train for n in NAMES.values())
    write_json(dest / 'source_hashes_before.json', initial_hashes)
    resource = xp.resource_snapshot()
    write_json(dest / 'resource_before.json', resource)
    write_json(dest / 'effective_config.json', {'dataset': args.dataset, 'modality': args.modality,
        'seed': SEED, 'strength': STRENGTH, 'repeats': REPEATS, 'split': 'validation',
        'smoke': args.smoke, 'batch_size': 16, 'rnn_backend': 'aten', 'precision': 'float32',
        'checkpoint_config': config})
    api = xp.imports()
    torch = api[0]
    assert torch.__version__ == '1.13.1+cu117' and torch.cuda.is_available()
    assert os.environ.get('CUDA_VISIBLE_DEVICES'), 'Explicit GPU visibility is required'
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    os.chdir(xp.CODE)
    opened_data, audit_active = set(), [True]
    def track_data(event, arguments):
        if event == 'open' and audit_active[0] and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(arguments[0])).resolve()
            if path.is_relative_to(xp.CODE.resolve()) and path.suffix in ('.pkl', '.npz'):
                assert str(path) in initial_hashes, ('Unfrozen data access', path)
                opened_data.add(str(path))
    sys.addaudithook(track_data)
    data = xp.dataset_for(args.dataset, 'validation', api[2])
    assert list(data.keys) == config['provenance']['split']['validation_ids']
    assert len(data) == expected_d and len(reference['labels']) == expected_n
    model = xp.load_model(args.dataset, SEED, api)
    before_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    fresh = xp.infer(model, data, api)
    parity = xp.compare_alignment(fresh, reference)
    assert parity['passed'], parity
    write_json(dest / 'validation_parity.json', parity)
    np.savez_compressed(dest / 'fresh_validation_predictions.npz', **fresh)
    original_fields = ['dataset', 'seed', 'dialogue_id', 'utterance_index', 'label', 'prediction', 'confidence', 'probabilities', 'log_probabilities']
    with (dest / 'original_predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=original_fields)
        writer.writeheader()
        for i in range(expected_n):
            p = reference['probabilities'][i].astype(np.float64)
            writer.writerow(dict(dataset=args.dataset, seed=SEED, dialogue_id=str(reference['dialogue_ids'][i]),
                utterance_index=int(reference['utterance_indices'][i]), label=int(reference['labels'][i]),
                prediction=int(reference['predictions'][i]), confidence=float(p.max()),
                probabilities=json.dumps(p.tolist()), log_probabilities=json.dumps(reference['log_probabilities'][i].tolist())))
    guard = xp.GraphGuard(model, torch)
    cache = install_graph_cache(model.graph_model)
    fields = ['dataset', 'seed', 'dialogue_id', 'utterance_index', 'modality', 'corruption', 'strength',
              'repeat_id', 'perturbation_seed', 'rng_key', 'visual_missing_skipped', 'prediction', 'confidence', 'probabilities', 'log_probabilities']
    audit_fields = fields[:8] + ['feature_name', 'feature_index', 'rng_key', 'rng_seed', 'eligible_count',
        'requested_count', 'actual_changed_count', 'vector_sha256', 'original_vector_sha256', 'delta_l2', 'stdzero_count', 'stdzero_unchanged']
    rowbase = rows = forwards = missing_count = cached_checks = mask_checks = audit_count = 0
    repeated_check = False
    started = time.monotonic()
    with (dest / 'perturbed_predictions.csv').open('w', newline='') as stream, \
         (dest / 'perturbation_audit.csv').open('w', newline='') as audit_stream, torch.no_grad():
        writer = csv.DictWriter(stream, fieldnames=fields)
        audit_writer = csv.DictWriter(audit_stream, fieldnames=audit_fields)
        writer.writeheader()
        audit_writer.writeheader()
        for batch in xp.loader_for(data, torch):
            tensors = [x.cuda() for x in batch[:-1]]
            lengths = [int(x.sum().item()) for x in tensors[7]]
            baseline = [x.clone() for x in tensors]
            guard.reset()
            cache.hyper_indices.clear()
            cache.gnn_indices.clear()
            cache.restore()
            raw_uncached = xp.forward(model, tensors, lengths)
            cache.enable()
            raw_cached = xp.forward(model, tensors, lengths)
            assert torch.allclose(raw_uncached, raw_cached, atol=1e-6, rtol=1e-5)
            assert np.allclose(raw_cached.cpu().numpy(), reference['log_probabilities'][rowbase:rowbase + sum(lengths)], atol=1e-6, rtol=1e-5)
            cached_checks += 1
            offset = 0
            for di, (dialogue, length) in enumerate(zip(batch[-1], lengths)):
                dialogue = str(dialogue)
                for ui in range(length):
                    row = rowbase + offset + ui
                    assert dialogue == str(reference['dialogue_ids'][row]) and ui == int(reference['utterance_indices'][row])
                    assert tensors[7][di, ui].item() == 1
                    active, missing = active_indices(args.modality, baseline[4][ui, di].cpu().numpy())
                    # Count actual missing visuals independently of selected modality.
                    missing_count += int(torch.all(baseline[4][ui, di] == 0).item())
                    vectors = {j: baseline[j][ui, di].cpu().numpy().copy() for j in active}
                    for kind in ('gaussian', 'dropout'):
                        for repeat in range(REPEATS):
                            values, audits = changed_vectors(vectors, stats, args.dataset, dialogue, ui, kind, repeat)
                            base = dict(dataset=args.dataset, seed=SEED, dialogue_id=dialogue,
                                utterance_index=ui, modality=args.modality, corruption=kind, strength=STRENGTH, repeat_id=repeat)
                            try:
                                for j, value in values.items():
                                    tensors[j][ui, di].copy_(torch.from_numpy(value).to(tensors[j].device))
                                if ui == 0 and repeat == 0:
                                    assert_target_only(tensors, baseline, values, ui, di, torch)
                                    mask_checks += 1
                                if active:
                                    all_logp = xp.forward(model, tensors, lengths)
                                    forwards += 1
                                    if di == 0 and ui == 0 and repeat == 0:
                                        cache.restore()
                                        uncached = xp.forward(model, tensors, lengths)
                                        cache.enable()
                                        assert torch.allclose(uncached, all_logp, atol=1e-6, rtol=1e-5)
                                        cached_checks += 1
                                    if not repeated_check:
                                        replay = xp.forward(model, tensors, lengths)
                                        assert torch.equal(replay, all_logp), 'Identical-input repeatability failed'
                                        repeated_check = True
                                    lp = all_logp[offset + ui].detach().cpu().numpy()
                                    p = np.exp(lp.astype(np.float64))
                                else:
                                    lp = reference['log_probabilities'][row].copy()
                                    p = np.exp(lp.astype(np.float64))
                                assert np.isfinite(lp).all() and np.isfinite(p).all() and abs(p.sum() - 1) < 1e-5
                            finally:
                                for j in active:
                                    tensors[j][ui, di].copy_(baseline[j][ui, di])
                            writer.writerow({**base, 'perturbation_seed': SEED,
                                'rng_key': json.dumps([key_for(args.dataset, dialogue, ui, NAMES[j], kind, repeat) for j in active]),
                                'visual_missing_skipped': missing, 'prediction': int(p.argmax()), 'confidence': float(p.max()),
                                'probabilities': json.dumps(p.tolist()), 'log_probabilities': json.dumps(lp.tolist())})
                            for audit in audits:
                                audit_writer.writerow({**base, **audit})
                                audit_count += 1
                            rows += 1
                    if (row + 1) % 10 == 0 or args.smoke:
                        stream.flush()
                        audit_stream.flush()
                        elapsed = time.monotonic() - started
                        status = dict(dataset=args.dataset, modality=args.modality, seed=SEED, utterances=row + 1,
                            total=expected_n, rows=rows, forwards=forwards, seconds=elapsed,
                            forwards_per_second=forwards / elapsed, peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
                        print(json.dumps(status), flush=True)
                        write_json(dest / 'progress.json', status)
                    if args.smoke:
                        break
                if args.smoke:
                    break
                offset += length
            assert all(torch.equal(x, y) for x, y in zip(tensors, baseline)), 'Batch restore mismatch'
            rowbase += sum(lengths)
            if args.smoke:
                break
    target_count = 1 if args.smoke else expected_n
    assert rows == target_count * 20
    if not args.smoke:
        assert rowbase == expected_n and missing_count == expected_missing
        assert forwards == (expected_n - (expected_missing if args.modality == 'visual' else 0)) * 20
    assert all(torch.equal(v.detach().cpu(), before_state[k]) for k, v in model.state_dict().items())
    guard.close()
    audit_active[0] = False
    imported = imported_sources(initial_hashes)
    after_hashes = {p: xp.sha(p) for p in initial_hashes}
    write_json(dest / 'source_hashes_after.json', after_hashes)
    assert after_hashes == initial_hashes, 'Frozen source/data/config/checkpoint changed'
    write_json(dest / 'perturbation_audit.json', {'rows': audit_count, 'file': 'perturbation_audit.csv',
        'sha256': xp.sha(dest / 'perturbation_audit.csv'), 'strength': STRENGTH,
        'dropout_requested_count': 'round(0.1 * count(training_std > 0))',
        'actual_changed_count_note': 'May be smaller when the original equals the replacement training mean or FP32 rounds noise away.',
        'stdzero_unchanged': True, 'rng': 'historical SHA256 first16 big-endian canonical JSON PCG64'})
    complete = dict(status='completed', dataset=args.dataset, modality=args.modality, seed=SEED,
        strength=STRENGTH, smoke=args.smoke, utterances=target_count, dialogues=expected_d,
        perturbed_rows=rows, target_forwards=forwards, seconds=time.monotonic() - started,
        visual_missing_retained=missing_count, graph_hook_checks=guard.count,
        cached_uncached_parity_checks=cached_checks, target_mask_checks=mask_checks,
        identical_input_repeatability=repeated_check, weights_unchanged=True, batch_inputs_restored=True,
        source_sha256=initial_hashes, actual_imported_source_sha256=imported,
        actual_opened_data_sha256={p: initial_hashes[p] for p in sorted(opened_data)},
        data_sha256={str((xp.CODE / f).resolve()): initial_hashes[str((xp.CODE / f).resolve())] for f in config['provenance']['data']},
        parity=parity, peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20, graph_cache_stats=cache.stats,
        resource=resource, output_sha256={p.name: xp.sha(p) for p in dest.iterdir() if p.is_file()})
    write_json(dest / 'complete.json', complete)
    print(json.dumps({'completed': args.dataset, 'modality': args.modality, 'seconds': complete['seconds']}), flush=True)
    del model, data
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['IEMOCAP', 'MELD'], required=True)
    parser.add_argument('--modality', choices=list(MODALITIES), required=True)
    parser.add_argument('--smoke', action='store_true', help='One target; separate modality/smoke directory; full baseline parity retained')
    run(parser.parse_args())
