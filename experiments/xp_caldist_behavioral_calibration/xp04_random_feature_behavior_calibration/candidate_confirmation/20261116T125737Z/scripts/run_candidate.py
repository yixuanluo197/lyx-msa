"""Frozen XP04 Gaussian candidate confirmation; shared RNG across model seeds."""
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

RNG_SEED, STRENGTH, REPEATS = 1475, 0.05, 10
SEEDS = [1475, 67137, 2024, 3407, 42]
CANDIDATES = {'IEMOCAP': 'text', 'MELD': 'audio'}
MODALITIES = {'text': [0, 1, 2, 3], 'audio': [5], 'visual': [4],
              'joint': [0, 1, 2, 3, 5, 4]}
NAMES = {0: 'text0', 1: 'text1', 2: 'text2', 3: 'text3', 5: 'audio', 4: 'visual'}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def key_for(dataset, dialogue, utterance, name, kind, repeat):
    return [str(dataset), RNG_SEED, str(dialogue), int(utterance), name, kind, STRENGTH, int(repeat)]


def canonical_key(key):
    return json.dumps(key, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def verify_reference_csv(path, reference, dataset, seed):
    """Require saved canonical probabilities and all IDs to exactly match the reference CSV."""
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = 527 if dataset == 'IEMOCAP' else 1109
    assert len(rows) == len(reference['labels']) == expected
    for i, row in enumerate(rows):
        assert row['dataset'] == dataset and int(row['seed']) == seed
        assert row['dialogue_id'] == str(reference['dialogue_ids'][i])
        assert int(row['utterance_index']) == int(reference['utterance_indices'][i])
        assert int(row['label']) == int(reference['labels'][i])
        assert int(row['prediction']) == int(reference['predictions'][i])
        assert np.array_equal(np.asarray(json.loads(row['probabilities']), dtype=np.float64), reference['probabilities'][i].astype(np.float64))
        assert np.array_equal(np.asarray(json.loads(row['log_probabilities']), dtype=np.float64), reference['log_probabilities'][i].astype(np.float64))
        assert float(row['confidence']) == float(reference['probabilities'][i].max())
    return {'passed': True, 'rows': expected, 'csv': str(Path(path).resolve()),
        'csv_sha256': xp.sha(path), 'probabilities_exact_equal': True,
        'log_probabilities_exact_equal': True, 'metadata_exact_equal': True}


def active_indices(modality, visual):
    missing = 4 in MODALITIES[modality] and bool(np.all(np.asarray(visual) == 0))
    return [j for j in MODALITIES[modality] if not (j == 4 and missing)], missing


def changed_vectors(vectors, stats, dataset, dialogue, utterance, kind, repeat):
    """Historical independent feature RNG; model seed deliberately absent from RNG key."""
    if kind != 'gaussian':
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
        requested = len(eligible)
        value = x + STRENGTH * std * rng.normal(size=x.shape)
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


def replay_diagnostics(model, tensors, lengths, cache, initial_output, torch, dest):
    """Observe exact inputs separately from CUDA floating-point output parity."""
    inputs = [x.detach().clone() for x in tensors]
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    def tensor_hash(value):
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    before_hashes = [tensor_hash(x) for x in tensors]
    outputs = {'cached': [initial_output.detach().cpu().numpy()] , 'uncached': []}
    inputs_equal = True
    for mode, count in [('cached', 4), ('uncached', 3)]:
        cache.enable() if mode == 'cached' else cache.restore()
        for _ in range(count):
            outputs[mode].append(xp.forward(model, tensors, lengths).detach().cpu().numpy())
            inputs_equal = inputs_equal and all(torch.equal(a, b) for a, b in zip(inputs, tensors))
    cache.enable()
    weights_equal = all(torch.equal(v.detach().cpu(), state[k]) for k, v in model.state_dict().items())
    reference = outputs['cached'][0]
    ref_p = np.exp(reference.astype(np.float64))
    records = []
    for mode, group in outputs.items():
        for repeat, value in enumerate(group):
            p = np.exp(value.astype(np.float64))
            records.append({'mode': mode, 'repeat': repeat,
                'exact_equal': bool(np.array_equal(reference, value)),
                'log_probability_max_abs_error': float(np.max(np.abs(reference - value))),
                'probability_max_abs_error': float(np.max(np.abs(ref_p - p))),
                'argmax_equal': bool(np.array_equal(reference.argmax(-1), value.argmax(-1))),
                'log_probability_tolerance_passed': bool(np.isfinite(value).all() and np.allclose(reference, value, atol=1e-6, rtol=1e-5)),
                'probability_tolerance_passed': bool(np.isfinite(p).all() and np.allclose(ref_p, p, atol=1e-6, rtol=1e-5))})
    def group_spread(group):
        return float(max(np.max(np.abs(a - b)) for a in group for b in group))
    eval_flags = {name: not module.training for name, module in model.named_modules()}
    result = {'criterion': 'Historical all-class log-probability/probability atol=1e-6, rtol=1e-5 plus identical argmax; feature RNG and input bytes exact',
        'atol': 1e-6, 'rtol': 1e-5, 'cached_repeats': 5, 'uncached_repeats': 3,
        'input_sha256_before': before_hashes, 'input_sha256_after': [tensor_hash(x) for x in tensors],
        'inputs_exact_equal': inputs_equal, 'weights_exact_equal': weights_equal,
        'all_modules_eval': all(eval_flags.values()), 'module_eval_flags': eval_flags,
        'parameter_dtypes': sorted(set(str(p.dtype) for p in model.parameters())),
        'bitwise_output_equal': all(r['exact_equal'] for r in records),
        'cached_pairwise_logp_max_abs_error': group_spread(outputs['cached']),
        'uncached_pairwise_logp_max_abs_error': group_spread(outputs['uncached']),
        'cached_uncached_logp_max_abs_error': float(max(np.max(np.abs(a - b)) for a in outputs['cached'] for b in outputs['uncached'])),
        'comparisons_to_initial_cached_output': records,
        'environment': {k: os.environ.get(k) for k in ['LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}}
    result['passed'] = bool(inputs_equal and weights_equal and result['all_modules_eval'] and
        result['input_sha256_before'] == result['input_sha256_after'] and all(r['argmax_equal'] and
        r['log_probability_tolerance_passed'] and r['probability_tolerance_passed'] for r in records))
    write_json(dest / 'replay_diagnostics.json', result)
    assert result['passed'], 'Identical-input CUDA output parity failed; see replay_diagnostics.json'
    return result


def freeze_inputs(dataset, seed, reference_path, reference_csv, stats_path):
    config_path = xp.BASE / f'configs/effective/{dataset}_cmerc_validation_seed{seed}.json'
    config = xp.config_for(dataset, seed)
    assert config['batch_size'] == 16 and config['rnn_backend'] == 'aten'
    assert not config['causal']
    split = config['provenance']['split']
    valid = set(map(str, split['validation_ids']))
    assert valid.isdisjoint(map(str, split['train_ids']))
    assert valid.isdisjoint(map(str, split['test_ids']))
    checkpoint = xp.BASE / f'runs/cmerc/{dataset}_validation_seed{seed}/best.pt'
    paths = [Path(__file__), ROOT / 'preregistration.json', ROOT / 'EXPERIMENT_PLAN.md',
             config_path, checkpoint, reference_path, reference_csv, stats_path]
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
    args.modality = CANDIDATES[args.dataset]
    assert args.seed in SEEDS
    assert (args.seed != 1475 and args.block == 0) or (args.seed == 1475 and args.block in (1, 2))
    dest = ROOT / 'runs' / args.dataset / f'seed{args.seed}' / f'block{args.block}'
    if args.smoke:
        dest /= 'smoke'
    dest.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in dest.iterdir() if p.name != 'smoke']
    assert not existing, 'Run exists; archive partial artifacts explicitly before recovery: ' + str(existing)
    prereg = json.loads((ROOT / 'preregistration.json').read_text())
    if args.seed == 1475:
        reference_path = Path(prereg['source_pilot']) / args.dataset / 'original_predictions.npz'
        reference_csv = reference_path.parent / 'original_predictions.csv'
    else:
        reference_path = xp.WORKSPACE / f'CMERC/experiments/xp_caldist_behavioral_calibration/20261107T172302Z/runs/{args.dataset}/seed{args.seed}/validation_predictions.npz'
        reference_csv = Path(prereg['source_five_seed']) / f'runs/{args.dataset}/seed{args.seed}/original_predictions.csv'
    stats_path = ROOT / f'provenance/{args.dataset}_train_statistics.npz'
    config, initial_hashes = freeze_inputs(args.dataset, args.seed, reference_path, reference_csv, stats_path)
    assert prereg['rng_seed'] == RNG_SEED and prereg['strength'] == STRENGTH and prereg['repeats'] == REPEATS
    assert prereg['datasets'][args.dataset]['modality'] == args.modality
    assert args.seed in prereg['blocks'][str(args.block)]['model_seeds']
    assert prereg['corruptions'] == ['gaussian']
    with np.load(reference_path) as z:
        reference = {k: z[k] for k in z.files}
    canonical_parity = verify_reference_csv(reference_csv, reference, args.dataset, args.seed)
    with np.load(stats_path) as z:
        stats = {k: z[k] for k in z.files}
    expected_n, expected_d, expected_missing, expected_train = ((527, 12, 1, 5283) if args.dataset == 'IEMOCAP' else (1109, 114, 95, 9989))
    assert all(int(stats[n + '_count']) == expected_train for n in NAMES.values())
    write_json(dest / 'source_hashes_before.json', initial_hashes)
    resource = xp.resource_snapshot()
    write_json(dest / 'resource_before.json', resource)
    write_json(dest / 'effective_config.json', {'dataset': args.dataset, 'modality': args.modality,
        'seed': args.seed, 'block': args.block, 'rng_seed': RNG_SEED, 'strength': STRENGTH, 'repeats': REPEATS, 'split': 'validation',
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
    expected_data_paths = {str((xp.CODE / relative).resolve()) for relative in config['provenance']['data']}
    def track_data(event, arguments):
        if event == 'open' and audit_active[0] and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(arguments[0])).resolve()
            if (str(path) in expected_data_paths or path.is_relative_to(xp.CODE.resolve())) and path.suffix in ('.pkl', '.npz'):
                assert str(path) in initial_hashes, ('Unfrozen data access', path)
                opened_data.add(str(path))
    sys.addaudithook(track_data)
    data = xp.dataset_for(args.dataset, 'validation', api[2])
    assert list(data.keys) == config['provenance']['split']['validation_ids']
    assert len(data) == expected_d and len(reference['labels']) == expected_n
    model = xp.load_model(args.dataset, args.seed, api)
    before_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    fresh = xp.infer(model, data, api)
    parity = xp.compare_alignment(fresh, reference)
    assert parity['passed'], parity
    parity['canonical_original'] = canonical_parity
    write_json(dest / 'validation_parity.json', parity)
    np.savez_compressed(dest / 'fresh_validation_predictions.npz', **fresh)
    np.savez_compressed(dest / 'original_predictions.npz', **reference)
    original_fields = ['dataset', 'seed', 'dialogue_id', 'utterance_index', 'label', 'prediction', 'confidence', 'probabilities', 'log_probabilities']
    with (dest / 'original_predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=original_fields)
        writer.writeheader()
        for i in range(expected_n):
            p = reference['probabilities'][i].astype(np.float64)
            writer.writerow(dict(dataset=args.dataset, seed=args.seed, dialogue_id=str(reference['dialogue_ids'][i]),
                utterance_index=int(reference['utterance_indices'][i]), label=int(reference['labels'][i]),
                prediction=int(reference['predictions'][i]), confidence=float(p.max()),
                probabilities=json.dumps(p.tolist()), log_probabilities=json.dumps(reference['log_probabilities'][i].tolist())))
    guard = xp.GraphGuard(model, torch)
    cache = install_graph_cache(model.graph_model)
    fields = ['dataset', 'seed', 'dialogue_id', 'utterance_index', 'modality', 'corruption', 'strength',
              'repeat_id', 'rng_repeat_id', 'block', 'perturbation_seed', 'rng_key', 'visual_missing_skipped', 'prediction', 'confidence', 'probabilities', 'log_probabilities']
    audit_fields = fields[:11] + ['feature_name', 'feature_index', 'rng_key', 'rng_seed', 'eligible_count',
        'requested_count', 'actual_changed_count', 'vector_sha256', 'original_vector_sha256', 'delta_l2', 'stdzero_count', 'stdzero_unchanged']
    rowbase = rows = forwards = missing_count = cached_checks = mask_checks = audit_count = 0
    repeated_check = None
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
                    for kind in ('gaussian',):
                        for repeat in range(REPEATS):
                            values, audits = changed_vectors(vectors, stats, args.dataset, dialogue, ui, kind, args.block * REPEATS + repeat)
                            base = dict(dataset=args.dataset, seed=args.seed, dialogue_id=dialogue,
                                utterance_index=ui, modality=args.modality, corruption=kind, strength=STRENGTH, repeat_id=repeat, rng_repeat_id=args.block * REPEATS + repeat, block=args.block)
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
                                        repeated_check = replay_diagnostics(model, tensors, lengths, cache, all_logp, torch, dest)
                                    lp = all_logp[offset + ui].detach().cpu().numpy()
                                    p = np.exp(lp.astype(np.float64))
                                else:
                                    lp = reference['log_probabilities'][row].copy()
                                    # Retain the saved original exp(FP32 logp) exactly when no feature is changed.
                                    p = reference['probabilities'][row].astype(np.float64).copy()
                                assert np.isfinite(lp).all() and np.isfinite(p).all() and abs(p.sum() - 1) < 1e-5
                            finally:
                                for j in active:
                                    tensors[j][ui, di].copy_(baseline[j][ui, di])
                            writer.writerow({**base, 'perturbation_seed': RNG_SEED,
                                'rng_key': json.dumps([key_for(args.dataset, dialogue, ui, NAMES[j], kind, args.block * REPEATS + repeat) for j in active]),
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
                        status = dict(dataset=args.dataset, modality=args.modality, seed=args.seed, utterances=row + 1,
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
    assert rows == target_count * REPEATS
    if not args.smoke:
        assert rowbase == expected_n and missing_count == expected_missing
        assert forwards == expected_n * REPEATS
    assert all(torch.equal(v.detach().cpu(), before_state[k]) for k, v in model.state_dict().items())
    guard.close()
    audit_active[0] = False
    imported = imported_sources(initial_hashes)
    after_hashes = {p: xp.sha(p) for p in initial_hashes}
    write_json(dest / 'source_hashes_after.json', after_hashes)
    assert after_hashes == initial_hashes, 'Frozen source/data/config/checkpoint changed'
    write_json(dest / 'perturbation_audit.json', {'rows': audit_count, 'file': 'perturbation_audit.csv',
        'sha256': xp.sha(dest / 'perturbation_audit.csv'), 'strength': STRENGTH,
        'rng_seed': RNG_SEED, 'block': args.block, 'rng_repeat_ids': list(range(args.block * REPEATS, (args.block + 1) * REPEATS)),
        'actual_changed_count_note': 'May be smaller when the original equals the replacement training mean or FP32 rounds noise away.',
        'stdzero_unchanged': True, 'rng': 'historical SHA256 first16 big-endian canonical JSON PCG64'})
    complete = dict(status='completed', dataset=args.dataset, modality=args.modality, seed=args.seed,
        strength=STRENGTH, block=args.block, rng_seed=RNG_SEED, smoke=args.smoke, utterances=target_count, dialogues=expected_d,
        perturbed_rows=rows, target_forwards=forwards, seconds=time.monotonic() - started,
        visual_missing_retained=missing_count, graph_hook_checks=guard.count,
        cached_uncached_parity_checks=cached_checks, target_mask_checks=mask_checks,
        identical_input_parity={'passed': repeated_check['passed'], 'bitwise_output_equal': repeated_check['bitwise_output_equal'],
            'criterion': repeated_check['criterion'], 'diagnostics_file': 'replay_diagnostics.json'},
        weights_unchanged=True, batch_inputs_restored=True,
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
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--block', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--smoke', action='store_true', help='One target; separate per-run smoke directory; full baseline parity retained')
    run(parser.parse_args())
