"""Runner safety gates, without launching training or using test data."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.round3_runner import (Runner, optimizer_statistics, sha, training_command,
                                  validate_identity, validate_test_authorization, write)


@pytest.fixture
def runner(tmp_path):
    code = tmp_path / 'code'
    (code / 'data').mkdir(parents=True)
    (code / 'runtime.py').write_text('# frozen source\n')
    feature = code / 'data' / 'features.pkl'
    feature.write_bytes(b'fixture feature bytes')
    inventory = tmp_path / 'data' / 'MELD' / 'DATA_ENVIRONMENT.json'
    write(inventory, {'data_provenance': {'feature_files': {'features.pkl': sha(feature)}}})
    (tmp_path / 'THIRD_ROUND_PROTOCOL.md').write_text('Frozen fixture protocol\n')
    import sys
    plan = {'python': sys.executable, 'code_dir': str(code),
            'protocol_sha256': sha(tmp_path / 'THIRD_ROUND_PROTOCOL.md'),
            'source_sha256': {'runtime.py': sha(code / 'runtime.py')},
            'data_inventories': {'MELD': {'path': str(inventory), 'sha256': sha(inventory)}},
            'seeds': [1937, 2025, 3407], 'gpu_allowlist': ['GPU-fixture']}
    write(tmp_path / 'PLAN.json', plan)
    args = SimpleNamespace(batch_dir=tmp_path, dataset='MELD', seed=1937,
        gpu_uuid='GPU-fixture', phase='validation', min_free_mb=5000,
        allow_shared_managed=False, test_authorization_sha256=None,
        resume_baseline=None, resume_cmerc=None)
    return Runner(args)


def test_frozen_source_and_actual_feature_bytes_are_checked(runner):
    runner.frozen()
    (runner.code / 'data' / 'features.pkl').write_bytes(b'changed actual data')
    with pytest.raises(ValueError, match='real feature bytes changed'):
        runner.frozen()


def test_replaced_inventory_rejected_even_if_feature_unchanged(runner):
    runner.inventory_path.write_text('{}')
    with pytest.raises(ValueError, match='inventory changed'):
        runner.frozen()


def test_unplanned_seed_gpu_and_completed_round2_retraining_rejected(runner):
    for dataset, seed, gpu, message in [('IEMOCAP', 1937, 'GPU-fixture', 'reuse'),
            ('MELD', 42, 'GPU-fixture', 'seed'), ('MELD', 2025, 'GPU-other', 'allowlist')]:
        with pytest.raises(ValueError, match=message):
            validate_identity(runner.plan, dataset, seed, gpu)


def test_training_flags_defer_test_and_use_official_wrapper():
    baseline = training_command('MELD', 2025, 'baseline')
    assert baseline[:5] == ['bash', 'scripts/run_training.sh', 'MELD', 'baseline', 'validation']
    assert '--defer-final-test' in baseline and '--fail-on-nonfinite' in baseline
    assert baseline[baseline.index('--seed') + 1] == '2025'
    assert '--difficulty-path' not in baseline
    with pytest.raises(ValueError, match='admitted real difficulty'):
        training_command('MELD', 2025, 'cmerc')
    full = training_command('MELD', 2025, 'cmerc', '/real/hash.pkl')
    assert full[full.index('--difficulty-path') + 1] == '/real/hash.pkl'


def test_successful_child_recorded_and_completed_stage_never_repeated(runner):
    output = runner.pair / 'result.txt'
    command = [runner.python, '-c', 'from pathlib import Path;Path(' + repr(str(output)) + ').write_text("done")']
    runner.run('fixture', command, artifacts=[output])
    attempt = runner.state['stages']['fixture'][0]
    assert attempt['status'] == 'completed' and attempt['exit_code'] == 0
    assert attempt['pid'] > 0 and Path(attempt['log']).is_file()
    runner.run('fixture', [runner.python, '-c', 'raise RuntimeError("must not run")'], artifacts=[output])
    assert len(runner.state['stages']['fixture']) == 1
    output.write_text('tampered')
    with pytest.raises(ValueError, match='artifact changed'):
        runner.run('fixture', command, artifacts=[output])


def test_failure_preserved_and_no_implicit_retry(runner):
    with pytest.raises(ValueError, match='failed'):
        runner.run('fixture', [runner.python, '-c', 'raise SystemExit(7)'], artifacts=[])
    attempt = runner.state['stages']['fixture'][0]
    assert attempt['exit_code'] == 7 and attempt['status'] == 'failed'
    with pytest.raises(RuntimeError, match='incomplete/failed attempt'):
        runner.run('fixture', [runner.python, '-c', 'pass'], artifacts=[])
    assert len(runner.state['stages']['fixture']) == 1


def test_gpu_preflight_refuses_other_users_processes(runner, monkeypatch):
    import scripts.round3_runner as module
    replies = iter(['GPU-fixture, 12000\n', 'GPU-fixture, 987, 200\n'])
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: next(replies))
    with pytest.raises(ValueError, match='unrelated GPU processes'):
        runner.available()


def test_test_phase_fails_before_inference_without_completed_validation(runner, monkeypatch):
    runner.args.phase = 'test'
    calls = []
    monkeypatch.setattr(runner, 'evaluate', lambda split: calls.append(split))
    with pytest.raises(ValueError, match='validation phase must complete'):
        runner.execute()
    assert calls == []


@pytest.fixture
def authorization(runner):
    pair = runner.pair
    best = {}
    inputs = {}
    for variant in ('baseline', 'cmerc'):
        best[variant] = pair / (variant + '.pt')
        best[variant].write_bytes(variant.encode())
        directory = pair / 'evaluation' / 'validation' / variant
        directory.mkdir(parents=True)
        inputs[variant] = {}
        for filename in ('metrics.json', 'predictions.npz'):
            (directory / filename).write_bytes(filename.encode())
            inputs[variant][filename] = sha(directory / filename)
    comparison = pair / 'comparison_validation' / 'comparison.json'
    write(comparison, {'dataset': 'MELD', 'split': 'validation', 'input_sha256': inputs})
    auth = {'mode': 'pilot', 'protocol_sha256': runner.plan['protocol_sha256'], 'pairs': {
        runner.pair_id: {'dataset': 'MELD', 'seed': 1937, 'comparison_sha256': sha(comparison),
                        'best_sha256': {v: sha(path) for v, path in best.items()}}}}
    path = runner.batch / 'TEST_AUTHORIZATION.json'
    write(path, auth)
    return path, best


def test_test_requires_external_hash_and_exact_seed(runner, authorization):
    path, best = authorization
    with pytest.raises(ValueError, match='explicit authorization'):
        validate_test_authorization(runner.batch, runner.pair_id, None, best)
    validate_test_authorization(runner.batch, runner.pair_id, sha(path), best)
    auth = json.loads(path.read_text())
    auth['pairs'][runner.pair_id]['seed'] = 2025
    write(path, auth)
    with pytest.raises(ValueError, match='dataset/seed mismatch'):
        validate_test_authorization(runner.batch, runner.pair_id, sha(path), best)


def test_test_authorization_pins_checkpoint_and_validation_inputs(runner, authorization):
    path, best = authorization
    digest = sha(path)
    target = runner.pair / 'evaluation' / 'validation' / 'baseline' / 'predictions.npz'
    target.write_bytes(b'replaced')
    with pytest.raises(ValueError, match='inference input changed'):
        validate_test_authorization(runner.batch, runner.pair_id, digest, best)
    target.write_bytes(b'predictions.npz')
    best['cmerc'].write_bytes(b'replaced checkpoint')
    with pytest.raises(ValueError, match='best checkpoint changed'):
        validate_test_authorization(runner.batch, runner.pair_id, digest, best)


def test_remaining_matrix_cannot_test_before_all_four_validation_freezes(runner, authorization):
    path, best = authorization
    auth = json.loads(path.read_text())
    auth['mode'] = 'remaining_matrix'
    write(path, auth)
    with pytest.raises(ValueError, match='complete preregistered'):
        validate_test_authorization(runner.batch, runner.pair_id, sha(path), best)


def test_adam_steps_are_checked_against_actual_curriculum_batches():
    import torch
    checkpoint = {'epoch': 1, 'args': {'batch_size': 16},
        'optimizer_state_dict': {'state': {0: {'step': torch.tensor(5.), 'exp_avg': torch.zeros(2)}}},
        'rng_state': {'python': (), 'numpy': (), 'torch': torch.zeros(5, dtype=torch.uint8),
                      'cuda': [torch.zeros(5, dtype=torch.uint8)]}}
    records = [{'train_dialogues': 16}, {'train_dialogues': 49}]
    result = optimizer_statistics(checkpoint, records)
    assert result['optimizer_steps_total'] == 5
    checkpoint['optimizer_state_dict']['state'][0]['step'] = torch.tensor(4.)
    with pytest.raises(ValueError, match='Adam step count'):
        optimizer_statistics(checkpoint, records)
