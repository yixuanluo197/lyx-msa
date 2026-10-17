"""Plan generation only; no subprocess or model invocation."""
import json
import shlex
import sys
import torch

import pytest


def complete_batch(tmp_path):
    for name, status in [('PIPELINE_STATE', 'completed'), ('BASELINE_VERIFICATION', 'passed'),
                         ('CMERC_VERIFICATION', 'passed')]:
        (tmp_path / (name + '.json')).write_text(json.dumps({'status': status}))
    return tmp_path


def test_gate_blocks_unfinished_work_without_writing(tmp_path):
    from scripts.round2_future_matrix import generate
    complete_batch(tmp_path)
    (tmp_path / 'PIPELINE_STATE.json').write_text('{"status":"running"}')
    with pytest.raises(ValueError, match='completed'):
        generate(tmp_path)
    assert not (tmp_path / 'FUTURE_MATRIX.json').exists()


def test_matrix_flags_seeds_and_provenance_stay_explicit(tmp_path):
    from scripts.round2_future_matrix import generate, DIFFICULTY_CHECK
    compile(DIFFICULTY_CHECK, '<planned-difficulty-check>', 'exec')
    matrix = generate(complete_batch(tmp_path))
    assert len(matrix['pairs']) == 6
    for pair in matrix['pairs']:
        assert pair['status'] == 'planned' and pair['split_seed'] == 1475
        assert pair['training_seed'] in [1937, 2025, 3407]
        baseline = pair['baseline_training']
        assert '--courselearning' not in baseline and '--contrastlearning' not in baseline
        for name, command in pair['extension_training'].items():
            assert 'train_cm.py' in command and 'run_training.sh' not in command
            assert '--defer-final-test' in command and '--fail-on-nonfinite' in command
            assert '--diagnostic-batches 1' in command and '${GPU_UUID:?' in command
            assert ('--courselearning' in command) == (name != 'without_cl')
            assert ('--contrastlearning' in command) == (name != 'without_hscl')
            assert ('--calibrate' in command) == (name != 'without_cc')
        assert '--auxiliary-policy same_step_same_mode' in pair['extension_training']['same_step_same_mode']
        assert 'DIFFICULTY_' in pair['full_training']
        assert 'DIFFICULTY_SHA_' in pair['full_training']
        assert '--expected-difficulty-sha256' in pair['completed_full_and_pair_admission']
        assert '--validation-freeze' in pair['test_comparison']
        assert 'VALIDATION_FREEZE_SHA_' in pair['test_evaluation']['baseline']
        assert 'scripts/round2_verify_run.py' in pair['strict_teacher_admission']
        assert f'--dataset {pair["dataset"]} --seed {pair["training_seed"]} --split-seed 1475' \
            in pair['completed_full_and_pair_admission']
        for key in ('strict_teacher_admission', 'strict_difficulty_admission', 'difficulty_generation'):
            assert shlex.split(pair[key])
    meld = next(p for p in matrix['pairs'] if p['dataset'] == 'MELD')
    assert '--epochs 40' in meld['baseline_training'] and '--dropout 0.4' in meld['full_training']
    assert '--rank_coff 0.002' in meld['full_training'] and '--epoch_ratio 0.4' in meld['full_training']
    assert (tmp_path / 'FUTURE_EXPERIMENTS.md').exists()
    with pytest.raises(FileExistsError):
        generate(tmp_path)


@pytest.mark.parametrize('corruption', [None, 'comparison', 'checkpoint', 'freeze', 'seed', 'validation_input'])
def test_native_validation_freeze_guards_content_and_pair(tmp_path, monkeypatch, corruption):
    from scripts.round2_future_matrix import FREEZE_CODE, sha
    identity = dict(dataset='IEMOCAP', split='validation', split_seed=1475, replacement_policy='prefix_mean_v1')
    best, evaluation, inputs = {}, {}, {}
    for name in ('baseline', 'cmerc'):
        best[name] = tmp_path / (name + '.pt')
        torch.save(dict(args=dict(Dataset='IEMOCAP', seed=1937, split_seed=1475,
            causal=True, selection_protocol='validation'), epoch=0, best_epoch=0), best[name])
        evaluation[name] = tmp_path / name
        evaluation[name].mkdir()
        (evaluation[name] / 'metrics.json').write_text(json.dumps({**identity, 'checkpoint': str(best[name])}))
        (evaluation[name] / 'predictions.npz').write_bytes(b'opaque fixture bytes, never used for inference')
        inputs[name] = {filename: sha(evaluation[name] / filename) for filename in ('metrics.json', 'predictions.npz')}
    comparison = tmp_path / 'comparison.json'
    comparison.write_text(json.dumps({**identity, 'input_sha256': inputs}))
    freeze = tmp_path / 'freeze.json'
    code = compile(FREEZE_CODE, '<planned-freeze>', 'exec')
    def execute(mode, seed='1937', digest=None):
        argv = ['-c', mode, 'IEMOCAP', seed, str(comparison), str(best['baseline']), str(best['cmerc']),
                str(evaluation['baseline']), str(evaluation['cmerc']), str(freeze)]
        if digest is not None:
            argv.append(digest)
        monkeypatch.setattr(sys, 'argv', argv)
        exec(code, {})
    execute('create')
    digest = sha(freeze)
    with pytest.raises(FileExistsError):
        execute('create')
    if corruption is None:
        execute('check', digest=digest)
        return
    if corruption == 'comparison':
        comparison.write_text(comparison.read_text() + ' ')
    elif corruption == 'checkpoint':
        best['baseline'].write_bytes(best['baseline'].read_bytes() + b'changed')
    elif corruption == 'freeze':
        freeze.write_text(freeze.read_text() + ' ')
    elif corruption == 'validation_input':
        (evaluation['cmerc'] / 'predictions.npz').write_bytes(b'changed')
    with pytest.raises(ValueError):
        execute('check', seed='2025' if corruption == 'seed' else '1937', digest=digest)
