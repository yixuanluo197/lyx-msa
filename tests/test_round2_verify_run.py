"""Verifier fixtures only: tiny checkpoints are never formal experiment inputs."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from causal_provenance import (architecture_from_args, build_data_provenance,
    create_difficulty_manifest, write_causal_difficulty)
from causal_run import prepare_run
from train_cm import model_state_sha256, save_training_checkpoint


def fixture_run(tmp_path, variant='baseline', paired=None, difficulty=None, overrides=None):
    feature = tmp_path / 'fixture_feature'
    feature.write_text('synthetic verifier fixture, not real features')
    data = build_data_provenance('IEMOCAP', ['train-a', 'train-b'], ['validation-a'],
                                 {'feature': feature}, 1475)
    expected = tmp_path / 'DATA_ENVIRONMENT.json'
    expected.write_text(json.dumps({'data_provenance': data, 'train_ids': ['private-id']}))
    torch.manual_seed(1937)
    model = torch.nn.Linear(2, 2)
    args = SimpleNamespace(Dataset='IEMOCAP', causal=True, norm='LN', base_model='GRU',
        graph_type='hyper', mm_fusion_mthd='concat_DHT', modals='avl', multi_modal=True,
        selection_protocol='validation', seed=1937, split_seed=1475, batch_size=16,
        epochs=2, lr=.0001, dropout=.5, num_L=3, num_K=4,
        auxiliary_policy='legacy_epoch_snapshot', replacement_policy='prefix_mean_v1',
        graph_construct='direct', use_residue=False, av_using_lstm=False,
        use_speaker=True, use_modal=False, l2=.00003, fail_on_nonfinite=True,
        defer_final_test=True,
        calibrate=variant == 'cmerc', contrastlearning=variant == 'cmerc',
        courselearning=variant == 'cmerc', rank_coff=.005 if variant == 'cmerc' else 0.,
        mscl_coff=.05 if variant == 'cmerc' else 0., cscl_coff=.05 if variant == 'cmerc' else 0.,
        epoch_ratio=.15 if variant == 'cmerc' else 0., scheduler_steps=1 if variant == 'cmerc' else 0,
        data_provenance=data, difficulty_artifact=None, output_dir=str(tmp_path / variant), resume=None)
    for key, value in (overrides or {}).items():
        setattr(args, key, value)
    if difficulty:
        from causal_provenance import resolve_causal_difficulty
        _, args.difficulty_artifact = resolve_causal_difficulty(difficulty, 'IEMOCAP', data)
    args.initial_model_sha256 = model_state_sha256(model)
    run = prepare_run(args)
    (run / 'initial_state.json').write_text(json.dumps({'model_state_sha256': args.initial_model_sha256}))
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(2):
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        if epoch == 0:
            save_training_checkpoint(run / 'best.pt', model, optimizer, epoch, args, 60., 0)
            payload = torch.load(run / 'best.pt', map_location='cpu')
            payload['rng_state']['cuda'] = []
            torch.save(payload, run / 'best.pt')
        save_training_checkpoint(run / 'latest.pt', model, optimizer, epoch, args, 60., 0)
        payload = torch.load(run / 'latest.pt', map_location='cpu')
        payload['rng_state']['cuda'] = []
        torch.save(payload, run / 'latest.pt')
        losses = dict(classification_loss=1., cc_loss=0., mscl_loss=0., cscl_loss=0., total_loss=1.)
        record = dict(epoch=epoch + 1, selection_protocol='validation', causal=True,
            selection_split='validation', selection_weighted_f1=60., validation_weighted_f1=60.,
            validation_accuracy=61., train_accuracy=62., train_weighted_f1=63.,
            test_accuracy=None, test_weighted_f1=None, elapsed_seconds=1.,
            sample_ratio=round(.15 * (epoch + 1), 2) if variant == 'cmerc' else 1.,
            train_dialogues=int(round(.15 * (epoch + 1), 2) * data['train_count'])
                if variant == 'cmerc' else data['train_count'],
            max_gpu_memory_mb=32., **losses)
        with (run / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        coefficients = dict(classification_loss=1., cc_loss=args.rank_coff,
                            mscl_loss=args.mscl_coff, cscl_loss=args.cscl_coff)
        raw = {name: 1. for name in coefficients if name == 'classification_loss' or variant == 'cmerc'}
        diagnostic = dict(epoch=epoch, batch=0, finite_losses=True, finite_gradients=True,
            auxiliary_policy='legacy_epoch_snapshot', losses=losses,
            main_training=True, aux_training=False,
            aux_modes=dict(mm=False, intra=False, inter=False), main_aux_max_abs_diff=0.,
            gradient_coefficients=coefficients, gradient_norms=raw,
            weighted_gradient_norms={name: value * coefficients[name] for name, value in raw.items()},
            parameter_update_norms={'weight': .01})
        with (run / 'training_diagnostics.jsonl').open('a') as handle:
            handle.write(json.dumps(diagnostic) + '\n')
    return run, expected, args


def verify(run, expected, **kwargs):
    from scripts.round2_verify_run import verify_run
    return verify_run(run, expected, epochs=2, **kwargs)


def test_complete_baseline_and_earliest_tie_selection_have_public_report(tmp_path):
    run, expected, _ = fixture_run(tmp_path)
    result = verify(run, expected, variant='baseline')
    assert result['status'] == 'passed'
    assert result['best_epoch'] == 1 and result['completed_epochs'] == 2
    assert 'private-id' not in json.dumps(result) and 'train-a' not in json.dumps(result)
    assert result['diagnostics']['epochs_verified'] == 2


@pytest.mark.parametrize('corruption', ['gap', 'test_exposure', 'wrong_best', 'nonfinite',
                                      'diagnostic', 'initial_hash', 'weighted_gradient',
                                      'schedule', 'loss_composition', 'mode', 'lag'])
def test_rejects_incomplete_or_unsafe_run(tmp_path, corruption):
    run, expected, _ = fixture_run(tmp_path)
    path = run / 'metrics.jsonl'
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == 'gap':
        records.pop(0)
    elif corruption == 'test_exposure':
        records[0]['test_weighted_f1'] = 99.
    elif corruption == 'wrong_best':
        records[1]['validation_weighted_f1'] = records[1]['selection_weighted_f1'] = 61.
    elif corruption == 'nonfinite':
        records[0]['total_loss'] = None
    elif corruption == 'schedule':
        records[0]['train_dialogues'] += 1
    elif corruption == 'loss_composition':
        records[0]['total_loss'] = 3.
    elif corruption == 'diagnostic':
        (run / 'training_diagnostics.jsonl').write_text('')
    elif corruption == 'initial_hash':
        (run / 'initial_state.json').write_text(json.dumps({'model_state_sha256': 'a' * 64}))
    else:
        diagnostic_path = run / 'training_diagnostics.jsonl'
        diagnostics = [json.loads(line) for line in diagnostic_path.read_text().splitlines()]
        if corruption == 'weighted_gradient':
            diagnostics[0]['weighted_gradient_norms']['classification_loss'] = 999.
        elif corruption == 'mode':
            diagnostics[0]['aux_modes']['mm'] = True
        else:
            diagnostics[0]['main_aux_max_abs_diff'] = .1
        diagnostic_path.write_text(''.join(json.dumps(item) + '\n' for item in diagnostics))
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))
    with pytest.raises(ValueError):
        verify(run, expected, variant='baseline')


@pytest.mark.parametrize('overrides', [dict(graph_construct='full'), dict(use_residue=True),
    dict(av_using_lstm=True), dict(fail_on_nonfinite=False), dict(defer_final_test=False)])
def test_rejects_changed_frozen_official_settings(tmp_path, overrides):
    run, expected, _ = fixture_run(tmp_path, overrides=overrides)
    with pytest.raises(ValueError, match='protocol'):
        verify(run, expected, variant='baseline')


def test_full_pair_requires_actual_pinned_validation_teacher(tmp_path):
    baseline, expected, args = fixture_run(tmp_path)
    source = torch.load(baseline / 'best.pt', map_location='cpu')
    manifest = create_difficulty_manifest(baseline / 'best.pt', source,
        args.data_provenance, architecture_from_args(args))
    difficulty = write_causal_difficulty({'train-a': .2, 'train-b': .3}, manifest,
                                        output_dir=tmp_path / 'difficulty')
    full, _, _ = fixture_run(tmp_path, 'cmerc', difficulty=difficulty)
    result = verify(full, expected, variant='cmerc', paired_run=baseline, difficulty=difficulty)
    assert result['paired_initialization_verified'] is True
    assert result['difficulty']['teacher_best_sha256'] == result['paired_best_sha256']
    Path(difficulty).write_bytes(b'replaced')
    with pytest.raises(ValueError, match='SHA256'):
        verify(full, expected, variant='cmerc', paired_run=baseline, difficulty=difficulty)
