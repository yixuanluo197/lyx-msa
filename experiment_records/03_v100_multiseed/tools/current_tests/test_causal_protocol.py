"""Synthetic CPU checks for causal inputs and checkpoint/curriculum isolation."""
import importlib.util
import pickle
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader import causal_mean_replacement, load_curriculum_difficulties, validate_checkpoint_protocol


def test_prefix_mean_ignores_future_other_dialogues_and_padding():
    features = torch.tensor([[[2.], [10.]], [[4.], [20.]], [[8.], [999.]]])
    mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
    expected = torch.tensor([[[2.], [10.]], [[3.], [15.]], [[14./3.], [0.]]])
    output = causal_mean_replacement(features, mask)
    torch.testing.assert_close(output, expected)
    changed = features.clone()
    changed[2] = -1000
    changed[:, 1] += 10000
    torch.testing.assert_close(causal_mean_replacement(changed, mask)[:2, 0], output[:2, 0])
    torch.testing.assert_close(causal_mean_replacement(features[:2], mask[:, :2]), output[:2])


def test_prefix_mean_gradient_has_no_future_dependency():
    features = torch.randn(4, 1, 3, requires_grad=True)
    causal_mean_replacement(features, torch.ones(1, 4))[1].sum().backward()
    torch.testing.assert_close(features.grad[2:], torch.zeros_like(features.grad[2:]))
    assert features.grad[:2].abs().sum() > 0


@pytest.mark.parametrize('checkpoint, causal', [
    ({'weight': torch.ones(1)}, True),
    ({'args': {'causal': False, 'norm': 'LN'}}, True),
    ({'args': {'causal': True, 'norm': 'LN'}}, False),
    ({'causal': False, 'args': {'causal': True, 'norm': 'LN'}}, True),
])
def test_checkpoint_temporal_protocol_mismatch_is_rejected(checkpoint, causal):
    with pytest.raises(ValueError, match='causal'):
        validate_checkpoint_protocol(checkpoint, causal=causal)


def test_causal_checkpoint_requires_effective_token_normalization():
    with pytest.raises(ValueError, match='norm=LN'):
        validate_checkpoint_protocol({'args': {'causal': True, 'norm': 'BN'}}, causal=True)


@pytest.mark.parametrize('key', ['base_model', 'norm', 'num_L', 'num_K', 'av_using_lstm', 'use_residue'])
def test_checkpoint_architecture_mismatch_is_rejected(key):
    args = {'causal': True, 'base_model': 'GRU', 'norm': 'LN', 'num_L': 3,
            'num_K': 4, 'av_using_lstm': False, 'use_residue': False}
    checkpoint = {'args': dict(args)}
    checkpoint['args'][key] = 'different'
    with pytest.raises(ValueError, match=key):
        validate_checkpoint_protocol(checkpoint, causal=True, expected_args=args)


def test_causal_checkpoint_missing_architecture_metadata_is_rejected():
    with pytest.raises(ValueError, match='missing num_L'):
        validate_checkpoint_protocol({'args': {'causal': True, 'norm': 'LN'}},
                                     causal=True, expected_args={'norm': 'LN', 'num_L': 3})


def test_legacy_baseline_state_dict_remains_accepted():
    validate_checkpoint_protocol({'linear.weight': torch.ones(2, 2)},
                                 expected_args={'norm': 'BN', 'base_model': 'GRU'})


def test_curriculum_files_are_separate_and_causal_wrapper_is_required(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'caldiff').mkdir()
    baseline = {'dialogue-a': 0.1}
    causal = {'dialogue-a': 0.7}
    with open('caldiff/IEMOCAP_diff.pkl', 'wb') as handle:
        pickle.dump(baseline, handle)
    with pytest.raises(FileNotFoundError):
        load_curriculum_difficulties('IEMOCAP', causal=True)
    with open('caldiff/IEMOCAP_causal_diff.pkl', 'wb') as handle:
        pickle.dump({'causal': True, 'dataset': 'IEMOCAP', 'difficulties': causal}, handle)
    assert load_curriculum_difficulties('IEMOCAP', causal=True) == causal
    assert load_curriculum_difficulties('IEMOCAP') == baseline
    with open('caldiff/IEMOCAP_causal_diff.pkl', 'wb') as handle:
        pickle.dump(baseline, handle)
    with pytest.raises(ValueError, match='causal difficulty wrapper'):
        load_curriculum_difficulties('IEMOCAP', causal=True)


def test_eval_cli_and_arguments_enable_causal_ln_on_cpu():
    path = Path(__file__).resolve().parents[1] / 'scripts/evaluate_checkpoint.py'
    spec = importlib.util.spec_from_file_location('causal_evaluator_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(['--dataset', 'IEMOCAP', '--checkpoint', 'future.pt',
                              '--output-dir', 'out', '--device', 'cpu', '--causal'])
    assert args.causal
    config = module.dataset_config(args.dataset)
    causal = module._model_arguments(config, torch.device('cpu'), causal=True)
    baseline = module._model_arguments(config, torch.device('cuda'))
    assert causal['causal'] is True and causal['norm'] == 'LN' and causal['no_cuda']
    assert 'causal' not in baseline and baseline['norm'] == 'BN'
    text = [torch.randn(3, 1, 2) for _ in range(4)]
    mask = torch.ones(1, 3)
    masked = module._masked_modalities(text, text[0], text[1], umask=mask, causal=True)
    shortened = module._masked_modalities([t[:2] for t in text], text[0][:2], text[1][:2],
                                         umask=mask[:, :2], causal=True)
    for full, prefix in zip(masked[0] + list(masked[1:]), shortened[0] + list(shortened[1:])):
        torch.testing.assert_close(full[:2], prefix)


def test_resume_rejects_causal_change_before_modifying_parameters(tmp_path):
    from train_cm import save_training_checkpoint, load_training_checkpoint
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / 'baseline.pt'
    save_training_checkpoint(path, model, optimizer, 2, {'causal': False}, None, None)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    with pytest.raises(ValueError, match='causal'):
        load_training_checkpoint(path, model, optimizer, map_location='cpu',
                                 expected_args={'causal': True, 'norm': 'LN'})
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name])


@pytest.mark.parametrize('key', ['seed', 'dropout', 'batch_size', 'lr', 'l2',
                                 'calibrate', 'rank_coff', 'contrastlearning',
                                 'mscl_coff', 'cscl_coff', 'courselearning',
                                 'epoch_ratio', 'scheduler_steps'])
def test_causal_resume_rejects_changed_training_protocol(tmp_path, key):
    from train_cm import save_training_checkpoint, load_training_checkpoint
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    saved = {'causal': True, 'norm': 'LN', key: 1}
    requested = {**saved, key: 2}
    path = tmp_path / 'causal.pt'
    save_training_checkpoint(path, model, optimizer, 2, saved, None, None)
    with pytest.raises(ValueError, match=key):
        load_training_checkpoint(path, model, optimizer, map_location='cpu',
                                 expected_args=requested)


def test_causal_checkpoint_roundtrip_and_architecture_rejection(tmp_path):
    from train_cm import save_training_checkpoint, load_training_checkpoint
    from caldiff.cal_diff import load_checkpoint
    args = {'causal': True, 'norm': 'LN', 'num_L': 3, 'num_K': 4,
            'base_model': 'GRU', 'av_using_lstm': False, 'use_residue': False}
    model = torch.nn.Linear(2, 1)
    model.causal = True
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / 'causal.pt'
    save_training_checkpoint(path, model, optimizer, 2, args, 10.0, 2)
    expected = model.weight.detach().clone()
    model.weight.data.zero_()
    load_checkpoint(model, path, torch.device('cpu'), expected_args=args)
    torch.testing.assert_close(model.weight, expected)
    restored = load_training_checkpoint(path, model, optimizer, map_location='cpu', expected_args=args)
    assert restored['start_epoch'] == 3
    with pytest.raises(ValueError, match='num_K'):
        load_checkpoint(model, path, torch.device('cpu'), expected_args={**args, 'num_K': 5})
