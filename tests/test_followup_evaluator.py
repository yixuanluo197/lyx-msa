"""Real-network evaluator regressions; synthetic validation data only."""
import importlib.util
import json
from pathlib import Path
import pytest
import torch
import numpy as np

SPEC = importlib.util.spec_from_file_location('followup_evaluator', Path(__file__).parents[1] / 'scripts/evaluate_checkpoint.py')
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)
torch.set_num_threads(1)


def checkpoint_args(base='GRU', residue=False, av=False):
    return dict(causal=True, Dataset='IEMOCAP', base_model=base, norm='LN',
                num_L=1, num_K=2, av_using_lstm=av, use_residue=residue,
                graph_type='hyper', modals='avl', mm_fusion_mthd='concat_DHT',
                use_speaker=True, use_modal=True, use_gcn=False,
                Deep_GCN_nlayers=4, multi_modal=True, dropout=0.0,
                architecture=dict(D_g=8, graph_hidden_size=8, D_m_a=6, D_m_v=5,
                                  n_speakers=2), replacement_policy='prefix_mean_v1')


def synthetic_batch():
    torch.manual_seed(310)
    return [*[torch.randn(3, 2, 1024) for _ in range(4)],
            torch.randn(3, 2, 5), torch.randn(3, 2, 6),
            torch.nn.functional.one_hot(torch.tensor([[0, 1], [1, 0], [0, 0]]), 2).float(),
            torch.tensor([[1., 1., 1.], [1., 0., 0.]]),
            torch.tensor([[0, 1, 2], [3, 0, 0]]), ['synthetic-validation-0', 'synthetic-validation-1']]


@pytest.mark.parametrize('raw,expected', [('cuda', 'cuda:2'), ('cuda:0', 'cuda:0'), ('cuda:1', 'cuda:1')])
def test_cuda_device_resolution_public_api(monkeypatch, raw, expected):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 3)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 2)
    assert str(E.resolve_device(raw, causal=True)) == expected


def test_cpu_and_invalid_devices(monkeypatch):
    assert E.resolve_device('cpu', causal=True) == torch.device('cpu')
    with pytest.raises(RuntimeError, match='official'):
        E.resolve_device('cpu', causal=False)
    with pytest.raises(ValueError, match='cpu|cuda'):
        E.resolve_device('meta', causal=True)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    with pytest.raises(ValueError, match='visible'):
        E.resolve_device('cuda:1', causal=True)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        E.resolve_device('cuda', causal=True)


@pytest.mark.parametrize('base', ['GRU', 'LSTM', 'Transformer'])
@pytest.mark.parametrize('residue', [False, True])
@pytest.mark.parametrize('av', [False, True])
def test_architecture_roundtrip_uses_strict_real_networks(tmp_path, base, residue, av):
    torch.manual_seed(91)
    config = E.dataset_config('IEMOCAP')
    checkpoint = {'args': checkpoint_args(base, residue, av)}
    source = E.build_models(config, torch.device('cpu'), causal=True, checkpoint=checkpoint)
    checkpoint['state_dict'] = source[0].state_dict()
    path = tmp_path / 'trusted-synthetic.pt'
    torch.save(checkpoint, path)
    loaded = torch.load(path, weights_only=False)
    models = E.build_models(config, torch.device('cpu'), causal=True, checkpoint=loaded)
    models[0].load_state_dict(E.extract_state_dict(loaded), strict=True)
    actual = E.run_inference(models, [synthetic_batch()], torch.device('cpu'))
    expected = E.run_inference(source, [synthetic_batch()], torch.device('cpu'))
    torch.testing.assert_close(torch.from_numpy(actual['log_probabilities']), torch.from_numpy(expected['log_probabilities']), atol=0, rtol=0)
    for view in expected['removed_log_probabilities']:
        np.testing.assert_array_equal(actual['removed_log_probabilities'][view],
                                      expected['removed_log_probabilities'][view])
    assert models[0].base_model == base
    assert models[0].use_residue == residue
    assert models[0].av_using_lstm == av
    bad = dict(loaded['state_dict'])
    bad.pop(next(iter(bad)))
    with pytest.raises(RuntimeError, match='Missing key'):
        models[0].load_state_dict(bad, strict=True)


@pytest.mark.parametrize('key,value', [('base_model','BERT'), ('num_L',True), ('num_K',0), ('use_residue','true'), ('norm','BN'), ('graph_type','GCN3'), ('replacement_policy','zero')])
def test_invalid_architecture_rejected(key, value):
    args = checkpoint_args()
    args[key] = value
    with pytest.raises(ValueError, match=key):
        E.causal_model_arguments(E.dataset_config('IEMOCAP'), torch.device('cpu'), {'args': args})


def test_missing_or_unsafe_architecture_rejected():
    for args in ({'causal': True}, {**checkpoint_args(), 'architecture': {'__class__': 'something'}},
                 {**checkpoint_args(), 'architecture': {'D_g': -1}},
                 {**checkpoint_args('Transformer'), 'architecture': {'D_g': 10}}):
        with pytest.raises(ValueError):
            E.causal_model_arguments(E.dataset_config('IEMOCAP'), torch.device('cpu'), {'args': args})


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_main_entrypoint_with_synthetic_validation_loader(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; no CUDA main execution claimed')
    config = E.dataset_config('IEMOCAP')
    checkpoint = {'args': {**checkpoint_args('LSTM', True, True), 'split_seed': 310}}
    models = E.build_models(config, torch.device('cpu'), causal=True, checkpoint=checkpoint)
    checkpoint['state_dict'] = models[0].state_dict()
    path = tmp_path / 'synthetic-validation.pt'
    torch.save(checkpoint, path)
    calls = []
    def loader(*args, **kwargs):
        calls.append(kwargs)
        return [synthetic_batch()]
    monkeypatch.setattr(E, 'build_test_loader', loader)
    output = tmp_path / device
    command = ['--dataset', 'IEMOCAP', '--causal', '--checkpoint', str(path),
               '--output-dir', str(output), '--split', 'validation', '--max-dialogues', '2',
               '--diagnostic-data-reason', 'synthetic validation loader fixture']
    if device == 'cpu':
        command += ['--device', 'cpu']
    # GPU branch deliberately exercises the CLI default --device cuda.
    assert E.main(command) == 0
    payload = json.loads((output / 'metrics.json').read_text())
    assert payload['replacement_policy'] == 'prefix_mean_v1'
    assert payload['architecture']['base_model'] == 'LSTM'
    assert payload['split'] == 'validation'
    assert payload['num_samples'] == 4
    assert calls[0]['split'] == 'validation'
    assert calls[0]['max_dialogues'] == 2
    assert calls[0]['split_seed'] == 310
    assert payload['evaluation_data']['verification'] == 'diagnostic_unverified'
    actual = np.load(output / 'predictions.npz')['log_probabilities']
    reference = E.run_inference(models, [synthetic_batch()], torch.device('cpu'))['log_probabilities']
    difference = np.abs(actual - reference)
    print(json.dumps({'check': 'main-entry-vs-cpu-reference', 'device': device,
                      'dtype': str(actual.dtype), 'seed': 310,
                      'max_abs_diff': float(difference.max()),
                      'max_rel_diff': float((difference / np.maximum(np.abs(reference), 1e-12)).max())}))
    np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=1e-5)


def test_split_seed_default_is_not_silent_zero():
    args = E.parse_args(['--dataset', 'IEMOCAP', '--causal', '--checkpoint', 'x', '--output-dir', 'y'])
    assert args.split_seed is None


def test_checkpoint_split_seed_is_inherited_and_conflicting_override_rejected():
    config = E.dataset_config('IEMOCAP')
    checkpoint = {'args': {**checkpoint_args(), 'split_seed': 1475}}
    resolved = E.resolve_evaluation_data(config, checkpoint, 'validation',
                                         diagnostic_data_reason='synthetic loader fixture')
    assert resolved['split_seed'] == 1475
    assert resolved['split_seed_source'] == 'checkpoint.args.split_seed'
    with pytest.raises(ValueError, match='split.seed'):
        E.resolve_evaluation_data(config, checkpoint, 'validation', split_seed=0,
                                  diagnostic_data_reason='synthetic loader fixture')


def test_missing_split_source_requires_explicit_diagnostic_configuration():
    config = E.dataset_config('IEMOCAP')
    checkpoint = {'args': checkpoint_args()}
    with pytest.raises(ValueError, match='split.seed'):
        E.resolve_evaluation_data(config, checkpoint, 'validation')
    with pytest.raises(ValueError, match='provenance'):
        E.resolve_evaluation_data(config, checkpoint, 'validation', split_seed=91)
    resolved = E.resolve_evaluation_data(config, checkpoint, 'validation', split_seed=91,
                                         diagnostic_data_reason='synthetic loader fixture')
    assert resolved['split_seed'] == 91
    assert resolved['verification'] == 'diagnostic_unverified'


def test_checkpoint_provenance_is_verified_before_any_subset(monkeypatch):
    import causal_provenance as P
    config = E.dataset_config('IEMOCAP')
    data = {'dataset': 'IEMOCAP', 'selection_protocol': 'validation', 'split_seed': 1475,
            'feature_files': {'real.pkl': 'a' * 64}, 'train_ids_sha256': 'b' * 64,
            'validation_ids_sha256': 'c' * 64, 'train_count': 10, 'validation_count': 1,
            'schema_version': 1}
    checkpoint = {'args': checkpoint_args(), 'data_provenance': data}
    resolved = E.resolve_evaluation_data(config, checkpoint, 'validation')
    calls = []
    def actual(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(data)
    monkeypatch.setattr(P, 'data_provenance_from_files', actual)
    verified = E.verify_evaluation_data(config, resolved)
    assert verified['verification'] == 'checkpoint_provenance_verified'
    assert calls[0][0] == ('IEMOCAP', 1475)
    assert calls[0][1]['selection_protocol'] == 'validation'
    changed = {**data, 'feature_files': {'real.pkl': 'd' * 64}}
    monkeypatch.setattr(P, 'data_provenance_from_files', lambda *a, **k: changed)
    with pytest.raises(ValueError, match='provenance'):
        E.verify_evaluation_data(config, resolved)
    changed = {**data, 'validation_ids_sha256': 'e' * 64}
    monkeypatch.setattr(P, 'data_provenance_from_files', lambda *a, **k: changed)
    with pytest.raises(ValueError, match='provenance'):
        E.verify_evaluation_data(config, resolved)
    with pytest.raises(ValueError, match='split.seed'):
        E.resolve_evaluation_data(config, {**checkpoint, 'args': {**checkpoint_args(), 'split_seed': 0}}, 'validation')


def test_legacy_split_seed_is_explicitly_identified():
    config = E.dataset_config('IEMOCAP')
    checkpoint = {'args': {**checkpoint_args(), 'seed': 1475, 'selection_protocol': 'validation'}}
    resolved = E.resolve_evaluation_data(config, checkpoint, 'validation',
                                         diagnostic_data_reason='legacy metadata audit')
    assert resolved['split_seed'] == 1475
    assert resolved['split_seed_source'] == 'legacy_checkpoint.args.seed'
    # Official test selection does not describe a held-out validation split.
    checkpoint['args']['selection_protocol'] = 'official_test'
    with pytest.raises(ValueError, match='selection_protocol'):
        E.resolve_evaluation_data(config, checkpoint, 'validation', split_seed=1475,
                                  diagnostic_data_reason='must not override known teacher provenance')


def test_real_feature_bytes_and_split_ids_provenance_are_checked(tmp_path, monkeypatch):
    import pickle
    import causal_provenance as P
    config = E.dataset_config('IEMOCAP')
    metadata, features = tmp_path / 'IEMOCAP_features.pkl', tmp_path / 'iemocap_features_roberta.pkl'
    # Minimal legal metadata for the real split/provenance helper; no dataset is opened.
    ids = ['synthetic-train-' + str(index) for index in range(20)]
    metadata.write_bytes(pickle.dumps([None] * 7 + [ids, ['synthetic-test-unused']]))
    features.write_bytes(b'synthetic feature version 1')
    paths = {'IEMOCAP_features.pkl': str(metadata), 'iemocap_features_roberta.pkl': str(features)}
    monkeypatch.setattr(P, 'dataset_feature_paths', lambda *a, **k: paths)
    provenance = P.data_provenance_from_files('IEMOCAP', 1475)
    checkpoint = {'args': {**checkpoint_args(), 'split_seed': 1475, 'selection_protocol': 'validation',
                           'data_provenance': provenance}}
    resolved = E.resolve_evaluation_data(config, checkpoint, 'validation')
    assert E.verify_evaluation_data(config, resolved)['verification'] == 'checkpoint_provenance_verified'
    with pytest.raises(ValueError, match='provenance'):
        E.verify_evaluation_data(config, {**resolved, 'valid_fraction': .2})
    features.write_bytes(b'synthetic feature version 2')
    with pytest.raises(ValueError, match='provenance'):
        E.verify_evaluation_data(config, resolved)
