import json
import os
import pickle
import random
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

import train_cm as training
from test_training_protocol import _FakeIemocapDataset, _FakeMeldDataset
from dataloader import split_iemocap_dialogues, select_curriculum_keys, resolve_meld_split


def test_capacity_sampler_preserves_first_safe_legacy_permutation_and_rng():
    assert hasattr(training, 'CapacitySafeBatchSampler')
    lengths = [10 + index for index in range(32)]
    dataset = list(range(len(lengths)))
    legacy_generator = torch.Generator().manual_seed(67142)
    safe_generator = torch.Generator().manual_seed(67142)
    legacy = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True,
                                         generator=legacy_generator)
    sampler = training.CapacitySafeBatchSampler(lengths, 16, safe_generator)
    safe = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, generator=safe_generator)
    global_rng = torch.get_rng_state().clone()
    assert [batch.tolist() for batch in safe] == [batch.tolist() for batch in legacy]
    assert sampler.retries == 0
    assert torch.equal(torch.get_rng_state(), global_rng)


def test_capacity_sampler_rejects_unsafe_candidate_without_dropping_samples():
    assert hasattr(training, 'CapacitySafeBatchSampler')
    lengths = [130] * 8 + [20] * 24
    observed_retries = []
    for seed in range(20):
        sampler = training.CapacitySafeBatchSampler(lengths, 16,
                                                     torch.Generator().manual_seed(seed))
        batches = list(sampler)
        assert sorted(sum(batches, [])) == list(range(32))
        assert all(len(batch) == 16 for batch in batches)
        assert max(sum(lengths[i] for i in batch) + 3 * len(batch) for batch in batches) <= 1000
        assert sampler.max_batch_edges <= 1000
        observed_retries.append(sampler.retries)
    assert max(observed_retries) > 0


def test_capacity_sampler_refuses_impossible_batch_size_instead_of_shrinking():
    assert hasattr(training, 'CapacitySafeBatchSampler')
    with pytest.raises(ValueError, match='capacity'):
        training.CapacitySafeBatchSampler([100] * 16, 16, torch.Generator())
    with pytest.raises(ValueError, match='capacity'):
        training.CapacitySafeBatchSampler([1001, 1, 1, 1], 16, torch.Generator())


def test_capacity_retry_limit_fails_before_yielding_any_partial_epoch(monkeypatch):
    lengths = [130] * 8 + [20] * 24
    monkeypatch.setattr(torch, 'randperm', lambda count, generator: torch.arange(count))
    sampler = training.CapacitySafeBatchSampler(lengths, 16, torch.Generator(), max_retries=1)
    with pytest.raises(RuntimeError, match='exceeded 1 retries'):
        next(iter(sampler))
    assert sampler.retries is None


@pytest.mark.skipif(not Path('data/IEMOCAP_features.pkl').exists(),
                    reason='trusted official feature files required for real-length integration')
def test_real_iemocap_capacity_all_five_seeds_eighty_epochs_full_and_curriculum():
    with Path('data/IEMOCAP_features.pkl').open('rb') as handle:
        feature = pickle.load(handle, encoding='latin1')
    labels, all_train_ids, test_ids = feature[2], feature[7], feature[8]
    train_ids, validation_ids = split_iemocap_dialogues(all_train_ids, 1475)
    with Path('caldiff/IEMOCAP_diff.pkl').open('rb') as handle:
        difficulty = pickle.load(handle)
    assert not set(train_ids) & set(validation_ids)
    assert not set(train_ids) & set(test_ids)
    assert len(train_ids) == 108
    global_rng = torch.get_rng_state().clone()
    for seed in [1475, 67137, 2024, 3407, 42]:
        for epoch in range(80):
            for ratio in [-1., min(1., round(.15 * (epoch + 1), 2))]:
                eligible = train_ids if ratio == -1 else select_curriculum_keys(train_ids, difficulty, ratio)
                lengths = [len(labels[key]) for key in eligible]
                generator = torch.Generator().manual_seed(seed + epoch)
                # Match DataLoader's base-seed draw before RandomSampler starts.
                torch.empty((), dtype=torch.int64).random_(generator=generator)
                sampler = training.CapacitySafeBatchSampler(lengths, 16, generator)
                batches = list(sampler)
                assert sorted(sum(batches, [])) == list(range(len(eligible)))
                assert all(len(batch) == 16 for batch in batches[:-1])
                assert 0 < len(batches[-1]) <= 16
                assert sampler.max_batch_edges <= 1000
                assert sampler.max_batch_nodes <= 5200
                assert sampler.retries <= 10000
                if seed == 67137 and epoch == 5 and ratio == -1:
                    assert sampler.retries > 0  # Exact real-data V1 failure.
    assert torch.equal(torch.get_rng_state(), global_rng)


@pytest.mark.skipif(not Path('data/MELD_features_raw1.pkl').exists(),
                    reason='trusted official feature files required for real-length integration')
def test_real_meld_capacity_all_five_seeds_forty_epochs_full_and_curriculum():
    with Path('data/MELD_features_raw1.pkl').open('rb') as handle:
        labels = pickle.load(handle)[2]
    with Path('data/meld_features_roberta.pkl').open('rb') as handle:
        roberta = pickle.load(handle, encoding='latin1')
    train_ids, test_ids, validation_ids = roberta[8:]
    del roberta
    train_ids = resolve_meld_split(train_ids, validation_ids, test_ids, 'train')
    with Path('caldiff/MELD_diff.pkl').open('rb') as handle:
        difficulty = pickle.load(handle)
    assert len(train_ids) == 1038
    assert not set(train_ids) & set(validation_ids)
    assert not set(train_ids) & set(test_ids)
    for seed in [1475, 67137, 2024, 3407, 42]:
        for epoch in range(40):
            for ratio in [-1., min(1., round(.4 * (epoch + 1), 2))]:
                eligible = train_ids if ratio == -1 else select_curriculum_keys(train_ids, difficulty, ratio)
                generator = torch.Generator().manual_seed(seed + epoch)
                torch.empty((), dtype=torch.int64).random_(generator=generator)
                sampler = training.CapacitySafeBatchSampler(
                    [len(labels[key]) for key in eligible], 16, generator)
                batches = list(sampler)
                assert sorted(sum(batches, [])) == list(range(len(eligible)))
                assert sampler.max_batch_edges <= 1000
                assert sampler.max_batch_nodes <= 5200
                assert sampler.retries == 0


def test_fixed_split_is_identical_across_training_seeds_and_pairs():
    with patch('train_cm.IEMOCAPDataset', _FakeIemocapDataset):
        loaders = [training.get_IEMOCAP_loaders(
            batch_size=4, selection_protocol='validation', seed=seed,
            split_seed=1475, epoch=epoch, rng_policy='continuous')
            for seed, epoch in [(1475, 0), (2024, 0), (2024, 1)]]
    assert loaders[0][0].dataset.keys == loaders[1][0].dataset.keys
    assert loaders[0][1].dataset.keys == loaders[2][1].dataset.keys
    assert list(loaders[1][0].sampler) != list(loaders[2][0].sampler)
    with patch('train_cm.MELDDataset', _FakeMeldDataset):
        first = training.get_MELD_loaders(selection_protocol='validation', seed=1,
                                         split_seed=1475, rng_policy='continuous')
        second = training.get_MELD_loaders(selection_protocol='validation', seed=2,
                                          split_seed=42, rng_policy='continuous')
    assert first[0].dataset.keys == second[0].dataset.keys
    assert first[1].dataset.keys == second[1].dataset.keys


def test_legacy_split_defaults_to_training_seed():
    with patch('train_cm.IEMOCAPDataset', _FakeIemocapDataset):
        first = training.get_IEMOCAP_loaders(selection_protocol='validation', seed=1)
        second = training.get_IEMOCAP_loaders(selection_protocol='validation', seed=2)
    assert first[1].dataset.keys != second[1].dataset.keys


def test_eval_restores_every_rng_and_train_advances_it():
    assert hasattr(training, 'preserve_rng_for_evaluation')
    training.seed_everything(5)
    saved = training._rng_state()
    with training.preserve_rng_for_evaluation(True):
        random.random(), np.random.rand(), torch.rand(3)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    training._restore_rng_state(saved)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])
    with training.preserve_rng_for_evaluation(False):
        torch.rand(3)
    assert not torch.equal(training._rng_state()['torch'], saved['torch'])


def test_graph_training_continues_dropout_rng_and_evaluation_restores(monkeypatch):
    class ToyGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)
            self.dropout = torch.nn.Dropout(.5)
        def forward(self, text, qmask, umask, lengths, audio, visual, epoch):
            hidden = self.dropout(text[0][:, 0])
            return torch.log_softmax(self.linear(hidden), -1), None, None, None, None, hidden
    config = Namespace(seed=17, rng_policy='continuous', multi_modal=True,
                       mm_fusion_mthd='concat_DHT', contrastlearning=False,
                       calibrate=False, rank_coff=0., mscl_coff=0., cscl_coff=0.)
    monkeypatch.setattr(training, 'args', config, raising=False)
    training.seed_everything(17)
    model = ToyGraph()
    replicas = [ToyGraph() for _ in range(3)]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.)
    batch = [torch.ones(5, 1, 3) for _ in range(7)]
    batch += [torch.ones(1, 5), torch.zeros(1, 5, dtype=torch.long), ['real-shape-dialogue']]
    def call(train):
        return training.train_or_eval_graph_model(
            model, *replicas, torch.nn.NLLLoss(), [batch], 0, False, 'avl',
            optimizer=optimizer if train else None, train=train)
    first = call(True)[0]
    state = training._rng_state()
    call(False)
    torch.testing.assert_close(training._rng_state()['torch'], state['torch'], rtol=0, atol=0)
    second = call(True)[0]
    assert first != second


def test_loader_workers_replay_python_numpy_draws_for_same_epoch(monkeypatch):
    class RandomDataset(_FakeIemocapDataset):
        def __getitem__(self, index):
            return self.keys[index], random.random(), float(np.random.rand())
    monkeypatch.setattr(training, 'IEMOCAPDataset', RandomDataset)
    def draw(epoch):
        loader, _, _ = training.get_IEMOCAP_loaders(
            batch_size=4, num_workers=2, selection_protocol='validation',
            seed=2024, split_seed=1475, epoch=epoch, rng_policy='continuous')
        return list(loader)
    assert draw(0) == draw(0)
    assert draw(0) != draw(1)


def test_new_run_rejects_existing_outputs_and_live_lock(tmp_path):
    assert hasattr(training, 'prepare_run_directory')
    run = tmp_path / 'run_seed1'
    lock = training.prepare_run_directory(run)
    with pytest.raises((FileExistsError, RuntimeError)):
        training.prepare_run_directory(run)
    lock.close()
    (run / 'metrics.jsonl').write_text('{"epoch":1}\n')
    with pytest.raises(FileExistsError):
        training.prepare_run_directory(run)


def test_resume_rejects_log_epoch_mismatch(tmp_path):
    assert hasattr(training, 'prepare_run_directory')
    run = tmp_path / 'run'
    run.mkdir()
    checkpoint = run / 'latest.pt'
    torch.save({'epoch': 1}, checkpoint)
    (run / 'metrics.jsonl').write_text('{"epoch":1}\n')
    with pytest.raises(ValueError, match='epoch'):
        training.prepare_run_directory(run, resume=checkpoint)


def test_continuous_resume_restores_exact_next_dropout_update(tmp_path):
    training.seed_everything(2024)
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(.5))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    args = {'causal': False, 'seed': 2024, 'split_seed': 1475,
            'rng_policy': 'continuous', 'batch_size': 16, 'lr': .01,
            'provenance': {'split': {'hash': 'fixed'}, 'source': {}, 'data': {}}}
    x = torch.ones(4, 3)
    def step():
        optimizer.zero_grad()
        loss = model(x).square().sum()
        loss.backward()
        optimizer.step()
        return loss.detach().clone()
    step()
    checkpoint = tmp_path / 'latest.pt'
    training.save_training_checkpoint(checkpoint, model, optimizer, 0, args, 1., 0)
    expected_loss = step()
    expected = [p.detach().clone() for p in model.parameters()]
    training.load_training_checkpoint(checkpoint, model, optimizer,
                                      map_location='cpu', expected_args=args)
    torch.testing.assert_close(step(), expected_loss, rtol=0, atol=0)
    for actual, wanted in zip(model.parameters(), expected):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    incompatible = dict(args, split_seed=42)
    with pytest.raises(ValueError, match='split_seed'):
        training.load_training_checkpoint(checkpoint, model, optimizer,
                                          expected_args=incompatible)


@pytest.mark.parametrize('changed', ['seed', 'rng_policy', 'rnn_backend', 'batch_size', 'lr',
                                    'batch_policy', 'edge_capacity', 'node_capacity',
                                    'batch_max_retries', 'provenance'])
def test_noncausal_resume_rejects_protocol_change_before_loading(tmp_path, changed):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    args = {'causal': False, 'seed': 1, 'split_seed': 1475,
            'rng_policy': 'continuous', 'rnn_backend': 'aten', 'batch_size': 16, 'lr': .001,
            'batch_policy': 'capacity_safe', 'edge_capacity': 1000,
            'node_capacity': 5200, 'batch_max_retries': 10000,
            'provenance': {'split': {'hash': 'a'}, 'source': {}, 'data': {}}}
    path = tmp_path / 'latest.pt'
    training.save_training_checkpoint(path, model, optimizer, 0, args, 1., 0)
    incompatible = dict(args)
    incompatible[changed] = 'changed'
    with pytest.raises(ValueError, match=changed):
        training.load_training_checkpoint(path, model, optimizer, expected_args=incompatible)


def test_recurrent_backend_wraps_only_recurrent_forward_and_preserves_state_dict():
    model = torch.nn.ModuleDict({'gru': torch.nn.GRU(3, 5, num_layers=2, dropout=.5),
                                'bn': torch.nn.BatchNorm1d(5)})
    state_keys = set(model.state_dict())
    seen = []
    model['gru'].forward = lambda *args: seen.append(('gru', torch.backends.cudnn.enabled))
    model['bn'].forward = lambda *args: seen.append(('bn', torch.backends.cudnn.enabled))
    with torch.backends.cudnn.flags(enabled=True):
        training.configure_recurrent_backend(model, 'aten')
        wrapped = model['gru'].forward
        training.configure_recurrent_backend(model, 'aten')
        assert model['gru'].forward is wrapped
        model['gru'](torch.ones(1))
        model['bn'](torch.ones(1))
        assert torch.backends.cudnn.enabled
    assert seen == [('gru', False), ('bn', True)]
    assert set(model.state_dict()) == state_keys


@pytest.mark.skipif(os.environ.get('CMERC_GPU_RESUME_TEST') != '1',
                    reason='explicit healthy single GPU required for process-restart RNN test')
def test_cuda_rnn_dropout_resume_across_fresh_processes(tmp_path):
    code = '''
import sys
from argparse import Namespace
import torch
import train_cm as training
mode, output = sys.argv[1:]
training.args = Namespace(seed=2024, rng_policy='continuous', rnn_backend='aten', causal=False)
training.seed_everything(2024)
model = torch.nn.GRU(3, 5, num_layers=2, dropout=.5).cuda()
training.configure_recurrent_backend(model, training.args.rnn_backend)
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
data = torch.ones(7, 2, 3, device='cuda')
@training.isolated_evaluation
def step(train=True):
    optimizer.zero_grad()
    loss = model(data)[0].square().sum()
    loss.backward()
    optimizer.step()
if mode == 'resume':
    training.load_training_checkpoint(output, model, optimizer, map_location='cpu',
                                      expected_args=training.args)
    step()
else:
    step()
    if mode == 'full':
        step()
training.save_training_checkpoint(output, model, optimizer,
                                 1 if mode != 'partial' else 0, training.args, 0., 0)
'''
    full = tmp_path / 'full.pt'
    interrupted = tmp_path / 'interrupted.pt'
    for mode, path in [('full', full), ('partial', interrupted), ('resume', interrupted)]:
        subprocess.run([sys.executable, '-c', code, mode, str(path)], check=True)
    reference = torch.load(full, map_location='cpu')
    actual = torch.load(interrupted, map_location='cpu')
    assert torch.equal(reference['rng_state']['torch'], actual['rng_state']['torch'])
    assert all(torch.equal(a, b) for a, b in zip(reference['rng_state']['cuda'], actual['rng_state']['cuda']))
    for key, value in reference['model_state_dict'].items():
        torch.testing.assert_close(value, actual['model_state_dict'][key], rtol=0, atol=0)
