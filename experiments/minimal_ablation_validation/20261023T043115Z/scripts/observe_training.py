"""Observe frozen CMERC forwards without introducing model calls or RNG draws."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import linecache
import os
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
REMOVALS = {'text': 'log_prob2t', 'audio': 'log_prob2a', 'visual': 'log_prob2v',
            'intra_context': 'log_prob2intra', 'inter_context': 'log_prob2inter'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def append(path, value):
    with Path(path).open('a') as out:
        out.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_hash():
    import random
    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode())
    digest.update(repr(np.random.get_state()).encode())
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_initialized():
        for value in torch.cuda.get_rng_state_all():
            digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def assert_scientific_pair(args, historical, run, full_cc=False):
    actual = vars(args).copy()
    expected = historical['args'].copy()
    allowed = {'output_dir', 'provenance', run['factor_key']}
    if full_cc:
        assert run['dataset'] == 'MELD'
        assert actual['calibrate'] is True
    else:
        assert actual[run['factor_key']] == run['factor_value']
    differences = {key: (expected.get(key), actual.get(key))
                   for key in set(actual) | set(expected)
                   if key not in allowed and actual.get(key) != expected.get(key)}
    if differences:
        raise AssertionError('Unregistered scientific argument differences: ' + repr(differences))
    provenance = actual['provenance']
    assert provenance['split'] == historical['split'], 'Data split changed'
    assert provenance['data'] == historical['data'], 'Input hashes changed'
    source = provenance['source']['files']
    old_source = historical['source']['files']
    assert set(source) == set(old_source), 'Source file inventory changed'
    for name, digest in old_source.items():
        if run['dataset'] == 'IEMOCAP' and name == 'train_cm.py':
            continue
        assert source[name] == digest, 'Unexpected source change: ' + name
    assert actual['selection_protocol'] == 'validation'
    assert actual['rng_policy'] == 'continuous' and actual['rnn_backend'] == 'aten'
    assert actual['epochs'] == run['epochs']


class PreflightComplete(Exception):
    pass


class Observer:
    def __init__(self, run, historical, output, preflight=False, full_cc=False, target=None):
        self.run, self.historical, self.output = run, historical, Path(output)
        self.preflight, self.full_cc = preflight, full_cc
        self.target = str(Path(target or Path(run['code_dir']) / 'train_cm.py').resolve())
        self.completed = {'train': 0, 'validation': 0, 'test': 0}
        self.active = None
        self.validated = False
        self.first = None
        self.output.mkdir(parents=True, exist_ok=False)
        self.original_flag = None
        self.test_metrics = None

    def split_for(self, train, keys):
        split = self.historical['split']
        if train:
            assert keys and len(keys) == len(set(keys))
            assert set(keys).issubset(set(split['train_ids']))
            assert not set(keys).intersection(split['validation_ids'] + split['test_ids'])
            return 'train'
        if keys == split['validation_ids']:
            return 'validation'
        assert keys == split['test_ids'], 'Unexpected evaluation IDs'
        assert self.completed['train'] == self.run['epochs']
        assert self.completed['validation'] == self.run['epochs']
        assert self.completed['test'] == 0, 'Repeated test evaluation rejected'
        return 'test'

    def metrics(self, labels, full, removed):
        module_path = Path(self.run['code_dir']) / 'evaluation_metrics.py'
        spec = importlib.util.spec_from_file_location('frozen_evaluation_metrics', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pred = full.argmax(1)
        calibration = module.calibration_metrics(labels, np.exp(full))
        cel = module.confidence_enhancement_level(labels, full, removed) if removed else None
        return {'weighted_f1': float(f1_score(labels, pred, average='weighted') * 100),
                'macro_f1': float(f1_score(labels, pred, average='macro') * 100),
                'accuracy': float(accuracy_score(labels, pred) * 100),
                'ece': calibration['ece'], 'cel': None if cel is None else cel['total'],
                'cel_by_removal': None if cel is None else cel['by_removal'],
                'n_utterances': int(len(labels)), 'ece_bins': 15}

    def trace(self, frame, event, arg):
        if frame.f_code.co_filename != self.target or frame.f_code.co_name != 'train_or_eval_graph_model':
            return None
        local = frame.f_locals
        args = frame.f_globals['args']
        if event == 'call':
            assert self.active is None, 'Nested training function'
            if not self.validated:
                assert_scientific_pair(args, self.historical, self.run, self.full_cc)
                write_json(self.output / 'effective_args.json', vars(args))
                self.validated = True
            keys = list(local['dataloader'].dataset.keys)
            split = self.split_for(local['train'], keys)
            epoch = int(local['epoch']) + 1
            if split in ('train', 'validation'):
                assert epoch == self.completed[split] + 1, 'Nonsequential epoch'
            self.active = {'split': split, 'epoch': epoch, 'keys': keys, 'batches': [],
                           'pending': False, 'captured': False, 'seen_ids': [], 'components': []}
            if self.first is None and split == 'train':
                self.first = {'initial_model_sha256': state_hash(local['model']),
                              'rng_before_train': rng_hash()}
            if split == 'test' and not args.calibrate and not args.contrastlearning:
                self.original_flag = args.calibrate
                args.calibrate = True
                self.active['evaluation_only_removal'] = True
            return self.trace
        if self.active is None:
            return self.trace
        state = self.active
        if event == 'line':
            line = linecache.getline(self.target, frame.f_lineno).strip()
            if state['pending']:
                self.after_step(local)
                state['pending'] = False
            if line.startswith('classification_loss = loss_function('):
                self.capture(local)
            if line == 'loss_components.append(components)':
                values = {key: float(value) for key, value in local['components'].items()}
                assert all(np.isfinite(value) for value in values.values())
                state['components'].append(values)
                if state['split'] != 'train':
                    batch = state['batches'][-1]
                    append(self.output / 'observer_batches.jsonl', {
                        'split': state['split'], 'epoch': state['epoch'],
                        'batch': len(state['batches']), 'dialogue_ids': batch['ids'],
                        'lengths': batch['lengths'], **values})
            if line == 'optimizer.step()':
                state['pending'] = True
            return self.trace
        if event == 'return':
            # Exception unwinding also reports return(None); never count it as completion.
            if arg is None:
                return self.trace
            if state['pending']:
                self.after_step(local)
            full = np.concatenate([b['full'] for b in state['batches']])
            labels = np.concatenate([b['labels'] for b in state['batches']])
            removed = {name: np.concatenate([b['removed'][name] for b in state['batches']])
                       for name in REMOVALS if all(name in b['removed'] for b in state['batches'])}
            assert state['seen_ids'] == state['keys'] if state['split'] != 'train' else (
                len(state['seen_ids']) == len(state['keys']) and set(state['seen_ids']) == set(state['keys']))
            metrics = self.metrics(labels, full, removed)
            assert round(metrics['weighted_f1'], 2) == arg[4]
            assert round(metrics['accuracy'], 2) == arg[1]
            averaged = {key: float(np.mean([component[key] for component in state['components']]))
                        for key in state['components'][0]} if state['components'] else {}
            row = {'split': state['split'], 'epoch': state['epoch'], **metrics,
                   'loss_components': averaged,
                   'dialogue_ids': state['seen_ids'], 'classification_loss': float(arg[0]),
                   'cel_context': ('epoch_start_auxiliary_snapshot_online_training'
                                   if state['split'] == 'train' else 'fixed_evaluation_weights'),
                   'evaluation_only_removal': state.get('evaluation_only_removal', False)}
            append(self.output / 'observer_epochs.jsonl', row)
            self.completed[state['split']] += 1
            if state['split'] == 'test':
                self.test_metrics = row
                arrays = {'labels': labels, 'full_log_probabilities': full,
                          'log_probabilities': full, 'probabilities': np.exp(full),
                          'predictions': full.argmax(1)}
                arrays.update({name + '_log_probabilities': value for name, value in removed.items()})
                arrays.update({'removed_' + name + '_log_probabilities': value for name, value in removed.items()})
                np.savez_compressed(self.output / 'predictions.npz', **arrays)
                write_json(self.output / 'metrics.json', row)
                if self.original_flag is not None:
                    args.calibrate = self.original_flag
                    self.original_flag = None
            self.active = None
        return self.trace

    def capture(self, local):
        state = self.active
        full = local['log_prob'].detach().cpu().numpy().copy()
        labels = local['label'].detach().cpu().numpy().copy()
        removed = {name: local[key].detach().cpu().numpy().copy()
                   for name, key in REMOVALS.items() if key in local}
        assert np.isfinite(full).all()
        assert all(np.isfinite(x).all() for x in removed.values())
        ids = list(local['data'][-1])
        state['seen_ids'].extend(ids)
        state['batches'].append({'full': full, 'labels': labels, 'removed': removed,
                                 'ids': ids, 'lengths': [int(x) for x in local['lengths']]})
        if len(state['batches']) == 1 and state['split'] == 'train' and state['epoch'] == 1:
            self.first['rng_after_forward'] = rng_hash()
            self.first['model_before_step_sha256'] = state_hash(local['model'])
            self.first['parameters_before_step'] = [p.detach().cpu().clone()
                                                     for p in local['model'].parameters()]

    def after_step(self, local):
        state = self.active
        components = {key: float(value) for key, value in local['components'].items()}
        assert all(np.isfinite(x) for x in components.values()), 'Nonfinite loss'
        params = list(local['model'].parameters())
        gradients_finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in params)
        assert gradients_finite, 'Nonfinite gradient'
        batch = state['batches'][-1]
        append(self.output / 'observer_batches.jsonl', {'split': 'train', 'epoch': state['epoch'],
               'batch': len(state['batches']), 'dialogue_ids': batch['ids'], 'lengths': batch['lengths'],
               **components, 'gradients_finite': gradients_finite})
        if state['epoch'] == 1 and len(state['batches']) == 1:
            previous = self.first.pop('parameters_before_step')
            self.first.update({'model_after_step_sha256': state_hash(local['model']),
                'optimizer_updated_parameters': any(not torch.equal(a, b.detach().cpu())
                                                     for a, b in zip(previous, params)),
                'rng_after_step': rng_hash(), 'components': components,
                'dialogue_ids': batch['ids'], 'lengths': batch['lengths'],
                'gradients_finite': gradients_finite})
            assert self.first['optimizer_updated_parameters']
            write_json(self.output / 'first_batch.json', self.first)
            arrays = {'full_log_probabilities': batch['full'], 'labels': batch['labels']}
            arrays.update({key + '_log_probabilities': val for key, val in batch['removed'].items()})
            np.savez_compressed(self.output / 'first_batch.npz', **arrays)
            if self.preflight:
                raise PreflightComplete()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--full-cc-control', action='store_true')
    options = parser.parse_args()
    registry = json.loads((ROOT / 'configs/registry.json').read_text())
    run = next(r for r in registry['runs'] if r['run_id'] == options.run_id)
    assert not options.full_cc_control or (options.preflight and run['dataset'] == 'MELD')
    historical = json.loads(Path(run['historical_manifest']).read_text())
    argv = list(run['argv'])
    code = Path(run['code_dir']).resolve()
    assert ROOT in code.parents
    if options.preflight:
        sandbox = ROOT / 'preflight' / (run['run_id'] + ('_full_cc' if options.full_cc_control else ''))
        assert '--output-dir' in argv
        argv[argv.index('--output-dir') + 1] = str(sandbox / 'runs')
        output = sandbox / 'observer'
    else:
        sandbox = ROOT / 'results' / run['run_id']
        output = sandbox
        assert Path(run['run_dir']).resolve().is_relative_to(ROOT / 'runs')
        actual_output_root = Path(argv[argv.index('--output-dir') + 1]).resolve()
        assert actual_output_root.is_relative_to(ROOT / 'runs')
        expected_run = actual_output_root / (run['dataset'] + '_validation_seed' + str(historical['args']['seed']))
        assert expected_run == Path(run['run_dir']).resolve(), 'argv output disagrees with registry'
    if options.full_cc_control and '--calibrate' not in argv:
        argv.append('--calibrate')
    observer = Observer(run, historical, output, options.preflight, options.full_cc_control)
    write_json(output / 'launch.json', {'argv': argv, 'code_dir': str(code),
               'preflight': options.preflight, 'full_cc_control': options.full_cc_control,
               'observer_sha256': sha(__file__), 'started_unix': time.time()})
    old_cwd, old_argv = Path.cwd(), sys.argv
    old_trace = sys.gettrace()
    try:
        os.chdir(code)
        sys.path.insert(0, str(code))
        sys.argv = [str(code / 'train_cm.py'), *argv]
        sys.settrace(observer.trace)
        runpy.run_path(str(code / 'train_cm.py'), run_name='__main__')
    except PreflightComplete:
        assert options.preflight
        sys.settrace(old_trace)
        write_json(output / 'status.json', {'phase': 'preflight_passed', 'first_step_completed': True})
    else:
        sys.settrace(old_trace)
        assert observer.completed == {'train': run['epochs'], 'validation': run['epochs'], 'test': 1}
        final_test = json.loads((Path(run['run_dir']) / 'final_test.json').read_text())
        assert final_test['test_weighted_f1'] == round(observer.test_metrics['weighted_f1'], 2)
        assert final_test['test_accuracy'] == round(observer.test_metrics['accuracy'], 2)
        write_json(output / 'status.json', {'phase': 'completed', 'counts': observer.completed,
                   'test_forward_passes': 1, 'final_test_matches': True})
    finally:
        sys.settrace(old_trace)
        os.chdir(old_cwd)
        sys.argv = old_argv


if __name__ == '__main__':
    main()
