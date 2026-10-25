"""Call/return-only guard for validation-only CMERC dropout diagnosis."""
from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import runpy
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
LOSSES = ('classification_loss', 'cc_loss', 'mscl_loss', 'cscl_loss', 'total_loss')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng_hash():
    digest = hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state())))
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_initialized():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_skip_patch(path, historical_sha):
    """Reverse only the registered three textual changes and check the old SHA."""
    raw = Path(path).read_bytes()
    assert raw.count(b'\n') == raw.count(b'\r\n'), 'Training CRLF changed'
    text = raw.decode().replace('\r\n', '\n')
    additions = (
        "    parser.add_argument('--skip-final-test', action='store_true',\n"
        "                        help='validation-only diagnosis: do not evaluate the test loader after training')\n",
        "    if args.skip_final_test and (args.selection_protocol != 'validation' or args.testing):\n"
        "        parser.error('--skip-final-test requires validation selection and forbids --testing')\n",
    )
    for addition in additions:
        assert text.count(addition) == 1, 'Unexpected skip-final-test parser patch'
        text = text.replace(addition, '')
    new = "    if args.selection_protocol == 'validation' and not args.skip_final_test and best_checkpoint_path.exists():\n"
    old = "    if args.selection_protocol == 'validation' and best_checkpoint_path.exists():\n"
    assert text.count(new) == 1, 'Unexpected final-test gate'
    text = text.replace(new, old)
    assert hashlib.sha256(text.replace('\n', '\r\n').encode()).hexdigest() == historical_sha
    return hashlib.sha256(raw).hexdigest()


def assert_scientific_pair(args, historical, run, patched_sha):
    actual, expected = vars(args), historical['args']
    permitted = {'dropout', 'skip_final_test', 'output_dir', 'provenance'}
    differences = {key: (expected.get(key), actual.get(key))
                   for key in set(actual) | set(expected)
                   if key not in permitted and actual.get(key) != expected.get(key)}
    assert not differences, 'Unregistered scientific differences: ' + repr(differences)
    assert actual['lr'] == expected['lr'] == run['lr'] == 1e-4
    assert expected['batch_size'] == actual['batch_size'] == run['batch_size'] == 16
    assert run['validation_batch_size'] == actual['batch_size']
    assert expected['dropout'] == .4 and actual['dropout'] == run['dropout'] == .5
    assert actual['rec_dropout'] == expected['rec_dropout'] == run['rec_dropout'] == .1
    assert actual['skip_final_test'] is True
    assert actual['seed'] == run['seed'] and actual['epochs'] == run['epochs']
    assert actual['selection_protocol'] == 'validation' and actual['testing'] is False
    assert actual['rng_policy'] == 'continuous' and actual['rnn_backend'] == 'aten'
    assert actual['batch_policy'] == 'capacity_safe' and actual['num_workers'] == 0
    assert actual['calibrate'] and actual['contrastlearning'] and actual['courselearning']
    assert (actual['rank_coff'], actual['mscl_coff'], actual['cscl_coff']) == (.002, .15, .15)
    provenance = actual['provenance']
    assert provenance['split'] == historical['split']
    assert provenance['split']['hash'] == run['split_hash']
    assert provenance['data'] == historical['data']
    files, old_files = provenance['source']['files'], historical['source']['files']
    assert set(files) == set(old_files)
    for name, digest in old_files.items():
        assert files[name] == (patched_sha if name == 'train_cm.py' else digest), name


def audit_dropout_models(args, models):
    """Read attributes only; no model construction, forward, mutation, or RNG draws."""
    before = rng_hash()
    assert args.dropout == .5 and args.rec_dropout == .1
    result = {'cli_dropout': args.dropout, 'cli_rec_dropout_metadata': args.rec_dropout, 'models': {}}
    assert set(models) == {'model', 'model_mm', 'model_intra', 'model_inter'}
    for name, model in models.items():
        gru, hidden_dropout = model.gru_l, model.dropout_
        assert isinstance(gru, torch.nn.GRU) and gru.dropout == .5
        assert gru.num_layers == 2 and gru.bidirectional is True
        assert isinstance(hidden_dropout, torch.nn.Dropout) and hidden_dropout.p == .5
        assert model.dropout == .5
        fixed = {}
        dropouts = {}
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                dropouts[module_name] = module.p
            for attr in ('dropouta', 'dropoutv', 'dropoutl'):
                if hasattr(module, attr):
                    value = getattr(module, attr)
                    assert isinstance(value, torch.nn.Dropout) and value.p == .5
                    fixed[(module_name + '.' + attr).lstrip('.')] = value.p
        result['models'][name] = {'class': type(model).__name__, 'language_gru': {
            'dropout': gru.dropout, 'num_layers': gru.num_layers, 'bidirectional': gru.bidirectional},
            'hidden_dropout_p': hidden_dropout.p, 'registered_dropout_modules': dropouts,
            'registered_fixed_modality_dropouts': fixed}
    tree = ast.parse((ROOT / 'code_xp04/model_cm.py').read_bytes())
    gated = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MMGatedAttention')
    definitions = {}
    for node in ast.walk(gated):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr in ('dropouta', 'dropoutv', 'dropoutl'):
                    assert ast.unparse(node.value.func) == 'nn.Dropout'
                    assert len(node.value.args) == 1 and ast.literal_eval(node.value.args[0]) == .5
                    definitions[target.attr] = .5
    assert definitions == {'dropouta': .5, 'dropoutv': .5, 'dropoutl': .5}
    result['fixed_modality_dropout_definition'] = {'class': 'MMGatedAttention', 'values': definitions,
        'active_instances': {name: bool(value['registered_fixed_modality_dropouts']) for name, value in result['models'].items()},
        'note': 'No UnimodalEncoder class in this frozen source; gated attention is not instantiated by concat_DHT.'}
    result['rng_before'] = before
    result['rng_after'] = rng_hash()
    assert result['rng_before'] == result['rng_after']
    return result


def check_components(components, args):
    values = {key: float(components[key]) for key in LOSSES}
    assert all(math.isfinite(value) for value in values.values()), 'Nonfinite loss component'
    expected = values['classification_loss']
    if args.calibrate:
        expected += args.rank_coff * values['cc_loss']
    if args.contrastlearning:
        expected += args.mscl_coff * values['mscl_loss'] + args.cscl_coff * values['cscl_loss']
    assert math.isclose(expected, values['total_loss'], rel_tol=1e-5, abs_tol=1e-6), values
    return values


class PreflightComplete(Exception):
    pass


class TrainingGuard:
    def __init__(self, run, historical, target, patched_sha, output=None, preflight=False):
        self.run, self.historical = run, historical
        self.target, self.patched_sha = str(target), patched_sha
        self.output = Path(output) if output is not None else None
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=False)
        self.preflight = preflight
        self.counts = {'train': 0, 'validation': 0, 'test': 0}
        self.active = None
        self.validated = False
        self.first = None
        self.memory_records = []
        self.line_events = 0

    def emit(self, name, value, append=False):
        if self.output is None:
            self.memory_records.append((name, value))
        elif append:
            with (self.output / name).open('a') as stream:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        else:
            write_json(self.output / name, value)

    def split_for(self, train, keys):
        split = self.historical['split']
        if train:
            assert keys and len(keys) == len(set(keys))
            assert set(keys).issubset(split['train_ids'])
            assert not set(keys).intersection(split['validation_ids'] + split['test_ids'])
            return 'train'
        assert keys == split['validation_ids'], 'Forbidden evaluation: only the complete validation split is allowed'
        return 'validation'

    def trace(self, frame, event, arg):
        if frame.f_code.co_filename != self.target or frame.f_code.co_name != 'train_or_eval_graph_model':
            return None
        frame.f_trace_lines = False
        frame.f_trace_opcodes = False
        if event == 'line':
            self.line_events += 1
            raise AssertionError('Target line tracing must be disabled')
        if event == 'call':
            assert self.active is None
            args = frame.f_globals['args']
            if not self.validated:
                assert_scientific_pair(args, self.historical, self.run, self.patched_sha)
                self.emit('effective_args.json', vars(args))
                self.validated = True
            local = frame.f_locals
            keys = list(local['dataloader'].dataset.keys)
            split = self.split_for(local['train'], keys)
            epoch = int(local['epoch']) + 1
            assert epoch == self.counts[split] + 1
            assert epoch <= self.run['epochs']
            if split == 'train':
                assert self.counts['train'] == self.counts['validation']
            else:
                assert self.counts['train'] == self.counts['validation'] + 1
            self.active = {'split': split, 'epoch': epoch, 'keys': keys,
                           'steps': [], 'seen_ids': [], 'frame': frame}
            if self.first is None and split == 'train':
                self.emit('module_audit.json', audit_dropout_models(args, {
                    name: local[name] for name in ('model', 'model_mm', 'model_intra', 'model_inter')}))
                self.first = {'initial_model_sha256': model_hash(local['model']),
                              'rng_before_train': rng_hash()}
            return self.trace
        if event == 'return' and arg is not None:
            self.finish_epoch(frame.f_locals, arg)
        return self.trace

    def adam_step(self, original, optimizer, *args, **kwargs):
        state = self.active
        assert state is not None and state['split'] == 'train', 'Optimizer step outside training'
        local = state['frame'].f_locals
        assert local['optimizer'] is optimizer
        values = check_components(local['components'], state['frame'].f_globals['args'])
        parameters = [p for group in optimizer.param_groups for p in group['params']]
        norm_squared = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                gradient = parameter.grad.detach()
                assert torch.isfinite(gradient).all().item(), 'Nonfinite gradient'
                norm_squared += gradient.double().square().sum().item()
        norm = math.sqrt(norm_squared)
        assert math.isfinite(norm)
        first_step = not state['steps']
        previous = [p.detach().cpu().clone() for p in parameters] if first_step else None
        first_of_run = state['epoch'] == 1 and first_step
        if first_of_run:
            self.first['rng_before_step'] = rng_hash()
        result = original(optimizer, *args, **kwargs)
        changed = any(not torch.equal(old, p.detach().cpu()) for old, p in zip(previous, parameters)) if first_step else None
        if first_step:
            assert changed, 'First Adam step did not change any parameter'
        ids = list(local['data'][-1])
        state['seen_ids'].extend(ids)
        row = {'split': 'train', 'epoch': state['epoch'], 'batch': len(state['steps']) + 1,
               'dialogue_ids': ids, 'lengths': [int(value) for value in local['lengths']],
               **values, 'gradients_finite': True, 'gradient_l2': norm,
               'first_step_parameters_changed': changed}
        state['steps'].append(row)
        self.emit('watch_batches.jsonl', row, append=True)
        if first_of_run:
            self.first.update({'rng_after_step': rng_hash(), 'components': values,
                               'gradient_l2': norm, 'gradients_finite': True,
                               'optimizer_updated_parameters': changed,
                               'dialogue_ids': ids, 'lengths': row['lengths'],
                               'model_after_step_sha256': model_hash(local['model'])})
            self.emit('first.json', self.first)
            if self.preflight:
                raise PreflightComplete()
        return result

    def finish_epoch(self, local, result):
        state = self.active
        assert state is not None
        components = [check_components(row, state['frame'].f_globals['args'])
                      for row in local['loss_components']]
        assert components, 'Empty train/validation evaluation'
        keys, dataset = state['keys'], local['dataloader'].dataset
        batch_size = state['frame'].f_globals['args'].batch_size
        assert len(components) == math.ceil(len(keys) / batch_size)
        expected_utterances = sum(len(dataset.videoLabels[key]) for key in keys)
        assert len(result[2]) == len(result[3]) == expected_utterances
        if state['split'] == 'train':
            assert len(state['steps']) == len(components)
            assert [len(step['dialogue_ids']) for step in state['steps']] == [
                min(batch_size, len(keys) - start) for start in range(0, len(keys), batch_size)]
            assert len(state['seen_ids']) == len(keys) and set(state['seen_ids']) == set(keys)
        averaged = {key: float(np.mean([row[key] for row in components])) for key in LOSSES}
        norms = [step['gradient_l2'] for step in state['steps']]
        row = {'split': state['split'], 'epoch': state['epoch'], 'dialogue_ids': keys,
               'weighted_f1': float(result[4]), 'accuracy': float(result[1]),
               'macro_f1': float(f1_score(result[2], result[3], average='macro') * 100),
               'classification_loss': float(result[0]), 'loss_components': averaged,
               'batches': len(components), 'n_utterances': expected_utterances,
               'gradients_finite': True if norms else None,
               'gradient_l2_mean': float(np.mean(norms)) if norms else None,
               'gradient_l2_max': max(norms) if norms else None,
               'first_step_parameters_changed': True if norms else None}
        assert all(math.isfinite(row[key]) for key in ('weighted_f1', 'accuracy', 'macro_f1', 'classification_loss'))
        self.emit('watch_epochs.jsonl', row, append=True)
        self.counts[state['split']] += 1
        self.active = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--preflight', action='store_true')
    options = parser.parse_args()
    registry = json.loads((ROOT / 'configs/registry.json').read_text())
    assert Path(sys.executable).resolve() == Path(registry['python']).resolve()
    for key, expected_value in registry['environment'].items():
        assert os.environ.get(key) == expected_value, 'Execution environment mismatch: ' + key
    run = next(row for row in registry['runs'] if row['run_id'] == options.run_id)
    historical = json.loads(Path(run['historical_manifest']).read_text())
    code = Path(registry['code_dir']).resolve()
    assert ROOT in code.parents
    target = str(code / 'train_cm.py')
    patched_sha = verify_skip_patch(target, historical['source']['files']['train_cm.py'])
    argv = list(run['argv'])
    assert '--skip-final-test' in argv and '--testing' not in argv
    output_index = argv.index('--output-dir') + 1
    assert Path(argv[output_index]).resolve() == Path(run['output_root']).resolve()
    expected = Path(argv[output_index]) / (run['dataset'] + '_validation_seed' + str(run['seed']))
    assert expected.resolve() == Path(run['run_dir']).resolve()
    assert expected.resolve().is_relative_to(ROOT / 'runs')
    if options.preflight:
        argv[output_index] = str(ROOT / 'preflight' / run['run_id'] / 'runs')
        output = ROOT / 'preflight' / run['run_id'] / 'observer'
    else:
        output = ROOT / 'results' / run['run_id']
    guard = TrainingGuard(run, historical, target, patched_sha, output, options.preflight)
    guard.emit('launch.json', {'argv': argv, 'code_dir': str(code), 'preflight': options.preflight,
                              'watcher_sha256': sha(__file__), 'training_sha256': patched_sha,
                              'python': sys.executable, 'environment': registry['environment']})
    old_cwd, old_argv, old_trace, old_step = Path.cwd(), sys.argv, sys.gettrace(), torch.optim.Adam.step
    @functools.wraps(old_step)
    def watched_step(optimizer, *positional, **keywords):
        return guard.adam_step(old_step, optimizer, *positional, **keywords)
    try:
        os.chdir(code)
        sys.path.insert(0, str(code))
        sys.argv = [target, *argv]
        torch.optim.Adam.step = watched_step
        sys.settrace(guard.trace)
        runpy.run_path(target, run_name='__main__')
    except PreflightComplete:
        assert options.preflight
        guard.emit('status.json', {'phase': 'preflight_passed', 'first_step_completed': True,
                                  'counts': guard.counts, 'test_forward_passes': 0,
                                  'target_line_events': guard.line_events})
    else:
        assert not options.preflight, 'Preflight should stop after its first Adam step'
        assert guard.counts == {'train': run['epochs'], 'validation': run['epochs'], 'test': 0}
        assert guard.active is None and guard.line_events == 0
        assert not (Path(run['run_dir']) / 'final_test.json').exists()
        epochs = [json.loads(line) for line in (Path(run['run_dir']) / 'metrics.jsonl').read_text().splitlines()]
        assert [row['epoch'] for row in epochs] == list(range(1, run['epochs'] + 1))
        assert all(row['test_accuracy'] is None and row['test_weighted_f1'] is None for row in epochs)
        selected = max(epochs, key=lambda row: row['validation_weighted_f1'])
        guard.emit('status.json', {'phase': 'completed', 'counts': guard.counts,
                                  'test_forward_passes': 0, 'no_final_test_artifacts': True,
                                  'selected_epoch': selected['epoch'],
                                  'selection_weighted_f1': selected['validation_weighted_f1'],
                                  'target_line_events': guard.line_events})
    finally:
        sys.settrace(old_trace)
        torch.optim.Adam.step = old_step
        os.chdir(old_cwd)
        sys.argv = old_argv


if __name__ == '__main__':
    main()
