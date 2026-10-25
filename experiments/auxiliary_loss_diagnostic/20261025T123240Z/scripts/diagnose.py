"""Fixed-checkpoint eval gradients using the untouched historical forward/loss body.

The original function is stopped at its loss-combiner return, before backward,
metrics aggregation or any optimizer action. Only explicit training keys are used.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RELATIONS = ('t', 'a', 'v', 'intra', 'inter')
LOSSES = ('classification', 'cc', 'mscl', 'cscl')


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def scalar(value):
    value = float(value)
    return value if math.isfinite(value) else None


def append_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    fields = list(rows[0])
    if path.exists():
        with path.open(newline='') as stream:
            assert next(csv.reader(stream)) == fields, str(path)
    fresh = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fresh:
            writer.writeheader()
        writer.writerows(rows)


def constructor_surface(path):
    """Reuse exact dimension block and four constructor calls from guarded main."""
    tree = ast.parse(Path(path).read_bytes())
    main = next(n for n in tree.body if isinstance(n, ast.If)
                and ast.unparse(n.test) == "__name__ == '__main__'")
    start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'feat2dim' for t in n.targets))
    stop = next(i for i in range(start + 1, len(main.body))
                if isinstance(main.body[i], ast.If) and ast.unparse(main.body[i].test) == 'args.graph_model')
    dimensions = ast.Module(body=main.body[start:stop], type_ignores=[])
    calls = {}
    expected = {'model': 'Model', 'model_mm': 'Model', 'model_intra': 'Model_Intra', 'model_inter': 'Model_Inter'}
    for name, cls in expected.items():
        matches = [n.value for n in ast.walk(main) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                   and n.value.func.id == cls]
        assert len(matches) == 1, name
        calls[name] = matches[0]
    return dimensions, calls


class CaseComplete(Exception):
    pass


def resource_admission(registry):
    policy = registry['hardware_policy']
    samples = []
    while len(samples) < policy['samples']:
        q = subprocess.run(['nvidia-smi', '--id=' + registry['gpu_uuid'],
                            '--query-gpu=index,uuid,name,memory.free,utilization.gpu',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
        if q.returncode:
            raise RuntimeError(q.stderr)
        f = [s.strip() for s in q.stdout.strip().split(',')]
        assert int(f[0]) == registry['gpu_index'] and f[1] == registry['gpu_uuid'] and 'TITAN Xp' in f[2]
        row = dict(time=now(), gpu_index=int(f[0]), gpu_uuid=f[1], free_mib=int(f[3]), utilization=int(f[4]))
        with (ROOT / 'logs/resources.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        if row['free_mib'] >= policy['minimum_free_mib'] and row['utilization'] <= policy['maximum_utilization_percent']:
            samples.append(row)
        else:
            samples = []
        if len(samples) < policy['samples']:
            time.sleep(policy['interval_seconds'])
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    import fcntl
    lock = (ROOT / 'state/diagnostic.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    registry = json.loads((ROOT / 'configs/registry.json').read_text())
    capsule = json.loads((ROOT / 'provenance/frozen_execution.json').read_text())
    for name, expected in capsule['files'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in registry['environment'].items():
        assert os.environ.get(name) == expected, name
    assert not (ROOT / 'results/case_summary.csv').exists(), 'No implicit duplicate execution/resume'
    with (ROOT / 'state/diagnostic_start.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), started_at_utc=now(), no_implicit_restart=True), stream)
    state = dict(active=True, phase='resource_admission', pid=os.getpid(), started_at_utc=now(),
                 training_runs=0, optimizer_steps=0, test_forwards=0, completed_cases=0,
                 model_forward_calls=0, autograd_calls=0, tensor_backward_calls=0, failed_cases=[])
    write(ROOT / 'state/status.json', state)
    state['admission_samples'] = resource_admission(registry)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from diagnostic_core import parameter_groups, gradient_norm_rows, gradient_pair_rows, pseudo_rows
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    assert 'TITAN Xp' in torch.cuda.get_device_properties(0).name
    torch.empty(1, device='cuda')
    visible = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                       '--format=csv,noheader,nounits'], text=True)
    identities = [line.split(',')[0].strip() for line in visible.splitlines()
                  if line.split(',')[-1].strip() == str(os.getpid())]
    assert identities and set(identities) == {registry['gpu_uuid']}
    code = ROOT / 'code_xp04'
    os.chdir(code)
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location('xp13_frozen_train_cm', code / 'train_cm.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dimensions, calls = constructor_surface(code / 'train_cm.py')
    reference = json.loads((ROOT / 'provenance/xp04_validation_reference.json').read_text())
    fixed = json.loads((ROOT / 'private/fixed_batches.json').read_text())['batches']
    assert len(fixed) == registry['n_batches'] == 32
    all_ids = [key for batch in fixed for key in batch['dialogue_ids']]
    assert len(all_ids) == len(set(all_ids)) == 512
    assert set(all_ids).issubset(reference['split']['train_ids'])
    assert not set(all_ids).intersection(reference['split']['validation_ids'] + reference['split']['test_ids'])
    dataset = module.MELDDataset(str(code / 'data/MELD_features_raw1.pkl'), keys=all_ids)
    id_to_index = {key: i for i, key in enumerate(dataset.keys)}
    old_backward = torch.Tensor.backward
    step_classes = [torch.optim.Adam, torch.optim.SGD, torch.optim.AdamW, torch.optim.RMSprop]
    old_steps = {cls: cls.step for cls in step_classes}

    def forbidden_backward(*args, **kwargs):
        state['tensor_backward_calls'] += 1
        raise AssertionError('Tensor.backward is forbidden in this diagnostic')

    def forbidden_step(*args, **kwargs):
        state['optimizer_steps'] += 1
        raise AssertionError('optimizer.step is forbidden in this diagnostic')

    def rng_hash():
        h = hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state())))
        h.update(torch.get_rng_state().numpy().tobytes())
        for r in torch.cuda.get_rng_state_all():
            h.update(r.cpu().numpy().tobytes())
        return h.hexdigest()

    def model_hash(model):
        h = hashlib.sha256()
        for name, value in model.state_dict().items():
            h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def gradient_hash(values):
        h = hashlib.sha256()
        for name, value in values.items():
            h.update(name.encode())
            h.update(b'None' if value is None else value.contiguous().numpy().tobytes())
        return h.hexdigest()

    torch.Tensor.backward = forbidden_backward
    for cls in step_classes:
        cls.step = forbidden_step
    try:
        for item in registry['checkpoints']:
            checkpoint = torch.load(item['path'], map_location='cpu')
            assert checkpoint['epoch'] + 1 == item['expected_epoch']
            args = SimpleNamespace(**checkpoint['args'])
            assert args.calibrate and args.contrastlearning == item['expected_contrastlearning']
            assert args.Dataset == 'MELD' and not args.causal and args.rnn_backend == 'aten'
            module.args = args
            scope = dict(module.__dict__, args=args, modals=args.modals)
            exec(compile(dimensions, str(code / 'train_cm.py'), 'exec'), scope)
            models = {name: eval(compile(ast.Expression(node), str(code / 'train_cm.py'), 'eval'), scope)
                      for name, node in calls.items()}
            for model in models.values():
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                module.configure_recurrent_backend(model, args.rnn_backend)
                model.cuda().eval()
            del checkpoint
            parameters = {name: p for name, p in models['model'].named_parameters() if p.requires_grad}
            groups = parameter_groups(list(parameters))
            random.seed(registry['diagnostic_seed']); np.random.seed(registry['diagnostic_seed'])
            torch.manual_seed(registry['diagnostic_seed']); torch.cuda.manual_seed_all(registry['diagnostic_seed'])
            for batch in fixed:
                context = dict(checkpoint_id=item['checkpoint_id'], batch_id=batch['batch_id'],
                               length_stratum=batch['length_stratum'])
                state.update(phase='diagnosing', current_case=context)
                write(ROOT / 'state/status.json', state)
                before = {name: model_hash(model) for name, model in models.items()}
                rng_before = rng_hash()
                modes_before = {name: [m.training for m in model.modules()] for name, model in models.items()}
                assert all(p.grad is None for p in parameters.values())
                base_outputs, base_weights, numeric, wrapper = {}, {}, [], []
                counts = dict(model_forward_calls=0, autograd_calls=0)
                roles = dict.fromkeys(models, 0)
                base_map = {id(module.CKSCL_m.ckscl_base1): 't', id(module.CKSCL_m.ckscl_base2): 'a',
                            id(module.CKSCL_m.ckscl_base3): 'v', id(module.CKSCL_c.ckscl_base1): 'intra',
                            id(module.CKSCL_c.ckscl_base2): 'inter'}
                wrapper_map = {id(module.CKSCL_m): 'mscl', id(module.CKSCL_c): 'cscl'}
                handles = []

                def forward_guard(model, inputs):
                    assert not model.training, 'Only eval mode allowed'
                    assert torch.is_grad_enabled() == (model is models['model']), 'Original main/teacher grad modes changed'
                    counts['model_forward_calls'] += 1
                    state['model_forward_calls'] += 1
                    roles[next(name for name, instance in models.items() if model is instance)] += 1

                for model in models.values():
                    handles.append(model.register_forward_pre_hook(forward_guard))
                output = {}

                def capture(local):
                    assert local['train'] is False
                    assert list(local['data'][-1]) == batch['dialogue_ids']
                    labels = local['label'].detach().cpu()
                    assert len(labels) == batch['n_utterances']
                    losses = dict(classification=local['classification_loss'], cc=local['rank_loss'],
                                  mscl=local['Mscl_tav'], cscl=local['Cscl_c'])
                    gradients = {}
                    gradient_status = {}
                    for loss_name, loss in losses.items():
                        if torch.is_tensor(loss) and loss.requires_grad and torch.isfinite(loss).item():
                            counts['autograd_calls'] += 1
                            state['autograd_calls'] += 1
                            values = torch.autograd.grad(loss, tuple(parameters.values()), retain_graph=True,
                                                         create_graph=False, allow_unused=True)
                            gradients[loss_name] = {name: None if value is None else value.detach().cpu()
                                                    for name, value in zip(parameters, values)}
                            gradient_status[loss_name] = 'observed'
                            del values
                        else:
                            gradients[loss_name] = dict.fromkeys(parameters, None)
                            value = loss.detach().item() if torch.is_tensor(loss) else float(loss)
                            gradient_status[loss_name] = 'nonfinite_loss_skipped' if not math.isfinite(value) else 'constant_or_nondifferentiable'
                    reference_weights = registry['reference_coefficients']
                    actual_weights = dict(reference_weights)
                    if not args.contrastlearning:
                        actual_weights.update(mscl=0., cscl=0.)
                    norms = gradient_norm_rows(gradients, groups, context, reference_weights, actual_weights)
                    pairs = gradient_pair_rows(gradients, groups, context, reference_weights, actual_weights)
                    removed = {r: local[key].detach().cpu() for r, key in zip(RELATIONS,
                               ['log_prob2t', 'log_prob2a', 'log_prob2v', 'log_prob2intra', 'log_prob2inter'])}
                    pseudo = {r: local[key].detach().cpu() for r, key in zip(RELATIONS,
                              ['Mscl_pred_t', 'Mscl_pred_a', 'Mscl_pred_v', 'Cscl_pred_intra', 'Cscl_pred_inter'])}
                    assert set(base_outputs) == set(base_weights) == set(RELATIONS)
                    records = pseudo_rows(context, labels, local['log_prob'].detach().cpu(), removed, pseudo,
                                          base_outputs, base_weights)
                    private_rows = records.pop('private_rows')
                    private_folder = ROOT / 'private' / item['checkpoint_id']
                    private_folder.mkdir(exist_ok=True)
                    write(private_folder / (batch['batch_id'] + '.json'),
                          dict(context=context, dialogue_ids=batch['dialogue_ids'],
                               utterance_records=private_rows, gradient_sha256={k: gradient_hash(v) for k, v in gradients.items()}))
                    output.update(norms=norms, pairs=pairs, records=records,
                                  summary={**context, 'n_dialogues':16, 'n_utterances':len(labels),
                                           **{'loss_' + name: scalar(value.detach().item() if torch.is_tensor(value) else value)
                                              for name, value in losses.items()},
                                           **{'requires_grad_' + name: bool(torch.is_tensor(value) and value.requires_grad)
                                              for name, value in losses.items()},
                                           **{'gradient_status_' + name: status for name, status in gradient_status.items()},
                                           **{'nonfinite_gradient_tensors_' + name: sum(value is not None and not torch.isfinite(value).all().item() for value in values.values())
                                              for name, values in gradients.items()},
                                           'actual_mscl_coefficient':actual_weights['mscl'],
                                           'actual_cscl_coefficient':actual_weights['cscl']})
                    raise CaseComplete()

                def trace(frame, event, value):
                    path = frame.f_code.co_filename
                    if path == str(code / 'clloss.py') and frame.f_code.co_name == 'forward':
                        frame.f_trace_lines = False; frame.f_trace_opcodes = False
                        if event == 'return' and value is not None:
                            local = frame.f_locals; obj = id(local.get('self'))
                            if obj in base_map:
                                relation = base_map[obj]
                                base_outputs[relation] = value.detach().cpu()
                                base_weights[relation] = list(local['weight'])
                                pos, neg = local['pos'].detach(), local['neg'].detach()
                                numeric.append({**context, 'relation':relation, 'n_anchors':int(value.numel()),
                                    'nonfinite_logits':int((~torch.isfinite(value)).sum().item()),
                                    'nonfinite_similarity_cells':int((~torch.isfinite(local['similarity'])).sum().item()),
                                    'negative_mask_cells':int((local['mask_neg'] < 0).sum().item()),
                                    'negative_neg_sum_anchors':int((neg < 0).sum().item()),
                                    'negative_denominator_anchors':int(((pos + neg) < 0).sum().item()),
                                    'zero_denominator_anchors':int(((pos + neg) == 0).sum().item()),
                                    'nonfinite_denominator_anchors':int((~torch.isfinite(pos + neg)).sum().item()),
                                    'pos_min':scalar(pos.min().item()), 'pos_max':scalar(pos.max().item()),
                                    'neg_min':scalar(neg.min().item()), 'neg_max':scalar(neg.max().item())})
                            elif obj in wrapper_map:
                                raw = -local['logits'].detach().mean().item()
                                wrapper.append({**context,'loss':wrapper_map[obj],
                                                'pre_fallback_loss':scalar(raw), 'returned_loss':scalar(value.item()),
                                                'nonfinite_before_fallback':not math.isfinite(raw),
                                                'fallback_observed':not math.isfinite(raw) and value.item()==0.})
                        return trace
                    if path == str(code / 'train_cm.py') and frame.f_code.co_name == 'combine_training_losses':
                        frame.f_trace_lines = False; frame.f_trace_opcodes = False
                        if event == 'return' and value is not None:
                            capture(frame.f_back.f_locals)
                        return trace
                    return None

                # Explicit train keys only; no validation/test loader is ever constructed.
                generator = torch.Generator().manual_seed(registry['diagnostic_seed'] + int(batch['batch_id'][1:]))
                loader = DataLoader(dataset, batch_sampler=[[id_to_index[key] for key in batch['dialogue_ids']]],
                                    collate_fn=dataset.collate_fn, num_workers=0, generator=generator)
                previous_trace = sys.gettrace()
                try:
                    sys.settrace(trace)
                    with torch.enable_grad():
                        module.train_or_eval_graph_model(**models, loss_function=module.FocalLoss(),
                            dataloader=loader, epoch=item['expected_epoch']-1, cuda=True, modals=args.modals,
                            optimizer=None, train=False, dataset='MELD')
                    raise AssertionError('Original function escaped the no-backward interception')
                except CaseComplete:
                    pass
                finally:
                    sys.settrace(previous_trace)
                    for handle in handles:
                        handle.remove()
                assert output and counts['model_forward_calls'] == 6
                assert roles == {'model':1, 'model_mm':3, 'model_intra':1, 'model_inter':1}
                after = {name:model_hash(model) for name,model in models.items()}
                assert before == after, 'Parameters or registered buffers changed'
                assert modes_before == {name:[m.training for m in model.modules()] for name,model in models.items()}
                rng_after = rng_hash()
                assert rng_after == rng_before, 'RNG was not restored by original isolated evaluation'
                assert all(p.grad is None for p in parameters.values())
                summary = output['summary']
                summary.update(**counts, parameters_buffers_unchanged=True, RNG_unchanged=True,
                               grad_buffers_unchanged=True,
                               status='completed',
                               **{'forward_' + name: value for name,value in roles.items()},
                               hscl_nonfinite_fallbacks=sum(row['fallback_observed'] for row in wrapper))
                append_csv(ROOT / 'results/gradient_norms.csv', output['norms'])
                append_csv(ROOT / 'results/gradient_pairs.csv', output['pairs'])
                for name, rows in output['records'].items():
                    append_csv(ROOT / 'results' / (name + '.csv'), rows)
                append_csv(ROOT / 'results/ckscl_numeric.csv', numeric)
                append_csv(ROOT / 'results/ckscl_wrapper_audit.csv', wrapper)
                append_csv(ROOT / 'results/state_audit.csv', [{**context,
                    **{'before_' + name: value for name,value in before.items()},
                    **{'after_' + name: value for name,value in after.items()},
                    'rng_before':rng_before, 'rng_after':rng_after, 'all_parameter_grad_buffers_none':True}])
                append_csv(ROOT / 'results/case_summary.csv', [summary])
                state['completed_cases'] += 1
                write(ROOT / 'state/status.json', state)
                print(json.dumps(dict(time=now(), **context, completed_cases=state['completed_cases'],
                                      model_forward_calls=state['model_forward_calls'], autograd_calls=state['autograd_calls'])), flush=True)
            for model in models.values():
                model.cpu()
            del models, parameters, scope
            torch.cuda.empty_cache()
        assert state['completed_cases'] == registry['expected_cases'] == 128
        assert state['model_forward_calls'] == 768 and state['autograd_calls'] <= 512
        assert state['training_runs'] == state['optimizer_steps'] == state['test_forwards'] == state['tensor_backward_calls'] == 0
        state.update(active=False, phase='completed', finished_at_utc=now())
        write(ROOT / 'state/status.json', state)
    except BaseException as error:
        state['failed_cases'].append(dict(case=state.get('current_case'), error=repr(error),
                                         completed_cases=state['completed_cases'],
                                         attempted_model_forward_calls=state['model_forward_calls'],
                                         attempted_autograd_calls=state['autograd_calls']))
        state.update(active=False, phase='failed', error=repr(error), finished_at_utc=now())
        write(ROOT / 'state/status.json', state)
        raise
    finally:
        torch.Tensor.backward = old_backward
        for cls, method in old_steps.items():
            cls.step = method


if __name__ == '__main__':
    main()
