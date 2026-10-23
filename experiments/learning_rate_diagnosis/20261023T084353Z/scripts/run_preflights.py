"""Serial first-Adam probes only; gate four formal, zero-test LR trials."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import time

from supervise_lr import (ROOT, INTERVAL, append_json, child_environment, digest, empty_admission,
                          execution_lock, now, resource_snapshot, validate_capsule, write_json)


def compare_first(left, right):
    exact = ('initial_model_sha256', 'rng_before_train', 'rng_before_step', 'rng_after_step', 'dialogue_ids', 'lengths')
    for key in exact:
        assert left[key] == right[key], 'LR first-step parity mismatch: ' + key
    for record in (left, right):
        assert record['gradients_finite'] and record['optimizer_updated_parameters']
        assert all(math.isfinite(float(value)) for value in record['components'].values())
        assert math.isfinite(record['gradient_l2'])
    assert set(left['components']) == set(right['components'])
    numeric = {key: {'left': left['components'][key], 'right': right['components'][key],
                     'difference': right['components'][key] - left['components'][key]}
               for key in left['components']}
    numeric['gradient_l2'] = {'left': left['gradient_l2'], 'right': right['gradient_l2'],
                              'difference': right['gradient_l2'] - left['gradient_l2']}
    for key, values in numeric.items():
        assert math.isclose(values['left'], values['right'], rel_tol=1e-5, abs_tol=1e-6), 'LR forward/gradient parity mismatch: ' + key
    return {'status': 'passed', 'exactly_equal_fields': list(exact), 'relative_tolerance': 1e-5,
            'absolute_tolerance': 1e-6, 'numeric_comparison': numeric,
            'post_Adam_parameters_expected_to_differ': True,
            'post_Adam_model_hashes_different': left['model_after_step_sha256'] != right['model_after_step_sha256']}


def evidence_passed(path):
    record = json.loads(path.read_text())
    assert record.get('status') == 'passed' or record.get('passed') is True, 'CPU evidence not passed: ' + str(path)
    return {'path': str(path), 'sha256': digest(path)}


def main():
    lock = execution_lock()
    registry = json.loads((ROOT / 'configs/registry.json').read_text())
    assert len(registry['runs']) == 4
    cpu = [evidence_passed(ROOT / 'preflight' / name) for name in ('capacity_cpu.json', 'guard_tests.json')]
    validate_capsule()
    assert not (ROOT / 'preflight/gate.json').exists(), 'Preflight gate already exists; refusing implicit re-probe'
    assert not any((ROOT / 'preflight' / run['run_id']).exists() for run in registry['runs']), 'Probe output already exists; no automatic retry'
    state = {'active': True, 'phase': 'resource_admission', 'pid': os.getpid(), 'started_at_utc': now(), 'runs': [], 'cpu_evidence': cpu,
             'foreign_process_samples': 0, 'foreign_process_observations': []}
    child, log = None, None
    row = None
    try:
        for run in registry['runs']:
            validate_capsule()
            state.update(phase='resource_admission', current_run=run['run_id'])
            while True:
                sample = resource_snapshot(registry)
                append_json(ROOT / 'logs/preflight_resources.jsonl', sample)
                if sample['unexpected_processes']:
                    state['foreign_process_samples'] += 1
                    state['foreign_process_observations'].append(sample)
                write_json(ROOT / 'state/preflight.json', state)
                assert state['foreign_process_samples'] == 0, 'Foreign GPU process observed before probe; formal gate refused'
                if empty_admission(sample):
                    break
                time.sleep(INTERVAL)
            env = child_environment(registry, run['run_id'], preflight=True)
            argv = [registry['python'], '-u', str(ROOT / 'scripts/memory_observer.py'),
                    str(ROOT / 'scripts/watch_training.py'), '--run-id', run['run_id'], '--preflight']
            path = ROOT / 'logs' / ('preflight_' + run['run_id'] + '.log')
            log = path.open('x')
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row = {'run_id': run['run_id'], 'pid': child.pid, 'argv': argv, 'started_at_utc': now(),
                   'environment': {key: env[key] for key in registry['environment']}, 'profile_output': env['CMERC_PROFILE_OUTPUT'],
                   'log': str(path), 'admission': sample}
            state['runs'].append(row)
            state.update(phase='probing', child_pid=child.pid)
            write_json(ROOT / 'state/preflight.json', state)
            resource_errors = 0
            while child.poll() is None:
                try:
                    sample = resource_snapshot(registry, [child.pid])
                    append_json(ROOT / 'logs/preflight_resources.jsonl', sample)
                    if sample['unexpected_processes']:
                        state['foreign_process_samples'] += 1
                        state['foreign_process_observations'].append(sample)
                        write_json(ROOT / 'state/preflight.json', state)
                except Exception as failure:
                    resource_errors += 1
                    append_json(ROOT / 'logs/preflight_resource_errors.jsonl', {'time': now(), 'error': repr(failure), 'child_pid': child.pid})
                time.sleep(INTERVAL)
            row.update(returncode=child.returncode, finished_at_utc=now())
            log.close()
            log = None
            assert child.returncode == 0, 'First-step probe failed: ' + run['run_id']
            assert resource_errors == 0, 'Resource query failed during probe; inspect before admitting formal work'
            child = None
            assert state['foreign_process_samples'] == 0, 'Foreign GPU process observed; probe naturally drained, formal gate refused'
            output = ROOT / 'preflight' / run['run_id'] / 'observer'
            observed = json.loads((output / 'status.json').read_text())
            assert observed['phase'] == 'preflight_passed' and observed['first_step_completed']
            assert observed['counts'] == {'train': 0, 'validation': 0, 'test': 0}
            assert observed['test_forward_passes'] == 0 and observed['target_line_events'] == 0
            first = json.loads((output / 'first.json').read_text())
            assert first['gradients_finite'] and first['optimizer_updated_parameters']
            assert all(math.isfinite(float(value)) for value in first['components'].values())
            assert math.isfinite(first['gradient_l2'])
            row['evidence'] = {'first_sha256': digest(output / 'first.json'), 'status_sha256': digest(output / 'status.json')}
            write_json(ROOT / 'state/preflight.json', state)
        pairs = {}
        for dataset in ('MELD', 'IEMOCAP'):
            runs = [run for run in registry['runs'] if run['dataset'] == dataset]
            firsts = [json.loads((ROOT / 'preflight' / run['run_id'] / 'observer/first.json').read_text()) for run in runs]
            pairs[dataset] = {'run_ids': [run['run_id'] for run in runs], **compare_first(*firsts)}
        parity = {'created_at_utc': now(), 'status': 'passed', 'pairs': pairs}
        write_json(ROOT / 'preflight/LR_first_step_parity.json', parity)
        validate_capsule()
        assert state['foreign_process_samples'] == 0, 'Foreign GPU process observed; formal gate refused'
        gate = {'status': 'passed', 'created_at_utc': now(), 'formal_runs_authorized': 4,
                'test_forward_passes': 0, 'foreign_process_samples': 0, 'cpu_evidence': cpu, 'first_step_runs': state['runs'],
                'parity_sha256': digest(ROOT / 'preflight/LR_first_step_parity.json'),
                'frozen_execution_sha256': digest(ROOT / 'provenance/frozen_execution.json')}
        write_json(ROOT / 'preflight/gate.json', gate)
        state.update(active=False, phase='passed', completed_at_utc=now(), child_pid=None)
    except BaseException as failure:
        state.update(phase='draining_after_error', error=repr(failure))
        # Do not lose a live child after a query/control exception; never signal any foreign task.
        while child is not None and child.poll() is None:
            state.update(active=True, child_pid=child.pid)
            write_json(ROOT / 'state/preflight.json', state)
            time.sleep(INTERVAL)
        if child is not None and row is not None:
            row.update(returncode=child.returncode, finished_at_utc=now())
        if log is not None:
            log.close()
        state.update(active=False, phase='failed', stopped_at_utc=now(), child_pid=None)
        write_json(ROOT / 'state/preflight.json', state)
        raise
    finally:
        lock.close()
    write_json(ROOT / 'state/preflight.json', state)


if __name__ == '__main__':
    main()
