"""Supervise one registered first-Adam probe and one validation-only formal run."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
INTERVAL = 10
ALLOWED_PID = 2933763


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def validate_capsule(expected=None):
    path = ROOT / 'provenance/frozen_execution.json'
    current = sha(path)
    assert expected is None or current == expected, 'Frozen capsule changed during execution'
    frozen = read(path)
    assert frozen['files']
    for name, digest in frozen['files'].items():
        target = Path(name) if Path(name).is_absolute() else ROOT / name
        assert sha(target) == digest, 'Frozen input changed: ' + name
    return current


def cpu_evidence():
    records = []
    for name in ('guard_tests.json', 'capacity_cpu.json'):
        path = ROOT / 'preflight' / name
        value = read(path)
        assert value['status'] == 'passed' and value['cuda_initialized'] is False
        if name == 'guard_tests.json':
            assert value['tests'] == 5 and value['failures'] == value['errors'] == 0
        else:
            assert value['epochs_checked'] == 40 and len(value['records']) == 40
        records.append({'path': str(path), 'sha256': sha(path)})
    return records


def resource_snapshot(registry, own_pids=(), stage=None):
    uuid = registry['gpu_uuid']
    query = subprocess.run(['nvidia-smi', '--id=' + uuid,
        '--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=20)
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=20)
    fields = [part.strip() for part in query.stdout.strip().split(',')]
    assert len(fields) == 7 and fields[0] == str(registry['gpu_index']) and fields[1] == uuid
    assert 'TITAN Xp' in fields[2]
    processes = []
    for line in apps.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if parts[0] == uuid:
            processes.append({'pid': int(parts[1]), 'name': parts[2], 'used_memory': parts[3]})
    own = set(own_pids)
    foreign = [value for value in processes if value['pid'] not in own]
    allowed = set(registry['resource_policy']['preexisting_gpu3_pids'])
    return {'time': now(), 'stage': stage, 'gpu_index': int(fields[0]), 'gpu_uuid': uuid,
            'gpu_name': fields[2], 'total_mib': int(fields[3]), 'free_mib': int(fields[4]),
            'used_mib': int(fields[5]), 'utilization_percent': int(fields[6]),
            'processes': processes, 'own_pids': sorted(own), 'all_foreign_processes': foreign,
            'allowed_foreign_processes': [value for value in foreign if value['pid'] in allowed],
            'unexpected_processes': [value for value in foreign if value['pid'] not in allowed]}


def admissible(sample, policy):
    return (not sample['own_pids'] and not sample['unexpected_processes']
            and sample['free_mib'] >= policy['admission_min_free_mib']
            and sample['utilization_percent'] <= policy['admission_max_utilization_percent'])


def observe(registry, state, stage, own_pids=()):
    try:
        sample = resource_snapshot(registry, own_pids, stage)
        append(ROOT / 'logs/resources.jsonl', sample)
    except Exception as error:
        failure = {'time': now(), 'stage': stage, 'own_pids': list(own_pids), 'error': repr(error)}
        append(ROOT / 'logs/resource_errors.jsonl', failure)
        state['last_resource_error'] = failure
        state['resource_query_errors'] += 1
        state['admission_paused'] = True
        write(ROOT / 'state/status.json', state)
        return None
    state['resource_samples'] += 1
    state['preexisting_pid_samples'] += bool(sample['allowed_foreign_processes'])
    state['unexpected_process_samples'] += bool(sample['unexpected_processes'])
    if sample['unexpected_processes']:
        state['unexpected_pids'] = sorted(set(state['unexpected_pids']) | {p['pid'] for p in sample['unexpected_processes']})
    policy = registry['resource_policy']
    pressure = (sample['free_mib'] < policy['admission_min_free_mib']
                or sample['utilization_percent'] > policy['admission_max_utilization_percent'])
    state['admission_conditions_currently_met'] = not sample['unexpected_processes'] and not pressure
    # The utilization threshold is an admission rule; active training may normally exceed it.
    state['admission_paused'] = bool(sample['unexpected_processes'] or (stage.endswith('_admission') and pressure))
    state['last_resource'] = sample
    write(ROOT / 'state/status.json', state)
    return sample


def wait_admission(registry, state, stage):
    policy = registry['resource_policy']
    good = []
    state.update(phase=stage, current_run=registry['runs'][0]['run_id'], active_pids=[])
    while len(good) < policy['samples_required']:
        sample = observe(registry, state, stage)
        if sample is not None and admissible(sample, policy):
            good.append(sample)
        else:
            good.clear()
        state['admission_consecutive_samples'] = len(good)
        write(ROOT / 'state/status.json', state)
        if len(good) < policy['samples_required']:
            time.sleep(policy['sample_seconds'])
    return good


def wait_released(registry, state, pid, stage):
    state.update(phase=stage, active_pids=[])
    query_failures = 0
    while True:
        sample = observe(registry, state, stage, [pid])
        query_failures = query_failures + 1 if sample is None else 0
        assert query_failures < 3, 'Child exited, but three resource queries failed; GPU release cannot be confirmed'
        if sample is not None and pid not in {p['pid'] for p in sample['processes']}:
            return sample
        time.sleep(INTERVAL)


def environment(registry, preflight=False):
    env = os.environ.copy()
    for name in ('PYTHONHASHSEED', 'CMERC_GRADIENT_PROBE', 'CMERC_STOP_AFTER_FIRST_BEST'):
        env.pop(name, None)
    env.update(registry['environment'])
    env['PATH'] = str(Path(registry['python']).parent) + ':' + env.get('PATH', '')
    prefix = 'preflight_' if preflight else ''
    env['CMERC_PROFILE_OUTPUT'] = str(ROOT / 'logs' / (prefix + registry['runs'][0]['run_id'] + '_memory.jsonl'))
    return env


def verify_probe(run):
    folder = ROOT / 'preflight' / run['run_id'] / 'observer'
    status, first = read(folder / 'status.json'), read(folder / 'first.json')
    assert status['phase'] == 'preflight_passed' and status['first_step_completed']
    assert status['counts'] == {'train': 0, 'validation': 0, 'test': 0}
    assert status['test_forward_passes'] == status['target_line_events'] == 0
    assert first['gradients_finite'] and first['optimizer_updated_parameters']
    assert math.isfinite(first['gradient_l2']) and first['gradient_l2'] > 0
    assert all(math.isfinite(value) for value in first['components'].values())
    loss = first['components']
    expected = loss['classification_loss'] + .002 * loss['cc_loss'] + .15 * loss['mscl_loss'] + .15 * loss['cscl_loss']
    assert math.isclose(expected, loss['total_loss'], rel_tol=1e-5, abs_tol=1e-6)
    assert not list((ROOT / 'preflight' / run['run_id']).rglob('final_test.json'))
    return {'status': 'passed', 'first_step_parameters_changed': True, 'test_forward_passes': 0,
            'first_sha256': sha(folder / 'first.json'), 'status_sha256': sha(folder / 'status.json')}


def verify_formal(run):
    folder = Path(run['run_dir'])
    observed = read(ROOT / 'results' / run['run_id'] / 'status.json')
    assert observed['phase'] == 'completed'
    assert observed['counts'] == {'train': 40, 'validation': 40, 'test': 0}
    assert observed['test_forward_passes'] == observed['target_line_events'] == 0
    assert observed['no_final_test_artifacts']
    assert all((folder / name).is_file() for name in ('best.pt', 'latest.pt'))
    assert not (folder / 'final_test.json').exists()
    epochs = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
    assert len(epochs) == 40 and all(row['test_weighted_f1'] is None and row['test_accuracy'] is None for row in epochs)


def validate_gate(capsule_sha, run):
    gate = read(ROOT / 'preflight/gate.json')
    assert gate['status'] == 'passed' and gate['formal_runs_authorized'] == 1 and gate['test_forward_passes'] == 0
    assert gate['frozen_execution_sha256'] == capsule_sha
    assert gate['cpu_evidence'] == cpu_evidence()
    assert gate['probe_evidence'] == verify_probe(run)
    validate_capsule(capsule_sha)
    return gate


def main():
    lock = (ROOT / 'state/execution.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    registry = read(ROOT / 'configs/registry.json')
    assert registry['formal_run_count'] == len(registry['runs']) == 1
    run = registry['runs'][0]
    assert (run['run_id'], run['dataset'], run['seed'], run['lr'], run['epochs']) == ('MELD_cmerc_lr2e-4_seed2024', 'MELD', 2024, .0002, 40)
    assert registry['gpu_index'] == 3 and registry['gpu_uuid'] == 'GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456'
    assert registry['environment']['CUDA_VISIBLE_DEVICES'] == registry['gpu_uuid']
    policy = registry['resource_policy']
    assert policy['preexisting_gpu3_pids'] == [ALLOWED_PID]
    assert (policy['admission_min_free_mib'], policy['admission_max_utilization_percent'], policy['samples_required'], policy['sample_seconds']) == (5000, 65, 3, 10)
    assert not (ROOT / 'state/status.json').exists(), 'Existing supervision state: no implicit retry'
    assert not Path(run['output_root']).exists() and not (ROOT / 'results' / run['run_id']).exists()
    assert not (ROOT / 'preflight' / run['run_id']).exists() and not (ROOT / 'preflight/gate.json').exists()
    capsule_sha = validate_capsule()
    cpu = cpu_evidence()
    state = {'active': True, 'phase': 'preflight_admission', 'pid': os.getpid(), 'started_at_utc': now(),
             'formal_started': 0, 'formal_completed': 0, 'runs': [], 'preflight_started': 0,
             'frozen_execution_sha256': capsule_sha, 'resource_samples': 0, 'resource_query_errors': 0,
             'preexisting_pid_samples': 0, 'unexpected_process_samples': 0, 'unexpected_pids': [],
             'co_resident_pid_explicitly_allowed': ALLOWED_PID, 'admission_paused': False,
             'active_pids': [], 'preflight': None}
    write(ROOT / 'state/status.json', state)
    child, log, row, mode = None, None, None, None
    try:
        for preflight in (True, False):
            mode = 'preflight' if preflight else 'formal'
            if not preflight:
                validate_gate(capsule_sha, run)
            validate_capsule(capsule_sha)
            admissions = wait_admission(registry, state, mode + '_admission')
            assert sha(ROOT / 'provenance/frozen_execution.json') == capsule_sha
            env = environment(registry, preflight)
            argv = [registry['python'], '-u', str(ROOT / 'scripts/memory_observer.py'),
                    str(ROOT / 'scripts/watch_training.py'), '--run-id', run['run_id']]
            if preflight:
                argv.append('--preflight')
            path = ROOT / 'logs' / ((mode + '_') + run['run_id'] + '.log')
            log = path.open('x')
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row = {'run_id': run['run_id'], 'stage': mode, 'pid': child.pid, 'argv': argv, 'started_at_utc': now(),
                   'admission_samples': admissions, 'environment': {name: env[name] for name in registry['environment']},
                   'profile_output': env['CMERC_PROFILE_OUTPUT'], 'log': str(path)}
            if preflight:
                state['preflight_started'] += 1
                state['preflight'] = row
                write(ROOT / 'state/preflight.json', {'active': True, 'phase': 'probing', **row})
            else:
                state['formal_started'] += 1
                state['runs'].append(row)
            state.update(phase=mode + '_running', active_pids=[child.pid])
            write(ROOT / 'state/status.json', state)
            while child.poll() is None:
                observe(registry, state, mode + '_running', [child.pid])
                time.sleep(INTERVAL)
            row.update(returncode=child.returncode, finished_at_utc=now())
            log.close()
            log = None
            assert child.returncode == 0, mode + ' failed: no automatic retry'
            released = wait_released(registry, state, child.pid, mode + '_release')
            row['gpu_release_confirmed'] = released
            child = None
            validate_capsule(capsule_sha)
            if preflight:
                probe = verify_probe(run)
                gate = {'status': 'passed', 'created_at_utc': now(), 'formal_runs_authorized': 1,
                        'test_forward_passes': 0, 'frozen_execution_sha256': capsule_sha,
                        'cpu_evidence': cpu, 'probe_evidence': probe, 'co_resident_pid_allowed': ALLOWED_PID,
                        'preflight_resource_disclosure': {'preexisting_pid_samples': state['preexisting_pid_samples'],
                            'unexpected_process_samples': state['unexpected_process_samples'], 'unexpected_pids': state['unexpected_pids'],
                            'resource_query_errors': state['resource_query_errors']}}
                write(ROOT / 'preflight/gate.json', gate)
                write(ROOT / 'state/preflight.json', {'active': False, 'phase': 'passed', **row, **probe})
            else:
                verify_formal(run)
                state['formal_completed'] += 1
            write(ROOT / 'state/status.json', state)
        state.update(active=False, phase='completed', completed_at_utc=now(), active_pids=[], admission_paused=False,
                     final_frozen_execution_sha256=validate_capsule(capsule_sha))
    except BaseException as failure:
        state.update(phase='draining_after_error', error=repr(failure))
        while child is not None and child.poll() is None:
            state.update(active=True, active_pids=[child.pid])
            observe(registry, state, 'draining_after_error', [child.pid])
            time.sleep(INTERVAL)
        if child is not None and row is not None:
            row.update(returncode=child.returncode, finished_at_utc=now())
        if log is not None:
            log.close()
        if mode == 'preflight' and row is not None:
            write(ROOT / 'state/preflight.json', {'active': False, 'phase': 'failed', **row, 'error': repr(failure)})
        state.update(active=False, phase='failed', stopped_at_utc=now(), active_pids=[])
        write(ROOT / 'state/status.json', state)
        raise
    finally:
        lock.close()
    write(ROOT / 'state/status.json', state)


if __name__ == '__main__':
    main()
