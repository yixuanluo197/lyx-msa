"""Run the four frozen LR trials; admit only registered GPU work, never retry."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
INTERVAL = 10


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def validate_capsule():
    frozen = json.loads((ROOT / 'provenance/frozen_execution.json').read_text())
    assert frozen['files'], 'Empty frozen capsule'
    for filename, expected in frozen['files'].items():
        path = Path(filename)
        if not path.is_absolute():
            path = ROOT / path
        assert digest(path) == expected, 'Frozen input changed: ' + filename


def validate_preflight_gate():
    gate = json.loads((ROOT / 'preflight/gate.json').read_text())
    assert gate['status'] == 'passed'
    assert gate['formal_runs_authorized'] == 4 and gate['test_forward_passes'] == 0
    assert gate['foreign_process_samples'] == 0, 'Preflight observed a foreign GPU process'
    assert gate['frozen_execution_sha256'] == digest(ROOT / 'provenance/frozen_execution.json'), 'Preflight authorized a different frozen capsule'
    expected_cpu = {(ROOT / 'preflight' / name).resolve() for name in ('capacity_cpu.json', 'guard_tests.json')}
    actual_cpu = set()
    for evidence in gate['cpu_evidence']:
        path = Path(evidence['path'])
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        assert path in expected_cpu and path not in actual_cpu, 'Unexpected or duplicate preflight CPU evidence'
        actual_cpu.add(path)
        assert digest(path) == evidence['sha256'], 'Preflight CPU evidence changed: ' + str(path)
        value = json.loads(path.read_text())
        assert value.get('status') == 'passed' or value.get('passed') is True, 'Preflight CPU evidence is not passed'
    assert actual_cpu == expected_cpu, 'Missing required preflight CPU evidence'
    parity_path = ROOT / 'preflight/LR_first_step_parity.json'
    assert digest(parity_path) == gate['parity_sha256'], 'Preflight parity evidence changed'
    parity = json.loads(parity_path.read_text())
    assert parity['status'] == 'passed' and set(parity['pairs']) == {'MELD', 'IEMOCAP'}
    assert all(value['status'] == 'passed' for value in parity['pairs'].values())
    return gate


def execution_lock():
    lock = (ROOT / 'state/execution.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return lock


def child_environment(registry, run_id, preflight=False):
    env = os.environ.copy()
    for key in ('PYTHONHASHSEED', 'CMERC_GRADIENT_PROBE', 'CMERC_STOP_AFTER_FIRST_BEST'):
        env.pop(key, None)
    env.update(registry['environment'])
    assert env['CUDA_VISIBLE_DEVICES'] == registry['gpu_uuid']
    env['PATH'] = str(Path(registry['python']).parent) + ':' + env.get('PATH', '')
    prefix = 'preflight_' if preflight else ''
    env['CMERC_PROFILE_OUTPUT'] = str(ROOT / 'logs' / (prefix + run_id + '_memory.jsonl'))
    return env


def resource_snapshot(registry, own_pids=()):
    uuid = registry['gpu_uuid']
    gpu = subprocess.run(['nvidia-smi', '--id=' + uuid,
        '--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=20)
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=20)
    fields = [part.strip() for part in gpu.stdout.strip().split(',')]
    assert len(fields) == 7 and fields[0] == str(registry['gpu_index']) and fields[1] == uuid
    assert 'TITAN Xp' in fields[2], 'Registered GPU model changed'
    processes = []
    for line in apps.stdout.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if parts[0] == uuid:
            processes.append({'pid': int(parts[1]), 'name': parts[2], 'used_memory': parts[3]})
    own = set(own_pids)
    return {'time': now(), 'gpu_index': int(fields[0]), 'gpu_uuid': uuid, 'gpu_name': fields[2],
            'total_mib': int(fields[3]), 'free_mib': int(fields[4]), 'used_mib': int(fields[5]),
            'utilization_percent': int(fields[6]), 'processes': processes, 'own_pids': sorted(own),
            'unexpected_processes': [row for row in processes if row['pid'] not in own]}


def empty_admission(sample):
    return not sample['processes'] and sample['free_mib'] >= 11500 and sample['utilization_percent'] <= 10


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A live append may be incomplete; the next observation reads it again.
            continue
    return rows


def verify_completed(run):
    observed = json.loads((ROOT / 'results' / run['run_id'] / 'status.json').read_text())
    assert observed['phase'] == 'completed'
    assert observed['counts'] == {'train': run['epochs'], 'validation': run['epochs'], 'test': 0}
    assert observed['test_forward_passes'] == 0 and observed['no_final_test_artifacts']
    assert observed['target_line_events'] == 0
    folder = Path(run['run_dir'])
    assert all((folder / name).is_file() for name in ('best.pt', 'latest.pt'))
    assert not (folder / 'final_test.json').exists()
    epochs = read_jsonl(folder / 'metrics.jsonl')
    assert len(epochs) == run['epochs']
    assert all(row.get('test_weighted_f1') is None for row in epochs)


def parallel_decision(first, samples):
    rows = read_jsonl(Path(first['run']['run_dir']) / 'metrics.jsonl')
    if len(rows) < 3:
        return None
    recent = [sample for sample in samples if sample.get('own_pids') == [first['process'].pid]][-12:]
    if len(recent) < 6:
        return None
    average = sum(sample['utilization_percent'] for sample in recent) / len(recent)
    last = recent[-1]
    memory = read_jsonl(ROOT / 'logs' / (first['run']['run_id'] + '_memory.jsonl'))
    peak = max((row['peak_reserved_mib'] for row in memory), default=float('inf'))
    allowed = (average < 80 and last['free_mib'] >= 3500 and not last['unexpected_processes']
               and peak <= 2300)
    return {'time': now(), 'parallel_allowed': allowed, 'after_completed_epochs': len(rows),
            'recent_sample_count': len(recent), 'recent_mean_utilization_percent': average,
            'free_mib': last['free_mib'], 'observed_peak_reserved_mib': peak if peak != float('inf') else None,
            'per_task_budget_mib': 2300, 'second_task_headroom_mib': 1200,
            'historical_MELD_peak_reserved_mib': 1738,
            'reason': 'Capacity and utilization gate passed' if allowed else 'Conservative serial fallback; capacity or utilization gate did not pass'}


def main():
    lock = execution_lock()
    registry = json.loads((ROOT / 'configs/registry.json').read_text())
    runs = registry['runs']
    assert len(runs) == registry['formal_run_count'] == 4
    assert [run['dataset'] for run in runs] == ['MELD', 'MELD', 'IEMOCAP', 'IEMOCAP']
    assert [run['lr'] for run in runs] == [5e-5, 2e-4, 5e-5, 2e-4]
    validate_preflight_gate()
    validate_capsule()
    assert not any(Path(run['run_dir']).exists() or (ROOT / 'results' / run['run_id']).exists() for run in runs), 'Duplicate/retry refused'
    state = {'active': True, 'phase': 'resource_admission', 'pid': os.getpid(), 'started_at_utc': now(),
             'formal_started': 0, 'formal_completed': 0, 'runs': [], 'concurrency_decision': None}
    active, samples, next_index = {}, [], 0
    error = None
    query_failures = 0
    try:
        while next_index < len(runs) or active:
            for run_id, task in list(active.items()):
                returncode = task['process'].poll()
                if returncode is None:
                    continue
                task['log'].close()
                task['row'].update(returncode=returncode, finished_at_utc=now())
                del active[run_id]
                if returncode:
                    error = 'Formal run failed; no automatic retry: ' + run_id
                else:
                    try:
                        verify_completed(task['run'])
                        state['formal_completed'] += 1
                    except BaseException as failure:
                        error = 'Artifact verification failed: ' + run_id + ': ' + repr(failure)
            try:
                sample = resource_snapshot(registry, [task['process'].pid for task in active.values()])
                append_json(ROOT / 'logs/resources.jsonl', sample)
                samples.append(sample)
                query_failures = 0
            except Exception as failure:
                query_failures += 1
                sample = None
                row = {'time': now(), 'error': repr(failure), 'active_pids': [task['process'].pid for task in active.values()], 'consecutive_failures': query_failures}
                append_json(ROOT / 'logs/resource_errors.jsonl', row)
                state['last_resource_error'] = row
                if query_failures >= 3:
                    error = 'Three resource query failures; draining own children without admitting further work'
            state['active_pids'] = [task['process'].pid for task in active.values()]
            state['current_runs'] = list(active)
            state['admission_paused'] = bool(error or sample is None or sample['unexpected_processes'])
            state['phase'] = 'draining_after_error' if error else ('foreign_process_admission_paused' if sample and sample['unexpected_processes'] else 'training' if active else 'waiting_for_registered_gpu')
            if error:
                state['error'] = error
                if not active:
                    raise RuntimeError(error)
            elif sample is not None and not sample['unexpected_processes'] and next_index < len(runs):
                admit = not active and empty_admission(sample)
                if next_index == 1 and len(active) == 1:
                    first = next(iter(active.values()))
                    if state['concurrency_decision'] is None:
                        decision = parallel_decision(first, samples)
                        if decision is not None:
                            state['concurrency_decision'] = decision
                            write_json(ROOT / 'provenance/MELD_concurrency_decision.json', decision)
                    decision = state['concurrency_decision']
                    admit = bool(decision and decision['parallel_allowed'] and sample['free_mib'] >= 3500)
                if admit:
                    validate_capsule()
                    run = runs[next_index]
                    env = child_environment(registry, run['run_id'])
                    argv = [registry['python'], '-u', str(ROOT / 'scripts/memory_observer.py'),
                            str(ROOT / 'scripts/watch_training.py'), '--run-id', run['run_id']]
                    log_path = ROOT / 'logs' / (run['run_id'] + '.log')
                    log = log_path.open('x')
                    try:
                        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        log.close()
                        raise
                    row = {'run_id': run['run_id'], 'started_at_utc': now(), 'pid': process.pid, 'argv': argv,
                           'environment': {key: env[key] for key in registry['environment']}, 'profile_output': env['CMERC_PROFILE_OUTPUT'],
                           'log': str(log_path), 'admission': sample}
                    active[run['run_id']] = {'run': run, 'process': process, 'row': row, 'log': log}
                    state['runs'].append(row)
                    state['formal_started'] += 1
                    next_index += 1
                    state.update(phase='training', current_runs=list(active),
                                 active_pids=[task['process'].pid for task in active.values()])
            write_json(ROOT / 'state/status.json', state)
            if next_index == len(runs) and not active:
                break
            time.sleep(INTERVAL)
        state.update(active=False, phase='completed', completed_at_utc=now(), current_runs=[], active_pids=[])
    except BaseException as failure:
        # Children remain owned and monitored until exit, even if resource queries fail.
        state.update(phase='draining_after_error', error=repr(failure))
        while active:
            state.update(active=True, current_runs=list(active), active_pids=[task['process'].pid for task in active.values()])
            write_json(ROOT / 'state/status.json', state)
            for run_id, task in list(active.items()):
                code = task['process'].poll()
                if code is not None:
                    task['row'].update(returncode=code, finished_at_utc=now())
                    task['log'].close()
                    del active[run_id]
            if active:
                time.sleep(INTERVAL)
        state.update(active=False, phase='failed', stopped_at_utc=now(), current_runs=[], active_pids=[])
        write_json(ROOT / 'state/status.json', state)
        raise
    finally:
        lock.close()
    write_json(ROOT / 'state/status.json', state)


if __name__ == '__main__':
    main()
