"""Run only the preregistered matrix, with GPU admission and no automatic retry."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def resource_snapshot(uuid):
    gpu = subprocess.run(['nvidia-smi', '--id=' + uuid,
        '--query-gpu=uuid,name,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True)
    apps = subprocess.run(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True)
    fields = [part.strip() for part in gpu.stdout.strip().split(',')]
    if len(fields) != 5 or fields[0] != uuid or 'TITAN Xp' not in fields[1]:
        raise RuntimeError('GPU identity query did not match the registered device')
    processes = [line for line in apps.stdout.splitlines() if line.split(',')[0].strip() == uuid]
    return {'time': now(), 'gpu': fields, 'processes': processes,
            'free_mib': int(fields[2]), 'used_mib': int(fields[3]),
            'utilization_percent': int(fields[4])}


def validate_capsule():
    frozen = json.loads((ROOT/'provenance/frozen_execution.json').read_text())
    for filename, expected in frozen['files'].items():
        if digest(ROOT/filename) != expected:
            raise RuntimeError('Frozen execution input changed: ' + filename)


def main():
    lock = (ROOT/'state/supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    registry = json.loads((ROOT/'configs/registry.json').read_text())
    gate = json.loads((ROOT/'preflight/gate.json').read_text())
    if gate['status'] != 'passed':
        raise RuntimeError('Preflight is not approved by its evidence gate')
    if any(Path(run['run_dir']).exists() for run in registry['runs']):
        raise RuntimeError('A formal output already exists; refusing duplicate/retry')
    validate_capsule()
    env = os.environ.copy()
    for key in ['PYTHONHASHSEED', 'CMERC_GRADIENT_PROBE', 'CMERC_STOP_AFTER_FIRST_BEST']:
        env.pop(key, None)
    env.update(registry['environment'])
    env['PATH'] = str(Path(registry['python']).parent) + ':' + env.get('PATH', '')
    state = {'active': True, 'phase': 'resource_admission', 'pid': os.getpid(),
             'started_at_utc': now(), 'formal_started': 0, 'formal_completed': 0, 'runs': []}
    write_json(ROOT/'state/status.json', state)
    try:
        for run in registry['runs']:
            validate_capsule()
            uuid = env['CUDA_VISIBLE_DEVICES']
            while True:
                sample = resource_snapshot(uuid)
                with (ROOT/'logs/resources.jsonl').open('a') as stream:
                    stream.write(json.dumps(sample) + '\n')
                if not sample['processes'] and sample['free_mib'] >= 11500 and sample['utilization_percent'] <= 10:
                    break
                state.update(phase='waiting_for_registered_gpu', waiting_run=run['run_id'])
                write_json(ROOT/'state/status.json', state)
                time.sleep(15)
            env['CMERC_PROFILE_OUTPUT'] = str(ROOT/'logs'/(run['run_id'] + '_memory.jsonl'))
            argv = [registry['python'], '-u', str(ROOT/'scripts/memory_observer.py'),
                    str(ROOT/'scripts/observe_training.py'), '--run-id', run['run_id']]
            row = {'run_id': run['run_id'], 'started_at_utc': now(), 'argv': argv,
                   'environment': {key: env[key] for key in registry['environment']},
                   'profile_output': env['CMERC_PROFILE_OUTPUT'], 'admission': sample,
                   'log': str(ROOT/'logs'/(run['run_id']+'.log'))}
            state['runs'].append(row)
            with Path(row['log']).open('x') as log:
                process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                row['pid'] = process.pid
                state['formal_started'] += 1
                state.update(phase='training', current_run=run['run_id'])
                write_json(ROOT/'state/status.json', state)
                while process.poll() is None:
                    sample = resource_snapshot(uuid)
                    sample.update(run_id=run['run_id'], own_pid=process.pid,
                        unexpected_processes=[line for line in sample['processes']
                            if int(line.split(',')[1].strip()) != process.pid])
                    with (ROOT/'logs/resources.jsonl').open('a') as stream:
                        stream.write(json.dumps(sample) + '\n')
                    time.sleep(10)
                row.update(returncode=process.returncode, finished_at_utc=now())
            if process.returncode:
                raise RuntimeError('Formal run failed; no automatic retry: ' + run['run_id'])
            observed = json.loads((ROOT/'results'/run['run_id']/'status.json').read_text())
            if observed['counts']['test'] != 1 or observed['test_forward_passes'] != 1:
                raise RuntimeError('Final test count is not exactly one')
            if not all((Path(run['run_dir'])/name).is_file() for name in ['best.pt','latest.pt','final_test.json']):
                raise RuntimeError('Expected checkpoint or final test artifact missing')
            state['formal_completed'] += 1
            write_json(ROOT/'state/status.json', state)
        state.update(active=False, phase='completed', completed_at_utc=now())
    except BaseException as error:
        state.update(active=False, phase='failed', error=repr(error), stopped_at_utc=now())
        write_json(ROOT/'state/status.json', state)
        raise
    write_json(ROOT/'state/status.json', state)


if __name__ == '__main__':
    main()
