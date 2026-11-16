"""Persistent, bounded queue for the eight preregistered local GPU runs."""
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/data2/yb/conda_envs/cmerc-xp-repro/bin/python'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def resources():
    text = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    return {int(row[0]): {'uuid': row[1].strip(), 'free_mib': int(row[2]),
                         'utilization': int(row[3])}
            for row in csv.reader(text.splitlines())}


def main():
    status_path = ROOT / 'logs/queue_status.json'
    if status_path.exists():
        raise RuntimeError('Queue record already exists; inspect it before any relaunch')
    queues = {gpu: [('MELD', modality), ('IEMOCAP', modality)]
              for gpu, modality in enumerate(['text', 'audio', 'visual', 'joint'])}
    active, records = {}, []
    initial = resources()
    assert all(gpu in initial and initial[gpu]['free_mib'] > 4000 for gpu in queues)
    dump(ROOT / 'provenance/scheduler.json', {
        'pid': os.getpid(), 'created_at': now(), 'server': 'local',
        'gpu_inventory': initial, 'queues': queues,
        'concurrency_reason': 'Prior same-card dual jobs saturated compute; one task per idle card.',
    })
    last_sample = 0.
    with (ROOT / 'logs/gpu_resource_samples.jsonl').open('x') as samples:
        while active or any(queues.values()):
            current = resources()
            for gpu in queues:
                if gpu in active:
                    process, stream, record = active[gpu]
                    code = process.poll()
                    if code is None:
                        continue
                    stream.close()
                    record.update(exit_code=code, ended_at=now())
                    dest = ROOT / 'runs' / record['dataset'] / record['modality'] / 'complete.json'
                    record['complete_exists'] = dest.is_file()
                    if code != 0 or not dest.is_file():
                        record['status'] = 'failed'
                        # Preserve other running tasks; do not launch further work on this card.
                        queues[gpu].clear()
                    else:
                        record['status'] = 'completed'
                    del active[gpu]
                if queues[gpu]:
                    if current[gpu]['free_mib'] < 4000:
                        continue
                    dataset, modality = queues[gpu].pop(0)
                    command = [PYTHON, str(ROOT / 'scripts/run_strength.py'),
                               '--dataset', dataset, '--modality', modality]
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                               PYTHONUNBUFFERED='1',
                               LD_LIBRARY_PATH='/data2/yb/conda_envs/cmerc-xp-repro/lib/python3.10/site-packages/nvidia/cusparse/lib')
                    stream = (ROOT / f'logs/{dataset}_{modality}.log').open('x')
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                               env=env, cwd=ROOT, start_new_session=True)
                    record = {'dataset': dataset, 'modality': modality, 'gpu': gpu,
                              'gpu_uuid': current[gpu]['uuid'], 'pid': process.pid,
                              'command': command, 'started_at': now(), 'status': 'running'}
                    records.append(record)
                    active[gpu] = (process, stream, record)
            if time.monotonic() - last_sample > 20:
                samples.write(json.dumps({'utc': now(), 'gpus': current,
                                          'active_pids': [x[0].pid for x in active.values()]}) + '\n')
                samples.flush()
                last_sample = time.monotonic()
            dump(status_path, {'updated_at': now(), 'runs': records,
                               'remaining': queues, 'active': len(active)})
            if active or any(queues.values()):
                time.sleep(3)
    passed = len(records) == 8 and all(r['status'] == 'completed' for r in records)
    dump(ROOT / 'logs/queue_complete.json', {'all_passed': passed, 'runs': records,
                                            'completed_at': now()})
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
