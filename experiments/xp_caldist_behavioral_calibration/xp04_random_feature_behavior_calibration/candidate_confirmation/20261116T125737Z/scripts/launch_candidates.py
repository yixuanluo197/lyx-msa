"""Persistent 12-job queue on explicitly inventoried local GPU UUIDs; no overwrite."""
import argparse
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/data2/yb/conda_envs/cmerc-xp-repro/bin/python'
GPU_INDICES = (0, 1, 2, 3)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def resources():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
    return {int(row[0]): {'uuid': row[1].strip(), 'free_mib': int(row[2]), 'utilization': int(row[3])}
            for row in csv.reader(output.splitlines())}


def jobs():
    return [{'dataset': dataset, 'seed': seed, 'block': block,
             'modality': 'text' if dataset == 'IEMOCAP' else 'audio'}
            for dataset in ['MELD', 'IEMOCAP']
            for seed, block in [(67137, 0), (2024, 0), (3407, 0), (42, 0), (1475, 1), (1475, 2)]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', type=Path, default=ROOT / 'results/preflight_verification.json')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    initial = resources()
    assert set(GPU_INDICES).issubset(initial), 'Local GPU0–3 inventory incomplete'
    assert len({initial[g]['uuid'] for g in GPU_INDICES}) == 4
    if args.dry_run:
        print(json.dumps({'jobs': jobs(), 'gpu_inventory': initial}, indent=2))
        return
    preflight = json.loads(args.preflight.read_text())
    assert preflight.get('all_passed') is True or preflight.get('passed') is True, 'Independent smoke preflight must pass first'
    (ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'logs/queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = ROOT / 'logs/queue_status.json'
    assert not status_path.exists(), 'Queue record exists; inspect before any relaunch'
    pending, active, records = jobs(), {}, []
    for job in pending:
        dest = ROOT / 'runs' / job['dataset'] / f"seed{job['seed']}" / f"block{job['block']}"
        assert not any(p.name != 'smoke' for p in dest.iterdir()) if dest.exists() else True, str(dest)
    dump(ROOT / 'provenance/scheduler.json', {
        'pid': os.getpid(), 'created_at': now(), 'server': 'local', 'gpu_inventory': initial,
        'healthy_gpu_indices': list(GPU_INDICES), 'jobs': pending,
        'preflight_file': str(args.preflight.resolve()),
        'concurrency_reason': 'Historical strength_response queue found same-card dual inference saturates compute; one job per healthy local card, free memory rechecked before each launch.',
        'historical_evidence': str(ROOT.parent.parent / 'strength_response/20261116T035211Z/provenance/scheduler.json'),
        'minimum_free_mib': 4000})
    last_sample, failed = 0., False
    with (ROOT / 'logs/gpu_resource_samples.jsonl').open('x') as samples:
        while active or (pending and not failed):
            current = resources()
            assert all(current[g]['uuid'] == initial[g]['uuid'] for g in GPU_INDICES), 'GPU index/UUID identity changed'
            for gpu in GPU_INDICES:
                if gpu in active:
                    process, stream, record = active[gpu]
                    code = process.poll()
                    if code is None:
                        continue
                    stream.close()
                    record.update(exit_code=code, ended_at=now())
                    complete_path = ROOT / 'runs' / record['dataset'] / f"seed{record['seed']}" / f"block{record['block']}" / 'complete.json'
                    record['complete_exists'] = complete_path.is_file()
                    valid_complete = False
                    if complete_path.is_file():
                        complete = json.loads(complete_path.read_text())
                        valid_complete = (complete.get('status') == 'completed' and complete.get('smoke') is False and
                                          all(complete.get(k) == record[k] for k in ['dataset', 'seed', 'block']))
                    record['status'] = 'completed' if code == 0 and valid_complete else 'failed'
                    failed = failed or record['status'] == 'failed'
                    del active[gpu]
                if not failed and pending and current[gpu]['free_mib'] >= 4000:
                    job = pending.pop(0)
                    command = [PYTHON, str(ROOT / 'scripts/run_candidate.py'), '--dataset', job['dataset'],
                               '--seed', str(job['seed']), '--block', str(job['block'])]
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=current[gpu]['uuid'], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                               LD_LIBRARY_PATH='/data2/yb/conda_envs/cmerc-xp-repro/lib/python3.10/site-packages/nvidia/cusparse/lib')
                    stream = (ROOT / 'logs' / f"{job['dataset']}_seed{job['seed']}_block{job['block']}.log").open('x')
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                               cwd=ROOT, start_new_session=True)
                    record = {**job, 'gpu': gpu, 'gpu_uuid': current[gpu]['uuid'], 'resource_at_start': current[gpu],
                              'pid': process.pid, 'command': command, 'started_at': now(), 'status': 'running'}
                    records.append(record)
                    active[gpu] = (process, stream, record)
            if time.monotonic() - last_sample > 20:
                samples.write(json.dumps({'utc': now(), 'gpus': current, 'active_pids': [x[0].pid for x in active.values()]}) + '\n')
                samples.flush()
                last_sample = time.monotonic()
            dump(status_path, {'updated_at': now(), 'runs': records, 'remaining': pending, 'active': len(active), 'failure_stops_new_launches': failed})
            if active or (pending and not failed):
                time.sleep(3)
    passed = len(records) == 12 and not pending and all(r['status'] == 'completed' for r in records)
    dump(ROOT / 'logs/queue_complete.json', {'all_passed': passed, 'runs': records, 'unlaunched': pending, 'completed_at': now()})
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
