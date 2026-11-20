"""Persistent fixed-protocol supervisor: isolated two-step checks then formal jobs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import time

D = Path(__file__).resolve().parents[1]
O = D.parent
PYTHON = '/LOCAL_HOME/miniconda3/envs/TEXT_repro/bin/python'


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    status = D / 'results/execution.json'
    if status.exists():
        raise RuntimeError('Execution already recorded: inspect existing processes before resubmitting')
    script = D / 'scripts/extend_overfit.py'
    lock = json.loads((D / 'provenance/launch_code_lock.json').read_text())
    assert hashlib.sha256(script.read_bytes()).hexdigest() == lock['training_script_sha256']
    state = dict(status='RUNNING', started_at=now(), supervisor_pid=os.getpid(), phases={})
    for phase in ['smoke', 'formal']:
        jobs = []
        state['phase'] = phase
        state['phases'][phase] = {}
        for dataset, gpu in [('IEMOCAP', '0'), ('MELD', '2')]:
            query = subprocess.check_output(['nvidia-smi', '-i', gpu,
                '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip()
            uuid, free = [s.strip() for s in query.split(',')]
            assert uuid == lock['gpu_uuids'][gpu] and int(free) > 7000, query
            output = D / 'local_only' / phase / dataset
            cmd = [PYTHON, str(script), '--dataset', dataset, '--old-diagnostic',
                   str(O), '--output', str(output)]
            if phase == 'smoke':
                cmd.append('--smoke')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                PYTHONDONTWRITEBYTECODE='1')
            log = D / 'logs' / f'{phase}_{dataset}.log'
            stream = log.open('x')
            process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                env=env, start_new_session=True)
            state['phases'][phase][dataset] = dict(pid=process.pid, command=cmd,
                gpu=gpu, gpu_uuid=uuid, log=str(log), status='RUNNING')
            jobs.append((dataset, process, stream, output))
            write(status, state)
            time.sleep(2)
        while any(p.poll() is None for _, p, _, _ in jobs):
            time.sleep(5)
        for dataset, process, stream, output in jobs:
            stream.close()
            row = state['phases'][phase][dataset]
            row.update(exit_code=process.returncode, finished_at=now(),
                       status='COMPLETED' if process.returncode == 0 else 'FAILED')
            if process.returncode == 0:
                complete = json.loads((output / 'completion.json').read_text())
                assert complete['status'] == 'COMPLETE'
                if phase == 'smoke':
                    assert complete['global_step'] == 502 and complete['new_optimizer_steps'] == 2
                else:
                    assert complete['global_step'] in (2000, 5000)
            write(status, state)
        if any(p.returncode for _, p, _, _ in jobs):
            state.update(status='FAILED', finished_at=now())
            write(status, state)
            return 1
    state.update(status='COMPLETED', finished_at=now())
    write(status, state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
