"""Persistent batch-local phase driver; stop at failed prerequisites."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
environment = dict(os.environ, TMPDIR=str(root / 'tmp'),
                   PYTORCH_CUDA_ALLOC_CONF='garbage_collection_threshold:0.8')
steps = [
    ('formal', [sys.executable, str(root / 'code/scripts/xp_multiseed.py'),
                'run', '--batch-root', str(root)]),
    ('summary', [sys.executable, str(root / 'code/scripts/xp_multiseed.py'),
                 'summarize', '--batch-root', str(root)]),
    ('finalize', [sys.executable, str(root / 'finalize.py')]),
]
if json.loads((root/'preflight/status.json').read_text()).get('status') != 'passed':
    raise RuntimeError('preflight must pass before the persistent formal driver starts')
status = {'pid':os.getpid(), 'server':'user23', 'started_at':time.time(), 'phases':{}}


def save():
    temp = root / 'pipeline_status.json.tmp'
    temp.write_text(json.dumps(status,indent=2)+'\n')
    temp.replace(root / 'pipeline_status.json')


for name, command in steps:
    status['phase'] = name
    status['status'] = 'running'
    with (root / 'logs' / ('pipeline_' + name + '.log')).open('x') as log:
        child = subprocess.Popen(command, env=environment, cwd=root / 'code',
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        status['phases'][name] = {'command':command, 'pid':child.pid,
                                 'started_at':time.time(), 'status':'running'}
        save()
        code = child.wait()
    status['phases'][name].update(exit_code=code, finished_at=time.time(),
                                 status='completed' if code == 0 else 'failed')
    save()
    if code and name == 'preflight':
        status.update(status='blocked_preflight',finished_at=time.time())
        save()
        raise SystemExit(code)
status.update(status='terminal',finished_at=time.time())
save()
