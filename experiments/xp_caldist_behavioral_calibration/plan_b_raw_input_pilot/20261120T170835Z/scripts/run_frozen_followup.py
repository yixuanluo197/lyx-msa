#!/usr/bin/env python3
"""Persistent, no-retry launcher. Verify pre-outcome locks before frozen inference."""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ENV = Path('/home/yangbin/miniconda3/envs')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_locks():
    count = 0
    for name in ['configs/COHORT_LOCK.json','provenance/TEXT_INPUT_LOCK.json']:
        obj = json.loads((RUN/name).read_text())
        for field in ['sources_sha256','outputs_sha256']:
            for path, expected in obj[field].items():
                if digest(path) != expected:
                    raise RuntimeError(f'Frozen input mismatch: {path}')
                count += 1
    review = json.loads((RUN/'provenance/engine_independent_review.json').read_text())
    # Independent review evidence is retained verbatim; machine gates live in engine.
    assert review
    return count


def main():
    destination = RUN/'provenance/execution_status.json'
    if destination.exists():
        raise FileExistsError('Inspect existing persistent run, never automatically duplicate')
    state = dict(status='STARTED',started_at=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(),host=os.uname().nodename,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        locks_checked=validate_locks(),script_sha256=digest(__file__),stages=[])
    def save():
        temp=destination.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(destination)
    save()
    commands = [('encode',ENV/'TEXT_repro/bin/python',['--mode','encode','--device','cuda:0']),
                ('infer',ENV/'multimodalerc310/bin/python',['--mode','infer','--device','cuda:0'])]
    try:
        for stage, python, arguments in commands:
            validate_locks()
            cmd=[str(python),str(RUN/'scripts/run_matched_text.py')]+arguments
            entry=dict(stage=stage,command=cmd,started_at=datetime.now(timezone.utc).isoformat(),
                engine_sha256=digest(RUN/'scripts/run_matched_text.py'))
            state['stages'].append(entry);state['current_stage']=stage;save()
            with (RUN/'logs'/f'{stage}.log').open('x') as log:
                process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                entry['pid']=process.pid;save();code=process.wait()
            entry.update(exit_code=code,ended_at=datetime.now(timezone.utc).isoformat());save()
            if code:
                raise RuntimeError(f'{stage} exited {code}; inspect logs before any new action')
        state['status']='COMPLETE'
    except Exception as error:
        state.update(status='FAILED',error=repr(error));raise
    finally:
        state['ended_at']=datetime.now(timezone.utc).isoformat();save()


if __name__=='__main__':
    main()
