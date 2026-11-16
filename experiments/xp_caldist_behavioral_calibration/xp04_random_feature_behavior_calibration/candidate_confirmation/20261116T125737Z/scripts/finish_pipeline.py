"""Persistent post-inference analysis and independent verification; no experiment expansion."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/data2/yb/conda_envs/cmerc-xp-repro/bin/python'

def state(**value):
    value['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    p=ROOT/'logs/pipeline_status.json'
    temporary=p.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(p)

def main():
    assert not (ROOT/'logs/pipeline_status.json').exists(),'Pipeline state exists; inspect before recovery'
    pid=json.loads((ROOT/'provenance/launcher_process.json').read_text())['pid']
    state(stage='waiting_for_inference',pid=os.getpid(),launcher_pid=pid)
    completion=ROOT/'logs/queue_complete.json'
    while not completion.exists():
        try:
            os.kill(pid,0)
            status=Path(f'/proc/{pid}/stat').read_text().split()[2]
            if status=='Z':raise RuntimeError('Inference launcher exited without queue_complete.json')
        except ProcessLookupError:
            raise RuntimeError('Inference launcher missing without queue_complete.json')
        time.sleep(5)
    queue=json.loads(completion.read_text())
    assert queue['all_passed'] and len(queue['runs'])==12,'Inference queue failed; inspect individual logs'
    historical=json.loads((ROOT/'historical_preflight/analysis_verification.json').read_text())
    assert historical['all_passed'],'Historical CPU parity required'
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
    for stage,command,log in [
        ('analysis',[PYTHON,str(ROOT/'scripts/analyze_candidates.py')],'full_analysis.log'),
        ('independent_verification',[PYTHON,str(ROOT/'scripts/verify_candidates.py'),'--stage','full'],'independent_verification.log')]:
        state(stage=stage,pid=os.getpid(),command=command)
        with (ROOT/'logs'/log).open('x') as stream:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        assert result.returncode==0,f'{stage} failed ({result.returncode}); inspect logs/{log}'
    audit=json.loads((ROOT/'results/independent_verification.json').read_text())
    assert audit['all_passed'] and audit['stage']=='full'
    state(stage='completed',all_passed=True,pid=os.getpid(),git_delivery_pending=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        state(stage='failed',all_passed=False,pid=os.getpid(),error=f'{type(error).__name__}: {error}')
        raise
