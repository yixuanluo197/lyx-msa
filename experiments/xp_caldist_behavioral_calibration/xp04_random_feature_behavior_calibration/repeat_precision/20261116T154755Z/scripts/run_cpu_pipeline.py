"""Persistent CPU-only run of the registered precision analysis and independent audit."""
import datetime
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/data2/yb/conda_envs/cmerc-xp-repro/bin/python'
def state(**value):
    value['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    path=ROOT/'logs/pipeline_status.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def main():
    assert not (ROOT/'logs/pipeline_status.json').exists(),'Inspect existing run before relaunch'
    assert json.loads((ROOT/'results/independent_preflight.json').read_text())['all_passed']
    assert json.loads((ROOT/'preflight/analysis_verification.json').read_text())['all_passed']
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1','CUDA_VISIBLE_DEVICES':''}
    for phase,script,arguments,log in [('analysis','analyze_precision.py',[],'full_analysis.log'),('independent_verification','verify_precision.py',['--stage','full'],'independent_verification.log')]:
        command=[PYTHON,str(ROOT/'scripts'/script),*arguments]
        state(stage=phase,pid=os.getpid(),cpu_only=True,command=command)
        with (ROOT/'logs'/log).open('x') as stream:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        assert result.returncode==0,f'{phase} failed; inspect logs/{log}'
    audit=json.loads((ROOT/'results/independent_verification.json').read_text())
    assert audit['all_passed'] and audit['stage']=='full'
    state(stage='completed',pid=os.getpid(),all_passed=True,cpu_only=True,git_delivery_pending=True)
if __name__=='__main__':
    try:main()
    except Exception as error:
        state(stage='failed',pid=os.getpid(),all_passed=False,error=f'{type(error).__name__}: {error}')
        raise
