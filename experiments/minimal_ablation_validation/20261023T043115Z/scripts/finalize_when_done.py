"""Wait for the fixed queue, then run only CPU verification and reporting."""
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def save(value):
    path=ROOT/'state/finalization.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    state={'active':True,'phase':'waiting_for_formal_runs','pid':os.getpid(),'commands':[]}
    save(state)
    try:
        while True:
            live=json.loads((ROOT/'state/status.json').read_text())
            if live['phase']=='failed':
                raise RuntimeError('Queue failed; no automatic rerun or extra evaluation')
            if live['phase']=='completed' and not live['active']:
                break
            time.sleep(20)
        config=json.loads((ROOT/'configs/registry.json').read_text())
        env=os.environ.copy();env.update(config['environment']);env['CUDA_VISIBLE_DEVICES']=''
        steps=[(config['python'],'verify_results.py'),(config['python'],'summarize_minimal.py'),
               ('/home/yangbin/miniconda3/envs/multimodalerc310/bin/python','plot_minimal.py')]
        for python,script in steps:
            argv=[python,'-u',str(ROOT/'scripts'/script)]
            item={'argv':argv,'started_at_utc':datetime.now(timezone.utc).isoformat()}
            state.update(phase=script);state['commands'].append(item);save(state)
            with (ROOT/'logs'/('final_'+script+'.log')).open('x') as log:
                result=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            item['returncode']=result.returncode;item['finished_at_utc']=datetime.now(timezone.utc).isoformat();save(state)
            if result.returncode:
                raise RuntimeError('CPU finalization failed: '+script)
        state.update(active=False,phase='completed',completed_at_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        state.update(active=False,phase='failed',error=repr(error));save(state);raise
    save(state)


if __name__=='__main__':
    main()
