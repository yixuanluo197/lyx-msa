"""Wait for the registered queue, then verify and render persisted data on CPU."""
from pathlib import Path
import datetime as dt
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(value):
    path = ROOT/'state/finalization.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    registry = json.loads((ROOT/'configs/registry.json').read_text())
    state = {'active':True,'phase':'waiting_for_single_run','pid':os.getpid(),'commands':[]}
    write(state)
    while True:
        try:
            queue = json.loads((ROOT/'state/status.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            time.sleep(20)
            continue
        if not queue['active']:
            if queue['phase'] != 'completed' or queue['formal_completed'] != 1:
                state.update(active=False,phase='queue_failed',queue=queue,finished_at_utc=now())
                write(state)
                return
            break
        time.sleep(20)
    env = os.environ.copy()
    env.update(registry['environment'])
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    commands = [(registry['python'],'verify_lr.py'),(registry['python'],'analyze_batch_size.py'),
                ('/home/yangbin/miniconda3/envs/multimodalerc310/bin/python','plot_batch_size.py')]
    for python, name in commands:
        argv = [python,'-u',str(ROOT/'scripts'/name)]
        row = {'argv':argv,'started_at_utc':now()}
        state['commands'].append(row)
        state['phase'] = name
        write(state)
        with (ROOT/'logs'/('final_'+name+'.log')).open('x') as stream:
            completed = subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        row.update(returncode=completed.returncode,finished_at_utc=now())
        write(state)
        if completed.returncode:
            state.update(active=False,phase='failed',finished_at_utc=now())
            write(state)
            return
    state.update(active=False,phase='completed',finished_at_utc=now())
    write(state)


if __name__ == '__main__':
    main()
