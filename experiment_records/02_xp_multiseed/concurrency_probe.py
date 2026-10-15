"""MELD throughput probe only; these altered-curriculum runs are not scores."""
import csv
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
GPU = 'GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a'
ENV = '/data2/yb/conda_envs/cmerc-xp-repro'


def check(owned):
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits']).decode()
    for uuid, pid in csv.reader(processes.splitlines()):
        if uuid.strip() == GPU and int(pid) not in owned:
            raise RuntimeError('unrelated process on concurrency probe GPU')
    free = int(subprocess.check_output(['nvidia-smi','-i',GPU,
        '--query-gpu=memory.free','--format=csv,noheader,nounits']).decode())
    if free < 3000:
        raise RuntimeError('insufficient free memory for MELD probe')


def trial(label, count):
    children = []
    started = time.monotonic()
    for index in range(count):
        check({child.pid for child in children})
        name = f'concurrency_{label}_{index}'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, CMERC_ENV_PREFIX=ENV,
            CMERC_PYTHON=str(ROOT/'profile_python'),
            CMERC_OUTPUT_DIR=str(ROOT/'preflight'/name),
            CMERC_PROFILE_OUTPUT=str(ROOT/'preflight'/(name+'_memory.jsonl')))
        command = ['bash','scripts/with_cmerc_env.sh','bash','scripts/run_training.sh',
                   'MELD','cmerc','official_test','--num-workers','0',
                   '--epochs','1','--epoch_ratio','1.0']
        with (ROOT/'logs'/(name+'.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT/'code',env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
        (ROOT/'preflight'/(name+'_launch.json')).write_text(json.dumps({
            'pid':child.pid,'gpu_uuid':GPU,'command':command,
            'engineering_only':True,'curriculum_override':1.0},indent=2))
        if index+1 < count:
            time.sleep(5)
    codes = [child.wait() for child in children]
    elapsed = time.monotonic()-started
    if any(codes):
        raise RuntimeError(f'MELD {label} probe exit codes {codes}')
    return {'count':count,'elapsed_seconds':elapsed,'dialogs_per_second':count*1152/elapsed,
            'exit_codes':codes}


def main():
    single = trial('single',1)
    paired = trial('pair',2)
    ratio = paired['dialogs_per_second']/single['dialogs_per_second']
    result = {'single':single,'paired':paired,'throughput_ratio':ratio,
              'single_task_throughput':single['dialogs_per_second'],
              'dual_task_total_throughput':paired['dialogs_per_second'],
              'enabled':ratio >= 1.10,'threshold':1.10,'gpu_uuid':GPU,
              'engineering_only':True,'batch_size':16,'precision':'float32',
              'note':'fresh 1-epoch full-curriculum throughput probe; never included in formal scores'}
    (ROOT/'preflight'/'meld_concurrency.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
