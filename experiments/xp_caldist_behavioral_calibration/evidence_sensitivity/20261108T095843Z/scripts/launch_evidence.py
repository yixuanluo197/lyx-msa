"""Persistent six-job validation-only supervisor; explicit --run required.

Jobs retain full original batch protocol, one target at a time. Phase barriers
are global across both anchors. No retries, training, Test or git operations.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/data2/yb/conda_envs/cmerc-xp-repro/bin/python'
GPU = 3
ENV = dict(CUDA_VISIBLE_DEVICES='3', PYTHONDONTWRITEBYTECODE='1',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           LD_LIBRARY_PATH='/data2/yb/conda_envs/cmerc-xp-repro/lib/python3.10/site-packages/nvidia/cusparse/lib',
           PYTORCH_CUDA_ALLOC_CONF='garbage_collection_threshold:0.8')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    import shutil
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    gpu=[x.strip() for x in raw.splitlines()[GPU].split(',')]
    assert int(gpu[0])==GPU
    processes=[]
    for line in apps.splitlines():
        cells=[x.strip() for x in line.split(',')]
        if len(cells)==3 and cells[1]==gpu[1]:processes.append(dict(pid=int(cells[0]),uuid=cells[1],memory_mib=int(cells[2])))
    disks={p:dict(zip(['total','used','free'],shutil.disk_usage(p))) for p in ['/data1','/data2']}
    return dict(utc=utc(),host=os.uname().nodename,gpu_index=GPU,gpu_uuid=gpu[1],gpu_name=gpu[2],
                gpu_memory_used_mib=int(gpu[3]),gpu_memory_free_mib=int(gpu[4]),gpu_utilization=int(gpu[5]),
                gpu_processes=processes,disks=disks)


def main():
    cfg=json.loads((ROOT/'configs/evidence_sensitivity_config.json').read_text())
    manifest=ROOT/'provenance/execution_jobs.json'
    assert not manifest.exists(),'Refusing to overwrite or silently resume a previous execution manifest'
    frozen=json.loads((ROOT/'provenance/source_hashes.json').read_text())
    assert sha(ROOT/'configs/evidence_sensitivity_config.json')==frozen['config_sha256']
    for p,h in frozen['inference_scripts'].items():assert sha(p)==h,p
    assert json.loads((ROOT/'provenance/audit.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'provenance/graph_cache_parity.json').read_text())['passed']
    data=dict(status='running',started_at=utc(),supervisor_pid=os.getpid(),gpu_host_index=GPU,
              server=os.uname().nodename,environment=ENV,jobs=[],events=[],
              source_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'scripts/run_evidence.py',ROOT/'scripts/evidence_core.py']})
    running={}
    def save():write_json(manifest,data)
    def resource_event(tag):
        r=snapshot();file=ROOT/f'logs/launch_resources_{tag}_{time.time_ns()}.json';write_json(file,r)
        data['events'].append(dict(type='resource_check',tag=tag,file=str(file),utc=utc()))
        save();return r
    try:
        for phase in cfg['phases']:
            data['phase']=phase;save()
            for ds,seed in cfg['anchors'].items():
                # Before every spawn, verify all occupied processes belong to this supervisor.
                r=resource_event(f'{phase}_{ds}_before')
                own={p.pid for p,_ in running.values()}
                assert all(p['pid'] in own for p in r['gpu_processes']),'Foreign GPU3 process appeared; stop before launch'
                assert r['gpu_memory_free_mib']>=4096,'Insufficient GPU3 headroom for safe same-card concurrency'
                assert r['disks']['/data2']['free']>=5*1024**3,'Insufficient /data2 headroom'
                assert r['disks']['/data1']['free']>=1024**3,'Insufficient /data1 headroom'
                done=ROOT/f'runs/{ds}/seed{seed}/{phase}_complete.json'
                assert not done.exists(),'Refusing to overwrite completed job'
                log=ROOT/f'logs/{phase}_{ds}.log'
                cmd=[PYTHON,'-u',str(ROOT/'scripts/run_evidence.py'),'--dataset',ds,'--phase',phase]
                env=os.environ.copy();env.update(ENV)
                with log.open('xb') as f:process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                job=dict(dataset=ds,seed=seed,phase=phase,pid=process.pid,status='running',started_at=utc(),command=cmd,log=str(log),complete_artifact=str(done))
                data['jobs'].append(job);running[ds]=(process,job);save()
                print(json.dumps(dict(event='job_started',**job)),flush=True)
                # Let the first model allocate before reviewing capacity for the second.
                time.sleep(8)
            last=0
            while running:
                for ds,(process,job) in list(running.items()):
                    code=process.poll()
                    if code is None:continue
                    job.update(exit_code=code,finished_at=utc())
                    assert code==0,f'{ds} {phase} exited {code}: {job["log"]}'
                    result=json.loads(Path(job['complete_artifact']).read_text());assert result['passed']
                    folder=Path(job['complete_artifact']).parent/'conditions'
                    for name,h in result['conditions'].items():assert sha(folder/(name+'.npz'))==h,name
                    job.update(status='complete',seconds=result['seconds'],target_forwards=result['target_forwards'],stored_vectors=result['stored_vectors'],skipped_visual_missing=result['skipped_visual_missing'])
                    del running[ds];save();print(json.dumps(dict(event='job_complete',**job)),flush=True)
                if time.monotonic()-last>=60:
                    r=resource_event(f'{phase}_progress');last=time.monotonic()
                    print(json.dumps(dict(event='phase_progress',phase=phase,running=list(running),gpu_memory_free_mib=r['gpu_memory_free_mib'],utc=utc())),flush=True)
                if running:time.sleep(10)
            # Explicit global phase barrier: both completed artifacts must be passed.
            for ds,seed in cfg['anchors'].items():
                assert json.loads((ROOT/f'runs/{ds}/seed{seed}/{phase}_complete.json').read_text())['passed']
            data['events'].append(dict(type='global_phase_barrier_passed',phase=phase,utc=utc()));save()
        data.update(status='complete',completed_at=utc(),successful_jobs=6)
        resource_event('pipeline_complete');save();print(json.dumps(dict(event='pipeline_complete',utc=utc())),flush=True)
    except BaseException as error:
        data.update(status='failed',error=repr(error),traceback=traceback.format_exc(),failed_at=utc())
        for process,job in running.values():
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:code=process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);code=process.wait()
                job.update(status='stopped_after_pipeline_failure',exit_code=code,finished_at=utc())
            else:job.update(status='failed',exit_code=process.returncode,finished_at=utc())
        save();print(data['traceback'],flush=True);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:main()
    else:print('Prepared supervisor. Explicit --run starts only the six frozen validation jobs.')
