"""Select one healthy local XP card before freezing/launching the sole trial."""
from pathlib import Path
import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEVICES = {0:'GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248',
           1:'GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a',
           3:'GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456'}


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def main():
    lock=(ROOT/'state/selection.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'state/status.json').exists(), 'No duplicate controller permitted'
    assert not (ROOT/'provenance/gpu_selection.json').exists(), 'Device already selected; no implicit relaunch'
    registry=read(ROOT/'provenance/selection_registry_template.json')
    run=registry['runs'][0]
    assert len(registry['runs'])==registry['formal_run_count']==1
    assert not Path(run['output_root']).exists()
    assert not (ROOT/'results'/run['run_id']).exists() and not (ROOT/'preflight'/run['run_id']).exists()
    base=read(ROOT/'provenance/frozen_selection_base_v3.json')
    for n,d in base['files'].items():assert sha(Path(n) if Path(n).is_absolute() else ROOT/n)==d,n
    good={index:[] for index in DEVICES}
    state={'active':True,'phase':'selecting_one_healthy_xp','pid':os.getpid(),'started_at_utc':now(),
           'candidates':DEVICES,'preflight_started':0,'formal_started':0}
    write(ROOT/'state/selection.json',state)
    while True:
        try:
            output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True,timeout=20)
            samples=[]
            for line in output.splitlines():
                fields=[x.strip() for x in line.split(',')];index=int(fields[0])
                if index not in DEVICES:continue
                assert fields[1]==DEVICES[index] and 'TITAN Xp' in fields[2], 'Healthy card identity mismatch'
                row={'time':now(),'gpu_index':index,'gpu_uuid':fields[1],'gpu_name':fields[2],
                     'free_mib':int(fields[3]),'utilization_percent':int(fields[4])}
                samples.append(row)
                if row['free_mib']>=3000 and row['utilization_percent']<=95:good[index].append(row)
                else:good[index]=[]
            assert len(samples)==len(DEVICES)
            with (ROOT/'logs/device_selection.jsonl').open('a') as stream:
                stream.write(json.dumps({'time':now(),'samples':samples})+'\n')
            state.update(last_samples=samples,consecutive={i:len(good[i]) for i in DEVICES})
            write(ROOT/'state/selection.json',state)
            ready=[index for index in DEVICES if len(good[index])>=3]
            if ready:
                # Prefer more spare memory, then lower utilization, then GPU0/1 on ties.
                selected=min(ready,key=lambda i:(-good[i][-1]['free_mib'],good[i][-1]['utilization_percent'],i))
                break
        except Exception as e:
            good={i:[] for i in DEVICES};state.update(last_query_error=repr(e),last_error_at_utc=now())
            write(ROOT/'state/selection.json',state)
        time.sleep(10)
    # Record co-residency, without killing/signalling any external process.
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader,nounits'],text=True,timeout=20)
    processes=[]
    for line in apps.splitlines():
        f=[x.strip() for x in line.split(',')]
        if f[0]==DEVICES[selected]:processes.append({'pid':int(f[1]),'name':f[2],'used_memory':f[3]})
    registry['gpu_index']=selected;registry['gpu_uuid']=DEVICES[selected]
    registry['environment']['CUDA_VISIBLE_DEVICES']=DEVICES[selected]
    registry['resource_policy']['preexisting_gpu3_pids']=sorted(p['pid'] for p in processes)
    registry['resource_policy']['preexisting_memory_mib']=sum(int(p['used_memory']) for p in processes)
    registry['resource_revision']='v3_healthy_xp_selection_before_any_gpu_call'
    registry['hardware_limit']=('Same host user23 and TITAN Xp type; control XP04 seed2024 used physical GPU3; '
        +'selected physical GPU'+str(selected)+'. Sharing, time and CUDA scatter nondeterminism limit strict causal/timing claims.')
    template=read(ROOT/'provenance/selection_registry_template.json')
    assert registry['runs']==template['runs']
    assert {k:v for k,v in registry['environment'].items() if k!='CUDA_VISIBLE_DEVICES'}=={
        k:v for k,v in template['environment'].items() if k!='CUDA_VISIBLE_DEVICES'}
    selection={'created_at_utc':now(),'selected_gpu_index':selected,'selected_gpu_uuid':DEVICES[selected],
        'candidates':DEVICES,'selection_samples':good[selected][-3:],'co_resident_processes_at_selection':processes,
        'same_host_same_gpu_type':True,'xp04_control_physical_gpu_index':3,
        'same_physical_gpu_as_control':selected==3,'preflight_started_before_selection':0,'formal_started_before_selection':0,
        'scientific_run_configuration_unchanged':True,'fixed_for_both_probe_and_formal':True}
    write(ROOT/'configs/registry.json',registry);write(ROOT/'provenance/gpu_selection.json',selection)
    # The chosen registry and evidence become immutable before any GPU process starts.
    files=dict(base['files'])
    for n in ['configs/registry.json','provenance/gpu_selection.json','provenance/frozen_selection_base_v3.json']:
        files[n]=sha(ROOT/n)
    write(ROOT/'provenance/frozen_execution.json',{'created_at_utc':now(),'revision':3,'files':files,
        'selected_gpu_index':selected,'selected_gpu_uuid':DEVICES[selected],
        'exclude':'Post-training analysis/plots/navigation, live logs/state and publication metadata.'})
    for n,d in files.items():assert sha(Path(n) if Path(n).is_absolute() else ROOT/n)==d,n
    state.update(active=False,phase='selected_and_frozen',selected=selection,
                 final_frozen_execution_sha256=sha(ROOT/'provenance/frozen_execution.json'),launched=[])
    for name,script in [('supervisor','run_single.py'),('finalizer','finalize_batch_size.py'),('delivery','finalize_delivery.py')]:
        argv=[sys.executable,'-u',str(ROOT/'scripts'/script)]
        with (ROOT/'logs'/(name+'.log')).open('x') as log:
            child=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launch={'pid':child.pid,'argv':argv,'started_at_utc':now(),'resource_revision':3,
                'frozen_execution_sha256':state['final_frozen_execution_sha256'],'selected_gpu_index':selected,
                'selected_gpu_uuid':DEVICES[selected]}
        write(ROOT/'provenance'/(name+'_launch.json'),launch);state['launched'].append(launch)
    state['completed_at_utc']=now();write(ROOT/'state/selection.json',state)


if __name__=='__main__':main()
