"""Durable fixed diagnostic supervisor; two-step engineering check then 500 steps."""
from pathlib import Path
import datetime,json,os,subprocess,time
D=Path(__file__).resolve().parents[1];P=D.parents[1]
PY='/LOCAL_HOME/miniconda3/envs/TEXT_repro/bin/python'
PRE='/LOCAL_HOME/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,data):
 t=Path(str(path)+'.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(path)
def main():
 status=D/'results/overfit_execution.json'
 if status.exists():raise RuntimeError('Existing execution; inspect before any resubmission')
 state={'started_at':now(),'supervisor_pid':os.getpid(),'status':'RUNNING','phases':{}}
 for phase,steps in [('smoke',2),('formal',500)]:
  jobs=[];state['phase']=phase;state['phases'][phase]={}
  for ds,seed,gpu in [('IEMOCAP',1475,'0'),('MELD',67137,'2')]:
   out=D/'local_only'/('overfit_smoke' if phase=='smoke' else 'overfit')/ds
   cmd=[PY,str(D/'scripts/small_sample_overfit.py'),'--dataset',ds,'--data',str(P/'local_only'/ds/'train_prepared.pt'),'--pretrained',PRE,'--output',str(out),'--seed',str(seed),'--steps',str(steps),'--count','128','--micro-batch-size','4','--accumulation','4']
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
   log=D/'logs'/f'overfit_{ds}_{phase}.log';f=log.open('x');p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   state['phases'][phase][ds]={'pid':p.pid,'command':cmd,'gpu':gpu,'log':str(log),'status':'RUNNING'};jobs.append((ds,p,f,out));write(status,state);time.sleep(2)
  while any(p.poll() is None for _,p,_,_ in jobs):time.sleep(5)
  for ds,p,f,out in jobs:
   f.close();record=state['phases'][phase][ds];record.update(exit_code=p.returncode,status='COMPLETED' if p.returncode==0 else 'FAILED',finished_at=now())
   if p.returncode==0:
    result=json.loads((out/'completion.json').read_text());grad=result['gradient_update'];assert result['actual_optimizer_steps']==steps and grad['gradient_all_finite'] and grad['maximum_parameter_update']>0
   write(status,state)
  if any(p.returncode for _,p,_,_ in jobs):state.update(status='FAILED',finished_at=now());write(status,state);return 1
 state.update(status='COMPLETED',finished_at=now());write(status,state);return 0
if __name__=='__main__':raise SystemExit(main())
