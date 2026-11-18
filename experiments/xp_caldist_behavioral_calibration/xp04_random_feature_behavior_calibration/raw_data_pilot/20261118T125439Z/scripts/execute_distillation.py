#!/usr/bin/env python3
"""Persistent supervisor: train both students, lock, evaluate once, then report."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, time
OUT = Path(__file__).resolve().parents[1]
PYTHON = '/LOCAL_HOME/miniconda3/envs/TEXT_repro/bin/python'
PRE = '/LOCAL_HOME/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59'
SPECS = [('IEMOCAP',1475,'0,1',29651),('MELD',67137,'2,3',29652)]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,obj):
 t=Path(str(p)+'.tmp'); t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)
def main():
 mode='smoke_b8' if '--profile-b8' in sys.argv else ('smoke' if '--smoke' in sys.argv else 'formal')
 smoke_mode=mode!='formal'
 micro,accum=(8,1) if mode=='smoke_b8' else (4,2) if mode=='formal' else (2,4)
 statuspath=OUT/'results'/(mode+'_supervisor.json' if smoke_mode else 'execution_status.json')
 if statuspath.exists(): raise RuntimeError('Existing supervisor state; inspect before resubmitting')
 processes=[];state={'mode':mode,'started_at':now(),'supervisor_pid':os.getpid(),'jobs':{},'status':'RUNNING'}
 script=OUT/'scripts/train_student.py'
 for ds,seed,gpus,port in SPECS:
  folder=OUT/'local_only'/ds/(mode if smoke_mode else 'training')
  if mode=='formal':
   smoke=json.loads((OUT/'local_only'/ds/'smoke_b8/smoke_result.json').read_text())
   if not smoke.get('smoke_pass'):raise RuntimeError(ds+' smoke not passed')
  prepared=OUT/'local_only'/ds/'train_prepared.pt'
  if not prepared.is_file():raise RuntimeError('Missing '+str(prepared))
  cmd=[PYTHON,'-m','torch.distributed.run','--nproc_per_node=2','--master_port='+str(port),str(script),
       '--data',str(prepared),'--pretrained',PRE,'--run-dir',str(folder),'--tokenizer',str(OUT/'local_only/tokenizer'),
       '--dataset',ds,'--seed',str(seed),'--micro-batch-size',str(micro),'--gradient-accumulation-steps',str(accum),
       '--eval-batch-size','1','--epochs','10','--patience','2','--log-every','20']
  if smoke_mode:cmd+=['--smoke-steps','2']
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',NCCL_IB_DISABLE='1')
  log=OUT/'logs'/f'{ds}_{mode}.log'
  stream=log.open('x');proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  processes.append((ds,proc,stream))
  state['jobs'][ds]={'pid':proc.pid,'command':cmd,'gpu_indices':gpus,'log':str(log),'status':'RUNNING','started_at':now()}
  write(statuspath,state);print(ds,mode,proc.pid,flush=True)
  time.sleep(3)
 while any(p.poll() is None for _,p,_ in processes):
  for ds,p,_ in processes:
   if p.poll() is not None and state['jobs'][ds]['status']=='RUNNING':
    state['jobs'][ds].update(status='COMPLETED' if p.returncode==0 else 'FAILED',exit_code=p.returncode,finished_at=now());write(statuspath,state)
  time.sleep(10)
 for ds,p,f in processes:
  f.close();state['jobs'][ds].update(status='COMPLETED' if p.returncode==0 else 'FAILED',exit_code=p.returncode,finished_at=now())
 if any(p.returncode for _,p,_ in processes):
  state.update(status='FAILED',finished_at=now());write(statuspath,state);return 1
 if smoke_mode:state.update(status='COMPLETED',finished_at=now());write(statuspath,state);return 0
 # Add all extraction decisions to the model lock BEFORE either Validation starts.
 extras=[OUT/'scripts/evaluate_student.py',OUT/'scripts/prepare_distillation_data.py',OUT/'configs/DISTILLATION_PROTOCOL.json',OUT/'configs/TEXT_RECOVERY_PROTOCOL.json',OUT/'configs/TOKENIZATION_PROTOCOL.json']
 extras+=list((OUT/'local_only/tokenizer').glob('*'))
 for ds,*_ in SPECS:
  lp=OUT/'local_only'/ds/'training/best_model_lock.json';lock=json.loads(lp.read_text())
  for path in extras:
   if path.is_file():lock['files'][str(path.resolve())]=sha(path)
  lock['both_students_training_complete_before_official_validation']=True;lock['extraction_frozen_at']=now();write(lp,lock)
 for ds,_,gpus,_ in SPECS:
  cmd=[PYTHON,str(OUT/'scripts/evaluate_student.py'),'--dataset',ds,'--run-dir',str(OUT/'local_only'/ds/'training'),'--output-root',str(OUT),'--gpu',gpus.split(',')[0],'--batch-size','4']
  with (OUT/'logs'/f'{ds}_validation.log').open('x') as stream:
   code=subprocess.call(cmd,stdout=stream,stderr=subprocess.STDOUT)
  state['jobs'][ds]['validation_exit_code']=code;write(statuspath,state)
  if code:state.update(status='VALIDATION_ENGINEERING_FAILURE',finished_at=now());write(statuspath,state);return code
  result=json.loads((OUT/'results'/ds/'feature_parity.json').read_text())
  if result['feature_gate_passed']:
   state.update(status='FEATURE_GATE_PASSED_NEEDS_XP04_PROBABILITY',dataset=ds);write(statuspath,state)
   # Parent handles the conditional probability adapter; never claim success early.
 # Reporting does not change models, features or acceptance gates.
 report=OUT/'scripts/write_distillation_reports.py'
 if report.exists():
  code=subprocess.call(['/LOCAL_HOME/miniconda3/envs/multimodalerc310/bin/python',str(report)])
  if code:state.update(status='REPORT_FAILURE');write(statuspath,state);return code
 state.update(status='FEATURE_VALIDATION_COMPLETED',finished_at=now());write(statuspath,state);return 0
if __name__=='__main__':raise SystemExit(main())
