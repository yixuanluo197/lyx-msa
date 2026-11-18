#!/usr/bin/env python3
"""Adopt existing detached trainers; never starts or repeats model training."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, time
OUT=Path(__file__).resolve().parents[1]
PY='/LOCAL_HOME/miniconda3/envs/TEXT_repro/bin/python'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,data):
 p=Path(str(path)+'.tmp');p.write_text(json.dumps(data,indent=2)+'\n');p.replace(path)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def live(pid):
 try:return Path('/proc/%s/stat'%pid).read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
def main():
 path=OUT/'results/execution_status.json';state=json.loads(path.read_text())
 if live(state['supervisor_pid']):raise RuntimeError('Original supervisor still alive; do not duplicate')
 state.update(recovery_supervisor_pid=os.getpid(),recovery_started_at=now(),recovery_reason='Interactive supervisor disappeared following turn interruption; original detached training workers continued. No training restarted. Child exit codes unavailable to adopted monitor; completion additionally requires summary, best lock and complete log event.')
 write(path,state)
 pending=set(state['jobs'])
 while pending:
  for ds in list(pending):
   job=state['jobs'][ds]
   if live(job['pid']):continue
   run=OUT/'local_only'/ds/'training';summary=run/'training_summary.json';lock=run/'best_model_lock.json'
   log=Path(job['log']).read_text()
   valid=summary.exists() and lock.exists() and json.loads(summary.read_text()).get('training_complete') is True and '"event": "complete"' in log
   job.update(status='COMPLETED_FROM_ARTIFACTS' if valid else 'FAILED',exit_code=None,completion_verified_at=now());pending.remove(ds);write(path,state)
   print(ds,job['status'],flush=True)
  if pending:time.sleep(10)
 if any(j['status']=='FAILED' for j in state['jobs'].values()):
  state.update(status='TRAINING_FAILURE',finished_at=now());write(path,state);return 1
 extras=[OUT/'scripts/evaluate_student.py',OUT/'scripts/prepare_distillation_data.py']+list((OUT/'configs').glob('*.json'))+list((OUT/'local_only/tokenizer').glob('*'))
 for ds in state['jobs']:
  if (OUT/'local_only'/ds/'validation/FINAL_VALIDATION_STARTED.json').exists():raise RuntimeError('Final Validation already started; manual saved-result recovery needed')
  p=OUT/'local_only'/ds/'training/best_model_lock.json';lock=json.loads(p.read_text())
  for f in extras:
   if f.is_file():lock['files'][str(f.resolve())]=sha(f)
  lock['both_students_training_complete_before_official_validation']=True;lock['extraction_frozen_at']=now();write(p,lock)
 state['status']='FINAL_VALIDATION';write(path,state)
 for ds,gpu in [('IEMOCAP','0'),('MELD','2')]:
  cmd=[PY,str(OUT/'scripts/evaluate_student.py'),'--dataset',ds,'--run-dir',str(OUT/'local_only'/ds/'training'),'--output-root',str(OUT),'--gpu',gpu,'--batch-size','4']
  with (OUT/'logs'/f'{ds}_validation.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
  state['jobs'][ds]['validation_exit_code']=code;write(path,state)
  if code:state.update(status='VALIDATION_ENGINEERING_FAILURE',finished_at=now());write(path,state);return code
 code=subprocess.call(['/LOCAL_HOME/miniconda3/envs/multimodalerc310/bin/python',str(OUT/'scripts/write_distillation_reports.py')])
 state.update(status='FEATURE_VALIDATION_COMPLETED' if code==0 else 'REPORT_FAILURE',finished_at=now());write(path,state)
 return code
if __name__=='__main__':raise SystemExit(main())
