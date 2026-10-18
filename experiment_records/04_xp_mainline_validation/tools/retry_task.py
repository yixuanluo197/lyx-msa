"""Preserve a technical failed attempt, then requeue that exact registered task."""
import argparse,datetime,fcntl,importlib.util,json,shutil
from pathlib import Path
B=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('xp_scheduler',B/'code/scripts/multiseed_scheduler.py')
import sys
scheduler=importlib.util.module_from_spec(spec);sys.modules[spec.name]=scheduler;spec.loader.exec_module(scheduler)
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True);p.add_argument('--reason',required=True);a=p.parse_args()
state_path=B/'state/queue.json';lock=scheduler.acquire_lock(state_path.with_suffix('.lock'))
if lock is None:raise RuntimeError('Queue is active; stop only this batch supervisor cleanly before explicit retry')
with lock:
 state=json.loads(state_path.read_text());job=next(j for j in state['jobs'] if j['id']==a.task)
 if job['status']!='failed':raise ValueError('Only a recorded technical failure can be requeued')
 if any(j['status'].startswith('running_') and scheduler.pid_identity(j['pid']) for j in state['jobs']):raise RuntimeError('Other batch tasks still active; do not disturb them')
 if scheduler.matching_live_pid(job,job.get('phase','training')):raise RuntimeError('Task process is still live')
 scheduler.verify_source_snapshot()
 if (Path(job['run_dir'])/'latest.pt').exists():
  raise RuntimeError('Checkpoint exists: inspect compatibility and prefer complete epoch-boundary resume; this restart helper only handles failures before a checkpoint')
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 attempt=B/'attempts'/a.task/stamp;attempt.mkdir(parents=True,exist_ok=False)
 (attempt/'failed_task.json').write_text(json.dumps(job,indent=2)+'\n')
 moved=[]
 for key in ['run_dir','result_dir','train_log','evaluation_log']:
  src=Path(job[key]);sources=[src]
  if key.endswith('_log'):sources.append(scheduler.memory_profile_path(src))
  for index,old in enumerate(sources):
   if old.exists():
    target=attempt/(key+('_memory' if index else ''));shutil.move(str(old),str(target));moved.append({'source':str(old),'preserved_as':str(target)})
 marker={'task_id':a.task,'reason':a.reason,'preserved_attempt':str(attempt),'moved':moved,'kind':'fresh restart of identical registered task; not checkpoint resume','scientific_changes_allowed':False}
 (attempt/'retry_record.json').write_text(json.dumps(marker,indent=2)+'\n')
 fresh={key:job[key] for key in ('id','stage','dataset','variant','seed','split_seed','batch_policy','edge_capacity','node_capacity','batch_max_retries','epochs','run_dir','result_dir','train_log','evaluation_log')}
 fresh.update(status='pending',prior_attempts=job.get('prior_attempts',[])+[marker])
 job.clear();job.update(fresh)
 state['phase']='technical_retry_pending';state.pop('first_seed_gate',None);state.pop('final_gate',None)
 scheduler.atomic_json(state_path,state)
 print(json.dumps(marker,indent=2))
