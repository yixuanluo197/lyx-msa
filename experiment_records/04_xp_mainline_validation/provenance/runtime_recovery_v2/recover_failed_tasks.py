"""Journaled resource-only recovery; no model/config/seed changes."""
import hashlib,importlib.util,json,shutil,sys,time
from pathlib import Path
B=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('xp_scheduler_recovery',B/'code/scripts/multiseed_scheduler.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
qpath=B/'state/queue.json';jp=B/'state/recovery/recovery_journal.json'
def write(p,x):m.atomic_json(p,x)
journal=m.read_json(jp) if jp.exists() else {'stages':{}}
queue_lock=m.acquire_lock(qpath.with_suffix('.lock'))
if queue_lock is None:raise RuntimeError('Original queue must be drained and stopped first')
with queue_lock:
 cfg=m.read_json(B/'configs/xp_mainline_validation.json');jobs=m.build_jobs(cfg,B/'runs',B/'logs',B/'results')
 scheduler=m.Scheduler(cfg,jobs,qpath,Path('/tmp/cmerc-gpu-locks'))
 assert not any(j['status'].startswith('running_') for j in scheduler.state['jobs']), 'Cannot recover around live tasks'
 assert not {j['id'] for j in scheduler.state['jobs'] if j['status']=='failed'}-{'IEMOCAP_cmerc_validation_seed2024','MELD_cmerc_validation_seed3407'}
 j=next(j for j in scheduler.state['jobs'] if j['id']=='IEMOCAP_cmerc_validation_seed2024')
 before=m.read_json(B/'state/recovery/iemocap2024_training_verified.json');checkpoint=Path(j['run_dir'])/'best.pt'
 assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before['best_sha256'];m.validate_training(j)
 if j['status']!='completed':
  assert j['status']=='failed' and j['training_exit_code']==0
  oldpath=B/'state/recovery/iemocap2024_original_resource_failure.json'
  if not oldpath.exists():write(oldpath,j)
  assert not Path(j['result_dir']).exists() and not Path(j['evaluation_log']).exists(), 'Interrupted evaluation outputs require explicit audit before retry'
  (B/'state/recovery/release_gpu3').touch();deadline=time.monotonic()+60
  while True:
   lease=m.acquire_lock(Path('/tmp/cmerc-gpu-locks')/'cmerc-GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456.lock')
   if lease is not None:break
   if time.monotonic()>deadline:raise TimeoutError('Exclusive GPU3 release failed')
   time.sleep(.2)
  deadline=time.monotonic()+120
  while True:
   gpu=next(g for g in m.query_gpus() if g.index==3)
   if gpu.idle_healthy(cfg['allowed_gpu_indices'],cfg['minimum_free_gpu_mb'],cfg['maximum_idle_utilization'],cfg['allowed_gpu_uuids']):break
   if time.monotonic()>deadline:raise TimeoutError('GPU3 not idle after controlled drain')
   time.sleep(1)
  j.setdefault('technical_attempt_history',[]).append({'kind':'independent_evaluation_resource_block','original_state':str(oldpath),'training_retained_unchanged':True})
  scheduler.start(j,'evaluation',gpu,lease)
  while j['status'].startswith('running_'):scheduler.refresh(j);time.sleep(1)
  assert j['status']=='completed',j.get('error')
 assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before['best_sha256']
 j.pop('error',None);j.pop('evaluation_idle_wait',None);j['resource_recovered_at']=m.utc_now();scheduler.save()
 journal['stages']['fixed_checkpoint_evaluation']={'status':'completed','task':j['id'],'best_sha256_unchanged':before['best_sha256'],'evaluation_pid':j['evaluation_pid'],'evaluation_exit_code':j['evaluation_exit_code']};write(jp,journal)
 j=next(j for j in scheduler.state['jobs'] if j['id']=='MELD_cmerc_validation_seed3407')
 assessment={'registered_task':j['id'],'checkpoint_assessment':'Partial checkpoint exists but attempt ran concurrently with a foreign GPU1 task, violating registered exclusivity; it is not an eligible continuation. Preserve it and restart identical seed/config from scratch on exclusive XP3.','score_used_to_decide_retry':False,'model_or_scientific_protocol_changes':False,'reason':'TOCTOU race between separate project GPU launchers'}
 if j['status']=='failed':
  assert j['training_exit_code']==-15
  stage=journal['stages'].setdefault('invalid_attempt_preservation',{'destination':str(B/'attempts'/j['id']/m.utc_now().replace(':','').replace('+','_')),'moves':[]});write(jp,journal)
  dest=Path(stage['destination']);dest.mkdir(parents=True,exist_ok=True)
  if not (dest/'failed_task.json').exists():write(dest/'failed_task.json',j)
  write(dest/'retry_assessment.json',assessment)
  for key in ('run_dir','result_dir','train_log','evaluation_log'):
   candidates=[Path(j[key])]+([m.memory_profile_path(j[key])] if key.endswith('_log') else [])
   for i,p in enumerate(candidates):
    target=dest/(key+('_memory' if i else ''))
    if p.exists():
     assert not target.exists(),f'Conflicting recovery paths: {p} / {target}'
     shutil.move(str(p),str(target))
    if target.exists():
     record={'source':str(p),'preserved_as':str(target)}
     if record not in stage['moves']:stage['moves'].append(record)
     write(jp,journal)
  base=next(x for x in jobs if x['id']==j['id']).copy();base['technical_attempt_history']=[{'kind':'invalid_partial_due_to_gpu_interference','preserved_attempt':str(dest),'original_training_exit_code':-15,'same_seed_retry':True}]
  j.clear();j.update(base);scheduler.save();stage['status']='completed';write(jp,journal)
 else:assert j['status']=='pending' and j.get('technical_attempt_history')
 scheduler.state['resource_recovery']={'controller_revision':'resource_exclusion_v2','science_source_snapshot_unchanged':m.verify_source_snapshot(),'effective_future_gpu_indices':[3],'enforced_by':'state/recovery/xp3_scheduler.py allocation overlay, independent of guard liveness','reason':'Exclude0/1 from our queue; no external process altered','recovered_evaluation':journal['stages']['fixed_checkpoint_evaluation'],'invalid_partial_retry':assessment,'attempt_preservation':journal['stages']['invalid_attempt_preservation'],'registered_tasks':20}
 scheduler.state['phase']='additional_seeds';scheduler.save();write(B/'provenance/resource_recovery.json',scheduler.state['resource_recovery'])
 journal['status']='completed';write(jp,journal)
 print(json.dumps({'status':'recovery_ready','complete_tasks':sum(x['status']=='completed' for x in scheduler.state['jobs']),'pending_same_registered_tasks':[x['id'] for x in scheduler.state['jobs'] if x['status']=='pending']}),flush=True)
