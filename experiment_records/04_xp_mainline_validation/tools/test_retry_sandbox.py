"""Exercise exit recording and same-task retry without models or any GPU."""
import hashlib,json,shutil,subprocess,sys,tempfile
from pathlib import Path
B=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='cmerc-xp-retry-') as tmp:
 p=Path(tmp)
 for name in ['tools','code/scripts','code/snapshots','state','runs/baseline/example','logs','results']:(p/name).mkdir(parents=True,exist_ok=True)
 shutil.copy2(B/'tools/retry_task.py',p/'tools/retry_task.py');shutil.copy2(B/'code/scripts/multiseed_scheduler.py',p/'code/scripts/multiseed_scheduler.py')
 files={};(p/'code/snapshots/source_snapshot.json').write_text(json.dumps({'files':files,'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}))
 run=p/'runs/baseline/example';(run/'failure.txt').write_text('engineering dummy only')
 log=p/'logs/dummy.train.log'
 with log.open('w') as h:failure=subprocess.run([sys.executable,'-c',"print('intentional preflight exit');raise SystemExit(17)"],stdout=h)
 assert failure.returncode==17
 job={'id':'IEMOCAP_baseline_validation_seed1475','stage':1,'dataset':'IEMOCAP','variant':'baseline','seed':1475,'split_seed':1475,'batch_policy':'capacity_safe','edge_capacity':1000,'node_capacity':5200,'batch_max_retries':10000,'epochs':80,'run_dir':str(run),'result_dir':str(p/'results/baseline'),'train_log':str(log),'evaluation_log':str(p/'logs/dummy.eval.log'),'status':'failed','training_exit_code':17}
 (p/'state/queue.json').write_text(json.dumps({'jobs':[job],'phase':'failed'}))
 result=subprocess.run([sys.executable,str(p/'tools/retry_task.py'),'--task',job['id'],'--reason','intentional engineering failure before any checkpoint'],text=True,capture_output=True)
 assert result.returncode==0,result.stderr
 state=json.loads((p/'state/queue.json').read_text());new=state['jobs'][0]
 assert new['id']==job['id'] and new['seed']==1475 and new['status']=='pending'
 archive=Path(new['prior_attempts'][0]['preserved_attempt'])
 assert json.loads((archive/'failed_task.json').read_text())['training_exit_code']==17
 assert (archive/'run_dir/failure.txt').exists() and (archive/'train_log').read_text().strip()=='intentional preflight exit'
 assert not run.exists() and not log.exists()
 payload={'status':'passed','gpu_workload':False,'intentional_exit_code':17,'same_registered_task_requeued':True,'failed_attempt_and_log_retained':True,'resume_policy':'Existing latest.pt is never archived/restarted by this helper; inspect full-state resume first'}
 (B/'preflight/retry_sandbox.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload))
