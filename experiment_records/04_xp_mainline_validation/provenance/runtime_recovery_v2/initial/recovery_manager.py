"""Durable resource drain, fixed-task recovery, then exclusive-XP3 pipeline."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
B=Path(__file__).resolve().parents[2];HERE=B/'state/recovery';PY='/data2/yb/conda_envs/cmerc-xp-repro/bin/python'
state={'pid':os.getpid(),'status':'waiting_for_valid_gpu3_task','started_at':time.time()}
def read(p):return json.loads(p.read_text())
def save():
 p=HERE/'manager.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
def live(pid):
 p=Path('/proc')/str(pid)/'stat'
 return p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z'
def run(name,script):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 env['LD_LIBRARY_PATH']='/data2/yb/conda_envs/cmerc-xp-repro/lib/python3.10/site-packages/nvidia/cusparse/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
 state['status']=name;save()
 with (B/'logs'/f'{name}_manager.log').open('x') as log:
  p=subprocess.Popen([PY,str(HERE/script)],cwd=B,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  state[name]={'pid':p.pid,'started_at':time.time()};save();rc=p.wait()
 state[name].update(exit_code=rc,finished_at=time.time());save()
 if rc:raise RuntimeError(f'{name} failed with {rc}; inspect dedicated log before retry')
try:
 save();old=read(B/'state/pipeline.json');oldpid=old['pid'];schedulerpid=old['phases']['formal']['pid']
 while True:
  guard=read(B/'state/resource_guard.json')
  if not live(guard['pid']):raise RuntimeError('Drain guard died; do not restart any controller')
  q=read(B/'state/queue.json');active=[j for j in q['jobs'] if j['status'].startswith('running_')]
  state['active']=[{'id':j['id'],'status':j['status'],'pid':j['pid']} for j in active];save()
  if guard['status']=='drained_waiting_recovery' and set(guard['held_indices'])=={0,1,3} and not active:break
  time.sleep(5)
 # These are our verified controller PIDs; no foreign training is signalled.
 cmd=(Path('/proc')/str(schedulerpid)/'cmdline').read_bytes().decode()
 assert str(B/'code/scripts/multiseed_scheduler.py') in cmd and '--state' in cmd
 os.kill(schedulerpid,signal.SIGTERM);state['status']='waiting_original_controller_exit';save()
 deadline=time.monotonic()+90
 while live(oldpid):
  if time.monotonic()>deadline:raise TimeoutError('Original controller did not terminate after drained scheduler stop')
  time.sleep(1)
 old=read(B/'state/pipeline.json');assert old['status']=='failed' and old['phases']['formal']['exit_code']==130
 for name,p in [('pipeline_v1.json',B/'state/pipeline.json'),('queue_before_recovery.json',B/'state/queue.json')]:
  target=HERE/name
  with target.open('x') as out:out.write(p.read_text())
 run('recover_registered_tasks','recover_failed_tasks.py')
 run('continue_registered_matrix','continue_pipeline.py')
 assert read(B/'state/pipeline.json')['status']=='completed'
 (HERE/'release_all').touch();state.update(status='completed',finished_at=time.time());save()
except BaseException as error:
 state.update(status='failed',error=repr(error),finished_at=time.time());save();raise
