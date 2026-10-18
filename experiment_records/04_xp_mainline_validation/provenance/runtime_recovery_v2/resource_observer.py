"""Read-only NVML CLI sampling for remaining XP3 formal tasks; no signals/launches."""
import datetime,json,os,subprocess,time
from pathlib import Path
B=Path(__file__).resolve().parents[2];dest=B/'logs/xp3_resource_observer.jsonl';statep=B/'state/recovery/resource_observer.json'
state={'pid':os.getpid(),'status':'running','interval_seconds':5,'coverage':'From observer start onward only; not continuous CUDA-kernel tracing','samples':0,'unexpected_pid_samples':0}
def save():
 t=statep.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(statep)
def descendants(pid):
 result={pid};pending=[pid]
 while pending:
  parent=pending.pop()
  try:children=(Path('/proc')/str(parent)/'task'/str(parent)/'children').read_text().split()
  except OSError:continue
  for x in children:
   child=int(x)
   if child not in result:result.add(child);pending.append(child)
 return result
with dest.open('x') as out:
 save()
 while True:
  record={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  try:
   queue=json.loads((B/'state/queue.json').read_text());active=[j for j in queue['jobs'] if j['status'].startswith('running_') and j.get('host_gpu_index')==3]
   expected=set().union(*(descendants(j['pid']) for j in active)) if active else set()
   record['active_tasks']=[{'id':j['id'],'phase':j['phase'],'pid':j['pid']} for j in active]
   record['gpu_csv']=subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=index,uuid,memory.total,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
   apps=subprocess.check_output(['nvidia-smi','--id=3','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10).strip();record['compute_apps_csv']=apps
   pids={int(row.split(',')[0]) for row in apps.splitlines() if row.split(',')[0].strip().isdigit()}
   unexpected=pids-expected
   # A queue phase can change between the two NVML queries; identify our batch directly too.
   for pid in list(unexpected):
    try:cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode()
    except OSError:cmd=''
    if str(B) in cmd:unexpected.remove(pid)
   record['unexpected_pids']=sorted(unexpected)
   state['unexpected_pid_samples']+=bool(unexpected)
  except Exception as error:record['sampling_error']=repr(error)
  out.write(json.dumps(record)+'\n');out.flush();state['samples']+=1;state['last_sample']=record;save()
  manager=json.loads((B/'state/recovery/manager.json').read_text())
  if manager['status'] in ('completed','failed'):break
  time.sleep(5)
state['status']='completed';state['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
