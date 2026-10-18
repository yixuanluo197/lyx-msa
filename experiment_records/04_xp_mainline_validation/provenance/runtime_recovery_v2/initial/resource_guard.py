"""Drain only our CMERC queue; leave all external projects untouched."""
import fcntl,json,os,time
from pathlib import Path
B=Path(__file__).resolve().parents[2]
UUIDS={0:'GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248',1:'GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a',3:'GPU-e53c5e2b-cabd-d4e7-ab11-b12f94557456'}
state={'pid':os.getpid(),'status':'draining_own_tasks','held_indices':[],'reason':'Other project dispatcher on0/1 does not coordinate CMERC GPU mutex. Exclude0/1 from our dispatch; GPU3 remains approved exclusive candidate.'}
handles={}
def save():
 p=B/'state/resource_guard.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
save()
for gpu in [0,1,3]:
 p=Path('/tmp/cmerc-gpu-locks')/f'cmerc-{UUIDS[gpu]}.lock';p.parent.mkdir(exist_ok=True)
 h=p.open('a+');fcntl.flock(h,fcntl.LOCK_EX);handles[gpu]=h;state['held_indices'].append(gpu);save()
state['status']='drained_waiting_recovery';save()
while True:
 # Root releases only3 after terminal checkpoints were inspected and evaluation resumed.
 if (B/'state/recovery/release_gpu3').exists() and 3 in handles:
  handles.pop(3).close();state['held_indices'].remove(3);state['status']='exclude_0_1_only';save()
 if (B/'state/recovery/release_all').exists():break
 time.sleep(1)
for h in handles.values():h.close()
state['status']='released';save()
