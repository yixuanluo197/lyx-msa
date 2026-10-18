import collections,json
from pathlib import Path
b=Path(__file__).resolve().parents[1]
p=b/'state/queue.json'
if p.exists():
 q=json.loads(p.read_text());print('Queue:',q['phase'],dict(collections.Counter(j['status'] for j in q['jobs'])))
 for j in q['jobs']:
  if j['status'].startswith('running_'):
   log=Path(j['run_dir'])/'metrics.jsonl';n=0
   if log.exists():n=len([l for l in log.read_text().splitlines(keepends=True) if l.endswith('\n') and l.strip()])
   print(j['id'],j['status'],f"{n}/{j['epochs']}",'GPU',j.get('host_gpu_index'),'PID',j.get('pid'))
else:print('Formal queue not started')
for name in ['gpu_preflight.json','gpu_preflight_iemocap.json','pipeline.json']:
 p=b/'state'/name
 if p.exists():
  s=json.loads(p.read_text());print(name,s['status'],s.get('phase',''))
