from pathlib import Path
import json,datetime
p=Path(__file__).resolve().parents[1]
print(datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S UTC'))
for ds in ['IEMOCAP','MELD']:
 rows=[]
 for line in (p/'logs'/f'{ds}_formal.log').read_text().splitlines():
  if line.startswith('{'):
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:pass
 r=rows[-1];hist=p/'local_only'/ds/'training/history.jsonl';epochs=[json.loads(s) for s in hist.read_text().splitlines()] if hist.exists() else []
 print(ds,'event',r.get('event'),'epoch',r.get('epoch'),'step',r.get('global_step'),'completed',len(epochs),'lastdev',epochs[-1]['inner_dev_mse'] if epochs else None,'skips',epochs[-1]['amp_skipped_steps_rank0'] if epochs else 0)
s=p/'results/execution_status.json'
if s.exists():
 state=json.loads(s.read_text());print('supervisor',state['status'],{ds:j['status'] for ds,j in state['jobs'].items()})
