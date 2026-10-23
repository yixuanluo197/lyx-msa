"""Read-only progress and resource summary."""
import json
from datetime import datetime,timezone
from pathlib import Path

root=Path(__file__).resolve().parents[1]
state=json.loads((root/'state/status.json').read_text())
value={'time':datetime.now(timezone.utc).isoformat(),'phase':state['phase'],
       'started':state['formal_started'],'completed':state['formal_completed'],'runs':[]}
for run in json.loads((root/'configs/registry.json').read_text())['runs']:
    path=Path(run['run_dir'])/'metrics.jsonl'
    if path.exists():
        records=[json.loads(line) for line in path.read_text().splitlines()]
        best=max(records,key=lambda row:row['validation_weighted_f1']) if records else None
        value['runs'].append({'id':run['run_id'],'epochs':len(records),'planned':run['epochs'],
            'best_validation_wf1':best['validation_weighted_f1'] if best else None,
            'best_epoch':best['epoch'] if best else None,
            'last_validation_wf1':records[-1]['validation_weighted_f1'] if records else None,
            'all_epoch_test_fields_null':all(r['test_weighted_f1'] is None for r in records)})
path=root/'logs/resources.jsonl'
if path.exists():
    records=[json.loads(line) for line in path.read_text().splitlines()]
    value['resource_samples']=len(records)
    value['unexpected_samples']=sum(bool(row.get('unexpected_processes')) for row in records)
path=root/'state/finalization.json'
if path.exists():value['finalization']=json.loads(path.read_text())['phase']
if 'error' in state:value['error']=state['error']
print(json.dumps(value,ensure_ascii=False))
