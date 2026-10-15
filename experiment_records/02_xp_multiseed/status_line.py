import collections
import json
from pathlib import Path

root = Path(__file__).resolve().parent
state = json.loads((root/'state.json').read_text())
manifest = json.loads((root/'batch_manifest.json').read_text())
print(dict(collections.Counter(v['status'] for v in state['tasks'].values())))
for task in manifest['tasks']:
    entry = state['tasks'][task['id']]
    if entry['status'] not in ('planned','completed'):
        path = root/'runs'/task['variant']/f"{task['dataset']}_official_test_seed{task['seed']}"/'metrics.jsonl'
        lines = [line for line in path.read_text().splitlines(True) if line.endswith('\n')] if path.exists() else []
        print(task['id'],entry['status'],f"{len(lines)}/{task['epochs']}",
              'GPU',entry.get('gpu_index'),entry.get('error',''))
pipeline = json.loads((root/'pipeline_status.json').read_text())
print('pipeline:',pipeline['status'],pipeline['phase'])
