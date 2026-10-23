"""Read-only, test-free progress for the registered LR matrix."""
import json
from datetime import datetime, timezone
from pathlib import Path

from supervise_lr import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def main():
    registry = json.loads((ROOT / 'configs/registry.json').read_text())
    path = ROOT / 'state/status.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    value = {'time': datetime.now(timezone.utc).isoformat(), 'phase': state.get('phase', 'not_started'),
             'started': state.get('formal_started', 0), 'completed': state.get('formal_completed', 0),
             'current_runs': state.get('current_runs', []), 'active_pids': state.get('active_pids', []), 'runs': []}
    for run in registry['runs']:
        folder = Path(run['run_dir'])
        rows = read_jsonl(folder / 'metrics.jsonl')
        best = max(rows, key=lambda row: row['validation_weighted_f1']) if rows else None
        watch = ROOT / 'results' / run['run_id'] / 'status.json'
        observed = json.loads(watch.read_text()) if watch.exists() else {}
        value['runs'].append({'id': run['run_id'], 'epochs': len(rows), 'planned': run['epochs'],
            'best_validation_wf1': best['validation_weighted_f1'] if best else None,
            'best_epoch': best['epoch'] if best else None,
            'last_validation_wf1': rows[-1]['validation_weighted_f1'] if rows else None,
            'last10_validation_mean': sum(row['validation_weighted_f1'] for row in rows[-10:]) / min(10, len(rows)) if rows else None,
            'all_epoch_test_fields_null': all(row.get('test_weighted_f1') is None for row in rows),
            'final_test_absent': not (folder / 'final_test.json').exists(),
            'observed_test_forward_passes': observed.get('test_forward_passes'), 'observed_phase': observed.get('phase')})
    samples = read_jsonl(ROOT / 'logs/resources.jsonl')
    value['resource_samples'] = len(samples)
    value['unexpected_process_samples'] = sum(bool(row.get('unexpected_processes')) for row in samples)
    value['last_resource'] = samples[-1] if samples else None
    value['resource_errors'] = len(read_jsonl(ROOT / 'logs/resource_errors.jsonl'))
    value['admission_paused'] = state.get('admission_paused', False)
    if state.get('error'):
        value['error'] = state['error']
    for name in ('finalization', 'delivery'):
        path = ROOT / 'state' / (name + '.json')
        if path.exists():
            value[name] = json.loads(path.read_text()).get('phase')
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
