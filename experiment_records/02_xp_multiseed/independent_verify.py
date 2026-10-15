"""Read-only CPU verification after the fixed XP batch completes."""
import csv
import gc
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'batch_manifest.json').read_text())
state = json.loads((ROOT / 'state.json').read_text())
summary = json.loads((ROOT / 'summary.json').read_text())
rows = {row['id']: row for row in csv.DictReader((ROOT / 'results.csv').open())}
assert len(state['tasks']) == len(manifest['tasks']) == len(rows) == 12
assert all(item['status'] == 'completed' for item in state['tasks'].values())
assert summary['all_tasks_terminal'] is True
checks = {}
reference_ids = {}
for task in manifest['tasks']:
    tid = task['id']
    entry = state['tasks'][tid]
    assert entry['training_exitcode'] == entry['evaluation_exitcode'] == 0
    assert entry['training_gpu_uuid'] in manifest['gpu_candidates'].values()
    assert entry['evaluation_gpu_uuid'] in manifest['gpu_candidates'].values()
    run = ROOT / 'runs' / task['variant'] / f"{task['dataset']}_official_test_seed{task['seed']}"
    records = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in records] == list(range(1, task['epochs'] + 1))
    for r in records:
        assert r['causal'] is False and r['effective_norm'] == 'BN'
        assert r['selection_protocol'] == 'official_test' and r['selection_split'] == 'test'
        for name in ('classification_loss', 'cc_loss', 'mscl_loss', 'cscl_loss', 'total_loss', 'test_weighted_f1'):
            assert math.isfinite(r[name]), (tid, r['epoch'], name)
    assert records[-1]['sample_ratio'] == 1.0
    assert records[-1]['train_dialogues'] == (120 if task['dataset'] == 'IEMOCAP' else 1152)
    selected = max(records, key=lambda r: r['selection_weighted_f1'])['epoch']
    assert selected == entry['selected_epoch']
    for filename, epoch in (('latest.pt', task['epochs']), ('best.pt', selected)):
        checkpoint = torch.load(run / filename, map_location='cpu')
        assert checkpoint['epoch'] + 1 == epoch
        assert checkpoint['best_epoch'] + 1 == selected
        assert checkpoint['best_metric'] == records[selected - 1]['selection_weighted_f1']
        assert checkpoint['causal'] is False
        args = checkpoint['args']
        assert args == entry['config']
        assert args['seed'] == task['seed'] and args['Dataset'] == task['dataset']
        assert args['batch_size'] == 16 and args['lr'] == 0.0001 and args['norm'] == 'BN'
        assert args['courselearning'] == (task['variant'] == 'cmerc')
        assert {'python', 'numpy', 'torch', 'cuda'} <= checkpoint['rng_state'].keys()
        assert len(checkpoint['rng_state']['cuda']) == 1
        assert checkpoint['optimizer_state_dict']['state']
        assert all(torch.isfinite(t).all().item() for t in checkpoint['model_state_dict'].values() if torch.is_floating_point(t))
        del checkpoint
        gc.collect()
    result = ROOT / 'results' / task['variant'] / task['dataset'] / 'official_test' / f"seed{task['seed']}"
    metrics = json.loads((result / 'metrics.json').read_text())
    assert metrics['causal'] is False and metrics['effective_norm'] == 'BN'
    assert Path(metrics['checkpoint']) == run / 'best.pt'
    with np.load(result / 'predictions.npz') as prediction:
        labels = prediction['labels']
        predictions = prediction['predictions']
        assert len(labels) == (1623 if task['dataset'] == 'IEMOCAP' else 2610)
        ids = (prediction['dialogue_ids'], prediction['utterance_indices'], labels)
        official = ROOT / 'preflight' / 'official_checkpoint' / task['dataset'] / 'predictions.npz'
        if task['dataset'] not in reference_ids:
            with np.load(official) as ref:
                reference_ids[task['dataset']] = tuple(ref[k].copy() for k in ('dialogue_ids', 'utterance_indices', 'labels'))
        assert all(np.array_equal(a, b) for a, b in zip(ids, reference_ids[task['dataset']]))
        logp = prediction['log_probabilities']
        assert np.isfinite(logp).all()
        assert np.allclose(np.exp(logp).sum(axis=1), 1.0, atol=1e-5)
        assert np.array_equal(logp.argmax(axis=1), predictions)
        wf1 = 100 * f1_score(labels, predictions, average='weighted')
        assert abs(wf1 - metrics['weighted_f1']) < 1e-10
        assert abs(wf1 - float(rows[tid]['weighted_f1'])) < 1e-10
        assert abs(100 * f1_score(labels, predictions, average='macro') - metrics['macro_f1']) < 1e-10
        assert abs(100 * accuracy_score(labels, predictions) - metrics['accuracy']) < 1e-10
        assert np.array_equal(confusion_matrix(labels, predictions), metrics['confusion_matrix'])
        assert abs(round(wf1, 2) - records[selected - 1]['test_weighted_f1']) < 0.011
        names = [name for name in prediction.files if name.startswith('removed_')]
        assert len(names) == 5
        by_removal = {}
        idx = np.arange(len(labels))
        for name in names:
            removed = prediction[name].astype(np.float64)
            assert np.isfinite(removed).all()
            assert np.allclose(np.exp(removed).sum(axis=1), 1.0, atol=1e-5)
            value = float(np.maximum(removed[idx, labels] - logp.astype(np.float64)[idx, labels], 0).sum())
            removal_name = name[len('removed_'):-len('_log_probabilities')]
            assert abs(value - metrics['cel']['by_removal'][removal_name]) < 1e-8
            by_removal[removal_name] = value
        assert abs(sum(by_removal.values()) - metrics['cel']['total']) < 1e-8
    checks[tid] = {'epochs': len(records), 'selected_epoch': selected, 'samples': len(labels), 'weighted_f1': wf1, 'cel': metrics['cel']['total'], 'checkpoint_rng_optimizer_finite': True, 'predictions_metrics_recomputed': True}
    print(json.dumps({'verified': tid, **checks[tid]}), flush=True)
for dataset, group in summary['groups'].items():
    values = [c['weighted_f1'] for tid, c in checks.items() if tid.startswith(dataset + '_cmerc_')]
    assert len(values) == group['completed_n'] == 5
    for key, value in [('mean', statistics.mean(values)), ('sample_std', statistics.stdev(values)), ('minimum', min(values)), ('maximum', max(values))]:
        assert abs(group[key] - value) < 1e-10
    delta = statistics.mean(values) - manifest['paper_weighted_f1'][dataset]
    assert abs(group['difference_points'] - delta) < 1e-10
    assert group['within_half_point'] == (abs(delta) <= 0.5)
for name, expected in manifest['file_sha256'].items():
    assert hashlib.sha256((ROOT / 'code' / name).read_bytes()).hexdigest() == expected, name
for name, expected in manifest['observer_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
patch = manifest['environment_patch']
assert hashlib.sha256(Path(patch['path']).read_bytes()).hexdigest() == patch['after_sha256']
output = {'status': 'passed', 'successful_runs': 12, 'cuda_used_by_verifier': False, 'runs': checks, 'groups_recomputed': summary['groups'], 'frozen_file_and_runtime_hashes': 'passed'}
(ROOT / 'independent_verification.json').write_text(json.dumps(output, indent=2) + '\n')
print('INDEPENDENT_VERIFICATION_PASSED', flush=True)
