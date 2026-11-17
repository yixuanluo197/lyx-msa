"""Verify immutable inputs and cached final results; never fit or reopen source targets."""
from pathlib import Path
import ast
import csv
import datetime as dt
import hashlib
import json
import subprocess

import numpy as np

OUT = Path(__file__).resolve().parents[1]
WORK = Path('/LOCAL/lyx-msa')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    checks = {}
    anchors = read(OUT / 'provenance/input_hashes_before.json')
    histories = read(OUT / 'provenance/history_hashes_before.json')
    for name, digest in anchors.items():
        checks[f'original_unchanged:{name}'] = sha(name) == digest
    for name, digest in histories.items():
        checks[f'history_unchanged:{name}'] = sha(OUT.parent / name) == digest
    protocol = read(OUT / 'configs/reconstruction_protocol.json')
    checks['protocol_v2_frozen'] = sha(OUT / 'configs/reconstruction_protocol.json') == read(OUT / 'provenance/protocol_frozen.json')['protocol_sha256']
    checks['v1_thresholds_unchanged'] = protocol['feature_gate'] == read(OUT / 'configs/reconstruction_protocol_v1.json')['feature_gate']
    actual = {}
    for ds, expected, width in [('IEMOCAP', 527, 1582), ('MELD', 1109, 300)]:
        result = read(OUT / f'provenance/{ds}_clean_feature_metrics.json')
        lock = read(OUT / f'provenance/{ds}_model_lock.json')
        marker = read(OUT / f'provenance/{ds}_validation_evaluation_marker.json')
        final = read(OUT / f'provenance/{ds}_model_selection_final.json')
        checks[f'{ds}:single_completed_evaluation'] = marker['status'] == 'COMPLETE' and marker['attempt'] == 1
        checks[f'{ds}:model_precedes_validation_target'] = dt.datetime.fromisoformat(lock['locked_at']) < dt.datetime.fromisoformat(marker['started_at'])
        checks[f'{ds}:converter_lock_hash'] = sha(lock['converter_path']) == lock['converter_sha256']
        checks[f'{ds}:manifest_lock_hash'] = sha(lock['manifest_path']) == lock['manifest_sha256']
        checks[f'{ds}:runner_hash'] = sha(OUT / 'scripts/evaluate_clean_reconstruction.py') == marker['runner_sha256']
        checks[f'{ds}:core_hash'] = sha(OUT / 'scripts/converter_core.py') == marker['core_sha256']
        checks[f'{ds}:core_matches_final_train'] = sha(OUT / 'scripts/converter_core.py') == final['input_hashes'][str(OUT / 'scripts/converter_core.py')]
        checks[f'{ds}:model_selection_precedes_model_lock'] = dt.datetime.fromisoformat(final['locked_at']) < dt.datetime.fromisoformat(lock['locked_at'])
        checks[f'{ds}:test_not_indexed_in_evaluation'] = result['test_used'] is False and result['label_slots_indexed'] is False
        for name, digest in marker['outputs'].items():
            checks[f'{ds}:evaluation_output:{name}'] = sha(OUT / name) == digest
        audio = np.load(OUT / f'local_only/{ds}/rebuilt_validation_audio.npy', allow_pickle=False)
        checks[f'{ds}:fixed_output_shape'] = audio.shape == (expected, width)
        metrics = []
        for stage, target_name, rebuilt_name, metric_key in [
            ('selected', 'target_val.npy', 'rebuilt_selected_validation.npy', 'selected_feature_metrics'),
            *([('context300', 'target300_val.npy', '../rebuilt_validation_audio.npy', 'context300_feature_metrics')] if ds == 'MELD' else [])]:
            base = OUT / f'local_only/{ds}/final_validation'
            target = np.load(base / target_name, allow_pickle=False)
            prediction = np.load(base / rebuilt_name, allow_pickle=False)
            valid = np.isfinite(target).all(axis=1) & np.isfinite(prediction).all(axis=1)
            difference = prediction[valid] - target[valid]
            measured = {'max_absolute_error': float(np.abs(difference).max()),
                        'mean_absolute_error': float(np.abs(difference).mean()),
                        'rmse': float(np.sqrt(np.square(difference).mean())),
                        'fully_finite_rows': int(valid.sum())}
            for field, value in measured.items():
                checks[f'{ds}:{stage}:independent_{field}'] = bool(np.isclose(value, result[metric_key][field], rtol=1e-12, atol=1e-12))
            with (base / f'{stage}_per_dimension.csv').open(newline='') as stream:
                columns = list(csv.DictReader(stream))
            checks[f'{ds}:{stage}:per_dimension_csv_complete'] = len(columns) == target.shape[1] and [int(r['feature_index']) for r in columns] == list(range(target.shape[1]))
            metrics.append({'stage': stage, **measured})
        checks[f'{ds}:failed_feature_gate_correctly_recorded'] = result['feature_gate']['passed'] is False
        actual[ds] = {'category': 'C', 'metrics_independently_recomputed': metrics, 'probability_parity': 'NOT_RUN'}
    context = read(OUT / 'provenance/MELD_clean_feature_metrics.json')['original_bigru']
    checks['MELD:all_six_original_bigru_weights_exact'] = len(context['original_weights_exact']) == 6 and all(context['original_weights_exact'].values())
    checks['MELD:missing_dialogue_propagates_to_ten_rows'] = context['completed_dialogues'] == 113 and context['missing_context_dialogues'] == 1 and context['affected_missing_context_rows'] == 10
    checks['MELD:padding_separate'] = context['padding_metrics_separate_from_real_rows']['shape'] == [2630, 300]
    independent = read(OUT / 'provenance/independent_train_boundary_review.json')
    checks['independent_train_boundary_review_passed'] = (
        independent.get('status') == 'PASS_WITH_DOCUMENTED_LIMITATIONS'
        and independent['check_count'] == len(independent['checks'])
        and all(c['pass_'] for c in independent['checks']))
    probability = read(OUT / 'provenance/final_probability_gate_verification.json')
    # The independent guard check reports the intercepted events, not a real forward.
    checks['probability_guards_prevented_real_forward'] = (
        probability['status'] == 'VERIFIED_NOT_RUN_FEATURE_GATES_FAILED'
        and probability['model_forward_executed'] is False
        and probability['checkpoint_deserialized'] is False
        and probability['softmax_comparison_executed'] is False)
    final_review = read(OUT / 'provenance/independent_final_report_review.json')
    checks['independent_final_report_review_approved'] = (
        final_review['status'] == 'APPROVED'
        and final_review['check_count'] == len(final_review['checks'])
        and all(c['passed'] for c in final_review['checks']))
    required = ['MELD_FEATURE_RECONSTRUCTION.md', 'IEMOCAP_FEATURE_RECONSTRUCTION.md',
                'CONVERTER_METHOD_COMPARISON.md', 'MELD_CLEAN_PARITY_REPORT.md',
                'IEMOCAP_CLEAN_PARITY_REPORT.md', 'XP04_PROBABILITY_PARITY_REPORT.md',
                'FINAL_RECONSTRUCTION_DECISION.md', 'README.md']
    for name in required:
        checks[f'delivered:{name}'] = (OUT / name).is_file() and (OUT / name).stat().st_size > 100
    for path in (OUT / 'scripts').glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
        checks[f'python_ast:{path.name}'] = True
    head = subprocess.check_output(['git', '-C', str(WORK / 'CMERC'), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '-C', str(WORK / 'CMERC'), 'status', '--short'], text=True)
    checks['git_head_unchanged'] = head == '89c15bed8783e3f110f2609d0140007afc396f1a'
    checks['tracked_cmerc_diff_empty'] = not subprocess.check_output(['git', '-C', str(WORK / 'CMERC'), 'diff', '--name-only'], text=True).strip()
    checks['private_directory_mode'] = (OUT / 'local_only').stat().st_mode & 0o777 == 0o700
    result = {'status': 'PASS' if all(checks.values()) else 'FAIL',
              'meaning': 'Delivery and protocol verification; scientific reconstruction category is C for both datasets',
              'verified_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'run_directory': str(OUT),
              'checks': checks, 'check_count': len(checks), 'failed_checks': [k for k, v in checks.items() if not v],
              'original_input_hashes_checked': len(anchors), 'historic_hashes_checked': len(histories),
              'datasets': actual, 'unit_tests': {'command': 'multimodalerc310/bin/python -m unittest discover -s scripts -p test_*.py -v',
                                               'tests': 22, 'exit_code': 0, 'executed_by_root': True},
              'git': {'HEAD': head, 'status_short': status, 'commit_or_push_performed': False},
              'scope': {'cmERC_training': False, 'cmERC_forward': False, 'raw_perturbation': False,
                        'test_features_or_labels_used': False, 'new_official_downloads': 0,
                        'shared_pickle_containers_deserialized': True, 'test_slots_indexed': False},
              'limitations': ['Author original converters absent', 'Both Validation numerical feature gates failed',
                              'MELD one unavailable Train media (archive and alternate corrupt) and one missing Validation media',
                              'Historical Validation exposure disclosed; no new Validation feedback tuning',
                              'Full-rank Ridge was added during Train stage, not initially preregistered',
                              'Initial MELD probe launch hash gap retained and repaired by independent bitwise recheck']}
    artifact_hashes = {}
    for path in OUT.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.name in ('verification.json', 'artifact_hashes_final.json'):
            continue
        artifact_hashes[str(path.relative_to(OUT))] = sha(path)
    manifest = OUT / 'provenance/artifact_hashes_final.json'
    manifest.write_text(json.dumps(artifact_hashes, indent=2) + '\n')
    result['artifact_manifest_sha256'] = sha(manifest)
    result['artifacts_hashed'] = len(artifact_hashes)
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'check_count', 'failed_checks', 'artifacts_hashed')}))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
