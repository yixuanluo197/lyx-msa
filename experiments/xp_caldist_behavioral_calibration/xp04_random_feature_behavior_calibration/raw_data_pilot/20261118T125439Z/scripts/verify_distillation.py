#!/usr/bin/env python3
"""Verify completed dual-dataset distillation artifacts without accessing Test or labels."""
import argparse
import csv
import json
import math
from pathlib import Path

from evaluate_student import SPECS, FROZEN_PROTOCOL, FROZEN_PROTOCOL_SHA256, sha256, verify_lock, write_json, now


def check_dataset(output_root, dataset):
    problems = []
    run_dir = output_root / 'local_only' / dataset / 'training'
    result_dir = output_root / 'results' / dataset
    local_validation = output_root / 'local_only' / dataset / 'validation'
    details = dict(dataset=dataset)
    try:
        lock, model_dir, _, lock_sha = verify_lock(run_dir, dataset)
        details['best_model_lock_sha256'] = lock_sha
        details['best_model_dir'] = str(model_dir)
    except Exception as exc:
        problems.append('Best model lock: ' + str(exc))
        return dict(verified=False, errors=problems, **details)
    # A lock alone cannot substitute for evidence of actual completed training.
    evidence = ['training_summary.json', 'training_config.json', 'history.jsonl', 'best_selection.json']
    missing = [name for name in evidence if not (run_dir / name).is_file()]
    if missing:
        problems.append('Missing training evidence: ' + ', '.join(missing))
    else:
        summary = json.loads((run_dir / 'training_summary.json').read_text())
        config = json.loads((run_dir / 'training_config.json').read_text())
        selection = json.loads((run_dir / 'best_selection.json').read_text())
        history = [json.loads(line) for line in (run_dir / 'history.jsonl').read_text().splitlines() if line.strip()]
        epochs = summary.get('completed_epochs', 0)
        if summary.get('training_complete') is not True or summary.get('smoke_only') is not False:
            problems.append('Training was not completed formally')
        if not (1 <= epochs <= 10) or [row.get('epoch') for row in history] != list(range(1, epochs + 1)):
            problems.append('Training history must contain complete epochs 1..N, N<=10')
        finite_losses = bool(history) and all(math.isfinite(row.get('inner_dev_mse', float('nan')))
                                             and math.isfinite(row.get('train_mse', float('nan'))) for row in history)
        if not finite_losses:
            problems.append('Training losses absent or nonfinite')
        else:
            # min_delta defaults to 0; allow the recorded early-stop tolerance if changed before training.
            minimum = min(row['inner_dev_mse'] for row in history)
            tolerance = float(config.get('min_delta', 0)) + 1e-12
            if abs(summary.get('best_inner_dev_mse', float('inf')) - minimum) > tolerance:
                problems.append('Best epoch does not minimize Train-internal dev loss')
        if (summary.get('best_epoch') != selection.get('epoch') or
                summary.get('best_epoch') != lock.get('best_epoch') or
                summary.get('best_inner_dev_mse') != selection.get('inner_dev_mse') or
                summary.get('best_inner_dev_mse') != lock.get('best_inner_dev_mse')):
            problems.append('Best selection, training summary and lock disagree')
        diagnostic = summary.get('gradient_update_check') or {}
        if (diagnostic.get('gradient_present') is not True or
                diagnostic.get('gradient_all_finite') is not True or
                not (diagnostic.get('maximum_parameter_update', 0) > 0) or
                not math.isfinite(diagnostic.get('gradient_l2') or float('nan'))):
            problems.append('No verified finite encoder gradient and nonzero parameter update')
        if summary.get('official_validation_accessed') is not False or summary.get('test_accessed') is not False:
            problems.append('Training boundary declarations invalid')
        latest_path = run_dir / 'latest_training.pt'
        # The trainer may use latest.pt; existence/size is enough here, without deserializing optimizer state.
        if not latest_path.is_file():
            latest_path = run_dir / 'latest.pt'
        if not latest_path.is_file() or latest_path.stat().st_size < 1000000:
            problems.append('Missing complete epoch-boundary resume checkpoint')
        details['training'] = dict(completed_epochs=epochs, best_epoch=summary.get('best_epoch'),
                                   best_inner_dev_mse=summary.get('best_inner_dev_mse'),
                                   global_steps=summary.get('global_steps'))
    parity_path = result_dir / 'feature_parity.json'
    if not parity_path.exists():
        problems.append('Missing final feature parity result')
        return dict(verified=False, errors=problems, **details)
    parity = json.loads(parity_path.read_text())
    spec = SPECS[dataset]
    for key, expected in [('dataset', dataset), ('rows', spec['rows']), ('dialogues', spec['dialogues']),
                          ('shape', [spec['rows'], 4, 1024]), ('best_model_lock_sha256', lock_sha),
                          ('protocol_sha256', FROZEN_PROTOCOL_SHA256), ('labels_accessed', False),
                          ('test_dialogues_indexed', False), ('validation_feedback_used_for_selection', False)]:
        if parity.get(key) != expected:
            problems.append('Parity field mismatch: ' + key)
    layers = parity.get('layers', [])
    if len(layers) != 4:
        problems.append('Missing four per-layer results')
    actual_passed = len(layers) == 4 and all(row.get('passed') is True for row in layers)
    if parity.get('feature_gate_passed') != actual_passed:
        problems.append('Feature gate summary disagrees with four individual layers')
    for row in layers:
        if row.get('passed') is True and (not row.get('checks') or not all(row['checks'].values())):
            problems.append('Passing layer contains failed/absent checks')
    features = local_validation / 'student_features.npz'
    if not features.exists() or sha256(features) != parity.get('student_features_sha256'):
        problems.append('Final student feature hash mismatch')
    for name in ['FINAL_VALIDATION_STARTED.json', 'FINAL_VALIDATION_COMPLETED.json']:
        path = local_validation / name
        if not path.exists():
            problems.append('Missing once-only marker ' + name)
        else:
            marker = json.loads(path.read_text())
            if marker.get('best_model_lock_sha256') != lock_sha:
                problems.append('Final Validation marker lock mismatch: ' + name)
            if name == 'FINAL_VALIDATION_COMPLETED.json' and marker.get('result_sha256') != sha256(parity_path):
                problems.append('Final parity result changed since completion')
    manifest = local_validation / 'validation_manifest.csv'
    if manifest.exists():
        with manifest.open() as stream:
            records = list(csv.DictReader(stream))
        keys = [(row['dialogue_id'], row['utterance_index']) for row in records]
        if len(keys) != spec['rows'] or len(set(keys)) != len(keys):
            problems.append('Validation manifest coverage/uniqueness failed')
    else:
        problems.append('Missing Validation manifest')
    probability_path = result_dir / 'probability_parity.json'
    if not actual_passed:
        if parity.get('probability_status') != 'NOT_RUN_FEATURE_GATE_FAILED' or parity.get('probability_metrics') is not None:
            problems.append('Failed feature gate must keep probability NOT_RUN and null')
        if probability_path.exists():
            probability = json.loads(probability_path.read_text())
            if probability.get('status', '').startswith('NOT_RUN') is False:
                problems.append('Probability computation performed after failed feature gate')
    elif not probability_path.exists():
        problems.append('Passing feature gate requires final frozen-XP04 probability parity result')
    details.update(feature_gate_passed=actual_passed,
                   probability_status=parity.get('probability_status'),
                   final_feature_rows=parity.get('rows'))
    # Train boundary manifests are allowed to use several prep naming conventions;
    # require exactly one effective Train manifest and partition it by dialogue.
    candidates = [output_root / 'local_only' / dataset / name for name in
                  ['train_manifest.csv', 'manifest_train.csv', 'train/train_manifest.csv', 'train/manifest.csv']]
    candidates += list((output_root / 'local_only' / dataset).glob('*train*manifest*.csv'))
    train_manifest = next((path for path in candidates if path.is_file()), None)
    if train_manifest is None:
        problems.append('Missing effective Train manifest')
    else:
        with train_manifest.open() as stream:
            train_rows = list(csv.DictReader(stream))
        if len(train_rows) != spec['train_rows']:
            problems.append('Effective Train manifest count mismatch')
        train_dialogues = {row['dialogue_id'] for row in train_rows}
        val_dialogues = set()
        if manifest.exists():
            with manifest.open() as stream:
                val_dialogues = {row['dialogue_id'] for row in csv.DictReader(stream)}
        if train_dialogues & val_dialogues:
            problems.append('Effective Train/Validation dialogue overlap')
        # Preserve training subset labels without ever reading emotion labels.
        split_column = next((name for name in ['inner_subset', 'inner_split', 'subset', 'training_subset', 'partition']
                             if train_rows and name in train_rows[0]), None)
        if split_column:
            per_dialogue = {}
            for row in train_rows:
                per_dialogue.setdefault(row['dialogue_id'], set()).add(row[split_column])
            if any(len(values) != 1 for values in per_dialogue.values()):
                problems.append('Internal Train fit/dev split cuts through a dialogue')
        details['train_manifest'] = str(train_manifest)
        details['train_rows'] = len(train_rows)
    return dict(verified=not problems, errors=problems, **details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-report-check', action='store_true')
    args = parser.parse_args()
    out = args.output_root.resolve()
    results = {dataset: check_dataset(out, dataset) for dataset in SPECS}
    protocol_ok = sha256(FROZEN_PROTOCOL) == FROZEN_PROTOCOL_SHA256
    input_errors = []
    inputs_path = out / 'provenance/input_hashes_before.json'
    if not inputs_path.is_file():
        input_errors.append('Missing original input pre-run hashes')
    else:
        inputs = json.loads(inputs_path.read_text())
        # Parent records a path -> {sha256,bytes} dictionary, optionally under files.
        inputs = inputs.get('files', inputs)
        checked_input_files = 0
        for name, record in inputs.items():
            if not isinstance(record, dict) or 'sha256' not in record:
                continue
            checked_input_files += 1
            path = Path(name)
            if not path.is_absolute():
                path = Path(record.get('path', name))
            try:
                if sha256(path) != record['sha256']:
                    input_errors.append('Original input changed: ' + str(path))
                if 'bytes' in record and path.stat().st_size != record['bytes']:
                    input_errors.append('Original input size changed: ' + str(path))
            except OSError as exc:
                input_errors.append('Original input unavailable: ' + str(exc))
        if checked_input_files < 6:
            input_errors.append('Insufficient original input hash records')
    report_errors = []
    if not args.skip_report_check:
        report_errors = ['Missing report ' + name for name in ['DISTILLATION_REPORT.md', 'FINAL_DISTILLATION_DECISION.md']
                         if not (out / name).is_file()]
    verified = protocol_ok and not input_errors and not report_errors and all(row['verified'] for row in results.values())
    result = dict(created_at=now(), verified=verified, protocol_hash_unchanged=protocol_ok,
                  datasets=results, original_input_integrity_errors=input_errors, report_errors=report_errors,
                  gates_are_not_research_success=True,
                  scope=dict(CMERC_retrained=False, Test_used=False, raw_perturbation_performed=False),
                  interpretation='verified=true means complete, honest, internally consistent artifacts; feature gates may fail')
    path = args.output or out / 'verification.json'
    write_json(path, result)
    print(json.dumps(result, indent=2))
    return 0 if verified else 1


if __name__ == '__main__':
    raise SystemExit(main())
