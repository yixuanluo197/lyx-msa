#!/usr/bin/env python3
"""CPU-only diagnostics of fixed author-checkpoint predictions; never run a model.

All rates and F1 values are fractions. No probabilities are renormalized or
clipped. High-confidence detail rows repeat at each qualifying threshold.
"""
import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from scripts.high_confidence_calibration import summarize_calibration
from scripts.high_confidence_followup_stats import BINS, THRESHOLDS, _validate


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows, fields=None):
    if not rows and fields is None:
        raise ValueError(f'Refusing to infer empty output schema: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def high_confidence_counts(rows):
    result = {}
    for threshold in THRESHOLDS:
        suffix = str(threshold).replace('.', 'p')
        confident = [r for r in rows if r['predicted_confidence'] >= threshold]
        errors = sum(not r['correct'] for r in confident)
        result.update({f'errors_ge_{suffix}': errors,
                       f'high_confidence_predictions_ge_{suffix}': len(confident),
                       f'high_confidence_error_rate_ge_{suffix}': errors / len(confident) if confident else None})
    return result


def error_distribution(rows):
    errors = [r for r in rows if not r['correct']]
    result = []
    for index, (low, high) in enumerate(BINS):
        count = sum(low <= r['predicted_confidence'] and
                    (r['predicted_confidence'] < high or
                     index == len(BINS)-1 and r['predicted_confidence'] <= high)
                    for r in errors)
        result.append(dict(dataset=rows[0]['dataset'], model=rows[0]['model'],
                           confidence_range=f'[{low:g},{high:g}{"]" if index == len(BINS)-1 else ")"}',
                           error_count=count, ratio_of_all_errors=count / len(errors) if errors else None))
    if sum(row['error_count'] for row in result) != len(errors):
        raise ValueError('Confidence bins do not partition all errors')
    return result


def validate_arrays(rows, probabilities, log_probabilities, labels, predictions):
    """Reject misaligned NPZ data and any alteration of the original scores."""
    _validate(rows, (rows[0]['dataset'], rows[0]['model']), 'full')
    fields = [field for field in rows[0] if field.startswith('probability_')]
    values = np.array([[r[field] for field in fields] for r in rows])
    expected_shape = (len(rows), len(fields))
    if probabilities.shape != expected_shape or log_probabilities.shape != expected_shape:
        raise ValueError('NPZ class probability dimensions differ')
    if labels.shape != (len(rows),) or predictions.shape != (len(rows),):
        raise ValueError('NPZ label dimensions differ')
    if not np.array_equal(labels, [r['true_label'] for r in rows]) or not np.array_equal(predictions, [r['predicted_label'] for r in rows]):
        raise ValueError('NPZ label/prediction order differs from CSV')
    if not np.isfinite(log_probabilities).all() or not np.isfinite(probabilities).all():
        raise ValueError('Nonfinite source probability or log probability')
    if not np.array_equal(values, probabilities):
        raise ValueError('CSV probabilities differ from saved NPZ')
    if not np.allclose(probabilities, np.exp(log_probabilities.astype(np.float64)), rtol=0, atol=1e-15):
        raise ValueError('NPZ probabilities differ from exp(original log probabilities)')
    if not np.array_equal(probabilities.argmax(axis=1), predictions):
        raise ValueError('Predictions differ from probability argmax')
    return float(np.max(np.abs(probabilities.sum(axis=1)-1)))


def load_dataset(source, dataset, project):
    metrics = json.loads((source/f'{dataset}_metrics.json').read_text())
    manifest = json.loads((source/f'{dataset}_run_manifest.json').read_text())
    with (source/f'{dataset}_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        for field in ('utterance_index', 'true_label', 'predicted_label'):
            row[field] = int(row[field])
        for field in list(row):
            if field.startswith('probability_') or field in ('predicted_confidence', 'true_label_probability'):
                row[field] = float(row[field])
        row['correct'] = row['predicted_label'] == row['true_label']
    expected_count = {'IEMOCAP': 1623, 'MELD': 2610}[dataset]
    if len(records) != expected_count:
        raise ValueError('Official test utterance count differs')
    expected_fields = ['probability_'+name for name in metrics['class_names']]
    if [field for field in records[0] if field.startswith('probability_')] != expected_fields:
        raise ValueError('Class mapping differs from fixed classification run')
    with np.load(source/f'{dataset}_predictions.npz', allow_pickle=False) as arrays:
        if set(arrays.files) != {'probabilities', 'log_probabilities', 'labels', 'predictions'}:
            raise ValueError('Unknown NPZ source schema')
        max_sum_deviation = validate_arrays(records, **{key: arrays[key] for key in arrays.files})
        logs = arrays['log_probabilities'].astype(np.float64).copy()
    for i, row in enumerate(records):
        row['true_label_log_probability'] = float(logs[i, row['true_label']])
    raw_relative = f'data/{"IEMOCAP_features.pkl" if dataset == "IEMOCAP" else "MELD_features_raw1.pkl"}'
    expected_raw_sha = json.loads((source/'preflight.json').read_text())['input_sha256'][raw_relative]
    raw_path = project/raw_relative
    if sha256(raw_path) != expected_raw_sha:
        raise ValueError('Original trusted raw features differ from verified evaluation')
    # These are existing, SHA-verified project feature files, not arbitrary input.
    with raw_path.open('rb') as stream:
        raw = pickle.load(stream, encoding='latin1')
    ids, labels, sentences, test_ids = raw[0], raw[2], raw[6], raw[8]
    expected_order = [(str(dialogue), i) for dialogue in test_ids for i in range(len(labels[dialogue]))]
    if [(row['dialogue_id'], row['utterance_index']) for row in records] != expected_order:
        raise ValueError('Source CSV dialogue-major flatten order differs from raw test data')
    raw_by_id = {str(dialogue): dialogue for dialogue in test_ids}
    if len(raw_by_id) != len(test_ids):
        raise ValueError('String dialogue IDs are not unique')
    for row in records:
        dialogue, index = raw_by_id[row['dialogue_id']], row['utterance_index']
        if (row['source_utterance_id'] != str(ids[dialogue][index]) or
                row['utterance_text'] != sentences[dialogue][index] or
                row['true_label'] != int(labels[dialogue][index])):
            raise ValueError('Original text, source ID or label does not align')
    batches = manifest['audit']['batches']
    observed_ids = [str(vid) for batch in batches for vid in batch['dialogue_ids']]
    if observed_ids != [str(vid) for vid in test_ids]:
        raise ValueError('Recorded official DataLoader order differs from raw test split')
    if not all(batch['batch_size'] == 16 for batch in batches):
        raise ValueError('Original official batch size differs')
    source_paths = [source/f'{dataset}_{suffix}' for suffix in
                    ('predictions.csv', 'predictions.npz', 'metrics.json', 'run_manifest.json')]
    return records, metrics, dict(dataset=dataset, valid_utterances=len(records),
        raw_data_path=str(raw_path), raw_data_sha256=expected_raw_sha, test_dialogues=len(test_ids),
        original_text_labels_source_ids_and_order_verified=True, padding_excluded=True,
        probability_sum_max_deviation=max_sum_deviation,
        source_sha256={str(path): sha256(path) for path in source_paths})


def generate_dataset_tables(rows, metrics):
    summary, bins = summarize_calibration(rows, n_bins=10)
    # Reuse exactly the established definition, verifying its log(p_true)
    # computation agrees with the stored model log_softmax at machine precision.
    nll_from_logs = -math.fsum(row['true_label_log_probability'] for row in rows) / len(rows)
    if not math.isclose(summary['nll'], nll_from_logs, rel_tol=0, abs_tol=1e-12):
        raise ValueError('Existing probability-based NLL differs from original log-softmax NLL')
    summary['nll'] = nll_from_logs
    support = Counter(row['true_label'] for row in rows)
    predicted = Counter(row['predicted_label'] for row in rows)
    tp = Counter(row['true_label'] for row in rows if row['correct'])
    f1 = {k: 2*tp[k]/(support[k]+predicted[k]) if support[k]+predicted[k] else 0
          for k in range(len(metrics['class_names']))}
    weighted = math.fsum(support[k]*f1[k] for k in f1) / len(rows)
    macro = math.fsum(f1.values()) / len(f1)
    for name, value in [('accuracy', summary['accuracy']), ('weighted_f1', weighted), ('macro_f1', macro)]:
        if not math.isclose(value*100, metrics[name], rel_tol=0, abs_tol=1e-10):
            raise ValueError(f'{name} differs from verified official classification result')
    errors = sorted((r for r in rows if not r['correct']),
                    key=lambda r: (-r['predicted_confidence'], r['dialogue_id'], r['utterance_index']))
    high = [dict(row, threshold=t, error_type='official_cmerc_residual_baseline_unavailable')
            for t in THRESHOLDS for row in errors if row['predicted_confidence'] >= t]
    class_rows = []
    for k, emotion in enumerate(metrics['class_names']):
        selected = [r for r in rows if r['true_label'] == k]
        count = sum(not r['correct'] for r in selected)
        class_rows.append(dict(dataset=rows[0]['dataset'], model=rows[0]['model'], true_label=k,
            true_emotion=emotion, support=len(selected), total_errors=count,
            error_rate=count/len(selected) if selected else None, **high_confidence_counts(selected)))
    summary.update(total_errors=len(errors), weighted_f1=weighted, macro_f1=macro,
        mean_error_confidence=math.fsum(r['predicted_confidence'] for r in errors)/len(errors) if errors else None,
        median_error_confidence=median(r['predicted_confidence'] for r in errors) if errors else None,
        **high_confidence_counts(rows))
    return summary, dict(all_predictions=rows, errors_all=errors, high_confidence_errors=high,
        class_error_summary=class_rows, error_confidence_distribution=error_distribution(rows),
        calibration_bins=bins)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=PROJECT/'checkpoints/analysis/official_checkpoint_verification')
    parser.add_argument('--output-dir', type=Path, default=PROJECT/'checkpoints/analysis/high_confidence_errors/official_checkpoint_analysis')
    parser.add_argument('--project-dir', type=Path, default=PROJECT)
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if source == output or output in source.parents:
        parser.error('Output must be independent of the original verification directory')
    original_sha = {str(p): sha256(p) for p in source.rglob('*') if p.is_file()}
    summary_rows, validations = [], []
    for dataset in ('IEMOCAP', 'MELD'):
        rows, metrics, validation = load_dataset(source, dataset, args.project_dir.resolve())
        summary, tables = generate_dataset_tables(rows, metrics)
        for name, table in tables.items():
            fields = list(rows[0]) + ['threshold', 'error_type'] if name == 'high_confidence_errors' else None
            if name == 'errors_all':
                fields = list(rows[0])
            write_csv(output/dataset/f'{name}.csv', table, fields=fields)
        validation['accuracy'] = summary['accuracy']
        validation['weighted_f1'] = summary['weighted_f1']
        validation['nll_from_original_log_probabilities'] = summary['nll']
        validation['calibration_matches_existing_definition'] = True
        summary_rows.append(summary)
        validations.append(validation)
    write_csv(output/'official_cmerc_summary.csv', summary_rows)
    if any(sha256(path) != expected for path, expected in original_sha.items()):
        raise ValueError('A protected original official verification file changed')
    payload = dict(status='passed', completed_at=datetime.now(timezone.utc).isoformat(),
        mode='CPU only; reused official full-input predictions, zero model forwards',
        probability_transform='None; use unchanged saved probabilities and original log_softmax',
        metric_units='All rates, confidence, gaps, F1, ECE and MCE are fractions [0,1]',
        brier_definition='Mean over utterances of sum across classes of (p_k - one_hot_k)^2',
        nll_definition='Negative mean original true-label log_probability, natural logarithm; no clipping',
        calibration_bins='10 equal-width left-closed/right-open bins; p=1 belongs to last bin; empty observed statistics blank',
        high_confidence_denominator='All predictions with max probability >= threshold; undefined when zero',
        high_confidence_detail='A sample is repeated once at every qualifying threshold',
        source_files_unchanged=len(original_sha), source_files_sha256=original_sha, datasets=validations,
        output_files_sha256={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*.csv'))
                             if p.name in {'official_cmerc_summary.csv', 'all_predictions.csv', 'errors_all.csv',
                                           'high_confidence_errors.csv', 'class_error_summary.csv',
                                           'error_confidence_distribution.csv', 'calibration_bins.csv'}})
    (output/'full_input_verification.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
