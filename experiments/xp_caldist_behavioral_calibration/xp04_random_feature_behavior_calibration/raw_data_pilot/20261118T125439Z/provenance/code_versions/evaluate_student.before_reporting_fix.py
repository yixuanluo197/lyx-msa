#!/usr/bin/env python3
"""One final, locked Validation evaluation. Never trains or reads label slots."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path('/LOCAL/lyx-msa')
BASE = ROOT / 'experiments/xp_caldist_behavioral_calibration/xp04_random_feature_behavior_calibration/raw_data_pilot'
FROZEN_PROTOCOL = BASE / '20261117T053929Z/configs/TEXT_RECOVERY_PROTOCOL.json'
FROZEN_PROTOCOL_SHA256 = 'e74ff548b9b43d253546d68a6ac9d701c8733097b3519bd6d62c6cbfe6e777bf'
XP = ROOT / 'CMERC/experiments/xp_mainline_validation/20261016T050823Z'
SPECS = {
    'IEMOCAP': dict(seed=1475, rows=527, dialogues=12, train_rows=5283,
                    filename='iemocap_features_roberta.pkl', slots=[2, 3, 4, 5], text_slot=6,
                    sha256='4bd60c581a5b7c70e217e18439f741b191605fbd6f2cf4dc1a6dcb7fcaa18f06'),
    'MELD': dict(seed=67137, rows=1109, dialogues=114, train_rows=9989,
                 filename='meld_features_roberta.pkl', slots=[3, 4, 5, 6], text_slot=7,
                 sha256='58b2bdadc2eaec6abb40ac0cb3ffa3f31e929d636ece8197aecf1ca373a92951'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_lock(run_dir, dataset):
    """Only inspect checkpoint after a completed training run explicitly locks it."""
    run_dir = Path(run_dir).resolve()
    lock_path = run_dir / 'best_model_lock.json'
    if not lock_path.is_file():
        raise RuntimeError('Missing explicit best_model_lock.json; Validation access forbidden')
    lock = json.loads(lock_path.read_text())
    if lock.get('dataset') != dataset or lock.get('training_complete') is not True:
        raise RuntimeError('Lock must identify this dataset and training_complete=true')
    summary_path = run_dir / 'training_summary.json'
    if not summary_path.is_file() or json.loads(summary_path.read_text()).get('training_complete') is not True:
        raise RuntimeError('Training summary must confirm completed formal training before Validation')
    raw_files = lock.get('files', lock.get('locked_files'))
    if not isinstance(raw_files, dict) or not raw_files:
        raise RuntimeError('Lock requires nonempty files {path: sha256} mapping')
    checked = {}
    for name, expected in raw_files.items():
        path = Path(name)
        if not path.is_absolute():
            path = run_dir / path
        expected = expected.get('sha256') if isinstance(expected, dict) else expected
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError('Locked file changed: ' + str(path))
        checked[str(path.resolve())] = actual
    model_dir = Path(lock.get('best_model_dir', lock.get('model_dir', 'best_student')))
    if not model_dir.is_absolute():
        model_dir = run_dir / model_dir
    # Every model file must be covered, not only an arbitrary sentinel.
    model_files = [path for path in model_dir.rglob('*') if path.is_file()]
    if not model_files or not any(path.name in ('pytorch_model.bin', 'model.safetensors',
                                                'pytorch_model.bin.index.json',
                                                'model.safetensors.index.json') for path in model_files):
        raise RuntimeError('No saved Hugging Face student weights in locked model directory')
    for path in model_files:
        if str(path.resolve()) not in checked:
            raise RuntimeError('Unhashed student file: ' + str(path))
    return lock, model_dir, checked, sha256(lock_path)


def load_validation(dataset):
    """Call only after verify_lock and one-time marker acquisition. No label/Test indexing."""
    spec = SPECS[dataset]
    cfg_path = XP / ('configs/effective/%s_cmerc_validation_seed%d.json' % (dataset, spec['seed']))
    cfg = json.loads(cfg_path.read_text())
    ids = cfg['provenance']['split']['validation_ids']
    train_ids = cfg['provenance']['split']['train_ids']
    if len(ids) != spec['dialogues'] or len(set(ids)) != len(ids) or set(ids) & set(train_ids):
        raise RuntimeError('Invalid effective Train/Validation dialogue partition')
    pkl_path = XP / 'code/data' / spec['filename']
    if sha256(pkl_path) != spec['sha256']:
        raise RuntimeError('Official feature pickle changed')
    # Container includes all splits. Only the four selected feature dictionaries and
    # text dictionary are indexed, exclusively by effective Validation dialogue IDs.
    with pkl_path.open('rb') as stream:
        package = pickle.load(stream, encoding='latin1')
    texts, keys, targets, rows = [], [], [], []
    for dialogue in ids:
        dialogue_text = package[spec['text_slot']][dialogue]
        layers = [np.asarray(package[slot][dialogue]) for slot in spec['slots']]
        n = len(dialogue_text)
        if any(layer.shape != (n, 1024) for layer in layers):
            raise RuntimeError('Feature/text alignment invalid for dialogue %s' % dialogue)
        targets.append(np.stack(layers, axis=1))
        for index, text in enumerate(dialogue_text):
            if not isinstance(text, str):
                raise RuntimeError('Original text must be a string')
            processed = text.encode('ascii', errors='ignore').decode('utf-8') if dataset == 'MELD' else text
            keys.append((str(dialogue), index))
            texts.append(processed)
            rows.append(dict(row_index=len(rows), dataset=dataset, split='validation',
                             dialogue_id=dialogue, utterance_index=index,
                             original_text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                             processed_text_sha256=hashlib.sha256(processed.encode('utf-8')).hexdigest()))
    target = np.concatenate(targets).astype(np.float32)
    if target.shape != (spec['rows'], 4, 1024):
        raise RuntimeError('Unexpected full Validation shape: ' + str(target.shape))
    return texts, keys, target, rows, {'config': str(cfg_path), 'config_sha256': sha256(cfg_path),
                                     'pickle': str(pkl_path.resolve()), 'pickle_sha256': spec['sha256']}


def layer_metrics(target, student, layer, gate):
    """Paired metrics in original feature space, with undefined cases kept explicit."""
    target = np.asarray(target, dtype=np.float64)
    student = np.asarray(student, dtype=np.float64)
    finite = bool(np.isfinite(target).all() and np.isfinite(student).all())
    shape_ok = target.shape == student.shape and target.ndim == 2 and target.shape[1] == 1024
    if not finite or not shape_ok:
        return dict(layer=layer, finite=finite, shape_ok=shape_ok, passed=False,
                    failure='nonfinite values or shape mismatch'), []
    delta = student - target
    abs_delta = np.abs(delta)
    tnorm, snorm = np.linalg.norm(target, axis=1), np.linalg.norm(student, axis=1)
    nonzero = (tnorm > 0) & (snorm > 0)
    cosines = np.sum(target[nonzero] * student[nonzero], axis=1) / (tnorm[nonzero] * snorm[nonzero])
    cosines = np.clip(cosines, -1., 1.)
    tmean, smean = target.mean(0), student.mean(0)
    tstd, sstd = target.std(0), student.std(0)
    floor = gate['standard_deviation_floor_for_constant_detection']
    target_nonconstant = tstd > floor
    defined = target_nonconstant & (sstd > floor)
    corr = np.full(1024, np.nan)
    corr[defined] = np.mean((target - tmean)[:, defined] * (student - smean)[:, defined], axis=0) / (tstd[defined] * sstd[defined])
    corr[defined] = np.clip(corr[defined], -1., 1.)
    all_corr_defined = bool(np.isfinite(corr[target_nonconstant]).all())
    valid_corr = corr[target_nonconstant & np.isfinite(corr)]
    corr_p05 = float(np.quantile(valid_corr, .05)) if valid_corr.size else None
    # Equal sized empirical distributions: W1 is the mean sorted absolute gap.
    w1 = np.mean(np.abs(np.sort(target, axis=0) - np.sort(student, axis=0)), axis=0)
    atol, rtol = gate['all_elements_allclose']['atol'], gate['all_elements_allclose']['rtol']
    mae, rmse = float(abs_delta.mean()), float(np.sqrt(np.mean(delta ** 2)))
    mean_cos = float(cosines.mean()) if cosines.size else None
    min_cos = float(cosines.min()) if cosines.size else None
    checks = dict(shape=shape_ok, all_finite=finite,
                  allclose=bool(np.allclose(student, target, atol=atol, rtol=rtol)),
                  MAE=mae <= gate['MAE_max'], RMSE=rmse <= gate['RMSE_max'],
                  mean_cosine=mean_cos is not None and mean_cos >= gate['mean_nonzero_row_cosine_min'],
                  min_cosine=min_cos is not None and min_cos >= gate['min_nonzero_row_cosine_min'],
                  all_target_nonconstant_correlations_defined=all_corr_defined,
                  correlation_p05=corr_p05 is not None and corr_p05 >= gate['target_nonconstant_dimension_correlation_p05_min'])
    summary = dict(layer=layer, hidden_offset=-layer, shape=list(target.shape), finite=finite,
                   max_abs_error=float(abs_delta.max()), MAE=mae, RMSE=rmse,
                   mean_nonzero_row_cosine=mean_cos, min_nonzero_row_cosine=min_cos,
                   zero_target_rows=int(np.sum(tnorm == 0)), zero_student_rows=int(np.sum(snorm == 0)),
                   both_zero_rows=int(np.sum((tnorm == 0) & (snorm == 0))),
                   one_zero_rows=int(np.sum((tnorm == 0) != (snorm == 0))),
                   target_constant_coordinates=int(np.sum(~target_nonconstant)),
                   undefined_target_nonconstant_correlations=int(np.sum(target_nonconstant & ~np.isfinite(corr))),
                   correlation_mean=float(valid_corr.mean()) if valid_corr.size else None,
                   correlation_p05=corr_p05, mean_abs_mean_difference=float(np.abs(smean - tmean).mean()),
                   mean_abs_std_difference=float(np.abs(sstd - tstd).mean()),
                   mean_coordinate_wasserstein1=float(w1.mean()),
                   failed_element_allclose_count=int(np.sum(abs_delta > atol + rtol * np.abs(target))),
                   checks=checks, passed=bool(all(checks.values())))
    coordinates = []
    for column in range(1024):
        coordinates.append(dict(layer=layer, hidden_offset=-layer, coordinate=column,
                                target_mean=float(tmean[column]), student_mean=float(smean[column]),
                                target_std=float(tstd[column]), student_std=float(sstd[column]),
                                mean_difference=float(smean[column] - tmean[column]),
                                std_difference=float(sstd[column] - tstd[column]),
                                MAE=float(abs_delta[:, column].mean()), RMSE=float(np.sqrt(np.mean(delta[:, column] ** 2))),
                                max_abs_error=float(abs_delta[:, column].max()),
                                target_nonconstant=bool(target_nonconstant[column]),
                                correlation=float(corr[column]) if np.isfinite(corr[column]) else None,
                                wasserstein1=float(w1[column])))
    return summary, coordinates


def extract_features(model_dir, tokenizer_dir, texts, batch_size):
    import torch
    from transformers import RobertaModel, RobertaTokenizer
    # FP32 final extraction, no autocast, no training dropout, no new head.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = RobertaTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    model = RobertaModel.from_pretrained(str(model_dir), add_pooling_layer=False, local_files_only=True)
    model = model.to(device='cuda:0', dtype=torch.float32).eval()
    outputs, token_rows = [], []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            untruncated = [len(tokenizer.encode(text, add_special_tokens=True, truncation=False)) for text in chunk]
            encoded = tokenizer(chunk, padding=True, truncation=True, max_length=512, return_tensors='pt')
            if not torch.all(encoded['input_ids'][:, 0] == 0):
                raise RuntimeError('Expected BOS token id 0 at position 0')
            encoded = {name: value.to('cuda:0') for name, value in encoded.items()}
            hidden = model(**encoded, output_hidden_states=True, return_dict=True).hidden_states
            values = torch.stack([hidden[-i][:, 0, :] for i in range(1, 5)], dim=1)
            outputs.append(values.cpu().numpy())
            for raw_length in untruncated:
                token_rows.append(dict(row_index=len(token_rows), tokens_before_truncation=raw_length,
                                       tokens_after_truncation=min(raw_length, 512), truncated=raw_length > 512))
            print(json.dumps({'event': 'validation_extraction', 'completed': min(start + batch_size, len(texts)),
                              'total': len(texts)}), flush=True)
    return np.concatenate(outputs).astype(np.float32), token_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=list(SPECS), required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--resume-saved-features', action='store_true',
                        help='Recover metrics from already saved final features; never re-extract')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.batch_size < 1:
        raise ValueError('Positive batch size required')
    out = args.output_root.resolve()
    local = out / 'local_only' / args.dataset / 'validation'
    results = out / 'results' / args.dataset
    result_path = results / 'feature_parity.json'
    marker = local / 'FINAL_VALIDATION_STARTED.json'
    features_path = local / 'student_features.npz'
    if result_path.exists():
        raise RuntimeError('Final Validation result already exists; repeated evaluation forbidden')
    if sha256(FROZEN_PROTOCOL) != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError('Frozen feature/probability gate changed')
    protocol = json.loads(FROZEN_PROTOCOL.read_text())
    lock, model_dir, checked_files, lock_hash = verify_lock(args.run_dir, args.dataset)
    tokenizer_dir = args.tokenizer or (out / 'local_only/tokenizer')
    tokenizer_files = [p for p in tokenizer_dir.rglob('*') if p.is_file()]
    if not tokenizer_files or any(str(p.resolve()) not in checked_files for p in tokenizer_files):
        raise RuntimeError('Tokenizer assets must exist and all be covered by best model lock')
    if str(Path(__file__).resolve()) not in checked_files:
        raise RuntimeError('Final evaluation code must be frozen in model lock')
    local.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        previous = json.loads(marker.read_text())
        if not args.resume_saved_features or not features_path.exists() or previous['best_model_lock_sha256'] != lock_hash:
            raise RuntimeError('Validation started previously; only saved-feature crash recovery with identical lock allowed')
    else:
        if args.resume_saved_features:
            raise RuntimeError('No previous final extraction to recover')
        with marker.open('x') as stream:
            json.dump(dict(started_at=now(), dataset=args.dataset, best_model_lock_sha256=lock_hash,
                           protocol_sha256=FROZEN_PROTOCOL_SHA256, evaluation_code_sha256=sha256(__file__)), stream, indent=2)
    texts, keys, target, manifest, inputs = load_validation(args.dataset)
    write_csv(local / 'validation_manifest.csv', manifest)
    if args.resume_saved_features:
        saved = np.load(features_path, allow_pickle=False)
        student = saved['features']
        saved_keys = list(zip(saved['dialogue_ids'].tolist(), saved['utterance_indices'].tolist()))
        if saved_keys != keys:
            raise RuntimeError('Saved feature dialogue/utterance order mismatch')
    else:
        student, token_rows = extract_features(model_dir, tokenizer_dir, texts, args.batch_size)
        temporary = local / 'student_features.tmp.npz'
        np.savez(temporary, features=student, dialogue_ids=np.array([k[0] for k in keys]),
                 utterance_indices=np.array([k[1] for k in keys]))
        temporary.replace(features_path)
        write_csv(local / 'token_lengths.csv', token_rows)
    if student.shape != target.shape:
        raise RuntimeError('Student full four-layer feature shape mismatch')
    summaries, coordinates = [], []
    for offset in range(4):
        summary, coord = layer_metrics(target[:, offset], student[:, offset], offset + 1,
                                       protocol['feature_gate_all_four_layers'])
        summaries.append(summary)
        coordinates.extend(coord)
    passed = all(row['passed'] for row in summaries)
    result = dict(dataset=args.dataset, completed_at=now(), status='PASS' if passed else 'FAIL',
                  feature_gate_passed=passed, representation='distilled RoBERTa student; not original CMERC weights',
                  rows=len(keys), dialogues=len(set(k[0] for k in keys)), shape=list(student.shape),
                  best_model_lock_sha256=lock_hash, protocol_sha256=FROZEN_PROTOCOL_SHA256,
                  inputs=inputs, student_features_sha256=sha256(features_path), layers=summaries,
                  probability_status='READY_AFTER_FEATURE_GATE' if passed else 'NOT_RUN_FEATURE_GATE_FAILED',
                  probability_metrics=None,
                  labels_accessed=False, test_dialogues_indexed=False, validation_feedback_used_for_selection=False)
    write_csv(results / 'per_coordinate_metrics.csv', coordinates)
    flat = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in summaries]
    write_csv(results / 'per_layer_metrics.csv', flat)
    write_json(result_path, result)
    write_json(local / 'FINAL_VALIDATION_COMPLETED.json', dict(completed_at=now(), result_sha256=sha256(result_path),
                                                             best_model_lock_sha256=lock_hash, feature_gate_passed=passed))
    print(json.dumps({'dataset': args.dataset, 'feature_gate_passed': passed,
                      'result': str(result_path)}), flush=True)


if __name__ == '__main__':
    main()
