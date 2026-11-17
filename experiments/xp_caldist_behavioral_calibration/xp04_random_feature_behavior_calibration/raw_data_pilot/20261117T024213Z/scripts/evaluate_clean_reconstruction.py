"""One final, lock-protected Validation feature evaluation; never fit or run CMERC.

Without --authorize-final-validation this only validates the lock and local inputs.
Full feature arrays and per-dimension values stay under local_only. A completed
evaluation cannot be repeated. An interrupted IO-only attempt may be resumed with
the exact same lock/input hashes; the original failure record is retained.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import traceback

import numpy as np

from converter_core import compute_metrics, predict_converter, stateless_transform

WORKSPACE = Path('/LOCAL/lyx-msa')
DEFAULT_OUT = Path(__file__).resolve().parents[1]
XP = WORKSPACE / 'CMERC/experiments/xp_mainline_validation/20261016T050823Z'
ASSETS = Path('/LOCAL/MELD_ASSETS/raw/MELD/MELD.Features.Models')
SPECS = {'IEMOCAP': (1475, 527, 1582, 1582, 'IEMOCAP_features.pkl'),
         'MELD': (67137, 1109, 6373, 1611, 'MELD_features_raw1.pkl')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w') as stream:
        json.dump(clean_json(value), stream, indent=2, allow_nan=False)
        stream.write('\n')
    path.chmod(0o600)


def load_pickle(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream, encoding='latin1')


def preflight(out, dataset):
    """No feature target pickle is opened here, including failed guards."""
    seed, nrows, inwidth, outwidth, base_name = SPECS[dataset]
    lockpath = out / 'provenance' / f'{dataset}_model_lock.json'
    lock = read_json(lockpath)
    require(lock.get('dataset') == dataset, 'Lock dataset differs')
    require(lock.get('validation_unopened') is True, 'Lock must precede new Validation access')
    required = ('converter_path', 'validation_features_path', 'validation_valid_path',
                'manifest_path', 'protocol_path', 'train_scale_path')
    paths = {key: Path(lock[key]).resolve() for key in required}
    for key, path in paths.items():
        require(path.is_relative_to(out.resolve()), f'{key} must be inside this run')
        require(path.is_file(), f'Missing locked input {key}: {path}')
    require(sha(paths['converter_path']) == lock['converter_sha256'], 'Converter changed after lock')
    require(paths['validation_features_path'].parent.name == lock['candidate'], 'Candidate directory differs from lock')
    require(sha(paths['protocol_path']) == lock['protocol_sha256'], 'Protocol changed after lock')
    frozen = read_json(out / 'provenance/protocol_frozen.json')
    require(frozen['protocol_sha256'] == lock['protocol_sha256'], 'Original protocol freeze mismatch')
    protocol = read_json(paths['protocol_path'])
    require(lock['transform'] in protocol['stateless_input_transforms'], 'Unknown locked input transform')
    require(protocol['datasets'][dataset]['expected_validation'] == nrows, 'Protocol split size differs')
    with paths['manifest_path'].open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    manifest_audit = read_json(out / 'provenance/data_manifest_audit.json')
    frozen_manifest = manifest_audit['artifact_sha256'][str(paths['manifest_path'].relative_to(out))]
    require(sha(paths['manifest_path']) == frozen_manifest, 'Frozen manifest changed')
    if 'manifest_sha256' in lock:
        require(sha(paths['manifest_path']) == lock['manifest_sha256'], 'Locked manifest changed')
    require(len(rows) == nrows, 'Incomplete fixed Validation manifest')
    seen = set()
    for i, row in enumerate(rows):
        require(row['dataset'] == dataset and row['split'] == 'validation', 'Only Validation rows authorized')
        require(int(row['row_index']) == i, 'Manifest is not contiguous')
        key = (row['dialogue_id'], int(row['utterance_index']))
        require(key not in seen, 'Duplicate manifest identity')
        seen.add(key)
    features = np.load(paths['validation_features_path'], allow_pickle=False, mmap_mode='r')
    valid = np.load(paths['validation_valid_path'], allow_pickle=False)
    require(features.shape == (nrows, inwidth), 'Candidate feature shape differs')
    require(valid.shape == (nrows,) and valid.dtype == np.dtype(bool), 'Valid mask must be boolean row vector')
    require(np.array_equal(valid, np.isfinite(features).all(axis=1)), 'Valid mask and finite features differ')
    scale = np.load(paths['train_scale_path'], allow_pickle=False)
    require(scale.shape == (outwidth,) and np.isfinite(scale).all() and (scale >= 0).all(), 'Invalid Train target scale')
    if 'train_scale_sha256' in lock:
        require(sha(paths['train_scale_path']) == lock['train_scale_sha256'], 'Train scale changed after lock')
    converter = load_pickle(paths['converter_path'])
    require(converter.get('reconstructed_method') is True, 'Converter must explicitly be reconstructed')
    require(converter['n_input_features'] == inwidth and converter['n_output_features'] == outwidth,
            'Locked converter dimensions differ')
    require(0 < converter['n_fit_rows'] <= protocol['datasets'][dataset]['expected_train'],
            'Converter fit count exceeds effective Train boundary')
    cfg_path = XP / f'configs/effective/{dataset}_cmerc_validation_seed{seed}.json'
    base_path = XP / 'code/data' / base_name
    expected_sources = read_json(out / 'provenance/input_hashes_before.json')
    source_paths = [cfg_path, base_path]
    if dataset == 'MELD':
        source_paths += [ASSETS / 'features/audio_embeddings_feature_selection_emotion.pkl',
                         ASSETS / 'features/audio_emotion.pkl', ASSETS / 'models/audio_weights_emotion.hdf5']
        path = Path(lock['train_scale300_path']).resolve()
        require(path.is_relative_to(out.resolve()) and path.is_file(), 'Missing locked Train 300D scale')
        require(sha(path) == lock['train_scale300_sha256'], 'Train 300D scale changed')
        paths['train_scale300_path'] = path
        scale300 = np.load(path, allow_pickle=False)
        require(scale300.shape == (300,) and np.isfinite(scale300).all() and (scale300 >= 0).all(), 'Invalid Train 300D scale')
    for path in source_paths:
        digest = expected_sources.get(str(path), expected_sources.get(str(path.resolve())))
        require(digest is not None and sha(path) == digest, f'Frozen source hash changed: {path.name}')
    config = read_json(cfg_path)
    require(config['seed'] == seed and config['Dataset'] == dataset, 'Effective config mismatch')
    ids = config['provenance']['split']['validation_ids']
    manifest_ids = list(dict.fromkeys(row['dialogue_id'] for row in rows))
    require(manifest_ids == list(map(str, ids)), 'Manifest effective Dialogue ordering differs')
    paths.update(config=cfg_path, base=base_path)
    binding = {'lock_sha256': sha(lockpath), 'inputs': {k: sha(v) for k, v in paths.items()},
               'protocol_sha256': lock['protocol_sha256'], 'converter_sha256': lock['converter_sha256'],
               'converter_core_sha256': sha(Path(__file__).with_name('converter_core.py'))}
    return lock, protocol, rows, paths, binding


def evaluation_marker(out, dataset, binding, resume):
    path = out / 'provenance' / f'{dataset}_validation_evaluation_marker.json'
    previous = None
    if path.exists():
        previous = read_json(path)
        require(resume, 'Validation already opened: explicit same-lock IO resume required')
        require(previous['status'] == 'FAILED_IO', 'Completed or non-IO failed Validation cannot be reopened')
        require(previous['binding'] == binding, 'Validation resume requires identical lock and input hashes')
    else:
        require(not resume, 'No failed evaluation exists to resume')
    record = {'status': 'OPENING_VALIDATION', 'started_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'binding': binding, 'attempt': previous['attempt'] + 1 if previous else 1,
              'runner_sha256': sha(__file__), 'core_sha256': sha(Path(__file__).with_name('converter_core.py')),
              'previous_failed_attempt': previous}
    write_json(path, record, exclusive=previous is None)
    return path, record


def load_validation_targets(dataset, rows, paths):
    """The only target accesses: frozen slot4 Val IDs; MELD selected slot1 keys."""
    base = load_pickle(paths['base'])
    values = []
    config = read_json(paths['config'])
    ids = config['provenance']['split']['validation_ids']
    keys = {str(key): key for key in ids}
    for row in rows:
        key, index = keys[row['dialogue_id']], int(row['utterance_index'])
        require(str(base[0][key][index]) == str(row['raw_utterance_id']), 'XP04 utterance identity changed')
        # No labels, Train features or Test records are selected.
        values.append(np.asarray(base[4][key][index], dtype=np.float64))
    audio = np.stack(values)
    if dataset == 'IEMOCAP':
        require(audio.shape == (527, 1582), 'IEMOCAP target shape differs')
        return audio, None
    selected = load_pickle(ASSETS / 'features/audio_embeddings_feature_selection_emotion.pkl')[1]
    expected_keys = {f"{row['raw_dialogue_id']}_{row['raw_utterance_id']}" for row in rows}
    require(set(selected) == expected_keys, 'Official selected Validation IDs differ')
    inputs = np.stack([np.asarray(selected[f"{r['raw_dialogue_id']}_{r['raw_utterance_id']}"], dtype=np.float64) for r in rows])
    require(inputs.shape == (1109, 1611) and audio.shape == (1109, 300), 'MELD target shape differs')
    return inputs, audio


def metric_gate(target, prediction, scale, groups, protocol):
    spec = protocol['feature_gate']
    scale = np.maximum(scale, spec['scale_floor'])
    metrics, columns = compute_metrics(target, prediction, groups=groups, target_scale=scale,
                                       return_per_dimension=True, wasserstein=True)
    finite = bool(np.isfinite(target).all() and np.isfinite(prediction).all())
    # Nonconstant is defined by the target, never jointly by target/prediction.
    # Otherwise a collapsed prediction column vanishes from the p05 summary.
    target_rows = np.isfinite(target).all(axis=1)
    prediction_rows = np.isfinite(prediction).all(axis=1)
    target_std = np.std(target[target_rows], axis=0) if target_rows.any() else np.full(target.shape[1], np.nan)
    prediction_std = np.std(prediction[prediction_rows], axis=0) if prediction_rows.any() else np.full(target.shape[1], np.nan)
    target_nonconstant = np.isfinite(target_std) & (target_std > spec['scale_floor'])
    prediction_constant = np.isfinite(prediction_std) & (prediction_std <= spec['scale_floor'])
    undefined_required = target_nonconstant & (~np.isfinite(columns['column_correlation']) | prediction_constant)
    required_corr = columns['column_correlation'][target_nonconstant & ~undefined_required]
    corr = float(np.quantile(required_corr, .05)) if required_corr.size else None
    columns['column_target_nonconstant_for_gate'] = target_nonconstant
    columns['column_prediction_constant_for_gate'] = prediction_constant
    columns['column_required_correlation_undefined'] = undefined_required
    checks = {'all_finite': finite,
              'full_coverage': metrics['fully_finite_rows'] == target.shape[0],
              'all_elements_allclose': bool(finite and np.allclose(prediction, target, **spec['all_elements_allclose'])),
              'mae': metrics['mean_absolute_error'] is not None and metrics['mean_absolute_error'] <= spec['MAE_max'],
              'train_scaled_rmse': metrics['global_train_scale_nrmse'] is not None and metrics['global_train_scale_nrmse'] <= spec['train_scaled_RMSE_max'],
              'mean_cosine': metrics['mean_cosine_similarity'] is not None and metrics['mean_cosine_similarity'] >= spec['mean_nonzero_row_cosine_min'],
              'min_cosine': metrics['min_cosine_similarity'] is not None and metrics['min_cosine_similarity'] >= spec['min_nonzero_row_cosine_min'],
              'required_target_correlations_defined': not bool(undefined_required.any()),
              'correlation_p05': (not bool(target_nonconstant.any())) or (
                  corr is not None and corr >= spec['nonconstant_dimension_correlation_p05_min'])}
    metrics['correlation_gate_definition'] = ('Target-nonconstant columns have population standard deviation > '
        'protocol feature_gate.scale_floor on all wholly finite target Validation rows; prediction population '
        'standard deviation <= the same floor is constant. Every target-nonconstant correlation must be '
        'defined; p05 is computed on these required columns only. No target-nonconstant columns means '
        'the correlation condition is not applicable, while all element-error gates still apply.')
    metrics['target_nonconstant_dimensions_for_correlation_gate'] = int(target_nonconstant.sum())
    metrics['target_nonconstant_but_prediction_constant_dimensions'] = int((target_nonconstant & prediction_constant).sum())
    metrics['target_nonconstant_but_correlation_undefined_dimensions'] = int(undefined_required.sum())
    metrics['required_target_dimension_correlation_p05'] = corr
    metrics['target_constant_dimensions'] = int(np.all(target == target[:1], axis=0).sum())
    metrics['prediction_constant_dimensions'] = int(np.all(prediction == prediction[:1], axis=0).sum())
    metrics['gate_checks'] = checks
    metrics['passed'] = bool(all(checks.values()))
    return metrics, columns


def write_columns(path, columns):
    width = len(columns['column_mae'])
    names = [key for key in columns if key.startswith('column_')]
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['feature_index'] + names)
        writer.writeheader()
        for i in range(width):
            writer.writerow(clean_json({'feature_index': i, **{name: columns[name][i] for name in names}}))
    Path(path).chmod(0o600)


def meld_context(rebuilt, rows):
    """Exact original HDF5; full Dialogue, right-zero padding, no target imputation."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import h5py
    import keras
    import tensorflow as tf
    require(not tf.config.list_physical_devices('GPU'), 'MELD feature evaluation must use CPU')
    model = keras.models.load_model(ASSETS / 'models/audio_weights_emotion.hdf5', compile=False)
    encoder = keras.Model(model.input, model.get_layer('utter').output)
    require(model.input_shape == (None, 33, 1611) and encoder.output_shape == (None, 33, 300), 'Original GRU shape differs')
    layer = model.get_layer('utter')
    require(layer.forward_layer.units == 150 and layer.backward_layer.units == 150, 'Original GRU units differ')
    for direction in (layer.forward_layer, layer.backward_layer):
        require(direction.reset_after is False and direction.implementation == 1
                and direction.recurrent_activation.__name__ == 'hard_sigmoid', 'Legacy GRU semantics differ')
    require(any(isinstance(x, keras.layers.Masking) and x.mask_value == 0 for x in model.layers),
            'Original Masking(0) layer missing')
    probe = np.array([-4., -2.5, -1., 0., 1., 2.5, 4.], np.float32)
    require(np.array_equal(keras.activations.hard_sigmoid(tf.constant(probe)).numpy(), np.clip(.2 * probe + .5, 0, 1)), 'Legacy hard sigmoid differs')
    with h5py.File(ASSETS / 'models/audio_weights_emotion.hdf5', 'r') as handle:
        exact = {}
        for direction, gru in [('forward', layer.forward_layer), ('backward', layer.backward_layer)]:
            for name, weight in zip(['kernel', 'recurrent_kernel', 'bias'], gru.get_weights()):
                key = f'{direction}/{name}'
                exact[key] = bool(np.array_equal(weight, handle[f'model_weights/utter/utter/{direction}_gru_1/{name}:0'][:]))
        require(all(exact.values()), 'Loaded original GRU weights differ')
    official_padding = load_pickle(ASSETS / 'features/audio_emotion.pkl')[1]
    result = np.full((len(rows), 300), np.nan, dtype=np.float32)
    dialogue_records, padding_pred, padding_target = [], [], []
    for dialogue in dict.fromkeys(r['dialogue_id'] for r in rows):
        index = [i for i, row in enumerate(rows) if row['dialogue_id'] == dialogue]
        raw_order = [int(rows[i]['raw_utterance_id']) for i in index]
        require(raw_order == sorted(raw_order) and len(set(raw_order)) == len(index), 'MELD actual utterance IDs not ordered')
        require([int(rows[i]['utterance_index']) for i in index] == list(range(len(index))), 'MELD real row ranks differ')
        require(0 < len(index) <= 33, 'MELD Dialogue length exceeds original model')
        finite = np.isfinite(rebuilt[index]).all()
        record = {'dialogue_id': dialogue, 'raw_dialogue_id': rows[index[0]]['raw_dialogue_id'],
                  'real_utterances': len(index), 'raw_utterance_ids': raw_order,
                  'complete_raw_input': bool(finite), 'affected_missing_context_rows': 0 if finite else len(index)}
        if finite:
            padded = np.zeros((1, 33, 1611), np.float32)
            padded[0, :len(index)] = rebuilt[index].astype(np.float32)
            output = encoder(padded, training=False).numpy()[0]
            result[index] = output[:len(index)]
            ref = np.asarray(official_padding[str(rows[index[0]]['raw_dialogue_id'])], dtype=np.float32)
            require(ref.shape == (33, 300), 'Original padding reference shape differs')
            record['padding_current_allzero'] = bool(np.all(output[len(index):] == 0))
            record['padding_reference_allzero'] = bool(np.all(ref[len(index):] == 0))
            record['padding_rows'] = 33 - len(index)
            padding_pred.append(output[len(index):])
            padding_target.append(ref[len(index):])
        dialogue_records.append(record)
    pad_metrics = compute_metrics(np.concatenate(padding_target), np.concatenate(padding_pred)) if padding_pred and sum(len(x) for x in padding_pred) else None
    return result, {'tensorflow': tf.__version__, 'keras': keras.__version__, 'gpu_devices': [],
                    'loaded_compile': False, 'original_weights_exact': exact, 'training': False,
                    'padding': 'right-zero to33, Masking(0), original model, training=False',
                    'dialogues': dialogue_records, 'completed_dialogues': sum(d['complete_raw_input'] for d in dialogue_records),
                    'missing_context_dialogues': sum(not d['complete_raw_input'] for d in dialogue_records),
                    'affected_missing_context_rows': sum(d['affected_missing_context_rows'] for d in dialogue_records),
                    'padding_metrics_separate_from_real_rows': pad_metrics}


def execute(out, dataset, preflight_result, resume=False):
    lock, protocol, rows, paths, binding = preflight_result
    marker_path, marker = evaluation_marker(out, dataset, binding, resume)
    try:
        dest = out / 'local_only' / dataset / 'final_validation'
        dest.mkdir(parents=True, exist_ok=True, mode=0o700)
        model = load_pickle(paths['converter_path'])
        require(model.get('reconstructed_method') is True, 'Converter must explicitly be reconstructed')
        features = np.load(paths['validation_features_path'], allow_pickle=False)
        valid = np.load(paths['validation_valid_path'], allow_pickle=False)
        speakers = np.asarray([(row['raw_utterance_id'][:5] + row['raw_utterance_id'].rsplit('_', 1)[1][0])
                               if dataset == 'IEMOCAP' else row['speaker_id'] for row in rows], dtype=str)
        rebuilt = np.full((len(rows), model['n_output_features']), np.nan, np.float64)
        if valid.any():
            inputs = stateless_transform(features[valid], lock['transform'])
            rebuilt[valid] = predict_converter(model, inputs, speakers=speakers[valid])
        target, target300 = load_validation_targets(dataset, rows, paths)
        np.save(dest / 'target_val.npy', target, allow_pickle=False)
        np.save(dest / 'rebuilt_selected_validation.npy', rebuilt, allow_pickle=False)
        groups = np.asarray([r['dialogue_id'] for r in rows])
        metrics, columns = metric_gate(target, rebuilt, np.load(paths['train_scale_path'], allow_pickle=False), groups, protocol)
        write_columns(dest / 'selected_per_dimension.csv', columns)
        result = {'dataset': dataset, 'selected_feature_metrics': metrics, 'source_valid_rows': int(valid.sum()),
                  'source_missing_rows': int((~valid).sum()), 'binding': binding,
                  'classification': 'RECONSTRUCTED_CANDIDATE_NOT_ORIGINAL', 'test_used': False,
                  'label_slots_indexed': False, 'cmERC_executed': False, 'parameters_fitted': False,
                  'validation_feedback_to_selection': False, 'historical_validation_exposure_disclosed': True}
        final_audio, passed = rebuilt, metrics['passed']
        if dataset == 'MELD':
            final_audio, context = meld_context(rebuilt, rows)
            np.save(dest / 'target300_val.npy', target300, allow_pickle=False)
            metrics300, columns300 = metric_gate(target300, final_audio, np.load(paths['train_scale300_path'], allow_pickle=False), groups, protocol)
            write_columns(dest / 'context300_per_dimension.csv', columns300)
            result.update(context300_feature_metrics=metrics300, original_bigru=context)
            passed = passed and metrics300['passed']
        final_path = out / 'local_only' / dataset / 'rebuilt_validation_audio.npy'
        np.save(final_path, final_audio, allow_pickle=False)
        final_path.chmod(0o600)
        gate_item = {'passed': bool(passed), 'clean_feature_parity_passed': bool(passed),
                     'validation_independent': True, 'full_validation_coverage': bool(np.isfinite(final_audio).all() and valid.all()),
                     'fit_effective_train_only': lock.get('fit_effective_train_only', False),
                     'train_dialogue_group_cv_only': lock.get('train_dialogue_group_cv_only', False),
                     'model_locked_before_validation': True, 'test_used': False,
                     'validation_rows': len(rows), 'output_width': final_audio.shape[1],
                     'rebuilt_audio_sha256': sha(final_path), 'manifest_sha256': sha(paths['manifest_path']),
                     'model_lock_sha256': binding['lock_sha256'], 'new_validation_untuned': True,
                     'historical_validation_exposure_disclosed': True}
        if dataset == 'MELD':
            gate_item.update(selected1611_parity_passed=metrics['passed'], context300_parity_passed=metrics300['passed'],
                             full_dialogue_bigru_context=context['missing_context_dialogues'] == 0,
                             real_lengths_and_order_preserved=True, padding33_and_masking_preserved=True)
        write_json(out / 'provenance' / f'{dataset}_feature_gate.json', {'datasets': {dataset: gate_item}})
        result['feature_gate'] = gate_item
        result['outputs'] = {str(p.relative_to(out)): sha(p) for p in dest.iterdir() if p.is_file()}
        result['outputs'][str(final_path.relative_to(out))] = sha(final_path)
        write_json(out / 'provenance' / f'{dataset}_clean_feature_metrics.json', result)
        for path in dest.iterdir():
            if path.is_file():
                path.chmod(0o600)
        marker.update(status='COMPLETE', completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                      feature_gate_passed=bool(passed), outputs=result['outputs'])
        write_json(marker_path, marker)
        return result
    except Exception as error:
        marker.update(status='FAILED_IO' if isinstance(error, OSError) else 'FAILED_NO_REOPEN',
                      failure_type=type(error).__name__, failure_message=str(error),
                      traceback=traceback.format_exc(), failed_at=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(out / 'provenance' / f'{dataset}_validation_failure_attempt{marker["attempt"]}.json', marker, exclusive=True)
        write_json(marker_path, marker)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=list(SPECS), required=True)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--authorize-final-validation', action='store_true')
    parser.add_argument('--resume-same-lock', action='store_true')
    args = parser.parse_args()
    checked = preflight(args.run_dir.resolve(), args.dataset)
    if not args.authorize_final_validation:
        print(json.dumps({'status': 'GUARDS_PASSED_TARGETS_UNOPENED', 'dataset': args.dataset,
                          'binding': checked[-1]}))
        return
    result = execute(args.run_dir.resolve(), args.dataset, checked, resume=args.resume_same_lock)
    print(json.dumps({'dataset': args.dataset, 'feature_gate': result['feature_gate']}))


if __name__ == '__main__':
    main()
