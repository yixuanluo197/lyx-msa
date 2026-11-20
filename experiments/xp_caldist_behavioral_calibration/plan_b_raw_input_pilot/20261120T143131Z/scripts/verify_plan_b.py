#!/usr/bin/env python3
"""Read-only staged Plan B verification; never import or execute model runners.

Only verification.json is written. New local tensors may be inspected on CPU;
neither historical pickle data nor model parameter tensors are deserialized.
Absent later-stage evidence is pending. A partial-stage pass is not completion.
"""
from __future__ import annotations
import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

OUT = Path(__file__).resolve().parents[1]
STAGES = {'data': 0, 'features': 1, 'models': 2, 'pilot': 3, 'final': 4}
VARIANTS = {'neutral_append': {'neutral_a', 'neutral_b'},
            'semantic_paraphrase': {'paraphrase_a', 'paraphrase_b'},
            'pitch_shift': {'pitch_plus_0.5', 'pitch_minus_0.5', 'pitch_plus_1', 'pitch_minus_1'},
            'timbre_tilt': {'timbre_light', 'timbre_moderate'}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def yes(value):
    return value is True or str(value).lower() in ('true', '1')


def load_local(path):
    """Only caller-selected new data/feature/difficulty/prediction tensors."""
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def array_sha(value):
    """Independent V2 interchange hash: dtype, shape, contiguous CPU bytes."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def historical_smoke_source(root, source, digest):
    """The two declared IO-only fixes may preserve a completed old smoke cache."""
    source = Path(source)
    allowed = {'raw_pilot.py': 'read_jsonl', 'raw_path_smoke.py': 'verify_baseline_complete'}
    if source.name not in allowed or source.resolve() != (root / 'scripts' / source.name).resolve():
        return None
    archive = root / 'provenance/jsonl_lf_io_fix_v2'
    registry_path = archive / 'ARCHIVE_MANIFEST.json'
    archived = archive / source.name
    if not registry_path.is_file() or not archived.is_file():
        return None
    registry = read(registry_path)
    recorded = registry['sha256'] if source.name == 'raw_pilot.py' else registry['raw_path_smoke.py']['sha256']
    if digest != recorded or sha(archived) != digest:
        return None
    def other_code(path):
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     or node.name != allowed[source.name]]
        return ast.dump(tree, include_attributes=False)
    if other_code(source) != other_code(archived):
        return None
    return archived


def prob(value):
    result = np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all() or (result < 0).any() or not np.isclose(result.sum(), 1, atol=1e-5):
        raise ValueError('Invalid stored probability vector')
    return result


def independent_behavior(original, perturbed):
    p, q = prob(original), np.stack([prob(v) for v in perturbed])
    maximum, changed = p.max(), q.max(axis=1)
    flip = q.argmax(axis=1) != p.argmax()
    divergences = []
    for row in q:
        middle = (p + row) / 2
        left, right = p > 0, row > 0
        divergences.append(.5 * (np.sum(p[left] * np.log(p[left] / middle[left])) +
                                  np.sum(row[right] * np.log(row[right] / middle[right]))))
    return {'mu': float(np.mean(flip * changed)), 'delta': float(abs(maximum - changed.mean())),
            'flip_rate': float(flip.mean()), 'original_class_retention': float(1 - flip.mean()),
            'js_mean': float(np.mean(divergences)), 'js_max': float(np.max(divergences)),
            'mean_confidence_change': float(np.mean(changed - maximum)),
            'mean_abs_confidence_change': float(np.mean(abs(changed - maximum))),
            'max_abs_confidence_change': float(np.max(abs(changed - maximum))),
            'mean_original_class_probability_change': float(np.mean(q[:, p.argmax()] - maximum))}


def optional_close(value, expected):
    if expected is None:
        return value in (None, '', 'NA', 'null')
    return value not in (None, '', 'NA', 'null') and np.isclose(float(value), expected, atol=1e-10, rtol=1e-9)


def bootstrap_multiplicities(seed, size, repeats=2000):
    random = np.random.RandomState(seed)
    draws = random.randint(0, size, (repeats, size))
    counts = np.zeros((repeats, size), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(repeats), size), draws.ravel()), 1)
    return counts


class Review:
    def __init__(self, root, stage):
        self.root, self.stage, self.checks, self.pending = root, STAGES[stage], [], []
        self.artifacts, self.datasets = {}, {}

    def check(self, name, value, detail=None):
        self.checks.append({'check': name, 'passed': bool(value), 'detail': detail})

    def file(self, path, name, stage=0):
        path = Path(path)
        if path.is_file():
            return True
        self.pending.append({'artifact': str(path.relative_to(self.root)), 'stage': stage, 'reason': name})
        if self.stage >= stage:
            self.check(name, False, 'Required evidence absent; no inferred success')
        return False

    def hashed(self, path, expected=None):
        path = Path(path)
        digest = sha(path)
        # Encoder provenance includes immutable external assets/executables.
        # Their hashes require the same check, while local display keys stay short.
        key = str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)
        self.artifacts[key] = digest
        if expected is not None:
            self.check('hash:' + key, digest == expected)
        return digest

    def protocol(self):
        path = self.root / 'configs/PLAN_B_PROTOCOL.json'
        protocol = read(path)
        self.hashed(path)
        self.check('protocol_train_fit_only', protocol['data_policy']['train_fit_only_gradient'] is True)
        self.check('protocol_same120', protocol['pilot']['samples_per_dataset'] == 120 and protocol['pilot']['same_cohort_text_audio'])
        self.check('protocol_no_old_features_or_weights', 'no old checkpoint' in protocol['downstream']['old_assets'])
        for path in (self.root / 'scripts').glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
            self.check('python_ast:' + path.name, True)
        snapshot = self.root / 'provenance/downstream_source_snapshot.json'
        if self.file(snapshot, 'frozen_core_manifest'):
            for name, item in read(snapshot)['files'].items():
                path = self.root / 'scripts/core_snapshot' / name
                if self.file(path, 'core_file:' + name):
                    self.hashed(path, item['sha256'])
                    self.check('core_original_bytes:' + name, item['byte_identical'])
        return protocol

    def data(self, ds, spec):
        import torch
        folder = self.root / 'local_only/data' / ds
        path = folder / 'dataset.pt'
        if not self.file(path, ds + ':prepared_data'):
            return None
        self.hashed(path, spec['data_sha256'])
        preparation = read(self.root / 'provenance/data_preparation.json')
        prepared = preparation['datasets'][ds]
        self.check(ds + ':only_raw_and_visual_source_slots', prepared['source_pickle_slots'] == [0, 1, 2, 5, 6]
                   and prepared['legacy_text_feature_slot_indexed'] is False
                   and prepared['legacy_audio_feature_slot_indexed'] is False
                   and prepared['test_dictionary_values_indexed'] is False
                   and prepared['previous_teacher_pt_loaded'] is False)
        data = load_local(path)
        self.check(ds + ':only_authorized_splits', set(data['splits']) == {'train', 'validation'})
        train = {str(d['dialogue_id']): d for d in data['splits']['train']}
        val = {str(d['dialogue_id']): d for d in data['splits']['validation']}
        fit, dev = set(map(str, data['fit_dialogue_ids'])), set(map(str, data['inner_dev_dialogue_ids']))
        self.check(ds + ':dialogue_partitions', not (set(train) & set(val)) and not (fit & dev)
                   and fit | dev == set(train))
        self.check(ds + ':identity_seed', data['dataset'] == ds and data['seed'] == spec['seed'])
        for split, dialogues in [('train', train), ('validation', val)]:
            self.check(ds + ':' + split + '_rows', sum(len(d['utterance_ids']) for d in dialogues.values()) == spec[split])
            identities = set()
            for did, d in dialogues.items():
                n = len(d['utterance_ids'])
                self.check(ds + ':' + split + ':' + did + ':shape',
                    len(d['texts']) == len(d['audio_paths']) == len(d['labels']) == n
                    and tuple(d['visual'].shape) == (n, 342) and tuple(d['speaker'].shape) == (n, 2 if ds == 'IEMOCAP' else 9))
                self.check(ds + ':' + split + ':' + did + ':finite_inputs',
                    torch.isfinite(d['visual']).all().item() and torch.isfinite(d['speaker']).all().item()
                    and torch.equal(d['speaker'].sum(1), torch.ones(n)))
                self.check(ds + ':' + split + ':' + did + ':no_old_TA_fields',
                    not (set(d) & {'videoText', 'videoAudio', 'text_features', 'audio_features', 'teacher_features'}))
                for uid in d['utterance_ids']:
                    identities.add((did, str(uid)))
            self.check(ds + ':' + split + ':unique_ids', len(identities) == spec[split])
        self.check(ds + ':fit_count', sum(len(train[d]['utterance_ids']) for d in fit) == spec['fit'])
        self.check(ds + ':dev_count', sum(len(train[d]['utterance_ids']) for d in dev) == spec['inner_dev'])
        if ds == 'IEMOCAP':
            amendment = self.root / 'configs/DOWNSTREAM_CLASS_WEIGHT_AMENDMENT.json'
            if self.file(amendment, 'TrainFit_class_weight_amendment', 0):
                amended = read(amendment)
                labels = torch.cat([train[did]['labels'] for did in data['fit_dialogue_ids']])
                counts = torch.bincount(labels, minlength=6).tolist()
                self.check('class_weights:Fit_only_counts', amended['counts'] == counts and sum(counts) == spec['fit'])
                self.check('class_weights:inverse_frequency', np.allclose(amended['weights'], [spec['fit'] / n for n in counts], rtol=1e-12, atol=1e-12))
                self.check('class_weights:partition_and_no_Val', amended['fit_dialogue_ids'] == data['fit_dialogue_ids']
                           and amended['validation_used'] is False and amended['PLAN_B_PROTOCOL_bytes_modified'] is False)
                self.hashed(amendment)
        self.datasets[ds] = {'fit_dialogues': len(fit), 'inner_dev_dialogues': len(dev), 'validation_dialogues': len(val)}
        return data

    def text_model(self, ds, data, spec):
        folder = self.root / 'local_only/text_models' / ds
        path = folder / 'training_config.json'
        if not self.file(path, ds + ':text_training_configuration', 1):
            return
        config = read(path); training = config['protocol']
        self.check(ds + ':text_fit_identity', training['fit_dialogue_ids'] == data['fit_dialogue_ids']
                   and training['inner_dev_dialogue_ids'] == data['inner_dev_dialogue_ids'])
        self.check(ds + ':text_public_initialization', training['public_weight_sha256'] == self.protocol_data['text']['initial_weights_sha256'])
        self.check(ds + ':text_no_Val_selection', training['official_validation_model_selection'] is False and training['test_accessed'] is False)
        self.check(ds + ':text_dataset_hash', training['data_sha256'] == spec['data_sha256'])
        self.check(ds + ':text_running_code_locked', training['script_sha256'] == sha(self.root / 'scripts/text_encoder.py'))
        supervisor = self.root / 'provenance' / ('text_stage_' + ds + '.json')
        if supervisor.exists():
            launched = read(supervisor)
            self.check(ds + ':text_master_protocol_unmodified', launched['protocol_sha256'] == sha(self.root / 'configs/PLAN_B_PROTOCOL.json'))
            self.check(ds + ':text_launch_data_code_hash', launched['script_sha256'] == training['script_sha256']
                       and launched['data_sha256'] == spec['data_sha256'])
        update = folder / 'gradient_update_verification.json'
        if self.file(update, ds + ':text_real_gradient_check', 1):
            verification = read(update)
            self.check(ds + ':text_full_backbone_trainable', verification['all_backbone_parameters_trainable'])
            for key in ('backbone', 'classifier'):
                self.check(ds + ':text_' + key + '_updated', verification[key]['gradient_present'] and verification[key]['gradient_finite']
                           and verification[key]['gradient_l2'] > 0 and verification[key]['max_parameter_update'] > 0)
        lock = folder / 'best_model_lock.json'
        if self.file(lock, ds + ':selected_text_encoder_complete', 1):
            locked = read(lock)
            self.check(ds + ':text_complete_not_refitted', locked['status'] == 'COMPLETE' and locked['dataset'] == ds
                       and locked['refit_full_train'] is False and locked['official_validation_used_for_selection'] is False)
            self.hashed(folder / 'best_classifier/model.safetensors', locked['model_sha256'])
            history = read(folder / 'history.json')
            winner = max(history, key=lambda row: row['inner_dev']['macro_f1'])
            self.check(ds + ':text_inner_dev_argmax', locked['best_epoch'] == winner['epoch']
                       and np.isclose(locked['best_internal_macro_f1'], winner['inner_dev']['macro_f1']))

    def features(self, ds, data):
        import torch
        folder = self.root / 'local_only/data' / ds / 'features'
        for kind in ('text', 'audio'):
            path = folder / (kind + '_features.pt')
            if not self.file(path, ds + ':' + kind + '_features', 1):
                continue
            # Audio partial checkpoints are intentionally not treated as completion.
            if kind == 'audio' and not self.file(path.with_suffix('.metadata.json'), ds + ':audio_extraction_complete', 1):
                continue
            values = load_local(path)
            self.hashed(path)
            for split in ('train', 'validation'):
                dialogues = {str(d['dialogue_id']): d for d in data['splits'][split]}
                self.check(ds + ':' + kind + ':' + split + ':all_dialogues', set(values[split]) == set(dialogues))
                for did, d in dialogues.items():
                    if did not in values[split]:
                        continue
                    n = len(d['utterance_ids']); array = values[split][did]
                    expected = (n, 4, 1024) if kind == 'text' else (n, 768)
                    self.check(ds + ':' + kind + ':' + split + ':' + did + ':array',
                               tuple(array.shape) == expected and array.dtype == torch.float32 and torch.isfinite(array).all().item())
                    if kind == 'audio':
                        mask = values['audio_missing_mask'][split][did]
                        self.check(ds + ':audio:' + split + ':' + did + ':missing_mask',
                            tuple(mask.shape) == (n,) and mask.dtype == torch.bool and torch.equal(array[mask], torch.zeros_like(array[mask])))
                        records = values['utterance_metadata'][split][did]
                        self.check(ds + ':audio:' + split + ':' + did + ':row_metadata', len(records) == n
                            and [str(row['utterance_id']) for row in records] == list(map(str, d['utterance_ids']))
                            and [bool(row['audio_missing']) for row in records] == mask.tolist())
            if kind == 'text':
                manifest_path = path.with_suffix('.manifest.json')
                if self.file(manifest_path, ds + ':text_feature_manifest', 1):
                    meta = read(manifest_path)
                    self.check(ds + ':text_layer_order_no_norm', meta['layer_order'] == [-1, -2, -3, -4]
                               and meta['extra_normalization'] is None and meta['fp32_eval'] is True)
                    self.check(ds + ':text_feature_hash', meta['feature_file_sha256'] == sha(path))
                    self.check(ds + ':text_model_hash_chain', meta['model_sha256'] == read(self.root / 'local_only/text_models' / ds / 'best_model_lock.json')['model_sha256'])
            else:
                meta = values['metadata']
                self.check(ds + ':audio_frozen', meta['trainable_parameters'] == 0)
                self.check(ds + ':audio_source_code', meta['encoder_code_sha256'] == sha(self.root / 'scripts/audio_encoder.py'))
                self.check(ds + ':audio_data_hash', meta['dataset_sha256'] == self.protocol_data['datasets'][ds]['data_sha256'])
            del values

    def models(self, ds, data):
        folder = self.root / 'local_only/data' / ds
        for stage in ('m3net', 'cmerc'):
            run = folder / 'runs' / stage
            if not self.file(run / 'provenance.json', ds + ':' + stage + '_provenance', 2):
                continue
            provenance = read(run / 'provenance.json'); c = provenance['effective_config']
            self.check(ds + ':' + stage + ':no_Val_selection', c['selection_protocol'] == 'effectiveTrain_inner_dialogue_dev'
                       and c['selection_metric'] == 'macro_f1' and c['initialization'] == 'fresh_random_no_M3Net_warmstart')
            self.check(ds + ':' + stage + ':mechanisms', all(c[key] == (stage == 'cmerc') for key in ('calibrate', 'contrastlearning', 'courselearning')))
            self.check(ds + ':' + stage + ':new_audio_size', c['audio_dim'] == 768 and c['text_layers'] == 4 and c['causal'] is False)
            if ds == 'IEMOCAP':
                amendment = self.root / 'configs/DOWNSTREAM_CLASS_WEIGHT_AMENDMENT.json'
                expected_weights = read(amendment)
                self.check(ds + ':' + stage + ':Fit_weight_statistics', c['class_weights']['counts'] == expected_weights['counts']
                           and np.allclose(c['class_weights']['weights'], expected_weights['weights'], rtol=1e-12, atol=1e-12))
                self.check(ds + ':' + stage + ':weight_amendment_bound', provenance['class_weight_amendment_sha256'] == sha(amendment))
            for name, digest in provenance['inputs'].items():
                path = Path(name)
                self.check(ds + ':' + stage + ':input_under_current_run:' + path.name, path.resolve().is_relative_to(self.root.resolve()))
                if path.is_file():
                    self.check(ds + ':' + stage + ':input_hash:' + path.name, sha(path) == digest)
            completion = run / 'completion.json'
            if self.file(completion, ds + ':' + stage + ':termination_evidence', 2):
                done = read(completion)
                self.check(ds + ':' + stage + ':completed', done.get('status') == 'COMPLETE')
                self.hashed(run / 'best.pt', done['best_checkpoint_sha256'])
                self.hashed(run / 'last.pt', done['last_checkpoint_sha256'])
                self.check(ds + ':' + stage + ':frozen_adapter_hash', done['adapter_sha256'] == sha(self.root / 'scripts/downstream.py'))
                self.check(ds + ':' + stage + ':frozen_protocol_hash', done['protocol_sha256'] == sha(self.root / 'configs/PLAN_B_PROTOCOL.json'))
                self.check(ds + ':' + stage + ':termination_reason', done['termination_reason'] in ('max_epochs', 'early_stopping_patience'))
            metric_path = run / 'metrics.jsonl'
            if self.file(metric_path, ds + ':' + stage + ':training_metrics', 2):
                rows = [json.loads(line) for line in metric_path.read_text().splitlines() if line.strip()]
                self.check(ds + ':' + stage + ':contiguous_epochs', [r['epoch'] for r in rows] == list(range(len(rows))))
                self.check(ds + ':' + stage + ':no_scl_nonfinite_hidden', all(r['official_nonfinite_fallback_count'] == 0 for r in rows))
                if rows:
                    best = max(rows, key=lambda row: row['inner_dev']['macro_f1'])
                    self.check(ds + ':' + stage + ':best_epoch_inner_dev', rows[-1]['best_epoch'] == best['epoch'])
                    if stage == 'cmerc':
                        self.check(ds + ':cmerc:active_SCL_checks', all(r['scl_raw_finite_checks'] > 0 for r in rows))
        path = folder / 'new_difficulty.pt'
        if self.file(path, ds + ':new_difficulty', 2):
            d = load_local(path)
            self.check(ds + ':difficulty_only_fit', set(d['difficulties']) == set(map(str, data['fit_dialogue_ids'])))
            self.check(ds + ':difficulty_new_source', d['new_features_only'] and d['m3net_checkpoint_sha256'] == sha(folder / 'runs/m3net/best.pt'))
            self.check(ds + ':difficulty_finite', all(np.isfinite(x) for x in d['difficulties'].values()))
            components_path = folder / 'difficulty_components.json'
            if self.file(components_path, ds + ':difficulty_component_evidence', 2):
                components = read(components_path)
                self.check(ds + ':difficulty_component_fit_coverage', {str(x['dialogue_id']) for x in components} == set(data['fit_dialogue_ids'])
                           and len(components) == len(data['fit_dialogue_ids']))
                train = {str(x['dialogue_id']): x for x in data['splits']['train']}
                for row in components:
                    did = str(row['dialogue_id']); dialogue = train[did]
                    n = len(dialogue['utterance_ids'])
                    speaker_term = int(dialogue['speaker'].nonzero()[:, 1].max()) + 1
                    gaps = [row[key] for key in ('text_gap_sum', 'audio_gap_sum', 'visual_gap_sum', 'intra_gap_sum', 'inter_gap_sum')]
                    value = (sum(gaps) + speaker_term) / (n + speaker_term)
                    self.check(ds + ':difficulty_formula:' + did, all(np.isfinite(gaps)) and min(gaps) >= 0
                               and row['length'] == n and row['speaker_term'] == speaker_term
                               and np.isclose(value, row['difficulty'], rtol=2e-6, atol=1e-7)
                               and row['difficulty'] == d['difficulties'][did])
        path = folder / 'validation_metrics.json'
        if self.file(path, ds + ':final_baseline_metrics', 2):
            baseline = read(path)
            self.check(ds + ':baseline_fixed_rows', baseline['n'] == self.protocol_data['datasets'][ds]['validation'])
            self.check(ds + ':baseline_metrics_finite', all(np.isfinite(baseline[k]) for k in ('accuracy', 'macro_f1', 'weighted_f1', 'ece15', 'brier_sum_classes', 'nll')))
            predictions_path = folder / 'validation_predictions.pt'
            if self.file(predictions_path, ds + ':final_baseline_probabilities', 2):
                records = load_local(predictions_path)
                probabilities = np.stack([prob(r['probability']) for r in records])
                y = np.asarray([int(r['label']) for r in records]); predicted = probabilities.argmax(1)
                confidence = probabilities.max(1); correct = predicted == y
                self.check(ds + ':baseline_probabilities_fixed_rows', len(y) == self.protocol_data['datasets'][ds]['validation'])
                classes = probabilities.shape[1]; f1 = []; support = []
                for label in range(classes):
                    tp = np.sum((y == label) & (predicted == label)); actual = np.sum(y == label); positive = np.sum(predicted == label)
                    f1.append(float(2 * tp / (actual + positive)) if actual + positive else 0.0); support.append(actual)
                bins = np.minimum((confidence * 15).astype(int), 14)
                ece = sum(float(np.abs(np.sum(confidence[bins == i] - correct[bins == i])) / len(y)) for i in range(15))
                calculated = {'accuracy': float(correct.mean()), 'macro_f1': float(np.mean(f1)),
                    'weighted_f1': float(np.average(f1, weights=support)), 'ece15': ece,
                    'brier_sum_classes': float(np.square(probabilities - np.eye(classes)[y]).sum(1).mean()),
                    'nll': float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1)).mean())}
                for key, value in calculated.items():
                    self.check(ds + ':baseline_recomputed:' + key, np.isclose(value, baseline[key], rtol=1e-9, atol=1e-10))

    def producer_bundles(self, ds, samples):
        """Audit V2 CPU interchange only; never load an encoder or model weights."""
        import torch
        found = []
        for path in self.root.rglob('*.manifest.json'):
            sidecar = read(path)
            if sidecar.get('format') == 'PLAN_B_PRECOMPUTED_FEATURES_V2' and sidecar.get('dataset') == ds:
                found.append((path, sidecar))
        pilots = [(p, m) for p, m in found if m.get('mode') == 'pilot']
        smokes = [(p, m) for p, m in found if m.get('mode') == 'smoke']
        self.check(ds + ':v2_one_smoke_and_one_pilot_bundle', len(smokes) == len(pilots) == 1)
        selected = {row['sample_id']: row for row in samples}
        clean_folder = self.root / 'local_only/data' / ds / 'features'
        clean_text = load_local(clean_folder / 'text_features.pt')
        clean_audio = load_local(clean_folder / 'audio_features.pt')
        selected_hash = sha(next(p for p in self.root.rglob('sample_manifest.csv')
                                  if csv_rows(p) and csv_rows(p)[0]['dataset'] == ds))
        for sidecar_path, sidecar in found:
            mode = sidecar['mode']; name = ds + ':v2:' + mode
            cache_path = sidecar_path.with_name(sidecar_path.name[:-len('.manifest.json')] + '.pt')
            if not self.file(cache_path, name + ':cache_present', 3):
                continue
            self.hashed(cache_path, sidecar['cache_sha256'])
            bundle = load_local(cache_path); metadata = bundle['metadata']
            self.check(name + ':sidecar_equal', sidecar.get('metadata') == metadata
                       and sidecar.get('status') == metadata.get('status') == 'COMPLETE'
                       and metadata.get('format') == sidecar.get('format')
                       and metadata.get('mode') == mode and metadata.get('dataset') == ds)
            runtime = metadata['runtime']
            self.check(name + ':pinned_encoder_runtime', (runtime.get('environment'), runtime.get('torch'), runtime.get('transformers'))
                       == ('TEXT_repro', '2.2.0+cu118', '4.42.3'))
            self.check(name + ':unchanged_encoder_state', metadata.get('encoder_states_unchanged') is True
                       and metadata.get('input_files_unchanged') is True
                       and metadata.get('encoder_states_before') == metadata.get('encoder_states_after')
                       and set(metadata.get('encoder_states_before', {})) == {'text', 'audio'})
            self.check(name + ':no_cmerc_or_test', metadata.get('no_cmerc_inference_performed') is True
                       and metadata.get('test_indexed') is False
                       and metadata.get('official_validation_indexed') is (mode == 'pilot'))
            self.hashed(self.root / 'configs/PLAN_B_PROTOCOL.json', metadata['scientific_protocol_sha256'])
            for source, digest in metadata['input_sha256'].items():
                archive = historical_smoke_source(self.root, source, digest) if mode == 'smoke' else None
                self.hashed(archive if archive is not None else Path(source), digest)
            completion_path = self.root / 'local_only/data' / ds / 'runs/cmerc/completion.json'
            completed = read(completion_path)
            actual_ref = {'path': str(completion_path.resolve()), 'sha256': sha(completion_path),
                          'completed_at': completed.get('completed_at_utc')}
            checkpoint_path = completion_path.parent / 'best.pt'
            self.check(name + ':actual_baseline_completed', completed.get('status') == 'COMPLETE'
                       and completed.get('stage') == 'cmerc'
                       and completed['best_checkpoint_sha256'] == sha(checkpoint_path)
                       and metadata['input_sha256'].get(str(checkpoint_path.resolve())) == completed['best_checkpoint_sha256'])
            if mode == 'pilot':
                self.check(name + ':true_completion_bound', metadata.get('baseline_completion') == actual_ref
                           and metadata['input_sha256'].get(actual_ref['path']) == actual_ref['sha256'])
            else:
                # Completed historical smoke is retained, with a separate explicit
                # provenance correction; never rewrite its cache or numerical output.
                errata_path = self.root / 'provenance/SMOKE_COMPLETION_PROVENANCE_ERRATA.json'
                if self.file(errata_path, name + ':historical_completion_erratum', 3):
                    errata = read(errata_path); entry = errata['datasets'][ds]
                    report_path = self.root / 'provenance' / ('raw_path_smoke_' + ds + '.json')
                    report = read(report_path)
                    self.hashed(report_path, entry['historical_smoke_report_sha256'])
                    self.hashed(Path(entry['baseline_lock_path']), entry['baseline_lock_sha256'])
                    lock = read(entry['baseline_lock_path'])
                    self.check(name + ':erratum_only_original_preserved', errata['status'] == 'PROVENANCE_ERRATUM_ONLY'
                               and errata['corrected_raw_path_smoke_sha256'] == sha(self.root / 'scripts/raw_path_smoke.py')
                               and Path(entry['historical_smoke_report']).resolve() == report_path.resolve()
                               and entry['historical_reports_and_caches_modified'] is False
                               and entry['feature_or_probability_results_changed'] is False
                               and entry['historical_precomputed_features_sha256'] == sha(cache_path)
                               and report['precomputed_features_sha256'] == sha(cache_path))
                    self.check(name + ':erratum_baseline_identity', entry['incorrect_historical_baseline_completion']
                               == metadata['baseline_completion'] == report['baseline_completion']
                               and entry['correct_baseline_completion'] == actual_ref
                               and entry['actual_completion_status'] == completed['status'] == 'COMPLETE'
                               and entry['actual_completion_stage'] == completed['stage'] == 'cmerc'
                               and entry['best_checkpoint_sha256'] == lock['checkpoint_sha256'] == completed['best_checkpoint_sha256'])
                    compared = entry['completion_inputs_match_original_smoke_inputs']
                    complete_inputs = {str(Path(path).resolve()): digest for path, digest in completed['input_hashes'].items()}
                    smoke_inputs = {str(Path(path).resolve()): digest for path, digest in report['input_hashes'].items()}
                    self.check(name + ':erratum_inputs_recomputed', bool(compared) and all(value is True for value in compared.values())
                               and all(complete_inputs.get(str(Path(source).resolve())) == smoke_inputs.get(str(Path(source).resolve()))
                                       == metadata['input_sha256'].get(str(Path(source).resolve())) == sha(source) for source in compared))
            text = metadata['encoder_provenance']['text']
            self.check(name + ':text_extraction_contract', text.get('normalization') == 'none'
                       and text.get('layer_order') == [-1, -2, -3, -4]
                       and text.get('all_parameters_frozen') is True and text.get('precision') == 'FP32 eval')
            manifest = read(self.root / 'local_only/data' / ds / 'features/text_features.manifest.json')
            self.check(name + ':same_clean_text_encoder', text.get('weights_sha256') == manifest['model_sha256']
                       and text.get('encoder_code_sha256') == manifest['script_sha256']
                       and text.get('tokenizer_files') == manifest['tokenizer_hashes'])
            audio = metadata['encoder_provenance']['audio']
            self.check(name + ':same_clean_audio_encoder', all(audio.get(key) == clean_audio['metadata'].get(key)
                       for key in ('protocol_hash', 'checkpoint_manifest', 'encoder_code_sha256', 'torch', 'transformers')))
            for source in (self.root / 'scripts/text_encoder.py', self.root / 'scripts/audio_encoder.py',
                           self.root / 'scripts/raw_pilot.py', self.root / 'scripts/raw_path_smoke.py',
                           self.root / 'scripts/extract_pilot_features.py',
                           clean_folder / 'text_features.pt', clean_folder / 'audio_features.pt'):
                bound = metadata['input_sha256'].get(str(source.resolve()))
                self.check(name + ':required_source_bound:' + source.name,
                           bound == sha(source) or (mode == 'smoke'
                           and historical_smoke_source(self.root, source, bound) is not None))
            if mode == 'pilot':
                self.check(name + ':same_fixed120', metadata['sample_manifest_sha256'] == selected_hash
                           and set(bundle['clean_controls']) == set(selected))
                expected = {(sid, family, variant, 'text' if family in ('neutral_append', 'semantic_paraphrase') else 'audio')
                            for sid in selected for family, variants in VARIANTS.items() for variant in variants}
            else:
                self.check(name + ':one_train_smoke_target', set(bundle['clean_controls']) == {'train_smoke'})
                expected = {('train_smoke', 'neutral_append', 'neutral_append', 'text'),
                            ('train_smoke', 'pitch_shift', 'pitch_plus_0.5', 'audio')}
            observed = [(r['sample_id'], r['perturbation_type'], r['variant_id'], r['modality']) for r in bundle['variants']]
            self.check(name + ':exact_fixed_variant_schema', len(set(observed)) == len(observed) and set(observed) == expected)
            self.check(name + ':counts', metadata['variants_planned'] == len(observed)
                       and metadata['variants_extracted'] == sum(r['status'] == 'EXTRACTED' for r in bundle['variants'])
                       and metadata['variants_failed'] == sum(r['status'] == 'FAILED' for r in bundle['variants']))
            records = []
            for sid, controls in bundle['clean_controls'].items():
                self.check(name + ':control_modalities:' + sid, set(controls) == {'text', 'audio'})
                for modality, row in controls.items():
                    self.check(name + ':control_identity:' + sid + ':' + modality,
                               row.get('sample_id') == sid and row.get('modality') == modality)
                    records.append((row, False))
            records.extend((row, True) for row in bundle['variants'])
            for row, is_variant in records:
                sid, modality = row['sample_id'], row['modality']
                key = name + ':record:' + sid + ':' + (row['variant_id'] if is_variant else 'clean_' + modality)
                self.check(key + ':known_status_modality', row['status'] in ('EXTRACTED', 'FAILED') and modality in ('text', 'audio'))
                if mode == 'pilot' and sid in selected:
                    self.check(key + ':fixed_target_identity', str(row['dialogue_id']) == selected[sid]['dialogue_id']
                               and int(row['utterance_index']) == int(selected[sid]['utterance_index']))
                split = 'validation' if mode == 'pilot' else 'train'
                cache = clean_text if modality == 'text' else clean_audio
                original = cache[split][str(row['dialogue_id'])][int(row['utterance_index'])]
                self.check(key + ':same_original_cache_row', row.get('original_feature_sha256') == array_sha(original))
                if row['status'] == 'FAILED':
                    self.check(key + ':failed_retained_ineligible', bool(row.get('error')) and not row.get('label_preserving_main_eligible'))
                    continue
                feature = row.get('feature'); shape = (4, 1024) if modality == 'text' else (768,)
                self.check(key + ':cpu_fp32_shape_hash', isinstance(feature, torch.Tensor)
                           and feature.device.type == 'cpu' and feature.dtype == torch.float32
                           and tuple(feature.shape) == shape and bool(torch.isfinite(feature).all())
                           and array_sha(feature) == row['target_feature_sha256'])
                if modality == 'text':
                    tokens = row.get('actual_token_ids', [])
                    self.check(key + ':actual_token_hash', bool(tokens) and tokens[0] == 0 and tokens[-1] == 2
                               and len(tokens) <= 512 and array_sha(np.asarray(tokens, dtype=np.int64)) == row['actual_token_ids_sha256'])
                if mode == 'smoke':
                    self.check(key + ':engineering_ineligible', not row.get('label_preserving_main_eligible'))
        smoke_report = self.root / 'provenance' / ('raw_path_smoke_' + ds + '.json')
        if self.file(smoke_report, ds + ':real_train_smoke_evidence', 3):
            report = read(smoke_report)
            self.check(ds + ':real_train_smoke_pass', report.get('status') == 'PASS' and report.get('pass_gate') is True
                       and report.get('official_validation_indexed') is False and report.get('inner_dev_indexed') is False
                       and report.get('test_indexed') is False and report.get('encoder_models_instantiated_in_this_process') is False)

    def pilot(self, ds, data):
        manifests = []
        for path in self.root.rglob('sample_manifest.csv'):
            rows = csv_rows(path)
            if rows and rows[0]['dataset'] == ds:
                manifests.append((path, rows))
        if not manifests:
            self.file(self.root / 'results' / ds / 'sample_manifest.csv', ds + ':pilot_same120_manifest', 3)
            return
        self.check(ds + ':pilot_exactly_one_cohort', len(manifests) == 1)
        path, samples = manifests[0]; folder = path.parent
        self.check(ds + ':pilot_120_unique', len(samples) == 120 and len({r['sample_id'] for r in samples}) == 120)
        self.check(ds + ':pilot_audio_available_only', all(not yes(r['original_audio_missing']) for r in samples))
        selected = {r['sample_id']: r for r in samples}
        self.producer_bundles(ds, samples)
        contexts = {str(d['dialogue_id']): len(d['utterance_ids']) for d in data['splits']['validation']}
        artifacts = {(p.parent.name, p.stem): p for p in self.root.glob('**/features_and_predictions/*/*.pt')}
        controls = {}
        for control_path in folder.glob('clean_reextraction_controls_*.csv'):
            for control in csv_rows(control_path):
                controls[(control['sample_id'], control['modality'])] = control
        predictions = sorted(folder.glob('perturbed_predictions_*.csv'))
        self.check(ds + ':pilot_no_mixed_execution_files', not (any(p.name.endswith('_both.csv') for p in predictions) and len(predictions) > 1))
        records = [r for p in predictions for r in csv_rows(p)]
        self.check(ds + ':pilot_planned_1200_records', len(records) == 1200)
        identities = [(r['sample_id'], r['variant_id']) for r in records]
        self.check(ds + ':pilot_unique_versions', len(set(identities)) == len(identities))
        families = {}
        derived_variants = {}
        for row in records:
            key = (row['sample_id'], row['perturbation_type'])
            families.setdefault(key, []).append(row)
            self.check(ds + ':pilot_member:' + ':'.join(key), row['sample_id'] in selected
                       and row['dialogue_id'] == selected[row['sample_id']]['dialogue_id']
                       and row['utterance_index'] == selected[row['sample_id']]['utterance_index'])
            if row['status'] == 'SUCCESS':
                p, q = prob(row['original_probabilities']), prob(row['perturbed_probabilities'])
                label = int(row['true_label']); original, changed = int(p.argmax()), int(q.argmax())
                scalars = {'original_prediction': original, 'original_correct': int(original == label),
                    'original_confidence': float(p.max()), 'perturbed_prediction': changed,
                    'perturbed_max_confidence': float(q.max()),
                    'perturbed_original_class_probability': float(q[original]),
                    'perturbed_original_label_probability': float(q[label])}
                self.check(ds + ':variant_probability_scalars:' + row['sample_id'] + ':' + row['variant_id'],
                           all(optional_close(row.get(field), value) for field, value in scalars.items()))
                eligible = yes(row.get('label_preserving_main_eligible'))
                derived_variants[(row['sample_id'], row['variant_id'])] = {
                    'original_correct': int(original == label), 'original_confidence': float(p.max()),
                    'flip': int(original != changed),
                    'correct_to_wrong': int(original == label and changed != label) if eligible else None,
                    'wrong_to_correct': int(original != label and changed == label) if eligible else None,
                    'confidence_change': float(q.max() - p.max()),
                    'abs_confidence_change': float(abs(q.max() - p.max())),
                    'original_class_probability_change': float(q[original] - p[original]),
                    'original_label_probability_change': float(q[label] - p[label]),
                    'js': independent_behavior(p, [q])['js_mean'],
                    'label_preserving_main_eligible': eligible,
                    'label_confirmed_for_perturbed_accuracy': eligible,
                    'automatic_qc_pass': yes(row.get('automatic_qc_pass'))}
                self.check(ds + ':target_only:' + row['sample_id'] + ':' + row['variant_id'],
                           yes(row['non_target_features_bit_exact']) and yes(row['dialogue_context_hashes_unchanged']))
                control = controls.get((row['sample_id'], row['modality']), {})
                self.check(ds + ':clean_negative_control:' + row['sample_id'] + ':' + row['modality'],
                    all(yes(control.get(k, False)) for k in ('clean_feature_allclose', 'clean_probability_allclose', 'clean_prediction_unchanged')))
                artifact = artifacts.get((row['sample_id'], row['variant_id']))
                self.check(ds + ':full_context_artifact_present:' + row['sample_id'] + ':' + row['variant_id'], artifact is not None)
                if artifact is not None:
                    self.hashed(artifact, row['artifact_sha256'])
                    stored = load_local(artifact)
                    probabilities = np.asarray(stored['full_dialogue_probabilities'])
                    expected_classes = 6 if ds == 'IEMOCAP' else 7
                    self.check(ds + ':full_context_shape:' + row['sample_id'] + ':' + row['variant_id'],
                        probabilities.shape == (contexts[row['dialogue_id']], expected_classes)
                        and np.isfinite(probabilities).all() and (probabilities >= 0).all()
                        and np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0)
                        and np.allclose(probabilities[int(row['utterance_index'])], prob(row['perturbed_probabilities']), atol=1e-10, rtol=1e-9))
                if yes(row['label_preserving_main_eligible']):
                    self.check(ds + ':manual_review:' + row['sample_id'] + ':' + row['variant_id'],
                               row['label_review_status'].upper() == 'PASS' and bool(row['reviewer'].strip()) and yes(row['automatic_qc_pass']))
            else:
                expected_error = ('Fixed variant failed automatic text/token QC' if row['modality'] == 'text'
                                  else 'Fixed audio variant failed signal QC')
                self.check(ds + ':legal_retained_QC_failure:' + row['sample_id'] + ':' + row['variant_id'],
                           row['status'] == 'FAILED' and row.get('error') == expected_error
                           and not yes(row.get('automatic_qc_pass')) and not yes(row.get('label_preserving_main_eligible')))
        for sid in selected:
            for kind, planned_versions in VARIANTS.items():
                found = families.get((sid, kind), [])
                self.check(ds + ':all_planned_versions:' + sid + ':' + kind,
                           {r['variant_id'] for r in found} == planned_versions and len(found) == len(planned_versions))
        for filename in ('prediction_flips.csv', 'confidence_changes.csv'):
            outputs = list(self.root.rglob(filename))
            self.check(ds + ':variant_derived_table_present:' + filename, len(outputs) == 1)
            for output in outputs:
                rows = [row for row in csv_rows(output) if row['dataset'] == ds]
                keys = [(row['sample_id'], row['variant_id']) for row in rows]
                self.check(ds + ':variant_derived_exact_success_denominator:' + filename,
                           len(set(keys)) == len(keys) and set(keys) == set(derived_variants))
                for row in rows:
                    key = (row['sample_id'], row['variant_id']); expected = derived_variants.get(key)
                    if expected is None:
                        continue
                    self.check(ds + ':variant_derived_recomputed:' + filename + ':' + ':'.join(key),
                               all(yes(row.get(metric)) == value if isinstance(value, bool)
                                   else optional_close(row.get(metric), value) for metric, value in expected.items()))
        analysis_files = list(self.root.rglob('behavior_features_all_complete.csv'))
        for p in analysis_files:
            for row in csv_rows(p):
                if row['dataset'] != ds:
                    continue
                key = (row['sample_id'], row['perturbation_type']); versions = families.get(key, [])
                planned = len(VARIANTS[row['perturbation_type']])
                self.check(ds + ':fixedK:' + ':'.join(key), len(versions) == planned and len({v['variant_id'] for v in versions}) == planned
                           and all(v['status'] == 'SUCCESS' for v in versions))
                if versions and all(v['status'] == 'SUCCESS' for v in versions):
                    actual = independent_behavior(versions[0]['original_probabilities'], [v['perturbed_probabilities'] for v in versions])
                    for metric, value in actual.items():
                        self.check(ds + ':behavior:' + ':'.join(key) + ':' + metric, np.isclose(float(row[metric]), value, rtol=1e-9, atol=1e-12))
        self.check(ds + ':behavior_analysis_present', bool(analysis_files))
        for p in self.root.rglob('analysis_verification.json'):
            a = read(p)
            self.check(ds + ':bootstrap_unit_and_repeats:' + str(p.parent.relative_to(self.root)), a['repetitions'] == 2000
                       and 'Dialogue' in a['bootstrap_unit'] and a['no_mixed_k'] and a['no_fitted_reliability_or_calibration'])
            self.check(ds + ':matched_Dialogue_adjustment:' + str(p.parent.relative_to(self.root)),
                       'full-cohort' in a.get('shared_bootstrap', '').lower()
                       and 'Dialogue x' in a.get('dialogue_effect_adjustment', '')
                       and a.get('unconfirmed_audio_not_used_for_perturbed_accuracy_or_calibration') is True)
            self.verify_statistics(ds, p.parent, a, samples)

    def verify_statistics(self, ds, folder, audit, samples):
        """Independent stored-CSV reconstruction, including exact cluster CI draws."""
        universe = sorted({r['dialogue_id'] for r in samples})
        positions = {did: i for i, did in enumerate(universe)}
        repeats, base_seed = int(audit['repetitions']), int(audit['seed'])
        draw_cache = {}

        def multiplicity(tag):
            seed = int(hashlib.sha256((str(base_seed) + ds + tag).encode()).hexdigest()[:8], 16)
            if seed not in draw_cache:
                draw_cache[seed] = bootstrap_multiplicities(seed, len(universe), repeats)
            return draw_cache[seed]

        pairs = [('behavior_features_main.csv', 'correct_wrong_comparison.csv', 'within_dialogue_confidence_comparison.csv'),
                 ('audio_exploratory_behavior_features.csv', 'audio_exploratory_correct_wrong_comparison.csv',
                  'audio_exploratory_within_dialogue_confidence_comparison.csv')]
        for source_name, comparison_name, matched_name in pairs:
            source_path = folder / source_name
            if not source_path.exists():
                continue
            source = [r for r in csv_rows(source_path) if r['dataset'] == ds]
            comparison_path = folder / comparison_name
            if comparison_path.exists():
                for row in csv_rows(comparison_path):
                    if row['dataset'] != ds:
                        continue
                    kind, metric, bin_name = row['perturbation_type'], row['metric'], row['confidence_bin']
                    values = [r for r in source if r['perturbation_type'] == kind and (bin_name == 'all' or r['confidence_bin'] == bin_name)]
                    correct = [r for r in values if int(r['original_correct']) == 1]
                    wrong = [r for r in values if int(r['original_correct']) == 0]
                    correct_ids = {r['dialogue_id'] for r in correct}; wrong_ids = {r['dialogue_id'] for r in wrong}
                    name = ds + ':cluster_CI:' + source_name + ':' + kind + ':' + bin_name + ':' + metric
                    self.check(name + ':support', int(row['correct_samples']) == len(correct) and int(row['error_samples']) == len(wrong)
                               and int(row['dialogues']) == len(universe) and int(row['shared_correct_error_dialogues']) == len(correct_ids & wrong_ids))
                    difference = np.mean([float(r[metric]) for r in wrong]) - np.mean([float(r[metric]) for r in correct]) if correct and wrong else None
                    self.check(name + ':estimate', optional_close(row['error_minus_correct'], difference))
                    if len(correct_ids) < 2 or len(wrong_ids) < 2:
                        self.check(name + ':insufficient_NA', int(row['bootstrap_valid']) == 0 and optional_close(row['ci_low'], None) and optional_close(row['ci_high'], None))
                        continue
                    stats = np.zeros((len(universe), 4), dtype=np.float64)
                    for r in values:
                        column = 0 if int(r['original_correct']) == 1 else 2
                        stats[positions[r['dialogue_id']], column] += float(r[metric])
                        stats[positions[r['dialogue_id']], column + 1] += 1
                    totals = multiplicity(bin_name) @ stats
                    valid = (totals[:, 1] > 0) & (totals[:, 3] > 0)
                    estimates = totals[valid, 2] / totals[valid, 3] - totals[valid, 0] / totals[valid, 1]
                    ci = np.quantile(estimates, [.025, .975]) if len(estimates) else [None, None]
                    self.check(name + ':paired_bootstrap_recomputed', int(row['bootstrap_valid']) == int(valid.sum())
                               and optional_close(row['ci_low'], ci[0]) and optional_close(row['ci_high'], ci[1]))
            matched_path = folder / matched_name
            if not matched_path.exists():
                continue
            for row in csv_rows(matched_path):
                if row['dataset'] != ds:
                    continue
                kind, metric = row['perturbation_type'], row['metric']
                values = [r for r in source if r['perturbation_type'] == kind]
                grouped = {}
                for r in values:
                    grouped.setdefault((r['dialogue_id'], r['confidence_bin']), []).append(r)
                cells = []
                for (did, _), group in grouped.items():
                    correct = [r for r in group if int(r['original_correct']) == 1]
                    wrong = [r for r in group if int(r['original_correct']) == 0]
                    if correct and wrong:
                        cells.append((did, float(np.mean([float(r[metric]) for r in wrong]) - np.mean([float(r[metric]) for r in correct])), len(correct) + len(wrong)))
                name = ds + ':matched_CI:' + source_name + ':' + kind + ':' + metric
                supported = {r[0] for r in cells}
                self.check(name + ':support', int(row['eligible_cells']) == len(cells)
                           and int(row['matched_dialogues']) == len(supported) and int(row['matched_samples']) == sum(c[2] for c in cells))
                self.check(name + ':fixed_equal_cell_estimate', optional_close(row['error_minus_correct'], np.mean([c[1] for c in cells]) if cells else None))
                if len(supported) < 2:
                    self.check(name + ':insufficient_NA', int(row['bootstrap_valid']) == 0 and optional_close(row['ci_low'], None) and optional_close(row['ci_high'], None))
                    continue
                stats = np.zeros((len(universe), 2), dtype=np.float64)
                for did, difference, _ in cells:
                    stats[positions[did], 0] += difference
                    stats[positions[did], 1] += 1
                totals = multiplicity('matched_cells') @ stats
                valid = totals[:, 1] > 0
                estimates = totals[valid, 0] / totals[valid, 1]
                ci = np.quantile(estimates, [.025, .975]) if len(estimates) else [None, None]
                self.check(name + ':paired_bootstrap_recomputed', int(row['bootstrap_valid']) == int(valid.sum())
                           and optional_close(row['ci_low'], ci[0]) and optional_close(row['ci_high'], ci[1]))

    def run(self):
        self.protocol_data = self.protocol()
        for ds, spec in self.protocol_data['datasets'].items():
            try:
                data = self.data(ds, spec)
                if data is None:
                    continue
                self.text_model(ds, data, spec)
                if self.stage >= 1:
                    self.features(ds, data)
                if self.stage >= 2:
                    self.models(ds, data)
                if self.stage >= 3:
                    self.pilot(ds, data)
            except Exception as error:
                self.check(ds + ':unexpected_verification_error', False, {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        failed = [c for c in self.checks if not c['passed']]
        requested_pending = [p for p in self.pending if p['stage'] <= self.stage]
        return {'status': 'FAIL' if failed else 'INCOMPLETE' if requested_pending else 'PASS' if self.stage == 4 else 'PASS_FOR_REQUESTED_STAGE',
                'meaning': 'Verification of actual artifacts and protocol; no inference, training, new Validation evaluation, or calibration fitting',
                'requested_stage': next(k for k, v in STAGES.items() if v == self.stage),
                'complete': self.stage == 4 and not failed and not requested_pending,
                'verified_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'checks': self.checks,
                'check_count': len(self.checks), 'failed_checks': failed, 'pending': self.pending,
                'datasets': self.datasets, 'artifacts_sha256': self.artifacts,
                'verification_code_sha256': sha(__file__),
                'verifier_execution': {'model_forward': False, 'encoder_load': False, 'training': False,
                    'original_pickle_or_old_model_load': False, 'test_access': False, 'network': False, 'git': False}}


def self_test():
    import tempfile
    p = [.8, .2]; q = [[.1, .9], [.9, .1]]
    m = independent_behavior(p, q)
    assert np.isclose(m['mu'], .45) and np.isclose(m['delta'], .1) and np.isclose(m['flip_rate'], .5)
    assert not np.isclose(m['mu'], m['flip_rate'])
    symmetric = independent_behavior([.5, .5], [[.4, .6], [.6, .4]])
    assert symmetric['js_mean'] >= 0 and np.isfinite(symmetric['js_mean'])
    zeros = independent_behavior([1., 0.], [[1., 0.], [0., 1.]])
    assert np.isclose(zeros['js_mean'], np.log(2)/2)
    cancellation = independent_behavior([.6, .4], [[.8, .2], [.5, .5]])
    assert not np.isclose(cancellation['delta'], cancellation['mean_abs_confidence_change'])
    counts = bootstrap_multiplicities(1234, 5, 7)
    random = np.random.RandomState(1234)
    expected = np.stack([np.bincount(random.randint(0, 5, 5), minlength=5) for _ in range(7)])
    assert np.array_equal(counts, expected)
    with tempfile.TemporaryDirectory(prefix='planb-verifier-source-') as temporary:
        root = Path(temporary); scripts = root / 'scripts'; scripts.mkdir()
        archive = root / 'provenance/jsonl_lf_io_fix_v2'; archive.mkdir(parents=True)
        before = 'def read_jsonl(p):\n    return p.splitlines()\ndef unchanged():\n    return 1\n'
        after = 'def read_jsonl(p):\n    return p.split(chr(10))\ndef unchanged():\n    return 1\n'
        archived = archive / 'raw_pilot.py'; archived.write_text(before)
        source = scripts / 'raw_pilot.py'; source.write_text(after)
        digest = sha(archived)
        (archive / 'ARCHIVE_MANIFEST.json').write_text(json.dumps({'sha256': digest}))
        assert historical_smoke_source(root, source, digest) == archived
        assert historical_smoke_source(root, source, 'wrong digest') is None
        source.write_text(after.replace('return 1', 'return 2'))
        assert historical_smoke_source(root, source, digest) is None
        review = Review(scripts, 'data')
        assert review.hashed(archived, digest) == digest and review.checks[-1]['passed']
        assert str(archived) in review.artifacts
    print(json.dumps({'synthetic_self_test': 'PASS', 'real_data_opened': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=OUT)
    parser.add_argument('--stage', choices=tuple(STAGES), default='data')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    review = Review(args.root.resolve(), args.stage)
    result = review.run()
    path = args.root / 'verification.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'requested_stage': args.stage, 'complete': result['complete'],
                      'checks': result['check_count'], 'failed': len(result['failed_checks']), 'pending': len(result['pending'])}))


if __name__ == '__main__':
    main()
