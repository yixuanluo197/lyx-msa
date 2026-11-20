#!/usr/bin/env python3
"""Fixed raw-input pilot components. True Validation requires a frozen baseline hash lock.

This module never trains a model. Text edits require explicit attributed semantic review; automatic
audio checks establish signal quality, not preservation of an emotion label.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

TYPES = {'neutral_append': ['neutral_a', 'neutral_b'],
         'semantic_paraphrase': ['paraphrase_a', 'paraphrase_b'],
         'pitch_shift': ['pitch_plus_0.5', 'pitch_minus_0.5', 'pitch_plus_1', 'pitch_minus_1'],
         'timbre_tilt': ['timbre_light', 'timbre_moderate']}
PITCH = {'pitch_plus_0.5': .5, 'pitch_minus_0.5': -.5,
         'pitch_plus_1': 1., 'pitch_minus_1': -1.}
TILT = {'timbre_light': (.75, 3.), 'timbre_moderate': (1.5, 6.)}
SEEDS = {'IEMOCAP': 1475, 'MELD': 67137}
REVIEW_STATUSES = {'PASS', 'FAIL', 'UNCERTAIN', 'PENDING'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def write_csv(path, records):
    records = list(records)
    if not records:
        raise ValueError('Refusing an undocumented empty CSV')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))


def safe_sample_id(dataset, did, index):
    return dataset + '_' + hashlib.sha256((str(did) + ':' + str(index)).encode()).hexdigest()[:16]


def confidence_bin(confidence):
    if not np.isfinite(confidence) or not 0 <= confidence <= 1 + 1e-6:
        raise ValueError('Confidence must be a probability')
    index = min(9, int(float(confidence) * 10))
    return f'{index / 10:.1f}-{(index + 1) / 10:.1f}'


def scalar(value):
    return value.item() if hasattr(value, 'item') else value


def checked_probabilities(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Expected finite normalized full softmax vectors')
    return p


def validate_frozen_inputs(args):
    """Machine provenance lock, created only after baseline selection has finished."""
    lock = json.loads(args.baseline_lock.read_text())
    if lock.get('baseline_frozen') is not True or lock.get('dataset') != args.dataset:
        raise RuntimeError('Baseline is not frozen for this dataset; do not open Validation')
    expected = {'dataset_sha256': args.dataset_dir / 'dataset.pt',
                'checkpoint_sha256': args.checkpoint,
                'text_features_sha256': args.text_features,
                'audio_features_sha256': args.audio_features}
    for key, path in expected.items():
        if lock.get(key) != sha256(path):
            raise RuntimeError('Frozen baseline input hash mismatch: ' + key)
    return lock


def load_inputs(args):
    validate_frozen_inputs(args)
    data = torch.load(args.dataset_dir / 'dataset.pt', map_location='cpu')
    text = torch.load(args.text_features, map_location='cpu')
    audio = torch.load(args.audio_features, map_location='cpu')
    if data['dataset'] != args.dataset or set(data['splits']) != {'train', 'validation'}:
        raise ValueError('Expected this dataset and Train/Validation only')
    dialogues = {str(row['dialogue_id']): row for row in data['splits']['validation']}
    for did, dialogue in dialogues.items():
        n = len(dialogue['texts'])
        if tuple(text['validation'][did].shape) != (n, 4, 1024) or tuple(audio['validation'][did].shape) != (n, 768):
            raise ValueError('New-feature shape / dialogue alignment mismatch')
    return data, dialogues, text, audio


def sample_fixed(rows, dataset, count=120):
    """Coverage then balance: no perturbed responses or effectiveness in selection."""
    if len(rows) < count:
        raise ValueError('Insufficient true Validation rows')
    rng = np.random.RandomState(SEEDS[dataset])
    tie = rng.permutation(len(rows))
    dialogue_counts, class_counts, strata_counts = Counter(), Counter(), Counter()
    remaining = set(range(len(rows))); chosen = []
    for _ in range(count):
        def key(index):
            row = rows[index]
            stratum = (int(row['true_label']), int(row['original_correct']), row['confidence_bin'])
            # Every unseen Dialogue takes precedence; subsequent rounds balance
            # per-Dialogue counts while selecting underrepresented observed strata.
            return (dialogue_counts[row['dialogue_id']],
                    class_counts[int(row['true_label'])], strata_counts[stratum], int(tie[index]))
        index = min(remaining, key=key)
        row = dict(rows[index]); chosen.append(row); remaining.remove(index)
        dialogue_counts[row['dialogue_id']] += 1
        class_counts[int(row['true_label'])] += 1
        strata_counts[(int(row['true_label']), int(row['original_correct']), row['confidence_bin'])] += 1
    return chosen


def select(args):
    from downstream import load_model, infer_dialogue
    data, dialogues, text, audio = load_inputs(args)
    if (args.output / 'sample_manifest.csv').exists():
        raise FileExistsError('Sample selection already frozen; reuse it without resampling')
    model = load_model(args.dataset_dir, args.checkpoint, device=args.device)
    rows = []
    for did, dialogue in dialogues.items():
        probs = checked_probabilities(infer_dialogue(model, dialogue, text['validation'][did], audio['validation'][did]))
        for index, p in enumerate(probs):
            label = int(scalar(dialogue['labels'][index])); pred = int(p.argmax())
            row = dict(dataset=args.dataset, seed=SEEDS[args.dataset], dialogue_id=did,
                       utterance_index=index, utterance_id=str(dialogue['utterance_ids'][index]),
                       sample_id=safe_sample_id(args.dataset, did, index),
                       speaker_id=int(torch.as_tensor(dialogue['speaker'][index]).argmax()),
                       true_label=label, original_prediction=pred, original_correct=int(pred == label),
                       original_confidence=float(p.max()), confidence_bin=confidence_bin(float(p.max())),
                       original_probabilities=json.dumps(p.tolist()),
                       original_text_sha256=hashlib.sha256(dialogue['texts'][index].encode()).hexdigest(),
                       original_audio_missing=bool(audio.get('audio_missing_mask', {}).get('validation', {}).get(did, torch.zeros(len(probs), dtype=torch.bool))[index]))
            rows.append(row)
    assert len(rows) == {'IEMOCAP': 527, 'MELD': 1109}[args.dataset]
    eligible_rows = [row for row in rows if not row['original_audio_missing']]
    selected = sample_fixed(eligible_rows, args.dataset)
    write_csv(args.output / 'original_predictions.csv', rows)
    write_csv(args.output / 'sample_manifest.csv', selected)
    save_json(args.output / 'selection_frozen.json', dict(dataset=args.dataset, seed=SEEDS[args.dataset], samples=120,
              algorithm='Lexicographic minimum selected Dialogue count, selected true-class count, selected (class,correct,confidence-bin) count, seeded random tie rank',
              true_validation_rows=len(rows), selected_dialogues=len({r['dialogue_id'] for r in selected}),
              eligible_audio_available_rows=len(eligible_rows), excluded_missing_audio_targets=len(rows)-len(eligible_rows),
              target_pool_rule='Predeclared audio_available only; unchanged full Dialogue contexts retain all missing rows; no effectiveness filtering',
              class_counts=dict(Counter(str(r['true_label']) for r in selected)),
              correct_counts=dict(Counter(str(r['original_correct']) for r in selected)),
              confidence_bin_counts=dict(Counter(r['confidence_bin'] for r in selected)),
              baseline_lock_sha256=sha256(args.baseline_lock), sample_manifest_sha256=sha256(args.output / 'sample_manifest.csv'),
              sampling_is_stratified=True, overall_sample_metrics_are_not_dataset_population_metrics=True,
              text_audio_share_exact_same_manifest=True, test_accessed=False))


def read_samples(args):
    selection = json.loads((args.output / 'selection_frozen.json').read_text())
    if selection['baseline_lock_sha256'] != sha256(args.baseline_lock) or selection['sample_manifest_sha256'] != sha256(args.output / 'sample_manifest.csv'):
        raise RuntimeError('Frozen selection inputs changed')
    rows = read_csv(args.output / 'sample_manifest.csv')
    if len(rows) != 120 or len({r['sample_id'] for r in rows}) != 120:
        raise ValueError('Expected the same 120 unique selected utterances')
    return rows


def text_template(args):
    from transformers import RobertaTokenizer
    from text_encoder import preprocessing
    data, dialogues, _, _ = load_inputs(args)
    validate_text_encoder_identity(args)
    model_dir = args.text_model / 'best_classifier' if (args.text_model / 'best_classifier').is_dir() else args.text_model
    tokenizer = RobertaTokenizer.from_pretrained(model_dir, local_files_only=True)
    samples = read_samples(args)
    records = []
    for sample in samples:
        did = sample['dialogue_id']; index = int(sample['utterance_index']); dialogue = dialogues[did]
        context = [dict(utterance_index=j, speaker_id=int(torch.as_tensor(dialogue['speaker'][j]).argmax()), text=value)
                   for j, value in enumerate(dialogue['texts'])]
        target_label = int(scalar(dialogue['labels'][index]))
        token_count = len(tokenizer.encode(preprocessing(dialogue['texts'][index], args.dataset), add_special_tokens=True, truncation=False))
        for kind in ['neutral_append', 'semantic_paraphrase']:
            for variant in TYPES[kind]:
                records.append(dict(sample_id=sample['sample_id'], dialogue_id=did, utterance_index=index,
                                    perturbation_type=kind, variant_id=variant,
                                    original_text=dialogue['texts'][index], original_text_sha256=sample['original_text_sha256'],
                                    original_true_label=target_label, original_emotion_name=data['class_names'][target_label],
                                    original_total_tokens=token_count,
                                    maximum_neutral_added_tokens=min(24, max(4, math.floor(.5 * max(1, token_count - 2)))),
                                    dialogue_context=context, processed_text='',
                                    review_status='PENDING', reviewer='', review_method='', review_notes='',
                                    instruction='Write/review using the real dialogue; preserve original meaning/emotion/strength/speaker, do not add emotion words to match the label. Explicitly attribute AI versus human review; no automatic synonym substitution counted as semantic review. Neutral append must retain original prefix and stay within token budget.'))
    path = args.local_dir / 'text_review_template.jsonl'
    if path.exists(): raise FileExistsError('Refuse to overwrite manual text review')
    write_jsonl(path, records)


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.


def transform_waveform(clean, variant, ffmpeg):
    clean = np.asarray(clean, dtype=np.float32)
    if clean.ndim != 1 or not len(clean) or not np.isfinite(clean).all():
        raise ValueError('Expected finite nonempty 16-kHz mono waveform')
    if variant in PITCH:
        semitones = PITCH[variant]; ratio = 2 ** (semitones / 12)
        shifted_sample_clock = int(round(16000 * ratio))
        actual_ratio = shifted_sample_clock / 16000
        filters = f'asetrate={shifted_sample_clock},aresample=16000,atempo={1 / actual_ratio:.12g}'
        command = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'f32le', '-ar', '16000', '-ac', '1',
                   '-i', 'pipe:0', '-af', filters, '-f', 'f32le', '-ar', '16000', '-ac', '1', 'pipe:1']
        output = subprocess.run(command, input=clean.astype('<f4').tobytes(), capture_output=True, check=True)
        changed = np.frombuffer(output.stdout, dtype='<f4').copy()
        parameters = dict(algorithm='FFmpeg asetrate + resampling + reciprocal WSOLA atempo', semitones=semitones,
                          requested_ratio=ratio, shifted_sample_clock=shifted_sample_clock,
                          actual_semitones=12 * math.log2(actual_ratio), actual_ratio=actual_ratio,
                          filter=filters, ffmpeg_sha256=sha256(ffmpeg),
                          ffmpeg_version=subprocess.check_output([ffmpeg, '-version'], text=True).splitlines()[0])
    elif variant in TILT:
        slope, cap = TILT[variant]
        frequencies = np.fft.rfftfreq(len(clean), d=1 / 16000)
        gain_db = np.clip(slope * np.log2(np.maximum(frequencies, 80) / 1000), -cap, cap)
        changed = np.fft.irfft(np.fft.rfft(clean) * 10 ** (gain_db / 20), n=len(clean)).astype(np.float32)
        parameters = dict(algorithm='Zero-phase whole-utterance FFT spectral tilt', db_per_octave=slope,
                          reference_hz=1000, low_frequency_floor_hz=80, gain_cap_db=cap)
    else:
        raise ValueError('Unknown fixed audio variant')
    raw_length = len(changed)
    if raw_length == 0 or not np.isfinite(changed).all():
        raise ValueError('Non-finite or empty transformed waveform')
    if len(changed) < len(clean): changed = np.pad(changed, (0, len(clean) - len(changed)))
    changed = changed[:len(clean)]
    gain = rms(clean) / max(rms(changed), 1e-12)
    changed = changed * gain
    peak = float(np.max(np.abs(changed)))
    peak_attenuation = min(1., .999 / max(peak, 1e-12))
    changed = (changed * peak_attenuation).astype(np.float32)
    rms_db = 20 * math.log10(max(rms(changed), 1e-12) / max(rms(clean), 1e-12))
    qc = dict(sample_rate=16000, channels=1, clean_samples=len(clean), raw_transformed_samples=raw_length,
              final_samples=len(changed), duration_seconds=len(clean) / 16000,
              raw_length_relative_difference=abs(raw_length - len(clean)) / len(clean),
              clean_rms=rms(clean), transformed_rms=rms(changed), rms_change_db=rms_db,
              clean_peak=float(np.max(np.abs(clean))),
              clean_clipping_fraction=float((np.abs(clean) >= .9999).mean()),
              peak_before_attenuation=peak,
              rms_restore_gain=gain, peak_attenuation=peak_attenuation,
              peak=float(np.max(np.abs(changed))), finite=bool(np.isfinite(changed).all()),
              clipping_fraction=float((np.abs(changed) >= .9999).mean()),
              emotion_label_preservation='UNCONFIRMED_REQUIRES_REVIEW')
    qc['automatic_signal_qc_pass'] = bool(qc['finite'] and rms(clean) > 1e-6 and len(changed) == len(clean)
               and abs(rms_db) <= .5 and qc['clipping_fraction'] == 0 and qc['raw_length_relative_difference'] <= .05)
    return changed, parameters, qc


def audio_transform(args):
    from audio_encoder import read_audio_16k
    _, dialogues, _, audio = load_inputs(args)
    samples = read_samples(args); records = []; review = []
    path = args.local_dir / 'audio_transformations.jsonl'
    if path.exists(): raise FileExistsError('Refuse to overwrite fixed audio transformations')
    ffmpeg = args.ffmpeg or shutil.which('ffmpeg') or '/home/yangbin/miniconda3/envs/msa_extract/bin/ffmpeg'
    if not Path(ffmpeg).is_file(): raise FileNotFoundError('Use an existing FFmpeg; do not install an unrecorded replacement')
    for sample in samples:
        did = sample['dialogue_id']; index = int(sample['utterance_index']); sid = sample['sample_id']
        source = dialogues[did]['audio_paths'][index]
        directory = args.local_dir / 'audio' / sid; directory.mkdir(parents=True, exist_ok=True)
        clean = None; clean_error = None
        try:
            missing = bool(audio.get('audio_missing_mask', {}).get('validation', {}).get(did, torch.zeros(len(dialogues[did]['texts']), dtype=torch.bool))[index])
            if missing: raise ValueError('Baseline audio_missing_mask=true; preserve missing modality')
            expected_source_sha = audio['utterance_metadata']['validation'][did][index]['source_sha256']
            if sha256(source) != expected_source_sha:
                raise ValueError('Raw audio bytes differ from the baseline clean feature source')
            clean, metadata = read_audio_16k(source, cache_dir=directory / 'decode_cache')
            clean = np.asarray(clean, dtype=np.float32)
            wavfile.write(directory / 'clean.wav', 16000, clean)
        except Exception as error:
            clean_error = str(error)
        for kind in ['pitch_shift', 'timbre_tilt']:
            for variant in TYPES[kind]:
                row = dict(sample_id=sid, dialogue_id=did, utterance_index=index, perturbation_type=kind,
                           variant_id=variant, review_status='PENDING', reviewer='', review_method='', review_notes='')
                try:
                    if clean_error: raise ValueError(clean_error)
                    changed, parameters, qc = transform_waveform(clean, variant, ffmpeg)
                    destination = directory / (variant + '.wav')
                    wavfile.write(destination, 16000, changed)
                    row.update(status='CREATED', source_path=str(source), source_sha256=sha256(source),
                               clean_path=str(directory / 'clean.wav'), clean_sha256=sha256(directory / 'clean.wav'),
                               processed_path=str(destination), processed_sha256=sha256(destination),
                               parameters=parameters, signal_qc=qc, decoding_metadata=metadata)
                except Exception as error:
                    row.update(status='FAILED', error=str(error), signal_qc={'automatic_signal_qc_pass': False})
                records.append(row)
                review_row = {key: row[key] for key in ['sample_id', 'dialogue_id', 'utterance_index',
                               'perturbation_type', 'variant_id', 'review_status', 'reviewer', 'review_method', 'review_notes']}
                review_row.update(clean_path=row.get('clean_path', ''), processed_path=row.get('processed_path', ''),
                                  original_text=dialogues[did]['texts'][index],
                                  automatic_signal_qc_pass=row['signal_qc']['automatic_signal_qc_pass'],
                                  label_preservation_instructions='Listen to clean/changed in context; automatic signal QC does not establish retained emotion or intensity.')
                review.append(review_row)
    write_jsonl(path, records)
    review_path = args.local_dir / 'audio_label_review.csv'
    if not review_path.exists(): write_csv(review_path, review)


def review_pass(row, modality):
    methods = (['AI_CONTEXT_SEMANTIC_REVIEW', 'HUMAN_CONTEXT_REVIEW'] if modality == 'text'
               else ['HUMAN_LISTENING_REVIEW', 'EXPERT_AUDIO_REVIEW'])
    return (row.get('review_status', '').upper() == 'PASS' and bool(row.get('reviewer', '').strip())
            and row.get('review_method') in methods and bool(row.get('review_notes', '').strip()))


def text_qc(original, changed, kind, tokenizer, dataset):
    from text_encoder import preprocessing
    original_processed = preprocessing(original, dataset); processed = preprocessing(changed, dataset)
    clean_ids = tokenizer.encode(original_processed, add_special_tokens=True, truncation=False)
    ids = tokenizer.encode(processed, add_special_tokens=True, truncation=False)
    qc = dict(nonempty=bool(processed.strip()), changed=processed != original_processed,
              token_ids_changed=ids != clean_ids,
              original_token_count=len(clean_ids), changed_token_count=len(ids), no_truncation=len(ids) <= 512,
              original_text_preserved=True, added_token_count=None, maximum_added_tokens=None)
    if kind == 'neutral_append':
        limit = min(24, max(4, math.floor(.5 * max(1, len(clean_ids) - 2))))
        added = len(ids) - len(clean_ids)
        qc.update(original_text_preserved=changed.startswith(original) and ids[:len(clean_ids)-1] == clean_ids[:-1],
                  added_token_count=added, maximum_added_tokens=limit,
                  controlled_addition_length=0 < added <= limit)
    qc['automatic_text_qc_pass'] = all(qc[key] for key in ['nonempty', 'changed', 'token_ids_changed', 'no_truncation', 'original_text_preserved']) and qc.get('controlled_addition_length', True)
    return qc


def assessed_text_reviews(args, dialogues, tokenizer):
    """Input-only validation, performed before inspecting any perturbed predictions."""
    from text_encoder import preprocessing
    samples = {row['sample_id']: row for row in read_samples(args)}
    variants = read_jsonl(args.text_review)
    expected = {(sid, variant) for sid in samples for kind in ['neutral_append', 'semantic_paraphrase'] for variant in TYPES[kind]}
    keys = [(row['sample_id'], row['variant_id']) for row in variants]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Reviewed text must retain every planned sample/variant, even failed/uncertain cases')
    assessments = {}; duplicate_groups = {}
    for row in variants:
        if row.get('review_status', 'PENDING').upper() not in REVIEW_STATUSES:
            raise ValueError('Unknown text review status; retain PASS/FAIL/UNCERTAIN/PENDING explicitly')
        sample = samples[row['sample_id']]; did = sample['dialogue_id']; index = int(sample['utterance_index'])
        original = dialogues[did]['texts'][index]; changed = row.get('processed_text', ''); kind = row['perturbation_type']
        if kind not in ['neutral_append', 'semantic_paraphrase'] or row['variant_id'] not in TYPES[kind]:
            raise ValueError('Unexpected text perturbation family')
        if str(row['dialogue_id']) != did or int(row['utterance_index']) != index:
            raise ValueError('Text review Dialogue/Utterance no longer matches frozen selection')
        if row.get('original_text_sha256') != hashlib.sha256(original.encode()).hexdigest():
            raise ValueError('Text review is based on a different original utterance')
        qc = text_qc(original, changed, kind, tokenizer, args.dataset)
        tokens = tuple(tokenizer.encode(preprocessing(changed, args.dataset), add_special_tokens=True, truncation=False))
        group_key = (row['sample_id'], kind, tokens)
        duplicate_groups.setdefault(group_key, []).append((row['sample_id'], row['variant_id']))
        assessments[(row['sample_id'], row['variant_id'])] = qc
    for keys in duplicate_groups.values():
        if len(keys) > 1:
            for key in keys:
                assessments[key].update(duplicate_within_family=True, automatic_text_qc_pass=False)
    return variants, assessments


def validate_text_reviews(args):
    from transformers import RobertaTokenizer
    _, dialogues, _, _ = load_inputs(args)
    validate_text_encoder_identity(args)
    model_dir = args.text_model / 'best_classifier' if (args.text_model / 'best_classifier').is_dir() else args.text_model
    tokenizer = RobertaTokenizer.from_pretrained(model_dir, local_files_only=True)
    variants, assessments = assessed_text_reviews(args, dialogues, tokenizer)
    records = []
    for row in variants:
        qc = assessments[(row['sample_id'], row['variant_id'])]
        records.append(dict(sample_id=row['sample_id'], perturbation_type=row['perturbation_type'],
             variant_id=row['variant_id'], review_status=row.get('review_status', 'PENDING'),
             review_method=row.get('review_method', ''), reviewer=row.get('reviewer', ''),
             attributed_semantic_review_pass=review_pass(row, 'text'), **qc))
    write_csv(args.local_dir / 'text_review_input_qc.csv', records)
    save_json(args.local_dir / 'text_review_input_qc.json', dict(total=len(records),
        automatic_qc_pass=sum(r['automatic_text_qc_pass'] for r in records),
        attributed_review_pass=sum(r['attributed_semantic_review_pass'] for r in records),
        no_perturbed_model_predictions_examined=True, no_hyperparameter_changes=True,
        review_input_sha256=sha256(args.text_review)))


def assert_target_only(original_text, original_audio, altered_text, altered_audio, index, modality):
    indices = [j for j in range(len(original_text)) if j != index]
    if not torch.equal(original_text[indices], altered_text[indices]) or not torch.equal(original_audio[indices], altered_audio[indices]):
        raise AssertionError('Another utterance feature changed')
    if modality == 'text' and not torch.equal(original_audio, altered_audio):
        raise AssertionError('Audio changed during text perturbation')
    if modality == 'audio' and not torch.equal(original_text, altered_text):
        raise AssertionError('Text changed during audio perturbation')


def validate_text_encoder_identity(args):
    """Clean cache and perturbed extraction must use the same selected encoder/tokenizer."""
    metadata = json.loads(args.text_features.with_suffix('.manifest.json').read_text())
    model_dir = args.text_model / 'best_classifier' if (args.text_model / 'best_classifier').is_dir() else args.text_model
    weight_file = model_dir / 'model.safetensors'
    if metadata['dataset'] != args.dataset or metadata['feature_file_sha256'] != sha256(args.text_features):
        raise ValueError('Clean text feature manifest has a different dataset/hash')
    if metadata['model_sha256'] != sha256(weight_file):
        raise ValueError('Perturbed and clean text extraction must use identical actual weights')
    for name, digest in metadata['tokenizer_hashes'].items():
        if sha256(model_dir / name) != digest:
            raise ValueError('Perturbed tokenizer differs from clean extraction: ' + name)
    return metadata['model_sha256']


def clean_reextraction_control(model, dialogue, original_text, original_audio, index, modality, reextracted, original_probabilities):
    """Fixed numerical negative control; never retune from pilot responses."""
    from downstream import infer_dialogue
    clean = original_text[index] if modality == 'text' else original_audio[index]
    reextracted = torch.as_tensor(reextracted, dtype=torch.float32)
    difference = (clean - reextracted).double().numpy()
    close = bool(torch.allclose(clean, reextracted, atol=1e-4, rtol=1e-4))
    altered_text, altered_audio = original_text.clone(), original_audio.clone()
    if modality == 'text': altered_text[index] = reextracted
    else: altered_audio[index] = reextracted
    assert_target_only(original_text, original_audio, altered_text, altered_audio, index, modality)
    probs = checked_probabilities(infer_dialogue(model, dialogue, altered_text, altered_audio))[index]
    original = np.asarray(json.loads(original_probabilities), dtype=np.float64)
    return dict(clean_feature_allclose=close, clean_feature_atol=1e-4, clean_feature_rtol=1e-4,
                clean_feature_max_abs=float(np.max(np.abs(difference))), clean_feature_mae=float(np.mean(np.abs(difference))),
                clean_feature_rmse=float(np.sqrt(np.mean(difference*difference))),
                clean_probability_max_abs=float(np.max(np.abs(probs-original))),
                clean_prediction_unchanged=bool(probs.argmax()==original.argmax()),
                clean_probability_allclose=bool(np.allclose(probs,original,atol=1e-4,rtol=1e-4)))


def infer(args):
    from downstream import load_model, infer_dialogue
    if (args.output / ('perturbed_predictions_' + args.modality + '.csv')).exists():
        raise FileExistsError('Inference file already exists; never silently overwrite completed conditions')
    data, dialogues, text, audio = load_inputs(args)
    samples = read_samples(args); lookup = {row['sample_id']: row for row in samples}
    model = load_model(args.dataset_dir, args.checkpoint, device=args.device)
    records, transformations, controls = [], [], {}
    modalities = ['text', 'audio'] if args.modality == 'both' else [args.modality]
    for modality in modalities:
        if modality == 'text':
            from text_encoder import load_encoder, extract_texts
            if not args.text_review: raise ValueError('Text inference requires explicit reviewed JSONL')
            text_encoder_sha256 = validate_text_encoder_identity(args)
            encoder, tokenizer = load_encoder(args.text_model, device=args.device)
            variants, text_assessments = assessed_text_reviews(args, dialogues, tokenizer)
            extractor_hash = sha256(args.text_review)
        else:
            from audio_encoder import load_audio_encoder, extract_waveform, read_audio_16k
            variants = read_jsonl(args.local_dir / 'audio_transformations.jsonl')
            reviewed = read_csv(args.audio_review or args.local_dir / 'audio_label_review.csv')
            review_lookup = {(r['sample_id'], r['variant_id']): r for r in reviewed}
            if len(review_lookup) != len(reviewed): raise ValueError('Duplicate/conflicting audio label reviews')
            for review in reviewed:
                if review.get('review_status', 'PENDING').upper() not in REVIEW_STATUSES:
                    raise ValueError('Unknown audio review status')
                sample = lookup.get(review['sample_id'])
                if sample is None or str(review['dialogue_id']) != sample['dialogue_id'] or int(review['utterance_index']) != int(sample['utterance_index']):
                    raise ValueError('Audio review no longer matches the same frozen target')
            encoder = load_audio_encoder(device=args.device)
            for identity_key in ['protocol_hash', 'checkpoint_manifest', 'encoder_code_sha256']:
                if audio['metadata'][identity_key] != encoder.metadata[identity_key]:
                    raise ValueError('Perturbed audio encoder differs from clean extraction: ' + identity_key)
            extractor_hash = sha256(args.audio_review or args.local_dir / 'audio_label_review.csv')
        expected = {(sid, v) for sid in lookup for kind in TYPES
                    if (kind in ['neutral_append', 'semantic_paraphrase']) == (modality == 'text') for v in TYPES[kind]}
        observed = [(r['sample_id'], r['variant_id']) for r in variants]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError('Every planned condition must remain present exactly once, including failures')
        for variant in variants:
            sample = lookup[variant['sample_id']]; sid = sample['sample_id']
            did = sample['dialogue_id']; index = int(sample['utterance_index']); dialogue = dialogues[did]
            kind = variant['perturbation_type']; version = variant['variant_id']
            base = dict(dataset=args.dataset, seed=SEEDS[args.dataset], sample_id=sid, dialogue_id=did,
                        utterance_index=index, utterance_id=sample['utterance_id'], speaker_id=int(sample['speaker_id']),
                        modality=modality, perturbation_type=kind, variant_id=version,
                        planned_k=len(TYPES[kind]), true_label=int(sample['true_label']),
                        original_prediction=int(sample['original_prediction']), original_correct=int(sample['original_correct']),
                        original_confidence=float(sample['original_confidence']), confidence_bin=sample['confidence_bin'],
                        original_probabilities=sample['original_probabilities'], review_input_sha256=extractor_hash)
            base['perturbation_strength'] = (f'{PITCH[version]:+g} semitones' if version in PITCH else
                     f'{TILT[version][0]:g} dB/octave, cap {TILT[version][1]:g} dB' if version in TILT else
                     'short neutral append' if kind == 'neutral_append' else 'reviewed semantic-preserving paraphrase')
            if modality == 'text': base['text_encoder_sha256'] = text_encoder_sha256
            original_text = torch.as_tensor(text['validation'][did]).clone()
            original_audio = torch.as_tensor(audio['validation'][did]).clone()
            altered_text, altered_audio = original_text.clone(), original_audio.clone()
            context_hash = {key: array_hash(dialogue[key]) for key in ['visual', 'speaker', 'labels']}
            try:
                if kind not in TYPES or version not in TYPES[kind]: raise ValueError('Unknown condition')
                if str(variant['dialogue_id']) != did or int(variant['utterance_index']) != index:
                    raise ValueError('Review row no longer matches frozen sample')
                if modality == 'text':
                    original = dialogue['texts'][index]
                    if variant.get('original_text_sha256') != hashlib.sha256(original.encode()).hexdigest():
                        raise ValueError('Manual text review targets a different clean utterance')
                    changed = variant.get('processed_text', '')
                    qc = text_assessments[(sid, version)]
                    if not qc['automatic_text_qc_pass']: raise ValueError('Automatic text/token QC failed: ' + json.dumps(qc))
                    if (sid,modality) not in controls:
                        reextracted=extract_texts(encoder,tokenizer,[original],dataset=args.dataset,batch_size=1)[0]
                        controls[(sid,modality)]=dict(sample_id=sid,modality=modality,
                           **clean_reextraction_control(model,dialogue,original_text,original_audio,index,modality,reextracted,sample['original_probabilities']))
                    if not all(controls[(sid,modality)][key] for key in ['clean_feature_allclose','clean_probability_allclose','clean_prediction_unchanged']):
                        raise ValueError('Clean re-extraction numerical control failed; do not interpret this as raw-input response')
                    altered_text[index] = extract_texts(encoder, tokenizer, [changed], dataset=args.dataset, batch_size=1)[0]
                    accepted = review_pass(variant, 'text')
                    base.update(label_review_status=variant.get('review_status', 'PENDING'), reviewer=variant.get('reviewer', ''),
                                review_method=variant.get('review_method', ''),
                                original_input_sha256=hashlib.sha256(original.encode()).hexdigest(),
                                processed_input_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                                original_feature_sha256=array_hash(original_text[index]), target_feature_sha256=array_hash(altered_text[index]),
                                quality_control=json.dumps(qc))
                else:
                    review = review_lookup.get((sid, version), {})
                    if variant.get('status') != 'CREATED': raise ValueError(variant.get('error', 'Audio transformation failed'))
                    if sha256(variant['processed_path']) != variant['processed_sha256']: raise ValueError('Processed audio hash changed')
                    if sha256(variant['source_path']) != variant['source_sha256'] or sha256(variant['clean_path']) != variant['clean_sha256']:
                        raise ValueError('Original/clean audio bytes changed after fixed transformation')
                    missing = bool(audio.get('audio_missing_mask', {}).get('validation', {}).get(did, torch.zeros(len(original_audio), dtype=torch.bool))[index])
                    if missing: raise ValueError('Preserve original missing-audio mask; cannot perturb missing audio')
                    if (sid,modality) not in controls:
                        clean_waveform,_=read_audio_16k(variant['clean_path'])
                        reextracted=extract_waveform(clean_waveform,encoder=encoder)
                        controls[(sid,modality)]=dict(sample_id=sid,modality=modality,
                           **clean_reextraction_control(model,dialogue,original_text,original_audio,index,modality,reextracted,sample['original_probabilities']))
                    if not all(controls[(sid,modality)][key] for key in ['clean_feature_allclose','clean_probability_allclose','clean_prediction_unchanged']):
                        raise ValueError('Clean re-extraction numerical control failed; do not interpret this as raw-input response')
                    waveform, _ = read_audio_16k(variant['processed_path'])
                    altered_audio[index] = extract_waveform(waveform, encoder=encoder)
                    qc = variant['signal_qc']; accepted = bool(qc.get('automatic_signal_qc_pass')) and review_pass(review, 'audio')
                    base.update(label_review_status=review.get('review_status', 'PENDING'), reviewer=review.get('reviewer', ''),
                                review_method=review.get('review_method', ''),
                                original_input_sha256=variant['source_sha256'], processed_input_sha256=variant['processed_sha256'],
                                original_feature_sha256=array_hash(original_audio[index]), target_feature_sha256=array_hash(altered_audio[index]),
                                quality_control=json.dumps(qc), transformation_parameters=json.dumps(variant['parameters']))
                assert_target_only(original_text, original_audio, altered_text, altered_audio, index, modality)
                if {key: array_hash(dialogue[key]) for key in context_hash} != context_hash:
                    raise AssertionError('Dialogue visual/speaker/labels changed')
                probabilities = checked_probabilities(infer_dialogue(model, dialogue, altered_text, altered_audio))
                if probabilities.shape != (len(dialogue['texts']), len(data['class_names'])):
                    raise ValueError('Full-Dialogue probability shape changed')
                assert_target_only(original_text, original_audio, altered_text, altered_audio, index, modality)
                if {key: array_hash(dialogue[key]) for key in context_hash} != context_hash:
                    raise AssertionError('Inference changed Dialogue visual/speaker/labels')
                p = probabilities[index]; pred = int(p.argmax()); label = int(base['true_label'])
                destination = args.local_dir / 'features_and_predictions' / sid / (version + '.pt')
                destination.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(target_feature=altered_text[index] if modality == 'text' else altered_audio[index],
                                full_dialogue_probabilities=torch.from_numpy(probabilities), dialogue_id=did, utterance_index=index), destination)
                base.update(status='SUCCESS', automatic_qc_pass=bool(qc.get('automatic_text_qc_pass', qc.get('automatic_signal_qc_pass', False))),
                            label_preserving_main_eligible=bool(accepted), non_target_features_bit_exact=True,
                            dialogue_context_hashes_unchanged=True, perturbed_prediction=pred,
                            perturbed_max_confidence=float(p.max()),
                            perturbed_original_class_probability=float(p[int(base['original_prediction'])]),
                            perturbed_original_label_probability=float(p[label]), perturbed_probabilities=json.dumps(p.tolist()),
                            artifact_sha256=sha256(destination))
            except Exception as error:
                base.update(status='FAILED', error=str(error), label_preserving_main_eligible=False)
            records.append(base)
            transformations.append({key: value for key, value in base.items() if key not in ['original_probabilities', 'perturbed_probabilities']})
        del encoder
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    # Each mode has its own file, preventing modality-specific execution from overwriting another.
    write_csv(args.output / ('perturbed_predictions_' + args.modality + '.csv'), records)
    write_csv(args.output / ('transformation_manifest_' + args.modality + '.csv'), transformations)
    if controls: write_csv(args.output / ('clean_reextraction_controls_' + args.modality + '.csv'), controls.values())
    save_json(args.output / ('inference_' + args.modality + '.json'), dict(planned=len(records),
              successful=sum(r['status']=='SUCCESS' for r in records), failed=sum(r['status']!='SUCCESS' for r in records),
              main_eligible=sum(r.get('label_preserving_main_eligible',False) for r in records),
              baseline_lock_sha256=sha256(args.baseline_lock), sample_manifest_sha256=sha256(args.output / 'sample_manifest.csv'),
              code_sha256=sha256(__file__), test_accessed=False, model_weights_updated=False))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['select', 'text-template', 'validate-text', 'audio-transform', 'infer'])
    p.add_argument('--dataset', required=True, choices=SEEDS)
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--text-features', type=Path, required=True)
    p.add_argument('--audio-features', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--baseline-lock', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--local-dir', type=Path, required=True)
    p.add_argument('--text-model', type=Path)
    p.add_argument('--text-review', type=Path)
    p.add_argument('--audio-review', type=Path)
    p.add_argument('--modality', choices=['text', 'audio', 'both'], default='both')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--ffmpeg')
    return p


def main():
    args = parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True); args.local_dir.mkdir(parents=True, exist_ok=True)
    {'select': select, 'text-template': text_template, 'validate-text': validate_text_reviews,
     'audio-transform': audio_transform, 'infer': infer}[args.mode](args)


if __name__ == '__main__': main()
