#!/usr/bin/env python3
"""Frozen MELD 24-pair semantic K5 extraction/inference, with explicit two-runtime boundary.

No training, encoder fallback, Test access, rematching or condition deletion exists.
Run encode in TEXT_repro; run infer in multimodalerc310. Full texts/features stay local.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PARENT = Path(__file__).resolve().parents[2] / '20261120T143131Z'
SOURCE_ENGINE = PARENT.parent / '20261120T170835Z/scripts/run_matched_text.py'
sys.path.insert(0, str(PARENT / 'scripts'))
import raw_pilot as shared

FAMILIES = {'semantic_paraphrase': ('paraphrase_a', 'paraphrase_b',
                                     'paraphrase_c', 'paraphrase_d', 'paraphrase_e')}
FROZEN_AB = frozenset(('paraphrase_a', 'paraphrase_b'))
EXPECTED_PAIRS = 24
EXPECTED_TARGETS = 48
PLANNED_K = 5
SCHEMA = 'MELD_MATCHED_SEMANTIC_K5_FEATURES_V1'


def now():
    return datetime.now(timezone.utc).isoformat()


def text_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def json_hash(value):
    return text_sha(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').split('\n') if line.strip()]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')


def save_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    if temp.exists():
        raise FileExistsError(temp)
    torch.save(value, temp)
    temp.rename(path)


def write_csv(path, rows):
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise ValueError('Undocumented empty output')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def paths(run):
    return SimpleNamespace(dataset='MELD', dataset_dir=PARENT/'local_only/data/MELD',
        text_features=PARENT/'local_only/features/MELD/text_features.pt',
        audio_features=PARENT/'local_only/features/MELD/audio_features.pt',
        checkpoint=PARENT/'local_only/data/MELD/runs/cmerc/best.pt',
        baseline_lock=PARENT/'configs/BASELINE_LOCK_MELD.json',
        text_model=PARENT/'local_only/text_models/MELD',
        cohort=run/'results/matched_sample_manifest.csv',
        review=run/'local_only/matched_text_reviewed.jsonl',
        protocol=run/'configs/EXPLORATORY_PROTOCOL.json')


def bindings(args):
    root = args.text_model/'best_classifier'
    files = [args.dataset_dir/'dataset.pt', args.text_features, args.audio_features,
        args.checkpoint, args.baseline_lock, args.text_features.with_suffix('.manifest.json'),
        args.text_model/'best_model_lock.json', args.cohort, args.review, args.protocol,
        Path(__file__), SOURCE_ENGINE, PARENT/'scripts/raw_pilot.py', PARENT/'scripts/text_encoder.py',
        PARENT/'scripts/downstream.py', PARENT/'configs/PLAN_B_PROTOCOL.json',
        PARENT/'results/pilot/MELD/original_predictions.csv']
    files += sorted(p for p in root.iterdir() if p.is_file())
    files += sorted((PARENT/'scripts/core_snapshot').glob('*.py'))
    return {str(p.resolve()): shared.sha256(p) for p in files}


def check_bindings(expected):
    changed = [p for p, digest in expected.items() if shared.sha256(p) != digest]
    if changed:
        raise RuntimeError('Frozen inputs/code changed: '+repr(changed))


def fingerprint(model):
    tensor_hashes = {name: shared.array_hash(value) for name, value in sorted(model.state_dict().items())}
    bn = {name: value for name, value in tensor_hashes.items()
          if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    return dict(state_sha256=json_hash(tensor_hashes), bn_sha256=json_hash(bn),
                tensor_count=len(tensor_hashes), tensor_hashes=tensor_hashes)


def ai_review_pass(row):
    """Attributed AI/PASS scope; never describes it as human confirmation."""
    return (row.get('review_status', '').upper() == 'PASS'
            and row.get('review_method') in {'AI_CONTEXT_SEMANTIC_REVIEW',
                                             'INDEPENDENT_AI_CONTEXT_QA'}
            and bool(row.get('reviewer', '').strip())
            and bool(row.get('review_notes', '').strip()))


def apply_duplicate_qc(rows, assessments):
    """Input-token-only fixed-menu collision policy, before any extraction.

    A collision between frozen a/b invalidates both a/b. New c/d/e must be
    distinct from *every* other version. They fail if they collide with any
    frozen version, including a frozen version already failing another check.
    All colliding newcomers fail, with no regeneration or reduced effective K.
    A new collision never changes an otherwise valid frozen a/b QC decision.
    """
    groups = {}
    for row in rows:
        key = (row['sample_id'], row['variant_id'])
        qc, tokens = assessments[key]
        qc['duplicate_rule'] = 'FROZEN_AB_PRIORITY_NEW_COLLISIONS_FAIL'
        qc['duplicate_within_family'] = False
        qc['same_tokens_other_variants'] = []
        groups.setdefault((row['sample_id'], row['perturbation_type'],
                           tuple(tokens)), []).append(key)
    for keys in groups.values():
        frozen = [key for key in keys if key[1] in FROZEN_AB]
        for key in keys:
            qc = assessments[key][0]
            qc['same_tokens_other_variants'] = sorted(k[1] for k in keys if k != key)
            fails = len(keys) > 1 and (key[1] not in FROZEN_AB or len(frozen) > 1)
            if fails:
                qc.update(duplicate_within_family=True, automatic_text_qc_pass=False)


def validate_protocol(protocol):
    if protocol['dataset'] != 'MELD' or protocol['model_seed'] != 67137 or protocol['test_access']:
        raise ValueError('Wrong frozen protocol identity')
    if (protocol['matching']['expected_targets'] != EXPECTED_TARGETS
            or protocol['matching']['expected_pairs'] != EXPECTED_PAIRS):
        raise ValueError('Frozen K5 24-pair/48-target count changed')
    if (protocol['perturbations']['semantic_paraphrase'] != list(FAMILIES['semantic_paraphrase'])
            or protocol['perturbations']['k_per_family'] != PLANNED_K):
        raise ValueError('Frozen K5 semantic variant menu changed')


def require_all_clean_controls(controls):
    gates = ('clean_feature_allclose', 'clean_probability_allclose',
             'original_probability_allclose', 'clean_prediction_unchanged')
    if len(controls) != EXPECTED_TARGETS or not all(all(row[g] for g in gates) for row in controls):
        raise RuntimeError('All 48 clean controls must pass before candidate inference')


def validate_cohort(samples):
    if len(samples) != EXPECTED_TARGETS:
        raise ValueError('Frozen K5 24-pair/48-target count changed')
    pairs = {}
    for sample in samples:
        pairs.setdefault(sample['pair_id'], []).append(sample)
    if len(pairs) != EXPECTED_PAIRS or len({s['dialogue_id'] for s in samples}) != EXPECTED_PAIRS:
        raise ValueError('Expected one fixed pair in each of 24 Dialogues')
    correctness_flags = {'true': True, '1': True, 'false': False, '0': False}
    for members in pairs.values():
        correctness = {correctness_flags.get(str(s['original_correct']).strip().lower()) for s in members}
        if (len(members) != 2 or len({s['dialogue_id'] for s in members}) != 1
                or correctness != {True, False}):
            raise ValueError('Pair must contain original correct/error targets within one Dialogue')


def validate_reviews(samples, rows, dialogues):
    lookup = {r['sample_id']: r for r in samples}
    if len(lookup) != len(samples):
        raise ValueError('Duplicate cohort target')
    expected = {(sid, kind, version) for sid in lookup for kind, versions in FAMILIES.items() for version in versions}
    keys = [(r['sample_id'], r['perturbation_type'], r['variant_id']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Must retain all five fixed semantic variants per target exactly once')
    for r in rows:
        sample = lookup[r['sample_id']]
        did, index = sample['dialogue_id'], int(sample['utterance_index'])
        original = dialogues[did]['texts'][index]
        if str(r['dialogue_id']) != did or int(r['utterance_index']) != index:
            raise ValueError('Reviewed target ID changed')
        if r['original_text'] != original or r['original_text_sha256'] != text_sha(original):
            raise ValueError('Original reviewed text differs from frozen source')
        if not isinstance(r.get('processed_text'), str):
            raise ValueError('Missing candidate string; FAIL/UNCERTAIN must also be retained')
        if r.get('review_status', '').upper() not in shared.REVIEW_STATUSES:
            raise ValueError('Unknown review status')
    return lookup


def prepare(run):
    args = paths(run)
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    shared.validate_frozen_inputs(args)
    shared.validate_text_encoder_identity(args)
    frozen = bindings(args)
    data, dialogues, text, audio = shared.load_inputs(args)
    samples = shared.read_csv(args.cohort)
    validate_cohort(samples)
    reference = {r['sample_id']: r for r in shared.read_csv(PARENT/'results/pilot/MELD/original_predictions.csv')}
    for sample in samples:
        if sample['sample_id'] not in reference:
            raise ValueError('Cohort target absent from frozen full Validation predictions')
        ref = reference[sample['sample_id']]
        for key, value in ref.items():
            if sample.get(key) != value:
                raise ValueError('Original prediction/source metadata changed: '+key)
        did, index = sample['dialogue_id'], int(sample['utterance_index'])
        d = dialogues[did]
        if str(d['utterance_ids'][index]) != sample['utterance_id']:
            raise ValueError('Utterance ID/order mismatch')
        if int(d['labels'][index]) != int(sample['true_label']) or int(d['speaker'][index].argmax()) != int(sample['speaker_id']):
            raise ValueError('Label/speaker mismatch')
        if text_sha(d['texts'][index]) != sample['original_text_sha256']:
            raise ValueError('Raw original text mismatch')
    rows = read_jsonl(args.review)
    lookup = validate_reviews(samples, rows, dialogues)
    check_bindings(frozen)
    return args, data, dialogues, text, audio, samples, rows, lookup, frozen


def source_snapshot(dialogues, text, audio):
    return {did: dict(text=shared.array_hash(text['validation'][did]),
                      audio=shared.array_hash(audio['validation'][did]),
                      visual=shared.array_hash(d['visual']),speaker=shared.array_hash(d['speaker']),
                      labels=shared.array_hash(d['labels']),
                      order=json_hash(dict(texts=d['texts'],utterance_ids=d['utterance_ids'])))
            for did,d in dialogues.items()}


def feature_record(row, feature, tokens, original_feature):
    if feature.shape != (4, 1024) or feature.dtype != torch.float32 or feature.device.type != 'cpu' or not torch.isfinite(feature).all():
        raise ValueError('Expected finite CPU FP32 four-layer feature')
    return dict(row, status='EXTRACTED', feature=feature.contiguous(),
        target_feature_sha256=shared.array_hash(feature),
        original_feature_sha256=shared.array_hash(original_feature),
        actual_token_ids=tokens, actual_token_ids_sha256=json_hash(tokens))


def encode(run, device):
    runtime = shared.require_runtime('encoder')
    args, data, dialogues, text, audio, samples, rows, lookup, frozen = prepare(run)
    destination = run/'local_only/encoder_features.pt'
    if destination.exists() or destination.with_suffix('.metadata.json').exists():
        raise FileExistsError('Preserve existing encoded features')
    import text_encoder as encoder
    model, tokenizer = encoder.load_encoder(args.text_model, device=device)
    model.requires_grad_(False).eval()
    before = fingerprint(model)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('Encoder not frozen')
    # Input-only QC happens before the first candidate feature or perturbed probability.
    assessments = {}
    for r in rows:
        key = (r['sample_id'], r['variant_id'])
        qc = shared.text_qc(r['original_text'], r['processed_text'], r['perturbation_type'], tokenizer, 'MELD')
        tokens, _ = encoder.tokenize(tokenizer, [r['processed_text']], 'MELD')
        full = tokenizer.encode(encoder.preprocessing(r['processed_text'], 'MELD'), add_special_tokens=True, truncation=False)
        qc['actual_token_ids_are_untruncated'] = full == tokens[0]
        qc['automatic_text_qc_pass'] &= qc['actual_token_ids_are_untruncated']
        assessments[key] = (qc, tokens[0])
    apply_duplicate_qc(rows, assessments)
    # Preserve failed-QC versions as explicit failed records, never regenerate or reduce K.
    controls, variants = {}, []
    for n, sample in enumerate(samples):
        sid, did, index = sample['sample_id'], sample['dialogue_id'], int(sample['utterance_index'])
        original = dialogues[did]['texts'][index]
        tokens, _ = encoder.tokenize(tokenizer, [original], 'MELD')
        feature = encoder.extract_texts(model, tokenizer, [original], dataset='MELD', batch_size=1)[0]
        controls[sid] = feature_record(dict(sample_id=sid, dialogue_id=did, utterance_index=index,
            original_input_sha256=text_sha(original)), feature, tokens[0], text['validation'][did][index])
        if n % 25 == 0:
            print(f'clean {n+1}/{len(samples)}', flush=True)
    valid_rows = []
    for r in rows:
        qc, tokens = assessments[(r['sample_id'], r['variant_id'])]
        sample = lookup[r['sample_id']]
        did, index = sample['dialogue_id'], int(sample['utterance_index'])
        base = dict(sample_id=r['sample_id'], dialogue_id=did, utterance_index=index,
            modality='text', perturbation_type=r['perturbation_type'], variant_id=r['variant_id'],
            label_review_status=r['review_status'].upper(), reviewer=r.get('reviewer', ''),
            review_method=r.get('review_method', ''), review_notes=r.get('review_notes', ''),
            review_input_sha256=shared.sha256(args.review), quality_control=qc,
            automatic_qc_pass=bool(qc['automatic_text_qc_pass']),
            label_preserving_main_eligible=bool(qc['automatic_text_qc_pass']) and ai_review_pass(r),
            original_input_sha256=r['original_text_sha256'], processed_input_sha256=text_sha(r['processed_text']),
            actual_token_ids=tokens, actual_token_ids_sha256=json_hash(tokens),
            original_feature_sha256=shared.array_hash(text['validation'][did][index]))
        if not qc['automatic_text_qc_pass']:
            variants.append(dict(base, status='FAILED_QC', error='Fixed automatic input QC failed'))
        else:
            variants.append(base)
            valid_rows.append((len(variants)-1, r))
    for start in range(0, len(valid_rows), 4):
        chunk = valid_rows[start:start+4]
        features = encoder.extract_texts(model, tokenizer, [r['processed_text'] for _, r in chunk], dataset='MELD', batch_size=4)
        for (pos, r), feature in zip(chunk, features):
            sample = lookup[r['sample_id']]
            original = text['validation'][sample['dialogue_id']][int(sample['utterance_index'])]
            variants[pos] = feature_record(variants[pos], feature, variants[pos]['actual_token_ids'], original)
        if start % 100 == 0:
            print(f'variants {min(start+4,len(valid_rows))}/{len(valid_rows)}', flush=True)
    after = fingerprint(model)
    if before != after:
        raise RuntimeError('Encoder state/BN changed')
    check_bindings(frozen)
    metadata = dict(schema=SCHEMA, dataset='MELD', seed=67137, runtime=runtime, created_at=now(),
        bindings=frozen, encoder_before=before, encoder_after=after, all_parameters_frozen=True,
        no_grad=True, dtype='float32', layer_order=[-1,-2,-3,-4], pooling='BOS', test_accessed=False,
        clean_count=len(controls), variant_count=len(variants), planned_k=PLANNED_K,
        frozen_versions=sorted(FROZEN_AB), duplicate_rule='FROZEN_AB_PRIORITY_NEW_COLLISIONS_FAIL',
        status_counts=dict(Counter(r.get('status') for r in variants)))
    save_torch(destination, dict(schema=SCHEMA, metadata=metadata, clean_controls=controls, variants=variants))
    file_sha = shared.sha256(destination)
    save_json(destination.with_suffix('.metadata.json'), dict(metadata, feature_file_sha256=file_sha))
    save_json(run/'provenance/text_engine_encode.json', dict(metadata, feature_file=str(destination), feature_file_sha256=file_sha))
    print(json.dumps(dict(status='COMPLETE',stage='encode',clean=len(controls),variants=len(variants),sha256=file_sha)), flush=True)


def check_cached_record(record):
    if json_hash(record['actual_token_ids']) != record['actual_token_ids_sha256']:
        raise ValueError('Actual Token ID integrity failure')
    tokens = record['actual_token_ids']
    if not tokens or tokens[0] != 0 or tokens[-1] != 2 or len(tokens) > 512:
        raise ValueError('Invalid special tokens/length')
    if record['status'] == 'EXTRACTED':
        feature = record['feature']
        if not isinstance(feature, torch.Tensor) or feature.dtype != torch.float32 or feature.device.type != 'cpu' or feature.shape != (4,1024) or not torch.isfinite(feature).all():
            raise ValueError('Invalid cached feature')
        if shared.array_hash(feature) != record['target_feature_sha256']:
            raise ValueError('Cached feature checksum failure')
    elif record['status'] != 'FAILED_QC':
        raise ValueError('Unexpected extraction status')


def infer(run, device):
    runtime = shared.require_runtime('cmerc')
    args, data, dialogues, text, audio, samples, rows, lookup, frozen = prepare(run)
    output = run/'results/perturbed_predictions.csv'
    if output.exists() or (run/'results/clean_controls.csv').exists() or (run/'local_only/dialogue_predictions.pt').exists():
        raise FileExistsError('Preserve completed inference artifacts')
    path = run/'local_only/encoder_features.pt'
    sidecar = json.loads(path.with_suffix('.metadata.json').read_text())
    file_sha = shared.sha256(path)
    if file_sha != sidecar['feature_file_sha256'] or sidecar['bindings'] != frozen:
        raise ValueError('Encoder cache input/hash binding mismatch')
    cache = torch.load(path, map_location='cpu')
    if cache['schema'] != SCHEMA or cache['metadata'] != {k:v for k,v in sidecar.items() if k!='feature_file_sha256'}:
        raise ValueError('Encoder cache schema/metadata mismatch')
    md = cache['metadata']
    if md['encoder_before'] != md['encoder_after'] or not md['all_parameters_frozen'] or md['runtime']['environment'] != 'TEXT_repro':
        raise ValueError('Encoder freeze/runtime provenance failed')
    reviewed = {(r['sample_id'], r['variant_id']): r for r in rows}
    keys = [(r['sample_id'], r['variant_id']) for r in cache['variants']]
    if len(keys) != len(set(keys)) or set(keys) != set(reviewed) or set(cache['clean_controls']) != set(lookup):
        raise ValueError('Incomplete or duplicated fixed candidate/clean cache')
    for r in list(cache['clean_controls'].values()) + cache['variants']:
        check_cached_record(r)
    for r in cache['variants']:
        source = reviewed[(r['sample_id'],r['variant_id'])]
        if r['processed_input_sha256'] != text_sha(source['processed_text']) or r['original_input_sha256'] != source['original_text_sha256']:
            raise ValueError('Cache source string mismatch')
        if r['perturbation_type'] != source['perturbation_type'] or r['label_review_status'] != source['review_status'].upper():
            raise ValueError('Cache family/review mismatch')
    from downstream import load_model, infer_dialogue
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.dataset_dir, args.checkpoint, device=device).eval().requires_grad_(False)
    before = fingerprint(model)
    source_before = source_snapshot(dialogues,text,audio)
    controls, clean_original, local_predictions = [], {}, {}
    for n, sample in enumerate(samples):
        sid, did, index = sample['sample_id'], sample['dialogue_id'], int(sample['utterance_index'])
        dialogue = dialogues[did]
        original_text, original_audio = text['validation'][did], audio['validation'][did]
        clean = cache['clean_controls'][sid]
        if clean['original_input_sha256'] != text_sha(dialogue['texts'][index]) or clean['original_feature_sha256'] != shared.array_hash(original_text[index]):
            raise ValueError('Clean cache source identity mismatch')
        if did not in clean_original:
            clean_original[did] = shared.checked_probabilities(infer_dialogue(model, dialogue, original_text, original_audio))
        ref = np.asarray(json.loads(sample['original_probabilities']), dtype=np.float64)
        original_p = clean_original[did][index]
        changed = original_text.clone(); changed[index] = clean['feature']
        shared.assert_target_only(original_text, original_audio, changed, original_audio, index, 'text')
        p = shared.checked_probabilities(infer_dialogue(model, dialogue, changed, original_audio))[index]
        ok_feature = bool(torch.allclose(clean['feature'], original_text[index], atol=1e-4, rtol=1e-4))
        ok_original = bool(np.allclose(original_p, ref, atol=1e-4, rtol=1e-4))
        ok_probability = bool(np.allclose(p, ref, atol=1e-4, rtol=1e-4))
        ok_class = int(p.argmax()) == int(ref.argmax()) == int(original_p.argmax())
        controls.append(dict(sample_id=sid,dialogue_id=did,utterance_index=index,modality='text',
            clean_feature_allclose=ok_feature, clean_probability_allclose=ok_probability,
            original_probability_allclose=ok_original, clean_prediction_unchanged=ok_class,
            feature_max_abs_error=float((clean['feature']-original_text[index]).abs().max()),
            probability_max_abs_error=float(np.abs(p-ref).max()),
            original_probability_max_abs_error=float(np.abs(original_p-ref).max()),
            feature_atol=1e-4,feature_rtol=1e-4,probability_atol=1e-4,probability_rtol=1e-4))
        if not all((ok_feature,ok_original,ok_probability,ok_class)):
            write_csv(run/'results/clean_controls.csv',controls)
            raise RuntimeError('Clean re-extraction negative control failed; perturbation inference prohibited')
        if n % 50 == 0:
            print(f'controls {n+1}/{len(samples)}',flush=True)
    require_all_clean_controls(controls)
    records = []
    for n, variant in enumerate(cache['variants']):
        sample = lookup[variant['sample_id']]
        sid, did, index = sample['sample_id'], sample['dialogue_id'], int(sample['utterance_index'])
        dialogue = dialogues[did]
        original_text, original_audio = text['validation'][did], audio['validation'][did]
        base = dict(sample, modality='text', perturbation_type=variant['perturbation_type'],
            variant_id=variant['variant_id'], planned_k=PLANNED_K, label_review_status=variant['label_review_status'],
            reviewer=variant['reviewer'], review_method=variant['review_method'],
            quality_control=json.dumps(variant['quality_control'],sort_keys=True),
            automatic_qc_pass=variant['automatic_qc_pass'],
            label_preserving_main_eligible=variant['label_preserving_main_eligible'],
            precomputed_feature_file_sha256=file_sha)
        for k in ['original_input_sha256','processed_input_sha256','original_feature_sha256',
                  'target_feature_sha256','actual_token_ids_sha256','review_input_sha256']:
            if k in variant:
                base[k] = variant[k]
        if variant['status'] != 'EXTRACTED':
            records.append(dict(base,status='FAILED',error=variant.get('error','Fixed input QC failure'),label_preserving_main_eligible=False))
            continue
        context_hash = {key: shared.array_hash(dialogue[key]) for key in ['visual','speaker','labels']}
        order_hash = json_hash(dict(texts=dialogue['texts'],utterance_ids=dialogue['utterance_ids']))
        changed = original_text.clone(); changed[index] = variant['feature']
        shared.assert_target_only(original_text,original_audio,changed,original_audio,index,'text')
        if variant['original_feature_sha256'] != shared.array_hash(original_text[index]):
            raise ValueError('Original feature row mismatch')
        probabilities = shared.checked_probabilities(infer_dialogue(model,dialogue,changed,original_audio))
        if probabilities.shape != (len(dialogue['texts']),7):
            raise ValueError('Unexpected full Dialogue softmax shape')
        shared.assert_target_only(original_text,original_audio,changed,original_audio,index,'text')
        if context_hash != {key:shared.array_hash(dialogue[key]) for key in context_hash} or order_hash != json_hash(dict(texts=dialogue['texts'],utterance_ids=dialogue['utterance_ids'])):
            raise RuntimeError('Dialogue non-target context mutated')
        p = probabilities[index]
        art = dict(dialogue_id=did,utterance_index=index,variant_id=variant['variant_id'],
                   full_dialogue_probabilities=torch.from_numpy(probabilities),
                   target_feature_sha256=variant['target_feature_sha256'],context_hashes=context_hash)
        local_predictions[sid+':'+variant['variant_id']] = art
        records.append(dict(base,status='SUCCESS',perturbed_prediction=int(p.argmax()),
            perturbed_max_confidence=float(p.max()),
            perturbed_original_class_probability=float(p[int(sample['original_prediction'])]),
            perturbed_original_label_probability=float(p[int(sample['true_label'])]),
            perturbed_probabilities=json.dumps(p.tolist()),non_target_features_bit_exact=True,
            dialogue_context_hashes_unchanged=True,
            full_dialogue_probability_sha256=shared.array_hash(probabilities)))
        if n % 100 == 0:
            print(f'inference {n+1}/{len(cache["variants"])}',flush=True)
    after = fingerprint(model)
    check_bindings(frozen)
    source_after = source_snapshot(dialogues,text,audio)
    if before != after or shared.sha256(path) != file_sha or source_before != source_after:
        raise RuntimeError('CMERC/BN/cache/original input tensors changed')
    if len(records) != len(reviewed):
        raise RuntimeError('A fixed condition was lost')
    save_torch(run/'local_only/dialogue_predictions.pt', dict(clean_original=clean_original,variants=local_predictions))
    write_csv(run/'results/clean_controls.csv',controls)
    write_csv(output,records)
    save_json(run/'provenance/text_engine_infer.json',dict(status='COMPLETE',created_at=now(),
        dataset='MELD',seed=67137,runtime=runtime,bindings=frozen,encoder_models_instantiated=False,
        encoder_feature_sha256=file_sha,encoder_before=md['encoder_before'],encoder_after=md['encoder_after'],
        model_before=before,model_after=after,all_parameters_frozen=True,test_accessed=False,
        original_context_before=source_before,original_context_after=source_after,
        original_context_bit_exact=source_before==source_after,
        clean_controls=len(controls),clean_controls_passed=len(controls),planned=len(records),
        planned_k=PLANNED_K,expected_pairs=EXPECTED_PAIRS,expected_targets=EXPECTED_TARGETS,
        duplicate_rule='FROZEN_AB_PRIORITY_NEW_COLLISIONS_FAIL',primary_review_scope='AI_PASS_EXPLORATORY',
        statuses=dict(Counter(r['status'] for r in records)),
        output_sha256=shared.sha256(output),dialogue_artifact_sha256=shared.sha256(run/'local_only/dialogue_predictions.pt')))
    print(json.dumps(dict(status='COMPLETE',stage='infer',planned=len(records),statuses=dict(Counter(r['status'] for r in records)))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['encode','infer'])
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--run-dir',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.mode=='encode':
        encode(args.run_dir.resolve(),args.device)
    else:
        infer(args.run_dir.resolve(),args.device)


if __name__=='__main__':
    main()
