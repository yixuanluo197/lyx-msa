#!/usr/bin/env python3
"""Build frozen split ID/media manifests and Train-only audio targets.

Shared historical pickle deserialization reconstructs all containers, but this
script never indexes Test IDs/features/labels or Validation feature vectors.
MELD CSV readers select ID and speaker fields only; label fields are untouched.
"""
import concurrent.futures
import csv
import hashlib
import json
import pickle
import platform
from collections import Counter
from pathlib import Path

import numpy as np

WORKSPACE = Path('/LOCAL/lyx-msa')
OUT = Path(__file__).resolve().parents[1]
XP = WORKSPACE / 'CMERC/experiments/xp_mainline_validation/20261016T050823Z'
IEM = Path('/LOCAL/IEMOCAP_AUDIO')
MELD = Path('/LOCAL/MELD_ASSETS/raw/MELD')
FIELDS = ['row_index', 'dataset', 'split', 'dialogue_id', 'utterance_index',
          'raw_dialogue_id', 'raw_utterance_id', 'speaker_id', 'media_path',
          'media_exists', 'media_sha256']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream, encoding='latin1')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    source_hashes, result = {}, {}
    selected_path = MELD / 'MELD.Features.Models/features/audio_embeddings_feature_selection_emotion.pkl'
    selected_hash = sha256(selected_path)
    assert selected_hash == '0cda206983988ab61de5b30ec32b8ac652e133f84cda4d3d8105a2db8fea363e'
    # No access to slot 1 (Validation) or slot 2 (Test).
    selected_train = load(selected_path)[0]
    source_hashes[str(selected_path)] = selected_hash
    all_rows = []
    for dataset, seed, base_name, dims, n_train, n_val in [
        ('IEMOCAP', 1475, 'IEMOCAP_features.pkl', 1582, 5283, 527),
        ('MELD', 67137, 'MELD_features_raw1.pkl', 1611, 9989, 1109),
    ]:
        cfg_path = XP / f'configs/effective/{dataset}_cmerc_validation_seed{seed}.json'
        cfg = json.loads(cfg_path.read_text())
        assert cfg['Dataset'] == dataset and cfg['seed'] == seed
        cfg_split = cfg['provenance']['split']
        # Only the authorized Train and Validation fields are selected.
        ids = {s: cfg_split[f'{s}_ids'] for s in ['train', 'validation']}
        assert len(set(ids['train'])) == len(ids['train'])
        assert len(set(ids['validation'])) == len(ids['validation'])
        assert set(ids['train']).isdisjoint(ids['validation'])
        base_path = XP / 'code/data' / base_name
        source_hashes[str(cfg_path)] = sha256(cfg_path)
        source_hashes[str(base_path)] = sha256(base_path)
        assert source_hashes[str(base_path)] == cfg['provenance']['data'][f'data/{base_name}']
        base = load(base_path)
        target_rows, groups, split_rows = [], [], {}
        for split in ['train', 'validation']:
            csv_metadata = {}
            if dataset == 'MELD':
                csv_path = MELD / 'MELD.Raw' / ('train_sent_emo.csv' if split == 'train' else 'dev_sent_emo.csv')
                source_hashes[str(csv_path)] = sha256(csv_path)
                with csv_path.open(encoding='utf-8-sig', newline='') as stream:
                    for row in csv.DictReader(stream):
                        key = (int(row['Dialogue_ID']), int(row['Utterance_ID']))
                        assert key not in csv_metadata
                        csv_metadata[key] = row['Speaker']
            rows = []
            for dialogue in ids[split]:
                utterance_ids = base[0][dialogue]
                speakers = base[1][dialogue]
                assert len(utterance_ids) == len(speakers)
                if dataset == 'IEMOCAP' and split == 'train':
                    targets = np.asarray(base[4][dialogue], dtype=np.float64)
                    assert targets.shape == (len(utterance_ids), dims)
                    assert np.isfinite(targets).all()
                if dataset == 'MELD':
                    raw_dialogue = int(dialogue) - (0 if split == 'train' else 1039)
                    assert sorted(u for d, u in csv_metadata if d == raw_dialogue) == list(map(int, utterance_ids))
                for index, uid in enumerate(utterance_ids):
                    if dataset == 'IEMOCAP':
                        raw_dialogue, raw_uid = str(dialogue), str(uid)
                        assert raw_uid.startswith(raw_dialogue + '_')
                        assert raw_uid.rsplit('_', 1)[-1][0] == speakers[index]
                        speaker_id = 0 if speakers[index] == 'M' else 1
                        media = IEM / f'Session{int(str(dialogue)[3:5])}/sentences/wav' / str(dialogue) / f'{uid}.wav'
                        if split == 'train':
                            target_rows.append(targets[index])
                    else:
                        raw_uid = int(uid)
                        assert (raw_dialogue, raw_uid) in csv_metadata
                        speaker = np.asarray(speakers[index])
                        assert speaker.shape == (9,) and speaker.sum() == 1
                        speaker_id = int(speaker.argmax())
                        media_folder = 'train_splits' if split == 'train' else 'dev_splits_complete'
                        media = MELD / 'MELD.Raw' / media_folder / f'dia{raw_dialogue}_utt{raw_uid}.mp4'
                        recovery_path = OUT / 'provenance/train_missing_media_recovery.json'
                        if not media.is_file() and split == 'train' and recovery_path.is_file():
                            recovered = json.loads(recovery_path.read_text())
                            if recovered['adopted'] and (raw_dialogue, raw_uid) == (recovered['raw_dialogue_id'], recovered['raw_utterance_id']):
                                assert recovered['adopted'] and recovered['ffprobe_returncode'] == 0
                                candidate = Path(recovered['recovered_path'])
                                assert sha256(candidate) == recovered['recovered_sha256']
                                media = candidate
                        if split == 'train':
                            target = np.asarray(selected_train[f'{raw_dialogue}_{raw_uid}'], dtype=np.float64)
                            assert target.shape == (dims,) and np.isfinite(target).all()
                            target_rows.append(target)
                    if split == 'train':
                        groups.append(str(dialogue))
                    rows.append(dict(row_index=len(rows), dataset=dataset, split=split,
                                     dialogue_id=str(dialogue), utterance_index=index,
                                     raw_dialogue_id=raw_dialogue, raw_utterance_id=raw_uid,
                                     speaker_id=speaker_id, media_path=str(media),
                                     media_exists=media.is_file(), media_sha256=''))
            assert len(rows) == (n_train if split == 'train' else n_val)
            split_rows[split] = rows
            all_rows.extend(rows)
        output_dir = OUT / 'local_only' / dataset
        output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        target_array = np.stack(target_rows).astype(np.float64, copy=False)
        groups_array = np.asarray(groups, dtype=str)
        assert target_array.shape == (n_train, dims)
        assert groups_array.shape == (n_train,)
        np.save(output_dir / 'target_train.npy', target_array, allow_pickle=False)
        np.save(output_dir / 'groups_train.npy', groups_array, allow_pickle=False)
        result[dataset] = dict(seed=seed, target_train_shape=list(target_array.shape),
                               target_train_dtype=str(target_array.dtype), groups_dtype=str(groups_array.dtype),
                               target_train_finite=bool(np.isfinite(target_array).all()),
                               train_dialogues=len(ids['train']), validation_dialogues=len(ids['validation']),
                               train_utterances=n_train, validation_utterances=n_val,
                               train_target_min=float(target_array.min()), train_target_max=float(target_array.max()),
                               train_target_zero_rows=int(np.all(target_array == 0, axis=1).sum()),
                               split_rows=split_rows)
    # Paths are determined before hashing. Never search other dataset splits.
    missing = [{k: r[k] for k in ['dataset', 'split', 'dialogue_id', 'raw_utterance_id', 'media_path']}
               for r in all_rows if not r['media_exists']]
    write_json(OUT / 'provenance/media_missing_initial.json', missing)
    print(json.dumps(dict(phase='media_presence_checked', counts=dict(Counter(r['dataset'] + '/' + r['split'] for r in all_rows)), missing=missing), default=str), flush=True)
    paths = sorted({r['media_path'] for r in all_rows if r['media_exists']})
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        media_hashes = dict(zip(paths, pool.map(sha256, paths)))
    artifacts = {}
    for dataset, info in result.items():
        for split, rows in info.pop('split_rows').items():
            for row in rows:
                row['media_sha256'] = media_hashes.get(row['media_path'], '')
            path = OUT / 'local_only' / dataset / f'manifest_{split}.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            path.chmod(0o600)
            artifacts[str(path.relative_to(OUT))] = sha256(path)
            info[split + '_media_present'] = sum(r['media_exists'] for r in rows)
            info[split + '_raw_id_differs_from_array_index'] = sum(str(r['raw_utterance_id']) != str(r['utterance_index']) for r in rows) if dataset == 'MELD' else None
        for filename in ['target_train.npy', 'groups_train.npy']:
            path = OUT / 'local_only' / dataset / filename
            path.chmod(0o600)
            artifacts[str(path.relative_to(OUT))] = sha256(path)
    audit = dict(status='COMPLETE_WITH_MISSING_MEDIA' if missing else 'COMPLETE',
                 environment=dict(python=platform.python_version(), numpy=np.__version__),
                 fieldnames=FIELDS, row_order='effective split IDs in stored order; each XP04 videoIDs list in stored order',
                 datasets=result, missing_media=missing,
                 source_sha256=source_hashes, artifact_sha256=artifacts,
                 media_files_hashed=len(media_hashes), media_hash_algorithm='SHA-256 complete file bytes',
                 targets_exported='Train only; IEMOCAP XP04 base audio slot 4, MELD official selected emotion slot 0',
                 shared_pickle_deserialization=True, test_ids_indexed=False, test_features_indexed=False,
                 labels_indexed=False, validation_target_values_indexed=False,
                 validation_target_values_exported=False, media_decode_executed=False,
                 training_executed=False, external_upload_executed=False)
    for key, filename in [('train_missing_media_recovery', 'train_missing_media_recovery.json'),
                          ('missing_media_candidate_qc', 'missing_media_candidate_qc.json')]:
        if (OUT / 'provenance' / filename).is_file():
            audit[key] = 'provenance/' + filename
    write_json(OUT / 'provenance/data_manifest_audit.json', audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
