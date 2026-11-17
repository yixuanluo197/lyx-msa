"""Read-only identity/index/source audit. Never deserialize target PKLs or models."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import subprocess
import zipfile

WORK = Path('/LOCAL/lyx-msa')
OUT = Path(__file__).resolve().parents[1]
BASE = OUT.parent
XP = WORK / 'CMERC/experiments/xp_mainline_validation/20261016T050823Z'
PRIOR = BASE / '20261116T174410Z'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    history = {}
    for timestamp in ('20261116T174410Z', '20261117T024213Z'):
        for path in (BASE / timestamp).rglob('*'):
            if (path.is_file() and path.suffix in ('.md', '.py', '.json')
                    and not {'local_only', '__pycache__'}.intersection(path.parts)):
                history[str(path.relative_to(BASE))] = sha(path)
    write(OUT / 'provenance/history_hashes_before.json', history)
    previous = read(PRIOR / 'provenance/official_feature_comparison.json')['comparisons']['CMERC']
    inputs, dataset_records = {}, {}
    for dataset, seed, expected_rows, expected_groups, target_file in (
            ('IEMOCAP', 1475, 527, 12, 'iemocap_features_roberta.pkl'),
            ('MELD', 67137, 1109, 114, 'meld_features_roberta.pkl')):
        config = XP / f'configs/effective/{dataset}_cmerc_validation_seed{seed}.json'
        checkpoint = XP / f'runs/cmerc/{dataset}_validation_seed{seed}/best.pt'
        target = XP / 'code/data' / target_file
        for path in (config, checkpoint, target):
            inputs[str(path.resolve())] = sha(path)
        cfg = read(config)
        assert cfg['seed'] == seed and cfg['Dataset'] == dataset
        allowed_ids = list(map(str, cfg['provenance']['split']['validation_ids']))
        assert len(allowed_ids) == expected_groups
        old_manifest = BASE / f'20261117T024213Z/local_only/{dataset}/manifest_validation.csv'
        with old_manifest.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == expected_rows
        assert list(dict.fromkeys(row['dialogue_id'] for row in rows)) == allowed_ids
        seen = set()
        public_rows = []
        for i, row in enumerate(rows):
            assert int(row['row_index']) == i and row['dataset'] == dataset and row['split'] == 'validation'
            key = row['dialogue_id'], int(row['utterance_index'])
            assert key not in seen
            seen.add(key)
            public_rows.append({k: row[k] for k in ('row_index', 'dataset', 'split', 'dialogue_id',
                               'utterance_index', 'raw_utterance_id', 'speaker_id')})
        dest = OUT / 'local_only' / f'{dataset}_validation_ids.csv'
        with dest.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(public_rows[0]))
            writer.writeheader()
            writer.writerows(public_rows)
        dest.chmod(0o600)
        identity = next(r for r in previous if r['file'] == target_file)
        assert inputs[str(target.resolve())] == identity['official_sha256'] == identity['local_sha256']
        dataset_records[dataset] = {'seed': seed, 'validation_rows': expected_rows,
            'validation_dialogues': expected_groups, 'effective_config': str(config),
            'config_sha256': sha(config), 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
            'reference_features': str(target.resolve()), 'reference_sha256': sha(target),
            'matches_previously_verified_official_file': True,
            'reference_shape_from_anchored_prior_audit': [[expected_rows, 1024]] * 4,
            'target_values_opened_this_round': False, 'original_text_opened_this_round': False,
            'validation_ids_source': str(old_manifest), 'validation_ids_source_sha256': sha(old_manifest),
            'private_manifest_sha256': sha(dest), 'new_reference_shape_measurement': False}
    for name in ('dataloader.py', 'model.py'):
        path = XP / 'code' / name
        inputs[str(path)] = sha(path)
    archive = PRIOR / 'local_only/CMERC_official_features.zip'
    old_download = read(PRIOR / 'provenance/cmerc_download.json')
    assert sha(archive) == old_download['archive_sha256']
    with zipfile.ZipFile(archive) as handle:
        members = [{'name': z.filename, 'bytes': z.file_size} for z in handle.infolist()]
    assert len(members) == 4 and all(z['name'].endswith('.pkl') for z in members)
    inputs[str(archive)] = sha(archive)
    write(OUT / 'provenance/input_hashes_before.json', inputs)
    write(OUT / 'provenance/input_index_audit.json', {
        'status': 'PASS_IDENTITY_AND_INDEX_AUDIT_ONLY',
        'checked_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'datasets': dataset_records,
        'official_archive_members': members, 'official_archive_contains_encoder_weights': False,
        'reference_value_arrays_loaded': False, 'text_or_feature_extraction_runs': 0,
        'checkpoint_deserialized': False, 'test_slots_or_labels_indexed': False,
        'historical_artifacts_anchored': len(history),
        'git_head': subprocess.check_output(['git', '-C', str(WORK / 'CMERC'), 'rev-parse', 'HEAD'], text=True).strip()})
    print(json.dumps({'status': 'PASS', 'frozen_inputs': len(inputs), 'historical_artifacts': len(history),
                      'validation_ids': {ds: r['validation_rows'] for ds, r in dataset_records.items()}}))


if __name__ == '__main__':
    main()
