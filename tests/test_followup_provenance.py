"""Content-addressed causal curriculum fixtures; no teacher is trained here."""
import copy
import json
import pickle
from pathlib import Path

import pytest

from causal_provenance import (architecture_from_args, build_data_provenance,
    create_difficulty_manifest, resolve_causal_difficulty, sha256_file,
    write_causal_difficulty)


@pytest.fixture
def fixture_artifact(tmp_path):
    feature = tmp_path / 'feature.pkl'
    feature.write_bytes(b'synthetic features for provenance fixture only')
    data = build_data_provenance('IEMOCAP', ['train-a', 'train-b'], ['valid-a'],
                                 {'feature.pkl': feature}, split_seed=17)
    architecture = architecture_from_args({'causal': True, 'Dataset': 'IEMOCAP',
        'base_model': 'GRU', 'norm': 'LN', 'num_L': 2, 'num_K': 2,
        'av_using_lstm': False, 'use_residue': False})
    checkpoint_path = tmp_path / 'fixture.pt'
    checkpoint_path.write_bytes(b'synthetic checkpoint provenance fixture only')
    checkpoint = {'epoch': 2, 'best_epoch': 1, 'data_provenance': data,
        'args': {**architecture, 'selection_protocol': 'validation', 'seed': 71,
                 'split_seed': 17, 'replacement_policy': 'prefix_mean_v1'}}
    manifest = create_difficulty_manifest(checkpoint_path, checkpoint, data, architecture)
    path = write_causal_difficulty({'train-a': .2, 'train-b': .7}, manifest,
                                    output_dir=tmp_path)
    return path, data, architecture, checkpoint, manifest


def test_explicit_artifact_roundtrip_records_source_and_content(fixture_artifact):
    path, data, architecture, _, manifest = fixture_artifact
    values, artifact = resolve_causal_difficulty(path, 'IEMOCAP', data, architecture)
    assert values == {'train-a': .2, 'train-b': .7}
    assert artifact['sha256'] == sha256_file(path)
    assert artifact['path'] == str(Path(path).resolve())
    assert artifact['manifest'] == manifest
    assert manifest['source_checkpoint']['training_data_provenance'] == data
    assert manifest['source_checkpoint']['training_seed'] == 71
    assert manifest['source_checkpoint']['selection_protocol'] == 'validation'


@pytest.mark.parametrize('change', ['official_test', 'different_split', 'missing_provenance',
                                   'replacement', 'architecture', 'training_seed'])
def test_manifest_rejects_incompatible_or_missing_teacher_evidence(fixture_artifact, change):
    path, data, architecture, checkpoint, _ = fixture_artifact
    checkpoint = copy.deepcopy(checkpoint)
    if change == 'official_test':
        checkpoint['args']['selection_protocol'] = 'official_test'
    elif change == 'different_split':
        checkpoint['data_provenance']['validation_ids_sha256'] = '0' * 64
    elif change == 'missing_provenance':
        del checkpoint['data_provenance']
    elif change == 'replacement':
        checkpoint['args']['replacement_policy'] = 'whole_dialogue_mean'
    elif change == 'architecture':
        checkpoint['args']['num_K'] += 1
    else:
        del checkpoint['args']['seed']
    with pytest.raises(ValueError):
        create_difficulty_manifest(path, checkpoint, data, architecture)


def test_train_only_values_do_not_prove_teacher_has_not_trained_on_validation(fixture_artifact):
    path, data, architecture, checkpoint, _ = fixture_artifact
    checkpoint = copy.deepcopy(checkpoint)
    checkpoint['data_provenance']['train_ids_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='training data provenance'):
        create_difficulty_manifest(path, checkpoint, data, architecture)


def test_missing_provenance_wrapper_rejected_even_with_train_only_ids(tmp_path, fixture_artifact):
    _, data, architecture, _, _ = fixture_artifact
    path = tmp_path / 'legacy.pkl'
    path.write_bytes(pickle.dumps({'causal': True, 'dataset': 'IEMOCAP',
                                  'difficulties': {'train-a': .1}}))
    with pytest.raises(ValueError, match='manifest'):
        resolve_causal_difficulty(path, 'IEMOCAP', data, architecture)


def test_replaced_artifact_rejected_by_sidecar_and_pinned_digest(fixture_artifact):
    path, data, architecture, _, _ = fixture_artifact
    original_sha = sha256_file(path)
    payload = pickle.loads(Path(path).read_bytes())
    payload['difficulties']['train-a'] = 999.
    Path(path).write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match='SHA256'):
        resolve_causal_difficulty(path, 'IEMOCAP', data, architecture)
    with pytest.raises(ValueError, match='SHA256'):
        resolve_causal_difficulty(path, 'IEMOCAP', data, architecture,
                                  expected_sha256=original_sha)


def test_writer_does_not_overwrite_existing_output(fixture_artifact):
    path, _, _, _, manifest = fixture_artifact
    before = Path(path).read_bytes()
    with pytest.raises(FileExistsError):
        write_causal_difficulty({'train-a': .3, 'train-b': .5}, manifest, output_path=path)
    assert Path(path).read_bytes() == before


def test_expected_split_and_policy_are_checked_on_read(fixture_artifact):
    path, data, architecture, _, _ = fixture_artifact
    changed = {**data, 'split_seed': 18}
    with pytest.raises(ValueError, match='data provenance'):
        resolve_causal_difficulty(path, 'IEMOCAP', changed, architecture)
    with pytest.raises(ValueError, match='replacement'):
        resolve_causal_difficulty(path, 'IEMOCAP', data, architecture,
                                  replacement_policy='zero_prefix_v1')


def test_feature_content_and_ordered_ids_are_part_of_provenance(tmp_path):
    feature = tmp_path / 'features'
    feature.write_bytes(b'features')
    first = build_data_provenance('IEMOCAP', ['a', 'b'], ['v'], [feature], 1)
    reordered = build_data_provenance('IEMOCAP', ['b', 'a'], ['v'], [feature], 1)
    feature.write_bytes(b'changed features')
    changed = build_data_provenance('IEMOCAP', ['a', 'b'], ['v'], [feature], 1)
    assert first != reordered
    assert first != changed
    with pytest.raises(ValueError, match='overlap'):
        build_data_provenance('IEMOCAP', ['v'], ['v'], [feature], 1)


def test_causal_dataset_requires_explicit_pinned_source_before_reading_features(monkeypatch):
    from dataloader import IEMOCAPDataset
    # Admission must fail before deserializing dataset features.
    with pytest.raises(ValueError, match='explicit difficulty path'):
        IEMOCAPDataset(causal=True, epoch_ratio=.5)


@pytest.mark.parametrize('condition', ['replaced', 'missing'])
def test_source_checkpoint_bytes_are_revalidated_on_read(fixture_artifact, condition):
    path, data, architecture, _, manifest = fixture_artifact
    teacher_path = Path(manifest['source_checkpoint']['path'])
    if condition == 'replaced':
        teacher_path.write_bytes(b'replaced teacher fixture')
    else:
        teacher_path.unlink()
    with pytest.raises(ValueError, match='source checkpoint.*SHA256'):
        resolve_causal_difficulty(path, 'IEMOCAP', data, architecture)


def test_expected_teacher_training_seed_must_match(fixture_artifact):
    path, data, architecture, _, _ = fixture_artifact
    with pytest.raises(ValueError, match='training seed'):
        resolve_causal_difficulty(path, 'IEMOCAP', data, architecture,
                                  expected_teacher_seed=72)
    _, artifact = resolve_causal_difficulty(path, 'IEMOCAP', data, architecture,
                                            expected_teacher_seed=71)
    assert artifact['manifest']['source_checkpoint']['training_seed'] == 71


@pytest.mark.parametrize('change', ['teacher_split', 'selection', 'missing_hash', 'missing_seed'])
def test_payload_manifest_is_validated_beyond_file_integrity(fixture_artifact, change):
    path, data, architecture, _, _ = fixture_artifact
    payload = pickle.loads(Path(path).read_bytes())
    source = payload['manifest']['source_checkpoint']
    if change == 'teacher_split':
        source['training_data_provenance'] = {**data, 'split_seed': 18}
    elif change == 'selection':
        source['selection_protocol'] = 'official_test'
    elif change == 'missing_hash':
        del source['sha256']
    else:
        del source['training_seed']
    Path(path).write_bytes(pickle.dumps(payload))
    Path(str(path) + '.manifest.json').write_text(json.dumps({
        'sha256': sha256_file(path), 'manifest': payload['manifest']}))
    with pytest.raises(ValueError):
        resolve_causal_difficulty(path, 'IEMOCAP', data, architecture)
