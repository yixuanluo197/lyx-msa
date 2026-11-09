import importlib.util
import copy
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/calibration_only.py'
SPEC = importlib.util.spec_from_file_location('calibration_only', PATH)
cal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cal)


def sample():
    rng = np.random.default_rng(42)
    logits = rng.normal(size=(48, 3)) * 3
    labels = rng.integers(0, 3, size=48)
    ids = np.repeat([f'd{i}' for i in range(8)], 6)
    return logits, labels, ids


def test_fit_is_bounded_argmax_invariant_and_mask_independent():
    logits, labels, ids = sample()
    mask = np.ones(len(labels), bool)
    mask[::7] = False
    labels[~mask] = -100
    a = cal.fit_calibrators(logits, labels, ids, mask, dataset='IEMOCAP', cv_folds=3)
    labels[~mask] = 99999
    b = cal.fit_calibrators(logits, labels, ids, mask, dataset='IEMOCAP', cv_folds=3)
    assert a['parameters'] == b['parameters']
    assert set(a['probabilities']) == {'E0', 'E1', 'E2', 'E3'}
    for method, probs in a['probabilities'].items():
        assert probs.shape == logits.shape
        assert np.array_equal(probs.argmax(1), logits.argmax(1))
        assert np.allclose(probs.sum(1), 1)
        assert a['metrics'][method]['n'] == int(mask.sum())
        assert np.all((a['temperatures'][method] >= .05) & (a['temperatures'][method] <= 20))
    assert a['parameters']['E2']['lambda'] in cal.LAMBDA_CANDIDATES
    assert np.all(a['oof_fold'][mask] >= 0)
    assert np.all(a['oof_fold'][~mask] == -1)
    assert np.isnan(a['oof_probabilities']['E2'][~mask]).all()


def test_meld_episode_groups_never_cross_folds():
    logits, labels, ids = sample()
    mapping = {f'd{i}': f'episode{i // 2}' for i in range(8)}
    result = cal.fit_calibrators(logits, labels, ids, np.ones(48, bool), dataset='MELD', episode_mapping=mapping)
    for episode in set(mapping.values()):
        rows = np.array([mapping[d] == episode for d in ids])
        assert len(set(result['oof_fold'][rows])) == 1
    with pytest.raises(ValueError, match='episode'):
        cal.fit_calibrators(logits, labels, ids, np.ones(48, bool), dataset='MELD')
    with pytest.raises(ValueError, match='episode'):
        cal.fit_calibrators(logits, labels, ids, np.ones(48, bool), dataset='MELD', episode_mapping={'d0': 'ep'})


def test_absent_predicted_class_falls_back_to_global():
    logits = np.array([[5., 0., -3.], [3., 0., -2.], [0., 4., -5.], [0., 1., -1.]])
    labels = np.array([0, 1, 1, 2])
    g = cal.fit_global(logits, labels)
    fit = cal.fit_predicted_class(logits, labels, logits.argmax(1), g['theta'], .1)
    assert fit['support'][2] == 0
    assert fit['fallback_global'][2]
    assert fit['temperatures'][2] == g['temperature']


def test_metrics_and_nll_are_stable():
    p = np.array([[.8, .2], [.1, .9]])
    m = cal.metrics(p, np.array([0, 1]))
    assert m['nll'] == pytest.approx(-np.log([.8, .9]).mean())
    assert m['brier'] == pytest.approx(.05)
    assert m['ece15'] == pytest.approx(.15)
    assert np.isfinite(cal.log_softmax(np.array([[1e4, -1e4]]))).all()


def test_calibration_role_and_checkpoint_guards():
    with pytest.raises(ValueError, match='Calibration'):
        cal.validate_role_artifact({'role': 'evaluation'})
    with pytest.raises(ValueError, match='selected'):
        cal.validate_checkpoint({'model_state_dict': {}}, {'dataset': 'IEMOCAP', 'fold': 'fold1'})
    meta = cal.validate_checkpoint({'model_state_dict': {}, 'selected_epoch': 2, 'is_best': True,
                                    'training_complete': True, 'complete': True,
                                    'epoch': 2, 'selection_metric': 'development_weighted_f1', 'selection_role': 'development',
                                    'dataset': 'IEMOCAP', 'fold': 'fold1'}, {'dataset': 'IEMOCAP', 'fold': 'fold1'})
    assert meta['selected_epoch'] == 2


def test_rejects_too_few_groups_and_bad_masks():
    z, y, ids = sample()
    with pytest.raises(ValueError, match='two'):
        cal.fit_calibrators(z, y, np.repeat('d0', len(y)), np.ones(48, bool), dataset='IEMOCAP')
    with pytest.raises(ValueError, match='boolean'):
        cal.fit_calibrators(z, y, ids, np.ones(48, int), dataset='IEMOCAP')


def approved_fixture(tmp_path):
    role = {'dialogue_ids': ['d0'], 'dialogue_count': 1, 'utterance_count': 2,
            'supervision_eligible_count': 1, 'unknown_masked_count': 1}
    role['sha256'] = cal.canonical_sha256(role)
    manifest = {'status': 'approved', 'protocols': {'IEMOCAP': {'folds': {'fold1': {'roles': {'calibration': role}}}}},
                'unknown_label_mask': {'utterances': [{'dialogue_id': 'd0', 'utterance_index': 1}]}}
    manifest['content_sha256'] = cal.canonical_sha256(manifest)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    artifact = {'role': 'calibration', 'dataset': 'IEMOCAP', 'fold': 'fold1', 'dialogue_ids': ['d0'],
                'dialogues': {'d0': {'dialogue_id': 'd0', 'labels': np.array([0, -100]), 'supervision_mask': np.array([True, False])}},
                'approved_role': role, 'approved_manifest_sha256': cal.sha256(path),
                'provenance': {'role_sha256': role['sha256'], 'approved_manifest_sha256': cal.sha256(path)}}
    return artifact, manifest, path


def test_manifest_binding_and_unknown_mask_are_enforced(tmp_path):
    artifact, manifest, path = approved_fixture(tmp_path)
    cal.validate_role_artifact(artifact, manifest, path)
    bad = copy.deepcopy(artifact)
    bad['provenance']['approved_manifest_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='manifest'):
        cal.validate_role_artifact(bad, manifest, path)
    bad = copy.deepcopy(artifact)
    bad['dialogues']['d0']['supervision_mask'][1] = True
    with pytest.raises(ValueError, match='unknown mask'):
        cal.validate_role_artifact(bad, manifest, path)
    bad_manifest = copy.deepcopy(manifest)
    bad_manifest['status'] = 'candidate'
    with pytest.raises(ValueError, match='manifest'):
        cal.validate_role_artifact(artifact, bad_manifest, path)


def test_exported_predictions_bind_checkpoint_and_exact_role_rows(tmp_path):
    artifact, _, _ = approved_fixture(tmp_path)
    checkpoint = tmp_path / 'best.pt'
    checkpoint.write_bytes(b'checkpoint-fixture')
    data = {**cal.artifact_rows(artifact), 'role': np.asarray('calibration'),
            'checkpoint_sha256': np.asarray(cal.sha256(checkpoint)), 'log_probabilities': np.array([[-.1, -2.], [-1., -.5]])}
    path = tmp_path / 'prediction.npz'
    np.savez(path, **data)
    assert cal.load_prediction_npz(path, artifact, checkpoint).shape == (2, 2)
    data['role'] = np.asarray('evaluation')
    np.savez(path, **data)
    with pytest.raises(ValueError, match='Calibration'):
        cal.load_prediction_npz(path, artifact, checkpoint)
    data['role'] = np.asarray('calibration')
    data['checkpoint_sha256'] = np.asarray('wrong')
    np.savez(path, **data)
    with pytest.raises(ValueError, match='checkpoint'):
        cal.load_prediction_npz(path, artifact, checkpoint)


def test_saved_results_include_all_probabilities_metadata_and_diagnostics(tmp_path):
    z, y, ids = sample()
    mask = np.ones(48, bool)
    result = cal.fit_calibrators(z, y, ids, mask, dataset='IEMOCAP')
    rows = {'labels': y, 'dialogue_ids': ids, 'utterance_indices': np.tile(np.arange(6), 8), 'supervision_mask': mask}
    output = tmp_path / 'calibration'
    cal.save_results(output, result, rows, {'role': 'calibration'})
    with np.load(output / 'calibration_probabilities.npz', allow_pickle=False) as source:
        for method in cal.METHODS:
            assert np.array_equal(source[f'{method}_predictions'], z.argmax(1))
            assert source[f'{method}_probabilities'].shape == z.shape
            assert source[f'{method}_log_probabilities'].shape == z.shape
        assert np.array_equal(source['input_scores'], z)
        assert np.array_equal(source['E0_oof_probabilities'], source['E0_probabilities'])
    diagnostic = json.loads((output / 'oof_diagnostic.json').read_text())
    assert diagnostic['diagnostic_only']
    assert len(diagnostic['folds']) == 5
    with pytest.raises(FileExistsError):
        cal.save_results(output, result, rows, {})


def test_extreme_nll_uses_log_probabilities_without_clipping():
    logits = np.array([[10000., -10000.], [-10000., 10000.], [10000., -10000.], [-10000., 10000.]])
    y = np.array([1, 0, 1, 0])
    result = cal.fit_calibrators(logits, y, np.array(['a', 'b', 'c', 'd']), np.ones(4, bool), dataset='IEMOCAP')
    assert result['metrics']['E0']['nll'] == pytest.approx(20000.)
    assert result['metrics']['E1']['nll'] == pytest.approx(1000.)
