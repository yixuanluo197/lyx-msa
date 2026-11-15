"""Build only the registered original mu/delta inputs from frozen perturbations."""

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'inputs'
CONFIG = json.loads((ROOT / 'preregistration.json').read_text())
SOURCE = Path(CONFIG['source_five_seed'])
PILOT = Path(CONFIG['source_pilot'])
KEY = ['dataset', 'seed', 'dialogue_id', 'utterance_index']
BEHAVIOR = CONFIG['features']['M_joint']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def response(confidence, predictions, maxima, original_prediction):
    maxima = np.asarray(maxima, float)
    return {'mu': float(np.mean((np.asarray(predictions) != original_prediction) * maxima)),
            'delta': float(abs(confidence - maxima.mean())),
            'flip_rate': float(np.mean(np.asarray(predictions) != original_prediction)),
            'confidence_change_mean': float(maxima.mean() - confidence),
            'confidence_change_std': float(maxima.std(ddof=0))}


def main():
    assert not (OUT / 'verification.json').exists(), 'Inputs already built; do not overwrite'
    old_manifest = json.loads((SOURCE / 'provenance/artifact_hashes.json').read_text())
    paths = [ROOT / 'preregistration.json', Path(__file__), SOURCE / 'baseline/original_predictions.csv',
             SOURCE / 'behavior/feature_matrix.csv', PILOT / 'behavior_features.csv']
    raw_paths = {}
    for ds in CONFIG['datasets']:
        for seed in CONFIG['seeds']:
            raw_paths[ds, seed] = (PILOT / ds if seed == 1475 else SOURCE / f'runs/{ds}/seed{seed}') / 'perturbed_predictions.csv'
    paths.extend(raw_paths.values())
    hashes = {str(path): sha(path) for path in paths}
    for path in paths:
        if path.is_relative_to(SOURCE):
            assert hashes[str(path)] == old_manifest[str(path.relative_to(SOURCE))]
        elif path.is_relative_to(PILOT):
            repo = ROOT.parents[4] / 'CMERC'
            relative = path.relative_to(ROOT.parents[4])
            blob = subprocess.check_output(['git', '-C', str(repo), 'show', f'HEAD:{relative}'])
            assert hashlib.sha256(blob).hexdigest() == hashes[str(path)]
    originals = pd.read_csv(SOURCE / 'baseline/original_predictions.csv', dtype={'dialogue_id': str})
    previous = pd.read_csv(SOURCE / 'behavior/feature_matrix.csv', dtype={'dialogue_id': str}).set_index(KEY)
    pilot = pd.read_csv(PILOT / 'behavior_features.csv', dtype={'dialogue_id': str})
    pilot = pilot.set_index(KEY + ['modality', 'corruption'])
    assert len(originals) == 8180 and not originals.duplicated(KEY).any()
    records, diagnostics, raw_count, checked = {}, [], 0, 0
    for (dataset, seed), path in raw_paths.items():
        raw = pd.read_csv(path, dtype={'dialogue_id': str}, usecols=[
            'dataset', 'seed', 'dialogue_id', 'utterance_index', 'modality', 'corruption', 'strength',
            'repeat_id', 'prediction', 'confidence', 'probabilities', 'visual_missing_skipped'])
        expected_modalities = {'text', 'audio', 'visual', 'joint'} if seed == 1475 else {'joint'}
        assert set(raw.modality) == expected_modalities and set(raw.corruption) == {'gaussian', 'dropout'}
        assert np.allclose(raw.strength, .05) and set(raw.seed) == {seed}
        origin = originals[(originals.dataset == dataset) & (originals.seed == seed)].set_index(['dialogue_id', 'utterance_index'])
        spec = CONFIG['datasets'][dataset]
        assert len(origin) == spec['utterances']
        assert len(raw) == len(origin) * len(expected_modalities) * 20
        raw_count += len(raw)
        for (dialogue, utterance, modality, kind), group in raw.groupby(['dialogue_id', 'utterance_index', 'modality', 'corruption'], sort=False):
            group = group.sort_values('repeat_id')
            assert group.repeat_id.tolist() == list(range(10))
            probabilities = np.array([json.loads(value) for value in group.probabilities])
            assert probabilities.shape == (10, 6 if dataset == 'IEMOCAP' else 7)
            assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0)
            assert np.array_equal(probabilities.argmax(axis=1), group.prediction)
            assert np.allclose(probabilities.max(axis=1), group.confidence, atol=1e-12, rtol=0)
            row = origin.loc[dialogue, utterance]
            original_key = (dataset, seed, dialogue, utterance)
            missing = bool(previous.loc[original_key, 'visual_missing'])
            skip = modality == 'visual' and missing
            if skip:
                assert group.visual_missing_skipped.all()
                assert np.allclose(probabilities, np.array(json.loads(row.probabilities))[None, :], atol=1e-12, rtol=0)
            values = response(float(row.confidence), group.prediction, group.confidence, int(row.prediction))
            for name in ['mu', 'delta']:
                expected = previous.loc[original_key, f'{kind}_{name}'] if modality == 'joint' else pilot.loc[original_key + (modality, kind), name]
                assert abs(values[name] - expected) < 1e-12
            checked += 1
            key = original_key + (modality,)
            if key not in records:
                records[key] = dict(zip(KEY + ['modality'], key))
                records[key].update({'label': int(row.label), 'prediction': int(row.prediction),
                                     'correct': int(row.label == row.prediction), 'confidence': float(row.confidence),
                                     'visual_missing': missing, 'perturbation_skipped': skip})
            records[key].update({f'{kind}_{name}': value for name, value in values.items()})
            diagnostics.append({**dict(zip(KEY + ['modality'], key)), 'corruption': kind,
                                'correct': int(row.label == row.prediction), 'visual_missing': missing,
                                'perturbation_skipped': skip, **values})
        print(json.dumps({'dataset': dataset, 'seed': seed, 'source_rows': len(raw)}), flush=True)
    frame = pd.DataFrame(records.values())
    assert len(frame) == 13088 and not frame.duplicated(KEY + ['modality']).any()
    assert np.isfinite(frame.select_dtypes(include=np.number)).all().all()
    joint = frame[frame.modality == 'joint'].copy()
    modality = frame[frame.seed == 1475].copy()
    assert len(joint) == 8180 and len(modality) == 6544
    assert len(diagnostics) == 26176 and raw_count == 261760
    joint.to_csv(OUT / 'joint_features.csv', index=False)
    modality.to_csv(OUT / 'modality_features.csv', index=False)
    diagnosis = pd.DataFrame(diagnostics)
    diagnosis.to_csv(OUT / 'response_records.csv', index=False)
    summary = diagnosis.groupby(['dataset', 'seed', 'modality', 'corruption']).agg(
        utterances=('mu', 'size'), mean_mu=('mu', 'mean'), mean_delta=('delta', 'mean'),
        mean_flip_rate=('flip_rate', 'mean'), any_flip_fraction=('flip_rate', lambda x: float((x > 0).mean())),
        confidence_change_mean=('confidence_change_mean', 'mean'),
        mean_within_repeat_confidence_sd=('confidence_change_std', 'mean'),
        skipped_visual=('perturbation_skipped', 'sum')).reset_index()
    summary.to_csv(OUT / 'behavior_response_summary.csv', index=False)
    diagnosis.groupby(['dataset', 'seed', 'modality', 'corruption', 'correct']).agg(
        utterances=('mu', 'size'), mean_mu=('mu', 'mean'), mean_delta=('delta', 'mean'),
        mean_flip_rate=('flip_rate', 'mean')).reset_index().to_csv(OUT / 'correct_wrong_response.csv', index=False)
    for path, expected in hashes.items():
        assert sha(path) == expected
    dump(OUT / 'source_hashes.json', hashes)
    dump(OUT / 'environment.json', {'python': sys.executable, 'numpy': np.__version__, 'pandas': pd.__version__,
                                   'host': platform.node(), 'gpu_used': False})
    dump(OUT / 'verification.json', {'status': 'input_integrity_passed', 'all_passed': True,
        'joint_rows': len(joint), 'modality_rows': len(modality), 'raw_probability_rows_rechecked': raw_count,
        'original_mu_delta_groups_matched': checked, 'one_utterance_per_fit_record': True,
        'only_original_mu_delta_are_scoring_features': BEHAVIOR,
        'source_files_unchanged': True, 'historical_artifact_and_git_sources_matched': True,
        'visual_missing_per_seed': {'IEMOCAP': 1, 'MELD': 95}})
    print(json.dumps({'all_passed': True, 'joint_rows': len(joint), 'modality_rows': len(modality)}))


if __name__ == '__main__':
    main()
