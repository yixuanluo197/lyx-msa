"""Independently reconstruct diagnostic evidence from historical prediction CSVs."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'diagnostic'
CONFIG = json.loads((ROOT / 'preregistration.json').read_text())
SOURCE = Path(CONFIG['source_run'])
PILOT = Path(CONFIG['seed1475_pilot'])
KEY = ['dataset', 'seed', 'dialogue_id', 'utterance_index']
MEASURES = ['mu', 'delta', 'D', 'A', 'JS']
TOL = 1e-12


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            value.update(block)
    return value.hexdigest()


def read(name):
    return pd.read_csv(OUT / name, dtype={'dialogue_id': str})


def equal(actual, expected):
    np.testing.assert_allclose(np.asarray(actual, float), np.asarray(expected, float),
                               rtol=0, atol=TOL, equal_nan=True)


def recalculate(q, p, predicted):
    maximum = q.max(axis=1)
    original_max = p[predicted]
    fixed = q[:, predicted]
    changes = q.argmax(axis=1) != predicted
    mid = (q + p) * .5
    # KL(p || mid) and KL(q || mid), natural logarithms, zero mass gives zero.
    pterm = np.where(p > 0, p * np.log(np.maximum(p, 1e-300) / mid), 0)
    qterm = np.where(q > 0, q * np.log(np.maximum(q, 1e-300) / mid), 0)
    return dict(mu=np.dot(changes.astype(float), maximum) / len(q),
                delta=abs(original_max - np.sum(maximum) / len(q)),
                D=np.sum(original_max - fixed) / len(q),
                A=np.linalg.norm(fixed - original_max) / np.sqrt(len(q)),
                JS=np.sum(pterm + qterm) / (2 * len(q)),
                flip_rate=np.count_nonzero(changes) / len(q),
                D_mc_se=np.sqrt(np.sum((fixed - fixed.mean()) ** 2) / (len(q) - 1) / len(q)))


def corr(x, y, ranked=False):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.ptp(x) <= CONFIG['zero_threshold'] or np.ptp(y) <= CONFIG['zero_threshold']:
        return np.nan
    if ranked:
        x, y = rankdata(x), rankdata(y)
    x, y = x - x.mean(), y - y.mean()
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def main():
    started = time.time()
    status = json.loads((OUT / 'verification.json').read_text())
    historical = json.loads((SOURCE / 'provenance/artifact_hashes.json').read_text())
    provenance_path = SOURCE / 'behavior/provenance/source_hashes.json'
    assert digest(provenance_path) == historical[str(provenance_path.relative_to(SOURCE))]
    new_sources = json.loads((OUT / 'source_hashes.json').read_text())
    hashed_old, git_verified = 0, 0
    repo = ROOT.parents[4] / 'CMERC'
    for filename, expected in new_sources.items():
        path = Path(filename)
        assert digest(path) == expected, filename
        if path.is_relative_to(SOURCE):
            assert historical[str(path.relative_to(SOURCE))] == expected, filename
            hashed_old += 1
        elif path.is_relative_to(PILOT):
            relative = path.relative_to(ROOT.parents[4])
            blob = subprocess.check_output(['git', '-C', str(repo), 'show', f'HEAD:{relative}'])
            assert hashlib.sha256(blob).hexdigest() == expected, filename
            git_verified += 1
    assert hashed_old == 11 and git_verified == 2
    originals = pd.read_csv(SOURCE / 'baseline/original_predictions.csv', dtype={'dialogue_id': str})
    features = read('feature_matrix.csv').set_index(KEY).sort_index()
    prior_features = pd.read_csv(SOURCE / 'behavior/feature_matrix.csv', dtype={'dialogue_id': str}).set_index(KEY)
    original_indexed = originals.set_index(KEY).sort_index()
    assert len(features) == len(original_indexed) == 8180
    assert not features.index.duplicated().any()
    assert features.index.equals(original_indexed.index)
    assert np.isfinite(features.select_dtypes(include=np.number).values).all()
    responses = read('utterance_responses.csv').set_index(KEY + ['corruption']).sort_index()
    halves = read('split_responses.csv').set_index(KEY + ['corruption', 'split']).sort_index()
    assert len(responses) == 16360 and not responses.index.duplicated().any()
    assert len(halves) == 49080 and not halves.index.duplicated().any()
    static_rows, groups, half_groups, rng_rows, raw_count = 0, 0, 0, 0, 0
    random_streams = {}
    for dataset, spec in CONFIG['datasets'].items():
        for seed in CONFIG['seeds']:
            current = original_indexed.loc[(dataset, seed)]
            assert len(current) == spec['n'] and len(current.index.get_level_values(0).unique()) == spec['dialogues']
            fs = features.loc[(dataset, seed)]
            assert fs.visual_missing.sum() == (1 if dataset == 'IEMOCAP' else 95)
            for (dialogue, utterance), row in current.iterrows():
                f = fs.loc[(dialogue, utterance)]
                p = np.array(json.loads(row.probabilities))
                pred = int(np.argmax(p))
                assert f.prediction == pred == row.prediction and f.label == row.label
                assert f.correct == int(pred == row.label)
                # Preserve the historical CSV confidence anchor; recomputing the
                # logit from the JSON probability can amplify a 1-ulp CSV parse
                # difference near confidence 1.
                c = float(row.confidence)
                equal(c, p.max())
                equal(f.confidence, c)
                clip = min(max(c, 1e-6), 1 - 1e-6)
                equal(f.logit_C, np.log(clip) - np.log1p(-clip))
                equal(f.margin, np.partition(p, -2)[-1] - np.partition(p, -2)[-2])
                equal(f.normalized_entropy, -np.dot(p, np.log(np.maximum(p, 1e-300))) / np.log(spec['classes']))
                equal([f[f'pred_class_{k}'] for k in range(1, 7)], [pred == k for k in range(1, 7)])
                assert f.visual_missing == prior_features.loc[(dataset, seed, dialogue, utterance)].visual_missing
                # Deliberately excludes prediction, label and model seed from the control key.
                encoded = json.dumps([dataset, str(dialogue), int(utterance), 'capacity_control_seed1475'],
                                     separators=(',', ':'), ensure_ascii=True).encode('utf8')
                rng_seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big')
                expected = np.random.Generator(np.random.PCG64(rng_seed)).standard_normal(4)
                actual = f[[f'rand_{k}' for k in range(4)]].to_numpy(float)
                equal(actual, expected)
                paired_key = (dataset, dialogue, utterance)
                if paired_key in random_streams:
                    equal(actual, random_streams[paired_key])
                random_streams[paired_key] = actual
                rng_rows += 1
                static_rows += 1
            path = PILOT / dataset / 'perturbed_predictions.csv' if seed == 1475 else SOURCE / f'runs/{dataset}/seed{seed}/perturbed_predictions.csv'
            raw = pd.read_csv(path, dtype={'dialogue_id': str})
            raw = raw[raw.modality.eq('joint')]
            assert len(raw) == spec['n'] * 20
            raw_count += len(raw)
            for (dialogue, utterance, kind), part in raw.groupby(['dialogue_id', 'utterance_index', 'corruption']):
                part = part.sort_values('repeat_id')
                assert part.repeat_id.tolist() == list(range(10))
                p = np.array(json.loads(current.loc[(dialogue, utterance)].probabilities))
                q = np.array([json.loads(v) for v in part.probabilities])
                assert q.shape == (10, spec['classes']) and np.isfinite(q).all()
                assert np.all((q >= 0) & (q <= 1))
                np.testing.assert_allclose(q.sum(axis=1), 1, rtol=0, atol=1e-5)
                pred = int(p.argmax())
                calculated = recalculate(q, p, pred)
                ident = (dataset, seed, dialogue, utterance, kind)
                record = responses.loc[ident]
                equal(record[list(calculated)], list(calculated.values()))
                f = fs.loc[(dialogue, utterance)]
                for name in MEASURES:
                    equal(f[f'{kind}_{name}'], calculated[name])
                for split, indices in enumerate(CONFIG['repeat_splits']):
                    half = halves.loc[ident + (split,)]
                    for side, selected in zip(('left', 'right'), indices):
                        output = recalculate(q[selected], p, pred)
                        equal(half[[f'{name}_{side}' for name in MEASURES]], [output[name] for name in MEASURES])
                    half_groups += 1
                groups += 1
            print(json.dumps({'independently_verified': dataset, 'seed': seed, 'groups': groups}), flush=True)
    split_summary = read('split_reliability.csv').set_index(['dataset', 'seed', 'corruption', 'split', 'measure'])
    summary = read('response_summary.csv').set_index(['dataset', 'seed', 'corruption', 'measure'])
    assert len(split_summary) == 300 and len(summary) == 120
    reset_responses = responses.reset_index()
    for (dataset, seed, kind), data in reset_responses.groupby(['dataset', 'seed', 'corruption']):
        fs = features.loc[(dataset, seed)].reindex(pd.MultiIndex.from_frame(data[['dialogue_id', 'utterance_index']]))
        for name in MEASURES:
            row = summary.loc[(dataset, seed, kind, name)]
            v = data[name].values
            assert row.n == len(v)
            equal([row['mean'], row.sample_sd, row.zero_fraction], [v.mean(), v.std(ddof=1), np.mean(abs(v) <= 1e-12)])
            for column, source_column in [('confidence_spearman', 'confidence'), ('entropy_spearman', 'normalized_entropy'), ('margin_spearman', 'margin')]:
                equal(row[column], corr(v, fs[source_column], True))
        variance = data.D.var(ddof=1)
        noise = np.square(data.D_mc_se).mean()
        row = summary.loc[(dataset, seed, kind, 'D_measurement_noise')]
        equal(row[['mean', 'sample_sd', 'observed_between_utterance_variance', 'mean_mc_variance', 'noise_corrected_reliability_ratio', 'fraction_abs_D_gt_1_96_mcse', 'no_flip_utterance_fraction', 'mean_flip_rate']],
              [data.D_mc_se.mean(), data.D_mc_se.std(ddof=1), variance, noise,
               max(0, variance - noise) / variance if variance > 0 else np.nan,
               np.mean(abs(data.D) > 1.96 * data.D_mc_se), np.mean(data.flip_rate == 0), data.flip_rate.mean()])
    for ident, data in halves.reset_index().groupby(['dataset', 'seed', 'corruption', 'split']):
        for measure in MEASURES:
            left, right = data[measure + '_left'], data[measure + '_right']
            pearson, spearman = corr(left, right), corr(left, right, True)
            row = split_summary.loc[ident + (measure,)]
            equal(row[['pearson', 'spearman', 'left_zero_fraction', 'right_zero_fraction']],
                  [pearson, spearman, np.mean(abs(left) <= 1e-12), np.mean(abs(right) <= 1e-12)])
            assert bool(row.correlation_estimable) == bool(np.isfinite(pearson) and np.isfinite(spearman))
            assert row.n == len(data)
    aggregates = read('aggregate_reliability.csv').set_index(['dataset', 'corruption', 'measure'])
    for ident, data in split_summary.reset_index().groupby(['dataset', 'corruption', 'measure']):
        row = aggregates.loc[ident]
        equal(row[['estimable_splits', 'spearman_mean', 'spearman_min', 'spearman_max', 'pearson_mean']],
              [data.correlation_estimable.sum(), data.spearman.mean(), data.spearman.min(), data.spearman.max(), data.pearson.mean()])
    evidence = dict(status='passed', static_original_rows=static_rows, response_groups=groups,
                    raw_joint_prediction_rows=raw_count, half_response_groups=half_groups,
                    split_correlation_rows=300, response_summary_rows=120,
                    label_free_random_rows=rng_rows, paired_control_ids=len(random_streams),
                    sources_rehashed=len(new_sources), prior_manifest_inputs_matched=hashed_old,
                    historical_git_seed1475_sources_matched=git_verified,
                    visual_missing_retained={'IEMOCAP_per_seed': 1, 'MELD_per_seed': 95},
                    all_numeric_features_finite=True, independent_formula_reconstruction=True,
                    no_diagnostic_function_imports=True, seconds=time.time() - started,
                    verifier_sha256=digest(Path(__file__)))
    status.update(independent_verification_pending=False, independent_verification=evidence,
                  all_passed=True, status='completed_and_independently_verified')
    (OUT / 'verification.json').write_text(json.dumps(status, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    artifacts = {str(p.relative_to(ROOT)): digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'artifact_hashes.json'}
    artifacts['scripts/verify_diagnostic.py'] = digest(Path(__file__))
    (OUT / 'artifact_hashes.json').write_text(json.dumps(artifacts, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
