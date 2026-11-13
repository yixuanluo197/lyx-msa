"""Independent spot checks against pinned official CaliDist and saved folds."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.special import expit

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('official_calidist', OUT / 'provenance/official_calidist.py')
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)
summary = json.loads((OUT / 'verification.json').read_text())
evidence = {}
for dataset in ('IEMOCAP', 'MELD'):
    folder = OUT / dataset
    original = pd.read_csv(folder / 'original_predictions.csv', dtype={'dialogue_id': str})
    pert = pd.read_csv(folder / 'perturbed_predictions.csv', dtype={'dialogue_id': str})
    behavior = pd.read_csv(folder / 'behavior_features.csv', dtype={'dialogue_id': str})
    reliability = pd.read_csv(folder / 'reliability_scores.csv', dtype={'dialogue_id': str})
    oof = pd.read_csv(folder / 'oof_calibration_predictions.csv', dtype={'dialogue_id': str})
    fold_parameters = json.loads((folder / 'fold_parameters.json').read_text())
    checked = 0
    for original_row in original.iloc[[0, len(original)//2, len(original)-1]].itertuples():
        mask = ((pert.dialogue_id == original_row.dialogue_id)
                & (pert.utterance_index == original_row.utterance_index))
        sample = pert[mask]
        p = np.asarray(json.loads(original_row.probabilities))
        for (corruption, modality), part in sample.groupby(['corruption', 'modality']):
            assert len(part) == 10 and list(part.repeat_id) == list(range(10))
            q = [np.asarray(json.loads(x)) for x in part.probabilities]
            preds = part.prediction.tolist()
            confs = part.confidence.tolist()
            expected_mu = official.get_mu(original_row.prediction, preds, confs)
            expected_delta = official.get_delta(original_row.confidence, confs)
            expected_js = float(np.mean([jensenshannon(p, x, base=np.e)**2 for x in q]))
            b = behavior[(behavior.dialogue_id == original_row.dialogue_id)
                         & (behavior.utterance_index == original_row.utterance_index)
                         & (behavior.corruption == corruption)
                         & (behavior.modality == modality)].iloc[0]
            assert np.isclose(b.mu, expected_mu, atol=1e-8)
            assert np.isclose(b.delta, expected_delta, atol=1e-8)
            assert np.isclose(b.js, expected_js, atol=1e-8)
            for row in part.itertuples():
                keys = json.loads(row.rng_key)
                seeds = json.loads(row.rng_seeds)
                assert len(keys) == len(seeds)
                for key, seed in zip(keys, seeds):
                    canonical = json.dumps(key, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
                    actual = int.from_bytes(hashlib.sha256(canonical.encode('utf-8')).digest()[:16], 'big')
                    assert actual == seed
            checked += 1
    prediction_checks = 0
    for f in fold_parameters:
        held = set(f['held_out_dialogues'])
        assert not (held & set(f['train_dialogues']))
        if f['method'] == 'E1':
            subset = oof[(oof.method == 'E1') & (oof.dialogue_id.isin(held))]
            original_subset = original[original.dialogue_id.isin(held)]
            assert np.array_equal(subset.utterance_index, original_subset.utterance_index)
            expected = expit(f['intercept'] + f['coefficient'] * original_subset.confidence.to_numpy())
            assert np.allclose(subset.confidence, expected, atol=1e-10)
        else:
            subset = reliability[(reliability.corruption == f['corruption'])
                                 & (reliability.modality == f['modality'])
                                 & (reliability.dialogue_id.isin(held))]
            expected = official.calibrate(subset.original_confidence, subset.lambda_oof,
                                          f['alpha'], f['beta'])
            assert np.allclose(subset.calibrated_confidence, expected, atol=1e-10)
            assert np.all(subset.calibrated_confidence <= subset.original_confidence + 1e-12)
        prediction_checks += len(subset)
    for method, subset in oof.groupby('method'):
        assert np.array_equal(subset.prediction.to_numpy().reshape(-1),
                              np.tile(original.prediction.to_numpy(), len(subset)//len(original))) if method != 'E2' else True
    evidence[dataset] = {'official_formula_spot_checks': checked,
                         'fold_confidence_predictions_checked': prediction_checks,
                         'folds': 12 if dataset == 'IEMOCAP' else 5,
                         'all_passed': True}
summary['independent_verification'] = evidence
(OUT / 'verification.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
print(json.dumps(evidence, indent=2))
