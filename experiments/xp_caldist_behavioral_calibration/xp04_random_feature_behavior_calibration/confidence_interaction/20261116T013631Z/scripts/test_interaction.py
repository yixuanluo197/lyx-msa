"""Small deterministic engineering checks; no experiment inputs or results used."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor'))
import mainline_core as historical
try:
    import interaction_core as core
except ModuleNotFoundError:
    core = None


def fixture():
    rng = np.random.default_rng(2901)
    n = 120
    confidence = rng.uniform(.2, .99, n)
    frame = pd.DataFrame({'confidence': confidence,
        'correct': rng.binomial(1, expit(-1 + 3 * confidence)),
        'dialogue_id': np.repeat([f'd{i:02d}' for i in range(12)], 10)})
    for feature in historical.FEATURES['M_joint']:
        frame[feature] = rng.normal(size=n)
    return frame


class InteractionTests(unittest.TestCase):
    def require_core(self):
        self.assertIsNotNone(core, 'interaction implementation has not been created')
        return core

    def test_training_only_transforms_ignore_held_rows_and_labels(self):
        c = self.require_core(); frame = fixture(); train = np.arange(80)
        original = c.fit_transforms(frame, train)
        altered = frame.copy(); altered.loc[80:, c.FEATURES] = 100000
        altered.loc[80:, 'confidence'] = -999
        altered['correct'] = 1 - altered.correct
        self.assertEqual(original, c.fit_transforms(altered, train))
        self.assertEqual(original['c0'], float(frame.iloc[train].confidence.mean()))
        z, t = c.transform(frame.iloc[train], original)
        np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(t.mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-14)
        np.testing.assert_allclose(t.std(axis=0), 1, atol=1e-14)

    def test_constant_behavior_has_exact_zero_interaction(self):
        c = self.require_core(); frame = fixture()
        frame[c.FEATURES] = [3., -2., 0., 8.]
        fitted = c.fit_transforms(frame, np.arange(len(frame)))
        z, t = c.transform(frame, fitted)
        np.testing.assert_array_equal(z, np.zeros_like(z))
        np.testing.assert_array_equal(t, np.zeros_like(t))

    def test_decomposition_and_bidirectional_changes_preserve_inputs(self):
        c = self.require_core(); frame = fixture(); original = frame.copy(deep=True)
        fitted = c.fit_model(frame, np.arange(len(frame)), 'M_interaction', .1)
        fitted['intercept'] = 0.; fitted['confidence_coefficient'] = 0.
        fitted['effective_behavior_coefficients'] = [3., 0., 0., 0.]
        fitted['effective_interaction_coefficients'] = [0., 2., 0., 0.]
        scores = c.score(frame, fitted)
        np.testing.assert_allclose(scores['behavior_score'], scores['additive_score'] + scores['interaction_score'])
        np.testing.assert_allclose(scores['reliability_score'], scores['base_score'] + scores['behavior_score'])
        np.testing.assert_allclose(scores['oof_confidence'], expit(scores['reliability_score']))
        shifts = scores['oof_confidence'] - frame.confidence.to_numpy()
        self.assertTrue((shifts > 0).any() and (shifts < 0).any())
        pd.testing.assert_frame_equal(frame, original)

    def test_objective_conversion_raw_confidence_and_shared_penalty(self):
        c = self.require_core(); frame = fixture(); n = len(frame); penalty = .1
        fitted = c.fit_model(frame, np.arange(n), 'M_interaction', penalty)
        z, t = c.transform(frame, fitted); u = np.column_stack([z, t])
        raw = np.asarray(fitted['raw_coefficients'])
        effective = np.r_[fitted['effective_behavior_coefficients'], fitted['effective_interaction_coefficients']]
        self.assertEqual(len(raw), 9)
        np.testing.assert_allclose(effective, raw[1:] / np.sqrt(n * penalty))
        design = np.column_stack([frame.confidence, u / np.sqrt(n * penalty)])
        scores = c.score(frame, fitted)
        np.testing.assert_allclose(design @ raw + fitted['intercept'], scores['reliability_score'], atol=1e-14)
        r = scores['reliability_score']; y = frame.correct.to_numpy()
        nll = np.mean(np.logaddexp(0, r) - y * r)
        sklearn_objective = nll + np.dot(raw, raw) / (2 * n)
        specified_objective = nll + raw[0]**2 / (2 * n) + penalty * np.dot(effective, effective) / 2
        self.assertAlmostEqual(sklearn_objective, specified_objective, places=14)

    def test_off_is_identical_e1_object_and_predictions(self):
        c = self.require_core(); frame = fixture(); train = np.arange(100); held = np.arange(100, 120)
        with patch.object(c, 'select_penalty', return_value='off'):
            scores, params = c.fit_outer(frame, train, held)
        for method in c.METHODS:
            self.assertIs(params['methods'][method]['outer_fit'], params['E1'])
            np.testing.assert_array_equal(scores[method]['oof_confidence'], scores['E1']['oof_confidence'])
            for candidates in params['methods'][method]['inner_candidate_fits']['off']:
                fold = params['inner_folds'][candidates['inner_fold']]
                self.assertIs(candidates['fit'], fold['E1'])

    def test_dialogue_isolation_and_fit_training_indices(self):
        c = self.require_core(); frame = fixture(); train = np.arange(100); held = np.arange(100, 120)
        _, params = c.fit_outer(frame, train, held)
        self.assertFalse(set(params['train_dialogues']) & set(params['held_dialogues']))
        for fold in params['inner_folds']:
            self.assertFalse(set(fold['train_dialogues']) & set(fold['held_dialogues']))
            self.assertLessEqual(set(fold['train_indices']) | set(fold['held_indices']), set(train))
        for method in c.METHODS:
            for candidates in params['methods'][method]['inner_candidate_fits'].values():
                for candidate in candidates:
                    self.assertFalse(set(candidate['fit']['train_dialogues']) & set(candidate['held_dialogues']))
        with self.assertRaises(AssertionError):
            c.fit_outer(frame, train, np.arange(95, 120))

    def test_historical_e1_and_joint_parity_on_tiny_fixture(self):
        c = self.require_core(); frame = fixture(); train = np.arange(100); held = np.arange(100, 120)
        for penalty in c.PENALTIES:
            method = 'E1' if penalty == 'off' else 'M_joint'
            features = [] if penalty == 'off' else c.FEATURES
            fitted = c.fit_model(frame, train, method, penalty)
            previous = historical.fit_model(frame, train, features, penalty)
            np.testing.assert_allclose(c.score(frame.iloc[held], fitted)['oof_confidence'],
                historical.score(frame.iloc[held], previous)['oof_confidence'], atol=1e-14, rtol=0)
        current, params = c.fit_outer(frame, train, held)
        old, previous_params = historical.fit_outer(frame, train, held, ['M_joint'])
        self.assertEqual(params['methods']['M_joint']['selected_penalty'], previous_params['methods']['M_joint']['selected_penalty'])
        np.testing.assert_allclose(current['M_joint']['oof_confidence'], old['M_joint']['oof_confidence'], atol=1e-14, rtol=0)

    def test_ties_prefer_off_then_stronger_penalty(self):
        c = self.require_core()
        self.assertEqual(c.select_penalty({'off': .2, '1.0': .2 - 5e-13, '0.1': .2, '0.01': .2}), 'off')
        self.assertEqual(c.select_penalty({'off': .3, '1.0': .2, '0.1': .2 - 5e-13, '0.01': .2}), 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
