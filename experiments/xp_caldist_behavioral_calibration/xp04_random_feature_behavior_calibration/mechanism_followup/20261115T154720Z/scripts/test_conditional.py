"""Engineering tests: train-only scaling, rare columns, serialization and correction semantics."""
import json
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

import conditional_core as core


class ConditionalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(51)
        self.groups = np.repeat(np.arange(8).astype(str), 12)
        n = len(self.groups)
        self.frame = pd.DataFrame({name: rng.normal(size=n) for name in ['logit_C', 'margin', 'normalized_entropy'] + core.BEHAVIOR + [f'rand_{k}' for k in range(4)]})
        for k in range(1, 6):
            self.frame[f'pred_class_{k}'] = (np.arange(n) % 6 == k).astype(float)
        self.frame['pred_class_5'] = 0.0
        self.frame.loc[self.groups == '7', 'pred_class_5'] = 1.0
        self.y = np.tile([0, 1], n // 2)

    def test_scaler_excludes_held_and_handles_unseen_rare_indicator(self):
        train = self.frame[self.groups != '7']
        held = self.frame[self.groups == '7'].copy()
        columns = core.feature_columns('IEMOCAP')['S']
        fit = core.fit_logistic(train[columns], self.y[self.groups != '7'], .1)
        np.testing.assert_allclose(fit['mean'], train[columns].mean(), atol=1e-14)
        self.assertEqual(fit['scale'][-1], 1.0)
        self.assertEqual(fit['mean'][-1], 0.0)
        held['logit_C'] = 1e6
        self.assertTrue(np.isfinite(core.logistic_logits(held[columns], fit)).all())
        again = core.fit_logistic(train[columns], self.y[self.groups != '7'], .1)
        self.assertEqual(fit, again)

    def test_serialized_prediction_matches_explicit_formula(self):
        candidate = core.candidates_for('S_DA', 'IEMOCAP')[-1]
        fit = json.loads(json.dumps(core.fit_candidate(self.frame, self.y, candidate)))
        p = fit['base']
        x = self.frame[fit['columns']].to_numpy()
        expected = expit(((x - p['mean']) / p['scale']) @ p['weights'] + p['intercept'])
        np.testing.assert_allclose(expected, core.predict(self.frame, fit), rtol=0, atol=1e-15)

    def test_offset_off_is_exact_static_fallback(self):
        candidate = core.candidates_for('S_DA_OFFSET', 'IEMOCAP')[2]
        self.assertEqual(candidate['penalty'], 'off')
        fit = core.fit_candidate(self.frame, self.y, candidate)
        expected = expit(core.logistic_logits(self.frame[fit['columns']], fit['base']))
        np.testing.assert_array_equal(core.predict(self.frame, fit), expected)
        self.assertEqual(fit['correction']['weights'], [0.0] * 4)

    def test_objective_and_gradient_are_registered_binary_logloss(self):
        x = self.frame[core.BEHAVIOR].to_numpy()
        offset = self.frame.logit_C.to_numpy()
        w = np.array([.1, -.2, .05, .3])
        lam = .1
        value, gradient = core.frozen.offset_objective(w, x, offset, self.y, lam)
        target = offset + x @ w
        expected = np.mean(np.logaddexp(0, target) - self.y * target) + lam * (w @ w) / 2
        self.assertAlmostEqual(value, expected, places=14)
        numerical = []
        for j in range(4):
            delta = np.eye(4)[j] * 1e-6
            numerical.append((core.frozen.offset_objective(w + delta, x, offset, self.y, lam)[0] - core.frozen.offset_objective(w - delta, x, offset, self.y, lam)[0]) / 2e-6)
        np.testing.assert_allclose(gradient, numerical, atol=1e-9)

    def test_tie_prefers_off_then_shrinkage(self):
        candidates = core.candidates_for('S_DA_OFFSET', 'IEMOCAP')
        self.assertEqual(len(candidates), 16)
        self.assertEqual(core.choose_candidate(candidates, np.ones(16)), 0)
        scores = np.ones(16)
        scores[1] -= 5e-13
        self.assertEqual(core.choose_candidate(candidates, scores), 0)
        scores[1] -= 2e-12
        self.assertEqual(core.choose_candidate(candidates, scores), 1)
        for method in ['T_DA', 'S_DA', 'S_RAND']:
            c = core.candidates_for(method, 'IEMOCAP')
            self.assertEqual(len(c), 8)
            self.assertTrue(all(x['mode'] == 'base' for x in c[:4]))

    def test_inner_training_scalers_and_pooled_selection(self):
        result = core.fit_nested_all(self.frame, self.y, self.groups, 'IEMOCAP')
        for inner in result['inner_folds']:
            tr, ho = inner['train_indices'], inner['held_indices']
            self.assertTrue(set(self.groups[tr]).isdisjoint(self.groups[ho]))
            for fits in inner['models'].values():
                for fit in fits:
                    expected = self.frame.iloc[tr][fit['columns']].mean()
                    np.testing.assert_allclose(fit['base']['mean'], expected, atol=1e-14)
        for method, registered in result['methods'].items():
            all_p = np.full((len(registered['candidates']), len(self.y)), np.nan)
            for inner in result['inner_folds']:
                for j, fit in enumerate(inner['models'][method]):
                    all_p[j, inner['held_indices']] = core.predict(self.frame.iloc[inner['held_indices']], fit)
            scores = np.mean((all_p - self.y[None, :]) ** 2, axis=1)
            np.testing.assert_allclose(scores, [x['binary_brier'] for x in registered['candidates']], atol=1e-15)
            self.assertEqual(registered['selected_index'], core.choose_candidate(registered['candidates'], scores))


if __name__ == '__main__':
    unittest.main()
