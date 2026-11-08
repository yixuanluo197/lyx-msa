import hashlib
import json
import unittest

import numpy as np

from evidence_core import (attenuate, condition_specs, mask_rng, mask_vectors,
                           sensitivity_metrics, target_indices)


class EvidenceCoreTests(unittest.TestCase):
    def test_attenuation_endpoints_formula_and_no_mutation(self):
        x = np.array([4., -3., 8.], np.float32)
        mean = np.array([1., 2., 0.], np.float32)
        for rho, wanted in [(0., x), (1., mean), (.5, np.array([2.5, -.5, 4.], np.float32))]:
            actual = attenuate([x], [mean], rho)[0]
            np.testing.assert_array_equal(actual, wanted)
            self.assertEqual(actual.dtype, np.float32)
            self.assertFalse(np.shares_memory(actual, x))
            self.assertFalse(np.shares_memory(actual, mean))
        np.testing.assert_array_equal(x, [4, -3, 8])
        np.testing.assert_array_equal(mean, [1, 2, 0])

    def test_rng_canonical_replay_and_every_key_component(self):
        key = ['IEMOCAP', 1475, 'dialogue-a', 2, 'T-All', .1, 0]
        canonical = json.dumps(key, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        seed = int.from_bytes(hashlib.sha256(canonical.encode('utf-8')).digest()[:16], 'big')
        expected = np.random.Generator(np.random.PCG64(seed)).random(12)
        np.testing.assert_array_equal(mask_rng(*key).random(12), expected)
        np.testing.assert_array_equal(mask_rng(*key).random(12), mask_rng(*key).random(12))
        for idx, value in enumerate(['MELD', 67137, 'dialogue-b', 3, 'A', .3, 1]):
            changed = key.copy(); changed[idx] = value
            self.assertFalse(np.array_equal(expected, mask_rng(*changed).random(12)))

    def test_exact_mask_count_independent_layers_and_no_rescaling(self):
        xs = [np.arange(1, 16, dtype=np.float32) + layer * 20 for layer in range(4)]
        means = [np.full(15, -1-layer, np.float32) for layer in range(4)]
        copies = [x.copy() for x in xs]
        changed, selected = mask_vectors(xs, means, .3, np.random.default_rng(5))
        for layer, indices in enumerate(selected):
            self.assertEqual(len(indices), 5)  # floor(0.3*15 + .5)
            self.assertEqual(len(np.unique(indices)), 5)
            expected = xs[layer].copy(); expected[indices] = means[layer][indices]
            np.testing.assert_array_equal(changed[layer], expected)
            np.testing.assert_array_equal(xs[layer], copies[layer])
            self.assertEqual(changed[layer].dtype, np.float32)
            self.assertFalse(np.shares_memory(changed[layer], xs[layer]))
        self.assertFalse(np.array_equal(selected[0], selected[1]))

    def test_mask_boundaries_and_unequal_layer_dimensions(self):
        xs = [np.arange(3, dtype=np.float32), np.arange(9, dtype=np.float32)]
        means = [np.ones(3, np.float32), np.ones(9, np.float32)]
        for ratio in [0., 1.]:
            changed, indices = mask_vectors(xs, means, ratio, np.random.default_rng(1))
            for layer in range(2):
                self.assertEqual(len(indices[layer]), int(ratio * len(xs[layer])))
                np.testing.assert_array_equal(changed[layer], xs[layer] if ratio == 0 else means[layer])

    def test_targets_and_exact_condition_groups(self):
        for target, wanted in [('T1', [0]), ('T2', [1]), ('T3', [2]), ('T4', [3]),
                               ('T-All', [0, 1, 2, 3]), ('A', [5]), ('V', [4])]:
            self.assertEqual(target_indices(target), wanted)
        overall = condition_specs('overall_attenuation')
        layers = condition_specs('layer_attenuation')
        masks = condition_specs('random_masking')
        self.assertEqual([len(overall), len(layers), len(masks)], [12, 16, 28])
        all_specs = overall + layers + masks
        self.assertEqual(len({c['id'] for c in all_specs}), 56)
        self.assertIn('decay_T-All_rho0p25', {c['id'] for c in overall})
        self.assertIn('mask_T1_ratio0p10', {c['id'] for c in masks})
        self.assertTrue(all(c['repeats'] == 1 and c['kind'] == 'attenuation' for c in overall + layers))
        self.assertTrue(all(c['repeats'] == 5 and c['kind'] == 'random_masking' for c in masks))
        with self.assertRaises(ValueError): target_indices('pitch')
        with self.assertRaises(ValueError): condition_specs('joint')

    def test_known_sensitivity_metrics_and_transitions(self):
        p = np.array([[.8, .2], [.7, .3]], np.float32)
        q = np.array([[[.2, .8], [.9, .1]], [[.4, .6], [.8, .2]]], np.float32)
        labels = np.array([0, 1])
        p_copy, q_copy = p.copy(), q.copy()
        out = sensitivity_metrics(labels, p, q)
        np.testing.assert_array_equal(out['original_prediction'], [0, 0])
        np.testing.assert_array_equal(out['perturbed_prediction'], [[1, 0], [1, 0]])
        np.testing.assert_array_equal(out['flip'], [[True, False], [True, False]])
        np.testing.assert_array_equal(out['correct_to_wrong'], [[True, False], [False, False]])
        np.testing.assert_array_equal(out['wrong_to_correct'], [[False, False], [True, False]])
        np.testing.assert_allclose(out['confidence_change'], [[0, .1], [-.1, .1]], atol=1e-7)
        np.testing.assert_allclose(out['original_true_probability'], [.8, .3], atol=1e-7)
        np.testing.assert_allclose(out['perturbed_true_probability'], [[.2, .9], [.6, .2]], atol=1e-7)
        np.testing.assert_allclose(out['true_probability_change'], [[-.6, .1], [.3, -.1]], atol=1e-7)
        for i in range(2):
            for r in range(2):
                a, b = p[i].astype(np.float64), q[i, r].astype(np.float64)
                mid = (a+b)/2
                expected = .5*(np.sum(a*np.log(a/mid)) + np.sum(b*np.log(b/mid)))
                self.assertAlmostEqual(out['js_divergence'][i, r], expected, places=14)
        np.testing.assert_array_equal(p, p_copy)
        np.testing.assert_array_equal(q, q_copy)
        for name in ['original_confidence', 'perturbed_confidence', 'original_true_probability',
                     'perturbed_true_probability', 'confidence_change', 'true_probability_change', 'js_divergence']:
            self.assertEqual(out[name].dtype, np.float64)

    def test_zero_probability_js_and_normalization_roundoff(self):
        out = sensitivity_metrics([0], np.array([[1., 0.]]), np.array([[[0., 1.]]]))
        self.assertAlmostEqual(out['js_divergence'][0, 0], np.log(2))
        p = np.array([[.2, .3, .50000006]], np.float32)
        out = sensitivity_metrics([0], p, p[:, None, :])
        self.assertEqual(out['js_divergence'][0, 0], 0.)

    def test_invalid_inputs_fail(self):
        for rho in [-.1, 1.1, np.nan]:
            with self.assertRaises(ValueError): attenuate([np.ones(2)], [np.zeros(2)], rho)
        with self.assertRaises(ValueError): attenuate([np.ones(2)], [np.zeros(3)], .5)
        with self.assertRaises(ValueError): mask_vectors([np.ones(2)], [np.zeros(2)], 1.1, np.random.default_rng(0))
        for p in [np.array([[np.nan, 0.]]), np.array([[-.1, 1.1]]), np.array([[.2, .2]])]:
            with self.assertRaises(ValueError): sensitivity_metrics([0], p, p[:, None, :])
        p = np.array([[.8, .2]])
        for labels in [[2], [-1], [0.5], [0, 1]]:
            with self.assertRaises(ValueError): sensitivity_metrics(labels, p, p[:, None, :])


if __name__ == '__main__':
    unittest.main()
