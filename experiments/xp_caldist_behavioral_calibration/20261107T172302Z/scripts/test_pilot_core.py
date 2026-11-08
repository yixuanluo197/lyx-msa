import hashlib
import json
import unittest

import numpy as np

from pilot_core import behavior_metrics, perturb_vectors, rng_for


class PilotCoreTests(unittest.TestCase):
    def test_rng_replay_key_isolation_and_canonical_seed(self):
        key = ['IEMOCAP', 1475, 'dialogue-a', 3, 'audio', 'gaussian', 0.05, 0]
        raw = json.dumps(key, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        seed = int.from_bytes(hashlib.sha256(raw.encode('utf-8')).digest()[:16], 'big')
        expected = np.random.Generator(np.random.PCG64(seed)).normal(size=10)
        np.testing.assert_array_equal(rng_for(*key).normal(size=10), expected)
        np.testing.assert_array_equal(rng_for(*key).normal(size=10), rng_for(*key).normal(size=10))
        for idx, replacement in enumerate(['MELD', 42, 'dialogue-b', 4, 'visual', 'dropout', 0.1, 1]):
            changed = key.copy()
            changed[idx] = replacement
            self.assertFalse(np.array_equal(expected, rng_for(*changed).normal(size=10)))

    def test_zero_noise_dropout_and_unit_attenuation_are_identity(self):
        x = np.array([1, -2, 0], dtype=np.float32)
        for kind, strength in [('gaussian', 0), ('dropout', 0), ('attenuation', 1)]:
            out = perturb_vectors([x], [np.ones(3)], kind, strength, np.random.default_rng(1))[0]
            np.testing.assert_array_equal(x, out)
            self.assertEqual(out.dtype, np.float32)
            self.assertFalse(np.shares_memory(x, out))

    def test_layers_use_own_std_independent_draws_and_leave_input_unchanged(self):
        vectors = [np.ones(4, np.float32) for _ in range(4)]
        stds = [np.full(4, value, dtype=np.float32) for value in [0, 1, 2, 3]]
        expected_rng = np.random.default_rng(7)
        expected = [(x.astype(np.float64) + 0.05 * std.astype(np.float64) * expected_rng.normal(size=x.shape)).astype(np.float32)
                    for x, std in zip(vectors, stds)]
        out = perturb_vectors(vectors, stds, 'gaussian', 0.05, np.random.default_rng(7))
        for x, std, actual, wanted in zip(vectors, stds, out, expected):
            np.testing.assert_array_equal(actual, wanted)
            np.testing.assert_array_equal(x, np.ones(4))
            self.assertFalse(np.shares_memory(x, actual))
        np.testing.assert_array_equal(stds[2], np.full(4, 2))

    def test_dropout_no_inverse_rescaling_and_attenuation(self):
        x = np.arange(1, 101, dtype=np.float32)
        out = perturb_vectors([x], [np.ones_like(x)], 'dropout', 0.5, np.random.default_rng(5))[0]
        self.assertTrue(np.any(out == 0))
        self.assertTrue(np.any(out != 0))
        np.testing.assert_array_equal(out[out != 0], x[out != 0])
        actual = perturb_vectors([x], [np.ones_like(x)], 'attenuation', 0.95, np.random.default_rng(0))[0]
        np.testing.assert_array_equal(actual, (x.astype(np.float64) * 0.95).astype(np.float32))

    def test_known_behavior_metrics_and_input_preservation(self):
        original = np.array([[0.8, 0.2]], dtype=np.float32)
        perturbed = np.array([[[0.2, 0.8], [0.7, 0.3]]], dtype=np.float32)
        original_copy, perturbed_copy = original.copy(), perturbed.copy()
        result = behavior_metrics(original, perturbed)
        self.assertAlmostEqual(result['flip_rate'][0], 0.5)
        self.assertAlmostEqual(result['mu'][0], 0.4)
        self.assertAlmostEqual(result['delta_original'][0], 0.05)
        self.assertAlmostEqual(result['confidence_mad'][0], 0.05)
        self.assertAlmostEqual(result['confidence_change'][0], -0.05)
        p = original.astype(np.float64)[0]
        js = []
        for q in perturbed.astype(np.float64)[0]:
            m = (p + q) / 2
            js.append((np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))) / 2)
        self.assertAlmostEqual(result['js_divergence'][0], np.mean(js), places=14)
        self.assertTrue(all(v.dtype == np.float64 for v in result.values()))
        np.testing.assert_array_equal(original, original_copy)
        np.testing.assert_array_equal(perturbed, perturbed_copy)

    def test_delta_vs_mad_cancellation_and_zero_probabilities(self):
        original = np.array([[0.8, 0.2], [1., 0.]])
        perturbed = np.array([[[0.9, 0.1], [0.7, 0.3]], [[1., 0.], [0., 1.]]])
        result = behavior_metrics(original, perturbed)
        self.assertAlmostEqual(result['delta_original'][0], 0)
        self.assertAlmostEqual(result['confidence_mad'][0], 0.1)
        self.assertAlmostEqual(result['js_divergence'][1], np.log(2) / 2)
        identity = behavior_metrics(original, original[:, None, :])
        for values in identity.values():
            np.testing.assert_array_equal(values, np.zeros(2))

    def test_invalid_inputs_rejected_and_float32_normalization_tolerated(self):
        p = np.array([[0.2, 0.3, 0.50000006]], dtype=np.float32)
        behavior_metrics(p, p[:, None, :])
        for bad in [np.array([[np.nan, 0]]), np.array([[np.inf, 0]]),
                    np.array([[-0.1, 1.1]]), np.array([[0.2, 0.2]])]:
            with self.assertRaises(ValueError):
                behavior_metrics(bad, bad[:, None, :])
        with self.assertRaises(ValueError):
            behavior_metrics(p, np.empty((1, 0, 3)))
        for kind, strength in [('unknown', 0.05), ('dropout', 1.1), ('gaussian', -1), ('attenuation', -0.1)]:
            with self.assertRaises(ValueError):
                perturb_vectors([np.ones(3)], [np.ones(3)], kind, strength, np.random.default_rng(1))
        with self.assertRaises(ValueError):
            perturb_vectors([np.ones(3)], [np.ones(2)], 'gaussian', 0.05, np.random.default_rng(1))
        with self.assertRaises(ValueError):
            perturb_vectors([np.ones(3)], [np.full(3, -1)], 'gaussian', 0.05, np.random.default_rng(1))


if __name__ == '__main__':
    unittest.main()
