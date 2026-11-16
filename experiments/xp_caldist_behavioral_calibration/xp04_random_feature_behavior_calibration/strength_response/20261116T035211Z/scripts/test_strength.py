"""CPU synthetic checks of the preregistered perturbation/target protocol."""
import hashlib
import unittest

import numpy as np
import run_strength as run


class StrengthTests(unittest.TestCase):
    def fixture(self, dimension=100):
        x = np.arange(dimension, dtype=np.float32) + 1
        return x, {'audio_std': np.ones(dimension, np.float32),
                   'audio_mean': np.full(dimension, -7, np.float32)}

    def test_exact_dropout_count_and_training_mean(self):
        x, stats = self.fixture(100)
        result, audits = run.changed_vectors({5: x}, stats, 'MELD', 'dialogue', 2, 'dropout', 0)
        changed = result[5] != x
        self.assertEqual(int(changed.sum()), 10)
        np.testing.assert_array_equal(result[5][changed], stats['audio_mean'][changed])
        self.assertEqual(audits[0]['requested_count'], 10)
        self.assertEqual(audits[0]['actual_changed_count'], 10)
        self.assertEqual(audits[0]['vector_sha256'], hashlib.sha256(result[5].tobytes()).hexdigest())
        np.testing.assert_array_equal(x, np.arange(100, dtype=np.float32) + 1)

    def test_zero_std_unchanged_and_nonconstant_denominator(self):
        x, stats = self.fixture(100)
        stats['audio_std'][:50] = 0
        for kind in ('gaussian', 'dropout'):
            result, audits = run.changed_vectors({5: x}, stats, 'IEMOCAP', 'd', 0, kind, 0)
            np.testing.assert_array_equal(result[5][:50], x[:50])
            self.assertTrue(audits[0]['stdzero_unchanged'])
            self.assertEqual(audits[0]['eligible_count'], 50)
            if kind == 'dropout':
                self.assertEqual(audits[0]['requested_count'], 5)

    def test_actual_changed_can_differ_from_requested(self):
        x, stats = self.fixture()
        stats['audio_mean'] = x.copy()
        _, audits = run.changed_vectors({5: x}, stats, 'MELD', 'd', 0, 'dropout', 0)
        self.assertEqual(audits[0]['requested_count'], 10)
        self.assertEqual(audits[0]['actual_changed_count'], 0)

    def test_historical_canonical_rng_and_gaussian(self):
        x, stats = self.fixture()
        key = ['MELD', 1475, 'dλ', 3, 'audio', 'gaussian', 0.1, 9]
        canonical = '["MELD",1475,"dλ",3,"audio","gaussian",0.1,9]'
        self.assertEqual(run.canonical_key(key), canonical)
        seed = int.from_bytes(hashlib.sha256(canonical.encode()).digest()[:16], 'big')
        expected_rng = np.random.Generator(np.random.PCG64(seed))
        expected = (x.astype(np.float64) + .1 * expected_rng.normal(size=x.shape)).astype(np.float32)
        a, audit = run.changed_vectors({5: x}, stats, 'MELD', 'dλ', 3, 'gaussian', 9)
        b, _ = run.changed_vectors({5: x}, stats, 'MELD', 'dλ', 3, 'gaussian', 9)
        np.testing.assert_array_equal(a[5], expected)
        np.testing.assert_array_equal(a[5], b[5])
        self.assertEqual(audit[0]['rng_seed'], str(seed))

    def test_exact_historical_dropout_rng(self):
        x, stats = self.fixture(91)
        eligible = np.arange(91)
        key = run.key_for('MELD', 'd', 7, 'audio', 'dropout', 2)
        seed = int.from_bytes(hashlib.sha256(run.canonical_key(key).encode()).digest()[:16], 'big')
        chosen = np.random.Generator(np.random.PCG64(seed)).choice(eligible, size=round(.1 * len(eligible)), replace=False)
        expected = x.copy()
        expected[chosen] = -7
        actual, _ = run.changed_vectors({5: x}, stats, 'MELD', 'd', 7, 'dropout', 2)
        np.testing.assert_array_equal(actual[5], expected)

    def test_modality_mapping_missing_visual(self):
        self.assertEqual(run.MODALITIES, {'text': [0, 1, 2, 3], 'audio': [5], 'visual': [4], 'joint': [0, 1, 2, 3, 5, 4]})
        self.assertEqual(run.active_indices('visual', np.zeros(8)), ([], True))
        self.assertEqual(run.active_indices('joint', np.zeros(8)), ([0, 1, 2, 3, 5], True))
        self.assertEqual(run.active_indices('text', np.zeros(8)), ([0, 1, 2, 3], False))
        self.assertEqual(run.active_indices('visual', np.ones(8)), ([4], False))

    def test_target_context_masks_and_restore(self):
        import torch
        base = [torch.arange(24, dtype=torch.float32).reshape(3, 2, 4) for _ in range(7)]
        base += [torch.ones(2, 3), torch.arange(6).reshape(2, 3)]
        tensors = [x.clone() for x in base]
        value = np.full(4, -3, np.float32)
        tensors[5][1, 0].copy_(torch.from_numpy(value))
        run.assert_target_only(tensors, base, {5: value}, 1, 0, torch)
        tensors[6][0, 0, 0] = -1
        with self.assertRaises(AssertionError):
            run.assert_target_only(tensors, base, {5: value}, 1, 0, torch)
        tensors[6].copy_(base[6])
        tensors[5][1, 0].copy_(base[5][1, 0])
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(tensors, base)))


if __name__ == '__main__':
    unittest.main()
