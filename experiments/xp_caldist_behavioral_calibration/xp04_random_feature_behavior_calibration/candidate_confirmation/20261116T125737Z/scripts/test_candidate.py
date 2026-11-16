"""Synthetic behavioral invariants and exact canonical reference alignment checks."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import run_candidate as run


class CandidateTests(unittest.TestCase):
    def fixture(self):
        return np.arange(100, dtype=np.float32), {'audio_std': np.ones(100), 'audio_mean': np.zeros(100)}

    def test_gaussian_matches_historical_rng_and_arithmetic(self):
        x, stats = self.fixture()
        key = ['MELD', 1475, 'dλ', 3, 'audio', 'gaussian', .05, 9]
        canonical = '["MELD",1475,"dλ",3,"audio","gaussian",0.05,9]'
        self.assertEqual(run.canonical_key(key), canonical)
        seed = int.from_bytes(hashlib.sha256(canonical.encode()).digest()[:16], 'big')
        expected = (x.astype(np.float64) + .05 * np.random.Generator(np.random.PCG64(seed)).normal(size=x.shape)).astype(np.float32)
        result, audit = run.changed_vectors({5: x}, stats, 'MELD', 'dλ', 3, 'gaussian', 9)
        np.testing.assert_array_equal(expected, result[5])
        self.assertEqual(audit[0]['rng_seed'], str(seed))
        self.assertEqual(audit[0]['vector_sha256'], hashlib.sha256(expected.tobytes()).hexdigest())

    def test_identical_feature_bytes_across_model_seeds(self):
        x, stats = self.fixture()
        hashes = []
        for model_seed in run.SEEDS:
            # Model seed cannot enter the perturbation function or key.
            result, audit = run.changed_vectors({5: x}, stats, 'MELD', 'd', 2, 'gaussian', 4)
            hashes.append(audit[0]['vector_sha256'])
            self.assertEqual(json.loads(audit[0]['rng_key'])[1], 1475)
        self.assertEqual(len(set(hashes)), 1)

    def test_disjoint_repeat_blocks_and_local_index_mapping(self):
        x, stats = self.fixture()
        keys, hashes = [], []
        for block in range(3):
            for local_repeat in range(10):
                global_repeat = block * run.REPEATS + local_repeat
                result, audit = run.changed_vectors({5: x}, stats, 'MELD', 'd', 2, 'gaussian', global_repeat)
                keys.append(audit[0]['rng_key'])
                hashes.append(audit[0]['vector_sha256'])
                self.assertEqual(json.loads(audit[0]['rng_key'])[-1], global_repeat)
        self.assertEqual(len(set(keys)), 30)
        self.assertEqual(len(set(hashes)), 30)

    def test_training_zero_std_unchanged_and_input_untouched(self):
        x, stats = self.fixture()
        original = x.copy()
        stats['audio_std'][::2] = 0
        result, audits = run.changed_vectors({5: x}, stats, 'MELD', 'd', 2, 'gaussian', 0)
        np.testing.assert_array_equal(result[5][::2], original[::2])
        np.testing.assert_array_equal(x, original)
        self.assertTrue(audits[0]['stdzero_unchanged'])
        self.assertEqual(audits[0]['eligible_count'], 50)
        self.assertTrue(np.any(result[5][1::2] != x[1::2]))

    def test_text_all_four_arrays_independent(self):
        x = np.arange(100, dtype=np.float32)
        stats = {f'text{i}_{kind}': np.ones(100) for i in range(4) for kind in ['std', 'mean']}
        result, audit = run.changed_vectors({i: x.copy() for i in range(4)}, stats, 'IEMOCAP', 'd', 1, 'gaussian', 0)
        self.assertEqual(set(result), {0, 1, 2, 3})
        self.assertEqual(len(set(a['vector_sha256'] for a in audit)), 4)
        self.assertEqual(run.CANDIDATES, {'IEMOCAP': 'text', 'MELD': 'audio'})
        self.assertEqual(run.active_indices('text', np.zeros(8)), ([0, 1, 2, 3], False))
        self.assertEqual(run.active_indices('audio', np.zeros(8)), ([5], False))

    def test_target_masks_context_padding_and_restore(self):
        import torch
        base = [torch.arange(24, dtype=torch.float32).reshape(3, 2, 4) for _ in range(7)]
        base += [torch.tensor([[1., 1., 0.], [1., 1., 1.]]), torch.arange(6).reshape(2, 3)]
        tensors = [x.clone() for x in base]
        value = np.full(4, -3, np.float32)
        tensors[5][1, 0].copy_(torch.from_numpy(value))
        run.assert_target_only(tensors, base, {5: value}, 1, 0, torch)
        for index in [0, 6, 7, 8]:
            corrupt = [x.clone() for x in tensors]
            corrupt[index].reshape(-1)[0] = -777
            with self.assertRaises(AssertionError):
                run.assert_target_only(corrupt, base, {5: value}, 1, 0, torch)
        with self.assertRaises(AssertionError):
            run.assert_target_only(tensors, base, {5: value}, 2, 0, torch)
        tensors[5][1, 0].copy_(base[5][1, 0])
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(tensors, base)))

    def test_prediction_and_audit_writer_schema_contains_block(self):
        import ast
        import io
        tree = ast.parse(Path(run.__file__).read_text())
        assignments = {node.targets[0].id: node.value for node in ast.walk(tree)
                       if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
        fields = ast.literal_eval(assignments['fields'])
        schema = eval(compile(ast.Expression(assignments['audit_fields']), '<audit schema>', 'eval'), {'fields': fields})
        x, stats = self.fixture()
        _, audits = run.changed_vectors({5: x}, stats, 'MELD', 'd', 2, 'gaussian', 17)
        base = dict(dataset='MELD', seed=1475, dialogue_id='d', utterance_index=2,
                    modality='audio', corruption='gaussian', strength=.05, repeat_id=7,
                    rng_repeat_id=17, block=1)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=schema)
        writer.writeheader()
        writer.writerow({**base, **audits[0]})
        stream.seek(0)
        row = next(csv.DictReader(stream))
        self.assertEqual(row['block'], '1')
        self.assertEqual(row['rng_repeat_id'], '17')
        self.assertEqual(row['repeat_id'], '7')

    def test_launcher_matrix_has_exact_preregistered_jobs(self):
        import launch_candidates
        matrix = launch_candidates.jobs()
        self.assertEqual(len(matrix), 12)
        self.assertEqual(len({(j['dataset'], j['seed'], j['block']) for j in matrix}), 12)
        for ds, modality in run.CANDIDATES.items():
            subset = [j for j in matrix if j['dataset'] == ds]
            self.assertEqual({(j['seed'], j['block']) for j in subset},
                             {(67137, 0), (2024, 0), (3407, 0), (42, 0), (1475, 1), (1475, 2)})
            self.assertTrue(all(j['modality'] == modality for j in subset))

    def test_refuses_overwrite_before_loading_or_inference(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dest = root / 'runs/MELD/seed2024/block0'
            dest.mkdir(parents=True)
            old = dest / 'original_predictions.csv'
            old.write_text('preserve-existing-artifact')
            with patch.object(run, 'ROOT', root), patch.object(run.xp, 'imports') as imports:
                with self.assertRaisesRegex(AssertionError, 'Run exists'):
                    run.run(SimpleNamespace(dataset='MELD', seed=2024, block=0, smoke=False))
                imports.assert_not_called()
            self.assertEqual(old.read_text(), 'preserve-existing-artifact')

    def test_reject_other_corruption_and_bad_stat_shape(self):
        x, stats = self.fixture()
        with self.assertRaises(ValueError):
            run.changed_vectors({5: x}, stats, 'MELD', 'd', 2, 'dropout', 0)
        stats['audio_std'] = np.ones(99)
        with self.assertRaises(AssertionError):
            run.changed_vectors({5: x}, stats, 'MELD', 'd', 2, 'gaussian', 0)


if __name__ == '__main__':
    unittest.main()
