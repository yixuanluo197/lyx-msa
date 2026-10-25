"""Pure synthetic sampling counterexamples; no model or data-file access."""
from __future__ import annotations

import importlib.util
import argparse
import hashlib
import io
import json
import os
import random
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


MODULE_PATH = Path(__file__).with_name('prepare_diagnostic.py')
if MODULE_PATH.exists():
    spec = importlib.util.spec_from_file_location('prepare_diagnostic', MODULE_PATH)
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
else:
    implementation = None


def fixture(n=640):
    ids = list(range(n))
    # Distinct lengths preserve a known rank ordering; all seven classes occur.
    labels = {i: [(i + j) % 7 for j in range(2 + i // 160)] for i in ids}
    return ids, labels


class SamplingTests(unittest.TestCase):
    def sample(self, ids, labels, seed=67137):
        self.assertIsNotNone(implementation, 'Sampling implementation is missing')
        return implementation.select_batches(ids, labels, selection_seed=seed)

    def test_reproducible_private_rng_and_no_input_mutation(self):
        ids, labels = fixture()
        original = json.dumps([ids, labels], sort_keys=True)
        before = random.getstate()
        one = self.sample(ids, labels)
        two = self.sample(ids, labels)
        self.assertEqual(one, two)
        self.assertEqual(before, random.getstate())
        self.assertEqual(original, json.dumps([ids, labels], sort_keys=True))

    def test_exact_32_batches_512_distinct_dialogues(self):
        ids, labels = fixture()
        result = self.sample(ids, labels)
        self.assertEqual(len(result['batches']), 32)
        chosen = [i for b in result['batches'] for i in b['dialogue_ids']]
        self.assertEqual(len(chosen), 512)
        self.assertEqual(len(set(chosen)), 512)
        self.assertEqual([b['batch_id'] for b in result['batches']],
                         [f'B{i:02d}' for i in range(1, 33)])
        self.assertTrue(all(len(b['dialogue_ids']) == 16 for b in result['batches']))

    def test_rank_strata_and_all_available_classes_covered(self):
        ids, labels = fixture()
        result = self.sample(ids, labels)
        ordered = sorted(ids, key=lambda k: (len(labels[k]), str(k)))
        for q in range(4):
            eligible = ordered[q * len(ids) // 4:(q + 1) * len(ids) // 4]
            batches = [b for b in result['batches'] if b['length_stratum'] == q + 1]
            chosen = [i for b in batches for i in b['dialogue_ids']]
            self.assertEqual(len(batches), 8)
            self.assertTrue(set(chosen) <= set(eligible))
            self.assertEqual({e for i in chosen for e in labels[i]},
                             {e for i in eligible for e in labels[i]})

    def test_never_accesses_validation_or_test_labels(self):
        ids, base = fixture()
        class TrainOnly(dict):
            def __getitem__(self, key):
                if key not in ids:
                    raise AssertionError('Attempted nontraining label access')
                return super().__getitem__(key)
        labels = TrainOnly(base)
        labels['validation_poison'] = object()
        labels['test_poison'] = object()
        result = self.sample(ids, labels)
        self.assertTrue(all(set(b['dialogue_ids']) <= set(ids) for b in result['batches']))

    def test_rare_available_class_is_not_lost(self):
        ids, labels = fixture()
        labels = {i: [0, 1] for i in ids}
        ordered = sorted(ids, key=str)
        rare = [ordered[(q + 1) * len(ids) // 4 - 1] for q in range(4)]
        # Keep equal lengths so class presence cannot move length strata.
        for i in rare:
            labels[i] = [0, 6]
        result = self.sample(ids, labels)
        chosen = {i for b in result['batches'] for i in b['dialogue_ids']}
        self.assertTrue(set(rare) <= chosen)

    def test_unavailable_classes_are_not_required(self):
        ids, _ = fixture()
        labels = {i: [0] for i in ids}
        result = self.sample(ids, labels)
        self.assertTrue(all(b['emotion_counts'][0] == 16 for b in result['batches']))
        self.assertTrue(all(sum(b['emotion_counts'][1:]) == 0 for b in result['batches']))

    def test_seed_changes_selection_without_changing_budget(self):
        ids, labels = fixture()
        one, two = self.sample(ids, labels), self.sample(ids, labels, seed=123)
        self.assertNotEqual(one['batches'], two['batches'])
        self.assertEqual(len(one['batches']), len(two['batches']))

    def test_capacity_violation_fails_without_resampling(self):
        ids, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'capacity'):
            self.sample(ids, {i: [0] * 100 for i in ids})

    def test_insufficient_stratum_fails(self):
        ids, labels = fixture(256)
        with self.assertRaisesRegex(ValueError, '128'):
            self.sample(ids, labels)

    def test_duplicate_ids_fail(self):
        ids, labels = fixture()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.sample(ids + [ids[0]], labels)

    def test_invalid_or_empty_label_sequence_fails(self):
        ids, labels = fixture()
        for bad in ([7], [-1], []):
            with self.subTest(bad=bad):
                modified = {**labels, ids[0]: bad}
                with self.assertRaises(ValueError):
                    self.sample(ids, modified)

    def test_public_batch_summary_excludes_dialogue_ids(self):
        ids, labels = fixture()
        result = self.sample(ids, labels)
        self.assertIsNotNone(implementation)
        public = implementation.public_batches(result)
        self.assertEqual(len(public), 32)
        self.assertTrue(all('dialogue_ids' not in b for b in public))
        self.assertTrue(all(len(b['dialogue_ids_sha256']) == 64 for b in public))
        self.assertTrue(all(sum(b['emotion_counts']) == b['n_utterances'] for b in public))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path)
    options = parser.parse_args()
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(SamplingTests))
    sys.stderr.write(output.getvalue())
    if options.evidence is not None:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        assert 'torch' not in sys.modules, 'Synthetic sampling tests must not import torch'
        options.evidence.parent.mkdir(parents=True, exist_ok=True)
        with options.evidence.open('x') as stream:
            json.dump({'status': 'passed' if result.wasSuccessful() else 'failed',
                'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
                'cuda_initialized': False, 'torch_imported': False,
                'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'Pure synthetic stdlib CPU sampling; no model, real data, or gradient access.',
                'red_stage': {'tests': 12, 'failures': 14, 'errors': 0,
                              'reason': 'Expected missing-implementation assertions before prepare_diagnostic.py existed.'},
                'test_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'implementation_sha256': hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest(),
                'output': output.getvalue()}, stream, indent=2)
            stream.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
