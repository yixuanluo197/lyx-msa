"""Synthetic tests only: no frozen feature/media/label files are opened."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_clean_reconstruction as evaluator


def protocol():
    return {'feature_gate': {'scale_floor': 1e-8,
        'all_elements_allclose': {'atol': 1e-4, 'rtol': 1e-4},
        'MAE_max': 1e-5, 'train_scaled_RMSE_max': .001,
        'mean_nonzero_row_cosine_min': .99999,
        'min_nonzero_row_cosine_min': .9999,
        'nonconstant_dimension_correlation_p05_min': .999}}


class SyntheticEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.target = np.array([[0., 1., 7.], [1., 2., 7.], [2., 4., 7.], [3., 8., 7.]])
        self.scale = np.array([1., 1., 0.])
        self.groups = np.array(['a', 'a', 'b', 'b'])

    def test_exact_and_constant_column(self):
        summary, columns = evaluator.metric_gate(self.target, self.target.copy(), self.scale,
                                                  self.groups, protocol())
        self.assertTrue(summary['passed'])
        self.assertEqual(summary['target_constant_dimensions'], 1)
        self.assertEqual(summary['mean_absolute_error'], 0)
        self.assertTrue(np.isnan(columns['column_correlation'][2]))
        self.assertEqual(columns['column_wasserstein'].tolist(), [0., 0., 0.])

    def test_missing_row_cannot_pass_finite_subset(self):
        prediction = self.target.copy()
        prediction[0] = np.nan
        summary, _ = evaluator.metric_gate(self.target, prediction, self.scale, self.groups, protocol())
        self.assertFalse(summary['passed'])
        self.assertEqual(summary['fully_finite_rows'], 3)
        self.assertEqual(summary['nonfinite_rows'], 1)
        self.assertEqual(summary['mean_absolute_error'], 0)
        self.assertFalse(summary['gate_checks']['all_elements_allclose'])

    def test_constant_column_difference_is_checked(self):
        prediction = self.target.copy()
        prediction[:, 2] += 1e-6
        summary, _ = evaluator.metric_gate(self.target, prediction, self.scale, self.groups, protocol())
        self.assertFalse(summary['passed'])
        self.assertTrue(summary['gate_checks']['all_elements_allclose'])
        self.assertFalse(summary['gate_checks']['train_scaled_rmse'])

    def test_collapsed_prediction_cannot_disappear_from_correlation_gate(self):
        target = self.target.copy()
        target[:, 0] = 1 + np.arange(4) * 1e-7
        prediction = target.copy()
        prediction[:, 0] = target[:, 0].mean()
        summary, columns = evaluator.metric_gate(target, prediction, np.ones(3), self.groups, protocol())
        self.assertTrue(summary['gate_checks']['all_elements_allclose'])
        self.assertTrue(summary['gate_checks']['mae'])
        self.assertEqual(summary['target_nonconstant_but_prediction_constant_dimensions'], 1)
        self.assertEqual(summary['target_nonconstant_but_correlation_undefined_dimensions'], 1)
        self.assertFalse(summary['gate_checks']['required_target_correlations_defined'])
        self.assertTrue(columns['column_required_correlation_undefined'][0])
        self.assertFalse(summary['passed'])

    def test_missing_lock_never_loads_target_pickle(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(evaluator, 'load_pickle', side_effect=AssertionError('No pickle expected')):
                with self.assertRaises(FileNotFoundError):
                    evaluator.preflight(Path(temporary), 'IEMOCAP')

    def test_changed_protocol_rejected_before_any_pickle(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            provenance = out / 'provenance'
            provenance.mkdir()
            candidate = out / 'local_only/IEMOCAP/candidates/is10'
            candidate.mkdir(parents=True)
            lock = {'dataset': 'IEMOCAP', 'validation_unopened': True, 'candidate': 'is10'}
            for key in ['converter_path', 'manifest_path', 'protocol_path', 'train_scale_path']:
                path = out / key
                path.write_bytes(b'synthetic input')
                lock[key] = str(path)
            for key, name in [('validation_features_path', 'validation.npy'),
                              ('validation_valid_path', 'validation_valid.npy')]:
                path = candidate / name
                path.write_bytes(b'synthetic input')
                lock[key] = str(path)
            lock.update(converter_sha256=evaluator.sha(lock['converter_path']), protocol_sha256='changed')
            evaluator.write_json(provenance / 'IEMOCAP_model_lock.json', lock)
            with patch.object(evaluator, 'load_pickle', side_effect=AssertionError('No pickle expected')):
                with self.assertRaisesRegex(ValueError, 'Protocol changed'):
                    evaluator.preflight(out, 'IEMOCAP')

    def test_marker_complete_forbids_second_evaluation(self):
        binding = {'lock_sha256': 'synthetic', 'inputs': {}}
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            path, record = evaluator.evaluation_marker(out, 'IEMOCAP', binding, False)
            record['status'] = 'COMPLETE'
            evaluator.write_json(path, record)
            for resume in [False, True]:
                with self.assertRaises(ValueError):
                    evaluator.evaluation_marker(out, 'IEMOCAP', binding, resume)

    def test_io_resume_requires_same_binding_and_preserves_failure(self):
        binding = {'lock_sha256': 'synthetic', 'inputs': {'features': 'abc'}}
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            path, record = evaluator.evaluation_marker(out, 'MELD', binding, False)
            record['status'] = 'FAILED_IO'
            evaluator.write_json(path, record)
            with self.assertRaises(ValueError):
                evaluator.evaluation_marker(out, 'MELD', {'lock_sha256': 'changed'}, True)
            _, resumed = evaluator.evaluation_marker(out, 'MELD', binding, True)
            self.assertEqual(resumed['attempt'], 2)
            self.assertEqual(resumed['previous_failed_attempt']['status'], 'FAILED_IO')

    def test_json_has_null_not_nan(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'test.json'
            evaluator.write_json(path, {'undefined': np.nan})
            self.assertIsNone(json.loads(path.read_text())['undefined'])


if __name__ == '__main__':
    unittest.main()
