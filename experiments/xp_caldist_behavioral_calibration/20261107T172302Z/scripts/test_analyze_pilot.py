"""Scientific regression checks for validation-only diagnostics."""
import unittest
import csv
import json
import tempfile
from pathlib import Path
import numpy as np
from analyze_pilot import (multiclass_metrics, repeated_metrics, confidence_bin,
                           lodo_success, paired_dialogue_bootstrap, validate_payload, run)


class AnalysisTests(unittest.TestCase):
    def test_multiclass_brier_and_nll_use_full_distribution(self):
        y = np.array([0, 1])
        p = np.array([[0.8, 0.2], [0.4, 0.6]])
        m = multiclass_metrics(y, p)
        self.assertAlmostEqual(m['brier'], .20)
        self.assertAlmostEqual(m['nll'], -np.log([.8, .6]).mean())
        self.assertTrue(np.isnan(m['auroc_success']))

    def test_repeated_metrics_do_not_average_probabilities(self):
        y = np.array([0, 1])
        q = np.array([[[.9, .1], [.1, .9]], [[.1, .9], [.9, .1]]])
        result = repeated_metrics(y, q)
        self.assertAlmostEqual(result['brier'], .82)
        self.assertAlmostEqual(result['nll'], -np.log([.9, .1]).mean())
        self.assertNotAlmostEqual(result['brier'], multiclass_metrics(y, q.mean(axis=1))['brier'])

    def test_fixed_confidence_bins_include_upper_edge(self):
        np.testing.assert_array_equal(confidence_bin([0., .1, .999, 1.]), [0, 1, 9, 9])

    def test_lodo_handles_single_class_training_without_leakage(self):
        result = lodo_success(np.array([.6, .8, .9]), np.zeros((3, 3)),
                              np.array([0, 1, 1]), np.array(['a', 'b', 'b']))
        self.assertTrue(np.isnan(result['base']).all())
        self.assertTrue(np.isnan(result['behavior']).all())
        self.assertEqual(result['skipped_folds'], 2)

    def test_lodo_held_out_labels_do_not_change_held_out_predictions(self):
        confidence = np.linspace(.52, .95, 8)
        behavior = np.arange(24).reshape(8, 3) / 24
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        dialogues = np.repeat(['a', 'b', 'c', 'd'], 2)
        a = lodo_success(confidence, behavior, labels, dialogues)
        changed = labels.copy(); changed[:2] = 1 - changed[:2]
        b = lodo_success(confidence, behavior, changed, dialogues)
        np.testing.assert_allclose(a['base'][:2], b['base'][:2])
        np.testing.assert_allclose(a['behavior'][:2], b['behavior'][:2])
        self.assertTrue(np.isfinite(a['behavior']).all())

    def test_bootstrap_is_paired_and_reports_zero_for_identical_methods(self):
        result = paired_dialogue_bootstrap(np.array([0, 1, 0, 1]),
                    np.array([.2, .7, .3, .8]), np.array([.2, .7, .3, .8]),
                    np.array(['a', 'a', 'b', 'b']), repeats=50, seed=20261107)
        for metric in ['auroc', 'brier', 'nll']:
            self.assertEqual(result[metric]['difference'], 0.)
            self.assertEqual(result[metric]['ci_low'], 0.)
            self.assertEqual(result[metric]['ci_high'], 0.)

    def test_offline_report_end_to_end_uses_only_validation_inputs(self):
        # Synthetic data live only in an automatically deleted temporary fixture.
        # They are never stored in the experimental runs or reported as results.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            config = dict(pilots={'IEMOCAP': 1475}, modality_order=['audio'],
                          perturbations={'attenuation': {'strength': .95, 'repeats': 1}},
                          statistics={'split': 'train'}, no_calibration_fit=True,
                          diagnostics={'ece_bins': 15, 'bootstrap_unit': 'dialogue',
                                       'bootstrap_repeats': 5, 'bootstrap_seed': 20261107})
            (root / 'configs/perturbation_config.json').write_text(json.dumps(config))
            run_dir = root / 'runs/IEMOCAP/seed1475'
            (run_dir / 'perturbations').mkdir(parents=True)
            n = 527
            p = np.tile([.7, .3], (n, 1))
            data = dict(labels=np.arange(n) % 2, predictions=p.argmax(1), probabilities=p,
                        log_probabilities=np.log(p), dialogue_ids=np.array([str(i % 12) for i in range(n)]),
                        utterance_indices=np.arange(n), speaker_ids=np.zeros(n, dtype=int),
                        batch_indices=np.zeros(n, dtype=int), dialogue_lengths=np.full(n, 44))
            np.savez(run_dir / 'validation_predictions.npz', **data)
            q = np.tile([.65, .35], (n, 1, 1))
            payload = dict(data, probabilities=q, log_probabilities=np.log(q), original_probabilities=p)
            np.savez(run_dir / 'perturbations/audio_attenuation.npz', **payload)
            result = run(root)
            self.assertEqual(result['sample_rows'], n)
            self.assertEqual(result['summary_rows'], 1)
            self.assertTrue((root / 'reports/SINGLE_MODAL_PILOT_REPORT.md').exists())
            with (root / 'results/modality_pilot_summary.csv').open() as handle:
                summary = next(csv.DictReader(handle))
            self.assertEqual(int(summary['stable_wrong_count']), 263)
            self.assertEqual(int(summary['correct_sensitive_count']), 0)
            self.assertAlmostEqual(float(summary['confidence_change']), -.05)

    def test_payload_rejects_reordered_targets_and_wrong_repetitions(self):
        baseline = dict(labels=np.array([0, 1]), probabilities=np.array([[.6, .4], [.4, .6]]),
                        dialogue_ids=np.array(['a', 'b']), utterance_indices=np.array([0, 0]))
        payload = dict(baseline, original_probabilities=baseline['probabilities'],
                       probabilities=np.ones((2, 5, 2)) / 2)
        validate_payload(baseline, payload, repeats=5)
        payload['dialogue_ids'] = np.array(['b', 'a'])
        with self.assertRaises(ValueError):
            validate_payload(baseline, payload, repeats=5)
        payload['dialogue_ids'] = baseline['dialogue_ids']
        with self.assertRaises(ValueError):
            validate_payload(baseline, payload, repeats=1)


if __name__ == '__main__':
    unittest.main()
