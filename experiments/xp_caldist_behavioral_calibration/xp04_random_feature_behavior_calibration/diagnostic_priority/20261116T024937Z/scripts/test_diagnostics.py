"""Focused mathematical checks for the frozen descriptive diagnostics."""
import itertools
import unittest

import numpy as np
import pandas as pd

import diagnose as d


class DiagnosticTests(unittest.TestCase):
    def test_matching_maximum_cardinality_then_distance(self):
        cases = [([.5, .52], [.48, .501]), ([.1, .11, .12], [.105, .14]),
                 ([.1, .5], [.12, .7]), ([.5], [.51, .49])]
        for correct, wrong in cases:
            pairs = d.match_confidences(correct, wrong)
            possibilities = []
            for count in range(min(len(correct), len(wrong)) + 1):
                for left in itertools.combinations(range(len(correct)), count):
                    for right in itertools.permutations(range(len(wrong)), count):
                        costs = [abs(correct[i] - wrong[j]) for i, j in zip(left, right)]
                        if all(x <= .02 for x in costs):
                            possibilities.append((-count, sum(costs)))
            optimum = min(possibilities)
            self.assertEqual(len(pairs), -optimum[0])
            self.assertAlmostEqual(sum(abs(correct[i] - wrong[j]) for i,j in pairs), optimum[1])
            self.assertEqual(len({i for i,j in pairs}), len(pairs))
            self.assertEqual(len({j for i,j in pairs}), len(pairs))
            self.assertEqual(pairs, d.match_confidences(correct, wrong))

    def test_matching_empty_and_caliper(self):
        self.assertEqual(d.match_confidences([], [.5]), [])
        self.assertEqual(d.match_confidences([.5], [.521]), [])
        self.assertEqual(d.match_confidences([.5], [.51]), [(0, 0)])

    def test_fixed_bin_boundaries(self):
        self.assertEqual(d.confidence_bins([0, .399, .4, .6, .8, 1]).tolist(), [0,0,1,2,3,3])
        with self.assertRaises(ValueError):
            d.confidence_bins([1.01])

    def test_population_sd_conversion_and_zero_se(self):
        sample_sd, se, ratio, zero = d.noise_statistics(np.array([.3, 0]), np.array([.1, 0]))
        np.testing.assert_allclose(sample_sd, [.3*np.sqrt(10/9),0])
        np.testing.assert_allclose(se, [.1,0])
        self.assertAlmostEqual(ratio[0], 1.)
        self.assertTrue(np.isnan(ratio[1]))
        np.testing.assert_array_equal(zero, [False,True])

    def test_constant_correlation_is_na(self):
        self.assertTrue(np.isnan(d.correlation([0,0,0],[1,2,3], 'spearman')))
        self.assertTrue(np.isnan(d.correlation([1,2,3],[0,0,0], 'pearson')))
        self.assertAlmostEqual(d.correlation([1,2,3],[3,2,1], 'spearman'), -1.)

    def test_shared_dialogue_bootstrap_and_all_seed_requirement(self):
        # Two seeds: first has dialogue totals (2,12)/(1,3), second (10,0)/(2,0).
        # Repeating dialogue0 gives (2+5)/2; dialogue1-only must be invalid.
        sums = np.array([[2.,12.], [10.,0.]])
        counts = np.array([[1,3], [2,0]])
        draws = np.array([[0,0], [1,1], [0,1], [1,0]])
        values = d.dialogue_bootstrap(sums, counts, draws)
        np.testing.assert_allclose(values[[0,2,3]], [3.5,4.25,4.25])
        self.assertTrue(np.isnan(values[1]))

    def test_auc_fixed_direction_and_ties(self):
        self.assertEqual(d.error_auc([0,0,1,1], [1,1,0,0]), 0.)
        self.assertEqual(d.error_auc([0,1], [0,0]), .5)
        self.assertTrue(np.isnan(d.error_auc([1,1], [0,1])))

    def test_pairs_ignore_behavior_and_sort_utterance_indices(self):
        frame = pd.DataFrame({'dataset':['toy']*4, 'seed':[1]*4,
            'dialogue_id':['d']*4, 'utterance_index':[3,0,2,1],
            'correct':[0,1,0,1], 'confidence':[.51,.5,.49,.5],
            'label':[0]*4, 'prediction':[0]*4,
            **{f'{c}_{m}':[0.,1.,2.,3.] for c in ['gaussian','dropout'] for m in ['mu','delta']}})
        first = d.build_pairs(frame)
        for c in ['gaussian','dropout']:
            for m in ['mu','delta']:
                frame[f'{c}_{m}'] = [100.,-100.,1000.,-1000.]
        second = d.build_pairs(frame.sample(frac=1, random_state=2))
        keys = ['dataset','seed','dialogue_id','correct_utterance_index','wrong_utterance_index']
        pd.testing.assert_frame_equal(first[keys], second[keys])
        self.assertEqual(first.correct_utterance_index.tolist(), [0,1])

    def test_empty_class_bins_retained_and_flagged(self):
        frame = pd.DataFrame({'dataset':['toy']*2, 'seed':[1]*2,
            'dialogue_id':['d']*2, 'correct':[1,0], 'confidence':[.3,.8],
            **{f'{c}_{m}':[0.,.2] for c in ['gaussian','dropout'] for m in ['mu','delta','flip_rate']}})
        distribution, bins, auc = d.distribution_tables(frame)
        self.assertEqual(len(bins),16)
        self.assertTrue(bins.insufficient_support.all())
        self.assertTrue(bins[bins.bin_index==1].correct_mean.isna().all())
        self.assertTrue((bins[bins.bin_index==3].n_wrong == 1).all())
        self.assertTrue((auc.auc_larger_is_error == 1).all())

    def test_split_mean_precedes_seed_mean(self):
        rows = pd.DataFrame({'dataset':['toy']*6,'seed':[1]*3+[2]*3,
            'corruption':['gaussian']*6,'measure':['mu']*6,
            'pearson':[.1,.4,.7,.4,np.nan,np.nan],
            'spearman':[.1,.4,.7,.4,np.nan,np.nan]})
        per_seed = rows.groupby(['dataset','seed','corruption','measure'],as_index=False)[['pearson','spearman']].mean()
        summary = d.seed_summary(per_seed,['dataset','corruption','measure'],['pearson','spearman'])
        self.assertAlmostEqual(summary.iloc[0].pearson_mean,.4)
        self.assertAlmostEqual(summary.iloc[0].pearson_sample_sd,0)
        self.assertEqual(summary.iloc[0].n_pearson_valid,2)

    def test_no_pairs_still_has_output_schema(self):
        frame = pd.DataFrame({'dataset':['toy'],'seed':[1],'dialogue_id':['d'],
            'correct':[1],'utterance_index':[0],'confidence':[.5]})
        pairs = d.build_pairs(frame)
        self.assertEqual(len(pairs),0)
        self.assertIn('dataset',pairs.columns)
        self.assertIn('gaussian_mu_difference',pairs.columns)


if __name__ == '__main__':
    unittest.main()
