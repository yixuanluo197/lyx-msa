"""Synthetic CPU tests; no experiment result access, model or GPU execution."""
from copy import deepcopy
import unittest
from analyze_dropout import first_max,last10_mean,qualification,recommendation,validate_pair


class DropoutAnalysisTests(unittest.TestCase):
    def test_candidate_peak_up_tail_equal(self):
        self.assertEqual(qualification(.1,0.),'second_seed_candidate')

    def test_mixed_peak_up_tail_down(self):
        self.assertEqual(qualification(.1,-.1),'mixed_evidence')

    def test_no_peak_gain_cannot_be_replaced_by_tail_gain(self):
        self.assertEqual(qualification(0.,1.),'no_best_improvement')
        self.assertEqual(qualification(-.1,1.),'no_best_improvement')

    def test_first_maximum_wins_rounded_tie(self):
        epochs=[{'epoch':1,'validation_weighted_f1':65.12},{'epoch':2,'validation_weighted_f1':66.01},{'epoch':3,'validation_weighted_f1':66.01}]
        self.assertEqual(first_max(epochs)['epoch'],2)

    def test_unrounded_selection_input_rejected(self):
        with self.assertRaises(AssertionError):first_max([{'epoch':1,'validation_weighted_f1':66.0123}])

    def test_last10_means_epochs31_to40(self):
        epochs=[{'epoch':i,'validation_weighted_f1':float(i)} for i in range(1,41)]
        self.assertEqual(last10_mean(epochs),35.5)

    def test_incomplete_tail_rejected(self):
        with self.assertRaises(AssertionError):last10_mean([{'validation_weighted_f1':1.}]*9)

    def base_args(self):
        return {'dropout':.4,'rec_dropout':.1,'lr':1e-4,'batch_size':16,'seed':2024,'selection_protocol':'validation','output_dir':'old','provenance':{},'epochs':40}

    def test_only_dropout_scientific_change_accepted(self):
        old=self.base_args();new={**old,'dropout':.5,'skip_final_test':True,'output_dir':'new'}
        before=deepcopy(new);validate_pair(new,old);self.assertEqual(new,before)

    def test_lr_batch_and_rec_dropout_changes_rejected(self):
        old=self.base_args()
        for key,value in [('lr',2e-4),('batch_size',8),('rec_dropout',.2),('seed',42)]:
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_pair({**old,'dropout':.5,'skip_final_test':True,key:value},old)

    def test_recommendation_does_not_authorize_additional_runs(self):
        self.assertIn('不自动追加',recommendation('second_seed_candidate'))
        self.assertIn('暂不推荐',recommendation('mixed_evidence'))
        self.assertIn('不支持扩大',recommendation('no_best_improvement'))


if __name__=='__main__':unittest.main(verbosity=2)
