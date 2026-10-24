"""Synthetic CPU qualification tests; no experiment reads or model execution."""
import unittest
from analyze_batch_size import qualification,recommendation


class QualificationTests(unittest.TestCase):
    def test_peak_gain_with_flat_tail_qualifies(self):
        self.assertEqual(qualification(.1,0.),'second_seed_candidate')

    def test_peak_gain_with_tail_loss_is_mixed(self):
        self.assertEqual(qualification(.1,-.01),'mixed_evidence')

    def test_tail_gain_does_not_replace_primary_metric(self):
        self.assertEqual(qualification(0.,1.),'no_best_improvement')
        self.assertEqual(qualification(-.1,1.),'no_best_improvement')

    def test_candidate_recommendation_never_launches_runs(self):
        text=recommendation(qualification(.1,.1))
        self.assertIn('另行登记',text);self.assertIn('不自动启动',text)
        self.assertIn('不是统计强证据',text)

    def test_mixed_and_no_gain_do_not_recommend_expansion(self):
        self.assertIn('暂不推荐',recommendation('mixed_evidence'))
        self.assertIn('不支持扩大',recommendation('no_best_improvement'))


if __name__=='__main__':unittest.main(verbosity=2)
