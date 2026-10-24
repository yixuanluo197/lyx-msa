"""Synthetic qualification checks; no experiment/result reads or model execution."""
import unittest
from analyze_second_seed import category,confirmation_decision


class ConfirmationTests(unittest.TestCase):
    def test_original_qualification(self):
        self.assertEqual(category(.1,0.),'strong_candidate')
        self.assertEqual(category(.1,-.1),'weak_mixed_candidate')
        self.assertEqual(category(0.,1.),'no_best_improvement')

    def test_both_qualified_seeds_support_only_further_review(self):
        result=confirmation_decision([{'seed':67137,'candidate_category':'strong_candidate'},{'seed':2024,'candidate_category':'strong_candidate'}])
        self.assertEqual(result['decision'],'both_seeds_meet_registered_qualification')
        self.assertIn('不是显著性',result['interpretation'])

    def test_mixed_second_seed_does_not_confirm(self):
        result=confirmation_decision([{'seed':67137,'candidate_category':'strong_candidate'},{'seed':2024,'candidate_category':'weak_mixed_candidate'}])
        self.assertEqual(result['decision'],'not_consistently_confirmed')
        self.assertIn('暂缓',result['next_step'])

    def test_no_peak_gain_does_not_confirm(self):
        result=confirmation_decision([{'seed':67137,'candidate_category':'strong_candidate'},{'seed':2024,'candidate_category':'no_best_improvement'}])
        self.assertEqual(result['decision'],'not_consistently_confirmed')
        self.assertIn('未提高',result['interpretation'])


if __name__=='__main__':unittest.main(verbosity=2)
