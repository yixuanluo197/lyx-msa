import unittest
import numpy as np
from verify_results import independent_classification, independent_success

class VerificationTests(unittest.TestCase):
    def test_independent_metrics_have_expected_units(self):
        m=independent_classification(np.array([0,0]),np.array([[.8,.2],[.4,.6]]))
        self.assertAlmostEqual(m['ece15'],40)
        self.assertAlmostEqual(m['brier'],.4)
        self.assertAlmostEqual(m['nll'],-np.log(.8*.4)/2)
        self.assertEqual(m['auroc_success'],100)
        self.assertEqual(m['accuracy'],.5)
    def test_auc_handles_ties_independently(self):
        m=independent_success(np.array([0,1,0,1]),np.array([.5,.5,.2,.8]))
        self.assertEqual(m['auroc'],.875)
    def test_single_class_auc_is_missing(self):
        self.assertTrue(np.isnan(independent_success(np.array([1,1]),np.array([.7,.8]))['auroc']))

if __name__=='__main__':unittest.main()
