import unittest
import numpy as np
from run_pilot import compare_alignment, perturbation_metadata

class AlignmentTests(unittest.TestCase):
    def payload(self):
        p=np.array([[.7,.3],[.2,.8]],dtype=np.float32)
        return dict(labels=np.array([0,1]), predictions=np.array([0,1]), probabilities=p,
                    log_probabilities=np.log(p),dialogue_ids=np.array(['a','b']),utterance_indices=np.array([0,0]))
    def test_perturbation_predictions_are_explicit(self):
        original=self.payload();values=np.log(np.array([[[.1,.9]],[[.8,.2]]],dtype=np.float32))
        result=perturbation_metadata(original,values)
        self.assertNotIn('predictions',result)
        np.testing.assert_array_equal(result['original_predictions'],[0,1])
        np.testing.assert_array_equal(result['perturbed_predictions'],[[1],[0]])

    def test_accepts_matching_all_fields(self):
        self.assertTrue(compare_alignment(self.payload(),self.payload())['passed'])
    def test_rejects_identity_or_distribution_changes(self):
        for k in ['labels','predictions','utterance_indices','probabilities','log_probabilities']:
            a,b=self.payload(),self.payload();b[k]=b[k].copy();b[k].flat[0]+=1
            self.assertFalse(compare_alignment(a,b)['passed'], k)
        a,b=self.payload(),self.payload();b['dialogue_ids'][0]='x'
        self.assertFalse(compare_alignment(a,b)['passed'])
    def test_small_numerical_error_within_tolerance(self):
        a,b=self.payload(),self.payload();b['probabilities']=b['probabilities']+1e-8
        self.assertTrue(compare_alignment(a,b)['passed'])

if __name__=='__main__':unittest.main()
