"""Synthetic regressions for original response definitions and independent matching."""
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from mainline_core import calibration_metrics
from analyze_strength import response, match_confidences, validate_probabilities, dialogue_means, safe_spearman, calibration_bootstrap, METHODS


class AnalysisTests(unittest.TestCase):
    def test_original_mu_delta_js_and_population_sd(self):
        p=np.array([[.8,.2]]*5+[[.3,.7]]*5)
        r=response(.8,0,p,np.array([.8,.2]))
        self.assertAlmostEqual(r['mu'],.35)
        self.assertAlmostEqual(r['delta'],.05)
        self.assertAlmostEqual(r['flip_rate'],.5)
        self.assertAlmostEqual(r['confidence_change_mean'],-.05)
        self.assertAlmostEqual(r['confidence_change_std'],.05)
        self.assertAlmostEqual(r['mu_left'],0)
        self.assertAlmostEqual(r['mu_right'],.7)
        self.assertAlmostEqual(r['delta_left'],0)
        self.assertAlmostEqual(r['delta_right'],.1)
        original=np.array([.8,.2]); changed=np.array([.3,.7]); midpoint=(original+changed)/2
        expected=.25*np.sum(original*np.log(original/midpoint)+changed*np.log(changed/midpoint))
        self.assertAlmostEqual(r['js'],expected)

    def test_delta_absolute_mean_not_mean_absolute_change(self):
        r=response(.7,0,np.array([[.6,.4]]*5+[[.8,.2]]*5),np.array([.7,.3]))
        self.assertAlmostEqual(r['delta'],0)
        self.assertAlmostEqual(r['confidence_change_std'],.1)

    def test_unchanged_visual_has_zero_response(self):
        p=np.tile(np.array([.8,.2]),(10,1))
        r=response(.8,0,p,np.array([.8,.2]))
        for key in ['mu','delta','js','flip_rate','confidence_change_std']:
            self.assertAlmostEqual(r[key],0)

    def test_matching_maximizes_cardinality_before_distance(self):
        # Greedy nearest pair (.50,.50) would leave .481 unmatched.
        pairs=match_confidences([.50,.481],[.50,.519])
        self.assertEqual(len(pairs),2)
        self.assertEqual(set(pairs),{(0,1),(1,0)})
        self.assertEqual(match_confidences([.1],[.9]),[])
        self.assertEqual(match_confidences([], [.5]),[])
        self.assertEqual(len(match_confidences([.51,.52],[.51])),1)

    def test_matching_no_reuse_and_caliper(self):
        a=np.array([.61,.63,.80,.88]); b=np.array([.615,.632,.819,.45])
        pairs=match_confidences(a,b)
        self.assertEqual(len({i for i,_ in pairs}),len(pairs))
        self.assertEqual(len({j for _,j in pairs}),len(pairs))
        self.assertTrue(all(abs(a[i]-b[j])<=.02 for i,j in pairs))

    def test_probability_identity_validation(self):
        validate_probabilities([[.7,.3]],[0],[.7],2)
        with self.assertRaises(ValueError):
            validate_probabilities([[.7,.3]],[1],[.7],2)
        with self.assertRaises(ValueError):
            validate_probabilities([[.7,.4]],[0],[.7],2)
        with self.assertRaises(ValueError):
            validate_probabilities([[.7,.3]],[0],[.6],2)

    def test_shared_dialogue_bootstrap_weights_utterances(self):
        f=pd.DataFrame({'dialogue_id':['a','a','b']})
        means=dialogue_means(f,np.array([1.,3.,10.]),np.array(['a','b']),np.array([[1,1],[2,0],[0,2]]))
        np.testing.assert_allclose(means,[14/3,2,10])

    def test_bootstrap_calibration_matches_direct_dialogue_repetition(self):
        rows=[]
        ids=np.array(['a','b'])
        multiplicities=np.array([[1,1],[2,0],[0,2]])
        for strength in [.05,.1]:
            for method in METHODS:
                q=np.array([.8,.3,.9])
                if method.startswith('M_'):
                    q=q+(.01 if strength==.05 else .02)
                for i in range(3):
                    rows.append(dict(dataset='D',seed=1475,modality='text',strength=strength,
                                     method=method,dialogue_id='a' if i<2 else 'b',utterance_index=i,
                                     correct=[1,0,0][i],oof_confidence=q[i]))
        config={'datasets':{'D':{}},'modalities':['text']}
        with patch('analyze_strength.load_draws',return_value=(ids,multiplicities)):
            result=calibration_bootstrap(config,pd.DataFrame(rows))
        indices=[[0,1,2],[0,1,0,1],[2,2]]
        y=np.array([1,0,0]); reference=np.array([.8,.3,.9]); candidate=reference+.01
        selected=result[(result.comparison=='behavior_minus_E1') &
                        (result.candidate=='M_joint') & (result.candidate_strength==.05)]
        for metric in ['ece15','binary_brier','binary_nll']:
            direct=np.array([calibration_metrics(candidate[i],y[i])[metric]-
                             calibration_metrics(reference[i],y[i])[metric] for i in indices])
            row=selected[selected.metric==metric].iloc[0]
            self.assertAlmostEqual(row.mean_difference,calibration_metrics(candidate,y)[metric]-
                                   calibration_metrics(reference,y)[metric])
            self.assertAlmostEqual(row.ci_low,np.quantile(direct,.025))
            self.assertAlmostEqual(row.ci_high,np.quantile(direct,.975))

    def test_constant_correlation_is_missing(self):
        self.assertTrue(np.isnan(safe_spearman(np.zeros(10),np.arange(10))))
        self.assertAlmostEqual(safe_spearman(np.arange(10),np.arange(10)),1.)


if __name__=='__main__':
    unittest.main()
