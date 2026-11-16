"""Focused semantic tests: original definitions, grouping, no-op, bootstrap."""
import unittest
import numpy as np
import pandas as pd
from analyze_candidates import response, validate_p, bootstrap_distributions, aggregate_tables, METHODS, native_condition, dialogue_concentration, cross_block_stability
from mainline_core import calibration_metrics, fit_outer, FEATURES

class AnalysisTests(unittest.TestCase):
    def test_mu_is_confidence_weighted_not_flip_rate(self):
        p=np.array([[.4,.6]]*4+[[.8,.2]]*6)
        r=response(.8,0,p,np.array([.8,.2]))
        self.assertAlmostEqual(r['mu'],.24)
        self.assertAlmostEqual(r['flip_rate'],.4)
        self.assertAlmostEqual(r['delta'],.08)

    def test_delta_is_absolute_mean_change_not_std(self):
        p=np.array([[.7,.3],[.9,.1]]*5)
        r=response(.8,0,p,np.array([.8,.2]))
        self.assertAlmostEqual(r['delta'],0)
        self.assertGreater(r['confidence_change_std'],0)
        self.assertGreater(r['js'],0)

    def test_identity_probabilities_have_zero_response(self):
        p=np.tile([.7,.3],(10,1))
        r=response(.7,0,p,np.array([.7,.3]))
        for c in ['mu','delta','js','flip_rate']:self.assertAlmostEqual(r[c],0)

    def test_probability_sum_rejected(self):
        with self.assertRaises(ValueError):validate_p([[.7,.7]],[0],[.7],2)

    def test_argmax_rejected(self):
        with self.assertRaises(ValueError):validate_p([[.7,.3]],[1],[.7],2)

    def test_bootstrap_is_group_resampling_with_unequal_sizes(self):
        f=pd.DataFrame(dict(dialogue_id=['a','a','b'],oof_confidence=[.8,.6,.4],correct=[1,0,0]))
        ids=np.array(['a','b']);multiplicities=np.array([[1,1],[2,0],[0,2]])
        actual=bootstrap_distributions(f,ids,multiplicities)
        for i,w in enumerate([[1,1,1],[2,2,0],[0,0,2]]):
            expected=calibration_metrics(f.oof_confidence,f.correct,np.array(w))
            for k,v in expected.items():self.assertAlmostEqual(actual[k][i],v)

    def test_off_path_reuses_e1_exactly(self):
        groups=np.repeat([f'd{i}' for i in range(7)],4)
        frame=pd.DataFrame(dict(dialogue_id=groups,confidence=np.tile([.2,.4,.6,.8],7),correct=np.tile([0,1,0,1],7),gaussian_mu=np.zeros(28),gaussian_delta=np.zeros(28)))
        scored,params=fit_outer(frame,np.arange(24),np.arange(24,28),['M_gaussian','M_gaussian_mu','M_gaussian_delta'])
        for method in ['M_gaussian','M_gaussian_mu','M_gaussian_delta']:
            self.assertEqual(params['methods'][method]['selected_penalty'],'off')
            np.testing.assert_array_equal(scored[method]['oof_confidence'],scored['E1']['oof_confidence'])
        self.assertFalse(set(params['train_dialogues'])&set(params['held_dialogues']))

    def test_ablation_features_exclude_js_classes_and_other_modality(self):
        self.assertEqual(FEATURES['M_gaussian_mu'],['gaussian_mu'])
        self.assertEqual(FEATURES['M_gaussian_delta'],['gaussian_delta'])
        self.assertEqual(FEATURES['M_gaussian'],['gaussian_mu','gaussian_delta'])

    def test_condition_metadata_is_json_serializable(self):
        import json
        condition=native_condition(('IEMOCAP',np.int64(1475),np.int64(0)))
        self.assertEqual(json.loads(json.dumps(condition)),dict(dataset='IEMOCAP',seed=1475,block=0))

    def test_dialogue_concentration_detects_single_dialogue_dependence(self):
        base=pd.DataFrame(dict(dataset=['X']*4,seed=[1475]*4,block=[0]*4,dialogue_id=['a','a','b','b'],utterance_index=[0,1,0,1],correct=[1,0,1,0],oof_confidence=[.5]*4))
        parts=[base.assign(method='E1')]
        for method in ['M_gaussian','M_gaussian_mu','M_gaussian_delta']:
            candidate=base.assign(method=method)
            candidate['oof_confidence']=[.9,.1,.4,.6]
            parts.append(candidate)
        result=dialogue_concentration(pd.concat(parts,ignore_index=True))
        summary=result['dialogue_concentration_summary']
        r=summary[summary.method.eq('M_gaussian')&summary.metric.eq('binary_brier')].iloc[0]
        self.assertLess(r.full_difference,0)
        self.assertEqual(r.omissions_reversing_full_direction,1)

    def test_cross_block_pairs_by_id_and_preserves_undefined_correlation(self):
        base=pd.DataFrame(dict(dataset=['X']*3,seed=[1475]*3,dialogue_id=['a']*3,utterance_index=[0,1,2],modality=['text']*3,mu=[0.,0.,0.],delta=[.1,.2,.3],js=[.01,.02,.03]))
        a=base.assign(block=0)
        b=base.iloc[::-1].assign(block=1)
        out=cross_block_stability(pd.concat([a,b],ignore_index=True))
        self.assertTrue(np.isnan(out[out.measure.eq('mu')].spearman.iloc[0]))
        self.assertAlmostEqual(out[out.measure.eq('delta')].spearman.iloc[0],1.)
        self.assertEqual(out[out.measure.eq('mu')].left_nonzero_fraction.iloc[0],0.)

    def test_top_label_ece_boundaries_include_one(self):
        r=calibration_metrics(np.array([0.,1.]),np.array([0,1]))
        self.assertEqual(r['ece15'],0)
        self.assertEqual(r['binary_brier'],0)

if __name__=='__main__':unittest.main()
