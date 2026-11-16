"""Focused tests for aggregation, uncertainty labels and paired statistics."""
import unittest
import json
import numpy as np
import pandas as pd
from analyze_precision import response, validate_p, unique, safe_spearman, bootstrap_distribution, comparisons, native_group
from mainline_core import calibration_metrics


class PrecisionTests(unittest.TestCase):
    def test_group_metadata_is_native_json_serializable(self):
        meta=native_group(('IEMOCAP',np.int64(1475),'K30_pooled'))
        self.assertEqual(json.loads(json.dumps(meta)),dict(dataset='IEMOCAP',seed=1475,condition='K30_pooled'))

    def test_delta_uses_absolute_after_pooled_mean(self):
        # All predictions unchanged; opposite signed shifts cancel only after pooling.
        p=np.array([[.9,.1]]*10+[[.6,.4]]*20)
        pooled=response(.7,0,p,[.7,.3])
        blocks=[response(.7,0,p[i:i+10],[.7,.3]) for i in [0,10,20]]
        self.assertAlmostEqual(pooled['delta'],0)
        self.assertAlmostEqual(np.mean([r['delta'] for r in blocks]),.13333333333333333)

    def test_mu_is_confidence_weighted_flip_not_flip_rate(self):
        p=np.array([[.8,.2]]*5+[[.4,.6]]*5)
        r=response(.8,0,p,[.8,.2])
        self.assertAlmostEqual(r['mu'],.3)
        self.assertAlmostEqual(r['flip_rate'],.5)
        self.assertAlmostEqual(r['delta'],.1)

    def test_confidence_mean_standard_error_ddof_one(self):
        p=np.array([[.8,.2]]*5+[[.6,.4]]*5)
        r=response(.7,0,p,[.7,.3])
        self.assertAlmostEqual(r['confidence_mean_standard_error'],np.std(p[:,0],ddof=1)/np.sqrt(10))

    def test_fixed_thirty_half_is_fifteen_fifteen(self):
        p=np.array([[.9,.1]]*15+[[.6,.4]]*15)
        r=response(.7,0,p,[.7,.3])
        self.assertAlmostEqual(r['delta_left'],.2)
        self.assertAlmostEqual(r['delta_right'],.1)
        self.assertAlmostEqual(r['delta'],.05)

    def test_identical_distribution_js_zero(self):
        r=response(.8,0,np.array([[.8,.2]]*10),[.8,.2])
        self.assertAlmostEqual(r['js'],0)
        self.assertAlmostEqual(r['delta'],0)
        self.assertAlmostEqual(r['mu'],0)

    def test_bad_repeat_count_rejected(self):
        with self.assertRaises(ValueError):response(.8,0,np.array([[.8,.2]]*20),[.8,.2])

    def test_duplicate_identity_rejected(self):
        with self.assertRaises(ValueError):unique(pd.DataFrame({'id':[1,1]}),['id'])

    def test_inconsistent_probability_prediction_rejected(self):
        with self.assertRaises(ValueError):validate_p([[.8,.2]],[1],[.8],2)

    def test_invalid_probability_normalization_rejected(self):
        with self.assertRaises(ValueError):validate_p([[.8,.8]],[0],[.8],2)

    def test_constant_spearman_is_missing_not_zero(self):
        self.assertTrue(np.isnan(safe_spearman(np.zeros(5),np.arange(5))))

    def test_dialogue_bootstrap_matches_repeated_rows(self):
        f=pd.DataFrame({'dialogue_id':['a','a','b'],'oof_confidence':[.4,.9,.7],'correct':[0,1,0]})
        result=bootstrap_distribution(f,np.array(['a','b']),np.array([[2,0],[1,1],[0,2]]))
        for i,idx in enumerate([[0,1,0,1],[0,1,2],[2,2]]):
            direct=calibration_metrics(f.iloc[idx].oof_confidence,f.iloc[idx].correct)
            for k in direct:self.assertAlmostEqual(result[k][i],direct[k])

    def test_no_twenty_or_best_block_comparison(self):
        values=list(comparisons({'conditions':dict.fromkeys(['K10_block0','K10_block1','K10_block2','K30_pooled'])}))
        primary=[v for v in values if v[-1]=='primary_repeat']
        self.assertEqual(primary,[('K30_pooled','M_gaussian','K10_block0','M_gaussian','primary_repeat')])
        self.assertFalse(any('20' in str(v) for v in values))


if __name__=='__main__':unittest.main()
