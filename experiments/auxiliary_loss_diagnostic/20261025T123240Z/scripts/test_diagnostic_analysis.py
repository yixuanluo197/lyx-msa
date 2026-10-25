"""Synthetic CPU tests: public privacy, pair denominators, missingness and case gates."""
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('analysis',Path(__file__).with_name('analyze_diagnostic.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class Rules(unittest.TestCase):
    def test_public_aggregate_allowed(self):a.assert_public([{'checkpoint_id':'XP04_best','batch_id':'b001','n_utterances':'200','emotion':'0','label0_count':'80'}])
    def test_private_ids_and_paths_rejected(self):
        for row in ({'dialogue_ids':'1,2'},{'utterance_id':'u1'},{'raw_gradient':'1.0'},{'main_pred':'0'},{'utterance_index':'1'},{'path':'private/raw.json'},{'value':'/data2/private/file'}):
            with self.assertRaises(AssertionError):a.assert_public([row])
    def test_valid_requires_shared_finite_nonzero_norms(self):
        row={'status':'valid','cosine':-.4,'common_tensors':2,'norm_a_common':3.,'norm_b_common':2.}
        self.assertTrue(a.valid_pair(row))
        for change in ({'status':'missing'},{'norm_a_common':0},{'norm_b_common':float('nan')},{'common_tensors':0},{'cosine':None}):self.assertFalse(a.valid_pair({**row,**change}))
    def test_conflict_denominator_excludes_missing_zero(self):
        base={'status':'valid','common_tensors':2,'norm_a_common':1.,'norm_b_common':1.}
        result=a.summarize_pairs([{**base,'cosine':-.5},{**base,'cosine':.5},{**base,'status':'zero','cosine':None}])
        self.assertEqual(result['valid_n'],2);self.assertEqual(result['conflict_n'],1);self.assertEqual(result['conflict_frequency'],.5);self.assertEqual(result['total_n'],3)
    def test_all_missing_is_na_not_zero_cosine(self):
        result=a.summarize_pairs([{'status':'missing','cosine':None}]);self.assertIsNone(result['cosine_mean']);self.assertIsNone(result['conflict_frequency'])
    def test_paired_delta_requires_both_valid_and_same_batch(self):
        base={'status':'valid','common_tensors':1,'norm_a_common':1.,'norm_b_common':1.,'loss_a':'classification','loss_b':'cc','group':'all'}
        result=a.paired_deltas([{**base,'checkpoint_id':'XP04_best','batch_id':'b1','cosine':-.4},{**base,'checkpoint_id':'XP04_latest','batch_id':'b1','cosine':.1},{**base,'checkpoint_id':'XP04_best','batch_id':'b2','cosine':.2}], 'XP04_best','XP04_latest')
        self.assertEqual(len(result),1);self.assertAlmostEqual(result[0]['delta_cosine'],.5)
    def test_nonfinite_ratio_is_missing(self):
        self.assertIsNone(a.ratio(1,0));self.assertIsNone(a.ratio(float('nan'),1));self.assertEqual(a.ratio(3,2),1.5)
    def test_gate_rejects_incomplete_and_test(self):
        state={'phase':'completed','completed_cases':128,'optimizer_steps':0,'test_forwards':0};verification={'status':'passed','cases':128,'optimizer_steps':0,'test_forwards':0}
        a.require_complete(state,verification,128)
        for changed in ({'completed_cases':127},{'phase':'running'},{'optimizer_steps':1},{'test_forwards':1}):
            with self.assertRaises(AssertionError):a.require_complete({**state,**changed},verification,128)
    def test_case_coverage_same_batches_all_checkpoints(self):
        cps=['a','b'];rows=[{'checkpoint_id':cp,'batch_id':b} for cp in cps for b in ('b1','b2')];a.case_coverage(rows,cps,2)
        with self.assertRaises(AssertionError):a.case_coverage(rows[:-1],cps,2)
        with self.assertRaises(AssertionError):a.case_coverage(rows+[rows[0]],cps,2)
    def test_reference_not_actual_xp12_weight(self):
        self.assertEqual(a.actual_coefficient('XP12_best','mscl'),0.);self.assertEqual(a.actual_coefficient('XP12_latest','cscl'),0.)
        self.assertEqual(a.reference_coefficient('mscl'),.15);self.assertEqual(a.actual_coefficient('XP04_best','cc'),.002)
    def test_priority_requires_consistent_direction_and_scale(self):
        rows=[]
        for cp in ('XP04_best','XP04_latest'):
            for loss,conflict,scale in [('cc',.8,1.),('mscl',.2,4.),('cscl',.3,2.)]:
                rows.append({'checkpoint_id':cp,'group':'shared_backbone','loss_b':loss,'valid_n':32,'conflict_frequency':conflict,'norm_ratio_reference_weighted_median':scale})
        self.assertEqual(a.priority_screen(rows)['status'],'insufficient')
        for r in rows:
            if r['loss_b']=='cc':r['norm_ratio_reference_weighted_median']=5.
        self.assertEqual(a.priority_screen(rows)['mechanism'],'cc')
        rows[0]['valid_n']=31;self.assertEqual(a.priority_screen(rows)['status'],'insufficient')
    def test_core_statuses_preserved_and_explicitly_certified(self):
        base={'checkpoint_id':'XP04_best','batch_id':'b1','group':'all','loss_a':'classification','loss_b':'cc','status':'partial_support','cosine':-.2,'common_tensors':2,'norm_a_common':1.,'norm_b_common':2.}
        normalized=a.normalize_pairs([base])[0]
        self.assertEqual(normalized['status'],'valid');self.assertEqual(normalized['source_status'],'partial_support')
        with self.assertRaises(AssertionError):a.normalize_pairs([{**base,'norm_a_common':0}])
    def test_original_wrapper_flags_not_recomputed_loss(self):
        base={'checkpoint_id':'XP04_best','loss':'mscl'}
        rows=[{**base,'batch_id':'b1','fallback_observed':'True','nonfinite_before_fallback':'True','pre_fallback_loss':'','returned_loss':'0'},{**base,'batch_id':'b2','fallback_observed':'False','nonfinite_before_fallback':'False','pre_fallback_loss':'2.0','returned_loss':'2.0'}]
        self.assertEqual(a.summarize_wrappers(rows)[0]['fallback_batches'],1)
        with self.assertRaises(AssertionError):a.summarize_wrappers([{**rows[0],'returned_loss':'1'}])
        with self.assertRaises(AssertionError):a.summarize_wrappers([{**rows[1],'pre_fallback_loss':''}])
        with self.assertRaises(AssertionError):a.summarize_wrappers([rows[0],rows[0]])
    def test_quantiles_and_nonsignificance(self):
        self.assertEqual(a.distribution([])['n'],0);self.assertIsNone(a.distribution([])['median']);self.assertEqual(a.distribution([1,2,3])['median'],2)

if __name__=='__main__':unittest.main(verbosity=2)
