"""Synthetic CPU checks; no model or experimental records are loaded."""
import copy
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('analysis',Path(__file__).with_name('analyze_hscl_ablation.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class Rules(unittest.TestCase):
    def test_positive_requires_strict_tail_increase(self):
        self.assertEqual(a.qualification(.01,.001),'positive')
        self.assertEqual(a.qualification(.01,0),'mixed_incomplete')
    def test_tail_fall_is_mixed(self):self.assertEqual(a.qualification(.01,-.01),'mixed_incomplete')
    def test_best_cannot_be_replaced_by_tail(self):
        self.assertEqual(a.qualification(0,2),'no_best_improvement')
        self.assertEqual(a.qualification(-.01,2),'no_best_improvement')
    def test_raw_hscl_is_retained_but_weighted_zero(self):
        raw={'cc_loss':100.,'mscl_loss':3.,'cscl_loss':2.}
        config={'calibrate':True,'contrastlearning':False,'rank_coff':.002,'mscl_coff':.15,'cscl_coff':.15}
        self.assertEqual(a.weighted(raw,config),{'weighted_cc_loss':.2,'weighted_mscl_loss':0.,'weighted_cscl_loss':0.})
        self.assertEqual(raw['mscl_loss'],3.)
    def test_first_rounded_maximum_wins(self):
        self.assertEqual(a.first_max([{'epoch':1,'validation_weighted_f1':65.24},{'epoch':2,'validation_weighted_f1':65.24}])['epoch'],1)
        with self.assertRaises(AssertionError):a.first_max([{'validation_weighted_f1':65.241}])
    def test_fixed_tail(self):
        self.assertEqual(a.last10_mean([{'validation_weighted_f1':x} for x in range(1,41)]),35.5)
    def test_only_switch_changes(self):
        old={'seed':67137,'Dataset':'MELD','lr':.0001,'batch_size':16,'dropout':.4,'rec_dropout':.1,'calibrate':True,'contrastlearning':True,'courselearning':True,'mscl_coff':.15,'cscl_coff':.15,'rank_coff':.002}
        new={**old,'contrastlearning':False,'skip_final_test':True};a.validate_pair(new,old)
        for key,value in [('dropout',.5),('lr',.0002),('calibrate',False),('courselearning',False),('mscl_coff',0.)]:
            with self.assertRaises(AssertionError):a.validate_pair({**new,key:value},old)
    def test_early_status_rejected(self):
        good={'phase':'completed','counts':{'train':40,'validation':40,'test':0},'test_forward_passes':0,'no_final_test_artifacts':True};a.require_complete(good)
        for key,value in [('phase','running'),('counts',{'train':39,'validation':39,'test':0}),('test_forward_passes',1)]:
            with self.assertRaises(AssertionError):a.require_complete({**good,key:value})
    def test_partial_watch_rejected(self):
        with self.assertRaises(AssertionError):a.verify_watch([{'split':'train','epoch':1}],[])
    def test_missing_history_validation_aux_not_zero(self):
        row=a.validation_component({'validation_loss':2.},None)
        self.assertEqual(row['classification_loss'],2.);self.assertIsNone(row['mscl_loss']);self.assertIsNone(row['total_loss'])
    def test_actual_flat_environment_schema(self):
        self.assertEqual(a.environment_versions({'python':'3.10.13 | conda','torch':'1.13.1+cu117','torch_cuda':'11.7','pyg':'2.4.0'}),('3.10.13','1.13.1+cu117','11.7','2.4.0'))
    def test_final_result_gate(self):
        run={'status':'passed','epochs':40,'selected_epoch':4,'validation_best_wf1':66.,'test_forwards':0,'loss_and_gradient_finite':True,'contrastlearning':False,'calibrate':True,'courselearning':True,'dropout':.4,
             'checkpoints':{'best.pt':{'epoch':4,'model_and_adam_finite':True,'full_rng_present':True},'latest.pt':{'epoch':40,'model_and_adam_finite':True,'full_rng_present':True}}}
        value={'status':'passed','formal_runs':1,'test_forwards':0,'frozen_sources_and_inputs_unchanged':True,'runs':{'r':run}}
        a.verify_final(value,'r',4,66.)
        bad=copy.deepcopy(value);bad['runs']['r']['contrastlearning']=True
        with self.assertRaises(AssertionError):a.verify_final(bad,'r',4,66.)

if __name__=='__main__':unittest.main(verbosity=2)
