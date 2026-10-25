"""CPU synthetic regression tests; never imports training code or reads result files."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('analysis',Path(__file__).with_name('analyze_dropout_confirmation.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class AnalysisRules(unittest.TestCase):
    def test_candidate_requires_both_metrics(self):
        self.assertEqual(a.qualification(.1,0),'qualified')
        self.assertEqual(a.qualification(.1,-.001),'mixed_evidence')
        self.assertEqual(a.qualification(0,2),'no_best_improvement')
        self.assertEqual(a.qualification(-.1,2),'no_best_improvement')
    def test_two_seed_confirmation(self):
        self.assertEqual(a.confirmation([{'seed':2024,'candidate_category':'qualified'},{'seed':67137,'candidate_category':'qualified'}])['decision'],'preliminary_consistency_support')
    def test_second_seed_tail_cannot_replace_best(self):
        self.assertEqual(a.confirmation([{'seed':2024,'candidate_category':'qualified'},{'seed':67137,'candidate_category':'no_best_improvement'}])['decision'],'not_confirmed')
    def test_second_seed_mixed(self):
        self.assertEqual(a.confirmation([{'seed':2024,'candidate_category':'qualified'},{'seed':67137,'candidate_category':'mixed_evidence'}])['decision'],'mixed_evidence')
    def test_first_max_tie(self):
        self.assertEqual(a.first_max([{'epoch':1,'validation_weighted_f1':66.01},{'epoch':2,'validation_weighted_f1':66.01}])['epoch'],1)
        with self.assertRaises(AssertionError):a.first_max([{'epoch':1,'validation_weighted_f1':66.011}])
    def test_fixed_last10(self):
        rows=[{'validation_weighted_f1':float(x)} for x in range(1,41)]
        self.assertEqual(a.last10_mean(rows),35.5)
        with self.assertRaises(AssertionError):a.last10_mean(rows[:9])
    def test_completion_rejects_early_and_test(self):
        good={'phase':'completed','counts':{'train':40,'validation':40,'test':0},'test_forward_passes':0,'no_final_test_artifacts':True}
        a.require_complete(good)
        for key,value in [('phase','running'),('counts',{'train':39,'validation':39,'test':0}),('test_forward_passes',1),('no_final_test_artifacts',False)]:
            bad=copy.deepcopy(good);bad[key]=value
            with self.assertRaises(AssertionError):a.require_complete(bad)
    def test_only_registered_dropout_change_allowed(self):
        old={'seed':67137,'dropout':.4,'rec_dropout':.1,'lr':.0001,'batch_size':16,'split_seed':1475}
        new={**old,'dropout':.5,'skip_final_test':True,'output_dir':'new'};a.validate_pair(new,old)
        for key,value in [('lr',.0002),('batch_size',8),('rec_dropout',.2),('seed',2024),('split_seed',2024)]:
            with self.assertRaises(AssertionError):a.validate_pair({**new,key:value},old)
    def test_final_verifier_gate_rejects_nonpassed_and_test(self):
        run={'status':'passed','epochs':40,'selected_epoch':3,'validation_best_wf1':66.,'test_forwards':0,
             'loss_and_gradient_finite':True,'every_epoch_order_matches_historical_and_registered_batch16_replay':True,
             'unchanged_sampler_AST_and_historical_batch16_replay_passed':True,'train_and_validation_batch_size':16,
             'lr_unchanged_from_xp04':.0001,'dropout':.5,'rec_dropout_metadata':.1,
             'checkpoints':{'best.pt':{'epoch':3,'model_and_adam_finite':True,'full_rng_present':True},'latest.pt':{'epoch':40,'model_and_adam_finite':True,'full_rng_present':True}}}
        v={'status':'passed','formal_runs':1,'test_forwards':0,'cuda_initialized':False,'frozen_sources_and_inputs_unchanged':True,'runs':{'r':run},'historical_files_unchanged':{'history':1881}}
        a.verify_final(v,'r',3,66.)
        for key,value in [('status','failed'),('test_forwards',1),('frozen_sources_and_inputs_unchanged',False)]:
            with self.assertRaises(AssertionError):a.verify_final({**v,key:value},'r',3,66.)
    def test_partial_watcher_cannot_pass(self):
        # Cardinality is checked before interpreting any per-epoch numeric payload.
        with self.assertRaises(AssertionError):a.verify_watch([{'split':'train','epoch':1}],[])

if __name__=='__main__':unittest.main(verbosity=2)
