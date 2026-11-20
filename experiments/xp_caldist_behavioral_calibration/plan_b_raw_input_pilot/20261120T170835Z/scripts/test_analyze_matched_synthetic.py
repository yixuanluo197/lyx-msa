#!/usr/bin/env python3
"""CPU-only synthetic invariants; no experiment feature/model/response access."""
import copy
import json
import unittest

import numpy as np

import analyze_matched as a


def prob(confidence, correct=True):
    values = np.zeros(7)
    values[0] = confidence
    values[1] = 1 - confidence
    return values.tolist()


def fixture():
    samples, pairs = [], []
    for i in range(3):
        c_conf, e_conf = .65 + .002 * i, .66 + .002 * i
        did = 'D' + str(i)
        pair_id = 'P' + str(i)
        for correct, confidence in ((1, c_conf), (0, e_conf)):
            samples.append(dict(sample_id=pair_id + ('C' if correct else 'E'), pair_id=pair_id,
                                dialogue_id=did, utterance_index=str(correct),
                                confidence_bin='0.6-0.7', true_label='0' if correct else '1',
                                original_correct=str(correct), original_confidence=str(confidence),
                                original_probabilities=json.dumps(prob(confidence))))
        pairs.append(dict(pair_id=pair_id, dialogue_id=did, confidence_bin='0.6-0.7',
                          correct_sample_id=pair_id+'C', error_sample_id=pair_id+'E',
                          correct_confidence=str(c_conf), error_confidence=str(e_conf),
                          confidence_signed_gap=str(e_conf-c_conf), confidence_abs_gap=str(abs(e_conf-c_conf))))
    predictions = []
    for sample in samples:
        for family, variants in a.FAMILIES.items():
            for v in variants:
                predictions.append(dict(sample, perturbation_type=family, variant_id=v, modality='text',
                                        status='SUCCESS', automatic_qc_pass='True', label_review_status='PASS',
                                        label_preserving_main_eligible='True', reviewer='synthetic_AI_review_1_and_2',
                                        review_method='AI_DOUBLE_CONTEXT_REVIEW',
                                        perturbed_probabilities=json.dumps(prob(.62 if sample['original_correct']=='1' else .59))))
    return samples, pairs, predictions


class FormalBehaviorTests(unittest.TestCase):
    def test_exact_clean_zero(self):
        p = prob(.65)
        result = a.behavior(p, [p, p])
        for metric in a.METRICS:
            self.assertEqual(result[metric], 0.)

    def test_mu_is_confidence_weighted_and_delta_not_abs_mean(self):
        p = prob(.65)
        # One flip with confidence .7, one unchanged at .6: max-confidence mean=.65.
        q1 = [.3,.7,0,0,0,0,0]
        q2 = [.6,.4,0,0,0,0,0]
        result = a.behavior(p,[q1,q2])
        self.assertAlmostEqual(result['mu'], .35)
        self.assertAlmostEqual(result['flip_rate'], .5)
        self.assertAlmostEqual(result['delta'], 0.)
        self.assertAlmostEqual(result['mean_abs_confidence_change'], .05)

    def test_js_natural_log_and_zero_entries(self):
        p = [1.,0,0,0,0,0,0]
        q = [0,1.,0,0,0,0,0]
        self.assertAlmostEqual(a.js_divergence(p,q),np.log(2.))
        self.assertEqual(a.js_divergence(p,p),0.)

    def test_fixed_k(self):
        with self.assertRaises(ValueError):
            a.behavior(prob(.6),[prob(.6)])

    def test_invalid_probability_fails_closed(self):
        for p in ([1.,0], [float('nan')]+[0]*6, [1.1,-.1]+[0]*5, [.1]*7):
            with self.assertRaises(ValueError):
                a.probability(p)

    def test_exact_rank_ties_and_direction(self):
        self.assertEqual(a.concordance(0.), .5)
        self.assertEqual(a.concordance(1.), 1.)
        self.assertEqual(a.concordance(-1.), 0.)


class GateAndCohortTests(unittest.TestCase):
    def test_valid_disjoint_cohort(self):
        samples,pairs,_ = fixture()
        self.assertEqual(len(a.validate_cohort(pairs,samples)),6)

    def test_reuse_and_cross_cell_rejected(self):
        samples,pairs,_=fixture()
        broken=copy.deepcopy(pairs)
        broken[1]['correct_sample_id']=pairs[0]['correct_sample_id']
        with self.assertRaises(ValueError):a.validate_cohort(broken,samples)
        broken=copy.deepcopy(pairs);broken[0]['confidence_bin']='0.7-0.8'
        with self.assertRaises(ValueError):a.validate_cohort(broken,samples)

    def test_complete_targets_ai_not_human(self):
        samples,pairs,predictions=fixture()
        rows=a.aggregate_targets(predictions,samples)
        self.assertEqual(len(rows),12)
        self.assertTrue(all(r['automatic_qc_eligible'] and r['ai_reviewed_exploratory_eligible'] for r in rows))
        self.assertFalse(any(r['human_confirmed_eligible'] for r in rows))

    def test_failed_version_does_not_reduce_k_or_keep_pair(self):
        samples,pairs,predictions=fixture()
        predictions[0]['status']='FAILED';predictions[0]['perturbed_probabilities']=''
        targets=a.aggregate_targets(predictions,samples)
        failed=next(r for r in targets if r['sample_id']==predictions[0]['sample_id'] and r['perturbation_type']=='neutral_append')
        self.assertFalse(failed['complete_planned_family'])
        self.assertIsNone(failed['mu'])
        rows=a.build_pair_records(pairs,targets)
        conditions=[r for r in rows if r['pair_id']=='P0' and r['perturbation_type']=='neutral_append']
        self.assertTrue(all(not r['eligible'] for r in conditions))
        self.assertTrue(all('INCOMPLETE_FIXED_K' in r['exclusion_reason'] for r in conditions))

    def test_uncertain_kept_in_diagnostic_not_ai(self):
        samples,pairs,predictions=fixture()
        predictions[0]['label_review_status']='UNCERTAIN';predictions[0]['label_preserving_main_eligible']='False'
        targets=a.aggregate_targets(predictions,samples)
        target=targets[0]
        self.assertTrue(target['automatic_qc_eligible'])
        self.assertFalse(target['ai_reviewed_exploratory_eligible'])
        rows=a.build_pair_records(pairs,targets)
        matching=[r for r in rows if r['pair_id']=='P0' and r['perturbation_type']=='neutral_append']
        self.assertEqual([r['eligible'] for r in matching],[True,False,False])

    def test_absent_variant_not_hidden(self):
        samples,pairs,predictions=fixture()
        removed=predictions.pop(0)
        targets=a.aggregate_targets(predictions,samples)
        target=next(r for r in targets if r['sample_id']==removed['sample_id'] and r['perturbation_type']==removed['perturbation_type'])
        self.assertEqual(target['recorded_k'],1)
        self.assertEqual(target['missing_variants'],removed['variant_id'])

    def test_duplicate_variant_and_changed_original_rejected(self):
        samples,pairs,predictions=fixture()
        with self.assertRaises(ValueError):a.aggregate_targets(predictions+[predictions[0]],samples)
        predictions[0]['original_probabilities']=json.dumps(prob(.69))
        with self.assertRaises(ValueError):a.aggregate_targets(predictions,samples)

    def test_human_requires_two_distinct_actual_ids(self):
        samples,pairs,predictions=fixture()
        for p in predictions:
            p.update(human_reviewer_1_status='PASS',human_reviewer_2_status='PASS',human_reviewer_1_id='H1',human_reviewer_2_id='H1')
        self.assertFalse(any(r['human_confirmed_eligible'] for r in a.aggregate_targets(predictions,samples)))
        for p in predictions:p['human_reviewer_2_id']='H2'
        self.assertTrue(all(r['human_confirmed_eligible'] for r in a.aggregate_targets(predictions,samples)))

    def test_pair_difference_changes_sign_if_values_swapped(self):
        samples,pairs,predictions=fixture();targets=a.aggregate_targets(predictions,samples)
        before=a.build_pair_records(pairs,targets)
        swapped=copy.deepcopy(targets)
        index={(r['sample_id'],r['perturbation_type']):r for r in targets}
        for pair in pairs:
            for family in a.FAMILIES:
                c=index[(pair['correct_sample_id'],family)];e=index[(pair['error_sample_id'],family)]
                for row in swapped:
                    if row['perturbation_type']!=family:continue
                    other=e if row['sample_id']==c['sample_id'] else c if row['sample_id']==e['sample_id'] else None
                    if other:
                        for metric in a.METRICS:row[metric]=other[metric]
        after=a.build_pair_records(pairs,swapped)
        for left,right in zip(before,after):
            if left['eligible']:
                for metric in a.METRICS:self.assertAlmostEqual(left[metric+'_difference'],-right[metric+'_difference'])


class BootstrapAndEstimandTests(unittest.TestCase):
    def test_shared_draws_deterministic(self):
        one=a.bootstrap_draws(['A','B','C'])
        self.assertEqual(one.shape,(2000,3))
        self.assertTrue(np.array_equal(one,a.bootstrap_draws(['A','B','C'])))

    def test_duplicate_cluster_full_weight_and_estimands(self):
        rows=[{'dialogue_id':'A','x':1.},{'dialogue_id':'A','x':3.},{'dialogue_id':'B','x':10.}]
        draws=np.array([[0,0],[0,1],[1,1]])
        stats=a.summarize_values(rows,'x',['A','B'],draws)
        self.assertAlmostEqual(stats['equal_dialogue'][0],6.)
        self.assertAlmostEqual(stats['equal_pair'][0],14/3)
        np.testing.assert_allclose(stats['equal_dialogue'][1],[2.,6.,10.])
        np.testing.assert_allclose(stats['equal_pair'][1],[2.,14/3,10.])

    def test_bootstrap_brute_force_independent_recompute(self):
        rows=[{'dialogue_id':d,'x':float(i+1)} for i,d in enumerate(['A','A','B','C','C','C'])]
        ids=['A','B','C','EMPTY'];draws=a.bootstrap_draws(ids)
        results=a.summarize_values(rows,'x',ids,draws)
        expected_d=[];expected_p=[]
        for draw in draws:
            groupmeans=[];flat=[]
            for i in draw:
                vals=[r['x'] for r in rows if r['dialogue_id']==ids[i]]
                if vals:groupmeans.append(np.mean(vals));flat.extend(vals)
            expected_d.append(np.mean(groupmeans) if groupmeans else np.nan)
            expected_p.append(np.mean(flat) if flat else np.nan)
        np.testing.assert_allclose(results['equal_dialogue'][1],expected_d,equal_nan=True)
        np.testing.assert_allclose(results['equal_pair'][1],expected_p,equal_nan=True)

    def test_empty_scope_all_nan_no_fake_human_pass(self):
        results=a.summarize_values([],'x',['A','B'],a.bootstrap_draws(['A','B']))
        self.assertIsNone(results['equal_dialogue'][0])
        self.assertTrue(np.isnan(results['equal_dialogue'][1]).all())
        self.assertEqual(a.interval(results['equal_dialogue'][1]),(None,None))

    def test_95_and_97_5_intervals(self):
        x=np.arange(2000,dtype=float)
        low,high=a.interval(x);wide_low,wide_high=a.interval(x,.975)
        self.assertLess(wide_low,low);self.assertGreater(wide_high,high)
        self.assertAlmostEqual(low,49.975)

    def test_fixed_calipers_and_residual_group_boundaries(self):
        self.assertEqual(a.gap_group(0.),'near_zero')
        self.assertEqual(a.gap_group(1e-6),'near_zero')
        self.assertEqual(a.gap_group(1.001e-6),'error_confidence_higher')
        self.assertEqual(a.gap_group(-1.001e-6),'error_confidence_lower')
        self.assertEqual(a.CALIPERS,(('all',None),('gap_le_0.02',.02),('gap_le_0.01',.01)))

    def test_all_scopes_replicates_share_master_array(self):
        samples,pairs,predictions=fixture()
        rows=a.build_pair_records(pairs,a.aggregate_targets(predictions,samples))
        ids=['D0','D1','D2'];draws=a.bootstrap_draws(ids)
        support,contrasts,ranks,reps=a.analyze_pairs(rows,pairs,ids,draws)
        self.assertEqual(len(support),36)
        self.assertEqual(len(contrasts),288)
        self.assertEqual(len(ranks),432)
        self.assertEqual(len(reps),720)
        human=[r for r in contrasts if r['analysis_scope']=='HUMAN_CONFIRMED']
        self.assertTrue(all(r['error_minus_correct'] is None and r['bootstrap_valid']==0 for r in human))
        priority=[r for r in contrasts if r['priority_two_family_js']]
        self.assertEqual(len(priority),2)
        self.assertTrue(all(r['bonferroni_ci97_5_low'] is not None for r in priority))
        # Exact same values for synthetic identical families under identical draws.
        np.testing.assert_array_equal(reps['C001__js_mean__equal_dialogue__difference'],
                                      reps['C019__js_mean__equal_dialogue__difference'])


if __name__=='__main__':
    unittest.main(verbosity=2)
