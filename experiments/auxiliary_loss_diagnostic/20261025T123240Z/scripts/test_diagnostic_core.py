"""Synthetic CPU mathematics and privacy tests; no data, model, result, or CUDA access."""
import json
import math
import unittest

import torch

from diagnostic_core import (BRANCHES, RELATIONS, gradient_norm_rows,
                             gradient_pair_rows, parameter_groups, pseudo_rows)

REFERENCE = {'classification':1.,'cc':.002,'mscl':.15,'cscl':.15}
ACTUAL = {'classification':1.,'cc':.002,'mscl':0.,'cscl':0.}
CONTEXT = {'checkpoint_id':'synthetic','batch_id':'batch000','length_stratum':'short',
           'dataset':'MELD','seed':67137,'mode':'eval'}


def relation_fixture(single_class=False, one=False):
    if single_class:
        labels = torch.zeros(1 if one else 3,dtype=torch.long)
        main_conf, removed_conf = [.1]*len(labels), [.2]*len(labels)
        main_correct, removed_correct = [False]*len(labels), [False]*len(labels)
        expected = [0]*len(labels)
    else:
        labels = torch.arange(7)
        main_conf, removed_conf = [.1,.1,.2,.1,.7,.6,.7],[.2,.1,.1,.7,.1,.7,.6]
        main_correct = [False,False,False,False,True,True,True]
        removed_correct = [False,False,False,True,False,True,True]
        expected = [0,0,0,0,1,0,1]
    def probabilities(confidence, correctness, offset):
        rows=[]
        for truth,p,is_correct in zip(labels.tolist(),confidence,correctness):
            if is_correct:
                row=[(1-p)/6]*7;row[truth]=p
            else:
                row=[(1-p-.6)/5]*7;row[truth]=p;row[(truth+offset)%7]=.6
            rows.append(row)
        return torch.tensor(rows,dtype=torch.float64).log()
    main=probabilities(main_conf,main_correct,1)
    removed=probabilities(removed_conf,removed_correct,2)
    pseudo=torch.tensor(expected)
    weights=[len(labels)/expected.count(label) if label in expected else 0. for label in range(max(expected)+1)]
    return (labels,main,{name:removed.clone() for name in RELATIONS},
            {name:pseudo.clone() for name in RELATIONS},
            {name:torch.full((len(labels),),-.5) for name in RELATIONS},
            {name:list(weights) for name in RELATIONS})


def maps(**updates):
    result={name:{} for name in ('classification','cc','mscl','cscl')}
    result.update(updates)
    return result


def row(rows,**keys):
    return next(value for value in rows if all(value[key]==expected for key,expected in keys.items()))


class CoreTests(unittest.TestCase):
    def test_parameter_groups_partition_and_exclude_heads_from_shared(self):
        names=['gru_l.weight_ih_l0','linear_a.weight','linear_v.bias','graph_model.conv.weight',
               'smax_fc.weight','hidfc.bias','normBNa.weight','linear_l.weight']
        groups=parameter_groups(names)
        self.assertEqual(groups['text_gru'],names[:1])
        self.assertEqual(groups['all'],names)
        self.assertNotIn('smax_fc.weight',groups['shared_backbone'])
        self.assertIn('normBNa.weight',groups['shared_backbone'])
        self.assertEqual(sum(len(groups[key]) for key in ('text_gru','audio_projection','visual_projection','graph_shared','classification_head','other')),len(names))
        with self.assertRaises(ValueError):parameter_groups(['normBNa.running_mean'])
        with self.assertRaises(ValueError):parameter_groups(['x','x'])

    def test_weighted_norms_and_parameterwise_auxiliary_sum(self):
        gradients=maps(classification={'p':torch.tensor([1.,2.])},cc={'p':torch.tensor([3.,4.])},
                       mscl={'p':torch.tensor([-1.,2.])},cscl={'p':None})
        norms=gradient_norm_rows(gradients,{'all':['p']},CONTEXT,REFERENCE,ACTUAL)
        self.assertAlmostEqual(row(norms,loss='cc')['raw_l2'],5.)
        self.assertAlmostEqual(row(norms,loss='cc')['reference_weighted_l2'],.01)
        expected=torch.tensor([3.,4.],dtype=torch.float64)*.002+torch.tensor([-1.,2.],dtype=torch.float64)*.15
        self.assertAlmostEqual(row(norms,loss='aux_reference_sum')['raw_l2'],float(expected.norm()))
        self.assertAlmostEqual(row(norms,loss='aux_actual_sum')['raw_l2'],.01)
        self.assertEqual(row(norms,loss='aux_actual_sum')['source_missing_tensors'],1)
        self.assertEqual(row(norms,loss='mscl')['actual_weighted_l2'],0.)
        self.assertFalse(row(norms,loss='mscl')['actual_objective_active'])

    def test_nan_not_replaced_by_zero_and_inactive_branch_not_confused_with_raw(self):
        gradients=maps(classification={'p':torch.ones(2)},cc={'p':torch.ones(2)},mscl={'p':torch.tensor([float('nan'),0.])})
        norms=gradient_norm_rows(gradients,{'all':['p']},CONTEXT,REFERENCE,ACTUAL)
        bad=row(norms,loss='mscl')
        self.assertEqual(bad['status'],'nonfinite')
        self.assertIsNone(bad['raw_l2']);self.assertIsNone(bad['actual_weighted_l2'])
        self.assertIsNone(row(norms,loss='aux_reference_sum')['raw_l2'])
        self.assertAlmostEqual(row(norms,loss='aux_actual_sum')['raw_l2'],.002*math.sqrt(2))
        self.assertEqual(row(norms,loss='aux_actual_sum')['inactive_source_tensors'],1)
        json.dumps(norms,allow_nan=False)

    def test_none_zero_and_cancelled_sum_have_distinct_status(self):
        gradients=maps(classification={'p':torch.ones(2)},cc={'p':torch.tensor([1.,-1.])},mscl={'p':torch.tensor([-1.,1.])})
        coefficients={'cc':1.,'mscl':1.,'cscl':0.}
        norms=gradient_norm_rows(gradients,{'all':['p'],'empty':[]},CONTEXT,coefficients,coefficients)
        self.assertEqual(row(norms,loss='aux_reference_sum',group='all')['status'],'zero')
        self.assertEqual(row(norms,loss='cscl',group='all')['status'],'missing')
        self.assertIsNone(row(norms,loss='cscl',group='all')['raw_l2'])
        self.assertEqual(row(norms,loss='cc',group='empty')['status'],'empty_group')

    def test_negative_cosine_joint_support_and_all_coordinates_preserved(self):
        gradients=maps(classification={'p':torch.tensor([1.,0.,1.]),'z':torch.zeros(2),'bad':torch.ones(1),'missing':None},
                       cc={'p':torch.tensor([-1.,0.,-1.]),'z':torch.ones(2),'bad':torch.tensor([float('inf')]),'missing':torch.ones(1)})
        pairs=gradient_pair_rows(gradients,{'all':['p','z','bad','missing']},CONTEXT,REFERENCE,ACTUAL)
        pair=row(pairs,loss_b='cc')
        self.assertAlmostEqual(pair['cosine'],-1.)
        self.assertTrue(pair['negative_cosine'])
        self.assertEqual(pair['common_tensors'],1);self.assertEqual(pair['common_coordinates'],3)
        self.assertEqual(pair['excluded_zero_tensors'],1);self.assertEqual(pair['excluded_missing_tensors'],1)
        self.assertEqual(pair['excluded_nonfinite_tensors'],1)
        self.assertAlmostEqual(pair['dot_reference_weighted'],-.004)
        self.assertEqual(pair['status'],'partial_nonfinite')

    def test_coordinatewise_disjoint_nonzero_vectors_have_valid_zero_cosine(self):
        gradients=maps(classification={'p':torch.tensor([1.,0.])},mscl={'p':torch.tensor([0.,1.])})
        pair=row(gradient_pair_rows(gradients,{'all':['p']},CONTEXT,REFERENCE,ACTUAL),loss_b='mscl')
        self.assertEqual(pair['common_coordinates'],2);self.assertEqual(pair['common_tensors'],1)
        self.assertEqual(pair['cosine'],0.)
        self.assertFalse(pair['negative_cosine'])
        self.assertIsNone(pair['cosine_actual_weighted'])
        self.assertEqual(pair['norm_ratio_actual_weighted'],0.)

    def test_no_joint_support_has_none_cosine_and_explicit_reason(self):
        for right,status in [(None,'no_common_missing'),(torch.zeros(1),'no_common_zero'),(torch.tensor([float('nan')]),'no_common_nonfinite')]:
            gradients=maps(classification={'p':torch.ones(1)},cc={'p':right})
            pair=row(gradient_pair_rows(gradients,{'all':['p']},CONTEXT,REFERENCE,ACTUAL),loss_b='cc')
            self.assertEqual(pair['status'],status)
            self.assertIsNone(pair['cosine']);self.assertIsNone(pair['dot_raw']);self.assertIsNone(pair['norm_a_common'])

    def test_shape_mismatch_and_sensitive_public_context_rejected(self):
        gradients=maps(classification={'p':torch.ones(2)},cc={'p':torch.ones(3)})
        with self.assertRaises(ValueError):gradient_norm_rows(gradients,{'all':['p']},CONTEXT,REFERENCE,ACTUAL)
        with self.assertRaises(ValueError):gradient_norm_rows(maps(),{'all':[]},{'dialogue_id':'secret'},REFERENCE,ACTUAL)

    def test_finite_values_with_overflowing_aggregate_are_not_reported_finite(self):
        gradients=maps(classification={'p':torch.tensor([1e308],dtype=torch.float64)},
                       cc={'p':torch.tensor([1e308],dtype=torch.float64)})
        norms=gradient_norm_rows(gradients,{'all':['p']},CONTEXT,REFERENCE,ACTUAL)
        pair=row(gradient_pair_rows(gradients,{'all':['p']},CONTEXT,REFERENCE,ACTUAL),loss_b='cc')
        self.assertEqual(row(norms,loss='cc')['status'],'nonfinite_aggregate')
        self.assertIsNone(row(norms,loss='cc')['raw_l2'])
        self.assertEqual(pair['status'],'nonfinite_aggregate');self.assertIsNone(pair['cosine'])
        json.dumps(norms+[pair],allow_nan=False)

    def test_five_original_branches_and_default_zero_probability_directions(self):
        output=pseudo_rows(CONTEXT,*relation_fixture())
        counts=row(output['pseudo_counts'],relation='t',pseudo_label=0)
        self.assertEqual(counts['n_default_zero'],3)
        self.assertEqual([counts['default_zero_delta_positive'],counts['default_zero_delta_zero'],counts['default_zero_delta_negative']],[1,1,1])
        branches={value['branch']:value['n_utterances'] for value in output['branch_counts'] if value['relation']=='t'}
        self.assertEqual(sum(branches.values()),7)
        self.assertEqual(branches[BRANCHES[0]],3)
        self.assertTrue(all(branches[branch]==1 for branch in BRANCHES[1:]))
        private=[value for value in output['private_rows'] if value['relation']=='t' and value['default_zero']]
        self.assertTrue(all(value['main_pred']!=value['emotion_class'] and value['removed_pred']!=value['emotion_class'] for value in private))

    def test_ordered_pair_counts_and_float32_epsilon_leak_cells(self):
        output=pseudo_rows(CONTEXT,*relation_fixture())
        pair=row(output['pair_counts'],relation='t')
        self.assertEqual(pair['ordered_offdiag_pairs'],42)
        self.assertEqual(pair['true_positive_pairs'],22);self.assertEqual(pair['true_negative_pairs'],20)
        self.assertEqual(pair['same_class_negative_weight_cells'],22)
        self.assertEqual(pair['different_class_positive_epsilon_cells'],20)
        self.assertLess(pair['same_negative_mask_weight'],0.)
        self.assertGreater(pair['different_positive_mask_weight'],0.)

    def test_single_relation_class_and_one_anchor_extremes(self):
        output=pseudo_rows(CONTEXT,*relation_fixture(single_class=True))
        pair=row(output['pair_counts'],relation='t')
        self.assertEqual(pair['true_positive_pairs'],6);self.assertEqual(pair['true_negative_pairs'],0)
        self.assertEqual(pair['zero_negative_fraction'],1.)
        self.assertEqual(row(output['pseudo_counts'],relation='t',pseudo_label=0)['class_weight'],1.)
        one=row(pseudo_rows(CONTEXT,*relation_fixture(single_class=True,one=True))['pair_counts'],relation='t')
        self.assertEqual(one['zero_positive_fraction'],1.);self.assertEqual(one['zero_negative_fraction'],1.)

    def test_contributions_use_actual_base_logits_N_denominator_and_class_weights(self):
        output=pseudo_rows(CONTEXT,*relation_fixture())
        overall=row(output['contribution_summary'],relation='t',stratum='all')
        self.assertAlmostEqual(overall['contribution_sum'],1.)
        pseudo_sums=[value['contribution_sum'] for value in output['contribution_summary'] if value['relation']=='t' and value['stratum']=='pseudo']
        self.assertAlmostEqual(sum(pseudo_sums),overall['contribution_sum'])
        self.assertEqual(overall['N_denominator'],7)
        mscl=row(output['contribution_summary'],relation='mscl_combined',stratum='all')
        self.assertAlmostEqual(mscl['original_return_after_fallback'],3.)

    def test_nonfinite_base_values_not_zeroed_and_original_fallback_is_disclosed(self):
        fixture=list(relation_fixture())
        fixture[4]['t'][0]=float('nan')
        output=pseudo_rows(CONTEXT,*fixture)
        relation=row(output['contribution_summary'],relation='t',stratum='all')
        self.assertEqual(relation['n_nonfinite_base_logits'],1)
        self.assertIsNone(relation['contribution_sum'])
        combined=row(output['contribution_summary'],relation='mscl_combined',stratum='all')
        self.assertTrue(combined['original_nonfinite_fallback_triggered'])
        self.assertIsNone(combined['raw_loss_before_fallback'])
        self.assertEqual(combined['original_return_after_fallback'],0.)
        json.dumps(output,allow_nan=False)

    def test_mismatched_pseudo_or_weights_are_rejected(self):
        fixture=list(relation_fixture())
        fixture[3]['t'][0]=1
        with self.assertRaises(ValueError):pseudo_rows(CONTEXT,*fixture)
        fixture=list(relation_fixture());fixture[5]['t'][0]=999.
        with self.assertRaises(ValueError):pseudo_rows(CONTEXT,*fixture)

    def test_fractional_class_indices_are_not_silently_truncated(self):
        fixture=list(relation_fixture());fixture[0]=fixture[0].double();fixture[0][0]=.5
        with self.assertRaises(ValueError):pseudo_rows(CONTEXT,*fixture)

    def test_tables_have_uniform_csv_columns_and_anchor_pair_counts_are_private(self):
        output=pseudo_rows(CONTEXT,*relation_fixture())
        for values in output.values():
            self.assertTrue(all(set(value)==set(values[0]) for value in values))
        self.assertTrue(all('true_positive_pairs_for_anchor' in value for value in output['private_rows']))
        self.assertEqual(row(output['pseudo_counts'],relation='t',pseudo_label=0)['positive_pairs_per_anchor'],4)

    def test_public_rows_have_no_private_anchor_keys_or_real_ids(self):
        output=pseudo_rows(CONTEXT,*relation_fixture())
        for name,values in output.items():
            if name=='private_rows':continue
            for value in values:
                self.assertNotIn('utterance_index',value)
                self.assertNotIn('main_pred',value)
                self.assertNotIn('removed_pred',value)
                self.assertFalse(any('dialogue' in key or 'speaker_id' in key for key in value))
        self.assertTrue(all('utterance_index' in value for value in output['private_rows']))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    unittest.main(verbosity=2)
