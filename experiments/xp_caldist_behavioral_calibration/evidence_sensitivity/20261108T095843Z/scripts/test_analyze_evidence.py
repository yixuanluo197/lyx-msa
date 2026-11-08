"""Scientific checks; synthetic arrays are tests, never experimental evidence."""
import unittest
import json
import tempfile
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from analyze_evidence import classification_from_confusion, summarize_condition, dialogue_bootstrap_weights, validate_payload, correctness_contrasts, compare_previous_pilot
from plot_evidence import curve_arrays


class EvidenceAnalysisTests(unittest.TestCase):
    def test_weak_stability_does_not_hide_strong_sensitivity(self):
        previous=[]
        for dialogue in ['a','b']:
            for kind in ['gaussian','dropout','attenuation']:
                previous.append(dict(dataset='IEMOCAP',modality='audio',dialogue_id=dialogue,
                                     utterance_index='0',perturbation_type=kind,flip_rate='0'))
        base=dict(dialogue_ids=np.array(['a','b']),utterance_indices=np.zeros(2,int),
                  labels=np.array([0,0]),predictions=np.array([0,1]))
        detail=dict(flip=np.array([[True],[False]]),true_probability_change=np.array([[-.3],[.01]]))
        row=compare_previous_pilot(previous,'IEMOCAP','A',base,detail)
        self.assertEqual(row['weak_stable_count'],2)
        self.assertEqual(row['strong_flip_among_weak_stable'],1)
        self.assertEqual(row['weak_stable_wrong_count'],1)

    def test_correctness_contrast_uses_paired_difference_not_overlapping_group_intervals(self):
        y = np.array([0,1,1,0,1,0])
        p = np.array([[.8,.2],[.8,.2],[.2,.8],[.2,.8],[.8,.2],[.8,.2]])
        q = np.array([[[.7,.3]],[[.4,.6]],[[.6,.4]],[[.4,.6]],[[.7,.3]],[[.7,.3]]])
        di = np.array([0,0,1,1,2,2])
        weights = np.array([[1,1,1],[2,1,0],[0,1,2],[1,0,2]])
        result = correctness_contrasts(y,p,q,di,weights)
        correct = p.argmax(1)==y
        flip=(q.argmax(2)!=p.argmax(1)[:,None]).mean(1)
        draws=[]
        for w in weights:
            ix=np.concatenate([np.tile(np.flatnonzero(di==d),int(n)) for d,n in enumerate(w)])
            draws.append(flip[ix][~correct[ix]].mean()-flip[ix][correct[ix]].mean())
        self.assertAlmostEqual(result['flip_rate']['difference_wrong_minus_correct'],flip[~correct].mean()-flip[correct].mean())
        self.assertAlmostEqual(result['flip_rate']['ci_low'],np.quantile(draws,.025))
        self.assertAlmostEqual(result['flip_rate']['ci_high'],np.quantile(draws,.975))

    def test_correctness_contrast_empty_group_is_na(self):
        p=np.array([[.8,.2],[.2,.8]])
        result=correctness_contrasts(np.array([0,1]),p,p[:,None],np.array([0,1]),np.ones((5,2)))
        self.assertTrue(np.isnan(result['flip_rate']['difference_wrong_minus_correct']))
        self.assertEqual(result['flip_rate']['valid_bootstraps'],0)

    def test_full_condition_io_in_automatically_deleted_fixture(self):
        from analyze_evidence import run
        from evidence_core import condition_specs
        source = Path(__file__).resolve().parent.parent
        cfg = json.loads((source/'configs/evidence_sensitivity_config.json').read_text())
        cfg['anchors']={'IEMOCAP':1475}
        cfg['validation_sizes']={'IEMOCAP':{'utterances':6,'dialogues':3}}
        cfg['visual_missingness']['expected_counts']={'IEMOCAP':1}
        cfg['analysis']['bootstrap_repeats']=7
        cfg['previous_pilot']='/does-not-exist-test-fixture'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'configs').mkdir()
            (root/'configs/evidence_sensitivity_config.json').write_text(json.dumps(cfg))
            directory=root/'runs/IEMOCAP/seed1475'; (directory/'conditions').mkdir(parents=True)
            p=np.array([[.95,.05],[.8,.2],[.3,.7],[.2,.8],[.7,.3],[.4,.6]])
            base=dict(labels=np.array([0,1,1,0,0,1]),probabilities=p,log_probabilities=np.log(p),
                      predictions=p.argmax(1),dialogue_ids=np.array(['a','a','b','b','c','c']),
                      utterance_indices=np.array([0,1,0,1,0,1]),speaker_ids=np.zeros(6,int),
                      dialogue_lengths=np.full(6,2),batch_indices=np.zeros(6,int))
            np.savez(directory/'baseline_validation.npz',**base)
            missing=np.array([False,False,False,False,False,True])
            for phase in cfg['phases']:
                for spec in condition_specs(phase):
                    active=~missing if spec['target']=='V' else np.ones(6,bool)
                    q=np.repeat(p[:,None,:],spec['repeats'],axis=1)
                    q[active]=q[active]*(1-spec['strength'])+np.array([.4,.6])*spec['strength']
                    data={k:v for k,v in base.items() if k not in ('probabilities','log_probabilities','predictions')}
                    data.update(phase=phase,condition_id=spec['id'],target=spec['target'],kind=spec['kind'],
                                strength=spec['strength'],repeats=spec['repeats'],active=active,visual_missing=missing,
                                probabilities=q,log_probabilities=np.log(q),perturbed_predictions=q.argmax(2),
                                original_probabilities=p,original_predictions=p.argmax(1))
                    np.savez(directory/'conditions'/f"{spec['id']}.npz",**data)
            result=run(root)
            self.assertEqual(result['conditions'],56)
            self.assertEqual(result['per_repeat_utterance_rows'],6*168)
            with (root/'results/sensitivity_summary.csv').open() as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),56*5)
            self.assertEqual((root/'results/sensitivity_summary.csv').read_bytes(),
                             (root/'results/modality_sensitivity_summary.csv').read_bytes())
            inactive=[r for r in rows if r['target']=='V' and r['subgroup']=='visual_missing']
            self.assertTrue(all(float(r['delta_accuracy'])==0 for r in inactive))
            self.assertTrue((root/'reports/FINAL_EVIDENCE_SENSITIVITY.md').exists())

    def test_vectorized_bootstrap_f1_matches_explicit_cluster_resampling(self):
        y = np.array([0, 1, 2, 1, 2, 0])
        p = np.array([[.8,.1,.1],[.2,.7,.1],[.1,.1,.8],[.6,.3,.1],[.1,.7,.2],[.6,.3,.1]])
        q = np.stack([p[:, [1,0,2]], p[:, [2,1,0]]], 1)
        di = np.array([0,0,1,1,2,2])
        weights = np.array([[1,1,1],[2,1,0],[0,0,3],[0,2,1]])
        selected = np.array([True,True,True,True,True,False])
        actual = summarize_condition(y,p,q,selected,di,weights)
        for average in ['weighted','macro']:
            draws = []
            for weight in weights:
                ix = np.concatenate([np.tile(np.flatnonzero((di==d)&selected),int(count))
                                     for d,count in enumerate(weight)])
                base = f1_score(y[ix],p[ix].argmax(1),average=average,zero_division=0)
                after = np.mean([f1_score(y[ix],q[ix,r].argmax(1),average=average,zero_division=0) for r in range(2)])
                draws.append(after-base)
            self.assertAlmostEqual(actual['delta_'+average+'_f1_ci_low'], np.quantile(draws,.025))
            self.assertAlmostEqual(actual['delta_'+average+'_f1_ci_high'], np.quantile(draws,.975))

    def test_curve_zero_is_baseline_not_invented_intervention(self):
        rows = [dict(strength=str(rho), original_accuracy='.7',original_accuracy_ci_low='.6',
                     original_accuracy_ci_high='.8',perturbed_accuracy='.5',
                     perturbed_accuracy_ci_low='.4',perturbed_accuracy_ci_high='.6') for rho in [.5,.25,1.,.75]]
        x,y,low,high = curve_arrays(rows,'perturbed_accuracy')
        np.testing.assert_array_equal(x,[0,.25,.5,.75,1])
        self.assertEqual(y[0],.7)
        self.assertEqual(low[0],.6)
        self.assertEqual(high[0],.8)

    def test_classification_matches_sklearn_including_absent_classes(self):
        labels = np.array([0, 0, 1, 1])
        predictions = np.array([0, 1, 1, 1])
        cm = np.zeros((3, 3))
        np.add.at(cm, (labels, predictions), 1)
        result = classification_from_confusion(cm)
        self.assertAlmostEqual(result['accuracy'], accuracy_score(labels, predictions))
        self.assertAlmostEqual(result['weighted_f1'], f1_score(labels, predictions, average='weighted'))
        self.assertAlmostEqual(result['macro_f1'], f1_score(labels, predictions, average='macro'))
        self.assertTrue(np.isnan(classification_from_confusion(cm * 0)['accuracy']))

    def test_repeat_classification_is_not_probability_ensemble(self):
        y = np.array([0, 1])
        p = np.array([[.8, .2], [.2, .8]])
        q = np.array([[[.9, .1], [.1, .9]], [[.1, .9], [.9, .1]]])
        result = summarize_condition(y, p, q, np.ones(2, bool), np.array([0, 1]), np.array([[1, 1], [2, 0]]))
        self.assertAlmostEqual(result['perturbed_accuracy'], .5)
        self.assertAlmostEqual(result['delta_accuracy'], -.5)
        self.assertAlmostEqual(result['flip_rate'], .5)
        self.assertAlmostEqual(result['true_probability_change'], -.3)

    def test_identical_predictions_have_zero_paired_deltas(self):
        y = np.array([0, 1, 1])
        p = np.array([[.8, .2], [.4, .6], [.7, .3]])
        w = dialogue_bootstrap_weights(np.array(['a', 'a', 'b']), 30, 20261108)[1]
        result = summarize_condition(y, p, p[:, None], np.ones(3, bool), np.array([0, 0, 1]), w)
        for metric in ['delta_accuracy', 'delta_weighted_f1', 'delta_macro_f1', 'flip_rate',
                       'true_probability_change', 'confidence_change', 'js_divergence']:
            self.assertAlmostEqual(result[metric], 0.)
            self.assertAlmostEqual(result[metric + '_ci_low'], 0.)
            self.assertAlmostEqual(result[metric + '_ci_high'], 0.)

    def test_subgroup_bootstrap_retains_original_dialogue_pool_and_empty_draw_is_na(self):
        y = np.array([0, 1])
        p = np.array([[.9, .1], [.1, .9]])
        result = summarize_condition(y, p, p[:, None], np.array([True, False]), np.array([0, 1]),
                                     np.array([[2, 0], [0, 2]]))
        self.assertEqual(result['n_utterances'], 1)
        self.assertEqual(result['delta_accuracy_valid_bootstraps'], 1)
        self.assertEqual(result['original_accuracy_valid_bootstraps'], 1)
        empty = summarize_condition(y, p, p[:, None], np.zeros(2, bool), np.array([0, 1]), np.ones((2, 2)))
        self.assertTrue(np.isnan(empty['perturbed_accuracy']))
        self.assertEqual(empty['delta_accuracy_valid_bootstraps'], 0)

    def test_shared_dialogue_weights_are_deterministic_and_clustered(self):
        a, w = dialogue_bootstrap_weights(np.array(['b', 'a', 'b']), 50, 20261108)
        b, z = dialogue_bootstrap_weights(np.array(['b', 'a', 'b']), 50, 20261108)
        np.testing.assert_array_equal(w, z)
        np.testing.assert_array_equal(a, [1, 0, 1])
        np.testing.assert_array_equal(w.sum(1), np.full(50, 2))

    def test_missing_visual_inactive_prediction_must_be_exact_original(self):
        base = dict(labels=np.array([0, 1]), dialogue_ids=np.array(['a', 'b']), utterance_indices=np.zeros(2, int),
                    probabilities=np.array([[.8, .2], [.2, .8]]), log_probabilities=np.log([[.8, .2], [.2, .8]]),
                    predictions=np.array([0, 1]), speaker_ids=np.zeros(2, int), batch_indices=np.zeros(2, int),
                    dialogue_lengths=np.ones(2, int))
        data = dict(base, probabilities=base['probabilities'][:, None], log_probabilities=base['log_probabilities'][:, None],
                    original_probabilities=base['probabilities'], original_predictions=base['predictions'],
                    perturbed_predictions=base['predictions'][:, None], active=np.array([True, False]),
                    visual_missing=np.array([False, True]))
        validate_payload(base, data, 1, 'V')
        data['probabilities'] = data['probabilities'].copy(); data['probabilities'][1, 0] = [.3, .7]
        data['log_probabilities'] = np.log(data['probabilities'])
        with self.assertRaises(ValueError):
            validate_payload(base, data, 1, 'V')


if __name__ == '__main__':
    unittest.main()
