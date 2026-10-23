"""Synthetic CPU regression tests for preregistered LR qualification and ranking.

Does not read experiment outputs, test artifacts, checkpoints or instantiate models.
"""
from copy import deepcopy
import unittest
from analyze_lr import choose_candidate


def result(name, lr, best, late, complete=True):
    best_delta, late_delta = best - 65., late - 63.
    return {'run_id':name, 'lr':lr, 'complete':complete, 'best_validation_wf1':best,
            'last10_validation_wf1':late, 'delta_best_validation_wf1':best_delta,
            'delta_last10_validation_wf1':late_delta,
            'candidate_category':('strong_candidate' if late_delta >= 0 else 'weak_mixed_candidate') if best_delta > 0 else 'no_best_improvement'}


class QualificationTests(unittest.TestCase):
    def test_higher_weak_peak_cannot_displace_qualified_strong_candidate(self):
        strong=result('qualified',5e-5,65.5,63.)
        weak=result('higher_peak_but_tail_falls',2e-4,66.,62.9)
        answer=choose_candidate([weak,strong])
        self.assertEqual(answer['run_id'],'qualified')
        self.assertEqual(answer['recommended_lr'],5e-5)
        self.assertEqual(answer['decision'],'strong_candidate')

    def test_only_weak_evidence_does_not_recommend_second_seed_candidate(self):
        weak=result('mixed',5e-5,66.,62.9)
        other=result('no_peak_gain',2e-4,65.,64.)
        answer=choose_candidate([weak,other])
        self.assertEqual(answer['decision'],'weak_mixed_candidate')
        self.assertIsNone(answer['recommended_lr'])
        self.assertIsNone(answer['run_id'])
        self.assertEqual(answer['weak_run_ids'],['mixed'])

    def test_strong_candidates_rank_by_best_then_tail(self):
        a=result('higher_peak',2e-4,66.,63.)
        b=result('higher_tail',5e-5,65.9,64.)
        self.assertEqual(choose_candidate([a,b])['run_id'],'higher_peak')
        b['best_validation_wf1']=66.
        b['delta_best_validation_wf1']=1.
        self.assertEqual(choose_candidate([a,b])['run_id'],'higher_tail')

    def test_exact_strong_tie_prefers_lower_registered_lr(self):
        a=result('low',5e-5,66.,63.5)
        b=result('high',2e-4,66.,63.5)
        for rows in ([a,b],[b,a]):
            self.assertEqual(choose_candidate(rows)['run_id'],'low')

    def test_neither_peak_improves_stops_search(self):
        answer=choose_candidate([result('a',5e-5,65.,64.),result('b',2e-4,64.9,64.)])
        self.assertEqual(answer['decision'],'stop_lr_search')
        self.assertIsNone(answer['recommended_lr'])

    def test_incomplete_matrix_never_recommends_candidate(self):
        answer=choose_candidate([result('a',5e-5,66.,64.),result('b',2e-4,66.,64.,False)])
        self.assertEqual(answer['decision'],'pending')
        self.assertIsNone(answer['recommended_lr'])

    def test_candidate_evaluation_does_not_mutate_input(self):
        rows=[result('a',5e-5,66.,64.),result('b',2e-4,66.,64.)]
        before=deepcopy(rows)
        choose_candidate(rows)
        self.assertEqual(rows,before)


if __name__=='__main__':
    unittest.main(verbosity=2)
