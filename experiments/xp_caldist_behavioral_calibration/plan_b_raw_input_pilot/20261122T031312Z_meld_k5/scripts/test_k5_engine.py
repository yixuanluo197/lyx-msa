"""CPU synthetic contract checks; no real inputs, encoder, or CMERC loading."""
import copy
import unittest
from unittest import mock

import torch

import run_k5 as engine


class K5Contracts(unittest.TestCase):
    def assessed(self, ids=(10, 11, 12, 13, 14), original_qc=None):
        rows, assessments = [], {}
        for variant, token in zip(engine.FAMILIES['semantic_paraphrase'], ids):
            row = dict(sample_id='synthetic', perturbation_type='semantic_paraphrase', variant_id=variant)
            rows.append(row)
            assessments[('synthetic', variant)] = (
                dict(automatic_text_qc_pass=True), [0, token, 2])
        if original_qc is not None:
            assessments[('synthetic', 'paraphrase_a')][0]['automatic_text_qc_pass'] = original_qc
        engine.apply_duplicate_qc(rows, assessments)
        return assessments

    def flags(self, assessments):
        return [assessments[('synthetic', v)][0]['automatic_text_qc_pass']
                for v in engine.FAMILIES['semantic_paraphrase']]

    def test_five_distinct_tokens_pass(self):
        self.assertEqual(self.flags(self.assessed()), [True] * 5)

    def test_new_collision_preserves_frozen_a(self):
        a = self.assessed((10, 11, 10, 13, 14))
        self.assertEqual(self.flags(a), [True, True, False, True, True])
        self.assertEqual(a[('synthetic', 'paraphrase_a')][0]['same_tokens_other_variants'], ['paraphrase_c'])

    def test_all_colliding_new_versions_fail(self):
        self.assertEqual(self.flags(self.assessed((10, 11, 12, 12, 12))),
                         [True, True, False, False, False])

    def test_new_collision_with_failed_old_still_fails_new(self):
        self.assertEqual(self.flags(self.assessed((10, 11, 10, 13, 14), False)),
                         [False, True, False, True, True])

    def test_frozen_ab_collision_fails_both_and_new_collision(self):
        self.assertEqual(self.flags(self.assessed((10, 10, 12, 10, 14))),
                         [False, False, True, False, True])

    def test_other_target_equal_tokens_are_independent(self):
        rows = [dict(sample_id=s, perturbation_type='semantic_paraphrase', variant_id='paraphrase_a')
                for s in ('one', 'two')]
        assessments = {(r['sample_id'], r['variant_id']):
                       (dict(automatic_text_qc_pass=True), [0, 10, 2]) for r in rows}
        engine.apply_duplicate_qc(rows, assessments)
        self.assertTrue(all(qc['automatic_text_qc_pass'] for qc, _ in assessments.values()))

    def test_ai_pass_requires_attribution_and_allows_no_human_substitution(self):
        row = dict(review_status='PASS', review_method='AI_CONTEXT_SEMANTIC_REVIEW',
                   reviewer='Synthetic AI', review_notes='Synthetic context audit.')
        self.assertTrue(engine.ai_review_pass(row))
        for change in (dict(review_status='UNCERTAIN'), dict(review_status='FAIL'),
                       dict(reviewer=' '), dict(review_notes=''),
                       dict(review_method='HUMAN_CONTEXT_REVIEW')):
            self.assertFalse(engine.ai_review_pass(dict(row, **change)))

    def test_protocol_requires_fixed_24_pairs_48_targets_semantic_five(self):
        p = dict(dataset='MELD', model_seed=67137, test_access=False,
                 matching=dict(expected_targets=48, expected_pairs=24),
                 perturbations=dict(semantic_paraphrase=list(engine.FAMILIES['semantic_paraphrase']), k_per_family=5))
        engine.validate_protocol(p)
        for section, key, value in (('matching', 'expected_targets', 49),
                                    ('matching', 'expected_pairs', 25),
                                    ('perturbations', 'k_per_family', 2),
                                    ('perturbations', 'semantic_paraphrase', ['paraphrase_a', 'paraphrase_b'])):
            altered = copy.deepcopy(p)
            altered[section][key] = value
            with self.assertRaises(ValueError):
                engine.validate_protocol(altered)
        for key, value in (('test_access', True), ('model_seed', 1475), ('dataset', 'IEMOCAP')):
            with self.assertRaises(ValueError):
                engine.validate_protocol(dict(p, **{key: value}))

    def test_all_48_clean_controls_gate_required(self):
        one = dict(clean_feature_allclose=True, clean_probability_allclose=True,
                   original_probability_allclose=True, clean_prediction_unchanged=True)
        rows = [dict(one) for _ in range(48)]
        engine.require_all_clean_controls(rows)
        for bad in (rows[:-1], rows + [dict(one)]):
            with self.assertRaises(RuntimeError):
                engine.require_all_clean_controls(bad)
        for field in one:
            altered = copy.deepcopy(rows)
            altered[-1][field] = False
            with self.assertRaises(RuntimeError):
                engine.require_all_clean_controls(altered)

    def test_24_pairs_each_contain_correct_error_in_one_distinct_dialogue(self):
        rows = [dict(pair_id=f'pair-{i}', dialogue_id=f'dialogue-{i}',
                     original_correct=str(correct))
                for i in range(24) for correct in (True, False)]
        engine.validate_cohort(rows)
        disk_rows = [dict(r, original_correct='1' if r['original_correct']=='True' else '0') for r in rows]
        engine.validate_cohort(disk_rows)
        for field, value in (('original_correct', 'True'), ('original_correct', 'invalid'),
                             ('dialogue_id', 'dialogue-1'),
                             ('pair_id', 'pair-1')):
            bad = copy.deepcopy(rows)
            bad[1][field] = value
            with self.assertRaises(ValueError):
                engine.validate_cohort(bad)

    def test_review_menu_retains_failed_candidates_and_rejects_lost_conditions(self):
        sample = dict(sample_id='synthetic', dialogue_id='dialogue', utterance_index='1')
        dialogues = {'dialogue': dict(texts=['context', 'Original.'])}
        rows = [dict(sample, perturbation_type='semantic_paraphrase', variant_id=v,
                     original_text='Original.', original_text_sha256=engine.text_sha('Original.'),
                     processed_text='Original.' if i else 'Changed.',
                     review_status='FAIL' if i else 'PASS')
                for i, v in enumerate(engine.FAMILIES['semantic_paraphrase'])]
        self.assertEqual(set(engine.validate_reviews([sample], rows, dialogues)), {'synthetic'})
        for bad in (rows[:-1], rows + [rows[0]],
                    [dict(rows[0], original_text='Wrong.')] + rows[1:]):
            with self.assertRaises(ValueError):
                engine.validate_reviews([sample], bad, dialogues)

    def test_target_only_feature_replacement_checks_other_rows_and_modalities(self):
        text = torch.arange(3 * 4 * 1024, dtype=torch.float32).reshape(3, 4, 1024)
        audio = torch.zeros(3, 768)
        changed = text.clone()
        changed[1] += 1
        engine.shared.assert_target_only(text, audio, changed, audio, 1, 'text')
        changed[0, 0, 0] += 1
        with self.assertRaises(AssertionError):
            engine.shared.assert_target_only(text, audio, changed, audio, 1, 'text')

    def test_cached_feature_shape_and_hash_are_enforced(self):
        feature = torch.zeros(4, 1024, dtype=torch.float32)
        record = engine.feature_record({}, feature, [0, 12, 2], feature)
        engine.check_cached_record(record)
        altered = dict(record, feature=feature.clone())
        altered['feature'][0, 0] = 1
        with self.assertRaises(ValueError):
            engine.check_cached_record(altered)
        with self.assertRaises(ValueError):
            engine.feature_record({}, torch.zeros(1024), [0, 12, 2], feature)
        failed = dict(status='FAILED_QC', actual_token_ids=[0, 12, 2],
                      actual_token_ids_sha256=engine.json_hash([0, 12, 2]))
        engine.check_cached_record(failed)

    def test_binding_detects_frozen_input_change(self):
        with mock.patch.object(engine.shared, 'sha256', return_value='new'):
            with self.assertRaises(RuntimeError):
                engine.check_bindings({'synthetic-input': 'old'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
