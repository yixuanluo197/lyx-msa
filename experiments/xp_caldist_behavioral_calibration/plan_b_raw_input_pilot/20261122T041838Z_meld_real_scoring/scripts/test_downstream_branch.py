"""CPU synthetic checks of the restricted adapter; no model construction."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import downstream_branch as branch


class BranchContracts(unittest.TestCase):
    def setUp(self):
        self.patch = mock.patch.multiple(branch, EXPECTED_TRAIN_UTTERANCES=5,
                                        EXPECTED_DIALOGUES=3, EXPECTED_FIT_DIALOGUES=2,
                                        EXPECTED_DEV_DIALOGUES=1)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def bundle(self):
        train = {}
        text, audio = {'train': {}, 'validation': {}}, {'train': {}, 'validation': {}}
        for i, n in enumerate((2, 1, 2)):
            did = str(i)
            train[did] = dict(dialogue_id=did, utterance_ids=list(range(n)), texts=['x'] * n,
                              labels=torch.arange(n), visual=torch.zeros(n, 4),
                              speaker=torch.eye(2)[torch.arange(n) % 2])
            text['train'][did] = torch.zeros(n, 4, 1024)
            audio['train'][did] = torch.zeros(n, 768)
        b = SimpleNamespace(data=dict(dataset='MELD', splits=dict(train=list(train.values()), validation=[])),
                            train=train, fit=['0', '1'], dev=['2'], text=text, audio=audio)
        b.features = lambda split, did: (b.text[split][did], b.audio[split][did])
        return b

    def test_restricted_counts_and_dialogue_relative_utterance_ids(self):
        report = branch.validate_bundle(self.bundle())
        self.assertEqual((report['fit_utterances'], report['inner_dev_utterances']), (3, 2))
        self.assertFalse(report['scoring_groups_loaded'])

    def test_new_protocol_identity_and_base_counts_are_strict(self):
        protocol = dict(dataset='MELD', model_seed=67137, test_access=False,
                        base=dict(fit_dialogues=656, dev_dialogues=72,
                                  fit_utterances=6313, dev_utterances=700))
        branch.validate_protocol(protocol)
        for key, value in [('dataset', 'IEMOCAP'), ('model_seed', 1), ('test_access', True)]:
            with self.assertRaises(ValueError):
                branch.validate_protocol(dict(protocol, **{key: value}))
        wrong = copy.deepcopy(protocol)
        wrong['base']['fit_utterances'] = 6314
        with self.assertRaises(ValueError): branch.validate_protocol(wrong)

    def test_rejects_validation_or_test_split(self):
        for key in ('validation', 'test'):
            b = self.bundle()
            b.data['splits'][key] = [b.train['0']]
            with self.assertRaises(ValueError):
                branch.validate_bundle(b)

    def test_rejects_feature_rows_outside_base(self):
        for name in ('text', 'audio'):
            b = self.bundle()
            getattr(b, name)['train']['held_out'] = torch.zeros(1)
            with self.assertRaises(ValueError):
                branch.validate_bundle(b)

    def test_rejects_fit_dev_overlap_or_duplicate(self):
        for fit, dev in ((['0', '1'], ['1']), (['0', '0'], ['2'])):
            b = self.bundle()
            b.fit, b.dev = fit, dev
            with self.assertRaises(ValueError):
                branch.validate_bundle(b)

    def test_rejects_bad_shape_nonfinite_and_duplicate_local_ids(self):
        b = self.bundle()
        b.audio['train']['0'] = torch.zeros(2, 300)
        with self.assertRaises(ValueError): branch.validate_bundle(b)
        b = self.bundle()
        b.text['train']['0'][0, 0, 0] = float('nan')
        with self.assertRaises(ValueError): branch.validate_bundle(b)
        b = self.bundle()
        b.train['0']['utterance_ids'] = [0, 0]
        with self.assertRaises(ValueError): branch.validate_bundle(b)

    def test_source_adaptation_is_only_guard_and_completion_identity(self):
        expected = branch.ORIGINAL_TRAIN_SOURCE.replace(
            branch._old_guard, '    _branch_validate_bundle(b,dataset)\n').replace(
                "'adapter_sha256':sha(__file__)", "'adapter_sha256':sha(_branch_entrypoint)")
        self.assertEqual(branch.ADAPTED_TRAIN_SOURCE, expected)
        self.assertIn("for did in b.fit:", __import__('inspect').getsource(branch.parent.build_difficulty))

    def test_training_delegates_to_same_parent_and_enforces_epoch_budget(self):
        b = self.bundle()
        with mock.patch.object(branch.parent, 'train', return_value='new-best') as train:
            self.assertEqual(branch.train(b, 'm3net', 'cuda:0', 'resume.pt'), 'new-best')
            train.assert_called_once_with(b, 'MELD', 'm3net', 'cuda:0', 'resume.pt', None)
        with self.assertRaises(ValueError): branch.train(b, 'm3net', max_epochs=1)
        with self.assertRaises(ValueError): branch.train(b, 'other')

    def test_same_anchor_and_flags_without_model_creation(self):
        with mock.patch.object(branch, 'branch_identity', return_value={'synthetic': True}):
            m = branch.config('MELD', 342, 9, 'm3net')
            c = branch.config('MELD', 342, 9, 'cmerc')
        for config in (m, c):
            self.assertEqual((config['epochs'], config['patience'], config['selection_metric']),
                             (40, 10, 'macro_f1'))
            self.assertEqual(config['audio_dim'], 768)
        self.assertFalse(any(m[k] for k in ('calibrate', 'contrastlearning', 'courselearning')))
        self.assertTrue(all(c[k] for k in ('calibrate', 'contrastlearning', 'courselearning')))

    def test_original_split_manifest_guards_exact_ids_and_order(self):
        b = self.bundle()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'local_only').mkdir()
            ds = [dict(dialogue_id=did, role='base_fit' if did in b.fit else 'base_inner_dev',
                       utterance_count=len(d['utterance_ids'])) for did, d in b.train.items()]
            us = [dict(dialogue_id=r['dialogue_id'], utterance_id=str(uid), utterance_index=i, role=r['role'])
                  for r in ds for i, uid in enumerate(b.train[r['dialogue_id']]['utterance_ids'])]
            for name, rows in [('SPLIT_DIALOGUES.csv', ds), ('SPLIT_UTTERANCES.csv', us)]:
                with (root / 'local_only' / name).open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
            lock = dict(dataset='MELD', no_test_data=True, official_validation_rows_used=0,
                        hashes={name: branch.sha(root / 'local_only' / name)
                                for name in ('SPLIT_DIALOGUES.csv', 'SPLIT_UTTERANCES.csv')})
            (root / 'SPLIT_LOCK.json').write_text(json.dumps(lock))
            with mock.patch.object(branch, 'RUN', root):
                branch.validate_split_lock(b)
                b.train['0']['utterance_ids'].reverse()
                with self.assertRaises(ValueError): branch.validate_split_lock(b)


if __name__ == '__main__':
    unittest.main(verbosity=2)
