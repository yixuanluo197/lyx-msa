"""CPU-only, in-memory interface tests; no training files or CUDA use."""
import argparse
import ast
import copy
import functools
import hashlib
import json
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch

from watch_training import (ROOT, PreflightComplete, TrainingGuard,
                            assert_scientific_pair, rng_hash, verify_skip_patch)

TOY = '''import torch
from sklearn.metrics import accuracy_score, f1_score
forwards = 0
steps = 0
def train_or_eval_graph_model(model, model_mm, model_intra, model_inter, loss_function, dataloader, epoch, cuda, modals, optimizer=None, train=False, dataset='IEMOCAP', return_components=False):
    global forwards, steps
    model.train(train)
    preds, labels, loss_components = [], [], []
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        lengths = [1 for _ in data[-1]]
        forwards += 1
        log_prob = model(data[0]).log_softmax(-1)
        label = data[1]
        classification_loss = loss_function(log_prob, label)
        cc, mscl, cscl = classification_loss*.1, classification_loss*.2, classification_loss*.3
        total_loss = classification_loss + args.rank_coff*cc + args.mscl_coff*mscl + args.cscl_coff*cscl
        components = dict(classification_loss=classification_loss.item(),cc_loss=cc.item(),mscl_loss=mscl.item(),cscl_loss=cscl.item(),total_loss=total_loss.item())
        loss_components.append(components)
        if train:
            total_loss.backward()
            optimizer.step()
            steps += 1
        labels.extend(label.tolist())
        preds.extend(log_prob.argmax(1).tolist())
    return (float(classification_loss),round(accuracy_score(labels,preds)*100,2),labels,preds,round(f1_score(labels,preds,average='weighted')*100,2))
'''


def fixture():
    split = {'train_ids': list(range(1,18)), 'validation_ids': list(range(101,110)), 'test_ids': [201], 'hash': 'split'}
    provenance = {'split': split, 'data': {},
                  'source': {'files': {'train_cm.py': 'old', 'model.py': 'unchanged'}}}
    old = dict(lr=.0001, batch_size=16, skip_final_test=False, output_dir='/old', provenance=provenance,
               seed=1475, epochs=2, selection_protocol='validation', testing=False,
               rng_policy='continuous', rnn_backend='aten', batch_policy='capacity_safe',
               num_workers=0, calibrate=True, contrastlearning=True, courselearning=True,
               rank_coff=.005, mscl_coff=.05, cscl_coff=.05)
    historical = {'args': copy.deepcopy(old), **copy.deepcopy(provenance)}
    actual = copy.deepcopy(old)
    actual.update(batch_size=8, skip_final_test=True, output_dir='/new')
    actual['provenance']['source']['files']['train_cm.py'] = 'patched'
    run = dict(lr=.0001, batch_size=8, validation_batch_size=8, seed=1475, epochs=2, split_hash='split')
    return argparse.Namespace(**actual), historical, run


class Loader:
    def __init__(self, keys):
        self.dataset = argparse.Namespace(keys=keys, videoLabels={key: [0] for key in keys})
        self.items = [(torch.ones(len(group), 2), torch.zeros(len(group), dtype=torch.long), group)
                      for group in [keys[start:start+8] for start in range(0,len(keys),8)]]

    def __iter__(self):
        return iter(self.items)


class GuardTests(unittest.TestCase):
    def test_scientific_gate_rejects_undeclared_config_or_source(self):
        args, old, run = fixture()
        assert_scientific_pair(args, old, run, 'patched')
        args.rank_coff = .01
        with self.assertRaises(AssertionError):
            assert_scientific_pair(args, old, run, 'patched')
        args, old, run = fixture()
        args.skip_final_test = False
        with self.assertRaises(AssertionError):
            assert_scientific_pair(args, old, run, 'patched')
        args, old, run = fixture()
        args.provenance['source']['files']['model.py'] = 'changed'
        with self.assertRaises(AssertionError):
            assert_scientific_pair(args, old, run, 'patched')

    def test_batch_factor_requires_eight_and_forbids_lr_seed_or_loss_changes(self):
        args, old, run = fixture()
        assert_scientific_pair(args, old, run, 'patched')
        for field, value in [('lr', .0002), ('seed', 2024), ('batch_size', 16),
                             ('mscl_coff', .2), ('calibrate', False), ('split_seed', 2024)]:
            actual, historical, registered = fixture()
            setattr(actual, field, value)
            with self.subTest(field=field), self.assertRaises(AssertionError):
                assert_scientific_pair(actual, historical, registered, 'patched')

    def test_test_loader_rejected_before_any_model_call(self):
        args, old, run = fixture()
        target = str(ROOT / '<memory-toy>')
        scope = {'args': args}
        exec(compile(TOY, target, 'exec'), scope)
        guard = TrainingGuard(run, old, target, 'patched')
        model = torch.nn.Linear(2, 2)
        original = sys.gettrace()
        sys.settrace(guard.trace)
        try:
            with self.assertRaisesRegex(AssertionError, 'Forbidden evaluation'):
                scope['train_or_eval_graph_model'](model,None,None,None,torch.nn.NLLLoss(),Loader([201]),0,False,'avl')
        finally:
            sys.settrace(original)
        self.assertEqual(scope['forwards'], 0)
        self.assertEqual(guard.counts['test'], 0)
        with self.assertRaises(AssertionError):
            guard.split_for(True, [1, 101])

    def execute(self, observed, preflight=False):
        torch.manual_seed(77)
        np.random.seed(77)
        random.seed(77)
        args, old, run = fixture()
        target = str(ROOT / '<memory-toy>')
        scope = {'args': args}
        exec(compile(TOY, target, 'exec'), scope)
        model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(.2), torch.nn.Linear(3, 2))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        guard = TrainingGuard(run, old, target, 'patched', preflight=preflight)
        old_step, old_trace = torch.optim.Adam.step, sys.gettrace()
        @functools.wraps(old_step)
        def watched_step(opt, *positional, **keywords):
            return guard.adam_step(old_step, opt, *positional, **keywords)
        if observed:
            torch.optim.Adam.step = watched_step
            sys.settrace(guard.trace)
        stopped = False
        try:
            for epoch in range(2):
                for train, keys in [(True, list(range(17,0,-1))), (False, list(range(101,110)))]:
                    scope['train_or_eval_graph_model'](model,None,None,None,torch.nn.NLLLoss(),Loader(keys),epoch,False,'avl',optimizer,train)
        except PreflightComplete:
            stopped = True
        finally:
            sys.settrace(old_trace)
            torch.optim.Adam.step = old_step
        return {'forwards': scope['forwards'], 'steps_counter': scope['steps'],
                'rng': rng_hash(), 'state': {k:v.clone() for k,v in model.state_dict().items()},
                'optimizer': copy.deepcopy(optimizer.state_dict()), 'guard': guard, 'stopped': stopped}

    def test_observation_preserves_rng_updates_and_call_counts(self):
        plain, watched = self.execute(False), self.execute(True)
        self.assertEqual(plain['forwards'], watched['forwards'])
        self.assertEqual(plain['steps_counter'], watched['steps_counter'])
        self.assertEqual(watched['forwards'], 10)
        self.assertEqual(watched['steps_counter'], 6)
        self.assertEqual(plain['rng'], watched['rng'])
        for key, value in plain['state'].items():
            self.assertTrue(torch.equal(value, watched['state'][key]))
        for key, values in plain['optimizer']['state'].items():
            for name, value in values.items():
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(value, watched['optimizer']['state'][key][name]))
                else:
                    self.assertEqual(value, watched['optimizer']['state'][key][name])
        guard = watched['guard']
        self.assertEqual(guard.counts, dict(train=2, validation=2, test=0))
        self.assertEqual(guard.line_events, 0)
        recorded = [row for name,row in guard.memory_records if name == 'watch_batches.jsonl']
        for epoch in (1, 2):
            steps = [row for row in recorded if row['epoch'] == epoch]
            self.assertEqual([len(row['dialogue_ids']) for row in steps], [8,8,1])
            self.assertEqual([key for row in steps for key in row['dialogue_ids']], list(range(17,0,-1)))
        evaluations = [row for name,row in guard.memory_records if name == 'watch_epochs.jsonl' and row['split']=='validation']
        self.assertTrue(all(row['batches'] == 2 and row['n_utterances'] == 9 for row in evaluations))
        self.assertTrue(guard.first['optimizer_updated_parameters'])
        self.assertEqual(guard.first['rng_before_step'], guard.first['rng_after_step'])

    def test_preflight_stops_after_the_first_completed_update(self):
        result = self.execute(True, preflight=True)
        self.assertTrue(result['stopped'])
        self.assertEqual(result['forwards'], 1)
        # The guard raises immediately after Adam returns, before this toy counter.
        self.assertEqual(result['steps_counter'], 0)
        self.assertEqual(result['guard'].counts, dict(train=0, validation=0, test=0))
        self.assertTrue(result['guard'].first['optimizer_updated_parameters'])
        self.assertEqual(len(result['optimizer']['state']), 4)

    def test_patch_reverses_exactly_and_training_loop_ast_is_identical(self):
        path = ROOT / 'code_xp04/train_cm.py'
        historical_path = ROOT.parents[2] / 'experiment_records/04_xp_mainline_validation/code/train_cm.py'
        old = historical_path.read_bytes()
        verify_skip_patch(path, hashlib.sha256(old).hexdigest())
        def epoch_loop(raw):
            tree = ast.parse(raw)
            block = next(n for n in tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                         and isinstance(n.test.left, ast.Name) and n.test.left.id == '__name__')
            return next(n for n in block.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'e')
        self.assertEqual(ast.dump(epoch_loop(old), include_attributes=False),
                         ast.dump(epoch_loop(path.read_bytes()), include_attributes=False))

    def test_original_and_current_batch8_samplers_match_with_partial_batch(self):
        from check_capacity import surface, replay
        original = ROOT.parents[2] / 'experiment_records/04_xp_mainline_validation/code/train_cm.py'
        old, old_ast = surface(original)
        new, new_ast = surface(ROOT / 'code_xp04/train_cm.py')
        self.assertEqual(old_ast, new_ast)
        keys = list(range(17))
        labels = {key: list(range(1 + key % 3)) for key in keys}
        before = torch.get_rng_state().clone()
        a, b = replay(old,keys,labels,8,2024), replay(new,keys,labels,8,2024)
        self.assertEqual(a[0], b[0])
        self.assertTrue(torch.equal(a[1], b[1]))
        self.assertEqual(a[2:], b[2:])
        self.assertEqual([len(batch) for batch in b[0]], [8,8,1])
        historical16 = replay(old,keys,labels,16,2024)
        self.assertEqual([len(batch) for batch in historical16[0]], [16,1])
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_validation_batch8_preserves_every_dialogue_and_tail(self):
        from check_capacity import validation_plan
        keys = list(range(17))
        labels = {key:[0,0] for key in keys}
        current, current_cap = validation_plan(keys,labels,8)
        historical, historical_cap = validation_plan(keys,labels,16)
        self.assertEqual(sum(current,[]), list(range(17)))
        self.assertEqual(sum(historical,[]), list(range(17)))
        self.assertEqual([len(batch) for batch in current], [8,8,1])
        self.assertEqual([len(batch) for batch in historical], [16,1])
        self.assertNotEqual(current_cap, historical_cap)


if __name__ == '__main__':
    unittest.main()
