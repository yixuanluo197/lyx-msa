"""CPU behavioral tests for the external trace observer; no CMERC training."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

from observe_training import Observer, assert_scientific_pair, rng_hash

ROOT = Path(__file__).resolve().parents[1]
TOY = '''import torch
from sklearn.metrics import accuracy_score, f1_score
forwards = 0
steps = 0
def train_or_eval_graph_model(model, model_mm, model_intra, model_inter, loss_function, dataloader, epoch, cuda, modals, optimizer=None, train=False, dataset='IEMOCAP', return_components=False):
    global forwards, steps
    model.train(train)
    preds, labels = [], []
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        lengths = [1 for _ in data[-1]]
        forwards += 1
        log_prob = model(data[0]).log_softmax(-1)
        label = data[1]
        if args.contrastlearning or args.calibrate:
            log_prob2t = log_prob.detach()
            log_prob2a = log_prob.detach()
            log_prob2v = log_prob.detach()
            log_prob2intra = log_prob.detach()
            log_prob2inter = log_prob.detach()
        classification_loss = loss_function(log_prob, label)
        total_loss = classification_loss
        components = dict(classification_loss=classification_loss.item(), cc_loss=0., mscl_loss=0., cscl_loss=0., total_loss=total_loss.item())
        if train:
            total_loss.backward()
            optimizer.step()
            steps += 1
        labels.extend(label.tolist())
        preds.extend(log_prob.argmax(1).tolist())
    return (float(classification_loss), round(accuracy_score(labels,preds)*100,2), labels, preds, round(f1_score(labels,preds,average='weighted')*100,2))
'''


class ObserverTests(unittest.TestCase):
    def base(self, directory):
        historic = {'split': {'train_ids': [1, 2], 'validation_ids': [3], 'test_ids': [4]},
                    'data': {}, 'source': {'files': {'train_cm.py': 'source'}}}
        run = {'code_dir': str(ROOT / 'code_xp04'), 'epochs': 2, 'dataset': 'MELD',
               'factor_key': 'calibrate', 'factor_value': False}
        observer = Observer(run, historic, directory, target=Path(directory).parent / 'toy.py')
        observer.validated = True
        return observer

    def test_trace_preserves_calls_updates_and_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            path = temp / 'toy.py'
            path.write_text(TOY)
            def execute(traced):
                torch.manual_seed(77)
                np.random.seed(77)
                import random
                random.seed(77)
                namespace = {'args': argparse.Namespace(calibrate=False, contrastlearning=True)}
                exec(compile(TOY, str(path), 'exec'), namespace)
                model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(.2), torch.nn.Linear(3, 2))
                optimizer = torch.optim.Adam(model.parameters(), lr=.01)
                observer = self.base(temp / 'observed') if traced else None
                class Loader:
                    def __init__(self, keys):
                        self.dataset = argparse.Namespace(keys=keys)
                        self.data = (torch.ones(len(keys), 2), torch.zeros(len(keys), dtype=torch.long), keys)
                    def __iter__(self):
                        return iter([self.data])
                if traced:
                    sys.settrace(observer.trace)
                try:
                    for epoch in range(2):
                        for train, keys in [(True, [2, 1]), (False, [3])]:
                            namespace['train_or_eval_graph_model'](model, None, None, None, torch.nn.NLLLoss(), Loader(keys), epoch, False, 'avl', optimizer, train)
                    namespace['train_or_eval_graph_model'](model, None, None, None, torch.nn.NLLLoss(), Loader([4]), 0, False, 'avl')
                finally:
                    sys.settrace(None)
                return namespace['forwards'], namespace['steps'], rng_hash(), {k:v.clone() for k,v in model.state_dict().items()}, observer
            plain = execute(False)
            traced = execute(True)
            self.assertEqual(plain[:3], traced[:3])
            for key in plain[3]:
                self.assertTrue(torch.equal(plain[3][key], traced[3][key]))
            self.assertEqual(traced[4].completed, {'train': 2, 'validation': 2, 'test': 1})
            self.assertTrue((temp / 'observed/predictions.npz').exists())

    def test_test_access_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            observer = self.base(Path(temp) / 'out')
            with self.assertRaises(AssertionError):
                observer.split_for(False, [4])
            observer.completed.update(train=2, validation=2)
            self.assertEqual(observer.split_for(False, [4]), 'test')
            observer.completed['test'] = 1
            with self.assertRaises(AssertionError):
                observer.split_for(False, [4])
            with self.assertRaises(AssertionError):
                observer.split_for(True, [1, 3])

    def test_scientific_difference_gate(self):
        split = {'train_ids': [1], 'validation_ids': [2], 'test_ids': [3]}
        provenance = {'split': split, 'data': {}, 'source': {'files': {'train_cm.py': 'sha'}}}
        old = dict(calibrate=True, lr=.0001, output_dir='/historical', provenance=provenance,
                   selection_protocol='validation', rng_policy='continuous', rnn_backend='aten', epochs=2)
        historical = {'args': old, **provenance}
        actual = argparse.Namespace(**{**old, 'calibrate': False, 'output_dir': '/new'})
        run = dict(dataset='MELD', factor_key='calibrate', factor_value=False, epochs=2)
        assert_scientific_pair(actual, historical, run)
        actual.lr = .01
        with self.assertRaises(AssertionError):
            assert_scientific_pair(actual, historical, run)


if __name__ == '__main__':
    unittest.main()
