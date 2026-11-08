"""CPU-only protocol checks; fake forwards never load CMERC or any data split."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

import run_evidence as runner


def synthetic_batch():
    # Two complete dialogues, with the second dialogue padded to length two.
    arrays = [torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) + 10*j + 1 for j in range(6)]
    arrays[4][0, 1] = 0  # One genuinely missing visual utterance.
    qmask = torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [0., 0.]]])
    umask = torch.tensor([[1., 1.], [1., 0.]])
    labels = torch.tensor([[0, 1], [1, 999]])
    return arrays + [qmask, umask, labels, ['dialogue-0', 'dialogue-1']]


class RunnerControlTests(unittest.TestCase):
    def test_metadata_is_dialogue_major_and_excludes_padding(self):
        batch = synthetic_batch()
        before = [x.clone() for x in batch[:-1]]
        md = runner.metadata(batch, [2, 1], 7)
        np.testing.assert_array_equal(md['dialogue_ids'], ['dialogue-0', 'dialogue-0', 'dialogue-1'])
        np.testing.assert_array_equal(md['utterance_indices'], [0, 1, 0])
        np.testing.assert_array_equal(md['labels'], [0, 1, 1])
        np.testing.assert_array_equal(md['speaker_ids'], [0, 1, 1])
        np.testing.assert_array_equal(md['dialogue_lengths'], [2, 2, 1])
        np.testing.assert_array_equal(md['batch_indices'], [7, 7, 7])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(batch[:-1], before)))

    def test_compare_rejects_nonfinite_shape_and_argmax_drift(self):
        a = np.array([[.5, .5]], np.float32)
        self.assertEqual(runner.compare(a, a.copy()), 0)
        # Numerically within tolerance, but changing a tied argmax is prohibited.
        for bad in [np.array([[.5, .5000001]], np.float32), np.array([[np.nan, .5]]), np.ones((2, 2))]:
            with self.assertRaises(AssertionError): runner.compare(a, bad)

    def test_loader_preserves_batch16_and_original_dialogue_order(self):
        class Dataset(torch.utils.data.Dataset):
            def __len__(self): return 18
            def __getitem__(self, i): return i
            @staticmethod
            def collate_fn(values): return values
        loader = runner.old.loader_for(Dataset(), torch)
        self.assertEqual(loader.batch_size, 16)
        self.assertEqual(list(loader), [list(range(16)), [16, 17]])

    def test_gate_checks_hashes_phase_order_and_rejects_unknown_dataset(self):
        with tempfile.TemporaryDirectory(prefix='evidence-gate-test-') as temp:
            root = Path(temp); (root/'provenance').mkdir()
            config = root/'config.json'; config.write_text('{}')
            artifact = root/'artifact.bin'; artifact.write_bytes(b'unit-test-only')
            item = {'data_sha256': {}}
            for name in ['checkpoint', 'effective_config', 'train_statistics', 'validation_baseline']:
                item[name+'_path'] = str(artifact); item[name+'_sha256'] = runner.sha(artifact)
            audit = dict(status='passed', datasets={'IEMOCAP': item}, frozen_source_sha256={})
            src = dict(config_sha256=runner.sha(config), inference_scripts={})
            (root/'provenance/audit.json').write_text(json.dumps(audit))
            (root/'provenance/source_hashes.json').write_text(json.dumps(src))
            parity = root/'provenance/graph_cache_parity.json'; parity.write_text('{"passed":true}')
            with mock.patch.object(runner, 'ROOT', root), mock.patch.object(runner, 'CONFIG', config):
                self.assertEqual(runner.gate('IEMOCAP', 'overall_attenuation'), item)
                with self.assertRaises(KeyError): runner.gate('Test', 'overall_attenuation')
                with self.assertRaises(FileNotFoundError): runner.gate('IEMOCAP', 'layer_attenuation')
                for ds, seed in runner.CFG['anchors'].items():
                    path = root/f'runs/{ds}/seed{seed}/overall_attenuation_complete.json'
                    path.parent.mkdir(parents=True); path.write_text('{"passed":true}')
                self.assertEqual(runner.gate('IEMOCAP', 'layer_attenuation'), item)
                parity.write_text('{"passed":false}')
                with self.assertRaises(AssertionError): runner.gate('IEMOCAP', 'overall_attenuation')
                parity.write_text('{"passed":true}'); artifact.write_bytes(b'changed')
                with self.assertRaises(AssertionError): runner.gate('IEMOCAP', 'overall_attenuation')

    def test_complete_runner_only_changes_one_target_and_preserves_missing_visual(self):
        batch = synthetic_batch(); snapshots = [x.clone() for x in batch[:-1]]
        modified_calls = []

        def forward(model, tensors, lengths):
            self.assertEqual(lengths, [2, 1])
            self.assertEqual([tuple(x.shape) for x in tensors], [tuple(x.shape) for x in snapshots])
            changed_positions = set()
            for j, (x, old) in enumerate(zip(tensors, snapshots)):
                different = x != old
                if j < 6:
                    changed_positions.update(tuple(row) for row in different.any(-1).nonzero().tolist())
                else:
                    self.assertTrue(torch.equal(x, old), 'Mask/label changed during forward')
            self.assertLessEqual(len(changed_positions), 1)
            for ui, di in changed_positions: self.assertLess(ui, lengths[di])
            modified_calls.append(changed_positions)
            total = sum(tensors[j].sum(-1) for j in range(6)) / 100
            flat = torch.cat([total[:n, di] for di, n in enumerate(lengths)])
            return torch.log_softmax(torch.stack([flat, -flat], -1), -1)

        model = torch.nn.Module(); model.graph_model = object()
        base = runner.metadata(batch, [2, 1], 0)
        base['log_probabilities'] = forward(model, batch[:-1], [2, 1]).numpy()
        base['probabilities'] = np.exp(base['log_probabilities'])
        base['predictions'] = base['probabilities'].argmax(-1)
        modified_calls.clear()

        class FakeDataset:
            def __init__(self, keys):
                if keys != batch[-1]: raise AssertionError('Requested a split other than validation')
                self.keys = keys
            def __len__(self): return 2

        class Guard:
            def __init__(self, *args): self.count = 0
            def reset(self): pass
            def close(self): pass

        cache = types.SimpleNamespace(stats={}, restore=lambda: None, enable=lambda: None)
        dl = types.SimpleNamespace(IEMOCAPDataset=FakeDataset)
        api = (torch, None, dl, None, None)
        with tempfile.TemporaryDirectory(prefix='evidence-runner-test-') as temp:
            root = Path(temp); (root/'provenance').mkdir(); (root/'logs').mkdir()
            config = root/'config.json'; config.write_text('{}')
            baseline = root/'validation.npz'; np.savez_compressed(baseline, **base)
            statistics = root/'train-statistics.npz'
            np.savez_compressed(statistics, **{name+'_mean': np.full(2, -10, np.float32)
                for name in ['text0', 'text1', 'text2', 'text3', 'audio', 'visual']})
            item = dict(validation_dialogues=2, validation_baseline_path=str(baseline),
                        train_statistics_path=str(statistics), all_zero_features={'visual': [dict(dialogue_id='dialogue-1', utterance_index=0)]})
            prior_cwd = Path.cwd()
            try:
                with contextlib.ExitStack() as stack:
                    for name, value in [('ROOT', root), ('CONFIG', config)]: stack.enter_context(mock.patch.object(runner, name, value))
                    stack.enter_context(mock.patch.object(runner, 'gate', return_value=item))
                    stack.enter_context(mock.patch.object(runner, 'install_graph_cache', return_value=cache))
                    stack.enter_context(mock.patch.object(runner.old, 'imports', return_value=api))
                    stack.enter_context(mock.patch.object(runner.old, 'CODE', root))
                    stack.enter_context(mock.patch.object(runner.old, 'config_for', return_value={'provenance': {'split': {'validation_ids': batch[-1], 'test_ids': ['forbidden']}}}))
                    stack.enter_context(mock.patch.object(runner.old, 'resource_snapshot', return_value={}))
                    stack.enter_context(mock.patch.object(runner.old, 'load_model', return_value=model))
                    stack.enter_context(mock.patch.object(runner.old, 'loader_for', return_value=[batch]))
                    stack.enter_context(mock.patch.object(runner.old, 'forward', side_effect=forward))
                    stack.enter_context(mock.patch.object(runner.old, 'GraphGuard', Guard))
                    # Mimic the independent storage of the real CPU -> CUDA copy.
                    stack.enter_context(mock.patch.object(torch.Tensor, 'cuda', lambda value, *args, **kwargs: value.clone()))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    runner.run('IEMOCAP', 'overall_attenuation')
            finally:
                import os
                os.chdir(prior_cwd)
            output = root/'runs/IEMOCAP/seed1475'
            done = json.loads((output/'overall_attenuation_complete.json').read_text())
            self.assertEqual(done['target_forwards'], 32)  # 3*12 minus 4 missing-visual targets
            self.assertEqual(done['skipped_visual_missing'], 4)
            self.assertFalse(done['test_inference'])
            self.assertTrue(any(changed for changed in modified_calls))
            for path in (output/'conditions').glob('*.npz'):
                with np.load(path) as z:
                    np.testing.assert_array_equal(z['perturbed_predictions'], z['probabilities'].argmax(-1))
                    self.assertEqual(z['probabilities'].shape, (3, 1, 2))
                    if str(z['target']) == 'V':
                        np.testing.assert_array_equal(z['active'], [True, True, False])
                        np.testing.assert_array_equal(z['probabilities'][2, 0], base['probabilities'][2])
                        np.testing.assert_array_equal(z['log_probabilities'][2, 0], base['log_probabilities'][2])
            self.assertTrue(all(torch.equal(x, old) for x, old in zip(batch[:-1], snapshots)))


if __name__ == '__main__': unittest.main()
