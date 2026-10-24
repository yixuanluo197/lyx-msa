"""CPU audit of every registered train/validation batch, using real lengths."""
import ast
import hashlib
import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


class Indices(Dataset):
    def __init__(self, keys, labels):
        self.keys, self.videoLabels = list(keys), labels

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return index

    @staticmethod
    def collate_fn(items):
        return list(items)


def surface(path):
    names = {'CapacitySafeBatchSampler', 'validation_training_loader'}
    nodes = [n for n in ast.parse(path.read_bytes()).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    scope = {'torch': torch, 'DataLoader': DataLoader, 'Sampler': Sampler}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope, {n.name: ast.dump(n, include_attributes=False) for n in nodes}


def limits(keys, labels, batches):
    totals = [sum(len(labels[keys[i]]) for i in batch) for batch in batches]
    return {'edges': max((n + 3*len(batch) for n, batch in zip(totals, batches)), default=0),
            'nodes': max((3*n for n in totals), default=0)}


def main():
    registry = json.loads((ROOT/'configs/registry.json').read_text())
    references = json.loads((ROOT/'provenance/xp04_validation_references.json').read_text())
    old, old_ast = surface(REPO/'experiment_records/04_xp_mainline_validation/code/train_cm.py')
    new, new_ast = surface(ROOT/'code_xp04/train_cm.py')
    assert old_ast == new_ast, 'Sampler or training-loader math changed'
    before = torch.get_rng_state().clone()
    datasets, records = {}, []
    for run in registry['runs']:
        ds = run['dataset']
        ref = references[f"{ds}_{run['seed']}"]
        if ds not in datasets:
            name = 'IEMOCAP_features.pkl' if ds == 'IEMOCAP' else 'MELD_features_raw1.pkl'
            with (ROOT/'code_xp04/data'/name).open('rb') as stream:
                raw = pickle.load(stream, encoding='latin1')
            with (ROOT/'code_xp04/caldiff'/f'{ds}_diff.pkl').open('rb') as stream:
                difficulty = pickle.load(stream)
            # Only train/validation lengths are examined; no test items are accessed.
            eligible = ref['split']['train_ids'] + ref['split']['validation_ids']
            labels = {key: raw[2][key] for key in eligible}
            del raw
            datasets[ds] = labels, difficulty
        labels, difficulty = datasets[ds]
        train, valid = ref['split']['train_ids'], ref['split']['validation_ids']
        assert not set(train).intersection(valid)
        assert not set(train+valid).intersection(ref['split']['test_ids'])
        valid_batches = [list(range(start, min(start+16, len(valid))))
                         for start in range(0, len(valid), 16)]
        valid_limit = limits(valid, labels, valid_batches)
        assert valid_limit['edges'] <= 1000 and valid_limit['nodes'] <= 5200
        for epoch, old_row in enumerate(ref['epochs']):
            ratio = old_row['sample_ratio']
            chosen = set(sorted(train, key=lambda key: difficulty[key])[:int(ratio*len(train))])
            keys = [key for key in train if key in chosen]
            results = []
            for code in [old, new]:
                generator = torch.Generator().manual_seed(run['seed']+epoch)
                loader = code['validation_training_loader'](
                    Indices(keys, labels), 16, {'generator': generator},
                    'capacity_safe', 1000, 5200, 10000, 0, False)
                batches = list(loader)
                assert sorted(sum(batches, [])) == list(range(len(keys)))
                cap = limits(keys, labels, batches)
                assert cap['edges'] <= 1000 and cap['nodes'] <= 5200
                assert cap['edges'] == loader.batch_sampler.max_batch_edges
                assert cap['nodes'] == loader.batch_sampler.max_batch_nodes
                results.append((batches, generator.get_state(), loader.batch_sampler.retries, cap))
            a, z = results
            assert a[0] == z[0] and torch.equal(a[1], z[1]) and a[2:] == z[2:]
            assert z[2] == old_row['sampler_retries']
            assert z[3]['edges'] == old_row['max_batch_edges']
            assert z[3]['nodes'] == old_row['max_batch_nodes']
            records.append({'run_id': run['run_id'], 'epoch': epoch+1,
                            'sample_ratio': ratio, 'train_dialogues': len(keys),
                            'train_batches': len(z[0]), 'validation_batches': len(valid_batches),
                            'validation_capacity': valid_limit, 'capacity': z[3],
                            'sampler_retries': z[2], 'private_seed': run['seed']+epoch,
                            'same_as_historical_sampler': True,
                            'ordered_train_ids_sha256': hashlib.sha256(json.dumps(
                                [keys[i] for i in sum(z[0], [])]).encode()).hexdigest()})
    assert len(records) == sum(r['epochs'] for r in registry['runs']) and torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()
    value = {'status': 'passed', 'epochs_checked': len(records),
             'original_sampling_ast_unchanged': True, 'global_cpu_rng_unchanged': True,
             'cuda_initialized': False, 'test_lengths_or_predictions_examined': False,
             'limits': 'CPU graph capacity does not establish GPU memory safety.',
             'records': records}
    output = ROOT/'preflight/capacity_cpu.json'
    with output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value[key] for key in ['status', 'epochs_checked', 'cuda_initialized']}))


if __name__ == '__main__':
    main()
