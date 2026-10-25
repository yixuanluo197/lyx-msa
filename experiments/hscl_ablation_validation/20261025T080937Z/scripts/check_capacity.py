"""CPU audit of identical current/historical batch16 sampling; no test access."""
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


def replay(code, keys, labels, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    # Real DataLoader iteration consumes its private base seed before randperm.
    loader = code['validation_training_loader'](
        Indices(keys, labels), batch_size, {'generator': generator},
        'capacity_safe', 1000, 5200, 10000, 0, False)
    batches = list(loader)
    assert sorted(sum(batches, [])) == list(range(len(keys)))
    assert [len(batch) for batch in batches] == [
        min(batch_size, len(keys)-start) for start in range(0,len(keys),batch_size)]
    cap = limits(keys, labels, batches)
    assert cap['edges'] <= 1000 and cap['nodes'] <= 5200
    assert cap['edges'] == loader.batch_sampler.max_batch_edges
    assert cap['nodes'] == loader.batch_sampler.max_batch_nodes
    return batches, generator.get_state(), loader.batch_sampler.retries, cap


def order_sha(keys, batches):
    return hashlib.sha256(json.dumps([keys[index] for index in sum(batches, [])]).encode()).hexdigest()


def validation_plan(keys, labels, batch_size):
    batches = [list(range(start, min(start+batch_size, len(keys))))
               for start in range(0,len(keys),batch_size)]
    cap = limits(keys, labels, batches)
    assert cap['edges'] <= 1000 and cap['nodes'] <= 5200
    assert sum(map(len,batches)) == len(keys)
    return batches, cap


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
        assert ds == 'MELD' and run['seed'] == 67137 and run['epochs'] == 40
        assert run['batch_size'] == run['validation_batch_size'] == 16
        assert ref['args']['batch_size'] == 16 and ref['args']['lr'] == run['lr'] == 1e-4
        assert ref['args']['dropout'] == run['dropout'] == .4
        assert ref['args']['contrastlearning'] is True and run['contrastlearning'] is False
        assert run['calibrate'] is True and run['courselearning'] is True
        assert ref['args']['rec_dropout'] == run['rec_dropout'] == .1
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
        valid_batches, valid_limit = validation_plan(valid, labels, 16)
        old_valid_batches, old_valid_limit = validation_plan(valid, labels, 16)
        assert valid_batches == old_valid_batches and valid_limit == old_valid_limit
        assert len(ref['epochs']) == run['epochs']
        for epoch, old_row in enumerate(ref['epochs']):
            assert old_row['epoch'] == epoch+1
            ratio = old_row['sample_ratio']
            expected_ratio = min(1,round(ref['args']['epoch_ratio'] * int(epoch/ref['args']['scheduler_steps']+1),2))
            assert ratio == expected_ratio
            chosen = set(sorted(train, key=lambda key: difficulty[key])[:int(ratio*len(train))])
            keys = [key for key in train if key in chosen]
            assert len(keys) == old_row['train_dialogues']
            historical16 = replay(old,keys,labels,16,run['seed']+epoch)
            assert historical16[2] == old_row['sampler_retries']
            assert historical16[3]['edges'] == old_row['max_batch_edges']
            assert historical16[3]['nodes'] == old_row['max_batch_nodes']
            a = historical16
            z = replay(new,keys,labels,16,run['seed']+epoch)
            assert a[0] == z[0] and torch.equal(a[1], z[1]) and a[2:] == z[2:]
            records.append({'run_id': run['run_id'], 'epoch': epoch+1,
                            'sample_ratio': ratio, 'train_dialogues': len(keys),
                            'batch_size': 16, 'validation_batch_size': 16,
                            'train_batches': len(z[0]), 'validation_batches': len(valid_batches),
                            'train_last_batch_dialogues': len(z[0][-1]),
                            'validation_last_batch_dialogues': len(valid_batches[-1]),
                            'validation_capacity': valid_limit, 'capacity': z[3],
                            'sampler_retries': z[2], 'private_seed': run['seed']+epoch,
                            'same_as_original_sampler_batch16': True,
                            'ordered_train_ids_sha256': order_sha(keys,z[0]),
                            'historical16': {
                                'batch_size':16, 'train_batches':len(historical16[0]),
                                'validation_batches':len(old_valid_batches),
                                'train_last_batch_dialogues':len(historical16[0][-1]),
                                'validation_last_batch_dialogues':len(old_valid_batches[-1]),
                                'capacity':historical16[3], 'validation_capacity':old_valid_limit,
                                'sampler_retries':historical16[2],
                                'ordered_train_ids_sha256':order_sha(keys,historical16[0]),
                                'replayed_metrics_match':True,
                                'step_count_source':'CPU reconstruction, not a historical step log'},
                            'same_groups_steps_private_rng_as_historical_batch16': True,
                            'eligible_dialogue_ids_sha256':hashlib.sha256(json.dumps(keys).encode()).hexdigest()})
    assert len(records) == sum(r['epochs'] for r in registry['runs']) and torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()
    value = {'status': 'passed', 'epochs_checked': len(records),
             'original_sampling_ast_unchanged': True, 'global_cpu_rng_unchanged': True,
             'cuda_initialized': False, 'test_lengths_or_predictions_examined': False,
             'historical_batch16_replay_passed':True,
             'batch_size':16, 'historical_batch_size':16, 'validation_batch_size':16,
             'expected_train_optimizer_steps_batch16':sum(row['train_batches'] for row in records),
             'historical_train_optimizer_steps_batch16_reconstructed':sum(row['historical16']['train_batches'] for row in records),
             'expected_validation_batches_batch16':sum(row['validation_batches'] for row in records),
             'historical_validation_batches_batch16_reconstructed':sum(row['historical16']['validation_batches'] for row in records),
             'limits': ['CPU graph capacity does not establish GPU memory safety.',
                        'Historical batch16 optimizer step counts are reconstructed, not measured from step logs.',
                        'Matching sampler order/private RNG does not imply identical model RNG trajectory under changed dropout.'],
             'records': records}
    output = ROOT/'preflight/capacity_cpu.json'
    with output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value[key] for key in ['status', 'epochs_checked', 'cuda_initialized']}))


if __name__ == '__main__':
    main()
