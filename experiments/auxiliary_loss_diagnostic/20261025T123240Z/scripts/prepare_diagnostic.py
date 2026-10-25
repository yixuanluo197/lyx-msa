"""CPU-only checkpoint audit and preregistered training-pool batch selection."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import operator
import os
import pickle
import random
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('text_gru', 'audio_projection', 'visual_projection', 'graph_shared',
          'classification_head', 'other')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def training_metadata(ids, labels):
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate training dialogue IDs')
    result = {}
    for key in ids:
        values = tuple(operator.index(label) for label in labels[key])
        if not values or any(label < 0 or label > 6 for label in values):
            raise ValueError('Emotion labels must be a nonempty sequence in 0..6')
        result[key] = values
    return result


def rank_strata(ids, labels):
    ordered = sorted(ids, key=lambda key: (len(labels[key]), str(key)))
    return [ordered[q * len(ordered) // 4:(q + 1) * len(ordered) // 4]
            for q in range(4)]


def aggregate(ids, labels):
    lengths = [len(labels[key]) for key in ids]
    counts = Counter(label for key in ids for label in labels[key])
    return {'n_dialogues': len(ids), 'n_utterances': sum(lengths),
            'emotion_counts': [counts[label] for label in range(7)],
            'length_min': min(lengths), 'length_max': max(lengths),
            'length_mean': sum(lengths) / len(lengths)}


def select_batches(ids, labels, selection_seed=67137):
    """Greedy class coverage, uniform fill, one shuffle; never capacity-resample."""
    ids = list(ids)
    metadata = training_metadata(ids, labels)
    strata = rank_strata(ids, metadata)
    if any(len(pool) < 128 for pool in strata):
        raise ValueError('Each length stratum must contain at least 128 dialogues')
    rng = random.Random(selection_seed)
    batches = []
    for q, pool in enumerate(strata, 1):
        remaining = list(pool)
        uncovered = {label for key in pool for label in metadata[key]}
        selected = []
        while uncovered:
            gains = [len(set(metadata[key]) & uncovered) for key in remaining]
            maximum = max(gains)
            if maximum == 0:
                raise ValueError('Available class coverage cannot be completed')
            candidates = [key for key, gain in zip(remaining, gains) if gain == maximum]
            chosen = rng.choice(candidates)
            selected.append(chosen)
            remaining.remove(chosen)
            uncovered -= set(metadata[chosen])
        selected.extend(rng.sample(remaining, 128 - len(selected)))
        rng.shuffle(selected)
        for start in range(0, 128, 16):
            chosen = selected[start:start + 16]
            stats = aggregate(chosen, metadata)
            capacity = {'edges': stats['n_utterances'] + 3 * len(chosen),
                        'nodes': 3 * stats['n_utterances']}
            if capacity['edges'] > 1000 or capacity['nodes'] > 5200:
                raise ValueError(f'Fixed batch capacity exceeded in stratum {q}; no resampling allowed')
            batches.append({'batch_id': f'B{len(batches) + 1:02d}',
                            'length_stratum': q, 'dialogue_ids': chosen,
                            **stats, 'capacity': capacity})
    selected = [key for batch in batches for key in batch['dialogue_ids']]
    assert len(batches) == 32 and len(selected) == len(set(selected)) == 512
    assert set(selected) <= set(ids)
    return {'selection_seed': selection_seed, 'batches': batches}


def public_batches(selection):
    return [{**{key: value for key, value in batch.items() if key != 'dialogue_ids'},
             'dialogue_ids_sha256': json_sha(batch['dialogue_ids'])}
            for batch in selection['batches']]


def parameter_group(name):
    if name.startswith('gru_l.'):
        return 'text_gru'
    if name.startswith('linear_a.'):
        return 'audio_projection'
    if name.startswith('linear_v.'):
        return 'visual_projection'
    if name.startswith('graph_model.'):
        return 'graph_shared'
    if name.startswith(('smax_fc.', 'hidfc.')):
        return 'classification_head'
    return 'other'


def construct_cpu_model(code_dir, args):
    """Only the registered MELD constructor; graph .cuda() paths are forwards."""
    import torch
    sys.path.insert(0, str(code_dir))
    from model_cm import Model
    assert args['Dataset'] == 'MELD' and args['causal'] is False
    assert args['base_model'] == 'GRU' and args['av_using_lstm'] is False
    assert args['mm_fusion_mthd'] == 'concat_DHT' and args['graph_type'] == 'hyper'
    with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
        model = Model(args['base_model'], 1024, 1024, 150, 100, 100, 100, 512,
            n_speakers=9, max_seq_len=200, window_past=args['windowp'],
            window_future=args['windowf'], n_classes=7,
            listener_state=args['active_listener'], context_attention=args['attention'],
            dropout=args['dropout'], nodal_attention=args['nodal_attention'], no_cuda=True,
            graph_type=args['graph_type'], use_topic=args['use_topic'], alpha=args['alpha'],
            multiheads=args['multiheads'], graph_construct=args['graph_construct'],
            use_GCN=args['use_gcn'], use_residue=args['use_residue'], D_m_v=342, D_m_a=300,
            modals=args['modals'], att_type=args['mm_fusion_mthd'],
            av_using_lstm=args['av_using_lstm'], Deep_GCN_nlayers=args['Deep_GCN_nlayers'],
            dataset=args['Dataset'], use_speaker=args['use_speaker'], use_modal=args['use_modal'],
            norm=args['norm'], num_L=args['num_L'], num_K=args['num_K'], causal=False)
    assert all(tensor.device.type == 'cpu' for tensor in model.state_dict().values())
    assert not torch.cuda.is_initialized()
    return model


def parameter_inventory(model, code_dir):
    parameters = [{'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
                   'numel': value.numel(), 'requires_grad': value.requires_grad,
                   'group': parameter_group(name)} for name, value in model.named_parameters()]
    groups = {group: [p['name'] for p in parameters if p['group'] == group] for group in GROUPS}
    groups['shared_backbone'] = [p['name'] for p in parameters if p['group'] != 'classification_head']
    groups['all'] = [p['name'] for p in parameters]
    buffers = [{'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
                'numel': value.numel()} for name, value in model.named_buffers()]
    return {'status': 'passed', 'parameters': parameters, 'buffers': buffers, 'groups': groups,
            'group_numel': {group: sum(p['numel'] for p in parameters if p['name'] in names)
                            for group, names in groups.items()},
            'n_parameter_tensors': len(parameters), 'n_parameters': sum(p['numel'] for p in parameters),
            'n_state_keys': len(model.state_dict()), 'state_keys': list(model.state_dict()),
            'source_sha256': {name: sha256(code_dir / name) for name in
                              ['model_cm.py', 'model_hyper.py', 'HypergraphConv.py', 'high_fre_conv.py']},
            'construction': 'Real CPU Model constructor and named_parameters/named_buffers; no forward.',
            'limits': ['Group membership does not imply an active gradient.',
                       'other includes linear_l and normalization parameters, as preregistered.',
                       'shared_backbone excludes only classification_head; all includes every parameter.']}


def finite_cpu_tree(value):
    import torch
    if torch.is_tensor(value):
        assert value.device.type == 'cpu' and bool(torch.isfinite(value).all())
        return value.numel()
    if isinstance(value, dict):
        return sum(finite_cpu_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(finite_cpu_tree(v) for v in value)
    if isinstance(value, float):
        assert math.isfinite(value)
    return 0


def original_train_sha(path):
    text = path.read_bytes().decode().replace('\r\n', '\n')
    for addition in (
        "    parser.add_argument('--skip-final-test', action='store_true',\n"
        "                        help='validation-only diagnosis: do not evaluate the test loader after training')\n",
        "    if args.skip_final_test and (args.selection_protocol != 'validation' or args.testing):\n"
        "        parser.error('--skip-final-test requires validation selection and forbids --testing')\n"):
        assert text.count(addition) == 1
        text = text.replace(addition, '')
    patched = "    if args.selection_protocol == 'validation' and not args.skip_final_test and best_checkpoint_path.exists():\n"
    assert text.count(patched) == 1
    text = text.replace(patched, "    if args.selection_protocol == 'validation' and best_checkpoint_path.exists():\n")
    return hashlib.sha256(text.replace('\n', '\r\n').encode()).hexdigest()


def checkpoint_inventory(registry, manifests, model, code_dir):
    import torch
    observed = []
    expected_ids = {'XP04_best', 'XP04_latest', 'XP12_best', 'XP12_latest'}
    assert {entry['checkpoint_id'] for entry in registry['checkpoints']} == expected_ids
    assert len(registry['checkpoints']) == 4
    for entry in registry['checkpoints']:
        path = Path(entry['path'])
        before = sha256(path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        manifest = manifests[entry['checkpoint_id']]
        assert checkpoint['epoch'] + 1 == entry['expected_epoch']
        assert checkpoint['args'] == manifest['args']
        assert checkpoint['provenance'] == manifest['provenance'] == manifest['args']['provenance']
        args = checkpoint['args']
        assert args['selection_protocol'] == 'validation' and args['seed'] == 67137
        assert args['split_seed'] == 1475 and args['Dataset'] == 'MELD'
        assert args['contrastlearning'] is entry['expected_contrastlearning']
        assert args['calibrate'] is True and args['courselearning'] is True
        assert (args['batch_size'], args['lr'], args['dropout'], args['rec_dropout']) == (16, 1e-4, .4, .1)
        for name, historical_sha in manifest['source']['files'].items():
            actual_sha = sha256(code_dir / name)
            assert actual_sha == historical_sha or (name == 'train_cm.py' and
                original_train_sha(code_dir / name) == historical_sha), name
        model_elements = finite_cpu_tree(checkpoint['model_state_dict'])
        adam_elements = finite_cpu_tree(checkpoint['optimizer_state_dict'])
        assert checkpoint['optimizer_state_dict']['state']
        assert all(group['lr'] == 1e-4 for group in checkpoint['optimizer_state_dict']['param_groups'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        rng = checkpoint['rng_state']
        assert {'python', 'numpy', 'torch', 'cuda'} <= set(rng)
        assert len(rng['python']) == 3 and len(rng['numpy']) == 5
        assert rng['torch'].dtype == torch.uint8 and rng['torch'].numel() > 0
        assert len(rng['cuda']) == 1 and all(t.dtype == torch.uint8 and t.numel() > 0 for t in rng['cuda'])
        after = sha256(path)
        assert before == after
        observed.append({'checkpoint_id': entry['checkpoint_id'], 'history_group': entry['history_group'],
            'path': str(path), 'sha256': before, 'size_bytes': path.stat().st_size,
            'manifest_path': entry['manifest_path'], 'manifest_sha256': sha256(entry['manifest_path']),
            'epoch': checkpoint['epoch'] + 1, 'epoch_zero_based': checkpoint['epoch'],
            'args_match_manifest': True, 'args_sha256': json_sha(args),
            'provenance_match_manifest': True, 'provenance_sha256': json_sha(checkpoint['provenance']),
            'source_compatible': True, 'model_strict_load_passed': True,
            'model_and_adam_finite': True, 'model_state_elements': model_elements,
            'adam_tensor_elements': adam_elements, 'adam_lr': 1e-4,
            'adam_steps': sorted({int(state['step']) for state in checkpoint['optimizer_state_dict']['state'].values()}),
            'rng_keys': sorted(rng), 'torch_rng_bytes': rng['torch'].numel(),
            'cuda_rng_states': len(rng['cuda']),
            'scientific_args': {key: args[key] for key in ['Dataset', 'seed', 'split_seed', 'batch_size', 'lr',
                'dropout', 'rec_dropout', 'calibrate', 'contrastlearning', 'courselearning',
                'rank_coff', 'mscl_coff', 'cscl_coff', 'selection_protocol', 'rng_policy', 'rnn_backend']}})
        del checkpoint
    return {'status': 'passed', 'checkpoints': observed, 'cpu_only': True,
            'cuda_initialized': torch.cuda.is_initialized(), 'model_forwards': 0,
            'test_scores_examined': False, 'historical_artifacts_modified': False}


def coverage_rows(ids, labels, selection):
    chosen = [key for batch in selection['batches'] for key in batch['dialogue_ids']]
    selected = set(chosen)
    rows = []
    for q, pool in [(0, list(ids)), *enumerate(rank_strata(ids, labels), 1)]:
        for scope, keys in [('pool', pool), ('selected', [key for key in pool if key in selected])]:
            data = aggregate(keys, labels)
            counts = data.pop('emotion_counts')
            rows.append({'length_stratum': q, 'scope': scope, **data,
                         'dialogue_coverage_of_stratum': len(keys) / len(pool),
                         'utterance_coverage_of_stratum': data['n_utterances'] / sum(len(labels[k]) for k in pool),
                         **{f'emotion_{label}_utterances': counts[label] for label in range(7)}})
    return rows


def write_exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
    assert not torch.cuda.is_initialized()
    registry = json.loads((ROOT / 'configs/registry.json').read_text())
    assert Path(sys.executable).resolve() == Path(registry['python']).resolve()
    for key, value in registry['environment'].items():
        if key != 'CUDA_VISIBLE_DEVICES':
            assert os.environ.get(key) == value, key
    assert (registry['sample_seed'], registry['batch_size'], registry['n_batches'],
            registry['length_strata'], registry['batches_per_stratum']) == (67137, 16, 32, 4, 8)
    outputs = ['preflight/checkpoint_inventory.json', 'configs/public_sampling_summary.json',
               'private/fixed_batches.json', 'results/sample_coverage.csv', 'provenance/parameter_inventory.json']
    assert all(not (ROOT / name).exists() for name in outputs), 'Preparation never overwrites an artifact'
    code_dir = Path(registry['code_dir'])
    manifests = {entry['checkpoint_id']: json.loads(Path(entry['manifest_path']).read_text())
                 for entry in registry['checkpoints']}
    reference = manifests['XP04_best']
    for manifest in manifests.values():
        assert manifest['data'] == reference['data'] and manifest['split'] == reference['split']
        allowed = {'contrastlearning', 'output_dir', 'skip_final_test', 'provenance'}
        assert {k: v for k, v in manifest['args'].items() if k not in allowed} == {
            k: v for k, v in reference['args'].items() if k not in allowed}
    train_ids = reference['split']['train_ids']
    assert len(train_ids) == 1038 and len(set(train_ids)) == 1038
    assert len(reference['split']['validation_ids']) == 114
    assert not set(train_ids) & set(reference['split']['validation_ids'] + reference['split']['test_ids'])
    raw_path = code_dir / 'data/MELD_features_raw1.pkl'
    raw_sha = sha256(raw_path)
    assert raw_sha == reference['data']['data/MELD_features_raw1.pkl']
    with raw_path.open('rb') as stream:
        raw = pickle.load(stream, encoding='latin1')
    # The pickle is indivisible; only raw[2] values indexed by registered TRAIN IDs are examined.
    labels = training_metadata(train_ids, raw[2])
    del raw
    before = random.getstate()
    selection = select_batches(train_ids, labels, selection_seed=registry['sample_seed'])
    assert before == random.getstate()
    model = construct_cpu_model(code_dir, reference['args'])
    parameters = parameter_inventory(model, code_dir)
    checkpoints = checkpoint_inventory(registry, manifests, model, code_dir)
    rows = coverage_rows(train_ids, labels, selection)
    public = {'status': 'passed', 'selection_seed': registry['sample_seed'],
        'rule': registry['selection_rule'], 'rank_boundary_rule': 'slice[floor(q*N/4):floor((q+1)*N/4)]',
        'n_batches': 32, 'batch_size': 16, 'n_unique_dialogues': 512,
        'training_pool_dialogues': 1038, 'train_split_hash': reference['split']['hash'],
        'raw_training_input_sha256': raw_sha, 'fixed_batches_content_sha256': json_sha(selection),
        'batches': public_batches(selection), 'coverage': rows,
        'validation_and_test_labels_examined': False, 'text_content_examined': False,
        'capacity_resampling_attempts': 0, 'global_python_rng_unchanged_by_sampling': True,
        'limitations': ['Class-covering selection is not an unbiased uniform pool sample.',
                       'The complete feature pickle is deserialized, but only registered training IDs index raw labels.',
                       'No batches are selected or replaced using gradients, predictions, or performance.']}
    assert not torch.cuda.is_initialized()
    for name, value in [('private/fixed_batches.json', selection), ('configs/public_sampling_summary.json', public),
                        ('provenance/parameter_inventory.json', parameters), ('preflight/checkpoint_inventory.json', checkpoints)]:
        write_exclusive(ROOT / name, value)
    output = ROOT / 'results/sample_coverage.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'status': 'passed', 'checkpoints': len(checkpoints['checkpoints']),
        'batches': 32, 'dialogues': 512, 'cuda_initialized': False,
        'artifacts_sha256': {name: sha256(ROOT / name) for name in outputs}}, indent=2))


if __name__ == '__main__':
    main()
