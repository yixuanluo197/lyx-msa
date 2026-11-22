"""Restricted MELD base-fit/base-dev M3Net -> difficulty -> CMERC adapter.

Only the frozen parent's Train/Validation size guard is adapted in its training
function. Model equations, losses, optimizer, curriculum, RNG and resume behavior
are imported from the exact frozen parent. No scoring group is loaded here.

CLI: ACTION --protocol PATH [--data-dir RUN/local_only/base_data]
ACTION: audit, smoke, train_m3net, build_difficulty, train_cmerc, evaluate.
evaluate means base-inner-dev only, never official Validation or scoring groups.
Runtime: multimodalerc310; the supervisor selects a healthy visible CUDA device.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import torch

RUN = Path(__file__).resolve().parents[1]
PARENT = RUN.parent / '20261120T143131Z'
SOURCE = PARENT / 'scripts/downstream.py'
SOURCE_SHA256 = '40632d490388f52c4c4ab99d96d1fced047782e3bf6ba782c9048add14bcc249'
DATA_ROOT = RUN / 'local_only/base_data'
EXPECTED_TRAIN_UTTERANCES = 7013
EXPECTED_DIALOGUES = 728
EXPECTED_FIT_DIALOGUES = 656
EXPECTED_DEV_DIALOGUES = 72
PROTOCOL = RUN / 'configs/REAL_SCORING_PROTOCOL.json'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


if sha(SOURCE) != SOURCE_SHA256:
    raise RuntimeError('Frozen parent adapter changed; do not silently adapt another version')
spec = importlib.util.spec_from_file_location('_restricted_meld_frozen_downstream', SOURCE)
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)


def branch_lock_paths():
    return [PROTOCOL, RUN / 'SPLIT_LOCK.json',
            RUN / 'local_only/SPLIT_DIALOGUES.csv', RUN / 'local_only/SPLIT_UTTERANCES.csv']


def validate_protocol(protocol):
    expected_base = dict(fit_dialogues=656, dev_dialogues=72,
                         fit_utterances=6313, dev_utterances=700)
    if (protocol.get('dataset') != 'MELD' or protocol.get('model_seed') != 67137
            or protocol.get('test_access') is not False
            or any(protocol.get('base', {}).get(k) != v for k, v in expected_base.items())):
        raise ValueError('Wrong independently frozen base-only MELD protocol')


def branch_identity():
    return dict(branch='MELD_REAL_SCORING_BASE_ONLY', dataset='MELD',
                fit_role='base_fit', selection_role='base_inner_dev', scoring_groups_loaded=False,
                expected_train_utterances=EXPECTED_TRAIN_UTTERANCES,
                expected_validation_utterances=0,
                expected_train_dialogues=EXPECTED_DIALOGUES,
                expected_fit_dialogues=EXPECTED_FIT_DIALOGUES,
                expected_dev_dialogues=EXPECTED_DEV_DIALOGUES,
                adapter_path=str(Path(__file__).resolve()), adapter_sha256=sha(__file__),
                parent_adapter_path=str(SOURCE), parent_adapter_sha256=SOURCE_SHA256,
                adapted_train_source_sha256=hashlib.sha256(ADAPTED_TRAIN_SOURCE.encode()).hexdigest(),
                split_and_protocol_hashes={str(p.resolve()): sha(p) for p in branch_lock_paths()})


def validate_bundle(bundle, dataset='MELD'):
    """Fail before model construction if scoring/test rows can enter training."""
    if dataset != 'MELD' or bundle.data.get('dataset') != 'MELD':
        raise ValueError('This independent branch is MELD only')
    splits = bundle.data['splits']
    if set(splits) != {'train', 'validation'} or splits['validation']:
        raise ValueError('Restricted bundle must contain Train and empty Validation only')
    train_rows = splits['train']
    ids = [str(d['dialogue_id']) for d in train_rows]
    fit, dev = list(bundle.fit), list(bundle.dev)
    actual = sum(len(d['utterance_ids']) for d in train_rows)
    if actual != EXPECTED_TRAIN_UTTERANCES or len(ids) != EXPECTED_DIALOGUES:
        raise ValueError(f'Expected restricted 7013/0 utterances, 728 Dialogues; got {actual}, {len(ids)}')
    if len(set(ids)) != len(ids) or set(ids) != set(bundle.train):
        raise ValueError('Duplicate or inconsistent restricted Train Dialogue IDs')
    if (len(fit) != EXPECTED_FIT_DIALOGUES or len(dev) != EXPECTED_DEV_DIALOGUES
            or len(set(fit)) != len(fit) or len(set(dev)) != len(dev)
            or set(fit) & set(dev) or set(fit) | set(dev) != set(ids)):
        raise ValueError('base_fit/base_dev must be disjoint 656/72 Dialogue partitions')
    for name, feature in [('text', bundle.text), ('audio', bundle.audio)]:
        if (set(feature) != {'train', 'validation'} or feature['validation']
                or set(map(str, feature['train'])) != set(ids)):
            raise ValueError(f'{name} feature dictionary exposes non-base rows or misses base rows')
    lengths = {}
    all_uids = []
    dimensions = set()
    for d in train_rows:
        did = str(d['dialogue_id'])
        n = len(d['utterance_ids'])
        if n < 1 or len(d['labels']) != n or len(d['texts']) != n:
            raise ValueError('Empty or misaligned restricted Dialogue')
        t, a = bundle.features('train', did)
        if tuple(t.shape) != (n, 4, 1024) or tuple(a.shape) != (n, 768):
            raise ValueError('Restricted text/audio feature shape mismatch')
        if (d['visual'].ndim != 2 or d['speaker'].ndim != 2
                or d['visual'].shape[0] != n or d['speaker'].shape[0] != n):
            raise ValueError('Restricted visual/speaker alignment mismatch')
        if not all(bool(torch.isfinite(value).all()) for value in (t, a, d['visual'], d['speaker'])):
            raise ValueError('Nonfinite restricted feature')
        labels = torch.as_tensor(d['labels'])
        if labels.ndim != 1 or bool(((labels < 0) | (labels >= 7)).any()):
            raise ValueError('Invalid MELD label range')
        if not torch.allclose(d['speaker'].sum(1), torch.ones(n)):
            raise ValueError('Restricted speaker rows must sum to one')
        dimensions.add((d['visual'].shape[1], d['speaker'].shape[1]))
        lengths[did] = n
        all_uids.extend((did, str(uid)) for uid in d['utterance_ids'])
    if len(dimensions) != 1 or len(set(all_uids)) != len(all_uids):
        raise ValueError('Inconsistent dimensions or duplicated restricted utterance IDs')
    return dict(dataset='MELD', train_utterances=actual, validation_utterances=0,
                train_dialogues=len(ids), fit_dialogues=len(fit), inner_dev_dialogues=len(dev),
                fit_utterances=sum(lengths[k] for k in fit),
                inner_dev_utterances=sum(lengths[k] for k in dev),
                fit_dialogue_ids=fit, inner_dev_dialogue_ids=dev,
                scoring_groups_loaded=False, test_accessed=False,
                text_shape_per_utterance=[4, 1024], audio_shape_per_utterance=[768])


def validate_split_lock(bundle):
    """Read ID-only split manifests; never deserialize scoring-group features."""
    path = RUN / 'SPLIT_LOCK.json'
    lock = json.loads(path.read_text())
    if (lock.get('dataset') != 'MELD' or lock.get('no_test_data') is not True
            or lock.get('official_validation_rows_used') != 0):
        raise ValueError('Wrong restricted Train split lock')
    manifests = {}
    for name in ('SPLIT_DIALOGUES.csv', 'SPLIT_UTTERANCES.csv'):
        file = RUN / 'local_only' / name
        if lock['hashes'][name] != sha(file):
            raise ValueError('Split manifest hash differs from original split lock')
        with file.open(newline='', encoding='utf-8') as stream:
            manifests[name] = list(csv.DictReader(stream))
    ds = manifests['SPLIT_DIALOGUES.csv']
    if len({r['dialogue_id'] for r in ds}) != len(ds):
        raise ValueError('Duplicate split Dialogue row')
    roles = {r['dialogue_id']: r['role'] for r in ds}
    expected_fit = {did for did, role in roles.items() if role == 'base_fit'}
    expected_dev = {did for did, role in roles.items() if role == 'base_inner_dev'}
    if set(bundle.fit) != expected_fit or set(bundle.dev) != expected_dev:
        raise ValueError('Dataset Fit/Dev IDs differ from the original frozen group split')
    for row in ds:
        if row['dialogue_id'] in bundle.train:
            if int(row['utterance_count']) != len(bundle.train[row['dialogue_id']]['utterance_ids']):
                raise ValueError('Base Dialogue length differs from the split lock')
    actual = {(did, str(uid), index, roles[did])
              for did, d in bundle.train.items() for index, uid in enumerate(d['utterance_ids'])}
    expected = {(r['dialogue_id'], r['utterance_id'], int(r['utterance_index']), r['role'])
                for r in manifests['SPLIT_UTTERANCES.csv']
                if r['role'] in {'base_fit', 'base_inner_dev'}}
    if actual != expected or len(actual) != EXPECTED_TRAIN_UTTERANCES:
        raise ValueError('Base utterance IDs/order differ from original frozen group split')


class Bundle(parent.Bundle):
    def __init__(self, root):
        if Path(root).resolve() != DATA_ROOT.resolve():
            raise ValueError('Training adapter may open only this branch local_only/base_data')
        super().__init__(root)
        self.audit = validate_bundle(self)
        validate_split_lock(self)

    def inputs(self):
        inputs = {str(Path(p).resolve()): value for p, value in super().inputs().items()}
        inputs.update({str(p.resolve()): sha(p) for p in branch_lock_paths()})
        return inputs


_original_config = parent.config
_original_provenance = parent.provenance
_original_load_model = parent.load_model


def config(dataset, visual_dim, speaker_dim, stage):
    if dataset != 'MELD':
        raise ValueError('This branch cannot change the IEMOCAP protocol')
    result = _original_config(dataset, visual_dim, speaker_dim, stage)
    if result['epochs'] != 40 or result['patience'] != 10 or result['selection_metric'] != 'macro_f1':
        raise ValueError('Frozen MELD epoch/selection hyperparameters changed')
    result.update(branch=branch_identity(),
                  selection_protocol='base_fit_gradient_base_dev_macro_f1_no_refit',
                  class_weight_source='MELD original FocalLoss unchanged')
    return result


def provenance(c, b):
    record = _original_provenance(c, b)
    record['source_plan_b_protocol_sha256'] = record['protocol_sha256']
    record['protocol_sha256'] = sha(PROTOCOL)
    record['branch'] = branch_identity()
    record['restricted_bundle'] = validate_bundle(b)
    record['code'][str(Path(__file__).resolve())] = sha(__file__)
    return record


def load_model(dataset_dir, checkpoint, device='cuda:0'):
    """Expose the new frozen CMERC for later scoring without loading score data."""
    path = Path(checkpoint).resolve()
    if path.parent.name not in {'m3net', 'cmerc'} or path.parent.parent != (DATA_ROOT / 'runs').resolve():
        raise ValueError('Refusing a checkpoint from another branch or old baseline')
    payload = torch.load(path, map_location='cpu')
    identity = payload.get('effective_config', {}).get('branch')
    if identity != branch_identity() or payload.get('new_features_only') is not True:
        raise ValueError('Checkpoint branch/source/split/protocol identity mismatch')
    completion_path = path.parent / 'completion.json'
    if not completion_path.is_file():
        raise ValueError('Best checkpoint presence is insufficient: training must be COMPLETE')
    completion = json.loads(completion_path.read_text())
    if (completion.get('status') != 'COMPLETE'
            or completion.get('protocol_sha256') != sha(PROTOCOL)
            or (path.name == 'best.pt' and completion.get('best_checkpoint_sha256') != sha(path))):
        raise ValueError('Checkpoint completion/hash/protocol mismatch')
    return _original_load_model(dataset_dir, path, device)


# An explicitly bounded source adaptation avoids duplicating the 120-line
# upstream training implementation. The exact parent file hash is checked above.
ORIGINAL_TRAIN_SOURCE = inspect.getsource(parent.train)
_old_guard = """    expected=(5283,527) if dataset=='IEMOCAP' else (9989,1109)
    actual=tuple(sum(len(d['utterance_ids']) for d in b.data['splits'][s]) for s in ('train','validation'))
    if actual!=expected:raise ValueError('effective Train/Validation size mismatch: '+str(actual))
"""
if ORIGINAL_TRAIN_SOURCE.count(_old_guard) != 1:
    raise RuntimeError('Frozen size guard differs; adaptation must be reviewed')
ADAPTED_TRAIN_SOURCE = ORIGINAL_TRAIN_SOURCE.replace(_old_guard, '    _branch_validate_bundle(b,dataset)\n')
_completion = "'adapter_sha256':sha(__file__)"
if ADAPTED_TRAIN_SOURCE.count(_completion) != 1:
    raise RuntimeError('Frozen completion identity differs')
ADAPTED_TRAIN_SOURCE = ADAPTED_TRAIN_SOURCE.replace(_completion, "'adapter_sha256':sha(_branch_entrypoint)")
parent._branch_validate_bundle = validate_bundle
parent._branch_entrypoint = str(Path(__file__).resolve())
parent.config = config
parent.provenance = provenance
parent.load_model = load_model
exec(compile(ADAPTED_TRAIN_SOURCE, str(__file__) + '::parent_train_guard_adaptation', 'exec'), parent.__dict__)
infer_dialogue = parent.infer_dialogue


def train(bundle, stage, device='cuda:0', resume=None, max_epochs=None):
    if stage not in {'m3net', 'cmerc'} or max_epochs not in (None, 40):
        raise ValueError('Preserve the frozen stage menu and 40-epoch MELD maximum')
    return parent.train(bundle, 'MELD', stage, device, resume, max_epochs)


def write_audit(bundle):
    report = dict(bundle=validate_bundle(bundle), identity=branch_identity(), inputs=bundle.inputs(),
                  effective_m3net=config('MELD', next(iter(bundle.train.values()))['visual'].shape[1],
                                         next(iter(bundle.train.values()))['speaker'].shape[1], 'm3net'),
                  effective_cmerc=config('MELD', next(iter(bundle.train.values()))['visual'].shape[1],
                                        next(iter(bundle.train.values()))['speaker'].shape[1], 'cmerc'))
    path = bundle.root / 'branch_input_audit.json'
    if path.exists():
        if json.loads(path.read_text()) != report:
            raise ValueError('Existing branch audit differs; do not replace frozen evidence')
    else:
        parent.dump(path, report)
    return report


def main():
    global PROTOCOL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'smoke', 'train_m3net', 'build_difficulty', 'train_cmerc', 'evaluate'])
    parser.add_argument('--dataset', choices=['MELD'], default='MELD')
    parser.add_argument('--data-dir', type=Path, default=DATA_ROOT)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint')
    parser.add_argument('--max-epochs', type=int)
    args = parser.parse_args()
    PROTOCOL = args.protocol.resolve()
    if PROTOCOL != (RUN / 'configs/REAL_SCORING_PROTOCOL.json').resolve():
        raise ValueError('Explicit protocol must belong to this exact independent branch')
    for path in branch_lock_paths():
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = json.loads(PROTOCOL.read_text())
    validate_protocol(protocol)
    if protocol['split_lock_sha256'] != sha(RUN / 'SPLIT_LOCK.json'):
        raise ValueError('New protocol does not bind the original group split lock')
    if Path(sys.prefix).name != 'multimodalerc310' or torch.__version__ != '2.2.2+cu121':
        raise RuntimeError('Use the existing frozen multimodalerc310 torch2.2.2+cu121 environment')
    torch.set_num_threads(2)
    bundle = Bundle(args.data_dir)
    report = write_audit(bundle)
    if args.action == 'audit':
        print(json.dumps(dict(status='PASS', **report['bundle'])), flush=True)
    elif args.action.startswith('train_'):
        train(bundle, args.action.split('_')[1], args.device, args.resume, args.max_epochs)
    elif args.action == 'build_difficulty':
        # The unchanged parent function iterates bundle.fit only and checks that
        # the new M3Net checkpoint is COMPLETE through the overridden loader.
        parent.build_difficulty(bundle, args.device)
    elif args.action == 'smoke':
        print(json.dumps(parent.smoke(bundle, 'MELD', args.device)), flush=True)
    else:
        checkpoint = args.checkpoint or bundle.root / 'runs/cmerc/best.pt'
        model = load_model(bundle.root, checkpoint, args.device)
        summary, records = parent.evaluate(model, bundle, [bundle.train[k] for k in bundle.dev], 'train')
        summary.update(scope='base_inner_dev_only_not_official_validation', scoring_groups_loaded=False)
        parent.dump(bundle.root / 'base_inner_dev_metrics.json', summary)
        parent.save(bundle.root / 'base_inner_dev_predictions.pt', records)


if __name__ == '__main__':
    main()
