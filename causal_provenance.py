"""Auditable, immutable causal curriculum artifacts.

Pickle/checkpoint inputs remain trusted local files, not an untrusted file format.
Provenance is a consistency contract, not proof about an upstream feature extractor.
"""
import hashlib
import json
import math
import pickle
from pathlib import Path


REPLACEMENT_POLICY = 'prefix_mean_v1'
ARCHITECTURE_FIELDS = (
    'Dataset', 'base_model', 'norm', 'num_L', 'num_K', 'av_using_lstm',
    'use_residue', 'graph_type', 'graph_construct', 'modals', 'mm_fusion_mthd',
    'use_speaker', 'use_modal', 'use_gcn', 'Deep_GCN_nlayers', 'multi_modal',
    'windowp', 'windowf', 'active_listener', 'attention', 'nodal_attention',
    'alpha', 'multiheads', 'causal',
)
TRAINING_PROTOCOL_FIELDS = (
    'selection_protocol', 'seed', 'split_seed', 'dropout', 'rec_dropout',
    'batch_size', 'lr', 'l2', 'class_weight', 'calibrate', 'rank_coff',
    'contrastlearning', 'mscl_coff', 'cscl_coff', 'courselearning',
    'epoch_ratio', 'scheduler_steps', 'epochs', 'auxiliary_policy',
    'replacement_policy', 'difficulty_sha256',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def hash_ids(ids):
    """Hash ordered IDs, including their JSON type (integer differs from string)."""
    return hashlib.sha256(_canonical(list(ids))).hexdigest()


def architecture_from_args(args):
    args = args if isinstance(args, dict) else vars(args)
    return {key: args[key] for key in ARCHITECTURE_FIELDS if key in args}


def dataset_feature_paths(dataset, root='data'):
    names = {'IEMOCAP': ('IEMOCAP_features.pkl', 'iemocap_features_roberta.pkl'),
             'MELD': ('MELD_features_raw1.pkl', 'meld_features_roberta.pkl')}
    if dataset not in names:
        raise ValueError('unsupported dataset: {}'.format(dataset))
    return {name: str(Path(root) / name) for name in names[dataset]}


def build_data_provenance(dataset, train_ids, validation_ids, feature_paths,
                          split_seed=None, selection_protocol='validation'):
    train_ids, validation_ids = list(train_ids), list(validation_ids)
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError('duplicate dialogue IDs in data provenance')
    if set(train_ids) & set(validation_ids):
        raise ValueError('training and validation ID overlap')
    if not train_ids or not validation_ids:
        raise ValueError('data provenance requires nonempty training and validation IDs')
    if split_seed is None:
        raise ValueError('data provenance requires explicit split_seed')
    paths = (feature_paths if isinstance(feature_paths, dict)
             else {Path(path).name: path for path in feature_paths})
    if not paths:
        raise ValueError('data provenance requires feature files')
    return {'schema_version': 1, 'dataset': dataset,
            'selection_protocol': selection_protocol, 'split_seed': int(split_seed),
            'train_ids_sha256': hash_ids(train_ids), 'train_count': len(train_ids),
            'validation_ids_sha256': hash_ids(validation_ids),
            'validation_count': len(validation_ids),
            'feature_files': {name: sha256_file(path) for name, path in sorted(paths.items())}}


def data_provenance_from_files(dataset, split_seed, valid_fraction=.1, root='data',
                               selection_protocol='validation'):
    """Read only split metadata; feature-file digests cover all stored feature bytes."""
    paths = dataset_feature_paths(dataset, root)
    if selection_protocol not in ('validation', 'official_test'):
        raise ValueError('unknown selection protocol')
    if dataset == 'IEMOCAP':
        from dataloader import split_iemocap_dialogues
        with open(paths['IEMOCAP_features.pkl'], 'rb') as handle:
            contents = pickle.load(handle, encoding='latin1')
        train_ids, test_ids = contents[7:9]
        if selection_protocol == 'validation':
            train_ids, validation_ids = split_iemocap_dialogues(train_ids, split_seed, valid_fraction)
        else:
            validation_ids = test_ids  # Explicit legacy selection set; not a safe teacher.
    else:
        from dataloader import resolve_meld_split
        if selection_protocol == 'validation':
            with open(paths['meld_features_roberta.pkl'], 'rb') as handle:
                contents = pickle.load(handle, encoding='latin1')
            train_ids, test_ids, validation_ids = contents[8:11]
            train_ids = resolve_meld_split(train_ids, validation_ids, test_ids, 'train')
        else:
            with open(paths['MELD_features_raw1.pkl'], 'rb') as handle:
                contents = pickle.load(handle, encoding='latin1')
            train_ids, validation_ids = contents[7:9]
    return build_data_provenance(dataset, train_ids, validation_ids, paths, split_seed,
                                 selection_protocol=selection_protocol)


def _validate_data_provenance(data):
    required = ('dataset', 'split_seed', 'train_ids_sha256', 'validation_ids_sha256',
                'train_count', 'validation_count', 'feature_files', 'selection_protocol')
    if not isinstance(data, dict) or any(key not in data for key in required):
        raise ValueError('missing training data provenance fields')
    if data['selection_protocol'] != 'validation':
        raise ValueError('causal curriculum requires validation selection protocol')
    if data['split_seed'] is None or min(data['train_count'], data['validation_count']) < 1:
        raise ValueError('invalid training data provenance split')
    digests = [data['train_ids_sha256'], data['validation_ids_sha256']]
    if not isinstance(data['feature_files'], dict) or not data['feature_files']:
        raise ValueError('missing feature file SHA256 provenance')
    digests.extend(data['feature_files'].values())
    if any(not isinstance(digest, str) or len(digest) != 64 or
           any(c not in '0123456789abcdef' for c in digest) for digest in digests):
        raise ValueError('invalid SHA256 in training data provenance')


def create_difficulty_manifest(checkpoint_path, checkpoint, data_provenance,
                               architecture, replacement_policy=REPLACEMENT_POLICY):
    """Admit a validation-selected causal teacher before any difficulty inference.

    Training IDs of the teacher are checked independently of which difficulty IDs
    are emitted; train-only difficulty values cannot establish teacher provenance.
    """
    _validate_data_provenance(data_provenance)
    args = checkpoint.get('args', {})
    source_data = checkpoint.get('data_provenance', args.get('data_provenance'))
    if source_data is None:
        raise ValueError('source checkpoint lacks training data provenance')
    if source_data != data_provenance:
        raise ValueError('source checkpoint training data provenance does not match requested split/features')
    if args.get('selection_protocol') != 'validation':
        raise ValueError('source checkpoint requires validation selection protocol; official_test is rejected')
    if args.get('causal') is not True or args.get('norm') != 'LN':
        raise ValueError('source checkpoint is not a causal LN architecture')
    if architecture_from_args(args) != architecture:
        raise ValueError('source checkpoint architecture does not match difficulty architecture')
    if args.get('replacement_policy') != replacement_policy:
        raise ValueError('source checkpoint replacement policy mismatch or missing provenance')
    if not isinstance(args.get('seed'), int) or args.get('split_seed') != data_provenance['split_seed']:
        raise ValueError('source checkpoint requires training seed and matching split seed')
    if not isinstance(checkpoint.get('epoch'), int) or checkpoint['epoch'] < 0:
        raise ValueError('source checkpoint has no completed training epoch provenance')
    if not isinstance(checkpoint.get('best_epoch'), int) or not 0 <= checkpoint['best_epoch'] <= checkpoint['epoch']:
        raise ValueError('source checkpoint has no valid best checkpoint selection provenance')
    return {'schema_version': 1, 'causal': True, 'dataset': data_provenance['dataset'],
            'replacement_policy': replacement_policy, 'architecture': architecture,
            'data_provenance': data_provenance,
            'source_checkpoint': {'path': str(Path(checkpoint_path).resolve()),
                'sha256': sha256_file(checkpoint_path),
                'selection_protocol': args['selection_protocol'],
                'training_seed': args['seed'], 'split_seed': args['split_seed'],
                'training_data_provenance': source_data, 'architecture': architecture,
                'training_protocol': {key: args[key] for key in TRAINING_PROTOCOL_FIELDS if key in args},
                'replacement_policy': replacement_policy, 'epoch': checkpoint['epoch'],
                'best_epoch': checkpoint['best_epoch']}}


def _validate_manifest(manifest, dataset, expected_data_provenance,
                       expected_architecture, replacement_policy):
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
        raise ValueError('causal difficulty requires provenance manifest schema_version=1')
    if manifest.get('causal') is not True or manifest.get('dataset') != dataset:
        raise ValueError('causal difficulty manifest dataset/protocol mismatch')
    _validate_data_provenance(expected_data_provenance)
    if manifest.get('data_provenance') != expected_data_provenance:
        raise ValueError('difficulty data provenance does not match requested split/features')
    if manifest.get('replacement_policy') != replacement_policy:
        raise ValueError('difficulty replacement policy mismatch')
    architecture = manifest.get('architecture')
    if not isinstance(architecture, dict) or architecture.get('causal') is not True or architecture.get('norm') != 'LN':
        raise ValueError('difficulty manifest missing causal architecture')
    if expected_architecture is not None and architecture != expected_architecture:
        raise ValueError('difficulty source architecture mismatch')
    source = manifest.get('source_checkpoint', {})
    if source.get('training_data_provenance') != expected_data_provenance:
        raise ValueError('source checkpoint training data provenance mismatch')
    if source.get('selection_protocol') != 'validation':
        raise ValueError('source checkpoint official_test or missing selection protocol is rejected')
    if source.get('split_seed') != expected_data_provenance['split_seed'] or not isinstance(source.get('training_seed'), int):
        raise ValueError('source checkpoint seed provenance mismatch')
    if source.get('architecture') != architecture or source.get('replacement_policy') != replacement_policy:
        raise ValueError('source checkpoint architecture/replacement provenance mismatch')
    training_protocol = source.get('training_protocol', {})
    for key, expected in (('seed', source['training_seed']),
                          ('split_seed', source['split_seed']),
                          ('selection_protocol', 'validation'),
                          ('replacement_policy', replacement_policy)):
        if training_protocol.get(key) != expected:
            raise ValueError('source checkpoint training protocol missing or inconsistent: {}'.format(key))
    digest = source.get('sha256', '')
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('source checkpoint SHA256 provenance missing or invalid')
    source_path = source.get('path')
    if not source_path or not Path(source_path).is_file():
        raise ValueError('source checkpoint file is unavailable for SHA256 validation')
    if sha256_file(source_path) != digest:
        raise ValueError('source checkpoint SHA256 differs from recorded teacher')
    if not isinstance(source.get('epoch'), int) or not isinstance(source.get('best_epoch'), int) or not 0 <= source['best_epoch'] <= source['epoch']:
        raise ValueError('source checkpoint training/selection epoch provenance missing')


def _validate_difficulties(difficulties):
    if not isinstance(difficulties, dict) or not difficulties:
        raise ValueError('difficulty values must be a nonempty dictionary')
    if any(isinstance(value, bool) or not isinstance(value, (float, int)) or
           not math.isfinite(value) or value < 0 for value in difficulties.values()):
        raise ValueError('difficulty values must be finite nonnegative numbers')


def write_causal_difficulty(difficulties, manifest, output_path=None, output_dir='caldiff'):
    _validate_manifest(manifest, manifest.get('dataset'), manifest.get('data_provenance'),
                       manifest.get('architecture'), manifest.get('replacement_policy'))
    _validate_difficulties(difficulties)
    if len(difficulties) != manifest['data_provenance']['train_count'] or hash_ids(difficulties) != manifest['data_provenance']['train_ids_sha256']:
        raise ValueError('difficulty IDs must exactly match the ordered source training split')
    payload = {'causal': True, 'dataset': manifest['dataset'],
               'difficulties': difficulties, 'manifest': manifest}
    contents = pickle.dumps(payload, protocol=4)
    digest = hashlib.sha256(contents).hexdigest()
    path = (Path(output_path) if output_path else Path(output_dir) /
            '{}_causal_diff_{}.pkl'.format(manifest['dataset'], digest))
    sidecar = Path(str(path) + '.manifest.json')
    if path.exists() or sidecar.exists():
        raise FileExistsError('refusing to overwrite immutable difficulty artifact: {}'.format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(contents)
    with sidecar.open('x') as handle:
        json.dump({'schema_version': 1, 'sha256': digest, 'manifest': manifest}, handle,
                  indent=2, sort_keys=True)
        handle.write('\n')
    return str(path)


def resolve_causal_difficulty(path, dataset, expected_data_provenance,
                              expected_architecture=None, replacement_policy=REPLACEMENT_POLICY,
                              expected_sha256=None, expected_teacher_seed=None):
    if not path:
        raise ValueError('causal curriculum requires an explicit difficulty path')
    path = Path(path).resolve()
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('difficulty SHA256 differs from pinned checkpoint/run manifest')
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or 'manifest' not in payload:
        raise ValueError('causal difficulty requires provenance manifest')
    sidecar_path = Path(str(path) + '.manifest.json')
    if not sidecar_path.is_file():
        raise ValueError('causal difficulty requires immutable SHA256 sidecar manifest')
    with sidecar_path.open() as handle:
        sidecar = json.load(handle)
    if sidecar.get('sha256') != digest:
        raise ValueError('difficulty SHA256 differs from immutable sidecar manifest')
    manifest = payload['manifest']
    if sidecar.get('manifest') != manifest:
        raise ValueError('embedded/sidecar difficulty manifests disagree')
    _validate_manifest(manifest, dataset, expected_data_provenance,
                       expected_architecture, replacement_policy)
    if expected_teacher_seed is not None and manifest['source_checkpoint']['training_seed'] != expected_teacher_seed:
        raise ValueError('source checkpoint training seed differs from expected teacher seed')
    if payload.get('causal') is not True or payload.get('dataset') != dataset:
        raise ValueError('causal difficulty payload protocol mismatch')
    difficulties = payload.get('difficulties')
    _validate_difficulties(difficulties)
    if len(difficulties) != expected_data_provenance['train_count'] or hash_ids(difficulties) != expected_data_provenance['train_ids_sha256']:
        raise ValueError('difficulty IDs do not match source training split')
    return difficulties, {'path': str(path), 'sha256': digest, 'manifest': manifest}
