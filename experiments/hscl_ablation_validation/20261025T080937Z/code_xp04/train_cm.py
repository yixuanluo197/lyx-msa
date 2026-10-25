import os
import json
import hashlib
import fcntl
import inspect
from contextlib import contextmanager
from functools import wraps
import numpy as np, argparse, time, pickle, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from dataloader import (IEMOCAPDataset, MELDDataset, split_iemocap_dialogues,
                        causal_mean_replacement, validate_checkpoint_protocol)
from model_cm import MaskedNLLLoss, LSTMModel, GRUModel, Model, MaskedMSELoss, FocalLoss, Model_Inter, Model_Intra
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
import pandas as pd
import pickle as pk
import datetime
import ipdb
from clloss import CKSCL_context, CKSCL_tav
from tqdm import tqdm
CKSCL_m = CKSCL_tav()
CKSCL_c = CKSCL_context()


def _rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def preserve_rng_for_evaluation(enabled):
    state = _rng_state() if enabled else None
    try:
        yield
    finally:
        if enabled:
            _restore_rng_state(state)


def isolated_evaluation(function):
    signature = inspect.signature(function)
    @wraps(function)
    def call(*positional, **keywords):
        train = signature.bind_partial(*positional, **keywords).arguments.get('train', False)
        continuous = getattr(globals().get('args'), 'rng_policy', 'legacy') == 'continuous'
        with preserve_rng_for_evaluation(continuous and not train):
            return function(*positional, **keywords)
    return call


def configure_recurrent_backend(model, backend='cudnn'):
    """Use native GRU/LSTM dropout with serializable RNG in continuous runs.

    cuDNN's per-process RNN dropout cache is absent from state_dict and CUDA
    generator state. Only recurrent forwards use the native backend; batch
    normalization, graph operations and the model equations are unchanged.
    """
    if backend not in ('aten', 'cudnn'):
        raise ValueError('rnn_backend must be aten or cudnn')
    if backend == 'cudnn':
        return model

    def native_forward(forward):
        @wraps(forward)
        def call(*inputs, **keywords):
            with torch.backends.cudnn.flags(enabled=False):
                return forward(*inputs, **keywords)
        return call

    for recurrent in model.modules():
        if isinstance(recurrent, nn.RNNBase) and not getattr(recurrent, '_native_rng_forward', False):
            recurrent.forward = native_forward(recurrent.forward)
            recurrent._native_rng_forward = True
    return model


def training_run_directory(args):
    path = Path(args.output_dir) / '{}_{}_seed{}'.format(args.Dataset, args.selection_protocol, args.seed)
    return path.with_name(path.name + '_causal') if args.causal else path


def prepare_run_directory(run_dir, resume=None):
    """Reserve a new run exclusively, or lock and verify an exact epoch resume."""
    run_dir = Path(run_dir)
    if resume is None:
        run_dir.mkdir(parents=True, exist_ok=False)
    elif Path(resume).resolve().parent != run_dir.resolve():
        raise ValueError('resume checkpoint must belong to the same run directory')
    lock = (run_dir / '.training.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            checkpoint = torch.load(resume, map_location='cpu')
            with (run_dir / 'metrics.jsonl').open() as handle:
                epochs = [json.loads(line)['epoch'] for line in handle if line.strip()]
            if epochs != list(range(1, checkpoint['epoch'] + 2)):
                raise ValueError('resume checkpoint epoch and contiguous metrics epochs do not match')
            if (run_dir / 'final_test.json').exists():
                raise ValueError('completed run already has final_test.json; create a new experiment')
    except Exception:
        lock.close()
        raise
    return lock


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def training_provenance(args, loaders):
    split = dict(zip(('train_ids', 'validation_ids', 'test_ids'),
                     [list(loader.dataset.keys) for loader in loaders]))
    split['hash'] = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    origin = Path('source_origin.json')
    source = {'base_commit': json.loads(origin.read_text()).get('base_commit') if origin.exists() else None,
              'files': {str(path): file_sha256(path) for path in sorted(Path('.').glob('*.py'))}}
    filenames = (['data/IEMOCAP_features.pkl', 'data/iemocap_features_roberta.pkl']
                 if args.Dataset == 'IEMOCAP' else ['data/MELD_features_raw1.pkl', 'data/meld_features_roberta.pkl'])
    if args.courselearning:
        filenames.append('caldiff/{}{}_diff.pkl'.format(args.Dataset, '_causal' if args.causal else ''))
    return {'split': split, 'source': source, 'data': {name: file_sha256(name) for name in filenames}}


def write_run_manifest(run_dir, args):
    manifest = {'args': vars(args).copy(), **args.provenance, 'provenance': args.provenance}
    path = Path(run_dir) / 'run_manifest.json'
    if args.resume:
        saved = json.loads(path.read_text())
        if saved['provenance'] != args.provenance:
            raise ValueError('resume provenance does not match run manifest')
        return
    with path.open('x') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)


def save_training_checkpoint(path, model, optimizer, epoch, args, best_metric, best_epoch,
                             sampling_metadata=None):
    """Atomically save all state required to resume at the next epoch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'args': dict(args) if isinstance(args, dict) else vars(args).copy(),
        'causal': args.get('causal', False) if isinstance(args, dict) else getattr(args, 'causal', False),
        'rng_state': _rng_state(),
        'best_metric': best_metric,
        'best_epoch': best_epoch,
        'provenance': args.get('provenance') if isinstance(args, dict) else getattr(args, 'provenance', None),
        'sampling_metadata': sampling_metadata,
    }
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def load_training_checkpoint(path, model, optimizer, map_location=None, expected_args=None):
    checkpoint = torch.load(path, map_location=map_location)
    causal = getattr(model, 'causal', False)
    if expected_args is not None:
        causal = expected_args.get('causal', False) if isinstance(expected_args, dict) else expected_args.causal
    validate_checkpoint_protocol(checkpoint, causal=causal, expected_args=expected_args)
    if expected_args is not None:
        requested = expected_args if isinstance(expected_args, dict) else vars(expected_args)
        saved = checkpoint['args']
        # Epoch budget may be extended; data split, stochastic training, loss
        # and curriculum settings must still describe the resumed experiment.
        protocol_keys = (
            'seed', 'split_seed', 'rng_policy', 'rnn_backend', 'dropout', 'rec_dropout', 'batch_size', 'lr', 'l2',
            'calibrate', 'rank_coff', 'contrastlearning', 'mscl_coff', 'cscl_coff',
            'courselearning', 'epoch_ratio', 'scheduler_steps', 'provenance',
            'num_workers', 'class_weight', 'graph_construct', 'windowp', 'windowf',
            'nodal_attention', 'active_listener', 'attention', 'alpha', 'multiheads', 'use_topic',
            'batch_policy', 'edge_capacity', 'node_capacity', 'batch_max_retries',
        )
        for key in protocol_keys:
            if key in requested and (key not in saved or saved[key] != requested[key]):
                raise ValueError('resume {} does not match the saved training protocol'.format(key))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    parameter_device = next(model.parameters()).device
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(parameter_device)
    _restore_rng_state(checkpoint['rng_state'])
    return {
        'start_epoch': checkpoint['epoch'] + 1,
        'best_metric': checkpoint.get('best_metric'),
        'best_epoch': checkpoint.get('best_epoch'),
        'args': checkpoint.get('args', {}),
    }


def write_jsonl_record(path, record):
    def json_safe(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        json.dump(json_safe(record), handle, ensure_ascii=False, sort_keys=True,
                  allow_nan=False)
        handle.write('\n')


def combine_training_losses(classification_loss, cc_loss=0.0, mscl_loss=0.0,
                            cscl_loss=0.0, calibrate=False,
                            contrastlearning=False, rank_coff=0.0,
                            mscl_coff=0.0, cscl_coff=0.0):
    """Apply the official CMERC coefficients and expose raw loss components."""
    def as_tensor(value):
        if torch.is_tensor(value):
            return value
        return classification_loss.new_tensor(value)

    cc_loss = as_tensor(cc_loss)
    mscl_loss = as_tensor(mscl_loss)
    cscl_loss = as_tensor(cscl_loss)
    total_loss = classification_loss
    if calibrate:
        total_loss = total_loss + rank_coff * cc_loss
    if contrastlearning:
        total_loss = total_loss + mscl_coff * mscl_loss + cscl_coff * cscl_loss
    components = {
        'classification_loss': classification_loss.detach().item(),
        'cc_loss': cc_loss.detach().item(),
        'mscl_loss': mscl_loss.detach().item(),
        'cscl_loss': cscl_loss.detach().item(),
        'total_loss': total_loss.detach().item(),
    }
    return total_loss, components


def choose_selection_metrics(protocol, valid_loss, valid_f1, test_loss, test_f1):
    if protocol == 'official_test':
        return 'test', test_loss, test_f1
    if protocol == 'validation':
        return 'validation', valid_loss, valid_f1
    raise ValueError('selection protocol must be official_test or validation')


def create_class_weight_SCL(label):
    unique = list(set(label.cpu().detach().numpy().tolist()))

    # one = sum(label)
    labels_dict = {l:(label==l).sum().item() for l in unique}
    # labels_dict = {0 : len(label) - one, 1: one}
    total = sum(list(labels_dict.values()))
    weights = []
    for i in range(max(unique)+1):
        if i not in unique:
            weights.append(0)
        else:
            weights.append(total/labels_dict[i])
    return weights

# seed = 1475 # We use seed = 1475 on IEMOCAP and seed = 67137 on MELD
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
        np.random.seed(int(args.seed)+worker_id)


def seed_loader_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def loader_randomness(seed, epoch, rng_policy):
    generator = torch.Generator()
    generator.manual_seed((0 if seed is None else seed) + (epoch if rng_policy == 'continuous' else 0))
    options = {'generator': generator}
    if rng_policy == 'continuous':
        options['worker_init_fn'] = seed_loader_worker
    return options


class CapacitySafeBatchSampler(Sampler):
    """Reject whole permutations that exceed fixed official graph parameters.

    Keep batch size and every eligible dialogue unchanged. The first candidate
    is the ordinary RandomSampler permutation, using the loader's private RNG
    after its worker/base-seed draw. Rejection never advances model/global RNG.
    """
    def __init__(self, lengths, batch_size, generator, edge_capacity=1000,
                 node_capacity=5200, max_retries=10000):
        self.lengths = [int(length) for length in lengths]
        self.batch_size = batch_size
        self.generator = generator
        self.edge_capacity = edge_capacity
        self.node_capacity = node_capacity
        self.max_retries = max_retries
        self.retries = self.max_batch_edges = self.max_batch_nodes = None
        if batch_size <= 0 or min(edge_capacity, node_capacity) <= 0 or max_retries < 0:
            raise ValueError('invalid batch capacity configuration')
        if any(length <= 0 for length in self.lengths):
            raise ValueError('dialogue lengths must be positive')
        if not self.lengths:
            return
        count = len(self.lengths)
        batches = len(self)
        smallest_batch = count % batch_size or min(count, batch_size)
        ordered = sorted(self.lengths)
        minimal_longest_batch = ordered[-1] + sum(ordered[:smallest_batch - 1])
        if (sum(ordered) + 3 * count > edge_capacity * batches
                or 3 * sum(ordered) > node_capacity * batches
                or minimal_longest_batch + 3 * smallest_batch > edge_capacity
                or 3 * minimal_longest_batch > node_capacity):
            raise ValueError('batch capacity impossible for the fixed batch size and dialogue lengths')

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for attempt in range(self.max_retries + 1):
            indices = torch.randperm(len(self.lengths), generator=self.generator).tolist()
            batches = [indices[start:start + self.batch_size]
                       for start in range(0, len(indices), self.batch_size)]
            totals = [sum(self.lengths[index] for index in batch) for batch in batches]
            edges = [total + 3 * len(batch) for total, batch in zip(totals, batches)]
            nodes = [3 * total for total in totals]
            if max(edges, default=0) <= self.edge_capacity and max(nodes, default=0) <= self.node_capacity:
                self.retries = attempt
                self.max_batch_edges = max(edges, default=0)
                self.max_batch_nodes = max(nodes, default=0)
                yield from batches
                return
        raise RuntimeError('batch capacity rejection exceeded {} retries; no batches were emitted'.format(self.max_retries))


def validation_training_loader(trainset, batch_size, randomness, batch_policy,
                               edge_capacity, node_capacity, batch_max_retries,
                               num_workers, pin_memory):
    if batch_policy == 'legacy':
        batching = {'batch_size': batch_size, 'shuffle': True}
    elif batch_policy == 'capacity_safe':
        lengths = [len(trainset.videoLabels[key]) for key in trainset.keys]
        batching = {'batch_sampler': CapacitySafeBatchSampler(
            lengths, batch_size, randomness['generator'], edge_capacity,
            node_capacity, batch_max_retries)}
    else:
        raise ValueError('batch_policy must be legacy or capacity_safe')
    return DataLoader(trainset, **batching, **randomness,
                      collate_fn=trainset.collate_fn, num_workers=num_workers,
                      pin_memory=pin_memory)

def get_train_valid_sampler(trainset, valid=0.1, dataset='IEMOCAP'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False,
                     epoch_ratio=-1, selection_protocol='official_test', seed=None, causal=False,
                     split_seed=None, epoch=0, rng_policy='legacy', batch_policy='legacy',
                     edge_capacity=1000, node_capacity=5200, batch_max_retries=10000):
    dataset_kwargs = {'causal': True} if causal else {}
    if batch_policy != 'legacy' and (selection_protocol != 'validation' or causal or rng_policy != 'continuous'):
        raise ValueError('capacity_safe batching requires noncausal continuous validation loaders')
    evaluation_randomness = loader_randomness(seed, epoch, rng_policy) if rng_policy == 'continuous' else {}
    if selection_protocol == 'validation':
        trainset = MELDDataset('data/MELD_features_raw1.pkl', split='train', epoch_ratio=epoch_ratio, **dataset_kwargs)
        validset = MELDDataset('data/MELD_features_raw1.pkl', split='validation', **dataset_kwargs)
        testset = MELDDataset('data/MELD_features_raw1.pkl', split='test', **dataset_kwargs)
        randomness = loader_randomness(seed, epoch, rng_policy)
        train_loader = validation_training_loader(trainset, batch_size, randomness,
                                                   batch_policy, edge_capacity, node_capacity,
                                                   batch_max_retries, num_workers, pin_memory)
        valid_loader = DataLoader(validset,
                                  **evaluation_randomness,
                                  batch_size=batch_size,
                                  collate_fn=validset.collate_fn,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
        test_loader = DataLoader(testset,
                                 **loader_randomness(seed, epoch, rng_policy) if rng_policy == 'continuous' else {},
                                 batch_size=batch_size,
                                 collate_fn=testset.collate_fn,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)
        return train_loader, valid_loader, test_loader
    if selection_protocol != 'official_test':
        raise ValueError('selection_protocol must be official_test or validation')

    trainset = MELDDataset('data/MELD_features_raw1.pkl', epoch_ratio=epoch_ratio, **dataset_kwargs)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    randomness = loader_randomness(seed, epoch, rng_policy) if rng_policy == 'continuous' else {}
    if randomness:
        train_sampler.generator = randomness['generator']
        valid_sampler.generator = evaluation_randomness['generator']

    train_loader = DataLoader(trainset,
                              **randomness,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              **evaluation_randomness,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/MELD_features_raw1.pkl', train=False, **dataset_kwargs)
    test_loader = DataLoader(testset,
                             **loader_randomness(seed, epoch, rng_policy) if rng_policy == 'continuous' else {},
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False,
                        epoch_ratio=-1, selection_protocol='official_test', seed=None, causal=False,
                        split_seed=None, epoch=0, rng_policy='legacy', batch_policy='legacy',
                        edge_capacity=1000, node_capacity=5200, batch_max_retries=10000):
    dataset_kwargs = {'causal': True} if causal else {}
    if batch_policy != 'legacy' and (selection_protocol != 'validation' or causal or rng_policy != 'continuous'):
        raise ValueError('capacity_safe batching requires noncausal continuous validation loaders')
    evaluation_randomness = loader_randomness(seed, epoch, rng_policy) if rng_policy == 'continuous' else {}
    if selection_protocol == 'validation':
        full_trainset = IEMOCAPDataset(**dataset_kwargs)
        train_keys, validation_keys = split_iemocap_dialogues(
            full_trainset.keys, seed=(0 if seed is None else seed) if split_seed is None else split_seed, valid_fraction=valid
        )
        trainset = IEMOCAPDataset(epoch_ratio=epoch_ratio, keys=train_keys, **dataset_kwargs)
        validset = IEMOCAPDataset(keys=validation_keys, **dataset_kwargs)
        testset = IEMOCAPDataset(train=False, **dataset_kwargs)
        randomness = loader_randomness(seed, epoch, rng_policy)
        if rng_policy == 'legacy':
            randomness['worker_init_fn'] = _init_fn
        train_loader = validation_training_loader(trainset, batch_size, randomness,
                                                   batch_policy, edge_capacity, node_capacity,
                                                   batch_max_retries, num_workers, pin_memory)
        valid_loader = DataLoader(validset,
                                  **evaluation_randomness,
                                  batch_size=batch_size,
                                  collate_fn=validset.collate_fn,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
        test_loader = DataLoader(testset,
                                 generator=loader_randomness(seed, epoch, rng_policy)['generator'] if rng_policy == 'continuous' else None,
                                 batch_size=batch_size,
                                 collate_fn=testset.collate_fn,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 worker_init_fn=seed_loader_worker if rng_policy == 'continuous' else _init_fn)
        return train_loader, valid_loader, test_loader
    if selection_protocol != 'official_test':
        raise ValueError('selection_protocol must be official_test or validation')

    trainset = IEMOCAPDataset(epoch_ratio=epoch_ratio, **dataset_kwargs)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    randomness = loader_randomness(seed, epoch, rng_policy) if rng_policy == 'continuous' else {'worker_init_fn': _init_fn}
    if rng_policy == 'continuous':
        valid_sampler.generator = evaluation_randomness['generator']

    train_loader = DataLoader(trainset,
                              **randomness,
                              shuffle=rng_policy == 'continuous',
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              **evaluation_randomness,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False, **dataset_kwargs)
    test_loader = DataLoader(testset,
                             generator=loader_randomness(seed, epoch, rng_policy)['generator'] if rng_policy == 'continuous' else None,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory, worker_init_fn=seed_loader_worker if rng_policy == 'continuous' else _init_fn)

    return train_loader, valid_loader, test_loader


@isolated_evaluation
def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    """
    
    """
    losses, preds, labels, masks = [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    if getattr(args, 'rng_policy', 'legacy') == 'legacy':
        seed_everything(args.seed)
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]        

        max_sequence_len.append(textf.size(0))
        
        log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask)
        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1)
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)
    
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]


@isolated_evaluation
def train_or_eval_graph_model(model, model_mm, model_intra, model_inter, loss_function, dataloader, epoch, cuda, modals, optimizer=None, train=False, dataset='IEMOCAP', return_components=False):
    losses, preds, labels = [], [], []
    loss_components = []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    model_mm.load_state_dict(model.state_dict())
    model_mm.eval()
    model_intra.load_state_dict(model.state_dict())
    model_intra.eval()
    model_inter.load_state_dict(model.state_dict())
    model_inter.eval()

    if getattr(args, 'rng_policy', 'legacy') == 'legacy':
        seed_everything(args.seed)
    for data in tqdm(dataloader):
        if train:
            optimizer.zero_grad()
        
        textf1,textf2,textf3,textf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        if args.multi_modal:
            if args.mm_fusion_mthd=='concat':
                if modals == 'avl':
                    textf = torch.cat([acouf, visuf, textf1,textf2,textf3,textf4],dim=-1)
                elif modals == 'av':
                    textf = torch.cat([acouf, visuf],dim=-1)
                elif modals == 'vl':
                    textf = torch.cat([visuf, textf1,textf2,textf3,textf4],dim=-1)
                elif modals == 'al':
                    textf = torch.cat([acouf, textf1,textf2,textf3,textf4],dim=-1)
                else:
                    raise NotImplementedError
            elif args.mm_fusion_mthd=='gated':
                textf = textf
        else:
            if modals == 'a':
                textf = acouf
            elif modals == 'v':
                textf = visuf
            elif modals == 'l':
                textf = textf
            else:
                raise NotImplementedError
        # 1. Cross Entropy Loss
        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
        if args.multi_modal and args.mm_fusion_mthd=='gated':
            log_prob, e_i, e_n, e_t, e_l, hidden = model(textf, qmask, umask, lengths, acouf, visuf)
        elif args.multi_modal and args.mm_fusion_mthd=='concat_subsequently':   
            log_prob, e_i, e_n, e_t, e_l, hidden = model([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
        elif args.multi_modal and args.mm_fusion_mthd=='concat_DHT':   
            log_prob, e_i, e_n, e_t, e_l, hidden = model([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
        else:
            log_prob, e_i, e_n, e_t, e_l, hidden = model(textf, qmask, umask, lengths)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        if args.contrastlearning or args.calibrate:
            with torch.no_grad():
                ft_scl = hidden
                mask_features = lambda feature: causal_mean_replacement(feature, umask) if getattr(args, 'causal', False) else feature.mean(dim=0).repeat(feature.shape[0], 1, 1)
                mask_textf1 = mask_features(textf1)
                mask_textf2 = mask_features(textf2)
                mask_textf3 = mask_features(textf3)
                mask_textf4 = mask_features(textf4)

                mask_acouf = mask_features(acouf)
                mask_visuf = mask_features(visuf)
                
                # mask multimodal data
                log_prob2t, e_i, e_n, e_t, e_l, hiddent = model_mm([mask_textf1, mask_textf2, mask_textf3, mask_textf4], qmask, umask, lengths, acouf, visuf, epoch)
                log_prob2a, e_i, e_n, e_t, e_l, hiddena = model_mm([textf1,textf2,textf3,textf4], qmask, umask, lengths, mask_acouf, visuf, epoch)
                log_prob2v, e_i, e_n, e_t, e_l, hiddenv = model_mm([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, mask_visuf, epoch)

                # mask inter context and intra contex
                log_prob2intra, e_i, e_n, e_t, e_l, hidden_intra = model_intra([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
                log_prob2inter, e_i, e_n, e_t, e_l, hidden_inter = model_inter([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)


                labels_ = label

                pred2t_ = torch.argmax(log_prob2t, 1) # batch*seq_len
                pred2a_ = torch.argmax(log_prob2a, 1) # batch*seq_len
                pred2v_ = torch.argmax(log_prob2v, 1) # batch*seq_len
                pred2c_intra = torch.argmax(log_prob2intra, 1)
                pred2c_inter = torch.argmax(log_prob2inter, 1)

            # 2. rank loss, indicate cf
            lp_ = log_prob
            pred_ = torch.argmax(lp_,1)
            rank_loss = 0
            for i in range(len(log_prob)):
                num = labels_[i]
                if lp_[i][num] <= log_prob2t[i][num]:
                    rank_loss += (torch.sub(log_prob2t[i][num], lp_[i][num]))
                if lp_[i][num] <= log_prob2a[i][num]:
                    rank_loss += (torch.sub(log_prob2a[i][num], lp_[i][num]))
                if lp_[i][num] <= log_prob2v[i][num]:
                    rank_loss += (torch.sub(log_prob2v[i][num], lp_[i][num]))

            rank_lossc = 0 
            for i in range(len(log_prob)):
                num = labels_[i]
                if lp_[i][num] <= log_prob2intra[i][num]:
                    rank_lossc += (torch.sub(log_prob2intra[i][num], lp_[i][num]))
                if lp_[i][num] <= log_prob2inter[i][num]:
                    rank_lossc += (torch.sub(log_prob2inter[i][num], lp_[i][num]))

            rank_loss = rank_loss + rank_lossc
            # 2. for SCL label ##########################################################
            # 0: confidence up 
            # 1: confidence drop
            Mscl_pred_t = torch.zeros_like(pred_)
            Mscl_pred_v = torch.zeros_like(pred_)
            Mscl_pred_a = torch.zeros_like(pred_)
            for index, x in enumerate(Mscl_pred_t):
                if pred_[index] != labels_[index] and pred2t_[index] == labels_[index]:
                    Mscl_pred_t[index] = 0
                if pred_[index] == labels_[index] and pred2t_[index] != labels_[index]:
                    Mscl_pred_t[index] = 1
                if pred_[index] == labels_[index] and pred2t_[index] == labels_[index]:
                    if lp_[index][labels_[index]] <= log_prob2t[index][labels_[index]]:
                        Mscl_pred_t[index] = 0
                    if lp_[index][labels_[index]] > log_prob2t[index][labels_[index]]:
                        Mscl_pred_t[index] = 1

                if pred_[index] != labels_[index] and pred2a_[index] == labels_[index]:
                    Mscl_pred_a[index] = 0
                if pred_[index] == labels_[index] and pred2a_[index] != labels_[index]:
                    Mscl_pred_a[index] = 1
                if pred_[index] == labels_[index] and pred2a_[index] == labels_[index]:
                    if lp_[index][labels_[index]] <= log_prob2a[index][labels_[index]]:
                        Mscl_pred_a[index] = 0
                    if lp_[index][labels_[index]] > log_prob2a[index][labels_[index]]:
                        Mscl_pred_a[index] = 1

                if pred_[index] != labels_[index] and pred2v_[index] == labels_[index]:
                    Mscl_pred_v[index] = 0
                if pred_[index] == labels_[index] and pred2v_[index] != labels_[index]:
                    Mscl_pred_v[index] = 1
                if pred_[index] == labels_[index] and pred2v_[index] == labels_[index]:
                    if lp_[index][labels_[index]] <= log_prob2v[index][labels_[index]]:
                        Mscl_pred_v[index] = 0
                    if lp_[index][labels_[index]] > log_prob2v[index][labels_[index]]:
                        Mscl_pred_v[index] = 1

            Cscl_pred_intra = torch.zeros_like(pred_)
            Cscl_pred_inter = torch.zeros_like(pred_)
            for index, x in enumerate(Cscl_pred_intra):
                if pred_[index] != labels_[index] and pred2c_intra[index] == labels_[index]:
                    Cscl_pred_intra[index] = 0
                if pred_[index] == labels_[index] and pred2c_intra[index] != labels_[index]:
                    Cscl_pred_intra[index] = 1
                if pred_[index] == labels_[index] and pred2c_intra[index] == labels_[index]:
                    if lp_[index][labels_[index]] <= log_prob2intra[index][labels_[index]]:
                        Cscl_pred_intra[index] = 0
                    if lp_[index][labels_[index]] > log_prob2intra[index][labels_[index]]:
                        Cscl_pred_intra[index] = 1

                if pred_[index] != labels_[index] and pred2c_inter[index] == labels_[index]:
                    Cscl_pred_inter[index] = 0
                if pred_[index] == labels_[index] and pred2c_inter[index] != labels_[index]:
                    Cscl_pred_inter[index] = 1
                if pred_[index] == labels_[index] and pred2c_inter[index] == labels_[index]:
                    if lp_[index][labels_[index]] <= log_prob2inter[index][labels_[index]]:
                        Cscl_pred_inter[index] = 0
                    if lp_[index][labels_[index]] > log_prob2inter[index][labels_[index]]:
                        Cscl_pred_inter[index] = 1
            

            # 2. HSCL
            Mscl_tav = CKSCL_m(ft_scl, labels1=Mscl_pred_t, labels2=Mscl_pred_a, labels3=Mscl_pred_v, weight1=create_class_weight_SCL(Mscl_pred_t), weight2=create_class_weight_SCL(Mscl_pred_a), weight3=create_class_weight_SCL(Mscl_pred_v))
            Cscl_c = CKSCL_c(ft_scl, labels1=Cscl_pred_intra, labels2=Cscl_pred_inter, weight1=create_class_weight_SCL(Cscl_pred_intra), weight2=create_class_weight_SCL(Cscl_pred_inter))

        classification_loss = loss_function(log_prob, label)
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(classification_loss.item())
        total_loss, components = combine_training_losses(
            classification_loss,
            cc_loss=rank_loss if (args.contrastlearning or args.calibrate) else 0.0,
            mscl_loss=Mscl_tav if (args.contrastlearning or args.calibrate) else 0.0,
            cscl_loss=Cscl_c if (args.contrastlearning or args.calibrate) else 0.0,
            calibrate=args.calibrate,
            contrastlearning=args.contrastlearning,
            rank_coff=args.rank_coff,
            mscl_coff=args.mscl_coff,
            cscl_coff=args.cscl_coff,
        )
        loss_components.append(components)
        if train:
            total_loss.backward()
            optimizer.step()
            

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        empty_result = (float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], [])
        if return_components:
            return empty_result + ({
                'classification_loss': float('nan'),
                'cc_loss': float('nan'),
                'mscl_loss': float('nan'),
                'cscl_loss': float('nan'),
                'total_loss': float('nan'),
            },)
        return empty_result

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_fscore = round(f1_score(labels,preds, average='weighted')*100, 2)

    result = (avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el)
    if return_components:
        averaged_components = {
            key: float(np.mean([component[key] for component in loss_components]))
            for key in loss_components[0]
        }
        return result + (averaged_components,)
    return result


if __name__ == '__main__':
    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--causal', action='store_true', help='use only history and current utterances')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', action='store_true', default=True, help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=True, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10, help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10, help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    
    parser.add_argument('--l2', type=float, default=0.00003, metavar='L2', help='L2 regularization weight')
    
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
    
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    
    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')
    
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--graph_type', default='relation', help='relation/GCN3/DeepGCN/MMGCN/MMGCN2')

    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')

    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')

    parser.add_argument('--graph_construct', default='full', help='single/window/fc for MMGCN2; direct/full for others')

    parser.add_argument('--use_gcn', action='store_true', default=False, help='whether to combine spectral and none-spectral methods or not')

    parser.add_argument('--use_residue', action='store_true', default=False, help='whether to use residue information or not')

    parser.add_argument('--multi_modal', action='store_true', default=False, help='whether to use multimodal information')

    parser.add_argument('--mm_fusion_mthd', default='concat', help='method to use multimodal information: concat, gated, concat_subsequently')

    parser.add_argument('--modals', default='avl', help='modals to fusion')

    parser.add_argument('--av_using_lstm', action='store_true', default=False, help='whether to use lstm in acoustic and visual modality')

    parser.add_argument('--Deep_GCN_nlayers', type=int, default=4, help='Deep_GCN_nlayers')

    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')

    parser.add_argument('--use_speaker', action='store_true', default=True, help='whether to use speaker embedding')

    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding')

    parser.add_argument('--norm', default='LN2', help='NORM type')

    parser.add_argument('--testing', action='store_true', default=False, help='testing')

    parser.add_argument('--num_L', type=int, default=3, help='num_hyperconvs')

    parser.add_argument('--num_K', type=int, default=4, help='num_convs')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--split-seed', type=int, default=None,
                        help='IEMOCAP validation split seed; omitted preserves legacy training-seed split')
    parser.add_argument('--rng-policy', choices=['legacy', 'continuous'], default='legacy',
                        help='legacy resets each call; continuous preserves training RNG across epochs')
    parser.add_argument('--batch-policy', choices=['legacy', 'capacity_safe'], default='legacy',
                        help='capacity_safe rejects whole epoch permutations exceeding fixed official graph weights')
    parser.add_argument('--edge-capacity', type=int, default=1000)
    parser.add_argument('--node-capacity', type=int, default=5200)
    parser.add_argument('--batch-max-retries', type=int, default=10000)
    parser.add_argument('--require-noncausal', action='store_true',
                        help='reject accidental causal flags in noncausal reproduction jobs')
    # parser.add_argument('--run_index', type=int)
    parser.add_argument('--calibrate', default=False, action='store_true')
    parser.add_argument('--rank_coff', type=float, default=0)
    parser.add_argument('--contrastlearning', default=False, action='store_true')
    parser.add_argument('--mscl_coff', type=float, default=0)
    parser.add_argument('--cscl_coff', type=float, default=0)
    parser.add_argument('--courselearning', default=False, action='store_true')
    parser.add_argument('--epoch_ratio', type=float, default=0)
    parser.add_argument('--scheduler_steps', type=int, default=0)
    parser.add_argument('--selection-protocol', choices=['official_test', 'validation'],
                        default='official_test',
                        help='official_test faithfully selects on test W-F1; validation selects on a held-out validation split')
    parser.add_argument('--skip-final-test', action='store_true',
                        help='validation-only diagnosis: do not evaluate the test loader after training')
    parser.add_argument('--output-dir', default='runs/train_cm',
                        help='root directory for JSONL logs and latest/best checkpoints')
    parser.add_argument('--resume', default=None,
                        help='resume a full training checkpoint created by this script')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader worker count')

    args = parser.parse_args()
    if args.seed is None:
        parser.error('--seed must be specified')
    if args.skip_final_test and (args.selection_protocol != 'validation' or args.testing):
        parser.error('--skip-final-test requires validation selection and forbids --testing')
    if args.require_noncausal and args.causal:
        parser.error('--require-noncausal forbids --causal')
    if args.batch_policy == 'capacity_safe' and (args.causal or args.selection_protocol != 'validation'
            or args.rng_policy != 'continuous' or args.modals != 'avl' or args.graph_type != 'hyper'):
        parser.error('capacity_safe requires noncausal validation continuous hyper avl protocol')
    if args.batch_policy == 'capacity_safe' and (not 0 < args.edge_capacity <= 1000
            or not 0 < args.node_capacity <= 5200 or args.batch_max_retries < 0):
        parser.error('capacity limits must fit official 1000 hyperedges / 5200 nodes and retry limit must be nonnegative')
    if args.split_seed is None:
        args.split_seed = args.seed
    args.rnn_backend = 'aten' if args.rng_policy == 'continuous' else 'cudnn'
    run_dir = training_run_directory(args)
    run_lock = prepare_run_directory(run_dir, args.resume) if not args.testing else None
    if args.causal:
        args.norm = 'LN'
        if not args.graph_model or not args.multi_modal or args.mm_fusion_mthd != 'concat_DHT' or args.graph_type != 'hyper':
            parser.error('--causal requires --multi_modal --mm_fusion_mthd concat_DHT --graph_type hyper')
    print('TEMPORAL PROTOCOL: {}; effective norm={}'.format('causal (history + current)' if args.causal else 'bidirectional', args.norm))
    today = datetime.datetime.now()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)
    initial_loader = get_IEMOCAP_loaders if args.Dataset == 'IEMOCAP' else get_MELD_loaders
    provenance_loaders = initial_loader(
        batch_size=args.batch_size, num_workers=0,
        valid=.1 if args.selection_protocol == 'validation' else 0.,
        selection_protocol=args.selection_protocol, seed=args.seed,
        split_seed=args.split_seed, rng_policy=args.rng_policy, causal=args.causal,
        batch_policy=args.batch_policy, edge_capacity=args.edge_capacity,
        node_capacity=args.node_capacity, batch_max_retries=args.batch_max_retries)
    args.provenance = training_provenance(args, provenance_loaders)
    del provenance_loaders
    if not args.testing:
        write_run_manifest(run_dir, args)
    seed_everything(args.seed)
    if args.av_using_lstm:
        name_ = args.mm_fusion_mthd+'_'+args.modals+'_'+args.graph_type+'_'+args.graph_construct+'using_lstm_'+args.Dataset
    else:
        name_ = args.mm_fusion_mthd+'_'+args.modals+'_'+args.graph_type+'_'+args.graph_construct+str(args.Deep_GCN_nlayers)+'_'+args.Dataset

    if args.use_speaker:
        name_ = name_+'_speaker'
    if args.use_modal:
        name_ = name_+'_modal'

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    feat2dim = {'IS10':1582,'3DCNN':512,'textCNN':100,'bert':768,'denseface':342,'MELD_text':600,'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024 #feat2dim['textCNN'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_text']

    if args.multi_modal:
        if args.mm_fusion_mthd=='concat':
            if modals == 'avl':
                D_m = D_audio+D_visual+D_text
            elif modals == 'av':
                D_m = D_audio+D_visual
            elif modals == 'al':
                D_m = D_audio+D_text
            elif modals == 'vl':
                D_m = D_visual+D_text
            else:
                raise NotImplementedError
        else:
            D_m = 1024
    else:
        if modals == 'a':
            D_m = D_audio
        elif modals == 'v':
            D_m = D_visual
        elif modals == 'l':
            D_m = D_text
        else:
            raise NotImplementedError
    D_g = 512 if args.Dataset=='IEMOCAP' else 1024
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 512
    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes  = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1

    if args.graph_model:
        model = Model(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=n_speakers,
                                 max_seq_len=200,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda,
                                 graph_type=args.graph_type,
                                 use_topic=args.use_topic,
                                 alpha=args.alpha,
                                 multiheads=args.multiheads,
                                 graph_construct=args.graph_construct,
                                 use_GCN=args.use_gcn,
                                 use_residue=args.use_residue,
                                 D_m_v = D_visual,
                                 D_m_a = D_audio,
                                 modals=args.modals,
                                 att_type=args.mm_fusion_mthd,
                                 av_using_lstm=args.av_using_lstm,
                                 Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                 dataset=args.Dataset,
                                 use_speaker=args.use_speaker,
                                 use_modal=args.use_modal,
                                 norm = args.norm,
                                 num_L = args.num_L,
                                 num_K = args.num_K,
                                 causal = args.causal)
        model_mm = Model(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=n_speakers,
                                 max_seq_len=200,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda,
                                 graph_type=args.graph_type,
                                 use_topic=args.use_topic,
                                 alpha=args.alpha,
                                 multiheads=args.multiheads,
                                 graph_construct=args.graph_construct,
                                 use_GCN=args.use_gcn,
                                 use_residue=args.use_residue,
                                 D_m_v = D_visual,
                                 D_m_a = D_audio,
                                 modals=args.modals,
                                 att_type=args.mm_fusion_mthd,
                                 av_using_lstm=args.av_using_lstm,
                                 Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                 dataset=args.Dataset,
                                 use_speaker=args.use_speaker,
                                 use_modal=args.use_modal,
                                 norm = args.norm,
                                 num_L = args.num_L,
                                 num_K = args.num_K,
                                 causal = args.causal)
        model_intra = Model_Intra(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=n_speakers,
                                 max_seq_len=200,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda,
                                 graph_type=args.graph_type,
                                 use_topic=args.use_topic,
                                 alpha=args.alpha,
                                 multiheads=args.multiheads,
                                 graph_construct=args.graph_construct,
                                 use_GCN=args.use_gcn,
                                 use_residue=args.use_residue,
                                 D_m_v = D_visual,
                                 D_m_a = D_audio,
                                 modals=args.modals,
                                 att_type=args.mm_fusion_mthd,
                                 av_using_lstm=args.av_using_lstm,
                                 Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                 dataset=args.Dataset,
                                 use_speaker=args.use_speaker,
                                 use_modal=args.use_modal,
                                 norm = args.norm,
                                 num_L = args.num_L,
                                 num_K = args.num_K,
                                 causal = args.causal)
        model_inter = Model_Inter(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=n_speakers,
                                 max_seq_len=200,
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda,
                                 graph_type=args.graph_type,
                                 use_topic=args.use_topic,
                                 alpha=args.alpha,
                                 multiheads=args.multiheads,
                                 graph_construct=args.graph_construct,
                                 use_GCN=args.use_gcn,
                                 use_residue=args.use_residue,
                                 D_m_v = D_visual,
                                 D_m_a = D_audio,
                                 modals=args.modals,
                                 att_type=args.mm_fusion_mthd,
                                 av_using_lstm=args.av_using_lstm,
                                 Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                 dataset=args.Dataset,
                                 use_speaker=args.use_speaker,
                                 use_modal=args.use_modal,
                                 norm = args.norm,
                                 num_L = args.num_L,
                                 num_K = args.num_K,
                                 causal = args.causal)
        print ('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    else:
        if args.base_model == 'GRU':
            model = GRUModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)

            print ('Basic GRU Model.')


        elif args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)

            print ('Basic LSTM Model.')

        else:
            print ('Base model must be one of DialogRNN/LSTM/GRU/Transformer')
            raise NotImplementedError

        name = 'Base'

    for backend_model in ([model, model_mm, model_intra, model_inter] if args.graph_model else [model]):
        configure_recurrent_backend(backend_model, args.rnn_backend)
    if cuda:
        model.cuda()
        model_mm.cuda()
        model_intra.cuda()
        model_inter.cuda()

    if args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])

    if args.Dataset == 'MELD':
        loss_function = FocalLoss()
    else:
        if args.class_weight:
            if args.graph_model:
                #loss_function = FocalLoss()
                loss_function  = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
            else:
                loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            if args.graph_model:
                loss_function = nn.NLLLoss()
            else:
                loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    lr = args.lr

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    best_epoch = None
    all_fscore, all_acc, all_loss = [], [], []
    start_epoch = 0
    metrics_path = run_dir / 'metrics.jsonl'
    latest_checkpoint_path = run_dir / 'latest.pt'
    best_checkpoint_path = run_dir / 'best.pt'
    if args.selection_protocol == 'official_test':
        print('SELECTION PROTOCOL: official_test (faithful official behavior; TEST W-F1 selects best checkpoint)')
    else:
        print('SELECTION PROTOCOL: validation (test metrics do not select checkpoints)')
    if args.resume:
        resume_metadata = load_training_checkpoint(
            args.resume, model, optimizer,
            map_location='cpu', expected_args=args
        )
        start_epoch = resume_metadata['start_epoch']
        best_fscore = resume_metadata['best_metric']
        best_epoch = resume_metadata['best_epoch']
        print('resumed from {} at epoch {}'.format(args.resume, start_epoch + 1))

    if args.testing:
        if args.causal:
            raise ValueError('use scripts/evaluate_checkpoint.py --causal with a causal checkpoint')
        if args.Dataset == 'MELD':
            train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                        batch_size=batch_size,
                                                                        num_workers=2)
        elif args.Dataset == 'IEMOCAP':
            train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                        batch_size=batch_size,
                                                                        num_workers=2)
        else:
            print("There is no such dataset")
        state = torch.load("best_model.pth.tar")
        model.load_state_dict(state)
        print('testing loaded model')
        test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, 0, cuda, args.modals, dataset=args.Dataset)
        print('test_acc:',test_acc,'test_fscore:',test_fscore)

    for e in range(start_epoch, n_epochs):
        if args.courselearning:
            epoch_ratio = min(1, round(args.epoch_ratio*int(e/args.scheduler_steps+1),2))
        else:
            epoch_ratio = -1
        print("current ratio = ", epoch_ratio)
        start_time = time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats()

        validation_fraction = 0.1 if args.selection_protocol == 'validation' else 0.0

        if args.Dataset == 'MELD':
            train_loader, valid_loader, test_loader = get_MELD_loaders(valid=validation_fraction,
                                                                        batch_size=batch_size,
                                                                        num_workers=args.num_workers,
                                                                        epoch_ratio=epoch_ratio,
                                                                        selection_protocol=args.selection_protocol,
                                                                        seed=args.seed, causal=args.causal,
                                                                        split_seed=args.split_seed, epoch=e,
                                                                        rng_policy=args.rng_policy,
                                                                        batch_policy=args.batch_policy,
                                                                        edge_capacity=args.edge_capacity,
                                                                        node_capacity=args.node_capacity,
                                                                        batch_max_retries=args.batch_max_retries)
        elif args.Dataset == 'IEMOCAP':
            train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=validation_fraction,
                                                                        batch_size=batch_size,
                                                                        num_workers=args.num_workers,
                                                                        epoch_ratio=epoch_ratio,
                                                                        selection_protocol=args.selection_protocol,
                                                                        seed=args.seed, causal=args.causal,
                                                                        split_seed=args.split_seed, epoch=e,
                                                                        rng_policy=args.rng_policy,
                                                                        batch_policy=args.batch_policy,
                                                                        edge_capacity=args.edge_capacity,
                                                                        node_capacity=args.node_capacity,
                                                                        batch_max_retries=args.batch_max_retries)
        else:
            print("There is no such dataset")
        if args.graph_model:
            train_loss, train_acc, _, _, train_fscore, _, _, _, _, _, train_components = train_or_eval_graph_model(model, model_mm, model_intra, model_inter, loss_function, train_loader, e, cuda, args.modals, optimizer, True, dataset=args.Dataset, return_components=True)
            valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, model_mm, model_intra, model_inter,loss_function, valid_loader, e, cuda, args.modals, dataset=args.Dataset)
            if args.selection_protocol == 'official_test':
                test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, model_mm, model_intra, model_inter,loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset)
                all_fscore.append(test_fscore)
            else:
                test_loss = test_acc = test_fscore = float('nan')
                test_label = test_pred = None
        else:
            train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e, optimizer, True)
            valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
            if args.selection_protocol == 'official_test':
                test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)
                all_fscore.append(test_fscore)
            else:
                test_loss = test_acc = test_fscore = float('nan')
                test_label = test_pred = None

            train_components = {
                'classification_loss': train_loss,
                'cc_loss': 0.0,
                'mscl_loss': 0.0,
                'cscl_loss': 0.0,
                'total_loss': train_loss,
            }

        selection_split, selection_loss, selection_fscore = choose_selection_metrics(
            args.selection_protocol, valid_loss, valid_fscore, test_loss, test_fscore
        )

        if best_loss == None or best_loss > selection_loss:
            best_loss = selection_loss

        improved = best_fscore is None or best_fscore < selection_fscore
        if improved:
            best_fscore = selection_fscore
            best_epoch = e
            if args.selection_protocol == 'official_test':
                best_label, best_pred = test_label, test_pred

        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)

        elapsed_seconds = round(time.time()-start_time, 2)
        max_gpu_memory_mb = (
            round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2) if cuda else 0.0
        )
        epoch_record = {
            'epoch': e + 1,
            'seed': args.seed,
            'split_seed': args.split_seed,
            'split_hash': args.provenance['split']['hash'],
            'rng_policy': args.rng_policy,
            'batch_policy': args.batch_policy,
            'sampler_retries': getattr(train_loader.batch_sampler, 'retries', None),
            'max_batch_edges': getattr(train_loader.batch_sampler, 'max_batch_edges', None),
            'max_batch_nodes': getattr(train_loader.batch_sampler, 'max_batch_nodes', None),
            'selection_protocol': args.selection_protocol,
            'causal': args.causal,
            'effective_norm': args.norm,
            'selection_split': selection_split,
            'selection_weighted_f1': selection_fscore,
            'classification_loss': train_components['classification_loss'],
            'cc_loss': train_components['cc_loss'],
            'mscl_loss': train_components['mscl_loss'],
            'cscl_loss': train_components['cscl_loss'],
            'total_loss': train_components['total_loss'],
            'train_accuracy': train_acc,
            'train_loss': train_loss,
            'train_weighted_f1': train_fscore,
            'validation_accuracy': valid_acc,
            'validation_loss': valid_loss,
            'validation_weighted_f1': valid_fscore,
            'test_accuracy': test_acc,
            'test_weighted_f1': test_fscore,
            'elapsed_seconds': elapsed_seconds,
            'sample_ratio': 1.0 if epoch_ratio == -1 else epoch_ratio,
            'train_dialogues': len(train_loader.dataset),
            'max_gpu_memory_mb': max_gpu_memory_mb,
        }
        write_jsonl_record(metrics_path, epoch_record)
        sampling_metadata = {
            key: epoch_record[key] for key in ('batch_policy', 'sampler_retries',
                                               'max_batch_edges', 'max_batch_nodes')
        }
        save_training_checkpoint(
            latest_checkpoint_path, model, optimizer, e, args, best_fscore, best_epoch,
            sampling_metadata=sampling_metadata
        )
        if improved:
            save_training_checkpoint(
                best_checkpoint_path, model, optimizer, e, args, best_fscore, best_epoch,
                sampling_metadata=sampling_metadata
            )

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, elapsed_seconds))
        if (e+1)%10 == 0:
            print ('----------best {} F-Score: {}'.format(selection_split, best_fscore))
            if best_label is not None:
                print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
                print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))

    if args.selection_protocol == 'validation' and not args.skip_final_test and best_checkpoint_path.exists():
        best_checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
        model.load_state_dict(best_checkpoint['model_state_dict'])
        if cuda:
            model.cuda()
        if args.graph_model:
            final_test_loss, final_test_acc, _, _, final_test_fscore, _, _, _, _, _ = train_or_eval_graph_model(
                model, model_mm, model_intra, model_inter, loss_function, test_loader,
                best_epoch, cuda, args.modals, dataset=args.Dataset
            )
        else:
            final_test_loss, final_test_acc, _, _, _, final_test_fscore, _ = train_or_eval_model(
                model, loss_function, test_loader, best_epoch
            )
        final_test_record = {
            'seed': args.seed,
            'split_seed': args.split_seed,
            'split_hash': args.provenance['split']['hash'],
            'rng_policy': args.rng_policy,
            'causal': args.causal,
            'effective_norm': args.norm,
            'selection_protocol': args.selection_protocol,
            'selected_epoch': best_epoch + 1,
            'selection_weighted_f1': best_fscore,
            'test_loss': final_test_loss,
            'test_accuracy': final_test_acc,
            'test_weighted_f1': final_test_fscore,
        }
        with (run_dir / 'final_test.json').open('x', encoding='utf-8') as handle:
            json.dump(final_test_record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print('final test from validation-selected epoch {}: accuracy {}, weighted F1 {}'.format(
            best_epoch + 1, final_test_acc, final_test_fscore
        ))
