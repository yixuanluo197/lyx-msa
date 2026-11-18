#!/usr/bin/env python3
"""Train an independent XP04 feature student using effective-Train data only.

Launch with torchrun for DDP. This program never indexes official Validation or Test records.
Full resume is supported at completed epoch boundaries, including each rank's RNG.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from transformers import RobertaModel


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'w') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def atomic_torch_save(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(data, temporary)
    os.replace(temporary, path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])


class PreparedDataset(Dataset):
    def __init__(self, prepared, indices):
        self.ids = prepared['input_ids']
        self.targets = prepared['targets']
        self.indices = [int(index) for index in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        return self.ids[index], self.targets[index], index


def collate(items):
    # RoBERTa padding ID is 1. Right padding leaves BOS at index zero.
    length = max(len(item[0]) for item in items)
    ids = torch.ones((len(items), length), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, (tokens, _, _) in enumerate(items):
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        ids[row, :len(tokens)] = tokens
        mask[row, :len(tokens)] = 1
    return {'input_ids': ids, 'attention_mask': mask,
            'targets': torch.stack([torch.as_tensor(item[1], dtype=torch.float32)
                                    for item in items]),
            'indices': torch.tensor([item[2] for item in items])}


class StridedSampler(Sampler):
    """Distributed evaluation without duplicate padding examples."""
    def __init__(self, size, rank, world):
        self.indices = list(range(rank, size, world))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def student_features(model, batch):
    result = model(input_ids=batch['input_ids'],
                   attention_mask=batch['attention_mask'],
                   output_hidden_states=True, return_dict=True)
    # Official pkl slot order is LAST, SECOND-LAST, THIRD-LAST, FOURTH-LAST.
    return torch.stack([result.hidden_states[-layer][:, 0, :]
                        for layer in (1, 2, 3, 4)], dim=1)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()
            if key != 'indices'}


@torch.no_grad()
def evaluate(model, loader, device, world):
    model.eval()
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    # No autocast: final student selection is based on FP32 hidden vectors.
    for batch in loader:
        batch = to_device(batch, device)
        predicted = student_features(model, batch).float()
        squared = (predicted - batch['targets']).square()
        if not torch.isfinite(squared).all():
            raise FloatingPointError('Non-finite FP32 internal-dev features/loss')
        totals[:4] += squared.double().sum(dim=(0, 2))
        totals[4] += predicted.shape[0] * predicted.shape[2]
    if world > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if totals[4].item() <= 0:
        raise ValueError('Internal dev set is empty')
    layers = (totals[:4] / totals[4]).cpu().tolist()
    return {'mse': float(sum(layers) / 4), 'layer_mse': layers,
            'samples': int(totals[4].item() / 1024)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--tokenizer', help='Directory containing prepared reference-compatible tokenizer')
    parser.add_argument('--dataset', required=True, choices=['IEMOCAP', 'MELD'])
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--micro-batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--min-delta', type=float, default=0.0)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-steps', type=int, default=0,
                        help='Train-only engineering check, not a formal model; optimizer step count')
    return parser.parse_args()


def validate_prepared(prepared, dataset):
    required = {'input_ids', 'targets', 'rows', 'fit_indices', 'dev_indices'}
    if not required <= set(prepared):
        raise ValueError(f'Missing prepared keys: {required - set(prepared)}')
    targets = prepared['targets']
    count = len(prepared['input_ids'])
    if tuple(targets.shape) != (count, 4, 1024) or not torch.isfinite(targets).all():
        raise ValueError('Expected finite [N, 4, 1024] Train targets')
    expected = {'IEMOCAP': 5283, 'MELD': 9989}[dataset]
    if count != expected or len(prepared['rows']) != count:
        raise ValueError(f'Expected effective Train count {expected}, got {count}')
    fit = [int(index) for index in prepared['fit_indices']]
    dev = [int(index) for index in prepared['dev_indices']]
    if not fit or not dev or len(fit) != len(set(fit)) or len(dev) != len(set(dev)):
        raise ValueError('Fit/dev index lists must be nonempty and unique')
    if set(fit) & set(dev) or set(fit) | set(dev) != set(range(count)):
        raise ValueError('Internal split must partition effective Train exactly')
    fit_dialogues = {str(prepared['rows'][index]['dialogue_id']) for index in fit}
    dev_dialogues = {str(prepared['rows'][index]['dialogue_id']) for index in dev}
    if fit_dialogues & dev_dialogues:
        raise ValueError('Dialogue leakage between inner fit and dev')
    for row, tokens in zip(prepared['rows'], prepared['input_ids']):
        if row.get('dataset', dataset) != dataset:
            raise ValueError('Cross-dataset prepared target detected')
        if len(tokens) < 2 or len(tokens) > 512 or tokens[0] != 0 or tokens[-1] != 2:
            raise ValueError('Expected BOS=0/EOS=2 with at most 512 total tokens')
    return fit, dev


def main():
    args = parse_args()
    if not 1 <= args.epochs <= 10 or args.micro_batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError('Invalid epoch/batch limits')
    if args.patience < 1 or args.learning_rate != 1e-5:
        raise ValueError('Protocol requires lr=1e-5 and positive patience')
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for authorized RoBERTa-large training')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(args.seed + rank)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'best_model_lock.json').exists():
        raise RuntimeError('Completed model is locked; refuse further training in this run')
    latest_path = run_dir / 'latest_training.pt'
    if latest_path.exists() and not args.resume:
        raise RuntimeError('Existing latest_training.pt requires explicit --resume')
    if args.resume and (not latest_path.exists() or args.smoke_steps):
        raise RuntimeError('Resume needs a full epoch checkpoint and cannot be smoke')
    data_hash = sha256(args.data)
    code_hash = sha256(__file__)
    prepared = torch.load(args.data, map_location='cpu', weights_only=False)
    fit_indices, dev_indices = validate_prepared(prepared, args.dataset)
    if args.smoke_steps:
        # The profile uses the longest Train-only sequences to expose activation peaks.
        needed = max(world * args.micro_batch_size * args.gradient_accumulation_steps * args.smoke_steps,
                     world * args.micro_batch_size)
        fit_indices = sorted(fit_indices, key=lambda index: len(prepared['input_ids'][index]), reverse=True)[:needed]
        dev_indices = sorted(dev_indices, key=lambda index: len(prepared['input_ids'][index]), reverse=True)[:max(world * args.micro_batch_size, 2)]
    fit_dataset = PreparedDataset(prepared, fit_indices)
    dev_dataset = PreparedDataset(prepared, dev_indices)
    fit_sampler = DistributedSampler(fit_dataset, num_replicas=world, rank=rank, shuffle=True,
                                     seed=args.seed, drop_last=False)
    dev_sampler = StridedSampler(len(dev_dataset), rank, world)
    # Explicit generators isolate loader bookkeeping from model dropout RNG.
    loader_generator = torch.Generator().manual_seed(args.seed + 10000 + rank)
    loader_kwargs = dict(batch_size=args.micro_batch_size, collate_fn=collate,
                         num_workers=args.workers, pin_memory=True, generator=loader_generator)
    fit_loader = DataLoader(fit_dataset, sampler=fit_sampler, **loader_kwargs)
    dev_kwargs = dict(loader_kwargs, batch_size=args.eval_batch_size)
    dev_loader = DataLoader(dev_dataset, sampler=dev_sampler, **dev_kwargs)
    model, loading = RobertaModel.from_pretrained(args.pretrained, add_pooling_layer=False,
                                                  local_files_only=True, output_loading_info=True)
    if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise RuntimeError(f'Pretrained encoder did not load completely: {loading}')
    allowed_unused = ('lm_head.', 'roberta.pooler.', 'pooler.')
    unexpected = [key for key in loading.get('unexpected_keys', [])
                  if not key.startswith(allowed_unused)]
    if unexpected:
        raise RuntimeError(f'Unexpected non-head pretrained keys: {unexpected}')
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=args.weight_decay, foreach=False)
    scaler = torch.cuda.amp.GradScaler(init_scale=4096.0)
    completed_epoch, global_step, best_loss, bad_epochs, best_epoch = 0, 0, math.inf, 0, None
    if args.resume:
        checkpoint = torch.load(latest_path, map_location='cpu', weights_only=False)
        if checkpoint['data_sha256'] != data_hash or checkpoint['world_size'] != world:
            raise RuntimeError('Resume data/world-size mismatch')
        immutable = ['dataset', 'seed', 'micro_batch_size', 'gradient_accumulation_steps',
                     'learning_rate', 'weight_decay', 'max_grad_norm', 'patience', 'min_delta', 'epochs', 'eval_batch_size']
        if any(checkpoint['arguments'][key] != vars(args)[key] for key in immutable):
            raise RuntimeError('Resume protocol configuration mismatch')
        if checkpoint['code_sha256'] != code_hash:
            raise RuntimeError('Resume script changed; use an explicitly reviewed new run')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        completed_epoch, global_step = checkpoint['completed_epoch'], checkpoint['global_step']
        best_loss, bad_epochs, best_epoch = checkpoint['best_loss'], checkpoint['bad_epochs'], checkpoint['best_epoch']
        restore_rng(checkpoint['rng_by_rank'][rank])
        loader_generator.set_state(checkpoint['loader_rng_by_rank'][rank])
        del checkpoint
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                       broadcast_buffers=False) if world > 1 else model
    configuration = dict(vars(args), data_sha256=data_hash, code_sha256=code_hash,
                         hostname=socket.gethostname(), world_size=world,
                         effective_batch_size=args.micro_batch_size * world * args.gradient_accumulation_steps,
                         precision={'training': 'fp16 autocast, FP32 parameters/AdamW states', 'inner_dev': 'FP32'},
                         loss='mean of four layer BOS MSE; no target normalization; no emotion labels',
                         layer_order=[-1, -2, -3, -4], fit_samples=len(fit_indices), dev_samples=len(dev_indices),
                         distributed_train_padding=world * len(fit_sampler) - len(fit_dataset),
                         initial_loading_info=loading,
                         encoder_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                         trainable_parameter_count=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                         all_encoder_parameters_trainable=all(parameter.requires_grad for parameter in model.parameters()),
                         initial_amp_loss_scale=4096.0,
                         resume_semantics='complete epoch boundary; model, optimizer, scaler, per-rank RNG, loader RNG',
                         scheduler=None, start_utc=utc_now())
    if rank == 0:
        atomic_json(run_dir / ('smoke_config.json' if args.smoke_steps else 'training_config.json'), configuration)
        model.config.save_pretrained(run_dir / 'encoder_config')
    torch.cuda.reset_peak_memory_stats(device)
    total_start = time.monotonic()
    initial = evaluate(model, dev_loader, device, world) if not args.resume else None
    if rank == 0 and initial is not None:
        atomic_json(run_dir / ('smoke_initial_dev.json' if args.smoke_steps else 'initial_inner_dev.json'), initial)
        print(json.dumps({'event': 'initial_inner_dev', **initial}), flush=True)
    diagnostic = None
    skip_count = 0
    smoke_finished = False
    for epoch in range(completed_epoch + 1, args.epochs + 1):
        if bad_epochs >= args.patience:
            break
        epoch_start = time.monotonic()
        fit_sampler.set_epoch(epoch - 1)
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        total_squared, total_samples = 0.0, 0
        for batch_index, batch in enumerate(fit_loader):
            batch = to_device(batch, device)
            group_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            group_length = min(args.gradient_accumulation_steps, len(fit_loader) - group_start)
            group_samples = min(group_length * args.micro_batch_size,
                                len(fit_sampler) - group_start * args.micro_batch_size)
            update = (batch_index + 1) % args.gradient_accumulation_steps == 0 or batch_index + 1 == len(fit_loader)
            sync = contextlib.nullcontext() if update or world == 1 else wrapped.no_sync()
            with sync:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    predicted = student_features(wrapped, batch)
                    loss = (predicted.float() - batch['targets']).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite distillation loss')
                scaler.scale(loss * batch['targets'].shape[0] / group_samples).backward()
            count = batch['targets'].shape[0]
            total_squared += loss.item() * count
            total_samples += count
            if update:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                before = None
                if diagnostic is None or diagnostic.get('maximum_parameter_update', 0) == 0:
                    probe = model.encoder.layer[-1].attention.self.query.weight
                    before = probe.detach().float().clone()
                    probe_grad = probe.grad
                    diagnostic = {'probe_parameter': 'encoder.layer.23.attention.self.query.weight',
                                  'gradient_present': probe_grad is not None,
                                  'gradient_all_finite': bool(torch.isfinite(probe_grad).all()) if probe_grad is not None else False,
                                  'gradient_l2': float(probe_grad.norm()) if probe_grad is not None else None}
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < previous_scale
                skip_count += int(skipped)
                if before is not None:
                    diagnostic['maximum_parameter_update'] = float((probe.detach().float() - before).abs().max())
                    diagnostic['optimizer_step_skipped'] = skipped
                    diagnostic['grad_norm_before_clip'] = float(grad_norm) if torch.isfinite(grad_norm) else None
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and (global_step % args.log_every == 0 or args.smoke_steps):
                    print(json.dumps({'event': 'train_step', 'epoch': epoch, 'global_step': global_step,
                                      'batch_index': batch_index, 'mse': loss.item(), 'scale': scaler.get_scale(),
                                      'skipped': skipped, 'seconds_elapsed': time.monotonic() - total_start,
                                      'peak_memory_gib': torch.cuda.max_memory_allocated(device) / 2**30}), flush=True)
                if args.smoke_steps and global_step >= args.smoke_steps:
                    smoke_finished = True
                    break
        aggregate = torch.tensor([total_squared, total_samples], dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(aggregate)
        dev = evaluate(model, dev_loader, device, world)
        epoch_record = {'epoch': epoch, 'global_step': global_step,
                        'train_mse': float(aggregate[0] / aggregate[1]),
                        'inner_dev_mse': dev['mse'], 'inner_dev_layer_mse': dev['layer_mse'],
                        'train_examples_including_distributed_padding': int(aggregate[1]),
                        'inner_dev_samples': dev['samples'], 'seconds': time.monotonic() - epoch_start,
                        'elapsed_seconds': time.monotonic() - total_start, 'amp_skipped_steps_rank0': skip_count,
                        'utc': utc_now()}
        improved = dev['mse'] < best_loss - args.min_delta
        if improved:
            best_loss, best_epoch, bad_epochs = dev['mse'], epoch, 0
        else:
            bad_epochs += 1
        if rank == 0:
            print(json.dumps({'event': 'epoch', **epoch_record, 'improved': improved}), flush=True)
            with open(run_dir / ('smoke_history.jsonl' if args.smoke_steps else 'history.jsonl'), 'a') as stream:
                stream.write(json.dumps(epoch_record, allow_nan=False) + '\n')
        if args.smoke_steps:
            break
        rngs, loader_rngs = [rng_state()], [loader_generator.get_state()]
        if world > 1:
            rngs, loader_rngs = [None] * world, [None] * world
            dist.all_gather_object(rngs, rng_state())
            dist.all_gather_object(loader_rngs, loader_generator.get_state())
        if rank == 0:
            if improved:
                best_dir = run_dir / 'best_student'
                best_dir.mkdir(exist_ok=True)
                model.save_pretrained(best_dir, safe_serialization=True)
                if args.tokenizer:
                    for token_file in Path(args.tokenizer).iterdir():
                        if token_file.is_file() and token_file.suffix in {'.json', '.txt', '.bpe'}:
                            # Tokenizer config is distinct from encoder config.json.
                            if token_file.name != 'config.json':
                                shutil.copy2(token_file, best_dir / token_file.name)
                atomic_json(run_dir / 'best_selection.json', {'epoch': best_epoch, 'inner_dev_mse': best_loss,
                                                             'criterion': 'minimum FP32 Train-internal Dialogue-held-out four-layer MSE'})
            atomic_torch_save(latest_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                           'scaler': scaler.state_dict(), 'rng_by_rank': rngs,
                                           'loader_rng_by_rank': loader_rngs, 'completed_epoch': epoch,
                                           'global_step': global_step, 'best_loss': best_loss,
                                           'best_epoch': best_epoch, 'bad_epochs': bad_epochs,
                                           'world_size': world, 'arguments': vars(args),
                                           'data_sha256': data_hash, 'code_sha256': code_hash,
                                           'scheduler': None, 'completed_utc': utc_now()})
        if world > 1:
            dist.barrier()
        completed_epoch = epoch
        if bad_epochs >= args.patience:
            break
    memory = {'rank': rank, 'local_rank': local_rank,
              'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
              'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30,
              'device': torch.cuda.get_device_name(device)}
    memory_by_rank = [memory]
    if world > 1:
        memory_by_rank = [None] * world
        dist.all_gather_object(memory_by_rank, memory)
    if rank == 0:
        summary = {'dataset': args.dataset, 'completed_epochs': completed_epoch,
                   'best_epoch': best_epoch, 'best_inner_dev_mse': best_loss,
                   'global_steps': global_step, 'early_stopped': bad_epochs >= args.patience,
                   'training_complete': not bool(args.smoke_steps), 'smoke_only': bool(args.smoke_steps),
                   'elapsed_seconds': time.monotonic() - total_start, 'memory_by_rank': memory_by_rank,
                   'gradient_update_check': diagnostic, 'official_validation_accessed': False,
                   'test_accessed': False, 'completed_utc': utc_now()}
        if args.smoke_steps:
            summary['smoke_pass'] = bool(diagnostic and diagnostic['gradient_present']
                                          and diagnostic['gradient_all_finite']
                                          and diagnostic.get('maximum_parameter_update', 0) > 0)
            atomic_json(run_dir / 'smoke_result.json', summary)
        else:
            best_dir = run_dir / 'best_student'
            files = {str(path.resolve()): sha256(path) for path in best_dir.iterdir() if path.is_file()}
            files[str(Path(__file__).resolve())] = code_hash
            files[str((run_dir / 'training_config.json').resolve())] = sha256(run_dir / 'training_config.json')
            atomic_json(run_dir / 'training_summary.json', summary)
            atomic_json(run_dir / 'best_model_lock.json', {'dataset': args.dataset, 'training_complete': True,
                        'best_model_dir': str(best_dir.resolve()), 'files': files,
                        'best_epoch': best_epoch, 'best_inner_dev_mse': best_loss,
                        'selection_split': 'effective Train internal Dialogue holdout only',
                        'prepared_train_sha256': data_hash, 'completed_utc': utc_now()})
        print(json.dumps({'event': 'complete', **summary}), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if args.smoke_steps and not (diagnostic and diagnostic['gradient_present']
                                  and diagnostic['gradient_all_finite']
                                  and diagnostic.get('maximum_parameter_update', 0) > 0):
        raise RuntimeError('Smoke failed: no finite nonzero encoder parameter update; see retained smoke evidence')


if __name__ == '__main__':
    main()
