#!/usr/bin/env python3
"""Fixed Train-Fit-only capacity diagnostic; never opens source pkl or held-out splits.

This program starts a new public RoBERTa-large backbone, with dropout disabled to
isolate deterministic fitting capacity. It does not resume/modify either student.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch
from transformers import RobertaConfig, RobertaModel


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def tokens_key(tokens):
    return tuple(int(value) for value in tokens)


def select(prepared, seed, count):
    fit = [int(index) for index in prepared['fit_indices']]
    dev = set(int(index) for index in prepared['dev_indices'])
    if set(fit) & dev:
        raise ValueError('Fit/dev overlap')
    candidates = fit.copy()
    random.Random(seed).shuffle(candidates)
    selected, seen_tokens, seen_text = [], set(), set()
    for index in candidates:
        row = prepared['rows'][index]
        key = tokens_key(prepared['input_ids'][index])
        text = row['processed_text_sha256']
        if key in seen_tokens or text in seen_text:
            continue
        if row['inner_subset'] != 'fit':
            raise ValueError('Selected a row outside Fit')
        selected.append(index)
        seen_tokens.add(key)
        seen_text.add(text)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError('Insufficient distinct Fit text/token sequences')
    return selected


def batch(prepared, indices, device):
    length = max(len(prepared['input_ids'][index]) for index in indices)
    ids = torch.ones((len(indices), length), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, index in enumerate(indices):
        tokens = torch.as_tensor(prepared['input_ids'][index], dtype=torch.long)
        if len(tokens) > 512 or tokens[0] != 0 or tokens[-1] != 2:
            raise ValueError('Unexpected cached tokenization')
        ids[row, :len(tokens)] = tokens
        mask[row, :len(tokens)] = 1
    targets = prepared['targets'][indices].to(dtype=torch.float32)
    return ids.to(device), mask.to(device), targets.to(device)


def features(model, ids, mask):
    result = model(input_ids=ids, attention_mask=mask,
                   output_hidden_states=True, return_dict=True)
    return torch.stack([result.hidden_states[-layer][:, 0, :]
                        for layer in (1, 2, 3, 4)], dim=1)


@torch.no_grad()
def evaluate(model, prepared, selected, device, micro_batch):
    model.eval()
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    for start in range(0, len(selected), micro_batch):
        indices = selected[start:start + micro_batch]
        ids, mask, targets = batch(prepared, indices, device)
        square = (features(model, ids, mask) - targets).square()
        if not torch.isfinite(square).all():
            raise FloatingPointError('Non-finite evaluation loss')
        sums += square.double().sum(dim=(0, 2))
    layers = (sums / (len(selected) * 1024)).cpu().tolist()
    return dict(layer_mse=layers, mse=float(np.mean(layers)), count=len(selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['IEMOCAP', 'MELD'])
    parser.add_argument('--data', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--count', type=int, default=128)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--micro-batch-size', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.seed != {'IEMOCAP': 1475, 'MELD': 67137}[args.dataset]:
        raise ValueError('Dataset seed must match the frozen protocol')
    if args.count not in range(128, 257) or args.steps not in (2, 500):
        raise ValueError('Protocol permits only a two-step engineering smoke or fixed 500-step run')
    if args.micro_batch_size * args.accumulation != 16:
        raise ValueError('Freeze effective batch size at 16')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run_config.json').exists():
        raise FileExistsError('Refuse to overwrite an existing diagnostic run')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Requires assigned local GPU')
    device = torch.device('cuda:0')
    data_hash = sha256(args.data)
    original_training = Path(args.data).parent / 'training' / 'training_config.json'
    if data_hash != json.loads(original_training.read_text())['data_sha256']:
        raise ValueError('Prepared Train file differs from the original training input')
    public_weight_path = Path(args.pretrained) / 'model.safetensors'
    public_weight_hash = sha256(public_weight_path)
    if public_weight_hash != '047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38':
        raise ValueError('Public initialization weights differ from the frozen snapshot')
    prepared = torch.load(args.data, map_location='cpu', weights_only=False)
    expected = {'IEMOCAP': (5283, 4747), 'MELD': (9989, 8989)}[args.dataset]
    if prepared['dataset'] != args.dataset or len(prepared['input_ids']) != expected[0]:
        raise ValueError('Effective Train data mismatch')
    if len(prepared['fit_indices']) != expected[1]:
        raise ValueError('Effective Train-Fit count mismatch')
    if tuple(prepared['targets'].shape) != (expected[0], 4, 1024):
        raise ValueError('Expected four-layer BOS targets')
    selected = select(prepared, args.seed, args.count)
    with open(output / 'selected_fit_manifest.csv', 'w') as stream:
        fields = ['selection_index', 'row_index', 'dataset', 'dialogue_id',
                  'utterance_index', 'processed_text_sha256', 'token_ids_sha256', 'tokens']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for order, index in enumerate(selected):
            row = prepared['rows'][index]
            token_bytes = np.asarray(prepared['input_ids'][index], dtype='<i8').tobytes()
            writer.writerow(dict(selection_index=order, row_index=index, dataset=args.dataset,
                                 dialogue_id=row['dialogue_id'], utterance_index=row['utterance_index'],
                                 processed_text_sha256=row['processed_text_sha256'],
                                 token_ids_sha256=hashlib.sha256(token_bytes).hexdigest(),
                                 tokens=len(prepared['input_ids'][index])))
    config = RobertaConfig.from_pretrained(args.pretrained, local_files_only=True)
    config.hidden_dropout_prob = 0.0
    config.attention_probs_dropout_prob = 0.0
    config.use_cache = False
    model, loading = RobertaModel.from_pretrained(
        args.pretrained, config=config, add_pooling_layer=False,
        local_files_only=True, output_loading_info=True)
    if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise RuntimeError(f'Incomplete public pretrained backbone: {loading}')
    unexpected = [key for key in loading.get('unexpected_keys', [])
                  if not key.startswith(('lm_head.', 'roberta.pooler.', 'pooler.'))]
    if unexpected:
        raise RuntimeError(f'Unexpected pretrained keys: {unexpected}')
    if model.config.hidden_dropout_prob != 0 or model.config.attention_probs_dropout_prob != 0:
        raise RuntimeError('Capacity test must disable configured dropout')
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0:
            raise RuntimeError(f'Dropout remained enabled in {name}')
        if hasattr(module, 'dropout_prob') and module.dropout_prob != 0:
            raise RuntimeError(f'Attention dropout remained enabled in {name}')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.to(device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.0, foreach=False)
    run_config = dict(vars(args), dataset=args.dataset, prepared_sha256=data_hash,
                      public_weight_sha256=public_weight_hash,
                      script_sha256=sha256(__file__), hostname=socket.gethostname(),
                      visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      gpu_name=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                      precision='FP32 parameters, activations, loss, AdamW; no AMP/TF32',
                      dropout={'hidden': 0.0, 'attention': 0.0}, weight_decay=0.0,
                      learning_rate=1e-5, scheduler=None, gradient_clip=1.0,
                      effective_batch_size=16, layer_order=[-1, -2, -3, -4],
                      encoder_parameters=sum(p.numel() for p in model.parameters()),
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      input_split='effective Train Fit only', target_normalization=None,
                      initial_loading_info=loading,
                      evaluation_steps=[0, 25, 50, 100, 200, 300, 400, 500],
                      capacity_test_changes='dropout and weight decay removed; no hyperparameter search')
    dump(output / 'run_config.json', run_config)
    model.config.save_pretrained(output / 'initial_encoder_config')
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    evaluations = []
    def record(step):
        value = evaluate(model, prepared, selected, device, args.micro_batch_size)
        value.update(step=step, elapsed_seconds=time.monotonic() - start,
                     peak_memory_gib=torch.cuda.max_memory_allocated(device) / 2**30)
        evaluations.append(value)
        dump(output / 'evaluations.json', evaluations)
        with open(output / 'evaluation_curve.csv', 'w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['step', 'mse', 'last_mse', 'second_last_mse',
                             'third_last_mse', 'fourth_last_mse', 'elapsed_seconds', 'peak_memory_gib'])
            for item in evaluations:
                writer.writerow([item['step'], item['mse'], *item['layer_mse'],
                                 item['elapsed_seconds'], item['peak_memory_gib']])
        print(json.dumps(dict(event='selected_fit_evaluation', **value)), flush=True)
        model.train()
    record(0)
    order = selected.copy()
    sampler_rng = random.Random(args.seed + 10000)
    sampler_rng.shuffle(order)
    offset = 0
    def next_indices():
        nonlocal offset
        result = []
        while len(result) < args.micro_batch_size:
            remaining = args.micro_batch_size - len(result)
            take = min(remaining, len(order) - offset)
            result.extend(order[offset:offset + take])
            offset += take
            if offset == len(order):
                offset = 0
                sampler_rng.shuffle(order)
        return result
    diagnostics = None
    with open(output / 'optimizer_steps.csv', 'w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['step', 'mean_microbatch_mse', 'gradient_norm_before_clip',
                         'elapsed_seconds', 'peak_memory_gib'])
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(args.accumulation):
                ids, mask, targets = batch(prepared, next_indices(), device)
                predicted = features(model, ids, mask)
                loss = (predicted - targets).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite FP32 training loss')
                (loss / args.accumulation).backward()
                step_loss += float(loss.detach()) / args.accumulation
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError('Non-finite FP32 gradients')
            probe = model.encoder.layer[-1].attention.self.query.weight
            before = probe.detach().clone() if diagnostics is None else None
            if diagnostics is None:
                diagnostics = dict(gradient_present=probe.grad is not None,
                                   gradient_all_finite=bool(torch.isfinite(probe.grad).all()),
                                   gradient_l2=float(probe.grad.norm()),
                                   probe='encoder.layer.23.attention.self.query.weight',
                                   all_encoder_parameters_trainable=all(p.requires_grad for p in model.parameters()))
            optimizer.step()
            if before is not None:
                diagnostics['maximum_parameter_update'] = float((probe.detach() - before).abs().max())
                dump(output / 'gradient_update_verification.json', diagnostics)
            writer.writerow([step, step_loss, float(norm), time.monotonic() - start,
                             torch.cuda.max_memory_allocated(device) / 2**30])
            stream.flush()
            if step % 10 == 0 or args.steps == 2:
                print(json.dumps(dict(event='optimizer_step', step=step, mse=step_loss,
                                      gradient_norm=float(norm), elapsed_seconds=time.monotonic()-start,
                                      peak_memory_gib=torch.cuda.max_memory_allocated(device)/2**30)), flush=True)
            if step in (25, 50, 100, 200, 300, 400, 500) or step == args.steps:
                record(step)
    final_model_hash = None
    if args.steps == 500:
        model.save_pretrained(output / 'final_student', safe_serialization=True)
        final_model_hash = sha256(output/'final_student'/'model.safetensors')
    dump(output / 'completion.json', dict(status='complete', actual_optimizer_steps=args.steps,
         initial_mse=evaluations[0]['mse'], final_mse=evaluations[-1]['mse'],
         reduction_fraction=1-evaluations[-1]['mse']/evaluations[0]['mse'],
         final_layer_mse=evaluations[-1]['layer_mse'], seconds=time.monotonic()-start,
         final_model_sha256=final_model_hash,
         gradient_update=diagnostics, official_validation_accessed=False, test_accessed=False))


if __name__ == '__main__':
    main()
