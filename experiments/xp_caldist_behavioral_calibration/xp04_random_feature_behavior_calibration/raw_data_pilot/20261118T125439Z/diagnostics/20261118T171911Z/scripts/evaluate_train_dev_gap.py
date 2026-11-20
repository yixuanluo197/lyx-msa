#!/usr/bin/env python3
"""Read only the original Train cache; FP32/eval loss for initial and epoch-8 encoders."""
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader
from transformers import RobertaModel


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent-run', required=True)
    ap.add_argument('--diagnostic-dir', required=True)
    ap.add_argument('--dataset', choices=['IEMOCAP', 'MELD'], required=True)
    ap.add_argument('--pretrained', required=True)
    ap.add_argument('--batch-size', type=int, default=4)
    args = ap.parse_args()
    p, d = Path(args.parent_run).resolve(), Path(args.diagnostic_dir).resolve()
    out = d / 'results'
    local = d / 'local_only' / f'gap_{args.dataset}'
    local.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / f'train_dev_gap_{args.dataset}.json'
    if result_path.exists():
        raise RuntimeError('Completed gap diagnostic already exists; refusing replacement')
    original_script = p / 'scripts/train_student.py'
    spec = importlib.util.spec_from_file_location('original_train_student', original_script)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    train_dir = p / 'local_only' / args.dataset / 'training'
    lock_path = train_dir / 'best_model_lock.json'
    lock = json.loads(lock_path.read_text())
    original_config_path = train_dir / 'training_config.json'
    original_config = json.loads(original_config_path.read_text())
    if lock['best_epoch'] != 8 or lock['dataset'] != args.dataset:
        raise RuntimeError('Unexpected best model identity')
    data_path = p / 'local_only' / args.dataset / 'train_prepared.pt'
    data_hash = sha256(data_path)
    if data_hash != lock['prepared_train_sha256'] or data_hash != original_config['data_sha256']:
        raise RuntimeError('Original Train cache does not match training lock/config')
    paths_to_verify = [train_dir / 'best_student/model.safetensors',
                       train_dir / 'best_student/config.json', original_script, original_config_path]
    verified = {}
    for path in paths_to_verify:
        actual = sha256(path)
        if actual != lock['files'][str(path)]:
            raise RuntimeError(f'Original locked input changed: {path}')
        verified[str(path)] = actual
    pretrained_weights = Path(args.pretrained) / 'model.safetensors'
    pretrained_hash = sha256(pretrained_weights)
    if pretrained_hash != '047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38':
        raise RuntimeError('Unexpected public pretrained model hash')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Expected allocated CUDA GPU')
    torch.cuda.set_device(0)
    prepared = torch.load(data_path, map_location='cpu', weights_only=False)
    fit, dev = helpers.validate_prepared(prepared, args.dataset)
    expected = {'IEMOCAP': (4747, 536), 'MELD': (8989, 1000)}[args.dataset]
    if (len(fit), len(dev)) != expected:
        raise RuntimeError('Unexpected original Fit/dev partition counts')
    gpu_query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader'], text=True)
    launch = {'dataset': args.dataset, 'start_utc': utc(), 'hostname': socket.gethostname(),
              'pid': os.getpid(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'gpu_inventory': gpu_query, 'torch_version': torch.__version__,
              'arguments': vars(args), 'script_sha256': sha256(__file__),
              'train_cache_sha256': data_hash, 'best_lock_sha256': sha256(lock_path),
              'locked_original_inputs': verified, 'pretrained_sha256': pretrained_hash,
              'fit_count': len(fit), 'inner_dev_count': len(dev),
              'official_validation_accessed': False, 'test_accessed': False,
              'model_precision': 'FP32', 'eval_mode': True, 'autocast_enabled': False,
              'layer_order': [-1, -2, -3, -4],
              'loss': 'mean of four BOS MSE; float32 squared differences, float64 sum / N / 1024',
              'original_loss_helper_reused': str(original_script)}
    save_json(local / 'launch.json', launch)
    print(json.dumps({'event': 'verified_inputs', 'dataset': args.dataset, 'fit': len(fit), 'dev': len(dev)}), flush=True)
    results = []
    for model_name, model_path in [('public_initial', Path(args.pretrained)),
                                   ('best_epoch8', train_dir / 'best_student')]:
        model, loading = RobertaModel.from_pretrained(str(model_path), add_pooling_layer=False,
                                                       local_files_only=True, output_loading_info=True)
        if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise RuntimeError(f'Incomplete model load: {loading}')
        unexpected = [key for key in loading.get('unexpected_keys', [])
                      if not key.startswith(('lm_head.', 'roberta.pooler.', 'pooler.'))]
        if unexpected:
            raise RuntimeError(f'Unexpected encoder weights: {unexpected}')
        model.float().to('cuda:0').eval()
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise RuntimeError('Non-FP32 parameter')
        torch.cuda.reset_peak_memory_stats()
        for subset, indices in [('fit', fit), ('inner_dev', dev)]:
            started = time.monotonic()
            loader = DataLoader(helpers.PreparedDataset(prepared, indices), batch_size=args.batch_size,
                                shuffle=False, collate_fn=helpers.collate, num_workers=0, pin_memory=True)
            values = helpers.evaluate(model, loader, torch.device('cuda:0'), world=1)
            if values['samples'] != len(indices):
                raise RuntimeError('Evaluation coverage mismatch')
            result = {'dataset': args.dataset, 'model': model_name, 'subset': subset,
                      'n': values['samples'], 'overall_mse': values['mse'],
                      **{f'layer{i+1}_mse': v for i, v in enumerate(values['layer_mse'])},
                      'seconds': time.monotonic() - started,
                      'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'precision': 'FP32', 'mode': 'eval', 'batch_size': args.batch_size}
            results.append(result)
            save_json(local / 'progress.json', {'results': results, 'last_update': utc()})
            print(json.dumps({'event': 'subset_complete', **result}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    # Check every source file again after inference; none may change during the diagnostic.
    after = {str(path): sha256(path) for path in paths_to_verify}
    if after != verified or sha256(data_path) != data_hash:
        raise RuntimeError('Original input changed during inference')
    gaps = []
    for name in ('public_initial', 'best_epoch8'):
        f, v = [next(r for r in results if r['model'] == name and r['subset'] == subset)
                for subset in ('fit', 'inner_dev')]
        gaps.append({'dataset': args.dataset, 'model': name, 'fit_mse': f['overall_mse'],
                     'inner_dev_mse': v['overall_mse'],
                     'dev_minus_fit_mse': v['overall_mse'] - f['overall_mse'],
                     'dev_to_fit_mse_ratio': v['overall_mse'] / f['overall_mse'],
                     'layer_dev_minus_fit_mse': [v[f'layer{i}_mse'] - f[f'layer{i}_mse'] for i in range(1, 5)]})
    final = {'status': 'COMPLETE', 'dataset': args.dataset, 'launch': launch, 'results': results,
             'gaps': gaps, 'source_inputs_unchanged': True, 'completed_utc': utc(),
             'old_best_inner_dev_mse': lock['best_inner_dev_mse'],
             'warning': 'FP32 eval Fit loss is descriptive; stochastic train-mode AMP logs are not directly comparable.'}
    save_json(result_path, final)
    with (out / f'train_dev_gap_{args.dataset}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({'event': 'diagnostic_complete', 'dataset': args.dataset, 'gaps': gaps}), flush=True)


if __name__ == '__main__':
    main()
