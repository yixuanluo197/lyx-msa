#!/usr/bin/env python3
"""Bounded Train-Fit capacity extension from the previous 500-step model.

Initialization is weights-only continuation with NEW AdamW states and reset sampler;
--resume instead restores this run's full saved step-2000 checkpoint.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import socket
import time
import traceback

import numpy as np
import torch
from transformers import RobertaModel

ACTIVE_OUTPUT = None


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def atomic_save(path, state):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def main():
    global ACTIVE_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['IEMOCAP', 'MELD'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--old-diagnostic', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    old = Path(args.old_diagnostic).resolve()
    prior = old / 'local_only' / 'overfit' / args.dataset
    if args.resume and args.smoke:
        raise ValueError('Smoke is not resumable')
    if args.resume:
        if not (output / 'stage_2000_checkpoint.pt').is_file() or (output / 'completion.json').exists():
            raise ValueError('Resume requires an unfinished run with full stage-2000 checkpoint')
    else:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError('Refuse nonempty output directory')
    ACTIVE_OUTPUT = output
    seed = {'IEMOCAP': 1475, 'MELD': 67137}[args.dataset]
    script_hash = sha256(__file__)
    helper_path = old / 'scripts' / 'small_sample_overfit.py'
    helper_hash = sha256(helper_path)
    if helper_hash != '617ee874d859564c6a9f9ff90f056bca966cdfe2b6f8bf7f49d52926393ad2af':
        raise ValueError('Old batch/features/evaluate implementation changed')
    spec = importlib.util.spec_from_file_location('original_overfit_helper', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    old_cfg = json.loads((prior / 'run_config.json').read_text())
    old_completion = json.loads((prior / 'completion.json').read_text())
    if old_cfg['dataset'] != args.dataset or old_cfg['seed'] != seed:
        raise ValueError('Old dataset/seed mismatch')
    if old_completion['status'] != 'complete' or old_completion['actual_optimizer_steps'] != 500:
        raise ValueError('Expected completed 500-step source model')
    data = Path(old_cfg['data']).resolve()
    data_hash = sha256(data)
    if data_hash != old_cfg['prepared_sha256']:
        raise ValueError('Train cache differs from the original fixed input')
    source_model = prior / 'final_student'
    source_model_hash = sha256(source_model / 'model.safetensors')
    if source_model_hash != old_completion['final_model_sha256']:
        raise ValueError('Old final model differs from original completion hash')
    manifest_path = prior / 'selected_fit_manifest.csv'
    manifest_hash = sha256(manifest_path)
    selected_rows = list(csv.DictReader(manifest_path.open()))
    prepared = torch.load(data, map_location='cpu', weights_only=False)
    if prepared['dataset'] != args.dataset:
        raise ValueError('Cached dataset mismatch')
    expected_total, expected_fit = {'IEMOCAP': (5283, 4747), 'MELD': (9989, 8989)}[args.dataset]
    if tuple(prepared['targets'].shape) != (expected_total, 4, 1024):
        raise ValueError('Wrong target shape')
    if len(prepared['fit_indices']) != expected_fit or len(selected_rows) != 128:
        raise ValueError('Wrong Fit/sample count')
    fit = {int(i) for i in prepared['fit_indices']}
    dev = {int(i) for i in prepared['dev_indices']}
    selected, seen_text, seen_tokens = [], set(), set()
    row_records = []
    for order, entry in enumerate(selected_rows):
        index = int(entry['row_index'])
        if int(entry['selection_index']) != order or index not in fit or index in dev:
            raise ValueError('Manifest order or Fit boundary mismatch')
        row = prepared['rows'][index]
        tokens = np.asarray(prepared['input_ids'][index], dtype='<i8')
        token_hash = hashlib.sha256(tokens.tobytes()).hexdigest()
        if (row['inner_subset'] != 'fit' or entry['dataset'] != args.dataset or
                str(row['dialogue_id']) != entry['dialogue_id'] or
                int(row['utterance_index']) != int(entry['utterance_index']) or
                row['processed_text_sha256'] != entry['processed_text_sha256'] or
                token_hash != entry['token_ids_sha256'] or
                len(tokens) != int(entry['tokens'])):
            raise ValueError('Manifest text/token/row alignment mismatch')
        if entry['processed_text_sha256'] in seen_text or token_hash in seen_tokens:
            raise ValueError('Duplicate selected text/token sequence')
        seen_text.add(entry['processed_text_sha256']); seen_tokens.add(token_hash)
        selected.append(index)
        target_array = prepared['targets'][index].float().contiguous().numpy().astype('<f4', copy=False)
        if not np.isfinite(target_array).all():
            raise ValueError('Non-finite selected target')
        row_records.append(dict(entry, target_sha256=hashlib.sha256(target_array.tobytes()).hexdigest()))
    target_array = prepared['targets'][selected].float().contiguous().numpy().astype('<f4', copy=False)
    targets_hash = hashlib.sha256(target_array.tobytes()).hexdigest()
    anchors = dict(script_sha256=script_hash, helper_sha256=helper_hash,
                   prepared_sha256=data_hash, source_model_sha256=source_model_hash,
                   selected_manifest_sha256=manifest_hash, selected_targets_sha256=targets_hash)
    protocol = dict(dataset=args.dataset, seed=seed, sample_count=128,
                    effective_batch=16, micro_batch=4, accumulation=4,
                    initial_global_step=500, first_stage_end=2000, maximum_global_step=5000,
                    first_eval_steps=[500, 750, 1000, 1250, 1500, 1750, 2000],
                    extension_eval_steps=[2500, 3000, 3500, 4000, 4500, 5000],
                    extension_rule='continue iff (mse1500-mse2000)/mse1500 > 0.05',
                    extension_threshold=0.05, precision='FP32; AMP/TF32 disabled',
                    dropout=0.0, weight_decay=0.0, learning_rate=1e-5,
                    adam_betas=[0.9, 0.999], adam_eps=1e-8, gradient_clip=1.0,
                    gradient_checkpointing='nonreentrant', loss='mean four original-scale BOS MSE',
                    layer_order=[-1, -2, -3, -4], smoke=args.smoke,
                    threads=args.threads,
                    initialization='old step500 weights only; NEW AdamW state from zero; sampler RNG reset seed+10000',
                    resume_semantics='--resume restores THIS run stage2000 model, optimizer, RNG and sampler exactly',
                    target_hash_anchor='old prepared file SHA anchors target content; selected FP32 byte hashes recorded here',
                    **anchors)
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if args.resume:
        previous = json.loads((output / 'run_config.json').read_text())
        if previous['protocol_sha256'] != protocol_hash or previous['protocol'] != protocol:
            raise ValueError('Resume protocol/code/input mismatch')
    else:
        dump(output / 'run_config.json', dict(protocol=protocol, protocol_sha256=protocol_hash,
             hostname=socket.gethostname(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
             torch_version=torch.__version__, threads=args.threads))
        with (output / 'selected_fit_manifest.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(row_records[0]))
            writer.writeheader(); writer.writerows(row_records)
        dump(output / 'decision.json', dict(status='PENDING_STEP_2000',
             rule=protocol['extension_rule'], threshold=0.05, rule_frozen_before_training=True,
             smoke=args.smoke, mse1500=None, mse2000=None, relative_decrease=None, continue_to_5000=None))
    torch.set_num_threads(args.threads)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    random.seed(seed); np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Assigned CUDA GPU required')
    device = torch.device('cuda:0')
    model, loading = RobertaModel.from_pretrained(source_model, local_files_only=True,
         add_pooling_layer=False, output_loading_info=True, torch_dtype=torch.float32)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Incomplete original step500 model load: {loading}')
    if model.config.hidden_dropout_prob != 0 or model.config.attention_probs_dropout_prob != 0:
        raise ValueError('Source model dropout config is not zero')
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0:
            raise ValueError(f'Enabled dropout in {name}')
        if hasattr(module, 'dropout_prob') and module.dropout_prob != 0:
            raise ValueError(f'Enabled attention dropout in {name}')
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.to(device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0, foreach=False)
    sampler_rng = random.Random(seed + 10000)
    order = selected.copy(); sampler_rng.shuffle(order)
    offset = 0
    start = time.monotonic()
    elapsed_before = 0.0
    evaluations = []
    step = 500
    diagnostics = None
    decision = json.loads((output / 'decision.json').read_text())
    if args.resume:
        checkpoint_manifest=json.loads((output/'stage_2000_checkpoint_manifest.json').read_text())
        if sha256(output/'stage_2000_checkpoint.pt') != checkpoint_manifest['sha256']:
            raise ValueError('Stage2000 checkpoint SHA mismatch')
        state = torch.load(output / 'stage_2000_checkpoint.pt', map_location='cpu', weights_only=False)
        if state['protocol_sha256'] != protocol_hash or state['global_step'] != 2000:
            raise ValueError('Full resume checkpoint protocol/step mismatch')
        if not state['decision']['continue_to_5000']:
            raise ValueError('Step2000 decision does not authorize extension')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        for parameter, values in optimizer.state.items():
            for key in ('exp_avg', 'exp_avg_sq'):
                if values[key].device != parameter.device:
                    raise RuntimeError('Optimizer momentum did not follow parameter device')
        random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch']); torch.cuda.set_rng_state(state['rng']['cuda'])
        order, offset = state['sampler_order'], state['sampler_offset']
        if len(order)!=128 or set(order)!=set(selected) or not 0<=offset<128:
            raise ValueError('Corrupt sampler state')
        sampler_rng.setstate(state['sampler_rng'])
        evaluations, decision = state['evaluations'], state['decision']
        diagnostics, step = state['gradient_verification'], state['global_step']
        elapsed_before = state['elapsed_seconds']
        rows = list(csv.DictReader((output / 'optimizer_steps.csv').open()))
        if len(rows)<1500 or [int(row['global_step']) for row in rows] != list(range(501,501+len(rows))):
            raise ValueError('Resume CSV does not contain a continuous saved step2000 prefix')
        if len(rows)>1500:
            # Preserve later interrupted work, then restore the exact saved boundary.
            archive=output/f'optimizer_steps_before_resume_{time.time_ns()}.csv'
            archive.write_bytes((output/'optimizer_steps.csv').read_bytes())
            with (output/'optimizer_steps.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows[:1500])
        del state
    torch.cuda.reset_peak_memory_stats(device)

    def elapsed():
        return elapsed_before + time.monotonic() - start

    def progress(status, **extra):
        dump(output / 'progress.json', dict(status=status, global_step=step,
             elapsed_seconds=elapsed(), peak_memory_gib=torch.cuda.max_memory_allocated(device)/2**30,
             **extra))

    def record(eval_step):
        value = helper.evaluate(model, prepared, selected, device, 4)
        value.update(global_step=eval_step, layer_rmse=[math.sqrt(x) for x in value['layer_mse']],
                     overall_rmse=math.sqrt(value['mse']), elapsed_seconds=elapsed())
        if evaluations:
            value['relative_decrease_from_previous_eval'] = 1-value['mse']/evaluations[-1]['mse']
        else:
            value['relative_decrease_from_previous_eval'] = None
        evaluations.append(value)
        dump(output / 'evaluations.json', evaluations)
        with (output / 'evaluation_curve.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['global_step','overall_mse','overall_rmse',
                  'last_mse','second_last_mse','third_last_mse','fourth_last_mse',
                  'last_rmse','second_last_rmse','third_last_rmse','fourth_last_rmse',
                  'relative_decrease_from_previous_eval','elapsed_seconds'])
            for item in evaluations:
                writer.writerow([item['global_step'],item['mse'],item['overall_rmse'],
                    *item['layer_mse'],*item['layer_rmse'],item['relative_decrease_from_previous_eval'],
                    item['elapsed_seconds']])
        print(json.dumps(dict(event='evaluation',**value)), flush=True)
        model.train()
        return value

    if not args.resume:
        baseline = record(500)
        old_layers = old_completion['final_layer_mse']
        matches = np.isclose(baseline['layer_mse'],old_layers,atol=1e-7,rtol=1e-5)
        overall_match = bool(np.isclose(baseline['mse'], old_completion['final_mse'], atol=1e-7, rtol=1e-5))
        dump(output/'step500_reproduction.json',dict(passed=bool(matches.all() and overall_match),
              atol=1e-7,rtol=1e-5,old_layer_mse=old_layers,new_layer_mse=baseline['layer_mse'],
              old_overall_mse=old_completion['final_mse'],new_overall_mse=baseline['mse']))
        if not matches.all() or not overall_match:
            raise RuntimeError('Step500 source loss reproduction failed; no training allowed')
        assert len(optimizer.state) == 0
    model.train()
    progress('RUNNING', initialization=protocol['initialization'], resumed=args.resume)

    def next_indices():
        nonlocal offset
        result=[]
        while len(result)<4:
            take=min(4-len(result),len(order)-offset)
            result.extend(order[offset:offset+take]); offset+=take
            if offset==len(order):
                offset=0; sampler_rng.shuffle(order)
        return result

    def save_full_checkpoint(name):
        state=dict(protocol_sha256=protocol_hash,global_step=step,model=model.state_dict(),
                   optimizer=optimizer.state_dict(),evaluations=evaluations,decision=decision,
                   gradient_verification=diagnostics,sampler_order=order.copy(),sampler_offset=offset,
                   sampler_rng=sampler_rng.getstate(),elapsed_seconds=elapsed(),
                   rng=dict(python=random.getstate(),numpy=np.random.get_state(),
                            torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
        path=output/f'{name}.pt'
        atomic_save(path,state)
        # Check actual readability, tensor shapes and optimizer progress before claiming a complete state.
        verified=torch.load(path,map_location='cpu',weights_only=False)
        if verified['global_step']!=step or verified['protocol_sha256']!=protocol_hash:
            raise RuntimeError('Saved full-state checkpoint metadata failed verification')
        if set(verified['model'])!=set(model.state_dict()) or not verified['optimizer']['state']:
            raise RuntimeError('Saved full-state checkpoint lacks model/optimizer tensors')
        if any(verified['model'][key].shape!=value.shape for key,value in model.state_dict().items()):
            raise RuntimeError('Saved full-state model tensor shapes differ')
        for parameter_state in verified['optimizer']['state'].values():
            if int(parameter_state['step'])!=step-500:
                raise RuntimeError('Saved optimizer step does not match actual new updates')
        del verified
        dump(output/f'{name}_manifest.json',dict(global_step=step,
              sha256=sha256(path),bytes=path.stat().st_size,complete_model_optimizer_rng_sampler=True,
              readability_verified=True,protocol_sha256=protocol_hash))

    limit = 502 if args.smoke else (5000 if args.resume else 2000)
    csv_mode='a' if args.resume else 'w'
    with (output/'optimizer_steps.csv').open(csv_mode) as f:
        writer=csv.writer(f)
        if not args.resume:
            writer.writerow(['global_step','new_optimizer_step','mean_microbatch_mse',
                             'gradient_norm_before_clip','elapsed_seconds','peak_memory_gib'])
        while step<limit:
            optimizer.zero_grad(set_to_none=True)
            average_loss=0.0
            for _ in range(4):
                ids,mask,targets=helper.batch(prepared,next_indices(),device)
                predicted=helper.features(model,ids,mask)
                loss=(predicted-targets).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite FP32 training loss')
                (loss/4).backward(); average_loss+=float(loss.detach())/4
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError('Non-finite gradient norm')
            probe=model.encoder.layer[-1].attention.self.query.weight
            before=probe.detach().clone() if diagnostics is None else None
            if diagnostics is None:
                diagnostics=dict(gradient_present=probe.grad is not None,
                     gradient_finite=bool(torch.isfinite(probe.grad).all()),
                     gradient_l2=float(probe.grad.norm()),
                     all_encoder_parameters_trainable=all(p.requires_grad for p in model.parameters()),
                     probe='encoder.layer.23.attention.self.query.weight')
            optimizer.step(); step+=1
            if before is not None:
                diagnostics['maximum_parameter_update']=float((probe.detach()-before).abs().max())
                diagnostics['first_new_optimizer_step']=step-500
                if diagnostics['maximum_parameter_update']<=0:
                    raise RuntimeError('No parameter update after first optimizer step')
                dump(output/'gradient_update_verification.json',diagnostics)
            writer.writerow([step,step-500,average_loss,float(norm),elapsed(),
                             torch.cuda.max_memory_allocated(device)/2**30]); f.flush()
            progress('RUNNING',last_microbatch_average_mse=average_loss)
            if step%25==0 or args.smoke:
                print(json.dumps(dict(event='optimizer_step',global_step=step,
                     new_optimizer_step=step-500,mse=average_loss,gradient_norm=float(norm),
                     elapsed_seconds=elapsed())),flush=True)
            if step in (750,1000,1250,1500,1750,2000,2500,3000,3500,4000,4500,5000) or args.smoke and step==502:
                record(step)
            if step==2000 and not args.smoke:
                by_step={item['global_step']:item for item in evaluations}
                mse1500,mse2000=by_step[1500]['mse'],by_step[2000]['mse']
                relative=(mse1500-mse2000)/mse1500 if mse1500>0 else 0.0
                decision=dict(status='DECIDED',rule=protocol['extension_rule'],threshold=.05,
                     rule_frozen_before_training=True,smoke=False,mse1500=mse1500,mse2000=mse2000,
                     relative_decrease=relative,continue_to_5000=bool(relative>.05),
                     final_authorized_global_step=5000 if relative>.05 else 2000)
                dump(output/'decision.json',decision)
                save_full_checkpoint('stage_2000_checkpoint')
                if decision['continue_to_5000']:
                    limit=5000
                else:
                    break
    final_model_hash=None
    final_checkpoint=None
    if not args.smoke:
        if step==2000:
            final_checkpoint='stage_2000_checkpoint.pt'
        else:
            save_full_checkpoint('final_training_checkpoint')
            final_checkpoint='final_training_checkpoint.pt'
        model.save_pretrained(output/'final_student',safe_serialization=True)
        final_model_hash=sha256(output/'final_student'/'model.safetensors')
    if data_hash!=sha256(data) or source_model_hash!=sha256(source_model/'model.safetensors'):
        raise RuntimeError('An original input changed during this diagnostic')
    completion=dict(status='COMPLETE',global_step=step,new_optimizer_steps=step-500,
         final_layer_mse=evaluations[-1]['layer_mse'],final_mse=evaluations[-1]['mse'],
         final_layer_rmse=evaluations[-1]['layer_rmse'],elapsed_seconds=elapsed(),
         source_model_sha256=source_model_hash,final_model_sha256=final_model_hash,
         final_training_checkpoint=final_checkpoint,
         protocol_sha256=protocol_hash,smoke=args.smoke,decision=decision,
         official_validation_accessed=False,test_accessed=False)
    dump(output/'completion.json',completion)
    progress('COMPLETE',final_mse=completion['final_mse'])


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        if ACTIVE_OUTPUT is not None:
            dump(ACTIVE_OUTPUT/'failure.json',dict(status='FAILED',exception=type(error).__name__,
                 message=str(error),traceback=traceback.format_exc()))
            dump(ACTIVE_OUTPUT/'progress.json',dict(status='FAILED',message=str(error)))
        raise
