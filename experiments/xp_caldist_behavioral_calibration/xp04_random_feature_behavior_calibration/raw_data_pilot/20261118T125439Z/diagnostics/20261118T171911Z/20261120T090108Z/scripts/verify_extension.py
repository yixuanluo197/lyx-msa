#!/usr/bin/env python3
"""CPU-only final verification of the fixed-128 Train-only extension.

This reads only the frozen old-diagnostic allowlist, the cached effective Train,
and new outputs. It does not perform model inference or read source PKLs.
"""
import argparse
import ast
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
from safetensors import safe_open


def read(path):
    return json.loads(Path(path).read_text())


def csvrows(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def finite_nonnegative(values):
    return all(math.isfinite(float(value)) and float(value) >= 0 for value in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument('--allow-pending', action='store_true')
    args = parser.parse_args()
    d = Path(args.directory).resolve()
    old = d.parent
    reports = ['TRAINING_EXTENSION_REPORT.md', 'FINAL_EXTENSION_DECISION.md']
    required = [d/name for name in reports]
    required += [d/'local_only/formal'/ds/'completion.json' for ds in ['IEMOCAP', 'MELD']]
    required += [d/'results/execution.json']
    pending = [str(path.relative_to(d)) for path in required if not path.exists()]
    if not pending and read(d/'results/execution.json').get('status') != 'COMPLETED':
        pending.append('supervisor has not completed')
    if pending:
        if args.allow_pending:
            print(json.dumps({'status': 'PENDING', 'pending': pending}))
            return
        raise RuntimeError(f'Final verification requires completed jobs and reports: {pending}')
    torch.set_num_threads(2)
    checks = []

    def require(name, condition, detail=None):
        checks.append({'name': name, 'pass': bool(condition), 'detail': detail})
        if not condition:
            raise AssertionError(f'{name}: {detail}')

    protocol = read(d/'configs/EXTENSION_PROTOCOL.json')
    lock = read(d/'provenance/launch_code_lock.json')
    require('launch_training_source_unchanged', sha(d/'scripts/extend_overfit.py') == lock['training_script_sha256'])
    require('launch_supervisor_source_unchanged', sha(d/'scripts/run_extension_jobs.py') == lock['supervisor_script_sha256'])
    require('launch_protocol_unchanged', sha(d/'configs/EXTENSION_PROTOCOL.json') == lock['protocol_sha256'])
    require('fixed_strict_stop_rule', protocol['stop_rule']['at_step'] == 2000 and
            protocol['stop_rule']['extend_to_5000_iff_strictly_greater_than'] == .05 and
            protocol['stop_rule']['maximum_total_steps'] == 5000)
    require('optimizer_reset_disclosed', protocol['initialization']['optimizer_state_available'] is False and
            'NOT exact' in protocol['initialization']['mode'])

    original_manifest = read(d/'provenance/original_input_hashes_before.json')
    require('467_original_allowlist_assets', len(original_manifest) == 467)
    original_after = {}
    for filename, expected in original_manifest.items():
        path = Path(filename)
        allowed = (path.is_relative_to(old) and not path.is_relative_to(d)) or path.name == 'train_prepared.pt'
        require('allowlisted_input:' + filename, allowed and path.suffix != '.pkl')
        actual = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        require('original_unchanged:' + filename, actual == expected)
        original_after[filename] = actual

    execution = read(d/'results/execution.json')
    summaries = {}
    expected_stages, expected_layers = [], []
    expected_summary = {}
    for ds in ['IEMOCAP', 'MELD']:
        run = d/'local_only/formal'/ds
        prior = old/'local_only/overfit'/ds
        config = read(run/'run_config.json')
        cfg = config['protocol']
        complete = read(run/'completion.json')
        decision = read(run/'decision.json')
        old_complete = read(prior/'completion.json')
        old_cfg = read(prior/'run_config.json')
        own_protocol_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        require(ds+':run_protocol_hash', own_protocol_hash == config['protocol_sha256'] == complete['protocol_sha256'])
        require(ds+':fixed_hyperparameters', cfg['dataset'] == ds and cfg['sample_count'] == 128 and
                cfg['initial_global_step'] == 500 and cfg['first_stage_end'] == 2000 and cfg['maximum_global_step'] == 5000 and
                cfg['micro_batch'] == cfg['accumulation'] == 4 and cfg['effective_batch'] == 16 and
                cfg['learning_rate'] == 1e-5 and cfg['dropout'] == 0 and cfg['weight_decay'] == 0 and
                cfg['adam_betas'] == [.9,.999] and cfg['adam_eps'] == 1e-8 and cfg['gradient_clip'] == 1 and
                cfg['layer_order'] == [-1,-2,-3,-4] and cfg['precision'].startswith('FP32') and cfg['smoke'] is False)
        require(ds+':run_source_anchors', cfg['script_sha256'] == lock['training_script_sha256'] and
                cfg['helper_sha256'] == sha(old/'scripts/small_sample_overfit.py'))
        require(ds+':new_optimizer_and_sampler_semantics', 'NEW AdamW state from zero' in cfg['initialization'] and 'sampler RNG reset' in cfg['initialization'])
        old_model = prior/'final_student/model.safetensors'
        require(ds+':unchanged_step500_weight_anchor', sha(old_model) == cfg['source_model_sha256'] ==
                old_complete['final_model_sha256'] == complete['source_model_sha256'])
        cache = Path(old_cfg['data'])
        require(ds+':unchanged_train_cache', str(cache) in original_manifest and sha(cache) == cfg['prepared_sha256'] == old_cfg['prepared_sha256'])
        prepared = torch.load(cache, map_location='cpu', weights_only=False)
        fit, dev = set(map(int, prepared['fit_indices'])), set(map(int, prepared['dev_indices']))
        original_selection = csvrows(prior/'selected_fit_manifest.csv')
        selection = csvrows(run/'selected_fit_manifest.csv')
        selected = [int(row['row_index']) for row in selection]
        require(ds+':identical_128_sample_order', len(selection) == len(set(selected)) == 128 and set(selected) <= fit and
                not set(selected)&dev and cfg['selected_manifest_sha256'] == sha(prior/'selected_fit_manifest.csv') and
                all(all(row[key] == new[key] for key in row) for row,new in zip(original_selection,selection)))
        all_texts, all_tokens = set(), set()
        for row,index in zip(selection,selected):
            source = prepared['rows'][index]
            tokens = np.asarray(prepared['input_ids'][index],dtype='<i8')
            targets = prepared['targets'][index].float().contiguous().numpy().astype('<f4',copy=False)
            require(ds+':exact_selected_token_target:' + str(index), row['token_ids_sha256'] == hashlib.sha256(tokens.tobytes()).hexdigest() and
                    row['target_sha256'] == hashlib.sha256(targets.tobytes()).hexdigest() and
                    row['processed_text_sha256'] == source['processed_text_sha256'] and source['inner_subset'] == 'fit')
            all_tokens.add(row['token_ids_sha256']); all_texts.add(row['processed_text_sha256'])
        require(ds+':distinct_text_and_tokens', len(all_tokens) == len(all_texts) == 128)
        selected_targets = prepared['targets'][selected].float().contiguous().numpy().astype('<f4',copy=False)
        require(ds+':exact_combined_target_hash', cfg['selected_targets_sha256'] == hashlib.sha256(selected_targets.tobytes()).hexdigest())
        del prepared, selected_targets
        gc.collect()
        evaluations = read(run/'evaluations.json')
        by_step = {item['global_step']: item for item in evaluations}
        require(ds+':all_mandatory_eval_points', set([500,750,1000,1250,1500,1750,2000]) <= set(by_step))
        for row in evaluations:
            layers = row['layer_mse']
            require(ds+':finite_original_scale_loss:' + str(row['global_step']), row['count'] == 128 and len(layers) == 4 and
                    finite_nonnegative(layers) and close(row['mse'],sum(layers)/4) and
                    all(close(x,math.sqrt(y)) for x,y in zip(row['layer_rmse'],layers)) and close(row['overall_rmse'],math.sqrt(row['mse'])))
        baseline = by_step[500]
        require(ds+':step500_reproduced', np.allclose(baseline['layer_mse'],old_complete['final_layer_mse'],atol=1e-7,rtol=1e-5) and
                np.isclose(baseline['mse'],old_complete['final_mse'],atol=1e-7,rtol=1e-5) and read(run/'step500_reproduction.json')['passed'])
        relative = (by_step[1500]['mse']-by_step[2000]['mse'])/by_step[1500]['mse'] if by_step[1500]['mse']>0 else 0.0
        should_extend = relative > .05
        final_step = 5000 if should_extend else 2000
        require(ds+':strict_5_percent_decision', decision['status'] == 'DECIDED' and decision['threshold'] == .05 and
                decision['rule_frozen_before_training'] and close(decision['relative_decrease'],relative) and
                close(decision['mse1500'],by_step[1500]['mse']) and close(decision['mse2000'],by_step[2000]['mse']) and
                decision['continue_to_5000'] == should_extend and decision['final_authorized_global_step'] == final_step)
        expected_evals = [500,750,1000,1250,1500,1750,2000] + ([2500,3000,3500,4000,4500,5000] if should_extend else [])
        require(ds+':no_missing_or_extra_evaluations', [row['global_step'] for row in evaluations] == expected_evals)
        for j,row in enumerate(evaluations):
            previous = evaluations[j-1] if j else None
            expected_stages.append({'dataset':ds,'global_step':row['global_step'],
                'overall_mse':row['mse'],'overall_rmse':math.sqrt(row['mse']),
                'previous_global_step':previous['global_step'] if previous else '',
                'relative_decrease_from_previous':1-row['mse']/previous['mse'] if previous else '',
                'relative_decrease_from_step500':1-row['mse']/baseline['mse']})
            for i,value in enumerate(row['layer_mse']):
                expected_layers.append({'dataset':ds,'global_step':row['global_step'],'layer_from_end':i+1,
                    'mse':value,'rmse':math.sqrt(value),
                    'relative_decrease_from_previous':1-value/previous['layer_mse'][i] if previous else '',
                    'relative_decrease_from_step500':1-value/baseline['layer_mse'][i],
                    'rmse_limit':1e-4,'rmse_ratio_to_limit':math.sqrt(value)/1e-4,
                    'rmse_necessary_condition_pass':math.sqrt(value)<=1e-4})
        rows = csvrows(run/'optimizer_steps.csv')
        require(ds+':every_actual_optimizer_step_retained', [int(r['global_step']) for r in rows] == list(range(501,final_step+1)) and
                [int(r['new_optimizer_step']) for r in rows] == list(range(1,final_step-499)))
        require(ds+':all_step_losses_and_gradients_finite', all(finite_nonnegative([row['mean_microbatch_mse'],row['gradient_norm_before_clip']]) for row in rows))
        require(ds+':completion_matches_decision_and_losses', complete['status'] == 'COMPLETE' and complete['global_step'] == final_step and
                complete['new_optimizer_steps'] == final_step-500 and complete['decision'] == decision and
                close(complete['final_mse'],by_step[final_step]['mse']) and
                all(close(x,y) for x,y in zip(complete['final_layer_mse'],by_step[final_step]['layer_mse'])) and
                complete['official_validation_accessed'] is False and complete['test_accessed'] is False)
        grad = read(run/'gradient_update_verification.json')
        require(ds+':nonzero_gradient_update', grad['all_encoder_parameters_trainable'] and grad['gradient_present'] and grad['gradient_finite'] and
                grad['gradient_l2']>0 and grad['maximum_parameter_update']>0 and grad['first_new_optimizer_step'] == 1)
        final_weights = run/'final_student/model.safetensors'
        require(ds+':final_model_weight_hash', sha(final_weights) == complete['final_model_sha256'])
        checkpoint_names = ['stage_2000_checkpoint'] + (['final_training_checkpoint'] if should_extend else [])
        for name in checkpoint_names:
            path = run/(name+'.pt')
            metadata = read(run/(name+'_manifest.json'))
            step = 2000 if name == 'stage_2000_checkpoint' else 5000
            require(ds+':full_checkpoint_hash:' + name, sha(path) == metadata['sha256'] and path.stat().st_size == metadata['bytes'] and
                    metadata['global_step'] == step and metadata['complete_model_optimizer_rng_sampler'] and metadata['readability_verified'] and
                    metadata['protocol_sha256'] == own_protocol_hash)
            state = torch.load(path,map_location='cpu',weights_only=False)
            require(ds+':full_checkpoint_metadata:' + name, state['global_step'] == step and state['protocol_sha256'] == own_protocol_hash and
                    set(state['rng']) == {'python','numpy','torch','cuda'} and state['sampler_rng'] is not None and
                    len(state['sampler_order']) == 128 and set(state['sampler_order']) == set(selected) and 0 <= state['sampler_offset'] < 128)
            require(ds+':FP32_finite_backbone:' + name, bool(state['model']) and all(t.dtype == torch.float32 and torch.isfinite(t).all().item() for t in state['model'].values()))
            optimizer = state['optimizer']
            require(ds+':new_AdamW_step_and_FP32_moments:' + name, bool(optimizer['state']) and
                    all(int(item['step']) == step-500 and item['exp_avg'].dtype == item['exp_avg_sq'].dtype == torch.float32 and
                        torch.isfinite(item['exp_avg']).all().item() and torch.isfinite(item['exp_avg_sq']).all().item() for item in optimizer['state'].values()) and
                    all(group['lr'] == 1e-5 and group['weight_decay'] == 0 and list(group['betas']) == [.9,.999] and group['eps'] == 1e-8 for group in optimizer['param_groups']))
            if step == final_step:
                with safe_open(str(final_weights),framework='pt',device='cpu') as weights:
                    require(ds+':final_saved_weights_equal_complete_state', set(weights.keys()) == set(state['model']) and
                            all(torch.equal(weights.get_tensor(key),value) for key,value in state['model'].items()))
                require(ds+':final_checkpoint_reference', complete['final_training_checkpoint'] == path.name)
            del state,optimizer
            gc.collect()
        for phase, expected_step in [('smoke',502),('formal',final_step)]:
            record = execution['phases'][phase][ds]
            phase_complete = read(d/'local_only'/phase/ds/'completion.json')
            require(ds+':exit_zero_and_expected_steps:' + phase, record['status'] == 'COMPLETED' and record['exit_code'] == 0 and
                    phase_complete['global_step'] == expected_step and Path(record['log']).is_file() and Path(record['log']).stat().st_size>0 and
                    record['gpu_uuid'] == lock['gpu_uuids'][record['gpu']])
        summaries[ds] = {'final_global_step':final_step,'new_optimizer_steps':final_step-500,
                         'mse1500_to_2000_relative_decrease':relative,'continue_to_5000':should_extend,
                         'final_mse':by_step[final_step]['mse'],'final_layer_rmse':by_step[final_step]['layer_rmse'],
                         'all_four_layers_meet_RMSE_necessary_condition':all(value<=1e-4 for value in by_step[final_step]['layer_rmse']),
                         'final_model_sha256':complete['final_model_sha256']}
        final = by_step[final_step]
        reduction = 1-final['mse']/by_step[final_step-500]['mse']
        expected_summary[ds] = {'final_global_step':final_step,'initial_mse':baseline['mse'],'final_mse':final['mse'],
            'final_overall_rmse':math.sqrt(final['mse']),'relative_reduction_from_step500':1-final['mse']/baseline['mse'],
            'final_window_start':final_step-500,'final_window_relative_decrease':reduction,
            'final_window_plateau_by_5percent_rule':reduction<=.05,'final_layer_mse':final['layer_mse'],
            'final_layer_rmse':final['layer_rmse'],'rmse_necessary_condition_pass':all(x<=1e-4 for x in final['layer_rmse']),
            'decision':decision,'optimizer_reset':True}

    for filename,expected_rows in [('stage_comparison.csv',expected_stages),('per_layer_comparison.csv',expected_layers)]:
        actual_rows = csvrows(d/'results'/filename)
        require('analysis_csv_complete:'+filename,len(actual_rows)==len(expected_rows))
        for i,(actual,expected) in enumerate(zip(actual_rows,expected_rows)):
            for key,value in expected.items():
                matching = actual[key] == str(value) if isinstance(value,(str,bool)) else close(actual[key],value)
                require('analysis_csv_value:'+filename+':'+str(i)+':'+key,matching)
    analysis_summary = read(d/'results/extension_summary.json')
    for ds,expected in expected_summary.items():
        for key,value in expected.items():
            actual = analysis_summary[ds][key]
            if isinstance(value,list):
                matching = len(actual)==len(value) and all(close(a,b) for a,b in zip(actual,value))
            elif isinstance(value,(dict,bool)):
                matching = actual == value
            else:
                matching = close(actual,value)
            require('analysis_summary_value:'+ds+':'+key,matching)
    for extension in ('png','pdf'):
        figure = d/'results'/('training_extension.'+extension)
        require('requested_loss_figure:'+extension,figure.is_file() and figure.stat().st_size>1000)

    artifacts = {}
    for path in sorted((d/'scripts').glob('*.py')):
        ast.parse(path.read_text(),filename=str(path))
        artifacts[str(path.relative_to(d))] = sha(path)
    for folder in ['configs','results']:
        for path in sorted((d/folder).glob('*')):
            if path.is_file():
                artifacts[str(path.relative_to(d))] = sha(path)
                if path.suffix == '.json':
                    read(path)
                elif path.suffix == '.csv':
                    require('nonempty_result_csv:'+path.name,bool(csvrows(path)))
    for filename in reports:
        artifacts[filename] = sha(d/filename)
    outcome = {'status':'PASS','delivery_complete':True,'verified_at':datetime.now(timezone.utc).isoformat(),
               'checks_passed':len(checks),'checks':checks,'dataset_summary':summaries,
               'original_input_file_count':len(original_after),'all_original_allowed_files_unchanged':True,
               'original_input_hashes_after':original_after,'artifacts_sha256':artifacts,
               'official_validation_accessed':False,'test_accessed':False,'original_pkl_opened':False,
               'new_model_inference_by_verifier':False,'gpu_used_by_verifier':False,
               'interpretation':'Delivery verification PASS is not full feature compatibility PASS; this is fixed128 Fit-only evidence.',
               'optimizer_semantics':'Old500 model weights continued with newly reset AdamW/sampler; saved new states contain only global_step minus500 updates.',
               'limitations':['Data access checked from frozen code/protocol and artifact boundaries, not operating-system file-access tracing.',
                              'RMSE<=1e-4 per layer is only a necessary condition, not full-dataset feature or XP04 probability parity.']}
    (d/'verification.json').write_text(json.dumps(outcome,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','checks_passed':len(checks),'original_files_unchanged':len(original_after),'datasets':summaries}))


if __name__ == '__main__':
    main()
