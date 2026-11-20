#!/usr/bin/env python3
"""Independent, CPU-only delivery checks; never opens original PKLs or held-out data."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
from safetensors import safe_open


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def csvrows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def finite_nonnegative(values):
    return all(math.isfinite(float(x)) and float(x) >= 0 for x in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-dir', default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument('--allow-pending', action='store_true')
    args = parser.parse_args()
    d = Path(args.diagnostic_dir).resolve()
    p = d.parent.parent
    reports = ['SMALL_SAMPLE_OVERFIT.md', 'FRAMEWORK_PARITY.md', 'TRAIN_DEV_GAP.md', 'DIAGNOSTIC_SUMMARY.md']
    needed = [d / name for name in reports]
    needed += [d / 'local_only/overfit' / ds / 'completion.json' for ds in ('IEMOCAP', 'MELD')]
    pending = [str(path.relative_to(d)) for path in needed if not path.exists()]
    if pending:
        if args.allow_pending:
            # Do not create a misleading final verification.json while experiments are active.
            print(json.dumps({'status': 'PENDING', 'missing': pending}))
            return
        raise RuntimeError(f'Final verification requires completed runs and all reports: {pending}')
    checks = []

    def require(name, condition, detail=None):
        checks.append({'name': name, 'pass': bool(condition), 'detail': detail})
        if not condition:
            raise AssertionError(f'{name}: {detail}')

    before_path = d / 'provenance/input_hashes_before.json'
    before = read(before_path)
    require('original_input_manifest_has_26_files', len(before) == 26)
    original_after = {}
    for filename, item in before.items():
        path = Path(filename)
        # The fixed allowlist includes only cached Train, saved student/config files,
        # and the public pretrained snapshot; never deserialize an official package.
        allowed = (path.name == 'train_prepared.pt' or
                   ('/training/' in str(path) and str(path).startswith(str(p / 'local_only'))) or
                   '/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59/' in str(path))
        require('original_input_allowlisted:' + path.name, allowed and path.suffix != '.pkl', str(path))
        actual = {'sha256': sha(path), 'bytes': path.stat().st_size}
        require('original_input_unchanged:' + filename, actual == item)
        original_after[filename] = actual
    prepared = {}
    overfit_summary = {}
    evaluation_steps = [0, 25, 50, 100, 200, 300, 400, 500]
    public_weights = None
    for ds, total, fit_count, dev_count in [('IEMOCAP', 5283, 4747, 536), ('MELD', 9989, 8989, 1000)]:
        cache = p / 'local_only' / ds / 'train_prepared.pt'
        data = torch.load(cache, map_location='cpu', weights_only=False)
        prepared[ds] = data
        fit, dev = set(map(int, data['fit_indices'])), set(map(int, data['dev_indices']))
        require(ds + ':original_train_partition', len(fit) == fit_count and len(dev) == dev_count and
                not fit & dev and fit | dev == set(range(total)))
        fit_dialogues = {str(data['rows'][i]['dialogue_id']) for i in fit}
        dev_dialogues = {str(data['rows'][i]['dialogue_id']) for i in dev}
        require(ds + ':dialogue_groups_disjoint', not fit_dialogues & dev_dialogues)
        lock = read(p / 'local_only' / ds / 'training/best_model_lock.json')
        require(ds + ':original_best_epoch8', lock['best_epoch'] == 8 and lock['training_complete'])
        run = d / 'local_only/overfit' / ds
        config, completion = read(run / 'run_config.json'), read(run / 'completion.json')
        grad = read(run / 'gradient_update_verification.json')
        require(ds + ':fixed_overfit_protocol', config['count'] == 128 and config['steps'] == 500 and
                config['learning_rate'] == 1e-5 and config['effective_batch_size'] == 16 and
                config['dropout'] == {'hidden': 0.0, 'attention': 0.0} and config['weight_decay'] == 0 and
                config['target_normalization'] is None and config['precision'].startswith('FP32') and
                config['layer_order'] == [-1, -2, -3, -4] and config['evaluation_steps'] == evaluation_steps)
        require(ds + ':overfit_source_frozen', config['script_sha256'] == sha(d / 'scripts/small_sample_overfit.py'))
        require(ds + ':overfit_train_input_frozen', config['prepared_sha256'] == before[str(cache)]['sha256'])
        public_weights = Path(config['pretrained']) / 'model.safetensors'
        require(ds + ':same_public_initializer', config['public_weight_sha256'] == before[str(public_weights)]['sha256'])
        require(ds + ':all_backbone_parameters_trainable', config['encoder_parameters'] == config['trainable_parameters'] and
                config['encoder_parameters'] > 350000000 and grad['all_encoder_parameters_trainable'])
        require(ds + ':nonzero_finite_gradient_and_parameter_update', grad['gradient_present'] and grad['gradient_all_finite'] and
                math.isfinite(grad['gradient_l2']) and grad['gradient_l2'] > 0 and grad['maximum_parameter_update'] > 0)
        selection = csvrows(run / 'selected_fit_manifest.csv')
        indices = [int(row['row_index']) for row in selection]
        require(ds + ':128_distinct_fit_indices', len(indices) == len(set(indices)) == 128 and set(indices) <= fit)
        texts, tokens = set(), set()
        for row, index in zip(selection, indices):
            source = data['rows'][index]
            token_values = tuple(map(int, data['input_ids'][index]))
            token_hash = hashlib.sha256(np.asarray(token_values, dtype='<i8').tobytes()).hexdigest()
            require(ds + ':selected_row_alignment:' + str(index), row['dataset'] == ds and
                    row['processed_text_sha256'] == source['processed_text_sha256'] and row['token_ids_sha256'] == token_hash and
                    source['inner_subset'] == 'fit' and str(row['dialogue_id']) == str(source['dialogue_id']))
            texts.add(source['processed_text_sha256'])
            tokens.add(token_values)
        require(ds + ':128_unique_texts_and_token_sequences', len(texts) == len(tokens) == 128)
        steps = csvrows(run / 'optimizer_steps.csv')
        require(ds + ':all_500_optimizer_steps_retained', [int(row['step']) for row in steps] == list(range(1, 501)))
        require(ds + ':optimizer_losses_and_gradients_finite', all(finite_nonnegative([
            row['mean_microbatch_mse'], row['gradient_norm_before_clip']]) for row in steps))
        evaluations = read(run / 'evaluations.json')
        require(ds + ':all_evaluation_checkpoints_retained', [r['step'] for r in evaluations] == evaluation_steps)
        for item in evaluations:
            require(ds + ':four_layer_eval_loss:' + str(item['step']), item['count'] == 128 and len(item['layer_mse']) == 4 and
                    finite_nonnegative(item['layer_mse']) and close(item['mse'], sum(item['layer_mse']) / 4))
        require(ds + ':actual_500_step_completion', completion['status'] == 'complete' and completion['actual_optimizer_steps'] == 500 and
                completion['official_validation_accessed'] is False and completion['test_accessed'] is False)
        require(ds + ':completion_losses_consistent', close(completion['initial_mse'], evaluations[0]['mse']) and
                close(completion['final_mse'], evaluations[-1]['mse']) and
                all(close(a, b) for a, b in zip(completion['final_layer_mse'], evaluations[-1]['layer_mse'])) and
                close(completion['reduction_fraction'], 1-completion['final_mse']/completion['initial_mse']))
        require(ds + ':diagnostic_final_model_saved_and_hashed', sha(run / 'final_student/model.safetensors') == completion['final_model_sha256'])
        overfit_summary[ds] = {key: completion[key] for key in ('actual_optimizer_steps', 'initial_mse', 'final_mse', 'reduction_fraction', 'final_layer_mse', 'final_model_sha256')}

        gap = read(d / 'results' / f'train_dev_gap_{ds}.json')
        require(ds + ':gap_complete_fp32_train_only', gap['status'] == 'COMPLETE' and gap['source_inputs_unchanged'] and
                gap['launch']['official_validation_accessed'] is False and gap['launch']['test_accessed'] is False and
                gap['launch']['model_precision'] == 'FP32' and gap['launch']['autocast_enabled'] is False)
        require(ds + ':gap_source_frozen', gap['launch']['script_sha256'] == sha(d / 'scripts/evaluate_train_dev_gap.py'))
        require(ds + ':gap_exact_four_model_subset_results', len(gap['results']) == 4 and
                {(r['model'], r['subset']) for r in gap['results']} ==
                {(m, s) for m in ['public_initial', 'best_epoch8'] for s in ['fit', 'inner_dev']})
        for row in gap['results']:
            layers = [row[f'layer{i}_mse'] for i in range(1, 5)]
            require(ds + ':gap_layer_values:' + row['model'] + ':' + row['subset'], row['n'] == {'fit': fit_count, 'inner_dev': dev_count}[row['subset']] and
                    finite_nonnegative(layers) and close(row['overall_mse'], sum(layers)/4) and row['precision'] == 'FP32' and row['mode'] == 'eval')
        for item in gap['gaps']:
            f, v = [next(r for r in gap['results'] if r['model'] == item['model'] and r['subset'] == s) for s in ['fit', 'inner_dev']]
            require(ds + ':gap_arithmetic:' + item['model'], close(item['fit_mse'], f['overall_mse']) and close(item['inner_dev_mse'], v['overall_mse']) and
                    close(item['dev_minus_fit_mse'], v['overall_mse']-f['overall_mse']) and close(item['dev_to_fit_mse_ratio'], v['overall_mse']/f['overall_mse']))

    framework = read(d / 'results/framework_summary.json')
    protocol = read(d / 'provenance/framework_protocol.json')
    mapping = read(d / 'provenance/framework_weight_mapping.json')
    require('framework_honest_status', framework['status'] in ('PASS', 'FAIL', 'BLOCKED'))
    if framework['status'] == 'BLOCKED':
        require('framework_block_has_reason', bool(framework.get('reason') or framework.get('blocker')))
    else:
        require('framework_same_actual_weights_proven', framework['actual_same_weights_verified'] and mapping['all_pairs_bit_exact'] and
                mapping['complete_fairseq_backbone_coverage'] and mapping['complete_hf_backbone_coverage'] and
                mapping['mapped_backbone_tensors'] == len(mapping['tensors']) == 388 and mapping['hf_only_zero_token_type_embedding'])
        require('framework_fixed_fp32_eval_protocol', protocol['dtype'] == 'float32' and protocol['mode'] == 'eval' and
                protocol['layers'] == [-1, -2, -3, -4] and protocol['pretrained_sha256'] == before[str(public_weights)]['sha256'])
        with safe_open(str(public_weights), framework='pt', device='cpu') as source:
            for pair in mapping['tensors']:
                tensor = source.get_tensor('roberta.' + pair['hf_key'])
                tensor_hash = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                require('framework_mapping_original_tensor:' + pair['hf_key'], pair['bit_exact'] and pair['dtype'] == 'torch.float32' and
                        list(tensor.shape) == pair['shape'] and tensor_hash == pair['tensor_sha256'])
            token_type = source.get_tensor('roberta.embeddings.token_type_embeddings.weight')
            require('framework_hf_only_embedding_really_zero', tuple(token_type.shape) == (1, 1024) and torch.count_nonzero(token_type).item() == 0)
        require('framework_protocol_mapping_source_hashes', framework['protocol_sha256'] == sha(d / 'provenance/framework_protocol.json') and
                framework['weight_mapping_sha256'] == sha(d / 'provenance/framework_weight_mapping.json') and
                framework['script_sha256'] == sha(d / 'scripts/framework_parity.py'))
        vendor = d / 'local_only/framework_vendor/fairseq-0.9.0'
        for name, expected in protocol['source_files'].items():
            require('native_fairseq_source_unchanged:' + name, sha(vendor / name) == expected)
        manifest = csvrows(d / 'results/framework_sample_manifest.csv')
        arrays = np.load(d / 'local_only/framework_outputs.npz')
        require('framework_all_eight_comparisons_retained', len(framework['metrics']) == 8 and len(manifest) == 64)
        for ds in ('IEMOCAP', 'MELD'):
            rows = [row for row in manifest if row['dataset'] == ds]
            fit = set(map(int, prepared[ds]['fit_indices']))
            require(ds + ':framework_32_fit_samples', len(rows) == 32 and {int(r['row_index']) for r in rows} <= fit)
            for row in rows:
                ids = tuple(map(int, prepared[ds]['input_ids'][int(row['row_index'])]))
                require(ds + ':framework_identical_cached_token_hash:' + row['row_index'],
                        row['token_ids_sha256'] == hashlib.sha256(json.dumps(ids).encode()).hexdigest())
            a, b = [arrays[ds + '_' + name].astype(np.float64) for name in ['huggingface', 'fairseq']]
            require(ds + ':framework_saved_output_shapes', a.shape == b.shape == (32, 4, 1024) and np.isfinite(a).all() and np.isfinite(b).all())
            for layer in range(4):
                r = next(row for row in framework['metrics'] if row['dataset'] == ds and row['layer_from_end'] == layer+1)
                delta = a[:, layer] - b[:, layer]
                cosine = (a[:, layer]*b[:, layer]).sum(1)/(np.linalg.norm(a[:, layer], axis=1)*np.linalg.norm(b[:, layer], axis=1))
                calculated = {'mae': np.abs(delta).mean(), 'rmse': np.sqrt((delta**2).mean()),
                              'max_abs_error': np.abs(delta).max(), 'mean_cosine': cosine.mean(), 'min_cosine': cosine.min()}
                require(ds + ':framework_metrics_recomputed:' + str(layer+1), r['samples'] == 32 and
                        all(close(r[key], value) for key, value in calculated.items()) and
                        r['allclose'] == bool(np.allclose(a[:, layer], b[:, layer], atol=1e-4, rtol=1e-4)))
        require('framework_scope_train_only', framework['official_validation_accessed'] is False and
                framework['test_accessed'] is False and framework['original_pkl_opened'] is False)

    artifact_hashes = {}
    for path in sorted((d / 'scripts').glob('*.py')):
        ast.parse(path.read_text(), filename=str(path))
        artifact_hashes[str(path.relative_to(d))] = sha(path)
    require('all_diagnostic_python_sources_parse', bool(artifact_hashes))
    for path in needed:
        artifact_hashes[str(path.relative_to(d))] = sha(path)
    for path in sorted((d / 'results').iterdir()):
        if path.is_file():
            artifact_hashes[str(path.relative_to(d))] = sha(path)
    for path in sorted((d / 'configs').glob('*.json')):
        read(path)
        artifact_hashes[str(path.relative_to(d))] = sha(path)
    outcome = {'status': 'PASS', 'delivery_complete': True, 'verified_at': datetime.now(timezone.utc).isoformat(),
               'checks_passed': len(checks), 'checks': checks, 'overfit_summary': overfit_summary,
               'original_input_file_count': len(original_after), 'all_original_allowed_inputs_unchanged': True,
               'original_input_hashes_after': original_after, 'artifact_hashes': artifact_hashes,
               'official_validation_accessed_by_verifier': False, 'test_accessed_by_verifier': False,
               'original_pkl_opened_by_verifier': False, 'gpu_inference_by_verifier': False,
               'research_interpretation': 'Delivery PASS is not feature compatibility PASS; bounded Train-only diagnostics do not authorize text perturbation or a new full run.',
               'limits': ['Runtime access claims are checked from code and recorded protocol; this is not an operating-system file-access trace.',
                          'Probe gradient confirms an update, not that every parameter has a nonzero gradient.',
                          'Native Fairseq v0.9.0 comparison does not establish the unknown historical COSMIC runtime version.']}
    serialized = json.dumps(outcome, indent=2, allow_nan=False) + '\n'
    (d / 'verification.json').write_text(serialized)
    (d / 'provenance/independent_diagnostic_review.json').write_text(serialized)
    print(json.dumps({'status': 'PASS', 'checks_passed': len(checks), 'original_inputs_unchanged': len(original_after), 'delivery_complete': True}))


if __name__ == '__main__':
    main()
