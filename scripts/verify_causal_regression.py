#!/usr/bin/env python3
"""Run numeric causal regressions against real CMERC modules, without training."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
# Reuse installed pytest only for importing the parametrized test module.
try:
    import pytest
except ImportError:
    sys.path.append('/home/yangbin/miniconda3/envs/msa_extract/lib/python3.10/site-packages')
from test_followup_causality import (configurations, check_config, check_graph_structure,
                                     check_auxiliary, negative_controls, check_cuda_eval_rnn_backend, SEED, ATOL, RTOL)
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/causal_review_followup')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_device = torch.device(args.device)
    if selected_device.type == 'cuda':
        if not torch.cuda.is_available():
            parser.error('CUDA requested but unavailable')
        index = torch.cuda.current_device() if selected_device.index is None else selected_device.index
        torch.cuda.set_device(index)
        args.device = 'cuda:{}'.format(index)
    elif selected_device.type != 'cpu':
        parser.error('Only CPU/CUDA are supported by this verification driver')
    start = time.monotonic()
    result = dict(seed=SEED, device=args.device, dtype='float32', atol=ATOL, rtol=RTOL,
                  domain='Pre-extracted aligned utterance features; fixed model weights.',
                  synthetic_protocol=dict(input_seed=SEED + 1, dialogue_lengths=[4, 2, 1],
                      text_stream_dims=[1024, 1024, 1024, 1024], audio_dim=6, visual_dim=5,
                      D_g=8, D_h=4, graph_hidden_size=8, D_p=4, D_e=4, D_a=4, n_classes=4, n_speakers=3,
                      speakers='(time // 2 + batch_index) % 3', dropout=0.0,
                      graph_structure_device='cpu', auxiliary_diagnostics_device='cpu'),
                  train_mode_dropout='All nn.Dropout, RNN internal dropout, attention dropout disabled.',
                  configurations=[], failures=[], skipped=[])
    def run(name, function):
        try:
            # Official constructors print branch labels; preserve them in raw log.
            value = function()
            print(json.dumps(dict(name=name, result=value)), flush=True)
            return value
        except Exception as error:
            record = dict(name=name, status='failed', error=str(error), traceback=traceback.format_exc())
            result['failures'].append(record)
            print(json.dumps(record), flush=True)
            return None
    for index, config in enumerate(configurations()):
        value = run('model_{}'.format(index), lambda config=config: check_config(config, args.device))
        if value is not None:
            result['configurations'].append(value)
    result['graph_structure'] = run('graph_structure', check_graph_structure)
    result['auxiliary'] = run('auxiliary_views', check_auxiliary)
    result['negative_controls'] = run('negative_controls', negative_controls)
    if torch.device(args.device).type == 'cuda':
        result['rnn_backend_diagnostics'] = [run('backend_' + base,
            lambda base=base: check_cuda_eval_rnn_backend(base)) for base in ('GRU', 'LSTM')]
    else:
        result['skipped'].append(dict(name='CUDA eval RNN backend diagnostics',
                                     reason='CPU invocation; CUDA-only tests are skipped'))
    result['elapsed_seconds'] = time.monotonic() - start
    result['status'] = 'failed' if result['failures'] else 'passed'
    result['configuration_counts'] = dict(planned=144, passed=len(result['configurations']),
                                          failed=sum(x['name'].startswith('model_') for x in result['failures']))
    target = args.output_dir / 'CAUSALITY_REGRESSION.json'
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    lines = ['# Causal CMERC regression evidence', '',
             f"Status: **{result['status']}**. Real model configurations: {len(result['configurations'])}/144.",
             f"Device `{args.device}`, float32, seed {SEED}, atol {ATOL}, rtol {RTOL}.",
             'Relative differences use denominator max(|reference|, 1e-8); acceptance uses PyTorch combined atol + rtol rule.', '',
             'Each row executes Model / Model_Intra / Model_Inter and their real HyperGCN. '
             'All GRU/LSTM/Transformer × audio/visual encoder × residue × train/eval × IEMOCAP/MELD settings use num_L=2, num_K=3. '
             'Lengths are 4, 2, 1 with repeated and differing speakers. No trained weights or datasets are loaded.', '',
             'Tests compare full log-probability vectors and graph hidden representations for individually perturbed future text/audio/visual/speaker and their joint perturbation. '
             'Every nonempty prefix is inferred separately (suffix addition/deletion), then dialogue reorder, other-dialogue changes, single-dialogue inference, finite padding values, and longer right padding are compared. '
             'Scalar gradient checks sample times 0 and 1, categories 0 and 3 and one hidden coordinate; every tested scalar has exactly zero future feature gradients and nonzero visible-feature influence. '
             'This requirement does not apply to a loss containing future training examples.', '',
             'Executed normalization hooks confirm feature-only LayerNorm; hooks inspect actual causal and padding masks at every executed Transformer. '
             'Unused legacy modules are not inferred to execute. Graph source membership and target delivery are checked separately; incoming-degree prefix invariance is exercised numerically.', '',
             '| Configuration | mode | max absolute difference | max relative difference | future gradient max |',
             '| --- | --- | ---: | ---: | ---: |']
    def diffs(value):
        if isinstance(value, dict):
            if 'max_abs_diff' in value:
                yield value
            for item in value.values():
                yield from diffs(item)
        elif isinstance(value, list):
            for item in value:
                yield from diffs(item)
    for row in result['configurations']:
        config = row['config']
        values = list(diffs(row['checks']))
        label = '/'.join(str(config[x]) for x in ('kind', 'base', 'av_encoder', 'residue', 'dataset'))
        lines.append('| {} | {} | {:.9g} | {:.9g} | {:.9g} |'.format(label,
            'train(dropout=0)' if config['training'] else 'eval',
            max(x['max_abs_diff'] for x in values), max(x['max_rel_diff'] for x in values),
            max(x['future_max'] for x in row['checks']['scalar_gradients'])))
    lines += ['', 'Six independent faulty fixtures (not production mutations) must trigger the same close-comparison detector:']
    for row in result['negative_controls'] or []:
        lines.append('- {}: detected={}, max_abs_diff={:.9g}, {}.'.format(row['fixture'], row['detected'], row['max_abs_diff'], row['intervention']))
    lines += ['', 'GPU nondeterminism: this invocation records only its selected device. A separate GPU invocation must preserve its own output directory. '
              'Normal-dropout training integration and restore validation are separate evidence. '
              'On CUDA, ordinary forward comparisons keep the default cuDNN-enabled backend; '
              'only the diagnostic scalar-gradient forward/backward locally disables cuDNN to use ATen RNN. '
              'Model weights and eval/train mode remain unchanged. Gradient-forward predictions are compared '
              'to ordinary forward predictions with the original unchanged tolerances. '
              'Separate GRU/LSTM fixtures profile actual aten::_cudnn_rnn execution, reproduce the eval-backward '
              'restriction, and verify finite nonzero ATen gradients without switching model mode.', '',
              'Re-run from `CMERC/`:', '', '```bash',
              "CUDA_VISIBLE_DEVICES='' /home/yangbin/miniconda3/envs/multimodalerc310/bin/python scripts/verify_causal_regression.py --device cpu",
              '```', '', 'Detailed per-check numbers and scalar-gradient records: `CAUSALITY_REGRESSION.json`.',
              'Raw output: `logs/causality_cpu.log`.']
    if result['failures']:
        lines += ['', 'Failures:', '```json', json.dumps(result['failures'], indent=2), '```']
    (args.output_dir / 'CAUSALITY_REGRESSION.md').write_text('\n'.join(lines) + '\n')
    aux = result['auxiliary']
    if aux:
        aux_lines = ['# Auxiliary-view core diagnostics', '',
                     f"prefix_mean_v1 formula max_abs_diff={aux['formula_max_abs_diff']:.9g}; inverse max_abs_diff={aux['inverse_max_abs_diff']:.9g}.",
                     '`m_t=(x_1+...+x_t)/t`, `x_t=t*m_t-(t-1)*m_(t-1)` (one-based t). '
                     'The prefix sequence is invertible up to floating point; prefix_mean_v1 is a prefix smoothing view, not complete modality absence. '
                     'Only valid utterances contribute; padding is excluded and output padding is zero.', '',
                     f"{len(aux['first_utterance_probability_noops'])} same-weight eval probability-vector checks pass: 12 backbone/av/residue configurations × three prefix-mean views and two GNN-edge views. "
                     'All use strict state_dict loading. The first utterance has no prior temporal edge and its prefix mean equals itself. '
                     'Agreement compares all four class probabilities, not argmax.', '',
                     f"{len(aux['retained_historical_information'])} Model_Intra/Inter historical-gradient checks pass. "
                     'Graph source/target incidence is identical across auxiliary context views; only same-modality temporal GNN edges are removed according to speaker. '
                     'Both encoder history and hypergraph history remain. These are GNN-edge ablations, not complete context removal.', '',
                     'No strict modality-absence policy is added. Train-function synchronization, dropout and weight lag are reported separately in AUXILIARY_VIEW_DIAGNOSTIC.md / TRAINING_INTEGRATION.md.', '',
                     'Full numeric first-utterance vectors-comparison errors and historical gradient maxima are in CAUSALITY_REGRESSION.json.']
        (args.output_dir / 'AUXILIARY_VIEW_CORE.md').write_text('\n'.join(aux_lines) + '\n')
    print(json.dumps(dict(status=result['status'], configurations=result['configuration_counts'],
                         failures=len(result['failures']), elapsed_seconds=result['elapsed_seconds'], output=str(target))), flush=True)
    return bool(result['failures'])


if __name__ == '__main__':
    raise SystemExit(main())
