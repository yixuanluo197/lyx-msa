#!/usr/bin/env python
"""Bounded real-function synthetic audit. Never launches a dataset experiment."""
import argparse
import copy
import json
import os
import sys
import tempfile
import traceback

import numpy as np
from pathlib import Path
from types import SimpleNamespace

# Must be set before CUDA context initialization; audit-local, never training default.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_cm as training
from causal_run import prepare_run
from model_cm import Model, Model_Intra, Model_Inter


def arguments(**changes):
    values = dict(causal=True, norm='LN', Dataset='IEMOCAP', base_model='GRU',
        num_L=2, num_K=2, av_using_lstm=True, use_residue=False, seed=25,
        split_seed=12, dropout=.15, rec_dropout=0., batch_size=2, lr=.001,
        l2=0., calibrate=True, contrastlearning=True, rank_coff=.005,
        mscl_coff=.05, cscl_coff=.05, courselearning=False, epoch_ratio=.5,
        scheduler_steps=1, selection_protocol='validation', multi_modal=True,
        mm_fusion_mthd='concat_DHT', modals='avl', tensorboard=False,
        auxiliary_policy='legacy_epoch_snapshot', replacement_policy='prefix_mean_v1')
    values.update(changes)
    return SimpleNamespace(**values)


def build_models(device='cpu', dropout=.15):
    return tuple(kind('GRU', 1024, 8, 4, 4, 4, 4, 8, n_speakers=3,
        max_seq_len=20, window_past=10, window_future=10, n_classes=4,
        dropout=dropout, no_cuda=torch.device(device).type == 'cpu', graph_type='hyper',
        use_residue=False, D_m_v=5, D_m_a=6, modals='avl', att_type='concat_DHT',
        av_using_lstm=True, dataset='IEMOCAP', use_modal=True, norm='LN',
        num_L=2, num_K=2, causal=True).to(device)
        for kind in (Model, Model, Model_Intra, Model_Inter))


def synthetic_batches(device='cpu', count=2):
    result = []
    for i in range(count):
        text = [torch.randn(3, 2, 1024, device=device) for _ in range(4)]
        visual = torch.randn(3, 2, 5, device=device)
        audio = torch.randn(3, 2, 6, device=device)
        speaker = torch.zeros(3, 2, 3, device=device)
        speaker[torch.arange(3), 0, torch.arange(3)] = 1
        speaker[:2, 1, :2] = torch.eye(2, device=device)
        mask = torch.tensor([[1.,1.,1.], [1.,1.,0.]], device=device)
        label = torch.tensor([[0,1,2], [1,3,0]], device=device)
        result.append([*text, visual, audio, speaker, mask, label,
                       [f'synthetic-{i}-a', f'synthetic-{i}-b']])
    return result


def run_epoch(models, optimizer, batches, epoch, device, records):
    return training.train_or_eval_graph_model(*models, torch.nn.NLLLoss(),
        batches, epoch, torch.device(device).type == 'cuda', 'avl',
        optimizer, True, return_components=True, diagnostic_callback=records.append)[-1]


def numerical_comparison(a, b, atol=2e-5, rtol=1e-5):
    """Report first, then decide. Discrete labels/modes/keys must match exactly."""
    maximum_absolute, maximum_relative, within_tolerance, semantic_equal = 0., 0., True, True
    def visit(left, right):
        nonlocal maximum_absolute, maximum_relative, within_tolerance, semantic_equal
        if isinstance(left, torch.Tensor):
            left, right = left.detach().cpu(), right.detach().cpu()
            if left.shape != right.shape:
                semantic_equal = False; return
            if left.is_floating_point():
                difference = (left-right).abs()
                maximum_absolute = max(maximum_absolute, float(difference.max()) if difference.numel() else 0.)
                relative = difference / right.abs().clamp_min(torch.finfo(right.dtype).tiny)
                maximum_relative = max(maximum_relative, float(relative.max()) if relative.numel() else 0.)
                within_tolerance = within_tolerance and bool(torch.allclose(left,right,atol=atol,rtol=rtol))
            else:
                semantic_equal = semantic_equal and torch.equal(left,right)
        elif isinstance(left, dict):
            if left.keys() != right.keys():
                semantic_equal=False; return
            for key in left: visit(left[key],right[key])
        elif isinstance(left, (tuple,list)):
            if len(left) != len(right):
                semantic_equal=False; return
            for x,y in zip(left,right): visit(x,y)
        elif isinstance(left, float):
            difference = abs(left-right)
            maximum_absolute=max(maximum_absolute,difference)
            maximum_relative=max(maximum_relative,difference/max(abs(right),1e-300))
            within_tolerance = within_tolerance and difference <= atol+rtol*abs(right)
        else:
            semantic_equal=semantic_equal and left == right
    visit(a,b)
    return {'max_abs_diff':maximum_absolute,'max_rel_diff':maximum_relative,
            'atol':atol,'rtol':rtol,'within_tolerance':within_tolerance and semantic_equal,
            'semantic_equal':semantic_equal,'exact':maximum_absolute == 0. and semantic_equal}


def recovery_check(device, batch_data, root):
    # Curriculum scheduling is exercised with a legal *synthetic fixture*;
    # it is never evidence for a trained teacher or a formal CL input.
    from causal_provenance import (build_data_provenance, architecture_from_args,
        create_difficulty_manifest, write_causal_difficulty, resolve_causal_difficulty)
    feature = root / 'synthetic_features'; feature.write_bytes(b'synthetic data fixture')
    data = build_data_provenance('IEMOCAP', ['synthetic-0-a','synthetic-0-b','synthetic-1-a','synthetic-1-b'],
        ['synthetic-valid'], [feature], split_seed=12)
    args = arguments(courselearning=True, data_provenance=data)
    checkpoint_fixture = root / 'synthetic_teacher_fixture'; checkpoint_fixture.write_bytes(b'NOT TRAINED; SYNTHETIC FLOW FIXTURE')
    teacher = {'args': vars(args).copy(), 'epoch': 0, 'best_epoch': 0}
    manifest = create_difficulty_manifest(checkpoint_fixture, teacher, data, architecture_from_args(args))
    path = write_causal_difficulty({'synthetic-0-a': .1, 'synthetic-0-b': .2, 'synthetic-1-a': .7, 'synthetic-1-b': .8}, manifest,
                                    output_dir=root)
    difficulty, artifact = resolve_causal_difficulty(path, 'IEMOCAP', data, architecture_from_args(args))
    args.difficulty_artifact = artifact
    args.difficulty_path = path
    args.output_dir = str(root / 'continuous')
    args.resume = None
    training.args = args
    prepare_run(args)
    training.seed_everything(args.seed)
    continuous = build_models(device, args.dropout)
    continuous_opt = torch.optim.Adam(continuous[0].parameters(), lr=args.lr)
    continuous_records, continuous_log = [], []
    selected_ids = []
    for epoch in range(2):
        ratio = min(1., round(args.epoch_ratio * int(epoch / args.scheduler_steps + 1), 2))
        # Use real curriculum key selection then feed its selected batches into
        # the production loss/label function; each epoch still has >=2 batches.
        from dataloader import select_curriculum_keys
        ids = select_curriculum_keys(list(difficulty), difficulty, ratio)
        selected_ids.append(ids)
        epoch_batches = [batch_data[index] for index in dict.fromkeys(int(k.split('-')[1]) for k in ids) for _ in range(2)]
        continuous_log.append(run_epoch(continuous, continuous_opt, epoch_batches, epoch, device, continuous_records))
    continuous_state = copy.deepcopy(continuous[0].state_dict())
    continuous_optimizer = copy.deepcopy(continuous_opt.state_dict())
    next_records = []
    next_loss = run_epoch(continuous, continuous_opt, batch_data, 2, device, next_records)

    args = copy.deepcopy(args)
    args.output_dir = str(root / 'interrupted')
    args.run_dir = ''
    args.run_manifest = None
    training.args = args
    run = prepare_run(args)
    training.seed_everything(args.seed)
    resumed = build_models(device, args.dropout)
    resumed_opt = torch.optim.Adam(resumed[0].parameters(), lr=args.lr)
    resumed_records, resumed_log = [], []
    epoch0_batches = [batch_data[index] for index in dict.fromkeys(int(k.split('-')[1]) for k in selected_ids[0]) for _ in range(2)]
    resumed_log.append(run_epoch(resumed, resumed_opt, epoch0_batches, 0, device, resumed_records))
    training.save_training_checkpoint(run/'best.pt', resumed[0], resumed_opt, 0, args, .7, 0)
    training.save_training_checkpoint(run/'latest.pt', resumed[0], resumed_opt, 0, args, .7, 0)
    training.write_jsonl_record(run/'metrics.jsonl', {'epoch':1, **resumed_log[0]})
    checkpoint = torch.load(run/'latest.pt', map_location='cpu')
    # Constructing a new process/model consumes RNG. Restore must undo that.
    resumed = build_models(device, args.dropout)
    resumed_opt = torch.optim.Adam(resumed[0].parameters(), lr=args.lr)
    args.resume = str(run/'latest.pt')
    assert prepare_run(args) == run
    metadata = training.load_training_checkpoint(args.resume, resumed[0], resumed_opt,
                                                 'cpu', args)
    assert metadata['start_epoch'] == 1
    assert metadata['curriculum']['next_epoch_ratio'] == 1.
    restored_rng, saved_rng = training._rng_state(), checkpoint['rng_state']
    rng_checks = {
        'python': restored_rng['python'] == saved_rng['python'],
        'numpy': (restored_rng['numpy'][0] == saved_rng['numpy'][0] and
            np.array_equal(restored_rng['numpy'][1], saved_rng['numpy'][1]) and
            restored_rng['numpy'][2:] == saved_rng['numpy'][2:]),
        'torch_cpu': torch.equal(restored_rng['torch'], saved_rng['torch']),
        'torch_cuda': (all(torch.equal(x,y) for x,y in zip(restored_rng['cuda'], saved_rng['cuda']))
                       if 'cuda' in saved_rng else 'skipped: CUDA unavailable'),
    }
    epoch1_batches = [batch_data[index] for index in dict.fromkeys(int(k.split('-')[1]) for k in selected_ids[1]) for _ in range(2)]
    resumed_log.append(run_epoch(resumed, resumed_opt, epoch1_batches, 1, device, resumed_records))
    training.write_jsonl_record(run/'metrics.jsonl', {'epoch':2, **resumed_log[1]})
    comparisons = {
        'weights': numerical_comparison(continuous_state, resumed[0].state_dict()),
        'optimizer': numerical_comparison(continuous_optimizer, resumed_opt.state_dict()),
        'epoch_logs': numerical_comparison(continuous_log, resumed_log),
        'batch_diagnostics': numerical_comparison(continuous_records, resumed_records),
    }
    resumed_next_records = []
    resumed_next_loss = run_epoch(resumed, resumed_opt, batch_data, 2, device, resumed_next_records)
    comparisons['next_step_loss'] = numerical_comparison(next_loss,resumed_next_loss)
    comparisons['next_step_diagnostics'] = numerical_comparison(next_records,resumed_next_records)
    require_exact = torch.device(device).type == 'cpu'
    passed = all(result['exact'] if require_exact else result['within_tolerance']
                 for result in comparisons.values()) and all(value is not False for value in rng_checks.values())
    # Resume from best creates a tracked branch and carries historical best.
    best_args = copy.deepcopy(args); best_args.resume = str(run/'best.pt')
    best_run = prepare_run(best_args)
    assert best_run != run and (best_run/'best.pt').is_file()
    branch = json.loads((best_run/'run_manifest.json').read_text())
    return {'status': 'passed' if passed else 'failed', 'epochs': 2, 'batches_per_epoch': [2,4],
        'comparison_policy': 'CPU exact' if require_exact else 'GPU predeclared atol=2e-5 rtol=1e-5 plus exact discrete semantics',
        'comparisons': comparisons,
        'dropout': args.dropout, 'weight_max_abs_diff': comparisons['weights']['max_abs_diff'],
        'optimizer_max_abs_diff': comparisons['optimizer']['max_abs_diff'],
        'logs_equal': comparisons['epoch_logs']['exact'],
        'rng_restored': all(value is not False for value in rng_checks.values()),
        'rng_checks':rng_checks, 'diagnostics_equal': comparisons['batch_diagnostics']['exact'],
        'next_step_loss_equal': comparisons['next_step_loss']['exact'],
        'next_epoch_components': next_loss, 'resumed_next_epoch_components': resumed_next_loss,
        'epoch_components': continuous_log, 'resumed_epoch_components': resumed_log,
        'continuous_batch_diagnostics': continuous_records,
        'resumed_batch_diagnostics': resumed_records, 'curriculum_selected_ids': selected_ids,
        'curriculum_restore': metadata['curriculum'],
        'historical_best': metadata['best_checkpoint'],
        'best_resume_branch': branch['parent_checkpoint'],
        'synthetic_difficulty_sha256': artifact['sha256'],
        'teacher_status': 'synthetic fixture only; NOT a real trained teacher'}


def audit(device):
    torch.set_num_threads(1)
    if torch.device(device).type == 'cuda':
        torch.use_deterministic_algorithms(True, warn_only=True)
    training.seed_everything(25)
    batch_data = synthetic_batches(device)
    cases = []
    active_case = None
    try:
        for policy in ('legacy_epoch_snapshot','same_step_same_mode'):
            for dropout in (0., .15):
                for cc, hscl in ((False,False),(True,False),(False,True),(True,True)):
                    active_case = {'policy':policy,'dropout':dropout,'CC':cc,'HSCL':hscl}
                    training.seed_everything(25)
                    args = arguments(calibrate=cc, contrastlearning=hscl,
                                     auxiliary_policy=policy, dropout=dropout)
                    training.args = args
                    current = build_models(device, dropout)
                    optimizer = torch.optim.Adam(current[0].parameters(), lr=args.lr)
                    records = []
                    components = run_epoch(current, optimizer, batch_data, 0, device, records)
                    assert len(records) == 2
                    assert all(torch.isfinite(torch.tensor(v)) for v in components.values())
                    assert all(r['finite_gradients'] for r in records)
                    assert all(p.grad is None for m in current[1:] for p in m.parameters())
                    assert all(sum(r['parameter_update_norms'].values()) > 0 for r in records)
                    assert records[0]['main_aux_max_abs_diff'] == 0.
                    if policy == 'legacy_epoch_snapshot':
                        assert records[1]['main_aux_max_abs_diff'] > 0.
                    else:
                        assert records[1]['main_aux_max_abs_diff'] == 0.
                    cases.append({'status':'passed','policy':policy,'dropout':dropout,
                        'CC':cc,'HSCL':hscl,'components':components,'batches':records})
        with tempfile.TemporaryDirectory(prefix='cmerc_integration_') as temporary:
            recovery = recovery_check(device, batch_data, Path(temporary))
    except Exception as error:
        recovery = {'status':'blocked', 'error_type':type(error).__name__,
                    'error':str(error), 'traceback':traceback.format_exc(),
                    'active_case':active_case, 'completed_cases':len(cases)}
    return {'device':str(device),'dtype':'torch.float32','seed':25,'split_seed':12,
            'input':'synthetic aligned utterance features only',
            'backend': {'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                'deterministic_warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
                'cudnn_deterministic':torch.backends.cudnn.deterministic,
                'cudnn_benchmark':torch.backends.cudnn.benchmark,
                'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                'classification':'default cuDNN train on GPU',
                'same_step_confidence':'ATen eval, full reference gradient only'},
            'cases':cases,
            'recovery':recovery,'formal_training_started':False,
            'real_CL_status':'blocked: no validated trained causal teacher/difficulty'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', default='reports/causal_review_followup')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    report = audit(device)
    suffix = device.type
    (output / ('TRAINING_INTEGRATION_'+suffix+'.json')).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'device':str(device),'cases':len(report['cases']),
        'recovery':report['recovery'],'status':report['recovery']['status']}, indent=2))
    if report['recovery']['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
