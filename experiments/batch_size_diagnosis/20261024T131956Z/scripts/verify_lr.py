"""Verify persisted validation-only trials on CPU; never run a model forward."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import os

import torch
from run_single import admissible, validate_device, validate_gate, validate_resource_policy
from watch_training import verify_skip_patch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
LOSS = ['classification_loss', 'cc_loss', 'mscl_loss', 'cscl_loss', 'total_loss']


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def finite(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all().item()
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


def resource_evidence(registry, state):
    policy = registry['resource_policy']
    validate_device(registry['gpu_index'], registry['gpu_uuid'])
    validate_resource_policy(policy)
    assert state['preregistered_co_resident_pids'] == policy['preexisting_gpu3_pids']
    assert state['new_foreign_pids_pause_admission'] is False
    resources = rows(ROOT/'logs/resources.jsonl')
    assert resources
    errors_path = ROOT/'logs/resource_errors.jsonl'
    errors = rows(errors_path) if errors_path.exists() else []
    allowed, unexpected, all_foreign = [], [], []
    for sample in resources:
        assert sample['gpu_index'] == registry['gpu_index'] and sample['gpu_uuid'] == registry['gpu_uuid']
        validate_device(sample['gpu_index'], sample['gpu_uuid'], sample['gpu_name'])
        own = set(sample['own_pids'])
        foreign = [p for p in sample['processes'] if p['pid'] not in own]
        permitted = [p for p in foreign if p['pid'] in policy['preexisting_gpu3_pids']]
        other = [p for p in foreign if p['pid'] not in policy['preexisting_gpu3_pids']]
        assert sample['all_foreign_processes'] == foreign
        assert sample['allowed_foreign_processes'] == permitted
        assert sample['unexpected_processes'] == other
        if foreign: all_foreign.append(sample)
        if permitted: allowed.append(sample)
        if other: unexpected.append(sample)
    for run in [state['preflight'], *state['runs']]:
        samples = run['admission_samples']
        assert len(samples) == policy['samples_required']
        for sample in samples:
            assert sample in resources and not sample['own_pids']
            assert admissible(sample, policy)
        stamps = [dt.datetime.fromisoformat(sample['time']) for sample in samples]
        assert all((later-earlier).total_seconds() >= 9.5 for earlier,later in zip(stamps,stamps[1:]))
        release = run['gpu_release_confirmed']
        assert release in resources and run['pid'] not in {p['pid'] for p in release['processes']}
        assert run['returncode'] == 0
    assert state['preflight']['finished_at_utc'] < state['runs'][0]['started_at_utc']
    assert state['preflight']['gpu_release_confirmed']['time'] <= state['runs'][0]['admission_samples'][0]['time']
    assert state['resource_samples'] == len(resources)
    assert state['resource_query_errors'] == len(errors)
    assert state['preexisting_pid_samples'] == len(allowed)
    assert state['unexpected_process_samples'] == len(unexpected)
    assert state['unexpected_pids'] == sorted({p['pid'] for sample in unexpected for p in sample['unexpected_processes']})
    observed_pids = sorted({p['pid'] for sample in all_foreign for p in sample['all_foreign_processes']})
    assert state['observed_foreign_pids'] == observed_pids
    return {'status': 'limited' if all_foreign or errors else 'sampled_no_foreign_observed',
            'admissions_passed_registered_three_sample_policy': True,
            'preregistered_co_resident_pids': policy['preexisting_gpu3_pids'],
            'observed_foreign_pids': observed_pids,
            'admission_min_free_mib': policy['admission_min_free_mib'],
            'admission_max_utilization_percent': policy['admission_max_utilization_percent'],
            'new_foreign_pids_pause_admission': policy['new_foreign_pids_pause_admission'],
            'resource_samples': len(resources), 'all_foreign_process_samples': len(all_foreign),
            'allowed_preexisting_pid_samples': len(allowed), 'new_foreign_pid_samples': len(unexpected),
            'new_foreign_pids': state['unexpected_pids'], 'resource_query_errors': len(errors),
            'minimum_free_mib': min(sample['free_mib'] for sample in resources),
            'minimum_free_mib_during_formal': min(sample['free_mib'] for sample in resources if sample['stage']=='formal_running'),
            'maximum_utilization_percent': max(sample['utilization_percent'] for sample in resources),
            'limits': ['GPU co-residency is allowed; preregistered PIDs are disclosure records, not an exclusive allowlist.',
                       'New foreign PIDs are recorded and do not block admission when the registered memory/utilization conditions pass.',
                       'Resource pressure and query errors are recorded without terminating any task; failed queries cannot authorize admission.',
                       'Discrete nvidia-smi samples cannot establish continuous resource isolation.']}


def main():
    assert not torch.cuda.is_initialized()
    registry = read(ROOT/'configs/registry.json')
    assert registry['formal_run_count'] == len(registry['runs']) == 1
    registered = registry['runs'][0]
    assert (registered['dataset'],registered['seed'],registered['lr'],registered['epochs'],registered['batch_size'],registered['validation_batch_size']) == ('MELD',2024,.0001,40,8,8)
    state = read(ROOT/'state/status.json')
    assert not state['active'] and state['phase'] == 'completed'
    assert state['formal_started'] == state['formal_completed'] == 1
    assert all(run['returncode'] == 0 for run in state['runs'])
    assert len(state['runs']) == 1
    assert {r['run_id'] for r in state['runs']} == {r['run_id'] for r in registry['runs']}
    capsule_sha = sha(ROOT/'provenance/frozen_execution.json')
    assert state['frozen_execution_sha256'] == state['final_frozen_execution_sha256'] == capsule_sha
    validate_gate(capsule_sha, registered)
    assert state['preflight_started'] == 1 and state['preflight']['returncode'] == 0
    probe_status = read(ROOT/'state/preflight.json')
    assert probe_status['phase'] == 'passed' and probe_status['test_forward_passes'] == 0
    for name, expected in read(ROOT/'provenance/frozen_execution.json')['files'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT/name
        assert sha(path) == expected, name
    history = read(ROOT/'provenance/history_before.json')['roots']
    assert len(history) == 10
    historical = {}
    for relative, snapshot in history.items():
        original = REPO/relative
        files, links = {}, {}
        for directory, _, names in os.walk(original, followlinks=False):
            for name in names:
                path = Path(directory)/name
                key = str(path.relative_to(original))
                if path.is_symlink():
                    links[key] = os.readlink(path)
                elif path.is_file():
                    files[key] = sha(path)
        assert files == snapshot['files'] and links == snapshot['symlinks'], relative
        historical[relative] = len(files)
    for value in read(ROOT/'provenance/data_inputs.json').values():
        assert sha(Path(value['path'])) == value['sha256']
        assert str(Path(value['path']).resolve()) == value['resolved']
    capacity = read(ROOT/'preflight/capacity_cpu.json')
    assert capacity['status'] == 'passed' and capacity['original_sampling_ast_unchanged'] is True
    assert capacity['historical_batch16_replay_passed'] is True
    expected = {(r['run_id'], r['epoch']): r for r in capacity['records']}
    results = {}
    for run in registry['runs']:
        directory = Path(run['run_dir'])
        observed = ROOT/'results'/run['run_id']
        status = read(observed/'status.json')
        assert status['phase'] == 'completed' and status['counts'] == {
            'train': run['epochs'], 'validation': run['epochs'], 'test': 0}
        assert status['test_forward_passes'] == 0 and status['target_line_events'] == 0
        assert not (directory/'final_test.json').exists()
        assert not list(observed.glob('*.npz'))
        manifest = read(directory/'run_manifest.json')
        args = manifest['args']
        old = read(run['historical_manifest'])
        patched_sha = verify_skip_patch(ROOT/'code_xp04/train_cm.py',old['source']['files']['train_cm.py'])
        assert set(manifest['source']['files']) == set(old['source']['files'])
        for name,digest in old['source']['files'].items():
            expected_sha = patched_sha if name == 'train_cm.py' else digest
            assert manifest['source']['files'][name] == expected_sha
            assert sha(ROOT/'code_xp04'/name) == expected_sha
        allowed = {'batch_size', 'skip_final_test', 'output_dir', 'provenance'}
        assert {k:v for k,v in args.items() if k not in allowed} == {
            k:v for k,v in old['args'].items() if k not in allowed}
        assert args['lr'] == run['lr'] == old['args']['lr'] == .0001
        assert args['batch_size'] == run['batch_size'] == 8 and old['args']['batch_size'] == 16
        assert args['skip_final_test'] is True
        assert args['selection_protocol'] == 'validation' and args['rng_policy'] == 'continuous'
        assert args['rnn_backend'] == 'aten' and args['batch_policy'] == 'capacity_safe'
        assert args['calibrate'] and args['contrastlearning'] and args['courselearning']
        assert manifest['split'] == old['split'] and manifest['data'] == old['data']
        assert manifest['split']['hash'] == run['split_hash']
        first = read(observed/'first.json')
        probe_first = read(ROOT/'preflight'/run['run_id']/'observer/first.json')
        for key in ['initial_model_sha256','rng_before_train','rng_before_step','rng_after_step','dialogue_ids','lengths']:
            assert first[key] == probe_first[key], key
        for entry in [first,probe_first]:
            assert entry['gradients_finite'] and entry['optimizer_updated_parameters']
            assert math.isfinite(entry['gradient_l2']) and all(math.isfinite(v) for v in entry['components'].values())
        for key in LOSS:
            assert math.isclose(first['components'][key],probe_first['components'][key],rel_tol=1e-5,abs_tol=1e-6), key
        assert math.isclose(first['gradient_l2'],probe_first['gradient_l2'],rel_tol=1e-5,abs_tol=1e-6)
        epochs = rows(directory/'metrics.jsonl')
        assert [e['epoch'] for e in epochs] == list(range(1, run['epochs']+1))
        selected = max(epochs, key=lambda e:e['validation_weighted_f1'])
        assert status['selected_epoch'] == selected['epoch']
        assert status['selection_weighted_f1'] == selected['validation_weighted_f1']
        watchers = rows(observed/'watch_epochs.jsonl')
        batches = rows(observed/'watch_batches.jsonl')
        assert len(watchers) == 2*run['epochs']
        for split in ['train', 'validation']:
            assert [e['epoch'] for e in watchers if e['split'] == split] == list(range(1,run['epochs']+1))
        assert not any(e['split'] == 'test' for e in watchers+batches)
        for epoch in epochs:
            assert epoch['test_accuracy'] is None and epoch['test_weighted_f1'] is None
            assert epoch['selection_protocol'] == epoch['selection_split'] == 'validation'
            cap = expected[run['run_id'], epoch['epoch']]
            assert cap['batch_size'] == cap['validation_batch_size'] == 8
            assert cap['same_as_original_sampler_batch8'] is True
            assert cap['historical16']['batch_size'] == 16 and cap['historical16']['replayed_metrics_match'] is True
            assert epoch['train_dialogues'] == cap['train_dialogues']
            assert epoch['sampler_retries'] == cap['sampler_retries']
            assert epoch['max_batch_edges'] == cap['capacity']['edges']
            assert epoch['max_batch_nodes'] == cap['capacity']['nodes']
            assert epoch['sample_ratio'] == cap['sample_ratio']
            for split in ['train', 'validation']:
                obs = next(e for e in watchers if e['epoch'] == epoch['epoch'] and e['split'] == split)
                assert obs['weighted_f1'] == epoch[split+'_weighted_f1']
                assert obs['accuracy'] == epoch[split+'_accuracy']
                assert obs['batches'] == cap[split+'_batches']
                if split == 'validation':
                    assert obs['dialogue_ids'] == manifest['split']['validation_ids']
                    assert obs['batches'] == math.ceil(len(obs['dialogue_ids']) / 8)
                    assert cap['validation_last_batch_dialogues'] == (len(obs['dialogue_ids'])-1) % 8 + 1
                assert all(math.isfinite(obs['loss_components'][k]) for k in LOSS)
                if split == 'train':
                    assert obs['gradients_finite'] is True and obs['first_step_parameters_changed'] is True
                    assert all(math.isclose(obs['loss_components'][k], epoch[k], rel_tol=1e-10, abs_tol=1e-10) for k in LOSS)
                    steps = [e for e in batches if e['epoch'] == epoch['epoch']]
                    assert len(steps) == cap['train_batches'] == math.ceil(cap['train_dialogues'] / 8)
                    assert all(len(step['dialogue_ids']) == 8 for step in steps[:-1])
                    assert len(steps[-1]['dialogue_ids']) == cap['train_last_batch_dialogues']
                    ids = [key for step in steps for key in step['dialogue_ids']]
                    assert len(ids) == len(set(ids)) == len(obs['dialogue_ids'])
                    assert set(ids) == set(obs['dialogue_ids'])
                    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == cap['ordered_train_ids_sha256']
        for batch in batches:
            assert 0 < len(batch['dialogue_ids']) <= 8
            assert batch['gradients_finite'] and math.isfinite(batch['gradient_l2'])
            assert all(math.isfinite(batch[k]) for k in LOSS)
            value = batch['classification_loss'] + args['rank_coff']*batch['cc_loss']
            value += args['mscl_coff']*batch['mscl_loss'] + args['cscl_coff']*batch['cscl_loss']
            assert math.isclose(value, batch['total_loss'], rel_tol=1e-5, abs_tol=1e-6)
        checkpoints = {}
        for filename, epoch in [('best.pt', selected['epoch']), ('latest.pt', run['epochs'])]:
            checkpoint = torch.load(directory/filename, map_location='cpu')
            assert checkpoint['epoch']+1 == epoch
            assert checkpoint['best_epoch']+1 == selected['epoch']
            assert checkpoint['best_metric'] == selected['validation_weighted_f1']
            assert checkpoint['args']['lr'] == run['lr'] and checkpoint['args']['skip_final_test']
            assert checkpoint['args']['batch_size'] == 8
            assert json.loads(json.dumps(checkpoint['args'])) == manifest['args']
            assert json.loads(json.dumps(checkpoint['provenance'])) == manifest['args']['provenance']
            assert checkpoint['optimizer_state_dict']['state'] and checkpoint['rng_state']
            rng = checkpoint['rng_state']
            assert {'python', 'numpy', 'torch', 'cuda'}.issubset(rng)
            assert len(rng['python']) == 3 and len(rng['numpy']) == 5
            assert rng['torch'].dtype == torch.uint8 and rng['torch'].numel() > 0
            assert len(rng['cuda']) == 1
            assert all(s.dtype == torch.uint8 and s.numel() > 0 for s in rng['cuda'])
            assert all(g['lr'] == run['lr'] for g in checkpoint['optimizer_state_dict']['param_groups'])
            finite(checkpoint['model_state_dict'])
            finite(checkpoint['optimizer_state_dict'])
            checkpoints[filename] = {'epoch':epoch, 'sha256':sha(directory/filename),
                                     'model_and_adam_finite':True, 'full_rng_present':True}
            del checkpoint
        results[run['run_id']] = {'status':'passed', 'epochs':run['epochs'],
            'selected_epoch':selected['epoch'], 'validation_best_wf1':selected['validation_weighted_f1'],
            'validation_last10_wf1':sum(e['validation_weighted_f1'] for e in epochs[-10:])/10,
            'train_batches':len(batches), 'loss_and_gradient_finite':True,
            'every_epoch_order_matches_registered_batch8_replay':True,
            'unchanged_sampler_AST_and_historical_batch16_replay_passed':True,
            'train_and_validation_batch_size':8,'lr_unchanged_from_xp04':.0001,
            'test_forwards':0,'checkpoints':checkpoints}
    resource_report = resource_evidence(registry,state)
    assert not torch.cuda.is_initialized()
    report = {'status':'passed','created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'formal_runs':1,'test_forwards':0,'cuda_initialized':False,
              'historical_files_unchanged':historical,'frozen_sources_and_inputs_unchanged':True,
              'resource_evidence':resource_report,'runs':results,
              'limits':['This is one registered Batch Size=8 diagnosis, not a completed five-seed study.',
                        'Train and validation batches both change from16 to8; grouping and optimizer step counts differ from the historical control.',
                        'The selected healthy XP GPU may be shared with observed foreign PIDs under the registered capacity gate; resource isolation is limited.',
                        'Author difficulty generation history unknown.',
                        'No test or final independent evaluation performed in this stage.']}
    (ROOT/'provenance/final_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','formal_runs':1,'test_forwards':0,
                      'history_files':sum(historical.values()),'cuda_initialized':False}))


if __name__ == '__main__':
    main()
