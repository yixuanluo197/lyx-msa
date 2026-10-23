"""Verify persisted validation-only trials on CPU; never run a model forward."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import os

import torch

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


def main():
    assert not torch.cuda.is_initialized()
    registry = read(ROOT/'configs/registry.json')
    state = read(ROOT/'state/status.json')
    assert not state['active'] and state['phase'] == 'completed'
    assert state['formal_started'] == state['formal_completed'] == 4
    assert all(run['returncode'] == 0 for run in state['runs'])
    assert len(state['runs']) == 4
    assert {r['run_id'] for r in state['runs']} == {r['run_id'] for r in registry['runs']}
    for name, expected in read(ROOT/'provenance/frozen_execution.json')['files'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT/name
        assert sha(path) == expected, name
    history = read(ROOT/'provenance/history_before.json')['roots']
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
    expected = {(r['run_id'], r['epoch']): r for r in read(ROOT/'preflight/capacity_cpu.json')['records']}
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
        allowed = {'lr', 'skip_final_test', 'output_dir', 'provenance'}
        assert {k:v for k,v in args.items() if k not in allowed} == {
            k:v for k,v in old['args'].items() if k not in allowed}
        assert args['lr'] == run['lr'] and args['skip_final_test'] is True
        assert manifest['split'] == old['split'] and manifest['data'] == old['data']
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
                assert all(math.isfinite(obs['loss_components'][k]) for k in LOSS)
                if split == 'train':
                    assert obs['gradients_finite'] is True and obs['first_step_parameters_changed'] is True
                    assert all(math.isclose(obs['loss_components'][k], epoch[k], rel_tol=1e-10, abs_tol=1e-10) for k in LOSS)
                    steps = [e for e in batches if e['epoch'] == epoch['epoch']]
                    ids = [key for step in steps for key in step['dialogue_ids']]
                    assert len(ids) == len(set(ids)) == len(obs['dialogue_ids'])
                    assert set(ids) == set(obs['dialogue_ids'])
                    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == cap['ordered_train_ids_sha256']
        for batch in batches:
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
            'every_epoch_order_matches_xp04_sampler':True,'test_forwards':0,'checkpoints':checkpoints}
    resources = rows(ROOT/'logs/resources.jsonl')
    foreign = [sample for sample in resources if sample['unexpected_processes']]
    assert not foreign, 'Resource conflict requires separate review before claiming clean comparison'
    assert not torch.cuda.is_initialized()
    report = {'status':'passed','created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'formal_runs':4,'test_forwards':0,'cuda_initialized':False,
              'historical_files_unchanged':historical,'frozen_sources_and_inputs_unchanged':True,
              'resource_samples':len(resources),'unexpected_process_samples':0,'runs':results,
              'limits':['Single seed per dataset, historical cross-card controls.',
                        'Author difficulty generation history unknown.',
                        'No test or final independent evaluation performed in this stage.']}
    (ROOT/'provenance/final_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','formal_runs':4,'test_forwards':0,
                      'history_files':sum(historical.values()),'cuda_initialized':False}))


if __name__ == '__main__':
    main()
