"""CPU-only verification of the fixed three-run experiment, without model inference."""
import ast
import hashlib
import json
import math
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Sampler

from observe_training import assert_scientific_pair, REMOVALS

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def check_history():
    original = json.loads((ROOT/'provenance/history_before.json').read_text())
    counts = {}
    for folder, manifest in original['roots'].items():
        base = Path(folder)
        files = {str(p.relative_to(base)):p for p in base.rglob('*') if p.is_file() and not p.is_symlink()}
        links = {str(p.relative_to(base)):str(p.readlink()) for p in base.rglob('*') if p.is_symlink()}
        assert set(files) == set(manifest['files']), ('history file inventory changed', folder)
        assert links == manifest['symlinks'], ('history links changed',folder)
        for name, meta in manifest['files'].items():
            assert sha(files[name]) == meta['sha256'], ('history bytes changed',folder,name)
        counts[folder] = len(files)
    return counts


def sampler_class():
    path = ROOT/'code_xp04/train_cm.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n,ast.ClassDef) and n.name=='CapacitySafeBatchSampler')
    namespace = {'torch':torch,'Sampler':Sampler}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['CapacitySafeBatchSampler']


def expected_batches(run, args, split, epoch, lengths, sampler, difficulty):
    keys = split['train_ids']
    if args['courselearning']:
        ratio = min(1,round(args['epoch_ratio']*int((epoch-1)/args['scheduler_steps']+1),2))
        chosen = set(sorted(keys,key=lambda k:difficulty[k])[:int(ratio*len(keys))])
        keys = [key for key in keys if key in chosen]
    if args['batch_policy']=='source_order':
        order = list(range(len(keys)))
        indices = [order[start:start+16] for start in range(0,len(keys),16)]
    else:
        generator = torch.Generator().manual_seed(args['seed']+epoch-1)
        # The actual DataLoader draws a worker/base seed before sampling.
        torch.empty((),dtype=torch.int64).random_(generator=generator)
        indices = list(sampler([lengths[k] for k in keys],16,generator,1000,5200,10000))
    return [[keys[index] for index in batch] for batch in indices]


def independent_metrics(labels, full, removed):
    probabilities = np.exp(full).astype(np.float64)
    confidence = probabilities.max(1)
    pred = full.argmax(1)
    correctness = (pred==labels).astype(float)
    ece = 0.
    for index in range(15):
        mask = (confidence>index/15)&(confidence<=(index+1)/15)
        if np.any(mask):
            ece += mask.mean()*abs(correctness[mask].mean()-confidence[mask].mean())*100
    rows = np.arange(len(labels))
    cel = sum(np.maximum(0.,value.astype(np.float64)[rows,labels]-full.astype(np.float64)[rows,labels]).sum()
              for value in removed.values())
    return {'weighted_f1':f1_score(labels,pred,average='weighted')*100,
            'macro_f1':f1_score(labels,pred,average='macro')*100,
            'accuracy':accuracy_score(labels,pred)*100,'ece':ece,'cel':float(cel)}


def main():
    assert not torch.cuda.is_initialized(), 'Verification must not initialize CUDA'
    registry = json.loads((ROOT/'configs/registry.json').read_text())
    state = json.loads((ROOT/'state/status.json').read_text())
    assert state['phase']=='completed' and not state['active']
    assert state['formal_started']==state['formal_completed']==3
    assert len(state['runs'])==3 and all(row['returncode']==0 for row in state['runs'])
    frozen = json.loads((ROOT/'provenance/frozen_execution.json').read_text())
    for name, expected in frozen['files'].items():
        assert sha(ROOT/name)==expected,('frozen input changed',name)
    for name, meta in json.loads((ROOT/'provenance/data_inputs.json').read_text()).items():
        assert sha(ROOT/name)==meta['sha256'],('data changed',name)
    historical_counts = check_history()
    reference = json.loads((ROOT/'provenance/xp04_references.json').read_text())
    sampler = sampler_class()
    verified = {}
    for run in registry['runs']:
        ident = run['run_id']; directory = Path(run['run_dir']); out = ROOT/'results'/ident
        manifest = json.loads((directory/'run_manifest.json').read_text())
        historical = json.loads(Path(run['historical_manifest']).read_text())
        args = manifest['args']; split = manifest['split']
        assert_scientific_pair(SimpleNamespace(**args),historical,run)
        epochs = read_rows(directory/'metrics.jsonl')
        assert [row['epoch'] for row in epochs]==list(range(1,run['epochs']+1))
        assert all(row['test_accuracy'] is None and row['test_weighted_f1'] is None for row in epochs)
        chosen = max(epochs,key=lambda row:row['validation_weighted_f1'])
        final = json.loads((directory/'final_test.json').read_text())
        assert final['selected_epoch']==chosen['epoch']
        status = json.loads((out/'status.json').read_text())
        assert status['phase']=='completed' and status['final_test_matches']
        assert status['counts']=={'train':run['epochs'],'validation':run['epochs'],'test':1}
        assert status['test_forward_passes']==1  # means one complete test-loader traversal
        observed = read_rows(out/'observer_epochs.jsonl')
        assert len(observed)==2*run['epochs']+1
        tests = [row for row in observed if row['split']=='test']
        assert len(tests)==1 and tests[0]['epoch']==chosen['epoch']
        assert tests[0]['dialogue_ids']==split['test_ids']
        for epoch in range(1,run['epochs']+1):
            for part in ['train','validation']:
                items = [row for row in observed if row['split']==part and row['epoch']==epoch]
                assert len(items)==1
                row = items[0]
                assert round(row['weighted_f1'],2)==epochs[epoch-1][part+'_weighted_f1']
                assert round(row['accuracy'],2)==epochs[epoch-1][part+'_accuracy']
        datafile = 'IEMOCAP_features.pkl' if run['dataset']=='IEMOCAP' else 'MELD_features_raw1.pkl'
        with (Path(run['code_dir'])/'data'/datafile).open('rb') as stream:
            raw = pickle.load(stream,encoding='latin1')
        lengths = {k:len(value) for k,value in raw[2].items()}
        difficulty = {}
        if args['courselearning']:
            with (Path(run['code_dir'])/'caldiff'/(run['dataset']+'_diff.pkl')).open('rb') as stream:
                difficulty = pickle.load(stream)
        batches = read_rows(out/'observer_batches.jsonl')
        train = defaultdict(list)
        for batch in batches:
            expected = batch['classification_loss']
            if args['calibrate'] or (batch['split']=='test' and run['variant']=='baseline'):
                expected += args['rank_coff']*batch['cc_loss']
            if args['contrastlearning']:
                expected += args['mscl_coff']*batch['mscl_loss']+args['cscl_coff']*batch['cscl_loss']
            assert math.isclose(expected,batch['total_loss'],rel_tol=1e-5,abs_tol=1e-6)
            if batch['split']=='train':
                assert batch['gradients_finite']
                train[batch['epoch']].append(batch)
        for epoch in range(1,run['epochs']+1):
            expected = expected_batches(run,args,split,epoch,lengths,sampler,difficulty)
            assert [row['dialogue_ids'] for row in train[epoch]]==expected,(ident,epoch,'batch ordering mismatch')
            for row in train[epoch]:
                assert row['lengths']==[lengths[k] for k in row['dialogue_ids']]
                assert sum(row['lengths'])+3*len(row['lengths'])<=1000
                assert 3*sum(row['lengths'])<=5200
        data = np.load(out/'predictions.npz')
        labels = data['labels']; full = data['full_log_probabilities']
        removed = {name:data[name+'_log_probabilities'] for name in REMOVALS}
        assert len(labels)==sum(lengths[k] for k in split['test_ids'])
        for array in [full,*removed.values()]:
            assert np.isfinite(array).all() and array.shape==full.shape
            assert np.allclose(np.exp(array).sum(1),1,rtol=1e-5,atol=1e-6)
        assert np.array_equal(data['predictions'],full.argmax(1))
        original_predictions = Path(reference[ident]['queue_job']['result_dir'])/'predictions.npz'
        old = np.load(original_predictions)
        assert np.array_equal(labels,old['labels'])
        independent = independent_metrics(labels,full,removed)
        saved = json.loads((out/'metrics.json').read_text())
        for key, value in independent.items():
            assert math.isclose(value,saved[key],rel_tol=1e-10,abs_tol=1e-8),(ident,key,value,saved[key])
        assert round(independent['weighted_f1'],2)==final['test_weighted_f1']
        assert round(independent['accuracy'],2)==final['test_accuracy']
        checks = {}
        for filename, expected_epoch in [('best.pt',chosen['epoch']),('latest.pt',run['epochs'])]:
            payload = torch.load(directory/filename,map_location='cpu')
            assert payload['epoch']+1==expected_epoch
            assert payload['best_epoch']+1==chosen['epoch']
            assert payload['args']==args
            assert payload['optimizer_state_dict']['state']
            assert all(torch.isfinite(t).all().item() for t in payload['model_state_dict'].values())
            for item in payload['optimizer_state_dict']['state'].values():
                assert all(not torch.is_tensor(value) or torch.isfinite(value).all().item() for value in item.values())
            assert {'python','numpy','torch','cuda'}<=set(payload['rng_state'])
            checks[filename]={'sha256':sha(directory/filename),'epoch':expected_epoch,'finite_model_and_adam':True}
            del payload
        verified[ident]={'status':'passed','epochs':len(epochs),'test_traversals':1,'selected_epoch':chosen['epoch'],
            'test_metrics_independently_recomputed':independent,'ordered_batch_coverage_every_epoch':True,
            'weighted_loss_identity_every_batch':True,'checkpoints':checks,
            'train_batches':sum(map(len,train.values())),
            'zero_mscl_train_batches':sum(b['mscl_loss']==0 for rows in train.values() for b in rows),
            'zero_cscl_train_batches':sum(b['cscl_loss']==0 for rows in train.values() for b in rows)}
    samples = read_rows(ROOT/'logs/resources.jsonl')
    unexpected = [row for row in samples if row.get('unexpected_processes')]
    assert not unexpected, 'Unexpected GPU context requires a separate scientific-validity review'
    result = {'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'formal_runs_started':3,'formal_runs_completed':3,'history_files_unchanged':historical_counts,
        'data_and_frozen_execution_unchanged':True,'runs':verified,'resource_samples':len(samples),
        'unexpected_gpu_process_samples':len(unexpected),'cuda_initialized':torch.cuda.is_initialized(),
        'limits':['single seed, historical controls','same-model cross-card comparison is not bitwise guaranteed',
                  'fixed_author generation history unknown','discrete resource samples are not global mutual exclusion']}
    assert result['cuda_initialized'] is False
    (ROOT/'provenance/final_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','runs':3,'historical_files_unchanged':sum(historical_counts.values()),
                     'unique_test_traversals':3,'cuda_initialized':False}))


if __name__=='__main__':
    main()
