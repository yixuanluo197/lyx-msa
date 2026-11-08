"""Explicit validation-only technical benchmark. Never reads Test predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parents[1] / '20261107T172302Z/scripts'
sys.path.insert(0, str(PREVIOUS))
import run_pilot as runner
from graph_index_cache import install_graph_cache
from accelerated_forward import FixedBatchExecution


def compare(torch, a, b):
    a, b = a.detach(), b.detach()
    return {'passed': bool(torch.allclose(a, b, atol=1e-6, rtol=1e-5)
        and torch.allclose(a.exp(), b.exp(), atol=1e-6, rtol=1e-5)
        and torch.equal(a.argmax(-1), b.argmax(-1))),
        'max_log_error': float((a-b).abs().max()),
        'max_probability_error': float((a.exp()-b.exp()).abs().max())}


def benchmark(datasets=None, cuda_graph=True):
    os.chdir(runner.CODE)
    api = runner.imports();torch = api[0]
    evidence = {'validation_only':True, 'original_batch_size':16,
                'fp32':True, 'gradients':False, 'batches':[], 'started_at':time.time()}
    dest = ROOT/'provenance/acceleration_benchmark.json'
    if dest.exists():
        previous=json.loads(dest.read_text())
        evidence['previous_attempts']=previous.get('previous_attempts',[])+[{k:v for k,v in previous.items() if k!='previous_attempts'}]
    def save():
        dest.write_text(json.dumps(evidence,indent=2)+'\n')
    try:
        with torch.no_grad():
            for ds, seed in [('IEMOCAP',1475),('MELD',67137)]:
                if datasets is not None and ds not in datasets:continue
                model = runner.load_model(ds,seed,api)
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                cache=install_graph_cache(model.graph_model)
                dataset=runner.dataset_for(ds,'validation',api[2])
                for bi,batch in enumerate(runner.loader_for(dataset,torch)):
                    if bi >= 2:break
                    tensors=[x.cuda() for x in batch[:-1]]
                    lengths=[int(x.sum()) for x in tensors[7]]
                    entry={'dataset':ds,'batch_index':bi,'lengths':lengths,'checks':[]}
                    evidence['batches'].append(entry)
                    reference=runner.forward(model,tensors,lengths)
                    start=time.monotonic()
                    execution=FixedBatchExecution(model,tensors,lengths,cuda_graph=cuda_graph)
                    torch.cuda.synchronize();entry['build_seconds']=time.monotonic()-start
                    entry.update(mode=execution.mode,capture_error=execution.capture_error)
                    # Compare complete batch output to the original uncached model.
                    groups={'T1':[0],'T2':[1],'T3':[2],'T4':[3],
                            'T-All':[0,1,2,3],'A':[5],'V':[4]}
                    for target in [0,min(1,lengths[0]-1)]:
                        for name,indices in groups.items():
                            for kind,rho in [('attenuation',.25),('attenuation',1.),('mask',.8)]:
                                changed=[x.clone() for x in tensors]
                                for index in indices:
                                    if kind=='attenuation':changed[index][target,0].mul_(1-rho)
                                    else:
                                        g=torch.Generator(device='cpu');g.manual_seed(1475+target+index)
                                        mask=torch.rand(changed[index][target,0].shape,generator=g)<rho
                                        changed[index][target,0].masked_fill_(mask.cuda(),0)
                                cache.restore()
                                try:original=runner.forward(model,changed,lengths)
                                finally:cache.enable()
                                actual=execution(changed,lengths)
                                result=compare(torch,actual,original)
                                result.update(target_utterance=target,group=name,kind=kind,rho=rho)
                                entry['checks'].append(result);save()
                                if not result['passed']:raise AssertionError('Accelerated full forward differs')
                    cache.restore()
                    try:original=runner.forward(model,tensors,lengths)
                    finally:cache.enable()
                    check=compare(torch,execution(tensors,lengths),original)
                    check['kind']='original';entry['checks'].append(check)
                    if not check['passed']:raise AssertionError('Original accelerated forward differs')
                    for mode,fn in [('cached_eager',lambda:runner.forward(model,tensors,lengths)),
                                    ('accelerated',lambda:execution(tensors,lengths))]:
                        for _ in range(3):fn()
                        torch.cuda.synchronize();start=time.monotonic()
                        for _ in range(20):fn()
                        torch.cuda.synchronize();entry[mode+'_seconds_per_forward']=(time.monotonic()-start)/20
                    entry['speedup']=entry['cached_eager_seconds_per_forward']/entry['accelerated_seconds_per_forward']
                    save();del execution,tensors
                evidence.setdefault('model_unchanged',{})[ds]=all(torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items())
                assert evidence['model_unchanged'][ds]
                cache.restore();del model;torch.cuda.empty_cache()
        evidence['passed']=True
    except BaseException as error:
        evidence.update(passed=False,error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        evidence['finished_at']=time.time()
        evidence['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('accelerated_forward.py')]}
        save()


def cache_parity():
    os.chdir(runner.CODE)
    api=runner.imports();torch=api[0]
    sys.path.insert(0,str(ROOT/'scripts'))
    from evidence_core import attenuate, mask_vectors, mask_rng
    evidence={'scope':'validation-only original complete batches', 'checks':[],
              'source_cache_sha256':hashlib.sha256((PREVIOUS/'graph_index_cache.py').read_bytes()).hexdigest(),
              'started_at':time.time(), 'fp32':True, 'model_unchanged':{}}
    dest=ROOT/'provenance/graph_cache_parity.json'
    def save():dest.write_text(json.dumps(evidence,indent=2)+'\n')
    try:
        with torch.no_grad():
            for ds,seed in [('IEMOCAP',1475),('MELD',67137)]:
                model=runner.load_model(ds,seed,api)
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                cache=install_graph_cache(model.graph_model)
                with np.load(PREVIOUS.parent/f'provenance/{ds}_train_statistics.npz') as z:
                    means={j:z[k+'_mean'] for j,k in enumerate(['text0','text1','text2','text3','visual','audio'])}
                dataset=runner.dataset_for(ds,'validation',api[2])
                for bi,batch in enumerate(runner.loader_for(dataset,torch)):
                    tensors=[x.cuda() for x in batch[:-1]]
                    lengths=[int(x.sum()) for x in tensors[7]]
                    groups={'T1':[0],'T-All':[0,1,2,3],'A':[5],'V':[4]}
                    cases=[('original',[],None,None)]
                    cases += [(name,idx,kind,value) for name,idx in groups.items()
                              for kind,value in [('attenuation',.25),('attenuation',1.),('random_masking',.8)]]
                    for name,indices,kind,value in cases:
                        changed=[x.clone() for x in tensors]
                        if indices:
                            vectors=[changed[j][0,0].cpu().numpy().copy() for j in indices]
                            m=[means[j] for j in indices]
                            if kind=='attenuation':values=attenuate(vectors,m,value)
                            else:values=mask_vectors(vectors,m,value,mask_rng(ds,seed,str(batch[-1][0]),0,name,value,0))[0]
                            for j,v in zip(indices,values):changed[j][0,0].copy_(torch.as_tensor(v,device=changed[j].device))
                        before=[x.clone() for x in changed]
                        cache.restore()
                        try:original=runner.forward(model,changed,lengths)
                        finally:cache.enable()
                        actual=runner.forward(model,changed,lengths)
                        result=compare(torch,actual,original)
                        result.update(dataset=ds,batch_index=bi,dialogue_ids=list(map(str,batch[-1])),
                                      lengths=lengths,target_utterance=0,target=name,kind=kind,value=value,
                                      full_inputs_unchanged=all(torch.equal(a,b) for a,b in zip(before,changed)))
                        assert result['full_inputs_unchanged']
                        evidence['checks'].append(result);save()
                        assert result['passed'],'Cached eager differs from original eager'
                evidence['model_unchanged'][ds]=all(torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items())
                assert evidence['model_unchanged'][ds]
                cache.restore();del model;torch.cuda.empty_cache()
        evidence['passed']=True
    except BaseException as error:
        evidence.update(passed=False,error=repr(error),traceback=traceback.format_exc());raise
    finally:
        evidence['finished_at']=time.time();save()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-benchmark',action='store_true');parser.add_argument('--dataset',choices=['IEMOCAP','MELD']);parser.add_argument('--no-cuda-graph',action='store_true');parser.add_argument('--cache-parity',action='store_true');args=parser.parse_args()
    if args.cache_parity:cache_parity()
    elif args.run_benchmark:benchmark(None if args.dataset is None else [args.dataset],not args.no_cuda_graph)
