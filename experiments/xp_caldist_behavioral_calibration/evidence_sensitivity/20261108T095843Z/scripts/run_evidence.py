"""Validation-only, target-only frozen CMERC interventions (no training/Test APIs)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from evidence_core import attenuate, mask_vectors, mask_rng, target_indices, condition_specs

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/evidence_sensitivity_config.json'
CFG = json.loads(CONFIG.read_text())
PREV = Path(CFG['previous_pilot'])
sys.path.insert(0, str(PREV/'scripts'))
import run_pilot as old
from graph_index_cache import install_graph_cache


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def write_json(p, obj):
    old.write_json(p, obj)


def compare(a,b):
    assert a.shape==b.shape and np.isfinite(a).all()
    assert np.allclose(a,b,atol=1e-6,rtol=1e-5), float(np.max(np.abs(a-b)))
    assert np.array_equal(a.argmax(-1),b.argmax(-1))
    return float(np.max(np.abs(a-b)))


def gate(ds, phase):
    audit=json.loads((ROOT/'provenance/audit.json').read_text());assert audit['status']=='passed'
    src=json.loads((ROOT/'provenance/source_hashes.json').read_text())
    assert sha(CONFIG)==src['config_sha256']
    for p,digest in src['inference_scripts'].items():assert sha(p)==digest,p
    item=audit['datasets'][ds]
    for key in ['checkpoint','effective_config','train_statistics','validation_baseline']:
        assert sha(item[key+'_path'])==item[key+'_sha256'],key
    for p,digest in audit['frozen_source_sha256'].items(): assert sha(p)==digest,p
    for p,digest in item['data_sha256'].items():assert sha(p)==digest,p
    phases=list(CFG['phases'])
    for prior in phases[:phases.index(phase)]:
        for name,seed in CFG['anchors'].items():
            done=ROOT/f'runs/{name}/seed{seed}/{prior}_complete.json'
            assert json.loads(done.read_text())['passed'],str(done)
    parity=json.loads((ROOT/'provenance/graph_cache_parity.json').read_text())
    assert parity['passed'], 'Eager graph cache parity must pass'
    return item


def metadata(batch, lengths, bi):
    d={k:[] for k in ['labels','dialogue_ids','utterance_indices','speaker_ids','batch_indices','dialogue_lengths']}
    for di,(key,n) in enumerate(zip(batch[-1],lengths)):
        d['labels'].extend(batch[8][di,:n].numpy().tolist())
        d['dialogue_ids'].extend([str(key)]*n);d['utterance_indices'].extend(range(n))
        d['speaker_ids'].extend(batch[6][:n,di].argmax(-1).numpy().tolist())
        d['batch_indices'].extend([bi]*n);d['dialogue_lengths'].extend([n]*n)
    return {k:np.asarray(v,dtype=str if k=='dialogue_ids' else np.int64) for k,v in d.items()}


def run(ds,phase):
    item=gate(ds,phase);seed=CFG['anchors'][ds]
    out=ROOT/f'runs/{ds}/seed{seed}';out.mkdir(parents=True,exist_ok=True)
    done=out/f'{phase}_complete.json';assert not done.exists(),'Do not overwrite completed job'
    write_json(ROOT/f'provenance/resources_{ds}_{phase}.json',old.resource_snapshot())
    api=old.imports();torch,ev,dl,Model,seed_everything=api;os.chdir(old.CODE)
    cfg=old.config_for(ds,seed);keys=cfg['provenance']['split']['validation_ids']
    dataset=dl.IEMOCAPDataset(keys=keys) if ds=='IEMOCAP' else dl.MELDDataset('data/MELD_features_raw1.pkl',keys=keys)
    assert list(dataset.keys)==keys and len(dataset)==item['validation_dialogues']
    with np.load(item['validation_baseline_path']) as f: base={k:f[k] for k in f.files}
    with np.load(item['train_statistics_path']) as f: stats={k:f[k] for k in f.files}
    missing_set={(x['dialogue_id'],x['utterance_index']) for x in item['all_zero_features']['visual']}
    missing=np.asarray([(str(k),int(u)) in missing_set for k,u in zip(base['dialogue_ids'],base['utterance_indices'])])
    baseline_path=out/'baseline_validation.npz'
    if not baseline_path.exists():np.savez_compressed(baseline_path,**base,visual_missing=missing)
    else:
        with np.load(baseline_path) as f:
            for k,v in base.items():assert np.array_equal(f[k],v),k
            assert np.array_equal(f['visual_missing'],missing)
    model=old.load_model(ds,seed,api)
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    cache=install_graph_cache(model.graph_model)
    guard=old.GraphGuard(model,torch)
    specs=condition_specs(phase);N,C=base['probabilities'].shape
    arrays={s['id']:np.full((N,s['repeats'],C),np.nan,dtype=np.float32) for s in specs}
    offsets=0;forwards=0;skip=0;checks=[];beg=time.monotonic();last=beg
    names={0:'text0',1:'text1',2:'text2',3:'text3',4:'visual',5:'audio'}
    for bi,batch in enumerate(old.loader_for(dataset,torch)):
        tensors=[x.cuda() for x in batch[:-1]]
        originals=[x.clone() for x in tensors]
        lengths=[int(x.sum().item()) for x in tensors[7]];count=sum(lengths)
        assert all(n>0 for n in lengths)
        md=metadata(batch,lengths,bi)
        for key,value in md.items():assert np.array_equal(value,base[key][offsets:offsets+count]),key
        guard.reset()
        with torch.no_grad():
            # Original instance methods provide the per-batch scientific reference.
            cache.restore();reference=old.forward(model,tensors,lengths).cpu().numpy()
            cache.enable();cached=old.forward(model,tensors,lengths).cpu().numpy()
            checks.append(dict(batch=bi,kind='original',log_error=compare(cached,reference),
                               probability_error=compare(np.exp(cached),np.exp(reference)),
                               previous_log_error=compare(reference,base['log_probabilities'][offsets:offsets+count]),
                               previous_probability_error=compare(np.exp(reference),base['probabilities'][offsets:offsets+count])))
            local=0
            for di,(dialogue_id,n) in enumerate(zip(batch[-1],lengths)):
                for ui in range(n):
                    row=offsets+local+ui
                    assert bool(np.all(batch[4][ui,di].numpy()==0))==bool(missing[row])
                    for s in specs:
                        target=s['target'];indices=target_indices(target)
                        if target=='V' and missing[row]:
                            arrays[s['id']][row]=base['log_probabilities'][row];skip+=s['repeats'];continue
                        vectors=[batch[j][ui,di].numpy() for j in indices]
                        means=[stats[names[j]+'_mean'] for j in indices]
                        for repeat in range(s['repeats']):
                            if s['kind']=='attenuation':changed=attenuate(vectors,means,s['strength'])
                            else:
                                rng=mask_rng(ds,seed,str(dialogue_id),ui,target,s['strength'],repeat)
                                changed,selected=mask_vectors(vectors,means,s['strength'],rng)
                            try:
                                for j,x in zip(indices,changed):tensors[j][ui,di].copy_(torch.from_numpy(x).cuda())
                                result=old.forward(model,tensors,lengths)
                                # Full-vector original/cached parity once per condition per batch.
                                if local+ui==0 and repeat==0:
                                    q=result.cpu().numpy();cache.restore()
                                    ref=old.forward(model,tensors,lengths).cpu().numpy();cache.enable()
                                    checks.append(dict(batch=bi,kind=s['id'],log_error=compare(q,ref),probability_error=compare(np.exp(q),np.exp(ref))))
                                arrays[s['id']][row,repeat]=result[local+ui].cpu().numpy();forwards+=1
                            finally:
                                for j in indices:tensors[j][ui,di].copy_(originals[j][ui,di])
                    now=time.monotonic()
                    if now-last>=30:
                        info=dict(dataset=ds,phase=phase,batch=bi,row=row,forwards=forwards,skipped=skip,seconds=now-beg,pid=os.getpid())
                        write_json(out/f'{phase}_progress.json',info);print(json.dumps(info),flush=True);last=now
                local+=n
            for j,(x,orig) in enumerate(zip(tensors,originals)):assert torch.equal(x,orig),f'Input {j} altered after target restoration'
            for k,v in model.state_dict().items():assert torch.equal(v.cpu(),state[k]),f'Model state changed: {k}'
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        offsets+=count
        # Recovery snapshots contain predictions only, never duplicate features.
        partial=out/'partial';partial.mkdir(exist_ok=True)
        for s in specs:np.savez_compressed(partial/(s['id']+'.npz'),log_probabilities=arrays[s['id']],completed_utterances=offsets)
        print(json.dumps(dict(dataset=ds,phase=phase,batch_complete=bi,utterances=offsets,seconds=time.monotonic()-beg)),flush=True)
        del tensors,originals
    assert offsets==N
    conditions=out/'conditions';conditions.mkdir(exist_ok=True)
    for s in specs:
        qlog=arrays[s['id']];assert np.isfinite(qlog).all()
        q=np.exp(qlog);assert np.allclose(q.sum(-1),1,atol=1e-6,rtol=1e-5)
        active=~missing if s['target']=='V' else np.ones(N,dtype=bool)
        if s['target']=='V':
            # Preserve stored original probabilities exactly for skipped inputs.
            q[missing]=base['probabilities'][missing,None,:]
        payload={k:v for k,v in base.items() if k not in ['probabilities','log_probabilities','predictions']}
        payload.update(original_predictions=base['predictions'],perturbed_predictions=q.argmax(-1),
            original_probabilities=base['probabilities'],probabilities=q,log_probabilities=qlog,
            active=active,visual_missing=missing,phase=np.asarray(phase),condition_id=np.asarray(s['id']),
            target=np.asarray(s['target']),kind=np.asarray(s['kind']),strength=np.asarray(s['strength']),repeats=np.asarray(s['repeats']))
        path=conditions/(s['id']+'.npz');assert not path.exists();np.savez_compressed(path,**payload)
    guard.close();cache.restore()
    result=dict(passed=True,dataset=ds,seed=seed,phase=phase,seconds=time.monotonic()-beg,
        target_forwards=forwards,skipped_visual_missing=skip,stored_vectors=sum(x.size//C for x in arrays.values()),
        complete_probability_parity_checks=checks,graph_guard_observations=guard.count,cache=cache.stats,
        inputs_restored_exact=True,parameters_and_buffers_unchanged=True,gradients_absent=True,
        test_inference=False,server=os.uname().nodename,pid=os.getpid(),gpu_visible=os.getenv('CUDA_VISIBLE_DEVICES'),
        config_sha256=sha(CONFIG),runner_sha256=sha(__file__),conditions={s['id']:sha(conditions/(s['id']+'.npz')) for s in specs})
    write_json(done,result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True,choices=list(CFG['anchors']))
    parser.add_argument('--phase',required=True,choices=list(CFG['phases']));args=parser.parse_args()
    try:run(args.dataset,args.phase)
    except Exception:
        failure=ROOT/f'logs/failure_{args.dataset}_{args.phase}_{int(time.time())}.txt'
        failure.write_text(traceback.format_exc());raise
