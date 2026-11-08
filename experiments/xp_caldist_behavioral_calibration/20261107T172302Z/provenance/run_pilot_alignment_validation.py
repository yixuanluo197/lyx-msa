"""Frozen XP04 GPU alignment and target-only validation pilot. No training paths."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

WORKSPACE = Path('/data2/yb/paper/lyx-msa')
BASE = WORKSPACE / 'CMERC/experiments/xp_mainline_validation/20261016T050823Z'
CODE = BASE / 'code'
SEEDS = [1475, 67137, 2024, 3407, 42]
PILOTS = {'IEMOCAP':1475, 'MELD':67137}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def compare_alignment(actual, expected):
    result={}
    for key in ['dialogue_ids','utterance_indices','labels','predictions']:
        result[key]={'equal':bool(np.array_equal(actual[key],expected[key]))}
    for key in ['probabilities','log_probabilities']:
        a,b=actual[key],expected[key]
        same=a.shape==b.shape
        result[key]={'equal':bool(same and np.isfinite(a).all() and np.allclose(a,b,atol=1e-6,rtol=1e-5)),
                     'max_abs_error':float(np.max(np.abs(a-b))) if same else None}
    result['argmax']={'equal':bool(np.array_equal(actual['probabilities'].argmax(1),expected['predictions']) and np.array_equal(actual['log_probabilities'].argmax(1),expected['predictions']))}
    return {'passed':all(x['equal'] for x in result.values()),'checks':result}


def imports():
    sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'scripts'))
    import torch
    import evaluate_checkpoint as ev
    import dataloader as dl
    from model import Model
    from train_cm import seed_everything
    torch.set_num_threads(1)
    return torch,ev,dl,Model,seed_everything


def config_for(ds,seed):
    return json.loads((BASE/f'configs/effective/{ds}_cmerc_validation_seed{seed}.json').read_text())


def dataset_for(ds,split,dl):
    cfg=config_for(ds,1475);keys=cfg['provenance']['split'][split+'_ids']
    dataset=dl.IEMOCAPDataset(keys=keys) if ds=='IEMOCAP' else dl.MELDDataset('data/MELD_features_raw1.pkl',keys=keys)
    assert list(dataset.keys)==keys
    return dataset


def loader_for(dataset,torch):
    return torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0,collate_fn=dataset.collate_fn)


def load_model(ds,seed,api):
    torch,ev,dl,Model,seed_everything=api
    seed_everything(seed)
    cp=BASE/f'runs/cmerc/{ds}_validation_seed{seed}/best.pt'
    checkpoint=torch.load(cp,map_location='cpu')
    dl.validate_checkpoint_protocol(checkpoint,causal=False,expected_args=config_for(ds,seed))
    model=Model(**ev._model_arguments(ev.dataset_config(ds),torch.device('cuda:0'))).cuda()
    ev.configure_evaluation_backend([model],checkpoint)
    model.load_state_dict(ev.extract_state_dict(checkpoint),strict=True)
    model.eval();model.requires_grad_(False)
    assert all(not m.training for m in model.modules())
    assert all(p.dtype==torch.float32 for p in model.parameters())
    del checkpoint
    return model


def forward(model,tensors,lengths):
    return model(tensors[:4],tensors[6],tensors[7],lengths,tensors[5],tensors[4],0)[0]


def infer(model,dataset,api):
    torch=api[0];parts=[];meta={k:[] for k in ['labels','dialogue_ids','utterance_indices','speaker_ids','batch_indices','dialogue_lengths']}
    with torch.no_grad():
        for bi,batch in enumerate(loader_for(dataset,torch)):
            tensors=[x.cuda() for x in batch[:-1]];lengths=[int(x.sum().item()) for x in tensors[7]]
            values=forward(model,tensors,lengths).cpu().numpy();parts.append(values)
            for di,(key,n) in enumerate(zip(batch[-1],lengths)):
                meta['labels'].extend(batch[8][di,:n].numpy().tolist());meta['dialogue_ids'].extend([str(key)]*n)
                meta['utterance_indices'].extend(range(n));meta['speaker_ids'].extend(batch[6][:n,di].argmax(-1).numpy().tolist())
                meta['batch_indices'].extend([bi]*n);meta['dialogue_lengths'].extend([n]*n)
    result={k:np.asarray(v,dtype=str if k=='dialogue_ids' else np.int64) for k,v in meta.items()}
    result['log_probabilities']=np.concatenate(parts);result['probabilities']=np.exp(result['log_probabilities']);result['predictions']=result['probabilities'].argmax(1)
    assert np.isfinite(result['log_probabilities']).all()
    assert np.allclose(result['probabilities'].sum(1),1,atol=1e-6)
    return result


def resource_snapshot():
    return dict(host=platform.node(),pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True),
        disks={p:dict(zip(['total','used','free'],shutil.disk_usage(p))) for p in ['/data1','/data2']},
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),python=sys.executable,python_version=sys.version,
        command=sys.argv,environment={k:os.environ.get(k) for k in ['LD_LIBRARY_PATH','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTORCH_CUDA_ALLOC_CONF']})


def freeze_inputs(root):
    origin=json.loads((BASE/'provenance/origin.json').read_text())['source_files_sha256']
    files={str(CODE/f):sha(CODE/f) for f in origin if (CODE/f).is_file()}
    imported = ['model.py','model_cm.py','dataloader.py','train_cm.py','model_hyper.py','model_GCN.py','HypergraphConv.py','causal_graph.py','high_fre_conv.py','clloss.py','evaluation_metrics.py','scripts/evaluate_checkpoint.py']
    mismatches={f:dict(expected=v,actual=files[str(CODE/f)],used_by_this_run=f in imported) for f,v in origin.items() if str(CODE/f) in files and files[str(CODE/f)]!=v}
    write_json(root/'provenance/origin_source_comparison.json',{'mismatches':mismatches,'enforced_imported_files':imported,'note':'Historical scheduling code is not imported or executed by this experiment.'})
    assert all(files[str(CODE/f)]==origin[f] for f in imported)
    for ds in PILOTS:
        cfg=config_for(ds,1475)
        for f,digest in cfg['provenance']['data'].items():
            path=CODE/f;actual=sha(path);assert actual==digest,(path,actual,digest);files[str(path)]=actual
        for seed in SEEDS:
            for path in [BASE/f'runs/cmerc/{ds}_validation_seed{seed}/best.pt',BASE/f'configs/effective/{ds}_cmerc_validation_seed{seed}.json',BASE/f'results/cmerc/{ds}/validation/seed{seed}/predictions.npz']:
                files[str(path)]=sha(path)
    write_json(root/'provenance/source_hashes.json',{'historical_inputs':files,'new_scripts':{str(p):sha(p) for p in (root/'scripts').glob('*.py')},'config_sha256':sha(root/'configs/perturbation_config.json')})
    return files


def align(root,api):
    evidence={};frozen=freeze_inputs(root)
    for ds in PILOTS:
        dataset=dataset_for(ds,'test',api[2])
        for seed in SEEDS:
            start=time.monotonic();model=load_model(ds,seed,api);actual=infer(model,dataset,api)
            reference=BASE/f'results/cmerc/{ds}/validation/seed{seed}/predictions.npz'
            with np.load(reference) as expected: result=compare_alignment(actual,expected)
            result.update(dataset=ds,seed=seed,reference=str(reference),seconds=time.monotonic()-start,
                          peak_allocated_mib=api[0].cuda.max_memory_allocated()/2**20)
            out=root/f'runs/{ds}/seed{seed}';out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/'test_alignment_predictions.npz',**actual)
            write_json(out/'alignment.json',result);evidence[f'{ds}_{seed}']=result
            print(json.dumps({'stage':'alignment',**result}),flush=True)
            del model;gc.collect();api[0].cuda.empty_cache()
            if not result['passed']:
                write_json(root/'results/alignment_summary.json',{'passed':False,'runs':evidence});raise RuntimeError('GPU alignment failed: all perturbations prohibited')
        del dataset;gc.collect()
    write_json(root/'results/alignment_summary.json',{'passed':True,'runs':evidence,'frozen_input_count':len(frozen)})
    lines=['# GPU 原预测对齐','', '全部10个checkpoint通过；FP32、eval、ATen、batch16、原始Test顺序。atol=1e-6，rtol=1e-5。','', '| Dataset | Seed | Prob max error | Log-prob max error | Passed |','|---|---:|---:|---:|---|']
    for r in evidence.values():lines.append(f"| {r['dataset']} | {r['seed']} | {r['checks']['probabilities']['max_abs_error']:.8g} | {r['checks']['log_probabilities']['max_abs_error']:.8g} | {r['passed']} |")
    lines.extend(['','逐项ID、话语索引、标签、prediction、argmax严格相同。完整向量逐元素通过容差。保存新Test概率只用于本次复现对齐，不参与任何Pilot诊断或拟合。'])
    (root/'reports/GPU_ALIGNMENT_REPORT.md').write_text('\n'.join(lines)+'\n')


def alignment_gate(root):
    d=json.loads((root/'results/alignment_summary.json').read_text());assert d['passed'] and len(d['runs'])==10
    for path,digest in json.loads((root/'provenance/source_hashes.json').read_text())['historical_inputs'].items():assert sha(path)==digest,path


def validation(root,api):
    alignment_gate(root);entries=[]
    for ds in PILOTS:
        cfg=config_for(ds,1475);split=cfg['provenance']['split'];sets=[set(split[k+'_ids']) for k in ['train','validation','test']]
        assert not any(sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)])
        dataset=dataset_for(ds,'validation',api[2]);expected=(12,527) if ds=='IEMOCAP' else (114,1109)
        assert len(dataset)==expected[0]
        for seed in SEEDS:
            model=load_model(ds,seed,api);actual=infer(model,dataset,api)
            assert len(actual['labels'])==expected[1]
            out=root/f'runs/{ds}/seed{seed}';np.savez_compressed(out/'validation_predictions.npz',**actual)
            entries.append(dict(dataset=ds,seed=seed,dialogues=len(dataset),utterances=len(actual['labels']),split_hash=split['hash'],accuracy=float(np.mean(actual['predictions']==actual['labels']))))
            print(json.dumps({'stage':'validation',**entries[-1]}),flush=True)
            del model;gc.collect();api[0].cuda.empty_cache()
        del dataset;gc.collect()
    write_json(root/'results/validation_summary.json',{'passed':True,'runs':entries})
    (root/'reports/VALIDATION_INFERENCE_REPORT.md').write_text('# 真正验证集原始推理\n\n全部10个checkpoint均已生成Validation概率。IEMOCAP为12对话/527话语；MELD为114对话/1109话语。按XP04 provenance中的validation_ids选取，保持原顺序，与train/test均不相交。\n\n每个run保存validation_predictions.npz，包含完整probability/log probability、标签、预测、dialogue_id、utterance_index、speaker_id、batch_index和对话长度。只有IEMOCAP1475与MELD67137进入本阶段扰动；其余seed没有执行扰动或正式实验。\n')


def training_stats(root,ds,api):
    out=root/f'provenance/{ds}_train_statistics.npz'
    if out.exists():
        with np.load(out) as z:return {k:z[k] for k in z.files}
    dataset=dataset_for(ds,'train',api[2]);collections=[dataset.videoAudio,dataset.videoVisual,dataset.roberta1,dataset.roberta2,dataset.roberta3,dataset.roberta4]
    stats={}
    for name,collection in zip(['audio','visual','text0','text1','text2','text3'],collections):
        x=np.concatenate([np.asarray(collection[k],dtype=np.float64) for k in dataset.keys])
        assert np.isfinite(x).all();stats[name+'_std']=x.std(axis=0,ddof=0).astype(np.float32);stats[name+'_mean']=x.mean(axis=0).astype(np.float32)
        stats[name+'_count']=np.asarray(len(x));del x
    np.savez_compressed(out,**stats);del dataset;gc.collect();return stats


class GraphGuard:
    """Observe graph indices without changing graph computation or outputs."""
    def __init__(self,model,torch):
        self.torch=torch;self.reference={};self.count=0;self.handles=[]
        for name,module in model.named_modules():
            if name.endswith('hyperconv1') or name.endswith('conv1') and 'graph_model' in name:
                self.handles.append(module.register_forward_pre_hook(self.hook(name)))
        assert self.handles,'No graph guard hooks found'
    def hook(self,name):
        def check(module,args):
            index=args[1]
            if name not in self.reference:self.reference[name]=index.detach().clone()
            else:assert self.torch.equal(index,self.reference[name]),'Graph structure changed: '+name
            self.count+=1
        return check
    def reset(self):self.reference={}
    def close(self):
        for h in self.handles:h.remove()


def pilot(root,ds,modality,api):
    from pilot_core import rng_for,perturb_vectors
    alignment_gate(root);torch=api[0];seed=PILOTS[ds]
    order=['audio','visual','text_representation'];idx=order.index(modality)
    for prior in order[:idx]:
        for pds,pseed in PILOTS.items():assert (root/f'runs/{pds}/seed{pseed}/{prior}_complete.json').exists(),'Previous modality incomplete'
    out=root/f'runs/{ds}/seed{seed}';dest=out/'perturbations';dest.mkdir(exist_ok=True)
    assert not (out/f'{modality}_complete.json').exists(),'Refusing to overwrite completed pilot'
    with np.load(out/'validation_predictions.npz') as z:original={k:z[k] for k in z.files}
    dataset=dataset_for(ds,'validation',api[2]);stats=training_stats(root,ds,api);model=load_model(ds,seed,api)
    state_before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};guard=GraphGuard(model,torch)
    indices={'audio':[5],'visual':[4],'text_representation':[0,1,2,3]}[modality]
    stds=[stats[k+'_std'] for k in ({'audio':['audio'],'visual':['visual'],'text_representation':['text0','text1','text2','text3']}[modality])]
    specs=[('gaussian',.05,5),('dropout',.05,5),('attenuation',.95,1)]
    n,c=original['probabilities'].shape;outputs={kind:np.empty((n,reps,c),np.float32) for kind,_,reps in specs};counts={kind:0 for kind,_,_ in specs}
    changed={kind:0 for kind,_,_ in specs};total=0;rowbase=0;start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for bi,batch in enumerate(loader_for(dataset,torch)):
            tensors=[x.cuda() for x in batch[:-1]];lengths=[int(x.sum().item()) for x in tensors[7]]
            baseline=[x.clone() for x in tensors];guard.reset();raw=forward(model,tensors,lengths).cpu().numpy();size=sum(lengths)
            assert np.allclose(raw,original['log_probabilities'][rowbase:rowbase+size],atol=1e-6,rtol=1e-5),'Validation replay mismatch'
            offset=0
            for di,(key,length) in enumerate(zip(batch[-1],lengths)):
                for ui in range(length):
                    row=rowbase+offset+ui
                    assert original['dialogue_ids'][row]==str(key) and original['utterance_indices'][row]==ui
                    vectors=[batch[j][ui,di].numpy().copy() for j in indices]
                    for kind,strength,reps in specs:
                        for repeat in range(reps):
                            rng=rng_for(ds,seed,str(key),ui,modality,kind,strength,repeat)
                            values=perturb_vectors(vectors,stds,kind,strength,rng)
                            changed[kind]+=int(any(not np.array_equal(a,b) for a,b in zip(vectors,values)))
                            try:
                                for j,v in zip(indices,values):tensors[j][ui,di].copy_(torch.from_numpy(v).to(tensors[j].device))
                                # Only a target row is ever assigned; all other tensors retain original storage and values.
                                outputs[kind][row,repeat]=forward(model,tensors,lengths)[offset+ui].cpu().numpy()
                            finally:
                                for j in indices:tensors[j][ui,di].copy_(baseline[j][ui,di])
                            counts[kind]+=1;total+=1
                    if total%352==0:
                        print(json.dumps(dict(stage='pilot',dataset=ds,modality=modality,targets=row+1,total_targets=n,forwards=total,seconds=time.monotonic()-start,seconds_per_forward=(time.monotonic()-start)/total)),flush=True)
                offset+=length
            assert all(torch.equal(x,y) for x,y in zip(tensors,baseline)),'Inputs, masks or padding changed after restore'
            rowbase+=size
    assert rowbase==n
    assert all(torch.equal(v.detach().cpu(),state_before[k]) for k,v in model.state_dict().items()),'Model state changed'
    metadata={k:v for k,v in original.items() if k not in ['probabilities','log_probabilities']}
    for kind,strength,reps in specs:
        values=outputs[kind];assert np.isfinite(values).all() and counts[kind]==n*reps
        path=dest/f'{modality}_{kind}.npz';assert not path.exists()
        np.savez_compressed(path,**metadata,log_probabilities=values,probabilities=np.exp(values),original_probabilities=original['probabilities'],strength=np.asarray(strength),repeats=np.asarray(reps),modality=np.asarray(modality),perturbation_type=np.asarray(kind))
    evidence=dict(passed=True,dataset=ds,seed=seed,modality=modality,targets=n,forwards=total,counts=counts,changed_target_counts=changed,
                  seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  graph_index_hook_checks=guard.count,unchanged_model_state=True,restored_full_input_exact=True,padding_untouched=True,
                  statistics_split='train',batch_size=16,parallel_targets=False,resource=resource_snapshot())
    guard.close();write_json(out/f'{modality}_complete.json',evidence)
    print(json.dumps({'stage':'pilot_complete',**evidence}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--stage',choices=['align','validation','pilot'],required=True)
    parser.add_argument('--dataset',choices=list(PILOTS));parser.add_argument('--modality',choices=['audio','visual','text_representation']);args=parser.parse_args();root=args.root.resolve()
    write_json(root/f'provenance/resource_{args.stage}_{args.dataset or "all"}_{args.modality or "none"}_{int(time.time())}.json',resource_snapshot())
    assert shutil.disk_usage('/data2').free>5*1024**3,'Insufficient disk headroom'
    os.chdir(CODE);api=imports();torch=api[0];assert torch.__version__=='1.13.1+cu117';assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    try:
        if args.stage=='align':align(root,api)
        elif args.stage=='validation':validation(root,api)
        else:pilot(root,args.dataset,args.modality,api)
    except BaseException as e:
        write_json(root/f'logs/failure_{args.stage}_{args.dataset}_{args.modality}_{int(time.time())}.json',dict(error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
