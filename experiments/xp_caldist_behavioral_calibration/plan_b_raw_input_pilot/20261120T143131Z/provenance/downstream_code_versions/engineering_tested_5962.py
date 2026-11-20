"""New-feature CMERC adapter; frozen XP04 equations, no old features/weights.

CLI ACTION --dataset IEMOCAP|MELD --data-dir DATASET_DIR [--device cuda:0]
ACTION: train_m3net, build_difficulty, train_cmerc, evaluate, smoke.
DATASET_DIR/dataset.pt contains splits=train/validation dialogue lists plus
fit_dialogue_ids and inner_dev_dialogue_ids. features/{text,audio}_features.pt
are split -> str(dialogue_id) -> float [N,4,1024] / [N,768]. Labels, visual and
speaker come from dataset.pt; missing audio must already be zero768 and listed
in its independent extraction manifest. Evaluation never changes BN statistics.
"""
from __future__ import annotations
import argparse, copy, fcntl, hashlib, json, os, random, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score

HERE=Path(__file__).resolve().parent
CORE=HERE/'core_snapshot'
for _source in CORE.glob('*.py'):
    _cached=sys.modules.get(_source.stem)
    if _cached is not None and Path(getattr(_cached,'__file__','')).resolve()!=_source.resolve():
        raise RuntimeError('Conflicting preimported module; frozen core required: '+_source.stem)
sys.path.insert(0,str(CORE))
import train_cm as official
from model_cm import Model, Model_Intra, Model_Inter, FocalLoss
from dataloader import select_curriculum_keys


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def dump(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def save(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)


def rng(): return official._rng_state()
def restore(state): official._restore_rng_state(state)


def metrics(prob,labels):
    prob=np.asarray(prob,dtype=np.float64);labels=np.asarray(labels,dtype=int)
    if len(prob)!=len(labels) or not np.isfinite(prob).all():raise ValueError('nonfinite/mismatched predictions')
    pred=prob.argmax(1);conf=prob.max(1);correct=pred==labels
    bins=np.minimum((conf*15).astype(int),14);ece=0.;bin_rows=[]
    for i in range(15):
        hit=bins==i;n=int(hit.sum())
        acc=float(correct[hit].mean()) if n else None;c=float(conf[hit].mean()) if n else None
        if n:ece+=n/len(labels)*abs(acc-c)
        bin_rows.append({'bin':i,'n':n,'accuracy':acc,'confidence':c})
    onehot=np.eye(prob.shape[1])[labels]
    return {'n':len(labels),'accuracy':float(accuracy_score(labels,pred)),
      'macro_f1':float(f1_score(labels,pred,labels=list(range(prob.shape[1])),average='macro',zero_division=0)),
      'weighted_f1':float(f1_score(labels,pred,labels=list(range(prob.shape[1])),average='weighted',zero_division=0)),
      'ece15':float(ece),'brier_sum_classes':float(np.square(prob-onehot).sum(1).mean()),
      'nll':float(-np.log(np.clip(prob[np.arange(len(labels)),labels],1e-12,1)).mean()),'bins':bin_rows}


def anchor(dataset):
    # Resolve from workspace rather than assuming the adapter's run depth.
    root=next(p for p in HERE.parents if (p/'CMERC').is_dir())/'CMERC/experiments/xp_mainline_validation/20261016T050823Z/configs/effective'
    seed=1475 if dataset=='IEMOCAP' else 67137
    path=root/f'{dataset}_cmerc_validation_seed{seed}.json'
    raw=json.loads(path.read_text())
    return {k:v for k,v in raw.items() if k not in ('provenance','output_dir','resume','testing')},str(path),sha(path)


def config(dataset,visual_dim,speaker_dim,stage):
    c,source,h=anchor(dataset)
    c.update(audio_dim=768,text_dim=1024,text_layers=4,visual_dim=int(visual_dim),speaker_dim=int(speaker_dim),
      causal=False,calibrate=stage=='cmerc',contrastlearning=stage=='cmerc',courselearning=stage=='cmerc',
      selection_protocol='effectiveTrain_inner_dialogue_dev',selection_metric='macro_f1',patience=10,
      stage=stage,anchor_config_sha256=h,anchor_config=source,old_feature_loading=False,
      initialization='fresh_random_no_M3Net_warmstart',rnn_backend='aten',rng_policy='continuous')
    c['class_weight_source']='IEMOCAP fixed XP04 historical reciprocal frequencies; MELD original FocalLoss; no new Validation estimation'
    return c


def make_model(c,device,cls=Model):
    if torch.device(device).type!='cuda':raise ValueError('Unmodified XP04 HyperGCN requires CUDA; set healthy CUDA_VISIBLE_DEVICES explicitly')
    torch.cuda.set_device(torch.device(device))
    dg=512 if c['Dataset']=='IEMOCAP' else 1024
    model=cls('GRU',1024,dg,150,100,100,100,512,n_speakers=c['speaker_dim'],max_seq_len=200,
      window_past=c['windowp'],window_future=c['windowf'],n_classes=6 if c['Dataset']=='IEMOCAP' else 7,
      listener_state=c['active_listener'],context_attention=c['attention'],dropout=c['dropout'],
      nodal_attention=c['nodal_attention'],no_cuda=False,graph_type=c['graph_type'],use_topic=False,
      alpha=c['alpha'],multiheads=c['multiheads'],graph_construct=c['graph_construct'],use_GCN=c['use_gcn'],
      use_residue=c['use_residue'],D_m_v=c['visual_dim'],D_m_a=768,modals=c['modals'],
      att_type=c['mm_fusion_mthd'],av_using_lstm=c['av_using_lstm'],Deep_GCN_nlayers=c['Deep_GCN_nlayers'],
      dataset=c['Dataset'],use_speaker=c['use_speaker'],use_modal=c['use_modal'],norm=c['norm'],
      num_L=c['num_L'],num_K=c['num_K'],causal=False)
    official.configure_recurrent_backend(model,'aten')
    return model.to(device)


def load_model(dataset_dir,checkpoint,device='cuda:0'):
    payload=torch.load(checkpoint,map_location='cpu')
    if payload.get('new_features_only') is not True:raise ValueError('Refusing an old/non-Plan-B checkpoint')
    model=make_model(payload['effective_config'],device)
    model.load_state_dict(payload['model_state_dict']);model.eval()
    return model


def batch(dialogues,texts,audios,device,labels=True):
    lengths=[len(d['utterance_ids']) for d in dialogues]
    if any(n<1 for n in lengths):raise ValueError('empty dialogue')
    ts=[torch.as_tensor(x,dtype=torch.float32) for x in texts]
    aa=[torch.as_tensor(x,dtype=torch.float32) for x in audios]
    for n,t,a,d in zip(lengths,ts,aa,dialogues):
        if t.shape!=(n,4,1024) or a.shape!=(n,768):raise ValueError('Feature shape mismatch')
        if not all(torch.isfinite(x).all() for x in (t,a,d['visual'],d['speaker'])):raise ValueError('nonfinite feature')
        if not torch.allclose(d['speaker'].sum(1),torch.ones(n)):raise ValueError('speaker must be one-hot')
    layers=[pad_sequence([t[:,i] for t in ts]).to(device) for i in range(4)]
    v=pad_sequence([d['visual'].float() for d in dialogues]).to(device)
    a=pad_sequence(aa).to(device);q=pad_sequence([d['speaker'].float() for d in dialogues]).to(device)
    mask=pad_sequence([torch.ones(n) for n in lengths],batch_first=True).to(device)
    lab=pad_sequence([d['labels'].long() if labels else torch.zeros(n,dtype=torch.long) for d,n in zip(dialogues,lengths)],batch_first=True).to(device)
    return (*layers,v,a,q,mask,lab,[str(d['dialogue_id']) for d in dialogues])


@torch.no_grad()
def infer_dialogue(model,dialogue,textfeatures,audiofeatures):
    model.eval();device=next(model.parameters()).device
    data=batch([dialogue],[textfeatures],[audiofeatures],device,labels=False)
    n=len(dialogue['utterance_ids'])
    output=model(list(data[:4]),data[6],data[7],[n],data[5],data[4],0)[0]
    return output.exp().cpu().numpy()


class Bundle:
    def __init__(self,root):
        self.root=Path(root);self.data=torch.load(self.root/'dataset.pt',map_location='cpu')
        if set(self.data['splits'])!={'train','validation'}:raise ValueError('Only Train/Validation may exist')
        self.text=torch.load(self.root/'features/text_features.pt',map_location='cpu')
        self.audio=torch.load(self.root/'features/audio_features.pt',map_location='cpu')
        self.train={str(d['dialogue_id']):d for d in self.data['splits']['train']}
        self.fit=[str(x) for x in self.data['fit_dialogue_ids']];self.dev=[str(x) for x in self.data['inner_dev_dialogue_ids']]
        if set(self.fit)&set(self.dev) or set(self.fit)|set(self.dev)!=set(self.train):raise ValueError('inner split leakage/incomplete')
    def features(self,split,did):return self.text[split][str(did)],self.audio[split][str(did)]
    def inputs(self):
        paths=[self.root/'dataset.pt',self.root/'features/text_features.pt',self.root/'features/audio_features.pt']
        paths+=sorted((self.root/'features').glob('*manifest*.json'))
        return {str(p):sha(p) for p in paths}
    def loader(self,keys,seed,epoch,device,batch_size):
        gen=torch.Generator().manual_seed(seed+epoch*100003)
        sampler=official.CapacitySafeBatchSampler([len(self.train[k]['utterance_ids']) for k in keys],batch_size,gen)
        return [batch([self.train[keys[i]] for i in indices],*[list(values) for values in zip(*[self.features('train',keys[i]) for i in indices])],device) for indices in sampler]


def evaluate(model,bundle,dialogues,split):
    probs=[];ys=[];records=[]
    with official.preserve_rng_for_evaluation(True):
        for d in dialogues:
            did=str(d['dialogue_id']);p=infer_dialogue(model,d,*bundle.features(split,did));probs.append(p);ys.extend(d['labels'].tolist())
            for uid,y,prob in zip(d['utterance_ids'],d['labels'].tolist(),p):records.append({'dialogue_id':did,'utterance_id':str(uid),'label':y,'probability':prob.tolist()})
    return metrics(np.concatenate(probs),ys),records


def provenance(c,b):
    return {'inputs':b.inputs(),'code':{str(p):sha(p) for p in [Path(__file__),*sorted(CORE.glob('*.py'))]},'effective_config':c}


def train(b,dataset,stage,device,resume=None,max_epochs=None):
    expected=(5283,527) if dataset=='IEMOCAP' else (9989,1109)
    actual=tuple(sum(len(d['utterance_ids']) for d in b.data['splits'][s]) for s in ('train','validation'))
    if actual!=expected:raise ValueError('effective Train/Validation size mismatch: '+str(actual))
    first=next(iter(b.train.values()));c=config(dataset,first['visual'].shape[1],first['speaker'].shape[1],stage)
    if max_epochs is not None:c['epochs']=max_epochs
    run=b.root/'runs'/stage
    if resume is None:run.mkdir(parents=True,exist_ok=False)
    elif Path(resume).resolve().parent!=run.resolve():raise ValueError('resume must belong to exact stage directory')
    lock=(run/'.training.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    difficulty=None
    prov=provenance(c,b)
    if stage=='cmerc':
        difficulty_path=b.root/'new_difficulty.pt'
        obj=torch.load(difficulty_path,map_location='cpu');difficulty=obj['difficulties']
        if (obj.get('new_features_only') is not True or obj['input_hashes']!=b.inputs()
                or set(difficulty)!=set(b.fit)
                or obj['m3net_checkpoint_sha256']!=sha(b.root/'runs/m3net/best.pt')):
            raise ValueError('stale/incomplete/non-new difficulty')
        prov['difficulty_sha256']=sha(difficulty_path)
    official.seed_everything(c['seed']);model=make_model(c,device)
    peers=[make_model(c,device,cls) for cls in (Model,Model_Intra,Model_Inter)]
    optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'],weight_decay=c['l2'])
    loss=FocalLoss() if dataset=='MELD' else torch.nn.NLLLoss(torch.tensor([1/.086747,1/.144406,1/.227883,1/.160585,1/.127711,1/.252668],device=device))
    official.args=SimpleNamespace(**c)
    start=0;best=-1.;best_epoch=-1;bad=0
    if resume:
        old=torch.load(resume,map_location='cpu')
        if old['provenance']!=prov:raise ValueError('resume input/code/config hashes differ')
        model.load_state_dict(old['model_state_dict']);optimizer.load_state_dict(old['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(device)
        restore(old['rng_state']);start=old['epoch']+1;best=old['best_metric'];best_epoch=old['best_epoch'];bad=old['bad_epochs']
        epochs=[json.loads(line)['epoch'] for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
        if epochs!=list(range(start)):raise ValueError('checkpoint/metric epoch mismatch; refusing ambiguous resume')
    else:dump(run/'effective_config.json',c);dump(run/'provenance.json',prov)
    finite_checks=[];finite_handles=[]
    def check_raw_scl(module,inputs,output):
        ok=bool(torch.isfinite(output).all());finite_checks.append(ok)
        if not ok:
            dump(run/'nonfinite_failure.json',{'component':'CKSCL_base raw logits before official fallback','epoch':epoch,'checks':finite_checks})
            raise FloatingPointError('nonfinite SCL before official fallback; not silently zeroed')
    for component in (official.CKSCL_m,official.CKSCL_c):
        for module in component.modules():
            if module.__class__.__name__=='CKSCL_base':finite_handles.append(module.register_forward_hook(check_raw_scl))
    for epoch in range(start,c['epochs']):
        finite_checks.clear()
        ratio=min(1,round(c['epoch_ratio']*int(epoch/c['scheduler_steps']+1),2)) if c['courselearning'] else -1
        keys=select_curriculum_keys(b.fit,difficulty,ratio) if difficulty is not None else list(b.fit)
        loader=b.loader(keys,c['seed'],epoch,device,c['batch_size'])
        output=official.train_or_eval_graph_model(model,*peers,loss,loader,epoch,True,c['modals'],optimizer=optimizer,train=True,dataset=dataset,return_components=True)
        score,_=evaluate(model,b,[b.train[k] for k in b.dev],'train');value=score['macro_f1']
        improved=value>best
        if improved:best=value;best_epoch=epoch;bad=0
        else:bad+=1
        record={'epoch':epoch,'train_loss':output[0],'train_weighted_f1_percent':output[4],'loss_components':output[-1],'inner_dev':score,'ratio':ratio,'selected_dialogues':len(keys),'best_epoch':best_epoch,'bad_epochs':bad,'scl_raw_finite_checks':len(finite_checks),'official_nonfinite_fallback_count':sum(not x for x in finite_checks)}
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        payload={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'rng_state':rng(),'epoch':epoch,'best_metric':best,'best_epoch':best_epoch,'bad_epochs':bad,'effective_config':c,'provenance':prov,'new_features_only':True,'stage':stage}
        save(run/'last.pt',payload)
        if improved:save(run/'best.pt',payload)
        print(json.dumps({'stage':stage,'epoch':epoch,'inner_dev_macro_f1':value,'best':best}),flush=True)
        if bad>=c['patience']:break
    lock.close()
    for handle in finite_handles:handle.remove()
    return run/'best.pt'


def build_difficulty(b,device):
    path=b.root/'runs/m3net/best.pt';model=load_model(b.root,path,device);c=torch.load(path,map_location='cpu')['effective_config']
    peers=[make_model(c,device,cls) for cls in (Model_Intra,Model_Inter)]
    for peer in peers:peer.load_state_dict(model.state_dict());peer.eval()
    result={}
    with torch.no_grad():
        for did in b.fit:
            d=b.train[did]
            data=batch([d],*[[x] for x in b.features('train',did)],device)
            t=list(data[:4]);v,a,q,mask,labels=data[4:9];n=len(d['utterance_ids'])
            full=model(t,q,mask,[n],a,v,0)[0];removed=[]
            mt=[x.mean(0,keepdim=True).expand_as(x) for x in t]
            removed.append(model(mt,q,mask,[n],a,v,0)[0]);removed.append(model(t,q,mask,[n],a.mean(0,keepdim=True).expand_as(a),v,0)[0]);removed.append(model(t,q,mask,[n],a,v.mean(0,keepdim=True).expand_as(v),0)[0])
            removed.extend(peer(t,q,mask,[n],a,v,0)[0] for peer in peers)
            row=torch.arange(n,device=device);y=labels[0,:n]
            gap=sum(torch.clamp(x[row,y]-full[row,y],min=0).sum() for x in removed)
            speakers=q[:n,0].nonzero()[:,1].max()+1
            result[did]=float(((gap+speakers)/(n+speakers)).cpu())
    save(b.root/'new_difficulty.pt',{'difficulties':result,'scope':'effectiveTrain Fit only; inner-dev excluded','input_hashes':b.inputs(),'m3net_checkpoint_sha256':sha(path),'new_features_only':True,'formula':'(sum positive true-class logprob gaps over 3 modality + 2 context removals + maxSpeakerIndex+1)/(length+maxSpeakerIndex+1)'})
    return result


def smoke(b,dataset,device):
    d=next(iter(b.train.values()));c=config(dataset,d['visual'].shape[1],d['speaker'].shape[1],'m3net');official.seed_everything(c['seed']);m=make_model(c,device)
    data=batch([d],*[[x] for x in b.features('train',str(d['dialogue_id']))],device)
    before=m.linear_a.weight.detach().clone();m.train();out=m(list(data[:4]),data[6],data[7],[len(d['utterance_ids'])],data[5],data[4],0)[0]
    loss=torch.nn.functional.nll_loss(out,data[8][0]);loss.backward();grad=m.linear_a.weight.grad
    if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum()>0:raise RuntimeError('no finite audio gradient')
    torch.optim.Adam(m.parameters(),lr=1e-4).step()
    if torch.equal(before,m.linear_a.weight):raise RuntimeError('parameter did not update')
    p=infer_dialogue(m,d,*b.features('train',str(d['dialogue_id'])));p2=infer_dialogue(m,d,*b.features('train',str(d['dialogue_id'])))
    assert np.allclose(p,p2,atol=1e-6,rtol=1e-6) and np.allclose(p.sum(1),1,atol=1e-6)
    # Real two-dialogue padding and all official auxiliary gradients.
    selected=list(b.train.values())[:2]
    inputs=[b.features('train',str(item['dialogue_id'])) for item in selected]
    padded=batch(selected,*[list(x) for x in zip(*inputs)],device)
    c=config(dataset,d['visual'].shape[1],d['speaker'].shape[1],'cmerc')
    official.args=SimpleNamespace(**c)
    peers=[make_model(c,device,cls) for cls in (Model,Model_Intra,Model_Inter)]
    aux={};base_checks=[];handles=[]
    for component in (official.CKSCL_m,official.CKSCL_c):
        for module in component.modules():
            if module.__class__.__name__=='CKSCL_base':
                handles.append(module.register_forward_hook(lambda module,args,out:base_checks.append(bool(torch.isfinite(out).all()))))
    original=official.combine_training_losses
    def capture(classification_loss,**kw):
        for name in ('cc_loss','mscl_loss','cscl_loss'):
            value=kw[name]
            finite=bool(torch.isfinite(value).all()) if torch.is_tensor(value) else np.isfinite(value)
            gradient=torch.autograd.grad(value,m.linear_a.weight,retain_graph=True,allow_unused=True)[0] if torch.is_tensor(value) and value.requires_grad else None
            aux[name]={'value':float(value.detach()) if torch.is_tensor(value) else float(value),'finite_before_combination':finite,'requires_grad':bool(torch.is_tensor(value) and value.requires_grad),'audio_gradient_norm':float(gradient.norm()) if gradient is not None else None}
        return original(classification_loss,**kw)
    official.combine_training_losses=capture
    try:
        objective=FocalLoss() if dataset=='MELD' else torch.nn.NLLLoss()
        official.train_or_eval_graph_model(m,*peers,objective,[padded],0,True,'avl',optimizer=torch.optim.Adam(m.parameters(),lr=1e-4),train=True,dataset=dataset)
    finally:
        official.combine_training_losses=original
        for handle in handles:handle.remove()
    if any(not x['finite_before_combination'] or x['value']==0 or not x['requires_grad'] or not x['audio_gradient_norm'] or not np.isfinite(x['audio_gradient_norm']) for x in aux.values()):raise RuntimeError('auxiliary finite/nonzero/gradient smoke failed: '+str(aux))
    if not all(base_checks):raise RuntimeError('nonfinite SCL before official fallback')
    buffers={name:value.clone() for name,value in m.named_buffers()}
    p=infer_dialogue(m,d,*b.features('train',str(d['dialogue_id'])))
    if len(selected)>1:infer_dialogue(m,selected[1],*b.features('train',str(selected[1]['dialogue_id'])))
    p2=infer_dialogue(m,d,*b.features('train',str(d['dialogue_id'])))
    assert np.allclose(p,p2,atol=1e-6,rtol=1e-6)
    assert all(torch.equal(buffers[name],value) for name,value in m.named_buffers())
    report={'gradient_finite':True,'audio_parameters_updated':True,'mask_real_count':int(padded[7].sum()),'label_count':sum(len(x['labels']) for x in selected),'padding_positions':int(padded[7].numel()-padded[7].sum()),'eval_bn_buffers_unchanged':True,'eval_other_dialogue_probability_maxabs':float(np.max(np.abs(p-p2))),'eval_numerical_tolerance':1e-6,'probability_shape':list(p.shape),'formal_training':False,'synthetic_engineering_smoke':getattr(b,'synthetic',False),'auxiliary':aux,'scl_raw_logits_finite_before_fallback':base_checks,'official_nonfinite_fallback_count':sum(not x for x in base_checks),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_name':torch.cuda.get_device_name(),'adapter_sha256':sha(__file__),'effective_config':c}
    dump(b.root/'smoke.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['train_m3net','build_difficulty','train_cmerc','evaluate','smoke'])
    parser.add_argument('--dataset',choices=['IEMOCAP','MELD'],required=True);parser.add_argument('--data-dir',required=True);parser.add_argument('--device',default='cuda:0');parser.add_argument('--resume');parser.add_argument('--checkpoint');parser.add_argument('--max-epochs',type=int)
    args=parser.parse_args()
    if args.action=='smoke' and not (Path(args.data_dir)/'features/audio_features.pt').exists():
        b=SimpleNamespace(root=Path(args.data_dir),train={},synthetic=True)
        b.root.mkdir(parents=True,exist_ok=True);features={}
        gen=torch.Generator().manual_seed(424242)
        for i,n in enumerate((4,2)):
            did='engineering_'+str(i);s=2 if args.dataset=='IEMOCAP' else 9
            b.train[did]={'dialogue_id':did,'utterance_ids':[did+'_'+str(j) for j in range(n)],'speaker':torch.eye(s)[torch.arange(n)%s],'labels':torch.arange(n)%(6 if args.dataset=='IEMOCAP' else 7),'visual':torch.randn(n,16,generator=gen)}
            features[did]=(torch.randn(n,4,1024,generator=gen),torch.randn(n,768,generator=gen))
        b.features=lambda split,did:features[did]
    else:b=Bundle(args.data_dir)
    if args.action.startswith('train_'):train(b,args.dataset,args.action.split('_')[1],args.device,args.resume,args.max_epochs)
    elif args.action=='build_difficulty':build_difficulty(b,args.device)
    elif args.action=='smoke':print(smoke(b,args.dataset,args.device))
    else:
        checkpoint=args.checkpoint or str(b.root/'runs/cmerc/best.pt');model=load_model(b.root,checkpoint,args.device)
        summary,records=evaluate(model,b,b.data['splits']['validation'],'validation')
        dump(b.root/'validation_metrics.json',summary);save(b.root/'validation_predictions.pt',records)

if __name__=='__main__':main()
