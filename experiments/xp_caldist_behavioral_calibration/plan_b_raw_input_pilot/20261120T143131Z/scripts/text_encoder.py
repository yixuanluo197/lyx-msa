#!/usr/bin/env python3
"""Plan B: supervised public RoBERTa-large training and deterministic text extraction.

No cached XP04 text vectors or distilled student weights are consumed. Official
Validation participates only in the post-training extraction operation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback

import numpy as np
import torch
import transformers
from transformers import (RobertaConfig, RobertaForSequenceClassification,
                          RobertaTokenizer, get_linear_schedule_with_warmup)

PUBLIC_WEIGHT_SHA = '047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38'
SEEDS = {'IEMOCAP': 1475, 'MELD': 67137}
ACTIVE_OUTPUT = None


def sha256(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def preprocessing(text, dataset):
    """COSMIC reference extraction: MELD ASCII removal; IEMOCAP original strings."""
    if not isinstance(text, str):
        raise TypeError('Original utterance text must be a string')
    if dataset not in SEEDS:
        raise ValueError('A supported dataset is required for fixed preprocessing')
    return text.encode('ascii', errors='ignore').decode('utf-8') if dataset == 'MELD' else text


def tokenize(tokenizer, texts, dataset):
    records, statistics = [], []
    for text in texts:
        processed = preprocessing(text, dataset)
        full = tokenizer.encode(processed, add_special_tokens=True, truncation=False)
        tokens = tokenizer.encode(processed, add_special_tokens=True, truncation=True,
                                  max_length=512)
        if len(tokens) > 512 or tokens[0] != 0 or tokens[-1] != 2:
            raise ValueError('Expected BOS=0, EOS=2 and maximum 512 total tokens')
        records.append(tokens)
        statistics.append(dict(tokens_before=len(full), tokens_after=len(tokens),
                               truncated=len(full)>512, empty_after_processing=not processed.strip(),
                               ascii_changed=processed!=text))
    return records, statistics


def collate_token_ids(tokens, device, labels=None):
    length = max(len(row) for row in tokens)
    ids = torch.ones((len(tokens), length), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(tokens):
        ids[index, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, :len(row)] = 1
    batch = dict(input_ids=ids.to(device), attention_mask=mask.to(device))
    if labels is not None:
        batch['labels'] = torch.as_tensor(labels, dtype=torch.long, device=device)
    return batch


def load_encoder(model_path, device='cuda:0'):
    """Load this run's supervised classifier backbone and its exact saved tokenizer.

    Returns (RobertaModel, RobertaTokenizer). model_path may be the training output
    directory or its best_classifier child. No public/old-student fallback occurs.
    """
    path = Path(model_path).resolve()
    if (path / 'best_classifier').is_dir():
        if not (path / 'best_model_lock.json').is_file():
            raise RuntimeError('Training output has not locked its selected model')
        path = path / 'best_classifier'
    classifier, loading = RobertaForSequenceClassification.from_pretrained(
        path, local_files_only=True, output_loading_info=True, torch_dtype=torch.float32)
    if any(loading.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError(f'Incomplete supervised classifier: {loading}')
    if classifier.config.planb_dataset not in SEEDS:
        raise ValueError('Missing Plan B dataset identity')
    tokenizer = RobertaTokenizer.from_pretrained(path, local_files_only=True)
    encoder = classifier.roberta.to(device, dtype=torch.float32).eval()
    encoder._planb_dataset = classifier.config.planb_dataset
    del classifier
    return encoder, tokenizer


@torch.no_grad()
def extract_texts(model, tokenizer, texts, dataset=None, batch_size=4):
    """Shared clean/perturbed API: CPU FP32 [N,4,1024], last four BOS vectors.

    Same scalar preprocessing/tokenization/padding is used by clean and perturbed
    callers. Inputs are independent utterances; no context concatenation occurs.
    """
    dataset = dataset or getattr(model, '_planb_dataset', getattr(model.config,'planb_dataset',None))
    if batch_size < 1:
        raise ValueError('Positive extraction batch size required')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Extraction requires FP32 model parameters')
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = next(model.parameters()).device
    encoded, _ = tokenize(tokenizer, texts, dataset)
    result=[]
    for start in range(0,len(encoded),batch_size):
        batch=collate_token_ids(encoded[start:start+batch_size],device)
        hidden=model(**batch,output_hidden_states=True,return_dict=True).hidden_states
        values=torch.stack([hidden[-layer][:,0,:] for layer in (1,2,3,4)],dim=1).float()
        if values.shape[1:]!=(4,1024) or not torch.isfinite(values).all():
            raise ValueError('Non-finite or unexpected four-layer BOS output')
        result.append(values.cpu())
    return torch.cat(result) if result else torch.empty((0,4,1024),dtype=torch.float32)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])


def get_training_rows(data, tokenizer):
    train={str(item['dialogue_id']):item for item in data['splits']['train']}
    if len(train)!=len(data['splits']['train']):
        raise ValueError('Duplicate Train Dialogue IDs')
    fit=[str(x) for x in data['fit_dialogue_ids']]
    dev=[str(x) for x in data['inner_dev_dialogue_ids']]
    if (not fit or not dev or set(fit)&set(dev) or set(fit)|set(dev)!=set(train)
            or len(fit)!=len(set(fit)) or len(dev)!=len(set(dev))):
        raise ValueError('Fit and internal dev must partition Train by Dialogue')
    result={}; stats={}
    for subset,ids in [('fit',fit),('inner_dev',dev)]:
        rows=[]; subset_stats=[]
        for did in ids:
            dialogue=train[did]
            texts=dialogue['texts']; labels=dialogue['labels']
            if len(texts)!=len(labels) or len(texts)!=len(dialogue['utterance_ids']):
                raise ValueError('Text/label/utterance alignment mismatch')
            encoded,token_stats=tokenize(tokenizer,texts,data['dataset'])
            for index,(tokens,label,item_stats) in enumerate(zip(encoded,labels,token_stats)):
                label=int(label)
                if label not in range(len(data['class_names'])):
                    raise ValueError('Class label outside declared class order')
                rows.append(dict(dialogue_id=did,utterance_index=index,
                    utterance_id=str(dialogue['utterance_ids'][index]),tokens=tokens,label=label))
                subset_stats.append(item_stats)
        result[subset]=rows
        stats[subset]=dict(samples=len(rows),dialogues=len(ids),
            truncated=sum(r['truncated'] for r in subset_stats),
            empty_after_processing=sum(r['empty_after_processing'] for r in subset_stats),
            ascii_changed=sum(r['ascii_changed'] for r in subset_stats),
            maximum_tokens_before=max(r['tokens_before'] for r in subset_stats))
    return result,stats


@torch.no_grad()
def evaluate_classifier(model,rows,device,classes,batch_size=4):
    model.eval(); confusion=torch.zeros((classes,classes),dtype=torch.long)
    ce_sum=0.0
    for start in range(0,len(rows),batch_size):
        selected=rows[start:start+batch_size]
        labels=[row['label'] for row in selected]
        output=model(**collate_token_ids([row['tokens'] for row in selected],device,labels))
        if not torch.isfinite(output.loss) or not torch.isfinite(output.logits).all():
            raise FloatingPointError('Non-finite FP32 classifier eval')
        ce_sum+=float(output.loss)*len(selected)
        for target,pred in zip(labels,output.logits.argmax(-1).cpu().tolist()):
            confusion[target,pred]+=1
    tp=confusion.diag().double()
    denom=confusion.sum(0)+confusion.sum(1)
    f1=torch.where(denom>0,2*tp/denom,torch.zeros_like(tp))
    return dict(ce=ce_sum/len(rows),macro_f1=float(f1.mean()),
                per_class_f1=f1.tolist(),confusion=confusion.tolist(),samples=len(rows))


def train(args,data):
    global ACTIVE_OUTPUT
    output=Path(args.output).resolve()
    if args.resume:
        if args.mode!='train' or not (output/'latest_training.pt').is_file() or (output/'best_model_lock.json').exists():
            raise RuntimeError('Resume requires unfinished formal training and full latest checkpoint')
    else:
        output.mkdir(parents=True,exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError('Refuse to overwrite nonempty training output')
    ACTIVE_OUTPUT=output
    pretrained=Path(args.pretrained).resolve()
    public_hash=sha256(pretrained/'model.safetensors')
    if public_hash!=PUBLIC_WEIGHT_SHA:
        raise ValueError('Expected the authorized public RoBERTa-large initializer')
    tokenizer=RobertaTokenizer.from_pretrained(pretrained,local_files_only=True,
                                              add_prefix_space=False,model_max_length=512)
    tokenizer.padding_side='right'; tokenizer.truncation_side='right'
    if (tokenizer.bos_token_id,tokenizer.pad_token_id,tokenizer.eos_token_id)!=(0,1,2):
        raise ValueError('Unexpected public special-token dictionary')
    rows,token_stats=get_training_rows(data,tokenizer)
    seed=SEEDS[args.dataset]
    micro=4; accumulation=4
    epoch_updates=math.ceil(math.ceil(len(rows['fit'])/micro)/accumulation)
    total_updates=10*epoch_updates
    warmup_updates=math.ceil(.1*total_updates)
    protocol=dict(dataset=args.dataset,seed=seed,data_sha256=sha256(args.data),
       public_weight_sha256=public_hash,script_sha256=sha256(__file__),
       class_names=list(data['class_names']),fit_dialogue_ids=list(data['fit_dialogue_ids']),
       inner_dev_dialogue_ids=list(data['inner_dev_dialogue_ids']),
       mode=args.mode,loss='supervised cross entropy on Train Fit emotion labels',
       model='public RoBERTa Large with native standard sequence-classification head',
       initial_head='new random classifier parameters; all backbone parameters fine-tuned',
       learning_rate=1e-5,weight_decay=.01,betas=[.9,.999],eps=1e-8,
       micro_batch=4,accumulation=4,effective_batch=16,epochs=10,patience=2,
       selection='strictly greatest internal-dev macro-F1; ties are non-improvement',
       macro_f1='unweighted mean over declared class_names; absent class F1=0',
       warmup_updates=warmup_updates,scheduler_total_updates=total_updates,
       scheduler='linear warmup 10% (ceil) followed by linear decay; fixed 10-epoch budget',
       precision='FP32 training/eval; no AMP/TF32',gradient_clip=1.0,
       dropout=.1,gradient_checkpointing='nonreentrant',
       shuffle='independent random.Random(seed + epoch) at each epoch; complete epoch resume',
       preprocessing='MELD ASCII errors=ignore; IEMOCAP unchanged; no case/whitespace normalization',
       tokenizer='public slow RoBERTa byte BPE, add_prefix_space=False, BOS/EOS, right truncate total512/right pad1',
       extraction='best encoder last four BOS [-1,-2,-3,-4], FP32, raw scale, no extra normalization',
       public_tokenizer_hashes={name:sha256(pretrained/name) for name in
           ('vocab.json','merges.txt','tokenizer_config.json')},
       official_validation_model_selection=False,test_accessed=False,
       refit_full_train=False,threads=args.threads)
    protocol_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if args.resume:
        previous=json.loads((output/'training_config.json').read_text())
        if previous['protocol_sha256']!=protocol_hash or previous['protocol']!=protocol:
            raise ValueError('Full resume protocol/code/input mismatch')
    else:
        dump(output/'training_config.json',dict(protocol=protocol,protocol_sha256=protocol_hash,
          token_statistics=token_stats,hostname=socket.gethostname(),
          visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),torch_version=torch.__version__,
          transformers_version=transformers.__version__,python=sys.version,executable=sys.executable,
          cuda_version=torch.version.cuda,gpu_name=torch.cuda.get_device_name(0),
          command=sys.argv))
        tokenizer.save_pretrained(output/'tokenizer')
    model_config=RobertaConfig.from_pretrained(pretrained,local_files_only=True)
    model_config.num_labels=len(data['class_names'])
    model_config.id2label={i:name for i,name in enumerate(data['class_names'])}
    model_config.label2id={name:i for i,name in enumerate(data['class_names'])}
    model_config.classifier_dropout=.1
    model_config.hidden_dropout_prob=.1;model_config.attention_probs_dropout_prob=.1
    model_config.planb_dataset=args.dataset;model_config.use_cache=False
    model,loading=RobertaForSequenceClassification.from_pretrained(pretrained,
        config=model_config,local_files_only=True,output_loading_info=True,torch_dtype=torch.float32)
    allowed_missing={'classifier.dense.weight','classifier.dense.bias',
                     'classifier.out_proj.weight','classifier.out_proj.bias'}
    if set(loading['missing_keys'])!=allowed_missing or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise RuntimeError(f'Unexpected public backbone/classifier loading: {loading}')
    if any(not key.startswith('lm_head.') for key in loading['unexpected_keys']):
        raise RuntimeError(f'Unexpected non-MLM unused weights: {loading}')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    device=torch.device('cuda:0');model.to(device,dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,
                                weight_decay=.01,foreach=False)
    scheduler=get_linear_schedule_with_warmup(optimizer,warmup_updates,total_updates)
    completed_epoch=0;global_step=0;best_epoch=None;best_score=-1.0;bad_epochs=0
    history=[];diagnostic=None
    if args.resume:
        state=torch.load(output/'latest_training.pt',map_location='cpu',weights_only=False)
        if state['protocol_sha256']!=protocol_hash:
            raise ValueError('Full checkpoint protocol mismatch')
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        completed_epoch=state['completed_epoch'];global_step=state['global_step']
        best_epoch=state['best_epoch'];best_score=state['best_score'];bad_epochs=state['bad_epochs']
        history=state['history'];diagnostic=state['gradient_verification']
        if state['sampler']['next_epoch']!=completed_epoch+1:
            raise ValueError('Checkpoint sampler/epoch mismatch')
        for parameter,values in optimizer.state.items():
            if any(values[key].device!=parameter.device for key in ('exp_avg','exp_avg_sq')):
                raise ValueError('Optimizer states failed device restoration')
        restore_rng(state['rng']);del state
    else:
        dump(output/'initial_loading_info.json',loading)
    torch.cuda.reset_peak_memory_stats(device)
    start=time.monotonic()
    def progress(status,**extra):
        dump(output/'progress.json',dict(status=status,epoch=completed_epoch,global_step=global_step,
             elapsed_seconds=time.monotonic()-start,peak_memory_gib=torch.cuda.max_memory_allocated(device)/2**30,**extra))
    progress('RUNNING')
    step_path=output/('smoke_steps.csv' if args.mode=='smoke' else 'optimizer_steps.csv')
    if args.resume:
        existing=list(csv.DictReader(step_path.open()))
        if len(existing)<global_step or [int(r['global_step']) for r in existing]!=list(range(1,len(existing)+1)):
            raise ValueError('Optimizer CSV lacks checkpoint prefix')
        if len(existing)>global_step:
            archive=output/f'optimizer_steps_interrupted_{time.time_ns()}.csv'
            archive.write_bytes(step_path.read_bytes())
            with step_path.open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(existing[0]));writer.writeheader();writer.writerows(existing[:global_step])
    with step_path.open('a' if args.resume else 'w') as stream:
        writer=csv.writer(stream)
        if not args.resume:
            writer.writerow(['epoch','global_step','ce','gradient_norm','learning_rate_used','elapsed_seconds'])
        for epoch in range(completed_epoch+1,11):
            if bad_epochs>=2:
                break
            indices=list(range(len(rows['fit'])))
            random.Random(seed+epoch).shuffle(indices)
            if args.mode=='smoke':
                indices=sorted(indices,key=lambda index:len(rows['fit'][index]['tokens']),reverse=True)[:32]
            model.train();epoch_loss=0.0;epoch_samples=0
            for group_start in range(0,len(indices),micro*accumulation):
                group=indices[group_start:group_start+micro*accumulation]
                optimizer.zero_grad(set_to_none=True);group_loss=0.0
                for micro_start in range(0,len(group),micro):
                    selected=[rows['fit'][i] for i in group[micro_start:micro_start+micro]]
                    output_values=model(**collate_token_ids([r['tokens'] for r in selected],device,
                                                           [r['label'] for r in selected]))
                    loss=output_values.loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite supervised FP32 loss')
                    (loss*len(selected)/len(group)).backward()
                    group_loss+=float(loss.detach())*len(selected)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Non-finite classifier/backbone gradients')
                probes={'backbone':model.roberta.encoder.layer[-1].attention.self.query.weight,
                        'classifier':model.classifier.out_proj.weight}
                before={name:p.detach().clone() for name,p in probes.items()} if diagnostic is None else None
                if diagnostic is None:
                    diagnostic={name:dict(gradient_present=p.grad is not None,
                       gradient_finite=bool(torch.isfinite(p.grad).all()),gradient_l2=float(p.grad.norm()))
                       for name,p in probes.items()}
                    diagnostic['all_backbone_parameters_trainable']=all(p.requires_grad for p in model.roberta.parameters())
                used_lr=optimizer.param_groups[0]['lr']
                optimizer.step();scheduler.step();global_step+=1
                if before is not None:
                    # Warmup begins at lr=0. Keep probing until the first positive-lr update.
                    for name,p in probes.items():
                        diagnostic[name]['max_parameter_update']=float((p.detach()-before[name]).abs().max())
                    dump(output/'gradient_update_verification.json',dict(global_step=global_step,learning_rate_used=used_lr,**diagnostic))
                    if used_lr==0:
                        diagnostic=None
                epoch_loss+=group_loss;epoch_samples+=len(group)
                writer.writerow([epoch,global_step,group_loss/len(group),float(norm),used_lr,time.monotonic()-start]);stream.flush()
                if global_step%20==0 or args.mode=='smoke':
                    print(json.dumps(dict(event='train_step',epoch=epoch,global_step=global_step,
                        ce=group_loss/len(group),lr=used_lr,gradient_norm=float(norm),
                        elapsed_seconds=time.monotonic()-start)),flush=True)
                    progress('RUNNING')
                if args.mode=='smoke' and global_step>=2:
                    dump(output/'smoke_completion.json',dict(status='COMPLETE',optimizer_steps=global_step,
                         peak_memory_gib=torch.cuda.max_memory_allocated(device)/2**30,
                         gradient_verification=diagnostic,official_validation_accessed=False))
                    progress('COMPLETE');return
            metrics=evaluate_classifier(model,rows['inner_dev'],device,len(data['class_names']))
            improved=metrics['macro_f1']>best_score
            if improved:
                best_score=metrics['macro_f1'];best_epoch=epoch;bad_epochs=0
                model.save_pretrained(output/'best_classifier',safe_serialization=True)
                tokenizer.save_pretrained(output/'best_classifier')
            else:
                bad_epochs+=1
            completed_epoch=epoch
            record=dict(epoch=epoch,global_step=global_step,fit_train_ce=epoch_loss/epoch_samples,
                        inner_dev=metrics,improved=improved,best_epoch=best_epoch,best_macro_f1=best_score,
                        elapsed_seconds=time.monotonic()-start)
            history.append(record);dump(output/'history.json',history)
            print(json.dumps(dict(event='epoch',**record)),flush=True)
            state=dict(protocol_sha256=protocol_hash,model=model.state_dict(),optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(),completed_epoch=epoch,global_step=global_step,
                 best_epoch=best_epoch,best_score=best_score,bad_epochs=bad_epochs,history=history,
                 rng=rng_state(),sampler=dict(next_epoch=epoch+1,shuffle_seed=seed+epoch+1,
                 completed_epoch_order=indices,offset=len(indices)),gradient_verification=diagnostic)
            save_torch(output/'latest_training.pt',state)
            progress('RUNNING',best_epoch=best_epoch,best_macro_f1=best_score)
    if best_epoch is None:
        raise RuntimeError('No trained classifier selected')
    model_file=output/'best_classifier'/'model.safetensors'
    dump(output/'best_model_lock.json',dict(status='COMPLETE',dataset=args.dataset,best_epoch=best_epoch,
       best_internal_macro_f1=best_score,completed_epochs=completed_epoch,global_step=global_step,
       model_sha256=sha256(model_file),protocol_sha256=protocol_hash,
       latest_training_sha256=sha256(output/'latest_training.pt'),refit_full_train=False,
       official_validation_used_for_selection=False,test_accessed=False))
    progress('COMPLETE',best_epoch=best_epoch,best_macro_f1=best_score)


def extract(args,data):
    output=Path(args.output).resolve(); feature_path=Path(args.features).resolve()
    if feature_path.exists() or feature_path.with_suffix('.manifest.json').exists():
        raise FileExistsError('Refuse existing text features')
    lock=json.loads((output/'best_model_lock.json').read_text())
    if lock['dataset']!=args.dataset or sha256(output/'best_classifier'/'model.safetensors')!=lock['model_sha256']:
        raise ValueError('Selected supervised model hash/dataset mismatch')
    encoder,tokenizer=load_encoder(output)
    features={};manifest={};all_stats=[]
    for split in ('train','validation'):
        features[split]={};manifest[split]=[]
        for dialogue in data['splits'][split]:
            did=str(dialogue['dialogue_id']);texts=dialogue['texts']
            if did in features[split] or len(texts)!=len(dialogue['utterance_ids']):
                raise ValueError('Dialogue/text alignment error')
            vectors=extract_texts(encoder,tokenizer,texts,args.dataset)
            _,stats=tokenize(tokenizer,texts,args.dataset)
            features[split][did]=vectors
            manifest[split].append(dict(dialogue_id=did,utterance_ids=list(dialogue['utterance_ids']),
                shape=list(vectors.shape),sha256=hashlib.sha256(vectors.numpy().astype('<f4',copy=False).tobytes()).hexdigest()))
            for index,item in enumerate(stats):
                all_stats.append(dict(split=split,dialogue_id=did,utterance_index=index,**item))
    feature_path.parent.mkdir(parents=True,exist_ok=True)
    save_torch(feature_path,features)
    metadata=dict(dataset=args.dataset,seed=SEEDS[args.dataset],model_sha256=lock['model_sha256'],
        data_sha256=sha256(args.data),script_sha256=sha256(__file__),feature_file_sha256=sha256(feature_path),
        feature_shape='per Dialogue [N,4,1024]',dtype='float32',layer_order=[-1,-2,-3,-4],
        pooling='BOS index0',extra_normalization=None,fp32_eval=True,model_selection_split='Train internal dev',
        split_statistics={split:dict(dialogues=len(features[split]),utterances=sum(len(t) for t in features[split].values()),
           truncated=sum(row['truncated'] for row in all_stats if row['split']==split),
           empty_after_processing=sum(row['empty_after_processing'] for row in all_stats if row['split']==split),
           ascii_changed=sum(row['ascii_changed'] for row in all_stats if row['split']==split)) for split in features},
        tokenizer_hashes={path.name:sha256(path) for path in (output/'best_classifier').iterdir()
                          if path.is_file() and path.name in ('vocab.json','merges.txt','tokenizer_config.json','special_tokens_map.json')},
        dialogue_features=manifest,test_extracted=False)
    dump(feature_path.with_suffix('.manifest.json'),metadata)
    with feature_path.with_suffix('.truncation.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_stats[0]));writer.writeheader();writer.writerows(all_stats)
    print(json.dumps(dict(event='extraction_complete',path=str(feature_path),split_statistics=metadata['split_statistics'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['smoke','train','extract'])
    parser.add_argument('--dataset',required=True,choices=list(SEEDS))
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--features')
    parser.add_argument('--pretrained',required=True)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.mode=='extract' and (not args.features or args.resume):
        raise ValueError('Extraction requires --features and forbids --resume')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed_all(SEEDS[args.dataset])
    if not torch.cuda.is_available():
        raise RuntimeError('Assigned CUDA GPU required')
    data=torch.load(args.data,map_location='cpu',weights_only=False)
    if data['dataset']!=args.dataset or int(data['seed'])!=SEEDS[args.dataset]:
        raise ValueError('Dataset/seed mismatch')
    if set(data['splits'])!={'train','validation'}:
        raise ValueError('Only effective Train and Validation are permitted')
    if args.mode=='extract':
        extract(args,data)
    else:
        train(args,data)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        if ACTIVE_OUTPUT is not None:
            dump(ACTIVE_OUTPUT/'failure.json',dict(status='FAILED',exception=type(error).__name__,
                 message=str(error),traceback=traceback.format_exc()))
        raise
