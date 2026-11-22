#!/usr/bin/env python3
"""Two-runtime, inference-only MELD score cohort producer.

TEXT_repro: --stage encode. multimodalerc310: --stage infer.
Only the fresh run's text and CMERC weights are allowed. The parent's pinned
preprocessing/forward are reused. Missing or unqualified K2 never removes clean
predictions: D_pred is an empty CSV cell. No reliability score is fitted here.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
DEFAULT_ROOT=HERE.parent
PARENT=DEFAULT_ROOT.parent/'20261120T143131Z'/'scripts'
PINNED={'text_encoder.py':'a8e87cb16064e3e54d52d63c1f6ccf62c2dd2eee4e600da2a0b80604f23adc34',
        'downstream.py':'40632d490388f52c4c4ab99d96d1fced047782e3bf6ba782c9048add14bcc249'}
ROLES=('score_fit','score_eval')
VARIANTS=('paraphrase_a','paraphrase_b')
FORMAT='MELD_SCORE_TEXT_K2_V1'


class FrozenInputMutation(RuntimeError):
    """A violation must abort the result rather than become missing behavior."""


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def text_sha(value):return hashlib.sha256(value.encode('utf-8')).hexdigest()


def tensor_sha(value):
    value=value.detach().cpu().contiguous().numpy() if isinstance(value,torch.Tensor) else np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:return [json.loads(line) for line in stream if line.strip()]


def dump_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=True,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def write_csv(path,rows):
    if not rows:raise ValueError('Missing clean cohort; refusing empty result')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def validate_cohort(data,role_rows,expected_count=2976):
    """Match every score utterance to the prelocked four-role Train manifest."""
    if data.get('dataset')!='MELD' or int(data.get('seed',-1))!=67137 or len(data.get('class_names',[]))!=7:
        raise ValueError('Wrong MELD dataset/seed/class order declaration')
    if set(data.get('splits',{}))!=set(ROLES):raise ValueError('Only score_fit/score_eval splits allowed; no Test')
    manifest={}
    for r in role_rows:
        key=(str(r['dialogue_id']),str(r['utterance_id']))
        if key in manifest:raise ValueError('duplicate role manifest utterance')
        manifest[key]=r
    index={};seen_dialogues=set()
    for role in ROLES:
        for dialogue in data['splits'][role]:
            did=str(dialogue['dialogue_id']);n=len(dialogue['utterance_ids'])
            if did in seen_dialogues:raise ValueError('Dialogue role overlap/duplicate')
            seen_dialogues.add(did)
            if n<1 or len(dialogue['texts'])!=n:raise ValueError('Empty/misaligned score Dialogue')
            if len(set(map(str,dialogue['utterance_ids'])))!=n:raise ValueError('duplicate Utterance ID')
            if tuple(dialogue['visual'].shape)[0]!=n or tuple(dialogue['speaker'].shape)[0]!=n:
                raise ValueError('Context modality length mismatch')
            for i,(uid,text) in enumerate(zip(dialogue['utterance_ids'],dialogue['texts'])):
                uid=str(uid);sid=f'MELD:train:{did}:{uid}';r=manifest.get((did,uid))
                if r is None or r['role']!=role:raise ValueError('Missing/mismatched locked role')
                if r.get('sample_id',sid)!=sid or ('utterance_index' in r and int(r['utterance_index'])!=i):
                    raise ValueError('Role manifest sample identity/order mismatch')
                if r.get('source_split','train')!='train':raise ValueError('Score utterance is not original Train')
                if sid in index or not isinstance(text,str):raise ValueError('duplicate/invalid score utterance')
                index[sid]=dict(sample_id=sid,role=role,dialogue_id=did,utterance_id=uid,
                    utterance_index=i,original_text=text,dialogue=dialogue)
    wanted={(str(r['dialogue_id']),str(r['utterance_id'])) for r in role_rows if r['role'] in ROLES}
    if wanted!={(r['dialogue_id'],r['utterance_id']) for r in index.values()}:
        raise ValueError('Incomplete score cohort versus locked roles')
    if expected_count is not None and len(index)!=expected_count:
        raise ValueError(f'Expected {expected_count} clean score utterances; got {len(index)}')
    return index


def validate_reviews(rows,index):
    """Partial coverage is legal. Wrong identity or duplicate versions are fatal."""
    reviewed={}
    required={'sample_id','role','dialogue_id','utterance_index','utterance_id','variant_id',
              'original_text_sha256','processed_text','review_status','review_method','reviewer','review_notes'}
    forbidden={'original_correct','original_confidence','original_probabilities','label','labels','prediction'}
    for r in rows:
        if not required<=set(r) or forbidden&set(r):raise ValueError('Review schema or blinding violation')
        sid=r['sample_id'];variant=r['variant_id']
        if sid not in index or variant not in VARIANTS:raise ValueError('Review outside fixed score cohort/K2')
        target=index[sid]
        if (r['role']!=target['role'] or str(r['dialogue_id'])!=target['dialogue_id']
            or int(r['utterance_index'])!=target['utterance_index'] or str(r['utterance_id'])!=target['utterance_id']):
            raise ValueError('Review role/Dialogue/Utterance identity mismatch')
        if r['original_text_sha256']!=text_sha(target['original_text']):raise ValueError('Review original text hash mismatch')
        if r['review_status'] not in ('PASS','FAIL','UNCERTAIN'):raise ValueError('Undecided/unknown review status')
        if not isinstance(r['processed_text'],str):raise ValueError('Candidate text must be a string')
        if not all(isinstance(r[k],str) for k in ('review_method','reviewer','review_notes')):
            raise ValueError('Invalid review attribution fields')
        if r['review_status']=='PASS' and (r['review_method']!='AI_CONTEXT_SEMANTIC_REVIEW'
            or not r['reviewer'].strip() or not r['review_notes'].strip()):
            raise ValueError('PASS needs attributed context semantic review')
        target_versions=reviewed.setdefault(sid,{})
        if variant in target_versions:raise ValueError('duplicate reviewed variant')
        target_versions[variant]=r
    return reviewed


def validate_probability(probability):
    p=np.asarray(probability,dtype=np.float64)
    if p.shape!=(7,) or not np.isfinite(p).all() or (p<0).any() or (p>1+1e-6).any() or abs(p.sum()-1)>1e-4:
        raise ValueError('Expected unchanged finite seven-class probability distribution')
    return p


def clean_summary(probability,label):
    p=validate_probability(probability);prediction=int(p.argmax());ordered=np.sort(p)
    if not 0<=int(label)<7:raise ValueError('Invalid score label')
    positive=p>0
    return dict(original_prediction=prediction,original_correct=int(prediction==int(label)),
                original_confidence=float(p[prediction]),clean_margin=float(ordered[-1]-ordered[-2]),
                clean_entropy=float(-np.sum(p[positive]*np.log(p[positive]))))


def d_pred(clean_probability,variants):
    p=validate_probability(clean_probability);c=int(p.argmax())
    if any(k not in variants or not variants[k].get('eligible') or variants[k].get('probabilities') is None for k in VARIANTS):
        return None
    return float(p[c]-np.mean([validate_probability(variants[k]['probabilities'])[c] for k in VARIANTS]))


def replace_target(clean,index,target):
    if (clean.dtype!=torch.float32 or target.dtype!=torch.float32 or clean.ndim!=3
        or tuple(clean.shape[1:])!=(4,1024) or tuple(target.shape)!=(4,1024)
        or not 0<=index<len(clean) or not torch.isfinite(clean).all() or not torch.isfinite(target).all()):
        raise ValueError('Expected finite FP32 target-only text replacement')
    changed=clean.clone();changed[index]=target
    keep=torch.arange(len(clean))!=index
    if not torch.equal(changed[keep],clean[keep]):raise RuntimeError('Non-target text changed')
    return changed


def token_qc(clean_ids,changed_ids,truncated,empty_after_processing):
    changed=changed_ids!=clean_ids
    return dict(token_ids_changed=changed,no_truncation=not truncated,nonempty=not empty_after_processing,
                eligible=bool(changed and not truncated and not empty_after_processing))


def enforce_distinct_versions(variants):
    a,b=(variants.get(k) for k in VARIANTS)
    if a and b and a.get('actual_token_ids')==b.get('actual_token_ids') and a.get('actual_token_ids') is not None:
        for r in (a,b):r.update(eligible=False,reason='DUPLICATE_K2_ACTUAL_TOKEN_IDS')


def require_runtime(stage):
    expected={'encode':('TEXT_repro','2.2.0+cu118','4.42.3'),
              'infer':('multimodalerc310','2.2.2+cu121','4.45.1')}[stage]
    actual=(Path(sys.prefix).name,torch.__version__,importlib.metadata.version('transformers'))
    if actual!=expected:raise RuntimeError(f'{stage} requires {expected}; actual {actual}')
    return dict(environment=actual[0],torch=actual[1],transformers=actual[2],python=sys.version,executable=sys.executable)


def load_torch(path):return torch.load(path,map_location='cpu',weights_only=False)


def seed_runtime():
    random.seed(67137);np.random.seed(67137);torch.manual_seed(67137)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(67137)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False


def inside_run(path,root):
    path=Path(path).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Fresh run assets must not resolve outside run directory')
    return path


def common(args):
    """Bind every input before either model is instantiated."""
    root=args.run_dir.resolve();adapter=args.adapter_dir.resolve()
    if adapter!=PARENT.resolve():raise ValueError('Use the explicit frozen parent adapter directory')
    inputs={}
    def bind(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        if expected is not None and digest!=expected:raise ValueError('Pinned asset hash mismatch: '+str(path))
        inputs[str(path)]=digest;return digest
    for name,digest in PINNED.items():bind(adapter/name,digest)
    core_files=sorted((adapter/'core_snapshot').glob('*.py'))
    if not core_files:raise ValueError('Missing frozen CMERC core')
    for path in core_files:bind(path)
    for path in (args.data,args.audio,args.review,args.roles,Path(__file__)):bind(inside_run(path,root))
    split_lock_path=root/'SPLIT_LOCK.json'
    split_lock=json.loads(split_lock_path.read_text());bind(split_lock_path)
    if (split_lock.get('dataset')!='MELD' or split_lock.get('no_test_data') is not True
        or split_lock.get('official_validation_rows_used')!=0
        or split_lock['hashes']['SPLIT_UTTERANCES.csv']!=sha(args.roles)):
        raise ValueError('Score role manifest differs from the independently frozen Train split')
    bind(root/'configs/REAL_SCORING_PROTOCOL.json')
    model=inside_run(root/'local_only/text_model/best_classifier',root)
    checkpoint=inside_run(root/'local_only/base_data/runs/cmerc/best.pt',root)
    lock_path=model.parent/'best_model_lock.json';lock=json.loads(lock_path.read_text());bind(lock_path)
    if lock.get('status')!='COMPLETE' or lock.get('dataset')!='MELD' or lock.get('test_accessed') is not False:
        raise ValueError('Fresh text training completion/dataset/no-Test lock missing')
    if lock.get('official_validation_used_for_selection') is not False:raise ValueError('Invalid text selection protocol')
    bind(model/'model.safetensors',lock['model_sha256'])
    for name in ('config.json','vocab.json','merges.txt','tokenizer_config.json','special_tokens_map.json'):
        bind(model/name)
    config=json.loads((model/'config.json').read_text())
    data=load_torch(args.data)
    if config.get('planb_dataset')!='MELD' or [config['id2label'][str(i)] for i in range(7)]!=data['class_names']:
        raise ValueError('Text model configuration/class order mismatch')
    completion=checkpoint.parent/'completion.json';done=json.loads(completion.read_text());bind(completion)
    if done.get('status')!='COMPLETE' or done.get('stage')!='cmerc':raise ValueError('Fresh CMERC not complete')
    bind(checkpoint,done['best_checkpoint_sha256'])
    with args.roles.open(newline='') as stream:role_rows=list(csv.DictReader(stream))
    index=validate_cohort(data,role_rows)
    training_path=model.parent/'training_config.json'
    training=json.loads(training_path.read_text());bind(training_path)
    protocol=training['protocol']
    protocol_sha=text_sha(json.dumps(protocol,sort_keys=True))
    base_path=root/'local_only/base_data/dataset.pt';bind(base_path)
    fit_ids={str(r['dialogue_id']) for r in role_rows if r['role']=='base_fit'}
    dev_ids={str(r['dialogue_id']) for r in role_rows if r['role']=='base_inner_dev'}
    if (protocol_sha!=lock['protocol_sha256'] or protocol_sha!=training['protocol_sha256']
        or protocol.get('data_sha256')!=sha(base_path) or protocol.get('script_sha256')!=PINNED['text_encoder.py']
        or set(map(str,protocol.get('fit_dialogue_ids',[])))!=fit_ids
        or set(map(str,protocol.get('inner_dev_dialogue_ids',[])))!=dev_ids):
        raise ValueError('Fresh text weights are not bound to this base_fit/base_inner_dev cohort')
    reviews=validate_reviews(read_jsonl(args.review),index)
    audio=load_torch(args.audio)
    if set(audio)!=set(ROLES):raise ValueError('Audio bundle must contain exactly score roles')
    for role in ROLES:
        wanted={str(d['dialogue_id']) for d in data['splits'][role]}
        if set(audio[role])!=wanted:raise ValueError('Audio score Dialogue coverage mismatch')
        for d in data['splits'][role]:
            a=audio[role][str(d['dialogue_id'])]
            if a.dtype!=torch.float32 or tuple(a.shape)!=(len(d['texts']),768) or not torch.isfinite(a).all():
                raise ValueError('Invalid FP32 audio features')
    return data,index,reviews,audio,inputs,lock,checkpoint


def import_parent(name,adapter):
    path=adapter/(name+'.py');spec=importlib.util.spec_from_file_location('score_frozen_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def ensure_inputs_unchanged(inputs):
    for path,digest in inputs.items():
        if sha(path)!=digest:raise RuntimeError('Input changed during scoring: '+path)


def public_identity(target):
    return {k:target[k] for k in ('sample_id','dialogue_id','utterance_id','utterance_index','role')}


def make_encoded_records(index,reviews,text_module,tokenizer,encoder):
    clean={role:{} for role in ROLES};records={};clean_tokens={}
    with torch.inference_mode():
        seen=set()
        for sid,target in index.items():
            key=(target['role'],target['dialogue_id'])
            if key not in seen:
                dialogue=target['dialogue'];features=text_module.extract_texts(encoder,tokenizer,dialogue['texts'],dataset='MELD',batch_size=1)
                if features.dtype!=torch.float32 or tuple(features.shape)!=(len(dialogue['texts']),4,1024) or not torch.isfinite(features).all():
                    raise ValueError('Clean extraction failed; do not drop clean utterances')
                clean[key[0]][key[1]]=features;seen.add(key)
            ids,stats=text_module.tokenize(tokenizer,[target['original_text']],'MELD')
            clean_tokens[sid]=ids[0]
            versions={}
            for variant in VARIANTS:
                review=reviews.get(sid,{}).get(variant)
                record=dict(public_identity(target),variant_id=variant,eligible=False,encoding_status='NOT_ATTEMPTED',
                            review_status='NOT_ATTEMPTED',reason='NO_REVIEWED_VERSION',probabilities=None)
                if review is not None:
                    record.update(review_status=review['review_status'],processed_text_sha256=text_sha(review['processed_text']),
                        original_text_sha256=review['original_text_sha256'],review_method=review['review_method'],
                        reviewer=review['reviewer'],review_notes_sha256=text_sha(review['review_notes']))
                    if review['review_status']!='PASS':
                        record.update(encoding_status='INELIGIBLE',reason='SEMANTIC_REVIEW_'+review['review_status'])
                    else:
                        candidate_ids,statistics=text_module.tokenize(tokenizer,[review['processed_text']],'MELD')
                        qc=token_qc(ids[0],candidate_ids[0],statistics[0]['truncated'],statistics[0]['empty_after_processing'])
                        record.update(actual_token_ids=candidate_ids[0],token_qc=qc,token_statistics=statistics[0],
                                      actual_token_ids_sha256=text_sha(json.dumps(candidate_ids[0])),
                                      eligible=qc['eligible'],encoding_status='INELIGIBLE',reason='TOKEN_QC_FAILED')
                        if qc['eligible']:
                            try:
                                feature=text_module.extract_texts(encoder,tokenizer,[review['processed_text']],dataset='MELD',batch_size=1)[0]
                                if feature.dtype!=torch.float32 or tuple(feature.shape)!=(4,1024) or not torch.isfinite(feature).all():
                                    raise ValueError('Invalid target feature')
                                record.update(feature=feature,feature_sha256=tensor_sha(feature),encoding_status='ENCODED',reason='OK')
                            except Exception as error:
                                record.update(eligible=False,encoding_status='FAILED',reason='FEATURE_EXTRACTION_FAILED',error=type(error).__name__+': '+str(error))
                versions[variant]=record
            enforce_distinct_versions(versions);records[sid]=versions
    return clean,records,clean_tokens


def encode(args):
    runtime=require_runtime('encode');data,index,reviews,audio,inputs,lock,checkpoint=common(args)
    if args.features.exists() or args.features.with_suffix('.manifest.json').exists():raise FileExistsError('Refusing to overwrite encoded score features')
    torch.set_num_threads(args.threads);seed_runtime()
    module=import_parent('text_encoder',args.adapter_dir)
    encoder,tokenizer=module.load_encoder(args.run_dir/'local_only/text_model',device=args.device)
    encoder.eval().requires_grad_(False)
    clean,records,clean_tokens=make_encoded_records(index,reviews,module,tokenizer,encoder)
    ensure_inputs_unchanged(inputs)
    bundle=dict(format=FORMAT,dataset='MELD',seed=67137,class_names=data['class_names'],clean=clean,variants=records,
                clean_token_ids=clean_tokens,inputs=inputs,text_model_sha256=lock['model_sha256'],runtime=runtime,
                fp32=True,text_batch_size=1,preprocessing='frozen parent MELD ASCII errors=ignore; last four BOS vectors [-1,-2,-3,-4]')
    args.features.parent.mkdir(parents=True,exist_ok=True);temp=args.features.with_name(args.features.name+'.tmp')
    torch.save(bundle,temp);temp.replace(args.features)
    dump_json(args.features.with_suffix('.manifest.json'),dict(format=FORMAT,feature_file_sha256=sha(args.features),inputs=inputs,
        clean_utterances=len(index),expected_k=2,reviewed_versions=sum(len(x) for x in reviews.values()),
        encoding_status_counts={s:sum(r['encoding_status']==s for vs in records.values() for r in vs.values()) for s in ('ENCODED','INELIGIBLE','FAILED','NOT_ATTEMPTED')},
        runtime=runtime,text_model_sha256=lock['model_sha256'],no_test=True,no_model_fit=True,fp32=True))
    print(json.dumps({'stage':'encode','status':'COMPLETE','clean_utterances':len(index),'features_sha256':sha(args.features)}),flush=True)


def model_digest(model):
    return text_sha(json.dumps({k:tensor_sha(v) for k,v in sorted(model.state_dict().items())},sort_keys=True))


def context_digest(dialogue,text,audio):
    values={k:tensor_sha(v) for k,v in [('text',text),('audio',audio),('visual',dialogue['visual']),('speaker',dialogue['speaker']),('labels',dialogue['labels'])]}
    values.update(utterance_ids=list(map(str,dialogue['utterance_ids'])),texts=[text_sha(x) for x in dialogue['texts']],
                  dialogue_id=str(dialogue['dialogue_id']))
    return text_sha(json.dumps(values,sort_keys=True))


def locked_forward(adapter,model,dialogue,text,audio):
    before=context_digest(dialogue,text,audio)
    with torch.inference_mode():prob=np.asarray(adapter.infer_dialogue(model,dialogue,text,audio))
    if context_digest(dialogue,text,audio)!=before:raise FrozenInputMutation('Frozen Dialogue/modalities mutated during forward')
    if prob.shape!=(len(dialogue['utterance_ids']),7):raise ValueError('Missing full Dialogue probability rows')
    for p in prob:validate_probability(p)
    return prob


def verify_cmerc_metadata(payload,data,index,lock,root,adapter):
    if payload.get('new_features_only') is not True or payload.get('stage')!='cmerc':raise ValueError('Refusing old/non-CMERC Plan B weights')
    c=payload['effective_config'];first=next(iter(index.values()))['dialogue']
    expected={'Dataset':'MELD','seed':67137,'text_dim':1024,'audio_dim':768,'text_layers':4,
              'visual_dim':int(first['visual'].shape[1]),'speaker_dim':int(first['speaker'].shape[1])}
    if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('CMERC configuration/dimension mismatch')
    branch=c.get('branch',{})
    if (branch.get('branch')!='MELD_REAL_SCORING_BASE_ONLY'
        or branch.get('scoring_groups_loaded') is not False
        or branch.get('fit_role')!='base_fit' or branch.get('selection_role')!='base_inner_dev'
        or branch.get('parent_adapter_sha256')!=PINNED['downstream.py']):
        raise ValueError('CMERC weights lack this fresh base-only branch identity')
    extra={}
    for path,digest in branch.get('split_and_protocol_hashes',{}).items():
        if sha(path)!=digest:raise ValueError('CMERC branch split/protocol lock changed')
        extra[str(Path(path).resolve())]=digest
    branch_adapter=inside_run(branch['adapter_path'],root)
    if sha(branch_adapter)!=branch['adapter_sha256']:raise ValueError('CMERC branch adapter hash mismatch')
    extra[str(branch_adapter)]=branch['adapter_sha256']
    # Base training text features must come from the same fresh fine-tuned encoder.
    metadata_path=root/'local_only/base_data/features/text_features.manifest.json'
    metadata=json.loads(metadata_path.read_text())
    if metadata.get('model_sha256')!=lock['model_sha256']:raise ValueError('CMERC base text features use different text weights')
    feature_path=metadata_path.with_name('text_features.pt')
    if sha(feature_path)!=metadata['feature_file_sha256']:raise ValueError('CMERC base text feature cache changed')
    bound=payload.get('provenance',{}).get('inputs',{})
    for path in (root/'local_only/base_data/dataset.pt',feature_path,metadata_path):
        digest=sha(path)
        if bound.get(str(path.resolve()))!=digest:raise ValueError('CMERC checkpoint training input hash mismatch')
        extra[str(path.resolve())]=digest
    for path,digest in payload.get('provenance',{}).get('code',{}).items():
        if Path(path).name in PINNED or Path(path).parent.name=='core_snapshot':
            if sha(path)!=digest:raise ValueError('CMERC training source changed: '+path)
    return c,extra


def infer(args):
    runtime=require_runtime('infer');data,index,reviews,audio,inputs,lock,checkpoint=common(args)
    output=args.run_dir/'results';result_path=output/'score_inputs.csv'
    if any((output/name).exists() for name in ('score_inputs.csv','score_transformations.csv','score_prediction_verification.json')):
        raise FileExistsError('Refusing existing scoring predictions')
    manifest=json.loads(args.features.with_suffix('.manifest.json').read_text())
    if manifest['feature_file_sha256']!=sha(args.features):raise ValueError('Encoded feature hash mismatch')
    bundle=load_torch(args.features)
    if bundle.get('format')!=FORMAT or bundle.get('fp32') is not True or bundle.get('inputs')!=inputs or manifest.get('inputs')!=inputs:
        raise ValueError('Encode/infer inputs/model/code differ')
    if bundle.get('class_names')!=data['class_names'] or set(bundle.get('variants',{}))!=set(index):raise ValueError('Encoded cohort/class mismatch')
    ensure_inputs_unchanged(bundle['inputs'])
    payload=load_torch(checkpoint)
    config,extra=verify_cmerc_metadata(payload,data,index,lock,args.run_dir,args.adapter_dir);del payload
    inputs.update(extra)
    torch.set_num_threads(args.threads);seed_runtime()
    adapter=import_parent('downstream',args.adapter_dir)
    model=adapter.load_model(args.run_dir/'local_only/base_data',checkpoint,args.device)
    model.eval().requires_grad_(False)
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('CMERC must run FP32')
    model_before=model_digest(model);clean_probabilities={}
    for role in ROLES:
        if set(bundle['clean'][role])!={str(d['dialogue_id']) for d in data['splits'][role]}:raise ValueError('Encoded clean Dialogue coverage mismatch')
        for dialogue in data['splits'][role]:
            did=str(dialogue['dialogue_id']);text=bundle['clean'][role][did];aud=audio[role][did]
            if text.dtype!=torch.float32 or tuple(text.shape)!=(len(dialogue['texts']),4,1024):raise ValueError('Invalid clean text shape/dtype')
            clean_probabilities[(role,did)]=locked_forward(adapter,model,dialogue,text,aud)
    clean_rows=[];risks=[]
    for sid,target in index.items():
        role=target['role'];did=target['dialogue_id'];i=target['utterance_index'];dialogue=target['dialogue']
        clean=bundle['clean'][role][did];aud=audio[role][did];original=clean_probabilities[(role,did)][i]
        summary=clean_summary(original,int(dialogue['labels'][i]));versions=bundle['variants'][sid]
        for variant in VARIANTS:
            record=versions[variant]
            if record['eligible']:
                feature=record['feature']
                if tensor_sha(feature)!=record['feature_sha256']:raise ValueError('Perturbed target feature hash mismatch')
                # Verify original tensors remain exact after the target-only clone is used.
                baseline=context_digest(dialogue,clean,aud)
                changed=replace_target(clean,i,feature)
                try:
                    record['probabilities']=locked_forward(adapter,model,dialogue,changed,aud)[i].tolist()
                    record['inference_status']='PREDICTED'
                except FrozenInputMutation:
                    raise
                except Exception as error:
                    record.update(eligible=False,inference_status='FAILED',reason='PERTURBED_FORWARD_FAILED',error=type(error).__name__+': '+str(error))
                if context_digest(dialogue,clean,aud)!=baseline:raise RuntimeError('Target replacement changed original context')
            else:record['inference_status']='NOT_RUN'
            probability=record.get('probabilities')
            risk=dict(public_identity(target),variant_id=variant,review_status=record['review_status'],
                      review_method=record.get('review_method',''),reviewer=record.get('reviewer',''),
                      review_notes_sha256=record.get('review_notes_sha256',''),
                      encoding_status=record['encoding_status'],inference_status=record['inference_status'],eligible=int(record['eligible']),
                      reason=record['reason'],error=record.get('error',''),
                      original_text_sha256=text_sha(target['original_text']),processed_text_sha256=record.get('processed_text_sha256',''),
                      actual_token_ids_sha256=record.get('actual_token_ids_sha256',''),feature_sha256=record.get('feature_sha256',''),
                      original_feature_sha256=tensor_sha(clean[i]),
                      original_prediction=summary['original_prediction'],original_probabilities=json.dumps(original.tolist()),
                      perturbed_prediction=int(np.argmax(probability)) if probability is not None else '',
                      perturbed_original_class_probability=probability[summary['original_prediction']] if probability is not None else '',
                      perturbed_probabilities=json.dumps(probability) if probability is not None else '')
            risks.append(risk)
        delta=d_pred(original,versions)
        reasons='|'.join(k+':'+versions[k]['reason'] for k in VARIANTS if not versions[k]['eligible'])
        clean_rows.append(dict(public_identity(target),**summary,D_pred=delta if delta is not None else '',
                               D_pred_missing_reason=reasons if delta is None else '',k2_available=int(delta is not None),
                               original_label=int(dialogue['labels'][i]),original_probabilities=json.dumps(original.tolist())))
    model_after=model_digest(model)
    if model_after!=model_before:raise RuntimeError('CMERC parameters/BN state mutated')
    ensure_inputs_unchanged(inputs)
    write_csv(result_path,clean_rows);write_csv(output/'score_transformations.csv',risks)
    verification=dict(status='COMPLETE',stage='infer',no_test=True,no_model_fit=True,dataset='MELD',seed=67137,
        clean_utterances=len(clean_rows),expected_clean_utterances=2976,planned_k2_versions=len(risks),
        available_D_pred=sum(r['k2_available'] for r in clean_rows),runtime=runtime,inputs=inputs,
        encoded_features_sha256=sha(args.features),encoder_manifest_sha256=sha(args.features.with_suffix('.manifest.json')),
        model_state_before=model_before,model_state_after=model_after,fp32=True,TF32=False,
        target_only_replacement=True,full_dialogue_context=True,text_modality_only=True,
        D_pred='p_clean(c)-mean(p_paraphrase_a(c),p_paraphrase_b(c)); c=clean argmax; signed; double-PASS distinct-token K2 only',
        entropy='-sum(p*ln(p)); natural logarithm; zero terms contribute zero',
        effective_cmerc_config=config,outputs={name:sha(output/name) for name in ('score_inputs.csv','score_transformations.csv')},
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    dump_json(output/'score_prediction_verification.json',verification)
    print(json.dumps({'stage':'infer','status':'COMPLETE','clean_utterances':len(clean_rows),'available_D_pred':verification['available_D_pred']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('encode','infer'))
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--adapter-dir',type=Path,default=PARENT)
    parser.add_argument('--data',type=Path);parser.add_argument('--audio',type=Path);parser.add_argument('--review',type=Path)
    parser.add_argument('--roles',type=Path);parser.add_argument('--features',type=Path)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args();args.run_dir=args.run_dir.resolve()
    for key,name in [('data','score_data.pt'),('audio','score_audio_features.pt'),('review','reviewed_text.jsonl'),
                     ('roles','SPLIT_UTTERANCES.csv'),('features','score_text_features.pt')]:
        if getattr(args,key) is None:setattr(args,key,args.run_dir/'local_only'/name)
    if args.threads<1:parser.error('--threads must be positive')
    inside_run(args.features,args.run_dir)
    (encode if args.stage=='encode' else infer)(args)


if __name__=='__main__':main()
