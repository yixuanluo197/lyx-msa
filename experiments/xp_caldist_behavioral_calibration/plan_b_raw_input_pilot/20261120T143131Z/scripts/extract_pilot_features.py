#!/usr/bin/env python3
"""Pinned encoder-only producer; CMERC inference consumes verified CPU tensors.

Smoke indexes one predeclared Train Fit target. Pilot requires an already frozen
baseline/120-target manifest and preserves all planned successful/failed variants.
No downstream/PyG import, model fitting, sampling or semantic review occurs here.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from scipy.io import wavfile

import raw_pilot as shared

ROOT=Path(__file__).resolve().parents[1]
TEXT_SHA='a8e87cb16064e3e54d52d63c1f6ccf62c2dd2eee4e600da2a0b80604f23adc34'
AUDIO_SHA='bbcdde2ee133290eac5dfdffd7f45f4bd2a204605e32bc0c213b0eca1171ca11'
FORMAT='PLAN_B_PRECOMPUTED_FEATURES_V2'


def now():
    return datetime.now(timezone.utc).isoformat()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def bind_file(inputs,path,expected=None):
    path=Path(path).resolve()
    digest=shared.sha256(path)
    if expected is not None and digest!=expected:
        raise ValueError('Bound input SHA mismatch: '+str(path))
    previous=inputs.setdefault(str(path),digest)
    if previous!=digest:
        raise ValueError('An input changed during producer execution: '+str(path))
    return digest


def actual_text_input(text_module,tokenizer,text,dataset):
    token_ids,statistics=text_module.tokenize(tokenizer,[text],dataset)
    actual=np.asarray(token_ids[0],dtype=np.int64)
    return dict(actual_token_ids=actual.tolist(),actual_token_ids_sha256=shared.array_hash(actual),
        extractor_processed_text_sha256=text_sha(text_module.preprocessing(text,dataset)),
        token_statistics=statistics[0])


def extracted_record(base,feature,original_feature):
    if base.get('modality') not in ('text','audio'):
        raise ValueError('Only text/audio target modalities are permitted')
    feature=torch.as_tensor(feature).detach().cpu().contiguous()
    expected=(4,1024) if base['modality']=='text' else (768,)
    if feature.dtype!=torch.float32 or tuple(feature.shape)!=expected or not torch.isfinite(feature).all():
        raise ValueError('Expected finite CPU FP32 target with shape '+str(expected))
    return dict(base,status='EXTRACTED',feature=feature,
                target_feature_sha256=shared.array_hash(feature),
                original_feature_sha256=shared.array_hash(original_feature))


def failed_record(base,error):
    return dict(base,status='FAILED',error=str(error),
                label_preserving_main_eligible=False)


def validate_planned(rows,samples,modality):
    if modality not in ('text','audio'):
        raise ValueError('Unsupported modality in fixed plan')
    for row in rows:
        family=row.get('perturbation_type')
        if (family not in shared.TYPES or row.get('variant_id') not in shared.TYPES[family]
            or (family in ('neutral_append','semantic_paraphrase'))!=(modality=='text')):
            raise ValueError('Fixed perturbation family/variant/modality mismatch')
        if row.get('modality',modality)!=modality:
            raise ValueError('Row modality differs from planned family')
    expected={(sid,variant) for sid in samples for family,variants in shared.TYPES.items()
              if (family in ('neutral_append','semantic_paraphrase'))==(modality=='text')
              for variant in variants}
    observed=[(row['sample_id'],row['variant_id']) for row in rows]
    if len(observed)!=len(set(observed)) or set(observed)!=expected:
        raise ValueError('Every fixed '+modality+' variant must be present exactly once, including failures')


def validate_target_identity(row,sample):
    if (row.get('sample_id')!=sample['sample_id'] or str(row.get('dialogue_id'))!=str(sample['dialogue_id'])
        or int(row.get('utterance_index',-1))!=int(sample['utterance_index'])):
        raise ValueError('Record target differs from fixed sample identity')


def atomic_bundle(path,bundle):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    torch.save(bundle,temporary);temporary.replace(path)


def state_fingerprint(model):
    tensors={name:shared.array_hash(value) for name,value in sorted(model.state_dict().items())}
    bn={}
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
            for key in ('running_mean','running_var','num_batches_tracked'):
                value=getattr(module,key)
                bn[name+'.'+key]=shared.array_hash(value) if value is not None else None
    digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    return dict(state_sha256=digest(tensors),tensor_count=len(tensors),tensor_hashes=tensors,
                bn_sha256=digest(bn),bn_buffers=bn)


def model_dir(path):
    path=Path(path)
    return path/'best_classifier' if (path/'best_classifier').is_dir() else path


def common_inputs(args):
    paths=[args.dataset_dir/'dataset.pt',args.text_features,args.audio_features,args.checkpoint,
           ROOT/'configs/PLAN_B_PROTOCOL.json',Path(__file__),Path(shared.__file__),
           ROOT/'scripts/raw_path_smoke.py',ROOT/'scripts/text_encoder.py',ROOT/'scripts/audio_encoder.py']
    inputs={}
    for path in paths:bind_file(inputs,path)
    bind_file(inputs,ROOT/'scripts/text_encoder.py',TEXT_SHA)
    bind_file(inputs,ROOT/'scripts/audio_encoder.py',AUDIO_SHA)
    for path in model_dir(args.text_model).iterdir():
        if path.is_file():bind_file(inputs,path)
    training_lock=args.text_model/'best_model_lock.json'
    if training_lock.exists():bind_file(inputs,training_lock)
    bind_file(inputs,args.text_features.with_suffix('.manifest.json'))
    bind_file(inputs,ROOT/'provenance/audio_model_files.json')
    return inputs


def prepare(args):
    runtime=shared.require_runtime('encoder')
    inputs=common_inputs(args)
    from raw_path_smoke import verify_baseline_complete,selected_target
    complete=verify_baseline_complete(args,inputs)
    bind_file(inputs,complete['path'],complete['sha256'])
    if args.mode=='pilot':
        if not args.baseline_lock or not args.pilot_output or not args.text_review:
            raise ValueError('Pilot needs baseline lock, fixed pilot output and text review')
        helper_args=SimpleNamespace(**vars(args));helper_args.output=args.pilot_output
        data,dialogues,text,audio=shared.load_inputs(helper_args)
        samples=shared.read_samples(helper_args)
        for path in [args.baseline_lock,args.pilot_output/'sample_manifest.csv',
                     args.pilot_output/'selection_frozen.json',args.text_review,
                     args.local_dir/'audio_transformations.jsonl',args.audio_review]:
            bind_file(inputs,path)
        selection=None
    else:
        # Shared containers deserialize; no Validation dictionary is indexed here.
        data=torch.load(args.dataset_dir/'dataset.pt',map_location='cpu',weights_only=False)
        text=torch.load(args.text_features,map_location='cpu',weights_only=False)
        audio=torch.load(args.audio_features,map_location='cpu',weights_only=False)
        if data['dataset']!=args.dataset:raise ValueError('Wrong smoke dataset')
        dialogue,index,selection=selected_target(data,audio)
        did=str(dialogue['dialogue_id']);dialogues={did:dialogue}
        samples=[dict(sample_id='train_smoke',dialogue_id=did,utterance_index=index,
                      utterance_id=str(dialogue['utterance_ids'][index]),
                      original_text_sha256=text_sha(dialogue['texts'][index]))]
    split='validation' if args.mode=='pilot' else 'train'
    text_metadata=json.loads(args.text_features.with_suffix('.manifest.json').read_text())
    if (text_metadata['dataset']!=args.dataset or text_metadata['data_sha256']!=inputs[str((args.dataset_dir/'dataset.pt').resolve())]
        or text_metadata['script_sha256']!=TEXT_SHA
        or text_metadata['feature_file_sha256']!=inputs[str(args.text_features.resolve())]
        or text_metadata['model_sha256']!=inputs[str((model_dir(args.text_model)/'model.safetensors').resolve())]):
        raise ValueError('Clean text cache is not bound to this data/encoder/weights')
    for name,digest in text_metadata['tokenizer_hashes'].items():
        bind_file(inputs,model_dir(args.text_model)/name,digest)
    if audio['metadata']['dataset_sha256']!=inputs[str((args.dataset_dir/'dataset.pt').resolve())]:
        raise ValueError('Clean audio cache has a different dataset source')
    text_dialogues={str(row['dialogue_id']):row for row in text_metadata['dialogue_features'][split]}
    for sample in samples:
        did=str(sample['dialogue_id']);index=int(sample['utterance_index']);dialogue=dialogues[did]
        if text_sha(dialogue['texts'][index])!=sample['original_text_sha256']:
            raise ValueError('Fixed target original text changed')
        if list(map(str,text_dialogues[did]['utterance_ids']))!=list(map(str,dialogue['utterance_ids'])):
            raise ValueError('Text cache utterance order mismatch')
        if (tuple(text[split][did].shape)!=(len(dialogue['texts']),4,1024)
            or tuple(audio[split][did].shape)!=(len(dialogue['texts']),768)):
            raise ValueError('Clean cache dimensions changed')
    return runtime,inputs,complete,data,dialogues,text,audio,samples,selection,split


def run(args):
    if args.output.exists() or args.output.with_suffix('.manifest.json').exists():
        raise FileExistsError('Refuse existing producer bundle/manifest')
    runtime,inputs,complete,data,dialogues,text,audio,samples,selection,split=prepare(args)
    import text_encoder as text_module
    import audio_encoder as audio_module
    # Source hashes are checked before importing/instantiating either real encoder.
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    text_model,tokenizer=text_module.load_encoder(args.text_model,device=args.device)
    text_model.eval().requires_grad_(False)
    audio_model=audio_module.load_audio_encoder(device=args.device)
    for key in ('protocol_hash','checkpoint_manifest','encoder_code_sha256'):
        if audio['metadata'][key]!=audio_model.metadata[key]:
            raise ValueError('Clean and perturbed audio encoder differ: '+key)
    for key in ('torch','transformers'):
        if audio['metadata'][key]!=audio_model.metadata[key]:
            raise ValueError('Clean and perturbed audio runtime differ: '+key)
    for item in audio_model.metadata['checkpoint_manifest']['files']:
        path=audio_module.MODEL_DIR/item['name']
        if path.is_file():bind_file(inputs,path,item['sha256'])
    bind_file(inputs,audio_module.FFMPEG)
    for name,module in [('text',text_model),('audio',audio_model.model)]:
        if (any(sub.training for sub in module.modules()) or any(p.requires_grad for p in module.parameters())
            or any(p.dtype!=torch.float32 for p in module.parameters())):
            raise RuntimeError(name+' encoder is not frozen FP32 eval')
    encoder_states_before={name:state_fingerprint(module) for name,module in
                           [('text',text_model),('audio',audio_model.model)]}
    args.local_dir.mkdir(parents=True,exist_ok=True)
    controls={};waves={}
    for sample in samples:
        sid=sample['sample_id'];did=str(sample['dialogue_id']);index=int(sample['utterance_index'])
        dialogue=dialogues[did];original=dialogue['texts'][index];source=dialogue['audio_paths'][index]
        controls[sid]={}
        for modality in ('text','audio'):
            base=dict(sample_id=sid,dialogue_id=did,utterance_index=index,modality=modality,
                      original_feature_sha256=shared.array_hash(text[split][did][index] if modality=='text' else audio[split][did][index]))
            try:
                if modality=='text':
                    base.update(original_input_sha256=text_sha(original),processed_input_sha256=text_sha(original),
                                **actual_text_input(text_module,tokenizer,original,args.dataset))
                    feature=text_module.extract_texts(text_model,tokenizer,[original],dataset=args.dataset,batch_size=1)[0]
                    original_feature=text[split][did][index]
                else:
                    meta=audio['utterance_metadata'][split][did][index]
                    if bool(audio['audio_missing_mask'][split][did][index]):raise ValueError('Preserve clean missing-audio mask')
                    if str(meta['utterance_id'])!=str(dialogue['utterance_ids'][index]):
                        raise ValueError('Audio cache utterance ID mismatch')
                    source_hash=bind_file(inputs,source,meta['source_sha256'])
                    wave,decode=audio_module.read_audio_16k(source,cache_dir=args.local_dir/'decode_cache')
                    feature=audio_module.extract_waveform(wave,encoder=audio_model)
                    if audio_model.last_metadata.get('audio_missing'):raise ValueError('Declared available source decoded as missing')
                    waves[sid]=wave
                    base.update(original_input_sha256=source_hash,processed_input_sha256=source_hash,
                                original_path=str(Path(source).resolve()),decode_metadata=decode,
                                waveform_sha256=shared.array_hash(wave),audio_extraction_metadata=dict(audio_model.last_metadata))
                    original_feature=audio[split][did][index]
                controls[sid][modality]=extracted_record(base,feature,original_feature)
            except Exception as error:
                controls[sid][modality]=failed_record(base,error)
    variants=[]
    if args.mode=='smoke':
        sample=samples[0];sid=sample['sample_id'];did=str(sample['dialogue_id']);index=int(sample['utterance_index'])
        original=dialogues[did]['texts'][index]
        changed=original+' Page two.'
        text_rows=[dict(sample_id=sid,dialogue_id=did,utterance_index=index,
                       perturbation_type='neutral_append',variant_id='neutral_append',processed_text=changed)]
        text_assessments={(sid,'neutral_append'):shared.text_qc(original,changed,'neutral_append',tokenizer,args.dataset)}
        audio_rows=[dict(sample_id=sid,dialogue_id=did,utterance_index=index,
                        perturbation_type='pitch_shift',variant_id='pitch_plus_0.5',status='SMOKE_GENERATE')]
        review_lookup={}
        text_asset=args.local_dir/'smoke_texts.json'
        if text_asset.exists():raise FileExistsError('Do not overwrite previous smoke text assets')
        shared.save_json(text_asset,dict(original_text=original,processed_text=changed,
            review_status='NOT_ASSESSED_ENGINEERING_ONLY',sample_id=sid))
        bind_file(inputs,text_asset)
    else:
        helper_args=SimpleNamespace(**vars(args));helper_args.output=args.pilot_output
        text_rows,text_assessments=shared.assessed_text_reviews(helper_args,dialogues,tokenizer)
        audio_rows=shared.read_jsonl(args.local_dir/'audio_transformations.jsonl')
        reviews=shared.read_csv(args.audio_review)
        review_lookup={(row['sample_id'],row['variant_id']):row for row in reviews}
        if len(review_lookup)!=len(reviews):raise ValueError('Duplicate audio label reviews')
        validate_planned(text_rows,{row['sample_id'] for row in samples},'text')
        validate_planned(audio_rows,{row['sample_id'] for row in samples},'audio')
        if set(review_lookup)!={(r['sample_id'],r['variant_id']) for r in audio_rows}:
            raise ValueError('Audio label review must preserve every fixed variant, even PENDING')
    lookup={row['sample_id']:row for row in samples}
    for modality,rows in [('text',text_rows),('audio',audio_rows)]:
        for row in rows:
            sid=row['sample_id'];sample=lookup[sid];did=str(sample['dialogue_id']);index=int(sample['utterance_index'])
            dialogue=dialogues[did];kind=row['perturbation_type'];vid=row['variant_id']
            engineering=args.mode=='smoke'
            review={} if engineering else (row if modality=='text' else review_lookup[(sid,vid)])
            qc=text_assessments[(sid,vid)] if modality=='text' else row.get('signal_qc',{})
            automatic=bool(qc.get('automatic_text_qc_pass',qc.get('automatic_signal_qc_pass',False)))
            base=dict(sample_id=sid,dialogue_id=did,utterance_index=index,modality=modality,
                 perturbation_type=kind,variant_id=vid,planned_k=1 if engineering else len(shared.TYPES[kind]),
                 quality_control=qc,automatic_qc_pass=automatic,label_preserving_main_eligible=False,
                 label_review_status='NOT_ASSESSED_ENGINEERING_ONLY' if engineering else review.get('review_status','PENDING'),
                 reviewer=review.get('reviewer',''),review_method=review.get('review_method',''),
                 review_notes=review.get('review_notes',''),
                 original_feature_sha256=shared.array_hash(text[split][did][index] if modality=='text' else audio[split][did][index]))
            try:
                if str(row['dialogue_id'])!=did or int(row['utterance_index'])!=index:
                    raise ValueError('Variant no longer targets the fixed utterance')
                if modality=='text':
                    original=dialogue['texts'][index];changed=row.get('processed_text','')
                    base.update(original_input_sha256=text_sha(original),processed_input_sha256=text_sha(changed),
                                **actual_text_input(text_module,tokenizer,changed,args.dataset))
                    if not automatic:raise ValueError('Fixed variant failed automatic text/token QC')
                    feature=text_module.extract_texts(text_model,tokenizer,[changed],dataset=args.dataset,batch_size=1)[0]
                    original_feature=text[split][did][index]
                else:
                    if not engineering:
                        if (review.get('review_status','PENDING').upper() not in shared.REVIEW_STATUSES or
                            str(review['dialogue_id'])!=did or int(review['utterance_index'])!=index):
                            raise ValueError('Audio review status/target identity differs from frozen sample')
                    if controls[sid]['audio']['status']!='EXTRACTED':
                        raise ValueError('Clean original audio control failed: '+controls[sid]['audio']['error'])
                    source=dialogue['audio_paths'][index]
                    source_hash=bind_file(inputs,source,audio['utterance_metadata'][split][did][index]['source_sha256'])
                    if engineering:
                        ffmpeg=args.ffmpeg or audio_module.FFMPEG
                        bind_file(inputs,ffmpeg)
                        changed,parameters,qc=shared.transform_waveform(waves[sid],vid,ffmpeg)
                        destination=args.local_dir/'smoke_pitch_plus_0.5.wav'
                        clean_path=args.local_dir/'smoke_clean.wav'
                        if destination.exists() or clean_path.exists():raise FileExistsError('Refuse previous smoke audio assets')
                        wavfile.write(destination,16000,changed);wavfile.write(clean_path,16000,waves[sid])
                        processed_hash=bind_file(inputs,destination);bind_file(inputs,clean_path)
                        base.update(processed_path=str(destination.resolve()),clean_path=str(clean_path.resolve()))
                    else:
                        if row.get('status')!='CREATED':raise ValueError(row.get('error','Fixed audio transformation failed'))
                        if Path(row['source_path']).resolve()!=Path(source).resolve():raise ValueError('Changed raw audio path')
                        bind_file(inputs,row['source_path'],row['source_sha256'])
                        bind_file(inputs,row['clean_path'],row['clean_sha256'])
                        processed_hash=bind_file(inputs,row['processed_path'],row['processed_sha256'])
                        if row['source_sha256']!=source_hash:raise ValueError('Variant raw source differs from clean baseline')
                        clean_wave,_=audio_module.read_audio_16k(row['clean_path'])
                        if shared.array_hash(clean_wave)!=shared.array_hash(waves[sid]):
                            raise ValueError('Saved transformation clean waveform differs from source decode')
                        changed,decode=audio_module.read_audio_16k(row['processed_path'])
                        parameters=row['parameters'];qc=row['signal_qc']
                        base.update(processed_path=str(Path(row['processed_path']).resolve()),
                                    clean_path=str(Path(row['clean_path']).resolve()),decode_metadata=decode)
                    automatic=bool(qc.get('automatic_signal_qc_pass',False))
                    base.update(original_input_sha256=source_hash,processed_input_sha256=processed_hash,
                         transformation_parameters=parameters,quality_control=qc,automatic_qc_pass=automatic,
                         waveform_sha256=shared.array_hash(changed))
                    if not automatic:raise ValueError('Fixed audio variant failed signal QC')
                    feature=audio_module.extract_waveform(changed,encoder=audio_model)
                    if audio_model.last_metadata.get('audio_missing'):raise ValueError('Processed waveform encoder marked missing')
                    base['audio_extraction_metadata']=dict(audio_model.last_metadata)
                    original_feature=audio[split][did][index]
                base['label_preserving_main_eligible']=bool(not engineering and automatic and shared.review_pass(review,modality))
                variants.append(extracted_record(base,feature,original_feature))
            except Exception as error:
                variants.append(failed_record(base,error))
    for path,digest in inputs.items():
        if shared.sha256(path)!=digest:raise ValueError('A bound source changed during extraction: '+path)
    for sid,sample in lookup.items():
        if set(controls[sid])!={'text','audio'}:raise ValueError('Missing fixed clean control modality')
        for modality,record in controls[sid].items():
            validate_target_identity(record,sample)
            if record['modality']!=modality:raise ValueError('Clean control modality/key mismatch')
    for record in variants:validate_target_identity(record,lookup[record['sample_id']])
    encoder_states_after={name:state_fingerprint(module) for name,module in
                          [('text',text_model),('audio',audio_model.model)]}
    if encoder_states_after!=encoder_states_before:
        raise RuntimeError('Loaded encoder weights/buffers changed during extraction')
    if any(sub.training for module in (text_model,audio_model.model) for sub in module.modules()):
        raise RuntimeError('An encoder left eval mode during extraction')
    metadata=dict(format=FORMAT,mode=args.mode,dataset=args.dataset,status='COMPLETE',
        runtime=runtime,scientific_protocol_sha256=shared.sha256(ROOT/'configs/PLAN_B_PROTOCOL.json'),
        input_sha256=inputs,encoder_provenance=dict(
           text=dict(weights_sha256=shared.sha256(model_dir(args.text_model)/'model.safetensors'),
             encoder_code_sha256=TEXT_SHA,clean_manifest_sha256=shared.sha256(args.text_features.with_suffix('.manifest.json')),
             tokenizer_files={path.name:shared.sha256(path) for path in model_dir(args.text_model).iterdir()
                              if path.is_file() and path.name in ('vocab.json','merges.txt','tokenizer_config.json','special_tokens_map.json')},
             normalization='none',layer_order=[-1,-2,-3,-4],precision='FP32 eval',all_parameters_frozen=True),
           audio=audio_model.metadata),baseline_completion=complete,
        encoder_states_unchanged=True,input_files_unchanged=True,
        encoder_states_before=encoder_states_before,encoder_states_after=encoder_states_after,
        baseline_lock_sha256=shared.sha256(args.baseline_lock) if args.mode=='pilot' else None,
        sample_manifest_sha256=shared.sha256(args.pilot_output/'sample_manifest.csv') if args.mode=='pilot' else None,
        producer_command=sys.argv,created_utc=now(),numpy=np.__version__,
        scipy=importlib.metadata.version('scipy'),actual_token_hash_format='array_hash(int64 actual input IDs including BOS/EOS after truncation)',
        variants_planned=len(variants),variants_extracted=sum(r['status']=='EXTRACTED' for r in variants),
        variants_failed=sum(r['status']=='FAILED' for r in variants),
        official_validation_indexed=args.mode=='pilot',test_indexed=False,no_cmerc_inference_performed=True)
    bundle=dict(metadata=metadata,clean_controls=controls,variants=variants)
    if selection is not None:bundle['selection']=selection
    atomic_bundle(args.output,bundle)
    shared.save_json(args.output.with_suffix('.manifest.json'),dict(format=FORMAT,
        status='COMPLETE',mode=args.mode,dataset=args.dataset,cache_sha256=shared.sha256(args.output),
        metadata=metadata))
    print(json.dumps(dict(status='COMPLETE',dataset=args.dataset,mode=args.mode,
       cache=str(args.output),variants=len(variants),failed=metadata['variants_failed'])),flush=True)


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['smoke','pilot'])
    parser.add_argument('--dataset',required=True,choices=['IEMOCAP','MELD'])
    for name in ('dataset-dir','text-features','audio-features','checkpoint','text-model','output','local-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('baseline-lock','pilot-output','text-review','audio-review'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--device',required=True)
    parser.add_argument('--ffmpeg')
    args=parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.resolve())
    if args.mode=='pilot' and args.audio_review is None:
        args.audio_review=args.local_dir/'audio_label_review.csv'
    if torch.device(args.device).type!='cuda' or not torch.cuda.is_available():
        raise ValueError('Use an explicitly assigned healthy CUDA GPU')
    return args


if __name__=='__main__':
    args=parse_args()
    try:
        run(args)
    except Exception as error:
        failure=args.output.with_suffix('.failure.json')
        if not args.output.exists() and not failure.exists():
            shared.save_json(failure,dict(status='FAILED',mode=args.mode,dataset=args.dataset,
                 error=str(error),traceback=traceback.format_exc(),created_utc=now()))
        raise
