#!/usr/bin/env python3
"""One fixed Train Fit target: real clean/raw-input path engineering check only.

No Validation dictionary is indexed, no training occurs and no emotion-preservation
PASS is assigned. Failures retain their fixed thresholds and all collected evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from raw_pilot import (array_hash, assert_target_only, checked_probabilities,
                       save_json, sha256, text_qc, transform_waveform)

ROOT = Path(__file__).resolve().parents[1]
NEUTRAL_SUFFIX = ' Page two.'
FEATURE_ATOL = 1e-4
PROBABILITY_ATOL = 1e-6


def now():
    return datetime.now(timezone.utc).isoformat()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def state_fingerprint(model):
    result = {}
    for name, tensor in sorted(model.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Unexpected non-Tensor model state: ' + name)
        result[name] = array_hash(tensor)
    return dict(sha256=json_hash(result), tensor_count=len(result), tensors=result)


def bn_fingerprint(model):
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for field in ['running_mean', 'running_var', 'num_batches_tracked']:
                value = getattr(module, field)
                result[name + '.' + field] = array_hash(value) if value is not None else None
    return dict(sha256=json_hash(result), buffers=result)


def dialogue_fingerprint(dialogue):
    return dict(dialogue_id=str(dialogue['dialogue_id']),
                utterance_ids=json_hash(list(map(str, dialogue['utterance_ids']))),
                speaker=array_hash(dialogue['speaker']), labels=array_hash(dialogue['labels']),
                visual=array_hash(dialogue['visual']), texts=json_hash(dialogue['texts']),
                audio_paths=json_hash([str(value) for value in dialogue['audio_paths']]))


def feature_comparison(original, rebuilt, layers=False):
    original = torch.as_tensor(original).detach().cpu()
    rebuilt = torch.as_tensor(rebuilt).detach().cpu()
    shape_ok = original.shape == rebuilt.shape
    dtype_ok = original.dtype == rebuilt.dtype == torch.float32
    result = dict(original_shape=list(original.shape), rebuilt_shape=list(rebuilt.shape),
                  original_dtype=str(original.dtype), rebuilt_dtype=str(rebuilt.dtype),
                  shape_pass=shape_ok, fp32_pass=dtype_ok, atol=FEATURE_ATOL, rtol=0.)
    if not shape_ok:
        result.update(pass_gate=False, max_abs_error=None, mae=None, rmse=None)
        return result
    a, b = original.numpy().astype(np.float64), rebuilt.numpy().astype(np.float64)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        result.update(finite=False, pass_gate=False, max_abs_error=None, mae=None, rmse=None)
        return result
    difference = a - b
    result.update(finite=finite, max_abs_error=float(np.abs(difference).max()),
                  mae=float(np.abs(difference).mean()), rmse=float(np.sqrt(np.mean(difference**2))),
                  pass_gate=bool(dtype_ok and np.allclose(a, b, atol=FEATURE_ATOL, rtol=0)))
    if layers:
        result['layers'] = [dict(layer_from_end=i+1,
            max_abs_error=float(np.max(np.abs(difference[i]))),
            mae=float(np.mean(np.abs(difference[i]))),
            rmse=float(np.sqrt(np.mean(difference[i]**2)))) for i in range(len(a))]
    return result


def probability_comparison(original, changed):
    original = checked_probabilities(original); changed = checked_probabilities(changed)
    if original.shape != changed.shape:
        raise ValueError('Full-Dialogue probability shape changed')
    difference = np.abs(original - changed)
    return dict(shape=list(original.shape), finite_normalized=True,
                max_abs_error=float(difference.max()), mean_abs_error=float(difference.mean()),
                class_agreement=float(np.mean(original.argmax(1) == changed.argmax(1))),
                confidence_max_abs_error=float(np.max(np.abs(original.max(1)-changed.max(1)))),
                atol=PROBABILITY_ATOL, rtol=0.,
                pass_gate=bool(np.allclose(original, changed, atol=PROBABILITY_ATOL, rtol=0)))


def selected_target(data, audio):
    """First declared Fit Dialogue; first available audio row therein, no effect search."""
    fit = [str(value) for value in data['fit_dialogue_ids']]
    inner_dev = {str(value) for value in data['inner_dev_dialogue_ids']}
    if not fit or fit[0] in inner_dev:
        raise ValueError('No first Fit Dialogue, or it overlaps internal dev')
    first = fit[0]
    matches = [item for item in data['splits']['train'] if str(item['dialogue_id']) == first]
    if len(matches) != 1:
        raise ValueError('First Fit Dialogue does not identify exactly one Train Dialogue')
    dialogue = matches[0]
    mask = audio['audio_missing_mask']['train'][first]
    if len(mask) != len(dialogue['utterance_ids']):
        raise ValueError('Train audio missing mask length mismatch')
    skipped = []
    for index, path in enumerate(dialogue['audio_paths']):
        if bool(mask[index]):
            skipped.append(dict(utterance_index=index, reason='Declared clean audio_missing_mask=true'))
            continue
        if path is None or not Path(path).is_file():
            skipped.append(dict(utterance_index=index, reason='Original audio file absent'))
            continue
        return dialogue, index, dict(first_fit_dialogue_id=first, utterance_index=index,
            utterance_id=str(dialogue['utterance_ids'][index]), earlier_rows_skipped=skipped,
            selection_rule='First fit_dialogue_ids entry, first row with declared nonmissing and present original audio; never skip an eligible row based on responses')
    raise ValueError('First Train Fit Dialogue has no valid audio target; do not choose another Dialogue')


def encoder_path(model_path):
    model_path = Path(model_path)
    return model_path / 'best_classifier' if (model_path / 'best_classifier').is_dir() else model_path


def verify_baseline_complete(args, inputs):
    path = args.checkpoint.parent / 'completion.json'
    complete = json.loads(path.read_text())
    if complete.get('status') != 'COMPLETE' or complete.get('stage') != 'cmerc':
        raise RuntimeError('CMERC baseline selection is unfinished; do not run raw-input smoke')
    if complete.get('best_checkpoint_sha256') != inputs[str(args.checkpoint)]:
        raise ValueError('Checkpoint is not the frozen selected baseline')
    recorded = {str(Path(path).resolve()): digest for path, digest in complete['input_hashes'].items()}
    for path in [args.dataset_dir / 'dataset.pt', args.text_features, args.audio_features]:
        if recorded.get(str(path.resolve())) != inputs[str(path)]:
            raise ValueError('Frozen baseline was trained on different inputs: ' + str(path))
    return dict(path=str(path), sha256=sha256(path), completed_at=complete.get('completed_at_utc'))


def run(args, report):
    import audio_encoder as audio_module
    import text_encoder as text_module
    import downstream
    from audio_encoder import load_audio_encoder, read_audio_16k, extract_waveform
    from text_encoder import load_encoder, extract_texts

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Frozen XP04 equations require an explicitly assigned healthy CUDA device')
    paths = [args.dataset_dir / 'dataset.pt', args.text_features, args.audio_features, args.checkpoint]
    text_dir = encoder_path(args.text_model)
    paths.append(text_dir / 'model.safetensors')
    input_hashes = {str(path): sha256(path) for path in paths}
    report['input_hashes'] = input_hashes
    report['baseline_completion'] = verify_baseline_complete(args, input_hashes)
    report['code_hashes'] = {str(Path(module.__file__)): sha256(module.__file__)
                           for module in [audio_module, text_module, downstream]}
    # These shared NEW-feature containers may contain Validation sections; only
    # Train values are indexed below. Never construct downstream.Bundle here.
    data = torch.load(args.dataset_dir / 'dataset.pt', map_location='cpu')
    text = torch.load(args.text_features, map_location='cpu')
    audio = torch.load(args.audio_features, map_location='cpu')
    if data['dataset'] != args.dataset:
        raise ValueError('Dataset identity mismatch')
    dialogue, index, selection = selected_target(data, audio)
    did = str(dialogue['dialogue_id']); report['selection'] = selection
    original_text = torch.as_tensor(text['train'][did]).clone()
    original_audio = torch.as_tensor(audio['train'][did]).clone()
    n = len(dialogue['utterance_ids'])
    if tuple(original_text.shape) != (n,4,1024) or tuple(original_audio.shape) != (n,768):
        raise ValueError('Expected new-feature dimensions')
    if original_text.dtype != torch.float32 or original_audio.dtype != torch.float32:
        raise TypeError('Clean caches must be FP32 without silent conversion')
    before_dialogue = dialogue_fingerprint(dialogue)
    before_cache = dict(text=array_hash(text['train'][did]), audio=array_hash(audio['train'][did]))
    report['original_dialogue_hashes'] = before_dialogue
    source = Path(dialogue['audio_paths'][index])
    source_sha = sha256(source)
    if audio['utterance_metadata']['train'][did][index]['source_sha256'] != source_sha:
        raise ValueError('Original source audio differs from clean cache provenance')
    if str(audio['utterance_metadata']['train'][did][index]['utterance_id']) != str(dialogue['utterance_ids'][index]):
        raise ValueError('Audio cache utterance index/order differs from raw input')
    if audio['metadata']['dataset_sha256'] != input_hashes[str(args.dataset_dir / 'dataset.pt')]:
        raise ValueError('Audio cache came from a different dataset container')
    input_hashes[str(source)] = source_sha
    text_manifest_path = args.text_features.with_suffix('.manifest.json')
    text_manifest = json.loads(text_manifest_path.read_text())
    if (text_manifest['model_sha256'] != input_hashes[str(text_dir / 'model.safetensors')]
            or text_manifest['feature_file_sha256'] != input_hashes[str(args.text_features)]
            or text_manifest['dataset'] != args.dataset
            or text_manifest['data_sha256'] != input_hashes[str(args.dataset_dir / 'dataset.pt')]
            or text_manifest['script_sha256'] != sha256(text_module.__file__)):
        raise ValueError('Text clean and smoke extractor source/weights/feature provenance differs')
    for name, digest in text_manifest['tokenizer_hashes'].items():
        if sha256(text_dir / name) != digest: raise ValueError('Tokenizer differs from clean cache')
    input_hashes[str(text_manifest_path)] = sha256(text_manifest_path)
    matching=[row for row in text_manifest['dialogue_features']['train'] if str(row['dialogue_id'])==did]
    if len(matching)!=1 or list(map(str,matching[0]['utterance_ids']))!=list(map(str,dialogue['utterance_ids'])):
        raise ValueError('Text cache utterance index/order differs from raw input')

    model = downstream.load_model(args.dataset_dir,args.checkpoint,device=args.device)
    text_encoder, tokenizer = load_encoder(args.text_model,device=args.device)
    audio_encoder = load_audio_encoder(device=args.device)
    for key in ['protocol_hash','checkpoint_manifest','encoder_code_sha256']:
        if audio['metadata'][key] != audio_encoder.metadata[key]:
            raise ValueError('Audio clean/smoke encoder identity differs: '+key)
    models={'cmerc':model,'text_encoder':text_encoder,'audio_encoder':audio_encoder.model}
    for module in models.values(): module.eval().requires_grad_(False)
    before_states={name:state_fingerprint(module) for name,module in models.items()}
    before_bn={name:bn_fingerprint(module) for name,module in models.items()}
    report['model_state_before']={name:{k:v for k,v in value.items() if k!='tensors'} for name,value in before_states.items()}
    report['bn_before']=before_bn
    report['models_all_eval_and_frozen']=all(all(not sub.training for sub in module.modules())
        and not any(p.requires_grad for p in module.parameters()) for module in models.values())
    report['models_fp32']=all(all(p.dtype==torch.float32 for p in module.parameters()) for module in models.values())
    if not report['models_all_eval_and_frozen'] or not report['models_fp32']:
        raise ValueError('All encoder/CMERC modules must be frozen FP32 eval')
    report['precision']='FP32, no autocast or TF32'

    baseline=checked_probabilities(downstream.infer_dialogue(model,dialogue,original_text,original_audio))
    repeat=checked_probabilities(downstream.infer_dialogue(model,dialogue,original_text,original_audio))
    report['baseline_repeat']=probability_comparison(baseline,repeat)
    case_results={}; local_artifacts={}
    input_text=dialogue['texts'][index]
    clean_wave,decode_metadata=read_audio_16k(source,cache_dir=args.local_dir/'decode_cache')
    if not len(clean_wave) or not np.isfinite(clean_wave).all():
        raise ValueError('Declared valid target decodes to invalid audio')
    clean_path=args.local_dir/'clean.wav';wavfile.write(clean_path,16000,np.asarray(clean_wave,dtype=np.float32))
    report['audio_decode']=decode_metadata
    report['audio_encoder_metadata']=audio_encoder.metadata
    texts_path=args.local_dir/'texts.json'
    save_json(texts_path,dict(original_text=input_text,neutral_append=input_text+NEUTRAL_SUFFIX,
       dialogue_context=[dict(utterance_index=i,speaker_id=int(torch.as_tensor(dialogue['speaker'][i]).argmax()),text=t)
                         for i,t in enumerate(dialogue['texts'])],semantic_review_status='NOT_PERFORMED_ENGINEERING_ONLY'))

    def forward_case(name,modality,feature):
        changed_text,changed_audio=original_text.clone(),original_audio.clone()
        if modality=='text': changed_text[index]=feature
        elif modality=='audio': changed_audio[index]=feature
        else: raise ValueError('Exactly one target modality per case')
        assert_target_only(original_text,original_audio,changed_text,changed_audio,index,modality)
        before=dialogue_fingerprint(dialogue)
        probabilities=checked_probabilities(downstream.infer_dialogue(model,dialogue,changed_text,changed_audio))
        assert_target_only(original_text,original_audio,changed_text,changed_audio,index,modality)
        if before!=dialogue_fingerprint(dialogue): raise AssertionError('Dialogue context mutated during inference')
        if probabilities.shape!=(n,len(data['class_names'])):raise ValueError('Invalid full-Dialogue probability shape')
        path=args.local_dir/(name+'.pt')
        torch.save(dict(target_feature=torch.as_tensor(feature).detach().cpu(),full_dialogue_probabilities=torch.from_numpy(probabilities),
                        dialogue_id=did,utterance_index=index,modality=modality),path)
        local_artifacts[str(path)]=sha256(path)
        return probabilities,dict(full_dialogue_shape=list(probabilities.shape),finite_normalized=True,
             only_target_feature_row_changed=True,other_modality_bit_exact=True,
             other_utterances_bit_exact=True,visual_speaker_labels_ids_order_bit_exact=True,
             original_target_feature_sha256=array_hash(original_text[index] if modality=='text' else original_audio[index]),
             replaced_target_feature_sha256=array_hash(feature),emotion_label_preservation='NOT_ASSESSED_ENGINEERING_ONLY')

    # Each case continues collecting evidence after a numerical failure; no model
    # setting/threshold is changed and the final status remains FAIL.
    for name in ['clean_text','clean_audio','neutral_append','pitch_plus_0.5']:
        try:
            if name=='clean_text':
                feature=extract_texts(text_encoder,tokenizer,[input_text],dataset=args.dataset,batch_size=1)[0]
                probabilities,record=forward_case(name,'text',feature)
                record['feature_comparison']=feature_comparison(original_text[index],feature,layers=True)
                record['probability_comparison']=probability_comparison(baseline,probabilities)
                record['pass_gate']=record['feature_comparison']['pass_gate'] and record['probability_comparison']['pass_gate']
            elif name=='clean_audio':
                feature=extract_waveform(clean_wave,encoder=audio_encoder)
                probabilities,record=forward_case(name,'audio',feature)
                record['feature_comparison']=feature_comparison(original_audio[index],feature)
                record['probability_comparison']=probability_comparison(baseline,probabilities)
                record['pass_gate']=record['feature_comparison']['pass_gate'] and record['probability_comparison']['pass_gate']
            elif name=='neutral_append':
                changed=input_text+NEUTRAL_SUFFIX
                qc=text_qc(input_text,changed,'neutral_append',tokenizer,args.dataset)
                if not qc['automatic_text_qc_pass']:raise ValueError('Fixed short suffix failed token/content engineering QC: '+json.dumps(qc))
                feature=extract_texts(text_encoder,tokenizer,[changed],dataset=args.dataset,batch_size=1)[0]
                probabilities,record=forward_case(name,'text',feature)
                record.update(token_qc=qc,pass_gate=True,not_required_to_flip_prediction=True)
            else:
                waveform,parameters,qc=transform_waveform(clean_wave,'pitch_plus_0.5',args.ffmpeg)
                wav_path=args.local_dir/'pitch_plus_0.5.wav';wavfile.write(wav_path,16000,waveform)
                feature=extract_waveform(waveform,encoder=audio_encoder)
                probabilities,record=forward_case(name,'audio',feature)
                record.update(parameters=parameters,signal_qc=qc,pass_gate=qc['automatic_signal_qc_pass'],
                              not_required_to_flip_prediction=True)
                local_artifacts[str(wav_path)]=sha256(wav_path)
            record['status']='PASS' if record['pass_gate'] else 'FAIL'
            case_results[name]=record
        except Exception as error:
            case_results[name]=dict(status='FAIL',pass_gate=False,error=str(error),traceback=traceback.format_exc(),
                                   emotion_label_preservation='NOT_ASSESSED_ENGINEERING_ONLY')
        report['cases']=case_results
        save_json(args.report,report)

    state_same={name:state_fingerprint(module)['sha256']==before_states[name]['sha256'] for name,module in models.items()}
    bn_same={name:bn_fingerprint(module)==before_bn[name] for name,module in models.items()}
    cache_same=before_cache==dict(text=array_hash(text['train'][did]),audio=array_hash(audio['train'][did]))
    context_same=before_dialogue==dialogue_fingerprint(dialogue)
    file_same={path:sha256(path)==digest for path,digest in input_hashes.items()}
    report['invariants']=dict(model_state_bit_exact=state_same,bn_buffers_bit_exact=bn_same,
        clean_feature_cache_bit_exact=cache_same,dialogue_context_bit_exact=context_same,
        input_files_unchanged=file_same,models_remain_eval=all(all(not sub.training for sub in m.modules()) for m in models.values()))
    local_artifacts[str(clean_path)]=sha256(clean_path);local_artifacts[str(texts_path)]=sha256(texts_path)
    report['local_artifacts_sha256']=local_artifacts
    report['pass_gate']=bool(report['baseline_repeat']['pass_gate'] and all(r['pass_gate'] for r in case_results.values())
        and all(state_same.values()) and all(bn_same.values()) and cache_same and context_same and all(file_same.values())
        and report['invariants']['models_remain_eval'])
    report['status']='PASS' if report['pass_gate'] else 'FAIL'


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=['IEMOCAP','MELD'])
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--text-features',type=Path,required=True)
    parser.add_argument('--audio-features',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--text-model',type=Path,required=True)
    parser.add_argument('--device',required=True)
    parser.add_argument('--ffmpeg',default='/home/yangbin/miniconda3/envs/msa_extract/bin/ffmpeg')
    args=parser.parse_args()
    for name in ['dataset_dir','text_features','audio_features','checkpoint','text_model']:
        setattr(args,name,getattr(args,name).resolve())
    args.local_dir=ROOT/'local_only/raw_smoke'/args.dataset
    args.report=ROOT/'provenance'/('raw_path_smoke_'+args.dataset+'.json')
    return args


def main():
    args=parse_args()
    if args.report.exists() or args.local_dir.exists():
        raise FileExistsError('Preserve prior smoke evidence; refuse to overwrite an existing attempt')
    args.local_dir.mkdir(parents=True)
    report=dict(status='RUNNING',started_utc=now(),dataset=args.dataset,
      scope='ONE_TRAIN_FIT_TARGET_ENGINEERING_CHECK_ONLY',selection='First Train Fit Dialogue and first valid audio row',
      official_validation_indexed=False,test_indexed=False,inner_dev_indexed=False,
      shared_container_note='New dataset/features Train+Validation containers deserialize; only Train values are indexed',
      labels_used_for_training_or_accuracy=False,no_training_or_weight_updates=True,
      no_emotion_preservation_pass_claim=True,no_pilot120_executed=True,
      thresholds=dict(feature_max_abs=FEATURE_ATOL,probability_max_abs=PROBABILITY_ATOL,rtol=0,
                      threshold_scope='Plan B exact-extractor repeatability engineering gate, not old XP04 feature compatibility'),
      fixed_variants=dict(neutral_suffix=NEUTRAL_SUFFIX,pitch_semitones=.5),
      environment=dict(host=platform.node(),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                       device=args.device,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')),
      script_sha256=sha256(__file__))
    try:
        report['environment']['host_gpu_inventory']=subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True).strip().splitlines()
        report['environment']['CUDA_DEVICE_ORDER']=os.environ.get('CUDA_DEVICE_ORDER')
    except Exception as error:
        report['environment']['gpu_inventory_error']=str(error)
    save_json(args.report,report)
    try:
        run(args,report)
    except Exception as error:
        report.update(status='FAIL',pass_gate=False,error=str(error),traceback=traceback.format_exc())
    report['completed_utc']=now();save_json(args.report,report)
    print(json.dumps({key:report.get(key) for key in ['status','dataset','pass_gate','error','selection']}),flush=True)
    raise SystemExit(0 if report.get('pass_gate') else 2)


if __name__=='__main__':main()
