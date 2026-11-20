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
from raw_pilot import (array_hash, assert_target_only, checked_probabilities,
                       save_json, sha256)

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
    """CMERC-only consumer: frozen TEXT_repro vectors are the sole encoder input."""
    import downstream
    from raw_pilot import load_precomputed_features,require_runtime
    report['inference_runtime']=require_runtime('cmerc')
    report['encoder_models_instantiated_in_this_process']=False
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.device(args.device).type!='cuda' or not torch.cuda.is_available():
        raise ValueError('Frozen CMERC equations require an explicitly assigned healthy CUDA device')
    paths=[args.dataset_dir/'dataset.pt',args.text_features,args.audio_features,args.checkpoint]
    input_hashes={str(path):sha256(path) for path in paths}
    report['input_hashes']=input_hashes
    report['baseline_completion']=verify_baseline_complete(args,input_hashes)
    report['code_hashes']={str(Path(downstream.__file__)):sha256(downstream.__file__)}
    bundle=load_precomputed_features(args.precomputed_features,args,'smoke')
    report['encoder_stage_runtime']=bundle['metadata']['runtime']
    report['encoder_stage_provenance']=bundle['metadata']['encoder_provenance']
    report['precomputed_features_sha256']=sha256(args.precomputed_features)
    report['encoder_stage_weights_unchanged']=bundle['metadata']['encoder_states_unchanged']
    # No Bundle construction: only Train values from shared containers are indexed.
    data=torch.load(args.dataset_dir/'dataset.pt',map_location='cpu')
    text=torch.load(args.text_features,map_location='cpu');audio=torch.load(args.audio_features,map_location='cpu')
    if data['dataset']!=args.dataset:raise ValueError('Dataset identity mismatch')
    dialogue,index,selection=selected_target(data,audio);did=str(dialogue['dialogue_id'])
    for key in ['first_fit_dialogue_id','utterance_index','utterance_id']:
        if str(selection[key])!=str(bundle['selection'][key]):raise ValueError('Encoder/CMERC smoke selected a different Train target')
    report['selection']=selection
    original_text=torch.as_tensor(text['train'][did]).clone();original_audio=torch.as_tensor(audio['train'][did]).clone()
    n=len(dialogue['utterance_ids'])
    if tuple(original_text.shape)!=(n,4,1024) or tuple(original_audio.shape)!=(n,768):raise ValueError('Unexpected new-feature shapes')
    if original_text.dtype!=torch.float32 or original_audio.dtype!=torch.float32:raise TypeError('Clean cache must be FP32')
    before_dialogue=dialogue_fingerprint(dialogue)
    before_cache=dict(text=array_hash(text['train'][did]),audio=array_hash(audio['train'][did]))
    report['original_dialogue_hashes']=before_dialogue
    source=Path(dialogue['audio_paths'][index]);input_hashes[str(source)]=sha256(source)
    if audio['utterance_metadata']['train'][did][index]['source_sha256']!=input_hashes[str(source)]:
        raise ValueError('Original audio source differs from clean feature provenance')
    model=downstream.load_model(args.dataset_dir,args.checkpoint,device=args.device).eval().requires_grad_(False)
    state=state_fingerprint(model);bn=bn_fingerprint(model)
    report['model_state_before']={'cmerc':{k:v for k,v in state.items() if k!='tensors'}}
    report['bn_before']={'cmerc':bn}
    report['models_all_eval_and_frozen']=all(not child.training for child in model.modules()) and not any(p.requires_grad for p in model.parameters())
    report['models_fp32']=all(p.dtype==torch.float32 for p in model.parameters())
    if not report['models_all_eval_and_frozen'] or not report['models_fp32']:raise ValueError('CMERC must be frozen FP32 eval')
    baseline=checked_probabilities(downstream.infer_dialogue(model,dialogue,original_text,original_audio))
    repeat=checked_probabilities(downstream.infer_dialogue(model,dialogue,original_text,original_audio))
    report['baseline_repeat']=probability_comparison(baseline,repeat)
    cases={};local_artifacts={}
    controls=bundle['clean_controls'].get('train_smoke',{})
    variants={r['variant_id']:r for r in bundle['variants']}
    if set(variants)!={'neutral_append','pitch_plus_0.5'}:raise ValueError('Smoke must preserve exactly its two fixed raw variants')
    specifications=[('clean_text','text',controls.get('text',{})),('clean_audio','audio',controls.get('audio',{})),
                    ('neutral_append','text',variants['neutral_append']),('pitch_plus_0.5','audio',variants['pitch_plus_0.5'])]
    for name,modality,record in specifications:
        try:
            if record.get('status')!='EXTRACTED':raise ValueError(record.get('error','Missing encoder-stage feature'))
            if record['original_feature_sha256']!=array_hash(original_text[index] if modality=='text' else original_audio[index]):
                raise ValueError('Precomputed feature references another clean cache row')
            feature=record['feature'];changed_text,changed_audio=original_text.clone(),original_audio.clone()
            if modality=='text':changed_text[index]=feature
            else:changed_audio[index]=feature
            assert_target_only(original_text,original_audio,changed_text,changed_audio,index,modality)
            probabilities=checked_probabilities(downstream.infer_dialogue(model,dialogue,changed_text,changed_audio))
            if probabilities.shape!=(n,len(data['class_names'])):raise ValueError('Full Dialogue probability shape mismatch')
            assert_target_only(original_text,original_audio,changed_text,changed_audio,index,modality)
            if before_dialogue!=dialogue_fingerprint(dialogue):raise AssertionError('Dialogue context mutated')
            result=dict(full_dialogue_shape=list(probabilities.shape),finite_normalized=True,
                only_target_feature_row_changed=True,other_modality_bit_exact=True,other_utterances_bit_exact=True,
                visual_speaker_labels_ids_order_bit_exact=True,original_target_feature_sha256=record['original_feature_sha256'],
                replaced_target_feature_sha256=record['target_feature_sha256'],
                emotion_label_preservation='NOT_ASSESSED_ENGINEERING_ONLY')
            for key in ['actual_token_ids_sha256','original_input_sha256','processed_input_sha256']:
                if key in record:result[key]=record[key]
            if name.startswith('clean_'):
                result['feature_comparison']=feature_comparison(original_text[index] if modality=='text' else original_audio[index],feature,layers=modality=='text')
                result['probability_comparison']=probability_comparison(baseline,probabilities)
                result['pass_gate']=result['feature_comparison']['pass_gate'] and result['probability_comparison']['pass_gate']
            else:
                result['input_qc']=record.get('quality_control',{})
                result['pass_gate']=bool(record.get('automatic_qc_pass',False))
                result['not_required_to_flip_prediction']=True
                if 'transformation_parameters' in record:result['parameters']=record['transformation_parameters']
            result['status']='PASS' if result['pass_gate'] else 'FAIL'
            path=args.local_dir/(name+'.pt')
            torch.save(dict(target_feature=feature,full_dialogue_probabilities=torch.from_numpy(probabilities),
                dialogue_id=did,utterance_index=index,modality=modality),path)
            local_artifacts[str(path)]=sha256(path);cases[name]=result
        except Exception as error:
            cases[name]=dict(status='FAIL',pass_gate=False,error=str(error),traceback=traceback.format_exc(),emotion_label_preservation='NOT_ASSESSED_ENGINEERING_ONLY')
        report['cases']=cases;save_json(args.report,report)
    model_same=state_fingerprint(model)['sha256']==state['sha256'];bn_same=bn_fingerprint(model)==bn
    cache_same=before_cache==dict(text=array_hash(text['train'][did]),audio=array_hash(audio['train'][did]))
    context_same=before_dialogue==dialogue_fingerprint(dialogue)
    file_same={path:sha256(path)==digest for path,digest in input_hashes.items()}
    precomputed_same=report['precomputed_features_sha256']==sha256(args.precomputed_features)
    report['invariants']=dict(model_state_bit_exact={'cmerc':model_same,'encoders_verified_in_producer':bundle['metadata']['encoder_states_unchanged']},
        bn_buffers_bit_exact={'cmerc':bn_same},clean_feature_cache_bit_exact=cache_same,
        dialogue_context_bit_exact=context_same,input_files_unchanged=file_same,precomputed_file_unchanged=precomputed_same,
        models_remain_eval=all(not child.training for child in model.modules()))
    report['local_artifacts_sha256']=local_artifacts
    report['pass_gate']=bool(report['baseline_repeat']['pass_gate'] and all(r['pass_gate'] for r in cases.values())
        and model_same and bn_same and cache_same and context_same and all(file_same.values()) and precomputed_same
        and report['invariants']['models_remain_eval'] and report['encoder_stage_weights_unchanged'])
    report['status']='PASS' if report['pass_gate'] else 'FAIL'


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=['IEMOCAP','MELD'])
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--text-features',type=Path,required=True)
    parser.add_argument('--audio-features',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--text-model',type=Path,required=True)
    parser.add_argument('--precomputed-features',type=Path,required=True,
                        help='TEXT_repro --mode smoke encoder cache; no inline encoder fallback')
    parser.add_argument('--device',required=True)
    parser.add_argument('--ffmpeg',default='/home/yangbin/miniconda3/envs/msa_extract/bin/ffmpeg')
    args=parser.parse_args()
    for name in ['dataset_dir','text_features','audio_features','checkpoint','text_model','precomputed_features']:
        setattr(args,name,getattr(args,name).resolve())
    args.local_dir=ROOT/'local_only/raw_smoke'/args.dataset/'cmerc_stage'
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
