#!/usr/bin/env python3
"""Frozen MELD pilot inference from physically score_fit-only exports.

TEXT_repro --stage encode; multimodalerc310 --stage infer. No old score_eval
table/text/cache is loaded. Exactly the same eligible K2 forwards generate the
signed D_pred and mean-absolute D_abs; all other clean rows remain with NA.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'20261122T151649Z_meld_dpred_corruption/scripts/infer_corruption.py'
PARENT_SHA='82559db547666ace42319243cee5df3ea73a6daf62e07a662a91c94379a602df'
PREVIOUS=ROOT.parent/'20261122T041838Z_meld_real_scoring'
ADAPTER=ROOT.parent/'20261120T143131Z/scripts'
ROLES=('pilot_fit','pilot_eval');VARIANTS=('corruption_a','corruption_b')
PILOT_FORMAT='MELD_SCORE_FIT_ONLY_D_PRED_D_ABS_PILOT_V1'


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:return [json.loads(line) for line in stream if line.strip()]


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def parent_adapter():
    if sha(PARENT)!=PARENT_SHA:raise ValueError('Frozen corruption adapter source changed')
    return load_module('frozen_corruption_parent',PARENT)


def validate_exports(data,audio,cache,cohort,clean_rows,targets,expected_roles=None):
    expected_roles=expected_roles or {'pilot_fit':75,'pilot_eval':60}
    if (data.get('dataset')!='MELD' or data.get('seed')!=67137 or len(data.get('class_names',[]))!=7
        or set(data.get('splits',{}))!=set(ROLES) or set(audio)!=set(ROLES)
        or set(cache.get('clean',{}))!=set(ROLES)):
        raise ValueError('Exports must contain only MELD pilot_fit/pilot_eval; no old score_eval/Val/Test')
    columns=('sample_id','role','dialogue_id','utterance_id','utterance_index')
    def key(row):return tuple(str(row[x]) for x in columns)
    if len({key(r) for r in cohort})!=len(cohort) or len({r['sample_id'] for r in cohort})!=len(cohort):
        raise ValueError('Duplicate frozen pilot cohort identity')
    if any(r['role'] not in ROLES for r in cohort):raise ValueError('Old/engineering role in pilot cohort')
    index={};dialogues=set();counts={}
    for role in ROLES:
        ds=data['splits'][role];ids={str(d['dialogue_id']) for d in ds}
        if len(ids)!=len(ds) or dialogues&ids:raise ValueError('Pilot Dialogue overlap/duplicates')
        dialogues.update(ids);counts[role]=len(ids)
        if set(audio[role])!=ids or set(cache['clean'][role])!=ids:raise ValueError('Exported modality Dialogue coverage mismatch')
        for d in ds:
            did=str(d['dialogue_id']);n=len(d['utterance_ids'])
            if n<1 or len(d['texts'])!=n or len(set(map(str,d['utterance_ids'])))!=n:raise ValueError('Bad original Dialogue length/order')
            for tensor,shape in ((audio[role][did],(n,768)),(cache['clean'][role][did],(n,4,1024))):
                if tensor.dtype!=torch.float32 or tuple(tensor.shape)!=shape or not torch.isfinite(tensor).all():
                    raise ValueError('Exported feature shape/precision/finiteness changed')
            if len(d['speaker'])!=n or len(d['visual'])!=n or len(d['labels'])!=n:raise ValueError('Context lengths changed')
            for i,(uid,text) in enumerate(zip(d['utterance_ids'],d['texts'])):
                sid=f'MELD:train:{did}:{uid}'
                index[sid]=dict(sample_id=sid,role=role,dialogue_id=did,utterance_id=str(uid),
                    utterance_index=i,original_text=text,dialogue=d)
    if counts!=expected_roles or {key(r) for r in cohort}!={key(r) for r in index.values()}:
        raise ValueError('Pilot exact clean cohort or 75/60 Dialogue split changed')
    if len(clean_rows)!=len(index) or len({r['sample_id'] for r in clean_rows})!=len(clean_rows):
        raise ValueError('Missing/duplicate exported clean numeric rows')
    forbidden={'D_pred','D_abs','corruption_d_pred','R','k2_available','D_pred_missing_reason'}
    for row in clean_rows:
        if row.get('original_role')!='score_fit' or row['sample_id'] not in index or key(row)!=key(index[row['sample_id']]):
            raise ValueError('Clean numeric source includes wrong original role/identity')
        if forbidden&set(row):raise ValueError('Previous behavioral/score outputs must not enter the new clean export')
        p=np.asarray(json.loads(row['original_probabilities']),float)
        if p.shape!=(7,) or not np.isfinite(p).all() or (p<0).any() or abs(p.sum()-1)>1e-4:
            raise ValueError('Bad exported seven-class clean probability')
        ordered=np.sort(p);nz=p>0
        scalars=dict(original_prediction=int(p.argmax()),original_confidence=float(p.max()),
            clean_margin=float(ordered[-1]-ordered[-2]),clean_entropy=float(-np.sum(p[nz]*np.log(p[nz]))))
        if str(row['original_correct']) not in ('0','1') or any(not np.isclose(float(row[k]),v,atol=1e-12,rtol=1e-10) for k,v in scalars.items()):
            raise ValueError('Exported raw confidence/margin/entropy scalar mismatch')
    roster={};blinds=set();target_ds=set()
    for row in targets:
        sid=row['sample_id'];target=index.get(sid)
        if (target is None or sid in roster or row['blind_id'] in blinds or row['dialogue_id'] in target_ds
            or key(row)!=key(target)):
            raise ValueError('Wrong/duplicate fixed Pilot target identity')
        roster[sid]=row;blinds.add(row['blind_id']);target_ds.add(row['dialogue_id'])
    if len(roster)!=sum(expected_roles.values()) or target_ds!=dialogues or set(cache.get('clean_token_ids',{}))!=set(roster):
        raise ValueError('Need one unchanged target/token record per Pilot Dialogue')
    return index,roster,{r['sample_id']:r for r in clean_rows}


def validate_reviews(rows,index,roster,ranked_candidates,options,text_sha):
    reviewed={};required={'sample_id','role','dialogue_id','utterance_index','utterance_id','variant_id','processed_text',
        'processed_text_sha256','original_text_sha256','edit','generation_status','review_status','reviewer','review_method','review_notes'}
    forbidden={'label','labels','original_correct','original_prediction','original_confidence','original_probabilities','score'}
    for row in rows:
        if not required<=set(row) or forbidden&set(row):raise ValueError('Blind review schema/outcome leakage')
        sid=row['sample_id'];v=row['variant_id']
        if sid not in roster or v not in VARIANTS:raise ValueError('Review outside fixed pilot targets')
        t=index[sid]
        if any(str(row[k])!=str(t[k]) for k in ('role','dialogue_id','utterance_index','utterance_id')):
            raise ValueError('Reviewed target/order/role differs')
        if row['original_text_sha256']!=text_sha(t['original_text']) or row['processed_text_sha256']!=text_sha(row['processed_text']):
            raise ValueError('Original/reviewed candidate exact bytes differ')
        if row['review_status'] not in ('PASS','FAIL','UNCERTAIN') or any(not isinstance(row[k],str) or not row[k].strip() for k in ('reviewer','review_notes')):
            raise ValueError('Need attributed decided AI review')
        offered=ranked_candidates(t['original_text'],roster[sid]['blind_id'],options);i=VARIANTS.index(v)
        if i<len(offered):
            text,edit=offered[i]
            if (row['processed_text']!=text or row['edit']!=edit or row['generation_status']!='GENERATED'
                or row['review_method']!='AI_CONTEXT_ORTHOGRAPHIC_REVIEW'):
                raise ValueError('Candidate is not the fixed ranked one-swap text/edit')
            if any(type(row['edit'][k]) is not int for k in ('position','word_start','word_end')):
                raise ValueError('Edit index must be an integer codepoint position')
        else:
            reason='NO_ELIGIBLE_EDIT' if not offered else 'NO_SECOND_DISTINCT_EDIT'
            if (row['edit'] is not None or row['processed_text']!=t['original_text'] or row['review_status']!='FAIL'
                or row['review_method']!='NO_CANDIDATE' or row['generation_status']!=reason):
                raise ValueError('Missing version must remain unchanged procedural FAIL')
        vs=reviewed.setdefault(sid,{})
        if v in vs:raise ValueError('Duplicate reviewed K2 version')
        vs[v]=row
    if set(reviewed)!=set(roster) or any(set(v)!=set(VARIANTS) for v in reviewed.values()) or len(rows)!=2*len(roster):
        raise ValueError('Retain all 270 planned versions, including missing and AI failures')
    return reviewed


def common(args,base):
    root=args.run_dir;inputs={};bind=lambda p,expected=None:base.bind(inputs,p,expected)
    bind(PARENT,PARENT_SHA);bind(Path(__file__));protocol=read_json(args.protocol);bind(args.protocol)
    if (protocol.get('dataset')!='MELD' or protocol.get('model_seed')!=67137 or protocol.get('source_role')!='score_fit'
        or protocol.get('historical_score_eval_used') is not False or protocol.get('test_access') is not False
        or protocol.get('official_validation_access') is not False or protocol.get('new_independent_evaluation') is not False
        or protocol['text']['k']!=2 or protocol['text']['corruption']['variants']!=list(VARIANTS)):
        raise ValueError('Wrong frozen score_fit-only exploratory protocol')
    source_lock_path=root/'provenance/SOURCE_LOCK.json';source_lock=read_json(source_lock_path);bind(source_lock_path)
    if (source_lock.get('status')!='SOURCE_EXPORT_VERIFIED' or source_lock.get('original_role')!='score_fit'
        or source_lock.get('old_score_eval_exported') is not False or source_lock.get('historical_score_eval_used') is not False
        or source_lock.get('protocol_sha256')!=sha(args.protocol) or source_lock.get('n_targets')!=135
        or source_lock.get('n_dialogues')!={'pilot_fit':75,'pilot_eval':60}):
        raise ValueError('Physically score_fit-only source export is not locked')
    required=('source_data.pt','source_audio.pt','source_cache.pt','source_clean.csv','pilot_cohort.csv','target_manifest.csv')
    for name in required:
        rel='local_only/'+name
        if rel not in source_lock['exports']:raise ValueError('Missing required frozen export')
        bind(root/rel,source_lock['exports'][rel])
    helper_path=root/'scripts/freeze_pilot.py';options=dict(protocol['text']['corruption'])
    bind(helper_path,options['generation_source_sha256'])
    collision_path=root/'local_only/collision_words.json';bind(collision_path,options['collision_words_sha256'])
    options['collision_words']=read_json(collision_path)
    helper=load_module('frozen_pilot_candidate_rules',helper_path)
    review_lock_path=root/'provenance/REVIEW_LOCK.json';review_lock=read_json(review_lock_path);bind(review_lock_path)
    if (review_lock.get('status')!='AI_REVIEW_LOCKED' or review_lock.get('human_review') is not False
        or review_lock.get('n_targets')!=135 or review_lock.get('n_versions')!=270
        or review_lock.get('protocol_sha256')!=sha(args.protocol)):
        raise ValueError('New Pilot AI review is not frozen')
    bind(args.review,review_lock['reviewed_sha256'])
    engine=base.parent_engine(PREVIOUS/'scripts/run_score_predictions.py')
    # All tensor deserializations here are NEW physical exports, never the old
    # all-role score_data/audio/cache. Old role/source tables are not parsed.
    data=engine.load_torch(root/'local_only/source_data.pt')
    audio=engine.load_torch(root/'local_only/source_audio.pt')
    cache=engine.load_torch(root/'local_only/source_cache.pt')
    source_clean_path=root/'local_only/source_clean.csv';source=read_csv(source_clean_path)
    index,roster,lookup=validate_exports(data,audio,cache,read_csv(root/'local_only/pilot_cohort.csv'),source,
        read_csv(root/'local_only/target_manifest.csv'))
    if source_lock.get('counts')!={role:sum(r['role']==role for r in source) for role in ROLES}:
        raise ValueError('Clean export fixed row counts differ')
    reviews=validate_reviews(read_jsonl(args.review),index,roster,helper.ranked_candidates,options,base.text_sha)
    def historic(path):
        path=Path(path).resolve();expected=source_lock['sources'].get(str(path))
        if expected is None:raise ValueError('Required frozen model/source metadata missing from SOURCE_LOCK: '+str(path))
        bind(path,expected)
    for name,value in engine.PINNED.items():
        historic(ADAPTER/name)
        if sha(ADAPTER/name)!=value:raise ValueError('Frozen original encoder/CMERC math changed')
    for path in sorted((ADAPTER/'core_snapshot').glob('*.py')):historic(path)
    historic(PREVIOUS/'scripts/run_score_predictions.py')
    model_dir=PREVIOUS/'local_only/text_model/best_classifier';text_lock_path=model_dir.parent/'best_model_lock.json'
    historic(text_lock_path);text_lock=read_json(text_lock_path)
    if (text_lock.get('status')!='COMPLETE' or text_lock.get('dataset')!='MELD'
        or text_lock.get('test_accessed') is not False or text_lock.get('official_validation_used_for_selection') is not False
        or cache.get('text_model_sha256')!=text_lock['model_sha256']):
        raise ValueError('Frozen text model lineage/cache model identity differs')
    historic(model_dir/'model.safetensors')
    if sha(model_dir/'model.safetensors')!=text_lock['model_sha256']:raise ValueError('Text weight hash mismatch')
    for name in ('config.json','vocab.json','merges.txt','tokenizer_config.json','special_tokens_map.json'):historic(model_dir/name)
    config=read_json(model_dir/'config.json')
    if config.get('planb_dataset')!='MELD' or [config['id2label'][str(i)] for i in range(7)]!=data['class_names']:
        raise ValueError('Frozen encoder class order changed')
    training_path=model_dir.parent/'training_config.json';historic(training_path);training=read_json(training_path)
    if base.text_sha(json.dumps(training['protocol'],sort_keys=True))!=text_lock['protocol_sha256']:
        raise ValueError('Original text training completion/config lock mismatch')
    checkpoint=PREVIOUS/'local_only/base_data/runs/cmerc/best.pt';completion=checkpoint.parent/'completion.json'
    historic(completion);historic(checkpoint);done=read_json(completion)
    if done.get('status')!='COMPLETE' or done.get('stage')!='cmerc' or sha(checkpoint)!=done['best_checkpoint_sha256']:
        raise ValueError('Original CMERC selected checkpoint/completion mismatch')
    return SimpleNamespace(base=base,engine=engine,inputs=inputs,protocol=protocol,data=data,index=index,
        roster=roster,reviews=reviews,audio=audio,cache=cache,source=source,clean_lookup=lookup,
        source_clean_path=source_clean_path,source_lock=source_lock,text_lock=text_lock,checkpoint=checkpoint)


def make_inputs(source,responses,reasons):
    result=[]
    for old in source:
        if {'D_pred','D_abs','k2_available','missing_reason'}&set(old):raise ValueError('Clean source contains prior behavior')
        response=responses.get(old['sample_id']);new=dict(old)
        new.update(D_pred='',D_abs='',k2_available=0,missing_reason=reasons.get(old['sample_id'],'NOT_SELECTED_FIXED_TARGET'))
        if response is not None:
            new.update(D_pred=response['corruption_d_pred'],D_abs=response['corruption_d_abs'],k2_available=1,missing_reason='')
            if new['D_abs']+1e-12<abs(new['D_pred']):raise RuntimeError('D_abs < abs(D_pred)')
        if {k:new[k] for k in old}!=old:raise RuntimeError('Frozen source clean columns changed')
        result.append(new)
    return result


def encode(args):
    base=parent_adapter();engine=base.parent_engine(PREVIOUS/'scripts/run_score_predictions.py')
    runtime=engine.require_runtime('encode');state=common(args,base)
    if args.features.exists() or args.features.with_suffix('.manifest.json').exists():raise FileExistsError('Refuse existing Pilot feature stage')
    if torch.device(args.device).type!='cuda' or not torch.cuda.is_available():raise ValueError('Root must assign a healthy CUDA device')
    torch.set_num_threads(args.threads);engine.seed_runtime()
    module=engine.import_parent('text_encoder',ADAPTER)
    encoder,tokenizer=module.load_encoder(PREVIOUS/'local_only/text_model',device=args.device)
    encoder.eval().requires_grad_(False)
    if any(m.training for m in encoder.modules()) or any(p.requires_grad or p.dtype!=torch.float32 for p in encoder.parameters()):raise RuntimeError('Encoder not frozen FP32')
    before=engine.model_digest(encoder);records=base.encode_records(state,module,tokenizer,encoder);after=engine.model_digest(encoder)
    if before!=after:raise RuntimeError('Encoder weights/BN buffers changed')
    base.unchanged(state.inputs)
    bundle=dict(format=base.FORMAT,pilot_format=PILOT_FORMAT,inputs=state.inputs,variants=records,dataset='MELD',seed=67137,
        class_names=state.data['class_names'],fp32=True,runtime=runtime,text_model_sha256=state.text_lock['model_sha256'],
        encoder_state_before=before,encoder_state_after=after,protocol_sha256=sha(args.protocol))
    args.features.parent.mkdir(parents=True,exist_ok=True);tmp=args.features.with_name(args.features.name+'.tmp')
    torch.save(bundle,tmp);tmp.replace(args.features)
    counts={k:sum(r['encoding_status']==k for vs in records.values() for r in vs.values()) for k in ('ENCODED','INELIGIBLE','FAILED')}
    base.write_json(args.features.with_suffix('.manifest.json'),dict(format=base.FORMAT,pilot_format=PILOT_FORMAT,
        feature_file_sha256=sha(args.features),source_bindings=state.inputs,protocol_sha256=sha(args.protocol),runtime=runtime,
        encoder_state_before=before,encoder_state_after=after,targets=len(records),planned_versions=270,encoding_status=counts,
        source_original_role='score_fit',old_score_eval_used=False,no_model_fit=True))
    print(json.dumps(dict(stage='encode',status='COMPLETE' if counts['FAILED']==0 else 'COMPLETE_WITH_FAILURES',targets=len(records),encoding_status=counts)))


def infer(args):
    base=parent_adapter();engine=base.parent_engine(PREVIOUS/'scripts/run_score_predictions.py')
    runtime=engine.require_runtime('infer');state=common(args,base);output=args.run_dir/'results/inference'
    if any((output/n).exists() for n in ('score_inputs.csv','corruption_variants.csv','clean_controls.csv','inference_verification.json')):
        raise FileExistsError('Refuse existing Pilot inference outputs')
    if torch.device(args.device).type!='cuda' or not torch.cuda.is_available():raise ValueError('Root must assign a healthy CUDA device')
    manifest=read_json(args.features.with_suffix('.manifest.json'))
    if manifest['feature_file_sha256']!=sha(args.features):raise ValueError('Encoded features changed')
    bundle=engine.load_torch(args.features)
    if bundle.get('pilot_format')!=PILOT_FORMAT or manifest.get('pilot_format')!=PILOT_FORMAT:raise ValueError('Wrong Pilot feature bundle')
    base.validate_encoded_records(bundle,manifest,state,sha(args.protocol))
    payload=engine.load_torch(state.checkpoint)
    config,extra=engine.verify_cmerc_metadata(payload,state.data,state.index,state.text_lock,PREVIOUS,ADAPTER);del payload
    # This legacy helper hashes original base-training files; it does not load
    # old scoring tensors, score_eval predictions, generated text, or role tables.
    for path,value in extra.items():base.bind(state.inputs,path,value)
    torch.set_num_threads(args.threads);engine.seed_runtime()
    adapter=engine.import_parent('downstream',ADAPTER)
    model=adapter.load_model(PREVIOUS/'local_only/base_data',state.checkpoint,args.device).eval().requires_grad_(False)
    if any(m.training for m in model.modules()) or any(p.requires_grad or p.dtype!=torch.float32 for p in model.parameters()):raise RuntimeError('CMERC not frozen FP32')
    before=engine.model_digest(model);controls=[];rows=[];responses={};reasons={};unexpected=0
    artifact_root=args.run_dir/'local_only/inference_artifacts'
    if artifact_root.exists():raise FileExistsError('Refuse existing Pilot inference artifacts')
    artifact_root.mkdir(parents=True)
    for sid in state.roster:
        t=state.index[sid];role,did,i=t['role'],t['dialogue_id'],t['utterance_index'];d=t['dialogue']
        clean=state.cache['clean'][role][did];aud=state.audio[role][did]
        p=engine.validate_probability(json.loads(state.clean_lookup[sid]['original_probabilities']))
        context=engine.context_digest(d,clean,aud);replay=engine.locked_forward(adapter,model,d,clean,aud)
        error=float(np.max(np.abs(replay[i]-p)))
        if error>1e-6 or int(replay[i].argmax())!=int(p.argmax()):raise RuntimeError('Pilot clean parity failed; abort before scoring')
        folder=artifact_root/base.text_sha(sid)[:20];folder.mkdir()
        clean_path=folder/'clean.pt';torch.save(dict(full_dialogue_probabilities=torch.as_tensor(replay),target_index=i),clean_path)
        controls.append(dict(sample_id=sid,role=role,dialogue_id=did,utterance_index=i,
            original_probabilities=json.dumps(p.tolist()),recomputed_probabilities=json.dumps(replay[i].tolist()),
            probability_max_abs_error=error,atol=1e-6,rtol=0.,argmax_identical=True,passed=True,
            original_feature_sha256=base.tensor_sha(clean[i]),original_context_sha256=context,
            full_dialogue_probability_sha256=base.tensor_sha(replay),artifact_sha256=sha(clean_path)))
        versions=bundle['variants'][sid]
        for version in VARIANTS:
            r=versions[version]
            if r['encoding_status']=='FAILED':unexpected+=1
            if r['eligible']:
                changed=engine.replace_target(clean,i,r['feature'])
                try:
                    full=engine.locked_forward(adapter,model,d,changed,aud)
                    r.update(probabilities=full[i].tolist(),inference_status='PREDICTED')
                    path=folder/(version+'.pt');torch.save(dict(target_feature=r['feature'],full_dialogue_probabilities=torch.as_tensor(full),
                        sample_id=sid,dialogue_id=did,utterance_index=i),path)
                    r.update(full_dialogue_probability_sha256=base.tensor_sha(full),artifact_sha256=sha(path))
                except engine.FrozenInputMutation:raise
                except Exception as err:
                    unexpected+=1;r.update(eligible=False,inference_status='FAILED',reason='PERTURBED_FORWARD_FAILED',error=type(err).__name__+': '+str(err))
                if engine.context_digest(d,clean,aud)!=context:raise RuntimeError('Original text/audio/vision/context changed')
            public={k:v for k,v in r.items() if k not in ('feature','actual_token_ids','probabilities')}
            for field in ('edit','token_qc','token_statistics','token_diff','feature_diff'):public[field]=json.dumps(public.get(field),sort_keys=True,allow_nan=False)
            q=r.get('probabilities')
            public.update(original_prediction=int(p.argmax()),original_confidence=float(p.max()),original_probabilities=json.dumps(p.tolist()),
                perturbed_prediction=int(np.argmax(q)) if q is not None else '',perturbed_probabilities=json.dumps(q) if q is not None else '',
                signed_response=float(p.max()-q[int(p.argmax())]) if q is not None else '',
                non_target_text_bit_exact=True,other_modalities_context_unchanged=True)
            rows.append(public)
        responses[sid]=base.k2_response(p,versions,engine.validate_probability)
        reasons[sid]='|'.join(v+':'+versions[v]['reason'] for v in VARIANTS if not versions[v]['eligible'])
    after=engine.model_digest(model)
    if before!=after:raise RuntimeError('CMERC weights/BN buffers changed')
    base.unchanged(state.inputs);merged=make_inputs(state.source,responses,reasons)
    base.write_csv(output/'score_inputs.csv',merged);base.write_csv(output/'corruption_variants.csv',rows);base.write_csv(output/'clean_controls.csv',controls)
    verification=dict(status='COMPLETE' if unexpected==0 else 'COMPLETE_WITH_FAILURES',unexpected_failures=unexpected,
        clean_utterances=len(merged),targets=len(state.roster),planned_versions=len(rows),clean_controls_passed=len(controls),
        pilot_fit_dialogues=75,pilot_eval_dialogues=60,available_k2=sum(r['k2_available'] for r in merged),
        no_model_fit=True,no_test=True,no_new_validation=True,old_score_eval_used=False,source_original_role='score_fit',
        source_clean_rows_unchanged=True,new_independent_evaluation=False,confirmatory_scoring='PAUSED',
        source_clean_path=str(state.source_clean_path.resolve()),source_clean_sha256=sha(state.source_clean_path),
        protocol_sha256=sha(args.protocol),source_bindings=state.inputs,
        historical_source_declarations=state.source_lock['sources'],historical_declarations_reparsed=False,
        encoded_features_sha256=sha(args.features),encoded_manifest_sha256=sha(args.features.with_suffix('.manifest.json')),
        runtime=runtime,encoder_runtime=bundle['runtime'],text_model_sha256=state.text_lock['model_sha256'],
        model_state_before=before,model_state_after=after,fp32=True,TF32=False,target_only=True,full_dialogue=True,
        clean_parity=dict(atol=1e-6,rtol=0.,argmax_required=True,targets=len(controls)),
        behavior='Same double-PASS token-distinct untruncated successful K2: D_pred=mean(p_clean(c)-p_k(c)); D_abs=mean(abs(p_clean(c)-p_k(c))). Both NA otherwise.',
        effective_cmerc_config=config,outputs={n:sha(output/n) for n in ('score_inputs.csv','corruption_variants.csv','clean_controls.csv')},
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    base.write_json(output/'inference_verification.json',verification)
    print(json.dumps({k:verification[k] for k in ('status','clean_utterances','targets','planned_versions','available_k2','unexpected_failures')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('encode','infer'))
    parser.add_argument('--run-dir',type=Path,default=ROOT);parser.add_argument('--protocol',type=Path)
    parser.add_argument('--review',type=Path);parser.add_argument('--features',type=Path)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args();args.run_dir=args.run_dir.resolve()
    args.protocol=(args.protocol or args.run_dir/'EXPERIMENT_PROTOCOL.json').resolve()
    args.review=(args.review or args.run_dir/'local_only/reviewed.jsonl').resolve()
    args.features=(args.features or args.run_dir/'local_only/pilot_features.pt').resolve()
    if args.threads<1:parser.error('--threads must be positive')
    if any(not p.is_relative_to(args.run_dir) for p in (args.protocol,args.review,args.features)):parser.error('All new artifacts must remain inside this Pilot run')
    (encode if args.stage=='encode' else infer)(args)


if __name__=='__main__':main()
