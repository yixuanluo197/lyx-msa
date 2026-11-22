#!/usr/bin/env python3
"""Frozen Train-only typo controls: TEXT_repro encode -> CMERC-only infer.

No generation, review, fitting, retry, target selection, or source mutation occurs.
The 2976 clean/semantic scoring rows are reused exactly; missing corruption stays
explicit. Real GPU execution is launched by the root, never by synthetic tests.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parent / '20261122T041838Z_meld_real_scoring'
ADAPTER = ROOT.parent / '20261120T143131Z/scripts'
PARENT_ENGINE_SHA = 'f149eb881f67310fdeb156bfb1001b5a0f5619f6d320955ca0b843a9b8c0cbf0'
VARIANTS = ('corruption_a', 'corruption_b')
FORMAT = 'MELD_FROZEN_CORRUPTION_K2_V1'
WORD = re.compile(r"(?<![\w'’\u0092\u02bc\u2018\u201b\uff07\-\u2010-\u2015\u2212])[a-z]{5,}(?![\w'’\u0092\u02bc\u2018\u201b\uff07\-\u2010-\u2015\u2212])")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): result.update(block)
    return result.hexdigest()


def text_sha(value): return hashlib.sha256(value.encode('utf-8')).hexdigest()


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream: return list(csv.DictReader(stream))


def read_jsonl(path):
    # Do not split on Unicode separators occurring in an utterance string.
    with Path(path).open(encoding='utf-8') as stream: return [json.loads(line) for line in stream if line.strip()]


def yes(value): return value is True or str(value).lower() in ('true','1')


def write_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def write_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def tensor_sha(value):
    array=value.detach().cpu().contiguous().numpy() if isinstance(value,torch.Tensor) else np.ascontiguousarray(value)
    return hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()


def bind(inputs,path,expected=None):
    path=Path(path).resolve();value=sha(path)
    if expected is not None and expected!=value:raise ValueError('Pinned input hash mismatch: '+str(path))
    if str(path) in inputs and inputs[str(path)]!=value:raise RuntimeError('Input changed: '+str(path))
    inputs[str(path)]=value;return value


def unchanged(inputs):
    for path,value in inputs.items():
        if sha(path)!=value:raise RuntimeError('Frozen input changed during stage: '+path)


def parent_engine(path):
    if sha(path)!=PARENT_ENGINE_SHA:raise ValueError('Frozen scoring engine source changed')
    spec=importlib.util.spec_from_file_location('frozen_real_score_engine',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def ranked_candidates(original,blind_id,options):
    """Independent replay of the frozen first-two ranking; never select by QC."""
    selected={}
    for match in WORD.finditer(original):
        word=match.group()
        if word not in set(options['allowlist']):continue
        for offset in range(1,len(word)-2):
            position=match.start()+offset;before=original[position:position+2];after=before[::-1]
            if before==after:continue
            changed=original[:position]+after+original[position+2:]
            selected.setdefault(changed,dict(position=position,before=before,after=after,
                original_word=word,edited_word=changed[match.start():match.end()],
                word_start=match.start(),word_end=match.end()))
    return sorted(selected.items(),key=lambda item:(text_sha(f"{options['seed']}|{blind_id}|{item[1]['position']}|{item[0]}"),item[0]))[:2]


def validate_roster(rows,index,expected=192):
    targets={};blinds=set();dialogues=set();counts={role:0 for role in ('score_fit','score_eval')}
    for row in rows:
        if not {'blind_id','sample_id','role','dialogue_id','utterance_index','utterance_id'}<=set(row):
            raise ValueError('Incomplete frozen target roster')
        sid=row['sample_id'];target=index.get(sid)
        if target is None or sid in targets or row['blind_id'] in blinds or row['dialogue_id'] in dialogues:
            raise ValueError('Duplicate/outside fixed target identity')
        for field in ('role','dialogue_id','utterance_id','utterance_index'):
            if str(row[field])!=str(target[field]):raise ValueError('Roster/source identity mismatch: '+field)
        if target['role'] not in counts:raise ValueError('Only historical Train score roles are permitted')
        targets[sid]=dict(row);blinds.add(row['blind_id']);dialogues.add(row['dialogue_id']);counts[target['role']]+=1
    if expected is not None and (len(targets)!=expected or counts!={'score_fit':96,'score_eval':96}):
        raise ValueError('Frozen 192 targets / 96-per-role coverage changed')
    return targets


def validate_reviews(rows,index,roster,options):
    reviewed={};forbidden={'label','labels','original_correct','original_prediction','original_confidence','original_probabilities','score'}
    required={'sample_id','role','dialogue_id','utterance_id','utterance_index','variant_id','processed_text',
              'original_text_sha256','processed_text_sha256','edit','generation_status','review_status','review_method','reviewer','review_notes'}
    for row in rows:
        if not required<=set(row) or forbidden&set(row):raise ValueError('Corruption review schema/blinding violation')
        sid=row['sample_id'];variant=row['variant_id']
        if sid not in roster or variant not in VARIANTS:raise ValueError('Review outside the frozen roster/K2')
        target=index[sid];official=roster[sid]
        if any(str(row[field])!=str(target[field]) for field in ('role','dialogue_id','utterance_id','utterance_index')):
            raise ValueError('Review target identity/role/order changed')
        if row.get('blind_id',official['blind_id'])!=official['blind_id']:raise ValueError('Wrong blinded target')
        if row['original_text_sha256']!=text_sha(target['original_text']) or not isinstance(row['processed_text'],str):
            raise ValueError('Original/candidate string hash mismatch')
        if row['processed_text_sha256']!=text_sha(row['processed_text']):raise ValueError('Reviewed candidate bytes changed')
        if row['review_status'] not in ('PASS','FAIL','UNCERTAIN') or not all(isinstance(row[k],str) and row[k].strip() for k in ('reviewer','review_notes')):
            raise ValueError('Review needs an attributed decided AI verdict')
        offered=ranked_candidates(target['original_text'],official['blind_id'],options);i=VARIANTS.index(variant)
        if i<len(offered):
            expected_text,edit=offered[i]
            if (row['processed_text']!=expected_text or row['edit']!=edit or row['generation_status']!='GENERATED'
                or row['review_method']!='AI_CONTEXT_ORTHOGRAPHIC_REVIEW'):
                raise ValueError('Candidate/edit differs from the frozen ranked one-swap operation')
            if any(type(row['edit'][key]) is not int for key in ('position','word_start','word_end')):
                raise ValueError('Edit indices must be exact zero-based integer codepoint positions')
        else:
            reason='NO_ELIGIBLE_EDIT' if not offered else 'NO_SECOND_DISTINCT_EDIT'
            if (row['processed_text']!=target['original_text'] or row['edit'] is not None or
                row['review_status']!='FAIL' or row['review_method']!='NO_CANDIDATE' or row['generation_status']!=reason):
                raise ValueError('Missing planned edit must remain explicit, unchanged and failed')
        versions=reviewed.setdefault(sid,{})
        if variant in versions:raise ValueError('Duplicate planned corruption version')
        versions[variant]=row
    if set(reviewed)!=set(roster) or any(set(vs)!=set(VARIANTS) for vs in reviewed.values()) or len(rows)!=2*len(roster):
        raise ValueError('Retain both planned rows for every target, including all failures')
    return reviewed


def token_difference(clean,changed):
    changes=[]
    for tag,i,j,a,b in difflib.SequenceMatcher(a=clean,b=changed,autojunk=False).get_opcodes():
        if tag!='equal':changes.append(dict(operation=tag,clean_start=i,clean_end=j,changed_start=a,changed_end=b,
                                            clean_token_ids=clean[i:j],changed_token_ids=changed[a:b]))
    return dict(clean_tokens=len(clean),changed_tokens=len(changed),actual_ids_changed=clean!=changed,changes=changes)


def feature_difference(clean,changed):
    a=np.asarray(clean.detach().cpu(),dtype=np.float64);b=np.asarray(changed.detach().cpu(),dtype=np.float64)
    if a.shape!=(4,1024) or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Feature comparison requires finite [4,1024] vectors')
    def summary(x,y):
        delta=y-x;norm=np.linalg.norm(x)*np.linalg.norm(y)
        return dict(max_abs=float(np.max(np.abs(delta))),mean_abs=float(np.mean(np.abs(delta))),
                    rmse=float(np.sqrt(np.mean(delta**2))),l2=float(np.linalg.norm(delta)),
                    cosine=float(np.dot(x.ravel(),y.ravel())/norm) if norm>0 else None)
    return dict(overall=summary(a,b),layers=[dict(layer_from_end=i+1,**summary(a[i],b[i])) for i in range(4)])


def k2_response(clean,versions,validate_probability):
    if any(not versions[v].get('eligible') or versions[v].get('probabilities') is None for v in VARIANTS):return None
    p=validate_probability(clean);label=int(p.argmax())
    changes=np.asarray([p[label]-validate_probability(versions[v]['probabilities'])[label] for v in VARIANTS])
    return dict(corruption_d_pred=float(changes.mean()),corruption_d_abs=float(np.abs(changes).mean()),
                corruption_cancellation=float(np.abs(changes).mean()-abs(changes.mean())))


def merge_score_inputs(source,responses,reasons):
    result=[]
    new_fields={'corruption_d_pred','corruption_d_abs','corruption_cancellation','corruption_k2_available','corruption_D_pred_missing_reason'}
    for row in source:
        if new_fields&set(row):raise ValueError('Source already has corruption output columns')
        sid=row['sample_id'];response=responses.get(sid)
        appended=dict(corruption_d_pred='',corruption_d_abs='',corruption_cancellation='',corruption_k2_available=0,
                      corruption_D_pred_missing_reason=reasons.get(sid,'NOT_IN_FROZEN_192_TARGET_ROSTER'))
        if response is not None:appended.update(response,corruption_k2_available=1,corruption_D_pred_missing_reason='')
        merged=dict(row,**appended)
        if {key:merged[key] for key in row}!=row:raise AssertionError('Frozen clean/semantic columns changed')
        result.append(merged)
    return result


def common(args):
    previous=args.previous.resolve();root=args.run_dir.resolve();inputs={}
    if previous!=PREVIOUS.resolve() or args.adapter_dir.resolve()!=ADAPTER.resolve():
        raise ValueError('Only the exact previous scoring instance and its frozen adapter are permitted')
    engine_path=previous/'scripts/run_score_predictions.py';bind(inputs,engine_path,PARENT_ENGINE_SHA)
    engine=parent_engine(engine_path)
    protocol=read_json(args.protocol);bind(inputs,args.protocol)
    if (protocol.get('dataset')!='MELD' or protocol.get('seed')!=67137 or protocol.get('test_access') is not False
        or protocol.get('official_validation_new_inference') is not False or protocol['text']['k']!=2
        or protocol['text']['corruption']['variants']!=list(VARIANTS)):
        raise ValueError('Wrong frozen Train-only corruption protocol')
    bind(inputs,Path(__file__))
    audit_path=root/'provenance/INPUT_AUDIT.json';audit=read_json(audit_path);bind(inputs,audit_path)
    if audit.get('status')!='INPUTS_VERIFIED' or audit.get('n_fixed_targets')!=192:
        raise ValueError('Independent source audit is incomplete')
    base=root.parent
    def audited(path):
        path=Path(path).resolve();key=str(path.relative_to(base))
        expected=audit['sha256_relative_to_pilot_parent'].get(key)
        if expected is None:raise ValueError('Required source missing from frozen input audit: '+key)
        bind(inputs,path,expected)
    audited(args.protocol);audited(args.roster)
    generation_path=root/'provenance/GENERATION_LOCK.json';generation=read_json(generation_path);bind(inputs,generation_path)
    if (generation.get('input_audit_sha256')!=sha(audit_path) or generation.get('n_targets')!=192
        or generation.get('n_planned_versions')!=384):raise ValueError('Generation lock differs from original audited cohort')
    bind(inputs,root/'scripts/orthographic_edits.py',generation['generation_sha256'])
    review_path=root/'provenance/CORRUPTION_REVIEW_LOCK.json';review_lock=read_json(review_path);bind(inputs,review_path)
    if (review_lock.get('status')!='AI_REVIEW_LOCKED' or review_lock.get('n_targets')!=192
        or review_lock.get('n_versions')!=384 or review_lock.get('human_review') is not False
        or review_lock.get('input_generation_sha256')!=sha(generation_path)):
        raise ValueError('Context-only corruption AI review is not frozen')
    bind(inputs,args.review,review_lock['reviewed_sha256'])
    for rel,value in review_lock.get('review_input_files',{}).items():bind(inputs,root/rel,value)
    old_args=SimpleNamespace(run_dir=previous,adapter_dir=args.adapter_dir,
        data=previous/'local_only/score_data.pt',audio=previous/'local_only/score_audio_features.pt',
        review=previous/'local_only/reviewed_text.jsonl',roles=previous/'local_only/SPLIT_UTTERANCES.csv')
    data,index,_,audio,parent_inputs,text_lock,checkpoint=engine.common(old_args)
    inputs.update(parent_inputs)
    verification_path=previous/'results/score_prediction_verification.json';audited(verification_path)
    old_verification=read_json(verification_path)
    if (old_verification.get('status')!='COMPLETE' or old_verification.get('no_test') is not True
        or old_verification.get('no_model_fit') is not True or old_verification.get('clean_utterances')!=2976):
        raise ValueError('Original frozen scoring inference is incomplete')
    for path,value in old_verification['inputs'].items():bind(inputs,path,value)
    for path,value in parent_inputs.items():
        if old_verification['inputs'].get(path)!=value:raise ValueError('Original exact-instance lineage changed')
    score_path=previous/'results/score_inputs.csv';semantic_path=previous/'results/score_transformations.csv'
    for path in (score_path,semantic_path):bind(inputs,path,old_verification['outputs'][path.name])
    source=read_csv(score_path)
    if len(source)!=2976 or len({r['sample_id'] for r in source})!=2976 or set(r['sample_id'] for r in source)!=set(index):
        raise ValueError('All original 2976 clean score inputs must be retained')
    clean_lookup={r['sample_id']:r for r in source}
    for sid,target in index.items():
        row=clean_lookup[sid]
        if any(str(row[field])!=str(target[field]) for field in ('sample_id','role','dialogue_id','utterance_id','utterance_index')):
            raise ValueError('Frozen clean row identity mismatch')
        p=engine.validate_probability(json.loads(row['original_probabilities']))
        if int(row['original_prediction'])!=int(p.argmax()) or abs(float(row['original_confidence'])-p.max())>1e-12:
            raise ValueError('Frozen original confidence is not max softmax')
    roster=validate_roster(read_csv(args.roster),index)
    reviews=validate_reviews(read_jsonl(args.review),index,roster,protocol['text']['corruption'])
    cache_path=previous/'local_only/score_text_features.pt';manifest_path=cache_path.with_suffix('.manifest.json')
    bind(inputs,cache_path,old_verification['encoded_features_sha256']);bind(inputs,manifest_path,old_verification['encoder_manifest_sha256'])
    manifest=read_json(manifest_path);cache=engine.load_torch(cache_path)
    if (manifest['feature_file_sha256']!=sha(cache_path) or manifest['inputs']!=parent_inputs
        or cache.get('format')!=engine.FORMAT or cache.get('inputs')!=parent_inputs
        or cache.get('text_model_sha256')!=text_lock['model_sha256'] or cache.get('class_names')!=data['class_names']
        or cache.get('fp32') is not True or cache.get('text_batch_size')!=1):
        raise ValueError('Original frozen clean text cache is not from this exact encoder instance')
    return SimpleNamespace(engine=engine,protocol=protocol,inputs=inputs,data=data,index=index,audio=audio,
        roster=roster,reviews=reviews,cache=cache,source=source,clean_lookup=clean_lookup,text_lock=text_lock,
        checkpoint=checkpoint,score_path=score_path,semantic_path=semantic_path,verification_path=verification_path,
        cache_path=cache_path)


def encode_records(state,text_module,tokenizer,encoder):
    records={}
    for sid,official in state.roster.items():
        target=state.index[sid];role,did,i=target['role'],target['dialogue_id'],target['utterance_index']
        original=target['original_text'];clean_ids,clean_stats=text_module.tokenize(tokenizer,[original],'MELD')
        if clean_ids[0]!=state.cache['clean_token_ids'][sid]:raise RuntimeError('Original tokenizer/preprocessing changed')
        clean_feature=state.cache['clean'][role][did][i];versions={}
        for variant in VARIANTS:
            review=state.reviews[sid][variant]
            candidate_ids,statistics=text_module.tokenize(tokenizer,[review['processed_text']],'MELD')
            qc=state.engine.token_qc(clean_ids[0],candidate_ids[0],statistics[0]['truncated'],statistics[0]['empty_after_processing'])
            qc['original_no_truncation']=not clean_stats[0]['truncated']
            qc['processed_ascii_content_changed']=text_module.preprocessing(original,'MELD')!=text_module.preprocessing(review['processed_text'],'MELD')
            qc['eligible']=bool(qc['eligible'] and qc['original_no_truncation'] and qc['processed_ascii_content_changed'])
            record={key:target[key] for key in ('sample_id','role','dialogue_id','utterance_id','utterance_index')}
            record.update(blind_id=official['blind_id'],variant_id=variant,
                review_status=review['review_status'],review_method=review['review_method'],reviewer=review['reviewer'],
                review_notes_sha256=text_sha(review['review_notes']),generation_status=review['generation_status'],
                edit=review['edit'],edit_validated=True,original_text_sha256=text_sha(original),
                processed_text_sha256=text_sha(review['processed_text']),actual_token_ids=candidate_ids[0],
                actual_token_ids_sha256=text_sha(json.dumps(candidate_ids[0])),
                token_qc=qc,token_statistics=statistics[0],token_diff=token_difference(clean_ids[0],candidate_ids[0]),
                original_feature_sha256=tensor_sha(clean_feature),eligible=False,encoding_status='INELIGIBLE',
                inference_status='NOT_RUN',probabilities=None,reason='SEMANTIC_REVIEW_'+review['review_status'])
            if review['review_method']=='NO_CANDIDATE':record['reason']=review['generation_status']
            elif review['review_status']=='PASS' and not qc['eligible']:record['reason']='TOKEN_QC_FAILED'
            elif review['review_status']=='PASS' and qc['eligible']:
                try:
                    with torch.inference_mode():feature=text_module.extract_texts(encoder,tokenizer,[review['processed_text']],dataset='MELD',batch_size=1)[0]
                    if feature.dtype!=torch.float32 or tuple(feature.shape)!=(4,1024) or not torch.isfinite(feature).all():
                        raise ValueError('Invalid finite CPU FP32 text feature')
                    feature=feature.detach().cpu().contiguous()
                    record.update(feature=feature,feature_sha256=tensor_sha(feature),
                        feature_diff=feature_difference(clean_feature,feature),eligible=True,encoding_status='ENCODED',reason='OK')
                except Exception as error:
                    record.update(encoding_status='FAILED',reason='FEATURE_EXTRACTION_FAILED',error=type(error).__name__+': '+str(error))
            versions[variant]=record
        distinct=versions[VARIANTS[0]]['actual_token_ids']!=versions[VARIANTS[1]]['actual_token_ids']
        for record in versions.values():
            record['token_qc']['distinct_k2_actual_token_ids']=distinct
            if not distinct and record['eligible']:record.update(eligible=False,reason='DUPLICATE_K2_ACTUAL_TOKEN_IDS')
        records[sid]=versions
    return records


def validate_encoded_records(bundle,manifest,state,protocol_sha):
    runtime=bundle.get('runtime',{})
    if (bundle.get('format')!=FORMAT or manifest.get('format')!=FORMAT
        or bundle.get('inputs')!=state.inputs or manifest.get('source_bindings')!=state.inputs
        or bundle.get('protocol_sha256')!=protocol_sha or manifest.get('protocol_sha256')!=protocol_sha
        or bundle.get('class_names')!=state.data['class_names'] or bundle.get('fp32') is not True
        or bundle.get('text_model_sha256')!=state.text_lock['model_sha256']
        or not bundle.get('encoder_state_before') or bundle.get('encoder_state_before')!=bundle.get('encoder_state_after')
        or manifest.get('encoder_state_before')!=bundle['encoder_state_before']
        or manifest.get('encoder_state_after')!=bundle['encoder_state_after']
        or (runtime.get('environment'),runtime.get('torch'),runtime.get('transformers'))!=('TEXT_repro','2.2.0+cu118','4.42.3')
        or manifest.get('runtime')!=runtime or set(bundle.get('variants',{}))!=set(state.roster)):
        raise ValueError('Encode/infer exact source/model/runtime/state/protocol binding changed')
    for sid,versions in bundle['variants'].items():
        if set(versions)!=set(VARIANTS):raise ValueError('Missing/extra encoded planned version')
        distinct=versions[VARIANTS[0]].get('actual_token_ids')!=versions[VARIANTS[1]].get('actual_token_ids')
        target=state.index[sid];clean=state.cache['clean'][target['role']][target['dialogue_id']][target['utterance_index']]
        for variant,record in versions.items():
            review=state.reviews[sid][variant];ids=record.get('actual_token_ids')
            if (not isinstance(ids,list) or not ids or ids[0]!=0 or ids[-1]!=2 or len(ids)>512
                or any(type(x) is not int or x<0 for x in ids)
                or text_sha(json.dumps(ids))!=record.get('actual_token_ids_sha256')):
                raise ValueError('Encoded actual token IDs/hash invalid')
            if (record.get('sample_id')!=sid or record.get('variant_id')!=variant
                or any(str(record.get(field))!=str(target[field]) for field in ('role','dialogue_id','utterance_id','utterance_index'))
                or record.get('encoding_status') not in ('ENCODED','INELIGIBLE','FAILED')
                or record.get('review_status')!=review['review_status'] or record.get('edit')!=review['edit']
                or record.get('processed_text_sha256')!=text_sha(review['processed_text'])
                or record.get('original_feature_sha256')!=tensor_sha(clean)
                or record['token_qc'].get('distinct_k2_actual_token_ids')!=distinct):
                raise ValueError('Encoded corruption record changed identity, review or frozen source')
            if record['eligible'] and (review['review_status']!='PASS' or not record['token_qc'].get('eligible')
                or not distinct or record['encoding_status']!='ENCODED'):
                raise ValueError('Ineligible/duplicate/not-encoded version cannot receive probabilities')
            if record['encoding_status']=='ENCODED':
                feature=record.get('feature')
                if (not isinstance(feature,torch.Tensor) or feature.device.type!='cpu' or feature.dtype!=torch.float32
                    or tuple(feature.shape)!=(4,1024) or not torch.isfinite(feature).all()
                    or record.get('feature_sha256')!=tensor_sha(feature)
                    or record.get('feature_diff')!=feature_difference(clean,feature)):
                    raise ValueError('Encoded target feature/delta evidence invalid')


def encode(args):
    state=common(args);runtime=state.engine.require_runtime('encode')
    if args.features.exists() or args.features.with_suffix('.manifest.json').exists():raise FileExistsError('Refuse existing encoded corruption')
    if torch.device(args.device).type!='cuda' or not torch.cuda.is_available():raise ValueError('Root must assign an explicit healthy CUDA device')
    torch.set_num_threads(args.threads);state.engine.seed_runtime()
    text_module=state.engine.import_parent('text_encoder',args.adapter_dir)
    encoder,tokenizer=text_module.load_encoder(args.previous/'local_only/text_model',device=args.device)
    encoder.eval().requires_grad_(False)
    if any(m.training for m in encoder.modules()) or any(p.requires_grad or p.dtype!=torch.float32 for p in encoder.parameters()):
        raise RuntimeError('Encoder must be frozen FP32 eval')
    before=state.engine.model_digest(encoder);records=encode_records(state,text_module,tokenizer,encoder)
    after=state.engine.model_digest(encoder)
    if before!=after:raise RuntimeError('Encoder parameters/buffers mutated')
    unchanged(state.inputs)
    bundle=dict(format=FORMAT,inputs=state.inputs,variants=records,dataset='MELD',seed=67137,
        class_names=state.data['class_names'],fp32=True,runtime=runtime,text_model_sha256=state.text_lock['model_sha256'],
        encoder_state_before=before,encoder_state_after=after,protocol_sha256=sha(args.protocol))
    args.features.parent.mkdir(parents=True,exist_ok=True);tmp=args.features.with_name(args.features.name+'.tmp')
    torch.save(bundle,tmp);tmp.replace(args.features)
    counts={status:sum(r['encoding_status']==status for vs in records.values() for r in vs.values()) for status in ('ENCODED','INELIGIBLE','FAILED')}
    write_json(args.features.with_suffix('.manifest.json'),dict(format=FORMAT,feature_file_sha256=sha(args.features),
        source_bindings=state.inputs,protocol_sha256=sha(args.protocol),runtime=runtime,targets=len(records),planned_versions=384,
        encoder_state_before=before,encoder_state_after=after,encoding_status=counts,no_model_fit=True,no_test=True))
    print(json.dumps(dict(stage='encode',status='COMPLETE' if counts['FAILED']==0 else 'COMPLETE_WITH_FAILURES',targets=len(records),encoding_status=counts)))


def infer(args):
    state=common(args);engine=state.engine;runtime=engine.require_runtime('infer')
    output=args.run_dir/'results/text_comparison'
    if any((output/name).exists() for name in ('score_inputs.csv','corruption_variants.csv','clean_controls.csv','inference_verification.json')):
        raise FileExistsError('Refuse existing corruption inference outputs')
    if torch.device(args.device).type!='cuda' or not torch.cuda.is_available():raise ValueError('Root must assign an explicit healthy CUDA device')
    manifest=read_json(args.features.with_suffix('.manifest.json'))
    if manifest['feature_file_sha256']!=sha(args.features):raise ValueError('Encoded corruption file hash mismatch')
    bundle=engine.load_torch(args.features)
    validate_encoded_records(bundle,manifest,state,sha(args.protocol))
    payload=engine.load_torch(state.checkpoint)
    config,extra=engine.verify_cmerc_metadata(payload,state.data,state.index,state.text_lock,args.previous,args.adapter_dir);del payload
    state.inputs.update(extra)
    torch.set_num_threads(args.threads);engine.seed_runtime()
    adapter=engine.import_parent('downstream',args.adapter_dir)
    model=adapter.load_model(args.previous/'local_only/base_data',state.checkpoint,args.device).eval().requires_grad_(False)
    if any(m.training for m in model.modules()) or any(p.requires_grad or p.dtype!=torch.float32 for p in model.parameters()):
        raise RuntimeError('CMERC must remain frozen FP32 eval')
    before=engine.model_digest(model);controls=[];variants=[];responses={};reasons={};unexpected=0
    artifact_root=args.run_dir/'local_only/corruption_inference'
    if artifact_root.exists():raise FileExistsError('Refuse existing inference artifacts')
    artifact_root.mkdir(parents=True)
    for sid in state.roster:
        target=state.index[sid];role,did,i=target['role'],target['dialogue_id'],target['utterance_index']
        dialogue=target['dialogue'];clean_text=state.cache['clean'][role][did];aud=state.audio[role][did]
        frozen=engine.validate_probability(json.loads(state.clean_lookup[sid]['original_probabilities']))
        original_context=engine.context_digest(dialogue,clean_text,aud)
        clean_prob=engine.locked_forward(adapter,model,dialogue,clean_text,aud)
        error=float(np.max(np.abs(clean_prob[i]-frozen)))
        if error>1e-6 or int(clean_prob[i].argmax())!=int(frozen.argmax()):
            raise RuntimeError('Shared-target clean parity failed; do not interpret or score corruption')
        folder=artifact_root/text_sha(sid)[:20];folder.mkdir()
        clean_artifact=folder/'clean.pt';torch.save(dict(full_dialogue_probabilities=torch.as_tensor(clean_prob),
            target_index=i,original_target_probability=torch.as_tensor(frozen)),clean_artifact)
        controls.append(dict(sample_id=sid,role=role,dialogue_id=did,utterance_index=i,
            original_probabilities=json.dumps(frozen.tolist()),recomputed_probabilities=json.dumps(clean_prob[i].tolist()),
            probability_max_abs_error=error,absolute_tolerance=1e-6,rtol=0.,argmax_identical=True,passed=True,
            source_clean_feature_sha256=tensor_sha(clean_text[i]),source_context_sha256=original_context,
            full_dialogue_probability_sha256=tensor_sha(clean_prob),artifact_sha256=sha(clean_artifact)))
        versions=bundle['variants'][sid]
        if set(versions)!=set(VARIANTS):raise ValueError('Missing/extra encoded planned version')
        for variant in VARIANTS:
            record=versions[variant]
            if (record['sample_id']!=sid or record['role']!=role or str(record['dialogue_id'])!=did
                or int(record['utterance_index'])!=i or str(record['utterance_id'])!=target['utterance_id']
                or record['original_feature_sha256']!=tensor_sha(clean_text[i])):
                raise ValueError('Encoded corruption targets another clean row')
            if record['encoding_status']=='FAILED':unexpected+=1
            if record['eligible']:
                feature=record['feature']
                if tensor_sha(feature)!=record['feature_sha256'] or feature.device.type!='cpu':raise ValueError('Encoded target feature corrupted')
                changed=engine.replace_target(clean_text,i,feature)
                try:
                    full=engine.locked_forward(adapter,model,dialogue,changed,aud)
                    record.update(probabilities=full[i].tolist(),inference_status='PREDICTED')
                    artifact=folder/(variant+'.pt')
                    torch.save(dict(target_feature=feature,full_dialogue_probabilities=torch.as_tensor(full),
                        sample_id=sid,dialogue_id=did,utterance_index=i),artifact)
                    record.update(full_dialogue_probability_sha256=tensor_sha(full),artifact_sha256=sha(artifact))
                except engine.FrozenInputMutation:raise
                except Exception as error:
                    unexpected+=1;record.update(eligible=False,inference_status='FAILED',reason='PERTURBED_FORWARD_FAILED',error=type(error).__name__+': '+str(error))
                if engine.context_digest(dialogue,clean_text,aud)!=original_context:raise RuntimeError('Target-only forward changed frozen context')
            public={key:value for key,value in record.items() if key not in ('feature','actual_token_ids','probabilities')}
            for key in ('edit','token_qc','token_statistics','token_diff','feature_diff'):
                public[key]=json.dumps(public.get(key),sort_keys=True,allow_nan=False)
            probability=record.get('probabilities')
            public.update(original_prediction=int(frozen.argmax()),original_confidence=float(frozen.max()),
                original_probabilities=json.dumps(frozen.tolist()),
                perturbed_prediction=int(np.argmax(probability)) if probability is not None else '',
                perturbed_original_class_probability=probability[int(frozen.argmax())] if probability is not None else '',
                perturbed_max_confidence=float(max(probability)) if probability is not None else '',
                perturbed_probabilities=json.dumps(probability) if probability is not None else '',
                signed_response=float(frozen.max()-probability[int(frozen.argmax())]) if probability is not None else '',
                non_target_text_bit_exact=True,other_modalities_context_unchanged=True)
            variants.append(public)
        responses[sid]=k2_response(frozen,versions,engine.validate_probability)
        reasons[sid]='|'.join(v+':'+versions[v]['reason'] for v in VARIANTS if not versions[v]['eligible'])
    after=engine.model_digest(model)
    if before!=after:raise RuntimeError('CMERC state/BN buffers changed')
    unchanged(state.inputs)
    merged=merge_score_inputs(state.source,responses,reasons)
    write_csv(output/'score_inputs.csv',merged);write_csv(output/'corruption_variants.csv',variants);write_csv(output/'clean_controls.csv',controls)
    verification=dict(status='COMPLETE' if unexpected==0 else 'COMPLETE_WITH_FAILURES',unexpected_failures=unexpected,
        clean_utterances=len(merged),corruption_targets=len(state.roster),planned_corruption_versions=len(variants),
        clean_controls_passed=len(controls),available_corruption_d_pred=sum(r['corruption_k2_available'] for r in merged),
        no_model_fit=True,no_test=True,no_new_validation=True,semantic_probabilities_reused=True,clean_semantic_rows_bit_exact=True,
        protocol_sha256=sha(args.protocol),source_bindings=state.inputs,
        source_score_inputs_path=str(state.score_path.resolve()),source_score_inputs_sha256=sha(state.score_path),
        source_semantic_variants_path=str(state.semantic_path.resolve()),source_semantic_variants_sha256=sha(state.semantic_path),
        source_prediction_verification_path=str(state.verification_path.resolve()),source_prediction_verification_sha256=sha(state.verification_path),
        source_clean_text_cache_sha256=sha(state.cache_path),encoded_features_sha256=sha(args.features),
        encoded_manifest_sha256=sha(args.features.with_suffix('.manifest.json')),runtime=runtime,encoder_runtime=bundle['runtime'],
        model_state_before=before,model_state_after=after,text_model_sha256=state.text_lock['model_sha256'],
        clean_parity=dict(scope='all 192 original target rows, full original Dialogue forward',atol=1e-6,rtol=0.,argmax_required=True),
        target_only=True,fp32=True,TF32=False,roles=['score_fit','score_eval'],original_source_split='Train',
        original_score_eval_previously_viewed=True,new_independent_evaluation=False,effective_cmerc_config=config,
        outputs={name:sha(output/name) for name in ('score_inputs.csv','corruption_variants.csv','clean_controls.csv')},
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/'inference_verification.json',verification)
    print(json.dumps({key:verification[key] for key in ('status','clean_utterances','corruption_targets','planned_corruption_versions','available_corruption_d_pred','unexpected_failures')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('encode','infer'))
    parser.add_argument('--run-dir',type=Path,default=ROOT)
    parser.add_argument('--previous',type=Path,default=PREVIOUS)
    parser.add_argument('--adapter-dir',type=Path,default=ADAPTER)
    parser.add_argument('--protocol',type=Path);parser.add_argument('--roster',type=Path)
    parser.add_argument('--review',type=Path);parser.add_argument('--features',type=Path)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    for name in ('run_dir','previous','adapter_dir'):setattr(args,name,getattr(args,name).resolve())
    args.protocol=(args.protocol or args.run_dir/'EXPERIMENT_PROTOCOL.json').resolve()
    args.roster=(args.roster or args.previous/'local_only/perturbation_attempt_manifest.csv').resolve()
    args.review=(args.review or args.run_dir/'local_only/corruption_reviewed.jsonl').resolve()
    args.features=(args.features or args.run_dir/'local_only/corruption_features.pt').resolve()
    if args.threads<1:parser.error('--threads must be positive')
    if not args.features.is_relative_to(args.run_dir) or not args.review.is_relative_to(args.run_dir):
        parser.error('New review/features must remain inside this new experiment')
    (encode if args.stage=='encode' else infer)(args)


if __name__=='__main__':main()
