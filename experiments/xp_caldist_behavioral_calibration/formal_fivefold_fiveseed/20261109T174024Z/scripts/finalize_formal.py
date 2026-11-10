#!/usr/bin/env python3
"""Read frozen training/calibration artifacts; write a sealed Evaluation readiness package.

No raw dataset, role feature artifact, Evaluation labels, or forward pass is opened.
Strict checkpoint reload constructs each recorded architecture on CPU only. Historical
Pilot checkpoints are checked against their original runner/helper/source lineage.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

MANIFEST_SHA='6d43928865915fe22a913c98b1165a05ba6e3e8f21d4d4c896cf51eb9c9b62be'
METHODS=('E0','E1','E2','E3')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def jsonout(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def csvout(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def resolve_output(row):
    if row['status']=='reused':return Path(row['reused_output'])
    if row['status']!='completed':raise ValueError('incomplete task cannot be finalized')
    valid=[a for a in row['attempts'] if a.get('returncode')==0]
    if not valid:raise ValueError('completed task has no verified successful attempt')
    output=Path(valid[-1]['directory'])/'output'
    return output/'difficulty.json' if row['kind']=='difficulty' else output

def audit_manifest_metadata(manifest,expected_folds=5):
    """Only role IDs/group IDs/count metadata; never inspect Evaluation labels."""
    out={}
    for dataset,protocol in manifest['protocols'].items():
        folds=protocol['folds']
        if len(folds)!=expected_folds:raise ValueError('unexpected fold count')
        counts=Counter();pool=set();evaluation=[]
        for fold,entry in folds.items():
            roles=entry['roles'];sets={r:set(v['dialogue_ids']) for r,v in roles.items()}
            for a in sets:
                if len(sets[a])!=len(roles[a]['dialogue_ids']):raise ValueError('duplicate dialogue ID')
                for b in sets:
                    if a<b and sets[a]&sets[b]:raise ValueError('illegal role overlap')
            groups={r:set(v['group_ids']) for r,v in roles.items()}
            for a in ('calibration','evaluation'):
                for b in groups:
                    if a!=b and groups[a]&groups[b]:raise ValueError('held-out group overlap')
            for v in sets.values():pool.update(v)
            counts.update(sets['evaluation'])
            evaluation.append({'fold':fold,'dialogue_ids':roles['evaluation']['dialogue_ids'],
                               'group_ids':roles['evaluation']['group_ids'],
                               'role_sha256':roles['evaluation'].get('sha256'),
                               'utterance_count':roles['evaluation'].get('utterance_count'),
                               'supervision_eligible_count':roles['evaluation'].get('supervision_eligible_count')})
        if set(counts)!=pool or any(n!=1 for n in counts.values()):raise ValueError('Evaluation coverage is not exactly once')
        out[dataset]={'evaluation_dialogues':len(counts),'exact_once':True,'folds':evaluation}
    return out

def audit_difficulty(artifact,expected_ids):
    rows=artifact['dialogues'];ids=[r['dialogue_id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected_ids):raise ValueError('Difficulty exact ID coverage failed')
    for row in rows:
        if row['valid_label_count']<=0:raise ValueError('no valid supervision')
        gaps=[row[k+'_gap'] for k in ('text','audio','visual','intra','inter')]
        if not np.isfinite(gaps+[row['speaker_term'],row['final_difficulty']]).all():raise ValueError('nonfinite Difficulty')
        if not np.isclose(sum(gaps),row['total_gap'],rtol=1e-6,atol=1e-7):raise ValueError('gap sum mismatch')
        expected=(row['total_gap']+row['speaker_term'])/(row['valid_label_count']+row['speaker_term'])
        if not np.isclose(expected,row['final_difficulty'],rtol=1e-10,atol=1e-12):raise ValueError('Difficulty formula mismatch')
    return {'count':len(ids),'exact_training_coverage':True,'formula_recomputed':True,
            'mean':float(np.mean([r['final_difficulty'] for r in rows]))}

def metrics(p,y,lp=None):
    p=np.asarray(p,dtype=np.float64);y=np.asarray(y,dtype=int)
    if len(y)==0:raise ValueError('empty valid Calibration set')
    if lp is None:lp=np.log(np.maximum(p,np.finfo(float).tiny))
    confidence=p.max(1);correct=p.argmax(1)==y
    ece=0.
    for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        mask=(confidence>lo)&(confidence<=hi)
        if mask.any():ece+=mask.mean()*abs(confidence[mask].mean()-correct[mask].mean())
    return {'n':int(len(y)),'nll':float(-lp[np.arange(len(y)),y].mean()),
            'brier':float(np.mean(np.sum((p-np.eye(p.shape[1])[y])**2,axis=1))),
            'ece15':float(ece),'accuracy':float(correct.mean())}

def audit_calibration_arrays(a):
    valid=np.asarray(a['supervision_mask'],bool);labels=np.asarray(a['labels'])[valid]
    original=np.asarray(a['E0_probabilities']).argmax(1);out={}
    for method in METHODS:
        p=np.asarray(a[method+'_probabilities']);oof=np.asarray(a[method+'_oof_probabilities']);t=np.asarray(a[method+'_temperatures'])
        # OOF intentionally omits unknown rows; full fitted probabilities retain them.
        for value,reference in ((p,original),(oof[valid],original[valid])):
            if not np.isfinite(value).all() or (value<0).any() or not np.allclose(value.sum(1),1.,atol=1e-7):raise ValueError('invalid probability vector')
            if not np.array_equal(value.argmax(1),reference):raise ValueError('argmax invariance failed')
        if not np.isfinite(t).all() or (t<=0).any():raise ValueError('invalid temperature')
        lp=np.asarray(a[method+'_log_probabilities'])[valid] if method+'_log_probabilities' in a else None
        out[method]=metrics(p[valid],labels,lp)
        out[method+'_oof']=metrics(oof[valid],labels)
    return out

def check_recorded_hashes(root,hashes):
    for name,expected in hashes.items():
        if sha(root/name)!=expected:raise ValueError('artifact hash mismatch: '+str(root/name))

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def strict_reload_cpu(checkpoint_path,runner_path,teacher_scripts,manifest_sha,expected_dataset,expected_fold,expected_seed=None):
    """Strict state loading without inference, honoring each checkpoint's original code."""
    import torch
    sys.path.insert(0,str(teacher_scripts))
    teacher=load_module('finalizer_teacher_helpers',teacher_scripts/'teacher_runner.py')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    dataset=checkpoint.get('dataset',checkpoint.get('architecture',{}).get('dataset'))
    if dataset!=expected_dataset:raise ValueError('checkpoint dataset mismatch')
    if checkpoint.get('fold',expected_fold)!=expected_fold:raise ValueError('checkpoint fold mismatch')
    bindings=checkpoint.get('bindings',{});provenance=checkpoint['provenance']
    if bindings:
        if sha(runner_path)!=bindings['runner_sha256']:raise ValueError('runner lineage mismatch')
        check_recorded_hashes(teacher_scripts,bindings['helper_sha256'])
        check_recorded_hashes(Path(provenance['source_root']),bindings['source_sha256'])
        if bindings['manifest_sha256']!=manifest_sha:raise ValueError('checkpoint manifest mismatch')
        if not all(checkpoint.get(k) is True for k in ('is_best','complete','training_complete')):raise ValueError('student is not completed best')
        if checkpoint.get('selection_role')!='development':raise ValueError('student selection role mismatch')
        if expected_seed is not None and checkpoint['config']['seed']!=expected_seed:raise ValueError('seed mismatch')
    else:
        if provenance['approved_manifest_sha256']!=manifest_sha:raise ValueError('teacher manifest mismatch')
        if sha(runner_path)!=provenance['runner_sha256']:raise ValueError('Teacher runner lineage mismatch')
        if provenance['roles_accessed']!=['training'] or not provenance['no_historical_checkpoint_loaded']:raise ValueError('Teacher role/initialization mismatch')
        if checkpoint['epoch']!=checkpoint['config']['epochs'] or not checkpoint['complete']:raise ValueError('teacher not fixed final')
        check_recorded_hashes(Path(provenance['source_root']),provenance['source_sha256'])
    module=teacher.load_frozen_module(Path(provenance['source_root']))
    model=teacher.build_model(module,dataset,checkpoint['config'],torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    digest=teacher.state_sha256(model.state_dict())
    expected=checkpoint.get('model_state_sha256')
    if expected and digest!=expected:raise ValueError('state hash mismatch')
    del model,checkpoint
    return {'strict_reload':True,'device':'cpu','forward_executed':False,'model_state_sha256':digest}

def audit_epochs(output,training_ids,development_ids,expected_epochs,config=None,difficulty=None):
    records=[json.loads(line) for line in (output/'epochs.jsonl').read_text().splitlines()]
    if [r['epoch'] for r in records]!=list(range(1,expected_epochs+1)):raise ValueError('epoch coverage mismatch')
    for r in records:
        if not r['train']['exact_curriculum_coverage'] or not r['development']['exact_development_coverage']:raise ValueError('curriculum/development coverage failed')
        ids=r['train']['accessed_ids']
        if config is not None:
            expected_ratio=min(1.,round(config['epoch_ratio']*int((r['epoch']-1)/config['scheduler_steps']+1),2))
            if r['ratio']!=expected_ratio:raise ValueError('curriculum schedule mismatch')
            expected_ids=set(sorted(training_ids,key=difficulty.__getitem__)[:int(expected_ratio*len(training_ids))])
            if set(ids)!=expected_ids:raise ValueError('curriculum difficulty selection mismatch')
        if len(ids)!=len(set(ids)) or not set(ids)<=set(training_ids):raise ValueError('curriculum IDs invalid')
        audit=r['train'].get('direct_unknown_gradient_audit')
        if audit and audit['status']!='passed':raise ValueError('unknown gradient audit failed')
    batches=[json.loads(line) for line in (output/'batch_access.jsonl').read_text().splitlines()]
    for b in batches:
        allowed=set(training_ids) if b['role']=='training' else development_ids if b['role']=='development' else set()
        if not set(b['ids'])<=allowed:raise ValueError('illegal role access')
        if b['role']=='training' and (b['edges']>1000 or b['nodes']>5200):raise ValueError('capacity exceeded')
    scores=[r['development']['weighted_f1_percent_rounded'] for r in records]
    selected=int(np.argmax(scores))+1
    return records,selected

def finalize(root,ledger_path,teacher_pilot,cmerc_pilot,strict=True):
    manifest_path=root/'configs/approved_split_manifests.json';manifest=read(manifest_path)
    if sha(manifest_path)!=MANIFEST_SHA:raise ValueError('approved manifest hash mismatch')
    evaluation=audit_manifest_metadata(manifest)
    ledger=read(ledger_path);tasks=ledger['tasks']
    expected={'teacher':10,'difficulty':10,'cmerc':50,'calibration':50}
    counts=Counter(row['kind'] for row in tasks.values())
    if dict(counts)!=expected:raise ValueError('formal task cardinality mismatch')
    # Fail before writing any final report if an authorized task is incomplete.
    outputs={key:resolve_output(row) for key,row in tasks.items()}
    result={k:[] for k in ('teacher_runs','difficulty_manifest','cmerc_runs','development_metrics','calibration_parameters','calibration_metrics','grouped_oof_predictions','gpu_resource_summary')}
    verification={'status':'passed' if strict else 'diagnostic_only_not_ready','manifest_sha256':MANIFEST_SHA,'counts':expected,'strict_reload_requested':strict,
                  'evaluation_forward_executed':False,'evaluation_labels_read':False,'evaluation_artifact_opened':False,
                  'evaluation_metadata_coverage':evaluation,'checkpoint_reload':[],'calibration_checks':[]}
    source_hashes={str(manifest_path):MANIFEST_SHA};teacher_scripts=teacher_pilot/'scripts'
    difficulty_hashes={};teacher_hashes={}
    for key,row in tasks.items():
        if row['kind']!='teacher':continue
        output=outputs[key];meta=read(output/'teacher_checkpoint_metadata.json');audit=read(output/'access_audit.json')
        role=manifest['protocols'][row['dataset']]['folds'][f"fold{row['fold']}"]['roles']['training']
        if not audit['exact_final_coverage'] or audit.get('rejected_ids'):raise ValueError('Teacher access audit failed')
        if meta['provenance']['training_role_sha256']!=role['sha256']:raise ValueError('Teacher fold role binding mismatch')
        if set(meta['actual_accessed_ids'])!=set(role['dialogue_ids']):raise ValueError('Teacher access IDs mismatch')
        checkpoint=output/'teacher_final.pt';digest=sha(checkpoint)
        if digest!=meta['checkpoint_sha256']:raise ValueError('Teacher checkpoint hash mismatch')
        teacher_hashes[(row['dataset'],row['fold'])]=digest
        check_recorded_hashes(output,{'epochs.jsonl':meta['epoch_log_sha256'],'batch_access.jsonl':meta['batch_access_log_sha256']})
        reload_report=strict_reload_cpu(checkpoint,teacher_scripts/'teacher_runner.py',teacher_scripts,MANIFEST_SHA,row['dataset'],f"fold{row['fold']}") if strict else {'strict_reload':False}
        verification['checkpoint_reload'].append(dict(task=key,**reload_report))
        result['teacher_runs'].append(dict(task=key,dataset=row['dataset'],fold=row['fold'],reused=row['status']=='reused',output=str(output),checkpoint_sha256=digest,epochs=meta['epoch'],**reload_report))
        p=meta['provenance'];result['gpu_resource_summary'].append(dict(task=key,gpu_uuid=p.get('gpu_uuid'),elapsed_seconds=p.get('total_elapsed_seconds'),peak_reserved_bytes=p.get('peak_reserved_bytes'),reused=row['status']=='reused'))
        source_hashes[str(checkpoint)]=digest
    for key,row in tasks.items():
        if row['kind']!='difficulty':continue
        path=outputs[key];artifact=read(path)
        roles=manifest['protocols'][row['dataset']]['folds'][f"fold{row['fold']}"]['roles']
        audit=audit_difficulty(artifact,roles['training']['dialogue_ids'])
        if artifact['manifest_sha256']!=MANIFEST_SHA or artifact['teacher_sha256']!=teacher_hashes[(row['dataset'],row['fold'])]:raise ValueError('Difficulty binding mismatch')
        digest=sha(path);difficulty_hashes[(row['dataset'],row['fold'])]=digest;source_hashes[str(path)]=digest
        result['difficulty_manifest'].append(dict(task=key,dataset=row['dataset'],fold=row['fold'],reused=row['status']=='reused',difficulty_sha256=digest,teacher_sha256=artifact['teacher_sha256'],**audit))
    for key,row in tasks.items():
        if row['kind']!='cmerc':continue
        output=outputs[key];completion=read(output/'completion.json');config=read(output/'effective_config.json');provenance=read(output/'provenance.json')
        check_recorded_hashes(output,completion['hashes'])
        if completion['status']!='completed' or not completion['first_epoch_gate_passed'] or read(output/'first_epoch_gate.json')['status']!='passed':raise ValueError('CMERC completion/first epoch gate failed')
        roles=manifest['protocols'][row['dataset']]['folds'][f"fold{row['fold']}"]['roles']
        if provenance['difficulty_sha256']!=difficulty_hashes[(row['dataset'],row['fold'])]:raise ValueError('CMERC Difficulty binding mismatch')
        if provenance['historical_checkpoint_loaded'] or provenance['roles_accessed']!=['training','development']:raise ValueError('student initialization/access policy failed')
        epochs,best_epoch=audit_epochs(output,roles['training']['dialogue_ids'],set(roles['development']['dialogue_ids']),config['epochs'],config,{r['dialogue_id']:r['final_difficulty'] for r in read(outputs[f"difficulty:{row['dataset']}:fold{row['fold']}"])['dialogues']})
        if completion['best_epoch']!=best_epoch:raise ValueError('Development checkpoint selection mismatch')
        runner=(cmerc_pilot/'scripts/cmerc_runner.py') if row['status']=='reused' else root/'scripts/cmerc_runner.py'
        reload_report=strict_reload_cpu(output/'best.pt',runner,teacher_scripts,MANIFEST_SHA,row['dataset'],f"fold{row['fold']}",row['seed']) if strict else {'strict_reload':False}
        verification['checkpoint_reload'].append(dict(task=key,**reload_report))
        result['cmerc_runs'].append(dict(task=key,dataset=row['dataset'],fold=row['fold'],seed=row['seed'],reused=row['status']=='reused',output=str(output),best_epoch=best_epoch,best_score=completion['best_score'],best_checkpoint_sha256=sha(output/'best.pt'),initialization_sha256=provenance['random_initialization_state_sha256'],difficulty_sha256=provenance['difficulty_sha256'],**reload_report))
        for e in epochs:result['development_metrics'].append(dict(dataset=row['dataset'],fold=row['fold'],seed=row['seed'],epoch=e['epoch'],curriculum_ratio=e['ratio'],**e['development']))
        final=epochs[-1];result['gpu_resource_summary'].append(dict(task=key,gpu_uuid=provenance.get('gpu_uuid'),elapsed_seconds=final['elapsed_seconds'],peak_reserved_bytes=final['peak_reserved_bytes'],reused=row['status']=='reused'))
        source_hashes[str(output/'best.pt')]=sha(output/'best.pt')
    for key,row in tasks.items():
        if row['kind']!='calibration':continue
        output=outputs[key];params=read(output/'parameters.json');saved=read(output/'calibration_metrics.json');oof=read(output/'oof_diagnostic.json');provenance=read(output/'provenance.json')
        model_key=f"cmerc:{row['dataset']}:fold{row['fold']}:seed{row['seed']}"
        if provenance['checkpoint_sha256']!=sha(outputs[model_key]/'best.pt'):raise ValueError('Calibration checkpoint binding mismatch')
        with np.load(output/'calibration_probabilities.npz',allow_pickle=False) as archive:a={name:archive[name] for name in archive.files}
        if str(a['role'])!='calibration':raise ValueError('non Calibration probability archive')
        role=manifest['protocols'][row['dataset']]['folds'][f"fold{row['fold']}"]['roles']['calibration']
        if set(map(str,a['dialogue_ids']))!=set(role['dialogue_ids']):raise ValueError('Calibration ID coverage mismatch')
        if len(set(zip(map(str,a['dialogue_ids']),map(int,a['utterance_indices']))))!=len(a['labels']):raise ValueError('duplicate Calibration utterance')
        valid=np.asarray(a['supervision_mask'],bool)
        unknown={(r['dialogue_id'],int(r['utterance_index'])) for r in manifest['unknown_label_mask']['utterances']} if row['dataset']=='IEMOCAP' else set()
        expected_mask=np.asarray([(str(d),int(u)) not in unknown for d,u in zip(a['dialogue_ids'],a['utterance_indices'])])
        if not np.array_equal(valid,expected_mask):raise ValueError('Calibration unknown mask identity mismatch')
        if provenance['approved_manifest_sha256']!=MANIFEST_SHA:raise ValueError('Calibration manifest binding mismatch')
        if valid.sum()!=role['supervision_eligible_count'] or len(valid)!=role['utterance_count']:raise ValueError('Calibration mask count mismatch')
        recomputed=audit_calibration_arrays(a)
        for method in METHODS:
            parameter=params[method]
            temperature=(np.full(len(valid),parameter['temperature']) if 'temperature' in parameter else np.asarray(parameter['temperatures'])[a['original_predictions']])
            if not np.allclose(temperature,a[method+'_temperatures'],rtol=1e-12,atol=1e-12):raise ValueError('parameter temperature replay mismatch')
            scaled=a['input_scores']/temperature[:,None];scaled=scaled-scaled.max(1,keepdims=True)
            probability=np.exp(scaled);probability/=probability.sum(1,keepdims=True)
            if not np.allclose(probability,a[method+'_probabilities'],rtol=1e-8,atol=1e-10):raise ValueError('parameter probability replay mismatch')
        held_seen=Counter()
        for cv in oof['folds']:
            if set(cv['train_groups'])&set(cv['held_groups']):raise ValueError('Grouped CV leakage')
            held_seen.update(cv['held_groups'])
        if set(held_seen)!=set(map(str,a['group_ids'][valid])) or any(n!=1 for n in held_seen.values()):raise ValueError('Grouped CV coverage mismatch')
        for method in METHODS:
            for metric in ('nll','brier','ece15','accuracy'):
                if not np.isclose(recomputed[method][metric],saved[method][metric],atol=1e-10,rtol=1e-8):raise ValueError('Calibration metric recomputation mismatch '+metric)
                if not np.isclose(recomputed[method+'_oof'][metric],oof['metrics'][method][metric],atol=1e-9,rtol=1e-7):raise ValueError('OOF metric recomputation mismatch '+metric)
            result['calibration_parameters'].append(dict(dataset=row['dataset'],fold=row['fold'],seed=row['seed'],method=method,parameters=params[method],checkpoint_sha256=provenance['checkpoint_sha256']))
            for scope in ('fit','grouped_oof'):
                value=recomputed[method] if scope=='fit' else recomputed[method+'_oof']
                result['calibration_metrics'].append(dict(dataset=row['dataset'],fold=row['fold'],seed=row['seed'],method=method,scope=scope,**value))
            p=a[method+'_oof_probabilities']
            for i in range(len(valid)):
                result['grouped_oof_predictions'].append(dict(dataset=row['dataset'],fold=row['fold'],seed=row['seed'],method=method,dialogue_id=str(a['dialogue_ids'][i]),utterance_index=int(a['utterance_indices'][i]),group_id=str(a['group_ids'][i]),oof_fold=int(a['oof_fold'][i]),supervision_mask=int(valid[i]),label=int(a['labels'][i]) if valid[i] else '',probabilities=p[i].tolist() if valid[i] else []))
        verification['calibration_checks'].append(dict(task=key,argmax_invariance=True,metrics_recomputed=True,valid_count=int(valid.sum()),masked_count=int((~valid).sum())))
        source_hashes[str(output/'calibration_probabilities.npz')]=sha(output/'calibration_probabilities.npz')
        source_hashes[str(output/'parameters.json')]=sha(output/'parameters.json')
    # Scheduler wall time covers each GPU-bound stage, including Difficulty and Calibration.
    resource_by_task={r['task']:r for r in result['gpu_resource_summary']}
    for key,row in tasks.items():
        resource=resource_by_task.setdefault(key,dict(task=key,reused=row['status']=='reused'))
        resource.update(kind=row['kind'],dataset=row['dataset'],fold=row['fold'],seed=row['seed'])
        if row['status']=='completed':
            attempt=[a for a in row['attempts'] if a.get('returncode')==0][-1]
            seconds=(datetime.fromisoformat(attempt['finished_at'])-datetime.fromisoformat(attempt['started_at'])).total_seconds()
            resource.update(gpu_uuid=attempt['gpu_uuid'],scheduled_gpu_seconds=seconds,scheduled_gpu_hours=seconds/3600.)
        else:
            resource.update(scheduled_gpu_seconds=0.,scheduled_gpu_hours=0.,historical_gpu_hours=float(resource.get('elapsed_seconds') or 0)/3600.)
    result['gpu_resource_summary']=list(resource_by_task.values())
    results=root/'results/final_verification'
    if results.exists():raise FileExistsError('refusing to overwrite finalized package: '+str(results))
    for name,rows in result.items():csvout(results/(name+'.csv'),rows)
    jsonout(results/'evaluation_readiness_manifest.json',dict(status='frozen_pending_researcher_approval',manifest_sha256=MANIFEST_SHA,datasets=evaluation,model_parameters=result['cmerc_runs'],calibrators=result['calibration_parameters']))
    jsonout(results/'verification.json',verification);jsonout(results/'source_hashes.json',source_hashes)
    jsonout(results/'report_data.json',{name:rows for name,rows in result.items() if name!='grouped_oof_predictions'})
    return verification

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--formal-root',type=Path,required=True);p.add_argument('--ledger',type=Path)
    p.add_argument('--teacher-pilot',type=Path,required=True);p.add_argument('--cmerc-pilot',type=Path,required=True)
    p.add_argument('--skip-strict-reload',action='store_true',help='diagnostic aggregation only; readiness is not accepted without strict reload')
    args=p.parse_args();root=args.formal_root.resolve()
    report=finalize(root,args.ledger or root/'state/scheduler/task_ledger.json',args.teacher_pilot.resolve(),args.cmerc_pilot.resolve(),strict=not args.skip_strict_reload)
    print(json.dumps({'status':report['status'],'counts':report['counts'],'evaluation_executed':False},indent=2))
if __name__=='__main__':main()
