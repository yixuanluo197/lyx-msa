"""Independent CPU audit of the preregistered candidate confirmation.

Does not import inference, behavior_metrics, analysis, or fitting implementations.
Preflight success is explicitly separate from a complete scientific result audit.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import traceback
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
KEY=['dataset','seed','dialogue_id','utterance_index']
CANDIDATES={'IEMOCAP':'text','MELD':'audio'}
NAMES={'text':['text0','text1','text2','text3'],'audio':['audio'],'visual':['visual'],
       'joint':['text0','text1','text2','text3','audio','visual']}
FEATURES={'M_gaussian':['gaussian_mu','gaussian_delta'],'M_gaussian_mu':['gaussian_mu'],'M_gaussian_delta':['gaussian_delta']}
COUNTS={'IEMOCAP':(527,12,1),'MELD':(1109,114,95)}
METRICS=['ece15','binary_brier','binary_nll']
CHECKS=[]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def read(path):
    return pd.read_csv(path,dtype={'dialogue_id':str},float_precision='round_trip')


def close(a,b,atol=1e-10):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape,(a.shape,b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert np.allclose(a,b,rtol=0,atol=atol),float(np.max(np.abs(a-b)))


def flag(v):
    if isinstance(v,(bool,np.bool_)): return bool(v)
    assert str(v).lower() in {'true','false','0','1'},v
    return str(v).lower() in {'true','1'}


def probs(values,classes):
    p=np.asarray([json.loads(v) if isinstance(v,str) else v for v in values],float)
    assert p.ndim==2 and p.shape[1]==classes
    assert np.isfinite(p).all() and (p>=0).all() and (p<=1+1e-6).all()
    assert np.allclose(p.sum(-1),1,atol=1e-6,rtol=1e-5)
    return p


def behavior(p,q):
    """Scalar utterance, literal repeat formulas, natural log JS in nats."""
    assert q.ndim==2 and len(q)>0 and q.shape[1]==len(p)
    maxima=q.max(1)
    flips=q.argmax(1)!=p.argmax()
    midpoint=(p[None,:]+q)/2
    js=[]
    for qi,mi in zip(q,midpoint):
        a=p>0; b=qi>0
        js.append((np.sum(p[a]*np.log(p[a]/mi[a]))+
                   np.sum(qi[b]*np.log(qi[b]/mi[b])))/2)
    mu=float(np.mean(flips*maxima)); rate=float(np.mean(flips))
    assert 0<=mu<=rate+1e-14
    return {'mu':mu,'delta':float(abs(p.max()-maxima.mean())),
            'js':float(np.mean(js)),'flip_rate':rate,
            'confidence_change_mean':float(np.mean(maxima-p.max())),
            'confidence_change_std':float(np.std(maxima-p.max()))}


def metrics(q,y):
    q=np.asarray(q,float); y=np.asarray(y,int)
    assert len(q)==len(y) and np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
    result=0.
    for i in range(15):
        m=(q>=i/15)&((q<(i+1)/15) if i<14 else (q<=1))
        if m.any(): result+=abs(float(np.sum(q[m]-y[m])))
    qc=np.clip(q,1e-15,1-1e-15)
    return {'ece15':result/len(q),'binary_brier':float(np.mean((q-y)**2)),
            'binary_nll':float(np.mean(-y*np.log(qc)-(1-y)*np.log1p(-qc)))}


def replay_fit(frame,fit,indices):
    indices=np.asarray(indices,int);train=frame.iloc[indices]
    assert fit['train_indices']==indices.tolist() and fit['n_train']==len(indices)
    assert fit['train_dialogues']==sorted(train.dialogue_id.unique().tolist())
    assert len(indices)==len(set(indices))
    fs=fit['features'];n=len(indices)
    assert fit['optimizer']=={'C':1.,'solver':'lbfgs','max_iter':2000,'tol':1e-8}
    if fit['penalty']=='off':
        assert fs==[] and fit['scaler_mean']==fit['scaler_scale']==[]
        assert fit['effective_behavior_coefficients']==[] and fit['design_divisor']==1
    else:
        assert fs and float(fit['penalty']) in [1.,.1,.01]
        x=train[fs].to_numpy(float);mean=x.mean(0);var=x.var(0)
        upper=n*np.finfo(float).eps*var+(n*mean*np.finfo(float).eps)**2
        scale=np.sqrt(var);scale[var<=upper]=1
        close(mean,fit['scaler_mean']);close(scale,fit['scaler_scale'])
        close(fit['design_divisor'],np.sqrt(n*float(fit['penalty'])))
    coeff=np.asarray(fit['raw_coefficients'])
    close(coeff[0],fit['confidence_coefficient'])
    close(coeff[1:]/fit['design_divisor'],fit['effective_behavior_coefficients'])
    def predict(held):
        base=fit['intercept']+fit['confidence_coefficient']*held.confidence.to_numpy(float)
        b=np.zeros(len(held))
        if fs: b=((held[fs].to_numpy(float)-fit['scaler_mean'])/fit['scaler_scale'])@np.asarray(fit['effective_behavior_coefficients'])
        return {'base_score':base,'behavior_score':b,'reliability_score':base+b,'oof_confidence':expit(base+b)}
    return predict


def cluster_statistics(frame,ids,draws):
    """Independent cluster weights; retain all 2000 registered draws."""
    idx=pd.Index(ids).get_indexer(frame.dialogue_id)
    assert (idx>=0).all()
    mult=np.stack([np.bincount(d,minlength=len(ids)) for d in draws])
    counts=np.bincount(idx,minlength=len(ids));den=mult@counts
    def mean(values):
        sums=np.bincount(idx,weights=np.asarray(values,float),minlength=len(ids))
        return np.divide(mult@sums,den,out=np.full(len(den),np.nan),where=den>0)
    return mean,idx,mult,den


def interval_check(row,distribution,point):
    valid=np.isfinite(distribution)
    assert row['resamples']==2000 and row['valid_resamples']==int(valid.sum()) and row['invalid_resamples']==int((~valid).sum())
    assert not flag(row['refit']) and flag(row['exploratory']) and not flag(row['multiple_testing_adjusted'])
    nullable_close(row['mean_difference'],point)
    lo,hi=np.quantile(distribution[valid],[.025,.975]) if valid.any() else (np.nan,np.nan)
    nullable_close(row['ci_low'],lo);nullable_close(row['ci_high'],hi)



def source_audit(root,config):
    path=root/'scripts/run_candidate.py';source=path.read_text();tree=ast.parse(source)
    calls=[];forbidden=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call):continue
        if isinstance(node.func,ast.Attribute):
            if node.func.attr=='dataset_for':
                assert len(node.args)>1 and isinstance(node.args[1],ast.Constant)
                calls.append(node.args[1].value)
            if node.func.attr in ['align','alignment_gate','validation','pilot','training_stats','fit','backward']:
                forbidden.append(node.func.attr)
    assert calls==['validation'] and not forbidden,(calls,forbidden)
    assert 'RNG_SEED, STRENGTH, REPEATS = 1475, 0.05, 10' in source
    assert 'assert_target_only' in source and 'finally:' in source and 'before_state' in source
    assert sha(root/'scripts/mainline_core.py')==sha(Path(config['source_mainline'])/'scripts/mainline_core.py')
    before=json.loads((root/'provenance/source_hashes_before.json').read_text())
    for p,d in before.items():assert sha(p)==d,p
    stats={}
    for ds in CANDIDATES:
        z=np.load(root/f'provenance/{ds}_train_statistics.npz')
        for name in ['text0','text1','text2','text3','audio','visual']:
            assert int(z[name+'_count'])==config['datasets'][ds]['train_count']
            assert (z[name+'_std']>=0).all() and np.isfinite(z[name+'_std']).all()
        stats[ds]=sha(root/f'provenance/{ds}_train_statistics.npz')
    return {'validation_only_executable_dataset_calls':calls,'no_training_or_Test_inference_call':True,
        'immutable_input_hashes_checked':len(before),'training_statistics_sha256':stats,
        'runner_sha256':sha(path),'core_sha256':sha(root/'scripts/mainline_core.py')}


def original_source(config,ds,seed):
    if seed==1475:return Path(config['source_pilot'])/ds/'original_predictions.csv'
    return Path(config['source_five_seed'])/f'runs/{ds}/seed{seed}/original_predictions.csv'


def inference_audit(root,config,smoke=False):
    runs=[];checked_hashes={};audits_by_ds={};prediction_rows=0
    if smoke:paths=sorted((root/'runs').glob('*/*/*/smoke/complete.json'))
    else:paths=sorted((root/'runs').glob('*/*/*/complete.json'))
    assert paths,'No completed inference evidence'
    if not smoke:assert len(paths)==12,len(paths)
    for complete in paths:
        path=complete.parent;result=json.loads(complete.read_text());ds=result['dataset'];seed=result['seed'];block=result['block']
        assert result['status']=='completed' and result['smoke']==smoke and ds in CANDIDATES
        assert result['modality']==CANDIDATES[ds] and result['strength']==.05 and result['rng_seed']==1475
        assert seed in config['blocks'][str(block)]['model_seeds'] and not(seed==1475 and block==0)
        assert result['utterances']==(1 if smoke else COUNTS[ds][0])
        assert result['perturbed_rows']==result['target_forwards']==result['utterances']*10
        assert result['weights_unchanged'] and result['batch_inputs_restored'] and result['identical_input_parity']['passed']
        assert result['graph_hook_checks']>0 and result['target_mask_checks']>0 and result['cached_uncached_parity_checks']>0
        assert result['parity']['passed'] and result['parity']['canonical_original']['passed']
        assert all(i['equal'] for i in result['parity']['checks'].values())
        replay=json.loads((path/'replay_diagnostics.json').read_text())
        assert replay['passed'] and replay['inputs_exact_equal'] and replay['weights_exact_equal'] and replay['all_modules_eval']
        assert replay['input_sha256_before']==replay['input_sha256_after']
        assert replay['parameter_dtypes']==['torch.float32'] and replay['atol']==1e-6 and replay['rtol']==1e-5
        assert len(replay['comparisons_to_initial_cached_output'])==8
        assert all(i['argmax_equal'] and i['probability_tolerance_passed'] and i['log_probability_tolerance_passed'] for i in replay['comparisons_to_initial_cached_output'])
        before=json.loads((path/'source_hashes_before.json').read_text());after=json.loads((path/'source_hashes_after.json').read_text())
        assert before==after==result['source_sha256']
        for source,digest in before.items():
            if source not in checked_hashes:checked_hashes[source]=sha(source)
            assert checked_hashes[source]==digest,source
        for section in ['actual_imported_source_sha256','data_sha256','actual_opened_data_sha256']:
            assert result[section]
            for source,digest in result[section].items():assert before[source]==digest
        for output,digest in result['output_sha256'].items():assert sha(path/output)==digest,output
        effective=json.loads((path/'effective_config.json').read_text())
        assert effective['split']=='validation' and effective['batch_size']==16 and effective['rnn_backend']=='aten' and effective['precision']=='float32'
        split=effective['checkpoint_config']['provenance']['split'];valid=set(map(str,split['validation_ids']))
        assert valid.isdisjoint(map(str,split['train_ids'])) and valid.isdisjoint(map(str,split['test_ids']))
        original=read(path/'original_predictions.csv');prior=read(original_source(config,ds,seed))
        pair=original.merge(prior,on=KEY,validate='one_to_one',suffixes=('','_prior'))
        assert len(original)==len(pair)==COUNTS[ds][0] and original.dialogue_id.nunique()==COUNTS[ds][1]
        assert set(original.dialogue_id)==valid
        for col in ['label','prediction','confidence']:close(pair[col],pair[col+'_prior'],0)
        classes=6 if ds=='IEMOCAP' else 7
        pp=probs(original.probabilities,classes);close(pp,probs(prior.probabilities,classes),0)
        fresh=np.load(path/'fresh_validation_predictions.npz')
        assert fresh['dialogue_ids'].astype(str).tolist()==original.dialogue_id.tolist()
        close(fresh['utterance_indices'],original.utterance_index,0);close(fresh['labels'],original.label,0)
        assert np.allclose(fresh['probabilities'],pp,atol=1e-6,rtol=1e-5)
        audit=read(path/'perturbation_audit.csv');raw=read(path/'perturbed_predictions.csv')
        assert len(raw)==result['utterances']*10 and not raw.duplicated(KEY+['block','repeat_id']).any()
        assert len(audit)==len(raw)*len(NAMES[CANDIDATES[ds]])
        assert not audit.duplicated(KEY+['block','repeat_id','feature_name']).any()
        assert audit.stdzero_unchanged.map(flag).all() and not raw.visual_missing_skipped.map(flag).any()
        assert set(raw.rng_repeat_id)==set(range(block*10,(block+1)*10)) and set(raw.perturbation_seed)=={1475}
        for row in audit.to_dict('records'):
            key=json.loads(row['rng_key']);canonical=json.dumps(key,ensure_ascii=False,separators=(',',':'),allow_nan=False)
            assert key==[ds,1475,row['dialogue_id'],int(row['utterance_index']),row['feature_name'],'gaussian',.05,int(row['rng_repeat_id'])]
            assert str(row['rng_seed'])==str(int.from_bytes(hashlib.sha256(canonical.encode()).digest()[:16],'big'))
            assert row['rng_repeat_id']==block*10+row['repeat_id'] and row['block']==block
            assert row['requested_count']==row['eligible_count'] and 0<=row['actual_changed_count']<=row['eligible_count']
            assert row['feature_name'] in NAMES[CANDIDATES[ds]]
            assert len(row['vector_sha256'])==len(row['original_vector_sha256'])==64
        for row in raw.to_dict('records'):
            expected=[[ds,1475,row['dialogue_id'],int(row['utterance_index']),name,'gaussian',.05,int(row['rng_repeat_id'])] for name in NAMES[CANDIDATES[ds]]]
            assert json.loads(row['rng_key'])==expected
        q=probs(raw.probabilities,classes);lp=np.array([json.loads(v) for v in raw.log_probabilities])
        close(q,np.exp(lp),1e-14);close(raw.prediction,q.argmax(1),0);close(raw.confidence,q.max(1),0)
        if not smoke:
            assert result['visual_missing_retained']==COUNTS[ds][2]
            assert set(map(tuple,raw[KEY].drop_duplicates().to_numpy()))==set(map(tuple,original[KEY].to_numpy()))
        audits_by_ds.setdefault(ds,[]).append(audit)
        prediction_rows+=len(raw);runs.append(f'{ds}/seed{seed}/block{block}'+('/smoke' if smoke else ''))
    cross=0;independent=0
    for ds,parts in audits_by_ds.items():
        frame=pd.concat(parts);keys=['dialogue_id','utterance_index','rng_repeat_id','feature_name']
        block0=frame[frame.block.eq(0)]
        if not smoke:assert set(block0.seed)==set(config['new_model_seeds'])
        for key,g in block0.groupby(keys,sort=False):
            assert g.vector_sha256.nunique()==g.original_vector_sha256.nunique()==g.rng_seed.nunique()==1
            if not smoke:assert len(g)==4
            cross+=1
        fresh=frame[frame.seed.eq(1475)]
        if not smoke:
            assert set(fresh.block)=={1,2}
            for _,g in fresh.groupby(['dialogue_id','utterance_index','feature_name'],sort=False):
                assert len(g)==20 and g.rng_seed.nunique()==20 and g.vector_sha256.nunique()==20
                independent+=1
    if not smoke:assert prediction_rows==98160
    return {'completed_runs':runs,'new_prediction_rows':prediction_rows,'cross_seed_vector_bytes_groups':cross,
        'independent_block_rng_vector_groups':independent,'weight_graph_mask_restore_parity':True,
        'historical1475_vector_bytes_not_recorded':'Historical keys/formula/stats verified, exact input bytes compared among four new model seeds.'}


def raw_audit(root,config,preflight=False):
    result=[];nrows=0
    for ds,modality in CANDIDATES.items():
        runs=[(1475,0,Path(config['source_pilot'])/ds/'perturbed_predictions.csv')]
        if not preflight:
            runs += [(s,0,root/f'runs/{ds}/seed{s}/block0/perturbed_predictions.csv') for s in config['new_model_seeds']]
            runs += [(1475,b,root/f'runs/{ds}/seed1475/block{b}/perturbed_predictions.csv') for b in [1,2]]
        for seed,block,path in runs:
            original=read(original_source(config,ds,seed));raw=read(path)
            if block==0 and seed==1475:raw=raw[raw.modality.eq(modality)&raw.corruption.eq('gaussian')].copy()
            assert len(raw)==len(original)*10 and set(raw.strength)=={.05} and set(raw.modality)=={modality} and set(raw.corruption)=={'gaussian'}
            assert not raw.duplicated(KEY+['repeat_id']).any() and set(raw.repeat_id)==set(range(10))
            assert not raw.visual_missing_skipped.map(flag).any()
            classes=6 if ds=='IEMOCAP' else 7
            p=probs(original.probabilities,classes);q=probs(raw.probabilities,classes)
            close(original.confidence,p.max(1),0);close(original.prediction,p.argmax(1),0)
            close(raw.confidence,q.max(1),0);close(raw.prediction,q.argmax(1),0)
            lookup={tuple(r[k] for k in KEY):(p[i],r) for i,r in enumerate(original.to_dict('records'))}
            raw['_position']=np.arange(len(raw))
            for identity,g in raw.groupby(KEY,sort=False):
                assert len(g)==10 and set(g.repeat_id)==set(range(10)) and identity in lookup
                g=g.sort_values('repeat_id');pp,orig=lookup[identity];qq=q[g._position]
                for row in g.to_dict('records'):
                    repeat=block*10+int(row['repeat_id'])
                    assert json.loads(row['rng_key'])==[[ds,1475,identity[2],int(identity[3]),name,'gaussian',.05,repeat] for name in NAMES[modality]]
                values=behavior(pp,qq)
                for suffix,half in [('left',qq[:5]),('right',qq[5:])]:
                    for m,v in behavior(pp,half).items():
                        if m in ['mu','delta','js','flip_rate']:values[m+'_'+suffix]=v
                result.append(dict(zip(KEY,identity),block=block,modality=modality,corruption='gaussian',strength=.05,
                    confidence=float(orig['confidence']),label=int(orig['label']),prediction=int(orig['prediction']),correct=int(orig['label']==orig['prediction']),**values))
            nrows+=len(raw)
    f=pd.DataFrame(result)
    assert len(f)==(1636 if preflight else 11452),len(f)
    return f,{'raw_probability_rows_recomputed':nrows,'utterance_behavior_records':len(f),'confidence_weighted_mu_and_original_delta':True}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--stage',choices=['smoke','preflight','full'],default='full')
    args=parser.parse_args();root=args.root.resolve();config=json.loads((root/'preregistration.json').read_text())
    result={'status':'failed','all_passed':False,'stage':args.stage,'checks':{}}
    output=root/'results'/('independent_verification.json' if args.stage=='full' else 'independent_preflight.json')
    if output.exists():result['prior_verification']=json.loads(output.read_text())
    try:
        assert config['seeds']==[1475,67137,2024,3407,42] and config['repeats']==10 and config['strength']==.05
        assert config['methods']==['E0','E1']+list(FEATURES)
        assert config['rng_seed']==1475 and config['expected_new_jobs']==12 and config['expected_new_prediction_rows']==98160
        result['checks']['source']=source_audit(root,config)
        result['checks']['inference']=inference_audit(root,config,smoke=args.stage!='full')
        if args.stage!='smoke':
            raw,result['checks']['raw_behavior']=raw_audit(root,config,preflight=args.stage=='preflight')
            result['checks']['calibration']=fit_audit(root,config,raw,preflight=args.stage=='preflight')
            result['checks']['diagnostics']=diagnostics_audit(root,preflight=args.stage=='preflight')
            result['checks']['concentration_and_random_blocks']=concentration_audit(root,preflight=args.stage=='preflight')
            result['checks']['bootstrap']=bootstrap_audit(root,config,preflight=args.stage=='preflight')
        result['status']='passed';result['all_passed']=True
        if args.stage!='full':result['formal_results_not_claimed']=True
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}';result['traceback']=traceback.format_exc()
    output.parent.mkdir(exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']!='passed':raise SystemExit(1)

def fit_audit(root,config,raw,preflight=False):
    out=root/('historical_preflight' if preflight else 'results')
    analysis=json.loads((out/'analysis_verification.json').read_text());assert analysis['all_passed']
    assert analysis['status']==('historical_preflight_passed' if preflight else 'analysis_complete')
    for p,d in json.loads((out/'source_hashes.json').read_text()).items():assert sha(p)==d,p
    for p,d in analysis['output_sha256'].items():assert sha(out/p)==d,p
    frame=read(out/'feature_matrix.csv');behavior_saved=read(out/'behavior_features.csv')
    pred=read(out/'oof_predictions.csv');saved_metrics=read(out/'calibration_metrics.csv');inner_saved=read(out/'inner_cv_results.csv')
    assert len(frame)==len(raw)==len(behavior_saved) and len(pred)==len(frame)*5
    assert not frame.duplicated(KEY+['block']).any() and not pred.duplicated(KEY+['block','method']).any()
    match=raw.merge(behavior_saved,on=KEY+['block'],validate='one_to_one',suffixes=('','_saved'));assert len(match)==len(raw)
    for m in ['mu','delta','js','flip_rate','confidence_change_mean','confidence_change_std','mu_left','mu_right','delta_left','delta_right','js_left','js_right']:
        close(match[m],match[m+'_saved'])
    merged=frame.merge(raw,on=KEY+['block'],validate='one_to_one',suffixes=('','_raw'))
    for m in ['mu','delta']:close(merged['gaussian_'+m],merged[m+'_raw'])
    historical=json.loads((Path(config['source_five_seed'])/'baseline/fold_parameters.json').read_text())
    predictions_replayed=inner_replayed=selected=0;fold_count=0
    for (ds,seed,block),g in frame.groupby(['dataset','seed','block'],sort=False):
        g=g.reset_index(drop=True);n=COUNTS[ds][0];assert len(g)==n
        old={f['fold']:f for f in historical if f['dataset']==ds and f['seed']==seed}
        records=json.loads((out/f'fold_parameters/{ds}_seed{seed}_block{block}.json').read_text())
        assert len(records)==len(old)==(12 if ds=='IEMOCAP' else 5)
        pp=pred[pred.dataset.eq(ds)&pred.seed.eq(seed)&pred.block.eq(block)];covered=np.zeros(n,int)
        for r in records:
            fold_count+=1;fold=r['fold'];tr=np.array(r['train_indices']);he=np.array(r['held_indices']);covered[he]+=1
            assert r['train_dialogues']==old[fold]['train_dialogues'] and r['held_dialogues']==old[fold]['held_dialogues']
            assert set(r['train_dialogues']).isdisjoint(r['held_dialogues']) and set(tr).isdisjoint(he)
            assert sorted(np.r_[tr,he])==list(range(n))
            assert tr.tolist()==np.flatnonzero(g.dialogue_id.isin(r['train_dialogues'])).tolist()
            assert he.tolist()==np.flatnonzero(g.dialogue_id.isin(r['held_dialogues'])).tolist()
            actual={'E1':replay_fit(g,r['E1'],tr)(g.iloc[he])}
            splits=list(GroupKFold(5).split(tr,groups=g.iloc[tr].dialogue_id));assert len(r['inner_folds'])==5
            for inn,(ii,jj) in zip(r['inner_folds'],splits):
                assert inn['train_indices']==tr[ii].tolist() and inn['held_indices']==tr[jj].tolist()
                assert set(inn['train_dialogues']).isdisjoint(inn['held_dialogues'])
                replay_fit(g,inn['E1'],tr[ii])
            assert set(r['methods'])==set(FEATURES)
            for method,item in r['methods'].items():
                assert item['features']==FEATURES[method]
                best='off';best_loss=np.inf
                for penalty in ['off',1.,.1,.01]:
                    candidates=item['inner_candidate_fits'][str(penalty)];assert len(candidates)==5;squared=[]
                    for candidate,inn in zip(candidates,r['inner_folds']):
                        assert candidate['held_indices']==inn['held_indices'] and candidate['held_dialogues']==inn['held_dialogues']
                        fit=candidate['fit'];assert fit['penalty']==penalty and fit['features']==([] if penalty=='off' else FEATURES[method])
                        if penalty=='off':assert fit==inn['E1']
                        held=g.iloc[inn['held_indices']];q=replay_fit(g,fit,inn['train_indices'])(held)['oof_confidence']
                        err=(q-held.correct.to_numpy())**2;close(err.mean(),candidate['held_brier']);squared.extend(err);inner_replayed+=len(err)
                    loss=np.mean(squared);close(loss,item['candidate_brier'][str(penalty)])
                    saved=inner_saved[inner_saved.dataset.eq(ds)&inner_saved.seed.eq(seed)&inner_saved.block.eq(block)&inner_saved.fold.eq(fold)&inner_saved.method.eq(method)&inner_saved.penalty.eq(str(penalty))]
                    assert len(saved)==1;close(saved.iloc[0].pooled_inner_brier,loss)
                    if loss<best_loss-1e-12:best,best_loss=penalty,loss
                assert best==item['selected_penalty']==item['outer_fit']['penalty'];selected+=1
                if best=='off':assert item['outer_fit']==r['E1']
                actual[method]=replay_fit(g,item['outer_fit'],tr)(g.iloc[he])
            for method in ['E0','E1']+list(FEATURES):
                saved=pp[pp.method.eq(method)&pp.fold.eq(fold)].set_index(KEY);held=g.iloc[he].set_index(KEY)
                assert set(saved.index)==set(held.index);saved=saved.loc[held.index]
                for c in ['confidence','label','prediction','correct']:close(saved[c],held[c],0)
                assert np.array_equal(saved.correct,(saved.prediction==saved.label).astype(int))
                if method=='E0':close(saved.oof_confidence,held.confidence,0)
                else:
                    for col,values in actual[method].items():close(saved[col],values)
                predictions_replayed+=len(saved)
        assert np.all(covered==1)
    from sklearn.metrics import roc_auc_score
    for (ds,seed,block,method),g in pred.groupby(['dataset','seed','block','method']):
        saved=saved_metrics[saved_metrics.dataset.eq(ds)&saved_metrics.seed.eq(seed)&saved_metrics.block.eq(block)&saved_metrics.method.eq(method)]
        assert len(saved)==1;row=saved.iloc[0]
        expected=metrics(g.oof_confidence,g.correct)
        expected.update(accuracy=float(np.mean(g.prediction==g.label)),weighted_f1=f1_score(g.label,g.prediction,average='weighted'),macro_f1=f1_score(g.label,g.prediction,average='macro'),correctness_auc=roc_auc_score(g.correct,g.oof_confidence),mean_confidence=float(g.oof_confidence.mean()))
        for m,v in expected.items():close(row[m],v)
    old=read(Path(config['source_mainline'])/'joint/oof_predictions.csv')
    e1=pred[pred.method.eq('E1')].merge(old[old.method.eq('E1')],on=KEY,validate='many_to_one',suffixes=('','_old'))
    assert len(e1)==len(frame);close(e1.oof_confidence,e1.oof_confidence_old,1e-8);close(e1.fold,e1.fold_old,0)
    old=read(Path(config['source_strength'])/'results/oof_predictions.csv')
    total=0;maxerror=0
    for ds,mod in CANDIDATES.items():
        a=pred[pred.dataset.eq(ds)&pred.seed.eq(1475)&pred.block.eq(0)&pred.method.eq('M_gaussian')]
        b=old[old.dataset.eq(ds)&old.modality.eq(mod)&old.strength.eq(.05)&old.method.eq('M_gaussian')]
        pair=a.merge(b,on=KEY,validate='one_to_one',suffixes=('','_old'));assert len(pair)==COUNTS[ds][0]
        close(pair.oof_confidence,pair.oof_confidence_old,1e-8);total+=len(pair)
        maxerror=max(maxerror,float(np.max(abs(pair.oof_confidence-pair.oof_confidence_old))))
    return {'outer_folds_replayed':fold_count,'oof_predictions_replayed':predictions_replayed,'inner_predictions_replayed':inner_replayed,
        'nested_penalty_selections_recomputed':selected,'historical1475_candidate_rows_reproduced':total,'historical_max_absolute_error':maxerror,
        'train_only_scalers_dialogue_grouping_off_exact_E1':True,'metrics_recomputed':len(saved_metrics)}

def nullable(actual,expected):
    if np.isfinite(expected):close(actual,expected)
    else:assert pd.isna(actual)


def diagnostics_audit(root,preflight=False):
    from scipy.stats import spearmanr
    out=root/('historical_preflight' if preflight else 'results')
    b=read(out/'behavior_features.csv');pred=read(out/'oof_predictions.csv');cm=read(out/'calibration_metrics.csv')
    checked=0
    for row in read(out/'behavior_summary.csv').to_dict('records'):
        f=b[b.dataset.eq(row['dataset'])&b.seed.eq(row['seed'])&b.block.eq(row['block'])]
        if row['population']!='all':f=f[f.correct.eq(int(row['population']=='correct'))]
        assert len(f)==row['utterances'] and f.dialogue_id.nunique()==row['dialogues']
        close(row['visual_missing'],f.visual_missing.map(flag).sum(),0)
        close(row['any_flip_fraction'],f.flip_rate.gt(0).mean());close(row['mu_zero_fraction'],f.mu.eq(0).mean())
        for m in ['mu','delta','js','flip_rate','confidence_change_mean','confidence_change_std']:
            close(row[m+'_mean'],f[m].mean());nullable(row[m+'_std'],f[m].std(ddof=1))
        checked+=1
    for row in read(out/'split_half_stability.csv').to_dict('records'):
        f=b[b.dataset.eq(row['dataset'])&b.seed.eq(row['seed'])&b.block.eq(row['block'])];m=row['measure']
        left,right=f[m+'_left'],f[m+'_right']
        nullable(row['spearman'],spearmanr(left,right).statistic if left.nunique()>1 and right.nunique()>1 else np.nan)
        for side,x in [('left',left),('right',right)]:
            close(row[side+'_mean'],x.mean());close(row[side+'_zero_fraction'],x.eq(0).mean())
        checked+=1
    for row in read(out/'conditional_behavior.csv').to_dict('records'):
        f=b[b.dataset.eq(row['dataset'])&b.seed.eq(row['seed'])&b.block.eq(row['block'])&b.correct.eq(row['correct'])]
        bins=np.minimum(np.floor(f.confidence.to_numpy()*10).astype(int),9);f=f[bins==row['confidence_bin']]
        assert len(f)==row['utterances']
        for m in ['confidence','mu','delta','js']:close(row[m],f[m].mean())
        checked+=1
    for row in read(out/'confidence_shift.csv').to_dict('records'):
        f=pred[pred.dataset.eq(row['dataset'])&pred.seed.eq(row['seed'])&pred.block.eq(row['block'])&pred.method.eq(row['method'])&pred.correct.eq(row['correct'])]
        assert len(f)==row['utterances'];diff=f.oof_confidence-f.confidence
        for m,v in {'original_mean_confidence':f.confidence.mean(),'calibrated_mean_confidence':f.oof_confidence.mean(),'mean_change':diff.mean(),'raised_fraction':(diff>1e-12).mean(),'lowered_fraction':(diff< -1e-12).mean()}.items():close(row[m],v)
        checked+=1
    for row in cm.to_dict('records'):
        f=pred[pred.dataset.eq(row['dataset'])&pred.seed.eq(row['seed'])&pred.block.eq(row['block'])&pred.method.eq(row['method'])]
        q=f.oof_confidence;change=q-f.confidence
        for m,v in {'correct_mean_confidence':q[f.correct.eq(1)].mean(),'wrong_mean_confidence':q[f.correct.eq(0)].mean(),'raised_fraction':(change>1e-12).mean(),'lowered_fraction':(change< -1e-12).mean()}.items():close(row[m],v)
        checked+=1
    for row in read(out/'aggregate_metrics.csv').to_dict('records'):
        f=cm[cm.dataset.eq(row['dataset'])&cm.block.eq(0)&cm.method.eq(row['method'])].set_index('seed')
        e1=cm[cm.dataset.eq(row['dataset'])&cm.block.eq(0)&cm.method.eq('E1')].set_index('seed')
        assert len(f)==row['seeds']==(1 if preflight else 5)
        for m in METRICS+['correctness_auc','mean_confidence']:
            close(row[m+'_mean'],f[m].mean());nullable(row[m+'_sample_std'],f[m].std(ddof=1))
            diff=f[m]-e1[m];close(row[m+'_difference_vs_E1'],diff.mean())
            close(row[m+'_improving_seeds'],int((diff<0).sum()) if m in METRICS else int((diff>0).sum()),0)
        close(row['all_three_improving_seeds'],int(((f[METRICS]-e1[METRICS])<0).all(axis=1).sum()),0);checked+=1
    for row in read(out/'random_block_comparison.csv').to_dict('records'):
        f=cm[cm.dataset.eq(row['dataset'])&cm.seed.eq(1475)&cm.block.eq(row['block'])].set_index('method')
        for m in METRICS:
            close(row[m],f.loc[row['method'],m]);close(row[m+'_difference_vs_E1'],f.loc[row['method'],m]-f.loc['E1',m])
        checked+=1
    counts=read(out/'behavior_selection_counts.csv')
    selected=pred[pred.method.isin(FEATURES)].drop_duplicates(['dataset','seed','block','fold','method'])
    for row in counts.to_dict('records'):
        f=selected[selected.dataset.eq(row['dataset'])&selected.seed.eq(row['seed'])&selected.block.eq(row['block'])&selected.method.eq(row['method'])&selected.selected_penalty.eq(str(row['selected_penalty']))]
        assert len(f)==row['outer_folds'];checked+=1
    assert counts.outer_folds.sum()==len(selected)
    dialogue=read(out/'per_dialogue_results.csv')
    for row in dialogue.to_dict('records'):
        f=pred[pred.dataset.eq(row['dataset'])&pred.seed.eq(row['seed'])&pred.block.eq(row['block'])&pred.method.eq(row['method'])&pred.dialogue_id.eq(row['dialogue_id'])]
        assert len(f)==row['utterances']
        for m,v in metrics(f.oof_confidence,f.correct).items():close(row[m],v)
        checked+=1
    return {'diagnostic_aggregate_confidence_shift_dialogue_rows_recomputed':checked,
        'sample_std_ddof1_and_paired_improvement_counts':True,'blocks_analyzed_separately':True}


def bootstrap_audit(root,config,preflight=False):
    out=root/('historical_preflight' if preflight else 'results')
    pred=read(out/'oof_predictions.csv');saved=read(out/'bootstrap_results.csv');checked=literal=0
    for ds in CANDIDATES:
        registry=np.load(Path(config['source_five_seed'])/f'baseline/bootstrap/{ds}_dialogue_draws.npz')
        ids=registry['dialogue_ids'].astype(str);draws=registry['draws'];assert draws.shape==(2000,COUNTS[ds][1])
        for block,part in pred[pred.dataset.eq(ds)].groupby('block'):
            distributions={};points={};seeds=sorted(part.seed.unique())
            for (seed,method),f in part.groupby(['seed','method']):
                mean,idx,mult,den=cluster_statistics(f,ids,draws);q=f.oof_confidence.to_numpy();y=f.correct.to_numpy()
                bi=np.minimum(np.floor(q*15).astype(int),14);residual=np.zeros((len(ids),15));np.add.at(residual,(idx,bi),q-y)
                qc=np.clip(q,1e-15,1-1e-15)
                distributions[seed,method]={'ece15':np.abs(mult@residual).sum(1)/den,'binary_brier':mean((q-y)**2),'binary_nll':mean(-y*np.log(qc)-(1-y)*np.log1p(-qc))}
                points[seed,method]=metrics(q,y)
                ii=np.concatenate([np.flatnonzero(idx==g) for g in draws[0]])
                direct=metrics(q[ii],y[ii])
                for m in METRICS:close(distributions[seed,method][m][0],direct[m]);literal+=1
            subset=saved[saved.dataset.eq(ds)&saved.block.eq(block)]
            expected=(len(seeds)+(1 if len(seeds)>1 else 0))*27
            assert len(subset)==expected,(ds,block,len(subset),expected)
            for row in subset.to_dict('records'):
                candidate,reference,m=row['candidate'],row['reference'],row['metric']
                assert row['resamples']==2000 and not flag(row['refit']) and flag(row['exploratory'])
                assert row['seeds']==len(seeds)
                if row['scope']=='per_seed':
                    seed=int(row['seed']);d=distributions[seed,candidate][m]-distributions[seed,reference][m]
                    point=points[seed,candidate][m]-points[seed,reference][m];assert row['unit']=='dialogue'
                else:
                    assert row['scope']=='mean_across_paired_seeds' and str(row['seed'])=='all' and len(seeds)>1
                    d=np.mean([distributions[s,candidate][m]-distributions[s,reference][m] for s in seeds],axis=0)
                    point=np.mean([points[s,candidate][m]-points[s,reference][m] for s in seeds]);assert row['unit']=='shared_dialogue_draw_across_seeds'
                close(row['mean_difference'],point);lo,hi=np.quantile(d,[.025,.975]);close(row['ci_low'],lo);close(row['ci_high'],hi)
                close(row['bootstrap_fraction_improving'],np.mean(d<0));checked+=1
    assert checked==len(saved)
    return {'bootstrap_rows_recomputed':checked,'literal_first_draw_checks':literal,'shared_2000_historical_dialogue_draws':True,
        'seeds_not_independent_bootstrap_units':True,'no_refit_and_no_RNG_block_pooling':True}




def concentration_audit(root,preflight=False):
    from scipy.stats import spearmanr
    out=root/('historical_preflight' if preflight else 'results')
    b=read(out/'behavior_features.csv');pred=read(out/'oof_predictions.csv')
    stability=read(out/'cross_block_stability.csv');influence=read(out/'dialogue_influence.csv');summary=read(out/'dialogue_concentration_summary.csv')
    assert len(stability)==(0 if preflight else 18)
    for row in stability.to_dict('records'):
        subset=b[b.dataset.eq(row['dataset'])&b.seed.eq(1475)]
        left=subset[subset.block.eq(row['left_block'])].set_index(KEY)
        right=subset[subset.block.eq(row['right_block'])].set_index(KEY)
        assert set(left.index)==set(right.index) and len(left)==row['utterances']
        right=right.loc[left.index];m=row['measure'];a=left[m];z=right[m]
        expected=spearmanr(a,z).statistic if a.nunique()>1 and z.nunique()>1 else np.nan
        nullable(row['spearman'],expected)
        for side,values in [('left',a),('right',z)]:
            close(row[side+'_zero_fraction'],values.eq(0).mean());close(row[side+'_nonzero_fraction'],values.ne(0).mean())
    # Generate the complete expected fixed-prediction influence registry independently.
    expected=[]
    for (ds,seed,block),f in pred.groupby(['dataset','seed','block']):
        base=f[f.method.eq('E1')].set_index(KEY)
        for method in FEATURES:
            part=f[f.method.eq(method)].set_index(KEY).loc[base.index]
            q=part.oof_confidence.to_numpy();e=base.oof_confidence.to_numpy();y=part.correct.to_numpy()
            dialogues=part.index.get_level_values('dialogue_id').to_numpy()
            full={m:metrics(q,y)[m]-metrics(e,y)[m] for m in METRICS}
            for dialogue in np.unique(dialogues):
                keep=dialogues!=dialogue
                omitted={m:metrics(q[keep],y[keep])[m]-metrics(e[keep],y[keep])[m] for m in METRICS}
                local={m:metrics(q[~keep],y[~keep])[m]-metrics(e[~keep],y[~keep])[m] for m in METRICS}
                for metric in METRICS:expected.append(dict(dataset=ds,seed=str(seed),block=int(block),scope='per_seed',method=method,dialogue_id=str(dialogue),metric=metric,dialogue_utterances=int((~keep).sum()),full_difference=full[metric],omitted_difference=omitted[metric],dialogue_difference=local[metric]))
    frame=pd.DataFrame(expected);pair=[]
    for key,f in frame[frame.block.eq(0)].groupby(['dataset','block','method','dialogue_id','metric']):
        if f.seed.nunique()>1:
            assert f.seed.nunique()==5
            r=dict(zip(['dataset','block','method','dialogue_id','metric'],key));r.update(seed='all',scope='mean_across_paired_seeds',dialogue_utterances=int(f.dialogue_utterances.iloc[0]))
            for col in ['full_difference','omitted_difference','dialogue_difference']:r[col]=f[col].mean()
            pair.append(r)
    if pair:frame=pd.concat([frame,pd.DataFrame(pair)],ignore_index=True)
    columns=['dataset','block','seed','scope','method','dialogue_id','metric'];influence['seed']=influence.seed.astype(str)
    assert not influence.duplicated(columns).any() and len(influence)==len(frame)
    joined=frame.merge(influence,on=columns,validate='one_to_one',suffixes=('','_saved'))
    assert len(joined)==len(frame) and set(joined.reference)=={'E1'} and not joined.refit.map(flag).any()
    for col in ['full_difference','omitted_difference','dialogue_difference','dialogue_utterances']:close(joined[col],joined[col+'_saved'])
    keys=['dataset','block','seed','scope','method','metric'];summary['seed']=summary.seed.astype(str)
    expected_groups=list(frame.groupby(keys));assert len(expected_groups)==len(summary)
    for key,f in expected_groups:
        selected=summary.copy()
        for col,value in zip(keys,key):selected=selected[selected[col].eq(value)]
        assert len(selected)==1;row=selected.iloc[0];full=f.full_difference.iloc[0];om=f.omitted_difference
        assert not flag(row.refit)
        values={'full_difference':full,'dialogues':len(f),'dialogues_improving':int(f.dialogue_difference.lt(0).sum()),
            'omission_difference_min':om.min(),'omission_difference_max':om.max(),'omissions_with_improvement':int(om.lt(0).sum()),
            'omissions_reversing_full_direction':int((om.gt(0) if full<0 else om.lt(0)).sum())}
        for col,value in values.items():close(row[col],value)
    return {'cross_block_rank_stability_rows_recomputed':len(stability),'fixed_OOF_dialogue_influence_rows_recomputed':len(influence),
        'concentration_summary_rows_recomputed':len(summary),'no_refit_or_RNG_block_pooling':True,'mean_across_seeds_uses_same_dialogue_omission':True}


if __name__=='__main__':main()
