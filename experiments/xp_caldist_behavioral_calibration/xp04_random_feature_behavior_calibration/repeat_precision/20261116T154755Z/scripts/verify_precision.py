"""Independent CPU audit of the preregistered 10 versus 30 precision analysis.

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



def immutable(root,config):
    registry=json.loads((root/'provenance/source_hashes_before.json').read_text())
    for path,digest in registry.items(): assert sha(path)==digest,path
    assert sha(root/'scripts/mainline_core.py')==sha(Path(config['source_mainline'])/'scripts/mainline_core.py')
    assert config['seed']==1475 and config['strength']==.05 and config['corruption']=='gaussian'
    assert config['conditions']=={'K10_block0':list(range(10)),'K10_block1':list(range(10,20)),'K10_block2':list(range(20,30)),'K30_pooled':list(range(30))}
    assert config['methods']==['E0','E1']+list(FEATURES) and config['penalties']==['off',1.,.1,.01]
    return {'immutable_historical_hashes_checked':len(registry),'frozen_calibrator_sha256':sha(root/'scripts/mainline_core.py'),'single_model_seed':1475,'no_new_inference_required':True}


def literal_response(p,q):
    out=behavior(p,q)
    n=len(q);half=n//2
    for name,part in [('left',q[:half]),('right',q[half:])]:
        for metric,value in behavior(p,part).items():
            if metric in ['mu','delta','js']:out[metric+'_'+name]=value
    out['confidence_sample_std']=q.max(1).std(ddof=1)
    out['confidence_mean_standard_error']=out['confidence_sample_std']/np.sqrt(n)
    return out


def raw_inputs(root,config):
    records=[];raw_rows=0;source_paths=[]
    historical=read(Path(config['source_candidate'])/'results/behavior_features.csv')
    for ds,modality in CANDIDATES.items():
        original=read(Path(config['source_pilot'])/ds/'original_predictions.csv')
        assert len(original)==COUNTS[ds][0] and original.dialogue_id.nunique()==COUNTS[ds][1]
        assert set(original.seed)=={1475} and not original.duplicated(KEY).any()
        p=probs(original.probabilities,6 if ds=='IEMOCAP' else 7)
        close(original.confidence,p.max(1),0);close(original.prediction,p.argmax(1),0)
        lookup={tuple(row[k] for k in KEY):(p[i],row) for i,row in enumerate(original.to_dict('records'))}
        parts=[]
        for block in range(3):
            path=(Path(config['source_pilot'])/ds/'perturbed_predictions.csv') if block==0 else (Path(config['source_candidate'])/f'runs/{ds}/seed1475/block{block}/perturbed_predictions.csv')
            source_paths.append(str(path));frame=read(path)
            frame=frame[frame.modality.eq(modality)&frame.corruption.eq('gaussian')&frame.strength.eq(.05)].copy()
            assert len(frame)==len(original)*10 and set(frame.seed)=={1475}
            assert not frame.duplicated(KEY+['repeat_id']).any() and set(frame.repeat_id)==set(range(10))
            assert not frame.visual_missing_skipped.map(flag).any()
            frame['global_repeat_id']=frame.repeat_id+block*10;frame['source_block']=block
            if 'rng_repeat_id' in frame:close(frame.rng_repeat_id,frame.global_repeat_id,0)
            q=probs(frame.probabilities,p.shape[1])
            if 'log_probabilities' in frame:
                lp=np.array([json.loads(x) for x in frame.log_probabilities]);close(q,np.exp(lp),1e-14)
            close(frame.prediction,q.argmax(1),0);close(frame.confidence,q.max(1),0)
            for row in frame.to_dict('records'):
                identity=tuple(row[k] for k in KEY);assert identity in lookup
                assert json.loads(row['rng_key'])==[[ds,1475,row['dialogue_id'],int(row['utterance_index']),name,'gaussian',.05,int(row['global_repeat_id'])] for name in NAMES[modality]]
            parts.append(frame);raw_rows+=len(frame)
        raw=pd.concat(parts,ignore_index=True);assert not raw.duplicated(KEY+['global_repeat_id']).any()
        for identity,g in raw.groupby(KEY,sort=False):
            g=g.sort_values('global_repeat_id');assert g.global_repeat_id.tolist()==list(range(30))
            pp,orig=lookup[identity];qq=probs(g.probabilities,len(pp))
            for condition,repeats in config['conditions'].items():
                values=literal_response(pp,qq[repeats])
                if condition!='K30_pooled':
                    h=historical[historical.dataset.eq(ds)&historical.seed.eq(1475)&historical.block.eq(int(condition[-1]))&historical.dialogue_id.eq(identity[2])&historical.utterance_index.eq(identity[3])]
                    assert len(h)==1
                    for col in ['mu','delta','js','flip_rate','confidence_change_mean','confidence_change_std','mu_left','mu_right','delta_left','delta_right','js_left','js_right']:close(h.iloc[0][col],values[col])
                records.append(dict(zip(KEY,identity),condition=condition,repeat_count=len(repeats),modality=modality,confidence=orig['confidence'],label=orig['label'],prediction=orig['prediction'],correct=int(orig['label']==orig['prediction']),**values))
    out=pd.DataFrame(records);assert len(out)==6544 and raw_rows==49080
    assert not out.duplicated(KEY+['condition']).any()
    pooled=out[out.condition.eq('K30_pooled')].set_index(KEY)
    blocks=out[out.condition.ne('K30_pooled')].groupby(KEY).delta.mean()
    cancellation=float(np.max(abs(pooled.delta-blocks.loc[pooled.index])))
    return out,{'selected_raw_rows':raw_rows,'one_utterance_per_condition_records':len(out),'historical_K10_behavior_all_blocks_recomputed':4908,'K30_delta_max_difference_from_wrong_mean_block_delta':cancellation,'global_rng_repeats_exact_0_to_29':True,'selected_source_files':source_paths}

def saved_fit_audit(root,config,expected,preflight=False):
    out=root/('preflight' if preflight else 'results');frame=read(out/'feature_matrix.csv');saved=read(out/'behavior_features.csv');pred=read(out/'oof_predictions.csv');cm=read(out/'calibration_metrics.csv');inner=read(out/'inner_cv_results.csv')
    keys=KEY+['condition'];assert len(frame)==len(saved)==len(expected)==(4908 if preflight else 6544) and len(pred)==len(frame)*5 and len(cm)==(30 if preflight else 40)
    assert not frame.duplicated(keys).any() and not saved.duplicated(keys).any() and not pred.duplicated(keys+['method']).any()
    for table in [frame,saved]:
        m=expected.merge(table,on=keys,validate='one_to_one',suffixes=('','_saved'));assert len(m)==len(expected)
        for col in ['mu','delta','js','flip_rate','confidence_change_mean','confidence_change_std','mu_left','mu_right','delta_left','delta_right','js_left','js_right','confidence_sample_std','confidence_mean_standard_error','confidence','label','prediction','correct','repeat_count']:close(m[col],m[col+'_saved'])
    close(frame.gaussian_mu,frame.mu,0);close(frame.gaussian_delta,frame.delta,0)
    history=json.loads((Path(config['source_five_seed'])/'baseline/fold_parameters.json').read_text())
    oldpred=read(Path(config['source_candidate'])/'results/oof_predictions.csv');oldmetric=read(Path(config['source_candidate'])/'results/calibration_metrics.csv')
    counts=dict(outer_folds=0,inner_predictions_replayed=0,nested_selections=0,oof_predictions_replayed=0,historical_K10_parity_predictions=0)
    for (ds,condition),f in frame.groupby(['dataset','condition'],sort=False):
        f=f.reset_index(drop=True);n=len(f);assert n==COUNTS[ds][0]
        folds=json.loads((out/f'fold_parameters/{ds}_{condition}.json').read_text());prior={r['fold']:r for r in history if r['dataset']==ds and r['seed']==1475};assert len(folds)==len(prior)
        cover=np.zeros(n,int)
        for r in folds:
            counts['outer_folds']+=1;tr=np.array(r['train_indices'],int);he=np.array(r['held_indices'],int);cover[he]+=1
            assert sorted(np.r_[tr,he])==list(range(n)) and set(tr).isdisjoint(he)
            assert r['train_dialogues']==prior[r['fold']]['train_dialogues'] and r['held_dialogues']==prior[r['fold']]['held_dialogues']
            assert tr.tolist()==np.flatnonzero(f.dialogue_id.isin(r['train_dialogues'])).tolist() and he.tolist()==np.flatnonzero(f.dialogue_id.isin(r['held_dialogues'])).tolist()
            actual={'E1':replay_fit(f,r['E1'],tr)(f.iloc[he])}
            splits=list(GroupKFold(5).split(tr,groups=f.iloc[tr].dialogue_id));assert len(r['inner_folds'])==5
            for row,(ii,jj) in zip(r['inner_folds'],splits):
                assert row['train_indices']==tr[ii].tolist() and row['held_indices']==tr[jj].tolist()
                assert set(row['train_dialogues']).isdisjoint(row['held_dialogues']);replay_fit(f,row['E1'],tr[ii])
            for method,item in r['methods'].items():
                assert item['features']==FEATURES[method];best='off';minimum=np.inf
                for penalty in ['off',1.,.1,.01]:
                    candidates=item['inner_candidate_fits'][str(penalty)];assert len(candidates)==5;errors=[]
                    for candidate,inn in zip(candidates,r['inner_folds']):
                        fit=candidate['fit'];assert fit['penalty']==penalty and fit['features']==([] if penalty=='off' else FEATURES[method])
                        assert candidate['held_indices']==inn['held_indices'] and candidate['held_dialogues']==inn['held_dialogues']
                        if penalty=='off':assert fit==inn['E1']
                        held=f.iloc[inn['held_indices']];q=replay_fit(f,fit,inn['train_indices'])(held)['oof_confidence'];err=(q-held.correct.to_numpy())**2
                        close(candidate['held_brier'],err.mean());errors.extend(err);counts['inner_predictions_replayed']+=len(err)
                    loss=np.mean(errors);close(loss,item['candidate_brier'][str(penalty)])
                    z=inner[inner.dataset.eq(ds)&inner.condition.eq(condition)&inner.fold.eq(r['fold'])&inner.method.eq(method)&inner.penalty.eq(str(penalty))];assert len(z)==1;close(z.iloc[0].pooled_inner_brier,loss)
                    if loss<minimum-1e-12:best,minimum=penalty,loss
                assert best==item['selected_penalty']==item['outer_fit']['penalty'];counts['nested_selections']+=1
                if best=='off':assert item['outer_fit']==r['E1']
                actual[method]=replay_fit(f,item['outer_fit'],tr)(f.iloc[he])
            for method in ['E0','E1']+list(FEATURES):
                z=pred[pred.dataset.eq(ds)&pred.condition.eq(condition)&pred.method.eq(method)&pred.fold.eq(r['fold'])].set_index(KEY);held=f.iloc[he].set_index(KEY)
                assert set(z.index)==set(held.index);z=z.loc[held.index]
                for col in ['label','prediction','correct','confidence']:close(z[col],held[col],0)
                if method=='E0':close(z.oof_confidence,held.confidence,0)
                else:
                    for col,value in actual[method].items():close(z[col],value)
                counts['oof_predictions_replayed']+=len(z)
        assert np.all(cover==1)
        if condition!='K30_pooled':
            a=pred[pred.dataset.eq(ds)&pred.condition.eq(condition)];b=oldpred[oldpred.dataset.eq(ds)&oldpred.seed.eq(1475)&oldpred.block.eq(int(condition[-1]))]
            m=a.merge(b,on=KEY+['method'],validate='one_to_one',suffixes=('','_old'));assert len(m)==n*5;close(m.oof_confidence,m.oof_confidence_old,1e-8);close(m.fold,m.fold_old,0);counts['historical_K10_parity_predictions']+=len(m)
        for method,g in pred[pred.dataset.eq(ds)&pred.condition.eq(condition)].groupby('method'):
            from sklearn.metrics import roc_auc_score
            z=cm[cm.dataset.eq(ds)&cm.condition.eq(condition)&cm.method.eq(method)];assert len(z)==1;r=z.iloc[0];q=g.oof_confidence
            expected_metrics=metrics(q,g.correct);expected_metrics.update(accuracy=np.mean(g.prediction==g.label),weighted_f1=f1_score(g.label,g.prediction,average='weighted'),macro_f1=f1_score(g.label,g.prediction,average='macro'),correctness_auc=roc_auc_score(g.correct,q),mean_confidence=q.mean(),correct_mean_confidence=q[g.correct.eq(1)].mean(),wrong_mean_confidence=q[g.correct.eq(0)].mean(),raised_fraction=(q>g.confidence+1e-12).mean(),lowered_fraction=(q<g.confidence-1e-12).mean())
            for col,value in expected_metrics.items():close(r[col],value)
            if condition!='K30_pooled':
                old=oldmetric[oldmetric.dataset.eq(ds)&oldmetric.seed.eq(1475)&oldmetric.block.eq(int(condition[-1]))&oldmetric.method.eq(method)].iloc[0]
                for col in METRICS:close(r[col],old[col],1e-8)
    return counts


def nullable(actual,expected):
    if np.isfinite(expected):close(actual,expected)
    else:assert pd.isna(actual)


def diagnostics(root):
    from scipy.stats import spearmanr
    out=root/'results';f=read(out/'feature_matrix.csv');count=0
    for row in read(out/'measurement_summary.csv').to_dict('records'):
        a=f[f.dataset.eq(row['dataset'])&f.condition.eq(row['condition'])];assert len(a)==row['utterances'];close(row['any_flip_fraction'],a.flip_rate.gt(0).mean());close(row['mu_nonzero_count'],a.mu.ne(0).sum(),0);close(row['mu_zero_fraction'],a.mu.eq(0).mean())
        for col in ['mu','delta','js','confidence_mean_standard_error','confidence_sample_std']:
            for name,value in [('mean',a[col].mean()),('median',a[col].median()),('p90',a[col].quantile(.9))]:close(row[col+'_'+name],value)
        count+=1
    for row in read(out/'split_half_stability.csv').to_dict('records'):
        a=f[f.dataset.eq(row['dataset'])&f.condition.eq(row['condition'])];x=a[row['measure']+'_left'];y=a[row['measure']+'_right']
        nullable(row['spearman'],spearmanr(x,y).statistic if x.nunique()>1 and y.nunique()>1 else np.nan)
        for side,values in [('left',x),('right',y)]:
            close(row[side+'_nonzero_count'],values.ne(0).sum(),0);close(row[side+'_zero_fraction'],values.eq(0).mean());close(row[side+'_tied_observation_fraction'],values.duplicated(keep=False).mean());close(row[side+'_unique_values'],values.nunique(),0)
        count+=1
    pred=read(out/'oof_predictions.csv')
    for row in read(out/'confidence_shift.csv').to_dict('records'):
        z=pred[pred.dataset.eq(row['dataset'])&pred.condition.eq(row['condition'])&pred.method.eq(row['method'])&pred.correct.eq(row['correct'])];change=z.oof_confidence-z.confidence
        for col,value in dict(original_mean_confidence=z.confidence.mean(),calibrated_mean_confidence=z.oof_confidence.mean(),mean_change=change.mean(),raised_fraction=(change>1e-12).mean(),lowered_fraction=(change< -1e-12).mean()).items():close(row[col],value)
        count+=1
    return {'measurement_and_stability_shift_rows_recomputed':count,'sample_std_ddof1_and_standard_error_of_mean_not_delta':True,'5_vs5_and15_vs15_fixed_halves':True,'sparse_mu_zero_and_ties_reported':True}


def intervals(root,config):
    pred=read(root/'results/oof_predictions.csv');table=read(root/'results/bootstrap_results.csv');influence=read(root/'results/dialogue_influence.csv');summary=read(root/'results/dialogue_concentration_summary.csv');count=literal=infcount=sumcount=0
    assert len(table)==234 and len(influence)==2646 and len(summary)==42
    expected_pairs=set()
    for condition in config['conditions']:
        for method in FEATURES:
            for reference in ['E0','E1']:expected_pairs.add((condition,method,condition,reference))
        expected_pairs.add((condition,'E1',condition,'E0'))
        for reference in ['M_gaussian_mu','M_gaussian_delta']:expected_pairs.add((condition,'M_gaussian',condition,reference))
    for block in range(3):expected_pairs.add(('K30_pooled','M_gaussian',f'K10_block{block}','M_gaussian'))
    for ds in CANDIDATES:
        z=np.load(Path(config['source_five_seed'])/f'baseline/bootstrap/{ds}_dialogue_draws.npz');ids=z['dialogue_ids'].astype(str);draws=z['draws'];assert draws.shape==(2000,COUNTS[ds][1])
        distributions={};points={};frames={}
        for key,f in pred[pred.dataset.eq(ds)].groupby(['condition','method']):
            f=f.set_index(KEY).sort_index().reset_index();frames[key]=f
            mean,idx,mult,den=cluster_statistics(f,ids,draws);q=f.oof_confidence.to_numpy();y=f.correct.to_numpy();binids=np.minimum(np.floor(q*15).astype(int),14);residual=np.zeros((len(ids),15));np.add.at(residual,(idx,binids),q-y);clip=np.clip(q,1e-15,1-1e-15)
            distributions[key]=dict(ece15=np.abs(mult@residual).sum(1)/den,binary_brier=mean((q-y)**2),binary_nll=mean(-y*np.log(clip)-(1-y)*np.log1p(-clip)));points[key]=metrics(q,y)
            direct=np.concatenate([np.flatnonzero(idx==g) for g in draws[0]])
            for metric,value in metrics(q[direct],y[direct]).items():close(distributions[key][metric][0],value);literal+=1
        subset=table[table.dataset.eq(ds)]
        actual_pairs=set(map(tuple,subset[['candidate_condition','candidate','reference_condition','reference']].to_numpy()))
        assert actual_pairs==expected_pairs and not subset.duplicated(['candidate_condition','candidate','reference_condition','reference','metric']).any()
        for row in subset.to_dict('records'):
            a=(row['candidate_condition'],row['candidate']);b=(row['reference_condition'],row['reference']);m=row['metric'];d=distributions[a][m]-distributions[b][m]
            close(row['difference'],points[a][m]-points[b][m]);lo,hi=np.quantile(d,[.025,.975]);close(row['ci_low'],lo);close(row['ci_high'],hi);close(row['bootstrap_fraction_improving'],np.mean(d<0));assert row['resamples']==2000 and not flag(row['refit']) and flag(row['exploratory']) and row['unit']=='dialogue';count+=1
        for row in influence[influence.dataset.eq(ds)].to_dict('records'):
            a=(row['candidate_condition'],row['candidate']);b=(row['reference_condition'],row['reference']);m=row['metric'];fa,fb=frames[a],frames[b];assert fa[KEY].equals(fb[KEY]);keep=fa.dialogue_id.ne(row['dialogue_id']).to_numpy()
            close(row['full_difference'],points[a][m]-points[b][m]);close(row['omitted_difference'],metrics(fa.oof_confidence[keep],fa.correct[keep])[m]-metrics(fb.oof_confidence[keep],fb.correct[keep])[m]);close(row['dialogue_difference'],metrics(fa.oof_confidence[~keep],fa.correct[~keep])[m]-metrics(fb.oof_confidence[~keep],fb.correct[~keep])[m]);close(row['dialogue_utterances'],(~keep).sum(),0);assert not flag(row['refit']);infcount+=1
    groups=['dataset','seed','candidate_condition','candidate','reference_condition','reference','scope','metric']
    for row in summary.to_dict('records'):
        f=influence.copy()
        for col in groups:f=f[f[col].eq(row[col])]
        assert len(f)==row['dialogues'];full=f.full_difference.iloc[0];om=f.omitted_difference
        values=dict(full_difference=full,dialogues_improving=f.dialogue_difference.lt(0).sum(),omission_difference_min=om.min(),omission_difference_max=om.max(),omissions_reversing_direction=(om.gt(0) if full<0 else om.lt(0)).sum())
        for col,value in values.items():close(row[col],value)
        sumcount+=1
    assert count==len(table) and infcount==len(influence)
    return {'paired_historical_2000_dialogue_bootstrap_rows_recomputed':count,'literal_first_draw_checks':literal,'fixed_OOF_dialogue_omission_rows_recomputed':infcount,'concentration_summary_rows':sumcount,'nested_conditions_not_independent_replicates':True,'no_refitting_or_seed_population_inference':True}


def full_audit(root,config,expected):
    v=json.loads((root/'results/analysis_verification.json').read_text());assert v['all_passed'] and v['status']=='analysis_complete'
    for name,digest in v['output_sha256'].items():assert sha(root/'results'/name)==digest,name
    for path,digest in json.loads((root/'results/source_hashes.json').read_text()).items():assert sha(path)==digest,path
    savedraw=read(root/'inputs/perturbed_predictions.csv');assert len(savedraw)==49080 and not savedraw.duplicated(KEY+['global_repeat_id']).any()
    originals=read(root/'inputs/original_predictions.csv');assert len(originals)==1636 and not originals.duplicated(KEY).any()
    # Validate delivery probabilities, RNG metadata and originals directly against frozen selected source files.
    for ds in CANDIDATES:
        part=originals[originals.dataset.eq(ds)];source=read(Path(config['source_pilot'])/ds/'original_predictions.csv');join=part.merge(source,on=KEY,validate='one_to_one',suffixes=('','_source'));assert len(join)==COUNTS[ds][0]
        for col in ['label','prediction','confidence']:close(join[col],join[col+'_source'],0)
        close(probs(join.probabilities,6 if ds=='IEMOCAP' else 7),probs(join.probabilities_source,6 if ds=='IEMOCAP' else 7),0)
    for source,part in savedraw.groupby('source_path'):
        original=read(source);ds=part.dataset.iloc[0];mod=CANDIDATES[ds];original=original[original.modality.eq(mod)&original.corruption.eq('gaussian')&original.strength.eq(.05)];join=part.merge(original,on=KEY+['repeat_id'],validate='one_to_one',suffixes=('','_source'));assert len(join)==len(part)
        assert part.source_sha256.eq(sha(source)).all();close(part.global_repeat_id,part.repeat_id+10*part.source_block,0)
        close(join.prediction,join.prediction_source,0);close(join.confidence,join.confidence_source,0)
        close(probs(join.probabilities,6 if ds=='IEMOCAP' else 7),probs(join.probabilities_source,6 if ds=='IEMOCAP' else 7),0)
        assert all(json.loads(a)==json.loads(b) for a,b in zip(join.rng_key,join.rng_key_source))
    return {'calibration':saved_fit_audit(root,config,expected),'diagnostics':diagnostics(root),'bootstrap_and_influence':intervals(root,config),'delivered_inputs_exactly_reproduce_saved_source':True,'analysis_output_hashes_verified':len(v['output_sha256'])}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--stage',choices=['preflight','full'],required=True)
    args=parser.parse_args();root=args.root.resolve();config=json.loads((root/'preregistration.json').read_text())
    result={'status':'failed','all_passed':False,'stage':args.stage,'checks':{}}
    target=root/'results'/('independent_preflight.json' if args.stage=='preflight' else 'independent_verification.json')
    try:
        result['checks']['source']=immutable(root,config)
        expected,result['checks']['raw']=raw_inputs(root,config)
        if args.stage=='full':result['checks'].update(full_audit(root,config,expected))
        elif (root/'preflight/analysis_verification.json').exists():
            result['checks']['historical_K10_fit_reproduction']=saved_fit_audit(root,config,expected[expected.condition.ne('K30_pooled')],preflight=True)
        result.update(status='passed',all_passed=True)
        if args.stage=='preflight':result['scope']='Input/formula/history preflight only. New K30 fits and scientific conclusions not yet verified.'
    except Exception as exc:
        result.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    if not result['all_passed']:raise SystemExit(1)

if __name__=='__main__':main()
