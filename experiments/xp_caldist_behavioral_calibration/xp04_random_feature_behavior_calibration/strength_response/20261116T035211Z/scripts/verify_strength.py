"""Independent CPU audit of the preregistered strength experiment.

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
MODALITIES=['text','audio','visual','joint']
NAMES={'text':['text0','text1','text2','text3'],'audio':['audio'],'visual':['visual'],
       'joint':['text0','text1','text2','text3','audio','visual']}
FEATURES={'M_joint':['gaussian_mu','gaussian_delta','dropout_mu','dropout_delta'],
          'M_gaussian':['gaussian_mu','gaussian_delta'],'M_dropout':['dropout_mu','dropout_delta']}
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


def historical_hashes(root):
    value=json.loads((root/'provenance/source_hashes.json').read_text())
    checked=0
    def visit(node):
        nonlocal checked
        if not isinstance(node,dict): return
        for path,digest in node.items():
            if isinstance(digest,dict): visit(digest)
            elif isinstance(digest,str) and len(digest)==64 and Path(path).is_absolute():
                assert sha(path)==digest,path
                checked+=1
    visit(value)
    assert checked>=17,'Historical input/source manifest incomplete'
    return {'historical_sources_unchanged':checked}


def raw_audit(root,config):
    old=Path(config['source_pilot'])
    flags=read(Path(config['source_mainline'])/'inputs/modality_features.csv')
    flags=flags[flags.modality.eq('joint')].set_index(KEY)
    all_features=[]; counts={}; condition_pairs=set(); probability_rows=0
    for ds,(n,groups,missing_n) in COUNTS.items():
        original=read(old/ds/'original_predictions.csv')
        assert len(original)==n and original.dialogue_id.nunique()==groups
        assert not original.duplicated(KEY).any() and set(original.seed)=={1475}
        assert original.label.between(0,5 if ds=='IEMOCAP' else 6).all()
        p=probs(original.probabilities,6 if ds=='IEMOCAP' else 7)
        close(original.confidence,p.max(1)); close(original.prediction,p.argmax(1))
        reference={tuple(row[k] for k in KEY):p[i] for i,row in enumerate(original.to_dict('records'))}
        old_frame=read(old/ds/'perturbed_predictions.csv')
        new_paths=[path for path in (root/'runs').glob(f'{ds}/*/perturbed_predictions.csv')]
        assert len(new_paths)==4,(ds,new_paths)
        new_frame=pd.concat([read(path) for path in new_paths],ignore_index=True)
        for strength,frame in [(.05,old_frame),(.1,new_frame)]:
            assert len(frame)==n*80,(ds,strength,len(frame))
            assert set(frame.strength)=={strength} and set(frame.seed)=={1475}
            assert not frame.duplicated(KEY+['modality','corruption','strength','repeat_id']).any()
            assert set(frame.modality)==set(MODALITIES) and set(frame.corruption)=={'gaussian','dropout'}
            q=probs(frame.probabilities,p.shape[1]); close(frame.prediction,q.argmax(1));close(frame.confidence,q.max(1))
            if 'log_probabilities' in frame:
                logp=np.asarray([json.loads(v) for v in frame.log_probabilities],float)
                assert logp.shape==q.shape and np.isfinite(logp).all()
                assert np.allclose(np.exp(logp),q,atol=1e-6,rtol=1e-5)
                close(logp.argmax(1),q.argmax(1),0)
            probability_rows+=len(frame)
            frame=frame.copy(); frame['_position']=np.arange(len(frame))
            for (modality,kind),condition in frame.groupby(['modality','corruption'],sort=False):
                assert len(condition)==n*10
                condition_pairs.add((ds,modality,kind))
                skip=condition.visual_missing_skipped.map(flag)
                expected_missing=missing_n if modality in ['visual','joint'] else 0
                assert int(skip.sum())==expected_missing*10
                for identity,repeats in condition.groupby(KEY,sort=False):
                    assert set(repeats.repeat_id)==set(range(10)) and len(repeats)==10
                    repeats=repeats.sort_values('repeat_id'); qq=q[repeats._position]
                    assert identity in reference,identity
                    pp=reference[identity]; result=behavior(pp,qq)
                    for suffix,half in [('left',qq[:5]),('right',qq[5:])]:
                        half_result=behavior(pp,half)
                        for metric in ['mu','delta','js','flip_rate']: result[metric+'_'+suffix]=half_result[metric]
                    is_missing=bool(repeats.visual_missing_skipped.map(flag).iloc[0])
                    assert repeats.visual_missing_skipped.map(flag).nunique()==1
                    expected_skip=flag(flags.loc[identity,'visual_missing']) and modality in ['visual','joint']
                    assert is_missing==expected_skip
                    if modality=='visual' and is_missing:
                        close(qq,np.repeat(pp[None,:],10,axis=0),0)
                        close([result[x] for x in ['mu','delta','js','flip_rate']],[0,0,0,0],1e-14)
                    for row in repeats.to_dict('records'):
                        key=json.loads(row['rng_key'])
                        names=[name for name in NAMES[modality] if not(name=='visual' and is_missing)]
                        assert key==[[ds,1475,identity[2],int(identity[3]),name,kind,strength,int(row['repeat_id'])] for name in names]
                        if 'rng_seeds' in row:
                            expected=[int.from_bytes(hashlib.sha256(json.dumps(k,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).digest()[:16],'big') for k in key]
                            assert json.loads(row['rng_seeds'])==expected
                    base=dict(zip(KEY,identity));base.update(modality=modality,corruption=kind,strength=strength,
                        original_confidence=float(pp.max()),visual_missing_skipped=is_missing,**result)
                    all_features.append(base)
            counts[f'{ds}_{strength:g}']=len(frame)
    assert len(condition_pairs)==16 and sum(v for k,v in counts.items() if k.endswith('_0.05'))==130880
    assert sum(v for k,v in counts.items() if k.endswith('_0.1'))==130880
    features=pd.DataFrame(all_features)
    source=read(old/'behavior_features.csv')
    pair=features[features.strength.eq(.05)].merge(source,on=KEY+['modality','corruption','strength'],validate='one_to_one',suffixes=('','_old'))
    assert len(pair)==13088
    for col in ['mu','delta','js','flip_rate']: close(pair[col],pair[col+'_old'])
    return features,{'raw_probability_rows_recomputed':probability_rows,'rows_by_dataset_strength':counts,
        'condition_pairs':len(condition_pairs),'historical_features_reproduced':len(pair),
        'mu_is_confidence_weighted_flip_not_flip_rate':True,'zero_visual_rows_retained':{'IEMOCAP':1,'MELD':95}}


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


def inference_audit(root,smoke=False):
    completed=[]
    for ds,(n,groups,missing_n) in COUNTS.items():
        prior=read(Path(json.loads((root/'preregistration.json').read_text())['source_pilot'])/ds/'original_predictions.csv')
        for modality in MODALITIES:
            path=root/'runs'/ds/modality
            if smoke: path/='smoke'
            if smoke and not (path/'complete.json').exists(): continue
            result=json.loads((path/'complete.json').read_text())
            assert result['status']=='completed' and result['smoke']==smoke
            assert result['dataset']==ds and result['modality']==modality and result['seed']==1475 and result['strength']==.1
            assert result['utterances']==(1 if smoke else n) and result['perturbed_rows']==(20 if smoke else n*20)
            assert result['weights_unchanged'] and result['batch_inputs_restored'] and result['identical_input_parity']['passed']
            replay=json.loads((path/'replay_diagnostics.json').read_text())
            assert replay['passed'] and replay['inputs_exact_equal'] and replay['weights_exact_equal'] and replay['all_modules_eval']
            assert replay['input_sha256_before']==replay['input_sha256_after']
            assert replay['parameter_dtypes']==['torch.float32'] and replay['atol']==1e-6 and replay['rtol']==1e-5
            assert len(replay['comparisons_to_initial_cached_output'])==8
            for item in replay['comparisons_to_initial_cached_output']:
                assert item['argmax_equal'] and item['probability_tolerance_passed'] and item['log_probability_tolerance_passed']
                assert 0<=item['probability_max_abs_error']<=1.1e-5
            assert replay['bitwise_output_equal']==all(item['exact_equal'] for item in replay['comparisons_to_initial_cached_output'])
            assert result['graph_hook_checks']>0 and result['target_mask_checks']>0 and result['cached_uncached_parity_checks']>0
            assert result['parity']['passed']
            assert all(item['equal'] for item in result['parity']['checks'].values())
            assert result['actual_imported_source_sha256'] and result['data_sha256'] and result['actual_opened_data_sha256']
            before=json.loads((path/'source_hashes_before.json').read_text())
            after=json.loads((path/'source_hashes_after.json').read_text())
            assert before==after==result['source_sha256']
            for source,digest in before.items(): assert sha(source)==digest,source
            for source,digest in result['actual_imported_source_sha256'].items(): assert before[source]==digest
            for source,digest in result['data_sha256'].items(): assert before[source]==digest
            for source,digest in result['actual_opened_data_sha256'].items(): assert before[source]==digest
            for output,digest in result['output_sha256'].items(): assert sha(path/output)==digest,output
            config=json.loads((path/'effective_config.json').read_text())
            assert config['split']=='validation' and config['batch_size']==16 and config['rnn_backend']=='aten' and config['precision']=='float32'
            split=config['checkpoint_config']['provenance']['split']
            assert set(split['validation_ids']).isdisjoint(split['train_ids'])
            assert set(split['validation_ids']).isdisjoint(split['test_ids'])
            original=read(path/'original_predictions.csv')
            pair=original.merge(prior,on=KEY,validate='one_to_one',suffixes=('','_old'))
            assert len(pair)==n and original.dialogue_id.nunique()==groups
            for col in ['label','prediction','confidence']: close(pair[col],pair[col+'_old'],0)
            close(probs(pair.probabilities,6 if ds=='IEMOCAP' else 7),probs(pair.probabilities_old,6 if ds=='IEMOCAP' else 7),0)
            fresh=np.load(path/'fresh_validation_predictions.npz')
            assert list(fresh['dialogue_ids'].astype(str))==original.dialogue_id.tolist()
            close(fresh['utterance_indices'],original.utterance_index,0);close(fresh['labels'],original.label,0)
            close(fresh['probabilities'],probs(original.probabilities,6 if ds=='IEMOCAP' else 7),1e-6)
            audits=read(path/'perturbation_audit.csv')
            assert len(audits)>0 and not audits.duplicated(KEY+['modality','corruption','repeat_id','feature_name']).any()
            assert audits.stdzero_unchanged.map(flag).all()
            for row in audits.to_dict('records'):
                key=json.loads(row['rng_key']);canonical=json.dumps(key,ensure_ascii=False,separators=(',',':'),allow_nan=False)
                assert str(row['rng_seed'])==str(int.from_bytes(hashlib.sha256(canonical.encode()).digest()[:16],'big'))
                assert key==[ds,1475,row['dialogue_id'],int(row['utterance_index']),row['feature_name'],row['corruption'],.1,int(row['repeat_id'])]
                assert row['requested_count']==(round(.1*row['eligible_count']) if row['corruption']=='dropout' else row['eligible_count'])
                assert 0<=row['actual_changed_count']<=row['eligible_count']
                for h in ['vector_sha256','original_vector_sha256']: assert len(row[h])==64
            if not smoke:
                assert result['visual_missing_retained']==missing_n
                assert result['target_forwards']==(n-(missing_n if modality=='visual' else 0))*20
                assert len(audits)==(n*len(NAMES[modality])-(missing_n if modality in ['visual','joint'] else 0))*20
            completed.append(f'{ds}/{modality}'+('/smoke' if smoke else ''))
    if smoke: assert completed,'No complete smoke evidence'
    else: assert len(completed)==8
    return {'independently_checked_runs':completed,'full_weight_graph_mask_restore_and_cache_parity':True,
            'raw_input_only_perturbation_and_validation_split':True}


def fit_audit(root,config,raw):
    out=root/'results'
    frame=read(out/'feature_matrix.csv')
    pred=read(out/'oof_predictions.csv');saved_metrics=read(out/'calibration_metrics.csv')
    assert len(frame)==13088 and not frame.duplicated(KEY+['modality','strength']).any()
    assert len(pred)==65440 and not pred.duplicated(KEY+['modality','strength','method']).any()
    assert set(pred.method)=={'E0','E1','M_joint','M_gaussian','M_dropout'}
    for corruption in ['gaussian','dropout']:
        match=frame.merge(raw[raw.corruption.eq(corruption)],on=KEY+['modality','strength'],validate='one_to_one',suffixes=('','_raw'))
        assert len(match)==len(frame)
        for variable in ['mu','delta']: close(match[f'{corruption}_{variable}'],match[variable])
    oldfold=json.loads((Path(config['source_five_seed'])/'baseline/fold_parameters.json').read_text())
    inner_saved=read(out/'inner_cv_results.csv')
    outer=inner=selected=0;seen_folds=set()
    for (ds,strength,modality),g in frame.groupby(['dataset','strength','modality'],sort=False):
        g=g.reset_index(drop=True);n,ng,_=COUNTS[ds]
        assert len(g)==n and g.dialogue_id.nunique()==ng
        files=list((out/'fold_parameters').glob(f'{ds}*{modality}*.json'))
        records=[]
        for path in files:
            candidate=json.loads(path.read_text())
            if isinstance(candidate,dict): candidate=candidate.get('folds',candidate.get('records',[candidate]))
            records.extend(r for r in candidate if float(r.get('strength',strength))==strength)
        assert len(records)==(12 if ds=='IEMOCAP' else 5),(ds,strength,modality,len(records))
        historical={r['fold']:r for r in oldfold if r['dataset']==ds and r['seed']==1475}
        seen=np.zeros(n,int)
        pp=pred[pred.dataset.eq(ds)&pred.strength.eq(strength)&pred.modality.eq(modality)]
        for record in records:
            fold=record['fold'];seen_folds.add((ds,fold));tr=np.asarray(record['train_indices'],int);he=np.asarray(record['held_indices'],int)
            definition=historical[fold]
            assert record['train_dialogues']==definition['train_dialogues'] and record['held_dialogues']==definition['held_dialogues']
            assert set(record['train_dialogues']).isdisjoint(record['held_dialogues'])
            assert set(tr).isdisjoint(he) and sorted(np.r_[tr,he])==list(range(n))
            assert np.array_equal(tr,np.flatnonzero(g.dialogue_id.isin(record['train_dialogues'])))
            assert np.array_equal(he,np.flatnonzero(g.dialogue_id.isin(record['held_dialogues'])))
            seen[he]+=1
            actual={'E1':replay_fit(g,record['E1'],tr)(g.iloc[he])}
            splits=list(GroupKFold(5).split(tr,groups=g.iloc[tr].dialogue_id))
            assert len(record['inner_folds'])==5
            for fold_inner,(ii,jj) in zip(record['inner_folds'],splits):
                assert fold_inner['train_indices']==tr[ii].tolist() and fold_inner['held_indices']==tr[jj].tolist()
                assert set(fold_inner['train_dialogues']).isdisjoint(fold_inner['held_dialogues'])
                replay_fit(g,fold_inner['E1'],tr[ii])
            assert set(record['methods'])==set(FEATURES)
            for method,item in record['methods'].items():
                assert item['features']==FEATURES[method]
                best='off';best_loss=np.inf
                for penalty in ['off',1.,.1,.01]:
                    fits=item['inner_candidate_fits'][str(penalty)]
                    assert len(fits)==5
                    squared=[]
                    for candidate,inn in zip(fits,record['inner_folds']):
                        assert candidate['held_indices']==inn['held_indices'] and candidate['held_dialogues']==inn['held_dialogues']
                        fitted=candidate['fit']
                        assert fitted['penalty']==penalty and fitted['features']==([] if penalty=='off' else item['features'])
                        if penalty=='off': assert fitted==inn['E1'],'off must be exact E1'
                        held=g.iloc[inn['held_indices']]
                        q=replay_fit(g,fitted,inn['train_indices'])(held)['oof_confidence']
                        loss=(q-held.correct.to_numpy(int))**2
                        close(candidate['held_brier'],loss.mean());squared.extend(loss.tolist());inner+=len(loss)
                    loss=float(np.mean(squared));close(loss,item['candidate_brier'][str(penalty)])
                    row=inner_saved[inner_saved.dataset.eq(ds)&inner_saved.strength.eq(strength)&inner_saved.modality.eq(modality)&inner_saved.fold.eq(fold)&inner_saved.method.eq(method)&inner_saved.penalty.eq(str(penalty))]
                    assert len(row)==1
                    close(row.iloc[0].pooled_inner_brier,loss)
                    if loss<best_loss-1e-12: best,best_loss=penalty,loss
                assert best==item['selected_penalty']==item['outer_fit']['penalty']
                if best=='off': assert item['outer_fit']==record['E1']
                actual[method]=replay_fit(g,item['outer_fit'],tr)(g.iloc[he]);selected+=1
            for method in ['E0','E1']+list(FEATURES):
                saved=pp[pp.method.eq(method)&pp.fold.eq(fold)].set_index(KEY)
                held=g.iloc[he].set_index(KEY)
                assert set(saved.index)==set(held.index)
                saved=saved.loc[held.index]
                for col in ['confidence','label','prediction','correct']: close(saved[col],held[col],1e-14)
                assert np.array_equal(saved.correct,(saved.prediction==saved.label).astype(int))
                if method=='E0': close(saved.oof_confidence,held.confidence,0)
                else:
                    for col,value in actual[method].items(): close(saved[col],value)
                outer+=len(saved)
        assert np.all(seen==1)
    assert len(seen_folds)==17 and outer==len(pred)
    assert len(saved_metrics)==80
    for identity,g in pred.groupby(['dataset','strength','modality','method']):
        mask=np.ones(len(saved_metrics),bool)
        for col,value in zip(['dataset','strength','modality','method'],identity): mask &= saved_metrics[col].eq(value)
        row=saved_metrics[mask];assert len(row)==1
        for col,value in metrics(g.oof_confidence,g.correct).items(): close(row.iloc[0][col],value)
        for col,value in {'accuracy':np.mean(g.prediction==g.label),
                          'weighted_f1':f1_score(g.label,g.prediction,average='weighted'),
                          'macro_f1':f1_score(g.label,g.prediction,average='macro')}.items(): close(row.iloc[0][col],value)
    old=read(Path(config['source_mainline'])/'modality/oof_predictions.csv')
    old=old[old.method.isin(['E0','E1','M_joint'])]
    match=pred[pred.strength.eq(.05)&pred.method.isin(['E0','E1','M_joint'])].merge(old,on=KEY+['modality','method'],validate='one_to_one',suffixes=('','_old'))
    assert len(match)==19632
    historical_max_error=float(np.max(np.abs(match.oof_confidence-match.oof_confidence_old)))
    close(match.oof_confidence,match.oof_confidence_old,1e-8);close(match.fold,match.fold_old,0)
    for (ds,modality),g in pred[pred.method.eq('E1')].groupby(['dataset','modality']):
        pair=g[g.strength.eq(.05)].merge(g[g.strength.eq(.1)],on=KEY+['modality','method'],validate='one_to_one',suffixes=('_05','_10'))
        assert len(pair)==COUNTS[ds][0]
        close(pair.oof_confidence_05,pair.oof_confidence_10,0)
    return {'oof_predictions_replayed':outer,'inner_candidate_predictions_replayed':inner,
            'nested_penalty_selections_recomputed':selected,'historical_outer_folds':len(seen_folds),
            'metrics_recomputed':len(saved_metrics),'unchanged_classification':True,
            'train_only_scaling_and_dialogue_grouping':True,'off_exact_E1':True,
            'historical_E1_and_005_Mjoint_modality_parity':True,
            'historical_oof_max_absolute_error':historical_max_error,'historical_oof_absolute_tolerance':1e-8}


def source_review(root):
    path=root/'scripts/run_strength.py'
    source=path.read_text();tree=ast.parse(source)
    # Review only executable call sites; historical helper definitions contain dormant Test routines.
    dataset_calls=[];forbidden=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call): continue
        function=node.func
        if isinstance(function,ast.Attribute) and function.attr=='dataset_for':
            assert len(node.args)>=2 and isinstance(node.args[1],ast.Constant)
            dataset_calls.append(node.args[1].value)
        if isinstance(function,ast.Attribute) and function.attr in ['align','alignment_gate','validation','pilot','training_stats','fit','backward']:
            forbidden.append(function.attr)
    assert dataset_calls==['validation'] and not forbidden,(dataset_calls,forbidden)
    assert 'SEED, STRENGTH, REPEATS = 1475, 0.1, 10' in source
    assert "config['batch_size'] == 16" in source and "config['rnn_backend'] == 'aten'" in source
    assert 'before_state' in source and 'model.state_dict()' in source and 'GraphGuard' in source
    assert 'finally:' in source and 'assert_target_only' in source
    frozen=root/'scripts/mainline_core.py';config=json.loads((root/'preregistration.json').read_text())
    assert sha(frozen)==sha(Path(config['source_mainline'])/'scripts/mainline_core.py')
    return {'live_inference_source_sha256':sha(path),'validation_only_executable_dataset_calls':dataset_calls,
            'no_training_or_Test_inference_call':True,'frozen_mainline_core_sha256':sha(frozen),
            'review_basis':'Static executable-path review plus runtime smoke evidence; dormant historical helper Test definitions are not called.'}


def self_test():
    # A flip is weighted by its perturbed confidence; equality with flip rate is wrong.
    p=np.array([.8,.2]);q=np.array([[.4,.6]]*5+[[.9,.1]]*5)
    b=behavior(p,q)
    close(b['mu'],.3);close(b['flip_rate'],.5);close(b['delta'],.05)
    assert b['mu']!=b['flip_rate']
    close(behavior(p,np.repeat(p[None,:],10,0))['js'],0)
    m=metrics([.2,.8],[0,1]);close(m['ece15'],.2);close(m['binary_brier'],.04)
    # Construct a train-only scaler and ensure held values cannot alter it.
    f=pd.DataFrame({'dialogue_id':['a','a','b'],'confidence':[.2,.7,.9],
                    'gaussian_mu':[0.,2.,100.], 'correct':[0,1,1]})
    fit={'train_indices':[0,1],'n_train':2,'train_dialogues':['a'],
         'optimizer':{'C':1.,'solver':'lbfgs','max_iter':2000,'tol':1e-8},
         'features':['gaussian_mu'],'penalty':1.,'scaler_mean':[1.],'scaler_scale':[1.],
         'design_divisor':np.sqrt(2),'raw_coefficients':[2.,np.sqrt(2)*3],
         'confidence_coefficient':2.,'effective_behavior_coefficients':[3.], 'intercept':-.5}
    scores=replay_fit(f,fit,[0,1])(f.iloc[[2]])
    close(scores['reliability_score'],[-.5+2*.9+3*99])
    print(json.dumps({'status':'passed','engineering_self_test':True,
                      'formal_results_not_claimed':True,'confidence_weighted_mu_checked':True}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--stage',choices=['preflight','full'],default='full')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test: self_test();return
    root=args.root.resolve();config=json.loads((root/'preregistration.json').read_text())
    result={'status':'failed','all_passed':False,'stage':args.stage,'checks':{}}
    try:
        result['checks']['protocol']={'seed':config['seed'],'reference_strength':config['reference_strength'],
                                     'new_strength':config['new_strength'],'repeats':config['repeats']}
        assert (config['seed'],config['reference_strength'],config['new_strength'],config['repeats'])==(1475,.05,.1,10)
        result['checks']['historical_hashes']=historical_hashes(root)
        result['checks']['source_review']=source_review(root)
        result['checks']['inference']=inference_audit(root,smoke=args.stage=='preflight')
        if args.stage=='full':
            analysis=json.loads((root/'results/analysis_verification.json').read_text())
            assert analysis['status']=='complete' and analysis['all_passed']
            for source,digest in json.loads((root/'results/source_hashes.json').read_text()).items(): assert sha(source)==digest,source
            for output,digest in analysis['output_sha256'].items(): assert sha(root/'results'/output)==digest,output
            raw,result['checks']['raw_behavior']=raw_audit(root,config)
            actual=read(root/'results/behavior_features.csv')
            merged=raw.merge(actual,on=KEY+['modality','corruption','strength'],validate='one_to_one',suffixes=('','_saved'))
            assert len(merged)==26176
            for col in ['mu','delta','js','flip_rate','confidence_change_mean','confidence_change_std']+[m+'_'+h for m in ['mu','delta','js','flip_rate'] for h in ['left','right']]: close(merged[col],merged[col+'_saved'])
            result['checks']['calibration']=fit_audit(root,config,raw)
            result['checks']['diagnostics']=diagnostic_audit(root,config)
            result['checks']['bootstrap']=bootstrap_audit(root,config)
        result['status']='passed';result['all_passed']=True
        if args.stage=='preflight': result['formal_results_not_claimed']=True
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
        result['traceback']=traceback.format_exc()
    output=root/'results'/('preflight_verification.json' if args.stage=='preflight' else 'independent_verification.json')
    output.parent.mkdir(exist_ok=True)
    if args.stage=='preflight' and output.exists():
        version=2
        while output.with_name(f'preflight_verification_v{version}.json').exists(): version+=1
        output=output.with_name(f'preflight_verification_v{version}.json')
    result['verification_output']=str(output.relative_to(root))
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']!='passed': raise SystemExit(1)





def nullable_close(actual,expected):
    if not np.isfinite(expected): assert pd.isna(actual)
    else: close(actual,expected)


def diagnostic_audit(root,config):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    b=read(root/'results/behavior_features.csv');summary=read(root/'results/behavior_summary.csv')
    stability=read(root/'results/split_half_stability.csv');pair=read(root/'results/matched_correct_wrong.csv')
    original=read(root/'results/original_predictions.csv').set_index(['dataset','dialogue_id','utterance_index'])
    checked=0
    for rows,kind in [(summary,'summary'),(stability,'stability')]:
        for row in rows.to_dict('records'):
            mask=np.ones(len(b),bool)
            for col in ['dataset','seed','modality','strength','corruption']: mask &= b[col].eq(row[col])
            f=b[mask]
            if row['population']=='valid_visual': f=f[~f.visual_missing.map(flag)]
            assert len(f)==row['utterances'] and f.dialogue_id.nunique()==row['dialogues']
            if kind=='summary':
                for metric in ['mu','delta','js','flip_rate']: close(row['mean_'+metric],f[metric].mean())
                close(row['mu_zero_fraction'],f.mu.eq(0).mean());close(row['nonzero_mu_fraction'],f.mu.ne(0).mean())
            else:
                m=row['measure'];left=f[m+'_left'];right=f[m+'_right']
                expected=np.nan if left.nunique()==1 or right.nunique()==1 else spearmanr(left,right).statistic
                nullable_close(row['spearman'],expected)
                close(row['left_zero_fraction'],left.eq(0).mean());close(row['right_zero_fraction'],right.eq(0).mean())
                close(row['full_zero_fraction'],f[m].eq(0).mean())
                nullable_close(row['auc_larger_is_error'],roc_auc_score(1-f.correct,f[m]) if f.correct.nunique()==2 else np.nan)
                assert row['auc_direction']=='fixed_larger_is_error' and row['n_repeats_each_half']==5
            checked+=1
    pairkey=['dataset','seed','modality','strength','corruption','population','dialogue_id']
    assert not pair.duplicated(pairkey+['correct_utterance_index']).any()
    assert not pair.duplicated(pairkey+['wrong_utterance_index']).any()
    lookup=b.set_index(['dataset','dialogue_id','utterance_index','modality','strength','corruption'])
    for r in pair.to_dict('records'):
        a=original.loc[(r['dataset'],r['dialogue_id'],r['correct_utterance_index'])]
        z=original.loc[(r['dataset'],r['dialogue_id'],r['wrong_utterance_index'])]
        assert a.correct==1 and z.correct==0
        distance=abs(a.confidence-z.confidence)
        assert distance<=.02 and r['confidence_distance']<=.02
        close(r['confidence_distance'],distance)
        if r['population']=='valid_visual': assert not flag(a.visual_missing) and not flag(z.visual_missing)
        for m in ['mu','delta','js','flip_rate']:
            common=(r['modality'],r['strength'],r['corruption'])
            av=lookup.loc[(r['dataset'],r['dialogue_id'],r['correct_utterance_index'])+common,m]
            zv=lookup.loc[(r['dataset'],r['dialogue_id'],r['wrong_utterance_index'])+common,m]
            close([r[m+'_correct'],r[m+'_wrong'],r[m+'_wrong_minus_correct']],[av,zv,zv-av])
    return {'summary_and_stability_rows_recomputed':checked,'matched_pair_rows_checked':len(pair),
            'same_dialogue_C_caliper_and_no_reuse':True,'positive_error_AUC_direction_fixed':True}


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


def bootstrap_audit(root,config):
    b=read(root/'results/behavior_features.csv');pairs=read(root/'results/matched_correct_wrong.csv')
    predictions=read(root/'results/oof_predictions.csv')
    response=read(root/'results/response_bootstrap.csv');calibration=read(root/'results/bootstrap_results.csv')
    checks=literal=0
    for ds in COUNTS:
        registry=np.load(Path(config['source_five_seed'])/f'baseline/bootstrap/{ds}_dialogue_draws.npz')
        ids=registry['dialogue_ids'].astype(str);draws=registry['draws']
        assert draws.shape==(2000,COUNTS[ds][1])
        for r in response[response.dataset.eq(ds)].to_dict('records'):
            frame=b if r['comparison']=='strength_0.10_minus_0.05' else pairs
            f=frame[frame.dataset.eq(ds)&frame.modality.eq(r['modality'])&frame.corruption.eq(r['corruption'])]
            if frame is pairs: f=f[f.population.eq(r['population'])]
            elif r['population']=='valid_visual': f=f[~f.visual_missing.map(flag)]
            keys=KEY if frame is b else ['dialogue_id','correct_utterance_index','wrong_utterance_index']
            old,new=[f[f.strength.eq(s)].sort_values(keys).reset_index(drop=True) for s in [.05,.1]]
            assert old[keys].equals(new[keys])
            m=r['measure']
            if r['comparison']=='strength_0.10_minus_0.05':
                if m=='mu_zero_fraction': a,z=old.mu.eq(0).astype(float),new.mu.eq(0).astype(float)
                elif m=='nonzero_mu_fraction': a,z=old.mu.ne(0).astype(float),new.mu.ne(0).astype(float)
                else:a,z=old[m],new[m]
                values=z.to_numpy()-a.to_numpy();base=old
            elif r['comparison']=='matched_wrong_minus_correct':
                base=old if r['strength']==.05 else new;values=base[m+'_wrong_minus_correct'].to_numpy()
            else:
                assert r['comparison']=='matched_difference_strength_0.10_minus_0.05'
                col=m.removeprefix('matched_')+'_wrong_minus_correct'
                values=new[col].to_numpy()-old[col].to_numpy();base=old
            mean,_,_,_=cluster_statistics(base,ids,draws)
            interval_check(r,mean(values),float(np.mean(values)) if len(values) else np.nan);checks+=1
        for modality in MODALITIES:
            f=predictions[predictions.dataset.eq(ds)&predictions.modality.eq(modality)]
            distributions={};points={}
            for (strength,method),part in f.groupby(['strength','method']):
                mean,idx,mult,den=cluster_statistics(part,ids,draws)
                q=part.oof_confidence.to_numpy();y=part.correct.to_numpy()
                bins=np.minimum((q*15).astype(int),14);residual=np.zeros((len(ids),15))
                np.add.at(residual,(idx,bins),q-y)
                qc=np.clip(q,1e-15,1-1e-15)
                distributions[strength,method]={'ece15':np.abs(mult@residual).sum(1)/den,
                    'binary_brier':mean((q-y)**2),'binary_nll':mean(-y*np.log(qc)-(1-y)*np.log1p(-qc))}
                points[strength,method]=metrics(q,y)
                # Literal first draw checks cluster weighting and nonlinear ECE.
                indices=np.concatenate([np.flatnonzero(idx==g) for g in draws[0]])
                direct=metrics(q[indices],y[indices])
                for m in METRICS: close(distributions[strength,method][m][0],direct[m]);literal+=1
            for r in calibration[calibration.dataset.eq(ds)&calibration.modality.eq(modality)].to_dict('records'):
                ca=(r['candidate_strength'],r['candidate']);re=(r['reference_strength'],r['reference']);m=r['metric']
                dist=distributions[ca][m]-distributions[re][m]
                interval_check(r,dist,points[ca][m]-points[re][m]);close(r['bootstrap_fraction_improving'],np.mean(dist<0));checks+=1
    assert len(calibration)==216 and len(response)==432,(len(calibration),len(response))
    return {'bootstrap_rows_recomputed':checks,'literal_resampling_metric_checks':literal,
            'shared_historical_2000_dialogue_draws':True,'no_refit':True}


if __name__=='__main__': main()
