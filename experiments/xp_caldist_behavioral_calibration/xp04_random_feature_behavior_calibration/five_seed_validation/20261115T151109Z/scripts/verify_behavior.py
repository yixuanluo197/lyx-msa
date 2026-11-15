"""Independent reconstruction of five-seed behavior features, nested OOF and bootstrap."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import f1_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'behavior'
OLD=ROOT.parents[1]/'unified_module_pilot/20261115T145425Z'
KEY=['dataset','seed','dialogue_id','utterance_index']
SEEDS=[1475,67137,2024,3407,42]
DATASETS=['IEMOCAP','MELD']
METHODS=['XP04','E1','L0','E1_MUDELTA','E1_DA','L0_MUDELTA','L0_DA']
METRICS=['ece15','binary_brier','binary_nll']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4194304),b''):h.update(chunk)
    return h.hexdigest()


def csv(path):return pd.read_csv(path,dtype={'dialogue_id':str})


def score(p,y):
    p=np.clip(np.asarray(p,float),1e-15,1-1e-15);y=np.asarray(y,int)
    b=np.minimum(np.floor(p*15).astype(int),14)
    residual=np.bincount(b,weights=p-y,minlength=15)
    return {'ece15':float(abs(residual).sum()/len(y)),
            'binary_brier':float(np.mean((p-y)**2)),
            'binary_nll':float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))}


def base(frame,p,train=None):
    def transformed(f):
        c=f.confidence.to_numpy(float)
        return logit(np.clip(c,1e-6,1-1e-6)) if p['name']=='L0' else c
    x=transformed(frame)
    if train is not None:
        xt=transformed(train)
        if p['name']=='L0':
            assert abs(xt.mean()-p['mean'][0])<1e-10
            assert abs((xt.std() if xt.std()>0 else 1)-p['scale'][0])<1e-10
        else:assert p['mean']==[0.] and p['scale']==[1.]
    return p['intercept']+(x-p['mean'][0])/p['scale'][0]*p['coefficient']


def corrected(frame,names,bp,cp,train):
    xt=train[names].to_numpy(float)
    mean,scale=np.asarray(cp['mean']),np.asarray(cp['scale'])
    assert np.allclose(xt.mean(axis=0),mean,atol=1e-10,rtol=0)
    sd=xt.std(axis=0);sd[sd==0]=1
    assert np.allclose(sd,scale,atol=1e-10,rtol=0)
    w=np.asarray(cp['weights']); x=(xt-mean)/scale
    offset=base(train,bp,train)
    if cp['penalty']=='off':assert np.array_equal(w,np.zeros(4))
    else:
        target=train.correct.to_numpy(int)
        residual=expit(offset+x@w)-target
        gradient=x.T@residual/len(train)+float(cp['penalty'])*w
        assert max(abs(gradient))<1e-6
        objective=np.mean(np.logaddexp(0,offset+x@w)-target*(offset+x@w))+float(cp['penalty'])*np.dot(w,w)/2
        assert abs(objective-cp['objective'])<1e-10
    return expit(base(frame,bp)+(frame[names].to_numpy(float)-mean)/scale@w)


def verify_features(features):
    assert len(features)==8180 and not features.duplicated(KEY).any()
    checked=0
    old=csv(OLD/'feature_matrix.csv')
    now=features[features.seed.eq(1475)]
    merged=now.merge(old,on=KEY,suffixes=('','_old'),validate='one_to_one')
    assert len(merged)==1636
    columns=[c for c in old.columns if c not in KEY]
    for col in columns:
        if pd.api.types.is_numeric_dtype(merged[col]):
            assert np.allclose(merged[col],merged[col+'_old'],atol=1e-12,rtol=0)
        else:assert merged[col].equals(merged[col+'_old'])
    for ds in DATASETS:
        for seed in SEEDS:
            f=features[features.dataset.eq(ds)&features.seed.eq(seed)]
            assert len(f)==(527 if ds=='IEMOCAP' else 1109)
            assert f.dialogue_id.nunique()==(12 if ds=='IEMOCAP' else 114)
            assert f.visual_missing.sum()==(1 if ds=='IEMOCAP' else 95)
            assert np.array_equal(f.correct,(f.label==f.prediction).astype(int))
            if seed==1475:continue
            folder=ROOT/f'runs/{ds}/seed{seed}'
            complete=json.loads((folder/'complete.json').read_text())
            assert complete['status']=='completed' and complete['weights_unchanged'] and complete['parity']['passed']
            raw=csv(folder/'perturbed_predictions.csv')
            assert len(raw)==20*len(f) and not raw.duplicated(KEY+['corruption','repeat_id']).any()
            # Recompute from complete saved probabilities independently of the analyzer.
            expected=f.set_index(KEY)
            for key,g in raw.groupby(KEY+['corruption'],sort=False):
                assert sorted(g.repeat_id.tolist())==list(range(10))
                k=key[:-1];kind=key[-1];row=expected.loc[k]
                probs=np.array([json.loads(value) for value in g.probabilities])
                lp=np.array([json.loads(value) for value in g.log_probabilities])
                assert np.allclose(np.exp(lp),probs,atol=1e-12,rtol=1e-12)
                assert np.array_equal(probs.argmax(axis=1),g.prediction)
                assert np.allclose(probs.max(axis=1),g.confidence,atol=1e-12,rtol=0)
                c=float(row.confidence); q=probs[:,int(row.prediction)];mx=probs.max(axis=1)
                values={'D':c-np.mean(q),'A':np.sqrt(np.mean((q-c)**2)),
                        'mu':np.mean((probs.argmax(axis=1)!=int(row.prediction))*mx),
                        'delta':abs(c-np.mean(mx))}
                for name,v in values.items():assert abs(row[f'{kind}_{name}']-v)<1e-12
                checked+=1
    return checked


def verify_bootstrap(predictions,bootstrap):
    expected_pairs={(m,'XP04') for m in METHODS if m!='XP04'} | {
        (base+'_'+variant,base) for base in ['E1','L0'] for variant in ['MUDELTA','DA']
    } | {('E1_DA','E1_MUDELTA'),('L0_DA','L0_MUDELTA')}
    assert len(expected_pairs)==12 and len(bootstrap)==72
    for dataset in DATASETS:
        subset=bootstrap[bootstrap.dataset.eq(dataset)]
        assert not subset.duplicated(['candidate','reference','metric']).any()
        assert {(r.candidate,r.reference) for r in subset.itertuples()}==expected_pairs
        for candidate,reference in expected_pairs:
            assert set(subset[subset.candidate.eq(candidate)&subset.reference.eq(reference)].metric)==set(METRICS)
    checks=0;direct_checks=0
    for ds in DATASETS:
        ps=predictions[predictions.dataset.eq(ds)]
        z=np.load(ROOT/f'baseline/bootstrap/{ds}_dialogue_draws.npz')
        groups=z['dialogue_ids'];indices=z['draws']
        assert indices.shape==(2000,len(groups))
        counts=np.stack([np.bincount(row,minlength=len(groups)) for row in indices])
        saved=json.loads((OUT/f'provenance/bootstrap_draws_{ds}.json').read_text())
        assert saved['dialogue_ids']==groups.tolist()
        assert np.array_equal(counts,np.array(saved['multiplicities']))
        draws={};full={}
        for seed in SEEDS:
            for method in METHODS:
                p=ps[ps.seed.eq(seed)&ps.method.eq(method)].copy()
                gi=np.searchsorted(groups,p.dialogue_id.to_numpy(str))
                n=np.bincount(gi,minlength=len(groups));total=counts@n
                y=p.correct.to_numpy(int);x=np.clip(p.oof_confidence.to_numpy(float),1e-15,1-1e-15)
                logs=-(y*np.log(x)+(1-y)*np.log1p(-x));b=np.minimum((x*15).astype(int),14)
                residual=np.zeros((len(groups),15));np.add.at(residual,(gi,b),x-y)
                draws[seed,method]={'ece15':np.abs(counts@residual).sum(axis=1)/total,
                    'binary_brier':counts@np.bincount(gi,weights=(x-y)**2,minlength=len(groups))/total,
                    'binary_nll':counts@np.bincount(gi,weights=logs,minlength=len(groups))/total}
                full[seed,method]=score(x,y)
                # Literal repeated utterances verify cluster weighting without linearizing ECE.
                if method in ['XP04','E1_DA']:
                    for which in range(3):
                        ix=np.concatenate([np.flatnonzero(gi==g) for g in indices[which]])
                        literal=score(x[ix],y[ix])
                        for metric in METRICS:
                            assert abs(literal[metric]-draws[seed,method][metric][which])<1e-10
                            direct_checks+=1
        for record in bootstrap[bootstrap.dataset.eq(ds)].to_dict('records'):
            candidate,reference,metric=record['candidate'],record['reference'],record['metric']
            distribution=np.mean([draws[s,candidate][metric]-draws[s,reference][metric] for s in SEEDS],axis=0)
            differences=np.array([full[s,candidate][metric]-full[s,reference][metric] for s in SEEDS])
            wanted={'mean_difference':differences.mean(),'seed_difference_sample_sd':differences.std(ddof=1),
                    'ci_low':np.quantile(distribution,.025),'ci_high':np.quantile(distribution,.975),
                    'bootstrap_fraction_improving':np.mean(distribution<0)}
            for k,v in wanted.items():assert abs(record[k]-v)<1e-10,(ds,candidate,reference,metric,k)
            assert record['improved_seeds']==np.sum(differences<0)
            checks+=1
    return checks,direct_checks


def self_test():
    report=json.loads((OUT/'engineering_self_test.json').read_text())
    assert report['engineering_fixture_only'] and not report['formal_five_seed_complete']
    assert report['seed1475_frozen_method_reproduced']
    for row in report['datasets']:
        assert row['max_prior_oof_error']<1e-12 and row['oof_rows']==7*row['utterances']
    assert report['core_sha256']==sha(ROOT/'provenance/analysis_code/calibration_core.py')==sha(OLD/'calibration_core.py')
    print(json.dumps({'engineering_fixture_verified':True,'formal_five_seed_results_not_claimed':True},indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test();return
    verification=json.loads((OUT/'verification.json').read_text())
    assert verification['status']=='completed' and verification['ten_inputs_complete']
    for p,h in json.loads((OUT/'provenance/source_hashes.json').read_text()).items():assert sha(p)==h
    snap=json.loads((OUT/'provenance/run_snapshot.json').read_text())
    for p,h in snap['code_sha256'].items():assert sha(p)==h
    f=csv(OUT/'feature_matrix.csv');p=csv(OUT/'oof_predictions.csv');m=csv(OUT/'per_seed_metrics.csv')
    cv=csv(OUT/'inner_cv_results.csv');agg=csv(OUT/'aggregate_metrics.csv');shift=csv(OUT/'correct_wrong_adjustment.csv')
    per=csv(OUT/'per_dialogue_results.csv');selected=csv(OUT/'selected_penalties.csv');off=csv(OUT/'behavior_off_folds.csv')
    assert len(p)==57260 and len(m)==70 and len(cv)==1360 and len(selected)==340 and len(off)==40
    assert not p.duplicated(KEY+['method']).any() and not m.duplicated(['dataset','seed','method']).any()
    features_checked=verify_features(f)
    outer_count,inner_count,fit_count,selection_count,max_error=0,0,0,0,0.
    for ds in DATASETS:
        for seed in SEEDS:
            frame=f[f.dataset.eq(ds)&f.seed.eq(seed)].copy()
            params=json.loads((OUT/f'fold_parameters/{ds}_seed{seed}.json').read_text())
            assert len(params)==(72 if ds=='IEMOCAP' else 30)
            for r in params:
                trainset,heldset=set(r['train_dialogues']),set(r['held_dialogues'])
                assert trainset.isdisjoint(heldset) and trainset|heldset==set(frame.dialogue_id)
                train=frame[frame.dialogue_id.isin(trainset)];held=frame[frame.dialogue_id.isin(heldset)]
                base(held,r['base'],train)
                if 'correction' in r:
                    actual=corrected(held,r['features'],r['base'],r['correction'],train)
                    inner_predictions={str(c['penalty']):pd.Series(index=train.index,dtype=float) for c in r['candidates']}
                    seen=set()
                    for fold in r['inner_folds']:
                        ts,hs=set(fold['train_dialogues']),set(fold['held_dialogues'])
                        assert ts.isdisjoint(hs) and ts|hs==trainset and seen.isdisjoint(hs)
                        seen|=hs
                        tr=frame[frame.dialogue_id.isin(ts)];ho=frame[frame.dialogue_id.isin(hs)]
                        assert len(tr)==fold['train_n'] and len(ho)==fold['held_n']
                        for penalty,cp in fold['corrections'].items():
                            inner_predictions[penalty].loc[ho.index]=corrected(ho,r['features'],fold['base'],cp,tr)
                            fit_count+=1;inner_count+=len(ho)
                    assert seen==trainset
                    chosen,best='off',np.inf
                    for candidate in r['candidates']:
                        penalty=candidate['penalty'];x=inner_predictions[str(penalty)]
                        assert x.notna().all()
                        values=score(x.to_numpy(),train.correct.to_numpy())
                        for k,v in values.items():assert abs(candidate[k]-v)<1e-10
                        saved=cv[cv.dataset.eq(ds)&cv.seed.eq(seed)&cv.method.eq(r['method'])&cv.outer_fold.eq(r['fold'])&cv.penalty.eq(str(penalty))]
                        assert len(saved)==1
                        assert all(abs(saved.iloc[0][k]-v)<1e-10 for k,v in values.items())
                        if values['binary_brier']<best-1e-12:chosen,best=penalty,values['binary_brier']
                    assert chosen==r['selected_penalty']==r['correction']['penalty']
                    selection_count+=1
                else:actual=expit(base(held,r['base']))
                predicted=p[p.dataset.eq(ds)&p.seed.eq(seed)&p.method.eq(r['method'])]
                aligned=held[KEY+['label','prediction','correct']].merge(predicted,on=KEY,validate='one_to_one',suffixes=('','_saved'))
                assert len(aligned)==len(held) and aligned.fold.eq(r['fold']).all()
                for col in ['label','prediction','correct']:assert aligned[col].equals(aligned[col+'_saved'])
                error=float(np.max(abs(actual-aligned.oof_confidence.to_numpy())))
                assert error<1e-10;(max_error:=max(max_error,error))
                outer_count+=len(held)
    for row in m.to_dict('records'):
        q=p[p.dataset.eq(row['dataset'])&p.seed.eq(row['seed'])&p.method.eq(row['method'])]
        x,y,c=q.oof_confidence.to_numpy(),q.correct.to_numpy(),q.original_confidence.to_numpy()
        for k,v in score(x,y).items():assert abs(row[k]-v)<1e-10
        extras={'accuracy':y.mean(),'weighted_f1':f1_score(q.label,q.prediction,average='weighted'),
                'macro_f1':f1_score(q.label,q.prediction,average='macro'),'correctness_auc':roc_auc_score(y,x),
                'mean_confidence':x.mean(),'correct_mean_confidence':x[y==1].mean(),'wrong_mean_confidence':x[y==0].mean(),
                'raised_fraction':np.mean(x>c+1e-12),'lowered_fraction':np.mean(x<c-1e-12)}
        for k,v in extras.items():assert abs(row[k]-v)<1e-10
        for correct in [0,1]:
            s=shift[shift.dataset.eq(row['dataset'])&shift.seed.eq(row['seed'])&shift.method.eq(row['method'])&shift.correct.eq(correct)].iloc[0]
            mask=y==correct
            values={'mean_original':c[mask].mean(),'mean_calibrated':x[mask].mean(),'mean_shift':(x[mask]-c[mask]).mean(),
                    'raised_fraction':np.mean(x[mask]>c[mask]+1e-12),'lowered_fraction':np.mean(x[mask]<c[mask]-1e-12)}
            assert s.n==mask.sum()
            for k,v in values.items():assert abs(s[k]-v)<1e-10
    for row in agg.to_dict('records'):
        source=m[m.dataset.eq(row['dataset'])&m.method.eq(row['method'])]
        for key in row:
            if key.endswith('_mean'):assert abs(row[key]-source[key[:-5]].mean())<1e-10
            elif key.endswith('_sample_sd'):assert abs(row[key]-source[key[:-10]].std(ddof=1))<1e-10
    for row in per.to_dict('records'):
        q=p[p.dataset.eq(row['dataset'])&p.seed.eq(row['seed'])&p.method.eq(row['method'])&p.dialogue_id.eq(row['dialogue_id'])]
        assert len(q)==row['n']
        for k,v in score(q.oof_confidence,q.correct).items():assert abs(row[k]-v)<1e-10
    # Baseline and historical parity for every eligible row, including fold IDs.
    baseline=csv(ROOT/'baseline/oof_predictions.csv')
    paired=p[p.method.isin(['XP04','E1','L0'])].merge(baseline,on=KEY+['method'],validate='one_to_one',suffixes=('','_base'))
    assert len(paired)==24540 and np.array_equal(paired.fold,paired.fold_base)
    assert np.max(abs(paired.oof_confidence-paired.oof_confidence_base))<1e-12
    old=csv(OLD/'oof_predictions.csv')
    paired=p[p.seed.eq(1475)].merge(old,on=KEY+['method'],validate='one_to_one',suffixes=('','_old'))
    assert len(paired)==11452 and np.max(abs(paired.oof_confidence-paired.oof_confidence_old))<1e-12
    for row in off.to_dict('records'):
        s=selected[selected.dataset.eq(row['dataset'])&selected.seed.eq(row['seed'])&selected.method.eq(row['method'])]
        assert len(s)==row['outer_folds'] and s.selected_penalty.eq('off').sum()==row['off_folds']
    boots,direct=verify_bootstrap(p,csv(OUT/'bootstrap_results.csv'))
    result={'all_passed':True,'behavior_groups_independently_recomputed':features_checked,
        'outer_predictions_reconstructed':outer_count,'inner_candidate_predictions_reconstructed':inner_count,
        'inner_candidate_fits_checked':fit_count,'nested_selections_recomputed':selection_count,
        'max_outer_reconstruction_error':max_error,'metric_rows_recomputed':len(m),
        'per_dialogue_metric_rows_recomputed':len(per),'shared_bootstrap_comparison_rows_recomputed':boots,
        'literal_dialogue_resampling_checks':direct,'seed1475_all_methods_historical_parity':True,
        'all_seeds_baseline_three_methods_parity':True,'training_only_scaler_and_correction_objective_verified':True,
        'source_and_code_hashes_verified':True}
    verification['independent_verification']=result;verification['independent_verification_pending']=False
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    hashes={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
    (OUT/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
