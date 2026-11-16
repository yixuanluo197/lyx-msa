"""Preregistered CPU-only precision analysis of saved perturbation probabilities."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import spearmanr
from mainline_core import FEATURES, METRICS, calibration_metrics, dump, fit_outer, metrics, sha

KEY = ['dataset', 'seed', 'dialogue_id', 'utterance_index']
GROUP = ['dataset', 'seed', 'condition']
METHODS = ['E0', 'E1', 'M_gaussian', 'M_gaussian_mu', 'M_gaussian_delta']
BEHAVIOR = METHODS[2:]
FEATURES.update(M_gaussian_mu=['gaussian_mu'], M_gaussian_delta=['gaussian_delta'])


def read(path, **kw):
    return pd.read_csv(path, dtype={'dataset': str, 'dialogue_id': str}, float_precision='round_trip', **kw)


def unique(frame, keys):
    if frame.duplicated(keys).any():
        raise ValueError(f'Duplicate identities {keys}')


def native_group(key):
    return dict(dataset=str(key[0]), seed=int(key[1]), condition=str(key[2]))


def validate_p(values, prediction, confidence, classes):
    p = np.asarray(values, float)
    if p.ndim != 2 or p.shape[1] != classes or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid complete probability vector')
    if not np.allclose(p.sum(1), 1, atol=1e-5, rtol=0):
        raise ValueError('Probability normalization failed')
    if not np.array_equal(p.argmax(1), prediction) or not np.allclose(p.max(1), confidence, atol=1e-12, rtol=0):
        raise ValueError('Stored prediction/confidence disagrees with probability vector')
    return p


def response(C, prediction, p, original):
    """Compute original definitions over all supplied repeats, then fixed half estimates."""
    p, original = np.asarray(p, float), np.asarray(original, float)
    n = len(p)
    if n not in [10, 30]:
        raise ValueError('Only preregistered 10 and 30 repeat conditions are permitted')
    confidence = p.max(1)
    flip = p.argmax(1) != prediction
    middle = (p + original[None, :]) / 2
    js = .5*np.sum(rel_entr(p, middle)+rel_entr(original[None, :], middle), axis=1)
    out = dict(mu=float(np.mean(flip*confidence)), delta=float(abs(C-confidence.mean())),
               js=float(js.mean()), flip_rate=float(flip.mean()),
               confidence_change_mean=float(confidence.mean()-C),
               confidence_change_std=float(confidence.std(ddof=0)),
               confidence_sample_std=float(confidence.std(ddof=1)),
               confidence_mean_standard_error=float(confidence.std(ddof=1)/np.sqrt(n)))
    for side, sl in [('left', slice(0,n//2)), ('right', slice(n//2,n))]:
        out.update({f'mu_{side}':float(np.mean((flip*confidence)[sl])),
                    f'delta_{side}':float(abs(C-confidence[sl].mean())),
                    f'js_{side}':float(js[sl].mean())})
    return out


def safe_spearman(x, y):
    return np.nan if len(x)<2 or np.ptp(x)==0 or np.ptp(y)==0 else float(spearmanr(x,y).statistic)


def load_inputs(config):
    history_path=Path(config['source_mainline'])/'inputs/joint_features.csv'
    history=read(history_path)
    hashes={str(history_path):sha(history_path)}
    originals, raws, rows, checks=[], [], [], []
    old_path=Path(config['source_candidate'])/'results/behavior_features.csv'
    old=read(old_path); hashes[str(old_path)]=sha(old_path)
    for ds,spec in config['datasets'].items():
        classes=6 if ds=='IEMOCAP' else 7
        path=Path(config['source_pilot'])/ds/'original_predictions.csv'
        original=read(path); hashes[str(path)]=sha(path); unique(original,KEY)
        if len(original)!=spec['utterances'] or original.dialogue_id.nunique()!=spec['dialogues'] or set(original.seed)!={1475} or set(original.dataset)!={ds}:
            raise ValueError('Original true Validation coverage mismatch')
        validate_p([json.loads(v) for v in original.probabilities],original.prediction,original.confidence,classes)
        original['correct']=original.prediction.eq(original.label).astype(int)
        flags=history[history.dataset.eq(ds)&history.seed.eq(1475)][KEY+['visual_missing']]
        original=original.merge(flags,on=KEY,validate='one_to_one')
        if len(original)!=spec['utterances'] or original.visual_missing.sum()!=spec['visual_missing']:
            raise ValueError('Missing visual retention mismatch')
        originals.append(original); origin=original.set_index(KEY)
        selected=[]
        for block in range(3):
            path=(Path(config['source_pilot'])/ds/'perturbed_predictions.csv' if block==0 else
                  Path(config['source_candidate'])/'runs'/ds/'seed1475'/f'block{block}'/'perturbed_predictions.csv')
            hashes[str(path)]=sha(path); raw=read(path)
            raw=raw[raw.modality.eq(spec['modality'])&raw.corruption.eq('gaussian')&raw.strength.eq(.05)].copy()
            if len(raw)!=len(original)*10 or set(raw.seed)!={1475} or set(raw.dataset)!={ds}:
                raise ValueError('Selected raw candidate coverage mismatch')
            unique(raw,KEY+['repeat_id'])
            if set(map(tuple,raw[KEY].drop_duplicates().to_numpy()))!=set(origin.index):
                raise ValueError('Selected raw identities mismatch')
            if raw.visual_missing_skipped.isna().any() or raw.visual_missing_skipped.astype(bool).any():
                raise ValueError('Text/audio must retain all visual-missing samples')
            raw['source_block']=block; raw['global_repeat_id']=raw.repeat_id+10*block
            if block and (not raw.block.eq(block).all() or not np.array_equal(raw.rng_repeat_id,raw.global_repeat_id)):
                raise ValueError('Saved block/global random repetition mismatch')
            raw['source_path']=str(path); raw['source_sha256']=hashes[str(path)]
            raw['log_probabilities_available']='log_probabilities' in raw.columns
            selected.append(raw)
        raw=pd.concat(selected,ignore_index=True); unique(raw,KEY+['global_repeat_id']); raws.append(raw)
        for key,group in raw.groupby(KEY,sort=False):
            group=group.sort_values('global_repeat_id')
            if group.global_repeat_id.tolist()!=list(range(30)):
                raise ValueError('Every utterance must contain global repeats 0 through 29 exactly once')
            p=validate_p([json.loads(v) for v in group.probabilities],group.prediction,group.confidence,classes)
            saved=group.log_probabilities_available.to_numpy(bool)
            lp=np.asarray([json.loads(v) for v in group.loc[saved,'log_probabilities']],float)
            if lp.shape!=p[saved].shape or not np.isfinite(lp).all() or not np.allclose(np.exp(lp),p[saved],atol=1e-12,rtol=1e-12):
                raise ValueError('Saved log probabilities do not reproduce saved probabilities')
            source=origin.loc[key]
            for condition,repeats in config['conditions'].items():
                row=dict(zip(KEY,key)); row.update({c:source[c] for c in ['label','prediction','correct','confidence','visual_missing']})
                row.update(condition=condition,modality=spec['modality'],corruption='gaussian',strength=.05,
                           repeat_count=len(repeats),perturbation_skipped=False)
                row.update(response(source.confidence,source.prediction,p[repeats],json.loads(source.probabilities)))
                rows.append(row)
        print(f'Validated {ds}: {len(original)} originals, {len(raw)} selected probability rows',flush=True)
    originals=pd.concat(originals,ignore_index=True); raw=pd.concat(raws,ignore_index=True)
    behavior=pd.DataFrame(rows); unique(behavior,KEY+['condition'])
    order=history[history.seed.eq(1475)][KEY].copy(); order['_order']=np.arange(len(order))
    features=behavior.merge(order,on=KEY,validate='many_to_one').sort_values(['condition','_order'],kind='stable').drop(columns='_order').reset_index(drop=True)
    for col in ['mu','delta','js','flip_rate','confidence_change_mean','confidence_change_std']:
        features['gaussian_'+col]=features[col]
    for block in range(3):
        new=features[features.condition.eq(f'K10_block{block}')]
        ref=old[old.seed.eq(1475)&old.block.eq(block)]
        join=new.merge(ref,on=KEY,suffixes=('_new','_old'),validate='one_to_one')
        if len(join)!=1636: raise ValueError('Historical K10 behavior coverage mismatch')
        for col in ['mu','delta','js','flip_rate','confidence_change_mean','confidence_change_std','mu_left','mu_right','delta_left','delta_right','js_left','js_right']:
            error=float(np.max(abs(join[col+'_new']-join[col+'_old'])))
            if error>1e-12: raise ValueError(f'K10 behavior parity {block} {col}: {error}')
            checks.append(dict(block=block,feature=col,max_absolute_error=error))
    return originals,raw,features,hashes,checks


def measurement_tables(features):
    summaries, stability=[],[]
    for key,f in features.groupby(GROUP,sort=False):
        common=dict(zip(GROUP,key)); common.update(modality=str(f.modality.iloc[0]),repeat_count=int(f.repeat_count.iloc[0]),utterances=len(f))
        row=dict(common,any_flip_fraction=float(f.flip_rate.gt(0).mean()),mu_nonzero_count=int(f.mu.ne(0).sum()),mu_zero_fraction=float(f.mu.eq(0).mean()))
        for col in ['mu','delta','js','confidence_mean_standard_error','confidence_sample_std']:
            row[col+'_mean']=float(f[col].mean());row[col+'_median']=float(f[col].median());row[col+'_p90']=float(f[col].quantile(.9))
        summaries.append(row)
        for measure in ['mu','delta','js']:
            a,b=f[measure+'_left'],f[measure+'_right']
            stability.append(dict(common,measure=measure,spearman=safe_spearman(a,b),
                                  left_nonzero_count=int(a.ne(0).sum()),right_nonzero_count=int(b.ne(0).sum()),
                                  left_zero_fraction=float(a.eq(0).mean()),right_zero_fraction=float(b.eq(0).mean()),
                                  left_tied_observation_fraction=float(a.duplicated(keep=False).mean()),
                                  right_tied_observation_fraction=float(b.duplicated(keep=False).mean()),
                                  left_unique_values=int(a.nunique()),right_unique_values=int(b.nunique())))
    return dict(measurement_summary=pd.DataFrame(summaries),split_half_stability=pd.DataFrame(stability))


def run_calibration(config,features,destination):
    folds_path=Path(config['source_five_seed'])/'baseline/fold_parameters.json'
    old_path=Path(config['source_candidate'])/'results/oof_predictions.csv'
    folds=json.loads(folds_path.read_text());old=read(old_path)
    outputs,metric_rows,selections,shifts,parity=[],[],[],[],[]
    for key,frame in features.groupby(GROUP,sort=False):
        common=native_group(key);common.update(modality=str(frame.modality.iloc[0]),repeat_count=int(frame.repeat_count.iloc[0]))
        frame=frame.reset_index(drop=True);parts,params=[],[]; covered=np.zeros(len(frame),int)
        for fold in [v for v in folds if v['dataset']==common['dataset'] and v['seed']==1475]:
            train=np.flatnonzero(frame.dialogue_id.isin(fold['train_dialogues']));held=np.flatnonzero(frame.dialogue_id.isin(fold['held_dialogues']))
            if len(train)!=fold['train_n'] or len(held)!=fold['held_n'] or len(train)+len(held)!=len(frame):
                raise ValueError('Historical outer dialogue fold mismatch')
            scored,fitted=fit_outer(frame,train,held,BEHAVIOR)
            fitted.update(common,fold=fold['fold']); params.append(fitted);covered[held]+=1
            scored['E0']=dict(oof_confidence=frame.iloc[held].confidence.to_numpy(float),base_score=np.full(len(held),np.nan),behavior_score=np.zeros(len(held)),reliability_score=np.full(len(held),np.nan))
            for method in METHODS:
                part=frame.iloc[held][KEY+['condition','modality','repeat_count','label','prediction','correct','confidence','visual_missing','perturbation_skipped']].copy()
                part['method'],part['fold']=method,fold['fold']
                part['selected_penalty']=fitted['methods'][method]['selected_penalty'] if method in BEHAVIOR else ('off' if method=='E1' else 'none')
                for col,values in scored[method].items():part[col]=values
                parts.append(part)
            for method,selection in fitted['methods'].items():
                for penalty,loss in selection['candidate_brier'].items():
                    selections.append(dict(common,fold=fold['fold'],method=method,penalty=penalty,pooled_inner_brier=loss,selected=str(selection['selected_penalty'])==penalty))
        if not np.all(covered==1):raise ValueError('OOF coverage differs from one')
        predictions=pd.concat(parts,ignore_index=True);outputs.append(predictions)
        dump(destination/'fold_parameters'/f"{common['dataset']}_{common['condition']}.json",params)
        for method,f in predictions.groupby('method',sort=False):
            metric_rows.append(dict(common,method=method,**metrics(f,f.oof_confidence.to_numpy(float))))
            for correct,g in f.groupby('correct'):
                change=g.oof_confidence-g.confidence
                shifts.append(dict(common,method=method,correct=int(correct),utterances=len(g),original_mean_confidence=float(g.confidence.mean()),calibrated_mean_confidence=float(g.oof_confidence.mean()),mean_change=float(change.mean()),raised_fraction=float((change>1e-12).mean()),lowered_fraction=float((change< -1e-12).mean())))
            block=0 if common['condition']=='K30_pooled' else int(common['condition'][-1])
            if common['condition']=='K30_pooled' and method!='E1': continue
            ref=old[old.dataset.eq(common['dataset'])&old.seed.eq(1475)&old.block.eq(block)&old.method.eq(method)]
            join=f.merge(ref[KEY+['oof_confidence']],on=KEY,suffixes=('_new','_old'),validate='one_to_one')
            error=float(np.max(abs(join.oof_confidence_new-join.oof_confidence_old)))
            if len(join)!=len(frame) or error>1e-8:raise ValueError(f'Historical OOF mismatch {common} {method}: {error}')
            parity.append(dict(common,method=method,rows=len(join),max_absolute_error=error,tolerance=1e-8))
        print(f"OOF completed {common['dataset']} {common['condition']}",flush=True)
    return dict(oof_predictions=pd.concat(outputs,ignore_index=True),calibration_metrics=pd.DataFrame(metric_rows),inner_cv_results=pd.DataFrame(selections),confidence_shift=pd.DataFrame(shifts)),parity


def load_draws(config,ds):
    with np.load(Path(config['source_five_seed'])/f'baseline/bootstrap/{ds}_dialogue_draws.npz') as f:
        ids,draws=f['dialogue_ids'].astype(str),f['draws']
    if draws.shape!=(2000,config['datasets'][ds]['dialogues']):raise ValueError('Bootstrap shape mismatch')
    return ids,np.stack([np.bincount(row,minlength=len(ids)) for row in draws])


def bootstrap_distribution(frame,ids,multiplicities):
    idx=pd.Index(ids).get_indexer(frame.dialogue_id)
    if (idx<0).any():raise ValueError('Unknown bootstrap dialogue')
    total=multiplicities@np.bincount(idx,minlength=len(ids));q=frame.oof_confidence.to_numpy(float);y=frame.correct.to_numpy(int)
    residual=np.zeros((len(ids),15));bi=np.minimum(np.floor(q*15).astype(int),14)
    np.add.at(residual,(idx,bi),q-y);clip=np.clip(q,1e-15,1-1e-15)
    return dict(ece15=np.abs(multiplicities@residual).sum(1)/total,
                binary_brier=multiplicities@np.bincount(idx,weights=(q-y)**2,minlength=len(ids))/total,
                binary_nll=multiplicities@np.bincount(idx,weights=-y*np.log(clip)-(1-y)*np.log1p(-clip),minlength=len(ids))/total)


def comparisons(config):
    for condition in config['conditions']:
        for method in BEHAVIOR:
            for reference in ['E0','E1']:
                yield condition,method,condition,reference,'same_condition'
        yield condition,'E1',condition,'E0','same_condition'
        for reference in ['M_gaussian_mu','M_gaussian_delta']:
            yield condition,'M_gaussian',condition,reference,'ablation'
    for block in range(3):
        yield 'K30_pooled','M_gaussian',f'K10_block{block}','M_gaussian','primary_repeat' if block==0 else 'descriptive_repeat'


def comparative_tables(config,predictions):
    bootstrap,influence=[],[]
    for ds in config['datasets']:
        ids,multiplicities=load_draws(config,ds);part=predictions[predictions.dataset.eq(ds)]
        indexed={(c,m):f.set_index(KEY).sort_index() for (c,m),f in part.groupby(['condition','method'],sort=False)}
        dist={key:bootstrap_distribution(f.reset_index(),ids,multiplicities) for key,f in indexed.items()}
        points={key:calibration_metrics(f.oof_confidence,f.correct) for key,f in indexed.items()}
        for ac,am,bc,bm,scope in comparisons(config):
            a,b=indexed[ac,am],indexed[bc,bm]
            if not a.index.equals(b.index) or not np.array_equal(a.correct,b.correct):raise ValueError('Unpaired confidence comparison')
            for metric in METRICS:
                d=dist[ac,am][metric]-dist[bc,bm][metric];lo,hi=np.quantile(d,[.025,.975])
                bootstrap.append(dict(dataset=ds,seed=1475,candidate_condition=ac,candidate=am,reference_condition=bc,reference=bm,scope=scope,metric=metric,difference=points[ac,am][metric]-points[bc,bm][metric],ci_low=float(lo),ci_high=float(hi),bootstrap_fraction_improving=float((d<0).mean()),resamples=2000,refit=False,unit='dialogue',exploratory=True))
            if not ((am=='M_gaussian' and bm=='E1') or scope in ['primary_repeat','descriptive_repeat']):continue
            dialogue=a.index.get_level_values('dialogue_id').to_numpy()
            for d in dict.fromkeys(dialogue):
                held=dialogue==d;keep=~held
                ma=calibration_metrics(a.oof_confidence.to_numpy()[keep],a.correct.to_numpy()[keep]);mb=calibration_metrics(b.oof_confidence.to_numpy()[keep],b.correct.to_numpy()[keep])
                la=calibration_metrics(a.oof_confidence.to_numpy()[held],a.correct.to_numpy()[held]);lb=calibration_metrics(b.oof_confidence.to_numpy()[held],b.correct.to_numpy()[held])
                for metric in METRICS:
                    influence.append(dict(dataset=ds,seed=1475,candidate_condition=ac,candidate=am,reference_condition=bc,reference=bm,scope=scope,dialogue_id=str(d),dialogue_utterances=int(held.sum()),metric=metric,full_difference=points[ac,am][metric]-points[bc,bm][metric],omitted_difference=ma[metric]-mb[metric],dialogue_difference=la[metric]-lb[metric],refit=False))
    influence=pd.DataFrame(influence);summary=[]
    groups=['dataset','seed','candidate_condition','candidate','reference_condition','reference','scope','metric']
    for key,f in influence.groupby(groups,sort=False):
        full=float(f.full_difference.iloc[0]);om=f.omitted_difference.to_numpy(float)
        summary.append(dict(zip(groups,key),full_difference=full,dialogues=len(f),dialogues_improving=int((f.dialogue_difference<0).sum()),omission_difference_min=float(om.min()),omission_difference_max=float(om.max()),omissions_reversing_direction=int(((om>0) if full<0 else (om<0)).sum()),refit=False))
    return dict(bootstrap_results=pd.DataFrame(bootstrap),dialogue_influence=influence,dialogue_concentration_summary=pd.DataFrame(summary))


def save_tables(destination,tables):
    destination.mkdir(parents=True,exist_ok=True)
    for name,frame in tables.items():
        frame.to_csv(destination/f'{name}.csv',index=False,float_format='%.17g',na_rep='NA')
        print(f'{name}: {len(frame)} rows',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    root=args.root.resolve();config=json.loads((root/'preregistration.json').read_text());started=time.time()
    if config['methods']!=METHODS or config['penalties']!=['off',1.,.1,.01] or config['seed']!=1475:raise ValueError('Fixed protocol changed')
    static=[root/'preregistration.json',root/'EXPERIMENT_PLAN.md']+[root/'scripts'/n for n in ['analyze_precision.py','mainline_core.py','test_precision.py','build_precision_reports.py']]
    static+=[Path(config['source_five_seed'])/'baseline/fold_parameters.json',Path(config['source_candidate'])/'results/oof_predictions.csv',Path(config['source_candidate'])/'results/calibration_metrics.csv']
    static += [Path(config['source_five_seed'])/f'baseline/bootstrap/{ds}_dialogue_draws.npz' for ds in config['datasets']]
    before={str(p):sha(p) for p in static}
    original,raw,features,hashes,feature_parity=load_inputs(config);before.update(hashes)
    if [len(original),len(raw),len(features)]!=[1636,49080,6544]:raise ValueError('Input/output expected coverage mismatch')
    destination=root/('preflight' if args.audit_only else 'results')
    save_tables(root/('preflight/inputs' if args.audit_only else 'inputs'),dict(original_predictions=original,perturbed_predictions=raw))
    tables=dict(behavior_features=features,feature_matrix=features.copy());tables.update(measurement_tables(features))
    if args.audit_only:
        tables={name:f[f.condition.str.startswith('K10')].copy() for name,f in tables.items()}
        fitted,parity=run_calibration(config,features[features.condition.str.startswith('K10')],destination);tables.update(fitted)
    else:
        fitted,parity=run_calibration(config,features,destination);tables.update(fitted)
        tables.update(comparative_tables(config,tables['oof_predictions']))
    # Reproduce the historical K10 metrics for all five methods, not merely their OOF predictions.
    old=read(Path(config['source_candidate'])/'results/calibration_metrics.csv');metric_parity=[]
    for _,row in tables['calibration_metrics'].iterrows():
        if row.condition=='K30_pooled':continue
        ref=old[old.dataset.eq(row.dataset)&old.seed.eq(1475)&old.block.eq(int(row.condition[-1]))&old.method.eq(row.method)]
        if len(ref)!=1:raise ValueError('Historical metric identity mismatch')
        for metric in METRICS:
            err=abs(float(row[metric])-float(ref.iloc[0][metric]));metric_parity.append(dict(dataset=row.dataset,condition=row.condition,method=row.method,metric=metric,max_absolute_error=err))
            if err>1e-8:raise ValueError('Historical metric parity failed')
    save_tables(destination,tables)
    if not all(sha(p)==h for p,h in before.items()):raise ValueError('Immutable source/code changed during analysis')
    dump(destination/'source_hashes.json',before)
    dump(destination/'analysis_verification.json',dict(status='preflight_passed' if args.audit_only else 'analysis_complete',all_passed=True,input_code_unchanged=True,cpu_only=True,new_inference=False,new_noise=False,test_labels_used=False,one_row_per_utterance_condition=True,conditions_nested=True,feature_parity=feature_parity,historical_oof_parity=parity,historical_metric_parity=metric_parity,output_rows={n:len(f) for n,f in tables.items()},output_sha256={n+'.csv':sha(destination/(n+'.csv')) for n in tables},elapsed_seconds=time.time()-started,environment=dict(python=sys.executable,host=platform.node(),numpy=np.__version__,pandas=pd.__version__)))
    if not args.audit_only:
        from build_precision_reports import build_reports
        build_reports(root)
    print(f'Complete: {destination}',flush=True)


if __name__=='__main__':main()
