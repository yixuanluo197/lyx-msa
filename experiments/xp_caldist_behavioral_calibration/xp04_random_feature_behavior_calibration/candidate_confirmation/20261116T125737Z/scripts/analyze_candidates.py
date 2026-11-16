"""Preregistered Gaussian candidate confirmation; CPU-only, dialogue OOF."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
for var in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[var] = '1'
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import spearmanr
from mainline_core import FEATURES, METRICS, calibration_metrics, dump, fit_outer, metrics, sha
FEATURES.update(M_gaussian_mu=['gaussian_mu'], M_gaussian_delta=['gaussian_delta'])
KEY = ['dataset', 'seed', 'dialogue_id', 'utterance_index']
CONDITION = ['dataset', 'seed', 'block']
METHODS = ['E0', 'E1', 'M_gaussian', 'M_gaussian_mu', 'M_gaussian_delta']
BEHAVIOR = METHODS[2:]


def read(path, **kw):
    return pd.read_csv(path, dtype={'dataset': str, 'dialogue_id': str}, float_precision='round_trip', **kw)


def unique(frame, key):
    if frame.duplicated(key).any():
        raise ValueError(f'Duplicate identity: {key}')


def validate_p(probabilities, pred, confidence, classes):
    p = np.asarray(probabilities, float)
    if p.ndim != 2 or p.shape[1] != classes or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid probability vectors')
    if not np.allclose(p.sum(1), 1, atol=1e-5, rtol=0):
        raise ValueError('Probability sum invalid')
    if not np.array_equal(p.argmax(1), np.asarray(pred)) or not np.allclose(p.max(1), confidence, atol=1e-12, rtol=0):
        raise ValueError('Prediction/confidence inconsistent with probabilities')
    return p


def response(C, prediction, p, original):
    p, original = np.asarray(p, float), np.asarray(original, float)
    maxp = p.max(1)
    flip = p.argmax(1) != prediction
    middle = (p + original[None, :]) / 2
    js = .5 * np.sum(rel_entr(p, middle) + rel_entr(original[None, :], middle), axis=1)
    out = dict(mu=float(np.mean(flip*maxp)), delta=float(abs(C-maxp.mean())), js=float(js.mean()),
               flip_rate=float(flip.mean()), confidence_change_mean=float(maxp.mean()-C),
               confidence_change_std=float(maxp.std(ddof=0)))
    for side, sl in [('left', slice(0, 5)), ('right', slice(5, 10))]:
        out.update({f'mu_{side}':float((flip*maxp)[sl].mean()),
                    f'delta_{side}':float(abs(C-maxp[sl].mean())), f'js_{side}':float(js[sl].mean())})
    return out


def safe_spearman(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def original_source(config, ds, seed):
    if seed == 1475:
        return Path(config['source_pilot']) / ds / 'original_predictions.csv'
    return Path(config['source_five_seed']) / 'runs' / ds / f'seed{seed}' / 'original_predictions.csv'


def run_specs(config, historical_only=False):
    for ds in config['datasets']:
        for block, block_spec in config['blocks'].items():
            for seed in block_spec['model_seeds']:
                if not historical_only or (int(block) == 0 and seed == 1475):
                    yield ds, int(seed), int(block)


def load_features(root, config, historical_only=False):
    historical = read(Path(config['source_mainline']) / 'inputs/joint_features.csv')
    modality_history = read(Path(config['source_mainline']) / 'inputs/modality_features.csv')
    original_parts, records, source_counts, hashes = [], [], [], {}
    for ds, seed, block in run_specs(config, historical_only):
        spec, modality = config['datasets'][ds], config['datasets'][ds]['modality']
        opath = original_source(config, ds, seed)
        hashes[str(opath)] = sha(opath)
        original = read(opath)
        unique(original, KEY)
        if len(original) != spec['utterances'] or original.dialogue_id.nunique() != spec['dialogues']:
            raise ValueError('Original coverage mismatch')
        if set(original.dataset) != {ds} or set(original.seed) != {seed}:
            raise ValueError('Original dataset/seed mismatch')
        validate_p([json.loads(x) for x in original.probabilities], original.prediction, original.confidence, 6 if ds == 'IEMOCAP' else 7)
        original['correct'] = original.prediction.eq(original.label).astype(int)
        old = historical[historical.dataset.eq(ds) & historical.seed.eq(seed)]
        flags = old[KEY + ['visual_missing']].copy()
        unique(flags, KEY)
        original = original.merge(flags, on=KEY, how='left', validate='one_to_one')
        if original.visual_missing.isna().any() or int(original.visual_missing.sum()) != spec['visual_missing']:
            raise ValueError('Visual missing retention mismatch')
        original['block'], original['modality'] = block, modality
        original_parts.append(original)
        origin = original.set_index(KEY)
        if seed == 1475 and block == 0:
            path = Path(config['source_pilot']) / ds / 'perturbed_predictions.csv'
        else:
            run = root / 'runs' / ds / f'seed{seed}' / f'block{block}'
            for name in ['complete.json', 'original_predictions.csv']:
                if not (run/name).exists():
                    raise FileNotFoundError(run/name)
                hashes[str(run/name)] = sha(run/name)
            run_orig = read(run/'original_predictions.csv')
            joined = original.merge(run_orig, on=KEY, suffixes=('_old','_new'), validate='one_to_one')
            if len(joined) != len(original) or len(run_orig) != len(original):
                raise ValueError('Run original IDs mismatch')
            for c in ['confidence', 'label', 'prediction']:
                if not np.array_equal(joined[c+'_old'].to_numpy(), joined[c+'_new'].to_numpy()):
                    raise ValueError('Canonical original values changed')
            for a,b in zip(joined.probabilities_old, joined.probabilities_new):
                if not np.array_equal(np.asarray(json.loads(a)), np.asarray(json.loads(b))):
                    raise ValueError('Canonical original probabilities changed')
            path = run / 'perturbed_predictions.csv'
        hashes[str(path)] = sha(path)
        raw = read(path)
        raw = raw[raw.modality.eq(modality) & raw.corruption.eq('gaussian') & raw.strength.eq(.05)].copy()
        if len(raw) != len(original)*10 or set(raw.dataset) != {ds} or set(raw.seed) != {seed}:
            raise ValueError('Raw candidate coverage mismatch')
        unique(raw, KEY+['repeat_id'])
        if set(map(tuple,raw[KEY].drop_duplicates().to_numpy())) != set(origin.index):
            raise ValueError('Raw IDs mismatch')
        if seed != 1475 or block:
            if not raw.block.eq(block).all() or not np.array_equal(raw.rng_repeat_id.to_numpy(), raw.repeat_id.to_numpy()+10*block):
                raise ValueError('Random block/repeat identity mismatch')
        if raw.visual_missing_skipped.isna().any() or raw.visual_missing_skipped.astype(bool).any():
            raise ValueError('Text/audio candidate must not skip a visual-missing utterance')
        for key, group in raw.groupby(KEY, sort=False):
            group = group.sort_values('repeat_id')
            if group.repeat_id.tolist() != list(range(10)):
                raise ValueError('Each utterance requires exactly 10 repeats')
            source = origin.loc[key]
            p = validate_p([json.loads(x) for x in group.probabilities], group.prediction, group.confidence, 6 if ds=='IEMOCAP' else 7)
            row = {c:source[c] for c in ['label','prediction','correct','confidence','visual_missing']}
            row.update(dict(zip(KEY,key)), block=block, modality=modality, corruption='gaussian', strength=.05,
                       repeat_count=10, perturbation_skipped=False)
            row.update(response(source.confidence, source.prediction, p, json.loads(source.probabilities)))
            records.append(row)
        source_counts.append(dict(dataset=ds, seed=seed, block=block, path=str(path), rows=len(raw)))
        print(f'Validated raw {ds} seed={seed} block={block} rows={len(raw)}', flush=True)
    behavior = pd.DataFrame(records)
    unique(behavior, KEY+['block'])
    # Historical model-specific order controls GroupKFold tie handling.
    order = historical[KEY].copy()
    order['_historical_order'] = np.arange(len(order))
    features = behavior.merge(order, on=KEY, validate='many_to_one').sort_values(['block','_historical_order'], kind='stable').drop(columns='_historical_order').reset_index(drop=True)
    for col in ['mu','delta','js','flip_rate','confidence_change_mean','confidence_change_std']:
        features['gaussian_'+col] = features[col]
    checks = []
    for ds in config['datasets']:
        a = features[features.dataset.eq(ds)&features.seed.eq(1475)&features.block.eq(0)]
        b = modality_history[modality_history.dataset.eq(ds)&modality_history.modality.eq(config['datasets'][ds]['modality'])]
        joined = a.merge(b, on=KEY, validate='one_to_one', suffixes=('_new','_old'))
        if len(joined) != config['datasets'][ds]['utterances']:
            raise ValueError('Historical behavior coverage mismatch')
        for col in FEATURES['M_gaussian']:
            error = float(np.max(abs(joined[col+'_new']-joined[col+'_old'])))
            if error > 1e-12:
                raise ValueError('Historical behavior derivation mismatch')
            checks.append(dict(dataset=ds, feature=col, max_absolute_error=error))
    return pd.concat(original_parts, ignore_index=True), behavior, features, hashes, dict(source_counts=source_counts, historical_feature_parity=checks)


def descriptive_tables(behavior):
    summaries, stability, bins = [], [], []
    for key, frame in behavior.groupby(CONDITION, sort=False):
        common = dict(zip(CONDITION,key))
        common['modality'] = frame.modality.iloc[0]
        for label, f in [('all',frame),('correct',frame[frame.correct.eq(1)]),('wrong',frame[frame.correct.eq(0)])]:
            row = dict(common, population=label, utterances=len(f), dialogues=f.dialogue_id.nunique(),
                       visual_missing=int(f.visual_missing.sum()), any_flip_fraction=float(f.flip_rate.gt(0).mean()),
                       mu_zero_fraction=float(f.mu.eq(0).mean()))
            for col in ['mu','delta','js','flip_rate','confidence_change_mean','confidence_change_std']:
                row[col+'_mean'] = float(f[col].mean())
                row[col+'_std'] = float(f[col].std(ddof=1))
            summaries.append(row)
        for col in ['mu','delta','js']:
            stability.append(dict(common, measure=col, spearman=safe_spearman(frame[col+'_left'],frame[col+'_right']),
                                  left_mean=float(frame[col+'_left'].mean()), right_mean=float(frame[col+'_right'].mean()),
                                  left_zero_fraction=float(frame[col+'_left'].eq(0).mean()),right_zero_fraction=float(frame[col+'_right'].eq(0).mean())))
        codes=np.minimum(np.floor(frame.confidence.to_numpy()*10).astype(int),9)
        for bi in sorted(set(codes)):
            for correct, f in frame[codes==bi].groupby('correct'):
                bins.append(dict(common, confidence_bin=int(bi), correct=int(correct), utterances=len(f),
                                 confidence=float(f.confidence.mean()),mu=float(f.mu.mean()),delta=float(f.delta.mean()),js=float(f.js.mean())))
    return dict(behavior_summary=pd.DataFrame(summaries),split_half_stability=pd.DataFrame(stability),conditional_behavior=pd.DataFrame(bins))


def cross_block_stability(behavior):
    rows=[]
    columns=['dataset','seed','modality','left_block','right_block','measure','utterances','spearman','left_zero_fraction','right_zero_fraction','left_nonzero_fraction','right_nonzero_fraction']
    for ds,frame in behavior[behavior.seed.eq(1475)].groupby('dataset',sort=False):
        blocks=sorted(frame.block.unique())
        for i,left_block in enumerate(blocks):
            for right_block in blocks[i+1:]:
                left=frame[frame.block.eq(left_block)].set_index(KEY)
                right=frame[frame.block.eq(right_block)].set_index(KEY).reindex(left.index)
                if len(left)!=len(right) or right[['mu','delta','js']].isna().any().any():raise ValueError('Cross-block identity mismatch')
                for measure in ['mu','delta','js']:
                    a,b=left[measure].to_numpy(float),right[measure].to_numpy(float)
                    rows.append(dict(dataset=ds,seed=1475,modality=str(left.modality.iloc[0]),left_block=int(left_block),right_block=int(right_block),measure=measure,utterances=len(left),spearman=safe_spearman(a,b),left_zero_fraction=float(np.mean(a==0)),right_zero_fraction=float(np.mean(b==0)),left_nonzero_fraction=float(np.mean(a!=0)),right_nonzero_fraction=float(np.mean(b!=0))))
    return pd.DataFrame(rows,columns=columns)


def native_condition(key):
    return dict(dataset=str(key[0]), seed=int(key[1]), block=int(key[2]))


def run_calibration(root, config, features, destination):
    folds = json.loads((Path(config['source_five_seed'])/'baseline/fold_parameters.json').read_text())
    historical = read(Path(config['source_mainline'])/'joint/oof_predictions.csv')
    strength = read(Path(config['source_strength'])/'results/oof_predictions.csv')
    outputs, metric_rows, shifts, selections, parity, dialogues = [], [], [], [], [], []
    for key, frame in features.groupby(CONDITION, sort=False):
        common=native_condition(key); common.update(modality=str(frame.modality.iloc[0]),strength=.05)
        frame=frame.reset_index(drop=True)
        parts, params = [], []
        covered=np.zeros(len(frame),int)
        for fold in [f for f in folds if f['dataset']==common['dataset'] and f['seed']==int(common['seed'])]:
            train=np.flatnonzero(frame.dialogue_id.isin(fold['train_dialogues']))
            held=np.flatnonzero(frame.dialogue_id.isin(fold['held_dialogues']))
            if len(train)!=fold['train_n'] or len(held)!=fold['held_n'] or len(train)+len(held)!=len(frame):
                raise ValueError('Historical dialogue fold mismatch')
            scored,fitted=fit_outer(frame,train,held,BEHAVIOR)
            fitted.update(common,fold=fold['fold']); params.append(fitted); covered[held]+=1
            scored['E0']=dict(oof_confidence=frame.iloc[held].confidence.to_numpy(float),base_score=np.full(len(held),np.nan),behavior_score=np.zeros(len(held)),reliability_score=np.full(len(held),np.nan))
            for method in METHODS:
                part=frame.iloc[held][KEY+['block','modality','label','prediction','correct','confidence','visual_missing','perturbation_skipped']].copy()
                part['method'],part['fold']=method,fold['fold']
                part['selected_penalty']=fitted['methods'][method]['selected_penalty'] if method in BEHAVIOR else ('off' if method=='E1' else 'none')
                for col,values in scored[method].items(): part[col]=values
                parts.append(part)
            for method, selection in fitted['methods'].items():
                for penalty, loss in selection['candidate_brier'].items():
                    selections.append(dict(common,fold=fold['fold'],method=method,penalty=penalty,pooled_inner_brier=loss,selected=str(selection['selected_penalty'])==penalty))
        if not np.all(covered==1): raise ValueError('OOF coverage !=1')
        predictions=pd.concat(parts,ignore_index=True); outputs.append(predictions)
        tag=f"{common['dataset']}_seed{common['seed']}_block{common['block']}"
        dump(destination/'fold_parameters'/f'{tag}.json',params)
        for method,f in predictions.groupby('method',sort=False):
            metric_rows.append(dict(common,method=method,**metrics(f,f.oof_confidence.to_numpy(float))))
            for correct,p in f.groupby('correct'):
                change=p.oof_confidence-p.confidence
                shifts.append(dict(common,method=method,correct=int(correct),utterances=len(p),original_mean_confidence=float(p.confidence.mean()),calibrated_mean_confidence=float(p.oof_confidence.mean()),mean_change=float(change.mean()),raised_fraction=float((change>1e-12).mean()),lowered_fraction=float((change< -1e-12).mean())))
            for dialogue,p in f.groupby('dialogue_id',sort=False):
                dialogues.append(dict(common,method=method,dialogue_id=dialogue,**metrics(p,p.oof_confidence.to_numpy(float))) if p.correct.nunique()==2 else dict(common,method=method,dialogue_id=dialogue,utterances=len(p),**calibration_metrics(p.oof_confidence,p.correct)))
        reference=historical[historical.dataset.eq(common['dataset'])&historical.seed.eq(common['seed'])&historical.method.eq('E1')]
        refs=[('E1',reference)]
        if common['seed']==1475 and common['block']==0:
            refs.append(('M_gaussian',strength[strength.dataset.eq(common['dataset'])&strength.modality.eq(common['modality'])&strength.strength.eq(.05)&strength.method.eq('M_gaussian')]))
        for method, reference in refs:
            joined=predictions[predictions.method.eq(method)].merge(reference[KEY+['oof_confidence']],on=KEY,validate='one_to_one',suffixes=('_new','_old'))
            error=float(np.max(abs(joined.oof_confidence_new-joined.oof_confidence_old)))
            if len(joined)!=len(frame) or error>1e-8: raise ValueError(f'Historical OOF parity failed {tag} {method}: {error}')
            parity.append(dict(common,method=method,rows=len(joined),max_absolute_error=error,tolerance=1e-8))
        print(f'OOF complete {tag}',flush=True)
    return dict(oof_predictions=pd.concat(outputs,ignore_index=True),calibration_metrics=pd.DataFrame(metric_rows),confidence_shift=pd.DataFrame(shifts),inner_cv_results=pd.DataFrame(selections),per_dialogue_results=pd.DataFrame(dialogues)),parity


def load_draws(config,ds):
    with np.load(Path(config['source_five_seed'])/f'baseline/bootstrap/{ds}_dialogue_draws.npz') as f:
        ids,draws=f['dialogue_ids'].astype(str),f['draws']
    if draws.shape!=(2000,config['datasets'][ds]['dialogues']): raise ValueError('Bootstrap registry shape mismatch')
    return ids,np.stack([np.bincount(row,minlength=len(ids)) for row in draws])


def bootstrap_distributions(frame,ids,multiplicities):
    idx=pd.Index(ids).get_indexer(frame.dialogue_id)
    if (idx<0).any(): raise ValueError('Bootstrap dialogue missing')
    total=multiplicities@np.bincount(idx,minlength=len(ids))
    q,y=frame.oof_confidence.to_numpy(float),frame.correct.to_numpy(int)
    residual=np.zeros((len(ids),15)); bi=np.minimum(np.floor(q*15).astype(int),14)
    np.add.at(residual,(idx,bi),q-y)
    clip=np.clip(q,1e-15,1-1e-15)
    return dict(ece15=np.abs(multiplicities@residual).sum(1)/total,
                binary_brier=multiplicities@np.bincount(idx,weights=(q-y)**2,minlength=len(ids))/total,
                binary_nll=multiplicities@np.bincount(idx,weights=-y*np.log(clip)-(1-y)*np.log1p(-clip),minlength=len(ids))/total)


def calibration_bootstrap(config,predictions):
    rows=[]
    for ds in config['datasets']:
        ids,multiplicities=load_draws(config,ds)
        subset=predictions[predictions.dataset.eq(ds)]
        for block,part in subset.groupby('block',sort=False):
            dist,points={},{}
            seeds=list(part.seed.unique())
            for (seed,method),f in part.groupby(['seed','method'],sort=False):
                dist[seed,method]=bootstrap_distributions(f,ids,multiplicities)
                points[seed,method]=calibration_metrics(f.oof_confidence,f.correct)
            comparisons=[(m,r) for m in BEHAVIOR for r in ['E0','E1']]+[('E1','E0'),('M_gaussian','M_gaussian_mu'),('M_gaussian','M_gaussian_delta')]
            for candidate,reference in comparisons:
                for metric in METRICS:
                    differences=[]; point_values=[]
                    for seed in seeds:
                        diff=dist[seed,candidate][metric]-dist[seed,reference][metric]
                        point=points[seed,candidate][metric]-points[seed,reference][metric]
                        differences.append(diff);point_values.append(point)
                        lo,hi=np.quantile(diff,[.025,.975])
                        rows.append(dict(dataset=ds,block=int(block),seed=int(seed),scope='per_seed',candidate=candidate,reference=reference,metric=metric,mean_difference=point,ci_low=float(lo),ci_high=float(hi),bootstrap_fraction_improving=float((diff<0).mean()),resamples=2000,seeds=len(seeds),refit=False,unit='dialogue',exploratory=True))
                    if len(seeds)>1:
                        diff=np.mean(differences,axis=0);lo,hi=np.quantile(diff,[.025,.975])
                        rows.append(dict(dataset=ds,block=int(block),seed='all',scope='mean_across_paired_seeds',candidate=candidate,reference=reference,metric=metric,mean_difference=float(np.mean(point_values)),ci_low=float(lo),ci_high=float(hi),bootstrap_fraction_improving=float((diff<0).mean()),resamples=2000,seeds=len(seeds),refit=False,unit='shared_dialogue_draw_across_seeds',exploratory=True))
    return pd.DataFrame(rows)


def aggregate_tables(tables):
    metrics_frame=tables['calibration_metrics']; rows=[]
    for (ds,method),f in metrics_frame[metrics_frame.block.eq(0)].groupby(['dataset','method'],sort=False):
        base=metrics_frame[metrics_frame.dataset.eq(ds)&metrics_frame.block.eq(0)&metrics_frame.method.eq('E1')].set_index('seed')
        row=dict(dataset=ds,block=0,method=method,seeds=len(f))
        for metric in METRICS+['correctness_auc','mean_confidence']:
            row[metric+'_mean']=float(f[metric].mean());row[metric+'_sample_std']=float(f[metric].std(ddof=1))
            diff=f.set_index('seed')[metric]-base[metric]
            row[metric+'_difference_vs_E1']=float(diff.mean())
            row[metric+'_improving_seeds']=int((diff<0).sum()) if metric in METRICS else int((diff>0).sum())
        row['all_three_improving_seeds']=int(((f.set_index('seed')[METRICS]-base[METRICS])<0).all(axis=1).sum())
        rows.append(row)
    selected=tables['oof_predictions'][KEY[:2]+['block','fold','method','selected_penalty']].drop_duplicates()
    selected=selected[selected.method.isin(BEHAVIOR)]
    counts=selected.groupby(CONDITION+['method','selected_penalty'],sort=False).size().rename('outer_folds').reset_index()
    blocks=[]
    for (ds,method),f in metrics_frame[metrics_frame.seed.eq(1475)].groupby(['dataset','method'],sort=False):
        base=metrics_frame[metrics_frame.dataset.eq(ds)&metrics_frame.seed.eq(1475)&metrics_frame.method.eq('E1')].set_index('block')
        for _,p in f.iterrows():
            row=dict(dataset=ds,seed=1475,block=int(p.block),method=method)
            for metric in METRICS:row[metric]=p[metric];row[metric+'_difference_vs_E1']=p[metric]-base.loc[p.block,metric]
            blocks.append(row)
    return dict(aggregate_metrics=pd.DataFrame(rows),behavior_selection_counts=counts,random_block_comparison=pd.DataFrame(blocks))


def dialogue_concentration(predictions):
    """Fixed-OOF leave-one-dialogue-out influence, without any refitting."""
    rows=[]
    for key,part in predictions.groupby(CONDITION,sort=False):
        common=native_condition(key)
        base=part[part.method.eq('E1')].set_index(KEY)
        for method in BEHAVIOR:
            f=part[part.method.eq(method)].set_index(KEY)
            if not f.index.equals(base.index):
                f=f.reindex(base.index)
            if f.oof_confidence.isna().any():raise ValueError('Dialogue influence pairing mismatch')
            q,b,y=f.oof_confidence.to_numpy(float),base.oof_confidence.to_numpy(float),f.correct.to_numpy(int)
            dialogues=f.index.get_level_values('dialogue_id').to_numpy()
            full_a,full_b=calibration_metrics(q,y),calibration_metrics(b,y)
            for dialogue in dict.fromkeys(dialogues):
                held=dialogues==dialogue;keep=~held
                omit_a,omit_b=calibration_metrics(q[keep],y[keep]),calibration_metrics(b[keep],y[keep])
                local_a,local_b=calibration_metrics(q[held],y[held]),calibration_metrics(b[held],y[held])
                for metric in METRICS:
                    rows.append(dict(common,method=method,reference='E1',dialogue_id=str(dialogue),metric=metric,
                                     dialogue_utterances=int(held.sum()),full_difference=full_a[metric]-full_b[metric],
                                     omitted_difference=omit_a[metric]-omit_b[metric],dialogue_difference=local_a[metric]-local_b[metric],
                                     refit=False,scope='per_seed'))
    frame=pd.DataFrame(rows)
    paired=[]
    for key,f in frame[frame.block.eq(0)].groupby(['dataset','block','method','reference','dialogue_id','metric'],sort=False):
        if f.seed.nunique()>1:
            row=dict(zip(['dataset','block','method','reference','dialogue_id','metric'],key))
            row.update(seed='all',scope='mean_across_paired_seeds',refit=False,seeds=int(f.seed.nunique()),dialogue_utterances=int(f.dialogue_utterances.iloc[0]))
            for col in ['full_difference','omitted_difference','dialogue_difference']:row[col]=float(f[col].mean())
            paired.append(row)
    if paired:frame=pd.concat([frame,pd.DataFrame(paired)],ignore_index=True)
    summary=[]
    for key,f in frame.groupby(['dataset','block','seed','scope','method','metric'],sort=False):
        row=dict(zip(['dataset','block','seed','scope','method','metric'],key))
        full=float(f.full_difference.iloc[0]);omitted=f.omitted_difference.to_numpy(float)
        row.update(full_difference=full,dialogues=len(f),dialogues_improving=int((f.dialogue_difference<0).sum()),
                   omission_difference_min=float(omitted.min()),omission_difference_max=float(omitted.max()),
                   omissions_with_improvement=int((omitted<0).sum()),
                   omissions_reversing_full_direction=int(((omitted>0) if full<0 else (omitted<0)).sum()),refit=False)
        summary.append(row)
    return dict(dialogue_influence=frame,dialogue_concentration_summary=pd.DataFrame(summary))


def save_tables(destination,tables):
    destination.mkdir(parents=True,exist_ok=True)
    for name,frame in tables.items():
        frame.to_csv(destination/f'{name}.csv',index=False,float_format='%.17g',na_rep='NA')
        print(f'{name}: {len(frame)} rows',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--historical-only',action='store_true')
    args=parser.parse_args();root=args.root.resolve();config=json.loads((root/'preregistration.json').read_text())
    if config['methods']!=METHODS or config['penalties']!=['off',1.,.1,.01]:raise ValueError('Protocol changed')
    started=time.time();destination=root/('historical_preflight' if args.historical_only else 'results')
    static=[root/'preregistration.json',root/'EXPERIMENT_PLAN.md',Path(config['source_mainline'])/'inputs/joint_features.csv',Path(config['source_mainline'])/'inputs/modality_features.csv',Path(config['source_mainline'])/'joint/oof_predictions.csv',Path(config['source_strength'])/'results/oof_predictions.csv',Path(config['source_five_seed'])/'baseline/fold_parameters.json']
    static += [root/'scripts'/name for name in ['analyze_candidates.py','mainline_core.py','test_candidate_analysis.py']]
    if not args.historical_only:
        static.append(root/'scripts/build_candidate_reports.py')
    static+=[Path(config['source_five_seed'])/f'baseline/bootstrap/{ds}_dialogue_draws.npz' for ds in config['datasets']]
    before={str(p):sha(p) for p in static}
    originals,behavior,features,input_hashes,verification=load_features(root,config,args.historical_only)
    before.update(input_hashes)
    tables=dict(original_predictions=originals,behavior_features=behavior,feature_matrix=features)
    tables.update(descriptive_tables(behavior))
    tables['cross_block_stability']=cross_block_stability(behavior)
    fitted,parity=run_calibration(root,config,features,destination);tables.update(fitted)
    tables.update(aggregate_tables(tables))
    tables.update(dialogue_concentration(tables['oof_predictions']))
    tables['bootstrap_results']=calibration_bootstrap(config,tables['oof_predictions'])
    save_tables(destination,tables)
    if not all(sha(p)==h for p,h in before.items()):raise ValueError('Analysis input/code mutation')
    dump(destination/'source_hashes.json',before)
    verification.update(status='historical_preflight_passed' if args.historical_only else 'analysis_complete',all_passed=True,historical_oof_parity=parity,output_rows={name:len(f) for name,f in tables.items()},output_sha256={name+'.csv':sha(destination/(name+'.csv')) for name in tables},input_code_unchanged=True,cpu_only=True,test_labels_used=False,cmerc_training=False,one_utterance_per_seed_block=True,blocks_not_pooled=True,no_js_or_class_inputs=True,bootstrap_refit=False,bootstrap_shared_across_seeds=True,elapsed_seconds=time.time()-started,environment=dict(python=sys.executable,host=platform.node(),numpy=np.__version__,pandas=pd.__version__))
    dump(destination/'analysis_verification.json',verification)
    if not args.historical_only:
        from build_candidate_reports import build_reports
        build_reports(root)
    print(f'Analysis complete {destination}',flush=True)

if __name__=='__main__':main()
