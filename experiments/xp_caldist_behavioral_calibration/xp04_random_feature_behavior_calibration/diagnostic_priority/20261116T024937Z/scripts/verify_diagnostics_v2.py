#!/usr/bin/env python3
"""Independent numerical audit; never imports the diagnostic implementation."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [1475, 67137, 2024, 3407, 42]
EDGES = np.array([0, .4, .6, .8, 1.])


def optimal_matching(a, b, caliper=.02):
    """Independent 1D noncrossing DP: maximize cardinality, then minimize L1."""
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    counts = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    costs = np.zeros_like(counts, dtype=float)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            candidates = [(counts[i-1, j], costs[i-1, j]),
                          (counts[i, j-1], costs[i, j-1])]
            distance = abs(a[i-1] - b[j-1])
            if distance <= caliper:
                candidates.append((counts[i-1, j-1]+1,
                                   costs[i-1, j-1]+distance))
            best = min(candidates, key=lambda x: (-x[0], x[1]))
            counts[i, j], costs[i, j] = best
    return int(counts[-1, -1]), float(costs[-1, -1])


def brute_matching(a, b, caliper=.02):
    best = (0, 0.)
    def visit(i, used, count, cost):
        nonlocal best
        if i == len(a):
            if (-count, cost) < (-best[0], best[1]):
                best = count, cost
            return
        visit(i+1, used, count, cost)
        for j in range(len(b)):
            d = abs(a[i]-b[j])
            if j not in used and d <= caliper:
                visit(i+1, used | {j}, count+1, cost+d)
    visit(0, set(), 0, 0.)
    return best


def auc_larger_is_error(score, correct):
    error = 1 - np.asarray(correct)
    n1, n0 = int(error.sum()), int((1-error).sum())
    if not n1 or not n0:
        return float('nan')
    return (rankdata(score)[error == 1].sum()-n1*(n1+1)/2)/(n1*n0)


def correlation(a, b, ranked=False):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float('nan')
    if ranked:
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def fixed_bin(values):
    return np.minimum(np.searchsorted(EDGES,values,side='right')-1,3)


def summarize(values):
    a=np.asarray(values,dtype=float)
    if not len(a):
        return dict(n=0,mean=np.nan,median=np.nan,sample_sd=np.nan,q05=np.nan,
                    q25=np.nan,q75=np.nan,q95=np.nan,zero_fraction=np.nan,
                    n_unique=0,minimum=np.nan,maximum=np.nan)
    q=np.quantile(a,[.05,.25,.75,.95])
    return dict(n=len(a),n_unique=len(np.unique(a)),minimum=np.min(a),maximum=np.max(a),mean=np.mean(a),median=np.median(a),
                sample_sd=np.std(a,ddof=1) if len(a)>1 else np.nan,
                q05=q[0],q25=q[1],q75=q[2],q95=q[3],zero_fraction=np.mean(a==0))


def selftest():
    rng = np.random.default_rng(941)
    cases = [([.1, .12], [.11, .13]), ([0, .2], [.02, .18]),
             ([], [.4]), ([.1, .1], [.1]), ([.1], [.5])]
    cases += [(rng.uniform(0, .1, rng.integers(1, 5)),
               rng.uniform(0, .1, rng.integers(1, 5))) for _ in range(80)]
    for a, b in cases:
        n, cost = optimal_matching(a, b)
        nb, cb = brute_matching(a, b)
        assert n == nb and math.isclose(cost, cb, abs_tol=1e-12)
    assert auc_larger_is_error([0, 1, 2, 3], [1, 1, 0, 0]) == 1
    assert auc_larger_is_error([0, 1, 2, 3], [0, 0, 1, 1]) == 0
    assert auc_larger_is_error([0, 0, 0, 0], [1, 1, 0, 0]) == .5
    assert np.isnan(correlation([0, 0], [1, 2]))
    assert np.isnan(correlation([1, 2], [0, 0], True))
    assert np.isclose(3 * np.sqrt(10/9) / np.sqrt(10), 3/np.sqrt(9))
    weighted_mu=np.mean(np.array([1,0,1,0])*np.array([.7,.8,.4,.6]))
    flip_rate=np.mean([1,0,1,0])
    assert 0 < weighted_mu < flip_rate and weighted_mu != flip_rate
    assert np.array_equal(fixed_bin([0,.399,.4,.6,.8,1]),[0,0,1,2,3,3])
    assert np.isclose(np.mean([np.mean([1,3]),np.mean([8])]),5)
    assert not np.all(np.array([2,0,4,3,1])>0)
    return {'matching_bruteforce_cases': len(cases), 'auc_direction_cases': 3,
            'constant_correlation_cases': 2, 'noise_identity_cases': 1,
            'fixed_bin_boundaries':6,'seed_equal_weight_and_invalid_cases':2,'weighted_mu_distinct_from_flip_rate_cases':1}


class Audit:
    def __init__(self):
        self.checks = []
    def check(self, name, condition, detail=None):
        entry = {'name': name, 'passed': bool(condition)}
        if detail is not None:
            entry['detail'] = detail
        self.checks.append(entry)
        if not condition:
            raise AssertionError(f'{name}: {detail}')
    def close(self, name, actual, expected, atol=1e-10):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.check(name, a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=1e-8, equal_nan=True),
                   {'n': int(a.size), 'max_absolute_error': float(np.nanmax(np.abs(a-b)))
                    if a.size and np.any(np.isfinite(a-b)) else None})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(name):
    return pd.read_csv(ROOT / 'results' / name)


def compare_keyed(audit, name, actual, expected, keys, columns):
    audit.check(name+'_rows', len(actual) == len(expected), {'actual':len(actual),'expected':len(expected)})
    audit.check(name+'_unique', not actual.duplicated(keys).any())
    merged = expected.merge(actual, on=keys, suffixes=('_expected','_actual'), how='outer', indicator=True)
    audit.check(name+'_keys', (merged['_merge'] == 'both').all())
    for c in columns:
        audit.close(name+'_'+c, merged[c+'_actual'], merged[c+'_expected'])


def run_audit(audit):
    features = pd.read_csv(ROOT/'inputs/features.csv')
    split = pd.read_csv(ROOT/'inputs/split_responses.csv')
    audit.check('canonical_8180_rows', len(features)==8180)
    audit.check('canonical_unique_keys', not features.duplicated(['dataset','seed','dialogue_id','utterance_index']).any())
    audit.check('split_49080_rows', len(split)==49080)
    audit.check('seeds_frozen', set(features.seed)==set(SEEDS))
    audit.check('correct_matches_labels', (features.correct == (features.label == features.prediction)).all())
    audit.check('confidence_in_unit_interval', features.confidence.between(0,1).all())
    for corrupt in ['gaussian','dropout']:
        audit.close('delta_abs_signed_change_'+corrupt,features[corrupt+'_delta'],np.abs(features[corrupt+'_confidence_change_mean']))
        audit.check('weighted_mu_bounds_'+corrupt,((features[corrupt+'_mu']>=0)&(features[corrupt+'_mu']<=features[corrupt+'_flip_rate']+1e-12)).all())
        audit.check('weighted_mu_and_flip_rate_zero_pattern_'+corrupt,((features[corrupt+'_mu']==0)==(features[corrupt+'_flip_rate']==0)).all())
        audit.check('mu_probability_'+corrupt,features[corrupt+'_mu'].between(0,1).all())
        audit.check('noise_population_sd_nonnegative_'+corrupt,(features[corrupt+'_confidence_change_std']>=0).all())
    for ds, (n, nd) in {'IEMOCAP':(527,12),'MELD':(1109,114)}.items():
        for seed in SEEDS:
            f = features[(features.dataset==ds)&(features.seed==seed)]
            audit.check(f'canonical_support_{ds}_{seed}', len(f)==n and f.dialogue_id.nunique()==nd)
    # Filled below with output-specific numerical contracts after schema agreement.
    audit_outputs(audit, features, split)
    return {'input_rows':len(features), 'split_rows':len(split)}


def audit_outputs(audit, features, split):
    audit_correlations(audit, split)
    audit_matching(audit, features)
    audit_descriptions(audit, features)


def audit_correlations(audit, split):
    keys=['dataset','seed','corruption','split','measure']
    rows=[]
    for (ds,seed,corrupt,s), group in split.groupby(keys[:-1]):
        for measure in ['mu','delta']:
            a,b=group[measure+'_left'],group[measure+'_right']
            p,r=correlation(a,b),correlation(a,b,True)
            rows.append(dict(zip(keys,[ds,seed,corrupt,s,measure]), n=len(group),
                             pearson=p,spearman=r,correlation_estimable=np.isfinite(p),
                             left_zero_fraction=np.mean(a==0),right_zero_fraction=np.mean(b==0)))
    expected=pd.DataFrame(rows)
    columns=['n','pearson','spearman','correlation_estimable','left_zero_fraction','right_zero_fraction']
    compare_keyed(audit,'split_correlations',load_csv('split_correlations.csv'),expected,keys,columns)
    compare_keyed(audit,'reference_split_reliability',pd.read_csv(ROOT/'inputs/reference_split_reliability.csv'),expected,keys,columns)
    seedrows=[]
    for key,g in expected.groupby(keys[:3]+['measure']):
        row=dict(zip(keys[:3]+['measure'],key),n_splits=len(g))
        for corr in ['pearson','spearman']:
            row[corr]=g[corr].mean()
            row['n_'+corr+'_valid']=g[corr].notna().sum()
        seedrows.append(row)
    se=pd.DataFrame(seedrows)
    cols=['n_splits','pearson','spearman','n_pearson_valid','n_spearman_valid']
    compare_keyed(audit,'stability_seed',load_csv('stability_seed.csv'),se,keys[:3]+['measure'],cols)
    summary=[]
    for key,g in se.groupby(['dataset','corruption','measure']):
        row=dict(zip(['dataset','corruption','measure'],key),n_seeds_expected=5)
        for corr in ['pearson','spearman']:
            row[corr+'_mean']=g[corr].mean()
            row[corr+'_sample_sd']=g[corr].std(ddof=1)
            row['n_'+corr+'_valid']=g[corr].notna().sum()
        summary.append(row)
    cols=[c for c in summary[0] if c not in ['dataset','corruption','measure']]
    compare_keyed(audit,'stability_summary',load_csv('stability_summary.csv'),pd.DataFrame(summary),['dataset','corruption','measure'],cols)


def audit_matching(audit, features):
    pairs=load_csv('matched_pairs.csv')
    key=['dataset','seed','dialogue_id']
    for label in ['correct','wrong']:
        audit.check('pairs_unique_'+label,not pairs.duplicated(key+[label+'_utterance_index']).any())
    allused=pd.concat([pairs[key+['correct_utterance_index']].rename(columns={'correct_utterance_index':'utterance_index'}),
                       pairs[key+['wrong_utterance_index']].rename(columns={'wrong_utterance_index':'utterance_index'})])
    audit.check('pairs_no_replacement_across_classes',not allused.duplicated().any())
    for label,correct in [('correct',1),('wrong',0)]:
        left=pairs.rename(columns={label+'_utterance_index':'utterance_index'})
        merged=left.merge(features,on=key+['utterance_index'],how='left',validate='many_to_one',indicator=True)
        audit.check('matched_source_'+label,(merged['_merge']=='both').all() and (merged.correct==correct).all())
        audit.close('matched_label_'+label,merged[label+'_label'],merged.label)
        audit.close('matched_prediction_'+label,merged[label+'_prediction'],merged.prediction)
        audit.close('matched_confidence_'+label,merged[label+'_rawC'],merged.confidence)
        for corrupt,measure in itertools.product(['gaussian','dropout'],['mu','delta']):
            audit.close('matched_value_'+label+'_'+corrupt+'_'+measure,
                        merged[f'{corrupt}_{measure}_{label}'],merged[f'{corrupt}_{measure}'])
    diff=pairs.wrong_rawC-pairs.correct_rawC
    audit.close('pair_confidence_signed_difference',pairs.rawC_wrong_minus_correct,diff)
    audit.close('pair_confidence_distance',pairs.rawC_abs_distance,np.abs(diff))
    audit.check('pair_caliper',(np.abs(diff)<=.02).all())
    objectives=[]
    for (ds,seed,dialogue),g in features.groupby(key):
        p=pairs[(pairs.dataset==ds)&(pairs.seed==seed)&(pairs.dialogue_id==dialogue)]
        n,c=optimal_matching(g.loc[g.correct==1,'confidence'],g.loc[g.correct==0,'confidence'])
        objectives.append((len(p)==n,np.isclose(p.rawC_abs_distance.sum(),c,atol=1e-12,rtol=1e-10)))
    audit.check('all_dialogue_maximum_cardinality_minimum_distance',all(all(x) for x in objectives),{'dialogue_seed_groups':len(objectives)})
    expected=[]
    for (ds,seed),f in features.groupby(['dataset','seed']):
        p=pairs[(pairs.dataset==ds)&(pairs.seed==seed)]
        for corrupt,measure in itertools.product(['gaussian','dropout'],['mu','delta']):
            c,w=p[f'{corrupt}_{measure}_correct'],p[f'{corrupt}_{measure}_wrong']
            d=w-c
            audit.close(f'pair_difference_{ds}_{seed}_{corrupt}_{measure}',p[f'{corrupt}_{measure}_difference'],d)
            expected.append(dict(dataset=ds,seed=seed,corruption=corrupt,measure=measure,
                                 n_pairs=len(p),n_dialogues=p.dialogue_id.nunique(),
                                 n_correct_available=int(f.correct.sum()),n_wrong_available=int((f.correct==0).sum()),
                                 n_dialogues_available=f.dialogue_id.nunique(),
                                 matched_correct_fraction=len(p)/int(f.correct.sum()),matched_wrong_fraction=len(p)/int((f.correct==0).sum()),
                                 mean_wrong_minus_correct=d.mean(),sample_sd_difference=d.std(ddof=1),median_difference=d.median(),
                                 positive_fraction=(d>0).mean(),negative_fraction=(d<0).mean(),tie_fraction=(d==0).mean(),
                                 tie_win_fraction=(d>0).mean()+.5*(d==0).mean(),correct_mean=c.mean(),wrong_mean=w.mean(),
                                 mean_confidence_difference=p.rawC_wrong_minus_correct.mean(),mean_abs_confidence_distance=p.rawC_abs_distance.mean(),
                                 max_confidence_distance=p.rawC_abs_distance.max(),insufficient_support=len(p)<10 or p.dialogue_id.nunique()<3))
    keys=['dataset','seed','corruption','measure']
    ex=pd.DataFrame(expected)
    compare_keyed(audit,'matched_seed_statistics',load_csv('matched_seed_statistics.csv'),ex,keys,[c for c in ex if c not in keys])
    coverage_columns=['dataset','seed','n_correct_available','n_wrong_available','n_dialogues_available','n_pairs','n_dialogues',
                      'matched_correct_fraction','matched_wrong_fraction','mean_confidence_difference',
                      'mean_abs_confidence_distance','max_confidence_distance','insufficient_support']
    coverage=ex[coverage_columns].drop_duplicates()
    compare_keyed(audit,'matched_coverage',load_csv('matched_coverage.csv'),coverage,['dataset','seed'],coverage_columns[2:])
    audit_bootstrap(audit,pairs,ex)


def audit_bootstrap(audit,pairs,seedstats):
    rows=[]
    alldraws=[]
    for ds in ['IEMOCAP','MELD']:
        z=np.load(ROOT/f'inputs/{ds}_dialogue_draws.npz',allow_pickle=False)
        ids=z['dialogue_ids'].astype(str)
        draws=z['draws']
        audit.check('frozen_draw_shape_'+ds,draws.shape==(2000,len(ids)))
        audit.check('frozen_draw_indices_'+ds,np.all((draws>=0)&(draws<len(ids))))
        weights=np.array([(draws==j).sum(axis=1) for j in range(len(ids))]).T
        audit.check('draws_total_weight_'+ds,np.all(weights.sum(axis=1)==len(ids)))
        for corrupt,measure in itertools.product(['gaussian','dropout'],['mu','delta']):
            seedvalues=[]
            denominators=[]
            for seed in SEEDS:
                p=pairs[(pairs.dataset==ds)&(pairs.seed==seed)]
                groups=p.groupby('dialogue_id')[f'{corrupt}_{measure}_difference'].agg(['sum','count']).reindex(ids,fill_value=0)
                numerator=weights@groups['sum'].to_numpy()
                denominator=weights@groups['count'].to_numpy()
                value=np.divide(numerator,denominator,out=np.full(2000,np.nan),where=denominator>0)
                seedvalues.append(value)
                denominators.append(denominator)
            valid=np.all(np.asarray(denominators)>0,axis=0)
            stats=np.mean(np.asarray(seedvalues),axis=0)
            audit.check('invalid_draw_statistics_NA_'+ds+'_'+corrupt+'_'+measure,np.isnan(stats[~valid]).all())
            s=seedstats[(seedstats.dataset==ds)&(seedstats.corruption==corrupt)&(seedstats.measure==measure)]
            low,high=np.quantile(stats[valid],[.025,.975]) if np.any(valid) else (np.nan,np.nan)
            rows.append(dict(dataset=ds,corruption=corrupt,measure=measure,point_estimate=s.mean_wrong_minus_correct.mean(),
                             ci_low=low,ci_high=high,n_valid=int(valid.sum()),n_invalid=int((~valid).sum()),n_resamples=2000,n_seeds=5,
                             min_seed_pairs=s.n_pairs.min(),min_seed_dialogues=s.n_dialogues.min(),any_low_support=s.insufficient_support.any()))
            for i in range(2000):
                alldraws.append(dict(dataset=ds,corruption=corrupt,measure=measure,draw_index=i,valid=valid[i],statistic=stats[i]))
    keys=['dataset','corruption','measure']
    expected=pd.DataFrame(rows)
    compare_keyed(audit,'matched_bootstrap',load_csv('matched_bootstrap.csv'),expected,keys,[c for c in expected if c not in keys])
    compare_keyed(audit,'matched_bootstrap_draws',load_csv('matched_bootstrap_draws.csv'),pd.DataFrame(alldraws),keys+['draw_index'],['valid','statistic'])


def audit_descriptions(audit, features):
    distributions,bins,risks,noise_summaries=[],[],[],[]
    noise=[]
    for (ds,seed),g in features.groupby(['dataset','seed']):
        for corrupt in ['gaussian','dropout']:
            for measure in ['mu','delta','flip_rate']:
                for class_name in ['all','correct','wrong']:
                    c=g if class_name=='all' else g[g.correct==(class_name=='correct')]
                    distributions.append(dict(dataset=ds,seed=seed,corruption=corrupt,measure=measure,class_name=class_name,
                                              dialogues=c.dialogue_id.nunique(),rawC_mean=c.confidence.mean(),**summarize(c[corrupt+'_'+measure])))
            for measure in ['mu','delta']:
                risks.append(dict(dataset=ds,seed=seed,corruption=corrupt,measure=measure,n=len(g),
                                  n_correct=g.correct.sum(),n_wrong=(g.correct==0).sum(),dialogues=g.dialogue_id.nunique(),
                                  auc_larger_is_error=auc_larger_is_error(g[corrupt+'_'+measure],g.correct)))
                for b in range(4):
                    f=g[fixed_bin(g.confidence)==b]
                    c,w=f[f.correct==1],f[f.correct==0]
                    row=dict(dataset=ds,seed=seed,corruption=corrupt,measure=measure,bin_index=b,bin_left=EDGES[b],bin_right=EDGES[b+1],
                             n_correct=len(c),n_wrong=len(w),dialogues=f.dialogue_id.nunique(),correct_dialogues=c.dialogue_id.nunique(),wrong_dialogues=w.dialogue_id.nunique(),
                             correct_rawC_mean=c.confidence.mean(),wrong_rawC_mean=w.confidence.mean(),
                             mean_wrong_minus_correct=w[corrupt+'_'+measure].mean()-c[corrupt+'_'+measure].mean(),
                             insufficient_support=len(c)<10 or len(w)<10 or f.dialogue_id.nunique()<3)
                    for label,frame in [('correct',c),('wrong',w)]:
                        row.update({label+'_'+k:v for k,v in summarize(frame[corrupt+'_'+measure]).items() if k!='n'})
                    bins.append(row)
            f=g[['dataset','seed','dialogue_id','utterance_index','correct','confidence']].copy()
            f['corruption']=corrupt
            f['signed_confidence_change_mean']=g[corrupt+'_confidence_change_mean']
            f['delta']=g[corrupt+'_delta']
            f['confidence_change_std_population']=g[corrupt+'_confidence_change_std']
            f['confidence_change_std_sample']=f.confidence_change_std_population*np.sqrt(10/9)
            f['signed_mean_se']=f.confidence_change_std_population/3
            f['zero_se']=f.signed_mean_se==0
            f['delta_over_signed_mean_se']=f.delta/f.signed_mean_se.replace(0,np.nan)
            noise.append(f)
            for class_name in ['all','correct','wrong']:
                c=f if class_name=='all' else f[f.correct==(class_name=='correct')]
                row=dict(dataset=ds,seed=seed,corruption=corrupt,class_name=class_name,n=len(c),dialogues=c.dialogue_id.nunique(),
                         n_zero_se=c.zero_se.sum(),zero_se_fraction=c.zero_se.mean(),n_ratio_valid=c.delta_over_signed_mean_se.notna().sum())
                for var in ['confidence_change_std_population','confidence_change_std_sample','signed_mean_se','delta_over_signed_mean_se']:
                    s=summarize(c[var].dropna())
                    row.update({var+'_'+k:s[k] for k in ['mean','median','sample_sd','q05','q25','q75','q95']})
                noise_summaries.append(row)
    keys=['dataset','seed','corruption','measure','class_name']
    d=pd.DataFrame(distributions)
    compare_keyed(audit,'distributions',load_csv('distributions.csv'),d,keys,[c for c in d if c not in keys])
    b=pd.DataFrame(bins)
    keys=['dataset','seed','corruption','measure','bin_index']
    compare_keyed(audit,'fixed_confidence_bins',load_csv('fixed_confidence_bins.csv'),b,keys,[c for c in b if c not in keys])
    r=pd.DataFrame(risks)
    keys=['dataset','seed','corruption','measure']
    compare_keyed(audit,'unconditional_risk_auc',load_csv('unconditional_risk_auc.csv'),r,keys,[c for c in r if c not in keys])
    n=pd.concat(noise)
    keys=['dataset','seed','dialogue_id','utterance_index','corruption']
    compare_keyed(audit,'measurement_noise',load_csv('measurement_noise.csv'),n,keys,[c for c in n if c not in keys])
    ns=pd.DataFrame(noise_summaries)
    keys=['dataset','seed','corruption','class_name']
    compare_keyed(audit,'measurement_noise_summary',load_csv('measurement_noise_summary.csv'),ns,keys,[c for c in ns if c not in keys])
    for stem,frame,groupkeys,variables in [('distribution_seed_summary',d,['dataset','corruption','measure','class_name'],['mean','median','sample_sd','zero_fraction','rawC_mean']),
                                          ('unconditional_risk_auc_summary',r,['dataset','corruption','measure'],['auc_larger_is_error'])]:
        expected=[]
        for key,group in frame.groupby(groupkeys):
            row=dict(zip(groupkeys,key),n_seeds_expected=5)
            for var in variables:
                row[var+'_mean']=group[var].mean()
                row[var+'_sample_sd']=group[var].std(ddof=1)
                row['n_'+var+'_valid']=group[var].notna().sum()
            expected.append(row)
        e=pd.DataFrame(expected)
        compare_keyed(audit,stem,load_csv(stem+'.csv'),e,groupkeys,[c for c in e if c not in groupkeys])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args=parser.parse_args()
    tests=selftest()
    if args.self_test:
        print(json.dumps({'status':'self_test_passed', **tests}, indent=2))
        return
    audit=Audit()
    result={'selftests':tests, 'verifier_sha256':digest(Path(__file__)),
            'independence':'Does not import diagnose.py; DP matching and rank-sum AUC independently recomputed.'}
    try:
        result.update(run_audit(audit))
        result['audited_files']={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'inputs').glob('*')) if p.is_file()}
        result['audited_files'].update({str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'results').glob('*.csv'))})
        result['interpretation_limits']=[
            'Validation has been repeatedly inspected; all diagnostics are exploratory.',
            'Fixed C bins retain mean-confidence residual confounding.',
            'Same Dialogue and approximate C matching does not prove independent information or causality.',
            'Unadjusted percentile intervals across multiple comparisons do not establish calibrated benefit.',
            'delta/SE is descriptive only and is neither a test nor an interval for absolute delta.']
        result['status']='passed'
    except Exception as exc:
        result['status']='failed'
        result['error']=repr(exc)
        raise
    finally:
        result['checks']=audit.checks
        result['passed_checks']=sum(x['passed'] for x in audit.checks)
        result['failed_checks']=sum(not x['passed'] for x in audit.checks)
        (ROOT/'results/independent_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':
    main()
