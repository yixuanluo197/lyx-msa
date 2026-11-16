"""Frozen descriptive diagnostics; no fitting, inference, or perturbation generation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, rankdata, spearmanr

CORRUPTIONS = ['gaussian', 'dropout']
MEASURES = ['mu', 'delta']
KEY = ['dataset', 'seed', 'dialogue_id', 'utterance_index']
PAIR_COLUMNS = ['dataset','seed','dialogue_id','correct_utterance_index','wrong_utterance_index',
    'correct_class','wrong_class','correct_label','wrong_label','correct_prediction','wrong_prediction',
    'correct_rawC','wrong_rawC','rawC_wrong_minus_correct','rawC_abs_distance'] + [
    f'{c}_{m}_{suffix}' for c in CORRUPTIONS for m in MEASURES for suffix in ['correct','wrong','difference']]
OUTPUT_NAMES = ['distributions','fixed_confidence_bins','unconditional_risk_auc','distribution_seed_summary',
    'unconditional_risk_auc_summary','split_correlations','stability_seed','stability_summary',
    'measurement_noise','measurement_noise_summary','matched_pairs','matched_seed_statistics',
    'matched_coverage','matched_bootstrap','matched_bootstrap_draws']


def match_confidences(correct, wrong, caliper=.02):
    """Return indices, maximizing valid cardinality before minimizing C distance."""
    correct, wrong = np.asarray(correct, float), np.asarray(wrong, float)
    nc, nw = len(correct), len(wrong)
    if not nc or not nw:
        return []
    distance = np.abs(correct[:, None] - wrong[None, :])
    dummy = 1 + (min(nc, nw) + 1) * caliper
    costs = np.full((nc, nw + nc), dummy)
    costs[:, :nw] = np.where(distance <= caliper, distance, 1e6)
    rows, cols = linear_sum_assignment(costs)
    return [(int(i), int(j)) for i,j in zip(rows, cols) if j < nw and distance[i,j] <= caliper]


def confidence_bins(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('raw confidence must be finite and in [0,1]')
    return np.minimum(np.searchsorted([0,.4,.6,.8,1.], values, side='right') - 1, 3)


def noise_statistics(pop_sd, delta):
    pop_sd, delta = np.asarray(pop_sd, float), np.asarray(delta, float)
    if np.any(pop_sd < 0):
        raise ValueError('negative population SD')
    sample_sd, se = pop_sd * np.sqrt(10/9), pop_sd / np.sqrt(9)
    zero_se = se == 0
    ratio = np.divide(delta, se, out=np.full_like(se, np.nan), where=~zero_se)
    return sample_sd, se, ratio, zero_se


def correlation(left, right, method):
    left, right = np.asarray(left, float), np.asarray(right, float)
    if len(left) < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        return np.nan
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return np.nan
    return float((spearmanr if method == 'spearman' else pearsonr)(left,right).statistic)


def error_auc(error, score):
    """Mann–Whitney AUC with larger scores fixed to mean greater error risk."""
    error, score = np.asarray(error, bool), np.asarray(score, float)
    n_error, n_correct = int(error.sum()), int((~error).sum())
    if not n_error or not n_correct or not np.isfinite(score).all():
        return np.nan
    return float((rankdata(score)[error].sum() - n_error*(n_error+1)/2)/(n_error*n_correct))


def dialogue_bootstrap(sums, counts, draws):
    """Use the same frozen dialogue multiplicities for every model seed."""
    sums, counts, draws = np.asarray(sums), np.asarray(counts), np.asarray(draws)
    result = np.full(len(draws), np.nan)
    for index, draw in enumerate(draws):
        denominator = counts[:, draw].sum(axis=1)
        if np.all(denominator > 0):
            result[index] = np.mean(sums[:, draw].sum(axis=1)/denominator)
    return result


def build_pairs(features):
    rows = []
    for (dataset, seed, dialogue), frame in features.groupby(['dataset','seed','dialogue_id'], sort=True):
        correct = frame[frame.correct == 1].sort_values('utterance_index')
        wrong = frame[frame.correct == 0].sort_values('utterance_index')
        for i,j in match_confidences(correct.confidence, wrong.confidence):
            left, right = correct.iloc[i], wrong.iloc[j]
            row = dict(dataset=dataset, seed=seed, dialogue_id=dialogue,
                       correct_utterance_index=int(left.utterance_index),
                       wrong_utterance_index=int(right.utterance_index),
                       correct_class=1, wrong_class=0,
                       correct_label=int(left.label), wrong_label=int(right.label),
                       correct_prediction=int(left.prediction), wrong_prediction=int(right.prediction),
                       correct_rawC=left.confidence, wrong_rawC=right.confidence,
                       rawC_wrong_minus_correct=right.confidence-left.confidence,
                       rawC_abs_distance=abs(right.confidence-left.confidence))
            for corruption in CORRUPTIONS:
                for measure in MEASURES:
                    column = f'{corruption}_{measure}'
                    row.update({f'{column}_correct':left[column], f'{column}_wrong':right[column],
                                f'{column}_difference':right[column]-left[column]})
            rows.append(row)
    return pd.DataFrame(rows,columns=PAIR_COLUMNS)


def describe(values):
    values = np.asarray(values, float)
    n = len(values)
    result = dict(n=n, n_unique=len(np.unique(values)), mean=np.nan, median=np.nan,
                  sample_sd=np.nan, minimum=np.nan, q05=np.nan, q25=np.nan,
                  q75=np.nan, q95=np.nan, maximum=np.nan, zero_fraction=np.nan)
    if n:
        quantiles = np.quantile(values, [.05,.25,.75,.95])
        result.update(mean=values.mean(), median=np.median(values),
                      sample_sd=values.std(ddof=1) if n > 1 else np.nan,
                      minimum=values.min(), q05=quantiles[0], q25=quantiles[1],
                      q75=quantiles[2], q95=quantiles[3], maximum=values.max(),
                      zero_fraction=np.mean(values == 0))
    return result


def selected_classes(frame):
    return [('all',frame), ('correct',frame[frame.correct == 1]), ('wrong',frame[frame.correct == 0])]


def seed_summary(frame, group_columns, value_columns, expected=5):
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns,keys))
        row['n_seeds_expected'] = expected
        for column in value_columns:
            values = group[column].dropna().to_numpy(float)
            row[f'{column}_mean'] = values.mean() if len(values) else np.nan
            row[f'{column}_sample_sd'] = values.std(ddof=1) if len(values)>1 else np.nan
            row[f'n_{column}_valid'] = len(values)
        rows.append(row)
    return pd.DataFrame(rows)


def distribution_tables(features):
    distributions, bins, aucs = [], [], []
    for (dataset,seed), frame in features.groupby(['dataset','seed'], sort=True):
        bin_ids = confidence_bins(frame.confidence)
        for corruption in CORRUPTIONS:
            for measure in [*MEASURES,'flip_rate']:
                column = f'{corruption}_{measure}'
                for class_name, subset in selected_classes(frame):
                    distributions.append(dict(dataset=dataset, seed=seed, corruption=corruption,
                        measure=measure, class_name=class_name, dialogues=subset.dialogue_id.nunique(),
                        rawC_mean=subset.confidence.mean(), **describe(subset[column])))
            for measure in MEASURES:
                column = f'{corruption}_{measure}'
                aucs.append(dict(dataset=dataset, seed=seed, corruption=corruption, measure=measure,
                    n=len(frame), n_correct=int(frame.correct.sum()), n_wrong=int((frame.correct==0).sum()),
                    dialogues=frame.dialogue_id.nunique(), auc_larger_is_error=error_auc(frame.correct==0, frame[column])))
                for bin_index,(lower,upper) in enumerate(zip([0,.4,.6,.8],[.4,.6,.8,1.])):
                    group = frame.loc[bin_ids == bin_index]
                    correct, wrong = group[group.correct==1], group[group.correct==0]
                    n_dialogues = group.dialogue_id.nunique()
                    reasons = (['correct<10'] if len(correct)<10 else []) + (['wrong<10'] if len(wrong)<10 else []) + (['dialogues<3'] if n_dialogues<3 else [])
                    row = dict(dataset=dataset, seed=seed, corruption=corruption, measure=measure,
                        bin_index=bin_index, bin_left=lower, bin_right=upper,
                        n_correct=len(correct), n_wrong=len(wrong), dialogues=n_dialogues,
                        correct_dialogues=correct.dialogue_id.nunique(), wrong_dialogues=wrong.dialogue_id.nunique(),
                        correct_rawC_mean=correct.confidence.mean(), wrong_rawC_mean=wrong.confidence.mean(),
                        mean_wrong_minus_correct=wrong[column].mean()-correct[column].mean(),
                        insufficient_support=bool(reasons), support_reason=';'.join(reasons))
                    for name, subgroup in [('correct',correct), ('wrong',wrong)]:
                        row.update({f'{name}_{k}':v for k,v in describe(subgroup[column]).items() if k != 'n'})
                    bins.append(row)
    return pd.DataFrame(distributions), pd.DataFrame(bins), pd.DataFrame(aucs)


def stability_tables(split_responses):
    rows = []
    for (dataset,seed,corruption,split), frame in split_responses.groupby(['dataset','seed','corruption','split'], sort=True):
        for measure in MEASURES:
            left, right = frame[f'{measure}_left'], frame[f'{measure}_right']
            pearson, spearman = correlation(left,right,'pearson'), correlation(left,right,'spearman')
            rows.append(dict(dataset=dataset, seed=seed, corruption=corruption, split=split,
                measure=measure, n=len(frame), pearson=pearson, spearman=spearman,
                correlation_estimable=np.isfinite(pearson) and np.isfinite(spearman),
                left_zero_fraction=np.mean(left==0), right_zero_fraction=np.mean(right==0)))
    splits = pd.DataFrame(rows)
    rows = []
    for keys, group in splits.groupby(['dataset','seed','corruption','measure'], sort=True):
        rows.append(dict(zip(['dataset','seed','corruption','measure'],keys),
            n_splits=len(group), pearson=group.pearson.mean(), spearman=group.spearman.mean(),
            n_pearson_valid=group.pearson.count(), n_spearman_valid=group.spearman.count()))
    per_seed = pd.DataFrame(rows)
    summary = seed_summary(per_seed, ['dataset','corruption','measure'], ['pearson','spearman'])
    return splits, per_seed, summary


def measurement_tables(features):
    details, summaries = [], []
    for corruption in CORRUPTIONS:
        frame = features[KEY + ['correct','confidence']].copy()
        frame['corruption'] = corruption
        frame['signed_confidence_change_mean'] = features[f'{corruption}_confidence_change_mean']
        frame['delta'] = features[f'{corruption}_delta']
        frame['confidence_change_std_population'] = features[f'{corruption}_confidence_change_std']
        sample_sd,se,ratio,zero = noise_statistics(frame.confidence_change_std_population, frame.delta)
        frame['confidence_change_std_sample'] = sample_sd
        frame['signed_mean_se'] = se
        frame['delta_over_signed_mean_se'] = ratio
        frame['zero_se'] = zero
        details.append(frame)
        for (dataset,seed), group in frame.groupby(['dataset','seed'], sort=True):
            for name, subset in selected_classes(group):
                row = dict(dataset=dataset, seed=seed, corruption=corruption, class_name=name,
                    n=len(subset), dialogues=subset.dialogue_id.nunique(), n_zero_se=int(subset.zero_se.sum()),
                    zero_se_fraction=subset.zero_se.mean(), n_ratio_valid=subset.delta_over_signed_mean_se.count())
                for column in ['confidence_change_std_population','confidence_change_std_sample','signed_mean_se','delta_over_signed_mean_se']:
                    valid = subset[column].dropna()
                    row.update({f'{column}_{k}':v for k,v in describe(valid).items() if k in ['mean','median','q05','q25','q75','q95','sample_sd']})
                summaries.append(row)
    return pd.concat(details,ignore_index=True), pd.DataFrame(summaries)


def matched_tables(features, pairs, input_directory, seeds):
    statistics = []
    for dataset, frame in features.groupby('dataset',sort=True):
        for seed in seeds:
            original = frame[frame.seed == seed]
            group = pairs[(pairs.dataset==dataset) & (pairs.seed==seed)]
            for corruption in CORRUPTIONS:
                for measure in MEASURES:
                    column = f'{corruption}_{measure}'
                    difference = group[f'{column}_difference'].to_numpy(float)
                    n, n_dialogues = len(group), group.dialogue_id.nunique()
                    reasons = (['pairs<10'] if n<10 else []) + (['dialogues<3'] if n_dialogues<3 else [])
                    positive = np.mean(difference>0) if n else np.nan
                    tie = np.mean(difference==0) if n else np.nan
                    statistics.append(dict(dataset=dataset,seed=seed,corruption=corruption,measure=measure,
                        n_pairs=n,n_dialogues=n_dialogues,
                        n_correct_available=int((original.correct==1).sum()),
                        n_wrong_available=int((original.correct==0).sum()),
                        n_dialogues_available=original.dialogue_id.nunique(),
                        matched_correct_fraction=n/(original.correct==1).sum() if (original.correct==1).any() else np.nan,
                        matched_wrong_fraction=n/(original.correct==0).sum() if (original.correct==0).any() else np.nan,
                        mean_wrong_minus_correct=difference.mean() if n else np.nan,
                        sample_sd_difference=difference.std(ddof=1) if n>1 else np.nan,
                        median_difference=np.median(difference) if n else np.nan,
                        positive_fraction=positive, negative_fraction=np.mean(difference<0) if n else np.nan,
                        tie_fraction=tie,tie_win_fraction=positive+.5*tie,
                        correct_mean=group[f'{column}_correct'].mean(), wrong_mean=group[f'{column}_wrong'].mean(),
                        mean_confidence_difference=group.rawC_wrong_minus_correct.mean(),
                        mean_abs_confidence_distance=group.rawC_abs_distance.mean(),
                        max_confidence_distance=group.rawC_abs_distance.max(),
                        insufficient_support=bool(reasons),support_reason=';'.join(reasons)))
    statistics = pd.DataFrame(statistics)
    bootstrap, replicates = [], []
    for dataset, frame in features.groupby('dataset',sort=True):
        with np.load(input_directory/f'{dataset}_dialogue_draws.npz') as loaded:
            dialogues, draws = loaded['dialogue_ids'].astype(str), loaded['draws']
        if draws.shape != (2000,len(dialogues)) or sorted(frame.dialogue_id.unique()) != dialogues.tolist():
            raise ValueError(f'{dataset}: invalid frozen dialogue draws')
        for corruption in CORRUPTIONS:
            for measure in MEASURES:
                sums, counts = np.zeros((len(seeds),len(dialogues))), np.zeros((len(seeds),len(dialogues)),int)
                for i,seed in enumerate(seeds):
                    group = pairs[(pairs.dataset==dataset) & (pairs.seed==seed)]
                    aggregate = group.groupby('dialogue_id')[f'{corruption}_{measure}_difference'].agg(['sum','count']).reindex(dialogues,fill_value=0)
                    sums[i], counts[i] = aggregate['sum'], aggregate['count']
                values = dialogue_bootstrap(sums,counts,draws)
                valid = np.isfinite(values)
                support = statistics[(statistics.dataset==dataset) & (statistics.corruption==corruption) & (statistics.measure==measure)]
                seed_means = support.mean_wrong_minus_correct.to_numpy(float)
                point = seed_means.mean() if np.isfinite(seed_means).all() else np.nan
                low,high = np.quantile(values[valid],[.025,.975]) if valid.any() else (np.nan,np.nan)
                bootstrap.append(dict(dataset=dataset,corruption=corruption,measure=measure,
                    point_estimate=point,ci_low=low,ci_high=high,n_valid=valid.sum(),n_invalid=(~valid).sum(),
                    n_resamples=len(values),n_seeds=len(seeds),min_seed_pairs=support.n_pairs.min(),
                    min_seed_dialogues=support.n_dialogues.min(),any_low_support=bool(support.insufficient_support.any())))
                replicates.extend(dict(dataset=dataset,corruption=corruption,measure=measure,draw_index=i,
                    valid=bool(valid[i]),statistic=value) for i,value in enumerate(values))
    return statistics, pd.DataFrame(bootstrap), pd.DataFrame(replicates)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_snapshot(root):
    paths = [root/'diagnostic_protocol.json',root/'DIAGNOSTIC_PROTOCOL.md']
    paths += sorted((root/'inputs').glob('*'))
    paths += sorted((root/'scripts').glob('*.py'))
    return {str(path.relative_to(root)):sha256(path) for path in paths if path.is_file()}


def check_formal_gate(root, formal):
    if not formal:
        raise ValueError('Formal computation requires explicit --formal after root review and verifier freeze')
    verification = json.loads((root/'inputs'/'verification.json').read_text())
    if verification.get('all_passed') is not True:
        raise ValueError('Input verification gate did not pass')
    destinations = [root/'results'/f'{name}.csv' for name in OUTPUT_NAMES]
    destinations += [root/'results'/'statistics_verification.json',
        root/'provenance'/'statistics_before.json',root/'provenance'/'statistics_after.json']
    present = [str(path.relative_to(root)) for path in destinations if path.exists()]
    if present:
        raise FileExistsError(f'Refusing to overwrite existing statistics: {present}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--formal',action='store_true',help='Only after root review and independent verifier freeze')
    args = parser.parse_args()
    root = args.root.resolve()
    inputs, results = root/'inputs', root/'results'
    check_formal_gate(root,args.formal)
    before = frozen_snapshot(root)
    (root/'provenance').mkdir(exist_ok=True)
    (root/'provenance'/'statistics_before.json').write_text(json.dumps(before,indent=2)+'\n')
    protocol = json.loads((root/'diagnostic_protocol.json').read_text())
    features = pd.read_csv(inputs/'features.csv',dtype={'dialogue_id':str})
    splits = pd.read_csv(inputs/'split_responses.csv',dtype={'dialogue_id':str})
    seeds = protocol['seeds']
    if features.duplicated(KEY).any() or splits.duplicated(KEY+['corruption','split']).any():
        raise ValueError('duplicate input key')
    if len(features)!=8180 or len(splits)!=49080 or set(features.seed)!=set(seeds):
        raise ValueError('input scope differs from frozen protocol')
    if not features.correct.eq(features.prediction.eq(features.label).astype(int)).all():
        raise ValueError('correctness differs from source prediction/label identity')
    if not features.modality.eq('joint').all():
        raise ValueError('only joint modality is frozen')
    confidence_bins(features.confidence)
    for dataset,spec in protocol['datasets'].items():
        for seed in seeds:
            group = features[(features.dataset==dataset) & (features.seed==seed)]
            if len(group)!=spec['utterances'] or group.dialogue_id.nunique()!=spec['dialogues']:
                raise ValueError('dataset/seed validation coverage differs from protocol')
    for corruption in CORRUPTIONS:
        if not np.allclose(features[f'{corruption}_delta'],np.abs(features[f'{corruption}_confidence_change_mean']),rtol=1e-12,atol=1e-15):
            raise ValueError('delta differs from absolute signed confidence change mean')
    tables = {}
    tables['distributions'],tables['fixed_confidence_bins'],tables['unconditional_risk_auc'] = distribution_tables(features)
    tables['distribution_seed_summary'] = seed_summary(tables['distributions'],['dataset','corruption','measure','class_name'],
        ['mean','median','sample_sd','zero_fraction','rawC_mean'])
    tables['unconditional_risk_auc_summary'] = seed_summary(tables['unconditional_risk_auc'],['dataset','corruption','measure'],['auc_larger_is_error'])
    tables['split_correlations'],tables['stability_seed'],tables['stability_summary'] = stability_tables(splits)
    tables['measurement_noise'],tables['measurement_noise_summary'] = measurement_tables(features)
    pairs = build_pairs(features)
    tables['matched_pairs'] = pairs
    tables['matched_seed_statistics'],tables['matched_bootstrap'],tables['matched_bootstrap_draws'] = matched_tables(features,pairs,inputs,seeds)
    coverage_columns = ['dataset','seed','n_correct_available','n_wrong_available','n_dialogues_available',
        'n_pairs','n_dialogues','matched_correct_fraction','matched_wrong_fraction',
        'mean_confidence_difference','mean_abs_confidence_distance','max_confidence_distance',
        'insufficient_support','support_reason']
    tables['matched_coverage'] = tables['matched_seed_statistics'][coverage_columns].drop_duplicates().reset_index(drop=True)
    results.mkdir(exist_ok=True)
    for name,frame in tables.items():
        frame.to_csv(results/f'{name}.csv',index=False,na_rep='NA',float_format='%.17g')
        print(f'{name}.csv: {len(frame)} rows',flush=True)
    after = frozen_snapshot(root)
    (root/'provenance'/'statistics_after.json').write_text(json.dumps(after,indent=2)+'\n')
    snapshots_equal = before == after
    verification = dict(all_passed=snapshots_equal,inputs_verification_all_passed=True,
        protocol_inputs_scripts_unchanged=snapshots_equal,
        output_rows={name:len(frame) for name,frame in tables.items()},
        output_sha256={f'{name}.csv':sha256(results/f'{name}.csv') for name in tables},
        environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
            numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
            threads={name:os.environ.get(name) for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}),
        completed_at=datetime.now(timezone.utc).isoformat(),
        no_new_inference=True,no_new_noise=True,no_fitting=True,no_test_labels=True,
        sd_convention='population ddof0 over10; sample_sd=pop_sd*sqrt(10/9); signed_mean_SE=pop_sd/sqrt9',
        interpretation='Exploratory descriptive statistics; delta/SE is not a delta CI or p-value; no direction flipping')
    (results/'statistics_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    if not snapshots_equal:
        raise RuntimeError('Frozen protocol/input/scripts changed during computation')
    print('Descriptive statistics complete; fixed larger-is-error direction; no fitting or new sampling.',flush=True)


if __name__ == '__main__':
    main()
