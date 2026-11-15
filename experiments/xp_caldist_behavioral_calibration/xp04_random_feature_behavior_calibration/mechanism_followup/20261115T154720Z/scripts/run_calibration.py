"""CPU-only bounded mechanism follow-up; fixed before viewing new OOF outcomes."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
import sklearn

import conditional_core as core

ROOT = Path(__file__).resolve().parents[1]
PRE = json.loads((ROOT / 'preregistration.json').read_text())
SOURCE = Path(PRE['source_run'])
OUT = ROOT / 'calibration'
KEY = ['dataset', 'seed', 'dialogue_id', 'utterance_index']
METHODS = PRE['methods']
NEW = METHODS[3:]


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def sources():
    paths = [ROOT / 'preregistration.json', ROOT / 'DESIGN_AND_PLAN.md', ROOT / 'diagnostic/feature_matrix.csv',
             ROOT / 'provenance/frozen_correction_core.py', Path(__file__), ROOT / 'scripts/conditional_core.py', ROOT / 'scripts/test_conditional.py',
             SOURCE / 'baseline/oof_predictions.csv', SOURCE / 'baseline/fold_parameters.json']
    paths += [SOURCE / f'baseline/bootstrap/{d}_dialogue_draws.npz' for d in PRE['datasets']]
    return {str(p): sha(p) for p in paths}


def checked_inputs():
    frame = pd.read_csv(ROOT / 'diagnostic/feature_matrix.csv', dtype={'dialogue_id': str})
    old = pd.read_csv(SOURCE / 'baseline/oof_predictions.csv', dtype={'dialogue_id': str})
    assert len(frame) == 8180 and not frame.duplicated(KEY).any()
    for ds, config in PRE['datasets'].items():
        for seed in PRE['seeds']:
            f = frame[(frame.dataset == ds) & (frame.seed == seed)]
            assert len(f) == config['n'] and f.dialogue_id.nunique() == config['dialogues']
            assert (f.correct == (f.label == f.prediction)).all()
            assert set(f.prediction).issubset(set(range(config['classes'])))
            cols = core.feature_columns(ds)['S'] + core.BEHAVIOR + [f'rand_{k}' for k in range(4)]
            assert np.isfinite(f[cols].to_numpy()).all()
    index = frame.set_index(KEY)
    for method in METHODS[:3]:
        x = old[old.method == method].set_index(KEY)
        assert len(x) == len(index) and set(x.index) == set(index.index)
        x = x.loc[index.index]
        for col in ['label', 'prediction', 'correct']:
            assert np.array_equal(x[col], index[col])
        assert np.max(np.abs(x.original_confidence - index.confidence)) < 1e-14
    return frame, old


def run_folds(frame, old):
    folds = json.loads((SOURCE / 'baseline/fold_parameters.json').read_text())
    predictions, selection, gaps = [], [], []
    for ds in PRE['datasets']:
        for seed in PRE['seeds']:
            data = frame[(frame.dataset == ds) & (frame.seed == seed)].reset_index(drop=True)
            y = data.correct.to_numpy(int)
            groups = data.dialogue_id.to_numpy(str)
            block = old[(old.dataset == ds) & (old.seed == seed)]
            indexed = block.set_index(['dialogue_id', 'utterance_index', 'method'])
            for method in METHODS[:3]:
                for i, row in data.iterrows():
                    prior = indexed.loc[(row.dialogue_id, row.utterance_index, method)]
                    predictions.append({**row[KEY + ['label', 'prediction', 'correct']].to_dict(), 'fold': int(prior.fold),
                                        'method': method, 'original_confidence': row.confidence,
                                        'oof_confidence': prior.oof_confidence, 'shift': prior.oof_confidence - row.confidence,
                                        'selected_mode': 'historical_frozen', 'selected_C': None})
            for fold in [x for x in folds if x['dataset'] == ds and x['seed'] == seed]:
                start = time.time()
                tr = np.flatnonzero(np.isin(groups, fold['train_dialogues']))
                ho = np.flatnonzero(np.isin(groups, fold['held_dialogues']))
                assert len(tr) + len(ho) == len(data) and set(tr).isdisjoint(ho)
                # Reconstruct historical baseline outputs from stored training-fold
                # parameters before using them as frozen fair comparisons.
                for name in ['E1', 'L0']:
                    reproduced = expit(core.frozen.base_logits(data.iloc[ho].confidence, fold['methods'][name]))
                    expected = [indexed.loc[(data.iloc[i].dialogue_id, data.iloc[i].utterance_index, name)].oof_confidence for i in ho]
                    assert np.max(np.abs(reproduced - expected)) < 1e-12
                raw = [indexed.loc[(data.iloc[i].dialogue_id, data.iloc[i].utterance_index, 'XP04')].oof_confidence for i in ho]
                assert np.max(np.abs(data.iloc[ho].confidence.to_numpy() - raw)) < 1e-14
                fit = core.fit_nested_all(data.iloc[tr], y[tr], groups[tr], ds)
                fit.update({'dataset': ds, 'seed': seed, 'fold': fold['fold'],
                            'train_indices': tr.tolist(), 'held_indices': ho.tolist(),
                            'train_dialogues': fold['train_dialogues'], 'held_dialogues': fold['held_dialogues'],
                            'historical_baselines': fold['methods']})
                dump(OUT / f'parameters/{ds}_seed{seed}_fold{fold["fold"]}.json', fit)
                for method, result in fit['methods'].items():
                    selected = result['fit']
                    p = core.predict(data.iloc[ho], selected)
                    held_m = core.frozen.metrics(p, y[ho])
                    candidate = result['candidates'][result['selected_index']]
                    selection.append({'dataset': ds, 'seed': seed, 'fold': fold['fold'], 'method': method,
                                      'selected_index': result['selected_index'], 'selected_C': candidate['C'],
                                      'selected_mode': candidate['mode'], 'selected_penalty': candidate.get('penalty', 'not_applicable'),
                                      'candidate_count': len(result['candidates']), 'train_n': len(tr), 'held_n': len(ho),
                                      'parameter_count': len(selected['base']['weights']) + 1 + (4 if selected['kind'] == 'offset' and candidate['penalty'] != 'off' else 0),
                                      'base_parameter_count': len(selected['base']['weights']) + 1,
                                      'parameters_per_training_row': (len(selected['base']['weights']) + 1 + (4 if selected['kind'] == 'offset' and candidate['penalty'] != 'off' else 0)) / len(tr)})
                    gaps.append({'dataset': ds, 'seed': seed, 'fold': fold['fold'], 'method': method,
                                 **{f'train_{k}': v for k, v in result['train_metrics'].items()},
                                 **{f'inner_oof_{k}': v for k, v in result['inner_metrics'].items()},
                                 **{f'outer_oof_{k}': v for k, v in held_m.items()},
                                 **{f'inner_minus_train_{k}': result['inner_metrics'][k] - result['train_metrics'][k] for k in held_m},
                                 **{f'outer_minus_train_{k}': held_m[k] - result['train_metrics'][k] for k in held_m}})
                    for i, probability in zip(ho, p):
                        row = data.iloc[i]
                        predictions.append({**row[KEY + ['label', 'prediction', 'correct']].to_dict(), 'fold': fold['fold'],
                                            'method': method, 'original_confidence': row.confidence,
                                            'oof_confidence': float(probability), 'shift': float(probability - row.confidence),
                                            'selected_mode': candidate['mode'], 'selected_C': candidate['C']})
                print(f'{ds} seed{seed} fold{fold["fold"]}: complete {time.time() - start:.2f}s', flush=True)
    oof = pd.DataFrame(predictions)
    assert len(oof) == 73620 and not oof.duplicated(KEY + ['method']).any()
    oof.to_csv(OUT / 'oof_predictions.csv', index=False)
    pd.DataFrame(selection).to_csv(OUT / 'selection_records.csv', index=False)
    pd.DataFrame(selection).groupby(['dataset', 'method', 'selected_mode']).size().rename('fold_count').reset_index().to_csv(OUT / 'selection_counts.csv', index=False)
    pd.DataFrame(gaps).to_csv(OUT / 'train_oof_gap.csv', index=False)
    return oof


def summaries(oof):
    per_seed, shifts, correct_wrong = [], [], []
    for (ds, seed, method), f in oof.groupby(['dataset', 'seed', 'method'], sort=False):
        p, y, c = f.oof_confidence.to_numpy(), f.correct.to_numpy(), f.original_confidence.to_numpy()
        per_seed.append({'dataset': ds, 'seed': seed, 'method': method, 'n': len(f),
                         **core.frozen.metrics(p, y), **core.frozen.auxiliary(p, y, c, f.label, f.prediction)})
        for correctness in [0, 1]:
            z = f[f.correct == correctness]
            correct_wrong.append({'dataset': ds, 'seed': seed, 'method': method, 'correct': correctness, 'n': len(z),
                                  'original_mean': z.original_confidence.mean(), 'calibrated_mean': z.oof_confidence.mean(),
                                  'mean_shift': z['shift'].mean(), 'raised_fraction': (z['shift'] > 1e-12).mean(),
                                  'lowered_fraction': (z['shift'] < -1e-12).mean()})
        shifts.append({'dataset': ds, 'seed': seed, 'method': method, 'mean_shift': float((p - c).mean()),
                       'raised_fraction': float(np.mean(p > c + 1e-12)), 'lowered_fraction': float(np.mean(p < c - 1e-12))})
    metrics = pd.DataFrame(per_seed)
    assert len(metrics) == 90
    metrics.to_csv(OUT / 'per_seed_metrics.csv', index=False)
    pd.DataFrame(shifts).to_csv(OUT / 'confidence_shift.csv', index=False)
    pd.DataFrame(correct_wrong).to_csv(OUT / 'correct_wrong_adjustment.csv', index=False)
    aggregate = []
    for (ds, method), f in metrics.groupby(['dataset', 'method'], sort=False):
        for metric in ['ece15', 'binary_brier', 'binary_nll', 'correctness_auc', 'mean_confidence']:
            aggregate.append({'dataset': ds, 'method': method, 'metric': metric, 'mean': f[metric].mean(), 'sample_sd': f[metric].std(ddof=1), 'seeds': len(f)})
    pd.DataFrame(aggregate).to_csv(OUT / 'aggregate_metrics.csv', index=False)
    return metrics, pd.DataFrame(aggregate)


def weighted_bootstrap(f, dialogue_ids, weights):
    """Per-draw metrics with unchanged OOF predictions; dialogue multiplicity shared across seeds."""
    p = np.clip(f.oof_confidence.to_numpy(), 1e-15, 1 - 1e-15)
    y = f.correct.to_numpy()
    group = pd.Categorical(f.dialogue_id, categories=dialogue_ids).codes
    assert (group >= 0).all()
    n = len(dialogue_ids)
    count = np.bincount(group, minlength=n)
    denominator = weights @ count
    brier = weights @ np.bincount(group, weights=(p - y) ** 2, minlength=n) / denominator
    loss = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    nll = weights @ np.bincount(group, weights=loss, minlength=n) / denominator
    bins = np.minimum((p * 15).astype(int), 14)
    error = np.zeros((n, 15))
    np.add.at(error, (group, bins), p - y)
    ece = np.abs(weights @ error).sum(axis=1) / denominator
    return {'ece15': ece, 'binary_brier': brier, 'binary_nll': nll}


def bootstrap(oof, per_seed):
    records = []
    for ds in PRE['datasets']:
        source = SOURCE / f'baseline/bootstrap/{ds}_dialogue_draws.npz'
        raw = np.load(source, allow_pickle=False)
        ids = raw['dialogue_ids'].astype(str)
        draws = raw['draws']
        weights = np.zeros((len(draws), len(ids)), int)
        for i, draw in enumerate(draws):
            weights[i] = np.bincount(draw, minlength=len(ids))
        sampled = {method: {metric: [] for metric in ['ece15', 'binary_brier', 'binary_nll']} for method in METHODS}
        for seed in PRE['seeds']:
            for method in METHODS:
                f = oof[(oof.dataset == ds) & (oof.seed == seed) & (oof.method == method)]
                for metric, values in weighted_bootstrap(f, ids, weights).items():
                    sampled[method][metric].append(values)
        differences = {}
        for first, second in PRE['comparisons']:
            for metric in ['ece15', 'binary_brier', 'binary_nll']:
                values = np.mean(sampled[first][metric], axis=0) - np.mean(sampled[second][metric], axis=0)
                a = per_seed[(per_seed.dataset == ds) & (per_seed.method == first)].set_index('seed')[metric]
                b = per_seed[(per_seed.dataset == ds) & (per_seed.method == second)].set_index('seed')[metric]
                delta = a - b
                lo, hi = np.quantile(values, [.025, .975])
                records.append({'dataset': ds, 'method': first, 'reference': second, 'metric': metric,
                                'mean_difference': delta.mean(), 'ci_low': lo, 'ci_high': hi,
                                'bootstrap_probability_improved': np.mean(values < 0), 'improved_seeds': int((delta < 0).sum()),
                                'repeats': len(values), 'draw_source_sha256': sha(source), 'refit': False})
                differences[f'{first}__vs__{second}__{metric}'] = values
        (OUT / 'bootstrap').mkdir(exist_ok=True)
        np.savez_compressed(OUT / f'bootstrap/{ds}_metric_differences.npz', **differences)
    results = pd.DataFrame(records)
    assert len(results) == 60
    results.to_csv(OUT / 'bootstrap_results.csv', index=False)
    return results


def reports(per_seed, aggregate, boot):
    rows = []
    for ds in PRE['datasets']:
        rows += [f'## {ds}', '', '| 方法 | ECE15 Mean ± SD | Brier Mean ± SD | NLL Mean ± SD |', '|---|---:|---:|---:|']
        for method in METHODS:
            vals = []
            for metric in ['ece15', 'binary_brier', 'binary_nll']:
                r = aggregate[(aggregate.dataset == ds) & (aggregate.method == method) & (aggregate.metric == metric)].iloc[0]
                vals.append(f'{r["mean"]:.6f} ± {r.sample_sd:.6f}')
            rows.append('| ' + method + ' | ' + ' | '.join(vals) + ' |')
        rows += ['', '主方法 S_DA 与强静态对照 S（差值<0为改善）：', '', '| 指标 | 平均差值 | Dialogue配对95%区间 | 改善Seed |', '|---|---:|---:|---:|']
        for metric in ['ece15', 'binary_brier', 'binary_nll']:
            r = boot[(boot.dataset == ds) & (boot.method == 'S_DA') & (boot.reference == 'S') & (boot.metric == metric)].iloc[0]
            rows.append(f'| {metric} | {r.mean_difference:+.6f} | [{r.ci_low:+.6f}, {r.ci_high:+.6f}] | {r.improved_seeds}/5 |')
        rows.append('')
    text = '# 条件行为校准五Seed机制诊断\n\n' + '\n'.join(rows)
    text += '\n## 解释边界\n\n本轮复用已用于checkpoint选择及多轮方法设计的Validation；外层OOF仅隔离校准器拟合，不能解释为独立确认。S是强静态对照，不预设优于E1。S_DA为预先固定主方法；T、S_RAND与OFFSET均为机制消融，不能事后替代主方法。OFFSET有16候选，其他增量方法8候选，其额外搜索容量限制已预登记。Bootstrap不重新拟合，五Seed共享同一Dialogue抽样，不是五份独立数据。SD为Seed间样本标准差。\n\n未运行CMERC、未新增扰动、未读取Test标签，Accuracy/F1保持原分类。所有拟合标准化仅使用对应训练Dialogue；内层合并OOF Brier决定超参数。\n'
    text += '\n所有Bootstrap对比未作多重比较校正。S_RAND只有一组固定无标签随机特征，属于容量诊断，不是正式随机化显著性检验。\n'
    (OUT / 'CONDITIONAL_CALIBRATION_REPORT.md').write_text(text)
    decisions = ['# 主方法固定判据', '']
    for ds in PRE['datasets']:
        sub = boot[(boot.dataset == ds) & (boot.method == 'S_DA') & (boot.reference == 'S')].set_index('metric')
        b, e, n = sub.loc['binary_brier'], sub.loc['ece15'], sub.loc['binary_nll']
        passed = b.mean_difference < 0 and e.mean_difference < 0 and n.mean_difference <= 0 and b.improved_seeds >= 4 and b.ci_high <= 0
        decisions.append(f'- {ds}：' + ('满足预定统计方向，建议另设独立数据确认；仍不自动处理原始数据。' if passed else '未同时满足预定增量判据；不进入原始数据干扰阶段。'))
    decisions += ['', '本结果为探索性机制诊断。即使满足方向，也不能将重复Validation上的选择视为独立有效性证明。完整10组比较和90行逐Seed指标均保留，不筛选赢家。']
    (OUT / 'FINAL_CONDITIONAL_DECISION.md').write_text('\n'.join(decisions) + '\n')


def main():
    assert not (OUT / 'verification.json').exists(), 'completed output is immutable; choose a new run to repeat'
    OUT.mkdir(exist_ok=True)
    before = sources()
    dump(OUT / 'source_hashes.json', before)
    dump(OUT / 'environment.json', {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'device': 'CPU'})
    frame, old = checked_inputs()
    oof = run_folds(frame, old)
    metrics, aggregate = summaries(oof)
    boot = bootstrap(oof, metrics)
    reports(metrics, aggregate, boot)
    after = sources()
    assert before == after
    verification = {'status': 'implementation_checks_passed_pending_independent_verification',
                    'utterances': len(frame), 'oof_rows': len(oof), 'per_seed_metric_rows': len(metrics),
                    'outer_folds': 85, 'bootstrap_rows': len(boot), 'source_hashes_unchanged': True,
                    'historical_baselines_reused_and_id_checked': True, 'gpu_inference': False,
                    'new_perturbations': False, 'uses_test_labels': False, 'independent_confirmation': False}
    dump(OUT / 'verification.json', verification)
    dump(OUT / 'artifact_hashes.json', {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'})
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
