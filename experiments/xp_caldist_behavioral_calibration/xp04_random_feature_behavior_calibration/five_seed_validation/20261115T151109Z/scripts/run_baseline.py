"""Registered CPU-only five-seed Validation confidence baselines."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[4]
OUT = RUN / 'baseline'
CORE = RUN / 'provenance/analysis_code/calibration_core.py'
spec = importlib.util.spec_from_file_location('locked_core', CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
CONFIG = json.loads((RUN / 'preregistration.json').read_text())
PILOT = ROOT / 'experiments/xp_caldist_behavioral_calibration/xp04_random_feature_behavior_calibration/pilot/20261113T025053Z'
SOURCE = ROOT / 'CMERC/experiments/xp_caldist_behavioral_calibration/20261107T172302Z/runs'
KEY = ['dialogue_id', 'utterance_index']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted_metrics(p, y, weights):
    n = weights.sum()
    bins = np.minimum((np.clip(p, core.EPS, 1-core.EPS)*15).astype(int), 14)
    count = np.bincount(bins, weights=weights, minlength=15)
    residual = np.bincount(bins, weights=weights*(p-y), minlength=15)
    return np.array([np.abs(residual).sum()/n, np.sum(weights*(p-y)**2)/n,
                     np.sum(weights*(-y*np.log(p)-(1-y)*np.log1p(-p)))/n])


def main():
    OUT.mkdir(exist_ok=True)
    originals, predictions, scores, fold_params = [], [], [], []
    sources = {str(CORE): sha(CORE), str(RUN/'preregistration.json'): sha(RUN/'preregistration.json'),
               str(Path(__file__).resolve()): sha(__file__),
               str(RUN/'scripts/verify_baseline.py'): sha(RUN/'scripts/verify_baseline.py')}
    pilot = pd.read_csv(PILOT/'original_predictions.csv', dtype={'dialogue_id': str})
    sources[str(PILOT/'original_predictions.csv')] = sha(PILOT/'original_predictions.csv')
    for dataset, expected in CONFIG['datasets'].items():
        canonical = pilot[pilot.dataset == dataset].reset_index(drop=True)
        assert len(canonical) == expected['utterances'] and canonical.dialogue_id.nunique() == expected['dialogues']
        assert not canonical.duplicated(KEY).any()
        for seed in CONFIG['seeds']:
            if seed == 1475:
                frame = canonical.copy()
            else:
                path = SOURCE/dataset/f'seed{seed}'/'validation_predictions.npz'
                sources[str(path)] = sha(path)
                arrays = np.load(path, allow_pickle=True)
                probs, logs = arrays['probabilities'].astype(float), arrays['log_probabilities'].astype(float)
                assert np.allclose(np.exp(logs), probs, atol=1e-7)
                frame = pd.DataFrame({'dataset': dataset, 'seed': seed,
                    'dialogue_id': arrays['dialogue_ids'].astype(str), 'utterance_index': arrays['utterance_indices'],
                    'label': arrays['labels'], 'prediction': arrays['predictions'], 'confidence': probs.max(1),
                    'probabilities': [json.dumps(x.tolist()) for x in probs],
                    'log_probabilities': [json.dumps(x.tolist()) for x in logs]})
                assert not frame.duplicated(KEY).any()
                frame = canonical[KEY].merge(frame, on=KEY, how='left', validate='one_to_one')
                assert not frame.isna().any().any()
                assert np.array_equal(frame.label, canonical.label)
            p = np.array([json.loads(s) for s in frame.probabilities])
            assert np.array_equal(p.argmax(1), frame.prediction) and np.allclose(p.max(1), frame.confidence)
            assert np.allclose(p.sum(1), 1, atol=1e-5)
            c = frame.confidence.to_numpy(float)
            y = (frame.label == frame.prediction).to_numpy(int)
            groups = frame.dialogue_id.to_numpy(str)
            splitter = LeaveOneGroupOut() if dataset == 'IEMOCAP' else GroupKFold(5)
            folds = list(splitter.split(c, groups=groups))
            oof = {'XP04': c.copy(), 'E1': np.full(len(c), np.nan), 'L0': np.full(len(c), np.nan)}
            fold_idx = np.full(len(c), -1)
            for fold, (tr, ho) in enumerate(folds):
                assert set(groups[tr]).isdisjoint(groups[ho])
                fold_idx[ho] = fold
                record = {'dataset': dataset, 'seed': seed, 'fold': fold,
                    'train_dialogues': sorted(set(groups[tr])), 'held_dialogues': sorted(set(groups[ho])),
                    'train_n': len(tr), 'held_n': len(ho), 'methods': {}}
                for method in ('E1', 'L0'):
                    params = core.fit_base(c[tr], y[tr], method)
                    oof[method][ho] = expit(core.base_logits(c[ho], params))
                    record['methods'][method] = params
                fold_params.append(record)
            assert np.all(fold_idx >= 0)
            originals.append(frame)
            for method, conf in oof.items():
                assert np.isfinite(conf).all()
                rows = frame[['dataset', 'seed', *KEY, 'label', 'prediction']].copy()
                rows['correct'], rows['fold'] = y, fold_idx
                rows['method'], rows['original_confidence'], rows['oof_confidence'] = method, c, conf
                rows['shift'] = conf-c
                predictions.append(rows)
                scores.append({'dataset': dataset, 'seed': seed, 'method': method, **core.metrics(conf, y),
                    **core.auxiliary(conf, y, c, frame.label.to_numpy(), frame.prediction.to_numpy())})
    originals = pd.concat(originals, ignore_index=True)
    predictions = pd.concat(predictions, ignore_index=True)
    scores = pd.DataFrame(scores)
    originals.to_csv(OUT/'original_predictions.csv', index=False)
    predictions.to_csv(OUT/'oof_predictions.csv', index=False)
    shift_rows = []
    for (dataset, seed, method, correct), frame in predictions.groupby(['dataset','seed','method','correct']):
        shift_rows.append({'dataset':dataset,'seed':seed,'method':method,'correct':correct,'n':len(frame),
            'original_mean_confidence':frame.original_confidence.mean(),
            'oof_mean_confidence':frame.oof_confidence.mean(),'mean_shift':frame['shift'].mean(),
            'raised_fraction':(frame['shift']>1e-12).mean(),'lowered_fraction':(frame['shift'] < -1e-12).mean()})
    shifts = pd.DataFrame(shift_rows)
    shifts.to_csv(OUT/'correct_wrong_confidence_shift.csv',index=False)
    scores.to_csv(OUT/'per_seed_metrics.csv', index=False)
    (OUT/'fold_parameters.json').write_text(json.dumps(fold_params, indent=2))
    metric_names = list(core.metrics([.5], [1]))
    aggregate = []
    for (dataset, method), frame in scores.groupby(['dataset', 'method'], sort=False):
        base = scores[(scores.dataset == dataset) & (scores.method == 'XP04')].set_index('seed')
        for metric in metric_names:
            difference = frame.set_index('seed')[metric] - base[metric]
            aggregate.append({'dataset': dataset, 'method': method, 'metric': metric,
                'mean': frame[metric].mean(), 'sample_sd': frame[metric].std(ddof=1),
                'mean_change_vs_XP04': difference.mean(), 'improved_seeds_vs_XP04': int((difference < -1e-12).sum()),
                'worsened_seeds_vs_XP04': int((difference > 1e-12).sum()), 'seeds': 5})
    pd.DataFrame(aggregate).to_csv(OUT/'aggregate_metrics.csv', index=False)
    bootstrap = []
    rng = np.random.default_rng(CONFIG['bootstrap']['seed'])
    samples_dir = OUT/'bootstrap'
    samples_dir.mkdir(exist_ok=True)
    for dataset in CONFIG['datasets']:
        ds = predictions[predictions.dataset == dataset]
        dialogues = sorted(ds.dialogue_id.unique())
        # One shared draw for all five model seeds, preserving their dependence.
        draws = rng.integers(0, len(dialogues), size=(2000, len(dialogues)))
        np.savez_compressed(samples_dir/f'{dataset}_dialogue_draws.npz',
                            draws=draws, dialogue_ids=np.array(dialogues))
        contrasts = [('E1', 'XP04'), ('L0', 'XP04'), ('L0', 'E1')]
        delta = {f'{a}_vs_{b}': np.zeros((2000, 3)) for a,b in contrasts}
        for seed in CONFIG['seeds']:
            seed_rows = ds[ds.seed == seed]
            xp = seed_rows[seed_rows.method == 'XP04']
            gi = pd.Categorical(xp.dialogue_id, categories=dialogues).codes
            y = xp.correct.to_numpy()
            p = {method: np.clip(seed_rows[seed_rows.method == method].oof_confidence.to_numpy(), core.EPS, 1-core.EPS)
                 for method in ('XP04', 'E1', 'L0')}
            for i, draw in enumerate(draws):
                weights = np.bincount(draw, minlength=len(dialogues))[gi]
                ms = {m: weighted_metrics(v, y, weights) for m,v in p.items()}
                for a,b in contrasts:
                    delta[f'{a}_vs_{b}'][i] += (ms[a]-ms[b])/5
        for a,b in contrasts:
            d = delta[f'{a}_vs_{b}']
            np.save(samples_dir/f'{dataset}_{a}_vs_{b}_metric_differences.npy', d)
            for j, metric in enumerate(metric_names):
                av = scores[(scores.dataset == dataset)&(scores.method == a)].set_index('seed')[metric]
                bv = scores[(scores.dataset == dataset)&(scores.method == b)].set_index('seed')[metric]
                bootstrap.append({'dataset': dataset, 'comparison': f'{a}_vs_{b}', 'metric': metric,
                    'point_mean_difference': (av-bv).mean(), 'ci025': np.quantile(d[:,j], .025),
                    'ci975': np.quantile(d[:,j], .975), 'fraction_below_zero': (d[:,j] < 0).mean(),
                    'improved_seeds': int((av-bv < -1e-12).sum()), 'resamples': 2000,
                    'unit': 'dialogue; shared draw across five seeds', 'refit': False})
    bootstrap = pd.DataFrame(bootstrap)
    bootstrap.to_csv(OUT/'bootstrap_results.csv', index=False)
    (OUT/'source_hashes.json').write_text(json.dumps(sources, indent=2))
    environment = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                   'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'CPU_threads': 1,
                   'GPU_used': False, 'training_CMerc': False, 'Test_used': False,
                   'source_probabilities': 'frozen true Validation; additional checkpoint parity audited in subsequent perturbation batch'}
    (OUT/'environment.json').write_text(json.dumps(environment, indent=2))
    report = ['# XP04 五Seed置信度基础校准实验', '',
        '固定使用真正Validation：IEMOCAP527条/12 Dialogue；MELD1109条/114 Dialogue。每个Seed单独拟合E1/L0；原模型类别不变。', '',
        'E1：原始C的L2 Logistic；L0：训练折StandardScaler后的logit(C)的L2 Logistic。均C=1，沿用冻结代码，不依据本实验结果选择超参数。', '',
        '## 五Seed均值与样本标准差', '', '|Dataset|Method|ECE15 Mean ± SD|Binary Brier Mean ± SD|Binary NLL Mean ± SD|', '|---|---|---|---|---|']
    for (dataset, method), frame in scores.groupby(['dataset','method'], sort=False):
        report.append('|'+ '|'.join([dataset,method]+[f'{frame[m].mean():.6f} ± {frame[m].std(ddof=1):.6f}' for m in metric_names])+'|')
    report += ['', '## 全部Seed结果', '', '|Dataset|Seed|Method|ECE15|Binary Brier|Binary NLL|', '|---|---|---|---|---|---|']
    for row in scores.to_dict('records'):
        report.append(f"|{row['dataset']}|{row['seed']}|{row['method']}|{row['ece15']:.6f}|{row['binary_brier']:.6f}|{row['binary_nll']:.6f}|")
    report += ['', '## Dialogue配对Bootstrap', '', '差值为前者减后者，负值表示改善。2000次、同一次Dialogue抽样跨五Seed共享；每次先计算各Seed指标差再取均值。区间不包含校准器重拟合不确定性。', '',
        '|Dataset|Comparison|Metric|Mean difference|95% percentile interval|Improved seeds|', '|---|---|---|---|---|---|']
    for row in bootstrap.to_dict('records'):
        report.append(f"|{row['dataset']}|{row['comparison']}|{row['metric']}|{row['point_mean_difference']:.6f}|[{row['ci025']:.6f}, {row['ci975']:.6f}]|{row['improved_seeds']}/5|")
    report += ['', '## 正确/错误样本置信度变化', '',
        '下表为五Seed对应组均值，完整每Seed数据见correct_wrong_confidence_shift.csv。', '',
        '|Dataset|Method|Correctness|Original mean C|OOF mean C|Mean shift|Raised fraction|',
        '|---|---|---|---|---|---|---|']
    for (dataset,method,correct), frame in shifts.groupby(['dataset','method','correct'],sort=False):
        report.append(f"|{dataset}|{method}|{'正确' if correct else '错误'}|{frame.original_mean_confidence.mean():.6f}|{frame.oof_mean_confidence.mean():.6f}|{frame.mean_shift.mean():.6f}|{frame.raised_fraction.mean():.6f}|")
    report += ['', '## 研究判断与限制', '',
        'IEMOCAP：E1的ECE15/Brier/NLL在全部5 Seed均优于XP04，ECE15改善的Dialogue bootstrap区间低于0，但Brier/NLL区间仍跨0。L0较E1的Brier/NLL在全部5 Seed变差。因此后续行为模块应同时保留E1作为重要基础消融。', '',
        'MELD：E1与L0的三项指标在全部5 Seed均优于XP04，且对应Dialogue bootstrap区间均低于0。L0较E1的Brier/NLL全部5 Seed改善且区间低于0；ECE15为4/5 Seed改善，区间跨0。该结果支持基础映射的稳定收益，不等于行为模块有效。', '',
        '本批只评估原始置信度映射，不能据此宣称随机扰动行为有效。行为模块仍须在冻结配置下与对应E1/L0基础比较。', '',
        'Validation已用于多轮方法设计，五Seed共享相同Dialogue，仅检验训练随机性稳定性；五Seed不是五份独立数据。IEMOCAP仅12 Dialogue，统计不确定性须结合上表区间解释。', '',
        'Seed1475复用经过checkpoint复核的原Pilot概率，其余4 Seed复用既有真正Validation NPZ。它们的fresh checkpoint parity在后续扰动作业单独审计。本批不使用GPU、Test标签或Test指标，不修改CMERC。', '',
        '独立验证结果见verification.json；逐话语原始完整概率/对数概率、OOF预测、折参数与源文件SHA256均保留。']
    (OUT/'BASELINE_FIVE_SEED_REPORT.md').write_text('\n'.join(report)+'\n')
    print(scores[['dataset','seed','method',*metric_names]].to_string(index=False))


if __name__ == '__main__':
    main()
