#!/usr/bin/env python
"""Offline, validation-only diagnostics. No model inference or parameter tuning."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FROZEN_CODE = Path(__file__).resolve().parents[4] / 'CMERC/experiment_records/04_xp_mainline_validation/code'
sys.path.insert(0, str(FROZEN_CODE))
from evaluation_metrics import calibration_metrics

BEHAVIOR = ('flip_rate', 'mu', 'delta_original', 'confidence_mad', 'js_divergence', 'confidence_change')


def multiclass_metrics(labels, probabilities):
    """Frozen ECE/SUROC; Brier is sum across classes, then mean samples."""
    labels = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or len(p) != len(labels) or not len(labels):
        raise ValueError('invalid probability shape or empty labels')
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('invalid probability distribution')
    frozen = calibration_metrics(labels, p, n_bins=15)
    onehot = np.eye(p.shape[1])[labels]
    return dict(ece15=frozen['ece'], brier=float(np.square(p - onehot).sum(1).mean()),
                nll=float(-np.log(np.maximum(p[np.arange(len(p)), labels], np.finfo(float).tiny)).mean()),
                auroc_success=frozen['auroc_success'], accuracy=float((p.argmax(1) == labels).mean()))


def repeated_metrics(labels, probabilities):
    """Mean of per-repeat metrics, never the metric of ensemble probabilities."""
    metrics = [multiclass_metrics(labels, probabilities[:, r]) for r in range(probabilities.shape[1])]
    return {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}


def confidence_bin(confidence):
    return np.minimum((np.asarray(confidence) * 10).astype(int), 9)


def lodo_success(confidence, behavior, correct, dialogues):
    """Held-out dialogue scores with scaler and logistic model fitted on train fold only."""
    confidence, behavior = np.asarray(confidence), np.asarray(behavior)
    correct, dialogues = np.asarray(correct, int), np.asarray(dialogues)
    features = {'base': confidence.reshape(-1, 1), 'behavior': np.column_stack([confidence, behavior])}
    result = {name: np.full(len(correct), np.nan) for name in features}
    skipped = 0
    for dialogue in np.unique(dialogues):
        held_out = dialogues == dialogue
        train = ~held_out
        if np.unique(correct[train]).size < 2:
            skipped += 1
            continue
        for name, x in features.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1., penalty='l2',
                                      solver='lbfgs', max_iter=1000, random_state=20261107))
            model.fit(x[train], correct[train])
            result[name][held_out] = model.predict_proba(x[held_out])[:, 1]
    result['skipped_folds'] = skipped
    result['n_dialogues'] = len(np.unique(dialogues))
    return result


def success_metrics(correct, score):
    correct, score = np.asarray(correct), np.asarray(score)
    if not len(correct):
        return dict(auroc=float('nan'), brier=float('nan'), nll=float('nan'))
    clipped = np.clip(score, 1e-15, 1 - 1e-15)
    return dict(auroc=float(roc_auc_score(correct, score)) if len(np.unique(correct)) == 2 else float('nan'),
                brier=float(np.mean((score - correct) ** 2)),
                nll=float(-np.mean(correct * np.log(clipped) + (1 - correct) * np.log1p(-clipped))))


def paired_dialogue_bootstrap(correct, base, behavior, dialogues, *, repeats, seed):
    """Paired CI on fixed OOF scores; does not refit models in bootstrap samples."""
    correct, base, behavior, dialogues = map(np.asarray, (correct, base, behavior, dialogues))
    good = np.isfinite(base) & np.isfinite(behavior)
    correct, base, behavior, dialogues = [x[good] for x in (correct, base, behavior, dialogues)]
    baseline, augmented = success_metrics(correct, base), success_metrics(correct, behavior)
    values = {name: [] for name in baseline}
    units = [np.flatnonzero(dialogues == d) for d in np.unique(dialogues)]
    rng = np.random.default_rng(seed)
    for _ in range(repeats if units else 0):
        ix = np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))])
        a, b = success_metrics(correct[ix], base[ix]), success_metrics(correct[ix], behavior[ix])
        for name in values:
            if np.isfinite(a[name]) and np.isfinite(b[name]):
                values[name].append(b[name] - a[name])
    result = {}
    for name, draws in values.items():
        lo, hi = np.quantile(draws, [.025, .975]) if draws else (np.nan, np.nan)
        result[name] = dict(base=baseline[name], behavior=augmented[name],
                            difference=augmented[name] - baseline[name], ci_low=float(lo),
                            ci_high=float(hi), valid_bootstraps=len(draws))
    return result


def validate_payload(baseline, payload, *, repeats):
    for field in ('labels', 'dialogue_ids', 'utterance_indices'):
        if not np.array_equal(baseline[field], payload[field]):
            raise ValueError(f'perturbation target order mismatch: {field}')
    for field in ('speaker_ids', 'batch_indices', 'dialogue_lengths'):
        if field in baseline and (field not in payload or not np.array_equal(baseline[field], payload[field])):
            raise ValueError(f'perturbation target metadata mismatch: {field}')
    original = baseline['probabilities']
    if payload['probabilities'].shape != (len(original), repeats, original.shape[1]):
        raise ValueError('incorrect perturbation shape or repeat count')
    if not np.array_equal(original, payload['original_probabilities']):
        raise ValueError('original probabilities changed')
    p = payload['probabilities']
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=1e-5):
        raise ValueError('invalid perturbation probabilities')
    if 'log_probabilities' in payload and not np.allclose(np.exp(payload['log_probabilities']), p, atol=1e-6, rtol=1e-5):
        raise ValueError('perturbation log probabilities disagree')


def load_npz(path):
    with np.load(path, allow_pickle=False) as values:
        return {k: values[k] for k in values.files}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f'no rows for {path}')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(x):
    x = np.asarray(x)
    return float(x.mean()) if len(x) else float('nan')


def group_statistics(rows, mask):
    result = {'count': int(mask.sum())}
    for key in BEHAVIOR:
        x = rows[key][mask]
        result[key + '_mean'] = finite_mean(x)
        for q in (10, 50, 90):
            result[f'{key}_q{q}'] = float(np.percentile(x, q)) if len(x) else float('nan')
    return result


def run(root):
    from pilot_core import behavior_metrics

    config = json.loads((root / 'configs/perturbation_config.json').read_text())
    if config['statistics']['split'] != 'train' or not config['no_calibration_fit']:
        raise ValueError('configuration is outside validation diagnostic protocol')
    diagnostics = config['diagnostics']
    if diagnostics['ece_bins'] != 15 or diagnostics['bootstrap_unit'] != 'dialogue':
        raise ValueError('unsupported diagnostic preregistration')
    output = root / 'results'
    rows, summaries, stratified, groups, cases, oof_rows, conditional = [], [], [], [], [], [], []
    for dataset, seed in config['pilots'].items():
        run_dir = root / 'runs' / dataset / f'seed{seed}'
        baseline = load_npz(run_dir / 'validation_predictions.npz')
        labels, original = baseline['labels'], baseline['probabilities']
        expected = {'IEMOCAP': (527, 12), 'MELD': (1109, 114)}[dataset]
        if (len(labels), len(np.unique(baseline['dialogue_ids']))) != expected:
            raise ValueError(f'{dataset}: incorrect validation size')
        if 'predictions' in baseline and not np.array_equal(baseline['predictions'], original.argmax(1)):
            raise ValueError('baseline predictions disagree')
        original_metrics = multiclass_metrics(labels, original)
        confidence, pred = original.max(1), original.argmax(1)
        correct = pred == labels
        for modality in config['modality_order']:
            for perturbation, specification in config['perturbations'].items():
                payload = load_npz(run_dir / 'perturbations' / f'{modality}_{perturbation}.npz')
                validate_payload(baseline, payload, repeats=specification['repeats'])
                p = payload['probabilities']
                metrics = behavior_metrics(original, p)
                if not all(key in metrics and np.asarray(metrics[key]).shape == (len(labels),) for key in BEHAVIOR):
                    raise ValueError('behavior metric contract mismatch')
                tag = dict(dataset=dataset, seed=seed, modality=modality, perturbation_type=perturbation,
                           perturbation_strength=specification['strength'], repeats=specification['repeats'])
                pert_metrics = repeated_metrics(labels, p)
                item = dict(tag, n_utterances=len(labels), n_dialogues=expected[1],
                            metric_units='ECE15 and AUROC percent; Brier/NLL/Accuracy natural units',
                            aggregation='mean of per-repeat metrics; no probability ensemble')
                item.update({'original_' + k: v for k, v in original_metrics.items()})
                item.update({'perturbed_' + k: v for k, v in pert_metrics.items()})
                item.update({key: finite_mean(metrics[key]) for key in BEHAVIOR})
                item.update(stable_wrong_count=int(((~correct) & (metrics['flip_rate'] == 0)).sum()),
                            wrong_count=int((~correct).sum()),
                            correct_sensitive_count=int((correct & (metrics['flip_rate'] > 0)).sum()),
                            correct_count=int(correct.sum()))
                for key in ('mu', 'delta_original', 'confidence_mad', 'js_divergence'):
                    item['auroc_success_negative_' + key] = (float(roc_auc_score(correct, -metrics[key]))
                                                            if len(np.unique(correct)) == 2 else float('nan'))
                summaries.append(item)
                for index in range(len(labels)):
                    row = dict(tag, dialogue_id=str(baseline['dialogue_ids'][index]),
                               utterance_index=int(baseline['utterance_indices'][index]),
                               speaker_id=int(baseline['speaker_ids'][index]), label=int(labels[index]),
                               correct=bool(correct[index]), original_prediction=int(pred[index]),
                               original_confidence=float(confidence[index]),
                               perturbed_predictions=json.dumps(p[index].argmax(1).tolist()),
                               mean_perturbed_confidence=float(p[index].max(1).mean()),
                               mean_perturbed_accuracy=float((p[index].argmax(1) == labels[index]).mean()))
                    row.update({key: float(metrics[key][index]) for key in BEHAVIOR})
                    rows.append(row)
                    if (not correct[index] and metrics['flip_rate'][index] == 0) or (correct[index] and metrics['flip_rate'][index] > 0):
                        cases.append(dict(row, case_type='stable_but_wrong' if not correct[index] else 'correct_but_sensitive'))
                for value in (False, True):
                    groups.append(dict(tag, correct=value, **group_statistics(metrics, correct == value)))
                bins = confidence_bin(confidence)
                for bin_id in range(10):
                    for value in (False, True):
                        mask = (bins == bin_id) & (correct == value)
                        stratified.append(dict(tag, confidence_bin=bin_id, lower=bin_id / 10,
                                               upper=(bin_id + 1) / 10, correct=value,
                                               **group_statistics(metrics, mask)))
                x = np.column_stack([metrics[k] for k in ('mu', 'confidence_mad', 'js_divergence')])
                oof = lodo_success(confidence, x, correct, baseline['dialogue_ids'])
                bootstrap = paired_dialogue_bootstrap(correct, oof['base'], oof['behavior'], baseline['dialogue_ids'],
                            repeats=diagnostics['bootstrap_repeats'], seed=diagnostics['bootstrap_seed'])
                for key, result in bootstrap.items():
                    conditional.append(dict(tag, metric=key, **result,
                                        n_oof=int(np.isfinite(oof['base']).sum()), skipped_folds=oof['skipped_folds'],
                                        ci_scope='paired dialogue bootstrap of fixed LODO OOF predictions; no refitting'))
                for index in range(len(labels)):
                    oof_rows.append(dict(tag, dialogue_id=str(baseline['dialogue_ids'][index]),
                                         utterance_index=int(baseline['utterance_indices'][index]),
                                         correct=bool(correct[index]), baseline_oof_success=float(oof['base'][index]),
                                         behavior_oof_success=float(oof['behavior'][index])))
    for filename, values in [('utterance_behavior_metrics.csv', rows), ('modality_pilot_summary.csv', summaries),
                             ('behavior_correctness_groups.csv', groups), ('confidence_stratified_behavior.csv', stratified),
                             ('conditional_information_diagnostics.csv', conditional), ('conditional_oof_predictions.csv', oof_rows)]:
        write_csv(output / filename, values)
    if cases:
        write_csv(output / 'stable_wrong_and_correct_sensitive_cases.csv', cases)
    else:
        (output / 'stable_wrong_and_correct_sensitive_cases.csv').write_text('case_type\n', encoding='utf-8')
    write_report(root, summaries, conditional)
    return dict(sample_rows=len(rows), summary_rows=len(summaries), conditional_rows=len(conditional))


def fmt(x):
    return f'{x:.6g}' if np.isfinite(x) else 'NA'


def write_report(root, summaries, conditional):
    lines = ['# 单模态验证集 Pilot 诊断', '',
             '仅覆盖 IEMOCAP seed1475 与 MELD seed67137 的真实验证集；不涉及正式测试、重新训练、校准拟合或联合模态。', '',
             '三种模态均为特征空间扰动。Text Representation 不等同于文本语义 Assertion/Probe，音视频扰动不等同于物理干扰。', '',
             '每个随机扰动重复5次；衰减重复1次。下表为话语平均行为指标，完整逐样本及逐重复概率保存于 runs。', '',
             '| 数据集 | 模态 | 类型 | Flip Rate | μ | δ原式 | MAD | JS | 置信度变化 | Stable/Wrong | Sensitive/Correct |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        lines.append('| ' + ' | '.join([row['dataset'], row['modality'], row['perturbation_type']] +
                      [fmt(row[k]) for k in BEHAVIOR] +
                      [f"{row['stable_wrong_count']}/{row['wrong_count']}",
                       f"{row['correct_sensitive_count']}/{row['correct_count']}"]) + ' |')
    lines += ['', '置信度变化为 mean(c_perturbed − c_original)，正值表示上升。稳定仅指零翻转，不能据此推断预测正确；仍需查看 MAD 与 JS。', '',
              '| 数据集 | 模态 | 类型 | 原始ECE15 | 扰动ECE15 | 原始Brier | 扰动Brier | 原始NLL | 扰动NLL | 原始AUROC | 扰动AUROC |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        keys = [side + '_' + metric for metric in ('ece15', 'brier', 'nll', 'auroc_success') for side in ('original', 'perturbed')]
        lines.append('| ' + ' | '.join([row['dataset'], row['modality'], row['perturbation_type']] + [fmt(row[k]) for k in keys]) + ' |')
    lines += ['', 'ECE15和AUROC采用冻结评估代码，以百分数报告；Brier是多分类概率平方误差之和的样本均值，NLL使用自然对数。扰动指标按每个重复分别计算后取平均，未对概率做ensemble。', '',
              '正确/错误行为的均值及10/50/90分位见 behavior_correctness_groups.csv；固定原始置信度0.1宽分层对照见 confidence_stratified_behavior.csv。分层中单组缺样本时为NA，不能视作零。', '',
              '独立信息诊断使用 leave-one-dialogue-out：原始置信度 logistic regression 对比原始置信度+μ+MAD+JS；每折只用其余对话拟合标准化和模型，固定L2 C=1，无搜索。', '',
              '| 数据集 | 模态 | 类型 | 指标 | 基础OOF | 行为OOF | 行为−基础 | 95%区间 |',
              '|---|---|---|---|---:|---:|---:|---:|']
    for row in conditional:
        lines.append('| ' + ' | '.join([row['dataset'], row['modality'], row['perturbation_type'], row['metric'],
                     fmt(row['base']), fmt(row['behavior']), fmt(row['difference']),
                     f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]"]) + ' |')
    lines += ['', 'OOF成功预测AUROC为0–1尺度；二元Brier和二元NLL与上表多分类指标的定义不同。AUROC差值正值较好，Brier/NLL差值负值较好。', '',
              '区间采用预注册种子20261107和2000次配对对话bootstrap，对固定OOF分数重采样，不重新训练每个bootstrap模型，不覆盖拟合过程全部不确定性。单类别样本的AUROC为NA，报告有效bootstrap数。', '',
              '本结果仅为两个单seed验证集的探索性诊断；多个模态、扰动和指标比较未经多重比较校正，不能宣称泛化、显著性或校准提升。', '',
              '继续研究应联合考察两个数据集上行为差异、同置信度分层及OOF增量信息的一致性；翻转最多不等于最有校准价值。A+V是否值得继续由这些诊断共同决定，当前脚本不启动联合实验。']
    path = root / 'reports/SINGLE_MODAL_PILOT_REPORT.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve()), indent=2))


if __name__ == '__main__':
    main()
