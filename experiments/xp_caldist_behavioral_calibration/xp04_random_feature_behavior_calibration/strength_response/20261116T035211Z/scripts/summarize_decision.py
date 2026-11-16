"""Compact interpretation of every preregistered strength/modality condition."""
from pathlib import Path

import numpy as np
import pandas as pd

from mainline_core import markdown_table

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['ece15', 'binary_brier', 'binary_nll']


def main():
    results = ROOT / 'results'
    response = pd.read_csv(results / 'behavior_summary.csv')
    metrics = pd.read_csv(results / 'calibration_metrics.csv')
    bootstrap = pd.read_csv(results / 'bootstrap_results.csv')
    all_rows = response[response.population.eq('all_retained')].copy()
    all_rows['any_flip_percent'] = 100 * (1 - all_rows.mu_zero_fraction)
    response_table = all_rows[['dataset', 'modality', 'corruption', 'strength',
                               'any_flip_percent', 'mean_flip_rate', 'mean_mu', 'mean_delta']]
    comparisons = []
    for (dataset, modality, strength), group in metrics.groupby(['dataset', 'modality', 'strength']):
        indexed = group.set_index('method')
        row = {'dataset': dataset, 'modality': modality, 'strength': strength}
        for name in METRICS:
            row['delta_' + name] = indexed.loc['M_joint', name] - indexed.loc['E1', name]
        comparisons.append(row)
    delta = pd.DataFrame(comparisons)
    delta.to_csv(results / 'm_joint_vs_e1_summary.csv', index=False)
    primary = metrics[metrics.modality.eq('joint') & metrics.method.isin(['E0', 'E1', 'M_joint'])]
    text = '# 强度响应实验：结果与后续判断\n\n'
    text += ('本次完成Seed1475、IEMOCAP全部527话语/12对话、MELD全部1109话语/114对话。'
             '历史0.05/5%为参照，新增0.10/10%，四种模态条件、两类扰动各10次。'
             '这是一次预登记敏感性实验，全部条件均报告；没有将强度或模态作为外层OOF择优参数。\n\n')
    text += '## 1. 加大强度是否产生更多行为变化\n\n'
    text += ('any_flip_percent是10次中至少出现一次预测翻转的话语百分比，'
             '不是逐次预测翻转率，也不是μ；mean_flip_rate为逐话语10次翻转比例的均值。\n\n')
    text += markdown_table(response_table) + '\n\n'
    for dataset in ['IEMOCAP', 'MELD']:
        selected = all_rows[all_rows.dataset.eq(dataset) & all_rows.modality.eq('joint')]
        text += f'**{dataset}联合模态：** '
        statements = []
        for kind in ['gaussian', 'dropout']:
            values = selected[selected.corruption.eq(kind)].set_index('strength')
            low, high = values.loc[.05], values.loc[.10]
            statements.append(f'{kind}至少一次翻转覆盖率{low.any_flip_percent:.2f}%→{high.any_flip_percent:.2f}%，'
                              f'平均δ {low.mean_delta:.6g}→{high.mean_delta:.6g}')
        text += '；'.join(statements) + '。\n\n'
    text += ('视觉缺失仍保留。上述表含全部话语；有效视觉子集另见behavior_summary.csv中的valid_visual，'
             '不得把缺失视觉的零响应解释为视觉可靠。更强扰动不保证每条话语响应单调增长：'
             '两个强度沿用包含strength的历史随机键，属于不同确定性随机流。\n\n')
    text += '## 2. 原主模型在两个强度下是否超过E1\n\n'
    text += ('主模型仍为M_joint：原始C与Gaussian/Dropout各自的原始μ、δ，'
             '通过同一双向Logistic可靠性映射输出预测正确概率。没有增加JS输入、交互项或类别特征。\n\n')
    text += markdown_table(primary[['dataset', 'strength', 'method'] + METRICS]) + '\n\n'
    text += '**全部模态的M_joint−E1差值：** 负值为改善。\n\n'
    text += markdown_table(delta) + '\n\n'
    text += ('M_gaussian与M_dropout是预登记的对应扰动消融，全部结果在calibration_metrics.csv和完整报告中列出，'
             '没有根据它们的最优结果替换M_joint。\n\n')
    choices = pd.read_csv(results / 'inner_cv_results.csv')
    selected_penalties = choices[choices.method.eq('M_joint') & choices.selected.eq(True)].copy()
    selection_counts = selected_penalties.groupby(['dataset', 'modality', 'strength']).agg(
        outer_folds=('fold', 'size'), behavior_off_folds=('penalty', lambda s: int(s.astype(str).eq('off').sum()))).reset_index()
    selection_counts.to_csv(results / 'behavior_selection_counts.csv', index=False)
    text += '**内层训练Dialogue选择关闭行为项的折数：** off时精确退回E1。\n\n'
    text += markdown_table(selection_counts) + '\n\n'
    shifts = pd.read_csv(results / 'confidence_shift.csv')
    joint_shifts = shifts[shifts.modality.eq('joint') & shifts.method.isin(['E1', 'M_joint'])]
    text += '**正确/错误样本的联合模态置信度调整：** correct=1为原始预测正确。\n\n'
    text += markdown_table(joint_shifts[['dataset', 'strength', 'method', 'correct', 'utterances',
                                       'original_mean_confidence', 'calibrated_mean_confidence',
                                       'raised_fraction', 'lowered_fraction']]) + '\n\n'
    text += ('错误样本置信度上升本身不能单独判定调整不合理；需结合同一留出样本的Brier/NLL、'
             '以及相对E1是否改善正确与错误预测的区分来判断。\n\n')
    text += '## 3. 单Seed证据判断\n\n'
    for dataset in ['IEMOCAP', 'MELD']:
        row = delta[delta.dataset.eq(dataset) & delta.modality.eq('joint') & delta.strength.eq(.10)].iloc[0]
        differences = row[['delta_' + name for name in METRICS]].to_numpy(float)
        if np.all(differences < 0):
            finding = '新增强度的联合模态M_joint三项点估计均优于E1，形成单Seed初步信号；仍需查看区间并在其他Seed确认。'
        elif np.any(differences < 0):
            finding = '新增强度的联合模态M_joint仅有部分指标点估计优于E1，尚未形成三项一致的增量证据。'
        else:
            finding = '新增强度的联合模态M_joint三项点估计均未优于E1，当前强度调整尚未解决增量校准问题。'
        text += f'- **{dataset}：** {finding}\n'
    text += '\n**联合模态bootstrap差值（全部预登记主模型比较）：**\n\n'
    selected = bootstrap[bootstrap.modality.eq('joint') & bootstrap.candidate.eq('M_joint')]
    text += markdown_table(selected[['dataset', 'comparison', 'candidate_strength', 'reference',
                                     'reference_strength', 'metric', 'mean_difference', 'ci_low', 'ci_high']])
    text += ('\n\n这些为共享2000次Dialogue抽样、无重新拟合的探索性区间；未校正多重比较，'
             '不包含Seed或拟合不确定性，也没有另外重抽扰动repeat。IEMOCAP只有12个Dialogue。Validation曾用于checkpoint选择及多轮研究，'
             '本轮不能作为独立测试证据。单模态出现的局部改善单独记录，不能升级为跨Seed结论。\n\n')
    text += '## 4. 研究结论与停止边界\n\n'
    text += ('本轮分别检验扰动是否产生可测响应、响应是否可重复、控制原始C后是否还有错误风险差异，'
             '以及同折OOF是否优于E1。响应变强与校准有效必须分别判断。'
             '重复一致性、近似C配对和响应差值区间见完整报告及results/。\n\n'
             '本轮结束后停止，不自动追加0.20、30次重复、五Seed或原始面部/音色干扰实验。'
             '后续是否确认某个数据集/模态，应基于全部结果和研究者审核，重新登记确认实验。\n')
    (ROOT / 'FINAL_STRENGTH_DECISION.md').write_text(text)
    import subprocess
    subprocess.run(['/home/yangbin/miniconda3/bin/python', str(Path(__file__).with_name('plot_strength.py'))], check=True)



if __name__ == '__main__':
    main()
