"""Build auditable Chinese result reports; no model fitting or selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from mainline_core import METRICS, markdown_table, sha, dump

LABELS={'E0':'原始XP04','E1':'仅原始置信度','M_gaussian':'C+μ+δ（主模型）','M_gaussian_mu':'C+μ','M_gaussian_delta':'C+δ'}
LIMITS='''## 证据范围与限制

- 真正Validation：IEMOCAP 527条/12个Dialogue；MELD 1109条/114个Dialogue。未使用Test。
- 两个候选来自已查看的同一Validation上的历史探索；新增模型Seed和随机块检验稳定性，不构成独立数据确认。
- 五个checkpoint评价同一批话语，不能视为五份独立数据。五Seed均值Bootstrap在每次共享Dialogue抽样后计算各Seed指标差，再取均值。
- Bootstrap为2000次固定OOF预测上的Dialogue配对抽样，不重新拟合；区间不包含校准器重拟合、Seed总体或随机扰动重复的不确定性。多项比较为探索性，未校正。
- Seed1475只有三个独立10次扰动块；不能据此断言任意随机噪声都稳定。三个块分别拟合和评价，没有合并成30次。
- 所有ECE均为0–1原始数值，例如0.04=4%；Brier/NLL为Top-label二元正确性口径。
- 调整原始预测正确的概率，不改变情绪类别。Accuracy、Weighted-F1、Macro-F1保持不变；AUROC只作区分能力诊断，不是额外成功门槛。
'''


def load(root,name):return pd.read_csv(root/'results'/f'{name}.csv',float_precision='round_trip',dtype={'dataset':str,'dialogue_id':str})

def table(frame):
    if not len(frame):return '无对应记录。'
    return markdown_table(frame)


def write(root,name,text):
    (root/name).write_text(text.rstrip()+'\n')


def build_reports(root):
    root=Path(root)
    config=json.loads((root/'preregistration.json').read_text())
    f=load(root,'calibration_metrics'); agg=load(root,'aggregate_metrics'); boot=load(root,'bootstrap_results')
    shifts=load(root,'confidence_shift'); blocks=load(root,'random_block_comparison'); response=load(root,'behavior_summary'); stability=load(root,'split_half_stability'); selected=load(root,'behavior_selection_counts'); concentration=load(root,'dialogue_concentration_summary'); block_stability=load(root,'cross_block_stability')
    summaries=[]
    for ds,spec in config['datasets'].items():
        a=agg[agg.dataset.eq(ds)&agg.method.eq('M_gaussian')].iloc[0]
        ci=boot[boot.dataset.eq(ds)&boot.scope.eq('mean_across_paired_seeds')&boot.candidate.eq('M_gaussian')&boot.reference.eq('E1')]
        valid_blocks=blocks[blocks.dataset.eq(ds)&blocks.method.eq('M_gaussian')]
        row=dict(dataset=ds,modality=spec['modality'])
        for metric in METRICS:
            c=ci[ci.metric.eq(metric)].iloc[0]
            row[metric+'_difference_vs_E1']=float(a[metric+'_difference_vs_E1'])
            row[metric+'_improving_seeds']=int(a[metric+'_improving_seeds'])
            row[metric+'_ci_low']=float(c.ci_low);row[metric+'_ci_high']=float(c.ci_high)
            row[metric+'_improving_blocks']=int((valid_blocks[metric+'_difference_vs_E1']<0).sum())
        row['all_three_improving_seeds']=int(a.all_three_improving_seeds)
        # These are descriptive sign checks, not a new significance threshold or tuning rule.
        ece_good=row['ece15_difference_vs_E1']<0 and row['ece15_improving_seeds']>=3
        loss_good=any(row[m+'_difference_vs_E1']<0 and row[m+'_improving_seeds']>=3 for m in ['binary_brier','binary_nll'])
        reverse=any(row[m+'_ci_low']>0 for m in METRICS)
        row['descriptive_assessment']='有跨Seed正向线索，需独立数据确认' if ece_good and loss_good and not reverse else '尚未确认预期的跨Seed增量校准收益'
        if reverse:row['descriptive_assessment']+='；存在指标反向风险'
        summaries.append(row)
    decision=pd.DataFrame(summaries)
    decision.to_csv(root/'results/candidate_decision_summary.csv',index=False,float_format='%.17g')
    method='''# 单模态Gaussian候选确认方法

## 固定输入与行为评分

IEMOCAP只扰动目标话语文本T1–T4；MELD只扰动目标话语音频。Gaussian σ=0.05，逐维Training标准差，零方差维度不加噪声。其他话语、模态、图、qmask、umask、长度、标签、原始权重保持不变。视觉缺失话语保留，缺失标记不作为校准特征。

每个Seed/随机块/话语的10次扰动先汇总为一条记录：

- μ=mean(1[扰动预测≠原始预测]×扰动最大置信度)。μ不是Flip Rate。
- δ=abs(C−mean(扰动最大置信度))。δ不是标准差，也不是逐次变化绝对值的均值。
- JS为原始与扰动完整分布的Jensen–Shannon散度均值（自然对数）；仅诊断，不输入校准器。
- Flip Rate、置信度变化均值/总体标准差、5+5分半稳定性另行记录。

## 可靠性映射

主模型：sigmoid(b+wC×C+行为项)，行为项使用训练折标准化后的μ、δ；C直接使用原始最大概率，不取logit。允许上调或下调置信度。这是CaliDist-inspired双向扩展，不是原始CaliDist的单向缩放。

固定对照E0、E1、C+μ+δ、C+μ、C+δ。所有模型使用冻结的历史Logistic/L2实现，内部候选off/1/0.1/0.01，仅以训练Dialogue内GroupKFold5的汇总Binary Brier选择行为惩罚；off逐元素复用E1输出。不能按外层指标重新选择超参数。

## OOF及随机块

沿用同一历史Dialogue划分和逐Seed样本顺序：IEMOCAP LODO12，MELD GroupKFold5。每个外层训练折内重新拟合标准化器与校准器；所有训练/留出Dialogue、样本索引、标准化器参数、LR系数、内部所有候选拟合均保存在results/fold_parameters/。

block0为五Seed，各10次；Seed1475 block0复用Pilot。block1/2只用于Seed1475，分别采用全局repeat索引10–19和20–29，固定同一SHA256/PCG64规则。模型Seed间使用相同扰动输入，随机块不合并。E1在所有块重新计算并核对历史一致性。
'''
    write(root,'METHOD_PROTOCOL.md',method+'\n'+LIMITS)
    cols=['dataset','method']+[x for m in METRICS for x in [m+'_mean',m+'_sample_std',m+'_difference_vs_E1',m+'_improving_seeds']]
    write(root,'CROSS_SEED_CONFIRMATION_REPORT.md','# 候选五Seed确认结果\n\n## 五Seed均值、样本标准差与相对E1变化\n\n'+table(agg[cols])+'\n\n## 每个Seed完整指标\n\n'+table(f[f.block.eq(0)][['dataset','seed','method']+METRICS+['correctness_auc','accuracy','weighted_f1','macro_f1']])+'\n\n## 五Seed共享Dialogue配对Bootstrap\n\n'+table(boot[boot.scope.eq('mean_across_paired_seeds')][['dataset','candidate','reference','metric','mean_difference','ci_low','ci_high','bootstrap_fraction_improving']])+'\n\n'+LIMITS)
    write(root,'RANDOM_REPEAT_STABILITY_REPORT.md','# 独立随机扰动块稳定性\n\n每个块单独10次扰动、单独汇总与OOF拟合。原始模型和样本不变；E0/E1是固定比较基线。不能把三块视为三份新数据。\n\n## 各块指标及相对E1变化\n\n'+table(blocks)+'\n\n## 三块行为响应\n\n'+table(response[response.seed.eq(1475)&response.population.eq('all')][['dataset','seed','block','modality','utterances','any_flip_fraction','mu_zero_fraction','mu_mean','delta_mean','js_mean']])+'\n\n## 独立随机块之间的Spearman\n\n'+table(block_stability)+'\n\n同一话语的两个独立随机块按ID配对；常量行为向量的Spearman记为NA，并同时报告零值/非零比例。\n\n## 5+5分半Spearman\n\n'+table(stability[stability.seed.eq(1475)])+'\n\n分半相关低或μ大量为零表示随机行为估计仍稀疏/不稳定；相关高不等于一定能改善校准。\n\n'+LIMITS)
    write(root,'MU_DELTA_ABLATION_REPORT.md','# μ/δ预登记消融\n\n主模型C+μ+δ与C、C+μ、C+δ采用相同外层Dialogue划分及训练折选择协议。这里只比较预登记配置，不按最佳外层结果替换主模型。\n\n'+table(agg[cols])+'\n\n## μ+δ相对单指标的配对区间\n\n'+table(boot[boot.scope.eq('mean_across_paired_seeds')&boot.reference.isin(['M_gaussian_mu','M_gaussian_delta'])][['dataset','candidate','reference','metric','mean_difference','ci_low','ci_high']])+'\n\n## 行为项开关与选择频次\n\n'+table(selected)+'\n\nμ和δ的互补性应从主模型相对单指标的收益判断，不能因为两者数值不同就认为互补；off较多说明训练折内未稳定选择行为项。\n\n'+LIMITS)
    write(root,'CONFIDENCE_ADJUSTMENT_ANALYSIS.md','# 正确/错误样本置信度变化\n\n## 所有Seed、随机块与方法\n\n'+table(shifts)+'\n\n## 行为响应：正确与错误样本\n\n'+table(response[['dataset','seed','block','population','utterances','any_flip_fraction','mu_mean','delta_mean','js_mean']])+'\n\n观察错误样本是否也提高置信度，必须结合Binary Brier/NLL判断。整体欠置信时，合理校准可能同时提高部分正确与错误样本；不能把“错误样本上调比例”单独作为否定依据。相同原始C区间内的行为对比另见results/conditional_behavior.csv。\n\n'+LIMITS)
    concise=['dataset','modality','ece15_difference_vs_E1','ece15_improving_seeds','binary_brier_difference_vs_E1','binary_brier_improving_seeds','binary_nll_difference_vs_E1','binary_nll_improving_seeds','all_three_improving_seeds','descriptive_assessment']
    conclusion='# 最终候选确认判断\n\n## 核心问题\n\n随机特征扰动的μ/δ能否在原始置信度之外提供增量可靠性信息？应优先比较C+μ+δ与E1，而不是只比较原始XP04。\n\n'+table(decision[concise])+'\n\n## 随机块方向一致性（Seed1475，三块）\n\n'+table(decision[['dataset']+[m+'_improving_blocks' for m in METRICS]])+'\n\n## Dialogue集中性检查\n\n'+table(concentration[concentration.scope.eq('mean_across_paired_seeds')&concentration.method.eq('M_gaussian')])+'\n\n该检查只从固定OOF预测中去掉一个Dialogue后重算指标，不重新拟合；omissions_reversing_full_direction表示去掉单个Dialogue后均值差值改变方向的次数。不能替代独立数据验证。'+'\n\n## 解释与停止边界\n\n表中正负号和一致Seed数量描述结果；“多数Seed有改善”不代表独立样本显著性。必须同时阅读各项配对区间、单Seed/随机块结果以及per_dialogue_results.csv，不只报告最佳条件。若ECE有改善但proper loss恶化，需要保留这一取舍，不能称为全面校准收益。\n\n本轮不自动训练或修改CMERC、不使用Test、不增加强度搜索、不处理原始文本/音频/视频。是否开展面部遮挡、音色处理等原始数据实验，等待研究者审核。即使出现跨Seed正向线索，独立数据上的最终证据仍需冻结方案后另行验证。\n\n'+LIMITS
    write(root,'FINAL_CANDIDATE_DECISION.md',conclusion)
    write(root,'RESULTS_SUMMARY.md','# 单模态随机特征扰动确认摘要\n\n'+table(agg[cols])+'\n\n## 主模型判断\n\n'+table(decision[concise])+'\n\n详细报告：METHOD_PROTOCOL.md、CROSS_SEED_CONFIRMATION_REPORT.md、RANDOM_REPEAT_STABILITY_REPORT.md、MU_DELTA_ABLATION_REPORT.md、CONFIDENCE_ADJUSTMENT_ANALYSIS.md、FINAL_CANDIDATE_DECISION.md。\n\n'+LIMITS)
    reports=['METHOD_PROTOCOL.md','CROSS_SEED_CONFIRMATION_REPORT.md','RANDOM_REPEAT_STABILITY_REPORT.md','MU_DELTA_ABLATION_REPORT.md','CONFIDENCE_ADJUSTMENT_ANALYSIS.md','FINAL_CANDIDATE_DECISION.md','RESULTS_SUMMARY.md']
    dump(root/'results/report_manifest.json',{name:sha(root/name) for name in reports})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=p.parse_args();build_reports(args.root)
