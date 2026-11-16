"""Chinese reports from complete, verified response/calibration outputs."""
import argparse
from pathlib import Path
import pandas as pd
from mainline_core import markdown_table


def build_reports(root):
    root = Path(root)
    results = root/'results'
    summary = pd.read_csv(results/'behavior_summary.csv')
    stability = pd.read_csv(results/'split_half_stability.csv')
    matched = pd.read_csv(results/'matched_statistics.csv')
    response_boots = pd.read_csv(results/'response_bootstrap.csv')
    metrics = pd.read_csv(results/'calibration_metrics.csv')
    boots = pd.read_csv(results/'bootstrap_results.csv')
    parity_path = results/'analysis_verification.json'
    if not parity_path.exists():
        raise FileNotFoundError('Analysis verification required before report generation')
    text = '# Seed1475 随机特征扰动强度响应与校准敏感性\n\n'
    text += ('0.05为历史主条件；0.10为预登记新增敏感性条件。完整报告两个强度、四种模态、两种干扰，'
             '不根据本次外层OOF结果选择强度。IEMOCAP 527条/12个Dialogue，MELD 1109条/114个Dialogue；'
             '只使用完整真实Validation、同一个Seed1475。\n\n')
    text += ('μ=mean(1[扰动预测≠原预测]×扰动最大置信度)，δ=|原始最大置信度−mean(扰动最大置信度)|；'
             'JS采用自然对数且仅用于响应诊断。置信度变化标准差按10repeat总体标准差ddof=0计算。'
             '每条话语每条件汇总成一条记录；没有把10次重复、四种模态或两种强度当成独立话语。\n\n')
    text += '## 行为响应与零值覆盖\n\n'
    text += markdown_table(summary[['dataset','modality','strength','corruption','population','utterances',
                       'mu_zero_fraction','mean_mu','mean_delta','mean_js','mean_flip_rate']])+'\n\n'
    text += ('all_retained包含全部话语；valid_visual仅在visual/joint条件下排除视觉缺失话语。'
             'IEMOCAP 1条、MELD 95条缺失视觉仍保留在校准和全部话语统计中；'
             'visual跳过扰动产生的零响应不能解释为视觉可靠。joint的有效视觉子集作为补充诊断报告。\n\n')
    text += '## 两个独立5repeat半组的响应稳定性\n\n'
    text += markdown_table(stability[['dataset','modality','strength','corruption','population','measure','spearman',
                          'left_zero_fraction','right_zero_fraction','auc_larger_is_error']])+'\n\n'
    text += ('固定重复0–4与5–9计算Spearman，不筛选更有利的拆分。常量半组的相关系数为NA；'
             '同时报告两半组零值占比。AUC方向固定为较大响应对应较高错误风险，未按结果翻转方向。'
             '这反映测量与无条件分离能力，不能代替控制原始置信度后的增量证据。\n\n')
    text += '## 相同Dialogue内近似原始置信度配对\n\n'
    text += markdown_table(matched[['dataset','modality','strength','corruption','population','measure','n_pairs',
                          'n_pair_dialogues','mean_wrong_minus_correct','insufficient_support']])+'\n\n'
    text += ('在原始置信度绝对差≤0.02内，用Hungarian算法先最大化配对数、再最小化总置信度距离；'
             '配对选择只依赖原始正确性、置信度与身份，不使用行为响应。每个配对条件内，每条原始话语最多使用一次。'
             '不同强度使用相同配对；valid_visual子集单独按同一规则配对。少于10对或3个Dialogue标为支持不足。'
             '配对记录会跨强度、干扰和模态复用，不能把CSV总行数当作独立样本数。'
             '完整逐对值见matched_correct_wrong.csv；固定置信度分箱结果见conditional_diagnostic.csv。\n\n')
    text += '## 同一OOF协议下的所有校准结果\n\n'
    text += markdown_table(metrics[['dataset','modality','strength','method','ece15','binary_brier','binary_nll']])+'\n\n'
    text += ('E0为原始置信度，E1仅拟合原始C；M_joint联合使用C及Gaussian/Dropout各自μ/δ四维，'
             'M_gaussian/M_dropout各自使用C与对应μ/δ。没有JS输入或交互项。'
             '复用历史17个外层Dialogue折，内层GroupKFold5按合并留出Brier选择off/1/0.1/0.01；'
             '仅行为特征标准化，每次拟合只使用对应训练Dialogue。off精确退回同训练子集的E1。'
             '完整内外模型、训练ID、标准化器与系数见fold_parameters/。\n\n')
    text += '## 配对Dialogue Bootstrap校准差值\n\n'
    text += markdown_table(boots[['dataset','modality','comparison','candidate','candidate_strength','reference',
                         'reference_strength','metric','mean_difference','ci_low','ci_high']])+'\n\n'
    text += ('差值方向为candidate−reference；ECE15、Brier、NLL负值表示改善。'
             '复用历史2000次Dialogue抽样，对所有条件共享同一组抽样；不重新拟合校准器。'
             '全部区间均为单Seed探索性区间，未校正多重比较，不包含模型Seed与拟合不确定性；'
             '将10次重复的行为汇总视为固定量，没有另外重抽扰动repeat来量化测量不确定性。\n\n')
    text += '## 强度变化与响应差值区间\n\n'
    text += markdown_table(response_boots[['dataset','modality','corruption','population','comparison','measure',
                          'mean_difference','ci_low','ci_high','valid_resamples']])+'\n\n'
    text += ('响应差值为0.10−0.05，匹配差值为错误−正确。强度间噪声采用独立的确定性随机流，'
             '配对仅发生于话语与Dialogue层面，没有宣称逐repeat噪声配对。'
             '无配对话语的重抽样可能没有有效分母，明确报告有效/无效抽样数。\n\n')
    text += '## 解释边界\n\n'
    text += ('更强干扰带来更多翻转、更大μ/δ/JS或更低零值占比，只证明模型行为对干扰有响应；'
             '不能据此证明响应包含原始C之外的校准信息。增量校准应直接查看同折OOF的行为方法−E1差值，'
             '以及同方法0.10−0.05敏感性差值。所有预登记结果均报告，不以最优模态、指标或强度替换主条件。\n\n'
             'Validation已用于checkpoint选择及多轮方法诊断；Dialogue OOF仅隔离可靠性映射的拟合和选参，'
             '不构成独立确认。只有一个模型Seed，不支持五Seed结论或正式性能提升声明。'
             '原始分类预测不变；本批没有训练CMERC、读取Test标签、修改原始数据、温度缩放，'
             '没有自动扩展0.20或更多repeat。独立验证结果由根目录verification.json给出。\n')
    (root/'STRENGTH_RESPONSE_REPORT.md').write_text(text)
    compact = ('# 强度响应实验交付\n\n'
               '0.05历史主条件与0.10敏感性条件；Seed1475、两个完整Validation、四模态、两干扰各10repeat。'
               '完整中文结果见[STRENGTH_RESPONSE_REPORT.md](STRENGTH_RESPONSE_REPORT.md)。\n\n'
               '结果CSV位于results/：原始预测1636条、行为汇总26176条、特征矩阵13088条、'
               'OOF预测65440条、校准指标80条。behavior_summary含全部话语及有效视觉子集，'
               'split_half_stability含零值占比、5vs5相关及固定方向AUC；matched_correct_wrong保存原始置信度配对。'
               'bootstrap_results和response_bootstrap复用历史2000次Dialogue抽样。\n\n'
               'CPU重算：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
               '/data2/yb/conda_envs/cmerc-xp-repro/bin/python scripts/analyze_strength.py`。'
               '`--prepare`仅验证历史0.05并输出prepared/；`--behavior-only`完成行为分析但不拟合校准器。\n\n'
               '更强响应不能替代增量校准证据；本次仅单Seed探索。完整拟合参数与输入哈希位于results/；'
               '独立审计和推理provenance见根目录交付文件。\n')
    (root/'README.md').write_text(compact)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    build_reports(parser.parse_args().root)
