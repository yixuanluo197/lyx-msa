# 最终候选确认判断

## 核心问题

随机特征扰动的μ/δ能否在原始置信度之外提供增量可靠性信息？应优先比较C+μ+δ与E1，而不是只比较原始XP04。

| dataset | modality | ece15_difference_vs_E1 | ece15_improving_seeds | binary_brier_difference_vs_E1 | binary_brier_improving_seeds | binary_nll_difference_vs_E1 | binary_nll_improving_seeds | all_three_improving_seeds | descriptive_assessment |
|---|---|---|---|---|---|---|---|---|---|
| IEMOCAP | text | -0.003966 | 3 | 0.000591 | 1 | 0.001951 | 1 | 1 | 尚未确认预期的跨Seed增量校准收益 |
| MELD | audio | -0.000777 | 3 | -0.000011 | 1 | -0.000071 | 1 | 1 | 尚未确认预期的跨Seed增量校准收益 |

## 随机块方向一致性（Seed1475，三块）

| dataset | ece15_improving_blocks | binary_brier_improving_blocks | binary_nll_improving_blocks |
|---|---|---|---|
| IEMOCAP | 3 | 1 | 1 |
| MELD | 2 | 3 | 3 |

## Dialogue集中性检查

| dataset | block | seed | scope | method | metric | full_difference | dialogues | dialogues_improving | omission_difference_min | omission_difference_max | omissions_with_improvement | omissions_reversing_full_direction | refit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IEMOCAP | 0 | all | mean_across_paired_seeds | M_gaussian | ece15 | -0.003966 | 12 | 7 | -0.004996 | -0.001925 | 12 | 0 | False |
| IEMOCAP | 0 | all | mean_across_paired_seeds | M_gaussian | binary_brier | 0.000591 | 12 | 6 | 0.000228 | 0.000814 | 0 | 0 | False |
| IEMOCAP | 0 | all | mean_across_paired_seeds | M_gaussian | binary_nll | 0.001951 | 12 | 7 | 0.000833 | 0.002565 | 0 | 0 | False |
| MELD | 0 | all | mean_across_paired_seeds | M_gaussian | ece15 | -0.000777 | 114 | 68 | -0.001531 | -0.000212 | 114 | 0 | False |
| MELD | 0 | all | mean_across_paired_seeds | M_gaussian | binary_brier | -0.000011 | 114 | 66 | -0.000066 | 0.000072 | 103 | 11 | False |
| MELD | 0 | all | mean_across_paired_seeds | M_gaussian | binary_nll | -0.000071 | 114 | 66 | -0.000207 | 0.000126 | 112 | 2 | False |

该检查只从固定OOF预测中去掉一个Dialogue后重算指标，不重新拟合；omissions_reversing_full_direction表示去掉单个Dialogue后均值差值改变方向的次数。不能替代独立数据验证。

## 解释与停止边界

表中正负号和一致Seed数量描述结果；“多数Seed有改善”不代表独立样本显著性。必须同时阅读各项配对区间、单Seed/随机块结果以及per_dialogue_results.csv，不只报告最佳条件。若ECE有改善但proper loss恶化，需要保留这一取舍，不能称为全面校准收益。

本轮不自动训练或修改CMERC、不使用Test、不增加强度搜索、不处理原始文本/音频/视频。是否开展面部遮挡、音色处理等原始数据实验，等待研究者审核。即使出现跨Seed正向线索，独立数据上的最终证据仍需冻结方案后另行验证。

## 证据范围与限制

- 真正Validation：IEMOCAP 527条/12个Dialogue；MELD 1109条/114个Dialogue。未使用Test。
- 两个候选来自已查看的同一Validation上的历史探索；新增模型Seed和随机块检验稳定性，不构成独立数据确认。
- 五个checkpoint评价同一批话语，不能视为五份独立数据。五Seed均值Bootstrap在每次共享Dialogue抽样后计算各Seed指标差，再取均值。
- Bootstrap为2000次固定OOF预测上的Dialogue配对抽样，不重新拟合；区间不包含校准器重拟合、Seed总体或随机扰动重复的不确定性。多项比较为探索性，未校正。
- Seed1475只有三个独立10次扰动块；不能据此断言任意随机噪声都稳定。三个块分别拟合和评价，没有合并成30次。
- 所有ECE均为0–1原始数值，例如0.04=4%；Brier/NLL为Top-label二元正确性口径。
- 调整原始预测正确的概率，不改变情绪类别。Accuracy、Weighted-F1、Macro-F1保持不变；AUROC只作区分能力诊断，不是额外成功门槛。
