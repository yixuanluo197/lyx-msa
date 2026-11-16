# 单模态Gaussian候选确认方法

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

## 证据范围与限制

- 真正Validation：IEMOCAP 527条/12个Dialogue；MELD 1109条/114个Dialogue。未使用Test。
- 两个候选来自已查看的同一Validation上的历史探索；新增模型Seed和随机块检验稳定性，不构成独立数据确认。
- 五个checkpoint评价同一批话语，不能视为五份独立数据。五Seed均值Bootstrap在每次共享Dialogue抽样后计算各Seed指标差，再取均值。
- Bootstrap为2000次固定OOF预测上的Dialogue配对抽样，不重新拟合；区间不包含校准器重拟合、Seed总体或随机扰动重复的不确定性。多项比较为探索性，未校正。
- Seed1475只有三个独立10次扰动块；不能据此断言任意随机噪声都稳定。三个块分别拟合和评价，没有合并成30次。
- 所有ECE均为0–1原始数值，例如0.04=4%；Brier/NLL为Top-label二元正确性口径。
- 调整原始预测正确的概率，不改变情绪类别。Accuracy、Weighted-F1、Macro-F1保持不变；AUROC只作区分能力诊断，不是额外成功门槛。
