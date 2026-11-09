# 训练均值与随机训练 Donor 参考敏感性

## 对照目的

历史实验把训练集逐维均值作为收缩参考，但该点不一定有情绪中性含义，也不保证位于特征流形。本阶段增加 Random Training Donor Replacement：每个验证话语、模态和重复从同数据集 Training split 均匀抽取一条话语，不使用任何标签；Text-All 的四个数组来自同一 Donor。每个条件固定 5 个 Donor，并在五个模型 Seed 间复用相同 Donor 身份。

独立验证确认每个数据集的 Donor 身份在五 Seed 间完全一致，训练话语与多模态数组对齐无冲突。视觉 Donor 全零状态被保留并单独记录，没有事后剔除。

## 结果

下表比较 `rho=1` 的五 Seed Flip Rate 均值 ± 样本标准差。

| 数据集 | 模态 | 训练均值 | 随机训练 Donor |
|---|---:|---:|---:|
| IEMOCAP | T-All | 0.1184 ± 0.0230 | 0.1861 ± 0.0305 |
| IEMOCAP | Audio | **0.1696 ± 0.0335** | **0.3014 ± 0.0329** |
| IEMOCAP | Visual | 0.0057 ± 0.0036 | 0.0071 ± 0.0026 |
| MELD | T-All | **0.5441 ± 0.0394** | **0.6747 ± 0.0097** |
| MELD | Audio | 0.0424 ± 0.0032 | 0.0519 ± 0.0059 |
| MELD | Visual | 0.0121 ± 0.0015 | 0.0155 ± 0.0012 |

Donor 替换比均值替换更强，但两个参考在所有 Seed 上给出相同的核心排序：IEMOCAP 为 Audio > T-All >> Visual，MELD 为 T-All >> Audio > Visual。因此，数据集级主导模态结论不依赖训练均值这一单一参考；绝对敏感性幅度明显依赖参考选择，不能跨参考合并成一个指标。

## 视觉 Donor 缺失性

IEMOCAP 每 Seed 有 2630 个有效目标×重复观察，其中仅 8 个 Donor 为全零视觉；其样本量过小，跨 Seed Flip Rate 为 0.0750 ± 0.1118，不能稳定解释。MELD 每 Seed 有 5070 个有效观察，其中 1755 个使用全零视觉 Donor：

- 视觉存在 Donor：Flip Rate 0.0183 ± 0.0015，JS 0.000320 ± 0.000092。
- 全零视觉 Donor：Flip Rate 0.0144 ± 0.0020，JS 0.000200 ± 0.000050。
- 全部 Donor：Flip Rate 0.0169 ± 0.0013。

这表明 MELD 的训练视觉池本身含大量全零样本，且 Donor 缺失性会改变幅度。保留该分层比把全零 Donor 当成普通有效视觉更准确。

## 对正确性预测的参考依赖

Donor 敏感性没有产生稳定的增量 AUROC。IEMOCAP 中相对 strict probability baseline 的 Donor Visual 平均 AUROC 差为 +0.0149，但只有 3/5 Seed 为正；其他 Donor 模态总体为负。MELD 最好的 strict+Donor Text 平均差仅 +0.0007，方向为 3/5 Seed 正。参考改变没有把敏感性转化为一致的错误识别信号。

## 解释边界

Random Donor 是分布参考敏感性检查，并非情绪中性替换，也没有保持目标话语的语义或说话人属性。训练均值和 Donor 的排序一致支持“模型依赖结构可复现”，但两者的幅度差异说明样本级敏感性受干预参考显著影响。
