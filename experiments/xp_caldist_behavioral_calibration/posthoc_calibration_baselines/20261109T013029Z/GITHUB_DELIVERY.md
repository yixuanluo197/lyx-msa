# CMERC 五 Seed Global Temperature Scaling 交付

本目录是十个 XP04 validation-selected CMERC checkpoint 的标准全局温度缩放实验。每个 dataset/seed 只使用真正的 Validation Split 拟合一个正温度，然后冻结参数并在 Test Split 上评价一次。

## 主要结果

| 数据集 | 指标 | Vanilla 五 Seed 均值 | Global TS 五 Seed 均值 | TS−Vanilla | 改善 Seed |
|---|---|---:|---:|---:|---:|
| IEMOCAP | ECE15 | 0.067169 | 0.078398 | +0.011228 | 1/5 |
| IEMOCAP | Brier | 0.443966 | 0.446304 | +0.002337 | 2/5 |
| IEMOCAP | NLL | 0.818251 | 0.831229 | +0.012978 | 1/5 |
| MELD | ECE15 | 0.186665 | 0.037421 | −0.149244 | 5/5 |
| MELD | Brier | 0.525287 | 0.478671 | −0.046616 | 5/5 |
| MELD | NLL | 1.130949 | 1.012850 | −0.118099 | 5/5 |

MELD 的 ECE15、Brier 和 NLL 在五个 Seed 上均改善；IEMOCAP 没有稳定改善且三项均值退化。全部 21,165 条逐话语记录的 argmax 保持不变，因此 Accuracy、Weighted-F1 和 Macro-F1 与原模型完全一致。

## 数据与防泄漏

- IEMOCAP Validation：12 对话／527 话语；Test：31 对话／1623 话语。
- MELD Validation：114 对话／1109 话语；Test：280 对话／2610 话语。
- Validation 与 Test 对话 ID 零重叠。
- 温度拟合脚本只读取 Validation；Test 不参与参数选择。
- 未重新训练模型，未使用行为信号、Reliability Score 或其他校准器。
- Experiment 6 仅用于两个 anchor checkpoint 的 Test 原始指标身份复核。

## 公开内容

目录包含预注册配置、只读输入审计、拟合与评价脚本、十个校准概率 NPZ、完整逐话语概率 CSV、2,000 次 dialogue-level paired bootstrap、论文图、报告和 SHA-256 来源清单。不包含 checkpoint、原始媒体或预处理输入特征。

独立验证状态见 [`verification.json`](verification.json)：温度重新拟合、Validation NLL、完整校准概率、Test 指标和全部 bootstrap 汇总均以零误差复算通过。主要研究判断见 [`reports/FINAL_RESEARCH_DECISION.md`](reports/FINAL_RESEARCH_DECISION.md)。

本地绝对路径保留用于来源追踪。重新执行完整输入审计需要本地 XP04 checkpoint 和历史预测文件；公开目录中的保存概率足以复核 Test 指标与结论。
