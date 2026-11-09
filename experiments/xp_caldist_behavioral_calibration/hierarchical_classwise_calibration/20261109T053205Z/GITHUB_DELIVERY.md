# CMERC Hierarchical Predicted-Class Temperature Scaling Pilot 交付

本目录公开 CMERC 层次化预测类别温度缩放 Pilot。实验严格限制在真正的 Validation Split：每个数据集和模型 Seed 独立执行外层 Leave-One-Dialogue-Out，内层五折 Grouped Dialogue CV 只在外层训练对话中选择正则化强度。

## 实验范围

- 数据集：IEMOCAP、MELD。
- 模型 Seed：1475、67137、2024、3407、42。
- 方法：Vanilla CMERC、Global TS、不正则化 Predicted-Class TS、Hierarchical Predicted-Class TS。
- 推理时仅按原始预测类别选择正温度；真实类别不参与温度选择。
- 共生成 8,180 条外层 OOF 预测、630 个外层对话折和 3,780 个内层 λ 评估。

没有读取历史 Test 预测或标签，没有重新训练 CMERC，没有修改或重新选择 checkpoint，也没有使用行为扰动或模态敏感性特征。

## 主要结论

- IEMOCAP 中 Hierarchical TS 相对 Global TS 的平均 NLL、Brier 和 ECE15 分别变化 `+0.017228`、`+0.004770`、`+0.001312`；NLL 与 Brier 均为 `0/5` Seed 改善。
- MELD 中对应变化为 `-0.000006`、`-0.000025`、`+0.000521`，增量接近零且方向不一致。
- 层次正则化将类别 log-temperature 偏移相对不正则化方法缩小约 77%（IEMOCAP）和 80%（MELD），但没有建立超越 Global TS 的收益。
- 当前研究决策是暂停该方法，不访问历史 Test；MELD 继续采用 Global TS，IEMOCAP 需要新的独立 grouped calibration/evaluation 数据才能确认类别级假设。

## 验证与公开内容

目录包含预注册配置、完整分析与验证脚本、逐话语 OOF 概率、类别温度、指标、2000 次对话级 paired bootstrap、报告和 PNG/PDF/SVG 图片。

[`verification.json`](verification.json) 状态为 `passed`：全部结果 CSV 已从原始 Validation NPZ 完整独立重算并逐字节一致；所有 OOF 样本覆盖一次；E1、E2、E3 均未改变任何 argmax；类别温度拟合无失败或边界解。

主要研究判断见 [`FINAL_RESEARCH_DECISION.md`](FINAL_RESEARCH_DECISION.md)，复现说明见 [`README.md`](README.md)。本地绝对路径保留在配置和 provenance 中用于来源追踪；公开 CSV 足以复算报告指标，无需重新执行 CMERC 推理。
