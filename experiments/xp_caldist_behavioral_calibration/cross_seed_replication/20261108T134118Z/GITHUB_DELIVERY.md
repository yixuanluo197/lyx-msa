# CMERC 跨 Seed 模态证据敏感性复现交付

本目录是 Validation-only 跨模型 Seed 复现实验的公开交付副本。目标仓库为 [`yixuanluo197/lyx-msa`](https://github.com/yixuanluo197/lyx-msa)，目录位置为 `experiments/xp_caldist_behavioral_calibration/cross_seed_replication/20261108T134118Z/`。

## 实验范围

- 数据集：IEMOCAP、MELD。
- 模型 Seed：1475、67137、2024、3407、42。
- 复用历史 Anchor：IEMOCAP seed1475、MELD seed67137 的训练均值干预结果。
- 新增运行：其余8个 checkpoint 的10项主敏感性条件，以及全部10个 checkpoint 的 Text/Audio/Visual 随机训练 Donor 对照。
- 正确性诊断：每个数据集和模型 Seed 独立执行 Leave-One-Dialogue-Out；Logistic Regression 固定 L2、`C=1`；2000次 dialogue-level paired bootstrap。

没有重新训练 CMERC，没有访问 Test split，没有执行联合模态干扰、Reliability Score 或 Temperature Scaling。

## 主要结论

- IEMOCAP 的 Audio 依赖在5/5 Seed中稳定，`rho=1` Flip Rate 为 `0.1696 ± 0.0335`。
- MELD 的 T-All 依赖在5/5 Seed中稳定，`rho=1` Flip Rate 为 `0.5441 ± 0.0394`。
- Visual 在两个数据集的全部 Seed 中均保持低敏感。
- 训练均值与随机训练 Donor 得到相同的模态排序，但绝对敏感性幅度不同。
- 中度敏感性特征在控制原始概率后没有稳定改善正确性预测；没有主扩展模型在5/5 Seed上一致改善AUROC。

研究判断为预注册的情况B：模态依赖可重复，但当前行为指标不支持继续开发统一 Reliability Score 或进入行为感知校准。

## 公开内容

本目录包含：

- 预注册配置与完整分析脚本；
- 原始 Validation 概率和所有干预概率；
- 逐话语、逐 Seed、类别级和跨 Seed CSV；
- 固定 OOF 预测、指标与配对 bootstrap 结果；
- SVG/PDF/PNG 论文图及视觉质检记录；
- 完整运行日志、输入哈希、条件哈希和独立验证结果。

不包含模型权重、原始媒体、原始文本或逐话语输入特征。NPZ 仅保存预测、元数据、干预身份与必要校验字段。

## 验证状态

[`verification.json`](verification.json) 的状态为 `passed`：

- 130个条件、204500个干预概率向量复算；
- 130条Seed级汇总复算；
- 180条OOF指标复算；
- 640条bootstrap结果复算；
- 24次Donor跨Seed身份检查通过；
- `validation_only=true`、`test_split_used=false`、`models_retrained=false`。

完整结论见 [`reports/FINAL_RESEARCH_DECISION.md`](reports/FINAL_RESEARCH_DECISION.md)，输入与源码哈希见 [`provenance/source_hashes.json`](provenance/source_hashes.json)。原配置保留本地绝对路径用于来源追踪；完整GPU重跑仍需本地XP04 checkpoint与预处理特征。
