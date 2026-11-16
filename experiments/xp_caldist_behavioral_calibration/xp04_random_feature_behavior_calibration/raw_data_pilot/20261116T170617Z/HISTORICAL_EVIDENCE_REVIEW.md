# 已有证据及复用边界

本轮阅读用户指定 evidence sensitivity 的 FINAL、TEXT、AUDIO、VISUAL 与 PREREGISTRATION 报告，并只读核对以下六阶段。未重跑其任何实验。

## 表征敏感性

`evidence_sensitivity/20261108T095843Z/reports/FINAL_EVIDENCE_SENSITIVITY.md` 中目标话语完全向训练均值收缩的翻转率：IEMOCAP A 18.22%、T-All 12.14%、V 0.57%；MELD T-All 55.01%、A 3.97%、V 1.44%。这些是预提取向量干预，不能当作原始音频 pitch/timbre 或文本语义编辑的结果。仅保留 IEMOCAP 音频、MELD 文本的研究优先级，不从模态依赖推出错误或校准增益。

## 六阶段随机特征结论

路径均相对于 `experiments/xp_caldist_behavioral_calibration/xp04_random_feature_behavior_calibration/`：

| 阶段及已读材料 | 已有结论 |
|---|---|
| `mainline_mu_delta/20261115T165025Z/MAINLINE_PROTOCOL.md`、`FINAL_MAINLINE_REPORT.md:25–27,60` | 未证明超过 confidence-only E1 的额外可靠性信息；IEMOCAP 的 Brier/NLL 五 Seed 均变差，MELD 三指标平均变差。Validation 已多轮用于设计。 |
| `confidence_interaction/20261116T013631Z/METHOD.md`、`FINAL_DECISION.md:5–9`、`RESULTS.md` | C×μδ 交互未超过 E1；MELD 三指标五 Seed 全变差。 |
| `diagnostic_priority/20261116T024937Z/FINAL_DIAGNOSTIC_DECISION.md:3–5`、`EXPERIMENT_PRIORITY_REPORT.md`、`CONDITIONAL_CONFIDENCE_REPORT.md:20–48` | MELD Gaussian μ 约 99.13% 为零、Dropout 约 96.12%；Gaussian δ 分半相关约 .0717。同 Dialogue 且置信度接近的正确/错误配对八项 μδ 差值 CI 全跨零，覆盖有限。 |
| `strength_response/20261116T035211Z/RESULTS_SUMMARY.md:7–22,37–40` | .05→.10 增强响应，但 .10 的 24 个新增登记条件没有三指标同时优于 E1。 |
| `candidate_confirmation/20261116T125737Z/FINAL_CANDIDATE_DECISION.md:5–10`、`XP04_BASELINE_AUDIT.md`、`RANDOM_REPEAT_STABILITY_REPORT.md` | IEMOCAP text Gaussian / MELD audio Gaussian 局部候选的 proper-loss 改善各仅 1/5 Seed；未确认跨 Seed 增量。 |
| `repeat_precision/20261116T154755Z/FINAL_PRECISION_DECISION.md`、`MEASUREMENT_PRECISION_REPORT.md`、`RESEARCH_INTERPRETATION.md:22–49` | 10→30 降低置信度均值 MC SE，未一致提升 δ 排序；μ 非零仍仅 IEMOCAP 1 条、MELD 4 条，未带来登记要求下同步校准收益。 |

## 可复用 Validation 预测

使用 `evidence_sensitivity/20261108T095843Z/runs/{dataset}/seed{seed}/baseline_validation.npz`。IEMOCAP seed1475 为 527×6、MELD seed67137 为 1109×7；包含完整概率、label、预测、Dialogue、utterance index、speaker、原 batch 等字段。本轮验证 NPZ ID 集与指定配置 Validation 完全一致、无重复，概率有限、非负、归一化且 argmax 一致，和冻结数据的标签/speaker 对齐。

旧 `five_seed_validation/20261115T151109Z/baseline/original_predictions.csv` 同身份样本可作为将来对照：已只读核验 ID 与顺序一致；不同历史推理批次 IEMOCAP 最大概率差约 2.98e-7、MELD 为 0。未来 fresh clean baseline 应登记容差，不能要求跨批概率字节相等，也不能把 FP32 数值差当成真实干扰效应。

本轮没有真实干扰预测，无法比较“原始干扰优于随机特征干扰”。历史负结果不等于真实输入实验的负结果；本次 NO-GO 是资料和来源链不足。
