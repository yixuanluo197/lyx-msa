# 最终研究判断

## 对十个研究问题的回答

1. **IEMOCAP 的音频依赖是否稳定？** 是。Audio 在 5/5 Seed 的 `rho=0.5` 和 `rho=1` 条件中都是三模态最高敏感项；完全收缩平均 Flip Rate 为 0.1696 ± 0.0335。
2. **MELD 的文本依赖是否稳定？** 是。T-All 在 5/5 Seed 中显著主导，完全收缩平均 Flip Rate 为 0.5441 ± 0.0394，Accuracy 平均下降 26.24 个百分点。
3. **视觉低敏感性是否一致？** 是。两个数据集、全部 Seed 均保持低敏感；MELD 全零视觉样本会把全体 Flip Rate 从视觉存在组的 0.0132 稀释到 0.0121，但不改变排序。
4. **T1–T4 排序是否稳定？** 大体稳定为 T1/T2 > T3/T4。IEMOCAP T1 在 5/5 Seed 最高或并列最高；MELD T1 在 4/5 Seed 最高、T2 在 1/5 Seed 最高。具体编码层身份尚未确认。
5. **训练均值和随机 Donor 是否产生类似结论？** 对模态排序是；对绝对幅度不是。Donor 替换系统性更强，因此两种参考不能直接合并。
6. **敏感性能否在原始置信度之外识别错误？** 当前不能。所有主 AUROC 扩展的五 Seed 平均差均不为正，且没有模型在 5/5 Seed 一致改善。
7. **哪个模态独立预测价值最大？** 没有可重复的胜者。Visual 的平均 AUROC 损失最小，但效应接近零且方向不一致；它不能被称为有效信号。
8. **轻度稳定性与强度敏感性是否互补？** 在行为描述上互补：轻度稳定样本在中度收缩下仍会翻转。对正确性预测的增量互补性尚未得到支持。
9. **是否值得开发样本级 Reliability Score？** 不建议基于当前三项行为指标和统一公式继续。它们可用于模型依赖、类别偏差和干预响应研究，但缺少错误识别的跨 Seed 证据。
10. **是否进入置信度校准阶段？** 目前不建议进入 Behavior-aware Calibration。若后续需要校准，应先保留 Global Temperature Scaling 作为独立基线，并在新的独立验证设计上证明行为信号增量价值。

## 决策门槛

结果符合预注册的 **情况 B**：模态敏感性排序跨 Seed 稳定，但控制原始置信度后不能改善正确性预测。部分结果也体现数据集条件化：IEMOCAP 主要依赖 Audio，MELD 主要依赖 Text；然而这不是情况 D 中“某模态可靠性信号有效”，因为两边的 OOF 增量诊断均未通过。

本阶段到此停止，不执行联合模态干扰、Reliability Score、Temperature Scaling 或 Test split 评估。

## 可继续研究的方向

- 将模态收缩作为模型行为与数据集偏差分析工具，研究 IEMOCAP neutral/excited/frustrated 及 MELD fear/anger 的非语言依赖。
- 追溯 T1–T4 的确切 RoBERTa 提取层与池化过程，再解释数组间差异。
- 对 MELD 视觉缺失建立显式 missingness 建模，避免把全零输入与有效低幅视觉表征混为一类。
- 若再次研究可靠性，先采用独立数据或嵌套交叉验证，并评估跨干预参考稳定的特征；不要根据本验证集上的 OOF 结果回选公式。

## 证据索引

- Seed 级敏感性：`results/per_seed_sensitivity.csv`
- 跨 Seed 汇总：`results/cross_seed_results.csv`
- Donor 对照：`results/donor_reference_results.csv`
- 公开 OOF 预测与指标：`results/correctness_oof_predictions.csv`、`results/correctness_oof_metrics.csv`
- 配对 bootstrap：`results/paired_bootstrap_results.csv`
- 轻度/中度联合诊断：`results/historical_joint_diagnostics.csv`
- 完整性与独立复算：`verification.json`
