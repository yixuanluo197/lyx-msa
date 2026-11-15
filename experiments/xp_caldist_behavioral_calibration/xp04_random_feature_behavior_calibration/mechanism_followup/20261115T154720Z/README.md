# XP04 后续机制实验（20261115T154720Z）

本轮仅复用已完成五Seed的真正Validation原始/扰动概率，不重新运行CMERC、不生成新扰动、不使用Test。研究两个具体原因：行为估计是否噪声过大，以及冻结基础映射的修正约束是否限制行为增量。

## 固定设计

[设计和实现计划](DESIGN_AND_PLAN.md)、[预登记参数](preregistration.json)。9方法及候选、主比较S_DA vs S、共享Dialogue bootstrap和停止规则在本轮新校准结果产生前固定。

## 第一批：重复测量诊断

[实验判断](DIAGNOSTIC_DECISION.md)、[完整报告](diagnostic/REPEATABILITY_REPORT.md)、[每话语模型特征](diagnostic/feature_matrix.csv)、[三种半样本汇总](diagnostic/split_reliability.csv)、[独立验证](diagnostic/verification.json)。8180条原话语、16360条两种扰动汇总、49080条半样本对比记录。

Gaussian D的重复一致性很低，Dropout D较稳定；μ因很少发生翻转而高度稀疏。可测敏感性不等于正确性信息，仍需与静态概率对照比较。

## 第二批：强静态对照与联合拟合

第一批提交时此项尚未正式执行。工程测试已通过，但不代表性能有效；后续完成后另行补齐校准报告、OOF数据和独立数值验证。

所有结果属于重复使用Validation的机制诊断，不能作为独立测试确认。五Seed共享同一批Dialogue，bootstrap也共享抽样；没有把Seed当独立数据。未修改原始特征、模型权重或情绪预测类别，不使用温度缩放。
