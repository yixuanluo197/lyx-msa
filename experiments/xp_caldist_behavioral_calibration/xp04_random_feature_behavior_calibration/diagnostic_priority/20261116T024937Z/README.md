# XP04 特征扰动：优先级和诊断统计

只用已有10次joint扰动、两个数据集完整Validation、五Seed。

- [统计报告](STATISTICAL_DIAGNOSTIC_REPORT.md)
- [相近置信度诊断](CONDITIONAL_CONFIDENCE_REPORT.md)
- [实验优先级](EXPERIMENT_PRIORITY_REPORT.md) / [最终判断](FINAL_DIAGNOSTIC_DECISION.md)
- results/：15个原统计CSV及diagnostic_summary.csv、experiment_priorities.csv；保留1279个配对和16000条bootstrap统计。
- inputs/：8180条话语模型记录、49080条既有拆分记录、历史参考、冻结2000次抽样。

只读交付检查：python verify_delivery.py（标准库）。环境/data2/yb/conda_envs/cmerc-xp-repro/bin/python，三个BLAS线程变量均1。测试：python scripts/test_diagnostics.py。独立审计有效入口为scripts/verify_diagnostics_v2.py；V1错误等同μ与翻转率，原源码/失败JSON/日志保留，勘误见provenance/verifier_erratum.json，统计数据未改。

正式统计命令python scripts/diagnose.py --formal拒绝覆盖。重现应复制代码、协议、inputs到新目录，保持results为空；深度审计会写报告，应在副本执行，历史绝对来源路径需换机映射。V2在正式统计后新增，原before/after快照仍对应V1，未回写历史。

本轮是探索性诊断，没有新增校准性能结果；增加重复和原始数据处理均未执行。
