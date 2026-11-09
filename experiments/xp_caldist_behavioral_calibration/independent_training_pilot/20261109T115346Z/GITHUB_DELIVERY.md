# CMERC Protocol B Pilot 前置审计交付

本目录交付 Protocol B 正式训练前的只读审计、IEMOCAP 标签政策分析、Fold 1 DataLoader dry-run、XP04 训练代码适配审计和 GPU 调度计划。

当前状态是 `awaiting_required_researcher_decision`：Protocol B 已获研究者确认，但 IEMOCAP 的 53 条 `xxx → happy` 样本仍需在 Legacy Label Policy 与 Unknown-Label Mask Policy 之间确定主实验政策。因此，本交付没有生成正式 `approved_split_manifests.json`，也没有启动训练、拟合校准器、访问 Evaluation 标签或执行 Evaluation 前向。

主要证据：

- 53 条异常话语的完整清单位于 `results/iemocap_xxx_anomalies.csv`。
- 五折角色交叠与隔离预检位于 `results/split_freeze_precheck.json`。
- 双数据集 Fold 1 DataLoader 检查位于 `results/data_loader_dry_run.json`。
- GPU 型号、UUID、占用状态和 Pilot 调度计划位于 `results/gpu_preflight.json`。
- 总体门控状态位于 `verification.json`。

报告建议以 Unknown-Label Mask Policy 作为新独立确认性实验的主政策，同时保留 Legacy Policy 作为可选的 XP04 可比性敏感性分析。该建议尚未替代研究者的政策确认。
