# 第09组：MELD Batch Size诊断

状态：已登记，资源准入排队；尚未执行GPU探针或正式训练，不存在本次性能结果。

本阶段仅允许一次正式完整CMERC训练：seed2024、split_seed1475、lr0.0001、Batch8、40轮，以验证集W-F1选模、零测试前向。控制复用XP04同Seed/lr的Batch16，不重训；其他科学配置逐项核对。

- [独立实验目录](../../experiments/batch_size_diagnosis/20261024T131956Z/README.md)
- [训练前登记](../../experiments/batch_size_diagnosis/20261024T131956Z/reports/PREREGISTRATION.md)
- [源码协议审计](../../experiments/batch_size_diagnosis/20261024T131956Z/reports/PROTOCOL_AUDIT.md)
- [配置与完整argv](../../experiments/batch_size_diagnosis/20261024T131956Z/configs/registry.json)
- [冻结清单](../../experiments/batch_size_diagnosis/20261024T131956Z/provenance/frozen_execution.json)
- [CPU容量预检](../../experiments/batch_size_diagnosis/20261024T131956Z/preflight/capacity_cpu.json)

实际同一--batch-size同时把train/validation分组16→8；原未pack双向GRU及BN/padding意味着验证预测也可能受分组影响，不能声称纯训练Batch效应。本次不自动追加Seed、搜索或测试评估，历史04～08保持不变。

[资源阻塞状态报告](../../reports/CMERC_BATCH_SIZE_DIAGNOSIS.md)；该报告尚无Batch8性能结果。
