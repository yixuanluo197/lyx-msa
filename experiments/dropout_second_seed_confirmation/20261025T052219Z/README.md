# 第11组：Dropout0.5第二Seed确认（已完成）

MELD seed67137 唯一40轮正式训练已完成并通过独立验收。Dropout0.5最佳验证W-F1 **66.41%（第17轮，+1.17 pp）**；末10轮 **62.440%（−0.469 pp）**。按预登记规则为**混合证据**：未确认完整跨Seed一致性，不直接扩大五Seed。零新增测试前向。

[完整中文报告](reports/CMERC_DROPOUT_SECOND_SEED_CONFIRMATION.md)、[比较CSV](results/comparison.csv)、[逐轮曲线CSV](results/curves.csv)、[Loss曲线CSV](results/loss_curves.csv)、[曲线及SHA清单](results/figures/manifest.json)、[独立验收](provenance/final_verification.json)、[预登记](reports/PREREGISTRATION.md)、[配置](configs/registry.json)、[源码审计](reports/SOURCE_AND_PROTOCOL_AUDIT.md)、[历史测试字段偶然访问说明](provenance/test_metric_access_incident.json)。

仅新增seed67137；seed2024 Dropout0.5复用XP10完成结果。两个Seed最佳值均升，但本Seed末段均值降、波动增大，且两Seed验证分类Loss均升高，不能宣称解决过拟合、统计显著或测试收益。原始数据、模型权重、逐Batch原始记录及日志留服务器，04～10历史文件未变（2011项哈希通过）。
