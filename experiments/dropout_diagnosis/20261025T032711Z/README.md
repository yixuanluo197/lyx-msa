# 第10组：MELD Dropout诊断

唯一正式40轮训练、独立CPU验收、报告和22件曲线已完成，零测试前向。Dropout0.5相比XP04同Seed的0.4：最佳验证66.49% vs66.01%，末10轮62.775% vs61.223%；满足第二Seed候选条件，验证分类Loss未改善。

[完整中文报告](reports/CMERC_DROPOUT_DIAGNOSIS.md)、[预登记](reports/PREREGISTRATION.md)、[源码审计](reports/SOURCE_AND_PROTOCOL_AUDIT.md)、[实际配置](configs/registry.json)、[CSV](results/comparison.csv)、[验收](provenance/final_verification.json)、[曲线](results/figures/manifest.json)、[公开交付入口](../../../experiment_records/10_dropout_diagnosis/GITHUB_DELIVERY.md)。

本组未追加第二Seed、五Seed、其他超参数或测试。数据、权重及原始日志保留本地独立批次。
