# 第08组：MELD 学习率第二Seed确认

唯一登记训练为完整非因果CMERC，seed2024、lr0.0002、40轮、验证集选模、零测试前向。

- [训练前登记](reports/PREREGISTRATION.md)
- [配置与完整argv](configs/registry.json)

本目录为独立批次；不覆盖历史、不自动扩展实验。正式运行已完成并通过[完整验收](provenance/final_verification.json)：40轮训练/验证、2548个Batch、两个Checkpoint、零测试前向。

结果：最佳验证W-F1为65.46%（第5轮），较XP04同Seed降低0.550个百分点；末10轮均值62.034%，提高0.811个百分点。按预登记规则，当前Seed未确认最佳F1改善，末段变化作为辅助线索保留，暂缓五Seed和Batch8。

- [完整中文报告](reports/CMERC_LR_SECOND_SEED_CONFIRMATION.md)
- [第08组入口](../../../experiment_records/08_lr_second_seed_confirmation/README.md)
- [两Seed对照CSV](results/comparison.csv)与[汇总JSON](results/metrics_summary.json)
- [训练曲线](results/curves.csv)、[全部分项Loss](results/loss_curves.csv)和[PNG/SVG图清单](results/figures/manifest.json)

源码与环境保持冻结版本。权重包含模型、Adam、进度和RNG，保留在本批次独立runs目录；训练后未评估测试集。资源共驻及作者课程生成史限制详见报告。本批完成不代表统计显著，也不自动启动后续实验。
