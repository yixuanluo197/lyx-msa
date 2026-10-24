# 第08组：MELD 学习率第二Seed独立确认

唯一登记的正式训练已完成：完整非因果CMERC，MELD、seed2024、split_seed1475、lr0.0002、batch16、40轮。其他科学参数保持XP04同Seed原值；采用验证集W-F1选模，零测试前向。

| Seed | 最佳验证W-F1：0.0001 → 0.0002 | 差值 | 末10轮均值：0.0001 → 0.0002 | 差值 |
|---:|---|---:|---|---:|
| 67137 | 65.24% → 66.07% | +0.830 pp | 62.909% → 62.983% | +0.074 pp |
| 2024 | 66.01% → 65.46% | −0.550 pp | 61.223% → 62.034% | +0.811 pp |

seed2024最佳Epoch为5，XP04控制为9。按第07组预登记规则，当前Seed属于主指标未改善，未确认首Seed的最佳验证F1收益；末段均值改善作为辅助事实保留。两Seed不构成统计显著或五Seed稳定提升证据。暂缓以0.0002开展最终五Seed和Batch8；本批未追加其他训练。

## 证据入口

- [完整中文报告](../../reports/CMERC_LR_SECOND_SEED_CONFIRMATION.md)
- [独立实验目录](../../experiments/lr_second_seed_confirmation/20261023T175409Z/README.md)
- [训练前登记](../../experiments/lr_second_seed_confirmation/20261023T175409Z/reports/PREREGISTRATION.md)
- [实际参数与完整argv](../../experiments/lr_second_seed_confirmation/20261023T175409Z/configs/registry.json)
- [对照结果CSV](../../experiments/lr_second_seed_confirmation/20261023T175409Z/results/comparison.csv)
- [最终验收](../../experiments/lr_second_seed_confirmation/20261023T175409Z/provenance/final_verification.json)
- [18张诊断图及PNG/SVG哈希](../../experiments/lr_second_seed_confirmation/20261023T175409Z/results/figures/manifest.json)

完整验收通过40轮训练/验证、2548个Batch、两个Checkpoint的模型/Adam/RNG及完整args/provenance，04–07共1555个历史文件保持原哈希。实际使用user23 GPU3，与seed2024历史控制同卡；已登记其他用户的小型共驻任务，270次采样未见新增外部PID，训练期间最小空闲9867 MiB。不将共享GPU下的耗时称为独占速度。

本入口为服务器本地交付；数据、权重和原始日志仍在独立批次保存。完整重算需要这些服务器产物，不宣称只凭导航或哈希即可重跑训练。此次没有执行测试评估或自动推送GitHub。
