# 第09组：MELD Batch Size诊断

唯一登记正式训练已完成并通过独立CPU验收：完整非因果CMERC，MELD seed2024/split_seed1475、LR=0.0001、Batch8、40轮，验证W-F1选模、零测试前向。

| 指标 | Batch16（XP04） | Batch8（09） |
|---|---:|---:|
| 最佳验证W-F1 | 66.01% | 65.89% |
| 最佳Epoch | 9 | 9 |
| 末10轮验证均值 | 61.223% | 63.048% |

本Seed最佳验证W-F1未改善，不支持主指标提升；不自动扩大搜索。

原--batch-size同时改变训练与验证分组；BN/padding及未pack双向GRU可能影响验证预测，另有更新次数、HSCL配对、CC尺度和RNG消耗耦合。单Seed不能证明某内部机制或论文差距根因。

- [完整中文报告](../../reports/CMERC_BATCH_SIZE_DIAGNOSIS.md)
- [独立批次](../../experiments/batch_size_diagnosis/20261024T131956Z/README.md)
- [训练前登记](../../experiments/batch_size_diagnosis/20261024T131956Z/reports/PREREGISTRATION.md)
- [完整argv](../../experiments/batch_size_diagnosis/20261024T131956Z/configs/registry.json)
- [结果对照CSV](../../experiments/batch_size_diagnosis/20261024T131956Z/results/comparison.csv)
- [独立验收](../../experiments/batch_size_diagnosis/20261024T131956Z/provenance/final_verification.json)
- [图及SHA清单](../../experiments/batch_size_diagnosis/20261024T131956Z/results/figures/manifest.json)

未追加任何训练或测试评估；公开交付范围见[GITHUB_DELIVERY.md](GITHUB_DELIVERY.md)。数据、权重和完整原始日志在服务器独立批次保留。
