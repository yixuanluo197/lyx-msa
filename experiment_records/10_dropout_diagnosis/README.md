# 第10组：MELD Dropout正则化对照

唯一正式训练已在本地user23 TITAN Xp GPU3完成40轮并通过独立CPU验收；完整非因果CMERC、seed2024/split1475、lr0.0001、Batch16、Dropout0.5、rec_dropout0.1，验证W-F1选模，零测试前向。科学配置唯一变化为Dropout0.4→0.5。

| 验证指标 | Dropout0.4（XP04） | Dropout0.5（XP10） | 差值 |
|---|---:|---:|---:|
| 最佳W-F1 | 66.01% | 66.49% | +0.480 pp |
| 最佳Epoch | 9 | 15 | — |
| 末10轮W-F1 | 61.223% | 62.775% | +1.552 pp |
| 末10轮分类Loss | 5.08928 | 5.32022 | +0.23094 |

按预登记满足第二Seed候选资格，但验证分类Loss反而上升，不能宣称整体过拟合已解决。仅单Seed开发证据，不启动第二Seed、五Seed或测试。实际--dropout同时控制文本GRU层间和分类头两个活动位置；rec_dropout0.1在当前构造路径仅作为未使用的元数据保留。

- [完整中文报告](../../reports/CMERC_DROPOUT_DIAGNOSIS.md)
- [独立批次](../../experiments/dropout_diagnosis/20261025T032711Z/README.md)与[预登记](../../experiments/dropout_diagnosis/20261025T032711Z/reports/PREREGISTRATION.md)
- [实际配置及命令](../../experiments/dropout_diagnosis/20261025T032711Z/configs/registry.json)与[源码协议审计](../../experiments/dropout_diagnosis/20261025T032711Z/reports/SOURCE_AND_PROTOCOL_AUDIT.md)
- [比较CSV](../../experiments/dropout_diagnosis/20261025T032711Z/results/comparison.csv)、[全量训练/验证曲线](../../experiments/dropout_diagnosis/20261025T032711Z/results/curves.csv)、[分项Loss](../../experiments/dropout_diagnosis/20261025T032711Z/results/loss_curves.csv)
- [独立验收](../../experiments/dropout_diagnosis/20261025T032711Z/provenance/final_verification.json)、[图表SHA](../../experiments/dropout_diagnosis/20261025T032711Z/results/figures/manifest.json)
- [GitHub交付范围](GITHUB_DELIVERY.md)与[当前发布清单](publication_manifest.json)

源输入49项冻结、04–09历史1881文件保全、40train/40validation/2548真实更新、best/latest第15/40轮及完整模型/Adam/RNG验收通过。原始数据、权重和原始日志仅保留服务器；公开22件图表、逐轮聚合CSV、配置、源码与验收信息。
