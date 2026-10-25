# 第13组：CMERC辅助损失梯度冲突与伪标签质量诊断

已完成且通过独立验收。固定四个真实Checkpoint与同一训练样本，只作前向及autograd.grad；没有新训练、optimizer.step或测试前向。

| 项目 | 实际范围 |
|---|---|
| 服务器 | user23，本地TITAN Xp GPU0，UUID见实际配置 |
| Checkpoint | XP04完整CMERC e8/e40；XP12 w/o HSCL e5/e40，MELD seed67137 |
| 样本 | XP04训练池固定32个Batch16，512个不同对话、4936话语，七类均覆盖 |
| 完成量 | 128个单元，768次模型前向、512次独立autograd |
| 完整性 | 2302个历史文件及四权重SHA未变；每case模型/buffer/RNG/.grad状态未变 |
| 主证据 | XP04两个快照CC共享主干负cos均28/32；后期MSCL加权尺度更高，唯一优先项规则未通过 |
| 下一机制建议 | 原分类Loss与CC的置信度条件方向关系；不直接确定新系数或扩大五Seed |

- [完整中文报告](../../reports/CMERC_AUXILIARY_LOSS_DIAGNOSTIC.md)
- [实际批次](../../experiments/auxiliary_loss_diagnostic/20261025T123240Z/README.md)
- [实际配置](../../experiments/auxiliary_loss_diagnostic/20261025T123240Z/configs/registry.json)
- [结果JSON](../../experiments/auxiliary_loss_diagnostic/20261025T123240Z/results/metrics_summary.json)
- [历史验证对照CSV](../../experiments/auxiliary_loss_diagnostic/20261025T123240Z/results/historical_validation_comparison.csv)
- [验收证据](../../experiments/auxiliary_loss_diagnostic/20261025T123240Z/provenance/final_verification.json)
- [图表及哈希](../../experiments/auxiliary_loss_diagnostic/20261025T123240Z/results/figures/manifest.json)
- [发布说明](GITHUB_DELIVERY.md)

二值伪标签描述信息移除前后的预测/置信度关系，不是情绪标签。默认0覆盖均错样本，不能等同可靠正关系。四离散快照的eval梯度不等于历史训练轨迹，XP12原训练HSCL有效贡献为0；离线参考梯度不代表其曾优化HSCL。

公共文件只含配置、源码、聚合CSV/JSON、中文报告和PNG/SVG；模型权重、原数据、逐话语诊断、采样ID和原始日志保留服务器。04～12历史产物未覆盖；公共manifest绑定本组文件SHA。
