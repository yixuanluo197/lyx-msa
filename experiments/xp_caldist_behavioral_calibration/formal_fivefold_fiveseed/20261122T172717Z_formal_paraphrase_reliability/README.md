# Protocol B 五折五 Seed 语义改写预审

**状态为 STOPPED_AT_PREFLIGHT。** 已核对 50 个 CMERC Checkpoint 的 SHA-256 和冻结分组。两个数据集都缺少通过旧四层 RoBERTa 特征兼容性验证的提取器。语义改写、模型推理、S0–S6 拟合、Bootstrap 和 Evaluation 均未执行。

- [预审决定](FINAL_DECISION.md)
- [文本兼容性审计](TEXT_ENCODER_COMPATIBILITY_AUDIT.md)
- [历史暴露与审计读取偏差](HISTORICAL_EXPOSURE_AUDIT.md)
- [权重完整性](results/checkpoint_integrity.csv)和[分组元数据](results/frozen_split_metadata.csv)
- [冻结计划](EXPERIMENT_PROTOCOL.json)、[验证记录](verification.json)和[公开文件哈希](PUBLICATION_MANIFEST.json)

公开产物包含报告、汇总元数据与审计代码。逐项掩码内容、Evaluation 输入、模型权重和运行日志保存在原环境。`verification.json` 中的原始清单哈希对应本地审计目录。

`scripts/audit_formal_assets.py` 使用 jq 的字段白名单核对分组，并按文件字节验证模型权重。`scripts/verify_preflight_delivery.py` 核对停止状态和输入哈希。重跑需采用原工作区结构，读取已有冻结元数据及本地权重。
