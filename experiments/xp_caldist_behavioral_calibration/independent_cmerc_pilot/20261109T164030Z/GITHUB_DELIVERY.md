# CMERC Independent Training Pilot GitHub 交付

本目录公开交付 Protocol B 下 IEMOCAP 与 MELD 的两个 Fold 1、seed 1475 CMERC Pilot，以及独立 Calibration 流程的实现、配置、审计证据和报告。

## 已验证结论

- 两个 CMERC 均从随机初始化开始，严格使用批准 Manifest 和对应 Fold-local Difficulty。
- IEMOCAP 完成 80 epochs，best epoch 79，Development Weighted-F1 为 67.94%。
- MELD 完成 40 epochs，best epoch 14，Development Weighted-F1 为 70.97%。
- CapacitySafeBatchSampler、Curriculum schedule、IEMOCAP 23 条 Fold 1 unknown utterance 的 supervision mask，以及 11 条标签依赖损失路径均通过真实训练验证。
- best/latest checkpoint 均通过 strict reload；18 项单元测试及 4,440 项独立断言通过。
- Calibration 内部 grouped OOF 诊断成功完成，所有正温度校准均保持原始 argmax。
- Evaluation DataLoader 未构建，Evaluation 标签未读取，未执行 Evaluation 前向或指标计算。

## 结果边界

Calibration 的 full-fit 与 grouped OOF 指标只用于 Pilot 内部诊断，不能解释为独立 Evaluation 成绩。正式结论仍需在完成五折五 Seed 训练并冻结方法后，于封存的 Evaluation 集上确认。

## 公开范围

GitHub 版本包含报告、配置、实现、测试、紧凑结果表、训练与校准元数据、审计 JSON、GPU/磁盘记录及可复算哈希。以下本地产物不进入 Git：

- CMERC checkpoint 与角色特征 artifact（`*.pt`）；
- Calibration 完整概率二进制（`*.npz`）；
- Python cache；
- 逐 batch 和逐 epoch 的大体积 JSONL 日志；
- 与 runs 中 artifact metadata 重复的 prestable metadata 副本。

排除文件的身份与 SHA-256 已由 `provenance/final_artifact_hashes.json`、`provenance/source_hashes.json` 和 `verification.json` 保留。公开文件的逐文件 SHA-256、大小和排除边界见 `publication_manifest.json`。

最终研究判断见 `reports/FINAL_CMERC_PILOT_DECISION.md`。
