# XP04 原始文本与音频 Pilot 审计报告

**审计已完成；状态为 STOPPED_AT_AUDIT_GATE，Pilot 未启动。**

本目录按研究者要求发布审计报告与汇总验证。两数据集原始特征提取链路尚未闭环，不能启动真实输入干扰。该状态不代表原始干扰无效，也不提供新的行为指标或校准收益证据。

| 本轮 Validation 核验 | IEMOCAP seed1475 | MELD seed67137 |
|---|---:|---:|
| 话语 / Dialogue | 527 / 12 | 1109 / 114 |
| 文本、speaker、原始标签一致 | 527 / 527 | 1109 / 1109 |
| 候选原始媒体存在 | 527 / 527 WAV | 1108 / 1109 MP4 |
| 完整 PCM 读取通过 | 527 | 未解码 MP4 音频流 |

- [原始数据可用性审计](RAW_DATA_AVAILABILITY_AUDIT.md)
- [特征提取链路审计与恢复条件](FEATURE_EXTRACTION_AUDIT.md)
- [实验范围及正式 μ、δ、JS 定义](EXPERIMENT_PROTOCOL.md)
- [抽样状态](SAMPLE_SELECTION_REPORT.md)
- [干扰协议状态](PERTURBATION_PROTOCOL.md)
- [历史结论与复用边界](HISTORICAL_EVIDENCE_REVIEW.md)
- [最终判断及八项研究问题](FINAL_RAW_DATA_PILOT_DECISION.md)
- [脱敏验证摘要](verification.json)
- [公开文件及本地源报告哈希](publication_manifest.json)

公开副本保留研究判断、统计汇总、checkpoint/config 哈希和必要的非文本异常标识，隐藏本地绝对路径。`$RAW_IEMOCAP`、`$RAW_MELD`、`$REFERENCE_FEATURE_EXTRACTION` 为本地路径占位符；仓库路径按仓库根目录表示。

原始报告、审计/验证脚本、完整逐行索引、完整概率、媒体哈希及环境/执行记录继续保存在原本地审计目录。正文提及的 `local_only/`、`provenance/` 和 `scripts/` 文件均未包含在此公开副本中。原始文本、音频、模型权重和逐样本数据没有上传。

`verification.json` 的 PASS 仅指本地审计包完整性验证；不表示提取链路可用或 Pilot 完成。审计阶段未进行 commit/push；本报告随后经研究者明确授权提交和推送。
