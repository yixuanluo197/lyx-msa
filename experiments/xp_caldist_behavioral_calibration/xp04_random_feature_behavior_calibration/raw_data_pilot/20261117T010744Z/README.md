# MELD 官方音频特征包与 XP04 来源确认

**官方 300 维音频与 XP04 全部真实话语精确相同；1611→官方网络→300 已作小规模 clean 验证；WAV→1611 生产环节仍待补齐。**

- [官方包、完整目录与模型资产](MELD_OFFICIAL_PACKAGE_AUDIT.md)
- [13708 条真实话语的 300 维精确对齐](AUDIO_300D_PARITY_REPORT.md)
- [官方音频网络恢复（emotion 实际为双向 GRU）](BC_LSTM_RECOVERY_REPORT.md)
- [最终三个问题的回答与后续边界](FINAL_MELD_AUDIO_DECISION.md)
- [机器验证](verification.json)

本目录独立于历史审计，不覆盖旧文件。Test 仅比较无标签音频特征；没有使用标签或计算模型性能。所有新模型检查仅使用 3 个完整真正 Validation 对话，不运行 CMERC，不训练、不干扰原始音频。

`local_only/` 保存下载文件、分段恢复证据和逐话语 CSV，`provenance/` 保存完整索引、哈希、失败及机器结果，`sources/` 保存本地证据。原始特征、完整原文、权重与逐话语记录不随公开交付上传；公开版本只含报告及脱敏统计。先前本地审计与 XP04 输入保留原样。

## 公开交付范围

本公开目录为本地审计的脱敏副本。`local_only/`、`sources/`、`provenance/` 与审计脚本均保留本地，不包含于此目录；报告中的这些路径表示本地证据位置。公开机器证据见：

- [完整成员目录、哈希与下载结果](evidence/package_inventory.json)
- [全量音频身份核验汇总](evidence/feature_parity_summary.json)
- [原网络结构与 Validation 重算汇总](evidence/encoder_recovery_summary.json)
- [发布文件哈希及本地报告来源哈希](publication_manifest.json)

公开报告仅删除服务器绝对路径并调整本地证据引用。未上传归档、特征、权重、完整原文或逐话语 CSV。公开验证记录覆盖审计完成时的 99 项检查；发布完整性另由文件哈希和 Git 提交确认。
