# MELD 与 IEMOCAP 音频生产链恢复

本轮为独立来源恢复审计，未覆盖历史实验。两个数据集尚未找回完整原始音频生产链，因此本轮没有启动raw Clean Feature Parity或CMERC推理。

- [MELD selector与声学前端来源](MELD_AUDIO_SELECTOR_RECOVERY.md)
- [IEMOCAP IS10生产来源](IEMOCAP_IS10_RECOVERY.md)
- [MELD Clean Feature Parity状态](MELD_CLEAN_PARITY_REPORT.md)
- [IEMOCAP Clean Feature Parity状态](IEMOCAP_CLEAN_PARITY_REPORT.md)
- [双数据集对照](CROSS_DATASET_AUDIO_PIPELINE_REPORT.md)
- [最终决定与缺失材料](FINAL_AUDIO_RECOVERY_DECISION.md)
- [机器验证](verification.json)
- [官方配置及版本静态比较](OPENSMILE_CONFIG_VERSION_COMPARISON.md)
- [执行范围与原链门控](EXPERIMENT_SCOPE.md)

`provenance/`保存来源检索记录、哈希与门控；`sources/`保存本地来源快照；`scripts/`保存静态核查和验证脚本；`local_only/`保存固定Validation候选与原媒体metadata，权限限制为本地。所有未运行数值为null，不填写0，也不称为失败。上轮13708条MELD身份对齐和24条末端编码器parity只复用既有记录。

本轮禁止训练、原始数据干扰、Test标签、改动XP04与历史实验、自动扩展数据集、Git Commit/Push。完成后等待审核；本地来源审计结束时未上传GitHub；本公开副本根据后续明确上传指令发布。

## 公开发布说明

研究者在来源审计完成后明确要求“上传到lyx的github”。报告中未Commit/Push等描述、验证中对应false值均指来源审计阶段；本次发布仅同步经过审核的报告和汇总，没有运行新实验。

`sources/`、`provenance/`、`scripts/`和`local_only/`均为本地审计证据位置，未随本公开副本上传。公开证据如下：

- [MELD来源恢复汇总](evidence/meld_recovery_summary.json)
- [IEMOCAP来源恢复汇总](evidence/iemocap_recovery_summary.json)
- [OpenSMILE版本与本地候选汇总](evidence/opensmile_summary.json)
- [已选特征静态核查](evidence/selected_assets_summary.json)
- [固定候选音轨metadata汇总](evidence/audio_stream_summary.json)
- [来源URL、版本和文件哈希目录](evidence/source_catalog.json)
- [公开文件及原始本地报告哈希](publication_manifest.json)

此目录不含原始媒体、特征向量、模型权重、论文全文、仓库镜像或逐话语记录。原始音频生产链仍不完整，未执行raw Clean Parity；所有未测量误差保持null。
