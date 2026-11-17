# XP04 双数据集音频转换器反向重建

**IEMOCAP 与 MELD 均为 C：本轮未建立通过 XP04 兼容性门槛的转换器。**

本目录是 20261117T024213Z 本地实验经后续明确授权上传的公开副本。科学结果和负结果保留；服务器路径替换为 `/LOCAL/` 占位符，完整媒体、特征、参数和逐话语记录不公开。报告中的“未Git提交/推送”描述实验完成时的状态；随后研究者明确授权本次发布。

- [最终研究决定](FINAL_RECONSTRUCTION_DECISION.md)
- [方法比较](CONVERTER_METHOD_COMPARISON.md)与[全部候选统计CSV](converter_method_comparison.csv)
- [IEMOCAP重建](IEMOCAP_FEATURE_RECONSTRUCTION.md)与[Clean Parity](IEMOCAP_CLEAN_PARITY_REPORT.md)
- [MELD重建](MELD_FEATURE_RECONSTRUCTION.md)与[Clean Parity](MELD_CLEAN_PARITY_REPORT.md)
- [XP04概率门控：NOT_RUN](XP04_PROBABILITY_PARITY_REPORT.md)
- [汇总CSV](clean_parity_summary.csv)、[本地实验验证摘要](verification.json)
- [协议](EXPERIMENT_PROTOCOL.md)、[冻结配置v2](configs/reconstruction_protocol.json)
- [本地保留边界](LOCAL_ONLY.md)、[发布与运行说明](PUBLICATION_NOTE.md)、[发布清单](publication_manifest.json)

代码位于 scripts/，不包含任何权重或可直接部署的兼容转换器。合成测试可独立运行；原始音频实验依赖本地许可数据、原始官方模型及记录的OpenSMILE配置，公开版本不会自动下载这些资产。
