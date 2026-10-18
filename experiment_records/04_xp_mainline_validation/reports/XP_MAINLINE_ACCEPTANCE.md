# XP 主线验收

通过：20/20，两个阶段及全矩阵协议验收、40个checkpoint、20份保存预测独立重算、1200个epoch容量/课程重放全部通过。

验收时间：2026-10-16T10:31:07.773879+00:00；源摘要 `fafddfee606270f5626a01e3611ecebcdc8ce2af72536ada3508fe31316481a9`。

验收不设分数阈值；模型/Adam有限、epoch与validation选模对齐、每epoch test字段为空、完整随机状态及sampling元数据均核对。

CPU检查不证明跨GPU逐位复现；fixed_author生成独立性未知，不宣称train-only。完整证据见 ../results/acceptance.json。

阶段为4个锚点+16项后续，20个有效训练与20次独评均退出0；无效部分尝试和受控调度交接单独留存，见 [RESOURCE_RECOVERY.md](RESOURCE_RECOVERY.md)。完整公开交付及独立备份的文件级验收见 `../provenance/large_artifact_backup.json` 与公开04的 `export_manifest.json`。
