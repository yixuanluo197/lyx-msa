# 04 XP 主研究基线

批次：`/data2/yb/paper/lyx-msa/CMERC/experiments/xp_mainline_validation/20261016T050823Z`。20个预注册任务均完成，完整验收passed；五seed配对结果见 `reports/XP_MAINLINE_REPRODUCTION.md`，逐项技术验收见 `results/acceptance.json`。

唯一批准的正式调度入口是 `provenance/runtime_recovery_v2/xp3_scheduler.py` 对应的原批次 `state/recovery/xp3_scheduler.py`：在冻结调度器之上强制排除GPU0/1，只分配健康、空闲的XP3。完整续跑控制器为同目录 `continue_pipeline.py`，验收使用 `finalize_mainline_v2.py`（4锚点+16任务）。本批已完成，不应重跑或覆盖当前run；新实验另建批次。

冻结 `tools/formal_pipeline.py` 和 `tools/finalize_mainline.py` 保留原初控制流程的历史证据，不再建议直接运行；资源竞争的失败attempt、原pipeline退出130和恢复审计均保留，未按分数选择重试。对应失败权重/日志位于批次与独立备份的 `attempts/`；公开包只含恢复元数据。

CPU测试排除及原因见 `preflight/test_scope.json`：`test_multiseed_scheduler.py` 为V100专用规则；`test_caldiff_batching.py` 及个例 `test_causal_checkpoint_roundtrip_and_architecture_rejection` 依赖快照未带的caldiff生成helper，均未被冒充为通过。原始失败见独立备份的 `logs/cpu_tests*.log`。XP工程回归入口为 `code/tests/test_xp_scheduler.py`；CPU测试不证明GPU数值或跨卡逐位等价。

`curriculum_source=fixed_author`：共用作者外部difficulty。生成过程未公开且文件包含当前validation键，不能称为train-only curriculum，也不能证明完整预处理validation-free。测试集未用于选择epoch/seed/超参数。

大型产物的独立副本在 `/data2/yb/paper/experiment_backups/CMERC/04_xp_mainline_validation/20261016T050823Z`，逐文件SHA256相同且inode独立；该目录仍在同一/data2文件系统，不是异盘或异地灾难恢复备份。备份清单：`provenance/large_artifact_backup.json`。

公开包不含权重、逐样本预测、原始日志、数据/特征/难度文件、symlink或缓存。路径与哈希仅作来源引用，不等于上游特征生成独立性已经证明。01/02/03历史树逐项核对 `provenance/history_trees.json`，未覆盖或修改。每个公开文件的source/target SHA256见 `export_manifest.json`；清单自身不自引用。
