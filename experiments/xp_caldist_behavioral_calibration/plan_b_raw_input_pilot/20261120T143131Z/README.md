# 方案B：自主特征、单Seed CMERC与原始干扰Pilot

本目录为2026-11-20已完成实验的公开交付副本。原计算目录位于仓库外的本地实验工作区，历史实验未覆盖。公开文件按原始SHA逐字节复制；模型、完整特征、完整原文、文本改写/上下文和媒体仍仅在本地。`local_only/`不会上传。

## 完成到哪一步

1. 两个独立RoBERTa Large监督微调、新四层文本特征、冻结Wav2Vec2 Base音频特征已完成。
2. IEMOCAP Seed1475和MELD Seed67137的新M³Net、新Dialogue难度、新CMERC已完成；旧XP04文本/音频、Checkpoint与难度不作新模型输入或初始化。
3. 完整Validation Baseline已冻结；IEMOCAP Accuracy58.06%/Macro-F155.22%，MELD62.85%/51.20%。
4. 各120条同批目标的文本/音频Pilot已完成计算，2400条件全部留档，2371成功、29个QC失败保留。μ/δ/JS及2000次Dialogue bootstrap已完成。
5. 独立数值核验62723项通过。音频情绪保持尚待人工听审，因此其结果属于QC探索；文本为双轮AI审核。可靠性增量尚未证实，未进入五Seed或CaliDist模块训练。

## 阅读入口

- [Baseline报告](BASELINE_REPORT.md)：完整Validation六项指标。
- [原始干扰报告](RAW_DATA_PERTURBATION_REPORT.md)：质量控制、响应、正确/错误、置信度及Dialogue分析。
- [最终判断](FINAL_PILOT_DECISION.md)：保留的正向、负向信号与下一步建议。
- [独立核验](verification.json)、[审查记录](REVIEW.md)、[阶段记录](STAGE_RESULTS.md)。
- [本地产物与执行索引](RUN_INDEX.md)：权重/媒体/特征路径与实际运行环境。

`configs/`、`scripts/`和`training_records/`保留协议、版本与实际训练记录；`results/`包含匿名话语ID、完整类别概率和统计，未包含原文或特征向量。`provenance/`保留源文件/模型哈希、启动记录及工程修正。完整原始日志/私有工件的引用不表示它们已公开；仅持有本公共副本不能运行依赖私有资产的全量数值验证。代码中原本的本地路径未改写，以保留运行哈希；重新运行需配置同版本本地授权资产及独立输出目录。

[发布文件清单](PUBLICATION_MANIFEST.json)记录复制文件SHA及本地来源。此前协议中的“不自动Commit/Push”描述实验运行时边界；本公开副本由研究者随后明确要求上传而发布，不改变实验结论。

`core_snapshot/`及`provenance/`内的上游/历史源码按原字节保留，仅用作数学、版本与工程修正的依据。归档中出现的上游Test入口不是本轮执行流程；本轮实际入口为scripts/downstream.py及记录的各阶段命令，未用Test训练、推理或评价。归档源码与CSV保留原换行及空白以维持运行SHA，不作格式化。
