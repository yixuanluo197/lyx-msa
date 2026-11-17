# CMERC 上游特征生产来源补充审计

**官方发布文件身份核验通过；原始生产链仍缺必要材料；Clean Parity 未运行。**

审计完成于 2026-11-16。本目录是随后经研究者要求上传的公开报告副本。

- [官方文件来源、哈希、结构和 split 比较](UPSTREAM_FEATURE_PROVENANCE.md)
- [RoBERTa、Tokenizer、BPE、Fairseq 与权重审计](ROBERTA_PIPELINE_AUDIT.md)
- [OpenSMILE、论文配置冲突及 MELD 300 维审计](OPENSMILE_PIPELINE_AUDIT.md)
- [Clean 特征及 CMERC 概率 Parity 状态](CLEAN_FEATURE_PARITY_REPORT.md)
- [最终判断与阻断材料](FINAL_UPSTREAM_AUDIT_DECISION.md)
- [MM-DFN 基础特征关系](MMD_FN_FEATURE_RELATIONSHIP.md)
- [脱敏验证摘要](verification.json)
- [公开来源、版本、文件哈希与结构摘要](source_summary.json)
- [Clean Parity 指标状态（均未运行）](clean_parity_metrics.json)
- [公开文件与本地源报告哈希](publication_manifest.json)

主要发现：CMERC/M3NET 两个官方 ZIP 内四文件与 XP04 一致，MM-DFN 两个基础文件也一致；COSMIC 参考代码及 tokenizer 已核对，但实际微调 encoder 权重/Fairseq 版本仍缺。音频存在文献与代码配置冲突，MELD 300 维生产链未确认。

IEMOCAP RoBERTa 包自带 Validation 为 647 条，XP04 为 527 条；XP04 的 12 个 Validation Dialogue 中 11 个位于该包自带 train 列表。这是上游微调可能重叠的条件性风险，缺原运行记录时不能宣布标签泄漏。

原始 ZIP、pickle、原文、音频、逐样本数据、模型权重、完整论文和本地源码镜像没有包含在本次交付中。文中提及的 `sources/`、`provenance/`、`local_only/` 和脚本继续留在本地；必要公开证据汇总于 `source_summary.json`。

本地绝对路径已替换为 `$REFERENCE_FEATURE_EXTRACTION`、`$PUBLISHED_CMERC_FEATURES` 等占位符；仓库路径按仓库根目录表示。原本地审计文件保持不变。

验证摘要的 PASS 只指证据完整性及发布文件身份，不能解释为生产链完整或 Clean Parity 通过；未测量指标为 null/NA。没有新训练、干扰或 CMERC 推理。审计阶段未提交推送，此公开副本随后经研究者明确授权发布。
