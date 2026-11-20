# Plan B：MELD文本匹配可靠性探索

本目录是2026-11-20已完成实验的公开交付副本。依据研究者后续明确要求上传，保留原始计算文件字节及SHA，不覆盖历史实验。原计算目录位于本地工作区；公开副本不包含完整原始文本、变换文本、完整Dialogue上下文、模型、媒体或特征向量。

## 研究结论

- Assertion、Probe、Sample只找到早期可行性条目，未找到可执行原始协议或可归属结果，保留待确认；没有自行定义或改名已有干扰。
- 冻结MELD Seed 67137。完整Validation按同Dialogue×固定0.1置信度区间匹配203对、406条目标、80个Dialogue；匹配在新干扰预测前锁定，不重训、不使用Test、不增加Seed。
- 两类×两版共1624条件，1544成功、80输入QC失败，全部保留。406条真实Clean负对照通过，独立61791项数值及产物核验通过。
- 语义改写出现局部μ/翻转率信号，但两个预先指定的主JS区间均跨零；行为排序相对原始置信度的增益尚未证明。中性追加未显示明确区分。
- 暂不正式开发或训练Reliability Score。建议先完成实际人工盲审，再考虑少量Seed的预注册复核；新Seed不能把复用Validation变成独立数据。

## 阅读入口

- [最终判断](FINAL_NEXT_STAGE_DECISION.md)
- [完整统计报告](MELD_TEXT_RELIABILITY_REPORT.md)
- [匹配队列与排除原因](MELD_TEXT_MATCHED_COHORT.md)
- [Assertion/Probe/Sample审计](ASSERTION_PROBE_SAMPLE_AUDIT.md)
- [数值核验](verification.json)、[最终报告复核](provenance/final_report_review.json)
- [冻结协议](configs/EXPLORATORY_PROTOCOL.json)、[配对锁](configs/COHORT_LOCK.json)
- [发布清单](PUBLICATION_MANIFEST.json)

`results/`保留全部1109条话语的匹配状态、冻结配对/原始概率、1624个条件及失败记录、逐目标和配对μ/δ/JS、支持率、置信区间和置信度排序对照。两个NPZ仅包含2000次Bootstrap的Dialogue抽样索引及统计值，不是原始特征或模型。

## 人工盲审与隐私边界

完整 `TEXT_REVIEW_PACKAGE.csv` 共1932行（历史480版与新增1452版），含原始全文和上下文，**只在本地，不上传GitHub**。本目录只发布不含数据行的 [字段模板](TEXT_REVIEW_PACKAGE_SCHEMA.csv) 和 [审核说明](TEXT_REVIEW_INSTRUCTIONS.md)。原始AI记录、独立AI二审、文本生成代码内嵌的全文及审核映射也全部保留本地。

所有人工字段仍空白、状态PENDING。公开报告中的AI审核探索性范围不能称为人工确认；人工确认范围为0对/NA。FAIL/UNCERTAIN版本仍保留统计身份和状态，不能把沿用原标签的参考转移解释为扰动后真实正确性或校准证据。

## 代码与复现限制

`scripts/`是本轮实际运行代码；冻结RoBERTa和CMERC流程依赖 [首轮Plan B](../20261120T143131Z/) 的脚本和本地授权资产。编码使用TEXT_repro，CMERC推理使用multimodalerc310；完整环境、命令、GPU UUID、PID、输入/代码/权重哈希见 `provenance/`。公开CSV/统计NPZ可用于核查结果和Bootstrap，但执行原始特征/模型核验需要未公开的同版本本地资产。

运行代码及哈希记录保留实际绝对路径，没有为发布静默修改。原目录包含私有资产的路径引用，不表示这些资产已上传；迁移环境需要显式配置路径和新输出目录，不能直接覆盖已冻结结果。

协议和原报告中的“不自动Commit/Push”“仅本地”描述实验结束时的执行边界。研究者随后授权本次公开交付；完整文本、特征和权重继续仅本地，实验科学结论没有改变。
