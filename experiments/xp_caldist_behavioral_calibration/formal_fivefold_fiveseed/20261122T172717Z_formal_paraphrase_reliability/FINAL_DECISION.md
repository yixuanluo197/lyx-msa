# Protocol B 五折五 Seed 语义改写评分的审计决定

**状态为 STOPPED_AT_PREFLIGHT。** 50 个 CMERC 最佳权重及冻结分组核验通过。MELD 和 IEMOCAP 均缺少与这些权重兼容的原始 RoBERTa 文本特征生产链。本轮在生成改写和打开 Evaluation 前停止。

## 核验结果

| 项目 | 结果 |
| --- | --- |
| IEMOCAP 权重 | 25/25 个实际 best.pt 文件的 SHA-256 与冻结记录逐一一致 |
| MELD 权重 | 25/25 个实际 best.pt 文件的 SHA-256 与冻结记录逐一一致 |
| 模型加载 | 已核对历史 CPU strict reload 记录。本轮只读哈希，未重新加载模型 |
| 冻结分组 | Protocol B 清单 SHA-256 与训练记录、历史验证和 readiness 记录一致。10 个 Fold 的角色 Dialogue 互斥，Calibration 和 Evaluation 的划组单元与本 Fold 其余角色互斥 |
| Evaluation 覆盖 | IEMOCAP 151 个 Dialogue，MELD 1,432 个，每个在本数据集的五个 Evaluation Fold 中出现一次 |
| 原始划分别名 | MELD 的 Evaluation 角色包括原始 Train 1,038、Validation 114、Test 280 个 Dialogue。根据历史来源元数据，IEMOCAP 包括 31 个原始 Test Dialogue。这些是 ID 计数 |
| 角色缓存 | 已核对正式索引中 32 个非 Evaluation 缓存的存在性及冻结角色汇总。未反序列化这些数据缓存 |

同 Fold 隔离能证明角色分割的结构正确。跨 Fold 轮转时，IEMOCAP 的 151 个、MELD 的 726 个 Evaluation Dialogue 曾属于另一 Fold 的历史 Calibration。轮转本身保留了同折隔离，但这批 Dialogue 并非整体从未参与历史分析。上游 RoBERTa 的标签暴露也未完全排除。封存的 readiness 状态仍有明确 Evaluation 准入边界，尤其需要确认重分组中的原始 Test 范围。[权重明细](results/checkpoint_integrity.csv) · [冻结角色汇总](results/frozen_split_metadata.csv) · [数据与模型核验](results/asset_audit_summary.json) · [历史暴露审计](HISTORICAL_EXPOSURE_AUDIT.md)

## 文本兼容性阻断

正式角色缓存直接从旧版官方 RoBERTa pkl 取四层 1024 维预计算向量。IEMOCAP 取槽位 2–5，MELD 取槽位 3–6。冻结 CMERC 使用这些向量，模型目录没有为新句子生产同源向量的编码链。

历史文本来源恢复未取得实际权重和可验证的抽取链。后续蒸馏候选也没有通过四层兼容性门。Plan B 的新 RoBERTa 和下游模型在自主特征空间内成套训练，现有单 Seed 结果不构成本轮 Protocol B 的兼容性证据。每条改写都需要由匹配编码器重新生成向量，语义标签保持无法弥补表示空间不一致。[文本兼容性审计](TEXT_ENCODER_COMPATIBILITY_AUDIT.md)

## 审计读取偏差

一次过宽的批准清单读取在工具输出中展开了 IEMOCAP 的逐项 unknown_label_mask 标签字段，可能涉及旧 Test Session。这是本轮的越界读取，字段未用于计算、分组选择或方案选择，也未在交付材料中重列。发现后停止展开这些字段，其余冻结分组复核使用角色 ID 和汇总计数白名单。独立的 Evaluation/Test 标签文件、特征缓存和 DataLoader 均未打开或构建。

## 研究问题的状态

| 问题 | 本轮判断 |
| --- | --- |
| μ、δ 是否有独立增益 | 未测得。没有合法的新干扰概率来拟合或评价 S2、S3 |
| μ、δ 是否互补 | 未测得。S4 未运行 |
| D_pred、D_abs 是否优于 μ、δ | 未测得。S5、S6 未运行 |
| 两数据集是否一致 | 没有同协议应答，无法比较 |
| 哪些指标可以冻结进入最终多分类校准 | 当前审计没有新增指标选择证据，既有单 Seed 结果只作为候选来源 |

K=2、K=5、S0–S6、Bootstrap 和 Fold/Seed 评分均未执行。恢复执行所需的条件是两个旧文本空间各有可验证的原始编码链，先在允许范围完成 clean 四层特征及冻结 CMERC 概率一致性检查，并明确确认 Evaluation 与原始 Test 别名的使用范围。[冻结实验计划与停止状态](EXPERIMENT_PROTOCOL.json) · [公开文件 SHA-256](PUBLICATION_MANIFEST.json)
