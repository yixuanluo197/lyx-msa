# 文本特征恢复：最终研究决定

**本轮完成来源核查，但两数据集的重新提取和Clean Feature Parity均因实际encoder权重未确认而阻断。** 不使用预训练或其他本地微调模型替代，不新增微调，也不通过仿射/回归制造Parity。

| 最终问题 | IEMOCAP | MELD |
|---|---|---|
| 找到CMERC实际使用的RoBERTa权重？ | 未找到可确认对象 | 未找到可确认对象 |
| 成功重提取目标Validation文本？ | 否；0/527，NOT_RUN | 否；0/1109，NOT_RUN |
| 四层是否通过Clean Feature Parity？ | 未验证，不能判通过 | 未验证，不能判通过 |
| 可否直接用原XP04开展文本干扰？ | 当前不具备兼容特征生成条件 | 当前不具备兼容特征生成条件 |

## 已增加的实证

1. 核对CMERC正式论文§3.2及其COSMIC引用，确认论文没有明确支持用未微调公开模型直接替代；COSMIC原文和代码明确要求数据集微调后的encoder。
2. 扩展本地权重盘点，找到了预训练大模型和自有微调权重，依据配置与生成代码逐项排除原生产身份；不能以“目录里有RoBERTa”认定恢复完成。
3. 仅读取COSMIC官方5.49GB包的128KiB尾部，完整列出36项中央目录，未见checkpoint命名条目；IEMOCAP/MELD特征与CMERC官方包大小和CRC32一致。没有下载/解包数据，也没有获得实际encoder。此元数据关系强于仅看README，仍不能充当payload SHA或权重身份。
4. 对现有官方目标特征、原始Checkpoint、配置及历史资料留档哈希；真正Validation IDs为Seed1475的527条和Seed67137的1109条，未使用文本包自带split替代。
5. 保存了最后四层顺序、BOS/pad/BPE、长度和归一化已知/未知项，明确模型内BN/四层平均与pkl生产的区别。

## 没有执行的部分

原始文本提取、分词、token导出、四层数值对比和CMERC概率验证均未执行。逐层CSV的全部数值指标为NA，机器状态为null；不是负的性能结果，更不是以缺数据证明模型无效。没有修改原始文本、音频、视觉、预提特征或checkpoint，没有使用Test标签，没有原始数据干扰、CMERC训练、转换器拟合或Git Commit/Push。

## 精确缺失项与下一步条件

需要每数据集实际生产encoder checkpoint的权重与可追溯来源/哈希、该run采用的训练split、Fairseq版本、BPE/dictionary参数，以及真实文本清洗、超长序列和四层输出处理记录。任务微调模型的名称相同或维度相同不足以证明身份。

得到这些资料后，才在新协议下固定方案并一次性验证真正Validation。若研究者另行授权“公开预训练候选探索”，必须单列为候选研究，不能改称本轮已恢复的CMERC原流程，也不能以这次Validation反复选参数后再宣称独立通过。

本轮停止，等待研究者审核。报告：[来源恢复](ROBERTA_SOURCE_RECOVERY.md)、[IEMOCAP](IEMOCAP_TEXT_CLEAN_PARITY.md)、[MELD](MELD_TEXT_CLEAN_PARITY.md)、[概率门控](XP04_TEXT_PROBABILITY_PARITY.md)、[验证证据](verification.json)。完整媒体与权重仍只保留本地，没有发布动作。
