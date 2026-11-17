# 上游特征补充审计最终判断

> 公开报告副本：本地绝对路径已替换；文中 `sources/`、`provenance/` 和 `local_only/` 详细记录仅保留本地。公开证据摘要见 `source_summary.json`。

**发布资产身份核验完成；完整原始特征生产链仍未闭环；Clean Parity 未执行。** 不启动真实输入干扰，不训练 CMERC 或替代 encoder，不修改历史实验。

## 相比上一轮新增的确定证据

1. 新下载 CMERC、M3NET 官方链接的两个 ZIP，字节数与 SHA-256 完全相同；其中四个 pickle 均与 XP04 对应文件一致，结构、样本数量、宽度及存储 split 已核对。现在可以确认 XP04 的发布特征版本身份。
2. MM-DFN 官方固定 commit 的两个基础 pickle 也完全一致，建立共享基础资产的直接证据。其旧文本是 100/600 维 TextCNN/GloVe 路径，不能当作四路 1024 维 RoBERTa。
3. COSMIC 六个指定脚本及辅助流程已按固定 commit 核对；共 14 项本地参考文件一致，3 项 tokenizer/dictionary 又与当前官方 Fairseq 下载一致。仍没有实际 dataset fine-tuned encoder checkpoint 和 Fairseq 生产版本。
4. M3NET 补充材料与代码的 IS10/IS13 说法存在冲突，必须保留。MELD 官方音频文档中的原始高维声学特征及学习后表示说明，不能被简化为“存在一个原生 300 维 OpenSMILE 配置”。未找到精确映射到 XP04 300 维资产的配置/权重。
5. IEMOCAP RoBERTa 文件自带 Validation 为 647 条，与 XP04 527 条不同。XP04 12 个 Validation Dialogue 中 11 个落在 RoBERTa 包自带 train 列表，须核对上游微调可能重叠；缺原运行材料时不宣布标签泄漏。

## 结论分层

| 问题 | 结论 |
|---|---|
| 本地 XP04 是否拿错当前官方特征文件？ | 四文件均一致，没有发现版本错拿。 |
| 找到的 COSMIC 代码是否结构匹配？ | 是；但结构匹配不证明 CMERC 实际 encoder 权重身份。 |
| Tokenizer/BPE 是否有明确参考来源？ | 有；当前参考资产一致，但历史 checkpoint 参数和 Fairseq 行为仍须闭环。 |
| IEMOCAP 1582 / MELD 300 能否直接确定音频配置？ | 不能；生产命令/配置缺失，论文和消费者说法有冲突。 |
| MM-DFN 能否解除阻断？ | 能闭环基础发布文件身份，不能补齐 RoBERTa 与音频生产配方。 |
| 能否重提取少量 clean 并比较预测？ | 当前不满足先决条件；四个数据集/模态组合均 NOT RUN，误差/cosine/概率差为 NA。 |

## 后续需要取得的具体材料

- IEMOCAP、MELD 实际生成发布特征的 fine-tuned RoBERTa encoder、字典/BPE 参数、Fairseq 版本、原命令和训练/选择 split manifest；预训练初始化包或 CMERC 分类权重均不等价。
- 实际 IEMOCAP 音频 OpenSMILE 版本、`.conf`、完整特征顺序及切段、采样率/声道、聚合、归一化参数；解释其与补充材料 IS13 声明的关系。
- MELD 300 维生产链、必要声学原特征与学习/投影/选择步骤及权重，不能假定是某个默认 OpenSMILE 配置或某层 bcLSTM。
- 同版本 MELD dev 缺失媒体的恢复，并验证 clean 输入身份。然后才做少量真正 Validation 的逐维特征及冻结 checkpoint 概率 parity。

## 交付与边界

已交付用户指定 5 份报告，另附 MM-DFN 独立报告、机器证据、固定来源快照和验证记录。本轮只增加新的时间戳目录，没有简化或改动 CMERC 代码，没有新依赖，没有运行 Test 标签评价或模型推理。

原始 ZIP、基础 pickle、源码镜像中的语料及本地路径/执行信息只保留本地；不能把整个 `sources/` 或 `local_only/` 目录整体发布。后续文件交付目标沿用 `yixuanluo197/lyx-msa`。审计阶段遵守“不自动 Git Commit 或 Push”，未提交或推送；此公开报告副本随后经研究者明确要求上传。

本轮到此停止，等待研究者审核。详细的 NA 指标与恢复验证步骤见 [Clean Parity 报告](CLEAN_FEATURE_PARITY_REPORT.md)，汇总验证状态见 `verification.json`。
