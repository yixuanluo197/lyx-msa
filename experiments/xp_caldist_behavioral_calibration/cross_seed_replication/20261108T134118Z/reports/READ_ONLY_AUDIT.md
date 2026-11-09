# 只读审计结果

审计通过。两轮历史实验的 artifact manifest 分别覆盖 151 和 429 个文件，所有 SHA-256 匹配；两个历史 `verification.json` 均为 `passed`。十个 XP04 `best.pt` 均成功反序列化、通过协议校验且参数有限，checkpoint selection protocol 为 Validation。

IEMOCAP 五个 Seed 共用同一 split hash，真正 Validation 为 12 个对话、527 条话语；MELD 为 114 个对话、1,109 条话语。各数据集的 train/validation/test 不相交，十份历史 Validation 原始概率的元数据顺序一致。训练均值和标准差从 Training Split 独立复算后与历史统计逐元素相同。

Donor 对齐可行：IEMOCAP 5,283 条、MELD 9,989 条训练话语均可由 `(dialogue_id, utterance_index)` 唯一索引，四个文本数组、Audio、Visual 没有长度错配，T-All 可以从同一 donor 取得四个数组。训练集中全零 Visual donor 为 IEMOCAP 20 条、MELD 3,407 条；协议保留全训练池并记录 donor missingness，后续分层报告。

Anchor 的 10 个主均值干预条件均已保存完整概率，可直接导入而不重新执行。Donor 对照历史上不存在，因此两个 Anchor 也需要运行 Donor。轻度扰动只覆盖 IEMOCAP seed1475 和 MELD seed67137，与中度敏感性的逐样本关联只能作为两个 Anchor 的诊断，不能称为五 Seed 证据。

审计时四张 TITAN Xp 均空闲。实验沿用本地服务器和历史环境；仅保存预测、指标和必要日志，不复制输入特征。
