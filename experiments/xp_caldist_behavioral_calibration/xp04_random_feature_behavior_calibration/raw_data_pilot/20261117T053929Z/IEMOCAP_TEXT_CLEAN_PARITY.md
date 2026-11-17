# IEMOCAP 文本 Clean Feature Parity

**状态：BLOCKED / NOT_RUN。** 没有找到已确认实际生产权重，论文也没有明确支持以未微调预训练模型直接替代。本轮成功重提取话语数为0；没有测量特征误差，不是观测到Parity失败。

## 固定目标与索引

使用Seed1475真正Validation，共527条/12个Dialogue。现有ID清单已逐行核对effective config的对话顺序、row index及唯一性，保留本地。没有打开本轮参考特征值或原始话语文本，也没有编码任何tokens。表中形状是基于已锚定官方pkl与历史结构审计的预期目标，不能写成新提取或新测量的形状。

目标文件 `iemocap_features_roberta.pkl` 当前SHA-256：`4bd60c581a5b7c70e217e18439f741b191605fbd6f2cf4dc1a6dcb7fcaa18f06`，与此前官方Dropbox文件的完整SHA一致。Checkpoint SHA-256：`2c0d1a7a39627e12d1244d23a225ad3da03bbce071a118909921a547a2bfd106`，仅按字节核验，没有反序列化。配置SHA-256：`59e40ae7b7d63bb5f56806eedefb2f70c6797095455f63c083917b1141275369`。

| 保存层 | 参考hidden索引 | pkl槽位 | 预期参考形状 | 重提取形状 | 数值指标 |
|---|---|---:|---|---|---|
| T1 | [-1] | 2 | (527,1024) | NA | NOT_RUN |
| T2 | [-2] | 3 | (527,1024) | NA | NOT_RUN |
| T3 | [-3] | 4 | (527,1024) | NA | NOT_RUN |
| T4 | [-4] | 5 | (527,1024) | NA | NOT_RUN |


IEMOCAP必须沿用Seed1475 effective split，不能改用RoBERTa pkl自带valid_ids。历史已确认当前12个Validation对话中，11个出现在文本包自带train_ids，1个在其valid_ids。若这些IDs对应encoder有监督微调训练集合，则存在表示学习阶段与下游Validation重叠的条件风险；没有实际encoder/训练日志时，不能据此宣告已发生标签泄漏。这里只复用历史split审计及现有配置哈希，不使用包内列表替换当前split。


## 预先固定的兼容标准

[协议](configs/TEXT_RECOVERY_PROTOCOL.json)在任何本轮Validation目标值访问前冻结。四层必须分别满足完整覆盖和有限性、每元素allclose(atol=rtol=1e-4)、MAE≤1e-5、RMSE≤1e-4、非零话语余弦平均≥0.99999/最小≥0.9999、所有目标非常数维相关有定义且p05≥0.999；常数/零向量单独计数。逐维分布记录均值/总体标准差差异与一阶Wasserstein。这里的相关性是坐标跨话语的描述统计，不是把同对话话语当独立样本做显著性检验。

当前缺权重导致这些检查均未执行。最大绝对误差、MAE、RMSE、平均/最小余弦、逐维相关性和分布差异全部为null/NA，不填0、1或Pass。参见[逐层状态CSV](layer_comparison.csv)和[机器可读状态](provenance/text_parity_status.json)。没有逐维数值CSV，因为不存在重新提取的数值。

## 执行边界与结论

无文本改写/追加，无训练、回归转换器或仿射匹配，无Validation反馈调参；未处理音频/视觉，未索引Test特征或标签。当前四层未通过（原因是尚未执行，而非已测数值不匹配）。因此不执行Seed1475原XP04的概率比较，也不能直接开始原始文本干扰。原始特征与checkpoint保持不变。

来源与缺失环节见[RoBERTa恢复报告](ROBERTA_SOURCE_RECOVERY.md)。历史已经存在Validation结构审计，“独立”仅指未来锁定提取方案后一次性验证、不把本轮结果用于调参，不声称研究者此前未见过该Validation。
