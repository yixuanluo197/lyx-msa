# MELD 文本 Clean Feature Parity

**状态：BLOCKED / NOT_RUN。** 没有找到已确认实际生产权重，论文也没有明确支持以未微调预训练模型直接替代。本轮成功重提取话语数为0；没有测量特征误差，不是观测到Parity失败。

## 固定目标与索引

使用Seed67137真正Validation，共1109条/114个Dialogue。现有ID清单已逐行核对effective config的对话顺序、row index及唯一性，保留本地。没有打开本轮参考特征值或原始话语文本，也没有编码任何tokens。表中形状是基于已锚定官方pkl与历史结构审计的预期目标，不能写成新提取或新测量的形状。

目标文件 `meld_features_roberta.pkl` 当前SHA-256：`58b2bdadc2eaec6abb40ac0cb3ffa3f31e929d636ece8197aecf1ca373a92951`，与此前官方Dropbox文件的完整SHA一致。Checkpoint SHA-256：`3f79f9eb3547e35ac15ad7ff4345d9b873d0c884a6f3a1ce60a7a7e47077e701`，仅按字节核验，没有反序列化。配置SHA-256：`5d6271460c882dc34bf55bbcdec96afd09bb6b963c238745e04a54fb20515f68`。

| 保存层 | 参考hidden索引 | pkl槽位 | 预期参考形状 | 重提取形状 | 数值指标 |
|---|---|---:|---|---|---|
| T1 | [-1] | 3 | (1109,1024) | NA | NOT_RUN |
| T2 | [-2] | 4 | (1109,1024) | NA | NOT_RUN |
| T3 | [-3] | 5 | (1109,1024) | NA | NOT_RUN |
| T4 | [-4] | 6 | (1109,1024) | NA | NOT_RUN |


MELD采用Seed67137的114个真正Validation对话，共1109条；不拼接或修改说话人、顺序与Dialogue。COSMIC参考ASCII处理是否属于CMERC原生产仍未确认，不能默认删除原文非ASCII内容，也不以重清洗后更接近目标为由选择方案。


## 预先固定的兼容标准

[协议](configs/TEXT_RECOVERY_PROTOCOL.json)在任何本轮Validation目标值访问前冻结。四层必须分别满足完整覆盖和有限性、每元素allclose(atol=rtol=1e-4)、MAE≤1e-5、RMSE≤1e-4、非零话语余弦平均≥0.99999/最小≥0.9999、所有目标非常数维相关有定义且p05≥0.999；常数/零向量单独计数。逐维分布记录均值/总体标准差差异与一阶Wasserstein。这里的相关性是坐标跨话语的描述统计，不是把同对话话语当独立样本做显著性检验。

当前缺权重导致这些检查均未执行。最大绝对误差、MAE、RMSE、平均/最小余弦、逐维相关性和分布差异全部为null/NA，不填0、1或Pass。参见[逐层状态CSV](layer_comparison.csv)和[机器可读状态](provenance/text_parity_status.json)。没有逐维数值CSV，因为不存在重新提取的数值。

## 执行边界与结论

无文本改写/追加，无训练、回归转换器或仿射匹配，无Validation反馈调参；未处理音频/视觉，未索引Test特征或标签。当前四层未通过（原因是尚未执行，而非已测数值不匹配）。因此不执行Seed67137原XP04的概率比较，也不能直接开始原始文本干扰。原始特征与checkpoint保持不变。

来源与缺失环节见[RoBERTa恢复报告](ROBERTA_SOURCE_RECOVERY.md)。历史已经存在Validation结构审计，“独立”仅指未来锁定提取方案后一次性验证、不把本轮结果用于调参，不声称研究者此前未见过该Validation。
