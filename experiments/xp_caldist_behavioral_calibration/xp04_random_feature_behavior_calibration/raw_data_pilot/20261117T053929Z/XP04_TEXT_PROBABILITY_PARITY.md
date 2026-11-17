# XP04 文本概率一致性验证

**IEMOCAP Seed1475、MELD Seed67137：均为NOT_RUN。** 四层文本特征没有完成Clean Feature Parity，因而按用户要求禁止继续概率验证；不是概率比较后发现失败。

| 数据集 | softmax最大误差 | softmax平均误差 | 类别一致率 | 最大置信度差异 | 原因 |
|---|---|---|---|---|---|
| IEMOCAP | NA | NA | NA | NA | 缺少已确认文本encoder，四层未验证 |
| MELD | NA | NA | NA | NA | 缺少已确认文本encoder，四层未验证 |

两checkpoint仅进行文件哈希核验，未反序列化、未执行CMERC模型前向。没有Softmax数组或预测CSV；不能把未测量写成零误差或100%一致，也不能宣称类别不一致。概率门未打开，因此本轮没有替换任何输入模态，更没有修改其他模态/话语/对话结构。

未来只有每数据集四层全部通过预定特征门，才能使用原模型冻结推理，保留音频/视觉、Speaker与Dialogue结构，仅替换四路文本；使用原norm=BN配置，不按论文LN描述改模型。原/重建两套softmax、类别和最大置信度须同时保存，概率最大误差≤1e-4、MAE≤1e-5、类别一致100%、最大置信度误差≤1e-4。

权重缺失与归一化阶段的证据见[来源恢复](ROBERTA_SOURCE_RECOVERY.md)；本轮状态为[JSON](provenance/text_parity_status.json)。
