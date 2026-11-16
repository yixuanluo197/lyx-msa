# 样本选择状态

**NOT_STARTED：原始输入/提取链路 gate 未通过，未抽取任何 Pilot 样本。**

审计池是 IEMOCAP 527 条 / 12 个 Dialogue 与 MELD 1109 条 / 114 个 Dialogue，均由各自 XP04 effective config 和真实 Validation NPZ 双重核验。`local_only/validation_index_audit.csv` 仅供输入审计，不能当作抽样结果或全量干扰实验。

保留用户的后续抽样要求：固定随机种子；每数据集 120 条；同一组样本用于文本和音频；尽量覆盖 Dialogue、六/七类情绪、正确/错误预测和置信度区间；保留原始标签、预测和身份字段。抽样算法、种子及分层配额尚未执行或冻结，不能生成伪造的 `sample_manifest.csv`。

恢复执行前必须解决 MELD 缺失 dev 媒体；不能悄悄删除缺失话语后把可用池称为原完整 Validation。质量状态与最终分析资格需独立记录，不能基于干预响应或正负研究结果替换样本。分层抽样后的全部指标必须分别报告正确/错误组，并说明其不代表总体估计。
