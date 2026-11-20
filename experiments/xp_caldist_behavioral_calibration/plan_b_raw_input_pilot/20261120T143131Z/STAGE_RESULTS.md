# Plan B 阶段记录（实际完成）

实验目录：20261120T143131Z；服务器user23；IEMOCAP Seed1475，MELD Seed67137。本轮未自动Commit或Push。

|阶段|状态|实际证据|
|---|---|---|
|Train/Validation索引与媒体准备|完成，保留2条MELD缺失|reports/DATA_PREPARATION_REPORT.md；3895项data检查|
|两套RoBERTa监督微调与文本特征|完成|IEMOCAP第8轮最佳；MELD第3轮最佳；provenance/text_stage_*.json|
|冻结Wav2Vec2 Base音频特征|完成|新768维、缺失零向量+mask；provenance/audio_stage_*.json|
|新M³Net、Train Fit难度、新CMERC|两数据集完成|各runs完成锁、新difficulty及哈希；models阶段10156项检查|
|完整真正Validation Baseline|完成，未据此重调|BASELINE_REPORT.md；527/1109条|
|固定Train原始输入工程smoke|两数据集通过|原smoke报告与独立IO勘误均保留|
|固定120目标/数据集|完成|IEMOCAP12 Dialogue、MELD114；两组均59正确61错误|
|输入变换及审核|完成本轮计算审核；音频听审未确认|文本480/数据集有署名AI双审；音频720/数据集客观QC；全部失败/不确定保留|
|冻结CMERC原始输入Pilot|2400条件全部留档|2371成功、29输入QC失败，无重新抽样或改K|
|μ/δ/JS与Dialogue Bootstrap|完成|results/analysis/analysis_verification.json；2000次；14个不完整家族保留|
|最终独立数值核验|见verification.json|核验不代表可靠性增量或音频情绪保持已证实|

Baseline：IEMOCAP Accuracy58.06%/Macro-F155.22%，MELD62.85%/51.20%。MELD文本响应明显，但多数正确/错误差异区间跨零且无同Dialogue×置信度匹配支持；温和音频翻转弱。最终判断与限制见FINAL_PILOT_DECISION.md。无Test、五Seed、可靠性模块或后续扩展任务。

原阶段中间记录完整保留于provenance/STAGE_RESULTS_before_final.md。初次Wav2Vec2默认loader失败、缓存并发临时文件错误、下游环境导入失败、同机GPU资源调整、JSONL读取与旧smoke完成路径勘误，以及验证器首轮显示错误均有独立记录；已解决的工程问题没有删去。正式运行使用冻结代码与新资产，原始特征和模型未被覆盖。

最终核验：62,723项PASS，0failed/0pending；verification.complete=true。全部计算阶段已结束，音频情绪保持仍待人工听审；本轮不扩大范围。
