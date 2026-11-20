# 新文本特征提取：已完成

|数据集|实际训练轮数|所选最佳轮数|内部Dev Macro-F1|Train特征数|Validation特征数|
|---|---:|---:|---:|---:|---:|
|IEMOCAP|10|8|0.547265|5283|527|
|MELD|5|3|0.478886|9989|1109|

两个独立RoBERTa Large均从同一公开预训练权重初始化，以各自Train Fit情绪标签进行监督交叉熵微调，未使用旧XP04文本特征、旧学生模型或官方Validation选择轮次。IEMOCAP训练10轮、MELD在第5轮触发patience2早停；各自取内部Dev Macro-F1严格改善的最佳模型。

提取最后四层BOS向量，顺序[-1,-2,-3,-4]，每话语[4,1024] FP32，无额外归一化。Tokenizer和BPE来自同一公开模型，BOS0/PAD1/EOS2、右侧padding和最大512 token右截断；实际Train/Validation均无截断、无空文本。IEMOCAP原文不变；MELD按固定参考流程移除非ASCII字符，不改大小写或空白，Train2742条、Validation309条受该固定预处理影响。

Fit/内部Dev分别为IEMOCAP4747/536、MELD8989/1000，均按Dialogue隔离。未对全Train再拟合，未使用Test。原始全部话语ID、Speaker与顺序保持。最佳权重、Tokenizer、配置及完整断点只留本地；模型、特征与代码哈希见provenance/text_stage_*.json和local_only/features/*/text_features.manifest.json。

独立特征阶段验证已通过9010项数据、权重、特征形状、缺失掩码和哈希检查（当时尚未宣称整个Pilot完成）。微调逐轮CE与内部指标见results/text_training_history.csv。完整下游分类性能仍须等待新CMERC冻结后的Validation评估。
