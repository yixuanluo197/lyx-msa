# Gaussian 单模态候选跨Seed与随机重复确认

## 固定研究范围

IEMOCAP 文本、MELD 音频，Gaussian σ=0.05，原始 XP04 五个 Seed，真正完整 Validation。两个候选来自既有探索结果，不能称为独立数据确认。

1. 历史 Seed1475 block0 复用原始 Pilot；其余四 Seed 各10次扰动。
2. Seed1475 新增 block1/block2，各10次，RNG repeat索引分别10–19/20–29；独立评价，不合并为30次。跨模型Seed保持扰动输入一致。
3. 同样 Dialogue OOF 比较 E0、E1、C+μ+δ（主模型）、C+μ、C+δ（预登记消融）。不增加JS、类别或交互项。
4. 沿用历史 raw-C Logistic、训练折标准化、固定内层GroupKFold5按Brier选择off/1/.1/.01正则。不得按外层结果更换方法。
5. 报告所有Seed/随机块、三项二元正确性校准指标、AUC诊断、行为覆盖与正确/错误置信度变化。共享Dialogue抽样2000次配对Bootstrap，五Seed不视为独立数据。

## 工程与停止边界

仅目标有效话语的指定已提取特征；文本同时处理T1–T4。其他上下文、qmask、umask、长度、图结构和标签保留。缺失视觉样本全部保留。Training统计量复用并哈希核对。逐run核验原始概率与冻结checkpoint一致、输入复原、权重及源文件不变、有限输出。GPU输出按历史FP32容差核验，不宣称bitwise一致。

仅本地运行；启动前确认GPU编号/UUID/资源，后台队列记录命令/PID/资源/退出状态。先smoke与独立预检，再全量运行。全部完成、独立复核后交付到既有GitHub仓库。禁止原始数据干扰、Test使用、模型重训和自动扩展新候选。
