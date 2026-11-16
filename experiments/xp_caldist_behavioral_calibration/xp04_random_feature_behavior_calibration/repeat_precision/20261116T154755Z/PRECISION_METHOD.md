# 随机扰动重复次数精度实验方法

固定IEMOCAP文本、MELD音频，Gaussian σ=0.05，Seed1475。保留历史K10_block0及两个新独立K10块，新增K30_pooled：每条话语先汇总全部30次，再形成一条训练记录。完整Validation为IEMOCAP527条/12Dialogue和MELD1109条/114Dialogue。

μ=mean(1[扰动预测≠原预测]×扰动最大置信度)；δ=abs(C−mean(扰动最大置信度))；JS只作诊断。δ不能平均三个10次δ，因绝对值非线性。K30的δ变小也可能来自正负置信度变化抵消，不能自动解释为可靠性提高。

校准器沿用冻结mainline_core：raw C +训练折标准化行为特征的双向Logistic，固定内层GroupKFold5、Brier选择off/1/.1/.01；off精确复用E1。IEMOCAP历史LODO12、MELD历史GroupKFold5保持原话语顺序。每个折保存完整训练/留出Dialogue、索引、标准化参数、全部内层候选与外层权重。没有选择最好块或搜索20次。

历史block0仅保存完整概率，未保存模型log-probabilities；inputs保留其缺失状态，以log_probabilities_available=False标识，不从概率伪造原始模型输出。blocks1/2保存的log-probabilities原样保留。

本轮只有Seed1475、同一批已探索的真正Validation。K30由三个K10块合并而来，条件相互嵌套，不能视为独立重复。仅有一组30次，15+15分半不是独立30次复现实验。Bootstrap复用2000次Dialogue抽样，不重新拟合，未涵盖校准器拟合、模型Seed或随机扰动的不确定性；多重比较未校正。Validation也用于原checkpoint选择和候选探索。本轮不使用Test、不重跑CMERC、不生成新噪声、不更改强度、不训练E3新结构。
