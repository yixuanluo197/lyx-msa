# Train配对音频转换器重建协议

本轮明确授权候选OpenSMILE流程及Train拟合的功能重建，不能把前轮缺原生产链当成停止尝试的理由；也不能将重建称为原算法恢复。最终A/B/C定义、候选范围、三折Dialogue GroupKFold、折内拟合规则、阈值、缺失处理及单次Validation顺序在 `configs/reconstruction_protocol.json` 冻结。

IEMOCAP只使用seed1475 effective Train5283/Val527；MELD只使用seed67137 Train9989/Val1109。共享pickle反序列化会载入历史容器，但只索引Train目标和许可的最终Validation目标，绝不索引Test特征或标签。转换器学习不需要情绪标签。

必须透明说明：历史审计已查看部分Validation汇总与24条末端GRU parity；它不是一个从未见过的新数据集。本轮协议冻结后，Validation目标及候选提取只在最终方案/参数/代码哈希锁定后打开一次，结果无论好坏不回流调参。

所有列匹配、仿射、speaker参数、scaler、投影及正则参数都在每折fit Train中确定。确定性逐行变换只能读取当前输入，不能使用held-out目标统计。MELD最多256条Train probe仅筛前端，不据小样本高维回归判成功；锁定前端后用完整可用Train做转换器CV。

特征门要求固定Validation全覆盖，逐元素allclose(atol=rtol=1e-4)、MAE≤1e-5、Train尺度NRMSE≤1e-3、非零行cosine均值≥0.99999且最小≥0.9999、非常数维相关第5百分位≥0.999。MELD还须完整Dialogue经原BiGRU后真实300维同过，33步padding单独报告。只有特征门通过才运行冻结XP04概率门，参数阈值见JSON。

任何Train推导关系即使零误差也属于重建候选；作者原资产来源未找回时不能判A。B必须特征和冻结概率门均通过；缺失Validation或任一失败判C，同时保留可评估子集与失败记录。Clean兼容不自动证明干扰输入上的重建有效。

候选wav处理不是原始数据干扰实验：它仅属于同一Clean音频的预处理候选（解码/下混/采样率/单位化），不做pitch/timbre/语义扰动。本轮不训练CMERC、不用Test、不扩其他seed、不Git提交或推送。

## Train阶段协议修订v2

初轮Train CV后、任何本轮Validation特征/目标打开前，增加全秩多输出Ridge(alpha=.01/1/100)，以免仅低秩rank256否定更一般线性关系。原协议和选型结果完整保留，CV每折scaler/拟合和原判定阈值不变。这个候选补充并非最初预注册，明确记录于 `provenance/protocol_revisions.json`；依然仅利用Train选择，最终Validation不回流。
