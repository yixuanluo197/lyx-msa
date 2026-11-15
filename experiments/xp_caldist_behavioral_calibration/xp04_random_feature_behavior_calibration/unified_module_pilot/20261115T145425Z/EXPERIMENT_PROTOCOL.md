# XP04 统一行为可靠性校准模块：固定实验协议

本次只执行 Seed 1475 CPU Pilot。主比较为 **XP04+E1_DA 对原始 XP04**；E1 是去掉行为分支的必要消融，L0 是替代基础映射对照。方法在 IEMOCAP/MELD 的结构、候选强度、选择规则相同，分别拟合参数。不基于外层结果选择数据集专用模态或修改模块。

保留原预测类别 k，q_j=p'_j[k]。修订行为量 D=C−mean(q_j)，A=sqrt(mean((q_j−C)^2))。联合模态的 Gaussian 和 Dropout 各产生 D/A，共4维；原 μ/δ 同样按两扰动类型分别产生4维，作为对照。每条原始话语只有一行，所有信息来自既有 Pilot CSV。D/A 是研究适配指标，μ/δ 对齐已核验的 CaliDist 定义，新的双向模块不称为原版 CaliDist。

基础映射 E1 完全沿用 raw-C L2 Logistic Regression，L0 沿用训练折标准化的 logit-C Logistic Regression。每个训练折拟合基础映射后冻结；行为修正模型为 sigmoid(logit(q_base)+z(B)·w)，只拟合4个权重，没有额外截距。优化目标是平均二元交叉熵+λ/2·||w||²。标准化器仅在相应训练 Dialogue 拟合。关闭分支时 w=0，逐行精确退回基础映射，但不能保证内层选择在外层永不恶化。

外层 IEMOCAP LODO12、MELD GroupKFold5；顺序与原 E1 一致。每个外层训练集内部做 GroupKFold5，完整重拟合基础映射、标准化器和修正器，比较 off/λ=1/0.1/0.01 的合并内层 OOF Brier。候选差值≤1e-12时优先off，再优先更强收缩。选定后才在完整外层训练 Dialogue 重拟合，预测外层留出 Dialogue。真实标签仅用于训练子集拟合及相应留出子集的评价。

主报告固定15区间 Top-label ECE、Binary Brier/NLL，同时保留 Accuracy/F1、正确性AUC、正确/错误置信度变化及逐Dialogue结果。Dialogue配对bootstrap2000次只重采样已经生成的外层OOF，不重新拟合整个嵌套过程；区间描述固定预测的Dialogue采样不确定性。该Validation已用于多轮方法设计，本次仍属探索性，不能作为全新独立确认集。

继续判据见preregistration.json。该文件中的“另一loss没有明显恶化”本轮具体固定为Brier增加不超过0.001、NLL增加不超过0.005；这是本轮结果之前定义的工程判断阈值，并非文献通用显著性阈值。点估计通过和bootstrap风险分别报告。实验结束后停止并给出五Seed建议，不自动启动后续实验或Git提交。
