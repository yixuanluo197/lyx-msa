# 实验范围与指标约束

状态：**审计未通过；本文件记录约束，不是已执行协议或完整预注册。** 提取器、输入可用性和 clean parity 未解决，不进入下游阶段。

只允许 IEMOCAP seed1475 和 MELD seed67137，真正 Validation 各 120 条，同数据集文本和音频共用样本。IEMOCAP 音频为重点、文本为对照；MELD 文本为重点、音频为对照。保留完整对话上下文，每次仅目标话语、单一模态受干扰。只读原始数据、特征、模型；新特征写隔离目录。没有视觉、Test 标签使用、训练、评分拟合、参数搜索、新随机特征实验、多 Seed 扩展或完整 Validation 干扰。

## 指标定义已核对

取原始概率 `p`、原预测 `yhat=argmax(p)`、原置信度 `C=max(p)`；同一登记条件的有效版本为 `p'_1 ... p'_K`，`C'_k=max(p'_k)`：

- `mu = mean_k(1[argmax(p'_k) != yhat] * C'_k)`。
- `delta = abs(C - mean_k(C'_k))`。
- `flip_rate = mean_k(1[argmax(p'_k) != yhat])`，独立报告，不能用它替代 μ。
- `mean_confidence_change = mean_k(C'_k - C)`；`mean_absolute_confidence_change = mean_k(abs(C'_k-C))`；同时保存每个版本、最大绝对变化和原始预测类别概率 `p'_k[yhat]` 变化。
- `JS(p,p') = 0.5*sum_c(p_c*ln(p_c/m_c)) + 0.5*sum_c(p'_c*ln(p'_c/m_c))`，`m=(p+p')/2`，自然对数，零项按极限处理。先算每版本 JS，再按登记版本求均值，不先平均概率向量。

正式定义来自 `mainline_mu_delta/20261115T165025Z/scripts/build_inputs.py:34–40`，C′ 含义另见 `confidence_interaction/20261116T013631Z/METHOD.md:7`。δ 与 `mean(abs(C'_k-C))` 不同。本轮未计算任何新 μ、δ 或 JS。

## 后续执行的必要约束

概率记录必须含每版本完整向量、预测、最大置信度、原类别与真实类别概率、身份字段及输入/参数哈希。版本是确定干预条件时，不称独立 Monte Carlo 重复。质量失败与标签保持不确定版本单独保存；按固定规则排除主分析，同时报告分母和原因，不能按预测结果筛除。

描述正确/错误两组及 Correct-to-Wrong、Wrong-to-Correct，并报告高置信错误与原置信度条件分析。应使用同一批 Dialogue-level 配对 bootstrap draw 计算组差及条件间差，空子组为 NA；不能把版本或同对话多话语当独立样本。匹配/分层所用置信度规则、bootstrap seed/次数、版本聚合和质量阈值需在执行前冻结；本轮尚未建立可执行的完整统计预注册。小规模分层样本整体均值不能当作全数据集指标。

先复用历史随机特征预测中同身份样本；若历史条件不是同 checkpoint/同 ID/同聚合口径，作为背景而非配对优越性证据。不得为比较补跑 Gaussian/Dropout。该 Validation 已用于 checkpoint 选择与多轮机制设计，后续 Pilot 仍属探索性诊断，不能宣称独立确认或校准收益。
