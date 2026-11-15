# XP04 随机扰动行为的重复测量可靠性

本轮只读取已经核验的五Seed163600条joint扰动记录，没有模型推理或新扰动。8180条原话语各形成一条特征记录。

对10次扰动使用预登记的三种5/5切分，全部报告。μ为flip×扰动最大置信度的均值，δ为原置信度与扰动最大置信度均值之差的绝对值；D/A取固定原预测类别的概率。JS为辅助扩展，不属原CaliDist核心。

| Dataset | Corruption | Measure | Mean Spearman | Min | Max | Estimable /15 |
|---|---|---|---:|---:|---:|---:|
| IEMOCAP | dropout | A | 0.8453 | 0.7269 | 0.9151 | 15/15 |
| IEMOCAP | dropout | D | 0.5577 | 0.3987 | 0.6686 | 15/15 |
| IEMOCAP | dropout | JS | 0.7825 | 0.6682 | 0.8842 | 15/15 |
| IEMOCAP | dropout | delta | 0.6358 | 0.5010 | 0.7723 | 15/15 |
| IEMOCAP | dropout | mu | 0.7435 | 0.6276 | 0.9077 | 15/15 |
| IEMOCAP | gaussian | A | 0.8374 | 0.7142 | 0.9313 | 15/15 |
| IEMOCAP | gaussian | D | 0.0136 | -0.0576 | 0.0994 | 15/15 |
| IEMOCAP | gaussian | JS | 0.7769 | 0.6485 | 0.9091 | 15/15 |
| IEMOCAP | gaussian | delta | 0.5215 | 0.2752 | 0.7297 | 15/15 |
| IEMOCAP | gaussian | mu | 0.7227 | 0.4063 | 0.8785 | 14/15 |
| MELD | dropout | A | 0.5955 | 0.5048 | 0.6818 | 15/15 |
| MELD | dropout | D | 0.7089 | 0.6050 | 0.8335 | 15/15 |
| MELD | dropout | JS | 0.7750 | 0.7182 | 0.8242 | 15/15 |
| MELD | dropout | delta | 0.5807 | 0.4624 | 0.7233 | 15/15 |
| MELD | dropout | mu | 0.7053 | 0.5671 | 0.8079 | 15/15 |
| MELD | gaussian | A | 0.3687 | 0.2770 | 0.5197 | 15/15 |
| MELD | gaussian | D | 0.0529 | 0.0157 | 0.1049 | 15/15 |
| MELD | gaussian | JS | 0.6563 | 0.6080 | 0.7508 | 15/15 |
| MELD | gaussian | delta | 0.0717 | 0.0205 | 0.1316 | 15/15 |
| MELD | gaussian | mu | 0.6407 | 0.3348 | 0.8809 | 15/15 |

这里的15项是5个模型Seed×3种切分的描述性汇总，彼此不独立，不是15份数据。常量或全零向量的相关性标记为不可估计，CSV留空；模型特征矩阵没有NaN/Inf。

## Monte Carlo 噪声

D的MC标准误为10次固定类概率的样本标准差/√10。response_summary.csv同时给出D跨话语方差、平均MC方差及max(0,var(D)−mean(SE²))/var(D)估计比例。这是描述性测量误差分解，受异方差与数值误差影响，不能据此判断某条预测错误。|D|>1.96SE比例也不是经过多重校正的逐话语显著检验。

完整半样本结果见split_reliability.csv及split_responses.csv；原始响应与置信度、margin、entropy的相关见response_summary.csv。高重复性只代表敏感性可测，不证明它在静态概率之外有可靠性信息；低重复性说明10次估计噪声可能掩盖信号，不证明增加重复必然改善校准。

本报告先于后续校准结果生成，不据诊断删特征、挑切分或重新设置扰动。后续固定强静态对照与联合拟合检验增量。所有数据仍来自参与过模型/方法选择的Validation，仅作探索性诊断。
