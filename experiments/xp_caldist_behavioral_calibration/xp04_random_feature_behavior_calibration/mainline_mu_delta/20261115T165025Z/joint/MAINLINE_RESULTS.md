# 原始μ/δ主线五 Seed结果

主方法只使用C与Gaussian μ/δ、Dropout μ/δ四维。μ为带扰动最大置信度权重的预测翻转均值；δ为原最大置信度与扰动最大置信度均值的绝对差。没有D/A、JS、熵、分类间隔或类别条件。

## 五Seed均值

| dataset | method | ece15_mean | binary_brier_mean | binary_nll_mean |
|---|---|---|---|---|
| IEMOCAP | E0 | 0.136499 | 0.234443 | 0.809438 |
| IEMOCAP | E1 | 0.072514 | 0.220685 | 0.634827 |
| IEMOCAP | M_joint | 0.069459 | 0.222093 | 0.638626 |
| MELD | E0 | 0.209270 | 0.237788 | 0.668629 |
| MELD | E1 | 0.041635 | 0.193374 | 0.569588 |
| MELD | M_joint | 0.043266 | 0.193714 | 0.570243 |

## 相对E1的行为增量

| dataset | metric | mean_difference | improved_seeds | ci_low | ci_high |
|---|---|---|---|---|---|
| IEMOCAP | ece15 | -0.003055 | 2 | -0.012115 | 0.010315 |
| IEMOCAP | binary_brier | 0.001408 | 0 | 0.000481 | 0.002846 |
| IEMOCAP | binary_nll | 0.003799 | 0 | 0.001150 | 0.008015 |
| MELD | ece15 | 0.001632 | 1 | -0.003093 | 0.002867 |
| MELD | binary_brier | 0.000340 | 0 | 0.000020 | 0.000679 |
| MELD | binary_nll | 0.000655 | 0 | -0.000036 | 0.001387 |

差值为主模型减E1，负值更好。标准差见aggregate_metrics.csv；完整消融均报告，不按最优结果替换M_joint。Bootstrap共2000次，以Dialogue为配对单位、五Seed共享抽样，再平均五Seed指标差；区间不含校准器重新拟合不确定性。所有bootstrap区间均为探索性区间，未做多重比较调整。

## 评分与校准

R=a+bC+Z(B)w，R_behavior=Z(B)w，q=sigmoid(R)。基础C与行为权重联合学习。λ由外训练集内GroupKFold5的合并留出Brier选定。仅行为列标准化；off精确退回同训练子集的E1。E1优化精度与旧实现的差异单独报告，不能记为行为收益。

## 限制

同一Validation已用于XP04 checkpoint选择及多轮方法设计，外层Dialogue OOF只隔离校准器拟合和选参，不是独立确认。所有Seed共享数据，不能当五份独立数据。仅调整置信度，Accuracy/Weighted-F1/Macro-F1不变；没有GPU推理、CMERC训练、重新扰动、Test标签或原始数据修改。
