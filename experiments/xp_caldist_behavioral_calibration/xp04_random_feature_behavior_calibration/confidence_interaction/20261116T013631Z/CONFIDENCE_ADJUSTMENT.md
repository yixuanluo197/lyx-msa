# 双向调整与行为分支诊断

以下为每Seed统计量的五Seed均值；raised/lowered相对原始C，不等同于相对E1。

| dataset | method | mean_confidence | correct_mean_confidence | wrong_mean_confidence | raised_fraction | lowered_fraction |
|---|---|---|---|---|---|---|
| IEMOCAP | E0 | 0.749630 | 0.773284 | 0.702641 | 0.000000 | 0.000000 |
| IEMOCAP | E1 | 0.666883 | 0.671300 | 0.658449 | 0.310816 | 0.689184 |
| IEMOCAP | M_interaction | 0.667661 | 0.671795 | 0.659787 | 0.310436 | 0.689564 |
| IEMOCAP | M_joint | 0.667744 | 0.671439 | 0.660727 | 0.312713 | 0.687287 |
| MELD | E0 | 0.461794 | 0.493221 | 0.397919 | 0.000000 | 0.000000 |
| MELD | E1 | 0.670528 | 0.703206 | 0.604197 | 1.000000 | 0.000000 |
| MELD | M_interaction | 0.670770 | 0.703164 | 0.605017 | 0.999459 | 0.000541 |
| MELD | M_joint | 0.670727 | 0.703268 | 0.604673 | 0.999459 | 0.000541 |

## 内层选择结果

| dataset | method | selected_penalty | folds |
|---|---|---|---|
| IEMOCAP | M_joint | off | 38 |
| IEMOCAP | M_interaction | off | 40 |
| IEMOCAP | M_joint | 1.0 | 15 |
| IEMOCAP | M_interaction | 1.0 | 18 |
| IEMOCAP | M_joint | 0.1 | 7 |
| IEMOCAP | M_interaction | 0.1 | 2 |
| MELD | M_joint | 0.01 | 1 |
| MELD | M_interaction | 0.01 | 1 |
| MELD | M_joint | 1.0 | 6 |
| MELD | M_interaction | 1.0 | 5 |
| MELD | M_interaction | off | 18 |
| MELD | M_interaction | 0.1 | 1 |
| MELD | M_joint | off | 18 |

交互模型IEMOCAP40/60外折、MELD18/25外折选择off，精确退回E1。只看外训练Dialogue的内部选择并不能保证外留出Dialogue不恶化。大多数折没有选择激活分支，提示目前的行为增量较弱，不直接证明所有潜在干扰都无价值。

固定原C区间[0,.4)、[.4,.6)、[.6,.8)、[.8,1]，分别按正确/错误分组的320个诊断单元见results/confidence_bins.csv；空单元数量0、均值留空。它们仅作描述，不参与拟合、选参或筛选子群。

分数拆分见OOF中的base_score/additive_score/interaction_score/behavior_score/reliability_score。样本真实标签只用于训练折拟合与事后评价，不参与推理时分数构造。没有预设敏感即错误；错误样本平均置信度增加也不能被解释为该样本获益。
