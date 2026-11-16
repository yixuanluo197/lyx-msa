# C×原μδ交互：五Seed结果

唯一新增结构是C与原μδ的四个交互项；Gaussian/Dropout仍各10次、.05强度。IEMOCAP527条/12Dialogue，MELD1109条/114Dialogue；五Seed完整Validation分别拟合。没有新增CMERC前向。

## 五Seed均值

| dataset | method | ece15_mean | binary_brier_mean | binary_nll_mean |
|---|---|---|---|---|
| IEMOCAP | E0 | 0.136499 | 0.234443 | 0.809438 |
| IEMOCAP | E1 | 0.072514 | 0.220685 | 0.634827 |
| IEMOCAP | M_interaction | 0.068676 | 0.221390 | 0.636603 |
| IEMOCAP | M_joint | 0.069459 | 0.222093 | 0.638626 |
| MELD | E0 | 0.209270 | 0.237788 | 0.668629 |
| MELD | E1 | 0.041635 | 0.193374 | 0.569588 |
| MELD | M_interaction | 0.043502 | 0.193942 | 0.570788 |
| MELD | M_joint | 0.043266 | 0.193714 | 0.570243 |

## 相对E1：原置信度之外的行为增量

| dataset | metric | mean_difference | seed_difference_sample_sd | improved_seeds | ci_low | ci_high |
|---|---|---|---|---|---|---|
| IEMOCAP | ece15 | -0.003838 | 0.004715 | 3 | -0.007569 | 0.005123 |
| IEMOCAP | binary_brier | 0.000705 | 0.001039 | 1 | 0.000143 | 0.001326 |
| IEMOCAP | binary_nll | 0.001776 | 0.002581 | 1 | 0.000436 | 0.003222 |
| MELD | ece15 | 0.001868 | 0.001553 | 0 | -0.002834 | 0.003929 |
| MELD | binary_brier | 0.000568 | 0.000783 | 0 | 0.000227 | 0.000963 |
| MELD | binary_nll | 0.001200 | 0.001719 | 0 | 0.000388 | 0.002127 |

## 相对旧M_joint：交互结构的修正效果

| dataset | metric | mean_difference | seed_difference_sample_sd | improved_seeds | ci_low | ci_high |
|---|---|---|---|---|---|---|
| IEMOCAP | ece15 | -0.000783 | 0.012640 | 2 | -0.010704 | 0.010247 |
| IEMOCAP | binary_brier | -0.000703 | 0.000460 | 4 | -0.001636 | -0.000094 |
| IEMOCAP | binary_nll | -0.002023 | 0.001728 | 4 | -0.005266 | 0.000029 |
| MELD | ece15 | 0.000236 | 0.001861 | 2 | -0.001862 | 0.003263 |
| MELD | binary_brier | 0.000227 | 0.000276 | 0 | 0.000047 | 0.000414 |
| MELD | binary_nll | 0.000545 | 0.000695 | 0 | 0.000127 | 0.000981 |

IEMOCAP相对旧模块三项均值均改善，Brier/NLL各4/5 Seed改善；但相对E1仅ECE均值改善，proper losses仍增加。MELD交互模型相对E1三项均值和全部Seed都变差。不能将“比旧M_joint好”写成“已经超过E1”。

## 统计与完整性

差值为candidate−reference，负值更好。所有40条每Seed指标及均值/样本标准差见results/per_seed_metrics.csv与aggregate_metrics.csv，四组完整OOF及内外参数均保留。2000次Dialogue成组配对抽样在五Seed间共享，再平均各Seed指标差；不把Seed当独立数据集。百分位95%区间为探索性、未作多重比较调整，不含重新拟合不确定性。

IEMOCAP交互−E1的ECE区间跨零，Brier/NLL区间位于恶化方向。MELD ECE区间跨零，Brier/NLL区间位于恶化方向。IEMOCAP交互−旧M_joint的Brier区间位于改善方向、NLL上限略高于0；这些都是探索性结果。

当前Validation已参与XP04 checkpoint选择与多轮方法设计；当前Dialogue OOF只隔离本轮拟合和选参，不构成独立确认。预测类别/Accuracy/F1不变；未用Test、未训练CMERC、未使用温度缩放、未改原始特征或原始数据。
