# μ/δ预登记消融

主模型C+μ+δ与C、C+μ、C+δ采用相同外层Dialogue划分及训练折选择协议。这里只比较预登记配置，不按最佳外层结果替换主模型。

| dataset | method | ece15_mean | ece15_sample_std | ece15_difference_vs_E1 | ece15_improving_seeds | binary_brier_mean | binary_brier_sample_std | binary_brier_difference_vs_E1 | binary_brier_improving_seeds | binary_nll_mean | binary_nll_sample_std | binary_nll_difference_vs_E1 | binary_nll_improving_seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IEMOCAP | E0 | 0.136499 | 0.022434 | 0.063985 | 0 | 0.234443 | 0.010082 | 0.013758 | 0 | 0.809438 | 0.126276 | 0.174610 | 0 |
| IEMOCAP | E1 | 0.072514 | 0.013909 | 0.000000 | 0 | 0.220685 | 0.002810 | 0.000000 | 0 | 0.634827 | 0.005702 | 0.000000 | 0 |
| IEMOCAP | M_gaussian | 0.068548 | 0.009881 | -0.003966 | 3 | 0.221276 | 0.001993 | 0.000591 | 1 | 0.636778 | 0.003161 | 0.001951 | 1 |
| IEMOCAP | M_gaussian_mu | 0.072516 | 0.013688 | 0.000002 | 1 | 0.221023 | 0.002778 | 0.000338 | 0 | 0.637047 | 0.006763 | 0.002220 | 0 |
| IEMOCAP | M_gaussian_delta | 0.069327 | 0.010547 | -0.003187 | 3 | 0.220987 | 0.002070 | 0.000303 | 1 | 0.635784 | 0.003493 | 0.000956 | 1 |
| MELD | E0 | 0.209270 | 0.023357 | 0.167635 | 0 | 0.237788 | 0.010166 | 0.044414 | 0 | 0.668629 | 0.021476 | 0.099041 | 0 |
| MELD | E1 | 0.041635 | 0.005157 | 0.000000 | 0 | 0.193374 | 0.001905 | 0.000000 | 0 | 0.569588 | 0.005167 | 0.000000 | 0 |
| MELD | M_gaussian | 0.040858 | 0.005093 | -0.000777 | 3 | 0.193363 | 0.001945 | -0.000011 | 1 | 0.569516 | 0.005249 | -0.000071 | 1 |
| MELD | M_gaussian_mu | 0.042049 | 0.005361 | 0.000414 | 1 | 0.193432 | 0.001991 | 0.000058 | 1 | 0.570007 | 0.005199 | 0.000419 | 1 |
| MELD | M_gaussian_delta | 0.040895 | 0.005352 | -0.000740 | 3 | 0.193517 | 0.001950 | 0.000143 | 0 | 0.569890 | 0.005313 | 0.000302 | 0 |

## μ+δ相对单指标的配对区间

| dataset | candidate | reference | metric | mean_difference | ci_low | ci_high |
|---|---|---|---|---|---|---|
| IEMOCAP | M_gaussian | M_gaussian_mu | ece15 | -0.003968 | -0.007480 | 0.002617 |
| IEMOCAP | M_gaussian | M_gaussian_mu | binary_brier | 0.000253 | -0.000364 | 0.001227 |
| IEMOCAP | M_gaussian | M_gaussian_mu | binary_nll | -0.000269 | -0.003774 | 0.003195 |
| IEMOCAP | M_gaussian | M_gaussian_delta | ece15 | -0.000779 | -0.001097 | 0.001612 |
| IEMOCAP | M_gaussian | M_gaussian_delta | binary_brier | 0.000289 | -0.000037 | 0.001057 |
| IEMOCAP | M_gaussian | M_gaussian_delta | binary_nll | 0.000994 | -0.000064 | 0.003549 |
| MELD | M_gaussian | M_gaussian_mu | ece15 | -0.001191 | -0.003358 | 0.003057 |
| MELD | M_gaussian | M_gaussian_mu | binary_brier | -0.000069 | -0.000473 | 0.000288 |
| MELD | M_gaussian | M_gaussian_mu | binary_nll | -0.000491 | -0.002143 | 0.000686 |
| MELD | M_gaussian | M_gaussian_delta | ece15 | -0.000037 | -0.002663 | 0.002627 |
| MELD | M_gaussian | M_gaussian_delta | binary_brier | -0.000154 | -0.000381 | 0.000040 |
| MELD | M_gaussian | M_gaussian_delta | binary_nll | -0.000374 | -0.000885 | 0.000072 |

## 行为项开关与选择频次

| dataset | seed | block | method | selected_penalty | outer_folds |
|---|---|---|---|---|---|
| IEMOCAP | 1475 | 0 | M_gaussian | 0.01 | 9 |
| IEMOCAP | 1475 | 0 | M_gaussian_mu | 0.01 | 5 |
| IEMOCAP | 1475 | 0 | M_gaussian_delta | 0.01 | 9 |
| IEMOCAP | 1475 | 0 | M_gaussian_mu | off | 7 |
| IEMOCAP | 1475 | 0 | M_gaussian | 0.1 | 3 |
| IEMOCAP | 1475 | 0 | M_gaussian_delta | 0.1 | 3 |
| IEMOCAP | 67137 | 0 | M_gaussian | 0.01 | 1 |
| IEMOCAP | 67137 | 0 | M_gaussian_mu | off | 11 |
| IEMOCAP | 67137 | 0 | M_gaussian_delta | 0.01 | 1 |
| IEMOCAP | 67137 | 0 | M_gaussian | off | 9 |
| IEMOCAP | 67137 | 0 | M_gaussian_mu | 0.01 | 1 |
| IEMOCAP | 67137 | 0 | M_gaussian_delta | off | 9 |
| IEMOCAP | 67137 | 0 | M_gaussian | 1.0 | 2 |
| IEMOCAP | 67137 | 0 | M_gaussian_delta | 1.0 | 2 |
| IEMOCAP | 2024 | 0 | M_gaussian | off | 9 |
| IEMOCAP | 2024 | 0 | M_gaussian_mu | off | 7 |
| IEMOCAP | 2024 | 0 | M_gaussian_delta | off | 11 |
| IEMOCAP | 2024 | 0 | M_gaussian | 1.0 | 2 |
| IEMOCAP | 2024 | 0 | M_gaussian_mu | 1.0 | 4 |
| IEMOCAP | 2024 | 0 | M_gaussian_delta | 1.0 | 1 |
| IEMOCAP | 2024 | 0 | M_gaussian | 0.1 | 1 |
| IEMOCAP | 2024 | 0 | M_gaussian_mu | 0.01 | 1 |
| IEMOCAP | 3407 | 0 | M_gaussian | off | 12 |
| IEMOCAP | 3407 | 0 | M_gaussian_mu | off | 12 |
| IEMOCAP | 3407 | 0 | M_gaussian_delta | off | 12 |
| IEMOCAP | 42 | 0 | M_gaussian | off | 11 |
| IEMOCAP | 42 | 0 | M_gaussian_mu | off | 12 |
| IEMOCAP | 42 | 0 | M_gaussian_delta | off | 11 |
| IEMOCAP | 42 | 0 | M_gaussian | 1.0 | 1 |
| IEMOCAP | 42 | 0 | M_gaussian_delta | 1.0 | 1 |
| MELD | 1475 | 0 | M_gaussian | 1.0 | 1 |
| MELD | 1475 | 0 | M_gaussian_mu | 0.01 | 5 |
| MELD | 1475 | 0 | M_gaussian_delta | off | 3 |
| MELD | 1475 | 0 | M_gaussian | 0.01 | 4 |
| MELD | 1475 | 0 | M_gaussian_delta | 0.1 | 2 |
| MELD | 67137 | 0 | M_gaussian | off | 3 |
| MELD | 67137 | 0 | M_gaussian_mu | off | 3 |
| MELD | 67137 | 0 | M_gaussian_delta | 1.0 | 3 |
| MELD | 67137 | 0 | M_gaussian | 1.0 | 2 |
| MELD | 67137 | 0 | M_gaussian_mu | 1.0 | 1 |
| MELD | 67137 | 0 | M_gaussian_delta | off | 2 |
| MELD | 67137 | 0 | M_gaussian_mu | 0.01 | 1 |
| MELD | 2024 | 0 | M_gaussian | off | 5 |
| MELD | 2024 | 0 | M_gaussian_mu | off | 5 |
| MELD | 2024 | 0 | M_gaussian_delta | 1.0 | 1 |
| MELD | 2024 | 0 | M_gaussian_delta | off | 4 |
| MELD | 3407 | 0 | M_gaussian | 1.0 | 4 |
| MELD | 3407 | 0 | M_gaussian_mu | 0.01 | 3 |
| MELD | 3407 | 0 | M_gaussian_delta | off | 3 |
| MELD | 3407 | 0 | M_gaussian_mu | off | 2 |
| MELD | 3407 | 0 | M_gaussian_delta | 0.1 | 1 |
| MELD | 3407 | 0 | M_gaussian | off | 1 |
| MELD | 3407 | 0 | M_gaussian_delta | 1.0 | 1 |
| MELD | 42 | 0 | M_gaussian | off | 2 |
| MELD | 42 | 0 | M_gaussian_mu | off | 1 |
| MELD | 42 | 0 | M_gaussian_delta | off | 5 |
| MELD | 42 | 0 | M_gaussian | 0.1 | 1 |
| MELD | 42 | 0 | M_gaussian_mu | 0.1 | 3 |
| MELD | 42 | 0 | M_gaussian | 1.0 | 2 |
| MELD | 42 | 0 | M_gaussian_mu | 0.01 | 1 |
| IEMOCAP | 1475 | 1 | M_gaussian | off | 4 |
| IEMOCAP | 1475 | 1 | M_gaussian_mu | 0.01 | 5 |
| IEMOCAP | 1475 | 1 | M_gaussian_delta | off | 4 |
| IEMOCAP | 1475 | 1 | M_gaussian | 0.1 | 1 |
| IEMOCAP | 1475 | 1 | M_gaussian_mu | off | 7 |
| IEMOCAP | 1475 | 1 | M_gaussian_delta | 0.1 | 1 |
| IEMOCAP | 1475 | 1 | M_gaussian | 1.0 | 1 |
| IEMOCAP | 1475 | 1 | M_gaussian_delta | 1.0 | 1 |
| IEMOCAP | 1475 | 1 | M_gaussian | 0.01 | 6 |
| IEMOCAP | 1475 | 1 | M_gaussian_delta | 0.01 | 6 |
| MELD | 1475 | 1 | M_gaussian | 1.0 | 4 |
| MELD | 1475 | 1 | M_gaussian_mu | 0.01 | 5 |
| MELD | 1475 | 1 | M_gaussian_delta | off | 5 |
| MELD | 1475 | 1 | M_gaussian | 0.1 | 1 |
| IEMOCAP | 1475 | 2 | M_gaussian | off | 11 |
| IEMOCAP | 1475 | 2 | M_gaussian_mu | 0.01 | 5 |
| IEMOCAP | 1475 | 2 | M_gaussian_delta | off | 11 |
| IEMOCAP | 1475 | 2 | M_gaussian_mu | off | 7 |
| IEMOCAP | 1475 | 2 | M_gaussian | 0.1 | 1 |
| IEMOCAP | 1475 | 2 | M_gaussian_delta | 0.01 | 1 |
| MELD | 1475 | 2 | M_gaussian | 1.0 | 2 |
| MELD | 1475 | 2 | M_gaussian_mu | 0.01 | 5 |
| MELD | 1475 | 2 | M_gaussian_delta | off | 4 |
| MELD | 1475 | 2 | M_gaussian | 0.01 | 3 |
| MELD | 1475 | 2 | M_gaussian_delta | 0.1 | 1 |

μ和δ的互补性应从主模型相对单指标的收益判断，不能因为两者数值不同就认为互补；off较多说明训练折内未稳定选择行为项。

## 证据范围与限制

- 真正Validation：IEMOCAP 527条/12个Dialogue；MELD 1109条/114个Dialogue。未使用Test。
- 两个候选来自已查看的同一Validation上的历史探索；新增模型Seed和随机块检验稳定性，不构成独立数据确认。
- 五个checkpoint评价同一批话语，不能视为五份独立数据。五Seed均值Bootstrap在每次共享Dialogue抽样后计算各Seed指标差，再取均值。
- Bootstrap为2000次固定OOF预测上的Dialogue配对抽样，不重新拟合；区间不包含校准器重拟合、Seed总体或随机扰动重复的不确定性。多项比较为探索性，未校正。
- Seed1475只有三个独立10次扰动块；不能据此断言任意随机噪声都稳定。三个块分别拟合和评价，没有合并成30次。
- 所有ECE均为0–1原始数值，例如0.04=4%；Brier/NLL为Top-label二元正确性口径。
- 调整原始预测正确的概率，不改变情绪类别。Accuracy、Weighted-F1、Macro-F1保持不变；AUROC只作区分能力诊断，不是额外成功门槛。
