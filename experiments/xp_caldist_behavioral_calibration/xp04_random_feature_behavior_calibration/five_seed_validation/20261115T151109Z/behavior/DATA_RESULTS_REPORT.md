# XP04 五 Seed 固定行为模块结果

仅使用真正 Validation：IEMOCAP 每 Seed 527 条 / 12 Dialogue；MELD 每 Seed 1109 条 / 114 Dialogue。原模型和预测类别保持不变。

主方法为预先登记的 E1_DA；Gaussian 与 Dropout 的 D/A 分开共 4 维。IEMOCAP 外层 LODO12、MELD GroupKFold5；行为分支正则强度在每个外训练集的内 GroupKFold5 中按 Brier 选择 off / 1 / 0.1 / 0.01。

## 五 Seed 指标均值

| dataset | method | ece15_mean | binary_brier_mean | binary_nll_mean |
|---|---|---|---|---|
| IEMOCAP | XP04 | 0.136499 | 0.234443 | 0.809438 |
| IEMOCAP | E1 | 0.072514 | 0.220685 | 0.634827 |
| IEMOCAP | L0 | 0.057706 | 0.225191 | 0.653854 |
| IEMOCAP | E1_DA | 0.074699 | 0.221335 | 0.636696 |
| MELD | XP04 | 0.209270 | 0.237788 | 0.668629 |
| MELD | E1 | 0.041635 | 0.193374 | 0.569588 |
| MELD | L0 | 0.027478 | 0.191467 | 0.564187 |
| MELD | E1_DA | 0.040324 | 0.193304 | 0.568657 |

完整均值和样本标准差见 aggregate_metrics.csv；逐 Seed 全部结果见 per_seed_metrics.csv。ECE 为固定15区间，Brier/NLL 为 Top-label 二元正确性口径。

## 主方法与 XP04 / E1 的共享 Dialogue bootstrap

| dataset | reference | metric | mean_difference | improved_seeds | ci_low | ci_high |
|---|---|---|---|---|---|---|
| IEMOCAP | XP04 | ece15 | -0.061800 | 5 | -0.112646 | -0.002586 |
| IEMOCAP | XP04 | binary_brier | -0.013108 | 5 | -0.042959 | 0.007134 |
| IEMOCAP | XP04 | binary_nll | -0.172741 | 5 | -0.454886 | 0.008804 |
| IEMOCAP | E1 | ece15 | 0.002185 | 1 | -0.006676 | 0.011816 |
| IEMOCAP | E1 | binary_brier | 0.000651 | 0 | -0.000579 | 0.002324 |
| IEMOCAP | E1 | binary_nll | 0.001869 | 0 | -0.001199 | 0.006344 |
| MELD | XP04 | ece15 | -0.168946 | 5 | -0.184513 | -0.129796 |
| MELD | XP04 | binary_brier | -0.044484 | 5 | -0.055632 | -0.033249 |
| MELD | XP04 | binary_nll | -0.099971 | 5 | -0.124821 | -0.074759 |
| MELD | E1 | ece15 | -0.001311 | 3 | -0.010169 | 0.005723 |
| MELD | E1 | binary_brier | -0.000070 | 2 | -0.001148 | 0.000988 |
| MELD | E1 | binary_nll | -0.000930 | 3 | -0.003353 | 0.001440 |

差值为候选减对照，负值更好。同一次抽到的 Dialogue 重数用于全部五 Seed，再对五个 Seed 的指标差取均值；没有把五 Seed 当作独立数据。2000次区间不包含重新拟合不确定性。

## 行为分支关闭比例

| dataset | seed | method | outer_folds | off_folds | active_folds |
|---|---|---|---|---|---|
| IEMOCAP | 1475 | E1_DA | 12 | 12 | 0 |
| IEMOCAP | 67137 | E1_DA | 12 | 9 | 3 |
| IEMOCAP | 2024 | E1_DA | 12 | 8 | 4 |
| IEMOCAP | 3407 | E1_DA | 12 | 1 | 11 |
| IEMOCAP | 42 | E1_DA | 12 | 1 | 11 |
| MELD | 1475 | E1_DA | 5 | 0 | 5 |
| MELD | 67137 | E1_DA | 5 | 1 | 4 |
| MELD | 2024 | E1_DA | 5 | 1 | 4 |
| MELD | 3407 | E1_DA | 5 | 0 | 5 |
| MELD | 42 | E1_DA | 5 | 0 | 5 |

Seed1475复用前轮冻结特征并逐条复现7方法 OOF；其余Seed从新完成的 joint2type×10repeat 概率计算。共享同一扰动随机种子1475用于配对。视觉缺失样本保留，其联合扰动仍作用于可用文本/音频。

这是固定配置的后续确认；Seed1475曾参与设计，四个新Seed为新增验证。只有一个固定Validation划分，不能把本结果视作独立测试集验证。外层 Dialogue OOF 仅隔离校准器拟合和调参；XP04 best.pt 本身曾按 Validation 选择 checkpoint，因此不等价于完全未参与模型选择的新 Dialogue 泛化确认。五 Seed 共享 Dialogue，跨 Seed 稳定性也不是新数据证据。
