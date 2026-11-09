# Classwise Temperature Analysis

温度由原始预测类别决定；下列 NLL 差值均为 E3−E1，负值代表层次类别温度优于 Global TS。

## IEMOCAP

| Predicted class | Mean E3-E1 NLL | Improved seeds | Range |
|---|---|---|---|
| angry | -0.005024 | 5/5 | [-0.018023, -0.000006] |
| excited | +0.024492 | 2/5 | [-0.001491, +0.101732] |
| frustrated | -0.000862 | 4/5 | [-0.002515, +0.000585] |
| happy | +0.017905 | 1/5 | [-0.001149, +0.038287] |
| neutral | -0.011177 | 4/5 | [-0.054962, +0.004136] |
| sad | +0.088877 | 0/5 | [+0.017496, +0.294236] |

angry 在 5/5 Seed 有小幅 NLL 改善，neutral 与 frustrated 分别为 4/5；但 sad 在 0/5 改善且平均明显退化，happy、excited 也不稳定。整体样本加权结果仍使 E3 在 5/5 Seed 输给 E1，不能据此部署 angry 专用温度。

## MELD

| Predicted class | Mean E3-E1 NLL | Improved seeds | Range |
|---|---|---|---|
| anger | -0.000908 | 2/5 | [-0.005151, +0.001513] |
| disgust | +0.003802 | 2/5 | [-0.001887, +0.015814] |
| fear | +0.011566 | 0/5 | [+0.000027, +0.042530] |
| joy | +0.000550 | 1/5 | [-0.000156, +0.001335] |
| neutral | -0.000699 | 2/5 | [-0.003415, +0.000723] |
| sadness | +0.001776 | 0/5 | [+0.000063, +0.004100] |
| surprise | -0.000318 | 2/5 | [-0.004745, +0.003409] |

没有 MELD 类别达到跨五 Seed 一致改善。fear 与 disgust 支持量最少，E3 已强制向 Global T 收缩；fear、sadness 的类别 NLL 在 0/5 Seed 改善。MELD 的类别温度五 Seed均值波动小于 IEMOCAP，但这主要反映稳定的全局欠置信方向，并未形成额外类别收益。

## 稳定性

| Predicted class | Mean train support | Mean E3 T | SD of seed means | Mean |log offset| |
|---|---|---|---|---|
| happy | 79.6 | 1.533 | 0.393 | 0.0905 |
| sad | 78.5 | 1.856 | 0.436 | 0.1169 |
| neutral | 66.9 | 1.476 | 0.465 | 0.1497 |
| angry | 121.0 | 1.611 | 0.418 | 0.0338 |
| excited | 67.1 | 1.761 | 0.454 | 0.0603 |
| frustrated | 70.0 | 1.607 | 0.379 | 0.0322 |

| Predicted class | Mean train support | Mean E3 T | SD of seed means | Mean |log offset| |
|---|---|---|---|---|
| neutral | 518.0 | 0.557 | 0.045 | 0.0256 |
| surprise | 163.2 | 0.594 | 0.071 | 0.0356 |
| fear | 24.6 | 0.560 | 0.041 | 0.0285 |
| sadness | 66.0 | 0.573 | 0.058 | 0.0186 |
| joy | 157.8 | 0.566 | 0.036 | 0.0175 |
| disgust | 18.0 | 0.592 | 0.063 | 0.0319 |
| anger | 151.7 | 0.571 | 0.031 | 0.0328 |

IEMOCAP 类别温度的 Seed 间波动仍大；MELD 较稳定但各类温度接近同一个全局温度。支持量与偏移图见 `figures/support_vs_temperature_offset.*`。
