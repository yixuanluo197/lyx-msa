# Predicted-Class Distribution Audit

校准器按原始预测类别选择温度，因此下表以预测类别为主要分组；真实类别分组仅保存在 `results/predicted_class_support.csv` 作辅助诊断。

## IEMOCAP

| Predicted class | Mean train support | Mean E3 T | SD of seed means | Mean |log offset| |
|---|---|---|---|---|
| happy | 79.6 | 1.533 | 0.393 | 0.0905 |
| sad | 78.5 | 1.856 | 0.436 | 0.1169 |
| neutral | 66.9 | 1.476 | 0.465 | 0.1497 |
| angry | 121.0 | 1.611 | 0.418 | 0.0338 |
| excited | 67.1 | 1.761 | 0.454 | 0.0603 |
| frustrated | 70.0 | 1.607 | 0.379 | 0.0322 |

五 Seed 的完整验证集预测类支持范围为：angry 116–144、excited 56–83、frustrated 58–94、happy 76–98、neutral 66–81、sad 84–87。IEMOCAP 没有验证集预测类别缺失，但每个外层训练折仅有 11 个对话，类别组成容易随留出对话改变。

## MELD

| Predicted class | Mean train support | Mean E3 T | SD of seed means | Mean |log offset| |
|---|---|---|---|---|
| neutral | 518.0 | 0.557 | 0.045 | 0.0256 |
| surprise | 163.2 | 0.594 | 0.071 | 0.0356 |
| fear | 24.6 | 0.560 | 0.041 | 0.0285 |
| sadness | 66.0 | 0.573 | 0.058 | 0.0186 |
| joy | 157.8 | 0.566 | 0.036 | 0.0175 |
| disgust | 18.0 | 0.592 | 0.063 | 0.0319 |
| anger | 151.7 | 0.571 | 0.031 | 0.0328 |

完整验证集支持范围为：neutral 491–556；surprise 151–191；joy 146–175；anger 121–172；sadness 46–79；fear 18–33；disgust 15–22。fear 与 disgust 是明确的小样本类别，不能依赖不受约束类别温度。
