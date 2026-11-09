# 模态敏感性的正确性预测增量价值

## 协议

目标为原始预测是否正确。每个数据集、每个模型 Seed 独立执行 Leave-One-Dialogue-Out；标准化只在训练折拟合；Logistic Regression 固定 L2、`C=1`，不调参。基础模型分别使用最大概率，以及更严格的“最大概率 + 预测熵 + Top-1/Top-2 margin”。扩展模型加入 `rho=0.5` 的 Prediction Flip、绝对置信度变化和 JS Divergence。`rho=1` 仅作强干预诊断，Donor 特征单独评估。

置信区间使用 2000 次 dialogue-level paired bootstrap。bootstrap 重采样固定 OOF 预测，不在每次重采样中重新拟合 Logistic Regression，因此区间不包含模型重拟合不确定性。IEMOCAP 只有 12 个对话，尤其需要谨慎。

## 基础模型

五 Seed 的 OOF 均值 ± 样本标准差：

| 数据集 | 模型 | AUROC Success | AUPRC Success | Binary Brier | Binary NLL |
|---|---|---:|---:|---:|---:|
| IEMOCAP | Confidence-only | 0.5588 ± 0.0411 | 0.6806 ± 0.0133 | 0.2212 ± 0.0026 | 0.6372 ± 0.0056 |
| IEMOCAP | Strict probability | 0.5630 ± 0.0394 | 0.6848 ± 0.0159 | 0.2226 ± 0.0019 | 0.6416 ± 0.0055 |
| MELD | Confidence-only | 0.7221 ± 0.0139 | 0.8315 ± 0.0150 | 0.1915 ± 0.0023 | 0.5642 ± 0.0066 |
| MELD | Strict probability | 0.7212 ± 0.0133 | 0.8304 ± 0.0153 | 0.1918 ± 0.0021 | 0.5647 ± 0.0060 |

## 中度训练均值敏感性的增量 AUROC

下表为扩展模型减基础模型的五 Seed AUROC 差；括号为改善 Seed 数/5。

| 数据集 | 扩展信号 | 相对 Confidence-only | 相对 Strict probability |
|---|---|---:|---:|
| IEMOCAP | Text | -0.0269 (0/5) | -0.0240 (0/5) |
| IEMOCAP | Audio | -0.0106 (2/5) | -0.0117 (1/5) |
| IEMOCAP | Visual | -0.0043 (1/5) | -0.0002 (2/5) |
| IEMOCAP | T+A+V features | -0.0298 (0/5) | -0.0262 (0/5) |
| MELD | Text | -0.0008 (1/5) | -0.0006 (2/5) |
| MELD | Audio | -0.0016 (1/5) | -0.0014 (1/5) |
| MELD | Visual | -0.0024 (0/5) | -0.0023 (0/5) |
| MELD | T+A+V features | -0.0049 (0/5) | -0.0046 (0/5) |

没有任何主扩展模型在五个 Seed 上一致改善 AUROC。对 AUROC 的 80 个主比较 Seed×模型结果中，没有任何“改善”置信区间排除零；反而 IEMOCAP Text、Audio/TAV 的部分 Seed 及 MELD Visual/TAV 的部分 Seed 显著变差。Brier 和 NLL 也没有给出相反的稳定正证据：例如 strict+TAV 在 IEMOCAP 的平均 Brier/NLL 差为 +0.00834/+0.02386，在 MELD 为 +0.00204/+0.00491，均表示恶化，且两个数据集均为 0/5 Seed 改善。

## 正确与错误样本的行为差异

`rho=0.5` 下，错误样本相对正确样本的平均 Flip Rate 差：

| 数据集 | Text | Audio | Visual |
|---|---:|---:|---:|
| IEMOCAP | +0.0096 | +0.0594 | +0.0017 |
| MELD | +0.1406 | +0.0196 | +0.0066 |

描述性分组中，错误样本常更易翻转；但这并未在控制原始概率特征后转化为稳定 OOF 增益。原因之一是敏感性和原始置信度相关，且“正确但敏感”和“稳定但错误”同时大量存在。`notable_cases.csv` 保留了所有中度干预下高置信稳定错误与正确但翻转样本，便于个案检查。

## 与轻度行为鲁棒性 Pilot 的关联

该关联只能在两个历史 Anchor 上计算。轻度 Gaussian/Dropout/Attenuation 指标与 `rho=0.5` 对应指标的 Spearman 相关随模态和扰动类型明显变化；部分条件因轻度 Flip 全为常量而相关系数未定义。即使轻度条件完全稳定，中度干预仍可翻转：

- IEMOCAP Audio：轻度稳定样本中约 7.9%–9.2% 在中度收缩下翻转。
- IEMOCAP Text：约 2.7%–4.0%。
- MELD Text：约 11.6%–12.4%。
- MELD Audio：约 1.1%–2.0%。

这说明轻度局部鲁棒性与较强证据依赖测量的是不同干预区间，具有描述性互补性；当前数据不支持它们对预测错误具有稳定的联合增量价值。所有历史联合相关仅作 Anchor 描述，未用话语级朴素 p 值作推断结论。

## 结论

模态敏感性可以稳定描述 CMERC 的模态依赖，却不能在本协议下稳定超越原始置信度识别错误。最接近零的主信号是 Visual，但其绝对效应极小、方向不一致，也没有支持进一步拟合统一 Reliability Score 的证据。
