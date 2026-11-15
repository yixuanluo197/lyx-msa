# 行为分支消融与启用情况

E1为raw-C Logistic基础映射，L0为标准化logit-C基础映射。MUDELTA使用两扰动各自的原始μ/δ；DA使用两扰动各自的固定原预测类别方向D与RMS幅度A。两个行为版本都为4维。没有混用每次扰动为训练样本。

| Dataset | Method | Penalty | Selected folds |
|---|---|---|---|
| IEMOCAP | E1_DA | off | 12 |
| IEMOCAP | E1_MUDELTA | 0.1 | 2 |
| IEMOCAP | E1_MUDELTA | off | 10 |
| IEMOCAP | L0_DA | off | 12 |
| IEMOCAP | L0_MUDELTA | 0.1 | 2 |
| IEMOCAP | L0_MUDELTA | off | 10 |
| MELD | E1_DA | 0.1 | 1 |
| MELD | E1_DA | 1.0 | 4 |
| MELD | E1_MUDELTA | 0.01 | 1 |
| MELD | E1_MUDELTA | 0.1 | 2 |
| MELD | E1_MUDELTA | 1.0 | 2 |
| MELD | L0_DA | 0.1 | 1 |
| MELD | L0_DA | 1.0 | 1 |
| MELD | L0_DA | off | 3 |
| MELD | L0_MUDELTA | 0.01 | 1 |
| MELD | L0_MUDELTA | 0.1 | 1 |
| MELD | L0_MUDELTA | 1.0 | 2 |
| MELD | L0_MUDELTA | off | 1 |

**IEMOCAP主DA分支：12/12折关闭，E1_DA与E1完全一致，L0_DA与L0完全一致。**原μ/δ有2折启用，但Brier/NLL比各自基础更差。

**MELD主DA分支：5/5折启用，其中4折λ=1、1折λ=0.1。**E1_DA相对E1的ECE变化−0.000980、Brier变化+0.000100、NLL变化−0.000078，变化很小且不是三项一致收益。L0_DA有3/5折关闭，相对L0的ECE更低，但Brier/NLL更高。原μ/δ对照也未稳定超过基础映射。

内层选择off是允许的模型选择结果，不是代码未接入。原始预测类别、输入D/A、优化参数和每折候选分数均已保存；外层留出从未参与λ选择。

`behavior_within_confidence_bins.csv`提供相近原始C范围内按行为中位数分组的描述性正确率；其分组只用于事后诊断，不能当作OOF增量证据或新的模型选择规则。
