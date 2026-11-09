# Calibration Pilot Report

所有结果来自独立 Calibration Set。Grouped OOF 仅是 Calibration 内部诊断，不是最终 Evaluation。四种方法均使用原始预测类别选择温度，所有温度为正，所有 argmax 与 E0 完全一致。

## IEMOCAP

Global T = 1.458735；Hierarchical λ = 100。

| 方法 | OOF NLL | OOF Brier | OOF ECE15 | Accuracy |
|---|---:|---:|---:|---:|
| E0 | 0.841706 | 0.448169 | 0.080338 | 69.2878% |
| E1 | 0.812070 | 0.443491 | 0.065726 | 69.2878% |
| E2 | 0.812099 | 0.443500 | 0.065325 | 69.2878% |
| E3 | 0.841374 | 0.453065 | 0.076331 | 69.2878% |

E2 选择 λ=100，类别温度强烈收缩到 Global T。相对 E1，OOF NLL/Brier 基本持平且略差，ECE15 仅有很小变化；E3 的 OOF NLL/Brier 明显更差，显示不受约束类别温度过拟合。

## MELD

Global T = 0.535567；Hierarchical λ = 0.1。

| 方法 | OOF NLL | OOF Brier | OOF ECE15 | Accuracy |
|---|---:|---:|---:|---:|
| E0 | 1.043578 | 0.476301 | 0.245820 | 72.6174% |
| E1 | 0.871688 | 0.394634 | 0.036164 | 72.6174% |
| E2 | 0.851258 | 0.390243 | 0.022895 | 72.6174% |
| E3 | 0.851710 | 0.390779 | 0.025741 | 72.6174% |

E2 在 Calibration grouped OOF 中相对 E1 同时降低 NLL、Brier 和 ECE15；E3 与 E2 接近但略差。该观察仍需封存 Evaluation 才能确认迁移性。
