# E3 Dialogue-level OOF 结果

Seed 1475 真正 Validation。IEMOCAP 527 条/12 Dialogue，MELD 1109 条/114 Dialogue。每条话语一条训练记录；行为特征来自原 Pilot 已完成的扰动汇总。所有数字均为 Top-label 二元正确性评价，较低 ECE15/Brier/NLL 更好。E3 是 CaliDist-inspired 双向扩展，并非原版 CaliDist。

## 主模型与对照

| Dataset | Method | ECE15 | Brier | NLL | Correctness AUC | Mean C |
|---|---|---|---|---|---|---|
| IEMOCAP | E0 | 0.10497 | 0.22230 | 0.71951 | 0.64519 | 0.72788 |
| IEMOCAP | E1 | 0.08977 | 0.22121 | 0.63706 | 0.59770 | 0.65852 |
| IEMOCAP | L0 | 0.05836 | 0.22884 | 0.67476 | 0.57808 | 0.65992 |
| IEMOCAP | E3-A | 0.06492 | 0.23105 | 0.68209 | 0.57524 | 0.66019 |
| IEMOCAP | E3-B | 0.07487 | 0.23069 | 0.68219 | 0.58467 | 0.65805 |
| IEMOCAP | E3-C | 0.08145 | 0.23966 | 0.73758 | 0.54771 | 0.65359 |
| IEMOCAP | E3-D | 0.13924 | 0.26103 | 0.81885 | 0.51554 | 0.63436 |
| MELD | E0 | 0.20136 | 0.23342 | 0.65945 | 0.72989 | 0.46840 |
| MELD | E1 | 0.04323 | 0.19276 | 0.56858 | 0.72662 | 0.66903 |
| MELD | L0 | 0.02560 | 0.19076 | 0.56380 | 0.72662 | 0.66897 |
| MELD | E3-A | 0.02283 | 0.19154 | 0.56554 | 0.72517 | 0.66958 |
| MELD | E3-B | 0.02692 | 0.19124 | 0.56461 | 0.72604 | 0.66970 |
| MELD | E3-C | 0.03554 | 0.19348 | 0.58295 | 0.72294 | 0.67116 |
| MELD | E3-D | 0.05104 | 0.19680 | 0.59307 | 0.71604 | 0.67128 |

IEMOCAP 的 Accuracy/Weighted-F1/Macro-F1 均为 0.66034/0.65021/0.64489；MELD 均为 0.66907/0.66104/0.56381。所有方法不改变预测类别，因此这些值逐方法相同。

## E3 相对 E1 的 Dialogue 配对 bootstrap

Δ=E3−E1，负数代表改善。2,000 次按 Dialogue 有放回抽样；95% percentile 区间。IEMOCAP 只有 12 个 Dialogue，区间应谨慎解释。

| Dataset | Method | Metric | Δ | 95% CI |
|---|---|---|---|---|
| IEMOCAP | E3-A | ece15 | -0.02485 | [-0.04822, 0.03506] |
| IEMOCAP | E3-A | binary_brier | 0.00984 | [-0.00063, 0.02514] |
| IEMOCAP | E3-A | binary_nll | 0.04503 | [-0.00176, 0.13029] |
| IEMOCAP | E3-B | ece15 | -0.01490 | [-0.04332, 0.04231] |
| IEMOCAP | E3-B | binary_brier | 0.00948 | [-0.00074, 0.02485] |
| IEMOCAP | E3-B | binary_nll | 0.04513 | [-0.00193, 0.13293] |
| IEMOCAP | E3-C | ece15 | -0.00832 | [-0.05957, 0.05491] |
| IEMOCAP | E3-C | binary_brier | 0.01845 | [0.00558, 0.03609] |
| IEMOCAP | E3-C | binary_nll | 0.10052 | [0.02695, 0.20906] |
| IEMOCAP | E3-D | ece15 | 0.04946 | [-0.04032, 0.13922] |
| IEMOCAP | E3-D | binary_brier | 0.03982 | [0.01534, 0.07311] |
| IEMOCAP | E3-D | binary_nll | 0.18179 | [0.06577, 0.34899] |
| MELD | E3-A | ece15 | -0.02040 | [-0.03177, 0.00902] |
| MELD | E3-A | binary_brier | -0.00122 | [-0.00385, 0.00149] |
| MELD | E3-A | binary_nll | -0.00304 | [-0.00983, 0.00409] |
| MELD | E3-B | ece15 | -0.01631 | [-0.03306, 0.01479] |
| MELD | E3-B | binary_brier | -0.00153 | [-0.00469, 0.00183] |
| MELD | E3-B | binary_nll | -0.00397 | [-0.01200, 0.00459] |
| MELD | E3-C | ece15 | -0.00769 | [-0.02909, 0.02384] |
| MELD | E3-C | binary_brier | 0.00072 | [-0.00451, 0.00617] |
| MELD | E3-C | binary_nll | 0.01437 | [-0.00888, 0.04436] |
| MELD | E3-D | ece15 | 0.00781 | [-0.02219, 0.03944] |
| MELD | E3-D | binary_brier | 0.00404 | [-0.00212, 0.01027] |
| MELD | E3-D | binary_nll | 0.02449 | [-0.00026, 0.05520] |

模型参数、训练折与留出 Dialogue ID、每折标准化器参数在 `fold_parameters.json`；逐话语结果在 `e3_oof_predictions.csv`。Bootstrap 的完整结果在 `bootstrap_results.csv`。
