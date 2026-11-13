# E0 versus E1 confidence shift

E1 is the existing dialogue-level OOF logistic confidence mapping. Its held-out sample label was not used in that fold's fit.

## IEMOCAP

E0 accuracy/mean C=0.660/0.728; E1 mean C=0.659; E1 global gap=0.002 (dialogue bootstrap 95% CI [-0.095, 0.082]). ECE15 0.10497 → 0.08977. Correctness AUC 0.645 → 0.598; OOF fold-specific mappings can change cross-fold ranking.
Correct predictions: mean C 0.760 → 0.667, mean shift -0.093, raised 0.210, lowered 0.790.
Wrong predictions: mean C 0.666 → 0.643, mean shift -0.024, raised 0.486, lowered 0.514.
This is an aggregate OOF calibration result: a higher confidence for a wrong prediction is not individually beneficial, even when aggregate Brier/NLL improve. Bins have different membership after calibration.

| Method | Interval | n | Mean C | Accuracy | Gap |
|---|---|---:|---:|---:|---:|
| E0 | [0.3, 0.4) | 14 | 0.373 | 0.429 | 0.055 |
| E0 | [0.4, 0.5) | 55 | 0.455 | 0.436 | -0.018 |
| E0 | [0.5, 0.6) | 87 | 0.549 | 0.529 | -0.020 |
| E0 | [0.6, 0.7) | 96 | 0.653 | 0.646 | -0.007 |
| E0 | [0.7, 0.8) | 71 | 0.748 | 0.789 | 0.041 |
| E0 | [0.8, 0.9) | 65 | 0.854 | 0.692 | -0.161 |
| E0 | [0.9, 1.0] | 139 | 0.966 | 0.784 | -0.182 |
| E1 | [0.4, 0.5) | 15 | 0.480 | 0.800 | 0.320 |
| E1 | [0.5, 0.6) | 136 | 0.561 | 0.485 | -0.076 |
| E1 | [0.6, 0.7) | 187 | 0.648 | 0.706 | 0.058 |
| E1 | [0.7, 0.8) | 158 | 0.738 | 0.791 | 0.053 |
| E1 | [0.8, 0.9) | 31 | 0.830 | 0.419 | -0.411 |

See `figures/e0_e1_confidence_shift.pdf` and `figures/reliability_diagram.pdf`.

IEMOCAP E1 has a 0.8–0.9 bin with 31 items and accuracy 0.419 against mean confidence 0.830; this local mismatch and the lower OOF correctness AUC warrant caution despite aggregate ECE improvement.

## MELD

E0 accuracy/mean C=0.669/0.468; E1 mean C=0.669; E1 global gap=0.000 (dialogue bootstrap 95% CI [-0.029, 0.028]). ECE15 0.20136 → 0.04323. Correctness AUC 0.730 → 0.727; OOF fold-specific mappings can change cross-fold ranking.
Correct predictions: mean C 0.501 → 0.703, mean shift 0.202, raised 1.000, lowered 0.000.
Wrong predictions: mean C 0.402 → 0.600, mean shift 0.198, raised 1.000, lowered 0.000.
This is an aggregate OOF calibration result: a higher confidence for a wrong prediction is not individually beneficial, even when aggregate Brier/NLL improve. Bins have different membership after calibration.

| Method | Interval | n | Mean C | Accuracy | Gap |
|---|---|---:|---:|---:|---:|
| E0 | [0.1, 0.2) | 2 | 0.191 | 0.000 | -0.191 |
| E0 | [0.2, 0.3) | 102 | 0.264 | 0.402 | 0.138 |
| E0 | [0.3, 0.4) | 268 | 0.354 | 0.466 | 0.113 |
| E0 | [0.4, 0.5) | 272 | 0.450 | 0.669 | 0.219 |
| E0 | [0.5, 0.6) | 277 | 0.550 | 0.819 | 0.270 |
| E0 | [0.6, 0.7) | 160 | 0.640 | 0.881 | 0.241 |
| E0 | [0.7, 0.8) | 27 | 0.721 | 0.926 | 0.205 |
| E0 | [0.8, 0.9) | 1 | 0.804 | 1.000 | 0.196 |
| E1 | [0.3, 0.4) | 18 | 0.386 | 0.333 | -0.052 |
| E1 | [0.4, 0.5) | 121 | 0.458 | 0.388 | -0.069 |
| E1 | [0.5, 0.6) | 220 | 0.552 | 0.491 | -0.061 |
| E1 | [0.6, 0.7) | 232 | 0.648 | 0.659 | 0.011 |
| E1 | [0.7, 0.8) | 305 | 0.751 | 0.793 | 0.042 |
| E1 | [0.8, 0.9) | 207 | 0.837 | 0.874 | 0.037 |
| E1 | [0.9, 1.0] | 6 | 0.915 | 0.833 | -0.082 |

See `figures/e0_e1_confidence_shift.pdf` and `figures/reliability_diagram.pdf`.

MELD E1 still overpredicts accuracy in the occupied 0.3–0.6 confidence bins by about 0.05–0.07, while the 0.7–0.9 bins remain mildly underconfident (gap about +0.04). The 0.9–1.0 bin has only six items. Aggregate improvement does not imply uniform bin calibration.
