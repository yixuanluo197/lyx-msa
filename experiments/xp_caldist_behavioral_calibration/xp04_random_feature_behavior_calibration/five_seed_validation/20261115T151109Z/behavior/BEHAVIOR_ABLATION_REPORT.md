# 固定消融与分支使用

| dataset | method | ece15_mean | binary_brier_mean | binary_nll_mean |
|---|---|---|---|---|
| IEMOCAP | XP04 | 0.136499 | 0.234443 | 0.809438 |
| IEMOCAP | E1 | 0.072514 | 0.220685 | 0.634827 |
| IEMOCAP | L0 | 0.057706 | 0.225191 | 0.653854 |
| IEMOCAP | E1_MUDELTA | 0.073084 | 0.222312 | 0.639431 |
| IEMOCAP | E1_DA | 0.074699 | 0.221335 | 0.636696 |
| IEMOCAP | L0_MUDELTA | 0.059930 | 0.226436 | 0.657485 |
| IEMOCAP | L0_DA | 0.062558 | 0.225154 | 0.654204 |
| MELD | XP04 | 0.209270 | 0.237788 | 0.668629 |
| MELD | E1 | 0.041635 | 0.193374 | 0.569588 |
| MELD | L0 | 0.027478 | 0.191467 | 0.564187 |
| MELD | E1_MUDELTA | 0.042086 | 0.193740 | 0.570252 |
| MELD | E1_DA | 0.040324 | 0.193304 | 0.568657 |
| MELD | L0_MUDELTA | 0.027524 | 0.191881 | 0.565047 |
| MELD | L0_DA | 0.026495 | 0.191859 | 0.564754 |

μ/δ 为原 CaliDist 定义，D/A 为本研究适配；没有平均两种干扰、添加JS、类别或重新选择扰动强度。四个行为变体共享完全相同的内外Dialogue划分。

| dataset | seed | method | outer_folds | off_folds | active_folds |
|---|---|---|---|---|---|
| IEMOCAP | 1475 | E1_MUDELTA | 12 | 10 | 2 |
| IEMOCAP | 1475 | E1_DA | 12 | 12 | 0 |
| IEMOCAP | 1475 | L0_MUDELTA | 12 | 10 | 2 |
| IEMOCAP | 1475 | L0_DA | 12 | 12 | 0 |
| IEMOCAP | 67137 | E1_MUDELTA | 12 | 8 | 4 |
| IEMOCAP | 67137 | E1_DA | 12 | 9 | 3 |
| IEMOCAP | 67137 | L0_MUDELTA | 12 | 7 | 5 |
| IEMOCAP | 67137 | L0_DA | 12 | 6 | 6 |
| IEMOCAP | 2024 | E1_MUDELTA | 12 | 12 | 0 |
| IEMOCAP | 2024 | E1_DA | 12 | 8 | 4 |
| IEMOCAP | 2024 | L0_MUDELTA | 12 | 12 | 0 |
| IEMOCAP | 2024 | L0_DA | 12 | 7 | 5 |
| IEMOCAP | 3407 | E1_MUDELTA | 12 | 1 | 11 |
| IEMOCAP | 3407 | E1_DA | 12 | 1 | 11 |
| IEMOCAP | 3407 | L0_MUDELTA | 12 | 0 | 12 |
| IEMOCAP | 3407 | L0_DA | 12 | 0 | 12 |
| IEMOCAP | 42 | E1_MUDELTA | 12 | 6 | 6 |
| IEMOCAP | 42 | E1_DA | 12 | 1 | 11 |
| IEMOCAP | 42 | L0_MUDELTA | 12 | 5 | 7 |
| IEMOCAP | 42 | L0_DA | 12 | 0 | 12 |
| MELD | 1475 | E1_MUDELTA | 5 | 0 | 5 |
| MELD | 1475 | E1_DA | 5 | 0 | 5 |
| MELD | 1475 | L0_MUDELTA | 5 | 1 | 4 |
| MELD | 1475 | L0_DA | 5 | 3 | 2 |
| MELD | 67137 | E1_MUDELTA | 5 | 3 | 2 |
| MELD | 67137 | E1_DA | 5 | 1 | 4 |
| MELD | 67137 | L0_MUDELTA | 5 | 4 | 1 |
| MELD | 67137 | L0_DA | 5 | 4 | 1 |
| MELD | 2024 | E1_MUDELTA | 5 | 3 | 2 |
| MELD | 2024 | E1_DA | 5 | 1 | 4 |
| MELD | 2024 | L0_MUDELTA | 5 | 5 | 0 |
| MELD | 2024 | L0_DA | 5 | 3 | 2 |
| MELD | 3407 | E1_MUDELTA | 5 | 5 | 0 |
| MELD | 3407 | E1_DA | 5 | 0 | 5 |
| MELD | 3407 | L0_MUDELTA | 5 | 5 | 0 |
| MELD | 3407 | L0_DA | 5 | 1 | 4 |
| MELD | 42 | E1_MUDELTA | 5 | 5 | 0 |
| MELD | 42 | E1_DA | 5 | 0 | 5 |
| MELD | 42 | L0_MUDELTA | 5 | 3 | 2 |
| MELD | 42 | L0_DA | 5 | 1 | 4 |
