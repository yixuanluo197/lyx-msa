# 全部OOF结果

ECE15为0–1单位；0.04=4%。Brier、NLL为Top-label二元正确性口径，越低越好。Accuracy/F1、AUC及置信度上/下调记录于calibration_metrics.csv；分类结果固定不变。

| dataset | condition | method | ece15 | binary_brier | binary_nll |
|---|---|---|---|---|---|
| IEMOCAP | K10_block0 | E0 | 0.104967 | 0.222303 | 0.719513 |
| IEMOCAP | K10_block0 | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | K10_block0 | M_gaussian | 0.070949 | 0.220315 | 0.634945 |
| IEMOCAP | K10_block0 | M_gaussian_mu | 0.089821 | 0.221226 | 0.637132 |
| IEMOCAP | K10_block0 | M_gaussian_delta | 0.075627 | 0.220327 | 0.634938 |
| MELD | K10_block0 | E0 | 0.201363 | 0.233420 | 0.659452 |
| MELD | K10_block0 | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | K10_block0 | M_gaussian | 0.039572 | 0.192479 | 0.567729 |
| MELD | K10_block0 | M_gaussian_mu | 0.044144 | 0.192071 | 0.566768 |
| MELD | K10_block0 | M_gaussian_delta | 0.041578 | 0.193074 | 0.569175 |
| IEMOCAP | K10_block1 | E0 | 0.104967 | 0.222303 | 0.719513 |
| IEMOCAP | K10_block1 | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | K10_block1 | M_gaussian | 0.068211 | 0.221608 | 0.638023 |
| IEMOCAP | K10_block1 | M_gaussian_mu | 0.089821 | 0.221226 | 0.637132 |
| IEMOCAP | K10_block1 | M_gaussian_delta | 0.068331 | 0.221752 | 0.638340 |
| MELD | K10_block1 | E0 | 0.201363 | 0.233420 | 0.659452 |
| MELD | K10_block1 | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | K10_block1 | M_gaussian | 0.044262 | 0.192599 | 0.568210 |
| MELD | K10_block1 | M_gaussian_mu | 0.044043 | 0.192201 | 0.567117 |
| MELD | K10_block1 | M_gaussian_delta | 0.043230 | 0.192764 | 0.568583 |
| IEMOCAP | K10_block2 | E0 | 0.104967 | 0.222303 | 0.719513 |
| IEMOCAP | K10_block2 | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | K10_block2 | M_gaussian | 0.087496 | 0.221291 | 0.637272 |
| IEMOCAP | K10_block2 | M_gaussian_mu | 0.089821 | 0.221226 | 0.637132 |
| IEMOCAP | K10_block2 | M_gaussian_delta | 0.085298 | 0.221438 | 0.637608 |
| MELD | K10_block2 | E0 | 0.201363 | 0.233420 | 0.659452 |
| MELD | K10_block2 | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | K10_block2 | M_gaussian | 0.042241 | 0.192634 | 0.568326 |
| MELD | K10_block2 | M_gaussian_mu | 0.043929 | 0.192036 | 0.566731 |
| MELD | K10_block2 | M_gaussian_delta | 0.042111 | 0.192897 | 0.568890 |
| IEMOCAP | K30_pooled | E0 | 0.104967 | 0.222303 | 0.719513 |
| IEMOCAP | K30_pooled | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | K30_pooled | M_gaussian | 0.081070 | 0.222261 | 0.639413 |
| IEMOCAP | K30_pooled | M_gaussian_mu | 0.089821 | 0.221226 | 0.637132 |
| IEMOCAP | K30_pooled | M_gaussian_delta | 0.078778 | 0.222410 | 0.639749 |
| MELD | K30_pooled | E0 | 0.201363 | 0.233420 | 0.659452 |
| MELD | K30_pooled | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | K30_pooled | M_gaussian | 0.045950 | 0.192202 | 0.567321 |
| MELD | K30_pooled | M_gaussian_mu | 0.043998 | 0.192042 | 0.566669 |
| MELD | K30_pooled | M_gaussian_delta | 0.043230 | 0.192764 | 0.568583 |

## K30与E1、历史K10的点差

负值表示改善。

| dataset | metric | K30_vs_E1 | K30_vs_historical_K10 |
|---|---|---|---|
| IEMOCAP | ece15 | -0.008703 | 0.010121 |
| IEMOCAP | binary_brier | 0.001051 | 0.001946 |
| IEMOCAP | binary_nll | 0.002349 | 0.004468 |
| MELD | ece15 | 0.002720 | 0.006379 |
| MELD | binary_brier | -0.000562 | -0.000277 |
| MELD | binary_nll | -0.001263 | -0.000409 |

## 配对Dialogue Bootstrap

| dataset | candidate_condition | candidate | reference_condition | reference | metric | difference | ci_low | ci_high |
|---|---|---|---|---|---|---|---|---|
| IEMOCAP | K30_pooled | M_gaussian | K30_pooled | E1 | ece15 | -0.008703 | -0.030728 | 0.000312 |
| IEMOCAP | K30_pooled | M_gaussian | K30_pooled | E1 | binary_brier | 0.001051 | 0.000000 | 0.002679 |
| IEMOCAP | K30_pooled | M_gaussian | K30_pooled | E1 | binary_nll | 0.002349 | 0.000000 | 0.005991 |
| IEMOCAP | K30_pooled | M_gaussian_mu | K30_pooled | E1 | ece15 | 0.000047 | -0.000178 | 0.002414 |
| IEMOCAP | K30_pooled | M_gaussian_mu | K30_pooled | E1 | binary_brier | 0.000016 | -0.000089 | 0.000127 |
| IEMOCAP | K30_pooled | M_gaussian_mu | K30_pooled | E1 | binary_nll | 0.000068 | -0.000185 | 0.000394 |
| IEMOCAP | K30_pooled | M_gaussian_delta | K30_pooled | E1 | ece15 | -0.010996 | -0.031873 | 0.000255 |
| IEMOCAP | K30_pooled | M_gaussian_delta | K30_pooled | E1 | binary_brier | 0.001200 | 0.000000 | 0.003061 |
| IEMOCAP | K30_pooled | M_gaussian_delta | K30_pooled | E1 | binary_nll | 0.002685 | 0.000000 | 0.006848 |
| IEMOCAP | K30_pooled | M_gaussian | K10_block0 | M_gaussian | ece15 | 0.010121 | -0.020279 | 0.032384 |
| IEMOCAP | K30_pooled | M_gaussian | K10_block0 | M_gaussian | binary_brier | 0.001946 | -0.001155 | 0.004591 |
| IEMOCAP | K30_pooled | M_gaussian | K10_block0 | M_gaussian | binary_nll | 0.004468 | -0.002185 | 0.010293 |
| IEMOCAP | K30_pooled | M_gaussian | K10_block1 | M_gaussian | ece15 | 0.012860 | -0.013656 | 0.030172 |
| IEMOCAP | K30_pooled | M_gaussian | K10_block1 | M_gaussian | binary_brier | 0.000652 | -0.000984 | 0.002512 |
| IEMOCAP | K30_pooled | M_gaussian | K10_block1 | M_gaussian | binary_nll | 0.001390 | -0.002162 | 0.005410 |
| IEMOCAP | K30_pooled | M_gaussian | K10_block2 | M_gaussian | ece15 | -0.006426 | -0.026088 | 0.000210 |
| IEMOCAP | K30_pooled | M_gaussian | K10_block2 | M_gaussian | binary_brier | 0.000970 | 0.000000 | 0.002474 |
| IEMOCAP | K30_pooled | M_gaussian | K10_block2 | M_gaussian | binary_nll | 0.002141 | 0.000000 | 0.005459 |
| MELD | K30_pooled | M_gaussian | K30_pooled | E1 | ece15 | 0.002720 | -0.013306 | 0.012538 |
| MELD | K30_pooled | M_gaussian | K30_pooled | E1 | binary_brier | -0.000562 | -0.001534 | 0.000258 |
| MELD | K30_pooled | M_gaussian | K30_pooled | E1 | binary_nll | -0.001263 | -0.003490 | 0.000656 |
| MELD | K30_pooled | M_gaussian_mu | K30_pooled | E1 | ece15 | 0.000768 | -0.006556 | 0.004053 |
| MELD | K30_pooled | M_gaussian_mu | K30_pooled | E1 | binary_brier | -0.000722 | -0.001664 | -0.000003 |
| MELD | K30_pooled | M_gaussian_mu | K30_pooled | E1 | binary_nll | -0.001914 | -0.004346 | -0.000059 |
| MELD | K30_pooled | M_gaussian_delta | K30_pooled | E1 | ece15 | 0.000000 | 0.000000 | 0.000000 |
| MELD | K30_pooled | M_gaussian_delta | K30_pooled | E1 | binary_brier | 0.000000 | 0.000000 | 0.000000 |
| MELD | K30_pooled | M_gaussian_delta | K30_pooled | E1 | binary_nll | 0.000000 | 0.000000 | 0.000000 |
| MELD | K30_pooled | M_gaussian | K10_block0 | M_gaussian | ece15 | 0.006379 | -0.014674 | 0.015989 |
| MELD | K30_pooled | M_gaussian | K10_block0 | M_gaussian | binary_brier | -0.000277 | -0.001411 | 0.000771 |
| MELD | K30_pooled | M_gaussian | K10_block0 | M_gaussian | binary_nll | -0.000409 | -0.003013 | 0.001987 |
| MELD | K30_pooled | M_gaussian | K10_block1 | M_gaussian | ece15 | 0.001688 | -0.012279 | 0.013361 |
| MELD | K30_pooled | M_gaussian | K10_block1 | M_gaussian | binary_brier | -0.000397 | -0.001129 | 0.000307 |
| MELD | K30_pooled | M_gaussian | K10_block1 | M_gaussian | binary_nll | -0.000889 | -0.002644 | 0.000728 |
| MELD | K30_pooled | M_gaussian | K10_block2 | M_gaussian | ece15 | 0.003709 | -0.010212 | 0.015003 |
| MELD | K30_pooled | M_gaussian | K10_block2 | M_gaussian | binary_brier | -0.000433 | -0.001166 | 0.000255 |
| MELD | K30_pooled | M_gaussian | K10_block2 | M_gaussian | binary_nll | -0.001005 | -0.002711 | 0.000572 |

## 正确/错误样本置信度

| dataset | seed | condition | modality | repeat_count | method | correct | utterances | original_mean_confidence | calibrated_mean_confidence | mean_change | raised_fraction | lowered_fraction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IEMOCAP | 1475 | K30_pooled | text | 30 | E1 | 0 | 179 | 0.666240 | 0.642721 | -0.023519 | 0.486034 | 0.513966 |
| IEMOCAP | 1475 | K30_pooled | text | 30 | E1 | 1 | 348 | 0.759579 | 0.666647 | -0.092932 | 0.209770 | 0.790230 |
| IEMOCAP | 1475 | K30_pooled | text | 30 | M_gaussian | 0 | 179 | 0.666240 | 0.643941 | -0.022299 | 0.486034 | 0.513966 |
| IEMOCAP | 1475 | K30_pooled | text | 30 | M_gaussian | 1 | 348 | 0.759579 | 0.665537 | -0.094042 | 0.206897 | 0.793103 |
| MELD | 1475 | K30_pooled | audio | 30 | E1 | 0 | 367 | 0.402171 | 0.599705 | 0.197534 | 1.000000 | 0.000000 |
| MELD | 1475 | K30_pooled | audio | 30 | E1 | 1 | 742 | 0.501153 | 0.703326 | 0.202173 | 1.000000 | 0.000000 |
| MELD | 1475 | K30_pooled | audio | 30 | M_gaussian | 0 | 367 | 0.402171 | 0.598371 | 0.196200 | 0.991826 | 0.008174 |
| MELD | 1475 | K30_pooled | audio | 30 | M_gaussian | 1 | 742 | 0.501153 | 0.704383 | 0.203230 | 1.000000 | 0.000000 |

## 固定OOF逐Dialogue省略影响

不重新拟合；ECE非加性，不将某Dialogue的局部差直接当作总体贡献。

| dataset | seed | candidate_condition | candidate | reference_condition | reference | scope | metric | full_difference | dialogues | dialogues_improving | omission_difference_min | omission_difference_max | omissions_reversing_direction | refit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IEMOCAP | 1475 | K30_pooled | M_gaussian | K30_pooled | E1 | same_condition | ece15 | -0.008703 | 12 | 1 | -0.017978 | 0.000000 | 0 | False |
| IEMOCAP | 1475 | K30_pooled | M_gaussian | K30_pooled | E1 | same_condition | binary_brier | 0.001051 | 12 | 0 | 0.000000 | 0.001168 | 0 | False |
| IEMOCAP | 1475 | K30_pooled | M_gaussian | K30_pooled | E1 | same_condition | binary_nll | 0.002349 | 12 | 0 | 0.000000 | 0.002612 | 0 | False |
| IEMOCAP | 1475 | K30_pooled | M_gaussian | K10_block0 | M_gaussian | primary_repeat | ece15 | 0.010121 | 12 | 3 | -0.006533 | 0.014182 | 1 | False |
| IEMOCAP | 1475 | K30_pooled | M_gaussian | K10_block0 | M_gaussian | primary_repeat | binary_brier | 0.001946 | 12 | 5 | 0.000618 | 0.002807 | 0 | False |
| IEMOCAP | 1475 | K30_pooled | M_gaussian | K10_block0 | M_gaussian | primary_repeat | binary_nll | 0.004468 | 12 | 4 | 0.001622 | 0.006311 | 0 | False |
| IEMOCAP | 1475 | K30_pooled | M_gaussian | K10_block1 | M_gaussian | descriptive_repeat | ece15 | 0.012860 | 12 | 3 | 0.003271 | 0.016753 | 0 | False |
| IEMOCAP | 1475 | K30_pooled | M_gaussian | K10_block1 | M_gaussian | descriptive_repeat | binary_brier | 0.000652 | 12 | 5 | -0.000465 | 0.000926 | 1 | False |
| IEMOCAP | 1475 | K30_pooled | M_gaussian | K10_block1 | M_gaussian | descriptive_repeat | binary_nll | 0.001390 | 12 | 5 | -0.001039 | 0.001970 | 1 | False |
| IEMOCAP | 1475 | K30_pooled | M_gaussian | K10_block2 | M_gaussian | descriptive_repeat | ece15 | -0.006426 | 12 | 1 | -0.015420 | 0.000000 | 0 | False |
| IEMOCAP | 1475 | K30_pooled | M_gaussian | K10_block2 | M_gaussian | descriptive_repeat | binary_brier | 0.000970 | 12 | 0 | 0.000000 | 0.001078 | 0 | False |
| IEMOCAP | 1475 | K30_pooled | M_gaussian | K10_block2 | M_gaussian | descriptive_repeat | binary_nll | 0.002141 | 12 | 0 | 0.000000 | 0.002380 | 0 | False |
| MELD | 1475 | K30_pooled | M_gaussian | K30_pooled | E1 | same_condition | ece15 | 0.002720 | 114 | 52 | 0.001547 | 0.004165 | 0 | False |
| MELD | 1475 | K30_pooled | M_gaussian | K30_pooled | E1 | same_condition | binary_brier | -0.000562 | 114 | 63 | -0.000674 | -0.000258 | 0 | False |
| MELD | 1475 | K30_pooled | M_gaussian | K30_pooled | E1 | same_condition | binary_nll | -0.001263 | 114 | 62 | -0.001502 | -0.000543 | 0 | False |
| MELD | 1475 | K30_pooled | M_gaussian | K10_block0 | M_gaussian | primary_repeat | ece15 | 0.006379 | 114 | 52 | 0.003868 | 0.009267 | 0 | False |
| MELD | 1475 | K30_pooled | M_gaussian | K10_block0 | M_gaussian | primary_repeat | binary_brier | -0.000277 | 114 | 59 | -0.000417 | -0.000074 | 0 | False |
| MELD | 1475 | K30_pooled | M_gaussian | K10_block0 | M_gaussian | primary_repeat | binary_nll | -0.000409 | 114 | 55 | -0.000731 | 0.000105 | 2 | False |
| MELD | 1475 | K30_pooled | M_gaussian | K10_block1 | M_gaussian | descriptive_repeat | ece15 | 0.001688 | 114 | 57 | 0.000444 | 0.003706 | 0 | False |
| MELD | 1475 | K30_pooled | M_gaussian | K10_block1 | M_gaussian | descriptive_repeat | binary_brier | -0.000397 | 114 | 66 | -0.000499 | -0.000250 | 0 | False |
| MELD | 1475 | K30_pooled | M_gaussian | K10_block1 | M_gaussian | descriptive_repeat | binary_nll | -0.000889 | 114 | 66 | -0.001117 | -0.000488 | 0 | False |
| MELD | 1475 | K30_pooled | M_gaussian | K10_block2 | M_gaussian | descriptive_repeat | ece15 | 0.003709 | 114 | 56 | 0.002533 | 0.005725 | 0 | False |
| MELD | 1475 | K30_pooled | M_gaussian | K10_block2 | M_gaussian | descriptive_repeat | binary_brier | -0.000433 | 114 | 67 | -0.000558 | -0.000266 | 0 | False |
| MELD | 1475 | K30_pooled | M_gaussian | K10_block2 | M_gaussian | descriptive_repeat | binary_nll | -0.001005 | 114 | 65 | -0.001273 | -0.000636 | 0 | False |

本轮只有Seed1475、同一批已探索的真正Validation。K30由三个K10块合并而来，条件相互嵌套，不能视为独立重复。仅有一组30次，15+15分半不是独立30次复现实验。Bootstrap复用2000次Dialogue抽样，不重新拟合，未涵盖校准器拟合、模型Seed或随机扰动的不确定性；多重比较未校正。Validation也用于原checkpoint选择和候选探索。本轮不使用Test、不重跑CMERC、不生成新噪声、不更改强度、不训练E3新结构。
