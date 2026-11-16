# Seed1475 随机特征扰动强度响应与校准敏感性

0.05为历史主条件；0.10为预登记新增敏感性条件。完整报告两个强度、四种模态、两种干扰，不根据本次外层OOF结果选择强度。IEMOCAP 527条/12个Dialogue，MELD 1109条/114个Dialogue；只使用完整真实Validation、同一个Seed1475。

μ=mean(1[扰动预测≠原预测]×扰动最大置信度)，δ=|原始最大置信度−mean(扰动最大置信度)|；JS采用自然对数且仅用于响应诊断。置信度变化标准差按10repeat总体标准差ddof=0计算。每条话语每条件汇总成一条记录；没有把10次重复、四种模态或两种强度当成独立话语。

## 行为响应与零值覆盖

| dataset | modality | strength | corruption | population | utterances | mu_zero_fraction | mean_mu | mean_delta | mean_js | mean_flip_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| IEMOCAP | audio | 0.050000 | dropout | all_retained | 527 | 0.952562 | 0.004783 | 0.006170 | 0.000301 | 0.011006 |
| IEMOCAP | audio | 0.050000 | gaussian | all_retained | 527 | 0.988615 | 0.001106 | 0.000679 | 0.000014 | 0.002657 |
| IEMOCAP | audio | 0.100000 | dropout | all_retained | 527 | 0.939279 | 0.010387 | 0.011286 | 0.000713 | 0.023529 |
| IEMOCAP | audio | 0.100000 | gaussian | all_retained | 527 | 0.982922 | 0.001884 | 0.001381 | 0.000060 | 0.004364 |
| IEMOCAP | joint | 0.050000 | dropout | all_retained | 527 | 0.954459 | 0.005912 | 0.006297 | 0.000311 | 0.013472 |
| IEMOCAP | joint | 0.050000 | dropout | valid_visual | 526 | 0.954373 | 0.005924 | 0.006300 | 0.000312 | 0.013498 |
| IEMOCAP | joint | 0.050000 | gaussian | all_retained | 527 | 0.990512 | 0.001101 | 0.000697 | 0.000014 | 0.002657 |
| IEMOCAP | joint | 0.050000 | gaussian | valid_visual | 526 | 0.990494 | 0.001103 | 0.000698 | 0.000014 | 0.002662 |
| IEMOCAP | joint | 0.100000 | dropout | all_retained | 527 | 0.925996 | 0.011819 | 0.011920 | 0.000747 | 0.026376 |
| IEMOCAP | joint | 0.100000 | dropout | valid_visual | 526 | 0.925856 | 0.011841 | 0.011913 | 0.000748 | 0.026426 |
| IEMOCAP | joint | 0.100000 | gaussian | all_retained | 527 | 0.982922 | 0.001845 | 0.001400 | 0.000061 | 0.004364 |
| IEMOCAP | joint | 0.100000 | gaussian | valid_visual | 526 | 0.982890 | 0.001849 | 0.001402 | 0.000061 | 0.004373 |
| IEMOCAP | text | 0.050000 | dropout | all_retained | 527 | 0.984820 | 0.001471 | 0.001439 | 0.000009 | 0.003226 |
| IEMOCAP | text | 0.050000 | gaussian | all_retained | 527 | 0.998102 | 0.000484 | 0.000105 | 0.000000 | 0.001139 |
| IEMOCAP | text | 0.100000 | dropout | all_retained | 527 | 0.977230 | 0.004292 | 0.002910 | 0.000028 | 0.009488 |
| IEMOCAP | text | 0.100000 | gaussian | all_retained | 527 | 0.998102 | 0.000081 | 0.000210 | 0.000001 | 0.000190 |
| IEMOCAP | visual | 0.050000 | dropout | all_retained | 527 | 0.998102 | 0.000403 | 0.000252 | 0.000000 | 0.000949 |
| IEMOCAP | visual | 0.050000 | dropout | valid_visual | 526 | 0.998099 | 0.000404 | 0.000252 | 0.000000 | 0.000951 |
| IEMOCAP | visual | 0.050000 | gaussian | all_retained | 527 | 0.998102 | 0.000322 | 0.000032 | 0.000000 | 0.000759 |
| IEMOCAP | visual | 0.050000 | gaussian | valid_visual | 526 | 0.998099 | 0.000323 | 0.000032 | 0.000000 | 0.000760 |
| IEMOCAP | visual | 0.100000 | dropout | all_retained | 527 | 0.998102 | 0.000161 | 0.000463 | 0.000001 | 0.000380 |
| IEMOCAP | visual | 0.100000 | dropout | valid_visual | 526 | 0.998099 | 0.000162 | 0.000464 | 0.000001 | 0.000380 |
| IEMOCAP | visual | 0.100000 | gaussian | all_retained | 527 | 0.998102 | 0.000161 | 0.000066 | 0.000000 | 0.000380 |
| IEMOCAP | visual | 0.100000 | gaussian | valid_visual | 526 | 0.998099 | 0.000162 | 0.000067 | 0.000000 | 0.000380 |
| MELD | audio | 0.050000 | dropout | all_retained | 1109 | 0.988278 | 0.001193 | 0.001390 | 0.000014 | 0.003968 |
| MELD | audio | 0.050000 | gaussian | all_retained | 1109 | 0.996393 | 0.000299 | 0.000146 | 0.000000 | 0.001082 |
| MELD | audio | 0.100000 | dropout | all_retained | 1109 | 0.981064 | 0.001867 | 0.002605 | 0.000033 | 0.006132 |
| MELD | audio | 0.100000 | gaussian | all_retained | 1109 | 0.994590 | 0.000501 | 0.000292 | 0.000002 | 0.001713 |
| MELD | joint | 0.050000 | dropout | all_retained | 1109 | 0.959423 | 0.003276 | 0.004797 | 0.000088 | 0.011091 |
| MELD | joint | 0.050000 | dropout | valid_visual | 1014 | 0.959566 | 0.003253 | 0.004800 | 0.000089 | 0.011144 |
| MELD | joint | 0.050000 | gaussian | all_retained | 1109 | 0.992786 | 0.000724 | 0.000376 | 0.000003 | 0.002525 |
| MELD | joint | 0.050000 | gaussian | valid_visual | 1014 | 0.993097 | 0.000652 | 0.000378 | 0.000003 | 0.002367 |
| MELD | joint | 0.100000 | dropout | all_retained | 1109 | 0.943192 | 0.005547 | 0.009714 | 0.000257 | 0.018485 |
| MELD | joint | 0.100000 | dropout | valid_visual | 1014 | 0.943787 | 0.005541 | 0.009657 | 0.000258 | 0.018639 |
| MELD | joint | 0.100000 | gaussian | all_retained | 1109 | 0.986474 | 0.000959 | 0.000748 | 0.000011 | 0.003246 |
| MELD | joint | 0.100000 | gaussian | valid_visual | 1014 | 0.986193 | 0.000838 | 0.000751 | 0.000011 | 0.002959 |
| MELD | text | 0.050000 | dropout | all_retained | 1109 | 0.966637 | 0.002673 | 0.004384 | 0.000071 | 0.009107 |
| MELD | text | 0.050000 | gaussian | all_retained | 1109 | 0.992786 | 0.000650 | 0.000334 | 0.000002 | 0.002254 |
| MELD | text | 0.100000 | dropout | all_retained | 1109 | 0.947701 | 0.004655 | 0.008970 | 0.000210 | 0.015690 |
| MELD | text | 0.100000 | gaussian | all_retained | 1109 | 0.988278 | 0.000848 | 0.000667 | 0.000009 | 0.002885 |
| MELD | visual | 0.050000 | dropout | all_retained | 1109 | 0.992786 | 0.000590 | 0.000382 | 0.000002 | 0.001984 |
| MELD | visual | 0.050000 | dropout | valid_visual | 1014 | 0.992110 | 0.000646 | 0.000418 | 0.000002 | 0.002170 |
| MELD | visual | 0.050000 | gaussian | all_retained | 1109 | 0.998197 | 0.000150 | 0.000051 | 0.000000 | 0.000451 |
| MELD | visual | 0.050000 | gaussian | valid_visual | 1014 | 0.998028 | 0.000164 | 0.000055 | 0.000000 | 0.000493 |
| MELD | visual | 0.100000 | dropout | all_retained | 1109 | 0.993688 | 0.000671 | 0.000692 | 0.000005 | 0.002344 |
| MELD | visual | 0.100000 | dropout | valid_visual | 1014 | 0.993097 | 0.000734 | 0.000757 | 0.000005 | 0.002564 |
| MELD | visual | 0.100000 | gaussian | all_retained | 1109 | 0.997295 | 0.000177 | 0.000103 | 0.000000 | 0.000541 |
| MELD | visual | 0.100000 | gaussian | valid_visual | 1014 | 0.997041 | 0.000193 | 0.000113 | 0.000000 | 0.000592 |

all_retained包含全部话语；valid_visual仅在visual/joint条件下排除视觉缺失话语。IEMOCAP 1条、MELD 95条缺失视觉仍保留在校准和全部话语统计中；visual跳过扰动产生的零响应不能解释为视觉可靠。joint的有效视觉子集作为补充诊断报告。

## 两个独立5repeat半组的响应稳定性

| dataset | modality | strength | corruption | population | measure | spearman | left_zero_fraction | right_zero_fraction | auc_larger_is_error |
|---|---|---|---|---|---|---|---|---|---|
| IEMOCAP | audio | 0.050000 | dropout | all_retained | mu | 0.648888 | 0.960152 | 0.969639 | 0.531633 |
| IEMOCAP | audio | 0.050000 | dropout | all_retained | delta | 0.540887 | 0.000000 | 0.000000 | 0.563989 |
| IEMOCAP | audio | 0.050000 | dropout | all_retained | js | 0.712108 | 0.000000 | 0.000000 | 0.569913 |
| IEMOCAP | audio | 0.050000 | dropout | all_retained | flip_rate | 0.648593 | 0.960152 | 0.969639 | 0.531730 |
| IEMOCAP | audio | 0.050000 | gaussian | all_retained | mu | 0.513913 | 0.990512 | 0.994307 | 0.504110 |
| IEMOCAP | audio | 0.050000 | gaussian | all_retained | delta | 0.461878 | 0.000000 | 0.000000 | 0.583606 |
| IEMOCAP | audio | 0.050000 | gaussian | all_retained | js | 0.716336 | 0.000000 | 0.000000 | 0.613947 |
| IEMOCAP | audio | 0.050000 | gaussian | all_retained | flip_rate | 0.514417 | 0.990512 | 0.994307 | 0.504094 |
| IEMOCAP | audio | 0.100000 | dropout | all_retained | mu | 0.860020 | 0.952562 | 0.941176 | 0.530469 |
| IEMOCAP | audio | 0.100000 | dropout | all_retained | delta | 0.678452 | 0.000000 | 0.000000 | 0.515122 |
| IEMOCAP | audio | 0.100000 | dropout | all_retained | js | 0.746425 | 0.000000 | 0.000000 | 0.559333 |
| IEMOCAP | audio | 0.100000 | dropout | all_retained | flip_rate | 0.859945 | 0.952562 | 0.941176 | 0.530574 |
| IEMOCAP | audio | 0.100000 | gaussian | all_retained | mu | 0.614204 | 0.986717 | 0.988615 | 0.504086 |
| IEMOCAP | audio | 0.100000 | gaussian | all_retained | delta | 0.489734 | 0.000000 | 0.000000 | 0.547181 |
| IEMOCAP | audio | 0.100000 | gaussian | all_retained | js | 0.717670 | 0.000000 | 0.000000 | 0.584313 |
| IEMOCAP | audio | 0.100000 | gaussian | all_retained | flip_rate | 0.613917 | 0.986717 | 0.988615 | 0.504086 |
| IEMOCAP | joint | 0.050000 | dropout | all_retained | mu | 0.769912 | 0.960152 | 0.965844 | 0.533182 |
| IEMOCAP | joint | 0.050000 | dropout | all_retained | delta | 0.554897 | 0.000000 | 0.000000 | 0.570186 |
| IEMOCAP | joint | 0.050000 | dropout | all_retained | js | 0.717645 | 0.000000 | 0.000000 | 0.569752 |
| IEMOCAP | joint | 0.050000 | dropout | all_retained | flip_rate | 0.770062 | 0.960152 | 0.965844 | 0.533174 |
| IEMOCAP | joint | 0.050000 | dropout | valid_visual | mu | 0.769908 | 0.960076 | 0.965779 | 0.533433 |
| IEMOCAP | joint | 0.050000 | dropout | valid_visual | delta | 0.559664 | 0.000000 | 0.000000 | 0.570176 |
| IEMOCAP | joint | 0.050000 | dropout | valid_visual | js | 0.718733 | 0.000000 | 0.000000 | 0.569579 |
| IEMOCAP | joint | 0.050000 | dropout | valid_visual | flip_rate | 0.770059 | 0.960076 | 0.965779 | 0.533425 |
| IEMOCAP | joint | 0.050000 | gaussian | all_retained | mu | 0.630626 | 0.990512 | 0.996205 | 0.505506 |
| IEMOCAP | joint | 0.050000 | gaussian | all_retained | delta | 0.430386 | 0.000000 | 0.000000 | 0.585741 |
| IEMOCAP | joint | 0.050000 | gaussian | all_retained | js | 0.723738 | 0.000000 | 0.000000 | 0.615312 |
| IEMOCAP | joint | 0.050000 | gaussian | all_retained | flip_rate | 0.631236 | 0.990512 | 0.996205 | 0.505506 |
| IEMOCAP | joint | 0.050000 | gaussian | valid_visual | mu | 0.630623 | 0.990494 | 0.996198 | 0.505553 |
| IEMOCAP | joint | 0.050000 | gaussian | valid_visual | delta | 0.428182 | 0.000000 | 0.000000 | 0.586804 |
| IEMOCAP | joint | 0.050000 | gaussian | valid_visual | js | 0.723076 | 0.000000 | 0.000000 | 0.613942 |
| IEMOCAP | joint | 0.050000 | gaussian | valid_visual | flip_rate | 0.631234 | 0.990494 | 0.996198 | 0.505553 |
| IEMOCAP | joint | 0.100000 | dropout | all_retained | mu | 0.763889 | 0.946869 | 0.933586 | 0.529402 |
| IEMOCAP | joint | 0.100000 | dropout | all_retained | delta | 0.632650 | 0.000000 | 0.000000 | 0.520500 |
| IEMOCAP | joint | 0.100000 | dropout | all_retained | js | 0.739291 | 0.000000 | 0.000000 | 0.564647 |
| IEMOCAP | joint | 0.100000 | dropout | all_retained | flip_rate | 0.762713 | 0.946869 | 0.933586 | 0.529466 |
| IEMOCAP | joint | 0.100000 | dropout | valid_visual | mu | 0.763882 | 0.946768 | 0.933460 | 0.529720 |
| IEMOCAP | joint | 0.100000 | dropout | valid_visual | delta | 0.631992 | 0.000000 | 0.000000 | 0.519114 |
| IEMOCAP | joint | 0.100000 | dropout | valid_visual | js | 0.738808 | 0.000000 | 0.000000 | 0.563848 |
| IEMOCAP | joint | 0.100000 | dropout | valid_visual | flip_rate | 0.762704 | 0.946768 | 0.933460 | 0.529785 |
| IEMOCAP | joint | 0.100000 | gaussian | all_retained | mu | 0.710390 | 0.986717 | 0.986717 | 0.504037 |
| IEMOCAP | joint | 0.100000 | gaussian | all_retained | delta | 0.457993 | 0.000000 | 0.000000 | 0.537308 |
| IEMOCAP | joint | 0.100000 | gaussian | all_retained | js | 0.725813 | 0.000000 | 0.000000 | 0.586287 |
| IEMOCAP | joint | 0.100000 | gaussian | all_retained | flip_rate | 0.712592 | 0.986717 | 0.986717 | 0.504045 |
| IEMOCAP | joint | 0.100000 | gaussian | valid_visual | mu | 0.710382 | 0.986692 | 0.986692 | 0.504100 |
| IEMOCAP | joint | 0.100000 | gaussian | valid_visual | delta | 0.459608 | 0.000000 | 0.000000 | 0.537679 |
| IEMOCAP | joint | 0.100000 | gaussian | valid_visual | js | 0.726321 | 0.000000 | 0.000000 | 0.585610 |
| IEMOCAP | joint | 0.100000 | gaussian | valid_visual | flip_rate | 0.712589 | 0.986692 | 0.986692 | 0.504109 |
| IEMOCAP | text | 0.050000 | dropout | all_retained | mu | 0.405117 | 0.988615 | 0.992410 | 0.509696 |
| IEMOCAP | text | 0.050000 | dropout | all_retained | delta | 0.825529 | 0.000000 | 0.000000 | 0.572866 |
| IEMOCAP | text | 0.050000 | dropout | all_retained | js | 0.931140 | 0.000000 | 0.000000 | 0.608120 |
| IEMOCAP | text | 0.050000 | dropout | all_retained | flip_rate | 0.406294 | 0.988615 | 0.992410 | 0.509680 |
| IEMOCAP | text | 0.050000 | gaussian | all_retained | mu | 1.000000 | 0.998102 | 0.998102 | 0.498563 |
| IEMOCAP | text | 0.050000 | gaussian | all_retained | delta | 0.615928 | 0.000000 | 0.000000 | 0.565883 |
| IEMOCAP | text | 0.050000 | gaussian | all_retained | js | 0.917925 | 0.000000 | 0.000000 | 0.581054 |
| IEMOCAP | text | 0.050000 | gaussian | all_retained | flip_rate | 1.000000 | 0.998102 | 0.998102 | 0.498563 |
| IEMOCAP | text | 0.100000 | dropout | all_retained | mu | 0.958287 | 0.977230 | 0.979127 | 0.508075 |
| IEMOCAP | text | 0.100000 | dropout | all_retained | delta | 0.854382 | 0.000000 | 0.000000 | 0.565787 |
| IEMOCAP | text | 0.100000 | dropout | all_retained | js | 0.943032 | 0.000000 | 0.000000 | 0.592773 |
| IEMOCAP | text | 0.100000 | dropout | all_retained | flip_rate | 0.957795 | 0.977230 | 0.979127 | 0.508091 |
| IEMOCAP | text | 0.100000 | gaussian | all_retained | mu | nan | 1.000000 | 0.998102 | 0.498563 |
| IEMOCAP | text | 0.100000 | gaussian | all_retained | delta | 0.547520 | 0.000000 | 0.000000 | 0.561806 |
| IEMOCAP | text | 0.100000 | gaussian | all_retained | js | 0.907574 | 0.000000 | 0.000000 | 0.583269 |
| IEMOCAP | text | 0.100000 | gaussian | all_retained | flip_rate | nan | 1.000000 | 0.998102 | 0.498563 |
| IEMOCAP | visual | 0.050000 | dropout | all_retained | mu | 1.000000 | 0.998102 | 0.998102 | 0.498563 |
| IEMOCAP | visual | 0.050000 | dropout | all_retained | delta | 0.605161 | 0.001898 | 0.001898 | 0.532171 |
| IEMOCAP | visual | 0.050000 | dropout | all_retained | js | 0.644328 | 0.001898 | 0.001898 | 0.565883 |
| IEMOCAP | visual | 0.050000 | dropout | all_retained | flip_rate | 1.000000 | 0.998102 | 0.998102 | 0.498563 |
| IEMOCAP | visual | 0.050000 | dropout | valid_visual | mu | 1.000000 | 0.998099 | 0.998099 | 0.498563 |
| IEMOCAP | visual | 0.050000 | dropout | valid_visual | delta | 0.602905 | 0.000000 | 0.000000 | 0.535161 |
| IEMOCAP | visual | 0.050000 | dropout | valid_visual | js | 0.642296 | 0.000000 | 0.000000 | 0.569062 |
| IEMOCAP | visual | 0.050000 | dropout | valid_visual | flip_rate | 1.000000 | 0.998099 | 0.998099 | 0.498563 |
| IEMOCAP | visual | 0.050000 | gaussian | all_retained | mu | 1.000000 | 0.998102 | 0.998102 | 0.498563 |
| IEMOCAP | visual | 0.050000 | gaussian | all_retained | delta | 0.444263 | 0.001898 | 0.001898 | 0.558370 |
| IEMOCAP | visual | 0.050000 | gaussian | all_retained | js | 0.723375 | 0.001898 | 0.001898 | 0.607317 |
| IEMOCAP | visual | 0.050000 | gaussian | all_retained | flip_rate | 1.000000 | 0.998102 | 0.998102 | 0.498563 |
| IEMOCAP | visual | 0.050000 | gaussian | valid_visual | mu | 1.000000 | 0.998099 | 0.998099 | 0.498563 |
| IEMOCAP | visual | 0.050000 | gaussian | valid_visual | delta | 0.441087 | 0.000000 | 0.000000 | 0.561507 |
| IEMOCAP | visual | 0.050000 | gaussian | valid_visual | js | 0.721795 | 0.000000 | 0.000000 | 0.610729 |
| IEMOCAP | visual | 0.050000 | gaussian | valid_visual | flip_rate | 1.000000 | 0.998099 | 0.998099 | 0.498563 |
| IEMOCAP | visual | 0.100000 | dropout | all_retained | mu | 1.000000 | 0.998102 | 0.998102 | 0.498563 |
| IEMOCAP | visual | 0.100000 | dropout | all_retained | delta | 0.623754 | 0.001898 | 0.001898 | 0.513164 |
| IEMOCAP | visual | 0.100000 | dropout | all_retained | js | 0.736737 | 0.001898 | 0.001898 | 0.537148 |
| IEMOCAP | visual | 0.100000 | dropout | all_retained | flip_rate | 1.000000 | 0.998102 | 0.998102 | 0.498563 |
| IEMOCAP | visual | 0.100000 | dropout | valid_visual | mu | 1.000000 | 0.998099 | 0.998099 | 0.498563 |
| IEMOCAP | visual | 0.100000 | dropout | valid_visual | delta | 0.621604 | 0.000000 | 0.000000 | 0.516047 |
| IEMOCAP | visual | 0.100000 | dropout | valid_visual | js | 0.735233 | 0.000000 | 0.000000 | 0.540165 |
| IEMOCAP | visual | 0.100000 | dropout | valid_visual | flip_rate | 1.000000 | 0.998099 | 0.998099 | 0.498563 |
| IEMOCAP | visual | 0.100000 | gaussian | all_retained | mu | nan | 0.998102 | 1.000000 | 0.498563 |
| IEMOCAP | visual | 0.100000 | gaussian | all_retained | delta | 0.475295 | 0.001898 | 0.001898 | 0.555705 |
| IEMOCAP | visual | 0.100000 | gaussian | all_retained | js | 0.759594 | 0.001898 | 0.001898 | 0.604636 |
| IEMOCAP | visual | 0.100000 | gaussian | all_retained | flip_rate | nan | 0.998102 | 1.000000 | 0.498563 |
| IEMOCAP | visual | 0.100000 | gaussian | valid_visual | mu | nan | 0.998099 | 1.000000 | 0.498563 |
| IEMOCAP | visual | 0.100000 | gaussian | valid_visual | delta | 0.472297 | 0.000000 | 0.000000 | 0.558827 |
| IEMOCAP | visual | 0.100000 | gaussian | valid_visual | js | 0.758220 | 0.000000 | 0.000000 | 0.608033 |
| IEMOCAP | visual | 0.100000 | gaussian | valid_visual | flip_rate | nan | 0.998099 | 1.000000 | 0.498563 |
| MELD | audio | 0.050000 | dropout | all_retained | mu | 0.770304 | 0.991885 | 0.989179 | 0.511613 |
| MELD | audio | 0.050000 | dropout | all_retained | delta | 0.292716 | 0.000000 | 0.000000 | 0.454611 |
| MELD | audio | 0.050000 | dropout | all_retained | js | 0.404176 | 0.000000 | 0.000000 | 0.534163 |
| MELD | audio | 0.050000 | dropout | all_retained | flip_rate | 0.770217 | 0.991885 | 0.989179 | 0.511604 |
| MELD | audio | 0.050000 | gaussian | all_retained | mu | 0.706465 | 0.998197 | 0.996393 | 0.505450 |
| MELD | audio | 0.050000 | gaussian | all_retained | delta | 0.034914 | 0.000000 | 0.000000 | 0.476090 |
| MELD | audio | 0.050000 | gaussian | all_retained | js | 0.250266 | 0.000000 | 0.000000 | 0.561892 |
| MELD | audio | 0.050000 | gaussian | all_retained | flip_rate | 0.706787 | 0.998197 | 0.996393 | 0.505450 |
| MELD | audio | 0.100000 | dropout | all_retained | mu | 0.833916 | 0.984671 | 0.982867 | 0.506347 |
| MELD | audio | 0.100000 | dropout | all_retained | delta | 0.516170 | 0.000000 | 0.000000 | 0.445317 |
| MELD | audio | 0.100000 | dropout | all_retained | js | 0.538741 | 0.000000 | 0.000000 | 0.552572 |
| MELD | audio | 0.100000 | dropout | all_retained | flip_rate | 0.833875 | 0.984671 | 0.982867 | 0.506347 |
| MELD | audio | 0.100000 | gaussian | all_retained | mu | 0.913283 | 0.995491 | 0.994590 | 0.508174 |
| MELD | audio | 0.100000 | gaussian | all_retained | delta | 0.080547 | 0.000000 | 0.000000 | 0.500801 |
| MELD | audio | 0.100000 | gaussian | all_retained | js | 0.260131 | 0.000000 | 0.000000 | 0.588629 |
| MELD | audio | 0.100000 | gaussian | all_retained | flip_rate | 0.912954 | 0.995491 | 0.994590 | 0.508174 |
| MELD | joint | 0.050000 | dropout | all_retained | mu | 0.730081 | 0.967538 | 0.968440 | 0.528812 |
| MELD | joint | 0.050000 | dropout | all_retained | delta | 0.490401 | 0.000000 | 0.000000 | 0.483159 |
| MELD | joint | 0.050000 | dropout | all_retained | js | 0.743002 | 0.000000 | 0.000000 | 0.638561 |
| MELD | joint | 0.050000 | dropout | all_retained | flip_rate | 0.728819 | 0.967538 | 0.968440 | 0.528840 |
| MELD | joint | 0.050000 | dropout | valid_visual | mu | 0.755285 | 0.967456 | 0.967456 | 0.527568 |
| MELD | joint | 0.050000 | dropout | valid_visual | delta | 0.482798 | 0.000000 | 0.000000 | 0.485954 |
| MELD | joint | 0.050000 | dropout | valid_visual | js | 0.742087 | 0.000000 | 0.000000 | 0.644149 |
| MELD | joint | 0.050000 | dropout | valid_visual | flip_rate | 0.753805 | 0.967456 | 0.967456 | 0.527598 |
| MELD | joint | 0.050000 | gaussian | all_retained | mu | 0.675653 | 0.995491 | 0.993688 | 0.508872 |
| MELD | joint | 0.050000 | gaussian | all_retained | delta | 0.079876 | 0.000000 | 0.000000 | 0.553974 |
| MELD | joint | 0.050000 | gaussian | all_retained | js | 0.640365 | 0.000000 | 0.000000 | 0.696167 |
| MELD | joint | 0.050000 | gaussian | all_retained | flip_rate | 0.675351 | 0.995491 | 0.993688 | 0.508872 |
| MELD | joint | 0.050000 | gaussian | valid_visual | mu | 0.757043 | 0.996055 | 0.993097 | 0.508172 |
| MELD | joint | 0.050000 | gaussian | valid_visual | delta | 0.064063 | 0.000000 | 0.000000 | 0.551201 |
| MELD | joint | 0.050000 | gaussian | valid_visual | js | 0.643498 | 0.000000 | 0.000000 | 0.703111 |
| MELD | joint | 0.050000 | gaussian | valid_visual | flip_rate | 0.756486 | 0.996055 | 0.993097 | 0.508172 |
| MELD | joint | 0.100000 | dropout | all_retained | mu | 0.746750 | 0.952209 | 0.956718 | 0.535031 |
| MELD | joint | 0.100000 | dropout | all_retained | delta | 0.694261 | 0.000000 | 0.000000 | 0.480379 |
| MELD | joint | 0.100000 | dropout | all_retained | js | 0.829366 | 0.000000 | 0.000000 | 0.631547 |
| MELD | joint | 0.100000 | dropout | all_retained | flip_rate | 0.746504 | 0.952209 | 0.956718 | 0.535132 |
| MELD | joint | 0.100000 | dropout | valid_visual | mu | 0.756524 | 0.950690 | 0.958580 | 0.533441 |
| MELD | joint | 0.100000 | dropout | valid_visual | delta | 0.683968 | 0.000000 | 0.000000 | 0.482356 |
| MELD | joint | 0.100000 | dropout | valid_visual | js | 0.825441 | 0.000000 | 0.000000 | 0.635567 |
| MELD | joint | 0.100000 | dropout | valid_visual | flip_rate | 0.756883 | 0.950690 | 0.958580 | 0.533553 |
| MELD | joint | 0.100000 | gaussian | all_retained | mu | 0.800106 | 0.989179 | 0.988278 | 0.512342 |
| MELD | joint | 0.100000 | gaussian | all_retained | delta | 0.112951 | 0.000000 | 0.000000 | 0.522999 |
| MELD | joint | 0.100000 | gaussian | all_retained | js | 0.619568 | 0.000000 | 0.000000 | 0.681511 |
| MELD | joint | 0.100000 | gaussian | all_retained | flip_rate | 0.799606 | 0.989179 | 0.988278 | 0.512346 |
| MELD | joint | 0.100000 | gaussian | valid_visual | mu | 0.782494 | 0.989152 | 0.988166 | 0.511935 |
| MELD | joint | 0.100000 | gaussian | valid_visual | delta | 0.121391 | 0.000000 | 0.000000 | 0.519275 |
| MELD | joint | 0.100000 | gaussian | valid_visual | js | 0.616487 | 0.000000 | 0.000000 | 0.675843 |
| MELD | joint | 0.100000 | gaussian | valid_visual | flip_rate | 0.781904 | 0.989152 | 0.988166 | 0.511940 |
| MELD | text | 0.050000 | dropout | all_retained | mu | 0.700915 | 0.973850 | 0.974752 | 0.525922 |
| MELD | text | 0.050000 | dropout | all_retained | delta | 0.530586 | 0.000000 | 0.000000 | 0.498447 |
| MELD | text | 0.050000 | dropout | all_retained | js | 0.775092 | 0.000000 | 0.000000 | 0.661743 |
| MELD | text | 0.050000 | dropout | all_retained | flip_rate | 0.700062 | 0.973850 | 0.974752 | 0.525944 |
| MELD | text | 0.050000 | gaussian | all_retained | mu | 0.675962 | 0.995491 | 0.993688 | 0.508868 |
| MELD | text | 0.050000 | gaussian | all_retained | delta | 0.070641 | 0.000000 | 0.000000 | 0.555223 |
| MELD | text | 0.050000 | gaussian | all_retained | js | 0.689552 | 0.000000 | 0.000000 | 0.695003 |
| MELD | text | 0.050000 | gaussian | all_retained | flip_rate | 0.675655 | 0.995491 | 0.993688 | 0.508872 |
| MELD | text | 0.100000 | dropout | all_retained | mu | 0.662495 | 0.959423 | 0.962128 | 0.534130 |
| MELD | text | 0.100000 | dropout | all_retained | delta | 0.710973 | 0.000000 | 0.000000 | 0.503364 |
| MELD | text | 0.100000 | dropout | all_retained | js | 0.850583 | 0.000000 | 0.000000 | 0.641785 |
| MELD | text | 0.100000 | dropout | all_retained | flip_rate | 0.662915 | 0.959423 | 0.962128 | 0.534137 |
| MELD | text | 0.100000 | gaussian | all_retained | mu | 0.817841 | 0.990081 | 0.990081 | 0.513657 |
| MELD | text | 0.100000 | gaussian | all_retained | delta | 0.093742 | 0.000000 | 0.000000 | 0.547122 |
| MELD | text | 0.100000 | gaussian | all_retained | js | 0.697461 | 0.000000 | 0.000000 | 0.685459 |
| MELD | text | 0.100000 | gaussian | all_retained | flip_rate | 0.817674 | 0.990081 | 0.990081 | 0.513659 |
| MELD | visual | 0.050000 | dropout | all_retained | mu | 0.666365 | 0.994590 | 0.994590 | 0.506838 |
| MELD | visual | 0.050000 | dropout | all_retained | delta | 0.526822 | 0.085663 | 0.085663 | 0.535889 |
| MELD | visual | 0.050000 | dropout | all_retained | js | 0.609570 | 0.085663 | 0.085663 | 0.547978 |
| MELD | visual | 0.050000 | dropout | all_retained | flip_rate | 0.666364 | 0.994590 | 0.994590 | 0.506841 |
| MELD | visual | 0.050000 | dropout | valid_visual | mu | 0.666337 | 0.994083 | 0.994083 | 0.507438 |
| MELD | visual | 0.050000 | dropout | valid_visual | delta | 0.381369 | 0.000000 | 0.000000 | 0.539341 |
| MELD | visual | 0.050000 | dropout | valid_visual | js | 0.489553 | 0.000000 | 0.000000 | 0.553770 |
| MELD | visual | 0.050000 | dropout | valid_visual | flip_rate | 0.666336 | 0.994083 | 0.994083 | 0.507443 |
| MELD | visual | 0.050000 | gaussian | all_retained | mu | 0.707426 | 0.999098 | 0.998197 | 0.502725 |
| MELD | visual | 0.050000 | gaussian | all_retained | delta | 0.376586 | 0.085663 | 0.085663 | 0.521240 |
| MELD | visual | 0.050000 | gaussian | all_retained | js | 0.463132 | 0.085663 | 0.085663 | 0.604361 |
| MELD | visual | 0.050000 | gaussian | all_retained | flip_rate | 0.706788 | 0.999098 | 0.998197 | 0.502725 |
| MELD | visual | 0.050000 | gaussian | valid_visual | mu | 0.707456 | 0.999014 | 0.998028 | 0.502967 |
| MELD | visual | 0.050000 | gaussian | valid_visual | delta | 0.184949 | 0.000000 | 0.000000 | 0.521856 |
| MELD | visual | 0.050000 | gaussian | valid_visual | js | 0.298100 | 0.000000 | 0.000000 | 0.621068 |
| MELD | visual | 0.050000 | gaussian | valid_visual | flip_rate | 0.706758 | 0.999014 | 0.998028 | 0.502967 |
| MELD | visual | 0.100000 | dropout | all_retained | mu | 0.925959 | 0.994590 | 0.993688 | 0.507512 |
| MELD | visual | 0.100000 | dropout | all_retained | delta | 0.618837 | 0.085663 | 0.085663 | 0.520326 |
| MELD | visual | 0.100000 | dropout | all_retained | js | 0.703784 | 0.085663 | 0.085663 | 0.545866 |
| MELD | visual | 0.100000 | dropout | all_retained | flip_rate | 0.926096 | 0.994590 | 0.993688 | 0.507512 |
| MELD | visual | 0.100000 | dropout | valid_visual | mu | 0.925972 | 0.994083 | 0.993097 | 0.508177 |
| MELD | visual | 0.100000 | dropout | valid_visual | delta | 0.501668 | 0.000000 | 0.000000 | 0.520765 |
| MELD | visual | 0.100000 | dropout | valid_visual | js | 0.612728 | 0.000000 | 0.000000 | 0.551249 |
| MELD | visual | 0.100000 | dropout | valid_visual | flip_rate | 0.926121 | 0.994083 | 0.993097 | 0.508177 |
| MELD | visual | 0.100000 | gaussian | all_retained | mu | 0.500000 | 0.998197 | 0.998197 | 0.504087 |
| MELD | visual | 0.100000 | gaussian | all_retained | delta | 0.359925 | 0.085663 | 0.085663 | 0.521578 |
| MELD | visual | 0.100000 | gaussian | all_retained | js | 0.421661 | 0.085663 | 0.085663 | 0.568355 |
| MELD | visual | 0.100000 | gaussian | all_retained | flip_rate | 0.499548 | 0.998197 | 0.998197 | 0.504087 |
| MELD | visual | 0.100000 | gaussian | valid_visual | mu | 0.500000 | 0.998028 | 0.998028 | 0.504451 |
| MELD | visual | 0.100000 | gaussian | valid_visual | delta | 0.163167 | 0.000000 | 0.000000 | 0.522260 |
| MELD | visual | 0.100000 | gaussian | valid_visual | js | 0.243881 | 0.000000 | 0.000000 | 0.578092 |
| MELD | visual | 0.100000 | gaussian | valid_visual | flip_rate | 0.499506 | 0.998028 | 0.998028 | 0.504451 |

固定重复0–4与5–9计算Spearman，不筛选更有利的拆分。常量半组的相关系数为NA；同时报告两半组零值占比。AUC方向固定为较大响应对应较高错误风险，未按结果翻转方向。这反映测量与无条件分离能力，不能代替控制原始置信度后的增量证据。

## 相同Dialogue内近似原始置信度配对

| dataset | modality | strength | corruption | population | measure | n_pairs | n_pair_dialogues | mean_wrong_minus_correct | insufficient_support |
|---|---|---|---|---|---|---|---|---|---|
| IEMOCAP | audio | 0.050000 | dropout | all_retained | mu | 114 | 12 | 0.002493 | False |
| IEMOCAP | audio | 0.050000 | dropout | all_retained | delta | 114 | 12 | 0.001525 | False |
| IEMOCAP | audio | 0.050000 | dropout | all_retained | js | 114 | 12 | 0.000105 | False |
| IEMOCAP | audio | 0.050000 | dropout | all_retained | flip_rate | 114 | 12 | 0.005263 | False |
| IEMOCAP | audio | 0.050000 | gaussian | all_retained | mu | 114 | 12 | -0.001891 | False |
| IEMOCAP | audio | 0.050000 | gaussian | all_retained | delta | 114 | 12 | -0.000101 | False |
| IEMOCAP | audio | 0.050000 | gaussian | all_retained | js | 114 | 12 | -0.000001 | False |
| IEMOCAP | audio | 0.050000 | gaussian | all_retained | flip_rate | 114 | 12 | -0.004386 | False |
| IEMOCAP | audio | 0.100000 | dropout | all_retained | mu | 114 | 12 | 0.007680 | False |
| IEMOCAP | audio | 0.100000 | dropout | all_retained | delta | 114 | 12 | 0.000835 | False |
| IEMOCAP | audio | 0.100000 | dropout | all_retained | js | 114 | 12 | 0.000173 | False |
| IEMOCAP | audio | 0.100000 | dropout | all_retained | flip_rate | 114 | 12 | 0.015789 | False |
| IEMOCAP | audio | 0.100000 | gaussian | all_retained | mu | 114 | 12 | -0.001872 | False |
| IEMOCAP | audio | 0.100000 | gaussian | all_retained | delta | 114 | 12 | -0.000107 | False |
| IEMOCAP | audio | 0.100000 | gaussian | all_retained | js | 114 | 12 | 0.000002 | False |
| IEMOCAP | audio | 0.100000 | gaussian | all_retained | flip_rate | 114 | 12 | -0.004386 | False |
| IEMOCAP | joint | 0.050000 | dropout | all_retained | mu | 114 | 12 | 0.005008 | False |
| IEMOCAP | joint | 0.050000 | dropout | all_retained | delta | 114 | 12 | 0.001728 | False |
| IEMOCAP | joint | 0.050000 | dropout | all_retained | js | 114 | 12 | 0.000105 | False |
| IEMOCAP | joint | 0.050000 | dropout | all_retained | flip_rate | 114 | 12 | 0.010526 | False |
| IEMOCAP | joint | 0.050000 | dropout | valid_visual | mu | 113 | 12 | 0.005052 | False |
| IEMOCAP | joint | 0.050000 | dropout | valid_visual | delta | 113 | 12 | 0.001710 | False |
| IEMOCAP | joint | 0.050000 | dropout | valid_visual | js | 113 | 12 | 0.000106 | False |
| IEMOCAP | joint | 0.050000 | dropout | valid_visual | flip_rate | 113 | 12 | 0.010619 | False |
| IEMOCAP | joint | 0.050000 | gaussian | all_retained | mu | 114 | 12 | -0.001868 | False |
| IEMOCAP | joint | 0.050000 | gaussian | all_retained | delta | 114 | 12 | -0.000123 | False |
| IEMOCAP | joint | 0.050000 | gaussian | all_retained | js | 114 | 12 | -0.000001 | False |
| IEMOCAP | joint | 0.050000 | gaussian | all_retained | flip_rate | 114 | 12 | -0.004386 | False |
| IEMOCAP | joint | 0.050000 | gaussian | valid_visual | mu | 113 | 12 | -0.001885 | False |
| IEMOCAP | joint | 0.050000 | gaussian | valid_visual | delta | 113 | 12 | -0.000121 | False |
| IEMOCAP | joint | 0.050000 | gaussian | valid_visual | js | 113 | 12 | -0.000001 | False |
| IEMOCAP | joint | 0.050000 | gaussian | valid_visual | flip_rate | 113 | 12 | -0.004425 | False |
| IEMOCAP | joint | 0.100000 | dropout | all_retained | mu | 114 | 12 | 0.006605 | False |
| IEMOCAP | joint | 0.100000 | dropout | all_retained | delta | 114 | 12 | 0.001059 | False |
| IEMOCAP | joint | 0.100000 | dropout | all_retained | js | 114 | 12 | 0.000166 | False |
| IEMOCAP | joint | 0.100000 | dropout | all_retained | flip_rate | 114 | 12 | 0.013158 | False |
| IEMOCAP | joint | 0.100000 | dropout | valid_visual | mu | 113 | 12 | 0.006664 | False |
| IEMOCAP | joint | 0.100000 | dropout | valid_visual | delta | 113 | 12 | 0.000987 | False |
| IEMOCAP | joint | 0.100000 | dropout | valid_visual | js | 113 | 12 | 0.000165 | False |
| IEMOCAP | joint | 0.100000 | dropout | valid_visual | flip_rate | 113 | 12 | 0.013274 | False |
| IEMOCAP | joint | 0.100000 | gaussian | all_retained | mu | 114 | 12 | -0.002252 | False |
| IEMOCAP | joint | 0.100000 | gaussian | all_retained | delta | 114 | 12 | -0.000126 | False |
| IEMOCAP | joint | 0.100000 | gaussian | all_retained | js | 114 | 12 | 0.000002 | False |
| IEMOCAP | joint | 0.100000 | gaussian | all_retained | flip_rate | 114 | 12 | -0.005263 | False |
| IEMOCAP | joint | 0.100000 | gaussian | valid_visual | mu | 113 | 12 | -0.002272 | False |
| IEMOCAP | joint | 0.100000 | gaussian | valid_visual | delta | 113 | 12 | -0.000129 | False |
| IEMOCAP | joint | 0.100000 | gaussian | valid_visual | js | 113 | 12 | 0.000002 | False |
| IEMOCAP | joint | 0.100000 | gaussian | valid_visual | flip_rate | 113 | 12 | -0.005310 | False |
| IEMOCAP | text | 0.050000 | dropout | all_retained | mu | 114 | 12 | 0.000117 | False |
| IEMOCAP | text | 0.050000 | dropout | all_retained | delta | 114 | 12 | 0.000132 | False |
| IEMOCAP | text | 0.050000 | dropout | all_retained | js | 114 | 12 | 0.000001 | False |
| IEMOCAP | text | 0.050000 | dropout | all_retained | flip_rate | 114 | 12 | 0.000000 | False |
| IEMOCAP | text | 0.050000 | gaussian | all_retained | mu | 114 | 12 | -0.002236 | False |
| IEMOCAP | text | 0.050000 | gaussian | all_retained | delta | 114 | 12 | -0.000037 | False |
| IEMOCAP | text | 0.050000 | gaussian | all_retained | js | 114 | 12 | -0.000000 | False |
| IEMOCAP | text | 0.050000 | gaussian | all_retained | flip_rate | 114 | 12 | -0.005263 | False |
| IEMOCAP | text | 0.100000 | dropout | all_retained | mu | 114 | 12 | -0.003044 | False |
| IEMOCAP | text | 0.100000 | dropout | all_retained | delta | 114 | 12 | 0.000220 | False |
| IEMOCAP | text | 0.100000 | dropout | all_retained | js | 114 | 12 | 0.000002 | False |
| IEMOCAP | text | 0.100000 | dropout | all_retained | flip_rate | 114 | 12 | -0.007018 | False |
| IEMOCAP | text | 0.100000 | gaussian | all_retained | mu | 114 | 12 | -0.000373 | False |
| IEMOCAP | text | 0.100000 | gaussian | all_retained | delta | 114 | 12 | 0.000008 | False |
| IEMOCAP | text | 0.100000 | gaussian | all_retained | js | 114 | 12 | -0.000000 | False |
| IEMOCAP | text | 0.100000 | gaussian | all_retained | flip_rate | 114 | 12 | -0.000877 | False |
| IEMOCAP | visual | 0.050000 | dropout | all_retained | mu | 114 | 12 | -0.001865 | False |
| IEMOCAP | visual | 0.050000 | dropout | all_retained | delta | 114 | 12 | -0.000017 | False |
| IEMOCAP | visual | 0.050000 | dropout | all_retained | js | 114 | 12 | 0.000000 | False |
| IEMOCAP | visual | 0.050000 | dropout | all_retained | flip_rate | 114 | 12 | -0.004386 | False |
| IEMOCAP | visual | 0.050000 | dropout | valid_visual | mu | 113 | 12 | -0.001881 | False |
| IEMOCAP | visual | 0.050000 | dropout | valid_visual | delta | 113 | 12 | -0.000016 | False |
| IEMOCAP | visual | 0.050000 | dropout | valid_visual | js | 113 | 12 | 0.000000 | False |
| IEMOCAP | visual | 0.050000 | dropout | valid_visual | flip_rate | 113 | 12 | -0.004425 | False |
| IEMOCAP | visual | 0.050000 | gaussian | all_retained | mu | 114 | 12 | -0.001491 | False |
| IEMOCAP | visual | 0.050000 | gaussian | all_retained | delta | 114 | 12 | -0.000005 | False |
| IEMOCAP | visual | 0.050000 | gaussian | all_retained | js | 114 | 12 | -0.000000 | False |
| IEMOCAP | visual | 0.050000 | gaussian | all_retained | flip_rate | 114 | 12 | -0.003509 | False |
| IEMOCAP | visual | 0.050000 | gaussian | valid_visual | mu | 113 | 12 | -0.001504 | False |
| IEMOCAP | visual | 0.050000 | gaussian | valid_visual | delta | 113 | 12 | -0.000004 | False |
| IEMOCAP | visual | 0.050000 | gaussian | valid_visual | js | 113 | 12 | -0.000000 | False |
| IEMOCAP | visual | 0.050000 | gaussian | valid_visual | flip_rate | 113 | 12 | -0.003540 | False |
| IEMOCAP | visual | 0.100000 | dropout | all_retained | mu | 114 | 12 | -0.000746 | False |
| IEMOCAP | visual | 0.100000 | dropout | all_retained | delta | 114 | 12 | -0.000080 | False |
| IEMOCAP | visual | 0.100000 | dropout | all_retained | js | 114 | 12 | -0.000000 | False |
| IEMOCAP | visual | 0.100000 | dropout | all_retained | flip_rate | 114 | 12 | -0.001754 | False |
| IEMOCAP | visual | 0.100000 | dropout | valid_visual | mu | 113 | 12 | -0.000753 | False |
| IEMOCAP | visual | 0.100000 | dropout | valid_visual | delta | 113 | 12 | -0.000080 | False |
| IEMOCAP | visual | 0.100000 | dropout | valid_visual | js | 113 | 12 | -0.000000 | False |
| IEMOCAP | visual | 0.100000 | dropout | valid_visual | flip_rate | 113 | 12 | -0.001770 | False |
| IEMOCAP | visual | 0.100000 | gaussian | all_retained | mu | 114 | 12 | -0.000745 | False |
| IEMOCAP | visual | 0.100000 | gaussian | all_retained | delta | 114 | 12 | 0.000003 | False |
| IEMOCAP | visual | 0.100000 | gaussian | all_retained | js | 114 | 12 | -0.000000 | False |
| IEMOCAP | visual | 0.100000 | gaussian | all_retained | flip_rate | 114 | 12 | -0.001754 | False |
| IEMOCAP | visual | 0.100000 | gaussian | valid_visual | mu | 113 | 12 | -0.000752 | False |
| IEMOCAP | visual | 0.100000 | gaussian | valid_visual | delta | 113 | 12 | 0.000003 | False |
| IEMOCAP | visual | 0.100000 | gaussian | valid_visual | js | 113 | 12 | -0.000000 | False |
| IEMOCAP | visual | 0.100000 | gaussian | valid_visual | flip_rate | 113 | 12 | -0.001770 | False |
| MELD | audio | 0.050000 | dropout | all_retained | mu | 137 | 66 | 0.001720 | False |
| MELD | audio | 0.050000 | dropout | all_retained | delta | 137 | 66 | -0.000412 | False |
| MELD | audio | 0.050000 | dropout | all_retained | js | 137 | 66 | -0.000000 | False |
| MELD | audio | 0.050000 | dropout | all_retained | flip_rate | 137 | 66 | 0.005109 | False |
| MELD | audio | 0.050000 | gaussian | all_retained | mu | 137 | 66 | 0.000000 | False |
| MELD | audio | 0.050000 | gaussian | all_retained | delta | 137 | 66 | -0.000016 | False |
| MELD | audio | 0.050000 | gaussian | all_retained | js | 137 | 66 | -0.000000 | False |
| MELD | audio | 0.050000 | gaussian | all_retained | flip_rate | 137 | 66 | 0.000000 | False |
| MELD | audio | 0.100000 | dropout | all_retained | mu | 137 | 66 | 0.001839 | False |
| MELD | audio | 0.100000 | dropout | all_retained | delta | 137 | 66 | -0.000741 | False |
| MELD | audio | 0.100000 | dropout | all_retained | js | 137 | 66 | -0.000003 | False |
| MELD | audio | 0.100000 | dropout | all_retained | flip_rate | 137 | 66 | 0.005109 | False |
| MELD | audio | 0.100000 | gaussian | all_retained | mu | 137 | 66 | 0.000586 | False |
| MELD | audio | 0.100000 | gaussian | all_retained | delta | 137 | 66 | 0.000016 | False |
| MELD | audio | 0.100000 | gaussian | all_retained | js | 137 | 66 | -0.000000 | False |
| MELD | audio | 0.100000 | gaussian | all_retained | flip_rate | 137 | 66 | 0.002190 | False |
| MELD | joint | 0.050000 | dropout | all_retained | mu | 137 | 66 | 0.001672 | False |
| MELD | joint | 0.050000 | dropout | all_retained | delta | 137 | 66 | -0.000539 | False |
| MELD | joint | 0.050000 | dropout | all_retained | js | 137 | 66 | -0.000000 | False |
| MELD | joint | 0.050000 | dropout | all_retained | flip_rate | 137 | 66 | 0.005109 | False |
| MELD | joint | 0.050000 | dropout | valid_visual | mu | 120 | 58 | 0.001909 | False |
| MELD | joint | 0.050000 | dropout | valid_visual | delta | 120 | 58 | -0.000618 | False |
| MELD | joint | 0.050000 | dropout | valid_visual | js | 120 | 58 | 0.000002 | False |
| MELD | joint | 0.050000 | dropout | valid_visual | flip_rate | 120 | 58 | 0.005833 | False |
| MELD | joint | 0.050000 | gaussian | all_retained | mu | 137 | 66 | 0.001214 | False |
| MELD | joint | 0.050000 | gaussian | all_retained | delta | 137 | 66 | 0.000056 | False |
| MELD | joint | 0.050000 | gaussian | all_retained | js | 137 | 66 | 0.000000 | False |
| MELD | joint | 0.050000 | gaussian | all_retained | flip_rate | 137 | 66 | 0.004380 | False |
| MELD | joint | 0.050000 | gaussian | valid_visual | mu | 120 | 58 | 0.001386 | False |
| MELD | joint | 0.050000 | gaussian | valid_visual | delta | 120 | 58 | 0.000051 | False |
| MELD | joint | 0.050000 | gaussian | valid_visual | js | 120 | 58 | 0.000000 | False |
| MELD | joint | 0.050000 | gaussian | valid_visual | flip_rate | 120 | 58 | 0.005000 | False |
| MELD | joint | 0.100000 | dropout | all_retained | mu | 137 | 66 | 0.000934 | False |
| MELD | joint | 0.100000 | dropout | all_retained | delta | 137 | 66 | -0.000393 | False |
| MELD | joint | 0.100000 | dropout | all_retained | js | 137 | 66 | 0.000002 | False |
| MELD | joint | 0.100000 | dropout | all_retained | flip_rate | 137 | 66 | 0.001460 | False |
| MELD | joint | 0.100000 | dropout | valid_visual | mu | 120 | 58 | 0.001359 | False |
| MELD | joint | 0.100000 | dropout | valid_visual | delta | 120 | 58 | -0.000793 | False |
| MELD | joint | 0.100000 | dropout | valid_visual | js | 120 | 58 | -0.000005 | False |
| MELD | joint | 0.100000 | dropout | valid_visual | flip_rate | 120 | 58 | 0.002500 | False |
| MELD | joint | 0.100000 | gaussian | all_retained | mu | 137 | 66 | 0.001016 | False |
| MELD | joint | 0.100000 | gaussian | all_retained | delta | 137 | 66 | 0.000008 | False |
| MELD | joint | 0.100000 | gaussian | all_retained | js | 137 | 66 | 0.000000 | False |
| MELD | joint | 0.100000 | gaussian | all_retained | flip_rate | 137 | 66 | 0.003650 | False |
| MELD | joint | 0.100000 | gaussian | valid_visual | mu | 120 | 58 | 0.001160 | False |
| MELD | joint | 0.100000 | gaussian | valid_visual | delta | 120 | 58 | 0.000012 | False |
| MELD | joint | 0.100000 | gaussian | valid_visual | js | 120 | 58 | 0.000000 | False |
| MELD | joint | 0.100000 | gaussian | valid_visual | flip_rate | 120 | 58 | 0.004167 | False |
| MELD | text | 0.050000 | dropout | all_retained | mu | 137 | 66 | 0.000330 | False |
| MELD | text | 0.050000 | dropout | all_retained | delta | 137 | 66 | -0.000301 | False |
| MELD | text | 0.050000 | dropout | all_retained | js | 137 | 66 | 0.000004 | False |
| MELD | text | 0.050000 | dropout | all_retained | flip_rate | 137 | 66 | 0.000730 | False |
| MELD | text | 0.050000 | gaussian | all_retained | mu | 137 | 66 | 0.000979 | False |
| MELD | text | 0.050000 | gaussian | all_retained | delta | 137 | 66 | 0.000059 | False |
| MELD | text | 0.050000 | gaussian | all_retained | js | 137 | 66 | 0.000000 | False |
| MELD | text | 0.050000 | gaussian | all_retained | flip_rate | 137 | 66 | 0.003650 | False |
| MELD | text | 0.100000 | dropout | all_retained | mu | 137 | 66 | -0.000734 | False |
| MELD | text | 0.100000 | dropout | all_retained | delta | 137 | 66 | 0.000046 | False |
| MELD | text | 0.100000 | dropout | all_retained | js | 137 | 66 | 0.000009 | False |
| MELD | text | 0.100000 | dropout | all_retained | flip_rate | 137 | 66 | -0.003650 | False |
| MELD | text | 0.100000 | gaussian | all_retained | mu | 137 | 66 | 0.001480 | False |
| MELD | text | 0.100000 | gaussian | all_retained | delta | 137 | 66 | 0.000012 | False |
| MELD | text | 0.100000 | gaussian | all_retained | js | 137 | 66 | 0.000001 | False |
| MELD | text | 0.100000 | gaussian | all_retained | flip_rate | 137 | 66 | 0.005109 | False |
| MELD | visual | 0.050000 | dropout | all_retained | mu | 137 | 66 | 0.000968 | False |
| MELD | visual | 0.050000 | dropout | all_retained | delta | 137 | 66 | -0.000024 | False |
| MELD | visual | 0.050000 | dropout | all_retained | js | 137 | 66 | -0.000000 | False |
| MELD | visual | 0.050000 | dropout | all_retained | flip_rate | 137 | 66 | 0.003650 | False |
| MELD | visual | 0.050000 | dropout | valid_visual | mu | 120 | 58 | 0.001106 | False |
| MELD | visual | 0.050000 | dropout | valid_visual | delta | 120 | 58 | 0.000003 | False |
| MELD | visual | 0.050000 | dropout | valid_visual | js | 120 | 58 | -0.000000 | False |
| MELD | visual | 0.050000 | dropout | valid_visual | flip_rate | 120 | 58 | 0.004167 | False |
| MELD | visual | 0.050000 | gaussian | all_retained | mu | 137 | 66 | 0.000000 | False |
| MELD | visual | 0.050000 | gaussian | all_retained | delta | 137 | 66 | 0.000006 | False |
| MELD | visual | 0.050000 | gaussian | all_retained | js | 137 | 66 | 0.000000 | False |
| MELD | visual | 0.050000 | gaussian | all_retained | flip_rate | 137 | 66 | 0.000000 | False |
| MELD | visual | 0.050000 | gaussian | valid_visual | mu | 120 | 58 | 0.000000 | False |
| MELD | visual | 0.050000 | gaussian | valid_visual | delta | 120 | 58 | 0.000011 | False |
| MELD | visual | 0.050000 | gaussian | valid_visual | js | 120 | 58 | 0.000000 | False |
| MELD | visual | 0.050000 | gaussian | valid_visual | flip_rate | 120 | 58 | 0.000000 | False |
| MELD | visual | 0.100000 | dropout | all_retained | mu | 137 | 66 | 0.001559 | False |
| MELD | visual | 0.100000 | dropout | all_retained | delta | 137 | 66 | -0.000044 | False |
| MELD | visual | 0.100000 | dropout | all_retained | js | 137 | 66 | -0.000001 | False |
| MELD | visual | 0.100000 | dropout | all_retained | flip_rate | 137 | 66 | 0.005839 | False |
| MELD | visual | 0.100000 | dropout | valid_visual | mu | 120 | 58 | 0.001780 | False |
| MELD | visual | 0.100000 | dropout | valid_visual | delta | 120 | 58 | -0.000002 | False |
| MELD | visual | 0.100000 | dropout | valid_visual | js | 120 | 58 | -0.000001 | False |
| MELD | visual | 0.100000 | dropout | valid_visual | flip_rate | 120 | 58 | 0.006667 | False |
| MELD | visual | 0.100000 | gaussian | all_retained | mu | 137 | 66 | 0.000000 | False |
| MELD | visual | 0.100000 | gaussian | all_retained | delta | 137 | 66 | -0.000002 | False |
| MELD | visual | 0.100000 | gaussian | all_retained | js | 137 | 66 | -0.000000 | False |
| MELD | visual | 0.100000 | gaussian | all_retained | flip_rate | 137 | 66 | 0.000000 | False |
| MELD | visual | 0.100000 | gaussian | valid_visual | mu | 120 | 58 | 0.000000 | False |
| MELD | visual | 0.100000 | gaussian | valid_visual | delta | 120 | 58 | 0.000005 | False |
| MELD | visual | 0.100000 | gaussian | valid_visual | js | 120 | 58 | 0.000000 | False |
| MELD | visual | 0.100000 | gaussian | valid_visual | flip_rate | 120 | 58 | 0.000000 | False |

在原始置信度绝对差≤0.02内，用Hungarian算法先最大化配对数、再最小化总置信度距离；配对选择只依赖原始正确性、置信度与身份，不使用行为响应。每个配对条件内，每条原始话语最多使用一次。不同强度使用相同配对；valid_visual子集单独按同一规则配对。少于10对或3个Dialogue标为支持不足。配对记录会跨强度、干扰和模态复用，不能把CSV总行数当作独立样本数。完整逐对值见matched_correct_wrong.csv；固定置信度分箱结果见conditional_diagnostic.csv。

## 同一OOF协议下的所有校准结果

| dataset | modality | strength | method | ece15 | binary_brier | binary_nll |
|---|---|---|---|---|---|---|
| IEMOCAP | text | 0.050000 | E0 | 0.104967 | 0.222303 | 0.719513 |
| IEMOCAP | text | 0.050000 | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | text | 0.050000 | M_joint | 0.080723 | 0.222036 | 0.638888 |
| IEMOCAP | text | 0.050000 | M_gaussian | 0.070949 | 0.220315 | 0.634945 |
| IEMOCAP | text | 0.050000 | M_dropout | 0.095112 | 0.221352 | 0.637351 |
| IEMOCAP | audio | 0.050000 | E0 | 0.104967 | 0.222303 | 0.719513 |
| IEMOCAP | audio | 0.050000 | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | audio | 0.050000 | M_joint | 0.088840 | 0.223429 | 0.642747 |
| IEMOCAP | audio | 0.050000 | M_gaussian | 0.093197 | 0.221404 | 0.637455 |
| IEMOCAP | audio | 0.050000 | M_dropout | 0.082178 | 0.223648 | 0.643465 |
| IEMOCAP | visual | 0.050000 | E0 | 0.104967 | 0.222303 | 0.719513 |
| IEMOCAP | visual | 0.050000 | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | visual | 0.050000 | M_joint | 0.070781 | 0.222937 | 0.640842 |
| IEMOCAP | visual | 0.050000 | M_gaussian | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | visual | 0.050000 | M_dropout | 0.057994 | 0.223125 | 0.641602 |
| IEMOCAP | joint | 0.050000 | E0 | 0.104967 | 0.222303 | 0.719513 |
| IEMOCAP | joint | 0.050000 | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | joint | 0.050000 | M_joint | 0.085903 | 0.223983 | 0.645440 |
| IEMOCAP | joint | 0.050000 | M_gaussian | 0.093274 | 0.221499 | 0.637647 |
| IEMOCAP | joint | 0.050000 | M_dropout | 0.085152 | 0.224293 | 0.645723 |
| MELD | text | 0.050000 | E0 | 0.201363 | 0.233420 | 0.659452 |
| MELD | text | 0.050000 | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | text | 0.050000 | M_joint | 0.042674 | 0.193197 | 0.569222 |
| MELD | text | 0.050000 | M_gaussian | 0.047061 | 0.191952 | 0.566552 |
| MELD | text | 0.050000 | M_dropout | 0.046241 | 0.193150 | 0.569359 |
| MELD | audio | 0.050000 | E0 | 0.201363 | 0.233420 | 0.659452 |
| MELD | audio | 0.050000 | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | audio | 0.050000 | M_joint | 0.043731 | 0.194260 | 0.573347 |
| MELD | audio | 0.050000 | M_gaussian | 0.039572 | 0.192479 | 0.567729 |
| MELD | audio | 0.050000 | M_dropout | 0.043818 | 0.194519 | 0.573969 |
| MELD | visual | 0.050000 | E0 | 0.201363 | 0.233420 | 0.659452 |
| MELD | visual | 0.050000 | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | visual | 0.050000 | M_joint | 0.042415 | 0.192715 | 0.568501 |
| MELD | visual | 0.050000 | M_gaussian | 0.040026 | 0.192628 | 0.568409 |
| MELD | visual | 0.050000 | M_dropout | 0.046286 | 0.192370 | 0.567720 |
| MELD | joint | 0.050000 | E0 | 0.201363 | 0.233420 | 0.659452 |
| MELD | joint | 0.050000 | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | joint | 0.050000 | M_joint | 0.049019 | 0.194164 | 0.571276 |
| MELD | joint | 0.050000 | M_gaussian | 0.048966 | 0.192093 | 0.566725 |
| MELD | joint | 0.050000 | M_dropout | 0.045398 | 0.193117 | 0.569344 |
| IEMOCAP | text | 0.100000 | E0 | 0.104967 | 0.222303 | 0.719513 |
| IEMOCAP | text | 0.100000 | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | text | 0.100000 | M_joint | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | text | 0.100000 | M_gaussian | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | text | 0.100000 | M_dropout | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | audio | 0.100000 | E0 | 0.104967 | 0.222303 | 0.719513 |
| IEMOCAP | audio | 0.100000 | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | audio | 0.100000 | M_joint | 0.090209 | 0.221757 | 0.638177 |
| IEMOCAP | audio | 0.100000 | M_gaussian | 0.091630 | 0.222158 | 0.639349 |
| IEMOCAP | audio | 0.100000 | M_dropout | 0.085909 | 0.221429 | 0.637506 |
| IEMOCAP | visual | 0.100000 | E0 | 0.104967 | 0.222303 | 0.719513 |
| IEMOCAP | visual | 0.100000 | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | visual | 0.100000 | M_joint | 0.074486 | 0.223403 | 0.642616 |
| IEMOCAP | visual | 0.100000 | M_gaussian | 0.086764 | 0.221335 | 0.637411 |
| IEMOCAP | visual | 0.100000 | M_dropout | 0.079490 | 0.223040 | 0.641412 |
| IEMOCAP | joint | 0.100000 | E0 | 0.104967 | 0.222303 | 0.719513 |
| IEMOCAP | joint | 0.100000 | E1 | 0.089773 | 0.221210 | 0.637064 |
| IEMOCAP | joint | 0.100000 | M_joint | 0.089443 | 0.221490 | 0.637631 |
| IEMOCAP | joint | 0.100000 | M_gaussian | 0.091039 | 0.221968 | 0.638841 |
| IEMOCAP | joint | 0.100000 | M_dropout | 0.089773 | 0.221210 | 0.637064 |
| MELD | text | 0.100000 | E0 | 0.201363 | 0.233420 | 0.659452 |
| MELD | text | 0.100000 | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | text | 0.100000 | M_joint | 0.040773 | 0.192899 | 0.568472 |
| MELD | text | 0.100000 | M_gaussian | 0.047738 | 0.192952 | 0.569015 |
| MELD | text | 0.100000 | M_dropout | 0.044679 | 0.193201 | 0.569398 |
| MELD | audio | 0.100000 | E0 | 0.201363 | 0.233420 | 0.659452 |
| MELD | audio | 0.100000 | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | audio | 0.100000 | M_joint | 0.042622 | 0.193126 | 0.569158 |
| MELD | audio | 0.100000 | M_gaussian | 0.044667 | 0.192547 | 0.568016 |
| MELD | audio | 0.100000 | M_dropout | 0.043661 | 0.193657 | 0.570534 |
| MELD | visual | 0.100000 | E0 | 0.201363 | 0.233420 | 0.659452 |
| MELD | visual | 0.100000 | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | visual | 0.100000 | M_joint | 0.049078 | 0.192277 | 0.567279 |
| MELD | visual | 0.100000 | M_gaussian | 0.045048 | 0.192807 | 0.568671 |
| MELD | visual | 0.100000 | M_dropout | 0.044279 | 0.191956 | 0.566570 |
| MELD | joint | 0.100000 | E0 | 0.201363 | 0.233420 | 0.659452 |
| MELD | joint | 0.100000 | E1 | 0.043230 | 0.192764 | 0.568583 |
| MELD | joint | 0.100000 | M_joint | 0.044513 | 0.193119 | 0.569348 |
| MELD | joint | 0.100000 | M_gaussian | 0.045950 | 0.193488 | 0.570157 |
| MELD | joint | 0.100000 | M_dropout | 0.042321 | 0.192842 | 0.568752 |

E0为原始置信度，E1仅拟合原始C；M_joint联合使用C及Gaussian/Dropout各自μ/δ四维，M_gaussian/M_dropout各自使用C与对应μ/δ。没有JS输入或交互项。复用历史17个外层Dialogue折，内层GroupKFold5按合并留出Brier选择off/1/0.1/0.01；仅行为特征标准化，每次拟合只使用对应训练Dialogue。off精确退回同训练子集的E1。完整内外模型、训练ID、标准化器与系数见fold_parameters/。

## 配对Dialogue Bootstrap校准差值

| dataset | modality | comparison | candidate | candidate_strength | reference | reference_strength | metric | mean_difference | ci_low | ci_high |
|---|---|---|---|---|---|---|---|---|---|---|
| IEMOCAP | text | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | ece15 | -0.009051 | -0.021411 | 0.006659 |
| IEMOCAP | text | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | binary_brier | 0.000827 | 0.000000 | 0.002446 |
| IEMOCAP | text | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | binary_nll | 0.001824 | 0.000000 | 0.005425 |
| IEMOCAP | text | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | ece15 | -0.018824 | -0.037127 | 0.008656 |
| IEMOCAP | text | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | binary_brier | -0.000895 | -0.002624 | 0.001302 |
| IEMOCAP | text | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | binary_nll | -0.002119 | -0.005969 | 0.002604 |
| IEMOCAP | text | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | ece15 | 0.005338 | -0.000127 | 0.014067 |
| IEMOCAP | text | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | binary_brier | 0.000142 | 0.000000 | 0.000445 |
| IEMOCAP | text | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | binary_nll | 0.000287 | 0.000000 | 0.000898 |
| IEMOCAP | text | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | ece15 | 0.000000 | 0.000000 | 0.000000 |
| IEMOCAP | text | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | binary_brier | 0.000000 | 0.000000 | 0.000000 |
| IEMOCAP | text | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | binary_nll | 0.000000 | 0.000000 | 0.000000 |
| IEMOCAP | text | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | ece15 | 0.000000 | 0.000000 | 0.000000 |
| IEMOCAP | text | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | binary_brier | 0.000000 | 0.000000 | 0.000000 |
| IEMOCAP | text | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | binary_nll | 0.000000 | 0.000000 | 0.000000 |
| IEMOCAP | text | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | ece15 | 0.000000 | 0.000000 | 0.000000 |
| IEMOCAP | text | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | binary_brier | 0.000000 | 0.000000 | 0.000000 |
| IEMOCAP | text | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | binary_nll | 0.000000 | 0.000000 | 0.000000 |
| IEMOCAP | text | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | ece15 | 0.009051 | -0.006659 | 0.021411 |
| IEMOCAP | text | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | binary_brier | -0.000827 | -0.002446 | 0.000000 |
| IEMOCAP | text | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | binary_nll | -0.001824 | -0.005425 | 0.000000 |
| IEMOCAP | text | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | ece15 | 0.018824 | -0.008656 | 0.037127 |
| IEMOCAP | text | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | binary_brier | 0.000895 | -0.001302 | 0.002624 |
| IEMOCAP | text | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | binary_nll | 0.002119 | -0.002604 | 0.005969 |
| IEMOCAP | text | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | ece15 | -0.005338 | -0.014067 | 0.000127 |
| IEMOCAP | text | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | binary_brier | -0.000142 | -0.000445 | 0.000000 |
| IEMOCAP | text | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | binary_nll | -0.000287 | -0.000898 | 0.000000 |
| IEMOCAP | audio | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | ece15 | -0.000934 | -0.006351 | 0.023181 |
| IEMOCAP | audio | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | binary_brier | 0.002220 | 0.000000 | 0.006943 |
| IEMOCAP | audio | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | binary_nll | 0.005683 | 0.000000 | 0.017994 |
| IEMOCAP | audio | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | ece15 | 0.003423 | 0.000000 | 0.008215 |
| IEMOCAP | audio | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | binary_brier | 0.000194 | 0.000000 | 0.000608 |
| IEMOCAP | audio | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | binary_nll | 0.000391 | 0.000000 | 0.001225 |
| IEMOCAP | audio | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | ece15 | -0.007596 | -0.014523 | 0.020683 |
| IEMOCAP | audio | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | binary_brier | 0.002438 | 0.000132 | 0.007000 |
| IEMOCAP | audio | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | binary_nll | 0.006401 | 0.000259 | 0.017676 |
| IEMOCAP | audio | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | ece15 | 0.000435 | -0.004123 | 0.012509 |
| IEMOCAP | audio | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | binary_brier | 0.000547 | 0.000000 | 0.001431 |
| IEMOCAP | audio | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | binary_nll | 0.001113 | 0.000000 | 0.002917 |
| IEMOCAP | audio | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | ece15 | 0.001857 | -0.004183 | 0.007377 |
| IEMOCAP | audio | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | binary_brier | 0.000948 | 0.000000 | 0.002895 |
| IEMOCAP | audio | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | binary_nll | 0.002285 | 0.000000 | 0.006943 |
| IEMOCAP | audio | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | ece15 | -0.003865 | -0.012295 | 0.000704 |
| IEMOCAP | audio | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | binary_brier | 0.000219 | 0.000000 | 0.000793 |
| IEMOCAP | audio | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | binary_nll | 0.000442 | 0.000000 | 0.001600 |
| IEMOCAP | audio | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | ece15 | 0.001369 | -0.014557 | 0.004731 |
| IEMOCAP | audio | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | binary_brier | -0.001672 | -0.005932 | 0.000103 |
| IEMOCAP | audio | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | binary_nll | -0.004570 | -0.015961 | 0.000198 |
| IEMOCAP | audio | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | ece15 | -0.001567 | -0.011551 | 0.007130 |
| IEMOCAP | audio | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | binary_brier | 0.000753 | 0.000000 | 0.002289 |
| IEMOCAP | audio | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | binary_nll | 0.001894 | 0.000000 | 0.005727 |
| IEMOCAP | audio | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | ece15 | 0.003731 | -0.022034 | 0.009365 |
| IEMOCAP | audio | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | binary_brier | -0.002219 | -0.006152 | -0.000132 |
| IEMOCAP | audio | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | binary_nll | -0.005959 | -0.016127 | -0.000259 |
| IEMOCAP | visual | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | ece15 | -0.018993 | -0.035327 | 0.007159 |
| IEMOCAP | visual | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | binary_brier | 0.001727 | 0.000000 | 0.003591 |
| IEMOCAP | visual | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | binary_nll | 0.003778 | 0.000000 | 0.007967 |
| IEMOCAP | visual | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | ece15 | 0.000000 | 0.000000 | 0.000000 |
| IEMOCAP | visual | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | binary_brier | 0.000000 | 0.000000 | 0.000000 |
| IEMOCAP | visual | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | binary_nll | 0.000000 | 0.000000 | 0.000000 |
| IEMOCAP | visual | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | ece15 | -0.031779 | -0.048056 | 0.007984 |
| IEMOCAP | visual | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | binary_brier | 0.001915 | -0.000307 | 0.004766 |
| IEMOCAP | visual | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | binary_nll | 0.004538 | -0.000477 | 0.010965 |
| IEMOCAP | visual | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | ece15 | -0.015287 | -0.031138 | 0.014716 |
| IEMOCAP | visual | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | binary_brier | 0.002193 | -0.000278 | 0.005056 |
| IEMOCAP | visual | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | binary_nll | 0.005552 | -0.000977 | 0.012814 |
| IEMOCAP | visual | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | ece15 | -0.003010 | -0.006444 | 0.000481 |
| IEMOCAP | visual | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | binary_brier | 0.000125 | 0.000000 | 0.000293 |
| IEMOCAP | visual | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | binary_nll | 0.000347 | 0.000000 | 0.000855 |
| IEMOCAP | visual | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | ece15 | -0.010283 | -0.031804 | 0.014659 |
| IEMOCAP | visual | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | binary_brier | 0.001830 | -0.001234 | 0.004976 |
| IEMOCAP | visual | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | binary_nll | 0.004348 | -0.003709 | 0.012287 |
| IEMOCAP | visual | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | ece15 | 0.003706 | -0.025117 | 0.033025 |
| IEMOCAP | visual | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | binary_brier | 0.000466 | -0.001307 | 0.002021 |
| IEMOCAP | visual | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | binary_nll | 0.001774 | -0.002798 | 0.006023 |
| IEMOCAP | visual | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | ece15 | -0.003010 | -0.006444 | 0.000481 |
| IEMOCAP | visual | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | binary_brier | 0.000125 | 0.000000 | 0.000293 |
| IEMOCAP | visual | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | binary_nll | 0.000347 | 0.000000 | 0.000855 |
| IEMOCAP | visual | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | ece15 | 0.021496 | -0.014807 | 0.046184 |
| IEMOCAP | visual | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | binary_brier | -0.000086 | -0.001522 | 0.001341 |
| IEMOCAP | visual | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | binary_nll | -0.000191 | -0.004410 | 0.003358 |
| IEMOCAP | joint | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | ece15 | -0.003871 | -0.011234 | 0.017091 |
| IEMOCAP | joint | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | binary_brier | 0.002774 | 0.000000 | 0.008968 |
| IEMOCAP | joint | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | binary_nll | 0.008376 | 0.000000 | 0.028048 |
| IEMOCAP | joint | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | ece15 | 0.003501 | 0.000000 | 0.008428 |
| IEMOCAP | joint | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | binary_brier | 0.000289 | 0.000000 | 0.000904 |
| IEMOCAP | joint | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | binary_nll | 0.000583 | 0.000000 | 0.001826 |
| IEMOCAP | joint | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | ece15 | -0.004622 | -0.009262 | 0.009097 |
| IEMOCAP | joint | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | binary_brier | 0.003083 | 0.000000 | 0.009086 |
| IEMOCAP | joint | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | binary_nll | 0.008659 | 0.000000 | 0.026672 |
| IEMOCAP | joint | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | ece15 | -0.000331 | -0.003378 | 0.013054 |
| IEMOCAP | joint | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | binary_brier | 0.000281 | 0.000000 | 0.000878 |
| IEMOCAP | joint | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | binary_nll | 0.000567 | 0.000000 | 0.001774 |
| IEMOCAP | joint | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | ece15 | 0.001266 | -0.005688 | 0.007293 |
| IEMOCAP | joint | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | binary_brier | 0.000758 | 0.000000 | 0.002309 |
| IEMOCAP | joint | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | binary_nll | 0.001777 | 0.000000 | 0.005378 |
| IEMOCAP | joint | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | ece15 | 0.000000 | 0.000000 | 0.000000 |
| IEMOCAP | joint | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | binary_brier | 0.000000 | 0.000000 | 0.000000 |
| IEMOCAP | joint | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | binary_nll | 0.000000 | 0.000000 | 0.000000 |
| IEMOCAP | joint | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | ece15 | 0.003540 | -0.005969 | 0.010229 |
| IEMOCAP | joint | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | binary_brier | -0.002493 | -0.008648 | 0.000000 |
| IEMOCAP | joint | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | binary_nll | -0.007809 | -0.027550 | 0.000000 |
| IEMOCAP | joint | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | ece15 | -0.002235 | -0.012988 | 0.007089 |
| IEMOCAP | joint | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | binary_brier | 0.000469 | 0.000000 | 0.001409 |
| IEMOCAP | joint | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | binary_nll | 0.001194 | 0.000000 | 0.003545 |
| IEMOCAP | joint | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | ece15 | 0.004622 | -0.009097 | 0.009262 |
| IEMOCAP | joint | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | binary_brier | -0.003083 | -0.009086 | 0.000000 |
| IEMOCAP | joint | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | binary_nll | -0.008659 | -0.026672 | 0.000000 |
| MELD | text | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | ece15 | -0.000556 | -0.015341 | 0.016003 |
| MELD | text | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | binary_brier | 0.000433 | -0.000876 | 0.001759 |
| MELD | text | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | binary_nll | 0.000639 | -0.002156 | 0.003537 |
| MELD | text | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | ece15 | 0.003831 | -0.013512 | 0.011327 |
| MELD | text | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | binary_brier | -0.000812 | -0.001994 | 0.000284 |
| MELD | text | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | binary_nll | -0.002032 | -0.004699 | 0.000479 |
| MELD | text | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | ece15 | 0.003011 | -0.004710 | 0.009540 |
| MELD | text | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | binary_brier | 0.000386 | -0.000262 | 0.001228 |
| MELD | text | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | binary_nll | 0.000775 | -0.000700 | 0.002636 |
| MELD | text | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | ece15 | -0.002457 | -0.021421 | 0.008065 |
| MELD | text | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | binary_brier | 0.000135 | -0.001372 | 0.001647 |
| MELD | text | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | binary_nll | -0.000111 | -0.003568 | 0.003323 |
| MELD | text | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | ece15 | 0.004508 | -0.010453 | 0.011756 |
| MELD | text | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | binary_brier | 0.000188 | -0.001043 | 0.001564 |
| MELD | text | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | binary_nll | 0.000431 | -0.002711 | 0.003987 |
| MELD | text | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | ece15 | 0.001449 | -0.008807 | 0.011934 |
| MELD | text | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | binary_brier | 0.000437 | -0.000418 | 0.001319 |
| MELD | text | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | binary_nll | 0.000814 | -0.001097 | 0.002737 |
| MELD | text | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | ece15 | -0.001901 | -0.021903 | 0.009277 |
| MELD | text | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | binary_brier | -0.000298 | -0.002064 | 0.001264 |
| MELD | text | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | binary_nll | -0.000750 | -0.004709 | 0.002705 |
| MELD | text | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | ece15 | 0.000677 | -0.011319 | 0.013526 |
| MELD | text | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | binary_brier | 0.001000 | -0.000186 | 0.002263 |
| MELD | text | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | binary_nll | 0.002463 | -0.000410 | 0.005756 |
| MELD | text | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | ece15 | -0.001562 | -0.009873 | 0.008611 |
| MELD | text | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | binary_brier | 0.000051 | -0.000545 | 0.000638 |
| MELD | text | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | binary_nll | 0.000039 | -0.001255 | 0.001274 |
| MELD | audio | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | ece15 | 0.000501 | -0.016688 | 0.014334 |
| MELD | audio | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | binary_brier | 0.001496 | -0.000730 | 0.004201 |
| MELD | audio | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | binary_nll | 0.004763 | -0.001956 | 0.013868 |
| MELD | audio | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | ece15 | -0.003658 | -0.013961 | 0.011999 |
| MELD | audio | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | binary_brier | -0.000286 | -0.001630 | 0.001061 |
| MELD | audio | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | binary_nll | -0.000854 | -0.004086 | 0.002405 |
| MELD | audio | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | ece15 | 0.000588 | -0.013906 | 0.015436 |
| MELD | audio | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | binary_brier | 0.001755 | -0.000201 | 0.004185 |
| MELD | audio | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | binary_nll | 0.005385 | -0.000564 | 0.013758 |
| MELD | audio | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | ece15 | -0.000608 | -0.017243 | 0.010380 |
| MELD | audio | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | binary_brier | 0.000362 | -0.001143 | 0.001849 |
| MELD | audio | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | binary_nll | 0.000575 | -0.002900 | 0.003921 |
| MELD | audio | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | ece15 | 0.001437 | -0.009321 | 0.007141 |
| MELD | audio | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | binary_brier | -0.000218 | -0.000727 | 0.000216 |
| MELD | audio | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | binary_nll | -0.000568 | -0.001786 | 0.000496 |
| MELD | audio | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | ece15 | 0.000431 | -0.016977 | 0.011016 |
| MELD | audio | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | binary_brier | 0.000893 | -0.000427 | 0.002202 |
| MELD | audio | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | binary_nll | 0.001951 | -0.000776 | 0.004809 |
| MELD | audio | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | ece15 | -0.001109 | -0.015948 | 0.011698 |
| MELD | audio | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | binary_brier | -0.001134 | -0.003569 | 0.000930 |
| MELD | audio | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | binary_nll | -0.004188 | -0.012997 | 0.002299 |
| MELD | audio | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | ece15 | 0.005096 | -0.012929 | 0.012965 |
| MELD | audio | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | binary_brier | 0.000068 | -0.001268 | 0.001399 |
| MELD | audio | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | binary_nll | 0.000286 | -0.002850 | 0.003509 |
| MELD | audio | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | ece15 | -0.000157 | -0.018409 | 0.010897 |
| MELD | audio | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | binary_brier | -0.000862 | -0.002783 | 0.000725 |
| MELD | audio | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | binary_nll | -0.003435 | -0.011066 | 0.001759 |
| MELD | visual | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | ece15 | -0.000815 | -0.016262 | 0.007634 |
| MELD | visual | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | binary_brier | -0.000049 | -0.001170 | 0.000985 |
| MELD | visual | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | binary_nll | -0.000082 | -0.002818 | 0.002266 |
| MELD | visual | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | ece15 | -0.003204 | -0.019033 | 0.008192 |
| MELD | visual | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | binary_brier | -0.000136 | -0.001479 | 0.001192 |
| MELD | visual | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | binary_nll | -0.000175 | -0.003377 | 0.002906 |
| MELD | visual | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | ece15 | 0.003056 | -0.013274 | 0.008615 |
| MELD | visual | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | binary_brier | -0.000394 | -0.001240 | 0.000366 |
| MELD | visual | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | binary_nll | -0.000864 | -0.002729 | 0.000808 |
| MELD | visual | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | ece15 | 0.005848 | -0.014586 | 0.012534 |
| MELD | visual | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | binary_brier | -0.000487 | -0.001308 | 0.000270 |
| MELD | visual | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | binary_nll | -0.001304 | -0.003246 | 0.000472 |
| MELD | visual | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | ece15 | 0.001818 | -0.007217 | 0.004336 |
| MELD | visual | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | binary_brier | 0.000043 | -0.000082 | 0.000173 |
| MELD | visual | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | binary_nll | 0.000087 | -0.000181 | 0.000361 |
| MELD | visual | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | ece15 | 0.001049 | -0.015647 | 0.012813 |
| MELD | visual | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | binary_brier | -0.000808 | -0.001756 | 0.000132 |
| MELD | visual | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | binary_nll | -0.002013 | -0.004173 | 0.000163 |
| MELD | visual | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | ece15 | 0.006664 | -0.009253 | 0.015824 |
| MELD | visual | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | binary_brier | -0.000438 | -0.001421 | 0.000433 |
| MELD | visual | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | binary_nll | -0.001222 | -0.003460 | 0.000793 |
| MELD | visual | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | ece15 | 0.005022 | -0.008329 | 0.016974 |
| MELD | visual | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | binary_brier | 0.000179 | -0.001156 | 0.001534 |
| MELD | visual | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | binary_nll | 0.000262 | -0.002770 | 0.003453 |
| MELD | visual | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | ece15 | -0.002007 | -0.009864 | 0.012072 |
| MELD | visual | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | binary_brier | -0.000415 | -0.001064 | 0.000216 |
| MELD | visual | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | binary_nll | -0.001150 | -0.002668 | 0.000316 |
| MELD | joint | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | ece15 | 0.005789 | -0.014175 | 0.014815 |
| MELD | joint | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | binary_brier | 0.001400 | -0.000184 | 0.003114 |
| MELD | joint | behavior_minus_E1 | M_joint | 0.050000 | E1 | 0.050000 | binary_nll | 0.002693 | -0.000665 | 0.006436 |
| MELD | joint | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | ece15 | 0.005736 | -0.012844 | 0.013433 |
| MELD | joint | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | binary_brier | -0.000671 | -0.001873 | 0.000412 |
| MELD | joint | behavior_minus_E1 | M_gaussian | 0.050000 | E1 | 0.050000 | binary_nll | -0.001859 | -0.004526 | 0.000579 |
| MELD | joint | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | ece15 | 0.002168 | -0.003226 | 0.010270 |
| MELD | joint | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | binary_brier | 0.000353 | -0.000186 | 0.001105 |
| MELD | joint | behavior_minus_E1 | M_dropout | 0.050000 | E1 | 0.050000 | binary_nll | 0.000761 | -0.000399 | 0.002339 |
| MELD | joint | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | ece15 | 0.001283 | -0.006839 | 0.009996 |
| MELD | joint | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | binary_brier | 0.000355 | -0.000481 | 0.001299 |
| MELD | joint | behavior_minus_E1 | M_joint | 0.100000 | E1 | 0.100000 | binary_nll | 0.000764 | -0.001064 | 0.002837 |
| MELD | joint | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | ece15 | 0.002720 | -0.008904 | 0.010131 |
| MELD | joint | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | binary_brier | 0.000724 | -0.000268 | 0.001926 |
| MELD | joint | behavior_minus_E1 | M_gaussian | 0.100000 | E1 | 0.100000 | binary_nll | 0.001574 | -0.000804 | 0.004485 |
| MELD | joint | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | ece15 | -0.000909 | -0.004898 | 0.008346 |
| MELD | joint | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | binary_brier | 0.000078 | -0.000577 | 0.000785 |
| MELD | joint | behavior_minus_E1 | M_dropout | 0.100000 | E1 | 0.100000 | binary_nll | 0.000168 | -0.001206 | 0.001656 |
| MELD | joint | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | ece15 | -0.004506 | -0.014551 | 0.015582 |
| MELD | joint | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | binary_brier | -0.001045 | -0.002865 | 0.000612 |
| MELD | joint | strength_0.10_minus_0.05 | M_joint | 0.100000 | M_joint | 0.050000 | binary_nll | -0.001928 | -0.005846 | 0.001671 |
| MELD | joint | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | ece15 | -0.003016 | -0.012264 | 0.013047 |
| MELD | joint | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | binary_brier | 0.001395 | 0.000131 | 0.002820 |
| MELD | joint | strength_0.10_minus_0.05 | M_gaussian | 0.100000 | M_gaussian | 0.050000 | binary_nll | 0.003432 | 0.000555 | 0.006814 |
| MELD | joint | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | ece15 | -0.003077 | -0.007766 | 0.004901 |
| MELD | joint | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | binary_brier | -0.000276 | -0.000665 | 0.000079 |
| MELD | joint | strength_0.10_minus_0.05 | M_dropout | 0.100000 | M_dropout | 0.050000 | binary_nll | -0.000592 | -0.001420 | 0.000154 |

差值方向为candidate−reference；ECE15、Brier、NLL负值表示改善。复用历史2000次Dialogue抽样，对所有条件共享同一组抽样；不重新拟合校准器。全部区间均为单Seed探索性区间，未校正多重比较，不包含模型Seed与拟合不确定性；将10次重复的行为汇总视为固定量，没有另外重抽扰动repeat来量化测量不确定性。

## 强度变化与响应差值区间

| dataset | modality | corruption | population | comparison | measure | mean_difference | ci_low | ci_high | valid_resamples |
|---|---|---|---|---|---|---|---|---|---|
| IEMOCAP | audio | dropout | all_retained | strength_0.10_minus_0.05 | mu | 0.005604 | 0.002478 | 0.009382 | 2000 |
| IEMOCAP | audio | dropout | all_retained | strength_0.10_minus_0.05 | delta | 0.005115 | 0.003876 | 0.006134 | 2000 |
| IEMOCAP | audio | dropout | all_retained | strength_0.10_minus_0.05 | js | 0.000412 | 0.000280 | 0.000624 | 2000 |
| IEMOCAP | audio | dropout | all_retained | strength_0.10_minus_0.05 | flip_rate | 0.012524 | 0.005724 | 0.020686 | 2000 |
| IEMOCAP | audio | dropout | all_retained | strength_0.10_minus_0.05 | mu_zero_fraction | -0.013283 | -0.031713 | 0.000000 | 2000 |
| IEMOCAP | audio | dropout | all_retained | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.013283 | 0.000000 | 0.031713 | 2000 |
| IEMOCAP | audio | dropout | all_retained | matched_wrong_minus_correct | mu | 0.002493 | -0.002759 | 0.009109 | 2000 |
| IEMOCAP | audio | dropout | all_retained | matched_wrong_minus_correct | mu | 0.007680 | -0.000707 | 0.020241 | 2000 |
| IEMOCAP | audio | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_mu | 0.005186 | -0.000545 | 0.012856 | 2000 |
| IEMOCAP | audio | dropout | all_retained | matched_wrong_minus_correct | delta | 0.001525 | -0.000155 | 0.003501 | 2000 |
| IEMOCAP | audio | dropout | all_retained | matched_wrong_minus_correct | delta | 0.000835 | -0.001651 | 0.003789 | 2000 |
| IEMOCAP | audio | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000690 | -0.002654 | 0.001440 | 2000 |
| IEMOCAP | audio | dropout | all_retained | matched_wrong_minus_correct | js | 0.000105 | 0.000023 | 0.000234 | 2000 |
| IEMOCAP | audio | dropout | all_retained | matched_wrong_minus_correct | js | 0.000173 | -0.000009 | 0.000438 | 2000 |
| IEMOCAP | audio | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_js | 0.000068 | -0.000058 | 0.000239 | 2000 |
| IEMOCAP | audio | dropout | all_retained | matched_wrong_minus_correct | flip_rate | 0.005263 | -0.006141 | 0.019824 | 2000 |
| IEMOCAP | audio | dropout | all_retained | matched_wrong_minus_correct | flip_rate | 0.015789 | -0.002223 | 0.042248 | 2000 |
| IEMOCAP | audio | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | 0.010526 | -0.001316 | 0.026088 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | strength_0.10_minus_0.05 | mu | 0.000778 | 0.000241 | 0.001424 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | strength_0.10_minus_0.05 | delta | 0.000703 | 0.000594 | 0.000860 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | strength_0.10_minus_0.05 | js | 0.000046 | 0.000019 | 0.000100 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | strength_0.10_minus_0.05 | flip_rate | 0.001708 | 0.000557 | 0.003083 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | strength_0.10_minus_0.05 | mu_zero_fraction | -0.005693 | -0.012078 | 0.000000 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.005693 | 0.000000 | 0.012078 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | matched_wrong_minus_correct | mu | -0.001891 | -0.004647 | 0.000000 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | matched_wrong_minus_correct | mu | -0.001872 | -0.006120 | 0.001159 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_mu | 0.000019 | -0.001898 | 0.002380 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | matched_wrong_minus_correct | delta | -0.000101 | -0.000272 | 0.000058 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | matched_wrong_minus_correct | delta | -0.000107 | -0.000347 | 0.000141 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000006 | -0.000241 | 0.000263 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | matched_wrong_minus_correct | js | -0.000001 | -0.000005 | 0.000002 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | matched_wrong_minus_correct | js | 0.000002 | -0.000011 | 0.000022 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_js | 0.000003 | -0.000007 | 0.000022 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | matched_wrong_minus_correct | flip_rate | -0.004386 | -0.010828 | 0.000000 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | matched_wrong_minus_correct | flip_rate | -0.004386 | -0.014286 | 0.002655 | 2000 |
| IEMOCAP | audio | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | 0.000000 | -0.004380 | 0.005357 | 2000 |
| IEMOCAP | joint | dropout | all_retained | strength_0.10_minus_0.05 | mu | 0.005906 | 0.002808 | 0.009698 | 2000 |
| IEMOCAP | joint | dropout | all_retained | strength_0.10_minus_0.05 | delta | 0.005623 | 0.004395 | 0.006661 | 2000 |
| IEMOCAP | joint | dropout | all_retained | strength_0.10_minus_0.05 | js | 0.000436 | 0.000297 | 0.000650 | 2000 |
| IEMOCAP | joint | dropout | all_retained | strength_0.10_minus_0.05 | flip_rate | 0.012903 | 0.006335 | 0.021011 | 2000 |
| IEMOCAP | joint | dropout | all_retained | strength_0.10_minus_0.05 | mu_zero_fraction | -0.028463 | -0.051441 | -0.009469 | 2000 |
| IEMOCAP | joint | dropout | all_retained | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.028463 | 0.009469 | 0.051441 | 2000 |
| IEMOCAP | joint | dropout | all_retained | matched_wrong_minus_correct | mu | 0.005008 | -0.001645 | 0.012901 | 2000 |
| IEMOCAP | joint | dropout | all_retained | matched_wrong_minus_correct | mu | 0.006605 | -0.003609 | 0.020421 | 2000 |
| IEMOCAP | joint | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_mu | 0.001598 | -0.002505 | 0.007716 | 2000 |
| IEMOCAP | joint | dropout | all_retained | matched_wrong_minus_correct | delta | 0.001728 | -0.000064 | 0.003729 | 2000 |
| IEMOCAP | joint | dropout | all_retained | matched_wrong_minus_correct | delta | 0.001059 | -0.001848 | 0.004361 | 2000 |
| IEMOCAP | joint | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000670 | -0.002931 | 0.001814 | 2000 |
| IEMOCAP | joint | dropout | all_retained | matched_wrong_minus_correct | js | 0.000105 | 0.000024 | 0.000232 | 2000 |
| IEMOCAP | joint | dropout | all_retained | matched_wrong_minus_correct | js | 0.000166 | -0.000006 | 0.000405 | 2000 |
| IEMOCAP | joint | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_js | 0.000060 | -0.000054 | 0.000215 | 2000 |
| IEMOCAP | joint | dropout | all_retained | matched_wrong_minus_correct | flip_rate | 0.010526 | -0.004202 | 0.028248 | 2000 |
| IEMOCAP | joint | dropout | all_retained | matched_wrong_minus_correct | flip_rate | 0.013158 | -0.008759 | 0.042275 | 2000 |
| IEMOCAP | joint | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | 0.002632 | -0.005839 | 0.014608 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | strength_0.10_minus_0.05 | mu | 0.005918 | 0.002813 | 0.009716 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | strength_0.10_minus_0.05 | delta | 0.005612 | 0.004378 | 0.006659 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | strength_0.10_minus_0.05 | js | 0.000436 | 0.000296 | 0.000651 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | strength_0.10_minus_0.05 | flip_rate | 0.012928 | 0.006354 | 0.021011 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | strength_0.10_minus_0.05 | mu_zero_fraction | -0.028517 | -0.051441 | -0.009500 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.028517 | 0.009500 | 0.051441 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | matched_wrong_minus_correct | mu | 0.005052 | -0.001648 | 0.013088 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | matched_wrong_minus_correct | mu | 0.006664 | -0.003609 | 0.020509 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_mu | 0.001612 | -0.002536 | 0.007840 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | matched_wrong_minus_correct | delta | 0.001710 | -0.000124 | 0.003729 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | matched_wrong_minus_correct | delta | 0.000987 | -0.001912 | 0.004325 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000723 | -0.002931 | 0.001754 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | matched_wrong_minus_correct | js | 0.000106 | 0.000024 | 0.000232 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | matched_wrong_minus_correct | js | 0.000165 | -0.000008 | 0.000405 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_js | 0.000059 | -0.000056 | 0.000214 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | matched_wrong_minus_correct | flip_rate | 0.010619 | -0.004238 | 0.028409 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | matched_wrong_minus_correct | flip_rate | 0.013274 | -0.008851 | 0.042861 | 2000 |
| IEMOCAP | joint | dropout | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | 0.002655 | -0.005932 | 0.014773 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | strength_0.10_minus_0.05 | mu | 0.000744 | 0.000252 | 0.001336 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | strength_0.10_minus_0.05 | delta | 0.000704 | 0.000587 | 0.000871 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | strength_0.10_minus_0.05 | js | 0.000047 | 0.000019 | 0.000101 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | strength_0.10_minus_0.05 | flip_rate | 0.001708 | 0.000552 | 0.003068 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | strength_0.10_minus_0.05 | mu_zero_fraction | -0.007590 | -0.016868 | 0.000000 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.007590 | 0.000000 | 0.016868 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | matched_wrong_minus_correct | mu | -0.001868 | -0.005282 | 0.000000 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | matched_wrong_minus_correct | mu | -0.002252 | -0.006363 | 0.000000 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_mu | -0.000384 | -0.001081 | 0.000000 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | matched_wrong_minus_correct | delta | -0.000123 | -0.000297 | 0.000016 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | matched_wrong_minus_correct | delta | -0.000126 | -0.000313 | 0.000066 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000003 | -0.000191 | 0.000238 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | matched_wrong_minus_correct | js | -0.000001 | -0.000004 | 0.000002 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | matched_wrong_minus_correct | js | 0.000002 | -0.000010 | 0.000022 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_js | 0.000003 | -0.000007 | 0.000021 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | matched_wrong_minus_correct | flip_rate | -0.004386 | -0.012399 | 0.000000 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | matched_wrong_minus_correct | flip_rate | -0.005263 | -0.014879 | 0.000000 | 2000 |
| IEMOCAP | joint | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | -0.000877 | -0.002480 | 0.000000 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | strength_0.10_minus_0.05 | mu | 0.000746 | 0.000254 | 0.001336 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | strength_0.10_minus_0.05 | delta | 0.000704 | 0.000587 | 0.000872 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | strength_0.10_minus_0.05 | js | 0.000047 | 0.000019 | 0.000101 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | strength_0.10_minus_0.05 | flip_rate | 0.001711 | 0.000554 | 0.003068 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | strength_0.10_minus_0.05 | mu_zero_fraction | -0.007605 | -0.016868 | 0.000000 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.007605 | 0.000000 | 0.016868 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | matched_wrong_minus_correct | mu | -0.001885 | -0.005282 | 0.000000 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | matched_wrong_minus_correct | mu | -0.002272 | -0.006363 | 0.000000 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_mu | -0.000387 | -0.001081 | 0.000000 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | matched_wrong_minus_correct | delta | -0.000121 | -0.000296 | 0.000021 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | matched_wrong_minus_correct | delta | -0.000129 | -0.000317 | 0.000065 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000008 | -0.000194 | 0.000234 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | matched_wrong_minus_correct | js | -0.000001 | -0.000004 | 0.000002 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | matched_wrong_minus_correct | js | 0.000002 | -0.000010 | 0.000022 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_js | 0.000003 | -0.000007 | 0.000021 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | matched_wrong_minus_correct | flip_rate | -0.004425 | -0.012399 | 0.000000 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | matched_wrong_minus_correct | flip_rate | -0.005310 | -0.014879 | 0.000000 | 2000 |
| IEMOCAP | joint | gaussian | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | -0.000885 | -0.002480 | 0.000000 | 2000 |
| IEMOCAP | text | dropout | all_retained | strength_0.10_minus_0.05 | mu | 0.002821 | 0.000748 | 0.005286 | 2000 |
| IEMOCAP | text | dropout | all_retained | strength_0.10_minus_0.05 | delta | 0.001471 | 0.001071 | 0.001869 | 2000 |
| IEMOCAP | text | dropout | all_retained | strength_0.10_minus_0.05 | js | 0.000019 | 0.000014 | 0.000026 | 2000 |
| IEMOCAP | text | dropout | all_retained | strength_0.10_minus_0.05 | flip_rate | 0.006262 | 0.001782 | 0.011454 | 2000 |
| IEMOCAP | text | dropout | all_retained | strength_0.10_minus_0.05 | mu_zero_fraction | -0.007590 | -0.013673 | -0.001992 | 2000 |
| IEMOCAP | text | dropout | all_retained | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.007590 | 0.001992 | 0.013673 | 2000 |
| IEMOCAP | text | dropout | all_retained | matched_wrong_minus_correct | mu | 0.000117 | -0.001873 | 0.002473 | 2000 |
| IEMOCAP | text | dropout | all_retained | matched_wrong_minus_correct | mu | -0.003044 | -0.010472 | 0.002849 | 2000 |
| IEMOCAP | text | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_mu | -0.003162 | -0.010227 | 0.001540 | 2000 |
| IEMOCAP | text | dropout | all_retained | matched_wrong_minus_correct | delta | 0.000132 | -0.000295 | 0.000557 | 2000 |
| IEMOCAP | text | dropout | all_retained | matched_wrong_minus_correct | delta | 0.000220 | -0.000579 | 0.000996 | 2000 |
| IEMOCAP | text | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_delta | 0.000088 | -0.000329 | 0.000499 | 2000 |
| IEMOCAP | text | dropout | all_retained | matched_wrong_minus_correct | js | 0.000001 | -0.000001 | 0.000004 | 2000 |
| IEMOCAP | text | dropout | all_retained | matched_wrong_minus_correct | js | 0.000002 | -0.000006 | 0.000011 | 2000 |
| IEMOCAP | text | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_js | 0.000001 | -0.000005 | 0.000008 | 2000 |
| IEMOCAP | text | dropout | all_retained | matched_wrong_minus_correct | flip_rate | 0.000000 | -0.004412 | 0.004960 | 2000 |
| IEMOCAP | text | dropout | all_retained | matched_wrong_minus_correct | flip_rate | -0.007018 | -0.023482 | 0.006062 | 2000 |
| IEMOCAP | text | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | -0.007018 | -0.022582 | 0.003670 | 2000 |
| IEMOCAP | text | gaussian | all_retained | strength_0.10_minus_0.05 | mu | -0.000403 | -0.001262 | 0.000000 | 2000 |
| IEMOCAP | text | gaussian | all_retained | strength_0.10_minus_0.05 | delta | 0.000106 | 0.000067 | 0.000145 | 2000 |
| IEMOCAP | text | gaussian | all_retained | strength_0.10_minus_0.05 | js | 0.000001 | 0.000001 | 0.000001 | 2000 |
| IEMOCAP | text | gaussian | all_retained | strength_0.10_minus_0.05 | flip_rate | -0.000949 | -0.002971 | 0.000000 | 2000 |
| IEMOCAP | text | gaussian | all_retained | strength_0.10_minus_0.05 | mu_zero_fraction | 0.000000 | 0.000000 | 0.000000 | 2000 |
| IEMOCAP | text | gaussian | all_retained | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.000000 | 0.000000 | 0.000000 | 2000 |
| IEMOCAP | text | gaussian | all_retained | matched_wrong_minus_correct | mu | -0.002236 | -0.006322 | 0.000000 | 2000 |
| IEMOCAP | text | gaussian | all_retained | matched_wrong_minus_correct | mu | -0.000373 | -0.001054 | 0.000000 | 2000 |
| IEMOCAP | text | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_mu | 0.001863 | 0.000000 | 0.005268 | 2000 |
| IEMOCAP | text | gaussian | all_retained | matched_wrong_minus_correct | delta | -0.000037 | -0.000073 | -0.000006 | 2000 |
| IEMOCAP | text | gaussian | all_retained | matched_wrong_minus_correct | delta | 0.000008 | -0.000037 | 0.000047 | 2000 |
| IEMOCAP | text | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_delta | 0.000045 | 0.000011 | 0.000074 | 2000 |
| IEMOCAP | text | gaussian | all_retained | matched_wrong_minus_correct | js | -0.000000 | -0.000000 | 0.000000 | 2000 |
| IEMOCAP | text | gaussian | all_retained | matched_wrong_minus_correct | js | -0.000000 | -0.000000 | 0.000000 | 2000 |
| IEMOCAP | text | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_js | 0.000000 | -0.000000 | 0.000000 | 2000 |
| IEMOCAP | text | gaussian | all_retained | matched_wrong_minus_correct | flip_rate | -0.005263 | -0.014879 | 0.000000 | 2000 |
| IEMOCAP | text | gaussian | all_retained | matched_wrong_minus_correct | flip_rate | -0.000877 | -0.002480 | 0.000000 | 2000 |
| IEMOCAP | text | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | 0.004386 | 0.000000 | 0.012399 | 2000 |
| IEMOCAP | visual | dropout | all_retained | strength_0.10_minus_0.05 | mu | -0.000242 | -0.000757 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | all_retained | strength_0.10_minus_0.05 | delta | 0.000211 | 0.000135 | 0.000314 | 2000 |
| IEMOCAP | visual | dropout | all_retained | strength_0.10_minus_0.05 | js | 0.000001 | 0.000000 | 0.000001 | 2000 |
| IEMOCAP | visual | dropout | all_retained | strength_0.10_minus_0.05 | flip_rate | -0.000569 | -0.001782 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | all_retained | strength_0.10_minus_0.05 | mu_zero_fraction | 0.000000 | 0.000000 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | all_retained | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.000000 | 0.000000 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | all_retained | matched_wrong_minus_correct | mu | -0.001865 | -0.005271 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | all_retained | matched_wrong_minus_correct | mu | -0.000746 | -0.002110 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_mu | 0.001118 | 0.000000 | 0.003161 | 2000 |
| IEMOCAP | visual | dropout | all_retained | matched_wrong_minus_correct | delta | -0.000017 | -0.000064 | 0.000029 | 2000 |
| IEMOCAP | visual | dropout | all_retained | matched_wrong_minus_correct | delta | -0.000080 | -0.000174 | -0.000005 | 2000 |
| IEMOCAP | visual | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000062 | -0.000158 | 0.000014 | 2000 |
| IEMOCAP | visual | dropout | all_retained | matched_wrong_minus_correct | js | 0.000000 | -0.000000 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | all_retained | matched_wrong_minus_correct | js | -0.000000 | -0.000000 | -0.000000 | 2000 |
| IEMOCAP | visual | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_js | -0.000000 | -0.000000 | -0.000000 | 2000 |
| IEMOCAP | visual | dropout | all_retained | matched_wrong_minus_correct | flip_rate | -0.004386 | -0.012399 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | all_retained | matched_wrong_minus_correct | flip_rate | -0.001754 | -0.004960 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | 0.002632 | 0.000000 | 0.007440 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | strength_0.10_minus_0.05 | mu | -0.000242 | -0.000759 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | strength_0.10_minus_0.05 | delta | 0.000211 | 0.000135 | 0.000314 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | strength_0.10_minus_0.05 | js | 0.000001 | 0.000000 | 0.000001 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | strength_0.10_minus_0.05 | flip_rate | -0.000570 | -0.001786 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | strength_0.10_minus_0.05 | mu_zero_fraction | 0.000000 | 0.000000 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.000000 | 0.000000 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | matched_wrong_minus_correct | mu | -0.001881 | -0.005271 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | matched_wrong_minus_correct | mu | -0.000753 | -0.002110 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_mu | 0.001128 | 0.000000 | 0.003161 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | matched_wrong_minus_correct | delta | -0.000016 | -0.000063 | 0.000030 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | matched_wrong_minus_correct | delta | -0.000080 | -0.000175 | -0.000005 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000064 | -0.000160 | 0.000013 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | matched_wrong_minus_correct | js | 0.000000 | -0.000000 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | matched_wrong_minus_correct | js | -0.000000 | -0.000000 | -0.000000 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_js | -0.000000 | -0.000000 | -0.000000 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | matched_wrong_minus_correct | flip_rate | -0.004425 | -0.012399 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | matched_wrong_minus_correct | flip_rate | -0.001770 | -0.004960 | 0.000000 | 2000 |
| IEMOCAP | visual | dropout | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | 0.002655 | 0.000000 | 0.007440 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | strength_0.10_minus_0.05 | mu | -0.000161 | -0.000505 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | strength_0.10_minus_0.05 | delta | 0.000034 | 0.000023 | 0.000048 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | strength_0.10_minus_0.05 | js | 0.000000 | 0.000000 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | strength_0.10_minus_0.05 | flip_rate | -0.000380 | -0.001188 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | strength_0.10_minus_0.05 | mu_zero_fraction | 0.000000 | 0.000000 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.000000 | 0.000000 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | matched_wrong_minus_correct | mu | -0.001491 | -0.004214 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | matched_wrong_minus_correct | mu | -0.000745 | -0.002107 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_mu | 0.000745 | 0.000000 | 0.002107 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | matched_wrong_minus_correct | delta | -0.000005 | -0.000014 | 0.000003 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | matched_wrong_minus_correct | delta | 0.000003 | -0.000013 | 0.000019 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_delta | 0.000007 | -0.000012 | 0.000029 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | matched_wrong_minus_correct | js | -0.000000 | -0.000000 | -0.000000 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | matched_wrong_minus_correct | js | -0.000000 | -0.000000 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_js | -0.000000 | -0.000000 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | matched_wrong_minus_correct | flip_rate | -0.003509 | -0.009919 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | matched_wrong_minus_correct | flip_rate | -0.001754 | -0.004960 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | 0.001754 | 0.000000 | 0.004960 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | strength_0.10_minus_0.05 | mu | -0.000162 | -0.000506 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | strength_0.10_minus_0.05 | delta | 0.000034 | 0.000023 | 0.000048 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | strength_0.10_minus_0.05 | js | 0.000000 | 0.000000 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | strength_0.10_minus_0.05 | flip_rate | -0.000380 | -0.001191 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | strength_0.10_minus_0.05 | mu_zero_fraction | 0.000000 | 0.000000 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.000000 | 0.000000 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | matched_wrong_minus_correct | mu | -0.001504 | -0.004214 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | matched_wrong_minus_correct | mu | -0.000752 | -0.002107 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_mu | 0.000752 | 0.000000 | 0.002107 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | matched_wrong_minus_correct | delta | -0.000004 | -0.000014 | 0.000003 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | matched_wrong_minus_correct | delta | 0.000003 | -0.000013 | 0.000019 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_delta | 0.000008 | -0.000012 | 0.000030 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | matched_wrong_minus_correct | js | -0.000000 | -0.000000 | -0.000000 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | matched_wrong_minus_correct | js | -0.000000 | -0.000000 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_js | -0.000000 | -0.000000 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | matched_wrong_minus_correct | flip_rate | -0.003540 | -0.009919 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | matched_wrong_minus_correct | flip_rate | -0.001770 | -0.004960 | 0.000000 | 2000 |
| IEMOCAP | visual | gaussian | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | 0.001770 | 0.000000 | 0.004960 | 2000 |
| MELD | audio | dropout | all_retained | strength_0.10_minus_0.05 | mu | 0.000673 | 0.000211 | 0.001220 | 2000 |
| MELD | audio | dropout | all_retained | strength_0.10_minus_0.05 | delta | 0.001215 | 0.001062 | 0.001394 | 2000 |
| MELD | audio | dropout | all_retained | strength_0.10_minus_0.05 | js | 0.000019 | 0.000017 | 0.000021 | 2000 |
| MELD | audio | dropout | all_retained | strength_0.10_minus_0.05 | flip_rate | 0.002164 | 0.000745 | 0.003821 | 2000 |
| MELD | audio | dropout | all_retained | strength_0.10_minus_0.05 | mu_zero_fraction | -0.007214 | -0.012940 | -0.001860 | 2000 |
| MELD | audio | dropout | all_retained | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.007214 | 0.001860 | 0.012940 | 2000 |
| MELD | audio | dropout | all_retained | matched_wrong_minus_correct | mu | 0.001720 | -0.003465 | 0.007858 | 2000 |
| MELD | audio | dropout | all_retained | matched_wrong_minus_correct | mu | 0.001839 | -0.003064 | 0.008333 | 2000 |
| MELD | audio | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_mu | 0.000120 | -0.002175 | 0.002652 | 2000 |
| MELD | audio | dropout | all_retained | matched_wrong_minus_correct | delta | -0.000412 | -0.000695 | -0.000133 | 2000 |
| MELD | audio | dropout | all_retained | matched_wrong_minus_correct | delta | -0.000741 | -0.001251 | -0.000216 | 2000 |
| MELD | audio | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000329 | -0.000739 | 0.000094 | 2000 |
| MELD | audio | dropout | all_retained | matched_wrong_minus_correct | js | -0.000000 | -0.000004 | 0.000003 | 2000 |
| MELD | audio | dropout | all_retained | matched_wrong_minus_correct | js | -0.000003 | -0.000011 | 0.000004 | 2000 |
| MELD | audio | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_js | -0.000003 | -0.000009 | 0.000002 | 2000 |
| MELD | audio | dropout | all_retained | matched_wrong_minus_correct | flip_rate | 0.005109 | -0.013429 | 0.025386 | 2000 |
| MELD | audio | dropout | all_retained | matched_wrong_minus_correct | flip_rate | 0.005109 | -0.010688 | 0.026017 | 2000 |
| MELD | audio | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | 0.000000 | -0.007194 | 0.008149 | 2000 |
| MELD | audio | gaussian | all_retained | strength_0.10_minus_0.05 | mu | 0.000202 | -0.000032 | 0.000530 | 2000 |
| MELD | audio | gaussian | all_retained | strength_0.10_minus_0.05 | delta | 0.000147 | 0.000131 | 0.000163 | 2000 |
| MELD | audio | gaussian | all_retained | strength_0.10_minus_0.05 | js | 0.000001 | 0.000001 | 0.000001 | 2000 |
| MELD | audio | gaussian | all_retained | strength_0.10_minus_0.05 | flip_rate | 0.000631 | -0.000198 | 0.001753 | 2000 |
| MELD | audio | gaussian | all_retained | strength_0.10_minus_0.05 | mu_zero_fraction | -0.001803 | -0.004481 | 0.000000 | 2000 |
| MELD | audio | gaussian | all_retained | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.001803 | 0.000000 | 0.004481 | 2000 |
| MELD | audio | gaussian | all_retained | matched_wrong_minus_correct | mu | 0.000000 | 0.000000 | 0.000000 | 2000 |
| MELD | audio | gaussian | all_retained | matched_wrong_minus_correct | mu | 0.000586 | 0.000000 | 0.001535 | 2000 |
| MELD | audio | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_mu | 0.000586 | 0.000000 | 0.001535 | 2000 |
| MELD | audio | gaussian | all_retained | matched_wrong_minus_correct | delta | -0.000016 | -0.000046 | 0.000013 | 2000 |
| MELD | audio | gaussian | all_retained | matched_wrong_minus_correct | delta | 0.000016 | -0.000039 | 0.000072 | 2000 |
| MELD | audio | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_delta | 0.000031 | -0.000028 | 0.000093 | 2000 |
| MELD | audio | gaussian | all_retained | matched_wrong_minus_correct | js | -0.000000 | -0.000000 | 0.000000 | 2000 |
| MELD | audio | gaussian | all_retained | matched_wrong_minus_correct | js | -0.000000 | -0.000000 | -0.000000 | 2000 |
| MELD | audio | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_js | -0.000000 | -0.000000 | 0.000000 | 2000 |
| MELD | audio | gaussian | all_retained | matched_wrong_minus_correct | flip_rate | 0.000000 | 0.000000 | 0.000000 | 2000 |
| MELD | audio | gaussian | all_retained | matched_wrong_minus_correct | flip_rate | 0.002190 | 0.000000 | 0.005714 | 2000 |
| MELD | audio | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | 0.002190 | 0.000000 | 0.005714 | 2000 |
| MELD | joint | dropout | all_retained | strength_0.10_minus_0.05 | mu | 0.002271 | 0.001310 | 0.003294 | 2000 |
| MELD | joint | dropout | all_retained | strength_0.10_minus_0.05 | delta | 0.004916 | 0.004585 | 0.005269 | 2000 |
| MELD | joint | dropout | all_retained | strength_0.10_minus_0.05 | js | 0.000169 | 0.000158 | 0.000180 | 2000 |
| MELD | joint | dropout | all_retained | strength_0.10_minus_0.05 | flip_rate | 0.007394 | 0.004136 | 0.010714 | 2000 |
| MELD | joint | dropout | all_retained | strength_0.10_minus_0.05 | mu_zero_fraction | -0.016231 | -0.026517 | -0.005966 | 2000 |
| MELD | joint | dropout | all_retained | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.016231 | 0.005966 | 0.026517 | 2000 |
| MELD | joint | dropout | all_retained | matched_wrong_minus_correct | mu | 0.001672 | -0.004061 | 0.007813 | 2000 |
| MELD | joint | dropout | all_retained | matched_wrong_minus_correct | mu | 0.000934 | -0.008202 | 0.011119 | 2000 |
| MELD | joint | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_mu | -0.000738 | -0.006209 | 0.005016 | 2000 |
| MELD | joint | dropout | all_retained | matched_wrong_minus_correct | delta | -0.000539 | -0.001528 | 0.000429 | 2000 |
| MELD | joint | dropout | all_retained | matched_wrong_minus_correct | delta | -0.000393 | -0.002255 | 0.001447 | 2000 |
| MELD | joint | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_delta | 0.000146 | -0.001125 | 0.001446 | 2000 |
| MELD | joint | dropout | all_retained | matched_wrong_minus_correct | js | -0.000000 | -0.000015 | 0.000014 | 2000 |
| MELD | joint | dropout | all_retained | matched_wrong_minus_correct | js | 0.000002 | -0.000042 | 0.000044 | 2000 |
| MELD | joint | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_js | 0.000002 | -0.000031 | 0.000035 | 2000 |
| MELD | joint | dropout | all_retained | matched_wrong_minus_correct | flip_rate | 0.005109 | -0.014187 | 0.025238 | 2000 |
| MELD | joint | dropout | all_retained | matched_wrong_minus_correct | flip_rate | 0.001460 | -0.026668 | 0.033572 | 2000 |
| MELD | joint | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | -0.003650 | -0.020837 | 0.013710 | 2000 |
| MELD | joint | dropout | valid_visual | strength_0.10_minus_0.05 | mu | 0.002288 | 0.001297 | 0.003346 | 2000 |
| MELD | joint | dropout | valid_visual | strength_0.10_minus_0.05 | delta | 0.004858 | 0.004513 | 0.005222 | 2000 |
| MELD | joint | dropout | valid_visual | strength_0.10_minus_0.05 | js | 0.000169 | 0.000158 | 0.000180 | 2000 |
| MELD | joint | dropout | valid_visual | strength_0.10_minus_0.05 | flip_rate | 0.007495 | 0.004177 | 0.010938 | 2000 |
| MELD | joint | dropout | valid_visual | strength_0.10_minus_0.05 | mu_zero_fraction | -0.015779 | -0.026693 | -0.004996 | 2000 |
| MELD | joint | dropout | valid_visual | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.015779 | 0.004996 | 0.026693 | 2000 |
| MELD | joint | dropout | valid_visual | matched_wrong_minus_correct | mu | 0.001909 | -0.004719 | 0.008793 | 2000 |
| MELD | joint | dropout | valid_visual | matched_wrong_minus_correct | mu | 0.001359 | -0.008854 | 0.012618 | 2000 |
| MELD | joint | dropout | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_mu | -0.000550 | -0.006535 | 0.005276 | 2000 |
| MELD | joint | dropout | valid_visual | matched_wrong_minus_correct | delta | -0.000618 | -0.001593 | 0.000267 | 2000 |
| MELD | joint | dropout | valid_visual | matched_wrong_minus_correct | delta | -0.000793 | -0.002501 | 0.000906 | 2000 |
| MELD | joint | dropout | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000175 | -0.001369 | 0.000997 | 2000 |
| MELD | joint | dropout | valid_visual | matched_wrong_minus_correct | js | 0.000002 | -0.000013 | 0.000016 | 2000 |
| MELD | joint | dropout | valid_visual | matched_wrong_minus_correct | js | -0.000005 | -0.000047 | 0.000038 | 2000 |
| MELD | joint | dropout | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_js | -0.000007 | -0.000040 | 0.000026 | 2000 |
| MELD | joint | dropout | valid_visual | matched_wrong_minus_correct | flip_rate | 0.005833 | -0.016532 | 0.028458 | 2000 |
| MELD | joint | dropout | valid_visual | matched_wrong_minus_correct | flip_rate | 0.002500 | -0.029286 | 0.037104 | 2000 |
| MELD | joint | dropout | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | -0.003333 | -0.022303 | 0.015108 | 2000 |
| MELD | joint | gaussian | all_retained | strength_0.10_minus_0.05 | mu | 0.000235 | -0.000071 | 0.000610 | 2000 |
| MELD | joint | gaussian | all_retained | strength_0.10_minus_0.05 | delta | 0.000372 | 0.000335 | 0.000411 | 2000 |
| MELD | joint | gaussian | all_retained | strength_0.10_minus_0.05 | js | 0.000008 | 0.000008 | 0.000008 | 2000 |
| MELD | joint | gaussian | all_retained | strength_0.10_minus_0.05 | flip_rate | 0.000721 | -0.000468 | 0.002070 | 2000 |
| MELD | joint | gaussian | all_retained | strength_0.10_minus_0.05 | mu_zero_fraction | -0.006312 | -0.011795 | -0.001833 | 2000 |
| MELD | joint | gaussian | all_retained | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.006312 | 0.001833 | 0.011795 | 2000 |
| MELD | joint | gaussian | all_retained | matched_wrong_minus_correct | mu | 0.001214 | 0.000000 | 0.003870 | 2000 |
| MELD | joint | gaussian | all_retained | matched_wrong_minus_correct | mu | 0.001016 | -0.000696 | 0.003290 | 2000 |
| MELD | joint | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_mu | -0.000198 | -0.001583 | 0.001491 | 2000 |
| MELD | joint | gaussian | all_retained | matched_wrong_minus_correct | delta | 0.000056 | -0.000017 | 0.000136 | 2000 |
| MELD | joint | gaussian | all_retained | matched_wrong_minus_correct | delta | 0.000008 | -0.000112 | 0.000129 | 2000 |
| MELD | joint | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000048 | -0.000199 | 0.000103 | 2000 |
| MELD | joint | gaussian | all_retained | matched_wrong_minus_correct | js | 0.000000 | -0.000000 | 0.000001 | 2000 |
| MELD | joint | gaussian | all_retained | matched_wrong_minus_correct | js | 0.000000 | -0.000001 | 0.000001 | 2000 |
| MELD | joint | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_js | 0.000000 | -0.000001 | 0.000001 | 2000 |
| MELD | joint | gaussian | all_retained | matched_wrong_minus_correct | flip_rate | 0.004380 | 0.000000 | 0.013959 | 2000 |
| MELD | joint | gaussian | all_retained | matched_wrong_minus_correct | flip_rate | 0.003650 | -0.002420 | 0.011629 | 2000 |
| MELD | joint | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | -0.000730 | -0.006000 | 0.005692 | 2000 |
| MELD | joint | gaussian | valid_visual | strength_0.10_minus_0.05 | mu | 0.000186 | -0.000121 | 0.000563 | 2000 |
| MELD | joint | gaussian | valid_visual | strength_0.10_minus_0.05 | delta | 0.000374 | 0.000335 | 0.000416 | 2000 |
| MELD | joint | gaussian | valid_visual | strength_0.10_minus_0.05 | js | 0.000008 | 0.000008 | 0.000009 | 2000 |
| MELD | joint | gaussian | valid_visual | strength_0.10_minus_0.05 | flip_rate | 0.000592 | -0.000627 | 0.002008 | 2000 |
| MELD | joint | gaussian | valid_visual | strength_0.10_minus_0.05 | mu_zero_fraction | -0.006903 | -0.012783 | -0.002012 | 2000 |
| MELD | joint | gaussian | valid_visual | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.006903 | 0.002012 | 0.012783 | 2000 |
| MELD | joint | gaussian | valid_visual | matched_wrong_minus_correct | mu | 0.001386 | 0.000000 | 0.004417 | 2000 |
| MELD | joint | gaussian | valid_visual | matched_wrong_minus_correct | mu | 0.001160 | -0.000811 | 0.003712 | 2000 |
| MELD | joint | gaussian | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_mu | -0.000226 | -0.001820 | 0.001728 | 2000 |
| MELD | joint | gaussian | valid_visual | matched_wrong_minus_correct | delta | 0.000051 | -0.000027 | 0.000132 | 2000 |
| MELD | joint | gaussian | valid_visual | matched_wrong_minus_correct | delta | 0.000012 | -0.000113 | 0.000144 | 2000 |
| MELD | joint | gaussian | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000040 | -0.000199 | 0.000124 | 2000 |
| MELD | joint | gaussian | valid_visual | matched_wrong_minus_correct | js | 0.000000 | 0.000000 | 0.000001 | 2000 |
| MELD | joint | gaussian | valid_visual | matched_wrong_minus_correct | js | 0.000000 | -0.000001 | 0.000001 | 2000 |
| MELD | joint | gaussian | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_js | -0.000000 | -0.000001 | 0.000001 | 2000 |
| MELD | joint | gaussian | valid_visual | matched_wrong_minus_correct | flip_rate | 0.005000 | 0.000000 | 0.015933 | 2000 |
| MELD | joint | gaussian | valid_visual | matched_wrong_minus_correct | flip_rate | 0.004167 | -0.002817 | 0.013275 | 2000 |
| MELD | joint | gaussian | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | -0.000833 | -0.006931 | 0.006505 | 2000 |
| MELD | text | dropout | all_retained | strength_0.10_minus_0.05 | mu | 0.001981 | 0.001025 | 0.002995 | 2000 |
| MELD | text | dropout | all_retained | strength_0.10_minus_0.05 | delta | 0.004586 | 0.004292 | 0.004888 | 2000 |
| MELD | text | dropout | all_retained | strength_0.10_minus_0.05 | js | 0.000140 | 0.000130 | 0.000149 | 2000 |
| MELD | text | dropout | all_retained | strength_0.10_minus_0.05 | flip_rate | 0.006583 | 0.003514 | 0.009813 | 2000 |
| MELD | text | dropout | all_retained | strength_0.10_minus_0.05 | mu_zero_fraction | -0.018936 | -0.029358 | -0.009830 | 2000 |
| MELD | text | dropout | all_retained | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.018936 | 0.009830 | 0.029358 | 2000 |
| MELD | text | dropout | all_retained | matched_wrong_minus_correct | mu | 0.000330 | -0.004103 | 0.004201 | 2000 |
| MELD | text | dropout | all_retained | matched_wrong_minus_correct | mu | -0.000734 | -0.008727 | 0.007909 | 2000 |
| MELD | text | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_mu | -0.001063 | -0.006718 | 0.004877 | 2000 |
| MELD | text | dropout | all_retained | matched_wrong_minus_correct | delta | -0.000301 | -0.001227 | 0.000605 | 2000 |
| MELD | text | dropout | all_retained | matched_wrong_minus_correct | delta | 0.000046 | -0.001702 | 0.001836 | 2000 |
| MELD | text | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_delta | 0.000347 | -0.000851 | 0.001544 | 2000 |
| MELD | text | dropout | all_retained | matched_wrong_minus_correct | js | 0.000004 | -0.000008 | 0.000016 | 2000 |
| MELD | text | dropout | all_retained | matched_wrong_minus_correct | js | 0.000009 | -0.000032 | 0.000048 | 2000 |
| MELD | text | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_js | 0.000005 | -0.000027 | 0.000036 | 2000 |
| MELD | text | dropout | all_retained | matched_wrong_minus_correct | flip_rate | 0.000730 | -0.013971 | 0.013670 | 2000 |
| MELD | text | dropout | all_retained | matched_wrong_minus_correct | flip_rate | -0.003650 | -0.028706 | 0.022604 | 2000 |
| MELD | text | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | -0.004380 | -0.022222 | 0.013671 | 2000 |
| MELD | text | gaussian | all_retained | strength_0.10_minus_0.05 | mu | 0.000198 | -0.000091 | 0.000469 | 2000 |
| MELD | text | gaussian | all_retained | strength_0.10_minus_0.05 | delta | 0.000332 | 0.000299 | 0.000367 | 2000 |
| MELD | text | gaussian | all_retained | strength_0.10_minus_0.05 | js | 0.000007 | 0.000006 | 0.000007 | 2000 |
| MELD | text | gaussian | all_retained | strength_0.10_minus_0.05 | flip_rate | 0.000631 | -0.000413 | 0.001620 | 2000 |
| MELD | text | gaussian | all_retained | strength_0.10_minus_0.05 | mu_zero_fraction | -0.004509 | -0.008475 | -0.000938 | 2000 |
| MELD | text | gaussian | all_retained | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.004509 | 0.000938 | 0.008475 | 2000 |
| MELD | text | gaussian | all_retained | matched_wrong_minus_correct | mu | 0.000979 | 0.000000 | 0.002809 | 2000 |
| MELD | text | gaussian | all_retained | matched_wrong_minus_correct | mu | 0.001480 | 0.000000 | 0.003772 | 2000 |
| MELD | text | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_mu | 0.000501 | 0.000000 | 0.001428 | 2000 |
| MELD | text | gaussian | all_retained | matched_wrong_minus_correct | delta | 0.000059 | -0.000003 | 0.000126 | 2000 |
| MELD | text | gaussian | all_retained | matched_wrong_minus_correct | delta | 0.000012 | -0.000102 | 0.000123 | 2000 |
| MELD | text | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000047 | -0.000186 | 0.000086 | 2000 |
| MELD | text | gaussian | all_retained | matched_wrong_minus_correct | js | 0.000000 | -0.000000 | 0.000001 | 2000 |
| MELD | text | gaussian | all_retained | matched_wrong_minus_correct | js | 0.000001 | -0.000000 | 0.000002 | 2000 |
| MELD | text | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_js | 0.000000 | -0.000000 | 0.000001 | 2000 |
| MELD | text | gaussian | all_retained | matched_wrong_minus_correct | flip_rate | 0.003650 | 0.000000 | 0.010476 | 2000 |
| MELD | text | gaussian | all_retained | matched_wrong_minus_correct | flip_rate | 0.005109 | 0.000000 | 0.012727 | 2000 |
| MELD | text | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | 0.001460 | -0.000000 | 0.005002 | 2000 |
| MELD | visual | dropout | all_retained | strength_0.10_minus_0.05 | mu | 0.000081 | -0.000144 | 0.000319 | 2000 |
| MELD | visual | dropout | all_retained | strength_0.10_minus_0.05 | delta | 0.000310 | 0.000270 | 0.000352 | 2000 |
| MELD | visual | dropout | all_retained | strength_0.10_minus_0.05 | js | 0.000003 | 0.000002 | 0.000003 | 2000 |
| MELD | visual | dropout | all_retained | strength_0.10_minus_0.05 | flip_rate | 0.000361 | -0.000509 | 0.001316 | 2000 |
| MELD | visual | dropout | all_retained | strength_0.10_minus_0.05 | mu_zero_fraction | 0.000902 | -0.001866 | 0.004234 | 2000 |
| MELD | visual | dropout | all_retained | strength_0.10_minus_0.05 | nonzero_mu_fraction | -0.000902 | -0.004234 | 0.001866 | 2000 |
| MELD | visual | dropout | all_retained | matched_wrong_minus_correct | mu | 0.000968 | -0.000540 | 0.003887 | 2000 |
| MELD | visual | dropout | all_retained | matched_wrong_minus_correct | mu | 0.001559 | 0.000000 | 0.003854 | 2000 |
| MELD | visual | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_mu | 0.000590 | -0.000963 | 0.002576 | 2000 |
| MELD | visual | dropout | all_retained | matched_wrong_minus_correct | delta | -0.000024 | -0.000148 | 0.000097 | 2000 |
| MELD | visual | dropout | all_retained | matched_wrong_minus_correct | delta | -0.000044 | -0.000240 | 0.000144 | 2000 |
| MELD | visual | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000020 | -0.000152 | 0.000095 | 2000 |
| MELD | visual | dropout | all_retained | matched_wrong_minus_correct | js | -0.000000 | -0.000001 | 0.000000 | 2000 |
| MELD | visual | dropout | all_retained | matched_wrong_minus_correct | js | -0.000001 | -0.000002 | 0.000000 | 2000 |
| MELD | visual | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_js | -0.000001 | -0.000002 | 0.000001 | 2000 |
| MELD | visual | dropout | all_retained | matched_wrong_minus_correct | flip_rate | 0.003650 | -0.002055 | 0.014634 | 2000 |
| MELD | visual | dropout | all_retained | matched_wrong_minus_correct | flip_rate | 0.005839 | 0.000000 | 0.014494 | 2000 |
| MELD | visual | dropout | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | 0.002190 | -0.003604 | 0.009559 | 2000 |
| MELD | visual | dropout | valid_visual | strength_0.10_minus_0.05 | mu | 0.000089 | -0.000158 | 0.000349 | 2000 |
| MELD | visual | dropout | valid_visual | strength_0.10_minus_0.05 | delta | 0.000339 | 0.000298 | 0.000383 | 2000 |
| MELD | visual | dropout | valid_visual | strength_0.10_minus_0.05 | js | 0.000003 | 0.000003 | 0.000003 | 2000 |
| MELD | visual | dropout | valid_visual | strength_0.10_minus_0.05 | flip_rate | 0.000394 | -0.000550 | 0.001448 | 2000 |
| MELD | visual | dropout | valid_visual | strength_0.10_minus_0.05 | mu_zero_fraction | 0.000986 | -0.002045 | 0.004589 | 2000 |
| MELD | visual | dropout | valid_visual | strength_0.10_minus_0.05 | nonzero_mu_fraction | -0.000986 | -0.004589 | 0.002045 | 2000 |
| MELD | visual | dropout | valid_visual | matched_wrong_minus_correct | mu | 0.001106 | -0.000623 | 0.004394 | 2000 |
| MELD | visual | dropout | valid_visual | matched_wrong_minus_correct | mu | 0.001780 | 0.000000 | 0.004359 | 2000 |
| MELD | visual | dropout | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_mu | 0.000674 | -0.001133 | 0.002944 | 2000 |
| MELD | visual | dropout | valid_visual | matched_wrong_minus_correct | delta | 0.000003 | -0.000137 | 0.000139 | 2000 |
| MELD | visual | dropout | valid_visual | matched_wrong_minus_correct | delta | -0.000002 | -0.000237 | 0.000227 | 2000 |
| MELD | visual | dropout | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000005 | -0.000162 | 0.000134 | 2000 |
| MELD | visual | dropout | valid_visual | matched_wrong_minus_correct | js | -0.000000 | -0.000001 | 0.000000 | 2000 |
| MELD | visual | dropout | valid_visual | matched_wrong_minus_correct | js | -0.000001 | -0.000003 | 0.000001 | 2000 |
| MELD | visual | dropout | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_js | -0.000000 | -0.000002 | 0.000001 | 2000 |
| MELD | visual | dropout | valid_visual | matched_wrong_minus_correct | flip_rate | 0.004167 | -0.002362 | 0.016515 | 2000 |
| MELD | visual | dropout | valid_visual | matched_wrong_minus_correct | flip_rate | 0.006667 | 0.000000 | 0.016328 | 2000 |
| MELD | visual | dropout | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | 0.002500 | -0.004238 | 0.010940 | 2000 |
| MELD | visual | gaussian | all_retained | strength_0.10_minus_0.05 | mu | 0.000027 | 0.000000 | 0.000085 | 2000 |
| MELD | visual | gaussian | all_retained | strength_0.10_minus_0.05 | delta | 0.000053 | 0.000046 | 0.000059 | 2000 |
| MELD | visual | gaussian | all_retained | strength_0.10_minus_0.05 | js | 0.000000 | 0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | all_retained | strength_0.10_minus_0.05 | flip_rate | 0.000090 | 0.000000 | 0.000284 | 2000 |
| MELD | visual | gaussian | all_retained | strength_0.10_minus_0.05 | mu_zero_fraction | -0.000902 | -0.002838 | 0.000000 | 2000 |
| MELD | visual | gaussian | all_retained | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.000902 | 0.000000 | 0.002838 | 2000 |
| MELD | visual | gaussian | all_retained | matched_wrong_minus_correct | mu | 0.000000 | 0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | all_retained | matched_wrong_minus_correct | mu | 0.000000 | 0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_mu | 0.000000 | 0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | all_retained | matched_wrong_minus_correct | delta | 0.000006 | -0.000005 | 0.000018 | 2000 |
| MELD | visual | gaussian | all_retained | matched_wrong_minus_correct | delta | -0.000002 | -0.000030 | 0.000025 | 2000 |
| MELD | visual | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000009 | -0.000038 | 0.000020 | 2000 |
| MELD | visual | gaussian | all_retained | matched_wrong_minus_correct | js | 0.000000 | -0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | all_retained | matched_wrong_minus_correct | js | -0.000000 | -0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_js | -0.000000 | -0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | all_retained | matched_wrong_minus_correct | flip_rate | 0.000000 | 0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | all_retained | matched_wrong_minus_correct | flip_rate | 0.000000 | 0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | all_retained | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | 0.000000 | 0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | valid_visual | strength_0.10_minus_0.05 | mu | 0.000029 | 0.000000 | 0.000093 | 2000 |
| MELD | visual | gaussian | valid_visual | strength_0.10_minus_0.05 | delta | 0.000057 | 0.000050 | 0.000065 | 2000 |
| MELD | visual | gaussian | valid_visual | strength_0.10_minus_0.05 | js | 0.000000 | 0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | valid_visual | strength_0.10_minus_0.05 | flip_rate | 0.000099 | 0.000000 | 0.000314 | 2000 |
| MELD | visual | gaussian | valid_visual | strength_0.10_minus_0.05 | mu_zero_fraction | -0.000986 | -0.003138 | 0.000000 | 2000 |
| MELD | visual | gaussian | valid_visual | strength_0.10_minus_0.05 | nonzero_mu_fraction | 0.000986 | 0.000000 | 0.003138 | 2000 |
| MELD | visual | gaussian | valid_visual | matched_wrong_minus_correct | mu | 0.000000 | 0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | valid_visual | matched_wrong_minus_correct | mu | 0.000000 | 0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_mu | 0.000000 | 0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | valid_visual | matched_wrong_minus_correct | delta | 0.000011 | -0.000001 | 0.000023 | 2000 |
| MELD | visual | gaussian | valid_visual | matched_wrong_minus_correct | delta | 0.000005 | -0.000025 | 0.000035 | 2000 |
| MELD | visual | gaussian | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_delta | -0.000006 | -0.000040 | 0.000027 | 2000 |
| MELD | visual | gaussian | valid_visual | matched_wrong_minus_correct | js | 0.000000 | -0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | valid_visual | matched_wrong_minus_correct | js | 0.000000 | -0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_js | -0.000000 | -0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | valid_visual | matched_wrong_minus_correct | flip_rate | 0.000000 | 0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | valid_visual | matched_wrong_minus_correct | flip_rate | 0.000000 | 0.000000 | 0.000000 | 2000 |
| MELD | visual | gaussian | valid_visual | matched_difference_strength_0.10_minus_0.05 | matched_flip_rate | 0.000000 | 0.000000 | 0.000000 | 2000 |

响应差值为0.10−0.05，匹配差值为错误−正确。强度间噪声采用独立的确定性随机流，配对仅发生于话语与Dialogue层面，没有宣称逐repeat噪声配对。无配对话语的重抽样可能没有有效分母，明确报告有效/无效抽样数。

## 解释边界

更强干扰带来更多翻转、更大μ/δ/JS或更低零值占比，只证明模型行为对干扰有响应；不能据此证明响应包含原始C之外的校准信息。增量校准应直接查看同折OOF的行为方法−E1差值，以及同方法0.10−0.05敏感性差值。所有预登记结果均报告，不以最优模态、指标或强度替换主条件。

Validation已用于checkpoint选择及多轮方法诊断；Dialogue OOF仅隔离可靠性映射的拟合和选参，不构成独立确认。只有一个模型Seed，不支持五Seed结论或正式性能提升声明。原始分类预测不变；本批没有训练CMERC、读取Test标签、修改原始数据、温度缩放，没有自动扩展0.20或更多repeat。独立验证结果由根目录verification.json给出。
