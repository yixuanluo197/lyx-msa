# Calibration and confidence-ranking diagnostic

Execution coverage: **complete**; missing evaluations: 0.

Validation is the primary analysis. Test is the prespecified replication analysis and is reported separately.
IEMOCAP seed 1937 test results were already observed in round 2 and were reused without retraining or reselection.
MELD seed 1937 is the technical pilot; its test may precede the remaining fixed matrix. Configurations do not change after that exposure.
Deltas are CMERC minus Baseline. Classification/calibration are percentages, with deltas in percentage points.
CEL is a sum of positive true-class log-probability differences under `prefix_mean_v1`, a prefix smoothing view, not complete modality removal.
Three seeds on a fixed split give uncertain estimates. Sample SD uses ddof=1; df=2 t intervals rely on normality and are descriptive.
The exact two-sided sign-flip test has minimum p=0.25 for three seeds. No robust statistical significance or proven innovation is claimed.

ECE/MCE/RMSCE use the frozen 15 equal-width confidence bins. AUROC/AUPRC-Success rank prediction correctness
using maximum softmax confidence. Calibration and correctness ranking are distinct properties.
All-correct/all-wrong segments have undefined success AUROC/AUPRC and remain NA, never zero.

| Dataset | Seed | Split | Segment | Metric | Baseline | CMERC | Delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | 1937 | validation | overall | ece | 21.367332 | 15.216119 | -6.151213 |
| IEMOCAP | 1937 | validation | overall | mce | 28.359297 | 73.788282 | 45.428985 |
| IEMOCAP | 1937 | validation | overall | rmsce | 22.212391 | 17.222625 | -4.989766 |
| IEMOCAP | 1937 | validation | overall | auroc_success | 64.133457 | 62.069286 | -2.064171 |
| IEMOCAP | 1937 | validation | overall | auprc_success | 73.171425 | 70.309299 | -2.862126 |
| IEMOCAP | 1937 | validation | early | ece | 28.806972 | 18.918251 | -9.888721 |
| IEMOCAP | 1937 | validation | early | mce | 40.423717 | 73.788282 | 33.364566 |
| IEMOCAP | 1937 | validation | early | rmsce | 30.300719 | 21.386577 | -8.914142 |
| IEMOCAP | 1937 | validation | early | auroc_success | 61.035806 | 61.701299 | 0.665493 |
| IEMOCAP | 1937 | validation | early | auprc_success | 57.139511 | 63.918580 | 6.779069 |
| IEMOCAP | 1937 | validation | middle | ece | 18.833274 | 15.717175 | -3.116099 |
| IEMOCAP | 1937 | validation | middle | mce | 25.491246 | 30.952606 | 5.461360 |
| IEMOCAP | 1937 | validation | middle | rmsce | 20.515495 | 18.637629 | -1.877867 |
| IEMOCAP | 1937 | validation | middle | auroc_success | 61.624528 | 59.826840 | -1.797688 |
| IEMOCAP | 1937 | validation | middle | auprc_success | 71.404248 | 67.472669 | -3.931579 |
| IEMOCAP | 1937 | validation | late | ece | 18.390918 | 14.810954 | -3.579964 |
| IEMOCAP | 1937 | validation | late | mce | 60.792953 | 23.268767 | -37.524186 |
| IEMOCAP | 1937 | validation | late | rmsce | 19.998344 | 16.444992 | -3.553352 |
| IEMOCAP | 1937 | validation | late | auroc_success | 65.778424 | 61.547727 | -4.230697 |
| IEMOCAP | 1937 | validation | late | auprc_success | 82.973510 | 75.891444 | -7.082065 |
| IEMOCAP | 1937 | test | overall | ece | 18.520027 | 14.550467 | -3.969560 |
| IEMOCAP | 1937 | test | overall | mce | 33.974854 | 24.416502 | -9.558352 |
| IEMOCAP | 1937 | test | overall | rmsce | 19.257726 | 15.614315 | -3.643410 |
| IEMOCAP | 1937 | test | overall | auroc_success | 67.208061 | 67.064640 | -0.143421 |
| IEMOCAP | 1937 | test | overall | auprc_success | 78.989156 | 79.212895 | 0.223739 |
| IEMOCAP | 1937 | test | early | ece | 20.460856 | 17.498728 | -2.962127 |
| IEMOCAP | 1937 | test | early | mce | 66.902366 | 31.365658 | -35.536708 |
| IEMOCAP | 1937 | test | early | rmsce | 21.814851 | 18.626813 | -3.188038 |
| IEMOCAP | 1937 | test | early | auroc_success | 67.041608 | 68.942280 | 1.900672 |
| IEMOCAP | 1937 | test | early | auprc_success | 76.402023 | 78.220531 | 1.818507 |
| IEMOCAP | 1937 | test | middle | ece | 16.690405 | 13.442743 | -3.247662 |
| IEMOCAP | 1937 | test | middle | mce | 67.169711 | 68.458299 | 1.288588 |
| IEMOCAP | 1937 | test | middle | rmsce | 18.014213 | 15.964131 | -2.050083 |
| IEMOCAP | 1937 | test | middle | auroc_success | 64.998482 | 65.547529 | 0.549047 |
| IEMOCAP | 1937 | test | middle | auprc_success | 78.327375 | 79.850398 | 1.523023 |
| IEMOCAP | 1937 | test | late | ece | 19.239080 | 12.948422 | -6.290659 |
| IEMOCAP | 1937 | test | late | mce | 32.147515 | 20.296182 | -11.851332 |
| IEMOCAP | 1937 | test | late | rmsce | 19.858145 | 14.625342 | -5.232802 |
| IEMOCAP | 1937 | test | late | auroc_success | 69.704096 | 66.156169 | -3.547927 |
| IEMOCAP | 1937 | test | late | auprc_success | 81.830587 | 80.162106 | -1.668481 |
| IEMOCAP | 2025 | validation | overall | ece | 25.104901 | 17.040716 | -8.064185 |
| IEMOCAP | 2025 | validation | overall | mce | 72.381660 | 35.558850 | -36.822811 |
| IEMOCAP | 2025 | validation | overall | rmsce | 26.619432 | 18.676318 | -7.943115 |
| IEMOCAP | 2025 | validation | overall | auroc_success | 59.945587 | 63.303859 | 3.358271 |
| IEMOCAP | 2025 | validation | overall | auprc_success | 68.382188 | 69.069600 | 0.687411 |
| IEMOCAP | 2025 | validation | early | ece | 32.906397 | 24.461327 | -8.445070 |
| IEMOCAP | 2025 | validation | early | mce | 72.381660 | 70.311424 | -2.070236 |
| IEMOCAP | 2025 | validation | early | rmsce | 35.579754 | 28.109277 | -7.470477 |
| IEMOCAP | 2025 | validation | early | auroc_success | 56.538953 | 58.273749 | 1.734796 |
| IEMOCAP | 2025 | validation | early | auprc_success | 56.391896 | 55.684780 | -0.707116 |
| IEMOCAP | 2025 | validation | middle | ece | 25.893720 | 15.645891 | -10.247829 |
| IEMOCAP | 2025 | validation | middle | mce | 63.359803 | 36.234181 | -27.125622 |
| IEMOCAP | 2025 | validation | middle | rmsce | 27.738266 | 17.076124 | -10.662142 |
| IEMOCAP | 2025 | validation | middle | auroc_success | 59.077259 | 66.475566 | 7.398307 |
| IEMOCAP | 2025 | validation | middle | auprc_success | 64.588955 | 68.666695 | 4.077740 |
| IEMOCAP | 2025 | validation | late | ece | 18.334338 | 15.949631 | -2.384707 |
| IEMOCAP | 2025 | validation | late | mce | 44.218093 | 60.825783 | 16.607690 |
| IEMOCAP | 2025 | validation | late | rmsce | 20.035571 | 17.694387 | -2.341184 |
| IEMOCAP | 2025 | validation | late | auroc_success | 61.781090 | 62.195122 | 0.414032 |
| IEMOCAP | 2025 | validation | late | auprc_success | 78.632585 | 77.639469 | -0.993116 |
| IEMOCAP | 2025 | test | overall | ece | 21.962087 | 12.713291 | -9.248795 |
| IEMOCAP | 2025 | test | overall | mce | 34.381594 | 19.665141 | -14.716452 |
| IEMOCAP | 2025 | test | overall | rmsce | 22.751155 | 13.886603 | -8.864552 |
| IEMOCAP | 2025 | test | overall | auroc_success | 67.536968 | 66.472315 | -1.064653 |
| IEMOCAP | 2025 | test | overall | auprc_success | 79.279014 | 79.341325 | 0.062311 |
| IEMOCAP | 2025 | test | early | ece | 24.096958 | 14.793629 | -9.303329 |
| IEMOCAP | 2025 | test | early | mce | 45.106721 | 30.841091 | -14.265630 |
| IEMOCAP | 2025 | test | early | rmsce | 25.738136 | 15.760495 | -9.977641 |
| IEMOCAP | 2025 | test | early | auroc_success | 61.079592 | 68.794117 | 7.714526 |
| IEMOCAP | 2025 | test | early | auprc_success | 74.069936 | 78.725144 | 4.655208 |
| IEMOCAP | 2025 | test | middle | ece | 20.466707 | 13.484737 | -6.981970 |
| IEMOCAP | 2025 | test | middle | mce | 67.914721 | 32.592553 | -35.322168 |
| IEMOCAP | 2025 | test | middle | rmsce | 21.562270 | 14.445424 | -7.116846 |
| IEMOCAP | 2025 | test | middle | auroc_success | 67.040197 | 64.539892 | -2.500305 |
| IEMOCAP | 2025 | test | middle | auprc_success | 79.972828 | 78.778007 | -1.194821 |
| IEMOCAP | 2025 | test | late | ece | 21.917091 | 12.997069 | -8.920021 |
| IEMOCAP | 2025 | test | late | mce | 42.180458 | 29.480251 | -12.700207 |
| IEMOCAP | 2025 | test | late | rmsce | 22.836867 | 13.748414 | -9.088454 |
| IEMOCAP | 2025 | test | late | auroc_success | 74.064706 | 65.934929 | -8.129777 |
| IEMOCAP | 2025 | test | late | auprc_success | 84.045776 | 80.820755 | -3.225020 |
| IEMOCAP | 3407 | validation | overall | ece | 19.405296 | 17.884412 | -1.520884 |
| IEMOCAP | 3407 | validation | overall | mce | 25.911030 | 35.436281 | 9.525251 |
| IEMOCAP | 3407 | validation | overall | rmsce | 20.616167 | 21.174045 | 0.557878 |
| IEMOCAP | 3407 | validation | overall | auroc_success | 63.400514 | 62.678406 | -0.722108 |
| IEMOCAP | 3407 | validation | overall | auprc_success | 73.434823 | 70.663731 | -2.771092 |
| IEMOCAP | 3407 | validation | early | ece | 23.542997 | 24.594171 | 1.051174 |
| IEMOCAP | 3407 | validation | early | mce | 41.409112 | 56.967259 | 15.558147 |
| IEMOCAP | 3407 | validation | early | rmsce | 26.134164 | 29.471482 | 3.337318 |
| IEMOCAP | 3407 | validation | early | auroc_success | 60.571429 | 63.079897 | 2.508468 |
| IEMOCAP | 3407 | validation | early | auprc_success | 63.251927 | 63.492476 | 0.240549 |
| IEMOCAP | 3407 | validation | middle | ece | 17.852474 | 19.152642 | 1.300168 |
| IEMOCAP | 3407 | validation | middle | mce | 26.415162 | 30.981009 | 4.565847 |
| IEMOCAP | 3407 | validation | middle | rmsce | 20.057412 | 21.227784 | 1.170372 |
| IEMOCAP | 3407 | validation | middle | auroc_success | 61.504684 | 62.728809 | 1.224125 |
| IEMOCAP | 3407 | validation | middle | auprc_success | 71.926573 | 68.791173 | -3.135399 |
| IEMOCAP | 3407 | validation | late | ece | 19.328578 | 17.714312 | -1.614266 |
| IEMOCAP | 3407 | validation | late | mce | 35.993873 | 43.883777 | 7.889903 |
| IEMOCAP | 3407 | validation | late | rmsce | 20.510686 | 19.200195 | -1.310491 |
| IEMOCAP | 3407 | validation | late | auroc_success | 65.923849 | 58.397933 | -7.525916 |
| IEMOCAP | 3407 | validation | late | auprc_success | 80.747444 | 76.543462 | -4.203982 |
| IEMOCAP | 3407 | test | overall | ece | 14.071656 | 19.248737 | 5.177081 |
| IEMOCAP | 3407 | test | overall | mce | 27.699423 | 28.275761 | 0.576338 |
| IEMOCAP | 3407 | test | overall | rmsce | 15.867242 | 20.117347 | 4.250106 |
| IEMOCAP | 3407 | test | overall | auroc_success | 65.213929 | 67.762833 | 2.548904 |
| IEMOCAP | 3407 | test | overall | auprc_success | 78.939522 | 79.421352 | 0.481830 |
| IEMOCAP | 3407 | test | early | ece | 17.307642 | 21.600751 | 4.293109 |
| IEMOCAP | 3407 | test | early | mce | 32.620012 | 72.363633 | 39.743621 |
| IEMOCAP | 3407 | test | early | rmsce | 19.357851 | 23.525251 | 4.167400 |
| IEMOCAP | 3407 | test | early | auroc_success | 64.006748 | 70.114455 | 6.107707 |
| IEMOCAP | 3407 | test | early | auprc_success | 75.570085 | 79.599924 | 4.029839 |
| IEMOCAP | 3407 | test | middle | ece | 13.416499 | 20.040806 | 6.624307 |
| IEMOCAP | 3407 | test | middle | mce | 37.048179 | 29.138986 | -7.909193 |
| IEMOCAP | 3407 | test | middle | rmsce | 15.499762 | 21.007990 | 5.508228 |
| IEMOCAP | 3407 | test | middle | auroc_success | 66.484540 | 67.886155 | 1.401615 |
| IEMOCAP | 3407 | test | middle | auprc_success | 78.790547 | 79.018550 | 0.228004 |
| IEMOCAP | 3407 | test | late | ece | 13.780557 | 16.929057 | 3.148500 |
| IEMOCAP | 3407 | test | late | mce | 67.305613 | 32.848954 | -34.456658 |
| IEMOCAP | 3407 | test | late | rmsce | 15.529533 | 18.524340 | 2.994807 |
| IEMOCAP | 3407 | test | late | auroc_success | 65.388018 | 65.175359 | -0.212659 |
| IEMOCAP | 3407 | test | late | auprc_success | 81.871202 | 80.050470 | -1.820732 |
| MELD | 1937 | validation | overall | ece | 20.542450 | 19.125483 | -1.416967 |
| MELD | 1937 | validation | overall | mce | 26.844391 | 26.707613 | -0.136778 |
| MELD | 1937 | validation | overall | rmsce | 21.658764 | 19.935583 | -1.723181 |
| MELD | 1937 | validation | overall | auroc_success | 70.762934 | 70.619253 | -0.143681 |
| MELD | 1937 | validation | overall | auprc_success | 79.373921 | 80.664448 | 1.290527 |
| MELD | 1937 | validation | early | ece | 23.919186 | 22.654974 | -1.264212 |
| MELD | 1937 | validation | early | mce | 29.991687 | 34.112903 | 4.121216 |
| MELD | 1937 | validation | early | rmsce | 24.940376 | 23.607418 | -1.332958 |
| MELD | 1937 | validation | early | auroc_success | 71.364221 | 72.014625 | 0.650404 |
| MELD | 1937 | validation | early | auprc_success | 82.420335 | 83.561283 | 1.140948 |
| MELD | 1937 | validation | middle | ece | 19.378941 | 16.975485 | -2.403456 |
| MELD | 1937 | validation | middle | mce | 32.959616 | 31.257584 | -1.702032 |
| MELD | 1937 | validation | middle | rmsce | 21.460301 | 19.450851 | -2.009450 |
| MELD | 1937 | validation | middle | auroc_success | 72.686891 | 75.116051 | 2.429160 |
| MELD | 1937 | validation | middle | auprc_success | 83.325986 | 85.082252 | 1.756266 |
| MELD | 1937 | validation | late | ece | 18.474758 | 17.816790 | -0.657968 |
| MELD | 1937 | validation | late | mce | 29.475969 | 25.753600 | -3.722369 |
| MELD | 1937 | validation | late | rmsce | 20.094569 | 18.666780 | -1.427790 |
| MELD | 1937 | validation | late | auroc_success | 68.251619 | 64.721444 | -3.530175 |
| MELD | 1937 | validation | late | auprc_success | 73.382287 | 73.694201 | 0.311914 |
| MELD | 1937 | test | overall | ece | 19.883207 | 19.185048 | -0.698159 |
| MELD | 1937 | test | overall | mce | 81.752169 | 26.782200 | -54.969969 |
| MELD | 1937 | test | overall | rmsce | 22.121570 | 20.699546 | -1.422024 |
| MELD | 1937 | test | overall | auroc_success | 75.684105 | 73.822607 | -1.861498 |
| MELD | 1937 | test | overall | auprc_success | 83.422704 | 83.074037 | -0.348668 |
| MELD | 1937 | test | early | ece | 20.885164 | 22.253107 | 1.367943 |
| MELD | 1937 | test | early | mce | 66.992611 | 29.549682 | -37.442929 |
| MELD | 1937 | test | early | rmsce | 23.573155 | 23.523045 | -0.050111 |
| MELD | 1937 | test | early | auroc_success | 76.474470 | 71.884926 | -4.589544 |
| MELD | 1937 | test | early | auprc_success | 83.008844 | 82.657219 | -0.351625 |
| MELD | 1937 | test | middle | ece | 19.046124 | 18.097462 | -0.948663 |
| MELD | 1937 | test | middle | mce | 32.053615 | 27.572463 | -4.481152 |
| MELD | 1937 | test | middle | rmsce | 21.013059 | 20.236617 | -0.776443 |
| MELD | 1937 | test | middle | auroc_success | 75.112931 | 76.487373 | 1.374442 |
| MELD | 1937 | test | middle | auprc_success | 83.414227 | 84.302051 | 0.887824 |
| MELD | 1937 | test | late | ece | 20.192077 | 17.796781 | -2.395297 |
| MELD | 1937 | test | late | mce | 81.752169 | 31.443039 | -50.309130 |
| MELD | 1937 | test | late | rmsce | 22.443764 | 19.451244 | -2.992520 |
| MELD | 1937 | test | late | auroc_success | 75.467785 | 73.183052 | -2.284733 |
| MELD | 1937 | test | late | auprc_success | 83.701071 | 82.554680 | -1.146390 |
| MELD | 2025 | validation | overall | ece | 20.799505 | 20.975420 | 0.175915 |
| MELD | 2025 | validation | overall | mce | 31.543480 | 31.334586 | -0.208895 |
| MELD | 2025 | validation | overall | rmsce | 21.517265 | 21.855834 | 0.338569 |
| MELD | 2025 | validation | overall | auroc_success | 71.488154 | 71.473373 | -0.014782 |
| MELD | 2025 | validation | overall | auprc_success | 82.360420 | 82.728840 | 0.368420 |
| MELD | 2025 | validation | early | ece | 23.682480 | 22.839527 | -0.842954 |
| MELD | 2025 | validation | early | mce | 31.464062 | 31.157171 | -0.306891 |
| MELD | 2025 | validation | early | rmsce | 24.431762 | 23.930576 | -0.501186 |
| MELD | 2025 | validation | early | auroc_success | 73.915572 | 72.765499 | -1.150073 |
| MELD | 2025 | validation | early | auprc_success | 86.223864 | 85.963152 | -0.260713 |
| MELD | 2025 | validation | middle | ece | 20.907827 | 21.510762 | 0.602935 |
| MELD | 2025 | validation | middle | mce | 37.490965 | 34.202626 | -3.288339 |
| MELD | 2025 | validation | middle | rmsce | 22.058801 | 23.512315 | 1.453514 |
| MELD | 2025 | validation | middle | auroc_success | 71.954289 | 72.846365 | 0.892076 |
| MELD | 2025 | validation | middle | auprc_success | 84.069980 | 83.502327 | -0.567653 |
| MELD | 2025 | validation | late | ece | 18.148051 | 18.656844 | 0.508793 |
| MELD | 2025 | validation | late | mce | 31.393179 | 82.159269 | 50.766090 |
| MELD | 2025 | validation | late | rmsce | 19.367725 | 19.797926 | 0.430201 |
| MELD | 2025 | validation | late | auroc_success | 68.794195 | 68.513722 | -0.280474 |
| MELD | 2025 | validation | late | auprc_success | 76.730722 | 77.727426 | 0.996703 |
| MELD | 2025 | test | overall | ece | 19.794715 | 18.400303 | -1.394412 |
| MELD | 2025 | test | overall | mce | 31.384727 | 30.690924 | -0.693803 |
| MELD | 2025 | test | overall | rmsce | 21.552970 | 19.913380 | -1.639590 |
| MELD | 2025 | test | overall | auroc_success | 74.989863 | 74.926200 | -0.063663 |
| MELD | 2025 | test | overall | auprc_success | 84.132992 | 84.343820 | 0.210828 |
| MELD | 2025 | test | early | ece | 21.499791 | 18.254795 | -3.244996 |
| MELD | 2025 | test | early | mce | 31.863489 | 80.495574 | 48.632085 |
| MELD | 2025 | test | early | rmsce | 23.114521 | 20.251266 | -2.863254 |
| MELD | 2025 | test | early | auroc_success | 76.884968 | 76.693948 | -0.191020 |
| MELD | 2025 | test | early | auprc_success | 86.420631 | 86.929497 | 0.508867 |
| MELD | 2025 | test | middle | ece | 19.278219 | 18.082228 | -1.195991 |
| MELD | 2025 | test | middle | mce | 31.755998 | 28.112612 | -3.643386 |
| MELD | 2025 | test | middle | rmsce | 21.105910 | 19.848804 | -1.257105 |
| MELD | 2025 | test | middle | auroc_success | 73.758628 | 75.041085 | 1.282457 |
| MELD | 2025 | test | middle | auprc_success | 82.495749 | 83.640359 | 1.144610 |
| MELD | 2025 | test | late | ece | 19.348197 | 18.902121 | -0.446076 |
| MELD | 2025 | test | late | mce | 31.037863 | 31.903135 | 0.865272 |
| MELD | 2025 | test | late | rmsce | 21.158380 | 20.279988 | -0.878392 |
| MELD | 2025 | test | late | auroc_success | 74.240137 | 73.070845 | -1.169293 |
| MELD | 2025 | test | late | auprc_success | 83.731645 | 81.877887 | -1.853757 |
| MELD | 3407 | validation | overall | ece | 25.008241 | 15.967028 | -9.041213 |
| MELD | 3407 | validation | overall | mce | 35.728990 | 21.498943 | -14.230047 |
| MELD | 3407 | validation | overall | rmsce | 26.296412 | 16.540431 | -9.755981 |
| MELD | 3407 | validation | overall | auroc_success | 71.954675 | 71.941236 | -0.013440 |
| MELD | 3407 | validation | overall | auprc_success | 82.069320 | 81.618303 | -0.451017 |
| MELD | 3407 | validation | early | ece | 27.023400 | 18.453869 | -8.569531 |
| MELD | 3407 | validation | early | mce | 41.180936 | 26.446279 | -14.734657 |
| MELD | 3407 | validation | early | rmsce | 28.724457 | 19.212439 | -9.512019 |
| MELD | 3407 | validation | early | auroc_success | 75.055612 | 74.529828 | -0.525784 |
| MELD | 3407 | validation | early | auprc_success | 86.433423 | 85.990182 | -0.443241 |
| MELD | 3407 | validation | middle | ece | 25.918952 | 17.212404 | -8.706548 |
| MELD | 3407 | validation | middle | mce | 41.095495 | 35.336399 | -5.759096 |
| MELD | 3407 | validation | middle | rmsce | 27.031816 | 18.798152 | -8.233664 |
| MELD | 3407 | validation | middle | auroc_success | 69.855201 | 70.576832 | 0.721630 |
| MELD | 3407 | validation | middle | auprc_success | 82.040443 | 83.604887 | 1.564444 |
| MELD | 3407 | validation | late | ece | 22.106786 | 14.047853 | -8.058933 |
| MELD | 3407 | validation | late | mce | 31.500662 | 25.687990 | -5.812672 |
| MELD | 3407 | validation | late | rmsce | 24.053582 | 15.874684 | -8.178898 |
| MELD | 3407 | validation | late | auroc_success | 70.530692 | 70.399399 | -0.131293 |
| MELD | 3407 | validation | late | auprc_success | 77.297253 | 74.560356 | -2.736897 |
| MELD | 3407 | test | overall | ece | 25.013815 | 16.231511 | -8.782305 |
| MELD | 3407 | test | overall | mce | 36.590477 | 80.765527 | 44.175050 |
| MELD | 3407 | test | overall | rmsce | 26.961842 | 17.070389 | -9.891453 |
| MELD | 3407 | test | overall | auroc_success | 75.090684 | 74.278291 | -0.812393 |
| MELD | 3407 | test | overall | auprc_success | 84.950117 | 84.176727 | -0.773390 |
| MELD | 3407 | test | early | ece | 25.761394 | 16.556223 | -9.205170 |
| MELD | 3407 | test | early | mce | 40.117679 | 80.765527 | 40.647848 |
| MELD | 3407 | test | early | rmsce | 27.474712 | 17.445399 | -10.029313 |
| MELD | 3407 | test | early | auroc_success | 75.510742 | 74.014557 | -1.496185 |
| MELD | 3407 | test | early | auprc_success | 86.425976 | 84.795136 | -1.630840 |
| MELD | 3407 | test | middle | ece | 24.611103 | 15.634243 | -8.976860 |
| MELD | 3407 | test | middle | mce | 34.250376 | 24.072833 | -10.177543 |
| MELD | 3407 | test | middle | rmsce | 26.854390 | 17.124222 | -9.730168 |
| MELD | 3407 | test | middle | auroc_success | 74.905050 | 76.124217 | 1.219166 |
| MELD | 3407 | test | middle | auprc_success | 84.095148 | 84.435579 | 0.340430 |
| MELD | 3407 | test | late | ece | 24.662046 | 16.802748 | -7.859298 |
| MELD | 3407 | test | late | mce | 37.742054 | 80.298787 | 42.556734 |
| MELD | 3407 | test | late | rmsce | 26.788454 | 18.172868 | -8.615587 |
| MELD | 3407 | test | late | auroc_success | 74.951747 | 72.565895 | -2.385852 |
| MELD | 3407 | test | late | auprc_success | 84.485743 | 83.542370 | -0.943373 |

| Dataset | Split | Segment | Metric | n | Baseline mean ± SD | CMERC mean ± SD | Paired delta mean ± SD | 95% t interval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | test | early | auprc_success | 3 | 75.347348 ± 1.181891 | 78.848533 ± 0.697926 | 3.501185 ± 1.490411 | [-0.201201, 7.203570] |
| IEMOCAP | test | early | auroc_success | 3 | 64.042649 ± 2.981171 | 69.283617 ± 0.723330 | 5.240968 ± 3.002274 | [-2.217095, 12.699031] |
| IEMOCAP | test | early | ece | 3 | 20.621819 ± 3.397519 | 17.964370 ± 3.427367 | -2.657449 ± 6.803338 | [-19.557877, 14.242979] |
| IEMOCAP | test | early | mce | 3 | 48.209700 ± 17.350542 | 44.856794 ± 23.823065 | -3.352905 ± 38.808474 | [-99.758499, 93.052688] |
| IEMOCAP | test | early | rmsce | 3 | 22.303613 ± 3.218101 | 19.304186 ± 3.926447 | -2.999426 ± 7.074406 | [-20.573226, 14.574373] |
| IEMOCAP | test | late | auprc_success | 3 | 82.582522 ± 1.267378 | 80.344444 ± 0.416257 | -2.238078 ± 0.858101 | [-4.369718, -0.106438] |
| IEMOCAP | test | late | auroc_success | 3 | 69.718940 ± 4.338363 | 65.755486 ± 0.514438 | -3.963454 ± 3.974882 | [-13.837609, 5.910700] |
| IEMOCAP | test | late | ece | 3 | 18.312243 ± 4.146693 | 14.291516 ± 2.284307 | -4.020727 ± 6.346396 | [-19.786048, 11.744595] |
| IEMOCAP | test | late | mce | 3 | 47.211195 ± 18.110886 | 27.541796 ± 6.497017 | -19.669399 ± 12.813174 | [-51.499087, 12.160289] |
| IEMOCAP | test | late | rmsce | 3 | 19.408182 ± 3.674389 | 15.632699 ± 2.542330 | -3.775483 ± 6.172045 | [-19.107692, 11.556725] |
| IEMOCAP | test | middle | auprc_success | 3 | 79.030250 ± 0.848512 | 79.215652 ± 0.562710 | 0.185402 ± 1.359423 | [-3.191591, 3.562395] |
| IEMOCAP | test | middle | auroc_success | 3 | 66.174406 ± 1.055598 | 65.991192 ± 1.716682 | -0.183214 ± 2.051438 | [-5.279270, 4.912841] |
| IEMOCAP | test | middle | ece | 3 | 16.857870 ± 3.528086 | 15.656095 ± 3.797329 | -1.201775 ± 7.030073 | [-18.665446, 16.261896] |
| IEMOCAP | test | middle | mce | 3 | 57.377537 ± 17.609681 | 43.396613 ± 21.772641 | -13.980924 ± 19.045638 | [-61.292912, 33.331063] |
| IEMOCAP | test | middle | rmsce | 3 | 18.358748 ± 3.045904 | 17.139182 ± 3.435458 | -1.219567 ± 6.353380 | [-17.002238, 14.563105] |
| IEMOCAP | test | overall | auprc_success | 3 | 79.069231 ± 0.183364 | 79.325191 ± 0.105161 | 0.255960 ± 0.211607 | [-0.269702, 0.781622] |
| IEMOCAP | test | overall | auroc_success | 3 | 66.652986 ± 1.257064 | 67.099929 ± 0.645982 | 0.446943 ± 1.877724 | [-4.217582, 5.111468] |
| IEMOCAP | test | overall | ece | 3 | 18.184590 ± 3.955896 | 15.504165 ± 3.370485 | -2.680425 ± 7.298827 | [-20.811717, 15.450868] |
| IEMOCAP | test | overall | mce | 3 | 32.018624 ± 3.746062 | 24.119135 ± 4.313005 | -7.899489 ± 7.780182 | [-27.226533, 11.427555] |
| IEMOCAP | test | overall | rmsce | 3 | 19.292041 ± 3.442085 | 16.539422 ± 3.216739 | -2.752619 ± 6.602552 | [-19.154267, 13.649030] |
| IEMOCAP | validation | early | auprc_success | 3 | 58.927778 ± 3.763434 | 61.031945 ± 4.635679 | 2.104167 ± 4.076217 | [-8.021718, 12.230052] |
| IEMOCAP | validation | early | auroc_success | 3 | 59.382062 ± 2.473129 | 61.018315 ± 2.474796 | 1.636252 ± 0.925431 | [-0.662646, 3.935151] |
| IEMOCAP | validation | early | ece | 3 | 28.418789 ± 4.693754 | 22.657916 ± 3.239326 | -5.760872 ± 5.943400 | [-20.525097, 9.003353] |
| IEMOCAP | validation | early | mce | 3 | 51.404830 ± 18.173148 | 67.022322 ± 8.879772 | 15.617492 ± 17.717475 | [-28.395157, 59.630141] |
| IEMOCAP | validation | early | rmsce | 3 | 30.671546 ± 4.733701 | 26.322445 ± 4.328510 | -4.349100 ± 6.695656 | [-20.982032, 12.283832] |
| IEMOCAP | validation | late | auprc_success | 3 | 80.784513 ± 2.170700 | 76.691459 ± 0.883360 | -4.093054 ± 3.045990 | [-11.659713, 3.473604] |
| IEMOCAP | validation | late | auroc_success | 3 | 64.494454 ± 2.350967 | 60.713594 ± 2.031377 | -3.780860 ± 3.989042 | [-13.690191, 6.128470] |
| IEMOCAP | validation | late | ece | 3 | 18.684611 ± 0.558409 | 16.158299 ± 1.462884 | -2.526312 ± 0.990470 | [-4.986776, -0.065849] |
| IEMOCAP | validation | late | mce | 3 | 47.001640 ± 12.631694 | 42.659442 ± 18.808418 | -4.342197 ± 29.065154 | [-76.544044, 67.859649] |
| IEMOCAP | validation | late | rmsce | 3 | 20.181534 ± 0.285661 | 17.779858 ± 1.379588 | -2.401676 ± 1.122653 | [-5.190501, 0.387149] |
| IEMOCAP | validation | middle | auprc_success | 3 | 69.306592 ± 4.093932 | 68.310179 ± 0.727970 | -0.996413 ± 4.412340 | [-11.957273, 9.964447] |
| IEMOCAP | validation | middle | auroc_success | 3 | 60.735490 ± 1.437320 | 63.010405 ± 3.333296 | 2.274915 ± 4.687185 | [-9.368698, 13.918528] |
| IEMOCAP | validation | middle | ece | 3 | 20.859823 ± 4.386979 | 16.838569 ± 2.004363 | -4.021254 ± 5.826966 | [-18.496240, 10.453733] |
| IEMOCAP | validation | middle | mce | 3 | 38.422070 ± 21.601650 | 32.722598 ± 3.041153 | -5.699472 ± 18.560992 | [-51.807532, 40.408589] |
| IEMOCAP | validation | middle | rmsce | 3 | 22.770391 ± 4.308399 | 18.980512 ± 2.096961 | -3.789879 ± 6.143610 | [-19.051451, 11.471694] |
| IEMOCAP | validation | overall | auprc_success | 3 | 71.662812 ± 2.844154 | 70.014210 ± 0.837031 | -1.648602 ± 2.023559 | [-6.675402, 3.378197] |
| IEMOCAP | validation | overall | auroc_success | 3 | 62.493186 ± 2.236514 | 62.683850 ± 0.617304 | 0.190664 ± 2.824108 | [-6.824809, 7.206137] |
| IEMOCAP | validation | overall | ece | 3 | 21.959176 ± 2.895528 | 16.713749 ± 1.363865 | -5.245427 ± 3.364377 | [-13.603002, 3.112148] |
| IEMOCAP | validation | overall | mce | 3 | 42.217329 ± 26.151743 | 48.261138 ± 22.107241 | 6.043809 ± 41.236268 | [-96.392761, 108.480378] |
| IEMOCAP | validation | overall | rmsce | 3 | 23.149330 ± 3.109371 | 19.024329 ± 1.998566 | -4.125001 ± 4.315968 | [-14.846461, 6.596459] |
| MELD | test | early | auprc_success | 3 | 85.285150 ± 1.971341 | 84.793951 ± 2.136139 | -0.491199 ± 1.076660 | [-3.165771, 2.183372] |
| MELD | test | early | auroc_success | 3 | 76.290060 ± 0.705429 | 74.197810 ± 2.409743 | -2.092250 ± 2.259031 | [-7.703994, 3.519495] |
| MELD | test | early | ece | 3 | 22.715450 ± 2.655706 | 19.021375 ± 2.924783 | -3.694075 ± 5.300843 | [-16.862098, 9.473949] |
| MELD | test | early | mce | 3 | 46.324593 ± 18.368673 | 63.603594 ± 29.491862 | 17.279001 ± 47.558430 | [-100.862687, 135.420690] |
| MELD | test | early | rmsce | 3 | 24.720796 ± 2.395960 | 20.406570 ± 3.041798 | -4.314226 ± 5.145397 | [-17.096100, 8.467648] |
| MELD | test | late | auprc_success | 3 | 83.972819 ± 0.444468 | 82.658313 ± 0.837067 | -1.314507 ± 0.477909 | [-2.501699, -0.127314] |
| MELD | test | late | auroc_success | 3 | 74.886556 ± 0.616415 | 72.939931 ± 0.328747 | -1.946626 ± 0.675086 | [-3.623633, -0.269619] |
| MELD | test | late | ece | 3 | 21.400774 ± 2.855689 | 17.833883 ± 1.050178 | -3.566890 ± 3.842973 | [-13.113364, 5.979584] |
| MELD | test | late | mce | 3 | 50.177362 ± 27.549281 | 47.881654 ± 28.075004 | -2.295708 ± 46.513557 | [-117.841790, 113.250373] |
| MELD | test | late | rmsce | 3 | 23.463533 ± 2.950319 | 19.301367 ± 1.061526 | -4.162166 ± 3.999013 | [-14.096264, 5.771932] |
| MELD | test | middle | auprc_success | 3 | 83.335041 ± 0.802635 | 84.125996 ± 0.425840 | 0.790955 ± 0.410748 | [-0.229400, 1.811309] |
| MELD | test | middle | auroc_success | 3 | 74.592203 ± 0.729341 | 75.884225 ± 0.752419 | 1.292022 ± 0.078078 | [1.098064, 1.485979] |
| MELD | test | middle | ece | 3 | 20.978482 ± 3.148082 | 17.271311 ± 1.417763 | -3.707171 ± 4.565360 | [-15.048153, 7.633811] |
| MELD | test | middle | mce | 3 | 32.686663 ± 1.362367 | 26.585969 ± 2.193132 | -6.100694 ± 3.555417 | [-14.932839, 2.731451] |
| MELD | test | middle | rmsce | 3 | 22.991120 ± 3.346012 | 19.069881 ± 1.696111 | -3.921239 ± 5.036418 | [-16.432394, 8.589917] |
| MELD | test | overall | auprc_success | 3 | 84.168604 ± 0.764329 | 83.864861 ± 0.689951 | -0.303743 ± 0.493644 | [-1.530024, 0.922537] |
| MELD | test | overall | auroc_success | 3 | 75.254884 ± 0.375119 | 74.342366 ± 0.554580 | -0.912518 ± 0.903090 | [-3.155917, 1.330882] |
| MELD | test | overall | ece | 3 | 21.563912 ± 2.988031 | 17.938954 ± 1.529862 | -3.624958 ± 4.479939 | [-14.753745, 7.503828] |
| MELD | test | overall | mce | 3 | 49.909124 ± 27.699451 | 46.079550 ± 30.102446 | -3.829574 ± 49.646838 | [-127.159157, 119.500009] |
| MELD | test | overall | rmsce | 3 | 23.545461 ± 2.972301 | 19.227772 ± 1.909251 | -4.317689 ± 4.828247 | [-16.311719, 7.676341] |
| MELD | validation | early | auprc_success | 3 | 85.025874 ± 2.258894 | 85.171539 ± 1.394588 | 0.145665 ± 0.866759 | [-2.007483, 2.298812] |
| MELD | validation | early | auroc_success | 3 | 73.445135 ± 1.890125 | 73.103317 ± 1.291183 | -0.341818 ± 0.914227 | [-2.612884, 1.929249] |
| MELD | validation | early | ece | 3 | 24.875022 ± 1.864311 | 21.316123 ± 2.480502 | -3.558899 ± 4.344444 | [-14.351095, 7.233298] |
| MELD | validation | early | mce | 3 | 34.212228 ± 6.079814 | 30.572118 ± 3.866652 | -3.640111 ± 9.859957 | [-28.133602, 20.853381] |
| MELD | validation | early | rmsce | 3 | 26.032198 ± 2.345393 | 22.250144 ± 2.635688 | -3.782054 ± 4.979692 | [-16.152295, 8.588186] |
| MELD | validation | late | auprc_success | 3 | 75.803421 ± 2.115811 | 75.327328 ± 2.123184 | -0.476093 ± 1.987627 | [-5.413631, 4.461445] |
| MELD | validation | late | auroc_success | 3 | 69.192169 ± 1.190517 | 67.878188 ± 2.891837 | -1.313980 ± 1.920730 | [-6.085337, 3.457376] |
| MELD | validation | late | ece | 3 | 19.576532 ± 2.197345 | 16.840496 ± 2.454702 | -2.736036 ± 4.646532 | [-14.278661, 8.806589] |
| MELD | validation | late | mce | 3 | 30.789937 ± 1.139198 | 44.533620 ± 32.584785 | 13.743683 ± 32.079375 | [-65.945901, 93.433267] |
| MELD | validation | late | rmsce | 3 | 21.171959 ± 2.521882 | 18.113130 ± 2.019370 | -3.058829 ± 4.530382 | [-14.312922, 8.195265] |
| MELD | validation | middle | auprc_success | 3 | 83.145470 ± 1.026740 | 84.063155 ± 0.884052 | 0.917685 ± 1.289912 | [-2.286632, 4.122003] |
| MELD | validation | middle | auroc_success | 3 | 71.498794 ± 1.469770 | 72.846416 ± 2.269610 | 1.347622 ± 0.940508 | [-0.988730, 3.683974] |
| MELD | validation | middle | ece | 3 | 22.068573 ± 3.421029 | 18.566217 ± 2.552801 | -3.502356 ± 4.751032 | [-15.304574, 8.299862] |
| MELD | validation | middle | mce | 3 | 37.182025 ± 4.076728 | 33.598870 ± 2.105368 | -3.583156 ± 2.044537 | [-8.662066, 1.495755] |
| MELD | validation | middle | rmsce | 3 | 23.516973 ± 3.058618 | 20.587106 ± 2.554239 | -2.929867 ± 4.908740 | [-15.123853, 9.264119] |
| MELD | validation | overall | auprc_success | 3 | 81.267887 ± 1.646668 | 81.670530 ± 1.033186 | 0.402644 ± 0.871276 | [-1.761727, 2.567014] |
| MELD | validation | overall | auroc_success | 3 | 71.401921 ± 0.600532 | 71.344620 ± 0.670330 | -0.057301 ± 0.074811 | [-0.243141, 0.128539] |
| MELD | validation | overall | ece | 3 | 22.116732 ± 2.507416 | 18.689310 ± 2.532525 | -3.427422 ± 4.926491 | [-15.665503, 8.810659] |
| MELD | validation | overall | mce | 3 | 31.372287 ± 4.444773 | 26.513714 ± 4.920687 | -4.858573 ± 8.116014 | [-25.019871, 15.302725] |
| MELD | validation | overall | rmsce | 3 | 23.157480 ± 2.719315 | 19.443949 ± 2.691590 | -3.713531 ± 5.333489 | [-16.962653, 9.535591] |

## Supplemental CEL per utterance

CEL remains a summed primary metric. Per-utterance values diagnose sample-count scale only; cross-dataset differences do not establish better causal calibration.

| Dataset | Seed | Split | Utterances | Metric | Baseline / utterance | CMERC / utterance | Delta / utterance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | 1937 | validation | 527 | cel.total | 0.929719 | 0.604793 | -0.324925 |
| IEMOCAP | 1937 | validation | 527 | cel.text | 0.183374 | 0.142897 | -0.040477 |
| IEMOCAP | 1937 | validation | 527 | cel.audio | 0.223574 | 0.232402 | 0.008828 |
| IEMOCAP | 1937 | validation | 527 | cel.visual | 0.011751 | 0.004019 | -0.007732 |
| IEMOCAP | 1937 | validation | 527 | cel.intra_context | 0.231786 | 0.074252 | -0.157534 |
| IEMOCAP | 1937 | validation | 527 | cel.inter_context | 0.279234 | 0.151224 | -0.128010 |
| IEMOCAP | 1937 | test | 1623 | cel.total | 0.917642 | 0.595594 | -0.322048 |
| IEMOCAP | 1937 | test | 1623 | cel.text | 0.166267 | 0.161341 | -0.004927 |
| IEMOCAP | 1937 | test | 1623 | cel.audio | 0.279566 | 0.175309 | -0.104257 |
| IEMOCAP | 1937 | test | 1623 | cel.visual | 0.010368 | 0.004760 | -0.005608 |
| IEMOCAP | 1937 | test | 1623 | cel.intra_context | 0.206422 | 0.111100 | -0.095322 |
| IEMOCAP | 1937 | test | 1623 | cel.inter_context | 0.255019 | 0.143085 | -0.111935 |
| IEMOCAP | 2025 | validation | 527 | cel.total | 1.367488 | 0.727516 | -0.639972 |
| IEMOCAP | 2025 | validation | 527 | cel.text | 0.252922 | 0.143164 | -0.109758 |
| IEMOCAP | 2025 | validation | 527 | cel.audio | 0.375284 | 0.221621 | -0.153663 |
| IEMOCAP | 2025 | validation | 527 | cel.visual | 0.011305 | 0.006358 | -0.004947 |
| IEMOCAP | 2025 | validation | 527 | cel.intra_context | 0.255336 | 0.132756 | -0.122580 |
| IEMOCAP | 2025 | validation | 527 | cel.inter_context | 0.472640 | 0.223617 | -0.249023 |
| IEMOCAP | 2025 | test | 1623 | cel.total | 1.130149 | 0.597808 | -0.532341 |
| IEMOCAP | 2025 | test | 1623 | cel.text | 0.223873 | 0.140102 | -0.083771 |
| IEMOCAP | 2025 | test | 1623 | cel.audio | 0.340743 | 0.155521 | -0.185223 |
| IEMOCAP | 2025 | test | 1623 | cel.visual | 0.011192 | 0.005651 | -0.005541 |
| IEMOCAP | 2025 | test | 1623 | cel.intra_context | 0.261677 | 0.133102 | -0.128575 |
| IEMOCAP | 2025 | test | 1623 | cel.inter_context | 0.292664 | 0.163431 | -0.129233 |
| IEMOCAP | 3407 | validation | 527 | cel.total | 0.902492 | 0.743014 | -0.159479 |
| IEMOCAP | 3407 | validation | 527 | cel.text | 0.130364 | 0.206337 | 0.075973 |
| IEMOCAP | 3407 | validation | 527 | cel.audio | 0.362819 | 0.266276 | -0.096543 |
| IEMOCAP | 3407 | validation | 527 | cel.visual | 0.011180 | 0.008855 | -0.002325 |
| IEMOCAP | 3407 | validation | 527 | cel.intra_context | 0.131734 | 0.141389 | 0.009656 |
| IEMOCAP | 3407 | validation | 527 | cel.inter_context | 0.266397 | 0.120156 | -0.146240 |
| IEMOCAP | 3407 | test | 1623 | cel.total | 0.774238 | 0.670300 | -0.103938 |
| IEMOCAP | 3407 | test | 1623 | cel.text | 0.135950 | 0.221819 | 0.085869 |
| IEMOCAP | 3407 | test | 1623 | cel.audio | 0.242973 | 0.197514 | -0.045458 |
| IEMOCAP | 3407 | test | 1623 | cel.visual | 0.008845 | 0.010351 | 0.001507 |
| IEMOCAP | 3407 | test | 1623 | cel.intra_context | 0.161444 | 0.109490 | -0.051953 |
| IEMOCAP | 3407 | test | 1623 | cel.inter_context | 0.225027 | 0.131126 | -0.093902 |
| MELD | 1937 | validation | 1109 | cel.total | 0.274418 | 0.235634 | -0.038784 |
| MELD | 1937 | validation | 1109 | cel.text | 0.108396 | 0.108265 | -0.000131 |
| MELD | 1937 | validation | 1109 | cel.audio | 0.013435 | 0.012655 | -0.000780 |
| MELD | 1937 | validation | 1109 | cel.visual | 0.005684 | 0.006588 | 0.000904 |
| MELD | 1937 | validation | 1109 | cel.intra_context | 0.063483 | 0.038200 | -0.025283 |
| MELD | 1937 | validation | 1109 | cel.inter_context | 0.083420 | 0.069926 | -0.013494 |
| MELD | 1937 | test | 2610 | cel.total | 0.304333 | 0.259473 | -0.044860 |
| MELD | 1937 | test | 2610 | cel.text | 0.127638 | 0.124831 | -0.002807 |
| MELD | 1937 | test | 2610 | cel.audio | 0.011893 | 0.010566 | -0.001327 |
| MELD | 1937 | test | 2610 | cel.visual | 0.004677 | 0.005414 | 0.000737 |
| MELD | 1937 | test | 2610 | cel.intra_context | 0.069793 | 0.040153 | -0.029640 |
| MELD | 1937 | test | 2610 | cel.inter_context | 0.090332 | 0.078509 | -0.011823 |
| MELD | 2025 | validation | 1109 | cel.total | 0.225358 | 0.255303 | 0.029945 |
| MELD | 2025 | validation | 1109 | cel.text | 0.107468 | 0.116626 | 0.009158 |
| MELD | 2025 | validation | 1109 | cel.audio | 0.013242 | 0.012164 | -0.001078 |
| MELD | 2025 | validation | 1109 | cel.visual | 0.006252 | 0.006623 | 0.000371 |
| MELD | 2025 | validation | 1109 | cel.intra_context | 0.034198 | 0.053244 | 0.019046 |
| MELD | 2025 | validation | 1109 | cel.inter_context | 0.064198 | 0.066646 | 0.002448 |
| MELD | 2025 | test | 2610 | cel.total | 0.246087 | 0.289140 | 0.043053 |
| MELD | 2025 | test | 2610 | cel.text | 0.124126 | 0.138977 | 0.014851 |
| MELD | 2025 | test | 2610 | cel.audio | 0.010969 | 0.009779 | -0.001190 |
| MELD | 2025 | test | 2610 | cel.visual | 0.005269 | 0.005350 | 0.000081 |
| MELD | 2025 | test | 2610 | cel.intra_context | 0.039012 | 0.061096 | 0.022084 |
| MELD | 2025 | test | 2610 | cel.inter_context | 0.066711 | 0.073938 | 0.007227 |
| MELD | 3407 | validation | 1109 | cel.total | 0.209036 | 0.245228 | 0.036192 |
| MELD | 3407 | validation | 1109 | cel.text | 0.091012 | 0.127333 | 0.036321 |
| MELD | 3407 | validation | 1109 | cel.audio | 0.011359 | 0.013701 | 0.002343 |
| MELD | 3407 | validation | 1109 | cel.visual | 0.005173 | 0.006672 | 0.001499 |
| MELD | 3407 | validation | 1109 | cel.intra_context | 0.035532 | 0.034728 | -0.000803 |
| MELD | 3407 | validation | 1109 | cel.inter_context | 0.065960 | 0.062792 | -0.003168 |
| MELD | 3407 | test | 2610 | cel.total | 0.227477 | 0.260155 | 0.032678 |
| MELD | 3407 | test | 2610 | cel.text | 0.104934 | 0.142395 | 0.037461 |
| MELD | 3407 | test | 2610 | cel.audio | 0.009955 | 0.011432 | 0.001477 |
| MELD | 3407 | test | 2610 | cel.visual | 0.004099 | 0.005364 | 0.001265 |
| MELD | 3407 | test | 2610 | cel.intra_context | 0.038748 | 0.035660 | -0.003088 |
| MELD | 3407 | test | 2610 | cel.inter_context | 0.069741 | 0.065303 | -0.004437 |

## ECE versus AUROC direction for every available seed

| Dataset | Seed | Split | ECE delta | AUROC-Success delta | ECE down but AUROC not up |
| --- | --- | --- | --- | --- | --- |
| IEMOCAP | 1937 | validation | -6.151213 | -2.064171 | True |
| IEMOCAP | 1937 | test | -3.969560 | -0.143421 | True |
| IEMOCAP | 2025 | validation | -8.064185 | 3.358271 | False |
| IEMOCAP | 2025 | test | -9.248795 | -1.064653 | True |
| IEMOCAP | 3407 | validation | -1.520884 | -0.722108 | True |
| IEMOCAP | 3407 | test | 5.177081 | 2.548904 | False |
| MELD | 1937 | validation | -1.416967 | -0.143681 | True |
| MELD | 1937 | test | -0.698159 | -1.861498 | True |
| MELD | 2025 | validation | 0.175915 | -0.014782 | False |
| MELD | 2025 | test | -1.394412 | -0.063663 | True |
| MELD | 3407 | validation | -9.041213 | -0.013440 | True |
| MELD | 3407 | test | -8.782305 | -0.812393 | True |

Improved ECE alone does not establish improved selective prediction or error detection. These are observations, not a causal attribution to CC.
