# Class and preregistered dialogue-position analysis

Execution coverage: **complete**; missing evaluations: 0.

Validation is the primary analysis. Test is the prespecified replication analysis and is reported separately.
IEMOCAP seed 1937 test results were already observed in round 2 and were reused without retraining or reselection.
MELD seed 1937 is the technical pilot; its test may precede the remaining fixed matrix. Configurations do not change after that exposure.
Deltas are CMERC minus Baseline. Classification/calibration are percentages, with deltas in percentage points.
CEL is a sum of positive true-class log-probability differences under `prefix_mean_v1`, a prefix smoothing view, not complete modality removal.
Three seeds on a fixed split give uncertain estimates. Sample SD uses ddof=1; df=2 t intervals rely on normality and are descriptive.
The exact two-sided sign-flip test has minimum p=0.25 for three seeds. No robust statistical significance or proven innovation is claimed.

Position = (zero-based utterance index + 0.5) / complete dialogue length: early [0,1/3), middle [1/3,2/3), late [2/3,1].
Length-one dialogues belong to middle. Every class and segment is retained, including negative deltas.

| Dataset | Seed | Split | Segment | Metric | Baseline | CMERC | Delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | 1937 | validation | overall | accuracy | 62.998102 | 63.187856 | 0.189753 |
| IEMOCAP | 1937 | validation | overall | weighted_f1 | 62.905474 | 63.393519 | 0.488045 |
| IEMOCAP | 1937 | validation | overall | macro_f1 | 61.469316 | 62.706679 | 1.237363 |
| IEMOCAP | 1937 | validation | overall | f1.happy | 58.571429 | 64.197531 | 5.626102 |
| IEMOCAP | 1937 | validation | overall | f1.sad | 74.666667 | 74.509804 | -0.156863 |
| IEMOCAP | 1937 | validation | overall | f1.neutral | 56.768559 | 53.266332 | -3.502227 |
| IEMOCAP | 1937 | validation | overall | f1.angry | 78.260870 | 76.315789 | -1.945080 |
| IEMOCAP | 1937 | validation | overall | f1.excited | 55.944056 | 57.333333 | 1.389277 |
| IEMOCAP | 1937 | validation | overall | f1.frustrated | 44.604317 | 50.617284 | 6.012967 |
| IEMOCAP | 1937 | validation | overall | ece | 21.367332 | 15.216119 | -6.151213 |
| IEMOCAP | 1937 | validation | overall | mce | 28.359297 | 73.788282 | 45.428985 |
| IEMOCAP | 1937 | validation | overall | rmsce | 22.212391 | 17.222625 | -4.989766 |
| IEMOCAP | 1937 | validation | overall | auroc_success | 64.133457 | 62.069286 | -2.064171 |
| IEMOCAP | 1937 | validation | overall | auprc_success | 73.171425 | 70.309299 | -2.862126 |
| IEMOCAP | 1937 | validation | early | accuracy | 51.977401 | 56.497175 | 4.519774 |
| IEMOCAP | 1937 | validation | early | weighted_f1 | 52.124575 | 56.634339 | 4.509764 |
| IEMOCAP | 1937 | validation | early | macro_f1 | 51.916186 | 55.153001 | 3.236815 |
| IEMOCAP | 1937 | validation | early | f1.happy | 61.818182 | 75.000000 | 13.181818 |
| IEMOCAP | 1937 | validation | early | f1.sad | 70.000000 | 60.000000 | -10.000000 |
| IEMOCAP | 1937 | validation | early | f1.neutral | 50.909091 | 60.000000 | 9.090909 |
| IEMOCAP | 1937 | validation | early | f1.angry | 60.714286 | 50.000000 | -10.714286 |
| IEMOCAP | 1937 | validation | early | f1.excited | 45.833333 | 54.545455 | 8.712121 |
| IEMOCAP | 1937 | validation | early | f1.frustrated | 22.222222 | 31.372549 | 9.150327 |
| IEMOCAP | 1937 | validation | early | ece | 28.806972 | 18.918251 | -9.888721 |
| IEMOCAP | 1937 | validation | early | mce | 40.423717 | 73.788282 | 33.364566 |
| IEMOCAP | 1937 | validation | early | rmsce | 30.300719 | 21.386577 | -8.914142 |
| IEMOCAP | 1937 | validation | early | auroc_success | 61.035806 | 61.701299 | 0.665493 |
| IEMOCAP | 1937 | validation | early | auprc_success | 57.139511 | 63.918580 | 6.779069 |
| IEMOCAP | 1937 | validation | middle | accuracy | 64.161850 | 63.583815 | -0.578035 |
| IEMOCAP | 1937 | validation | middle | weighted_f1 | 63.589599 | 63.233200 | -0.356400 |
| IEMOCAP | 1937 | validation | middle | macro_f1 | 63.005102 | 62.218605 | -0.786497 |
| IEMOCAP | 1937 | validation | middle | f1.happy | 58.823529 | 51.282051 | -7.541478 |
| IEMOCAP | 1937 | validation | middle | f1.sad | 79.310345 | 83.333333 | 4.022989 |
| IEMOCAP | 1937 | validation | middle | f1.neutral | 61.538462 | 51.851852 | -9.686610 |
| IEMOCAP | 1937 | validation | middle | f1.angry | 68.888889 | 71.428571 | 2.539683 |
| IEMOCAP | 1937 | validation | middle | f1.excited | 73.469388 | 70.588235 | -2.881152 |
| IEMOCAP | 1937 | validation | middle | f1.frustrated | 36.000000 | 44.827586 | 8.827586 |
| IEMOCAP | 1937 | validation | middle | ece | 18.833274 | 15.717175 | -3.116099 |
| IEMOCAP | 1937 | validation | middle | mce | 25.491246 | 30.952606 | 5.461360 |
| IEMOCAP | 1937 | validation | middle | rmsce | 20.515495 | 18.637629 | -1.877867 |
| IEMOCAP | 1937 | validation | middle | auroc_success | 61.624528 | 59.826840 | -1.797688 |
| IEMOCAP | 1937 | validation | middle | auprc_success | 71.404248 | 67.472669 | -3.931579 |
| IEMOCAP | 1937 | validation | late | accuracy | 72.881356 | 69.491525 | -3.389831 |
| IEMOCAP | 1937 | validation | late | weighted_f1 | 72.863976 | 68.681185 | -4.182790 |
| IEMOCAP | 1937 | validation | late | macro_f1 | 68.561294 | 64.902482 | -3.658812 |
| IEMOCAP | 1937 | validation | late | f1.happy | 54.901961 | 61.016949 | 6.114988 |
| IEMOCAP | 1937 | validation | late | f1.sad | 73.076923 | 75.471698 | 2.394775 |
| IEMOCAP | 1937 | validation | late | f1.neutral | 62.962963 | 40.000000 | -22.962963 |
| IEMOCAP | 1937 | validation | late | f1.angry | 95.327103 | 92.000000 | -3.327103 |
| IEMOCAP | 1937 | validation | late | f1.excited | 47.826087 | 45.454545 | -2.371542 |
| IEMOCAP | 1937 | validation | late | f1.frustrated | 77.272727 | 75.471698 | -1.801029 |
| IEMOCAP | 1937 | validation | late | ece | 18.390918 | 14.810954 | -3.579964 |
| IEMOCAP | 1937 | validation | late | mce | 60.792953 | 23.268767 | -37.524186 |
| IEMOCAP | 1937 | validation | late | rmsce | 19.998344 | 16.444992 | -3.553352 |
| IEMOCAP | 1937 | validation | late | auroc_success | 65.778424 | 61.547727 | -4.230697 |
| IEMOCAP | 1937 | validation | late | auprc_success | 82.973510 | 75.891444 | -7.082065 |
| IEMOCAP | 1937 | test | overall | accuracy | 64.510166 | 65.249538 | 0.739372 |
| IEMOCAP | 1937 | test | overall | weighted_f1 | 63.958180 | 65.059289 | 1.101108 |
| IEMOCAP | 1937 | test | overall | macro_f1 | 62.150484 | 63.065101 | 0.914617 |
| IEMOCAP | 1937 | test | overall | f1.happy | 40.366972 | 45.323741 | 4.956769 |
| IEMOCAP | 1937 | test | overall | f1.sad | 68.837209 | 75.319149 | 6.481940 |
| IEMOCAP | 1937 | test | overall | f1.neutral | 64.677804 | 63.198960 | -1.478845 |
| IEMOCAP | 1937 | test | overall | f1.angry | 63.841808 | 55.762082 | -8.079726 |
| IEMOCAP | 1937 | test | overall | f1.excited | 73.981191 | 73.344652 | -0.636539 |
| IEMOCAP | 1937 | test | overall | f1.frustrated | 61.197917 | 65.442021 | 4.244104 |
| IEMOCAP | 1937 | test | overall | ece | 18.520027 | 14.550467 | -3.969560 |
| IEMOCAP | 1937 | test | overall | mce | 33.974854 | 24.416502 | -9.558352 |
| IEMOCAP | 1937 | test | overall | rmsce | 19.257726 | 15.614315 | -3.643410 |
| IEMOCAP | 1937 | test | overall | auroc_success | 67.208061 | 67.064640 | -0.143421 |
| IEMOCAP | 1937 | test | overall | auprc_success | 78.989156 | 79.212895 | 0.223739 |
| IEMOCAP | 1937 | test | early | accuracy | 62.222222 | 61.111111 | -1.111111 |
| IEMOCAP | 1937 | test | early | weighted_f1 | 61.341273 | 60.412819 | -0.928454 |
| IEMOCAP | 1937 | test | early | macro_f1 | 53.803730 | 51.064509 | -2.739221 |
| IEMOCAP | 1937 | test | early | f1.happy | 16.666667 | 27.027027 | 10.360360 |
| IEMOCAP | 1937 | test | early | f1.sad | 57.142857 | 63.235294 | 6.092437 |
| IEMOCAP | 1937 | test | early | f1.neutral | 61.309524 | 58.193980 | -3.115544 |
| IEMOCAP | 1937 | test | early | f1.angry | 46.666667 | 17.142857 | -29.523810 |
| IEMOCAP | 1937 | test | early | f1.excited | 77.366255 | 74.336283 | -3.029972 |
| IEMOCAP | 1937 | test | early | f1.frustrated | 63.670412 | 66.451613 | 2.781201 |
| IEMOCAP | 1937 | test | early | ece | 20.460856 | 17.498728 | -2.962127 |
| IEMOCAP | 1937 | test | early | mce | 66.902366 | 31.365658 | -35.536708 |
| IEMOCAP | 1937 | test | early | rmsce | 21.814851 | 18.626813 | -3.188038 |
| IEMOCAP | 1937 | test | early | auroc_success | 67.041608 | 68.942280 | 1.900672 |
| IEMOCAP | 1937 | test | early | auprc_success | 76.402023 | 78.220531 | 1.818507 |
| IEMOCAP | 1937 | test | middle | accuracy | 66.298343 | 67.403315 | 1.104972 |
| IEMOCAP | 1937 | test | middle | weighted_f1 | 66.200751 | 67.549553 | 1.348801 |
| IEMOCAP | 1937 | test | middle | macro_f1 | 64.226021 | 64.512003 | 0.285982 |
| IEMOCAP | 1937 | test | middle | f1.happy | 46.428571 | 45.945946 | -0.482625 |
| IEMOCAP | 1937 | test | middle | f1.sad | 72.992701 | 80.263158 | 7.270457 |
| IEMOCAP | 1937 | test | middle | f1.neutral | 70.250896 | 67.938931 | -2.311965 |
| IEMOCAP | 1937 | test | middle | f1.angry | 61.904762 | 52.083333 | -9.821429 |
| IEMOCAP | 1937 | test | middle | f1.excited | 75.121951 | 74.594595 | -0.527357 |
| IEMOCAP | 1937 | test | middle | f1.frustrated | 58.657244 | 66.246057 | 7.588813 |
| IEMOCAP | 1937 | test | middle | ece | 16.690405 | 13.442743 | -3.247662 |
| IEMOCAP | 1937 | test | middle | mce | 67.169711 | 68.458299 | 1.288588 |
| IEMOCAP | 1937 | test | middle | rmsce | 18.014213 | 15.964131 | -2.050083 |
| IEMOCAP | 1937 | test | middle | auroc_success | 64.998482 | 65.547529 | 0.549047 |
| IEMOCAP | 1937 | test | middle | auprc_success | 78.327375 | 79.850398 | 1.523023 |
| IEMOCAP | 1937 | test | late | accuracy | 65.000000 | 67.222222 | 2.222222 |
| IEMOCAP | 1937 | test | late | weighted_f1 | 64.431818 | 67.067793 | 2.635975 |
| IEMOCAP | 1937 | test | late | macro_f1 | 64.286284 | 67.007448 | 2.721164 |
| IEMOCAP | 1937 | test | late | f1.happy | 47.368421 | 55.384615 | 8.016194 |
| IEMOCAP | 1937 | test | late | f1.sad | 74.251497 | 80.219780 | 5.968283 |
| IEMOCAP | 1937 | test | late | f1.neutral | 62.780269 | 64.423077 | 1.642808 |
| IEMOCAP | 1937 | test | late | f1.angry | 71.428571 | 68.115942 | -3.312629 |
| IEMOCAP | 1937 | test | late | f1.excited | 68.421053 | 70.786517 | 2.365464 |
| IEMOCAP | 1937 | test | late | f1.frustrated | 61.467890 | 63.114754 | 1.646864 |
| IEMOCAP | 1937 | test | late | ece | 19.239080 | 12.948422 | -6.290659 |
| IEMOCAP | 1937 | test | late | mce | 32.147515 | 20.296182 | -11.851332 |
| IEMOCAP | 1937 | test | late | rmsce | 19.858145 | 14.625342 | -5.232802 |
| IEMOCAP | 1937 | test | late | auroc_success | 69.704096 | 66.156169 | -3.547927 |
| IEMOCAP | 1937 | test | late | auprc_success | 81.830587 | 80.162106 | -1.668481 |
| IEMOCAP | 2025 | validation | overall | accuracy | 59.203036 | 59.013283 | -0.189753 |
| IEMOCAP | 2025 | validation | overall | weighted_f1 | 59.349495 | 59.440172 | 0.090677 |
| IEMOCAP | 2025 | validation | overall | macro_f1 | 58.714327 | 58.669214 | -0.045113 |
| IEMOCAP | 2025 | validation | overall | f1.happy | 60.139860 | 61.077844 | 0.937984 |
| IEMOCAP | 2025 | validation | overall | f1.sad | 71.724138 | 75.816993 | 4.092856 |
| IEMOCAP | 2025 | validation | overall | f1.neutral | 49.218750 | 49.549550 | 0.330800 |
| IEMOCAP | 2025 | validation | overall | f1.angry | 72.649573 | 71.889401 | -0.760172 |
| IEMOCAP | 2025 | validation | overall | f1.excited | 56.953642 | 48.226950 | -8.726692 |
| IEMOCAP | 2025 | validation | overall | f1.frustrated | 41.600000 | 45.454545 | 3.854545 |
| IEMOCAP | 2025 | validation | overall | ece | 25.104901 | 17.040716 | -8.064185 |
| IEMOCAP | 2025 | validation | overall | mce | 72.381660 | 35.558850 | -36.822811 |
| IEMOCAP | 2025 | validation | overall | rmsce | 26.619432 | 18.676318 | -7.943115 |
| IEMOCAP | 2025 | validation | overall | auroc_success | 59.945587 | 63.303859 | 3.358271 |
| IEMOCAP | 2025 | validation | overall | auprc_success | 68.382188 | 69.069600 | 0.687411 |
| IEMOCAP | 2025 | validation | early | accuracy | 50.847458 | 50.282486 | -0.564972 |
| IEMOCAP | 2025 | validation | early | weighted_f1 | 50.588422 | 49.918231 | -0.670191 |
| IEMOCAP | 2025 | validation | early | macro_f1 | 50.576768 | 48.186227 | -2.390541 |
| IEMOCAP | 2025 | validation | early | f1.happy | 62.068966 | 65.573770 | 3.504805 |
| IEMOCAP | 2025 | validation | early | f1.sad | 66.666667 | 63.414634 | -3.252033 |
| IEMOCAP | 2025 | validation | early | f1.neutral | 49.122807 | 56.140351 | 7.017544 |
| IEMOCAP | 2025 | validation | early | f1.angry | 55.319149 | 41.025641 | -14.293508 |
| IEMOCAP | 2025 | validation | early | f1.excited | 45.283019 | 40.740741 | -4.542278 |
| IEMOCAP | 2025 | validation | early | f1.frustrated | 25.000000 | 22.222222 | -2.777778 |
| IEMOCAP | 2025 | validation | early | ece | 32.906397 | 24.461327 | -8.445070 |
| IEMOCAP | 2025 | validation | early | mce | 72.381660 | 70.311424 | -2.070236 |
| IEMOCAP | 2025 | validation | early | rmsce | 35.579754 | 28.109277 | -7.470477 |
| IEMOCAP | 2025 | validation | early | auroc_success | 56.538953 | 58.273749 | 1.734796 |
| IEMOCAP | 2025 | validation | early | auprc_success | 56.391896 | 55.684780 | -0.707116 |
| IEMOCAP | 2025 | validation | middle | accuracy | 57.225434 | 57.225434 | 0.000000 |
| IEMOCAP | 2025 | validation | middle | weighted_f1 | 56.941494 | 57.216632 | 0.275138 |
| IEMOCAP | 2025 | validation | middle | macro_f1 | 57.573297 | 56.816509 | -0.756789 |
| IEMOCAP | 2025 | validation | middle | f1.happy | 62.857143 | 48.888889 | -13.968254 |
| IEMOCAP | 2025 | validation | middle | f1.sad | 74.074074 | 83.333333 | 9.259259 |
| IEMOCAP | 2025 | validation | middle | f1.neutral | 46.511628 | 45.901639 | -0.609989 |
| IEMOCAP | 2025 | validation | middle | f1.angry | 59.259259 | 60.526316 | 1.267057 |
| IEMOCAP | 2025 | validation | middle | f1.excited | 73.469388 | 60.869565 | -12.599823 |
| IEMOCAP | 2025 | validation | middle | f1.frustrated | 29.268293 | 41.379310 | 12.111018 |
| IEMOCAP | 2025 | validation | middle | ece | 25.893720 | 15.645891 | -10.247829 |
| IEMOCAP | 2025 | validation | middle | mce | 63.359803 | 36.234181 | -27.125622 |
| IEMOCAP | 2025 | validation | middle | rmsce | 27.738266 | 17.076124 | -10.662142 |
| IEMOCAP | 2025 | validation | middle | auroc_success | 59.077259 | 66.475566 | 7.398307 |
| IEMOCAP | 2025 | validation | middle | auprc_success | 64.588955 | 68.666695 | 4.077740 |
| IEMOCAP | 2025 | validation | late | accuracy | 69.491525 | 69.491525 | 0.000000 |
| IEMOCAP | 2025 | validation | late | weighted_f1 | 69.331027 | 68.677368 | -0.653659 |
| IEMOCAP | 2025 | validation | late | macro_f1 | 65.808316 | 64.573709 | -1.234607 |
| IEMOCAP | 2025 | validation | late | f1.happy | 56.000000 | 65.573770 | 9.573770 |
| IEMOCAP | 2025 | validation | late | f1.sad | 73.469388 | 76.923077 | 3.453689 |
| IEMOCAP | 2025 | validation | late | f1.neutral | 53.571429 | 38.297872 | -15.273556 |
| IEMOCAP | 2025 | validation | late | f1.angry | 90.566038 | 92.156863 | 1.590825 |
| IEMOCAP | 2025 | validation | late | f1.excited | 53.061224 | 43.902439 | -9.158785 |
| IEMOCAP | 2025 | validation | late | f1.frustrated | 68.181818 | 70.588235 | 2.406417 |
| IEMOCAP | 2025 | validation | late | ece | 18.334338 | 15.949631 | -2.384707 |
| IEMOCAP | 2025 | validation | late | mce | 44.218093 | 60.825783 | 16.607690 |
| IEMOCAP | 2025 | validation | late | rmsce | 20.035571 | 17.694387 | -2.341184 |
| IEMOCAP | 2025 | validation | late | auroc_success | 61.781090 | 62.195122 | 0.414032 |
| IEMOCAP | 2025 | validation | late | auprc_success | 78.632585 | 77.639469 | -0.993116 |
| IEMOCAP | 2025 | test | overall | accuracy | 63.894023 | 65.804067 | 1.910043 |
| IEMOCAP | 2025 | test | overall | weighted_f1 | 63.327134 | 65.807453 | 2.480319 |
| IEMOCAP | 2025 | test | overall | macro_f1 | 61.047973 | 64.261643 | 3.213671 |
| IEMOCAP | 2025 | test | overall | f1.happy | 34.977578 | 49.006623 | 14.029044 |
| IEMOCAP | 2025 | test | overall | f1.sad | 70.308789 | 75.159236 | 4.850447 |
| IEMOCAP | 2025 | test | overall | f1.neutral | 63.426488 | 61.942257 | -1.484231 |
| IEMOCAP | 2025 | test | overall | f1.angry | 61.261261 | 58.867925 | -2.393337 |
| IEMOCAP | 2025 | test | overall | f1.excited | 75.465839 | 73.476703 | -1.989136 |
| IEMOCAP | 2025 | test | overall | f1.frustrated | 60.847880 | 67.117117 | 6.269237 |
| IEMOCAP | 2025 | test | overall | ece | 21.962087 | 12.713291 | -9.248795 |
| IEMOCAP | 2025 | test | overall | mce | 34.381594 | 19.665141 | -14.716452 |
| IEMOCAP | 2025 | test | overall | rmsce | 22.751155 | 13.886603 | -8.864552 |
| IEMOCAP | 2025 | test | overall | auroc_success | 67.536968 | 66.472315 | -1.064653 |
| IEMOCAP | 2025 | test | overall | auprc_success | 79.279014 | 79.341325 | 0.062311 |
| IEMOCAP | 2025 | test | early | accuracy | 63.148148 | 63.148148 | 0.000000 |
| IEMOCAP | 2025 | test | early | weighted_f1 | 62.675967 | 62.349625 | -0.326343 |
| IEMOCAP | 2025 | test | early | macro_f1 | 56.383285 | 53.336682 | -3.046604 |
| IEMOCAP | 2025 | test | early | f1.happy | 31.578947 | 32.876712 | 1.297765 |
| IEMOCAP | 2025 | test | early | f1.sad | 66.666667 | 64.788732 | -1.877934 |
| IEMOCAP | 2025 | test | early | f1.neutral | 58.252427 | 60.869565 | 2.617138 |
| IEMOCAP | 2025 | test | early | f1.angry | 39.215686 | 18.181818 | -21.033868 |
| IEMOCAP | 2025 | test | early | f1.excited | 77.551020 | 76.855895 | -0.695125 |
| IEMOCAP | 2025 | test | early | f1.frustrated | 65.034965 | 66.447368 | 1.412403 |
| IEMOCAP | 2025 | test | early | ece | 24.096958 | 14.793629 | -9.303329 |
| IEMOCAP | 2025 | test | early | mce | 45.106721 | 30.841091 | -14.265630 |
| IEMOCAP | 2025 | test | early | rmsce | 25.738136 | 15.760495 | -9.977641 |
| IEMOCAP | 2025 | test | early | auroc_success | 61.079592 | 68.794117 | 7.714526 |
| IEMOCAP | 2025 | test | early | auprc_success | 74.069936 | 78.725144 | 4.655208 |
| IEMOCAP | 2025 | test | middle | accuracy | 65.561694 | 66.482505 | 0.920810 |
| IEMOCAP | 2025 | test | middle | weighted_f1 | 65.271564 | 66.891047 | 1.619483 |
| IEMOCAP | 2025 | test | middle | macro_f1 | 60.786957 | 63.783840 | 2.996884 |
| IEMOCAP | 2025 | test | middle | f1.happy | 29.090909 | 43.373494 | 14.282585 |
| IEMOCAP | 2025 | test | middle | f1.sad | 69.696970 | 77.027027 | 7.330057 |
| IEMOCAP | 2025 | test | middle | f1.neutral | 69.718310 | 65.151515 | -4.566795 |
| IEMOCAP | 2025 | test | middle | f1.angry | 56.896552 | 56.250000 | -0.646552 |
| IEMOCAP | 2025 | test | middle | f1.excited | 77.832512 | 72.189349 | -5.643163 |
| IEMOCAP | 2025 | test | middle | f1.frustrated | 61.486486 | 68.711656 | 7.225170 |
| IEMOCAP | 2025 | test | middle | ece | 20.466707 | 13.484737 | -6.981970 |
| IEMOCAP | 2025 | test | middle | mce | 67.914721 | 32.592553 | -35.322168 |
| IEMOCAP | 2025 | test | middle | rmsce | 21.562270 | 14.445424 | -7.116846 |
| IEMOCAP | 2025 | test | middle | auroc_success | 67.040197 | 64.539892 | -2.500305 |
| IEMOCAP | 2025 | test | middle | auprc_success | 79.972828 | 78.778007 | -1.194821 |
| IEMOCAP | 2025 | test | late | accuracy | 62.962963 | 67.777778 | 4.814815 |
| IEMOCAP | 2025 | test | late | weighted_f1 | 62.113816 | 67.931280 | 5.817464 |
| IEMOCAP | 2025 | test | late | macro_f1 | 62.028607 | 67.969686 | 5.941080 |
| IEMOCAP | 2025 | test | late | f1.happy | 39.639640 | 60.273973 | 20.634333 |
| IEMOCAP | 2025 | test | late | f1.sad | 73.885350 | 81.767956 | 7.882605 |
| IEMOCAP | 2025 | test | late | f1.neutral | 62.608696 | 59.296482 | -3.312213 |
| IEMOCAP | 2025 | test | late | f1.angry | 71.084337 | 70.588235 | -0.496102 |
| IEMOCAP | 2025 | test | late | f1.excited | 70.408163 | 70.000000 | -0.408163 |
| IEMOCAP | 2025 | test | late | f1.frustrated | 54.545455 | 65.891473 | 11.346018 |
| IEMOCAP | 2025 | test | late | ece | 21.917091 | 12.997069 | -8.920021 |
| IEMOCAP | 2025 | test | late | mce | 42.180458 | 29.480251 | -12.700207 |
| IEMOCAP | 2025 | test | late | rmsce | 22.836867 | 13.748414 | -9.088454 |
| IEMOCAP | 2025 | test | late | auroc_success | 74.064706 | 65.934929 | -8.129777 |
| IEMOCAP | 2025 | test | late | auprc_success | 84.045776 | 80.820755 | -3.225020 |
| IEMOCAP | 3407 | validation | overall | accuracy | 63.187856 | 62.998102 | -0.189753 |
| IEMOCAP | 3407 | validation | overall | weighted_f1 | 63.027623 | 63.280601 | 0.252978 |
| IEMOCAP | 3407 | validation | overall | macro_f1 | 62.605711 | 63.006095 | 0.400385 |
| IEMOCAP | 3407 | validation | overall | f1.happy | 66.242038 | 60.689655 | -5.552383 |
| IEMOCAP | 3407 | validation | overall | f1.sad | 74.838710 | 75.675676 | 0.836966 |
| IEMOCAP | 3407 | validation | overall | f1.neutral | 51.063830 | 55.502392 | 4.438563 |
| IEMOCAP | 3407 | validation | overall | f1.angry | 75.518672 | 72.398190 | -3.120482 |
| IEMOCAP | 3407 | validation | overall | f1.excited | 57.971014 | 59.119497 | 1.148482 |
| IEMOCAP | 3407 | validation | overall | f1.frustrated | 50.000000 | 54.651163 | 4.651163 |
| IEMOCAP | 3407 | validation | overall | ece | 19.405296 | 17.884412 | -1.520884 |
| IEMOCAP | 3407 | validation | overall | mce | 25.911030 | 35.436281 | 9.525251 |
| IEMOCAP | 3407 | validation | overall | rmsce | 20.616167 | 21.174045 | 0.557878 |
| IEMOCAP | 3407 | validation | overall | auroc_success | 63.400514 | 62.678406 | -0.722108 |
| IEMOCAP | 3407 | validation | overall | auprc_success | 73.434823 | 70.663731 | -2.771092 |
| IEMOCAP | 3407 | validation | early | accuracy | 56.497175 | 54.802260 | -1.694915 |
| IEMOCAP | 3407 | validation | early | weighted_f1 | 55.774385 | 54.606405 | -1.167980 |
| IEMOCAP | 3407 | validation | early | macro_f1 | 54.914089 | 54.180440 | -0.733649 |
| IEMOCAP | 3407 | validation | early | f1.happy | 73.684211 | 65.384615 | -8.299595 |
| IEMOCAP | 3407 | validation | early | f1.sad | 66.666667 | 68.292683 | 1.626016 |
| IEMOCAP | 3407 | validation | early | f1.neutral | 55.652174 | 59.047619 | 3.395445 |
| IEMOCAP | 3407 | validation | early | f1.angry | 59.259259 | 39.024390 | -20.234869 |
| IEMOCAP | 3407 | validation | early | f1.excited | 52.000000 | 53.333333 | 1.333333 |
| IEMOCAP | 3407 | validation | early | f1.frustrated | 22.222222 | 40.000000 | 17.777778 |
| IEMOCAP | 3407 | validation | early | ece | 23.542997 | 24.594171 | 1.051174 |
| IEMOCAP | 3407 | validation | early | mce | 41.409112 | 56.967259 | 15.558147 |
| IEMOCAP | 3407 | validation | early | rmsce | 26.134164 | 29.471482 | 3.337318 |
| IEMOCAP | 3407 | validation | early | auroc_success | 60.571429 | 63.079897 | 2.508468 |
| IEMOCAP | 3407 | validation | early | auprc_success | 63.251927 | 63.492476 | 0.240549 |
| IEMOCAP | 3407 | validation | middle | accuracy | 64.739884 | 61.271676 | -3.468208 |
| IEMOCAP | 3407 | validation | middle | weighted_f1 | 64.363052 | 61.252630 | -3.110423 |
| IEMOCAP | 3407 | validation | middle | macro_f1 | 64.716302 | 61.216057 | -3.500244 |
| IEMOCAP | 3407 | validation | middle | f1.happy | 65.000000 | 52.631579 | -12.368421 |
| IEMOCAP | 3407 | validation | middle | f1.sad | 83.333333 | 78.571429 | -4.761905 |
| IEMOCAP | 3407 | validation | middle | f1.neutral | 46.153846 | 53.571429 | 7.417582 |
| IEMOCAP | 3407 | validation | middle | f1.angry | 68.235294 | 63.291139 | -4.944155 |
| IEMOCAP | 3407 | validation | middle | f1.excited | 76.595745 | 73.076923 | -3.518822 |
| IEMOCAP | 3407 | validation | middle | f1.frustrated | 48.979592 | 46.153846 | -2.825746 |
| IEMOCAP | 3407 | validation | middle | ece | 17.852474 | 19.152642 | 1.300168 |
| IEMOCAP | 3407 | validation | middle | mce | 26.415162 | 30.981009 | 4.565847 |
| IEMOCAP | 3407 | validation | middle | rmsce | 20.057412 | 21.227784 | 1.170372 |
| IEMOCAP | 3407 | validation | middle | auroc_success | 61.504684 | 62.728809 | 1.224125 |
| IEMOCAP | 3407 | validation | middle | auprc_success | 71.926573 | 68.791173 | -3.135399 |
| IEMOCAP | 3407 | validation | late | accuracy | 68.361582 | 72.881356 | 4.519774 |
| IEMOCAP | 3407 | validation | late | weighted_f1 | 68.531787 | 72.487512 | 3.955725 |
| IEMOCAP | 3407 | validation | late | macro_f1 | 64.581327 | 69.191987 | 4.610660 |
| IEMOCAP | 3407 | validation | late | f1.happy | 60.000000 | 61.818182 | 1.818182 |
| IEMOCAP | 3407 | validation | late | f1.sad | 71.698113 | 78.431373 | 6.733259 |
| IEMOCAP | 3407 | validation | late | f1.neutral | 47.272727 | 50.000000 | 2.727273 |
| IEMOCAP | 3407 | validation | late | f1.angry | 90.196078 | 93.069307 | 2.873228 |
| IEMOCAP | 3407 | validation | late | f1.excited | 43.902439 | 51.063830 | 7.161391 |
| IEMOCAP | 3407 | validation | late | f1.frustrated | 74.418605 | 80.769231 | 6.350626 |
| IEMOCAP | 3407 | validation | late | ece | 19.328578 | 17.714312 | -1.614266 |
| IEMOCAP | 3407 | validation | late | mce | 35.993873 | 43.883777 | 7.889903 |
| IEMOCAP | 3407 | validation | late | rmsce | 20.510686 | 19.200195 | -1.310491 |
| IEMOCAP | 3407 | validation | late | auroc_success | 65.923849 | 58.397933 | -7.525916 |
| IEMOCAP | 3407 | validation | late | auprc_success | 80.747444 | 76.543462 | -4.203982 |
| IEMOCAP | 3407 | test | overall | accuracy | 66.112138 | 65.003081 | -1.109057 |
| IEMOCAP | 3407 | test | overall | weighted_f1 | 66.277336 | 64.656869 | -1.620467 |
| IEMOCAP | 3407 | test | overall | macro_f1 | 65.018768 | 62.375741 | -2.643027 |
| IEMOCAP | 3407 | test | overall | f1.happy | 49.006623 | 42.187500 | -6.819123 |
| IEMOCAP | 3407 | test | overall | f1.sad | 75.313808 | 75.221239 | -0.092569 |
| IEMOCAP | 3407 | test | overall | f1.neutral | 63.130660 | 63.659148 | 0.528488 |
| IEMOCAP | 3407 | test | overall | f1.angry | 63.192182 | 55.303030 | -7.889152 |
| IEMOCAP | 3407 | test | overall | f1.excited | 73.943662 | 72.516556 | -1.427106 |
| IEMOCAP | 3407 | test | overall | f1.frustrated | 65.525672 | 65.366972 | -0.158700 |
| IEMOCAP | 3407 | test | overall | ece | 14.071656 | 19.248737 | 5.177081 |
| IEMOCAP | 3407 | test | overall | mce | 27.699423 | 28.275761 | 0.576338 |
| IEMOCAP | 3407 | test | overall | rmsce | 15.867242 | 20.117347 | 4.250106 |
| IEMOCAP | 3407 | test | overall | auroc_success | 65.213929 | 67.762833 | 2.548904 |
| IEMOCAP | 3407 | test | overall | auprc_success | 78.939522 | 79.421352 | 0.481830 |
| IEMOCAP | 3407 | test | early | accuracy | 63.518519 | 62.407407 | -1.111111 |
| IEMOCAP | 3407 | test | early | weighted_f1 | 63.553333 | 61.542293 | -2.011040 |
| IEMOCAP | 3407 | test | early | macro_f1 | 57.362851 | 52.950287 | -4.412564 |
| IEMOCAP | 3407 | test | early | f1.happy | 35.616438 | 27.692308 | -7.924131 |
| IEMOCAP | 3407 | test | early | f1.sad | 67.153285 | 67.175573 | 0.022288 |
| IEMOCAP | 3407 | test | early | f1.neutral | 58.745875 | 58.631922 | -0.113953 |
| IEMOCAP | 3407 | test | early | f1.angry | 38.297872 | 22.857143 | -15.440729 |
| IEMOCAP | 3407 | test | early | f1.excited | 76.363636 | 74.678112 | -1.685525 |
| IEMOCAP | 3407 | test | early | f1.frustrated | 68.000000 | 66.666667 | -1.333333 |
| IEMOCAP | 3407 | test | early | ece | 17.307642 | 21.600751 | 4.293109 |
| IEMOCAP | 3407 | test | early | mce | 32.620012 | 72.363633 | 39.743621 |
| IEMOCAP | 3407 | test | early | rmsce | 19.357851 | 23.525251 | 4.167400 |
| IEMOCAP | 3407 | test | early | auroc_success | 64.006748 | 70.114455 | 6.107707 |
| IEMOCAP | 3407 | test | early | auprc_success | 75.570085 | 79.599924 | 4.029839 |
| IEMOCAP | 3407 | test | middle | accuracy | 66.666667 | 64.272560 | -2.394107 |
| IEMOCAP | 3407 | test | middle | weighted_f1 | 67.356358 | 64.144990 | -3.211368 |
| IEMOCAP | 3407 | test | middle | macro_f1 | 65.043254 | 59.244428 | -5.798826 |
| IEMOCAP | 3407 | test | middle | f1.happy | 46.666667 | 30.303030 | -16.363636 |
| IEMOCAP | 3407 | test | middle | f1.sad | 78.911565 | 73.758865 | -5.152699 |
| IEMOCAP | 3407 | test | middle | f1.neutral | 67.169811 | 67.647059 | 0.477248 |
| IEMOCAP | 3407 | test | middle | f1.angry | 59.259259 | 47.826087 | -11.433172 |
| IEMOCAP | 3407 | test | middle | f1.excited | 73.626374 | 71.134021 | -2.492353 |
| IEMOCAP | 3407 | test | middle | f1.frustrated | 64.625850 | 64.797508 | 0.171657 |
| IEMOCAP | 3407 | test | middle | ece | 13.416499 | 20.040806 | 6.624307 |
| IEMOCAP | 3407 | test | middle | mce | 37.048179 | 29.138986 | -7.909193 |
| IEMOCAP | 3407 | test | middle | rmsce | 15.499762 | 21.007990 | 5.508228 |
| IEMOCAP | 3407 | test | middle | auroc_success | 66.484540 | 67.886155 | 1.401615 |
| IEMOCAP | 3407 | test | middle | auprc_success | 78.790547 | 79.018550 | 0.228004 |
| IEMOCAP | 3407 | test | late | accuracy | 68.148148 | 68.333333 | 0.185185 |
| IEMOCAP | 3407 | test | late | weighted_f1 | 68.040595 | 68.136094 | 0.095500 |
| IEMOCAP | 3407 | test | late | macro_f1 | 68.076020 | 68.039673 | -0.036348 |
| IEMOCAP | 3407 | test | late | f1.happy | 57.553957 | 56.000000 | -1.553957 |
| IEMOCAP | 3407 | test | late | f1.sad | 78.350515 | 82.222222 | 3.871707 |
| IEMOCAP | 3407 | test | late | f1.neutral | 64.390244 | 65.753425 | 1.363181 |
| IEMOCAP | 3407 | test | late | f1.angry | 73.684211 | 68.613139 | -5.071072 |
| IEMOCAP | 3407 | test | late | f1.excited | 71.084337 | 71.186441 | 0.102103 |
| IEMOCAP | 3407 | test | late | f1.frustrated | 63.392857 | 64.462810 | 1.069953 |
| IEMOCAP | 3407 | test | late | ece | 13.780557 | 16.929057 | 3.148500 |
| IEMOCAP | 3407 | test | late | mce | 67.305613 | 32.848954 | -34.456658 |
| IEMOCAP | 3407 | test | late | rmsce | 15.529533 | 18.524340 | 2.994807 |
| IEMOCAP | 3407 | test | late | auroc_success | 65.388018 | 65.175359 | -0.212659 |
| IEMOCAP | 3407 | test | late | auprc_success | 81.871202 | 80.050470 | -1.820732 |
| MELD | 1937 | validation | overall | accuracy | 65.464382 | 65.374211 | -0.090171 |
| MELD | 1937 | validation | overall | weighted_f1 | 63.980093 | 64.073416 | 0.093324 |
| MELD | 1937 | validation | overall | macro_f1 | 51.157017 | 50.988507 | -0.168510 |
| MELD | 1937 | validation | overall | f1.neutral | 78.243902 | 77.514793 | -0.729110 |
| MELD | 1937 | validation | overall | f1.surprise | 65.346535 | 64.686469 | -0.660066 |
| MELD | 1937 | validation | overall | f1.fear | 36.065574 | 33.898305 | -2.167269 |
| MELD | 1937 | validation | overall | f1.sadness | 40.437158 | 45.026178 | 4.589020 |
| MELD | 1937 | validation | overall | f1.joy | 60.625000 | 59.934853 | -0.690147 |
| MELD | 1937 | validation | overall | f1.disgust | 25.000000 | 21.428571 | -3.571429 |
| MELD | 1937 | validation | overall | f1.anger | 52.380952 | 54.430380 | 2.049427 |
| MELD | 1937 | validation | overall | ece | 20.542450 | 19.125483 | -1.416967 |
| MELD | 1937 | validation | overall | mce | 26.844391 | 26.707613 | -0.136778 |
| MELD | 1937 | validation | overall | rmsce | 21.658764 | 19.935583 | -1.723181 |
| MELD | 1937 | validation | overall | auroc_success | 70.762934 | 70.619253 | -0.143681 |
| MELD | 1937 | validation | overall | auprc_success | 79.373921 | 80.664448 | 1.290527 |
| MELD | 1937 | validation | early | accuracy | 69.436997 | 68.900804 | -0.536193 |
| MELD | 1937 | validation | early | weighted_f1 | 68.468528 | 67.803145 | -0.665383 |
| MELD | 1937 | validation | early | macro_f1 | 53.347540 | 50.407661 | -2.939879 |
| MELD | 1937 | validation | early | f1.neutral | 80.862534 | 80.758808 | -0.103726 |
| MELD | 1937 | validation | early | f1.surprise | 65.909091 | 61.363636 | -4.545455 |
| MELD | 1937 | validation | early | f1.fear | 42.105263 | 50.000000 | 7.894737 |
| MELD | 1937 | validation | early | f1.sadness | 36.000000 | 36.000000 | 0.000000 |
| MELD | 1937 | validation | early | f1.joy | 67.741935 | 66.666667 | -1.075269 |
| MELD | 1937 | validation | early | f1.disgust | 25.000000 | 0.000000 | -25.000000 |
| MELD | 1937 | validation | early | f1.anger | 55.813953 | 58.064516 | 2.250563 |
| MELD | 1937 | validation | early | ece | 23.919186 | 22.654974 | -1.264212 |
| MELD | 1937 | validation | early | mce | 29.991687 | 34.112903 | 4.121216 |
| MELD | 1937 | validation | early | rmsce | 24.940376 | 23.607418 | -1.332958 |
| MELD | 1937 | validation | early | auroc_success | 71.364221 | 72.014625 | 0.650404 |
| MELD | 1937 | validation | early | auprc_success | 82.420335 | 83.561283 | 1.140948 |
| MELD | 1937 | validation | middle | accuracy | 63.911846 | 63.360882 | -0.550964 |
| MELD | 1937 | validation | middle | weighted_f1 | 61.820329 | 61.417611 | -0.402718 |
| MELD | 1937 | validation | middle | macro_f1 | 49.611271 | 49.352867 | -0.258404 |
| MELD | 1937 | validation | middle | f1.neutral | 77.987421 | 75.555556 | -2.431866 |
| MELD | 1937 | validation | middle | f1.surprise | 69.841270 | 72.440945 | 2.599675 |
| MELD | 1937 | validation | middle | f1.fear | 46.153846 | 33.333333 | -12.820513 |
| MELD | 1937 | validation | middle | f1.sadness | 32.142857 | 37.931034 | 5.788177 |
| MELD | 1937 | validation | middle | f1.joy | 52.173913 | 51.685393 | -0.488520 |
| MELD | 1937 | validation | middle | f1.disgust | 20.000000 | 25.000000 | 5.000000 |
| MELD | 1937 | validation | middle | f1.anger | 48.979592 | 49.523810 | 0.544218 |
| MELD | 1937 | validation | middle | ece | 19.378941 | 16.975485 | -2.403456 |
| MELD | 1937 | validation | middle | mce | 32.959616 | 31.257584 | -1.702032 |
| MELD | 1937 | validation | middle | rmsce | 21.460301 | 19.450851 | -2.009450 |
| MELD | 1937 | validation | middle | auroc_success | 72.686891 | 75.116051 | 2.429160 |
| MELD | 1937 | validation | middle | auprc_success | 83.325986 | 85.082252 | 1.756266 |
| MELD | 1937 | validation | late | accuracy | 63.002681 | 63.806971 | 0.804290 |
| MELD | 1937 | validation | late | weighted_f1 | 61.372167 | 62.433953 | 1.061786 |
| MELD | 1937 | validation | late | macro_f1 | 48.112420 | 49.288298 | 1.175878 |
| MELD | 1937 | validation | late | f1.neutral | 75.595238 | 75.757576 | 0.162338 |
| MELD | 1937 | validation | late | f1.surprise | 58.426966 | 56.818182 | -1.608784 |
| MELD | 1937 | validation | late | f1.fear | 12.500000 | 13.333333 | 0.833333 |
| MELD | 1937 | validation | late | f1.sadness | 49.350649 | 55.421687 | 6.071037 |
| MELD | 1937 | validation | late | f1.joy | 59.615385 | 59.183673 | -0.431711 |
| MELD | 1937 | validation | late | f1.disgust | 28.571429 | 28.571429 | 0.000000 |
| MELD | 1937 | validation | late | f1.anger | 52.727273 | 55.932203 | 3.204931 |
| MELD | 1937 | validation | late | ece | 18.474758 | 17.816790 | -0.657968 |
| MELD | 1937 | validation | late | mce | 29.475969 | 25.753600 | -3.722369 |
| MELD | 1937 | validation | late | rmsce | 20.094569 | 18.666780 | -1.427790 |
| MELD | 1937 | validation | late | auroc_success | 68.251619 | 64.721444 | -3.530175 |
| MELD | 1937 | validation | late | auprc_success | 73.382287 | 73.694201 | 0.311914 |
| MELD | 1937 | test | overall | accuracy | 65.019157 | 65.938697 | 0.919540 |
| MELD | 1937 | test | overall | weighted_f1 | 63.683365 | 64.830827 | 1.147462 |
| MELD | 1937 | test | overall | macro_f1 | 47.176699 | 47.429033 | 0.252334 |
| MELD | 1937 | test | overall | f1.neutral | 78.365204 | 79.043280 | 0.678076 |
| MELD | 1937 | test | overall | f1.surprise | 57.487091 | 57.435897 | -0.051194 |
| MELD | 1937 | test | overall | f1.fear | 24.691358 | 17.721519 | -6.969839 |
| MELD | 1937 | test | overall | f1.sadness | 34.104046 | 38.016529 | 3.912483 |
| MELD | 1937 | test | overall | f1.joy | 61.832061 | 62.140992 | 0.308931 |
| MELD | 1937 | test | overall | f1.disgust | 25.263158 | 24.444444 | -0.818713 |
| MELD | 1937 | test | overall | f1.anger | 48.493976 | 53.200569 | 4.706593 |
| MELD | 1937 | test | overall | ece | 19.883207 | 19.185048 | -0.698159 |
| MELD | 1937 | test | overall | mce | 81.752169 | 26.782200 | -54.969969 |
| MELD | 1937 | test | overall | rmsce | 22.121570 | 20.699546 | -1.422024 |
| MELD | 1937 | test | overall | auroc_success | 75.684105 | 73.822607 | -1.861498 |
| MELD | 1937 | test | overall | auprc_success | 83.422704 | 83.074037 | -0.348668 |
| MELD | 1937 | test | early | accuracy | 65.600000 | 68.800000 | 3.200000 |
| MELD | 1937 | test | early | weighted_f1 | 64.492449 | 67.783301 | 3.290852 |
| MELD | 1937 | test | early | macro_f1 | 46.762917 | 49.141944 | 2.379026 |
| MELD | 1937 | test | early | f1.neutral | 79.518072 | 81.718062 | 2.199989 |
| MELD | 1937 | test | early | f1.surprise | 48.044693 | 51.111111 | 3.066418 |
| MELD | 1937 | test | early | f1.fear | 26.086957 | 24.000000 | -2.086957 |
| MELD | 1937 | test | early | f1.sadness | 36.893204 | 41.904762 | 5.011558 |
| MELD | 1937 | test | early | f1.joy | 62.406015 | 65.648855 | 3.242840 |
| MELD | 1937 | test | early | f1.disgust | 23.529412 | 20.689655 | -2.839757 |
| MELD | 1937 | test | early | f1.anger | 50.862069 | 58.921162 | 8.059093 |
| MELD | 1937 | test | early | ece | 20.885164 | 22.253107 | 1.367943 |
| MELD | 1937 | test | early | mce | 66.992611 | 29.549682 | -37.442929 |
| MELD | 1937 | test | early | rmsce | 23.573155 | 23.523045 | -0.050111 |
| MELD | 1937 | test | early | auroc_success | 76.474470 | 71.884926 | -4.589544 |
| MELD | 1937 | test | early | auprc_success | 83.008844 | 82.657219 | -0.351625 |
| MELD | 1937 | test | middle | accuracy | 64.186047 | 64.534884 | 0.348837 |
| MELD | 1937 | test | middle | weighted_f1 | 62.510335 | 63.117624 | 0.607289 |
| MELD | 1937 | test | middle | macro_f1 | 45.827730 | 46.149354 | 0.321624 |
| MELD | 1937 | test | middle | f1.neutral | 77.598152 | 77.699531 | 0.101378 |
| MELD | 1937 | test | middle | f1.surprise | 60.093897 | 58.986175 | -1.107722 |
| MELD | 1937 | test | middle | f1.fear | 23.529412 | 18.181818 | -5.347594 |
| MELD | 1937 | test | middle | f1.sadness | 36.923077 | 42.647059 | 5.723982 |
| MELD | 1937 | test | middle | f1.joy | 61.538462 | 59.388646 | -2.149815 |
| MELD | 1937 | test | middle | f1.disgust | 14.814815 | 14.814815 | 0.000000 |
| MELD | 1937 | test | middle | f1.anger | 46.296296 | 51.327434 | 5.031137 |
| MELD | 1937 | test | middle | ece | 19.046124 | 18.097462 | -0.948663 |
| MELD | 1937 | test | middle | mce | 32.053615 | 27.572463 | -4.481152 |
| MELD | 1937 | test | middle | rmsce | 21.013059 | 20.236617 | -0.776443 |
| MELD | 1937 | test | middle | auroc_success | 75.112931 | 76.487373 | 1.374442 |
| MELD | 1937 | test | middle | auprc_success | 83.414227 | 84.302051 | 0.887824 |
| MELD | 1937 | test | late | accuracy | 65.257143 | 64.457143 | -0.800000 |
| MELD | 1937 | test | late | weighted_f1 | 64.015298 | 63.560603 | -0.454695 |
| MELD | 1937 | test | late | macro_f1 | 48.531329 | 46.263725 | -2.267604 |
| MELD | 1937 | test | late | f1.neutral | 77.927928 | 77.574371 | -0.353557 |
| MELD | 1937 | test | late | f1.surprise | 63.492063 | 61.702128 | -1.789936 |
| MELD | 1937 | test | late | f1.fear | 25.000000 | 9.523810 | -15.476190 |
| MELD | 1937 | test | late | f1.sadness | 28.318584 | 29.508197 | 1.189613 |
| MELD | 1937 | test | late | f1.joy | 61.538462 | 61.090909 | -0.447552 |
| MELD | 1937 | test | late | f1.disgust | 35.294118 | 35.294118 | 0.000000 |
| MELD | 1937 | test | late | f1.anger | 48.148148 | 49.152542 | 1.004394 |
| MELD | 1937 | test | late | ece | 20.192077 | 17.796781 | -2.395297 |
| MELD | 1937 | test | late | mce | 81.752169 | 31.443039 | -50.309130 |
| MELD | 1937 | test | late | rmsce | 22.443764 | 19.451244 | -2.992520 |
| MELD | 1937 | test | late | auroc_success | 75.467785 | 73.183052 | -2.284733 |
| MELD | 1937 | test | late | auprc_success | 83.701071 | 82.554680 | -1.146390 |
| MELD | 2025 | validation | overall | accuracy | 66.366096 | 66.907124 | 0.541028 |
| MELD | 2025 | validation | overall | weighted_f1 | 64.681259 | 65.699105 | 1.017846 |
| MELD | 2025 | validation | overall | macro_f1 | 51.281867 | 53.799082 | 2.517215 |
| MELD | 2025 | validation | overall | f1.neutral | 78.404669 | 79.603960 | 1.199291 |
| MELD | 2025 | validation | overall | f1.surprise | 65.551839 | 66.019417 | 0.467578 |
| MELD | 2025 | validation | overall | f1.fear | 31.578947 | 33.333333 | 1.754386 |
| MELD | 2025 | validation | overall | f1.sadness | 38.372093 | 40.229885 | 1.857792 |
| MELD | 2025 | validation | overall | f1.joy | 60.606061 | 60.481100 | -0.124961 |
| MELD | 2025 | validation | overall | f1.disgust | 25.000000 | 37.735849 | 12.735849 |
| MELD | 2025 | validation | overall | f1.anger | 59.459459 | 59.190031 | -0.269428 |
| MELD | 2025 | validation | overall | ece | 20.799505 | 20.975420 | 0.175915 |
| MELD | 2025 | validation | overall | mce | 31.543480 | 31.334586 | -0.208895 |
| MELD | 2025 | validation | overall | rmsce | 21.517265 | 21.855834 | 0.338569 |
| MELD | 2025 | validation | overall | auroc_success | 71.488154 | 71.473373 | -0.014782 |
| MELD | 2025 | validation | overall | auprc_success | 82.360420 | 82.728840 | 0.368420 |
| MELD | 2025 | validation | early | accuracy | 69.973190 | 70.777480 | 0.804290 |
| MELD | 2025 | validation | early | weighted_f1 | 68.754341 | 70.167033 | 1.412692 |
| MELD | 2025 | validation | early | macro_f1 | 53.375483 | 56.766232 | 3.390748 |
| MELD | 2025 | validation | early | f1.neutral | 81.401617 | 82.417582 | 1.015965 |
| MELD | 2025 | validation | early | f1.surprise | 62.921348 | 64.444444 | 1.523096 |
| MELD | 2025 | validation | early | f1.fear | 37.500000 | 42.105263 | 4.605263 |
| MELD | 2025 | validation | early | f1.sadness | 34.782609 | 33.333333 | -1.449275 |
| MELD | 2025 | validation | early | f1.joy | 67.226891 | 68.376068 | 1.149178 |
| MELD | 2025 | validation | early | f1.disgust | 28.571429 | 42.857143 | 14.285714 |
| MELD | 2025 | validation | early | f1.anger | 61.224490 | 63.829787 | 2.605297 |
| MELD | 2025 | validation | early | ece | 23.682480 | 22.839527 | -0.842954 |
| MELD | 2025 | validation | early | mce | 31.464062 | 31.157171 | -0.306891 |
| MELD | 2025 | validation | early | rmsce | 24.431762 | 23.930576 | -0.501186 |
| MELD | 2025 | validation | early | auroc_success | 73.915572 | 72.765499 | -1.150073 |
| MELD | 2025 | validation | early | auprc_success | 86.223864 | 85.963152 | -0.260713 |
| MELD | 2025 | validation | middle | accuracy | 66.391185 | 66.942149 | 0.550964 |
| MELD | 2025 | validation | middle | weighted_f1 | 64.410168 | 64.871494 | 0.461326 |
| MELD | 2025 | validation | middle | macro_f1 | 51.370037 | 53.447028 | 2.076991 |
| MELD | 2025 | validation | middle | f1.neutral | 78.095238 | 79.113924 | 1.018686 |
| MELD | 2025 | validation | middle | f1.surprise | 74.193548 | 74.418605 | 0.225056 |
| MELD | 2025 | validation | middle | f1.fear | 38.461538 | 38.461538 | 0.000000 |
| MELD | 2025 | validation | middle | f1.sadness | 33.962264 | 37.735849 | 3.773585 |
| MELD | 2025 | validation | middle | f1.joy | 56.470588 | 53.658537 | -2.812052 |
| MELD | 2025 | validation | middle | f1.disgust | 20.000000 | 33.333333 | 13.333333 |
| MELD | 2025 | validation | middle | f1.anger | 58.407080 | 57.407407 | -0.999672 |
| MELD | 2025 | validation | middle | ece | 20.907827 | 21.510762 | 0.602935 |
| MELD | 2025 | validation | middle | mce | 37.490965 | 34.202626 | -3.288339 |
| MELD | 2025 | validation | middle | rmsce | 22.058801 | 23.512315 | 1.453514 |
| MELD | 2025 | validation | middle | auroc_success | 71.954289 | 72.846365 | 0.892076 |
| MELD | 2025 | validation | middle | auprc_success | 84.069980 | 83.502327 | -0.567653 |
| MELD | 2025 | validation | late | accuracy | 62.734584 | 63.002681 | 0.268097 |
| MELD | 2025 | validation | late | weighted_f1 | 60.739745 | 61.781905 | 1.042160 |
| MELD | 2025 | validation | late | macro_f1 | 47.145505 | 49.019366 | 1.873860 |
| MELD | 2025 | validation | late | f1.neutral | 75.438596 | 76.969697 | 1.531100 |
| MELD | 2025 | validation | late | f1.surprise | 55.813953 | 55.555556 | -0.258398 |
| MELD | 2025 | validation | late | f1.fear | 13.333333 | 13.333333 | 0.000000 |
| MELD | 2025 | validation | late | f1.sadness | 43.835616 | 46.575342 | 2.739726 |
| MELD | 2025 | validation | late | f1.joy | 55.913978 | 56.521739 | 0.607761 |
| MELD | 2025 | validation | late | f1.disgust | 26.666667 | 37.037037 | 10.370370 |
| MELD | 2025 | validation | late | f1.anger | 59.016393 | 57.142857 | -1.873536 |
| MELD | 2025 | validation | late | ece | 18.148051 | 18.656844 | 0.508793 |
| MELD | 2025 | validation | late | mce | 31.393179 | 82.159269 | 50.766090 |
| MELD | 2025 | validation | late | rmsce | 19.367725 | 19.797926 | 0.430201 |
| MELD | 2025 | validation | late | auroc_success | 68.794195 | 68.513722 | -0.280474 |
| MELD | 2025 | validation | late | auprc_success | 76.730722 | 77.727426 | 0.996703 |
| MELD | 2025 | test | overall | accuracy | 66.091954 | 64.980843 | -1.111111 |
| MELD | 2025 | test | overall | weighted_f1 | 64.862030 | 64.390863 | -0.471167 |
| MELD | 2025 | test | overall | macro_f1 | 47.455983 | 48.122129 | 0.666145 |
| MELD | 2025 | test | overall | f1.neutral | 79.635950 | 79.150579 | -0.485371 |
| MELD | 2025 | test | overall | f1.surprise | 56.384743 | 55.934959 | -0.449784 |
| MELD | 2025 | test | overall | f1.fear | 16.216216 | 25.287356 | 9.071140 |
| MELD | 2025 | test | overall | f1.sadness | 35.598706 | 38.509317 | 2.910611 |
| MELD | 2025 | test | overall | f1.joy | 62.312925 | 59.916782 | -2.396143 |
| MELD | 2025 | test | overall | f1.disgust | 29.411765 | 26.573427 | -2.838338 |
| MELD | 2025 | test | overall | f1.anger | 52.631579 | 51.482480 | -1.149099 |
| MELD | 2025 | test | overall | ece | 19.794715 | 18.400303 | -1.394412 |
| MELD | 2025 | test | overall | mce | 31.384727 | 30.690924 | -0.693803 |
| MELD | 2025 | test | overall | rmsce | 21.552970 | 19.913380 | -1.639590 |
| MELD | 2025 | test | overall | auroc_success | 74.989863 | 74.926200 | -0.063663 |
| MELD | 2025 | test | overall | auprc_success | 84.132992 | 84.343820 | 0.210828 |
| MELD | 2025 | test | early | accuracy | 67.657143 | 66.400000 | -1.257143 |
| MELD | 2025 | test | early | weighted_f1 | 66.656163 | 65.966331 | -0.689832 |
| MELD | 2025 | test | early | macro_f1 | 47.249819 | 46.977842 | -0.271977 |
| MELD | 2025 | test | early | f1.neutral | 81.423804 | 80.808081 | -0.615723 |
| MELD | 2025 | test | early | f1.surprise | 50.267380 | 49.180328 | -1.087052 |
| MELD | 2025 | test | early | f1.fear | 25.000000 | 22.222222 | -2.777778 |
| MELD | 2025 | test | early | f1.sadness | 33.333333 | 34.951456 | 1.618123 |
| MELD | 2025 | test | early | f1.joy | 65.369650 | 63.492063 | -1.877586 |
| MELD | 2025 | test | early | f1.disgust | 17.647059 | 21.739130 | 4.092072 |
| MELD | 2025 | test | early | f1.anger | 57.707510 | 56.451613 | -1.255897 |
| MELD | 2025 | test | early | ece | 21.499791 | 18.254795 | -3.244996 |
| MELD | 2025 | test | early | mce | 31.863489 | 80.495574 | 48.632085 |
| MELD | 2025 | test | early | rmsce | 23.114521 | 20.251266 | -2.863254 |
| MELD | 2025 | test | early | auroc_success | 76.884968 | 76.693948 | -0.191020 |
| MELD | 2025 | test | early | auprc_success | 86.420631 | 86.929497 | 0.508867 |
| MELD | 2025 | test | middle | accuracy | 65.348837 | 64.302326 | -1.046512 |
| MELD | 2025 | test | middle | weighted_f1 | 63.766913 | 63.733541 | -0.033372 |
| MELD | 2025 | test | middle | macro_f1 | 46.925080 | 47.930121 | 1.005041 |
| MELD | 2025 | test | middle | f1.neutral | 79.346558 | 79.091995 | -0.254563 |
| MELD | 2025 | test | middle | f1.surprise | 58.035714 | 57.894737 | -0.140977 |
| MELD | 2025 | test | middle | f1.fear | 13.333333 | 26.315789 | 12.982456 |
| MELD | 2025 | test | middle | f1.sadness | 41.818182 | 50.000000 | 8.181818 |
| MELD | 2025 | test | middle | f1.joy | 59.154930 | 56.872038 | -2.282892 |
| MELD | 2025 | test | middle | f1.disgust | 25.806452 | 18.181818 | -7.624633 |
| MELD | 2025 | test | middle | f1.anger | 50.980392 | 47.154472 | -3.825921 |
| MELD | 2025 | test | middle | ece | 19.278219 | 18.082228 | -1.195991 |
| MELD | 2025 | test | middle | mce | 31.755998 | 28.112612 | -3.643386 |
| MELD | 2025 | test | middle | rmsce | 21.105910 | 19.848804 | -1.257105 |
| MELD | 2025 | test | middle | auroc_success | 73.758628 | 75.041085 | 1.282457 |
| MELD | 2025 | test | middle | auprc_success | 82.495749 | 83.640359 | 1.144610 |
| MELD | 2025 | test | late | accuracy | 65.257143 | 64.228571 | -1.028571 |
| MELD | 2025 | test | late | weighted_f1 | 64.159346 | 63.486258 | -0.673088 |
| MELD | 2025 | test | late | macro_f1 | 47.702013 | 48.736299 | 1.034286 |
| MELD | 2025 | test | late | f1.neutral | 78.093076 | 77.494200 | -0.598877 |
| MELD | 2025 | test | late | f1.surprise | 60.416667 | 59.803922 | -0.612745 |
| MELD | 2025 | test | late | f1.fear | 10.000000 | 27.272727 | 17.272727 |
| MELD | 2025 | test | late | f1.sadness | 31.067961 | 29.126214 | -1.941748 |
| MELD | 2025 | test | late | f1.joy | 61.886792 | 58.914729 | -2.972064 |
| MELD | 2025 | test | late | f1.disgust | 43.243243 | 37.735849 | -5.507394 |
| MELD | 2025 | test | late | f1.anger | 49.206349 | 50.806452 | 1.600102 |
| MELD | 2025 | test | late | ece | 19.348197 | 18.902121 | -0.446076 |
| MELD | 2025 | test | late | mce | 31.037863 | 31.903135 | 0.865272 |
| MELD | 2025 | test | late | rmsce | 21.158380 | 20.279988 | -0.878392 |
| MELD | 2025 | test | late | auroc_success | 74.240137 | 73.070845 | -1.169293 |
| MELD | 2025 | test | late | auprc_success | 83.731645 | 81.877887 | -1.853757 |
| MELD | 3407 | validation | overall | accuracy | 66.005410 | 65.284040 | -0.721371 |
| MELD | 3407 | validation | overall | weighted_f1 | 64.422252 | 64.113608 | -0.308644 |
| MELD | 3407 | validation | overall | macro_f1 | 51.093533 | 51.361376 | 0.267843 |
| MELD | 3407 | validation | overall | f1.neutral | 78.320312 | 77.614858 | -0.705454 |
| MELD | 3407 | validation | overall | f1.surprise | 67.289720 | 67.313916 | 0.024196 |
| MELD | 3407 | validation | overall | f1.fear | 36.363636 | 34.782609 | -1.581028 |
| MELD | 3407 | validation | overall | f1.sadness | 39.285714 | 42.708333 | 3.422619 |
| MELD | 3407 | validation | overall | f1.joy | 61.442006 | 61.129568 | -0.312438 |
| MELD | 3407 | validation | overall | f1.disgust | 21.052632 | 24.390244 | 3.337612 |
| MELD | 3407 | validation | overall | f1.anger | 53.900709 | 51.590106 | -2.310603 |
| MELD | 3407 | validation | overall | ece | 25.008241 | 15.967028 | -9.041213 |
| MELD | 3407 | validation | overall | mce | 35.728990 | 21.498943 | -14.230047 |
| MELD | 3407 | validation | overall | rmsce | 26.296412 | 16.540431 | -9.755981 |
| MELD | 3407 | validation | overall | auroc_success | 71.954675 | 71.941236 | -0.013440 |
| MELD | 3407 | validation | overall | auprc_success | 82.069320 | 81.618303 | -0.451017 |
| MELD | 3407 | validation | early | accuracy | 69.168901 | 69.168901 | 0.000000 |
| MELD | 3407 | validation | early | weighted_f1 | 68.010416 | 68.513207 | 0.502791 |
| MELD | 3407 | validation | early | macro_f1 | 52.136921 | 56.002603 | 3.865681 |
| MELD | 3407 | validation | early | f1.neutral | 80.862534 | 80.323450 | -0.539084 |
| MELD | 3407 | validation | early | f1.surprise | 64.516129 | 64.444444 | -0.071685 |
| MELD | 3407 | validation | early | f1.fear | 40.000000 | 47.619048 | 7.619048 |
| MELD | 3407 | validation | early | f1.sadness | 34.782609 | 36.363636 | 1.581028 |
| MELD | 3407 | validation | early | f1.joy | 68.253968 | 69.421488 | 1.167519 |
| MELD | 3407 | validation | early | f1.disgust | 22.222222 | 40.000000 | 17.777778 |
| MELD | 3407 | validation | early | f1.anger | 54.320988 | 53.846154 | -0.474834 |
| MELD | 3407 | validation | early | ece | 27.023400 | 18.453869 | -8.569531 |
| MELD | 3407 | validation | early | mce | 41.180936 | 26.446279 | -14.734657 |
| MELD | 3407 | validation | early | rmsce | 28.724457 | 19.212439 | -9.512019 |
| MELD | 3407 | validation | early | auroc_success | 75.055612 | 74.529828 | -0.525784 |
| MELD | 3407 | validation | early | auprc_success | 86.433423 | 85.990182 | -0.443241 |
| MELD | 3407 | validation | middle | accuracy | 66.666667 | 66.391185 | -0.275482 |
| MELD | 3407 | validation | middle | weighted_f1 | 64.866612 | 64.700191 | -0.166421 |
| MELD | 3407 | validation | middle | macro_f1 | 52.809939 | 52.416979 | -0.392961 |
| MELD | 3407 | validation | middle | f1.neutral | 77.707006 | 77.258567 | -0.448439 |
| MELD | 3407 | validation | middle | f1.surprise | 76.470588 | 76.562500 | 0.091912 |
| MELD | 3407 | validation | middle | f1.fear | 48.275862 | 41.379310 | -6.896552 |
| MELD | 3407 | validation | middle | f1.sadness | 44.444444 | 47.457627 | 3.013183 |
| MELD | 3407 | validation | middle | f1.joy | 55.555556 | 57.142857 | 1.587302 |
| MELD | 3407 | validation | middle | f1.disgust | 16.666667 | 18.181818 | 1.515152 |
| MELD | 3407 | validation | middle | f1.anger | 50.549451 | 48.936170 | -1.613280 |
| MELD | 3407 | validation | middle | ece | 25.918952 | 17.212404 | -8.706548 |
| MELD | 3407 | validation | middle | mce | 41.095495 | 35.336399 | -5.759096 |
| MELD | 3407 | validation | middle | rmsce | 27.031816 | 18.798152 | -8.233664 |
| MELD | 3407 | validation | middle | auroc_success | 69.855201 | 70.576832 | 0.721630 |
| MELD | 3407 | validation | middle | auprc_success | 82.040443 | 83.604887 | 1.564444 |
| MELD | 3407 | validation | late | accuracy | 62.198391 | 60.321716 | -1.876676 |
| MELD | 3407 | validation | late | weighted_f1 | 60.206077 | 59.104983 | -1.101095 |
| MELD | 3407 | validation | late | macro_f1 | 45.824773 | 44.657472 | -1.167300 |
| MELD | 3407 | validation | late | f1.neutral | 76.106195 | 74.924471 | -1.181723 |
| MELD | 3407 | validation | late | f1.surprise | 56.521739 | 57.142857 | 0.621118 |
| MELD | 3407 | validation | late | f1.fear | 11.764706 | 10.526316 | -1.238390 |
| MELD | 3407 | validation | late | f1.sadness | 38.235294 | 43.589744 | 5.354449 |
| MELD | 3407 | validation | late | f1.joy | 58.252427 | 54.166667 | -4.085761 |
| MELD | 3407 | validation | late | f1.disgust | 23.529412 | 20.000000 | -3.529412 |
| MELD | 3407 | validation | late | f1.anger | 56.363636 | 52.252252 | -4.111384 |
| MELD | 3407 | validation | late | ece | 22.106786 | 14.047853 | -8.058933 |
| MELD | 3407 | validation | late | mce | 31.500662 | 25.687990 | -5.812672 |
| MELD | 3407 | validation | late | rmsce | 24.053582 | 15.874684 | -8.178898 |
| MELD | 3407 | validation | late | auroc_success | 70.530692 | 70.399399 | -0.131293 |
| MELD | 3407 | validation | late | auprc_success | 77.297253 | 74.560356 | -2.736897 |
| MELD | 3407 | test | overall | accuracy | 66.398467 | 66.360153 | -0.038314 |
| MELD | 3407 | test | overall | weighted_f1 | 65.247282 | 65.344859 | 0.097577 |
| MELD | 3407 | test | overall | macro_f1 | 49.151710 | 49.122690 | -0.029020 |
| MELD | 3407 | test | overall | f1.neutral | 79.210926 | 79.097744 | -0.113181 |
| MELD | 3407 | test | overall | f1.surprise | 57.049180 | 58.175559 | 1.126379 |
| MELD | 3407 | test | overall | f1.fear | 23.809524 | 26.804124 | 2.994600 |
| MELD | 3407 | test | overall | f1.sadness | 38.414634 | 40.883978 | 2.469344 |
| MELD | 3407 | test | overall | f1.joy | 64.118372 | 62.222222 | -1.896150 |
| MELD | 3407 | test | overall | f1.disgust | 29.906542 | 23.300971 | -6.605571 |
| MELD | 3407 | test | overall | f1.anger | 51.552795 | 53.374233 | 1.821438 |
| MELD | 3407 | test | overall | ece | 25.013815 | 16.231511 | -8.782305 |
| MELD | 3407 | test | overall | mce | 36.590477 | 80.765527 | 44.175050 |
| MELD | 3407 | test | overall | rmsce | 26.961842 | 17.070389 | -9.891453 |
| MELD | 3407 | test | overall | auroc_success | 75.090684 | 74.278291 | -0.812393 |
| MELD | 3407 | test | overall | auprc_success | 84.950117 | 84.176727 | -0.773390 |
| MELD | 3407 | test | early | accuracy | 67.885714 | 68.228571 | 0.342857 |
| MELD | 3407 | test | early | weighted_f1 | 66.872774 | 67.416645 | 0.543871 |
| MELD | 3407 | test | early | macro_f1 | 47.931336 | 49.849740 | 1.918405 |
| MELD | 3407 | test | early | f1.neutral | 81.367144 | 80.962801 | -0.404344 |
| MELD | 3407 | test | early | f1.surprise | 50.810811 | 52.513966 | 1.703156 |
| MELD | 3407 | test | early | f1.fear | 23.076923 | 27.586207 | 4.509284 |
| MELD | 3407 | test | early | f1.sadness | 36.363636 | 43.243243 | 6.879607 |
| MELD | 3407 | test | early | f1.joy | 66.423358 | 64.885496 | -1.537861 |
| MELD | 3407 | test | early | f1.disgust | 21.621622 | 22.222222 | 0.600601 |
| MELD | 3407 | test | early | f1.anger | 55.855856 | 57.534247 | 1.678391 |
| MELD | 3407 | test | early | ece | 25.761394 | 16.556223 | -9.205170 |
| MELD | 3407 | test | early | mce | 40.117679 | 80.765527 | 40.647848 |
| MELD | 3407 | test | early | rmsce | 27.474712 | 17.445399 | -10.029313 |
| MELD | 3407 | test | early | auroc_success | 75.510742 | 74.014557 | -1.496185 |
| MELD | 3407 | test | early | auprc_success | 86.425976 | 84.795136 | -1.630840 |
| MELD | 3407 | test | middle | accuracy | 65.813953 | 65.232558 | -0.581395 |
| MELD | 3407 | test | middle | weighted_f1 | 64.520778 | 63.958699 | -0.562079 |
| MELD | 3407 | test | middle | macro_f1 | 49.780103 | 48.153634 | -1.626469 |
| MELD | 3407 | test | middle | f1.neutral | 78.830409 | 78.110599 | -0.719810 |
| MELD | 3407 | test | middle | f1.surprise | 58.407080 | 58.215962 | -0.191117 |
| MELD | 3407 | test | middle | f1.fear | 23.529412 | 29.268293 | 5.738881 |
| MELD | 3407 | test | middle | f1.sadness | 46.280992 | 48.854962 | 2.573970 |
| MELD | 3407 | test | middle | f1.joy | 61.344538 | 60.360360 | -0.984177 |
| MELD | 3407 | test | middle | f1.disgust | 30.303030 | 12.500000 | -17.803030 |
| MELD | 3407 | test | middle | f1.anger | 49.765258 | 49.765258 | 0.000000 |
| MELD | 3407 | test | middle | ece | 24.611103 | 15.634243 | -8.976860 |
| MELD | 3407 | test | middle | mce | 34.250376 | 24.072833 | -10.177543 |
| MELD | 3407 | test | middle | rmsce | 26.854390 | 17.124222 | -9.730168 |
| MELD | 3407 | test | middle | auroc_success | 74.905050 | 76.124217 | 1.219166 |
| MELD | 3407 | test | middle | auprc_success | 84.095148 | 84.435579 | 0.340430 |
| MELD | 3407 | test | late | accuracy | 65.485714 | 65.600000 | 0.114286 |
| MELD | 3407 | test | late | weighted_f1 | 64.369381 | 64.702609 | 0.333228 |
| MELD | 3407 | test | late | macro_f1 | 49.427038 | 48.867051 | -0.559987 |
| MELD | 3407 | test | late | f1.neutral | 77.345538 | 78.132118 | 0.786581 |
| MELD | 3407 | test | late | f1.surprise | 61.306533 | 63.492063 | 2.185531 |
| MELD | 3407 | test | late | f1.fear | 25.000000 | 22.222222 | -2.777778 |
| MELD | 3407 | test | late | f1.sadness | 31.481481 | 30.000000 | -1.481481 |
| MELD | 3407 | test | late | f1.joy | 64.214047 | 61.209964 | -3.004082 |
| MELD | 3407 | test | late | f1.disgust | 37.837838 | 34.285714 | -3.552124 |
| MELD | 3407 | test | late | f1.anger | 48.803828 | 52.727273 | 3.923445 |
| MELD | 3407 | test | late | ece | 24.662046 | 16.802748 | -7.859298 |
| MELD | 3407 | test | late | mce | 37.742054 | 80.298787 | 42.556734 |
| MELD | 3407 | test | late | rmsce | 26.788454 | 18.172868 | -8.615587 |
| MELD | 3407 | test | late | auroc_success | 74.951747 | 72.565895 | -2.385852 |
| MELD | 3407 | test | late | auprc_success | 84.485743 | 83.542370 | -0.943373 |

## Across-seed descriptive summaries

| Dataset | Split | Segment | Metric | n | Baseline mean ± SD | CMERC mean ± SD | Paired delta mean ± SD | 95% t interval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | test | early | accuracy | 3 | 62.962963 ± 0.667695 | 62.222222 ± 1.031067 | -0.740741 ± 0.641500 | [-2.334316, 0.852834] |
| IEMOCAP | test | early | auprc_success | 3 | 75.347348 ± 1.181891 | 78.848533 ± 0.697926 | 3.501185 ± 1.490411 | [-0.201201, 7.203570] |
| IEMOCAP | test | early | auroc_success | 3 | 64.042649 ± 2.981171 | 69.283617 ± 0.723330 | 5.240968 ± 3.002274 | [-2.217095, 12.699031] |
| IEMOCAP | test | early | ece | 3 | 20.621819 ± 3.397519 | 17.964370 ± 3.427367 | -2.657449 ± 6.803338 | [-19.557877, 14.242979] |
| IEMOCAP | test | early | f1.angry | 3 | 41.393408 ± 4.589775 | 19.393939 ± 3.043878 | -21.999469 ± 7.091021 | [-39.614541, -4.384397] |
| IEMOCAP | test | early | f1.excited | 3 | 77.093637 ± 0.638914 | 75.290097 ± 1.366750 | -1.803541 ± 1.171889 | [-4.714674, 1.107592] |
| IEMOCAP | test | early | f1.frustrated | 3 | 65.568459 ± 2.213548 | 66.521883 ± 0.125405 | 0.953424 ± 2.095315 | [-4.251627, 6.158475] |
| IEMOCAP | test | early | f1.happy | 3 | 27.954017 ± 9.981410 | 29.198682 ± 3.202589 | 1.244665 ± 9.142361 | [-21.466219, 23.955549] |
| IEMOCAP | test | early | f1.neutral | 3 | 59.435942 ± 1.641220 | 59.231822 ± 1.435131 | -0.204120 ± 2.867404 | [-7.327147, 6.918908] |
| IEMOCAP | test | early | f1.sad | 3 | 63.654269 ± 5.644295 | 65.066533 ± 1.984774 | 1.412264 ± 4.163019 | [-8.929249, 11.753776] |
| IEMOCAP | test | early | macro_f1 | 3 | 55.849956 ± 1.838523 | 52.450493 ± 1.215746 | -3.399463 ± 0.890731 | [-5.612161, -1.186765] |
| IEMOCAP | test | early | mce | 3 | 48.209700 ± 17.350542 | 44.856794 ± 23.823065 | -3.352905 ± 38.808474 | [-99.758499, 93.052688] |
| IEMOCAP | test | early | rmsce | 3 | 22.303613 ± 3.218101 | 19.304186 ± 3.926447 | -2.999426 ± 7.074406 | [-20.573226, 14.574373] |
| IEMOCAP | test | early | weighted_f1 | 3 | 62.523525 ± 1.113881 | 61.434912 ± 0.972857 | -1.088612 ± 0.853691 | [-3.209299, 1.032075] |
| IEMOCAP | test | late | accuracy | 3 | 65.370370 ± 2.612359 | 67.777778 ± 0.555556 | 2.407407 ± 2.320364 | [-3.356696, 8.171510] |
| IEMOCAP | test | late | auprc_success | 3 | 82.582522 ± 1.267378 | 80.344444 ± 0.416257 | -2.238078 ± 0.858101 | [-4.369718, -0.106438] |
| IEMOCAP | test | late | auroc_success | 3 | 69.718940 ± 4.338363 | 65.755486 ± 0.514438 | -3.963454 ± 3.974882 | [-13.837609, 5.910700] |
| IEMOCAP | test | late | ece | 3 | 18.312243 ± 4.146693 | 14.291516 ± 2.284307 | -4.020727 ± 6.346396 | [-19.786048, 11.744595] |
| IEMOCAP | test | late | f1.angry | 3 | 72.065706 ± 1.412194 | 69.105772 ± 1.307698 | -2.959934 ± 2.307787 | [-8.692796, 2.772927] |
| IEMOCAP | test | late | f1.excited | 3 | 69.971184 ± 1.384371 | 70.657653 ± 0.603626 | 0.686468 ± 1.476267 | [-2.980782, 4.353718] |
| IEMOCAP | test | late | f1.frustrated | 3 | 59.802067 ± 4.652994 | 64.489679 ± 1.388554 | 4.687612 ± 5.773560 | [-9.654705, 19.029929] |
| IEMOCAP | test | late | f1.happy | 3 | 48.187339 ± 8.985191 | 57.219529 ± 2.663061 | 9.032190 ± 11.128982 | [-18.613733, 36.678114] |
| IEMOCAP | test | late | f1.neutral | 3 | 63.259736 ± 0.982800 | 63.157661 ± 3.409396 | -0.102075 ± 2.783575 | [-7.016858, 6.812708] |
| IEMOCAP | test | late | f1.sad | 3 | 75.495788 ± 2.479036 | 81.403319 ± 1.049840 | 5.907532 ± 2.006139 | [0.924005, 10.891058] |
| IEMOCAP | test | late | macro_f1 | 3 | 64.796970 ± 3.055880 | 67.672269 ± 0.576815 | 2.875299 ± 2.991693 | [-4.556479, 10.307076] |
| IEMOCAP | test | late | mce | 3 | 47.211195 ± 18.110886 | 27.541796 ± 6.497017 | -19.669399 ± 12.813174 | [-51.499087, 12.160289] |
| IEMOCAP | test | late | rmsce | 3 | 19.408182 ± 3.674389 | 15.632699 ± 2.542330 | -3.775483 ± 6.172045 | [-19.107692, 11.556725] |
| IEMOCAP | test | late | weighted_f1 | 3 | 64.862076 ± 2.986723 | 67.711722 ± 0.566984 | 2.849646 ± 2.866960 | [-4.272278, 9.971570] |
| IEMOCAP | test | middle | accuracy | 3 | 66.175568 ± 0.562624 | 66.052793 ± 1.609005 | -0.122775 ± 1.969185 | [-5.014502, 4.768953] |
| IEMOCAP | test | middle | auprc_success | 3 | 79.030250 ± 0.848512 | 79.215652 ± 0.562710 | 0.185402 ± 1.359423 | [-3.191591, 3.562395] |
| IEMOCAP | test | middle | auroc_success | 3 | 66.174406 ± 1.055598 | 65.991192 ± 1.716682 | -0.183214 ± 2.051438 | [-5.279270, 4.912841] |
| IEMOCAP | test | middle | ece | 3 | 16.857870 ± 3.528086 | 15.656095 ± 3.797329 | -1.201775 ± 7.030073 | [-18.665446, 16.261896] |
| IEMOCAP | test | middle | f1.angry | 3 | 59.353524 ± 2.505435 | 52.053140 ± 4.212038 | -7.300384 ± 5.818466 | [-21.754255, 7.153486] |
| IEMOCAP | test | middle | f1.excited | 3 | 75.526946 ± 2.132115 | 72.639321 ± 1.773626 | -2.887624 ± 2.580707 | [-9.298456, 3.523207] |
| IEMOCAP | test | middle | f1.frustrated | 3 | 61.589860 ± 2.985646 | 66.585074 ± 1.978974 | 4.995213 ± 4.181277 | [-5.391655, 15.382082] |
| IEMOCAP | test | middle | f1.happy | 3 | 40.728716 ± 10.079339 | 39.874157 ± 8.388040 | -0.854559 ± 15.326496 | [-38.927685, 37.218567] |
| IEMOCAP | test | middle | f1.neutral | 3 | 69.046339 ± 1.646794 | 66.912502 ± 1.532026 | -2.133837 ± 2.526735 | [-8.410594, 4.142919] |
| IEMOCAP | test | middle | f1.sad | 3 | 73.867078 ± 4.669110 | 77.016350 ± 3.252159 | 3.149272 ± 7.189780 | [-14.711131, 21.009674] |
| IEMOCAP | test | middle | macro_f1 | 3 | 63.352077 ± 2.258728 | 62.513424 ± 2.854348 | -0.838653 ± 4.504412 | [-12.028234, 10.350928] |
| IEMOCAP | test | middle | mce | 3 | 57.377537 ± 17.609681 | 43.396613 ± 21.772641 | -13.980924 ± 19.045638 | [-61.292912, 33.331063] |
| IEMOCAP | test | middle | rmsce | 3 | 18.358748 ± 3.045904 | 17.139182 ± 3.435458 | -1.219567 ± 6.353380 | [-17.002238, 14.563105] |
| IEMOCAP | test | middle | weighted_f1 | 3 | 66.276224 ± 1.044444 | 66.195197 ± 1.805801 | -0.081028 ± 2.714330 | [-6.823798, 6.661743] |
| IEMOCAP | test | overall | accuracy | 3 | 64.838776 ± 1.144987 | 65.352228 ± 0.410248 | 0.513452 ± 1.522177 | [-3.267844, 4.294749] |
| IEMOCAP | test | overall | auprc_success | 3 | 79.069231 ± 0.183364 | 79.325191 ± 0.105161 | 0.255960 ± 0.211607 | [-0.269702, 0.781622] |
| IEMOCAP | test | overall | auroc_success | 3 | 66.652986 ± 1.257064 | 67.099929 ± 0.645982 | 0.446943 ± 1.877724 | [-4.217582, 5.111468] |
| IEMOCAP | test | overall | ece | 3 | 18.184590 ± 3.955896 | 15.504165 ± 3.370485 | -2.680425 ± 7.298827 | [-20.811717, 15.450868] |
| IEMOCAP | test | overall | f1.angry | 3 | 62.765084 ± 1.342243 | 56.644346 ± 1.939307 | -6.120738 ± 3.229431 | [-14.143088, 1.901612] |
| IEMOCAP | test | overall | f1.excited | 3 | 74.463564 ± 0.868198 | 73.112637 ± 0.520426 | -1.350927 ± 0.679509 | [-3.038920, 0.337066] |
| IEMOCAP | test | overall | f1.frustrated | 3 | 62.523823 ± 2.605562 | 65.975370 ± 0.989494 | 3.451547 ± 3.286442 | [-4.712428, 11.615522] |
| IEMOCAP | test | overall | f1.happy | 3 | 41.450391 ± 7.076996 | 45.505955 ± 3.413211 | 4.055563 ± 10.453260 | [-21.911774, 30.022900] |
| IEMOCAP | test | overall | f1.neutral | 3 | 63.744984 ± 0.821276 | 62.933455 ± 0.888706 | -0.811529 ± 1.160492 | [-3.694352, 2.071293] |
| IEMOCAP | test | overall | f1.sad | 3 | 71.486602 ± 3.395146 | 75.233208 ± 0.080626 | 3.746606 ± 3.423432 | [-4.757671, 12.250883] |
| IEMOCAP | test | overall | macro_f1 | 3 | 62.739075 ± 2.049789 | 63.234162 ± 0.954250 | 0.495087 ± 2.950802 | [-6.835111, 7.825285] |
| IEMOCAP | test | overall | mce | 3 | 32.018624 ± 3.746062 | 24.119135 ± 4.313005 | -7.899489 ± 7.780182 | [-27.226533, 11.427555] |
| IEMOCAP | test | overall | rmsce | 3 | 19.292041 ± 3.442085 | 16.539422 ± 3.216739 | -2.752619 ± 6.602552 | [-19.154267, 13.649030] |
| IEMOCAP | test | overall | weighted_f1 | 3 | 64.520884 ± 1.553512 | 65.174537 ± 0.583886 | 0.653653 ± 2.086690 | [-4.529971, 5.837278] |
| IEMOCAP | validation | early | accuracy | 3 | 53.107345 ± 2.989550 | 53.860640 ± 3.212565 | 0.753296 ± 3.310432 | [-7.470274, 8.976866] |
| IEMOCAP | validation | early | auprc_success | 3 | 58.927778 ± 3.763434 | 61.031945 ± 4.635679 | 2.104167 ± 4.076217 | [-8.021718, 12.230052] |
| IEMOCAP | validation | early | auroc_success | 3 | 59.382062 ± 2.473129 | 61.018315 ± 2.474796 | 1.636252 ± 0.925431 | [-0.662646, 3.935151] |
| IEMOCAP | validation | early | ece | 3 | 28.418789 ± 4.693754 | 22.657916 ± 3.239326 | -5.760872 ± 5.943400 | [-20.525097, 9.003353] |
| IEMOCAP | validation | early | f1.angry | 3 | 58.430898 ± 2.791328 | 43.350010 ± 5.845342 | -15.080888 ± 4.808883 | [-27.026814, -3.134961] |
| IEMOCAP | validation | early | f1.excited | 3 | 47.705451 ± 3.729353 | 49.539843 ± 7.644309 | 1.834392 ± 6.641391 | [-14.663737, 18.332521] |
| IEMOCAP | validation | early | f1.frustrated | 3 | 23.148148 ± 1.603751 | 31.198257 ± 8.890170 | 8.050109 ± 10.321849 | [-17.590786, 33.691004] |
| IEMOCAP | validation | early | f1.happy | 3 | 65.857119 ± 6.779620 | 68.652795 ± 5.497654 | 2.795676 ± 10.758249 | [-23.929297, 29.520649] |
| IEMOCAP | validation | early | f1.neutral | 3 | 51.894691 ± 3.374420 | 58.395990 ± 2.010644 | 6.501299 ± 2.882613 | [-0.659509, 13.662107] |
| IEMOCAP | validation | early | f1.sad | 3 | 67.777778 ± 1.924501 | 63.902439 ± 4.167807 | -3.875339 ± 5.838017 | [-18.377778, 10.627100] |
| IEMOCAP | validation | early | macro_f1 | 3 | 52.469014 ± 2.220879 | 52.506556 ± 3.772983 | 0.037542 ± 2.891857 | [-7.146229, 7.221312] |
| IEMOCAP | validation | early | mce | 3 | 51.404830 ± 18.173148 | 67.022322 ± 8.879772 | 15.617492 ± 17.717475 | [-28.395157, 59.630141] |
| IEMOCAP | validation | early | rmsce | 3 | 30.671546 ± 4.733701 | 26.322445 ± 4.328510 | -4.349100 ± 6.695656 | [-20.982032, 12.283832] |
| IEMOCAP | validation | early | weighted_f1 | 3 | 52.829127 ± 2.663803 | 53.719658 ± 3.444745 | 0.890531 ± 3.144214 | [-6.920131, 8.701193] |
| IEMOCAP | validation | late | accuracy | 3 | 70.244821 ± 2.352165 | 70.621469 ± 1.957120 | 0.376648 ± 3.968231 | [-9.480985, 10.234281] |
| IEMOCAP | validation | late | auprc_success | 3 | 80.784513 ± 2.170700 | 76.691459 ± 0.883360 | -4.093054 ± 3.045990 | [-11.659713, 3.473604] |
| IEMOCAP | validation | late | auroc_success | 3 | 64.494454 ± 2.350967 | 60.713594 ± 2.031377 | -3.780860 ± 3.989042 | [-13.690191, 6.128470] |
| IEMOCAP | validation | late | ece | 3 | 18.684611 ± 0.558409 | 16.158299 ± 1.462884 | -2.526312 ± 0.990470 | [-4.986776, -0.065849] |
| IEMOCAP | validation | late | f1.angry | 3 | 92.029740 ± 2.861585 | 92.408723 ± 0.577434 | 0.378984 ± 3.272987 | [-7.751568, 8.509535] |
| IEMOCAP | validation | late | f1.excited | 3 | 48.263250 ± 4.595016 | 46.806938 ± 3.767374 | -1.456312 ± 8.198492 | [-21.822495, 18.909871] |
| IEMOCAP | validation | late | f1.frustrated | 3 | 73.291050 ± 4.649160 | 75.609721 ± 5.091901 | 2.318671 ± 4.076536 | [-7.808005, 12.445348] |
| IEMOCAP | validation | late | f1.happy | 3 | 56.967320 ± 2.683148 | 62.802967 ± 2.432798 | 5.835647 ± 3.885333 | [-3.816055, 15.487349] |
| IEMOCAP | validation | late | f1.neutral | 3 | 54.602373 ± 7.895759 | 42.765957 ± 6.322408 | -11.836415 ± 13.185504 | [-44.591022, 20.918191] |
| IEMOCAP | validation | late | f1.sad | 3 | 72.748141 ± 0.930283 | 76.942049 ± 1.479928 | 4.193908 ± 2.261980 | [-1.425162, 9.812978] |
| IEMOCAP | validation | late | macro_f1 | 3 | 66.316979 ± 2.038158 | 66.222726 ± 2.576704 | -0.094253 ± 4.251041 | [-10.654423, 10.465917] |
| IEMOCAP | validation | late | mce | 3 | 47.001640 ± 12.631694 | 42.659442 ± 18.808418 | -4.342197 ± 29.065154 | [-76.544044, 67.859649] |
| IEMOCAP | validation | late | rmsce | 3 | 20.181534 ± 0.285661 | 17.779858 ± 1.379588 | -2.401676 ± 1.122653 | [-5.190501, 0.387149] |
| IEMOCAP | validation | late | weighted_f1 | 3 | 70.242263 ± 2.305370 | 69.948689 ± 2.198686 | -0.293575 ± 4.081189 | [-10.431811, 9.844661] |
| IEMOCAP | validation | middle | accuracy | 3 | 62.042389 ± 4.181606 | 60.693642 ± 3.218361 | -1.348748 ± 1.858122 | [-5.964577, 3.267082] |
| IEMOCAP | validation | middle | auprc_success | 3 | 69.306592 ± 4.093932 | 68.310179 ± 0.727970 | -0.996413 ± 4.412340 | [-11.957273, 9.964447] |
| IEMOCAP | validation | middle | auroc_success | 3 | 60.735490 ± 1.437320 | 63.010405 ± 3.333296 | 2.274915 ± 4.687185 | [-9.368698, 13.918528] |
| IEMOCAP | validation | middle | ece | 3 | 20.859823 ± 4.386979 | 16.838569 ± 2.004363 | -4.021254 ± 5.826966 | [-18.496240, 10.453733] |
| IEMOCAP | validation | middle | f1.angry | 3 | 65.461147 ± 5.380925 | 65.082009 ± 5.667469 | -0.379139 ± 4.004301 | [-10.326373, 9.568096] |
| IEMOCAP | validation | middle | f1.excited | 3 | 74.511507 ± 1.805003 | 68.178241 ± 6.450655 | -6.333266 ± 5.436355 | [-19.837921, 7.171389] |
| IEMOCAP | validation | middle | f1.frustrated | 3 | 38.082628 ± 10.019323 | 44.120248 ± 2.464608 | 6.037619 ± 7.849501 | [-13.461621, 25.536860] |
| IEMOCAP | validation | middle | f1.happy | 3 | 62.226891 ± 3.136098 | 50.934173 ± 1.895441 | -11.292718 ± 3.345701 | [-19.603900, -2.981536] |
| IEMOCAP | validation | middle | f1.neutral | 3 | 51.401312 ± 8.780852 | 50.441640 ± 4.024666 | -0.959672 ± 8.557456 | [-22.217571, 20.298228] |
| IEMOCAP | validation | middle | f1.sad | 3 | 78.905917 ± 4.642859 | 81.746032 ± 2.749287 | 2.840114 ± 7.085030 | [-14.760076, 20.440305] |
| IEMOCAP | validation | middle | macro_f1 | 3 | 61.764900 ± 3.729504 | 60.083724 ± 2.873551 | -1.681177 ± 1.575429 | [-5.594759, 2.232405] |
| IEMOCAP | validation | middle | mce | 3 | 38.422070 ± 21.601650 | 32.722598 ± 3.041153 | -5.699472 ± 18.560992 | [-51.807532, 40.408589] |
| IEMOCAP | validation | middle | rmsce | 3 | 22.770391 ± 4.308399 | 18.980512 ± 2.096961 | -3.789879 ± 6.143610 | [-19.051451, 11.471694] |
| IEMOCAP | validation | middle | weighted_f1 | 3 | 61.631382 ± 4.079932 | 60.567487 ± 3.066242 | -1.063895 ± 1.800255 | [-5.535976, 3.408186] |
| IEMOCAP | validation | overall | accuracy | 3 | 61.796331 ± 2.247863 | 61.733080 ± 2.357324 | -0.063251 ± 0.219108 | [-0.607546, 0.481044] |
| IEMOCAP | validation | overall | auprc_success | 3 | 71.662812 ± 2.844154 | 70.014210 ± 0.837031 | -1.648602 ± 2.023559 | [-6.675402, 3.378197] |
| IEMOCAP | validation | overall | auroc_success | 3 | 62.493186 ± 2.236514 | 62.683850 ± 0.617304 | 0.190664 ± 2.824108 | [-6.824809, 7.206137] |
| IEMOCAP | validation | overall | ece | 3 | 21.959176 ± 2.895528 | 16.713749 ± 1.363865 | -5.245427 ± 3.364377 | [-13.603002, 3.112148] |
| IEMOCAP | validation | overall | f1.angry | 3 | 75.476371 ± 2.805888 | 73.534460 ± 2.422099 | -1.941911 ± 1.180158 | [-4.873587, 0.989765] |
| IEMOCAP | validation | overall | f1.excited | 3 | 56.956238 ± 1.013482 | 54.893260 ± 5.841863 | -2.062977 ± 5.772202 | [-16.401922, 12.275967] |
| IEMOCAP | validation | overall | f1.frustrated | 3 | 45.401439 ± 4.256354 | 50.240997 ± 4.609841 | 4.839559 ± 1.091474 | [2.128186, 7.550931] |
| IEMOCAP | validation | overall | f1.happy | 3 | 61.651109 ± 4.052464 | 61.988343 ± 1.923033 | 0.337234 ± 5.613404 | [-13.607235, 14.281704] |
| IEMOCAP | validation | overall | f1.neutral | 3 | 52.350380 ± 3.935900 | 52.772758 ± 3.006958 | 0.422378 ± 3.971187 | [-9.442597, 10.287354] |
| IEMOCAP | validation | overall | f1.sad | 3 | 73.743171 ± 1.750649 | 75.334158 ± 0.717399 | 1.590986 ± 2.222934 | [-3.931088, 7.113061] |
| IEMOCAP | validation | overall | macro_f1 | 3 | 60.929785 ± 2.001009 | 61.460663 ± 2.422097 | 0.530878 ± 0.651120 | [-1.086594, 2.148350] |
| IEMOCAP | validation | overall | mce | 3 | 42.217329 ± 26.151743 | 48.261138 ± 22.107241 | 6.043809 ± 41.236268 | [-96.392761, 108.480378] |
| IEMOCAP | validation | overall | rmsce | 3 | 23.149330 ± 3.109371 | 19.024329 ± 1.998566 | -4.125001 ± 4.315968 | [-14.846461, 6.596459] |
| IEMOCAP | validation | overall | weighted_f1 | 3 | 61.760864 ± 2.089200 | 62.038097 ± 2.250578 | 0.277233 ± 0.199791 | [-0.219076, 0.773543] |
| MELD | test | early | accuracy | 3 | 67.047619 ± 1.258873 | 67.809524 ± 1.253675 | 0.761905 ± 2.257926 | [-4.847095, 6.370905] |
| MELD | test | early | auprc_success | 3 | 85.285150 ± 1.971341 | 84.793951 ± 2.136139 | -0.491199 ± 1.076660 | [-3.165771, 2.183372] |
| MELD | test | early | auroc_success | 3 | 76.290060 ± 0.705429 | 74.197810 ± 2.409743 | -2.092250 ± 2.259031 | [-7.703994, 3.519495] |
| MELD | test | early | ece | 3 | 22.715450 ± 2.655706 | 19.021375 ± 2.924783 | -3.694075 ± 5.300843 | [-16.862098, 9.473949] |
| MELD | test | early | f1.anger | 3 | 54.808478 ± 3.540871 | 57.635674 ± 1.237895 | 2.827196 ± 4.762570 | [-9.003684, 14.658075] |
| MELD | test | early | f1.disgust | 3 | 20.932697 ± 3.001080 | 21.550336 ± 0.783532 | 0.617639 ± 3.465946 | [-7.992247, 9.227524] |
| MELD | test | early | f1.fear | 3 | 24.721293 ± 1.524248 | 24.602810 ± 2.732328 | -0.118483 ± 4.022621 | [-10.111228, 9.874262] |
| MELD | test | early | f1.joy | 3 | 64.733008 ± 2.082966 | 64.675472 ± 1.093627 | -0.057536 ± 2.863252 | [-7.170249, 7.055177] |
| MELD | test | early | f1.neutral | 3 | 80.769674 ± 1.084289 | 81.162981 ± 0.486899 | 0.393307 ± 1.568198 | [-3.502312, 4.288927] |
| MELD | test | early | f1.sadness | 3 | 35.530058 ± 1.920758 | 40.033154 ± 4.451474 | 4.503096 ± 2.667340 | [-2.122944, 11.129136] |
| MELD | test | early | f1.surprise | 3 | 49.707628 ± 1.465552 | 50.935135 ± 1.673772 | 1.227507 ± 2.117194 | [-4.031894, 6.486908] |
| MELD | test | early | macro_f1 | 3 | 47.314691 ± 0.586904 | 48.656509 ± 1.496224 | 1.341818 ± 1.416437 | [-2.176807, 4.860443] |
| MELD | test | early | mce | 3 | 46.324593 ± 18.368673 | 63.603594 ± 29.491862 | 17.279001 ± 47.558430 | [-100.862687, 135.420690] |
| MELD | test | early | rmsce | 3 | 24.720796 ± 2.395960 | 20.406570 ± 3.041798 | -4.314226 ± 5.145397 | [-17.096100, 8.467648] |
| MELD | test | early | weighted_f1 | 3 | 66.007129 ± 1.316215 | 67.055426 ± 0.960835 | 1.048297 ± 2.037718 | [-4.013677, 6.110270] |
| MELD | test | late | accuracy | 3 | 65.333333 ± 0.131966 | 64.761905 ± 0.734754 | -0.571429 ± 0.604743 | [-2.073694, 0.930837] |
| MELD | test | late | auprc_success | 3 | 83.972819 ± 0.444468 | 82.658313 ± 0.837067 | -1.314507 ± 0.477909 | [-2.501699, -0.127314] |
| MELD | test | late | auroc_success | 3 | 74.886556 ± 0.616415 | 72.939931 ± 0.328747 | -1.946626 ± 0.675086 | [-3.623633, -0.269619] |
| MELD | test | late | ece | 3 | 21.400774 ± 2.855689 | 17.833883 ± 1.050178 | -3.566890 ± 3.842973 | [-13.113364, 5.979584] |
| MELD | test | late | f1.anger | 3 | 48.719442 ± 0.534124 | 50.895422 ± 1.789025 | 2.175981 ± 1.542382 | [-1.655508, 6.007469] |
| MELD | test | late | f1.disgust | 3 | 38.791733 ± 4.059506 | 35.771894 ± 1.773995 | -3.019839 ± 2.792014 | [-9.955587, 3.915908] |
| MELD | test | late | f1.fear | 3 | 20.000000 ± 8.660254 | 19.672920 ± 9.144956 | -0.327080 ± 16.511431 | [-41.343748, 40.689587] |
| MELD | test | late | f1.joy | 3 | 62.546434 ± 1.454659 | 60.405201 ± 1.292159 | -2.141233 ± 1.466858 | [-5.785109, 1.502644] |
| MELD | test | late | f1.neutral | 3 | 77.788847 ± 0.392697 | 77.733563 ± 0.347479 | -0.055284 ± 0.739323 | [-1.891864, 1.781295] |
| MELD | test | late | f1.sadness | 3 | 30.289342 ± 1.719205 | 29.544803 ± 0.438042 | -0.744539 ± 1.690759 | [-4.944618, 3.455541] |
| MELD | test | late | f1.surprise | 3 | 61.738421 ± 1.582533 | 61.666038 ± 1.844336 | -0.072383 ± 2.042077 | [-5.145183, 5.000416] |
| MELD | test | late | macro_f1 | 3 | 48.553460 ± 0.862725 | 47.955691 ± 1.466744 | -0.597768 ± 1.651269 | [-4.699749, 3.504212] |
| MELD | test | late | mce | 3 | 50.177362 ± 27.549281 | 47.881654 ± 28.075004 | -2.295708 ± 46.513557 | [-117.841790, 113.250373] |
| MELD | test | late | rmsce | 3 | 23.463533 ± 2.950319 | 19.301367 ± 1.061526 | -4.162166 ± 3.999013 | [-14.096264, 5.771932] |
| MELD | test | late | weighted_f1 | 3 | 64.181342 ± 0.178063 | 63.916490 ± 0.681813 | -0.264852 ± 0.529338 | [-1.579800, 1.050096] |
| MELD | test | middle | accuracy | 3 | 65.116279 ± 0.838500 | 64.689922 ± 0.484108 | -0.426357 ± 0.710477 | [-2.191279, 1.338566] |
| MELD | test | middle | auprc_success | 3 | 83.335041 ± 0.802635 | 84.125996 ± 0.425840 | 0.790955 ± 0.410748 | [-0.229400, 1.811309] |
| MELD | test | middle | auroc_success | 3 | 74.592203 ± 0.729341 | 75.884225 ± 0.752419 | 1.292022 ± 0.078078 | [1.098064, 1.485979] |
| MELD | test | middle | ece | 3 | 20.978482 ± 3.148082 | 17.271311 ± 1.417763 | -3.707171 ± 4.565360 | [-15.048153, 7.633811] |
| MELD | test | middle | f1.anger | 3 | 49.013982 ± 2.430741 | 49.415721 ± 2.108325 | 0.401739 ± 4.442175 | [-10.633234, 11.436712] |
| MELD | test | middle | f1.disgust | 3 | 23.641432 ± 7.967853 | 15.165544 ± 2.857100 | -8.475888 ± 8.931990 | [-30.664181, 13.712406] |
| MELD | test | middle | f1.fear | 3 | 20.130719 ± 5.886709 | 24.588633 ± 5.741496 | 4.457914 ± 9.231919 | [-18.475445, 27.391274] |
| MELD | test | middle | f1.joy | 3 | 60.679310 ± 1.323708 | 58.873682 ± 1.800275 | -1.805628 ± 0.714502 | [-3.580550, -0.030707] |
| MELD | test | middle | f1.neutral | 3 | 78.591707 ± 0.898312 | 78.300708 ± 0.715434 | -0.290998 ± 0.411805 | [-1.313978, 0.731982] |
| MELD | test | middle | f1.sadness | 3 | 41.674083 ± 4.680621 | 47.167340 ± 3.956322 | 5.493257 ± 2.811035 | [-1.489740, 12.476254] |
| MELD | test | middle | f1.surprise | 3 | 58.845564 ± 1.096918 | 58.365625 ± 0.560900 | -0.479939 ± 0.544254 | [-1.831940, 0.872062] |
| MELD | test | middle | macro_f1 | 3 | 47.510971 ± 2.040285 | 47.411036 ± 1.098349 | -0.099935 ± 1.365465 | [-3.491939, 3.292069] |
| MELD | test | middle | mce | 3 | 32.686663 ± 1.362367 | 26.585969 ± 2.193132 | -6.100694 ± 3.555417 | [-14.932839, 2.731451] |
| MELD | test | middle | rmsce | 3 | 22.991120 ± 3.346012 | 19.069881 ± 1.696111 | -3.921239 ± 5.036418 | [-16.432394, 8.589917] |
| MELD | test | middle | weighted_f1 | 3 | 63.599342 ± 1.015643 | 63.603288 ± 0.435404 | 0.003946 ± 0.585576 | [-1.450707, 1.458598] |
| MELD | test | overall | accuracy | 3 | 65.836526 ± 0.724263 | 65.759898 ± 0.706825 | -0.076628 ± 1.015868 | [-2.600184, 2.446927] |
| MELD | test | overall | auprc_success | 3 | 84.168604 ± 0.764329 | 83.864861 ± 0.689951 | -0.303743 ± 0.493644 | [-1.530024, 0.922537] |
| MELD | test | overall | auroc_success | 3 | 75.254884 ± 0.375119 | 74.342366 ± 0.554580 | -0.912518 ± 0.903090 | [-3.155917, 1.330882] |
| MELD | test | overall | ece | 3 | 21.563912 ± 2.988031 | 17.938954 ± 1.529862 | -3.624958 ± 4.479939 | [-14.753745, 7.503828] |
| MELD | test | overall | f1.anger | 3 | 50.892783 ± 2.146311 | 52.685761 ± 1.045683 | 1.792977 ± 2.927950 | [-5.480453, 9.066408] |
| MELD | test | overall | f1.disgust | 3 | 28.193822 ± 2.550058 | 24.772947 ± 1.660776 | -3.420874 ± 2.937080 | [-10.716987, 3.875238] |
| MELD | test | overall | f1.fear | 3 | 21.572366 ± 4.659470 | 23.271000 ± 4.865460 | 1.698634 ± 8.098636 | [-18.419493, 21.816760] |
| MELD | test | overall | f1.joy | 3 | 62.754453 ± 1.205411 | 61.426666 ± 1.308228 | -1.327787 ± 1.439317 | [-4.903249, 2.247674] |
| MELD | test | overall | f1.neutral | 3 | 79.070693 ± 0.646875 | 79.097201 ± 0.053652 | 0.026508 ± 0.594169 | [-1.449490, 1.502506] |
| MELD | test | overall | f1.sadness | 3 | 36.039129 ± 2.188783 | 39.136608 ± 1.533195 | 3.097479 ± 0.739495 | [1.260473, 4.934486] |
| MELD | test | overall | f1.surprise | 3 | 56.973672 ± 0.555040 | 57.182139 ± 1.141651 | 0.208467 ± 0.819536 | [-1.827374, 2.244309] |
| MELD | test | overall | macro_f1 | 3 | 47.928131 ± 1.068812 | 48.224617 ± 0.851467 | 0.296486 ± 0.349680 | [-0.572166, 1.165139] |
| MELD | test | overall | mce | 3 | 49.909124 ± 27.699451 | 46.079550 ± 30.102446 | -3.829574 ± 49.646838 | [-127.159157, 119.500009] |
| MELD | test | overall | rmsce | 3 | 23.545461 ± 2.972301 | 19.227772 ± 1.909251 | -4.317689 ± 4.828247 | [-16.311719, 7.676341] |
| MELD | test | overall | weighted_f1 | 3 | 64.597559 ± 0.814811 | 64.855516 ± 0.477477 | 0.257957 ± 0.821146 | [-1.781883, 2.297797] |
| MELD | validation | early | accuracy | 3 | 69.526363 ± 0.409524 | 69.615728 ± 1.014997 | 0.089366 ± 0.674695 | [-1.586669, 1.765400] |
| MELD | validation | early | auprc_success | 3 | 85.025874 ± 2.258894 | 85.171539 ± 1.394588 | 0.145665 ± 0.866759 | [-2.007483, 2.298812] |
| MELD | validation | early | auroc_success | 3 | 73.445135 ± 1.890125 | 73.103317 ± 1.291183 | -0.341818 ± 0.914227 | [-2.612884, 1.929249] |
| MELD | validation | early | ece | 3 | 24.875022 ± 1.864311 | 21.316123 ± 2.480502 | -3.558899 ± 4.344444 | [-14.351095, 7.233298] |
| MELD | validation | early | f1.anger | 3 | 57.119810 ± 3.632290 | 58.580152 ± 5.011751 | 1.460342 ± 1.685271 | [-2.726103, 5.646787] |
| MELD | validation | early | f1.disgust | 3 | 25.264550 ± 3.182860 | 27.619048 ± 23.961420 | 2.354497 ± 23.753947 | [-56.653579, 61.362574] |
| MELD | validation | early | f1.fear | 3 | 39.868421 ± 2.305449 | 46.574770 ± 4.049642 | 6.706349 ± 1.824808 | [2.173276, 11.239423] |
| MELD | validation | early | f1.joy | 3 | 67.740931 ± 0.513539 | 68.154741 ± 1.390683 | 0.413809 ± 1.289612 | [-2.789765, 3.617384] |
| MELD | validation | early | f1.neutral | 3 | 81.042228 ± 0.311240 | 81.166613 ± 1.105023 | 0.124385 ± 0.802228 | [-1.868460, 2.117231] |
| MELD | validation | early | f1.sadness | 3 | 35.188406 ± 0.702861 | 35.232323 ± 1.654594 | 0.043917 ± 1.515629 | [-3.721113, 3.808948] |
| MELD | validation | early | f1.surprise | 3 | 64.448856 ± 1.495007 | 63.417508 ± 1.778705 | -1.031348 ± 3.146036 | [-8.846534, 6.783839] |
| MELD | validation | early | macro_f1 | 3 | 52.953315 ± 0.707155 | 54.392165 ± 3.471741 | 1.438850 ± 3.799519 | [-7.999677, 10.877378] |
| MELD | validation | early | mce | 3 | 34.212228 ± 6.079814 | 30.572118 ± 3.866652 | -3.640111 ± 9.859957 | [-28.133602, 20.853381] |
| MELD | validation | early | rmsce | 3 | 26.032198 ± 2.345393 | 22.250144 ± 2.635688 | -3.782054 ± 4.979692 | [-16.152295, 8.588186] |
| MELD | validation | early | weighted_f1 | 3 | 68.411095 ± 0.375273 | 68.827795 ± 1.212937 | 0.416700 ± 1.041709 | [-2.171048, 3.004449] |
| MELD | validation | late | accuracy | 3 | 62.645219 ± 0.409524 | 62.377122 ± 1.824895 | -0.268097 ± 1.418633 | [-3.792177, 3.255984] |
| MELD | validation | late | auprc_success | 3 | 75.803421 ± 2.115811 | 75.327328 ± 2.123184 | -0.476093 ± 1.987627 | [-5.413631, 4.461445] |
| MELD | validation | late | auroc_success | 3 | 69.192169 ± 1.190517 | 67.878188 ± 2.891837 | -1.313980 ± 1.920730 | [-6.085337, 3.457376] |
| MELD | validation | late | ece | 3 | 19.576532 ± 2.197345 | 16.840496 ± 2.454702 | -2.736036 ± 4.646532 | [-14.278661, 8.806589] |
| MELD | validation | late | f1.anger | 3 | 56.035768 ± 3.157354 | 55.109104 ± 2.547081 | -0.926663 ± 3.748939 | [-10.239543, 8.386217] |
| MELD | validation | late | f1.disgust | 3 | 26.255836 ± 2.545991 | 28.536155 ± 8.518573 | 2.280320 ± 7.225018 | [-15.667619, 20.228258] |
| MELD | validation | late | f1.fear | 3 | 12.532680 ± 0.784824 | 12.397661 ± 1.620632 | -0.135019 ± 1.042440 | [-2.724585, 2.454547] |
| MELD | validation | late | f1.joy | 3 | 57.927263 ± 1.872004 | 56.624026 ± 2.510067 | -1.303237 ± 2.465148 | [-7.427003, 4.820529] |
| MELD | validation | late | f1.neutral | 3 | 75.713343 ± 0.349118 | 75.883915 ± 1.028449 | 0.170572 ± 1.356431 | [-3.198989, 3.540132] |
| MELD | validation | late | f1.sadness | 3 | 43.807187 ± 5.557732 | 48.528924 ± 6.153136 | 4.721738 ± 1.753469 | [0.365880, 9.077595] |
| MELD | validation | late | f1.surprise | 3 | 56.920886 ± 1.351461 | 56.505532 ± 0.838567 | -0.415355 ± 1.123207 | [-3.205554, 2.374845] |
| MELD | validation | late | macro_f1 | 3 | 47.027566 ± 1.148375 | 47.655045 ± 2.599454 | 0.627479 ± 1.593022 | [-3.329808, 4.584766] |
| MELD | validation | late | mce | 3 | 30.789937 ± 1.139198 | 44.533620 ± 32.584785 | 13.743683 ± 32.079375 | [-65.945901, 93.433267] |
| MELD | validation | late | rmsce | 3 | 21.171959 ± 2.521882 | 18.113130 ± 2.019370 | -3.058829 ± 4.530382 | [-14.312922, 8.195265] |
| MELD | validation | late | weighted_f1 | 3 | 60.772663 ± 0.583742 | 61.106947 ± 1.764139 | 0.334284 ± 1.243113 | [-2.753780, 3.422347] |
| MELD | validation | middle | accuracy | 3 | 65.656566 ± 1.517237 | 65.564738 ± 1.928375 | -0.091827 ± 0.573462 | [-1.516385, 1.332730] |
| MELD | validation | middle | auprc_success | 3 | 83.145470 ± 1.026740 | 84.063155 ± 0.884052 | 0.917685 ± 1.289912 | [-2.286632, 4.122003] |
| MELD | validation | middle | auroc_success | 3 | 71.498794 ± 1.469770 | 72.846416 ± 2.269610 | 1.347622 ± 0.940508 | [-0.988730, 3.683974] |
| MELD | validation | middle | ece | 3 | 22.068573 ± 3.421029 | 18.566217 ± 2.552801 | -3.502356 ± 4.751032 | [-15.304574, 8.299862] |
| MELD | validation | middle | f1.anger | 3 | 52.645374 ± 5.051144 | 51.955796 ± 4.730368 | -0.689578 ± 1.111674 | [-3.451129, 2.071972] |
| MELD | validation | middle | f1.disgust | 3 | 18.888889 ± 1.924501 | 25.505051 ± 7.588373 | 6.616162 ± 6.072589 | [-8.468986, 21.701309] |
| MELD | validation | middle | f1.fear | 3 | 44.297082 ± 5.163905 | 37.724727 ± 4.073279 | -6.572355 ± 6.416402 | [-22.511581, 9.366871] |
| MELD | validation | middle | f1.joy | 3 | 54.733352 ± 2.263265 | 54.162262 ± 2.763382 | -0.571090 ± 2.200839 | [-6.038276, 4.896096] |
| MELD | validation | middle | f1.neutral | 3 | 77.929889 ± 0.200408 | 77.309349 ± 1.779728 | -0.620540 ± 1.731702 | [-4.922325, 3.681246] |
| MELD | validation | middle | f1.sadness | 3 | 36.849855 ± 6.639721 | 41.041504 ± 5.557383 | 4.191648 ± 1.433957 | [0.629503, 7.753794] |
| MELD | validation | middle | f1.surprise | 3 | 73.501802 ± 3.368360 | 74.474017 ± 2.061336 | 0.972214 ± 1.410994 | [-2.532888, 4.477317] |
| MELD | validation | middle | macro_f1 | 3 | 51.263749 ± 1.601981 | 51.738958 ± 2.129629 | 0.475209 ± 1.388815 | [-2.974798, 3.925215] |
| MELD | validation | middle | mce | 3 | 37.182025 ± 4.076728 | 33.598870 ± 2.105368 | -3.583156 ± 2.044537 | [-8.662066, 1.495755] |
| MELD | validation | middle | rmsce | 3 | 23.516973 ± 3.058618 | 20.587106 ± 2.554239 | -2.929867 ± 4.908740 | [-15.123853, 9.264119] |
| MELD | validation | middle | weighted_f1 | 3 | 63.699036 ± 1.642937 | 63.663098 ± 1.946535 | -0.035938 ± 0.446556 | [-1.145244, 1.073369] |
| MELD | validation | overall | accuracy | 3 | 65.945296 ± 0.453852 | 65.855125 ± 0.912173 | -0.090171 ± 0.631199 | [-1.658157, 1.477815] |
| MELD | validation | overall | auprc_success | 3 | 81.267887 ± 1.646668 | 81.670530 ± 1.033186 | 0.402644 ± 0.871276 | [-1.761727, 2.567014] |
| MELD | validation | overall | auroc_success | 3 | 71.401921 ± 0.600532 | 71.344620 ± 0.670330 | -0.057301 ± 0.074811 | [-0.243141, 0.128539] |
| MELD | validation | overall | ece | 3 | 22.116732 ± 2.507416 | 18.689310 ± 2.532525 | -3.427422 ± 4.926491 | [-15.665503, 8.810659] |
| MELD | validation | overall | f1.anger | 3 | 55.247040 ± 3.726362 | 55.070172 ± 3.840145 | -0.176868 ± 2.181489 | [-5.595986, 5.242250] |
| MELD | validation | overall | f1.disgust | 3 | 23.684211 ± 2.279014 | 27.851555 ± 8.687193 | 4.167344 ± 8.185241 | [-16.165921, 24.500610] |
| MELD | validation | overall | f1.fear | 3 | 34.669386 ± 2.680544 | 34.004749 ± 0.730478 | -0.664637 ± 2.115342 | [-5.919438, 4.590165] |
| MELD | validation | overall | f1.joy | 3 | 60.891022 ± 0.477260 | 60.515174 ± 0.598086 | -0.375849 ± 0.287879 | [-1.090980, 0.339283] |
| MELD | validation | overall | f1.neutral | 3 | 78.322961 ± 0.080416 | 78.244537 ± 1.178358 | -0.078424 ± 1.106597 | [-2.827364, 2.670516] |
| MELD | validation | overall | f1.sadness | 3 | 39.364989 ± 1.034813 | 42.654799 ± 2.398595 | 3.289810 ± 1.370449 | [-0.114573, 6.694193] |
| MELD | validation | overall | f1.surprise | 3 | 66.062698 ± 1.067579 | 66.006601 ± 1.313770 | -0.056097 ± 0.568094 | [-1.467320, 1.355126] |
| MELD | validation | overall | macro_f1 | 3 | 51.177472 ± 0.095819 | 52.049655 ± 1.526476 | 0.872183 ± 1.441249 | [-2.708079, 4.452445] |
| MELD | validation | overall | mce | 3 | 31.372287 ± 4.444773 | 26.513714 ± 4.920687 | -4.858573 ± 8.116014 | [-25.019871, 15.302725] |
| MELD | validation | overall | rmsce | 3 | 23.157480 ± 2.719315 | 19.443949 ± 2.691590 | -3.713531 ± 5.333489 | [-16.962653, 9.535591] |
| MELD | validation | overall | weighted_f1 | 3 | 64.361201 ± 0.354548 | 64.628710 ± 0.927207 | 0.267509 ± 0.680183 | [-1.422161, 1.957178] |

## IEMOCAP Angry and MELD anger

| Dataset | Seed | Split | Segment | Metric | Baseline | CMERC | Delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | 1937 | validation | overall | f1.angry | 78.260870 | 76.315789 | -1.945080 |
| IEMOCAP | 1937 | validation | early | f1.angry | 60.714286 | 50.000000 | -10.714286 |
| IEMOCAP | 1937 | validation | middle | f1.angry | 68.888889 | 71.428571 | 2.539683 |
| IEMOCAP | 1937 | validation | late | f1.angry | 95.327103 | 92.000000 | -3.327103 |
| IEMOCAP | 1937 | test | overall | f1.angry | 63.841808 | 55.762082 | -8.079726 |
| IEMOCAP | 1937 | test | early | f1.angry | 46.666667 | 17.142857 | -29.523810 |
| IEMOCAP | 1937 | test | middle | f1.angry | 61.904762 | 52.083333 | -9.821429 |
| IEMOCAP | 1937 | test | late | f1.angry | 71.428571 | 68.115942 | -3.312629 |
| IEMOCAP | 2025 | validation | overall | f1.angry | 72.649573 | 71.889401 | -0.760172 |
| IEMOCAP | 2025 | validation | early | f1.angry | 55.319149 | 41.025641 | -14.293508 |
| IEMOCAP | 2025 | validation | middle | f1.angry | 59.259259 | 60.526316 | 1.267057 |
| IEMOCAP | 2025 | validation | late | f1.angry | 90.566038 | 92.156863 | 1.590825 |
| IEMOCAP | 2025 | test | overall | f1.angry | 61.261261 | 58.867925 | -2.393337 |
| IEMOCAP | 2025 | test | early | f1.angry | 39.215686 | 18.181818 | -21.033868 |
| IEMOCAP | 2025 | test | middle | f1.angry | 56.896552 | 56.250000 | -0.646552 |
| IEMOCAP | 2025 | test | late | f1.angry | 71.084337 | 70.588235 | -0.496102 |
| IEMOCAP | 3407 | validation | overall | f1.angry | 75.518672 | 72.398190 | -3.120482 |
| IEMOCAP | 3407 | validation | early | f1.angry | 59.259259 | 39.024390 | -20.234869 |
| IEMOCAP | 3407 | validation | middle | f1.angry | 68.235294 | 63.291139 | -4.944155 |
| IEMOCAP | 3407 | validation | late | f1.angry | 90.196078 | 93.069307 | 2.873228 |
| IEMOCAP | 3407 | test | overall | f1.angry | 63.192182 | 55.303030 | -7.889152 |
| IEMOCAP | 3407 | test | early | f1.angry | 38.297872 | 22.857143 | -15.440729 |
| IEMOCAP | 3407 | test | middle | f1.angry | 59.259259 | 47.826087 | -11.433172 |
| IEMOCAP | 3407 | test | late | f1.angry | 73.684211 | 68.613139 | -5.071072 |
| MELD | 1937 | validation | overall | f1.anger | 52.380952 | 54.430380 | 2.049427 |
| MELD | 1937 | validation | early | f1.anger | 55.813953 | 58.064516 | 2.250563 |
| MELD | 1937 | validation | middle | f1.anger | 48.979592 | 49.523810 | 0.544218 |
| MELD | 1937 | validation | late | f1.anger | 52.727273 | 55.932203 | 3.204931 |
| MELD | 1937 | test | overall | f1.anger | 48.493976 | 53.200569 | 4.706593 |
| MELD | 1937 | test | early | f1.anger | 50.862069 | 58.921162 | 8.059093 |
| MELD | 1937 | test | middle | f1.anger | 46.296296 | 51.327434 | 5.031137 |
| MELD | 1937 | test | late | f1.anger | 48.148148 | 49.152542 | 1.004394 |
| MELD | 2025 | validation | overall | f1.anger | 59.459459 | 59.190031 | -0.269428 |
| MELD | 2025 | validation | early | f1.anger | 61.224490 | 63.829787 | 2.605297 |
| MELD | 2025 | validation | middle | f1.anger | 58.407080 | 57.407407 | -0.999672 |
| MELD | 2025 | validation | late | f1.anger | 59.016393 | 57.142857 | -1.873536 |
| MELD | 2025 | test | overall | f1.anger | 52.631579 | 51.482480 | -1.149099 |
| MELD | 2025 | test | early | f1.anger | 57.707510 | 56.451613 | -1.255897 |
| MELD | 2025 | test | middle | f1.anger | 50.980392 | 47.154472 | -3.825921 |
| MELD | 2025 | test | late | f1.anger | 49.206349 | 50.806452 | 1.600102 |
| MELD | 3407 | validation | overall | f1.anger | 53.900709 | 51.590106 | -2.310603 |
| MELD | 3407 | validation | early | f1.anger | 54.320988 | 53.846154 | -0.474834 |
| MELD | 3407 | validation | middle | f1.anger | 50.549451 | 48.936170 | -1.613280 |
| MELD | 3407 | validation | late | f1.anger | 56.363636 | 52.252252 | -4.111384 |
| MELD | 3407 | test | overall | f1.anger | 51.552795 | 53.374233 | 1.821438 |
| MELD | 3407 | test | early | f1.anger | 55.855856 | 57.534247 | 1.678391 |
| MELD | 3407 | test | middle | f1.anger | 49.765258 | 49.765258 | 0.000000 |
| MELD | 3407 | test | late | f1.anger | 48.803828 | 52.727273 | 3.923445 |

## Prespecified IEMOCAP new-seed sensitivity (2025 and 3407 only)

This descriptive sensitivity excludes the already observed seed 1937. It never replaces the full three-seed main table and has no inferential p-value.

| Split | Metric | Available new seeds n | Baseline mean ± SD | CMERC mean ± SD | Delta mean ± SD |
| --- | --- | --- | --- | --- | --- |
| validation | accuracy | 2 | 61.195446 ± 2.817693 | 61.005693 ± 2.817693 | -0.189753 ± 0.000000 |
| validation | weighted_f1 | 2 | 61.188559 ± 2.600829 | 61.360386 ± 2.715594 | 0.171827 ± 0.114764 |
| validation | macro_f1 | 2 | 60.660019 ± 2.751624 | 60.837655 ± 3.066638 | 0.177636 ± 0.315015 |
| validation | ece | 2 | 22.255099 ± 4.030229 | 17.462564 ± 0.596583 | -4.792535 ± 4.626812 |
| validation | mce | 2 | 49.146345 ± 32.859698 | 35.497566 ± 0.086669 | -13.648780 ± 32.773029 |
| validation | rmsce | 2 | 23.617800 ± 4.244949 | 19.925181 ± 1.766160 | -3.692618 ± 6.011109 |
| validation | auroc_success | 2 | 61.673051 ± 2.443002 | 62.991132 ± 0.442262 | 1.318082 ± 2.885264 |
| validation | auprc_success | 2 | 70.908506 ± 3.572752 | 69.866665 ± 1.127221 | -1.041840 ± 2.445531 |
| validation | cel.total | 2 | 598.139773 ± 173.278437 | 387.484546 ± 5.775188 | -210.655227 ± 179.053625 |
| validation | cel.text | 2 | 100.995843 ± 45.670985 | 92.093541 ± 23.540834 | -8.902302 ± 69.211819 |
| validation | cel.audio | 2 | 194.490152 ± 4.645028 | 128.560864 ± 16.640755 | -65.929287 ± 21.285783 |
| validation | cel.visual | 2 | 5.924808 ± 0.046867 | 4.008631 ± 0.930496 | -1.916176 ± 0.977363 |
| validation | cel.intra_context | 2 | 101.992730 ± 46.059752 | 72.237206 ± 3.217347 | -29.755524 ± 49.277098 |
| validation | cel.inter_context | 2 | 194.736241 ± 76.855805 | 90.584303 ± 38.554243 | -104.151937 ± 38.301562 |
| validation | f1.angry | 2 | 74.084122 ± 2.028760 | 72.143795 ± 0.359768 | -1.940327 ± 1.668992 |
| test | accuracy | 2 | 65.003081 ± 1.568444 | 65.403574 ± 0.566383 | 0.400493 ± 2.134826 |
| test | weighted_f1 | 2 | 64.802235 ± 2.086108 | 65.232161 ± 0.813586 | 0.429926 ± 2.899694 |
| test | macro_f1 | 2 | 63.033370 ± 2.807776 | 63.318692 ± 1.333534 | 0.285322 ± 4.141310 |
| test | ece | 2 | 18.016872 ± 5.579377 | 15.981014 ± 4.621258 | -2.035857 ± 10.200635 |
| test | mce | 2 | 31.040508 ± 4.725009 | 23.970451 ± 6.088627 | -7.070057 ± 10.813636 |
| test | rmsce | 2 | 19.309198 ± 4.867662 | 17.001975 ± 4.405802 | -2.307223 ± 9.273463 |
| test | auroc_success | 2 | 66.375449 ± 1.642637 | 67.117574 ± 0.912534 | 0.742125 ± 2.555171 |
| test | auprc_success | 2 | 79.109268 ± 0.240056 | 79.381339 ± 0.056588 | 0.272071 ± 0.296645 |
| test | cel.total | 2 | 1545.410111 ± 408.455594 | 1029.069662 ± 83.195158 | -516.340449 ± 491.650753 |
| test | cel.text | 2 | 291.996231 ± 100.904061 | 293.698813 ± 93.780428 | 1.702581 ± 194.684489 |
| test | cel.audio | 2 | 473.685543 ± 112.205143 | 286.488028 ± 48.193081 | -187.197514 ± 160.398224 |
| test | cel.visual | 2 | 16.259508 ± 2.693857 | 12.986050 ± 5.394385 | -3.273458 ± 8.088241 |
| test | cel.intra_context | 2 | 343.362241 ± 115.030589 | 196.863767 ± 27.097685 | -146.498474 ± 87.932904 |
| test | cel.inter_context | 2 | 420.106587 ± 77.621945 | 239.033004 ± 37.075050 | -181.073584 ± 40.546895 |
| test | f1.angry | 2 | 62.226722 ± 1.365367 | 57.085477 ± 2.520761 | -5.141244 ± 3.886128 |

The second-round Angry decline is a prespecified diagnostic. A repeat across seeds is a robustness observation; a class-specific mechanism requires a new controlled ablation.
Fixed-split seed variation does not measure split/session uncertainty or establish deployment generalization.
