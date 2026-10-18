# MELD Causal Baseline versus CMERC

Execution coverage: **complete**; missing evaluations: 0.

Validation is the primary analysis. Test is the prespecified replication analysis and is reported separately.
IEMOCAP seed 1937 test results were already observed in round 2 and were reused without retraining or reselection.
MELD seed 1937 is the technical pilot; its test may precede the remaining fixed matrix. Configurations do not change after that exposure.
Deltas are CMERC minus Baseline. Classification/calibration are percentages, with deltas in percentage points.
CEL is a sum of positive true-class log-probability differences under `prefix_mean_v1`, a prefix smoothing view, not complete modality removal.
Three seeds on a fixed split give uncertain estimates. Sample SD uses ddof=1; df=2 t intervals rely on normality and are descriptive.
The exact two-sided sign-flip test has minimum p=0.25 for three seeds. No robust statistical significance or proven innovation is claimed.

## All completed paired validation and test results

| Dataset | Seed | Split | Segment | Metric | Baseline | CMERC | Delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
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
| MELD | 1937 | validation | overall | cel.total | 304.329374 | 261.317901 | -43.011473 |
| MELD | 1937 | validation | overall | cel.text | 120.211619 | 120.066198 | -0.145421 |
| MELD | 1937 | validation | overall | cel.audio | 14.899596 | 14.034321 | -0.865275 |
| MELD | 1937 | validation | overall | cel.visual | 6.303065 | 7.305895 | 1.002830 |
| MELD | 1937 | validation | overall | cel.intra_context | 70.402860 | 42.363762 | -28.039098 |
| MELD | 1937 | validation | overall | cel.inter_context | 92.512233 | 77.547725 | -14.964508 |
| MELD | 1937 | validation | overall | auroc_success | 70.762934 | 70.619253 | -0.143681 |
| MELD | 1937 | validation | overall | auprc_success | 79.373921 | 80.664448 | 1.290527 |
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
| MELD | 1937 | test | overall | cel.total | 794.309417 | 677.225481 | -117.083935 |
| MELD | 1937 | test | overall | cel.text | 333.134501 | 325.808955 | -7.325545 |
| MELD | 1937 | test | overall | cel.audio | 31.041133 | 27.577673 | -3.463460 |
| MELD | 1937 | test | overall | cel.visual | 12.207420 | 14.130956 | 1.923536 |
| MELD | 1937 | test | overall | cel.intra_context | 182.158618 | 104.799349 | -77.359270 |
| MELD | 1937 | test | overall | cel.inter_context | 235.767744 | 204.908548 | -30.859196 |
| MELD | 1937 | test | overall | auroc_success | 75.684105 | 73.822607 | -1.861498 |
| MELD | 1937 | test | overall | auprc_success | 83.422704 | 83.074037 | -0.348668 |
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
| MELD | 2025 | validation | overall | cel.total | 249.922033 | 283.130780 | 33.208747 |
| MELD | 2025 | validation | overall | cel.text | 119.182183 | 129.338759 | 10.156576 |
| MELD | 2025 | validation | overall | cel.audio | 14.685511 | 13.489572 | -1.195939 |
| MELD | 2025 | validation | overall | cel.visual | 6.933740 | 7.344718 | 0.410978 |
| MELD | 2025 | validation | overall | cel.intra_context | 37.925148 | 59.047292 | 21.122143 |
| MELD | 2025 | validation | overall | cel.inter_context | 71.195451 | 73.910439 | 2.714989 |
| MELD | 2025 | validation | overall | auroc_success | 71.488154 | 71.473373 | -0.014782 |
| MELD | 2025 | validation | overall | auprc_success | 82.360420 | 82.728840 | 0.368420 |
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
| MELD | 2025 | test | overall | cel.total | 642.287825 | 754.655434 | 112.367609 |
| MELD | 2025 | test | overall | cel.text | 323.968347 | 362.729889 | 38.761542 |
| MELD | 2025 | test | overall | cel.audio | 28.630214 | 25.524402 | -3.105812 |
| MELD | 2025 | test | overall | cel.visual | 13.752592 | 13.963620 | 0.211028 |
| MELD | 2025 | test | overall | cel.intra_context | 101.822243 | 159.460571 | 57.638328 |
| MELD | 2025 | test | overall | cel.inter_context | 174.114429 | 192.976952 | 18.862523 |
| MELD | 2025 | test | overall | auroc_success | 74.989863 | 74.926200 | -0.063663 |
| MELD | 2025 | test | overall | auprc_success | 84.132992 | 84.343820 | 0.210828 |
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
| MELD | 3407 | validation | overall | cel.total | 231.821120 | 271.957823 | 40.136703 |
| MELD | 3407 | validation | overall | cel.text | 100.932541 | 141.212768 | 40.280227 |
| MELD | 3407 | validation | overall | cel.audio | 12.596930 | 15.194956 | 2.598026 |
| MELD | 3407 | validation | overall | cel.visual | 5.737240 | 7.399493 | 1.662253 |
| MELD | 3407 | validation | overall | cel.intra_context | 39.404716 | 38.513796 | -0.890920 |
| MELD | 3407 | validation | overall | cel.inter_context | 73.149692 | 69.636810 | -3.512882 |
| MELD | 3407 | validation | overall | auroc_success | 71.954675 | 71.941236 | -0.013440 |
| MELD | 3407 | validation | overall | auprc_success | 82.069320 | 81.618303 | -0.451017 |
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
| MELD | 3407 | test | overall | cel.total | 593.715956 | 679.004393 | 85.288436 |
| MELD | 3407 | test | overall | cel.text | 273.877732 | 371.651615 | 97.773883 |
| MELD | 3407 | test | overall | cel.audio | 25.983658 | 29.838572 | 3.854914 |
| MELD | 3407 | test | overall | cel.visual | 10.699123 | 14.000072 | 3.300950 |
| MELD | 3407 | test | overall | cel.intra_context | 101.132657 | 93.072995 | -8.059662 |
| MELD | 3407 | test | overall | cel.inter_context | 182.022786 | 170.441138 | -11.581648 |
| MELD | 3407 | test | overall | auroc_success | 75.090684 | 74.278291 | -0.812393 |
| MELD | 3407 | test | overall | auprc_success | 84.950117 | 84.176727 | -0.773390 |

## Supplementary CEL per utterance

Divide summed CEL by the identical paired split's utterance count for a scale diagnostic. The main CEL definition remains the sum.

| Seed | Split | Utterances | Metric | Baseline / utterance | CMERC / utterance | Delta / utterance |
| --- | --- | --- | --- | --- | --- | --- |
| 1937 | validation | 1109 | cel.total | 0.274418 | 0.235634 | -0.038784 |
| 1937 | validation | 1109 | cel.text | 0.108396 | 0.108265 | -0.000131 |
| 1937 | validation | 1109 | cel.audio | 0.013435 | 0.012655 | -0.000780 |
| 1937 | validation | 1109 | cel.visual | 0.005684 | 0.006588 | 0.000904 |
| 1937 | validation | 1109 | cel.intra_context | 0.063483 | 0.038200 | -0.025283 |
| 1937 | validation | 1109 | cel.inter_context | 0.083420 | 0.069926 | -0.013494 |
| 1937 | test | 2610 | cel.total | 0.304333 | 0.259473 | -0.044860 |
| 1937 | test | 2610 | cel.text | 0.127638 | 0.124831 | -0.002807 |
| 1937 | test | 2610 | cel.audio | 0.011893 | 0.010566 | -0.001327 |
| 1937 | test | 2610 | cel.visual | 0.004677 | 0.005414 | 0.000737 |
| 1937 | test | 2610 | cel.intra_context | 0.069793 | 0.040153 | -0.029640 |
| 1937 | test | 2610 | cel.inter_context | 0.090332 | 0.078509 | -0.011823 |
| 2025 | validation | 1109 | cel.total | 0.225358 | 0.255303 | 0.029945 |
| 2025 | validation | 1109 | cel.text | 0.107468 | 0.116626 | 0.009158 |
| 2025 | validation | 1109 | cel.audio | 0.013242 | 0.012164 | -0.001078 |
| 2025 | validation | 1109 | cel.visual | 0.006252 | 0.006623 | 0.000371 |
| 2025 | validation | 1109 | cel.intra_context | 0.034198 | 0.053244 | 0.019046 |
| 2025 | validation | 1109 | cel.inter_context | 0.064198 | 0.066646 | 0.002448 |
| 2025 | test | 2610 | cel.total | 0.246087 | 0.289140 | 0.043053 |
| 2025 | test | 2610 | cel.text | 0.124126 | 0.138977 | 0.014851 |
| 2025 | test | 2610 | cel.audio | 0.010969 | 0.009779 | -0.001190 |
| 2025 | test | 2610 | cel.visual | 0.005269 | 0.005350 | 0.000081 |
| 2025 | test | 2610 | cel.intra_context | 0.039012 | 0.061096 | 0.022084 |
| 2025 | test | 2610 | cel.inter_context | 0.066711 | 0.073938 | 0.007227 |
| 3407 | validation | 1109 | cel.total | 0.209036 | 0.245228 | 0.036192 |
| 3407 | validation | 1109 | cel.text | 0.091012 | 0.127333 | 0.036321 |
| 3407 | validation | 1109 | cel.audio | 0.011359 | 0.013701 | 0.002343 |
| 3407 | validation | 1109 | cel.visual | 0.005173 | 0.006672 | 0.001499 |
| 3407 | validation | 1109 | cel.intra_context | 0.035532 | 0.034728 | -0.000803 |
| 3407 | validation | 1109 | cel.inter_context | 0.065960 | 0.062792 | -0.003168 |
| 3407 | test | 2610 | cel.total | 0.227477 | 0.260155 | 0.032678 |
| 3407 | test | 2610 | cel.text | 0.104934 | 0.142395 | 0.037461 |
| 3407 | test | 2610 | cel.audio | 0.009955 | 0.011432 | 0.001477 |
| 3407 | test | 2610 | cel.visual | 0.004099 | 0.005364 | 0.001265 |
| 3407 | test | 2610 | cel.intra_context | 0.038748 | 0.035660 | -0.003088 |
| 3407 | test | 2610 | cel.inter_context | 0.069741 | 0.065303 | -0.004437 |

## Paired seed summaries

| Dataset | Split | Segment | Metric | n | Baseline mean ± SD | CMERC mean ± SD | Paired delta mean ± SD | 95% t interval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MELD | test | overall | accuracy | 3 | 65.836526 ± 0.724263 | 65.759898 ± 0.706825 | -0.076628 ± 1.015868 | [-2.600184, 2.446927] |
| MELD | test | overall | auprc_success | 3 | 84.168604 ± 0.764329 | 83.864861 ± 0.689951 | -0.303743 ± 0.493644 | [-1.530024, 0.922537] |
| MELD | test | overall | auroc_success | 3 | 75.254884 ± 0.375119 | 74.342366 ± 0.554580 | -0.912518 ± 0.903090 | [-3.155917, 1.330882] |
| MELD | test | overall | cel.audio | 3 | 28.551668 ± 2.529652 | 27.646882 ± 2.157918 | -0.904786 ± 4.125898 | [-11.154085, 9.344513] |
| MELD | test | overall | cel.inter_context | 3 | 197.301653 ± 33.546470 | 189.442213 ± 17.503468 | -7.859440 ± 25.068974 | [-70.134224, 54.415343] |
| MELD | test | overall | cel.intra_context | 3 | 128.371173 ± 46.582570 | 119.110972 ± 35.432252 | -9.260201 ± 67.506806 | [-176.956403, 158.436001] |
| MELD | test | overall | cel.text | 3 | 310.326860 ± 31.896846 | 353.396820 ± 24.304670 | 43.069960 ± 52.682011 | [-87.799411, 173.939331] |
| MELD | test | overall | cel.total | 3 | 676.771066 ± 104.648242 | 703.628436 ± 44.199627 | 26.857370 ± 125.389971 | [-284.628586, 338.343326] |
| MELD | test | overall | cel.visual | 3 | 12.219712 ± 1.526772 | 14.031549 ± 0.087997 | 1.811838 ± 1.547986 | [-2.033573, 5.657249] |
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
| MELD | validation | overall | accuracy | 3 | 65.945296 ± 0.453852 | 65.855125 ± 0.912173 | -0.090171 ± 0.631199 | [-1.658157, 1.477815] |
| MELD | validation | overall | auprc_success | 3 | 81.267887 ± 1.646668 | 81.670530 ± 1.033186 | 0.402644 ± 0.871276 | [-1.761727, 2.567014] |
| MELD | validation | overall | auroc_success | 3 | 71.401921 ± 0.600532 | 71.344620 ± 0.670330 | -0.057301 ± 0.074811 | [-0.243141, 0.128539] |
| MELD | validation | overall | cel.audio | 3 | 14.060679 ± 1.272155 | 14.239617 ± 0.871030 | 0.178937 ± 2.101506 | [-5.041493, 5.399367] |
| MELD | validation | overall | cel.inter_context | 3 | 78.952459 ± 11.783691 | 73.698325 ± 3.959721 | -5.254134 ± 8.967448 | [-27.530509, 17.022242] |
| MELD | validation | overall | cel.intra_context | 3 | 49.244241 ± 18.338829 | 46.641617 ± 10.914721 | -2.602625 ± 24.625279 | [-63.775209, 58.569959] |
| MELD | validation | overall | cel.text | 3 | 113.442114 ± 10.845829 | 130.205908 ± 10.599921 | 16.763794 ± 21.007136 | [-35.420824, 68.948412] |
| MELD | validation | overall | cel.total | 3 | 262.024176 ± 37.738682 | 272.135501 ± 10.907525 | 10.111326 ± 46.135918 | [-104.496648, 124.719300] |
| MELD | validation | overall | cel.visual | 3 | 6.324682 ± 0.598543 | 7.350035 ± 0.047025 | 1.025353 ± 0.625941 | [-0.529571, 2.580278] |
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

## Training budgets and resource use

Both methods use 40 epochs. Curriculum sampling changes actual optimizer update counts; equal epoch budgets do not imply equal steps.

| Seed | Model | Epochs | Best epoch | Selection W-F1 (rounded) | Epoch seconds sum | Peak allocated MB | Derived steps | Actual Adam steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1937 | baseline | 40 | 8 | 63.980000 | 200.960000 | 798.710000 | 2600 | 2600 |
| 1937 | cmerc | 40 | 8 | 64.070000 | 1316.020000 | 895.630000 | 2548 | 2548 |
| 2025 | baseline | 40 | 8 | 64.680000 | 237.290000 | 801.340000 | 2600 | 2600 |
| 2025 | cmerc | 40 | 8 | 65.700000 | 1940.040000 | 896.060000 | 2548 | 2548 |
| 3407 | baseline | 40 | 5 | 64.420000 | 242.580000 | 800.050000 | 2600 | 2600 |
| 3407 | cmerc | 40 | 9 | 64.110000 | 2094.140000 | 895.490000 | 2548 | 2548 |

Shared-server timing is descriptive and can reflect concurrency. No hyperparameters are changed to improve results.
