# Third-round paired statistics

Status: complete.

Validation is the primary analysis; test is a separately reported replication family.
IEMOCAP seed 1937 test results were already observed in round 2; this is not a wholly blind study.
Three paired training seeds on a fixed split give uncertain estimates. The exact two-sided sign-flip test
has only 8 assignments and minimum p = 0.25; no robust significance claim is supported.
95% paired t intervals use df = 2 and a normality assumption; all other metrics are descriptive.
`prefix_mean_v1` is prefix smoothing, not complete modality removal. CEL uses summed log-probability differences.

| Dataset | Split | Metric | n | Baseline mean ± SD | CMERC mean ± SD | Paired delta ± SD | 95% t interval | Exact p | Holm p |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| IEMOCAP | test | accuracy | 3 | 64.838776 ± 1.144987 | 65.352228 ± 0.410248 | 0.513452 ± 1.522177 | [-3.267844, 4.294749] | NA | NA |
| IEMOCAP | test | auroc_success | 3 | 66.652986 ± 1.257064 | 67.099929 ± 0.645982 | 0.446943 ± 1.877724 | [-4.217582, 5.111468] | NA | NA |
| IEMOCAP | test | cel.total | 3 | 1526.717601 ± 290.630721 | 1008.262681 ± 68.989191 | -518.454919 ± 347.668872 | [-1382.112274, 345.202436] | 0.250000 | 1.000000 |
| IEMOCAP | test | ece | 3 | 18.184590 ± 3.955896 | 15.504165 ± 3.370485 | -2.680425 ± 7.298827 | [-20.811717, 15.450868] | 0.750000 | 1.000000 |
| IEMOCAP | test | macro_f1 | 3 | 62.739075 ± 2.049789 | 63.234162 ± 0.954250 | 0.495087 ± 2.950802 | [-6.835111, 7.825285] | NA | NA |
| IEMOCAP | test | weighted_f1 | 3 | 64.520884 ± 1.553512 | 65.174537 ± 0.583886 | 0.653653 ± 2.086690 | [-4.529971, 5.837278] | 0.750000 | 1.000000 |
| IEMOCAP | validation | accuracy | 3 | 61.796331 ± 2.247863 | 61.733080 ± 2.357324 | -0.063251 ± 0.219108 | [-0.607546, 0.481044] | NA | NA |
| IEMOCAP | validation | auroc_success | 3 | 62.493186 ± 2.236514 | 62.683850 ± 0.617304 | 0.190664 ± 2.824108 | [-6.824809, 7.206137] | NA | NA |
| IEMOCAP | validation | cel.total | 3 | 562.080434 ± 137.526493 | 364.565065 ± 39.907195 | -197.515369 ± 128.639294 | [-517.073090, 122.042351] | 0.250000 | 1.000000 |
| IEMOCAP | validation | ece | 3 | 21.959176 ± 2.895528 | 16.713749 ± 1.363865 | -5.245427 ± 3.364377 | [-13.603002, 3.112148] | 0.250000 | 1.000000 |
| IEMOCAP | validation | macro_f1 | 3 | 60.929785 ± 2.001009 | 61.460663 ± 2.422097 | 0.530878 ± 0.651120 | [-1.086594, 2.148350] | NA | NA |
| IEMOCAP | validation | weighted_f1 | 3 | 61.760864 ± 2.089200 | 62.038097 ± 2.250578 | 0.277233 ± 0.199791 | [-0.219076, 0.773543] | 0.250000 | 1.000000 |
| MELD | test | accuracy | 3 | 65.836526 ± 0.724263 | 65.759898 ± 0.706825 | -0.076628 ± 1.015868 | [-2.600184, 2.446927] | NA | NA |
| MELD | test | auroc_success | 3 | 75.254884 ± 0.375119 | 74.342366 ± 0.554580 | -0.912518 ± 0.903090 | [-3.155917, 1.330882] | NA | NA |
| MELD | test | cel.total | 3 | 676.771066 ± 104.648242 | 703.628436 ± 44.199627 | 26.857370 ± 125.389971 | [-284.628586, 338.343326] | 1.000000 | 1.000000 |
| MELD | test | ece | 3 | 21.563912 ± 2.988031 | 17.938954 ± 1.529862 | -3.624958 ± 4.479939 | [-14.753745, 7.503828] | 0.250000 | 1.000000 |
| MELD | test | macro_f1 | 3 | 47.928131 ± 1.068812 | 48.224617 ± 0.851467 | 0.296486 ± 0.349680 | [-0.572166, 1.165139] | NA | NA |
| MELD | test | weighted_f1 | 3 | 64.597559 ± 0.814811 | 64.855516 ± 0.477477 | 0.257957 ± 0.821146 | [-1.781883, 2.297797] | 0.750000 | 1.000000 |
| MELD | validation | accuracy | 3 | 65.945296 ± 0.453852 | 65.855125 ± 0.912173 | -0.090171 ± 0.631199 | [-1.658157, 1.477815] | NA | NA |
| MELD | validation | auroc_success | 3 | 71.401921 ± 0.600532 | 71.344620 ± 0.670330 | -0.057301 ± 0.074811 | [-0.243141, 0.128539] | NA | NA |
| MELD | validation | cel.total | 3 | 262.024176 ± 37.738682 | 272.135501 ± 10.907525 | 10.111326 ± 46.135918 | [-104.496648, 124.719300] | 1.000000 | 1.000000 |
| MELD | validation | ece | 3 | 22.116732 ± 2.507416 | 18.689310 ± 2.532525 | -3.427422 ± 4.926491 | [-15.665503, 8.810659] | 0.500000 | 1.000000 |
| MELD | validation | macro_f1 | 3 | 51.177472 ± 0.095819 | 52.049655 ± 1.526476 | 0.872183 ± 1.441249 | [-2.708079, 4.452445] | NA | NA |
| MELD | validation | weighted_f1 | 3 | 64.361201 ± 0.354548 | 64.628710 ± 0.927207 | 0.267509 ± 0.680183 | [-1.422161, 1.957178] | 0.750000 | 1.000000 |

## All-seed diagnostic findings

Context CEL share is the combined intra/inter GNN-edge-ablation delta divided by net CEL delta, only when net CEL falls.
These ablations remove selected GNN temporal edges; encoders and hypergraphs may retain speaker history.
Share can exceed 100% when modality views worsen; it does not prove a causal mechanism.

- IEMOCAP seed 1937 validation: angry F1 delta -1.945080; ECE delta -6.151213; AUROC-Success delta -2.064171; context share of net CEL reduction 87.879779%.
- IEMOCAP seed 1937 test: angry F1 delta -8.079726; ECE delta -3.969560; AUROC-Success delta -0.143421; context share of net CEL reduction 64.355698%.
- IEMOCAP seed 2025 validation: angry F1 delta -0.760172; ECE delta -8.064185; AUROC-Success delta 3.358271; context share of net CEL reduction 58.065549%.
- IEMOCAP seed 2025 test: angry F1 delta -2.393337; ECE delta -9.248795; AUROC-Success delta -1.064653; context share of net CEL reduction 48.428915%.
- IEMOCAP seed 3407 validation: angry F1 delta -3.120482; ECE delta -1.520884; AUROC-Success delta -0.722108; context share of net CEL reduction 85.644393%.
- IEMOCAP seed 3407 test: angry F1 delta -7.889152; ECE delta 5.177081; AUROC-Success delta 2.548904; context share of net CEL reduction 140.329542%.
- MELD seed 1937 validation: angry F1 delta 2.049427; ECE delta -1.416967; AUROC-Success delta -0.143681; context share of net CEL reduction 99.981711%.
- MELD seed 1937 test: angry F1 delta 4.706593; ECE delta -0.698159; AUROC-Success delta -1.861498; context share of net CEL reduction 92.428108%.
- MELD seed 2025 validation: angry F1 delta -0.269428; ECE delta 0.175915; AUROC-Success delta -0.014782; context share of net CEL reduction NA%.
- MELD seed 2025 test: angry F1 delta -1.149099; ECE delta -1.394412; AUROC-Success delta -0.063663; context share of net CEL reduction NA%.
- MELD seed 3407 validation: angry F1 delta -2.310603; ECE delta -9.041213; AUROC-Success delta -0.013440; context share of net CEL reduction NA%.
- MELD seed 3407 test: angry F1 delta 1.821438; ECE delta -8.782305; AUROC-Success delta -0.812393; context share of net CEL reduction NA%.

Full class and position metrics, all epoch loss/convergence/resource trajectories, and every recorded raw/weighted gradient norm
are retained in JSON/CSV. Gradient samples cover the recorded batches (first batch each epoch), not the entire epoch.
Per-component norm magnitudes do not add to a total gradient norm; directions may reinforce or cancel.
