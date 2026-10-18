# Next research candidates: hypotheses, not established innovations

Execution coverage: **complete**; missing evaluations: 0.

Validation is the primary analysis. Test is the prespecified replication analysis and is reported separately.
IEMOCAP seed 1937 test results were already observed in round 2 and were reused without retraining or reselection.
MELD seed 1937 is the technical pilot; its test may precede the remaining fixed matrix. Configurations do not change after that exposure.
Deltas are CMERC minus Baseline. Classification/calibration are percentages, with deltas in percentage points.
CEL is a sum of positive true-class log-probability differences under `prefix_mean_v1`, a prefix smoothing view, not complete modality removal.
Three seeds on a fixed split give uncertain estimates. Sample SD uses ddof=1; df=2 t intervals rely on normality and are descriptive.
The exact two-sided sign-flip test has minimum p=0.25 for three seeds. No robust statistical significance or proven innovation is claimed.

## Measured diagnostic evidence

| Dataset | Seed | Split | Angry/anger F1 delta | CEL total delta | Context CEL delta | Context share of net reduction % | ECE delta | AUROC-Success delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | 1937 | validation | -1.945080 | -171.235653 | -150.481514 | 87.879779 | -6.151213 | -2.064171 |
| IEMOCAP | 1937 | test | -8.079726 | -522.683861 | -336.376846 | 64.355698 | -3.969560 | -0.143421 |
| IEMOCAP | 2025 | validation | -0.760172 | -337.265259 | -195.834926 | 58.065549 | -8.064185 | 3.358271 |
| IEMOCAP | 2025 | test | -2.393337 | -863.990030 | -418.420994 | 48.428915 | -9.248795 | -1.064653 |
| IEMOCAP | 3407 | validation | -3.120482 | -84.045195 | -71.979997 | 85.644393 | -1.520884 | -0.722108 |
| IEMOCAP | 3407 | test | -7.889152 | -168.690867 | -236.723121 | 140.329542 | 5.177081 | 2.548904 |
| MELD | 1937 | validation | 2.049427 | -43.011473 | -43.003606 | 99.981711 | -1.416967 | -0.143681 |
| MELD | 1937 | test | 4.706593 | -117.083935 | -108.218466 | 92.428108 | -0.698159 | -1.861498 |
| MELD | 2025 | validation | -0.269428 | 33.208747 | 23.837132 | NA | 0.175915 | -0.014782 |
| MELD | 2025 | test | -1.149099 | 112.367609 | 76.500851 | NA | -1.394412 | -0.063663 |
| MELD | 3407 | validation | -2.310603 | 40.136703 | -4.403802 | NA | -9.041213 | -0.013440 |
| MELD | 3407 | test | 1.821438 | 85.288436 | -19.641310 | NA | -8.782305 | -0.812393 |

Context share is defined only for a meaningful net CEL reduction; near-zero threshold is 1e-12 × max(1, |baseline CEL|).
The ratio can exceed 100% when other views worsen. It is not a causal attribution. CEL magnitudes are not directly comparable across dataset sample counts.

## Candidates requiring a separately frozen follow-up

1. **Class trade-offs:** test whether CC/HSCL/CL change Angry/anger errors systematically. Compare paired component ablations with per-class recall/precision; retain the current negative results.
2. **Context-view sensitivity:** investigate why GNN-edge ablations contribute to CEL changes. Encoders and hypergraphs retain history, so this is not complete context removal. Use explicit edge and representation controls before mechanism claims.
3. **Calibration versus correctness ranking:** study ECE improvements alongside AUROC/AUPRC-Success and selective-risk curves. A lower ECE does not prove better error detection.
4. **Auxiliary snapshot semantics:** independently compare legacy_epoch_snapshot and same_step_same_mode with matched confidence-pass weights/modes and explicit gradient rules; do not silently substitute the auxiliary policy.
5. **Replacement semantics:** prefix_mean_v1 is invertible prefix smoothing. Strict missing-modality diagnostics need an explicitly named fixed replacement and cannot directly reuse cross-policy CEL comparisons.

These are candidate questions. This round does not establish novelty, mechanism, statistical significance, or cross-split robustness and does not launch any ablation.
