# Final Pilot decision

## IEMOCAP

Original accuracy=0.66034; mean top-label confidence=0.72788; accuracy−confidence=-0.06753.
E1: ECE15=0.08977, Brier=0.22121, NLL=0.63706.
Lowest-Brier E2 (dropout/joint): ECE15=0.08905, Brier=0.22048, NLL=0.65225.
E2 conditions improving all three metrics over E1: 0/8.
E2 scale factor ≥0.999 in 0.5% of utterance-condition records. The original CaliDist scaling factor is at most 1, so it cannot correct underconfidence by increasing C.

## MELD

Original accuracy=0.66907; mean top-label confidence=0.46840; accuracy−confidence=+0.20067.
E1: ECE15=0.04323, Brier=0.19276, NLL=0.56858.
Lowest-Brier E2 (gaussian/text): ECE15=0.20136, Brier=0.23342, NLL=0.65945.
E2 conditions improving all three metrics over E1: 0/8.
E2 scale factor ≥0.999 in 100.0% of utterance-condition records. The original CaliDist scaling factor is at most 1, so it cannot correct underconfidence by increasing C.

## Decision

暂无收益：暂不进入五 Seed 或原始数据干扰。

Qualifying conditions: none.
This single-seed Pilot has no uncertainty interval and does not establish cross-seed robustness. The full experiment remains subject to researcher review.

## Questions answered

1. Both Gaussian noise and feature dropout changed some predictions and confidences; dropout was stronger in both datasets, while most predictions remained stable.
2. Under dropout, IEMOCAP audio was the most responsive single modality; MELD text was the most responsive single modality. Joint perturbation had the largest flip rate in both.
3. μ and δ formed a numerical Reliability Score, but E2 did not show consistent calibration value beyond C-only E1.
4. No E2 condition improved ECE15, binary Brier, and binary NLL together relative to E1.
5. IEMOCAP had small E2 ECE/Brier gains in some conditions, with worse NLL; MELD E2 was essentially E0 and worse than E1 on all three metrics.
6. The datasets differ: IEMOCAP is mildly overconfident on average and responds most to audio; MELD is strongly underconfident on average and responds most to text. CaliDist’s original scaling can only reduce confidence, explaining its MELD limitation.
