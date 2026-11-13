# E3 design recommendation

This is a single-seed Validation diagnostic, not an E3 training result.

## Findings

1. IEMOCAP is overconfident in aggregate: accuracy−mean C=-0.068. This is not a claim that every bin or class is overconfident.
2. MELD is underconfident in aggregate: accuracy−mean C=0.201. Its correct predictions have mean C=0.501; 0.469 are below 0.5. Wrong predictions have mean C=0.402.
3. Among MELD predicted classes with n≥20, positive gaps: surprise, anger, joy, sadness, neutral, fear; negative gaps: none. Class gaps are descriptive and can reflect prediction-mix effects; inspect sample counts before treating them as a correction rule.
4. MELD E1 raises confidence in 1.000 of all samples, including 1.000 of wrong predictions. It improves aggregate ECE/Brier/NLL in the Pilot, but the increase is not targeted only at correct predictions.
5. A future E3 should allow both upward and downward adjustment around C because the datasets have opposite aggregate bias and class/bin biases may differ. Start with a low-capacity, regularized correctness model using C plus pre-registered behavior features, evaluated dialogue-level OOF; do not alter predicted labels.
6. Class-conditional information is plausible but not yet established as useful. Test predicted-class features as a separate pre-registered ablation with shrinkage and minimum support; never use true class at inference. Compare against C-only E1 and behavior-only variants on the same dialogue folds, checking ECE15, binary Brier, and binary NLL together.

No E3 was trained. This diagnostic alone does not authorize the five-seed or raw-data stage; review the Pilot decision first.
