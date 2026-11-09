# IEMOCAP Label Policy Audit

## Decision status

IEMOCAP training is **not authorized to start** until the researcher chooses one of the two policies below. Protocol B itself is confirmed, but `approved_split_manifests.json` is intentionally not frozen yet because the requested gate includes the label policy.

## Reproduced discrepancy

The frozen XP04 feature pickle contains 7,433 utterances in the six target classes. Raw `EmoEvaluation` primary labels contain 7,380 utterances in those classes. Exact utterance-ID alignment identifies 53 additional frozen samples labeled `happy` whose raw primary label is `xxx`.

- Frozen happy support: 648.
- Raw primary-label happy support: 595.
- Difference: 53 utterances, or 0.713% of the frozen corpus and 8.179% of frozen happy support.
- Session distribution: Ses01 8, Ses02 8, Ses03 36, Ses04 0, Ses05 1.
- The complete Dialogue ID, zero-based utterance index, utterance ID, annotation path and raw label are in [results/iemocap_xxx_anomalies.csv](results/iemocap_xxx_anomalies.csv).

The original IEMOCAP README defines `xxx` as failure to obtain a majority categorical ground truth. CMERC does not implement a label conversion; it reads the frozen pickle. The same label object predates CMERC and is present in the upstream DialogueRNN/COSMIC prepared artifacts. All 53 samples have Happiness as the unique plurality among categorical annotations, but seven other raw `xxx` samples with the same plurality pattern were not included. No executable conversion rule or authoritative manual inclusion list was found. The mapping therefore cannot be reproduced as a complete deterministic upstream rule.

## Policy alternatives

### Legacy Label Policy

Retain the frozen XP04 labels: 7,433 supervised samples and 648 happy samples. This gives the closest comparison with XP04 and its DialogueRNN-derived artifact, but treats 53 raw `xxx` items as happy without a recoverable authoritative rule.

### Unknown-Label Mask Policy

Retain all 7,433 utterances in the input dialogue and graph, while excluding the 53 items from every label-dependent operation. Effective supervision becomes 7,380 samples and happy support becomes 595.

Implementation must use an independent `supervision_mask`; `umask` and `qmask` remain unchanged. The mask must cover classification NLL, rank loss, MSCL, CSCL, fold-local class weights, Development checkpoint metrics, Calibration NLL/CV, and Evaluation metrics. The 53 probabilities may be stored with `metric_eligible=false`.

Fold-level counts are in [results/label_policy_impact.csv](results/label_policy_impact.csv). The distribution is uneven: Fold 3 would mask 36 Evaluation utterances, while Fold 4 would mask none. This is a material protocol choice rather than a cosmetic preprocessing option.

## Recommendation

Use **Unknown-Label Mask Policy** for the new independent confirmatory protocol because it follows the published primary ground truth and avoids silently supervising unknown labels as happy. Keep Legacy Policy as a separately named comparability sensitivity analysis if resources later permit. This recommendation is not an approval; the researcher must choose the main policy before manifest freeze or training.
