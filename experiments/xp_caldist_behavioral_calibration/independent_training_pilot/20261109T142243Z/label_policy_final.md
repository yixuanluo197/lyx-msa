# Final IEMOCAP Label Policy

The approved policy is **Unknown-Label Mask Policy**.

All 53 raw-primary `xxx` utterances remain in their original dialogues with unchanged features, ordering, speakers, `qmask`, `umask`, padding and graph context. They are excluded from every ground-truth-dependent loss, Development checkpoint metric, Calibration NLL/CV objective and future Evaluation metric through a separate boolean `supervision_mask`.

The approved mask contains 53 unique utterance IDs in 16 dialogues. Frozen-pickle alignment passed. Fold 1 contains 23 masked Training utterances, 14 Development, 8 Calibration and 8 sealed Evaluation utterances. The frozen pickle and its labels were not modified.

The full immutable list is embedded in `approved_split_manifests.json`; alignment evidence is in `results/mask_alignment_verification.json`.
