# Final data readiness decision

## Decision

**CONDITIONALLY READY for a restricted next phase; NOT READY for the full CaliDist behavior-aware raw-signal perturbation experiment. Stop here for researcher review.**

The project is ready for non-destructive feature-space perturbation design using copied/derived inputs. It is not ready to claim semantic text, pitch/timbre, facial dynamics, or local facial interventions under exact XP04 provenance until the missing encoder/extractor and raw-ID chains are rebuilt and a clean re-extraction baseline is verified.

## Direct answers

1. **IEMOCAP raw text:** YES in the frozen pickle; a complete matching local transcript candidate also exists. Its directory is not referenced by XP04, so historical producer provenance is not closed.
2. **MELD raw text:** YES in the frozen pickle; candidate CSV exists. One of 13,708 candidate rows differs in text/speaker from XP04.
3. **Original WAV:** No WAV is part of the proven XP04 runtime chain. A local IEMOCAP candidate has all 7,433 matching sentence-WAV IDs; MELD candidate audio is embedded in MP4. Neither collection is linked to frozen features by a producer manifest.
4. **Original MP4/frames:** Candidate MELD MP4 and IEMOCAP session video exist locally; no frames/face crops/landmarks/AUs are in the proven chain. Two expected MELD MP4 files are missing.
5. **RoBERTa extraction code:** A schema- and content-structure-matching COSMIC/Fairseq pipeline exists. Exact released-pickle producer checkpoint/version linkage is UNVERIFIED.
6. **Audio extraction code:** MISSING from the audited provenance chain.
7. **Visual extraction code:** MISSING from the audited provenance chain.
8. **T1-T4:** matching producer maps T1/T2/T3/T4 to RoBERTa-large transformer layers 24/23/22/21, respectively, using BOS `[:,0,:]`, width 1024. The exact pickle-to-producer linkage remains UNVERIFIED.
9. **One-to-one alignment:** YES inside frozen text/audio/visual arrays. External raw-text alignment is complete except 53 IEMOCAP `xxx` label remaps and one MELD text/speaker version mismatch. External raw A/V→feature identity is UNVERIFIED.
10. **Directly implementable:** text/audio/visual feature dropout; text feature Gaussian noise; other explicitly opaque feature-space perturbations. Text semantic editing is conditional on re-encoding.
11. **Must re-extract:** every semantic text perturbation; every physical audio pitch/timbre perturbation; every video/frame facial dynamics or occlusion perturbation.
12. **Ready to enter behavior perturbation experiments?** Only the restricted feature-space phase. Full behavior-aware semantic/acoustic/visual signal perturbations: **NO-GO until prerequisites below are satisfied.**

## Required preparation

1. Freeze a canonical ID manifest containing dataset, split, dialogue key, utterance index/ID, speaker representation, label, raw path, timestamp, and clean-output hash.
2. Resolve the 53 IEMOCAP `xxx` labels and the MELD dialogue 556/utterance 6 version discrepancy by an explicit protocol; do not silently inherit them into new indices.
3. Pin the text encoder environment: producer code commit, Fairseq version, RoBERTa tokenizer assets, fine-tuned checkpoint hashes, ASCII policy, max-length behavior, and clean re-extraction comparison.
4. For audio, establish attributable raw segments and a documented extractor. Existing 1,582-D/300-D vectors cannot distinguish pitch from timbre in an interpretable manner.
5. For visual work, retain frame timestamps, face boxes/landmarks or region masks, target-speaker tracks, frame-level features, aggregation, and missing reasons. Existing 342-D vectors cannot support local facial evidence analysis.
6. Write every new feature set to a new immutable directory; never overwrite the four XP04 pickles. Keep clean and perturbed manifests and hashes.

## T1-T4 intervention recommendation

Layer-wise interventions are mechanically possible because arrays are separate. Apply them to copied T1-T4 tensors before CMERC's independent BN and record which layer is changed. Interpret with caution: the model then equal-averages all four layers, so an intervention tests the stored layer representation under this fusion rule, not an isolated transformer computation.

No model training or perturbation run was performed in this audit.
