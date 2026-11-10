# XP04 data provenance audit

Audit time: `2026-11-10T05:56:04Z`. Scope: the completed XP04 validation batch and the exact inputs reached from its frozen configuration. This is a read-only audit; no data, feature, source, checkpoint, or historical experiment was changed.

## Executive result

**Decision: CONDITIONALLY READY, but NOT READY for the full raw-signal CaliDist perturbation protocol.** The frozen utterance-level tensors and all 20 formal best checkpoints are readable and structurally aligned. Feature-space noise/dropout is immediately possible. Text semantic perturbation is possible only after rebuilding and pinning the fine-tuned RoBERTa encoder path. True pitch/timbre and face-dynamics/region perturbations are blocked because XP04 contains no attributable A/V extraction code or raw-to-feature manifest.

Evidence terms used throughout:

- `VERIFIED`: directly established from the frozen XP04 source/file and rechecked.
- `CANDIDATE`: locally present raw material that aligns strongly but is not referenced by XP04.
- `MISSING`: the required artifact is absent from the audited dependency chain.
- `UNVERIFIED`: plausible or structurally matching, but the available artifacts cannot prove the claim.

## Actual dependency boundary

- Public record: `CMERC/experiment_records/04_xp_mainline_validation/`.
- Executed batch: `CMERC/experiments/xp_mainline_validation/20261016T050823Z/`.
- Frozen source: the executed batch's `code/`, archived in the public record.
- Feature targets: `/data2/yb/paper/05_reproduction/external_repos/CMERC/data/*.pkl` through `CMERC/data` symlinks.
- Five seeds: `42, 1475, 2024, 3407, 67137`; baseline and CMERC variants for IEMOCAP and MELD, 20 `best.pt` files total.

## Deliverables

- [RAW_TEXT_AUDIT.md](RAW_TEXT_AUDIT.md)
- [RAW_AUDIO_AUDIT.md](RAW_AUDIO_AUDIT.md)
- [RAW_VIDEO_AUDIT.md](RAW_VIDEO_AUDIT.md)
- [TEXT_FEATURE_EXTRACTION_AUDIT.md](TEXT_FEATURE_EXTRACTION_AUDIT.md)
- [AUDIO_FEATURE_EXTRACTION_AUDIT.md](AUDIO_FEATURE_EXTRACTION_AUDIT.md)
- [VISUAL_FEATURE_EXTRACTION_AUDIT.md](VISUAL_FEATURE_EXTRACTION_AUDIT.md)
- [T1_T4_LAYER_MAPPING.md](T1_T4_LAYER_MAPPING.md)
- [MULTIMODAL_ID_ALIGNMENT_AUDIT.md](MULTIMODAL_ID_ALIGNMENT_AUDIT.md)
- [PERTURBATION_FEASIBILITY_MATRIX.md](PERTURBATION_FEASIBILITY_MATRIX.md)
- [FINAL_DATA_READINESS_DECISION.md](FINAL_DATA_READINESS_DECISION.md)
- Machine-readable: `file_inventory.csv`, `utterance_alignment_audit.csv`, `modality_feature_statistics.csv`, `t1_t4_mapping.csv`, `perturbation_feasibility.csv`, `verification.json`, and `provenance/source_hashes.json`.

The final go/no-go decision is in `FINAL_DATA_READINESS_DECISION.md`; research execution should remain stopped until that file is reviewed.
