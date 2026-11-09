# CMERC Independent Calibration/Evaluation Split Design

This directory is a read-only protocol-design deliverable. It audits the frozen IEMOCAP and MELD inputs used by XP04, reconstructs grouping metadata, records historical exposure, and creates **candidate** manifests for independent Training/Development/Calibration/Evaluation experiments.

No CMERC model was trained or executed. No calibration parameter was fitted, no saved Test prediction or Test performance metric was read, and no historical split or checkpoint was changed. `candidate_split_manifests.json` is marked `candidate_not_approved`; it is not a formal split until the researcher reviews `FINAL_PROTOCOL_RECOMMENDATION.md`.

The audit indexes 21,141 utterances: 7,433 IEMOCAP utterances in 151 dialogues and 13,708 MELD utterances in 1,432 dialogues. MELD Season–Episode metadata is recovered from the pinned official CSV snapshot; IEMOCAP Session IDs are recovered directly from dialogue and utterance IDs.

Recommended future protocol: five-fold group-level rotating cross-fitting. IEMOCAP reserves one whole Session for Evaluation and the next Session for Calibration. MELD uses ten balanced Season–Episode blocks, with two blocks for Evaluation, one for Calibration, one for Development, and six for Training in each fold. Every fold requires a newly trained CMERC model because old XP04 checkpoints saw samples that the new protocol may reserve.

Run the audit again with:

```bash
python scripts/build_protocol_audit.py
```

The script reads only frozen data and provenance files and regenerates all CSV/JSON planning artifacts.
