# Visual feature extraction audit

## Established facts

- Both datasets use a source label `denseface`, width 342 (`train_cm.py:999-1002`).
- One vector is stored per utterance; no frame dimension survives.
- IEMOCAP has 79 all-zero visual rows; MELD has 3,760. Split-level details are in the alignment report and CSVs.
- XP04 has no special missing-visual branch: a zero vector is passed as data. Dialogue tail padding is distinguishable through `umask`.
- CMERC applies a single learned linear projection to the entire vector (`model_cm.py:647-651,808-813`).

## Missing provenance

No attributable visual extraction/pickle-save script exists in the audited source chain. Detector, face alignment, speaker-face selection, tracking, frame sampling, temporal pooling, normalization, DenseFace checkpoint, missing policy, and coordinate semantics are all UNVERIFIED. Later local DenseFace re-extraction utilities are not the producer of XP04 and are excluded as causal evidence.

## Interpretation

The 342 dimensions are an opaque utterance-level latent vector. They may encode aggregate facial evidence, but they cannot demonstrate temporal dynamics, and no coordinates can be assigned to eyes, brows, mouth, or another face region. Consequently neither dynamic smoothing nor local facial evidence analysis is valid on the current vectors. Only whole-vector or coordinate-level feature perturbation can be reported, without anatomical interpretation.
