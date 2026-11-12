# XP04 baseline and feature audit

**Gate: stopped before Pilot.** The 10 XP04 prediction files under a path named `validation` contain Test dialogue IDs, while the later independently inferred Validation files contain the expected Validation IDs. The path component denotes the checkpoint selection protocol, not the prediction split: the frozen evaluator calls `build_test_loader`. These historical files cannot serve as original Validation predictions under the requested gate.

The source is the frozen XP04 batch `20261016T050823Z`, not Protocol B. All 10 best checkpoints and all four raw feature files exist and were SHA-256 hashed. No checkpoint was loaded in this audit.

| Dataset | Seed | Original `validation` rows/dialogues | Overlap with Validation/Test IDs | Later true Validation rows/dialogues | Later overlap with Validation/Test IDs |
|---|---:|---:|---:|---:|---:|
| IEMOCAP | 1475 | 1623/31 | 0/31 | 527/12 | 12/0 |
| IEMOCAP | 67137 | 1623/31 | 0/31 | 527/12 | 12/0 |
| IEMOCAP | 2024 | 1623/31 | 0/31 | 527/12 | 12/0 |
| IEMOCAP | 3407 | 1623/31 | 0/31 | 527/12 | 12/0 |
| IEMOCAP | 42 | 1623/31 | 0/31 | 527/12 | 12/0 |
| MELD | 1475 | 2610/280 | 0/280 | 1109/114 | 114/0 |
| MELD | 67137 | 2610/280 | 0/280 | 1109/114 | 114/0 |
| MELD | 2024 | 2610/280 | 0/280 | 1109/114 | 114/0 |
| MELD | 3407 | 2610/280 | 0/280 | 1109/114 | 114/0 |
| MELD | 42 | 2610/280 | 0/280 | 1109/114 | 114/0 |

The overlaps are dialogue counts. For every seed, historical IDs exactly equal the registered Test IDs; later IDs exactly equal the registered Validation IDs. Train, Validation and Test dialogue sets are disjoint.

## Feature inventory

The loader reads T1–T4 as four separate 1024-dimensional arrays; their source layer identity is not inferred. The loader pads each batch with `pad_sequence`; `umask` is one on real utterances and zero on padding. `qmask` is constructed separately and must remain unchanged. The model applies the configured BN to text after input; the loader does not standardize audio or visual. Original feature zero is therefore not assumed to equal the Training mean. No noise statistics were calculated or applied after the mismatch gate.

| Dataset | Feature | Dimension | Train utterances | Validation utterances | Train all-zero | Validation all-zero |
|---|---|---:|---:|---:|---:|---:|
| IEMOCAP | audio | 1582 | 5283 | 527 | 0 | 0 |
| IEMOCAP | visual | 342 | 5283 | 527 | 20 | 1 |
| IEMOCAP | T1 | 1024 | 5283 | 527 | 0 | 0 |
| IEMOCAP | T2 | 1024 | 5283 | 527 | 0 | 0 |
| IEMOCAP | T3 | 1024 | 5283 | 527 | 0 | 0 |
| IEMOCAP | T4 | 1024 | 5283 | 527 | 0 | 0 |
| MELD | audio | 300 | 9989 | 1109 | 1 | 0 |
| MELD | visual | 342 | 9989 | 1109 | 3407 | 95 |
| MELD | T1 | 1024 | 9989 | 1109 | 0 | 0 |
| MELD | T2 | 1024 | 9989 | 1109 | 0 | 0 |
| MELD | T3 | 1024 | 9989 | 1109 | 0 | 0 |
| MELD | T4 | 1024 | 9989 | 1109 | 0 | 0 |

Audio all-zero features and visual all-zero features are counted as missing evidence for future perturbation masking; they are not deleted. Feature arrays in both splits are finite. The model produces log-probabilities; later saved full probabilities are finite, normalized and agree with argmax predictions.

## Stop reason and required resolution

The historical file path cannot be used as evidence that its rows are Validation. The code confirms it evaluates Test after Validation-based checkpoint selection. Establish whether the later Validation inference artifact is accepted as the authoritative prediction source under the XP04-only requirement, and independently verify its probabilities against fresh checkpoint inference. Only then can the requested Pilot start. No Test label entered a calibration fit.

See `verification.json` for per-seed machine-readable checks and `provenance/source_hashes.json` for input hashes.
