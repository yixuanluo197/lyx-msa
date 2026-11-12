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
