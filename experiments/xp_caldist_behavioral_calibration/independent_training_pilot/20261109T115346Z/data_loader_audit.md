# Protocol B Fold 1 DataLoader Dry-run

The structural dry-run passed for Training, Development, Calibration and Evaluation in both datasets. It used the XP04 frozen DataLoader implementation and precomputed features. No model forward was run.

| Dataset | Role | Dialogues | Utterances | Length range |
|---|---:|---:|---:|---:|
| IEMOCAP | Training | 84 | 4,216 | 8–110 |
| IEMOCAP | Development | 9 | 488 | 28–91 |
| IEMOCAP | Calibration | 30 | 1,356 | 24–74 |
| IEMOCAP | Evaluation | 28 | 1,373 | 26–79 |
| MELD | Training | 852 | 8,176 | 1–33 |
| MELD | Development | 141 | 1,363 | 1–22 |
| MELD | Calibration | 153 | 1,406 | 1–24 |
| MELD | Evaluation | 286 | 2,763 | 1–23 |

Verified dimensions were four text arrays of 1,024, visual 342, IEMOCAP audio 1,582, MELD audio 300, IEMOCAP qmask 2 and MELD qmask 9. Feature lengths, speaker masks, valid-utterance `umask`, padding and within-dialogue order were consistent. Training/Development/Calibration batches used batch size 16 and `num_workers=0`.

Evaluation was audited through IDs and non-label feature structure only. Its labels were not inspected and no Evaluation DataLoader forward path was created. Full batch shapes and checks are in [results/data_loader_dry_run.json](results/data_loader_dry_run.json).
