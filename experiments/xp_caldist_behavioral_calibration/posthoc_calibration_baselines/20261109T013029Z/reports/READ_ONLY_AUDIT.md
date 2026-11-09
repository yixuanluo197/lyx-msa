# Read-only input audit

Status: **passed**.

The ten XP04 checkpoints and twenty saved prediction files form an intact provenance chain. All prediction inputs match the prior artifact manifest; Validation and Test dialogue sets are disjoint.

| Dataset | Validation | Test | Seeds |
|---|---:|---:|---:|
| IEMOCAP | 527 / 12 dialogues | 1623 / 31 dialogues | 5 |
| MELD | 1109 / 114 dialogues | 2610 / 280 dialogues | 5 |

Experiment 6 is accepted only as a Test-only historical metric identity check for IEMOCAP seed1475 and MELD seed67137. It supplies no fitting data or calibration parameters.

Existing complete probability vectors are sufficient; no GPU inference is required.
