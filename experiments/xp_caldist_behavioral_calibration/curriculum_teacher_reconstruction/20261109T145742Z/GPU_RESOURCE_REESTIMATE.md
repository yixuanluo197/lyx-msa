# GPU Resource Re-estimate

No GPU was queried or used in this audit-only round. Historical local TITAN Xp process-wall measurements are used as planning evidence.

| Work | Count | Per model planning basis | Process GPU-hours |
|---|---:|---:|---:|
| IEMOCAP teachers | 5 | 0.154 h (XP04 baseline × 0.9) | 0.77 |
| MELD teachers | 5 | 0.080 h (XP04 baseline × 0.9) | 0.40 |
| IEMOCAP CMERC | 25 | 0.786 h | 19.64 |
| MELD CMERC | 25 | 0.366 h | 9.16 |
| Total | 60 | — | 29.97 |

With a 25% contingency, plan 37.46 process GPU-hours, plus batch-one difficulty inference. These are estimates, not measured amended-protocol throughput. Four perfectly occupied cards imply a theoretical lower bound of about 9.4 hours; dependencies, IEMOCAP's roughly 11.2 GiB reserved peak, scoring, and queues make actual wall time longer.

The two-teacher plus two-CMERC Pilot is approximately 1.39 process GPU-hours before contingency. Existing 50-model artifact planning is about 15.1 GB. Ten teacher best/latest checkpoints, optimizer states, and logs add an estimated 1–3 GB; difficulty CSV/JSON artifacts are small. Recheck GPUs and `/data1`/`/data2` immediately before any future launch.
