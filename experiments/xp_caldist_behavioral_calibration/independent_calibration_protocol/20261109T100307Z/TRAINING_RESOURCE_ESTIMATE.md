# Training Resource Estimate

The estimates use observed XP04 CMERC wall time and checkpoint size, without launching GPU work. Five XP04 IEMOCAP runs took 0.834–0.909 GPU-hours each (median 0.873); five MELD runs took 0.391–0.421 (median 0.407). Best checkpoints are approximately 59.8 MB and 142.1 MB, respectively.

| Protocol | Dataset | Models | Planned GPU-hours | With 25% contingency | Estimated full artifacts |
|---|---|---:|---:|---:|---:|
| A | IEMOCAP | 5 | 3.93 | 4.91 | 0.90 GB |
| A | MELD | 5 | 1.83 | 2.29 | 2.13 GB |
| B | IEMOCAP | 25 | 19.64 | 24.55 | 4.48 GB |
| B | MELD | 25 | 9.16 | 11.45 | 10.66 GB |

The recommended Protocol B therefore needs 50 CMERC models, about 28.8 GPU-hours before contingency and 36.0 GPU-hours with contingency. The estimate applies a conservative 0.9 factor to XP04 duration despite smaller Training pools; it excludes queue time and temporary prediction artifacts.

Current free space is about 75 GB on `/data1` and 135 GB on `/data2`. The estimated 15.1 GB of complete Protocol B run artifacts fits, but `/data2` is 98% occupied. A future execution must reserve space and recheck it before launch. No GPU scheduling decision was made in this audit.

A resource-saving staged plan can run one fixed preregistered seed per fold first for pipeline validation, but those results cannot be used to redesign the method. The formal five-seed analysis still requires all 50 models.
