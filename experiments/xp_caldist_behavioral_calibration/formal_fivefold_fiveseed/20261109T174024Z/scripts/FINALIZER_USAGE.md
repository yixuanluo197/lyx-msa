# Formal artifact verification

Run `finalize_formal.py` only after all scheduler tasks complete. It fails before report writing when any task is incomplete. It never opens raw datasets or role feature artifacts and never performs model forward inference. CPU architecture construction and strict state loading preserve each checkpoint's recorded code lineage, including the two historical Pilot students.

Use the XP04 environment, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, and the existing `LD_LIBRARY_PATH`. Example from repository root:

```bash
/data2/yb/conda_envs/cmerc-xp-repro/bin/python \
  experiments/xp_caldist_behavioral_calibration/formal_fivefold_fiveseed/20261109T174024Z/scripts/finalize_formal.py \
  --formal-root experiments/xp_caldist_behavioral_calibration/formal_fivefold_fiveseed/20261109T174024Z \
  --teacher-pilot experiments/xp_caldist_behavioral_calibration/curriculum_teacher_pilot/20261109T154754Z \
  --cmerc-pilot experiments/xp_caldist_behavioral_calibration/independent_cmerc_pilot/20261109T164030Z
```

Outputs are created once under `results/final_verification/`. A second invocation refuses to overwrite that package. The package contains all eight requested CSVs, `verification.json`, `source_hashes.json`, `report_data.json`, and `evaluation_readiness_manifest.json`. The parent report-generation workflow can publish these CSVs to the formal root's `results/` after verification.

Verification includes fixed-final Teacher epochs/access IDs, checkpoint/log hashes, full Training-only Difficulty formula recomputation, per-epoch curriculum schedule and exact Difficulty-selected IDs, capacity limits, Development-only earliest rounded Weighted-F1 checkpoint selection, Calibration unknown-mask identity, Grouped CV disjointness/coverage, full-probability parameter replay, argmax invariance, and independent saved metric recomputation. Unknown rows retain fitted probabilities; their intentionally absent OOF probabilities are exported as empty lists and excluded from OOF metrics.

`--skip-strict-reload` is only a diagnostic mode. Its verification status is `diagnostic_only_not_ready` and cannot certify readiness. Readiness remains pending researcher approval even after every check passes.

Evaluation checks use approved Manifest IDs, group IDs, role hashes, and sample counts only. The final manifest contains no Evaluation labels or probabilities. Calibration OOF is internal diagnostic evidence; it is not an independent Evaluation result. GPU accounting distinguishes new scheduler-bound GPU time from historical Pilot training time.
