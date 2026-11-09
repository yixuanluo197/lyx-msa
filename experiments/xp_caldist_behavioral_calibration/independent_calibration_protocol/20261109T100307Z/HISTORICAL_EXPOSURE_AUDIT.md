# Historical Exposure Audit

## Exposure flags

`sample_lineage.csv` distinguishes four historical uses plus a narrow `not_previously_evaluated` flag:

- XP04 Training samples updated original CMERC weights. They were also used to estimate train-only feature statistics in later perturbation studies.
- XP04 Validation samples selected checkpoints and were repeatedly used for behavioral, evidence-sensitivity, Global TS, transfer-diagnostic, and hierarchical-calibration development.
- XP04 Test samples were repeatedly evaluated and their outcomes influenced the research sequence. They are marked as calibration/method-development exposed even when they were not used to numerically fit a temperature.
- `not_previously_evaluated=true` means no known prediction-level validation/test analysis; it does not mean model-unseen, dataset-unseen, or researcher-unseen.

## What remains valid

All previous XP04, perturbation, evidence-sensitivity, Global TS, transfer-diagnostic, and hierarchical-classwise results remain valid as exploratory findings under their documented splits. They can motivate hypotheses and provide engineering baselines.

## What is no longer confirmatory

Existing XP04 checkpoints cannot be used for candidate Calibration/Evaluation roles drawn from the original Training pool because those weights already saw the samples. Existing Validation cannot serve as independent Calibration after checkpoint selection. Existing Test cannot be described as a new confirmatory evaluation because it has been inspected repeatedly and influenced method development.

Retraining each candidate fold establishes model-unseen and calibration-unseen Evaluation data. It does not erase researcher exposure to the datasets or the prior use of historical outcomes in choosing the next question. A genuinely researcher-unseen confirmation requires an external cohort or a prospectively sequestered dataset.
