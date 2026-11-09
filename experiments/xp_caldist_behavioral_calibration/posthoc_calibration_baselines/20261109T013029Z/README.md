# CMERC five-seed Global Temperature Scaling

This directory is a self-contained post-hoc calibration baseline for ten XP04 validation-selected CMERC checkpoints. It uses true Validation predictions to fit one scalar temperature per dataset/seed and evaluates the frozen parameter once on Test.

## Result

|Dataset|Metric|Vanilla mean±SD|Global TS mean±SD|Δ TS−Vanilla|Improved Seeds|
|---|---|---|---|---|---|
|IEMOCAP|ece15|0.067169±0.024055|0.078398±0.031217|+0.011228|1/5|
|IEMOCAP|brier|0.443966±0.015820|0.446304±0.013811|+0.002337|2/5|
|IEMOCAP|nll|0.818251±0.034006|0.831229±0.020622|+0.012978|1/5|
|MELD|ece15|0.186665±0.029827|0.037421±0.000930|-0.149244|5/5|
|MELD|brier|0.525287±0.007938|0.478671±0.007576|-0.046616|5/5|
|MELD|nll|1.130949±0.016868|1.012850±0.021256|-0.118099|5/5|

MELD improves comprehensively in all five seeds. IEMOCAP does not improve consistently and degrades on average. All Test argmax predictions and classification metrics remain unchanged.

## Reproduction from saved probabilities

Run the scripts in order with `/data2/yb/conda_envs/cmerc-xp-repro/bin/python`: `audit_inputs.py`, `fit_temperatures.py`, `evaluate_test.py`, `verify_results.py`; figures use `python scripts/plot_results.py`. The fitting script reads Validation only. Complete original and calibrated distributions are stored in each run directory and in `results/per_utterance_calibrated_probabilities.csv`.

See `reports/PREREGISTRATION.md`, `reports/READ_ONLY_AUDIT.md`, `reports/VALIDATION_FITTING_REPORT.md`, `reports/TEST_CALIBRATION_REPORT.md`, `reports/FIVE_SEED_COMPARISON.md`, and `reports/FINAL_RESEARCH_DECISION.md`.
