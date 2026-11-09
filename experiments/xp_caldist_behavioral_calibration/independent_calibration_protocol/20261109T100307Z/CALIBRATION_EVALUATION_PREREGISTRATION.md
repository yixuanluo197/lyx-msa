# Calibration/Evaluation Preregistration

## Status

This is a design preregistration. Candidate assignments are frozen in `candidate_split_manifests.json` with status `candidate_not_approved`. No formal split, training run, probability prediction, or calibration fit may begin until researcher review.

## Data roles

1. **Training:** the only role allowed to update CMERC weights.
2. **Development:** checkpoint selection and training control only.
3. **Calibration:** fit Global TS and class-conditional calibration parameters; grouped CV within this role selects regularization.
4. **Evaluation:** one final application of frozen checkpoint and frozen calibrator settings.

No dialogue crosses roles. IEMOCAP Calibration/Evaluation use whole Sessions. MELD uses whole Season–Episode groups for every role.

## Methods

- E0: Vanilla CMERC.
- E1: Global Temperature Scaling.
- E2: strongly regularized Hierarchical Predicted-Class Temperature Scaling.
- E3: unregularized predicted-class TS, reported only as an overfitting diagnostic.

Each sample uses its original predicted class to select a class temperature. True labels may enter calibration NLL but never select a temperature at inference.

Within Calibration, grouped CV selects the fixed, preregistered regularization candidates for E2 using NLL. Evaluation labels and probabilities are inaccessible until the checkpoint, global temperature, E2 regularization choice, and analysis code are frozen. Primary comparison uses NLL; multiclass Brier and ECE15 are prespecified supporting metrics. Accuracy and argmax invariance are verification checks.

## Statistical unit

IEMOCAP uncertainty is clustered by Session/fold and dialogue; MELD is clustered by Season–Episode. Five seeds are repeated model initializations on the same data, not five independent datasets. Fold-level results and seed directions are reported separately before any aggregate.

## Confirmation labels

- Model-unseen: required and achieved only by a newly trained fold model.
- Calibration-unseen: Evaluation never enters calibration fitting or grouped CV.
- Method-development-unseen: not fully achievable with historically studied IEMOCAP/MELD; the candidate assignments were designed after earlier results.
- Researcher-unseen: not achievable within these two already studied datasets.

The future study may be called a model-independent grouped re-evaluation. It must not be described as a wholly researcher-unseen confirmatory test.
