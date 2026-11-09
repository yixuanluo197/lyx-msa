# Hierarchical Predicted-Class Temperature Scaling preregistration

This method pilot uses only the ten saved true-Validation probability files. Historical Test behavior motivates the research question, but no Test probability, label, or metric enters fitting, lambda selection, or OOF evaluation.

Every dataset and model seed is processed independently. The outer loop leaves one complete dialogue out. Within the remaining dialogues, six fixed shrinkage candidates `{0.001, 0.01, 0.1, 1, 10, 100}` are compared using five deterministic inner folds grouped by dialogue. Inner OOF NLL selects lambda; the outer dialogue is unavailable to every inner operation. A numerical tie within `1e-12` selects the larger lambda.

E0 is the original distribution. E1 fits one global temperature on the outer training dialogues. E2 independently fits temperatures for each original predicted class and is an overfitting control. E3 first fits the same global temperature, then minimizes

`mean sample NLL + lambda * mean_k((log(T_k)-log(T_global))^2)`.

The penalty average includes all dataset classes. Classes absent from a training fold are fixed to the global temperature. The sample NLL remains averaged over all training utterances, so low-support predicted classes receive proportionally less likelihood evidence and therefore shrink more strongly. All temperatures lie in `[0.05, 20]`.

At inference, temperature selection uses only the original predicted class. The selected positive scalar scales every logit of that utterance, so its argmax must remain unchanged. True labels are used only for Validation fitting and evaluation.

Primary decisions use outer-OOF NLL and multiclass Brier together with ECE15. Dialogue-level paired bootstrap uses 2,000 resamples of fixed OOF predictions for E1−E0, E3−E1, and E3−E2. These intervals do not include calibrator refitting uncertainty. Five seeds share the same Validation utterances and are never pooled as independent datasets. IEMOCAP's 12 dialogues and small predicted-class folds require explicit uncertainty reporting.

This phase does not fit a true-class-conditioned calibrator, access Test, retrain CMERC, or launch a subsequent method.
