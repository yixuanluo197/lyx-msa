# Calibration transfer diagnostic preregistration

This exploratory study is motivated by the already observed IEMOCAP/MELD Test-transfer difference. It nevertheless reads and evaluates only the ten true Validation prediction files. Test probabilities and labels are excluded from every script input.

For every dataset and model seed, ordinary temperature stability uses outer leave-one-dialogue-out (LODO). Each held-out dialogue receives a temperature fitted on all remaining Validation dialogues. The concatenated held-out probabilities form one complete OOF prediction set. Single-dialogue ECE is retained as a noisy descriptive statistic; primary OOF comparisons use the concatenated set and prioritize NLL and multiclass Brier.

The sample-size diagnostic draws dialogue subsets at fixed proportions 50%, 75%, and 100%. The first two use 100 deterministic repetitions; 100% is fit once. Counts use round-half-up. No subset is selected using downstream performance.

The exploratory regularization objective is `NLL + lambda * log(T)^2`, with lambda fixed to `{0, 0.001, 0.01, 0.1, 1}`. In each outer LODO fold, lambda is selected using five deterministic inner folds grouped by dialogue. Inner candidates minimize utterance-weighted OOF NLL; a numerical tie within `1e-12` chooses the larger lambda. The outer dialogue is absent from fitting, inner selection, and normalization. Ordinary and regularized temperatures are then compared only through outer OOF predictions.

Dialogue bootstrap uses 2,000 paired resamples of fixed OOF predictions. These intervals condition on the fitted fold temperatures. Temperature-refitting uncertainty is diagnosed separately through LODO temperature dispersion and the repeated dialogue-subset experiment; it is not claimed to be included in the fixed-prediction bootstrap.

Five seeds share the same Validation utterances and are reported as model-seed replications, never pooled as five independent datasets. Classes with support below 20 are flagged as uncertain. This phase does not fit classwise temperatures or perform any new Test evaluation.
