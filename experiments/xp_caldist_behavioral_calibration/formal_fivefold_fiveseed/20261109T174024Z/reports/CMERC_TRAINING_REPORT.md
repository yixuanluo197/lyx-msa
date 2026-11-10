# CMERC Training Report

All results in this report stop at Training, Development selection, and Calibration-internal diagnostics. No Evaluation DataLoader was built; no Evaluation label, feature, probability, or forward pass was accessed. Calibration OOF values are internal diagnostics and are not independent Evaluation results.

All 50 CMERC models completed or were verified historical Fold 1 seed 1475 Pilots. Every model started from its recorded random initialization, used its own fold Teacher/Difficulty, completed the frozen Curriculum schedule, used Development only for checkpoint selection, and passed CPU strict reload.

Mean Development best Weighted-F1 scores were 68.576% for IEMOCAP and 71.609% for MELD. These are Development results, not final performance estimates.
