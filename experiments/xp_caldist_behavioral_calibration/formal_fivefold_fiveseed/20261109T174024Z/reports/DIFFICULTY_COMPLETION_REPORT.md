# Difficulty Completion Report

All results in this report stop at Training, Development selection, and Calibration-internal diagnostics. No Evaluation DataLoader was built; no Evaluation label, feature, probability, or forward pass was accessed. Calibration OOF values are internal diagnostics and are not independent Evaluation results.

All 10 fold-local Difficulty artifacts exactly cover their fold Training dialogues and exclude Development, Calibration, and Evaluation. The mask-aware formula was independently recomputed for every dialogue; all values are finite. IEMOCAP dialogue counts by fold are 84, 80, 80, 81, 83; MELD counts are 852, 842, 855, 876, 871.
