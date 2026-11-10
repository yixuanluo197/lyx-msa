# Teacher Completion Report

All results in this report stop at Training, Development selection, and Calibration-internal diagnostics. No Evaluation DataLoader was built; no Evaluation label, feature, probability, or forward pass was accessed. Calibration OOF values are internal diagnostics and are not independent Evaluation results.

All 10 fold-local Teachers completed or were verified historical Fold 1 artifacts. The 8 new Teachers used only fold Training IDs and fixed final epochs; IEMOCAP used 80 epochs and MELD 40. All 10 checkpoints passed CPU strict reload.
