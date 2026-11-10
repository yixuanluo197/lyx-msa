# Approved Protocol

All results in this report stop at Training, Development selection, and Calibration-internal diagnostics. No Evaluation DataLoader was built; no Evaluation label, feature, probability, or forward pass was accessed. Calibration OOF values are internal diagnostics and are not independent Evaluation results.

The frozen Protocol B manifest SHA-256 is `6d43928865915fe22a913c98b1165a05ba6e3e8f21d4d4c896cf51eb9c9b62be`. IEMOCAP uses Session-level five-fold rotation and the Unknown-Label Mask Policy. MELD uses Season-Episode blocks. Teacher, mask-aware Difficulty, CapacitySafeBatchSampler, 80/40 epoch CMERC schedules, Development-only selection, and Calibration-only fitting remained frozen.
