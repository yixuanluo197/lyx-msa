# GPU Scheduling Report

All results in this report stop at Training, Development selection, and Calibration-internal diagnostics. No Evaluation DataLoader was built; no Evaluation label, feature, probability, or forward pass was accessed. Calibration OOF values are internal diagnostics and are not independent Evaluation results.

Four TITAN Xp GPUs were bound by UUID with one independent process per GPU, no DDP, four CPU threads per task, and zero DataLoader workers. New scheduled work consumed 13.816 GPU-hours; verified historical Pilot work accounts for 0.703 GPU-hours. Peak reserved memory was 11.555 GiB. One dependency-placeholder defect caused four Calibration attempts to fail before loading data; the root cause was fixed under a failing regression test and only affected tasks were rerun in fresh attempt directories.
