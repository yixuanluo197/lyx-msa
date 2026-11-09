# Calibration Pilot Report

Status: **not started**. Calibration depends on trained, Development-selected Pilot checkpoints. No temperature was fitted, no Calibration probability was produced, and no Evaluation forward or label access occurred.

After the training gate clears, E0–E3 will be fitted only from Calibration probabilities. Hierarchical lambda selection will use grouped CV by IEMOCAP Dialogue or MELD Episode. Every temperature must remain positive and the calibrated argmax must equal the original argmax.
