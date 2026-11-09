# Training Code Safety Audit

The XP04 frozen `train_cm.py`, `dataloader.py`, `model_cm.py` and `evaluation_metrics.py` remain the authoritative implementation source. No historical file was modified.

Required adaptations were identified and bounded to a new runner: manifest-driven Training/Development loaders, post-forward supervision mask, Training-only class weights, explicit Calibration inference, and complete removal of XP04's automatic Test/Evaluation loader and final forward.

All five rank branches and all MSCL/CSCL relation labels use the true class. Filtering only classification loss would leak the 53 unknown labels. The mask core therefore selects eligible rows before every true-label index and before contrastive pools are constructed, while retaining the full unmasked model forward.

The runner itself was not completed or executed after the mandatory curriculum-difficulty fidelity gate failed. Model structure, optimizer and historical checkpoints were untouched.
