# XP04 Training Code Audit for Protocol B

The authoritative basis is the XP04 frozen source snapshot. The current working-tree `train_cm.py` and `dataloader.py` differ from that snapshot, so the Pilot runner must be derived from the frozen files and kept inside this new experiment directory.

XP04 settings to preserve are FP32, batch size 16, Adam learning rate `1e-4`, weight decay `3e-5`, capacity-safe batching, continuous RNG, `num_workers=0`, GRU CMERC, BN, and the dataset-specific epoch/loss/curriculum settings in the effective configs. Each Fold × Seed must instantiate a new model with `parent_checkpoint=null`; Development Weighted-F1 alone selects the best checkpoint.

Required minimal adaptations:

1. Manifest-driven four-role loaders. The training process receives only Training and Development. Calibration is handled after checkpoint freeze. Evaluation is never constructed by the Pilot process.
2. Removal of XP04's automatic final Test forward. The frozen runner currently constructs a Test loader and evaluates it after model selection, which violates this Pilot boundary.
3. Fold-local Training-only class weights for IEMOCAP. The frozen code uses corpus-level hard-coded weights.
4. Fold-local Training-only curriculum difficulty generation. Existing `caldiff` files follow the old split, include now-held-out groups, and omit groups newly assigned to Training. Reusing them would violate the model-level independence claim. Curriculum remains enabled with XP04 ratios; only its difficulty artifact is regenerated from the fold's Training role.
5. If Unknown-Label Mask is selected, use an independent supervision mask across all label-dependent objectives and metrics while retaining the utterances in sequence/graph construction.
6. Save initial-state hash before the first optimizer step, best/latest checkpoint hashes, manifest hash, source/data/difficulty hashes, GPU UUID, peak memory, wall time and epoch metrics.

The model body does not require architectural changes. BatchNorm running statistics update only under `model.train()` on Training; Development and Calibration must run under `eval()` with no optimizer step.
