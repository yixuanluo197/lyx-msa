# Training Runner Integration Audit

The existing `protocol_b_masking.py` is isolated and not integrated into the frozen XP04 runner. Its five tests establish basic filtering only.

A new defect was found: it implements rank gaps with `ReLU`. At exact equality, its loss value matches XP04 but its student gradient is zero; XP04's `<=` branch gives a subtraction gradient. The new core preserves XP04 with `torch.where(student<=teacher, teacher-student, 0)` and includes a tie-gradient regression test.

Before CMERC training, the experiment-local runner must:

- construct only Training and Development role-scoped loaders; Calibration is a later phase and Evaluation has no entry point;
- keep unknown nodes through all six forwards and filter afterward;
- filter classification NLL, five rank branches, three MSCL relations, two CSCL relations, class weights, and Development metrics;
- skip backward and optimizer step for an empty supervised batch; skip contrastive terms for singleton supervision;
- compute fold-local class weights from all eligible Training labels before curriculum filtering;
- retain the frozen MELD FocalLoss, teacher-refresh timing, BN/GRU structure, and curriculum schedule;
- read only the exact fold-local difficulty IDs and never fall back to author PKLs.

The curriculum sampler mechanics already rank eligible keys, take `floor(ratio*N)`, and restore source order. IEMOCAP ratios are `.15,.30,.45,.60,.75,.90,1`; MELD ratios are `.40,.80,1` under the one-step schedule. Full runner parity, mixed sentinel, contrastive backward, optimizer-empty-batch, and role-access tripwire tests remain open. Therefore the mask does not yet cover every executed training path, even though the pure helper coverage is improved.
