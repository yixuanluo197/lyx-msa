# Fold-local Teacher Protocol

## Unique recommendation

Train one Training-only M³Net-family teacher for each Dataset × Fold, from random initialization, and save the fixed final epoch. The ten teachers are shared by the five downstream CMERC seeds in their fold.

- IEMOCAP: 80 epochs, seed 1475, dropout 0.5, batch 16, Adam, LR `1e-4`, weight decay `3e-5`, BN, L3/K4.
- MELD: 40 epochs, seed 67137, dropout 0.4, batch 16, Adam, LR `1e-4`, weight decay `3e-5`, BN, L3/K3.
- Access: Training dialogue/episode IDs only. Do not construct Development, Calibration, or Evaluation loaders.
- Endpoint: final epoch, irrespective of any metric. No validation/test checkpoint selection.
- Provenance: initialization hash, final checkpoint hash, approved-manifest hash, Training-role hash, effective config, code hashes, RNG state, and epoch logs.

This preserves the author model family, five removal views, difficulty definition, and curriculum schedule. Training-only access, fixed endpoint, unknown-label masking, batch-one difficulty inference, and fold sharing are Protocol B amendments.

The teacher runner itself still requires a role-scoped loader and an all-true/mixed-mask integration test before launch. The public external script is prohibited because it constructs Test loaders and reads Test every epoch.
