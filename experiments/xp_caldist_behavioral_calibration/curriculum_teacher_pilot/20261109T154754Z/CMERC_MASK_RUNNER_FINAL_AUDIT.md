# CMERC Mask Runner Final Audit

The mask integration gate passed without starting CMERC training. The experiment-local adapter performs all six complete contextual forwards before applying `supervision_mask` to label-dependent operations.

Actual tensor backward tests cover classification NLL, five rank branches, three MSCL relations, and two CSCL relations. Across all 11 paths, masked rows have exactly zero direct label-dependent gradient and eligible rows have non-zero gradient. Teacher/removal scores remain detached. The original `<=` rank branch, including its tie gradient, is preserved.

Additional tests cover fold-local class weights, empty-supervision optimizer skip, singleton contrastive zero behavior, masked Development NLL/accuracy/F1, sentinel-label invariance, full-context unknown-node participation, and exact Training-only curriculum sampling. Frozen `CKSCL_tav` and `CKSCL_context` aggregate behavior is checked against the experiment adapter.

This lane uses real tensor backward and frozen contrastive kernels but does not run a complete CMERC graph-training epoch. The real graph-context requirement is independently covered by the six-view parity gate, where masked utterances remain in every real model forward. Together these checks close the loss-mask integration gate; final verification remains required again when the full CMERC Pilot Runner is instantiated.
