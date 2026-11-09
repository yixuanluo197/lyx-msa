# Official Training Protocol Comparison

| Item | External M³Net B | CMERC command A | Protocol B amendment |
|---|---|---|---|
| Architecture | M³Net `Model` | weight-compatible M³Net path in `train_cm.py` | same non-causal architecture, frozen adapter |
| IEMOCAP endpoint | 80 epochs | 80 epochs, seed 1475 | fixed epoch 80, seed 1475 |
| MELD endpoint | 15 epochs in README | 40 epochs, seed 67137 | fixed epoch 40, seed 67137 |
| Selection | Test inspected each epoch; no model save | public author checkpoint rule absent | no selection; final epoch only |
| Data access | original train plus Test diagnostics | original split semantics | current Fold Training only |
| Checkpoint output | not implemented in public script | current local runner can save, upstream provenance absent | explicit final checkpoint with hashes |

The 80/40 endpoints are adopted from the CMERC commands because the teacher exists to reproduce the CMERC curriculum inputs and the MELD external command conflicts with CMERC. This is a preregistered reconstruction choice, not a recovered author checkpoint rule.

Candidate comparison:

1. **Training-only fixed final epoch:** closest structural match, leakage-free, ten teachers, deterministic endpoint. Recommended.
2. **Training-only dialogue cross-fitting:** reduces in-sample teacher scoring concerns but changes the teacher ensemble and multiplies training/inference cost; it is farther from the author method.
3. **CMERC-NoCurriculum:** clean ablation, but removes a named part of CMERC and is not the primary reproduction.
